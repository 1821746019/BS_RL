import jax, jax.numpy as jnp
from jax import Array
import flax.core
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from functools import partial
from .common import jax_jit
from typing import Union, Optional, Tuple, Callable
import tensorflow_probability.substrates.jax.distributions as tfd
import numpy as np
from .common import profile
from .config import AlgoConfig, NetworkConfig
from .networks import Actor, Critic, SharedEncoder, VectorCritic
from .nn.s5 import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel, S5Summarizer
from .nn.lstm import LSTMSummarizer
from .nn.Simba import RSNorm
from .jax_utils import concat_valid
import chex
MAX_NORM = 0.4
class TrainStateWithBatchStats(TrainState):
    batch_stats: Optional[flax.core.FrozenDict] = None

class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict

class EncoderTrainState(TrainState):
    target_params: flax.core.FrozenDict

@chex.dataclass
class AgentState:
    actor_state: TrainState
    critic_state: CriticTrainState
    encoder_state: EncoderTrainState
    rsnorm_state: TrainStateWithBatchStats
    log_alpha_state: Optional[TrainState]
    def replace(self, **kwargs) -> 'AgentState':...

# Add edac_penalty sum slot
CarryType = tuple[AgentState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
class RSACAgent:
    def __init__(self,
                 action_dim: int,
                 key: jax.Array,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 is_discrete: bool,
                 obs_split_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 dummy_obs: np.ndarray,
                 norm_limit: float = MAX_NORM):
        self.actor_model: nn.Module
        self.critic_model: nn.Module
        self.algo_config = algo_config
        self.action_dim = action_dim
        self.network_config = network_config
        self.norm_limit = norm_limit
        self.is_discrete = is_discrete
        self.obs_split_fn = obs_split_fn
        key_actor, key_critic, key_encoder, key_rsnorm, key_log_alpha = jax.random.split(key, 5)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.sgd(learning_rate=self.algo_config.policy_lr, momentum=0.9) if self.algo_config.use_SGD else optax.adamw(learning_rate=self.algo_config.policy_lr, eps=self.algo_config.adam_eps),
        )
        self.actor_optimizer = optimizer
        self.critic_optimizer = optimizer
        self.alpha_optimizer = optimizer
        self.encoder_optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.sgd(learning_rate=self.algo_config.summarizer_lr, momentum=0.9) if self.algo_config.use_SGD else optax.adamw(learning_rate=self.algo_config.summarizer_lr, eps=self.algo_config.adam_eps),
        )

        self.agent_state = self._create_models_and_states(key_actor, key_critic, key_encoder, key_rsnorm, dummy_obs)

        if algo_config.autotune:
            log_alpha_params = {'log_alpha': jnp.log(0.2)}
            log_alpha_state = TrainState.create(apply_fn=None, params=log_alpha_params, tx=self.alpha_optimizer)
            self.agent_state = self.agent_state.replace(log_alpha_state=log_alpha_state)
            self.curr_alpha = jnp.exp(self.agent_state.log_alpha_state.params['log_alpha'])
        else:
            self.curr_alpha = jnp.array(algo_config.alpha)
        
        if self.is_discrete:
            self.target_entropy = -self.algo_config.target_entropy_scale_for_disc * jnp.log(1.0 / self.action_dim)
        else:
            self.target_entropy = -float(self.action_dim) * self.algo_config.target_entropy_scale_for_cont

    def _init_model_with_batch_stats(self, model: nn.Module, key, *args, **kwargs):
        variables = model.init({'params': key, 'dropout': key}, *args, **kwargs)
        params = variables.get('params', flax.core.FrozenDict()) # RSNorm没有param故用.get
        batch_stats = variables.get('batch_stats')
        return params, batch_stats

    def _create_models_and_states(self, key_actor, key_critic, key_encoder, key_rsnorm, dummy_obs: np.ndarray) -> AgentState:
        # Shared encoder and target encoder
        self.shared_encoder = SharedEncoder(network_cfg=self.network_config)

        dummy_obs_batched = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
        dummy_cnn, dummy_mem, dummy_instant = self.obs_split_fn(dummy_obs_batched)
        
        encoder_params, _ = self._init_model_with_batch_stats(self.shared_encoder, key_encoder, dummy_mem, dummy_cnn, hidden_state=None, training=False)
        
        # 动态推断 summarizer 的输出维度
        dummy_encoder_outputs, _ = self.shared_encoder.apply({'params': encoder_params}, dummy_mem, dummy_cnn, hidden_state=None, training=False)
        encoder_output_dim = dummy_encoder_outputs.shape[-1]  if dummy_encoder_outputs is not None else 0
        
        if self.network_config.use_pretrained_summarizer_path:
            try:
                loaded = checkpoints.restore_checkpoint(self.network_config.use_pretrained_summarizer_path, target={'summarizer_params': encoder_params})
                encoder_params = loaded.get('summarizer_params', encoder_params)
                print(f"Loaded pretrained encoder from {self.network_config.use_pretrained_summarizer_path}")
            except Exception as e:
                print(f"Warning: failed to load pretrained encoder: {e}")
        
        encoder_state = EncoderTrainState.create(
            apply_fn=self.shared_encoder.apply, 
            params=encoder_params, 
            target_params=encoder_params, 
            tx=self.encoder_optimizer
        )

        # Actor head
        self.actor_model = Actor(network_config=self.network_config, action_dim=self.action_dim, is_discrete=self.is_discrete)
        actor_input_dim = int(encoder_output_dim + (0 if dummy_instant is None else dummy_instant.shape[-1]))
        actor_params, _ = self._init_model_with_batch_stats(self.actor_model,key_actor,jnp.zeros((1, actor_input_dim)),deterministic=True,)
        actor_state = TrainState.create(apply_fn=self.actor_model.apply, params=actor_params, tx=self.actor_optimizer)

        # Critic heads (two critics)
        self.critic_model = VectorCritic(network_config=self.network_config, action_dim=self.action_dim, is_discrete=self.is_discrete, n_critics=self.algo_config.n_critics)
        dummy_critic_input = jnp.zeros((1, actor_input_dim))
        
        if self.is_discrete:
            critic_params, _ = self._init_model_with_batch_stats(self.critic_model, key_critic, dummy_critic_input, deterministic=True)
        else:
            dummy_action = jnp.zeros((1, self.action_dim))
            critic_params, _ = self._init_model_with_batch_stats(self.critic_model, key_critic, dummy_critic_input, action=dummy_action, deterministic=True)

        critic_state = CriticTrainState.create(apply_fn=self.critic_model.apply, params=critic_params, target_params=critic_params, tx=self.critic_optimizer)

        self.train_encoder = bool(self.network_config.train_summarizer)

        # RSNorm for observations
        self.rsnorm_model = RSNorm()
        rsnorm_params, rsnorm_batch_stats = self._init_model_with_batch_stats(self.rsnorm_model, key_rsnorm, dummy_obs[None,...], update_stats=False)
        rsnorm_state = TrainStateWithBatchStats.create(
            apply_fn=self.rsnorm_model.apply,
            params=rsnorm_params,
            batch_stats=rsnorm_batch_stats,
            tx=optax.sgd(1e-4)  # Dummy optimizer, not used
        )
        return AgentState(
            actor_state=actor_state,
            critic_state=critic_state,
            encoder_state=encoder_state,
            rsnorm_state=rsnorm_state,
            log_alpha_state=None,
        )

    def _norm_obs(self, rsnorm_state: TrainStateWithBatchStats, obs: jnp.ndarray, update_stats: bool):
        def _update():
            norm_obs, new_vars = self.rsnorm_model.apply(
                {'params': rsnorm_state.params, 'batch_stats': rsnorm_state.batch_stats},
                obs,
                update_stats=True,
                mutable=['batch_stats']
            )
            new_state = rsnorm_state.replace(batch_stats=new_vars['batch_stats'])
            return new_state, norm_obs

        def _no_update():
            norm_obs = self.rsnorm_model.apply(
                {'params': rsnorm_state.params, 'batch_stats': rsnorm_state.batch_stats},
                obs,
                update_stats=False
            )
            return rsnorm_state, norm_obs

        return _update() if update_stats else _no_update()

    def _compute_edac_penalty_continuous(self,
                                         critic_params: flax.core.FrozenDict,
                                         actor_params: flax.core.FrozenDict,
                                         x_t: jnp.ndarray,
                                         loss_calc_m: jnp.ndarray,
                                         key: jax.Array) -> jnp.ndarray:
        """EDAC penalty for continuous actions using action-gradients of critics.
        Computes mean of unit gradients across critics and penalizes its squared norm.
        """
        if not self.algo_config.use_edac:
            return jnp.array(0.0)

        B, T = loss_calc_m.shape
        # Sample actions from current policy at s_t
        mean_t, log_std_t = self.actor_model.apply({'params': actor_params}, x_t.reshape(-1, x_t.shape[-1]), deterministic=True)
        mean_t = mean_t.reshape(B, T, -1)
        log_std_t = log_std_t.reshape(B, T, -1)
        dist_t = tfd.MultivariateNormalDiag(loc=mean_t, scale_diag=jnp.exp(log_std_t))
        u_t = dist_t.sample(seed=key)
        a_hat = jnp.tanh(u_t)  # (B, T, A)

        # Flatten for per-sample jacobian
        BT = B * T
        x_flat = x_t.reshape(BT, -1)
        a_flat = a_hat.reshape(BT, -1)
        mask_flat = loss_calc_m.reshape(BT)

        def q_fn_single(a_single, x_single):
            # returns (n_critics,)
            return self.critic_model.apply({'params': critic_params}, x_single[None, :], action=a_single[None, :], deterministic=True).squeeze(1)

        jacobian_fn = jax.jacrev(q_fn_single, argnums=0)
        grads = jax.vmap(jacobian_fn, in_axes=(0, 0))(a_flat, x_flat)  # (BT, n_critics, A)

        grad_norm = jnp.linalg.norm(grads, axis=-1, keepdims=True) + 1e-8
        grads_unit = grads / grad_norm
        mean_grad = jnp.mean(grads_unit, axis=1)  # (BT, A)
        penalty_per_sample = jnp.sum(mean_grad ** 2, axis=-1)  # (BT,)

        denom = mask_flat.sum() + 1e-8
        penalty_avg = (penalty_per_sample * mask_flat).sum() / denom
        scale = self.algo_config.edac_coef / jnp.maximum(1, self.algo_config.n_critics - 1)
        return scale * penalty_avg

    def _compute_edac_penalty_discrete(self,
                                       critic_params: flax.core.FrozenDict,
                                       x_t: jnp.ndarray,
                                       loss_calc_m: jnp.ndarray) -> jnp.ndarray:
        """EDAC-like penalty for discrete actions by diversifying Q-vectors across actions.
        Uses unit-normalized Q-vectors per critic and penalizes squared norm of their mean.
        """
        if not self.algo_config.use_edac:
            return jnp.array(0.0)

        B, T = loss_calc_m.shape
        BT = B * T
        q_all = self.critic_model.apply({'params': critic_params}, x_t.reshape(-1, x_t.shape[-1]), deterministic=True)  # (nC, BT, A)
        nC = q_all.shape[0]
        q_all = q_all.reshape(nC, B, T, -1)  # (nC, B, T, A)
        # Move critic axis to end for normalization convenience
        q_perm = jnp.transpose(q_all, (1, 2, 0, 3))  # (B, T, nC, A)
        q_norm = jnp.linalg.norm(q_perm, axis=-1, keepdims=True) + 1e-8  # (B, T, nC, 1)
        q_unit = q_perm / q_norm  # (B, T, nC, A)
        mean_vec = jnp.mean(q_unit, axis=2)  # (B, T, A)
        penalty_bt = jnp.sum(mean_vec ** 2, axis=-1)  # (B, T)
        denom = loss_calc_m.sum() + 1e-8
        penalty_avg = (penalty_bt * loss_calc_m).sum() / denom
        scale = self.algo_config.edac_coef / jnp.maximum(1, self.algo_config.n_critics - 1)
        return scale * penalty_avg

    @partial(jax_jit, static_argnames=('self', 'deterministic'))
    def select_action(self, actor_state: TrainState, encoder_params: flax.core.FrozenDict, obs: jnp.ndarray, hidden_state: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key: jax.Array, deterministic: bool = False):
        key, dropout_key = jax.random.split(key)
        # 若obs的shape为(B,)，需增加一个轴。也就是把标量视为1d数组
        if len(obs.shape) == 1: obs = obs[:, None]
        obs_cnn_mem, obs_mem, obs_instant = self.obs_split_fn(obs)

        # Add sequence dimension for summarizer
        obs_cnn_mem = obs_cnn_mem[:, None, :] if obs_cnn_mem is not None else None
        obs_mem = obs_mem[:, None, :] if obs_mem is not None else None
        
        outputs, new_hidden_state = self.shared_encoder.apply(
            {'params': encoder_params},
            obs_mem,
            obs_cnn_mem,
            hidden_state,
            training=not deterministic,
            rngs={'dropout': dropout_key}
        )
        summary_t = outputs[:, -1, :] if outputs is not None else None
        x = concat_valid([summary_t, obs_instant], axis=-1)
        det_flag = jnp.asarray(deterministic)

        if self.is_discrete:
            logits = self.actor_model.apply({'params': actor_state.params}, x, deterministic=True)
            def take_argmax(k):
                return jnp.argmax(logits, axis=-1)
            def take_sample(k):
                return jax.random.categorical(k, logits, axis=-1)
            actions = jax.lax.cond(det_flag, take_argmax, take_sample, key)
            return actions, new_hidden_state
        else:
            mean, log_std = self.actor_model.apply({'params': actor_state.params}, x, deterministic=True)
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
            def take_mean(k):
                return dist.mean()
            def take_sample(k):
                return dist.sample(seed=k)
            actions = jax.lax.cond(det_flag, take_mean, take_sample, key)
            squashed = jnp.tanh(actions)
            return squashed, new_hidden_state

    @partial(jax_jit, static_argnames=('self', 'deterministic'))
    def select_action_for_eval(self,
                               actor_state: TrainState,
                               encoder_params: flax.core.FrozenDict,
                               rsnorm_state: TrainStateWithBatchStats,
                               obs: jnp.ndarray,
                               hidden_state: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
                               key: jax.Array,
                               deterministic: bool = True):
        key_action, new_key = jax.random.split(key)
        _, norm_obs = self._norm_obs(rsnorm_state, obs, update_stats=False)

        actions, new_hidden_state = self.select_action(
            actor_state,
            encoder_params,
            norm_obs,
            hidden_state,
            key_action,
            deterministic=deterministic
        )
        return actions, new_hidden_state, new_key

    def _create_calc_loss_mask(self, is_real: jnp.ndarray):
        B, T = is_real.shape
        burn_in_steps = int(self.algo_config.burn_in)
        if burn_in_steps <= 0:
            return is_real
        
        # Create a mask that is True for the first `burn_in_steps`
        burn_in_mask = jnp.arange(T) < burn_in_steps
        # 真实数据且不在burn-in阶段时计算损失
        calc_loss = jnp.logical_and(is_real.astype(bool), jnp.logical_not(burn_in_mask))
        return calc_loss.astype(jnp.float32)
    def _update_critic(self, actor_state: TrainState, critic_state: CriticTrainState, encoder_state: EncoderTrainState, curr_alpha: Array, obs_cnn: Array, obs_mem: Array, obs_instant: Array, a: Array, r: Array, term: Array, loss_calc_m: Array, key: Array):
        B, T = loss_calc_m.shape[0], loss_calc_m.shape[1]
        key, dropout_key = jax.random.split(key)
         # Pre-calculate summaries for the target network, which should be detached from grad calculations
        s_tp1_targ, _ = self.shared_encoder.apply({'params': encoder_state.target_params}, obs_mem, obs_cnn, hidden_state=None, training=False)
        s_tp1 = s_tp1_targ[:, 1:, :] if s_tp1_targ is not None else None
        obs_instant_tp1 = obs_instant[:, 1:, :] if obs_instant is not None else None
        x_tp1 = concat_valid([s_tp1, obs_instant_tp1], axis=-1)

        def maybe_freeze_encoder(params):
            if self.train_encoder:
                return params
            else:
                return jax.lax.stop_gradient(params)
        if self.is_discrete:
            def critic_loss_fn(critic_params, encoder_params):
                encoder_params = maybe_freeze_encoder(encoder_params)
                summaries, _ = self.shared_encoder.apply({'params': encoder_params}, obs_mem, obs_cnn, hidden_state=None, training=True, rngs={'dropout': dropout_key})
                s_t = summaries[:, :-1, :] if summaries is not None else None
                obs_instant_t = obs_instant[:, :-1, :] if obs_instant is not None else None
                x_t = concat_valid([s_t, obs_instant_t], axis=-1)
                
                next_logits = self.actor_model.apply({'params': actor_state.params}, x_tp1.reshape(-1, x_tp1.shape[-1]), deterministic=True)
                next_logits = next_logits.reshape(B, T, -1)
                next_probs = nn.softmax(next_logits, axis=-1)
                next_log_probs = nn.log_softmax(next_logits, axis=-1)

                q_next_all = self.critic_model.apply({'params': critic_state.target_params}, x_tp1.reshape(-1, x_tp1.shape[-1]), deterministic=True).reshape(self.algo_config.n_critics, B, T, -1)
                q_next = jnp.median(q_next_all, axis=0)
                v_next = jnp.sum(next_probs * (q_next - curr_alpha * next_log_probs), axis=-1)
                target = r + (1.0 - term) * self.algo_config.gamma * v_next

                q_all = self.critic_model.apply({'params': critic_params}, x_t.reshape(-1, x_t.shape[-1]), deterministic=False)
                q_all = q_all.reshape(self.algo_config.n_critics, B, T, -1)
                a_idx = a[..., None]
                a_idx_expanded = a_idx[None, ...] # (1, B, T, 1)
                q_taken = jnp.take_along_axis(q_all, a_idx_expanded, axis=-1).squeeze(-1)

                target_expanded = target[None, ...]
                mse = (q_taken - target_expanded)**2
                mse = (mse * loss_calc_m[None, ...]).sum(axis=(1,2)) / (loss_calc_m.sum() + 1e-8)
                loss = mse.mean()
                qf1_value_mean = (q_taken[0] * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                qf2_value_mean = (q_taken[1] * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                # Discrete EDAC-like penalty based on Q-vector alignment across critics
                edac_penalty = self._compute_edac_penalty_discrete(critic_params, x_t, loss_calc_m) if self.algo_config.use_edac else jnp.array(0.0)
                loss = loss + edac_penalty
                return loss, (qf1_value_mean, qf2_value_mean, x_t, edac_penalty)
        else: # continuous
            def critic_loss_fn(critic_params, encoder_params):
                encoder_params = maybe_freeze_encoder(encoder_params)
                summaries, _ = self.shared_encoder.apply({'params': encoder_params}, obs_mem, obs_cnn, hidden_state=None, training=True, rngs={'dropout': dropout_key})
                s_t = summaries[:, 0:-1, :] if summaries is not None else None
                obs_instant_t = obs_instant[:, :-1, :] if obs_instant is not None else None
                x_t = concat_valid([s_t, obs_instant_t], axis=-1)

                mean_tp1, log_std_tp1 = self.actor_model.apply({'params': actor_state.params}, x_tp1.reshape(-1, x_tp1.shape[-1]), deterministic=True)
                mean_tp1 = mean_tp1.reshape(B, T, -1)
                log_std_tp1 = log_std_tp1.reshape(B, T, -1)
                dist_tp1 = tfd.MultivariateNormalDiag(loc=mean_tp1, scale_diag=jnp.exp(log_std_tp1))
                u = dist_tp1.sample(seed=key)
                squashed_tp1 = jnp.tanh(u)
                log_prob = dist_tp1.log_prob(u)
                log_prob -= jnp.sum(jnp.log(1 - jnp.tanh(u) ** 2 + 1e-6), axis=-1)

                q_next_all = self.critic_model.apply({'params': critic_state.target_params}, x_tp1.reshape(-1, x_tp1.shape[-1]), action=squashed_tp1.reshape(-1, squashed_tp1.shape[-1]), deterministic=True).reshape(self.algo_config.n_critics, B, T)
                q_next = jnp.median(q_next_all, axis=0)
                target = r + (1.0 - term) * self.algo_config.gamma * (q_next - curr_alpha * log_prob)

                q_cur = self.critic_model.apply({'params': critic_params}, x_t.reshape(-1, x_t.shape[-1]), action=a.reshape(-1, a.shape[-1]), deterministic=False)
                q_cur = q_cur.reshape(self.algo_config.n_critics, B, T)

                target_expanded = target[None, ...]
                mse = (q_cur - target_expanded)**2
                mse = (mse * loss_calc_m[None, ...]).sum(axis=(1,2)) / (loss_calc_m.sum() + 1e-8)
                loss = mse.mean()
                qf1_value_mean = (q_cur[0] * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                qf2_value_mean = (q_cur[1] * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                # EDAC penalty (continuous): compute via helper
                edac_penalty = self._compute_edac_penalty_continuous(critic_params, actor_state.params, x_t, loss_calc_m, key) if self.algo_config.use_edac else jnp.array(0.0)
                loss = loss + edac_penalty
                return loss, (qf1_value_mean, qf2_value_mean, x_t, edac_penalty)

        (critic_loss_val, (qf1_value_mean, qf2_value_mean, x_t, edac_penalty_val)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True, argnums=(0,1))(critic_state.params, encoder_state.params)
        g_critic_params, g_enc_params = critic_grads
        critic_state_new = critic_state.apply_gradients(grads=g_critic_params)
        encoder_state_new = encoder_state.apply_gradients(grads=g_enc_params)
        return critic_state_new, encoder_state_new, (critic_loss_val, qf1_value_mean, qf2_value_mean, x_t, edac_penalty_val)
    def _update_actor_and_alpha(self, actor_state: TrainState, critic_state: CriticTrainState, log_alpha_state: Optional[TrainState], curr_alpha: Array, x_t: Array, loss_calc_m: Array, key: Array):
        B, T = loss_calc_m.shape
        if self.is_discrete:
            def actor_loss_fn(actor_params):
                logits = self.actor_model.apply(
                    {'params': actor_params},
                    x_t.reshape(-1, x_t.shape[-1]),
                    deterministic=False,
                ).reshape(B, T, -1)
                probs = nn.softmax(logits, axis=-1)
                log_probs = nn.log_softmax(logits, axis=-1)
                q_all = self.critic_model.apply({'params': critic_state.params}, x_t.reshape(-1, x_t.shape[-1]), deterministic=True).reshape(self.algo_config.n_critics, B, T, -1)
                min_q = jnp.min(q_all, axis=0)
                actor_loss_t = jnp.sum(probs * (curr_alpha * log_probs - min_q), axis=-1)
                actor_loss = (actor_loss_t * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                entropy = (-jnp.sum((probs + 1e-8) * log_probs, axis=-1) * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                return actor_loss, entropy
        else: # continuous
            def actor_loss_fn(actor_params):
                mean, log_std = self.actor_model.apply(
                    {'params': actor_params},
                    x_t.reshape(-1, x_t.shape[-1]),
                    deterministic=False,
                )
                mean = mean.reshape(B, T, -1)
                log_std = log_std.reshape(B, T, -1)
                dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
                u = dist.sample(seed=key)
                squashed = jnp.tanh(u)
                log_prob = dist.log_prob(u)
                log_prob -= jnp.sum(jnp.log(1 - jnp.tanh(u) ** 2 + 1e-6), axis=-1)

                q_pi = self.critic_model.apply({'params': critic_state.params}, x_t.reshape(-1, x_t.shape[-1]), action=squashed.reshape(-1, squashed.shape[-1]), deterministic=True).reshape(self.algo_config.n_critics, B, T)
                min_q = jnp.min(q_pi, axis=0)
                loss_t = (curr_alpha * log_prob - min_q)
                loss = (loss_t * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                entropy = (-(log_prob) * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                return loss, entropy

        (actor_loss_val, entropy_val), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state_new = actor_state.apply_gradients(grads=actor_grads)

        alpha_loss_val = jnp.array(0.0)
        log_alpha_state_new = log_alpha_state
        curr_alpha_new = curr_alpha # 没autotune时alpha值是固定的
        if self.algo_config.autotune:
            if self.is_discrete:
                logits_det = self.actor_model.apply({'params': actor_state.params}, x_t.reshape(-1, x_t.shape[-1]), deterministic=True).reshape(B, T, -1)
                log_probs_det = nn.log_softmax(logits_det, axis=-1)
                probs_det = nn.softmax(logits_det, axis=-1)
                def alpha_loss_fn(log_alpha_params):
                    lp = log_probs_det
                    pr = probs_det
                    loss_t = pr * (-jnp.exp(log_alpha_params['log_alpha']) * (lp + self.target_entropy))
                    loss = (jnp.sum(loss_t, axis=-1) * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
                    return loss
            else: # continuous
                mean_det, log_std_det = self.actor_model.apply({'params': actor_state.params}, x_t.reshape(-1, x_t.shape[-1]), deterministic=True)
                mean_det = mean_det.reshape(B, T, -1)
                log_std_det = log_std_det.reshape(B, T, -1)
                dist_det = tfd.MultivariateNormalDiag(loc=mean_det, scale_diag=jnp.exp(log_std_det))
                u_det = dist_det.sample(seed=key)
                log_prob_det = dist_det.log_prob(u_det)
                log_prob_det -= jnp.sum(jnp.log(1 - jnp.tanh(u_det) ** 2 + 1e-6), axis=-1)
                def alpha_loss_fn(log_alpha_params):
                    return ((-jnp.exp(log_alpha_params['log_alpha']) * (log_prob_det + self.target_entropy)) * loss_calc_m).sum() / (loss_calc_m.sum() + 1e-8)
            
            alpha_loss_val, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha_state.params)
            log_alpha_state_new = log_alpha_state.apply_gradients(grads=alpha_grads)
            # clamp updated log_alpha and alpha
            alpha_clamped = jnp.clip(jnp.exp(log_alpha_state_new.params['log_alpha']), self.algo_config.alpha_min, self.algo_config.alpha_max)
            log_alpha_state_new = log_alpha_state_new.replace(params={'log_alpha': jnp.log(alpha_clamped)})
            curr_alpha_new = alpha_clamped
        return actor_state_new, log_alpha_state_new, curr_alpha_new, (actor_loss_val, entropy_val, alpha_loss_val)

    @partial(jax_jit, static_argnames=('self',))
    def _update(self, agent_state: AgentState, batch: dict, key: jax.Array) -> Tuple[AgentState, jnp.ndarray, dict]:
        key, dropout_key = jax.random.split(key)
        o = batch['o']
        a = batch['a']
        r = batch['r']
        term = batch['term']
        # trunc = batch['trunc']
        is_real = batch['m'] # rb的m为1时表示对应索引的数据是真实的，为0时表示是padding的
        
        # Apply burn-in mask, m from rb only indicates padding, not burn-in
        loss_calc_m = self._create_calc_loss_mask(is_real)
        
        _, o_normalized = self._norm_obs(agent_state.rsnorm_state, o, update_stats=False)
        obs_cnn, obs_mem, obs_instant = self.obs_split_fn(o_normalized)

        if self.algo_config.autotune:
            curr_alpha = jnp.exp(agent_state.log_alpha_state.params['log_alpha'])
        else:
            curr_alpha = self.curr_alpha

        critic_state_new, encoder_state_new, (critic_loss_val, qf1_value_mean, qf2_value_mean, x_t, edac_penalty_val) = self._update_critic(agent_state.actor_state, agent_state.critic_state, agent_state.encoder_state, curr_alpha, obs_cnn, obs_mem, obs_instant, a, r, term, loss_calc_m, key)
        # 首次update不更新actor和alpha
        update_actor_and_alpha = agent_state.critic_state.step != 0
        def no_update_aa(actor_state, critic_state, log_alpha_state, curr_alpha, x_t, loss_calc_m, key):
            return actor_state, log_alpha_state, curr_alpha, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        defer_update = False 
        def maybe_defer_to_update_aa(actor_state, critic_state, log_alpha_state, curr_alpha, x_t, loss_calc_m, key):
            if defer_update: # 流程C0->C1A0->C2A1 critic评估的是上上步的actor。共用x_t减少了一次summarizer的前向传播，能带来10%的SPS提升(500->550)
                return jax.lax.cond(update_actor_and_alpha, self._update_actor_and_alpha, no_update_aa, actor_state, critic_state, log_alpha_state, curr_alpha, x_t, loss_calc_m, key)
            else: # 标准流程C0A0->C1A1 critic评估的是上一步的(最新的)actor
                summaries, _ = self.shared_encoder.apply({'params': encoder_state_new.params}, obs_mem, obs_cnn, hidden_state=None, training=True, rngs={'dropout': dropout_key})
                s_t_new = summaries[:, :-1, :] if summaries is not None else None
                obs_instant_t = obs_instant[:, :-1, :] if obs_instant is not None else None
                x_t_new = concat_valid([s_t_new, obs_instant_t], axis=-1)
                return self._update_actor_and_alpha(actor_state, critic_state_new, log_alpha_state, curr_alpha, x_t_new, loss_calc_m, key)
        actor_state_new, log_alpha_state_new, curr_alpha_new, (actor_loss_val, entropy_val, alpha_loss_val) = maybe_defer_to_update_aa(agent_state.actor_state, agent_state.critic_state, agent_state.log_alpha_state, curr_alpha, x_t, loss_calc_m, key)
        
        agent_state_new = agent_state.replace(
            actor_state=actor_state_new,
            critic_state=critic_state_new,
            encoder_state=encoder_state_new,
            log_alpha_state=log_alpha_state_new,
        )
        metrics = {
            'critic_loss': critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha_loss': alpha_loss_val,
            'alpha': curr_alpha_new,
            'entropy': entropy_val,
            'qf1_value_mean': qf1_value_mean,
            'qf2_value_mean': qf2_value_mean,
            'edac_penalty': edac_penalty_val,
        }
        return agent_state_new, curr_alpha_new, metrics

    @partial(jax_jit, static_argnames=('self', 'do_update', 'deterministic'))
    def update_then_select_action(self, obs, hidden_state, batches, do_update, agent_state: AgentState, key: jax.Array, updates_per_call: int, deterministic: bool = False):
        """Combined action selection and agent update to reduce CPU-TPU communication with multiple updates per call.

        batches: if do_update, a dict with leading dim K=updates_per_call; otherwise can be None or a single batch.
        """
        # Split rng on device to avoid host-device traffic
        key_action, key_update, new_key = jax.random.split(key, 3)
        
        # If put _norm_obs in select_action, the first update will use raw obs to calc loss, which cause instability. So put _norm_obs here to make update func(loss calc) use the newest rsnorm_state
        updated_rsnorm_state, norm_obs_for_action = self._norm_obs(
            agent_state.rsnorm_state, obs, update_stats=deterministic
        )
        agent_state_new = agent_state.replace(rsnorm_state=updated_rsnorm_state)
        # 1. First perform agent update(s) if requested
        if do_update:
            num_updates = jnp.asarray(updates_per_call, dtype=jnp.int32)
            init_carry: CarryType = (
                agent_state_new,
                jnp.array(0.0),  # critic_loss_sum
                jnp.array(0.0),  # actor_loss_sum
                jnp.array(0.0),  # alpha_loss_sum
                jnp.array(0.0),  # entropy_sum
                jnp.array(0.0),  # qf1_mean_sum
                jnp.array(0.0),  # qf2_mean_sum
                jnp.array(0.0),  # current_alpha placeholder
                key_update,
                jnp.array(0.0),  # edac_penalty_sum
            )

            def body_fun(i, carry: CarryType):
                # 每次更新应使用不同的key
                (ag_state,
                 cl_sum, al_sum, aloss_sum, ent_sum, qf1m_sum, qf2m_sum, cur_alpha, key_update_in, edac_sum) = carry
                key_update_i, key_update_out = jax.random.split(key_update_in)
                b = {
                    'o': batches['o'][i],
                    'a': batches['a'][i],
                    'r': batches['r'][i],
                    'term': batches['term'][i],
                    'trunc': batches['trunc'][i],
                    'm': batches['m'][i],
                }
                ag_state_n, cur_alpha_n, metric_i = self._update(ag_state, b, key_update_i)
                ag_state_n: AgentState 
                cur_alpha_n: jnp.ndarray 
                metric_i: dict
                # Target network update logic inside the loop
                target_update_freq = self.algo_config.target_network_frequency // max(1, self.algo_config.update_frequency)
                
                def _update_targets(states: Tuple[CriticTrainState, EncoderTrainState]):
                    critic, s = states
                    tau = self.algo_config.tau
                    critic = critic.replace(target_params=optax.incremental_update(critic.params, critic.target_params, tau))
                    s = s.replace(target_params=optax.incremental_update(s.params, s.target_params, tau))
                    return critic, s

                def _no_update_targets(states):
                    return states

                critic_state_n, s_state_n = jax.lax.cond(
                    (ag_state_n.actor_state.step % target_update_freq == 0),
                    _update_targets,
                    _no_update_targets,
                    (ag_state_n.critic_state, ag_state_n.encoder_state)
                )
                ag_state_n = ag_state_n.replace(critic_state=critic_state_n, encoder_state=s_state_n)
                
                cl_sum = cl_sum + metric_i['critic_loss']
                al_sum = al_sum + metric_i['actor_loss']
                aloss_sum = aloss_sum + metric_i['alpha_loss']
                ent_sum = ent_sum + metric_i['entropy']
                qf1m_sum = qf1m_sum + metric_i['qf1_value_mean']
                qf2m_sum = qf2m_sum + metric_i['qf2_value_mean']
                edac_sum = edac_sum + metric_i.get('edac_penalty', jnp.array(0.0))
                return (ag_state_n,
                        cl_sum, al_sum, aloss_sum, ent_sum, qf1m_sum, qf2m_sum, cur_alpha_n, key_update_out, edac_sum)

            (agent_state_new,
             critic_loss_sum, actor_loss_sum, alpha_loss_sum, entropy_sum, qf1_mean_sum, qf2_mean_sum, curr_alpha, _, edac_penalty_sum) = \
                jax.lax.fori_loop(0, num_updates, body_fun, init_carry)

            kf = jnp.maximum(1, num_updates)
            metrics = {
                'critic_loss': critic_loss_sum / kf,
                'actor_loss': actor_loss_sum / kf,
                'alpha_loss': alpha_loss_sum / kf,
                'alpha': curr_alpha,
                'entropy': entropy_sum / kf,
                'qf1_value_mean': qf1_mean_sum / kf,
                'qf2_value_mean': qf2_mean_sum / kf,
                'edac_penalty': edac_penalty_sum / kf,
            }
        else:
            curr_alpha = jnp.exp(agent_state_new.log_alpha_state.params['log_alpha']) if self.algo_config.autotune and agent_state_new.log_alpha_state else jnp.array(self.algo_config.alpha)
            metrics = {}
        
        # 2. Then perform action selection using potentially updated states
        actions, new_hidden_state = self.select_action(agent_state_new.actor_state, agent_state_new.encoder_state.params, norm_obs_for_action, hidden_state, key_action, deterministic=deterministic)
        
        return actions, new_hidden_state, agent_state_new, metrics, new_key