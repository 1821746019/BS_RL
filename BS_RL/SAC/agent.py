import jax
import jax.numpy as jnp
import flax.core
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from functools import partial
from .common import jax_jit
from typing import Union, Optional, Tuple
import tensorflow_probability.substrates.jax.distributions as tfd
from .common import profile
from .config import AlgoConfig, NetworkConfig
from .networks import TradingActorDiscrete, TradingCriticDiscrete, TradingActorContinuous, TradingCriticContinuous
from .s5 import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel
MAX_NORM = 0.4
class TrainStateWithBatchStats(TrainState):
    batch_stats: Optional[flax.core.FrozenDict] = None

class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Optional[flax.core.FrozenDict] = None
    target_batch_stats: Optional[flax.core.FrozenDict] = None

class SummarizerTrainState(TrainState):
    target_params: flax.core.FrozenDict

class _LSTMStackStep(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x_t: jnp.ndarray):
        # carry: (h_layers, c_layers) with shape [L,B,H]
        h_layers, c_layers = carry
        cur = x_t  # [B,H]
        new_h_layers = []
        new_c_layers = []
        for i in range(self.num_layers):
            cell = nn.OptimizedLSTMCell(features=self.hidden_dim, name=f"lstm_{i}")
            (h_new, c_new), cur = cell((h_layers[i], c_layers[i]), cur)
            new_h_layers.append(h_new)
            new_c_layers.append(c_new)
        new_h = jnp.stack(new_h_layers, axis=0)
        new_c = jnp.stack(new_c_layers, axis=0)
        summary_t = new_h_layers[-1]  # [B,H]
        return (new_h, new_c), summary_t

class LSTMSummarizer(nn.Module):
    hidden_dim: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        """
        x: [B, T, D]
        Returns:
          outputs: [B, T+1, H] (includes initial top-layer hidden as t=0)
          final_state: (h, c) with shapes [L,B,H]
        """
        B, T, _ = x.shape
        H = self.hidden_dim
        L = self.num_layers
        x_proj = nn.Dense(H, name="in_proj")(x)  # [B,T,H]

        def init_state(bs):
            h0 = jnp.zeros((L, bs, H))
            c0 = jnp.zeros((L, bs, H))
            return (h0, c0)

        if initial_state is None:
            h, c = init_state(B)
        else:
            h, c = initial_state

        # Create a scanned module over time
        Scanned = nn.scan(
            _LSTMStackStep,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        time_scan = Scanned(hidden_dim=H, num_layers=L, name="time_scan")
        (final_h, final_c), ys = time_scan((h, c), x_proj)  # ys: [B,T,H]
        outputs = jnp.concatenate([h[-1][:, None, :], ys], axis=1)  # [B,T+1,H]
        return outputs, (final_h, final_c)

class S5Summarizer(nn.Module):
    hidden_dim: int
    num_layers: int = 1
    delta_min: float = 0.001
    delta_max: float = 0.1

    def setup(self):
        H = self.hidden_dim
        if H % 2 != 0:
            raise ValueError(f"S5Summarizer.hidden_dim must be even for conjugate symmetry, got {H}")
        P = H // 2  # ensure H = 2P to pack complex state
        # Build DPLR HiPPO
        Lambda, _, _, V, _ = make_DPLR_HiPPO(P)
        Vinv = jnp.conj(jnp.transpose(V))
        ssm_init_fn = init_S5SSM(
            H=H,
            P=P,
            Lambda_re_init=jnp.real(Lambda),
            Lambda_im_init=jnp.imag(Lambda),
            V=V,
            Vinv=Vinv,
            C_init="lecun_normal",
            discretization="zoh",
            dt_min=self.delta_min,
            dt_max=self.delta_max,
            conj_sym=True,
            clip_eigs=False,
            bidirectional=False,
        )
        self.encoder = nn.Dense(H, name="in_proj")
        self.s5_stack = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=H,
            n_layers=self.num_layers,
            activation="full_glu",
            do_norm=True,
            prenorm=True,
            do_gtrxl_norm=True,
        )

    def _pack_hidden_real_to_complex(self, h_real: jnp.ndarray) -> jnp.ndarray:
        """h_real: [B, H] -> complex [1, B, P] where H=2P"""
        H = h_real.shape[-1]
        P = H // 2
        real = h_real[..., :P]
        imag = h_real[..., P: P * 2]
        h_complex = real + 1j * imag
        return h_complex[None, ...]

    def _unpack_hidden_complex_to_real(self, h_complex: jnp.ndarray) -> jnp.ndarray:
        """h_complex: [1, B, P] -> real [B, H] with H=2P"""
        real = jnp.real(h_complex[0])
        imag = jnp.imag(h_complex[0])
        return jnp.concatenate([real, imag], axis=-1)

    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        """
        x: [B, T, D]
        Returns:
          outputs: [B, T+1, H]
          final_state: (h, c_dummy) with shapes [L,B,H]
        """
        B, T, _ = x.shape
        H = self.hidden_dim
        L = self.num_layers

        x_proj = self.encoder(x)  # [B, T, H]
        # Transpose to [T, B, H] for S5 stack
        x_tbH = jnp.swapaxes(x_proj, 0, 1)

        # Prepare initial hidden states per S5 layer as complex [1,B,P]
        if initial_state is None:
            h0_layers = [jnp.zeros((1, B, max(H // 2, 1)), dtype=jnp.complex64) for _ in range(L)]
        else:
            h_init, _ = initial_state  # [L, B, H]
            h0_layers = [self._pack_hidden_real_to_complex(h_init[i]) for i in range(L)]

        # Resets: zeros [T, B]
        d_tb = jnp.zeros((T, B), dtype=jnp.float32)

        # Apply S5 stack
        new_h_layers, y_tbH = self.s5_stack(h0_layers, x_tbH, d_tb)

        # Convert new hidden back to real [L, B, H]
        final_h_real = jnp.stack([self._unpack_hidden_complex_to_real(hc) for hc in new_h_layers], axis=0)

        # To outputs [B, T+1, H] with prepended initial top layer hidden
        top_init = final_h_real[-1] * 0.0  # default zeros for t=0
        if initial_state is not None:
            top_init = initial_state[0][-1]
        y_bTH = jnp.swapaxes(y_tbH, 0, 1)
        outputs = jnp.concatenate([top_init[:, None, :], y_bTH], axis=1)
        return outputs, (final_h_real, jnp.zeros_like(final_h_real))

class RSACAgentBase:
    def __init__(self,
                 action_dim: int,
                 observation_space_shape,
                 key: jax.Array,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 actor_model_cls,
                 critic_model_cls,
                 is_discrete: bool,
                 norm_limit: float):
        self.actor_model: nn.Module
        self.critic_model: nn.Module
        self.algo_config = algo_config
        self.action_dim = action_dim
        self.network_config = network_config
        self.norm_limit = norm_limit
        self.is_discrete = is_discrete
        key_actor, key_qf1, key_qf2, key_summarizer, key_log_alpha = jax.random.split(key, 5)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.sgd(learning_rate=self.algo_config.policy_lr, momentum=0.9) if self.algo_config.use_SGD else optax.adamw(learning_rate=self.algo_config.policy_lr, eps=self.algo_config.adam_eps),
        )
        self.actor_optimizer = optimizer
        self.critic_optimizer = optimizer
        self.alpha_optimizer = optimizer
        self.summarizer_optimizer = optimizer

        self.obs_dim = int(jnp.prod(jnp.array(observation_space_shape))) if len(observation_space_shape) == 1 else observation_space_shape[-1]
        self.market_feature_dim = self.network_config.market_feature_dim if self.network_config.market_feature_dim > 0 else self.obs_dim - self.network_config.agent_feature_dim
        self.agent_feature_dim = self.network_config.agent_feature_dim
        if self.market_feature_dim + self.agent_feature_dim != self.obs_dim:
            self.market_feature_dim = self.obs_dim
            self.agent_feature_dim = 0

        self._create_models_and_states(key_actor, key_qf1, key_qf2, key_summarizer, actor_model_cls, critic_model_cls)

        self.log_alpha_state: TrainState
        self.target_entropy = 0.0
        if algo_config.autotune:
            log_alpha_params = {'log_alpha': jnp.zeros(())}
            self.log_alpha_state = TrainState.create(apply_fn=None, params=log_alpha_params, tx=self.alpha_optimizer)
            self.current_alpha = jnp.exp(self.log_alpha_state.params['log_alpha'])
        else:
            self.current_alpha = jnp.array(algo_config.alpha)

    def _init_model_with_batch_stats(self, model, key, *args, **kwargs):
        variables = model.init({'params': key, 'dropout': key}, *args, **kwargs)
        params = variables['params']
        batch_stats = variables.get('batch_stats')
        return params, batch_stats

    def _apply_model_with_batch_stats(self, model: nn.Module, variables, *args, deterministic=True, mutable=None, **kwargs):
        if mutable:
            return model.apply(variables, *args, deterministic=deterministic, mutable=mutable, **kwargs)
        else:
            return model.apply(variables, *args, deterministic=deterministic, **kwargs)

    def _create_models_and_states(self, key_actor, key_qf1, key_qf2, key_summarizer, actor_model_cls, critic_model_cls):
        raise NotImplementedError


    def select_action(self, actor_state: TrainStateWithBatchStats, summarizer_params: flax.core.FrozenDict,
                      obs: jnp.ndarray, hidden_h: jnp.ndarray, hidden_c: jnp.ndarray,
                      key: jax.Array, deterministic: bool = False):
        raise NotImplementedError

    def _update(self, states, batch, key):
        raise NotImplementedError
    
    def update_agent_then_get_action(self, obs, hidden_h, hidden_c, batch, do_update, do_target_update,  actor_state, qf1_state, qf2_state, summarizer_state, log_alpha_state, key, deterministic: bool = False):
        """Combined function to reduce CPU-TPU communication overhead"""
        raise NotImplementedError

class RSACAgentDiscrete(RSACAgentBase):
    def __init__(self,
                 action_dim: int,
                 observation_space_shape,
                 key: jax.Array,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 actor_model_cls=TradingActorDiscrete,
                 critic_model_cls=TradingCriticDiscrete):
        super().__init__(action_dim, observation_space_shape, key, network_config, algo_config, actor_model_cls, critic_model_cls, True, norm_limit=MAX_NORM)

    def _create_models_and_states(self, key_actor, key_qf1, key_qf2, key_summarizer, actor_model_cls, critic_model_cls):
        # Shared summarizer and target summarizer
        if self.network_config.use_s5_summarizer:
            print("Using S5 Summarizer.")
            self.summarizer = S5Summarizer(
                hidden_dim=self.network_config.s5_hidden_dim,
                num_layers=self.network_config.s5_num_layers,
                delta_min=self.network_config.s5_delta_min,
                delta_max=self.network_config.s5_delta_max,
            )
            summarizer_hidden_dim = self.network_config.s5_hidden_dim
        else:
            print("Using LSTM Summarizer.")
            self.summarizer = LSTMSummarizer(hidden_dim=self.network_config.lstm_hidden_dim, num_layers=self.network_config.lstm_num_layers)
            summarizer_hidden_dim = self.network_config.lstm_hidden_dim
        dummy_seq = jnp.zeros((1, 1, self.market_feature_dim))
        summarizer_params, _ = self._init_model_with_batch_stats(self.summarizer, key_summarizer, dummy_seq)
        # Load pretrained summarizer if configured
        if self.network_config.use_pretrained_summarizer_path:
            try:
                loaded = checkpoints.restore_checkpoint(self.network_config.use_pretrained_summarizer_path, target={'summarizer_params': summarizer_params})
                summarizer_params = loaded.get('summarizer_params', summarizer_params)
                print(f"Loaded pretrained summarizer from {self.network_config.use_pretrained_summarizer_path}")
            except Exception as e:
                print(f"Warning: failed to load pretrained summarizer: {e}")
        self.summarizer_state = SummarizerTrainState.create(apply_fn=self.summarizer.apply, params=summarizer_params, target_params=summarizer_params, tx=self.summarizer_optimizer)

        # Actor head
        self.actor_model = actor_model_cls(network_config=self.network_config, action_dim=self.action_dim)
        actor_params, actor_batch_stats = self._init_model_with_batch_stats(self.actor_model, key_actor, jnp.zeros((1, summarizer_hidden_dim + self.agent_feature_dim)), deterministic=True)
        self.actor_state = TrainStateWithBatchStats.create(apply_fn=self.actor_model.apply, params=actor_params, batch_stats=actor_batch_stats, tx=self.actor_optimizer)

        # Critic heads (two critics)
        self.critic_model = critic_model_cls(network_config=self.network_config, action_dim=self.action_dim)
        qf1_params, qf1_batch_stats = self._init_model_with_batch_stats(self.critic_model, key_qf1, jnp.zeros((1, summarizer_hidden_dim + self.agent_feature_dim)), deterministic=True)
        self.qf1_state = CriticTrainState.create(apply_fn=self.critic_model.apply, params=qf1_params, batch_stats=qf1_batch_stats, target_params=qf1_params, target_batch_stats=qf1_batch_stats, tx=self.critic_optimizer)
        qf2_params, qf2_batch_stats = self._init_model_with_batch_stats(self.critic_model, key_qf2, jnp.zeros((1, summarizer_hidden_dim + self.agent_feature_dim)), deterministic=True)
        self.qf2_state = CriticTrainState.create(apply_fn=self.critic_model.apply, params=qf2_params, batch_stats=qf2_batch_stats, target_params=qf2_params, target_batch_stats=qf2_batch_stats, tx=self.critic_optimizer)

        # Freeze summarizer if not training
        self.train_summarizer = bool(self.network_config.train_summarizer)

        # target entropy for discrete
        self.target_entropy = -self.algo_config.target_entropy_scale * jnp.log(1.0 / self.action_dim)

    @partial(jax_jit, static_argnums=(0, 7))
    def select_action(self, actor_state: TrainStateWithBatchStats, summarizer_params: flax.core.FrozenDict,
                      obs: jnp.ndarray, hidden_h: jnp.ndarray, hidden_c: jnp.ndarray,
                      key: jax.Array, deterministic: bool = False):
        market = obs[..., :self.market_feature_dim]
        agent_feat = obs[..., self.market_feature_dim: self.market_feature_dim + self.agent_feature_dim] if self.agent_feature_dim > 0 else jnp.zeros((obs.shape[0], 0))
        seq = market[:, None, :]
        outputs, (new_h, new_c) = self.summarizer.apply({'params': summarizer_params}, seq, (hidden_h, hidden_c))
        summary_t = outputs[:, -1, :]
        x = jnp.concatenate([summary_t, agent_feat], axis=-1)
        logits = self.actor_model.apply({'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, x, deterministic=True)
        if deterministic:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(key, logits, axis=-1)
        return actions, new_h, new_c

    def _split_obs(self, o_seq: jnp.ndarray):
        market = o_seq[..., :self.market_feature_dim]
        agent_feat = o_seq[..., self.market_feature_dim: self.market_feature_dim + self.agent_feature_dim] if self.agent_feature_dim > 0 else jnp.zeros(o_seq.shape[:-1] + (0,))
        return market, agent_feat

    @partial(jax_jit, static_argnums=(0,))
    def _update(self,
                actor_state: TrainStateWithBatchStats,
                qf1_state: CriticTrainState,
                qf2_state: CriticTrainState,
                summarizer_state: SummarizerTrainState,
                log_alpha_state: Optional[TrainState],
                batch: dict,
                key: jax.Array):
        o = batch['o']
        a = batch['a']
        r = batch['r']
        term = batch['term']
        trunc = batch['trunc']
        m = batch['m']
        B, T = a.shape[0], a.shape[1]
        market_o, agent_o = self._split_obs(o)

        summaries, _ = self.summarizer.apply({'params': summarizer_state.params}, market_o)
        summaries_t = summaries[:, 1:-1, :]
        summaries_tp1_targ, _ = self.summarizer.apply({'params': summarizer_state.target_params}, market_o)
        summaries_tp1 = summaries_tp1_targ[:, 2:, :]

        actor_input_t = jnp.concatenate([summaries_t, agent_o[:, :-1, :]], axis=-1)
        actor_input_tp1 = jnp.concatenate([summaries_tp1, agent_o[:, 1:, :]], axis=-1)

        if self.algo_config.autotune:
            current_alpha = jnp.exp(log_alpha_state.params['log_alpha'])
        else:
            current_alpha = self.current_alpha

        def critic_loss_fn(q1_params, q1_bs, q2_params, q2_bs, summarizer_params):
            next_logits = self.actor_model.apply({'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, actor_input_tp1.reshape(-1, actor_input_tp1.shape[-1]), deterministic=True)
            next_logits = next_logits.reshape(B, T, -1)
            next_probs = nn.softmax(next_logits, axis=-1)
            next_log_probs = nn.log_softmax(next_logits, axis=-1)

            q1_next = self.critic_model.apply({'params': qf1_state.target_params, 'batch_stats': qf1_state.target_batch_stats}, actor_input_tp1.reshape(-1, actor_input_tp1.shape[-1]), deterministic=True).reshape(B, T, -1)
            q2_next = self.critic_model.apply({'params': qf2_state.target_params, 'batch_stats': qf2_state.target_batch_stats}, actor_input_tp1.reshape(-1, actor_input_tp1.shape[-1]), deterministic=True).reshape(B, T, -1)
            min_q_next = jnp.minimum(q1_next, q2_next)
            v_next = jnp.sum(next_probs * (min_q_next - current_alpha * next_log_probs), axis=-1)
            target = r + (1.0 - term) * self.algo_config.gamma * v_next

            q1_all, new_q1_vars = self.critic_model.apply({'params': q1_params, 'batch_stats': q1_bs}, actor_input_t.reshape(-1, actor_input_t.shape[-1]), deterministic=False, mutable=['batch_stats'])
            q2_all, new_q2_vars = self.critic_model.apply({'params': q2_params, 'batch_stats': q2_bs}, actor_input_t.reshape(-1, actor_input_t.shape[-1]), deterministic=False, mutable=['batch_stats'])
            q1_all = q1_all.reshape(B, T, -1)
            q2_all = q2_all.reshape(B, T, -1)
            a_idx = a[..., None]
            q1_taken = jnp.take_along_axis(q1_all, a_idx, axis=-1).squeeze(-1)
            q2_taken = jnp.take_along_axis(q2_all, a_idx, axis=-1).squeeze(-1)

            mse1 = (q1_taken - target) ** 2
            mse2 = (q2_taken - target) ** 2
            mse1 = (mse1 * m).sum() / (m.sum() + 1e-8)
            mse2 = (mse2 * m).sum() / (m.sum() + 1e-8)
            loss = 0.5 * (mse1 + mse2)
            # Masked mean Q values for metrics logging
            qf1_value_mean = (q1_taken * m).sum() / (m.sum() + 1e-8)
            qf2_value_mean = (q2_taken * m).sum() / (m.sum() + 1e-8)
            return loss, (new_q1_vars, new_q2_vars, qf1_value_mean, qf2_value_mean)

        (critic_loss_val, (new_q1_vars, new_q2_vars, qf1_value_mean, qf2_value_mean)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True, argnums=(0,1,2,3,4))(qf1_state.params, qf1_state.batch_stats, qf2_state.params, qf2_state.batch_stats, summarizer_state.params)
        g_q1_params, g_q1_bs, g_q2_params, g_q2_bs, g_sum_params = critic_grads
        qf1_state_new = qf1_state.apply_gradients(grads=g_q1_params).replace(batch_stats=new_q1_vars['batch_stats'])
        qf2_state_new = qf2_state.apply_gradients(grads=g_q2_params).replace(batch_stats=new_q2_vars['batch_stats'])
        if self.train_summarizer:
            summarizer_state_new = summarizer_state.apply_gradients(grads=g_sum_params)
        else:
            summarizer_state_new = summarizer_state

        summaries_t_detached = jax.lax.stop_gradient(summaries_t)
        actor_input_t_detached = jnp.concatenate([summaries_t_detached, agent_o[:, :-1, :]], axis=-1)

        def actor_loss_fn(actor_params, actor_bs):
            outputs, new_actor_vars = self.actor_model.apply(
                {'params': actor_params, 'batch_stats': actor_bs},
                actor_input_t_detached.reshape(-1, actor_input_t_detached.shape[-1]),
                deterministic=False,
                mutable=['batch_stats']
            )
            logits = outputs.reshape(B, T, -1)
            probs = nn.softmax(logits, axis=-1)
            log_probs = nn.log_softmax(logits, axis=-1)
            q1_all = self.critic_model.apply({'params': qf1_state_new.params, 'batch_stats': qf1_state_new.batch_stats}, actor_input_t_detached.reshape(-1, actor_input_t_detached.shape[-1]), deterministic=True).reshape(B, T, -1)
            q2_all = self.critic_model.apply({'params': qf2_state_new.params, 'batch_stats': qf2_state_new.batch_stats}, actor_input_t_detached.reshape(-1, actor_input_t_detached.shape[-1]), deterministic=True).reshape(B, T, -1)
            min_q = jnp.minimum(q1_all, q2_all)
            actor_loss_t = jnp.sum(probs * (current_alpha * log_probs - min_q), axis=-1)
            actor_loss = (actor_loss_t * m).sum() / (m.sum() + 1e-8)
            entropy = (-jnp.sum((probs + 1e-8) * log_probs, axis=-1) * m).sum() / (m.sum() + 1e-8)
            return actor_loss, (entropy, new_actor_vars)

        (actor_loss_val, (entropy_val, new_actor_vars)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, actor_state.batch_stats)
        actor_state_new = actor_state.apply_gradients(grads=actor_grads).replace(batch_stats=new_actor_vars['batch_stats'])

        alpha_loss_val = jnp.array(0.0)
        log_alpha_state_to_return = log_alpha_state
        current_alpha_to_return = current_alpha
        if self.algo_config.autotune:
            logits_det = self.actor_model.apply({'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, actor_input_t_detached.reshape(-1, actor_input_t_detached.shape[-1]), deterministic=True).reshape(B, T, -1)
            log_probs_det = nn.log_softmax(logits_det, axis=-1)
            probs_det = nn.softmax(logits_det, axis=-1)
            def alpha_loss_fn(log_alpha_params):
                lp = log_probs_det
                pr = probs_det
                loss_t = pr * (-jnp.exp(log_alpha_params['log_alpha']) * (lp + self.target_entropy))
                loss = (jnp.sum(loss_t, axis=-1) * m).sum() / (m.sum() + 1e-8)
                return loss
            alpha_loss_val, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha_state.params)
            log_alpha_state_updated = log_alpha_state.apply_gradients(grads=alpha_grads)
            log_alpha_state_to_return = log_alpha_state_updated
            current_alpha_to_return = jnp.exp(log_alpha_state_updated.params['log_alpha'])

        metrics = {
            'critic_loss': critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha_loss': alpha_loss_val,
            'alpha': current_alpha_to_return,
            'entropy': entropy_val,
            'qf1_value_mean': qf1_value_mean,
            'qf2_value_mean': qf2_value_mean,
        }
        return actor_state_new, qf1_state_new, qf2_state_new, summarizer_state_new, log_alpha_state_to_return, current_alpha_to_return, metrics

    @partial(jax_jit, static_argnums=(0, 5, 6, 13, 14))
    def update_agent_then_get_action(self, obs, hidden_h, hidden_c, batches, do_update, do_target_update, actor_state: TrainStateWithBatchStats, qf1_state: CriticTrainState, qf2_state: CriticTrainState, summarizer_state: SummarizerTrainState, log_alpha_state: TrainState, key, updates_per_call: int, deterministic: bool = False):
        """Combined action selection and agent update to reduce CPU-TPU communication with multiple updates per call.

        batches: if do_update, a dict with leading dim K=updates_per_call; otherwise can be None or a single batch.
        """
        # Split rng on device to avoid host-device traffic
        key_action, key_update, new_key = jax.random.split(key, 3)
        
        # 1. First perform agent update(s) if requested
        if do_update:
            updated_actor_state, updated_qf1_state, updated_qf2_state = actor_state, qf1_state, qf2_state
            updated_summarizer_state, updated_log_alpha_state = summarizer_state, log_alpha_state
            critic_loss_sum = jnp.array(0.0)
            actor_loss_sum = jnp.array(0.0)
            alpha_loss_sum = jnp.array(0.0)
            entropy_sum = jnp.array(0.0)
            qf1_mean_sum = jnp.array(0.0)
            qf2_mean_sum = jnp.array(0.0)
            for i in range(updates_per_call):
                if batches is None:
                    break
                b = {
                    'o': batches['o'][i],
                    'a': batches['a'][i],
                    'r': batches['r'][i],
                    'term': batches['term'][i],
                    'trunc': batches['trunc'][i],
                    'm': batches['m'][i],
                }
                updated_actor_state, updated_qf1_state, updated_qf2_state, updated_summarizer_state, updated_log_alpha_state, current_alpha, metric_i = self._update(
                    updated_actor_state, updated_qf1_state, updated_qf2_state, updated_summarizer_state, updated_log_alpha_state, b, key_update
                )
                critic_loss_sum += metric_i['critic_loss']
                actor_loss_sum += metric_i['actor_loss']
                alpha_loss_sum += metric_i['alpha_loss']
                entropy_sum += metric_i['entropy']
                qf1_mean_sum += metric_i['qf1_value_mean']
                qf2_mean_sum += metric_i['qf2_value_mean']
            kf = jnp.maximum(1, updates_per_call)
            metrics = {
                'critic_loss': critic_loss_sum / kf,
                'actor_loss': actor_loss_sum / kf,
                'alpha_loss': alpha_loss_sum / kf,
                'alpha': current_alpha,
                'entropy': entropy_sum / kf,
                'qf1_value_mean': qf1_mean_sum / kf,
                'qf2_value_mean': qf2_mean_sum / kf,
            }
        else:
            updated_actor_state, updated_qf1_state, updated_qf2_state, updated_summarizer_state, updated_log_alpha_state = actor_state, qf1_state, qf2_state, summarizer_state, log_alpha_state
            current_alpha = jnp.exp(log_alpha_state.params['log_alpha']) if self.algo_config.autotune and log_alpha_state else jnp.array(self.algo_config.alpha)
            metrics = {}
        
        # Apply target network updates if requested
        if do_update and do_target_update:
            tau = self.algo_config.tau
            final_qf1_state = updated_qf1_state.replace(
                target_params=optax.incremental_update(updated_qf1_state.params, updated_qf1_state.target_params, tau),
                target_batch_stats=optax.incremental_update(updated_qf1_state.batch_stats, updated_qf1_state.target_batch_stats, tau) if updated_qf1_state.batch_stats is not None else updated_qf1_state.target_batch_stats
            )
            final_qf2_state = updated_qf2_state.replace(
                target_params=optax.incremental_update(updated_qf2_state.params, updated_qf2_state.target_params, tau),
                target_batch_stats=optax.incremental_update(updated_qf2_state.batch_stats, updated_qf2_state.target_batch_stats, tau) if updated_qf2_state.batch_stats is not None else updated_qf2_state.target_batch_stats
            )
            new_target = optax.incremental_update(updated_summarizer_state.params, updated_summarizer_state.target_params, tau)
            final_summarizer_state = updated_summarizer_state.replace(target_params=new_target)
        else:
            final_qf1_state, final_qf2_state, final_summarizer_state = updated_qf1_state, updated_qf2_state, updated_summarizer_state
        
        # 2. Then perform action selection using potentially updated states
        market = obs[..., :self.market_feature_dim]
        agent_feat = obs[..., self.market_feature_dim: self.market_feature_dim + self.agent_feature_dim] if self.agent_feature_dim > 0 else jnp.zeros((obs.shape[0], 0))
        seq = market[:, None, :]
        outputs, (new_h, new_c) = self.summarizer.apply({'params': final_summarizer_state.params}, seq, (hidden_h, hidden_c))
        summary_t = outputs[:, -1, :]
        x = jnp.concatenate([summary_t, agent_feat], axis=-1)
        logits = self.actor_model.apply({'params': updated_actor_state.params, 'batch_stats': updated_actor_state.batch_stats}, x, deterministic=True)
        if deterministic:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(key_action, logits, axis=-1)
        
        return actions, new_h, new_c, updated_actor_state, final_qf1_state, final_qf2_state, final_summarizer_state, updated_log_alpha_state, metrics, new_key

class RSACAgentContinuous(RSACAgentBase):
    def __init__(self,
                 action_dim: int,
                 observation_space_shape,
                 key: jax.Array,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 actor_model_cls=TradingActorContinuous,
                 critic_model_cls=TradingCriticContinuous):
        super().__init__(action_dim, observation_space_shape, key, network_config, algo_config, actor_model_cls, critic_model_cls, False, norm_limit=MAX_NORM)

    def _create_models_and_states(self, key_actor, key_qf1, key_qf2, key_summarizer, actor_model_cls, critic_model_cls):
        if self.network_config.use_s5_summarizer:
            print("Using S5 Summarizer.")
            self.summarizer = S5Summarizer(
                hidden_dim=self.network_config.s5_hidden_dim,
                num_layers=self.network_config.s5_num_layers,
                delta_min=self.network_config.s5_delta_min,
                delta_max=self.network_config.s5_delta_max,
            )
            summarizer_hidden_dim = self.network_config.s5_hidden_dim
        else:
            print("Using LSTM Summarizer.")
            self.summarizer = LSTMSummarizer(hidden_dim=self.network_config.lstm_hidden_dim, num_layers=self.network_config.lstm_num_layers)
            summarizer_hidden_dim = self.network_config.lstm_hidden_dim
        dummy_seq = jnp.zeros((1, 1, self.market_feature_dim))
        summarizer_params, _ = self._init_model_with_batch_stats(self.summarizer, key_summarizer, dummy_seq)
        if self.network_config.use_pretrained_summarizer_path:
            try:
                loaded = checkpoints.restore_checkpoint(self.network_config.use_pretrained_summarizer_path, target={'summarizer_params': summarizer_params})
                summarizer_params = loaded.get('summarizer_params', summarizer_params)
                print(f"Loaded pretrained summarizer from {self.network_config.use_pretrained_summarizer_path}")
            except Exception as e:
                print(f"Warning: failed to load pretrained summarizer: {e}")
        self.summarizer_state = SummarizerTrainState.create(apply_fn=self.summarizer.apply, params=summarizer_params, target_params=summarizer_params, tx=self.summarizer_optimizer)

        # Actor head
        self.actor_model = actor_model_cls(network_config=self.network_config, action_dim=self.action_dim)
        actor_params, actor_batch_stats = self._init_model_with_batch_stats(self.actor_model, key_actor, jnp.zeros((1, summarizer_hidden_dim + self.agent_feature_dim)), deterministic=True)
        self.actor_state = TrainStateWithBatchStats.create(apply_fn=self.actor_model.apply, params=actor_params, batch_stats=actor_batch_stats, tx=self.actor_optimizer)

        # Critic heads (two critics)
        self.critic_model = critic_model_cls(network_config=self.network_config)
        dummy_action = jnp.zeros((1, self.action_dim))
        qf1_params, qf1_batch_stats = self._init_model_with_batch_stats(self.critic_model, key_qf1, jnp.zeros((1, summarizer_hidden_dim + self.agent_feature_dim)), dummy_action, deterministic=True)
        self.qf1_state = CriticTrainState.create(apply_fn=self.critic_model.apply, params=qf1_params, batch_stats=qf1_batch_stats, target_params=qf1_params, target_batch_stats=qf1_batch_stats, tx=self.critic_optimizer)
        qf2_params, qf2_batch_stats = self._init_model_with_batch_stats(self.critic_model, key_qf2, jnp.zeros((1, summarizer_hidden_dim + self.agent_feature_dim)), dummy_action, deterministic=True)
        self.qf2_state = CriticTrainState.create(apply_fn=self.critic_model.apply, params=qf2_params, batch_stats=qf2_batch_stats, target_params=qf2_params, target_batch_stats=qf2_batch_stats, tx=self.critic_optimizer)

        if self.network_config.use_pretrained_summarizer_path:
            pass
        self.train_summarizer = bool(self.network_config.train_summarizer)
        self.target_entropy = -float(self.action_dim)

    @partial(jax_jit, static_argnums=(0, 7))
    def select_action(self, actor_state: TrainStateWithBatchStats, summarizer_params: flax.core.FrozenDict,
                      obs: jnp.ndarray, hidden_h: jnp.ndarray, hidden_c: jnp.ndarray,
                      key: jax.Array, deterministic: bool = False):
        market = obs[..., :self.market_feature_dim]
        agent_feat = obs[..., self.market_feature_dim: self.market_feature_dim + self.agent_feature_dim] if self.agent_feature_dim > 0 else jnp.zeros((obs.shape[0], 0))
        seq = market[:, None, :]
        outputs, (new_h, new_c) = self.summarizer.apply({'params': summarizer_params}, seq, (hidden_h, hidden_c))
        summary_t = outputs[:, -1, :]
        x = jnp.concatenate([summary_t, agent_feat], axis=-1)
        mean, log_std = self.actor_model.apply({'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, x, deterministic=True)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        if deterministic:
            action = dist.mean()
        else:
            action = dist.sample(seed=key)
        squashed = jnp.tanh(action)
        return squashed, new_h, new_c

    def _split_obs(self, o_seq: jnp.ndarray):
        market = o_seq[..., :self.market_feature_dim]
        agent_feat = o_seq[..., self.market_feature_dim: self.market_feature_dim + self.agent_feature_dim] if self.agent_feature_dim > 0 else jnp.zeros(o_seq.shape[:-1] + (0,))
        return market, agent_feat

    @partial(jax_jit, static_argnums=(0,))
    def _update(self,
                actor_state: TrainStateWithBatchStats,
                qf1_state: CriticTrainState,
                qf2_state: CriticTrainState,
                summarizer_state: SummarizerTrainState,
                log_alpha_state: Optional[TrainState],
                batch: dict,
                key: jax.Array):
        o = batch['o']
        a = batch['a']  # [B, T, A]
        r = batch['r']
        term = batch['term']
        trunc = batch['trunc']
        m = batch['m']
        B, T = r.shape
        market_o, agent_o = self._split_obs(o)

        summaries, _ = self.summarizer.apply({'params': summarizer_state.params}, market_o)
        s_t = summaries[:, 1:-1, :]
        s_tp1_targ, _ = self.summarizer.apply({'params': summarizer_state.target_params}, market_o)
        s_tp1 = s_tp1_targ[:, 2:, :]

        x_t = jnp.concatenate([s_t, agent_o[:, :-1, :]], axis=-1)
        x_tp1 = jnp.concatenate([s_tp1, agent_o[:, 1:, :]], axis=-1)

        if self.algo_config.autotune:
            current_alpha = jnp.exp(log_alpha_state.params['log_alpha'])
        else:
            current_alpha = self.current_alpha

        def critic_loss_fn(q1_params, q1_bs, q2_params, q2_bs, summarizer_params):
            mean_tp1, log_std_tp1 = self.actor_model.apply({'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, x_tp1.reshape(-1, x_tp1.shape[-1]), deterministic=True)
            mean_tp1 = mean_tp1.reshape(B, T, -1)
            log_std_tp1 = log_std_tp1.reshape(B, T, -1)
            dist_tp1 = tfd.MultivariateNormalDiag(loc=mean_tp1, scale_diag=jnp.exp(log_std_tp1))
            u = dist_tp1.sample(seed=key)
            squashed_tp1 = jnp.tanh(u)
            log_prob = dist_tp1.log_prob(u)
            log_prob -= jnp.sum(jnp.log(1 - jnp.tanh(u) ** 2 + 1e-6), axis=-1)

            q1_next = self.critic_model.apply({'params': qf1_state.target_params, 'batch_stats': qf1_state.target_batch_stats}, x_tp1.reshape(-1, x_tp1.shape[-1]), squashed_tp1.reshape(-1, squashed_tp1.shape[-1]), deterministic=True).reshape(B, T)
            q2_next = self.critic_model.apply({'params': qf2_state.target_params, 'batch_stats': qf2_state.target_batch_stats}, x_tp1.reshape(-1, x_tp1.shape[-1]), squashed_tp1.reshape(-1, squashed_tp1.shape[-1]), deterministic=True).reshape(B, T)
            min_q_next = jnp.minimum(q1_next, q2_next)
            target = r + (1.0 - term) * self.algo_config.gamma * (min_q_next - current_alpha * log_prob)

            q1_cur, new_q1_vars = self.critic_model.apply({'params': q1_params, 'batch_stats': q1_bs}, x_t.reshape(-1, x_t.shape[-1]), a.reshape(-1, a.shape[-1]), deterministic=False, mutable=['batch_stats'])
            q2_cur, new_q2_vars = self.critic_model.apply({'params': q2_params, 'batch_stats': q2_bs}, x_t.reshape(-1, x_t.shape[-1]), a.reshape(-1, a.shape[-1]), deterministic=False, mutable=['batch_stats'])
            q1_cur = q1_cur.reshape(B, T)
            q2_cur = q2_cur.reshape(B, T)

            mse1 = (q1_cur - target) ** 2
            mse2 = (q2_cur - target) ** 2
            mse1 = (mse1 * m).sum() / (m.sum() + 1e-8)
            mse2 = (mse2 * m).sum() / (m.sum() + 1e-8)
            loss = 0.5 * (mse1 + mse2)
            # Masked mean Q values for metrics logging
            qf1_value_mean = (q1_cur * m).sum() / (m.sum() + 1e-8)
            qf2_value_mean = (q2_cur * m).sum() / (m.sum() + 1e-8)
            return loss, (new_q1_vars, new_q2_vars, qf1_value_mean, qf2_value_mean)

        (critic_loss_val, (new_q1_vars, new_q2_vars, qf1_value_mean, qf2_value_mean)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True, argnums=(0,1,2,3,4))(qf1_state.params, qf1_state.batch_stats, qf2_state.params, qf2_state.batch_stats, summarizer_state.params)
        g_q1_params, g_q1_bs, g_q2_params, g_q2_bs, g_sum_params = critic_grads
        qf1_state_new = qf1_state.apply_gradients(grads=g_q1_params).replace(batch_stats=new_q1_vars['batch_stats'])
        qf2_state_new = qf2_state.apply_gradients(grads=g_q2_params).replace(batch_stats=new_q2_vars['batch_stats'])
        if self.train_summarizer:
            summarizer_state_new = summarizer_state.apply_gradients(grads=g_sum_params)
        else:
            summarizer_state_new = summarizer_state

        s_t_det = jax.lax.stop_gradient(s_t)
        x_t_det = jnp.concatenate([s_t_det, agent_o[:, :-1, :]], axis=-1)
        def actor_loss_fn(actor_params, actor_bs):
            (mean, log_std), new_actor_vars = self.actor_model.apply(
                {'params': actor_params, 'batch_stats': actor_bs},
                x_t_det.reshape(-1, x_t_det.shape[-1]),
                deterministic=False,
                mutable=['batch_stats']
            )
            mean = mean.reshape(B, T, -1)
            log_std = log_std.reshape(B, T, -1)
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
            u = dist.sample(seed=key)
            squashed = jnp.tanh(u)
            log_prob = dist.log_prob(u)
            log_prob -= jnp.sum(jnp.log(1 - jnp.tanh(u) ** 2 + 1e-6), axis=-1)

            q1_pi = self.critic_model.apply({'params': qf1_state_new.params, 'batch_stats': qf1_state_new.batch_stats}, x_t_det.reshape(-1, x_t_det.shape[-1]), squashed.reshape(-1, squashed.shape[-1]), deterministic=True).reshape(B, T)
            q2_pi = self.critic_model.apply({'params': qf2_state_new.params, 'batch_stats': qf2_state_new.batch_stats}, x_t_det.reshape(-1, x_t_det.shape[-1]), squashed.reshape(-1, squashed.shape[-1]), deterministic=True).reshape(B, T)
            min_q = jnp.minimum(q1_pi, q2_pi)
            loss_t = (current_alpha * log_prob - min_q)
            loss = (loss_t * m).sum() / (m.sum() + 1e-8)
            entropy = (-(log_prob) * m).sum() / (m.sum() + 1e-8)
            return loss, (entropy, new_actor_vars)

        (actor_loss_val, (entropy_val, new_actor_vars)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, actor_state.batch_stats)
        actor_state_new = actor_state.apply_gradients(grads=actor_grads).replace(batch_stats=new_actor_vars['batch_stats'])

        alpha_loss_val = jnp.array(0.0)
        log_alpha_state_to_return = log_alpha_state
        current_alpha_to_return = current_alpha
        if self.algo_config.autotune:
            mean_det, log_std_det = self.actor_model.apply({'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, x_t_det.reshape(-1, x_t_det.shape[-1]), deterministic=True)
            mean_det = mean_det.reshape(B, T, -1)
            log_std_det = log_std_det.reshape(B, T, -1)
            dist_det = tfd.MultivariateNormalDiag(loc=mean_det, scale_diag=jnp.exp(log_std_det))
            u_det = dist_det.sample(seed=key)
            log_prob_det = dist_det.log_prob(u_det)
            log_prob_det -= jnp.sum(jnp.log(1 - jnp.tanh(u_det) ** 2 + 1e-6), axis=-1)
            def alpha_loss_fn(log_alpha_params):
                return ((-jnp.exp(log_alpha_params['log_alpha']) * (log_prob_det + self.target_entropy)) * m).sum() / (m.sum() + 1e-8)
            alpha_loss_val, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha_state.params)
            log_alpha_state_updated = log_alpha_state.apply_gradients(grads=alpha_grads)
            log_alpha_state_to_return = log_alpha_state_updated
            current_alpha_to_return = jnp.exp(log_alpha_state_updated.params['log_alpha'])

        metrics = {
            'critic_loss': critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha_loss': alpha_loss_val,
            'alpha': current_alpha_to_return,
            'entropy': entropy_val,
            'qf1_value_mean': qf1_value_mean,
            'qf2_value_mean': qf2_value_mean,
        }
        return actor_state_new, qf1_state_new, qf2_state_new, summarizer_state_new, log_alpha_state_to_return, current_alpha_to_return, metrics

    @partial(jax_jit, static_argnums=(0, 5, 6, 13, 14))
    def update_agent_then_get_action(self, obs, hidden_h, hidden_c, batches, do_update, do_target_update, actor_state: TrainStateWithBatchStats, qf1_state: CriticTrainState, qf2_state: CriticTrainState, summarizer_state: SummarizerTrainState, log_alpha_state: TrainState, key, updates_per_call: int, deterministic: bool = False):
        """Combined action selection and agent update to reduce CPU-TPU communication with multiple updates per call.

        batches: if do_update, a dict with leading dim K=updates_per_call; otherwise can be None or a single batch.
        """
        # Split rng on device to avoid host-device traffic
        key_action, key_update, new_key = jax.random.split(key, 3)
        
        # 1. First perform agent update(s) if requested
        if do_update:
            updated_actor_state, updated_qf1_state, updated_qf2_state = actor_state, qf1_state, qf2_state
            updated_summarizer_state, updated_log_alpha_state = summarizer_state, log_alpha_state
            # metrics accumulator
            critic_loss_sum = jnp.array(0.0)
            actor_loss_sum = jnp.array(0.0)
            alpha_loss_sum = jnp.array(0.0)
            entropy_sum = jnp.array(0.0)
            qf1_mean_sum = jnp.array(0.0)
            qf2_mean_sum = jnp.array(0.0)
            # Iterate K updates; batches has shape [K, ...]
            for i in range(updates_per_call):
                if batches is None:
                    break
                # Per-update batch slicing along leading K
                b = {
                    'o': batches['o'][i],
                    'a': batches['a'][i],
                    'r': batches['r'][i],
                    'term': batches['term'][i],
                    'trunc': batches['trunc'][i],
                    'm': batches['m'][i],
                }
                updated_actor_state, updated_qf1_state, updated_qf2_state, updated_summarizer_state, updated_log_alpha_state, current_alpha, metric_i = self._update(
                    updated_actor_state, updated_qf1_state, updated_qf2_state, updated_summarizer_state, updated_log_alpha_state, b, key_update
                )
                critic_loss_sum += metric_i['critic_loss']
                actor_loss_sum += metric_i['actor_loss']
                alpha_loss_sum += metric_i['alpha_loss']
                entropy_sum += metric_i['entropy']
                qf1_mean_sum += metric_i['qf1_value_mean']
                qf2_mean_sum += metric_i['qf2_value_mean']
            kf = jnp.maximum(1, updates_per_call)
            metrics = {
                'critic_loss': critic_loss_sum / kf,
                'actor_loss': actor_loss_sum / kf,
                'alpha_loss': alpha_loss_sum / kf,
                'alpha': current_alpha,
                'entropy': entropy_sum / kf,
                'qf1_value_mean': qf1_mean_sum / kf,
                'qf2_value_mean': qf2_mean_sum / kf,
            }
        else:
            updated_actor_state, updated_qf1_state, updated_qf2_state, updated_summarizer_state, updated_log_alpha_state = actor_state, qf1_state, qf2_state, summarizer_state, log_alpha_state
            current_alpha = jnp.exp(log_alpha_state.params['log_alpha']) if self.algo_config.autotune and log_alpha_state else jnp.array(self.algo_config.alpha)
            metrics = {}
        
        # Apply target network updates if requested
        if do_update and do_target_update:
            tau = self.algo_config.tau
            final_qf1_state = updated_qf1_state.replace(
                target_params=optax.incremental_update(updated_qf1_state.params, updated_qf1_state.target_params, tau),
                target_batch_stats=optax.incremental_update(updated_qf1_state.batch_stats, updated_qf1_state.target_batch_stats, tau) if updated_qf1_state.batch_stats is not None else updated_qf1_state.target_batch_stats
            )
            final_qf2_state = updated_qf2_state.replace(
                target_params=optax.incremental_update(updated_qf2_state.params, updated_qf2_state.target_params, tau),
                target_batch_stats=optax.incremental_update(updated_qf2_state.batch_stats, updated_qf2_state.target_batch_stats, tau) if updated_qf2_state.batch_stats is not None else updated_qf2_state.target_batch_stats
            )
            new_target = optax.incremental_update(updated_summarizer_state.params, updated_summarizer_state.target_params, tau)
            final_summarizer_state = updated_summarizer_state.replace(target_params=new_target)
        else:
            final_qf1_state, final_qf2_state, final_summarizer_state = updated_qf1_state, updated_qf2_state, updated_summarizer_state
        
        # 2. Then perform action selection using potentially updated states
        market = obs[..., :self.market_feature_dim]
        agent_feat = obs[..., self.market_feature_dim: self.market_feature_dim + self.agent_feature_dim] if self.agent_feature_dim > 0 else jnp.zeros((obs.shape[0], 0))
        seq = market[:, None, :]
        outputs, (new_h, new_c) = self.summarizer.apply({'params': final_summarizer_state.params}, seq, (hidden_h, hidden_c))
        summary_t = outputs[:, -1, :]
        x = jnp.concatenate([summary_t, agent_feat], axis=-1)
        mean, log_std = self.actor_model.apply({'params': updated_actor_state.params, 'batch_stats': updated_actor_state.batch_stats}, x, deterministic=True)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        if deterministic:
            action = dist.mean()
        else:
            action = dist.sample(seed=key_action)
        squashed = jnp.tanh(action)
        
        return squashed, new_h, new_c, updated_actor_state, final_qf1_state, final_qf2_state, final_summarizer_state, updated_log_alpha_state, metrics, new_key

RSACAgent = Union[RSACAgentDiscrete, RSACAgentContinuous]