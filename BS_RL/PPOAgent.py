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
from .config import AlgoConfig, NetworkConfig
from .networks import ActorCritic
from flax.core import FrozenDict

class TrainStateWithBatchStats(TrainState):
    batch_stats: Optional[FrozenDict] = None

class PPOAgent:
    def __init__(self,
                 action_dim: int,
                 observation_space_shape,
                 key: jax.Array,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 is_continuous: bool):
        
        self.network_config = network_config
        self.algo_config = algo_config
        self.action_dim = action_dim
        self.is_continuous = is_continuous

        self.ac_model = ActorCritic(
            action_dim=self.action_dim,
            network_config=self.network_config,
            is_continuous=self.is_continuous
        )

        key_ac, key_summarizer_norm = jax.random.split(key)
        
        # Init train state
        dummy_obs = jnp.zeros((1,) + observation_space_shape)
        dummy_hidden_state = self.ac_model.get_initial_state(batch_size=1)
        
        variables = self.ac_model.init({'params': key_ac, 'dropout': key_ac}, dummy_hidden_state, dummy_obs, deterministic=True)
        params = variables['params']
        batch_stats = variables.get('batch_stats')

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.algo_config.max_grad_norm),
            optax.sgd(learning_rate=self.algo_config.learning_rate, momentum=0.9) if self.algo_config.use_SGD else optax.adam(learning_rate=self.algo_config.learning_rate, eps=self.algo_config.adam_eps),
        )

        self.train_state = TrainStateWithBatchStats.create(
            apply_fn=self.ac_model.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer
        )
    
    @partial(jax_jit, static_argnums=(0,))
    def get_action_and_value(self,
                             train_state: TrainStateWithBatchStats,
                             hidden_state: Tuple[jnp.ndarray, jnp.ndarray],
                             obs: jnp.ndarray,
                             key: jax.Array
                             ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        
        new_hidden_state, pi, value = train_state.apply_fn(
            {'params': train_state.params, 'batch_stats': train_state.batch_stats},
            hidden_state,
            obs,
            deterministic=True
        )
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value, new_hidden_state

    @partial(jax_jit, static_argnums=(0,))
    def get_value(self,
                  train_state: TrainStateWithBatchStats,
                  hidden_state: Tuple[jnp.ndarray, jnp.ndarray],
                  obs: jnp.ndarray
                  ) -> jnp.ndarray:
        _, _, value = train_state.apply_fn(
            {'params': train_state.params, 'batch_stats': train_state.batch_stats},
            hidden_state,
            obs,
            deterministic=True
        )
        return value

    @partial(jax_jit, static_argnums=(0,))
    def update(self,
               train_state: TrainStateWithBatchStats,
               initial_hidden_state: Tuple[jnp.ndarray, jnp.ndarray],
               batch: dict,
               key: jax.Array):

        def ppo_loss(params, batch_stats, obs, actions, log_probs_old, advantages, returns, values_old):
            
            # For recurrent PPO, we need to reshape data to sequence format
            # obs shape: [batch_size, obs_dim] -> [batch_size // num_steps, num_steps, obs_dim]
            batch_size = obs.shape[0]
            num_envs = batch_size // self.algo_config.num_steps
            
            # Reshape to sequence format for recurrent model
            obs_seq = obs.reshape(num_envs, self.algo_config.num_steps, -1)
            actions_seq = actions.reshape(num_envs, self.algo_config.num_steps, -1) if actions.ndim > 1 else actions.reshape(num_envs, self.algo_config.num_steps)
            
            _, pi, values_pred = self.ac_model.apply(
                {'params': params, 'batch_stats': batch_stats},
                initial_hidden_state,
                obs_seq,  # Now properly shaped as [B, T, D]
                deterministic=True
            )
            
            # Flatten back for loss computation
            log_probs_new = pi.log_prob(actions_seq).reshape(-1)
            entropy = pi.entropy().mean()
            values_pred = values_pred.reshape(-1)

            # Policy loss
            logratio = log_probs_new - log_probs_old
            ratio = jnp.exp(logratio)
            
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            pg_loss1 = -adv_norm * ratio
            pg_loss2 = -adv_norm * jnp.clip(ratio, 1 - self.algo_config.clip_coef, 1 + self.algo_config.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss_unclipped = (values_pred - returns) ** 2
            v_clipped = values_old + jnp.clip(values_pred - values_old, -self.algo_config.clip_coef, self.algo_config.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

            total_loss = pg_loss - self.algo_config.ent_coef * entropy + self.algo_config.vf_coef * v_loss
            
            return total_loss, (pg_loss, v_loss, entropy)

        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        
        (total_loss, (pg_loss, v_loss, entropy)), grads = grad_fn(
            train_state.params,
            train_state.batch_stats,
            batch['obs'],
            batch['actions'],
            batch['log_probs'],
            batch['advantages'],
            batch['returns'],
            batch['values']
        )
        
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = {
            'total_loss': total_loss,
            'policy_loss': pg_loss,
            'value_loss': v_loss,
            'entropy': entropy,
        }
        return train_state, metrics