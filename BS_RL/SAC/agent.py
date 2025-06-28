import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from functools import partial
from typing import Union
import tensorflow_probability.substrates.jax.distributions as tfd

from .config import AlgoConfig, NetworkConfig
from .networks import DiscreteTradingActor, DiscreteTradingCritic, ContinuousTradingActor, ContinuousTradingCritic

class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict

class SACAgentDiscrete:
    def __init__(self,
                 action_dim: int,
                 observation_space_shape,
                 key: jax.random.PRNGKey,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 actor_model_cls,
                 critic_model_cls
                 ):

        self.algo_config = algo_config
        self.action_dim = action_dim
        self.network_config = network_config
        self.norm_limit = 0.6 # 限制梯度范数<0.6
        key_actor, key_qf1, key_qf2, key_log_alpha = jax.random.split(key, 4)
        self.key_buffer = key

        if len(observation_space_shape) == 3:
            dummy_obs = jnp.zeros((1, *observation_space_shape), dtype=jnp.float32)
        else:
            dummy_obs = jnp.zeros((1, *observation_space_shape), dtype=jnp.float32)

        self.actor_model = actor_model_cls(network_config=self.network_config, action_dim=self.action_dim)
        actor_params = self.actor_model.init({'params': key_actor, 'dropout': key_actor}, dummy_obs, deterministic=True)['params']
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.adamw(learning_rate=algo_config.policy_lr, eps=algo_config.adam_eps),
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            tx=actor_optimizer
        )

        self.critic_model = critic_model_cls(network_config=self.network_config, action_dim=self.action_dim)
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.adamw(learning_rate=algo_config.q_lr, eps=algo_config.adam_eps),
        )

        qf1_params = self.critic_model.init({'params': key_qf1, 'dropout': key_qf1}, dummy_obs, deterministic=True)['params']
        self.qf1_state = CriticTrainState.create(
            apply_fn=self.critic_model.apply,
            params=qf1_params,
            target_params=qf1_params,
            tx=critic_optimizer
        )

        qf2_params = self.critic_model.init({'params': key_qf2, 'dropout': key_qf2}, dummy_obs, deterministic=True)['params']
        self.qf2_state = CriticTrainState.create(
            apply_fn=self.critic_model.apply,
            params=qf2_params,
            target_params=qf2_params,
            tx=critic_optimizer
        )

        if algo_config.autotune:
            self.target_entropy = -algo_config.target_entropy_scale * jnp.log(1.0 / action_dim)
            log_alpha_params = {'log_alpha': jnp.zeros((), dtype=jnp.float32)}
            self.log_alpha_state = TrainState.create(
                apply_fn=None,
                params=log_alpha_params,
                tx=critic_optimizer
            )
            self.current_alpha = jnp.exp(self.log_alpha_state.params['log_alpha'])
        else:
            self.current_alpha = jnp.array(algo_config.alpha, dtype=jnp.float32)
            self.log_alpha_state = None
            self.target_entropy = 0.0

    @partial(jax.jit, static_argnums=(0, 4))
    def select_action(self, actor_params: flax.core.FrozenDict, obs: jnp.ndarray, key: jax.random.PRNGKey, deterministic: bool = False):
        key_dropout, key_sample = jax.random.split(key)
        logits = self.actor_model.apply(
            {'params': actor_params}, obs,
            deterministic=deterministic,
            rngs={'dropout': key_dropout}
        )
        if deterministic:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(key_sample, logits, axis=-1)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(self,
                       actor_state: TrainState,
                       qf1_state: CriticTrainState,
                       qf2_state: CriticTrainState,
                       log_alpha_input: Union[flax.core.FrozenDict, jnp.ndarray],
                       data: dict,
                       key: jax.random.PRNGKey):

        if self.algo_config.autotune:
            current_alpha = jnp.exp(log_alpha_input['log_alpha'])
        else:
            current_alpha = log_alpha_input

        key_next_logits, key_q_target = jax.random.split(key, 2)
        next_logits = self.actor_model.apply(
            {'params': actor_state.params}, data['next_observations'],
            deterministic=False, # Use dropout for stochasticity in target
            rngs={'dropout': key_next_logits}
        )
        next_action_probs = nn.softmax(next_logits, axis=-1)
        next_action_log_probs = nn.log_softmax(next_logits, axis=-1)

        qf1_next_target_values = self.critic_model.apply(
            {'params': qf1_state.target_params}, data['next_observations'],
            deterministic=True, rngs={'dropout': key_q_target}
        )
        qf2_next_target_values = self.critic_model.apply(
            {'params': qf2_state.target_params}, data['next_observations'],
            deterministic=True, rngs={'dropout': key_q_target}
        )
        min_qf_next_target = jnp.minimum(qf1_next_target_values, qf2_next_target_values)
        
        next_q_value_components = next_action_probs * (min_qf_next_target - current_alpha * next_action_log_probs)
        next_q_value = jnp.sum(next_q_value_components, axis=1)
        
        target_q_values = data['rewards'] + (1.0 - data['dones']) * self.algo_config.gamma * next_q_value
        target_q_values = jax.lax.stop_gradient(target_q_values)

        def qf1_loss_fn(params):
            qf1_all_actions = self.critic_model.apply(
                {'params': params}, data['observations'],
                deterministic=True, rngs={'dropout': key}
            )
            qf1_taken_action = jnp.take_along_axis(qf1_all_actions, data['actions'], axis=1).squeeze(-1)
            loss = ((qf1_taken_action - target_q_values) ** 2).mean()
            return loss, qf1_taken_action.mean()
        (qf1_loss_val, qf1_values_mean), qf1_grads = jax.value_and_grad(qf1_loss_fn, has_aux=True)(qf1_state.params)
        qf1_grads = jax.lax.pmean(qf1_grads, axis_name='batch')
        qf1_loss_val = jax.lax.pmean(qf1_loss_val, axis_name='batch')
        qf1_values_mean = jax.lax.pmean(qf1_values_mean, axis_name='batch')
        qf1_state_new = qf1_state.apply_gradients(grads=qf1_grads)
        
        def qf2_loss_fn(params):
            qf2_all_actions = self.critic_model.apply(
                {'params': params}, data['observations'],
                deterministic=True, rngs={'dropout': key}
            )
            qf2_taken_action = jnp.take_along_axis(qf2_all_actions, data['actions'], axis=1).squeeze(-1)
            loss = ((qf2_taken_action - target_q_values) ** 2).mean()
            return loss, qf2_taken_action.mean()
        (qf2_loss_val, qf2_values_mean), qf2_grads = jax.value_and_grad(qf2_loss_fn, has_aux=True)(qf2_state.params)
        qf2_grads = jax.lax.pmean(qf2_grads, axis_name='batch')
        qf2_loss_val = jax.lax.pmean(qf2_loss_val, axis_name='batch')
        qf2_values_mean = jax.lax.pmean(qf2_values_mean, axis_name='batch')
        qf2_state_new = qf2_state.apply_gradients(grads=qf2_grads)

        critic_loss = (qf1_loss_val + qf2_loss_val) / 2.0
        critic_loss = jax.lax.pmean(critic_loss, axis_name='batch')
        
        return qf1_state_new, qf2_state_new, critic_loss, \
               {'qf1_loss': qf1_loss_val, 'qf2_loss': qf2_loss_val,
                'qf1_values': qf1_values_mean, 'qf2_values': qf2_values_mean}

    @partial(jax.jit, static_argnums=(0,))
    def _update_actor_and_alpha(self,
                                actor_state: TrainState,
                                qf1_state: CriticTrainState,
                                qf2_state: CriticTrainState,
                                log_alpha_input: Union[TrainState, jnp.ndarray],
                                data: dict,
                                key: jax.random.PRNGKey):
        
        key_actor, key_alpha = jax.random.split(key)
        
        if self.algo_config.autotune:
            actor_effective_alpha = jnp.exp(log_alpha_input.params['log_alpha'])
        else:
            actor_effective_alpha = log_alpha_input

        def actor_loss_fn(actor_params):
            logits = self.actor_model.apply(
                {'params': actor_params}, data['observations'],
                deterministic=False, rngs={'dropout': key_actor}
            )
            action_probs = nn.softmax(logits, axis=-1)
            action_log_probs = nn.log_softmax(logits, axis=-1)

            qf1_all_actions = self.critic_model.apply(
                {'params': qf1_state.params}, data['observations'],
                deterministic=True, rngs={'dropout': key_actor}
            )
            qf2_all_actions = self.critic_model.apply(
                {'params': qf2_state.params}, data['observations'],
                deterministic=True, rngs={'dropout': key_actor}
            )
            min_qf_values = jnp.minimum(qf1_all_actions, qf2_all_actions)
            min_qf_values = jax.lax.stop_gradient(min_qf_values)

            actor_loss_components = action_probs * (actor_effective_alpha * action_log_probs - min_qf_values)
            loss = jnp.sum(actor_loss_components, axis=1).mean()
            
            # Add a small epsilon to action_probs for numerical stability.
            entropy = -jnp.sum((action_probs + 1e-8) * action_log_probs, axis=1).mean()
            return loss, entropy

        (actor_loss_val, entropy_val), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_grads = jax.lax.pmean(actor_grads, axis_name='batch')
        actor_loss_val = jax.lax.pmean(actor_loss_val, axis_name='batch')
        entropy_val = jax.lax.pmean(entropy_val, axis_name='batch')
        actor_state_new = actor_state.apply_gradients(grads=actor_grads)

        alpha_loss_val = 0.0
        log_alpha_state_to_return = log_alpha_input
        current_alpha_to_return = actor_effective_alpha

        if self.algo_config.autotune:
            def alpha_loss_fn(log_alpha_params_dict):
                logits_old_actor = self.actor_model.apply(
                    {'params': actor_state.params}, data['observations'],
                    deterministic=True, rngs={'dropout': key_alpha}
                )
                action_log_probs_old_actor = nn.log_softmax(logits_old_actor, axis=-1)
                detached_log_probs = jax.lax.stop_gradient(action_log_probs_old_actor)
                
                # We are using action probabilities for the expectation.
                action_probs_old_actor = nn.softmax(logits_old_actor, axis=-1)
                detached_probs = jax.lax.stop_gradient(action_probs_old_actor)
                
                alpha_loss_components = detached_probs * \
                                        (-jnp.exp(log_alpha_params_dict['log_alpha']) * (detached_log_probs + self.target_entropy))
                loss = jnp.sum(alpha_loss_components, axis=1).mean()
                return loss

            alpha_loss_val_scalar, alpha_grads_dict = jax.value_and_grad(alpha_loss_fn)(log_alpha_input.params)
            alpha_grads_dict = jax.lax.pmean(alpha_grads_dict, axis_name='batch')
            alpha_loss_val = jax.lax.pmean(alpha_loss_val_scalar, axis_name='batch')
            
            log_alpha_state_updated = log_alpha_input.apply_gradients(grads=alpha_grads_dict)
            log_alpha_state_to_return = log_alpha_state_updated
            current_alpha_to_return = jnp.exp(log_alpha_state_updated.params['log_alpha'])
        
        actor_metrics = {'actor_loss': actor_loss_val, 'alpha_loss': alpha_loss_val, 
                         'alpha': current_alpha_to_return, 'entropy': entropy_val}
        
        return actor_state_new, log_alpha_state_to_return, current_alpha_to_return, actor_loss_val, actor_metrics

    @partial(jax.jit, static_argnums=(0,))
    def update_target_networks(self, qf1_state: CriticTrainState, qf2_state: CriticTrainState):
        qf1_state_new = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, self.algo_config.tau)
        )
        qf2_state_new = qf2_state.replace(
            target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, self.algo_config.tau)
        )
        return qf1_state_new, qf2_state_new

    # Combined update function
    @partial(jax.jit, static_argnums=(0,))
    def update_all(self,
                   actor_state: TrainState,
                   qf1_state: CriticTrainState,
                   qf2_state: CriticTrainState,
                   log_alpha_input: Union[TrainState, flax.core.FrozenDict, jnp.ndarray],
                   data: dict,
                   key: jax.random.PRNGKey
                   ):

        key_critic, key_actor = jax.random.split(key)

        alpha_arg_for_critic = log_alpha_input.params if self.algo_config.autotune and hasattr(log_alpha_input, 'params') else log_alpha_input
        
        qf1_state, qf2_state, critic_loss, critic_metrics = self._update_critic(
            actor_state, qf1_state, qf2_state, 
            alpha_arg_for_critic, 
            data, key_critic
        )
        
        actor_state, returned_log_alpha_state, returned_current_alpha, actor_loss, actor_alpha_metrics = self._update_actor_and_alpha(
            actor_state, qf1_state, qf2_state, 
            log_alpha_input, 
            data,
            key_actor
        )
        
        all_metrics = {**critic_metrics, **actor_alpha_metrics, 'critic_loss_combined': critic_loss}
        return actor_state, qf1_state, qf2_state, returned_log_alpha_state, returned_current_alpha, all_metrics

class SACAgentContinuous:
    def __init__(self,
                 action_dim: int,
                 observation_space_shape,
                 key: jax.random.PRNGKey,
                 network_config: NetworkConfig,
                 algo_config: AlgoConfig,
                 actor_model_cls=ContinuousTradingActor,
                 critic_model_cls=ContinuousTradingCritic
                 ):

        self.algo_config = algo_config
        self.action_dim = action_dim
        self.network_config = network_config
        self.norm_limit = 1.0 
        key_actor, key_qf1, key_qf2, key_log_alpha = jax.random.split(key, 4)
        self.key_buffer = key

        dummy_obs = jnp.zeros((1, *observation_space_shape), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        self.actor_model = actor_model_cls(network_config=self.network_config, action_dim=self.action_dim)
        actor_params = self.actor_model.init({'params': key_actor, 'dropout': key_actor}, dummy_obs, deterministic=True)['params']
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.adamw(learning_rate=algo_config.policy_lr, eps=algo_config.adam_eps),
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            tx=actor_optimizer
        )

        self.critic_model = critic_model_cls(network_config=self.network_config)
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.norm_limit),
            optax.adamw(learning_rate=algo_config.q_lr, eps=algo_config.adam_eps),
        )

        qf1_params = self.critic_model.init({'params': key_qf1, 'dropout': key_qf1}, dummy_obs, dummy_action, deterministic=True)['params']
        self.qf1_state = CriticTrainState.create(
            apply_fn=self.critic_model.apply,
            params=qf1_params,
            target_params=qf1_params,
            tx=critic_optimizer
        )

        qf2_params = self.critic_model.init({'params': key_qf2, 'dropout': key_qf2}, dummy_obs, dummy_action, deterministic=True)['params']
        self.qf2_state = CriticTrainState.create(
            apply_fn=self.critic_model.apply,
            params=qf2_params,
            target_params=qf2_params,
            tx=critic_optimizer
        )
        
        if algo_config.autotune:
            self.target_entropy = -float(self.action_dim)
            log_alpha_params = {'log_alpha': jnp.zeros((), dtype=jnp.float32)}
            self.log_alpha_state = TrainState.create(
                apply_fn=None,
                params=log_alpha_params,
                tx=critic_optimizer
            )
            self.current_alpha = jnp.exp(self.log_alpha_state.params['log_alpha'])
        else:
            self.current_alpha = jnp.array(algo_config.alpha, dtype=jnp.float32)
            self.log_alpha_state = None
            self.target_entropy = 0.0

    def _get_action_dist(self, actor_params, obs, key_dropout, deterministic):
        mean, log_std = self.actor_model.apply(
            {'params': actor_params}, obs,
            deterministic=deterministic,
            rngs={'dropout': key_dropout}
        )
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return dist
    
    def _sample_action(self, dist, key_sample, deterministic):
        if deterministic:
            action = dist.mean()
        else:
            action = dist.sample(seed=key_sample)
        
        squashed_action = jnp.tanh(action)
        return squashed_action, action

    @partial(jax.jit, static_argnums=(0, 4))
    def select_action(self, actor_params: flax.core.FrozenDict, obs: jnp.ndarray, key: jax.random.PRNGKey, deterministic: bool = False):
        key_dropout, key_sample = jax.random.split(key)
        dist = self._get_action_dist(actor_params, obs, key_dropout, deterministic)
        squashed_action, _ = self._sample_action(dist, key_sample, deterministic)
        return squashed_action

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(self, actor_state, qf1_state, qf2_state, log_alpha_input, data, key):
        if self.algo_config.autotune:
            current_alpha = jnp.exp(log_alpha_input['log_alpha'])
        else:
            current_alpha = log_alpha_input

        key_dropout, key_sample, key_q_target = jax.random.split(key, 3)
        next_dist = self._get_action_dist(actor_state.params, data['next_observations'], key_dropout, deterministic=False)
        next_squashed_action, next_action = self._sample_action(next_dist, key_sample, deterministic=False)
        
        next_log_prob = next_dist.log_prob(next_action)
        next_log_prob -= jnp.sum(jnp.log(1 - jnp.tanh(next_action)**2 + 1e-6), axis=1)

        qf1_next_target = self.critic_model.apply(
            {'params': qf1_state.target_params}, data['next_observations'], next_squashed_action,
            deterministic=True, rngs={'dropout': key_q_target}
        )
        qf2_next_target = self.critic_model.apply(
            {'params': qf2_state.target_params}, data['next_observations'], next_squashed_action,
            deterministic=True, rngs={'dropout': key_q_target}
        )
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        
        next_q_value = min_qf_next_target - current_alpha * next_log_prob
        target_q_value = data['rewards'] + (1.0 - data['dones']) * self.algo_config.gamma * next_q_value
        target_q_value = jax.lax.stop_gradient(target_q_value)

        def qf_loss_fn(params, key_dropout):
            q_val = self.critic_model.apply(
                {'params': params}, data['observations'], data['actions'],
                deterministic=True, rngs={'dropout': key_dropout}
            )
            loss = ((q_val - target_q_value) ** 2).mean()
            return loss, q_val.mean()

        (qf1_loss_val, qf1_values_mean), qf1_grads = jax.value_and_grad(qf_loss_fn, has_aux=True)(qf1_state.params, key)
        qf1_grads = jax.lax.pmean(qf1_grads, axis_name='batch')
        qf1_loss_val = jax.lax.pmean(qf1_loss_val, axis_name='batch')
        qf1_state_new = qf1_state.apply_gradients(grads=qf1_grads)
        
        (qf2_loss_val, qf2_values_mean), qf2_grads = jax.value_and_grad(qf_loss_fn, has_aux=True)(qf2_state.params, key)
        qf2_grads = jax.lax.pmean(qf2_grads, axis_name='batch')
        qf2_loss_val = jax.lax.pmean(qf2_loss_val, axis_name='batch')
        qf2_state_new = qf2_state.apply_gradients(grads=qf2_grads)

        critic_loss = (qf1_loss_val + qf2_loss_val) / 2.0
        
        return qf1_state_new, qf2_state_new, critic_loss, {
            'qf1_loss': qf1_loss_val, 'qf2_loss': qf2_loss_val,
            'qf1_values': qf1_values_mean, 'qf2_values': qf2_values_mean
        }

    @partial(jax.jit, static_argnums=(0,))
    def _update_actor_and_alpha(self, actor_state, qf1_state, qf2_state, log_alpha_input, data, key):
        key_actor, key_alpha, key_sample = jax.random.split(key, 3)
        
        if self.algo_config.autotune:
            actor_effective_alpha = jnp.exp(log_alpha_input.params['log_alpha'])
        else:
            actor_effective_alpha = log_alpha_input
            
        def actor_loss_fn(actor_params):
            dist = self._get_action_dist(actor_params, data['observations'], key_actor, deterministic=False)
            squashed_action, action = self._sample_action(dist, key_sample, deterministic=False)
            
            log_prob = dist.log_prob(action)
            log_prob -= jnp.sum(jnp.log(1 - jnp.tanh(action)**2 + 1e-6), axis=1)
            entropy = -log_prob.mean()

            qf1_pi = self.critic_model.apply({'params': qf1_state.params}, data['observations'], squashed_action, deterministic=True, rngs={'dropout': key_actor})
            qf2_pi = self.critic_model.apply({'params': qf2_state.params}, data['observations'], squashed_action, deterministic=True, rngs={'dropout': key_actor})
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            
            actor_loss = (actor_effective_alpha * log_prob - min_qf_pi).mean()
            return actor_loss, (entropy, log_prob)

        (actor_loss_val, (entropy_val, log_prob_val)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_grads = jax.lax.pmean(actor_grads, axis_name='batch')
        actor_state_new = actor_state.apply_gradients(grads=actor_grads)
        
        alpha_loss_val = 0.0
        log_alpha_state_to_return = log_alpha_input
        current_alpha_to_return = actor_effective_alpha

        if self.algo_config.autotune:
            detached_log_prob = jax.lax.stop_gradient(log_prob_val)
            def alpha_loss_fn(log_alpha_params_dict):
                alpha_loss = (-jnp.exp(log_alpha_params_dict['log_alpha']) * (detached_log_prob + self.target_entropy)).mean()
                return alpha_loss

            alpha_loss_val_scalar, alpha_grads_dict = jax.value_and_grad(alpha_loss_fn)(log_alpha_input.params)
            alpha_grads_dict = jax.lax.pmean(alpha_grads_dict, axis_name='batch')
            alpha_loss_val = jax.lax.pmean(alpha_loss_val_scalar, axis_name='batch')
            
            log_alpha_state_updated = log_alpha_input.apply_gradients(grads=alpha_grads_dict)
            log_alpha_state_to_return = log_alpha_state_updated
            current_alpha_to_return = jnp.exp(log_alpha_state_updated.params['log_alpha'])
        
        actor_metrics = {'actor_loss': actor_loss_val, 'alpha_loss': alpha_loss_val, 'alpha': current_alpha_to_return, 'entropy': entropy_val}
        
        return actor_state_new, log_alpha_state_to_return, current_alpha_to_return, actor_loss_val, actor_metrics
        
    @partial(jax.jit, static_argnums=(0,))
    def update_target_networks(self, qf1_state: CriticTrainState, qf2_state: CriticTrainState):
        qf1_state_new = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, self.algo_config.tau)
        )
        qf2_state_new = qf2_state.replace(
            target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, self.algo_config.tau)
        )
        return qf1_state_new, qf2_state_new

    @partial(jax.jit, static_argnums=(0,))
    def update_all(self, actor_state, qf1_state, qf2_state, log_alpha_input, data, key):
        key_critic, key_actor = jax.random.split(key)

        alpha_arg_for_critic = log_alpha_input.params if self.algo_config.autotune and hasattr(log_alpha_input, 'params') else log_alpha_input
        
        qf1_state, qf2_state, critic_loss, critic_metrics = self._update_critic(
            actor_state, qf1_state, qf2_state, alpha_arg_for_critic, data, key_critic
        )
        
        actor_state, returned_log_alpha_state, returned_current_alpha, actor_loss, actor_alpha_metrics = self._update_actor_and_alpha(
            actor_state, qf1_state, qf2_state, log_alpha_input, data, key_actor
        )
        
        all_metrics = {**critic_metrics, **actor_alpha_metrics, 'critic_loss_combined': critic_loss}
        return actor_state, qf1_state, qf2_state, returned_log_alpha_state, returned_current_alpha, all_metrics