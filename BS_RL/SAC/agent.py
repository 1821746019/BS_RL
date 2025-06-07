import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from functools import partial
from typing import Union

from .config import AlgoConfig
# Networks will be passed as arguments to SACAgent init

class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict

class SACAgent:
    def __init__(self,
                 action_dim: int,
                 observation_space_shape, # For network init
                 key: jax.random.PRNGKey,
                 algo_config: AlgoConfig,
                 actor_model_cls, # e.g. ActorCNN or ActorMLP
                 critic_model_cls # e.g. CriticCNN or CriticMLP
                 ):

        self.algo_config = algo_config
        self.action_dim = action_dim
        
        # PRNG keys
        key_actor, key_qf1, key_qf2, key_log_alpha = jax.random.split(key, 4)
        self.key_buffer = key # For action selection randomness

        # Dummy input for network initialization
        # For CNNs, obs_shape is (C, H, W), add batch dim
        # For MLPs, obs_shape is (Features,), add batch dim
        if len(observation_space_shape) == 3: # Image obs
            dummy_obs = jnp.zeros((1, *observation_space_shape), dtype=jnp.float32)
        else: # Vector obs
            dummy_obs = jnp.zeros((1, *observation_space_shape), dtype=jnp.float32)


        # Actor
        self.actor_model = actor_model_cls(action_dim=action_dim)
        actor_params = self.actor_model.init(key_actor, dummy_obs)['params']
        self.actor_state = TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            tx=optax.adamw(learning_rate=algo_config.policy_lr, eps=algo_config.adam_eps)
        )

        # Critic 1
        self.critic_model = critic_model_cls(action_dim=action_dim) # Q-network
        qf1_params = self.critic_model.init(key_qf1, dummy_obs)['params']
        self.qf1_state = CriticTrainState.create(
            apply_fn=self.critic_model.apply,
            params=qf1_params,
            target_params=qf1_params, # Initialize target params same as online params
            tx=optax.adamw(learning_rate=algo_config.q_lr, eps=algo_config.adam_eps)
        )

        # Critic 2
        qf2_params = self.critic_model.init(key_qf2, dummy_obs)['params'] # Re-init for different weights
        self.qf2_state = CriticTrainState.create(
            apply_fn=self.critic_model.apply,
            params=qf2_params,
            target_params=qf2_params,
            tx=optax.adamw(learning_rate=algo_config.q_lr, eps=algo_config.adam_eps)
        )

        # Alpha (Entropy Temperature)
        if algo_config.autotune:
            self.target_entropy = -algo_config.target_entropy_scale * jnp.log(1.0 / action_dim)
            # log_alpha is a scalar, TrainState expects params to be a pytree (like a dict)
            log_alpha_params = {'log_alpha': jnp.zeros((), dtype=jnp.float32)}
            self.log_alpha_state = TrainState.create(
                apply_fn=None, # Not applying a network model here
                params=log_alpha_params,
                tx=optax.adamw(learning_rate=algo_config.q_lr, eps=algo_config.adam_eps) # Same LR as Q
            )
            self.current_alpha = jnp.exp(self.log_alpha_state.params['log_alpha'])
        else:
            self.current_alpha = jnp.array(algo_config.alpha, dtype=jnp.float32)
            self.log_alpha_state = None # To satisfy type hints or checks
            self.target_entropy = 0.0 # Placeholder

    @partial(jax.jit, static_argnums=(0, 4))
    def select_action(self, actor_params: flax.core.FrozenDict, obs: jnp.ndarray, key: jax.random.PRNGKey, deterministic: bool = False):
        logits = self.actor_model.apply({'params': actor_params}, obs)
        # Gumbel-softmax trick for sampling is not strictly needed for SAC discrete action selection
        # Categorical sampling is fine
        if deterministic:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(key, logits, axis=-1)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(self,
                       actor_state: TrainState,
                       qf1_state: CriticTrainState,
                       qf2_state: CriticTrainState,
                       log_alpha_input: Union[flax.core.FrozenDict, jnp.ndarray], # .params if autotune, else fixed value
                       data: dict,
                       key_next_actions: jax.random.PRNGKey): # Not used for discrete SAC target Q

        if self.algo_config.autotune: # log_alpha_input is log_alpha_state.params
            current_alpha = jnp.exp(log_alpha_input['log_alpha'])
        else: # log_alpha_input is the fixed alpha value
            current_alpha = log_alpha_input

        # Get next action probabilities and log probabilities from current policy
        next_logits = self.actor_model.apply({'params': actor_state.params}, data['next_observations'])
        next_action_probs = nn.softmax(next_logits, axis=-1)
        next_action_log_probs = nn.log_softmax(next_logits, axis=-1)

        # Target Q-values from target Q-networks
        qf1_next_target_values = self.critic_model.apply({'params': qf1_state.target_params}, data['next_observations'])
        qf2_next_target_values = self.critic_model.apply({'params': qf2_state.target_params}, data['next_observations'])
        min_qf_next_target = jnp.minimum(qf1_next_target_values, qf2_next_target_values)
        
        # Expected value for next state: sum_a' [ P(a'|s') * (Q_target(s',a') - alpha * log P(a'|s')) ]
        next_q_value_components = next_action_probs * (min_qf_next_target - current_alpha * next_action_log_probs)
        next_q_value = jnp.sum(next_q_value_components, axis=1)
        
        # TD target
        target_q_values = data['rewards'] + (1.0 - data['dones']) * self.algo_config.gamma * next_q_value
        target_q_values = jax.lax.stop_gradient(target_q_values) # Important!

        # --- QF1 Loss ---
        def qf1_loss_fn(params):
            # Get Q-values for ALL actions from current Q-network
            qf1_all_actions = self.critic_model.apply({'params': params}, data['observations'])
            # Gather Q-values for the specific actions taken (from replay buffer)
            qf1_taken_action = jnp.take_along_axis(qf1_all_actions, data['actions'], axis=1).squeeze(-1)
            loss = ((qf1_taken_action - target_q_values) ** 2).mean()
            return loss, qf1_taken_action.mean()

        (qf1_loss_val, qf1_values_mean), qf1_grads = jax.value_and_grad(qf1_loss_fn, has_aux=True)(qf1_state.params)
        qf1_grads = jax.lax.pmean(qf1_grads, axis_name='batch')
        qf1_loss_val = jax.lax.pmean(qf1_loss_val, axis_name='batch')
        qf1_values_mean = jax.lax.pmean(qf1_values_mean, axis_name='batch')
        qf1_state_new = qf1_state.apply_gradients(grads=qf1_grads)
        
        # --- QF2 Loss ---
        def qf2_loss_fn(params):
            qf2_all_actions = self.critic_model.apply({'params': params}, data['observations'])
            qf2_taken_action = jnp.take_along_axis(qf2_all_actions, data['actions'], axis=1).squeeze(-1)
            loss = ((qf2_taken_action - target_q_values) ** 2).mean()
            return loss, qf2_taken_action.mean()

        (qf2_loss_val, qf2_values_mean), qf2_grads = jax.value_and_grad(qf2_loss_fn, has_aux=True)(qf2_state.params)
        qf2_grads = jax.lax.pmean(qf2_grads, axis_name='batch')
        qf2_loss_val = jax.lax.pmean(qf2_loss_val, axis_name='batch')
        qf2_values_mean = jax.lax.pmean(qf2_values_mean, axis_name='batch')
        qf2_state_new = qf2_state.apply_gradients(grads=qf2_grads)

        critic_loss = (qf1_loss_val + qf2_loss_val) / 2.0
        # critic_loss is already an average of averages, pmean not strictly needed if components are.
        # However, to be safe if one component was scalar and other not:
        critic_loss = jax.lax.pmean(critic_loss, axis_name='batch')
        
        return qf1_state_new, qf2_state_new, critic_loss, \
               {'qf1_loss': qf1_loss_val, 'qf2_loss': qf2_loss_val,
                'qf1_values': qf1_values_mean, 'qf2_values': qf2_values_mean}


    @partial(jax.jit, static_argnums=(0,))
    def _update_actor_and_alpha(self,
                                actor_state: TrainState,
                                qf1_state: CriticTrainState, # Pass online Q-network states
                                qf2_state: CriticTrainState,
                                log_alpha_input: Union[TrainState, jnp.ndarray], # TrainState if autotune, else fixed value
                                data: dict):
        
        # Determine effective alpha for actor loss calculation
        if self.algo_config.autotune: # log_alpha_input is TrainState
            actor_effective_alpha = jnp.exp(log_alpha_input.params['log_alpha'])
        else: # log_alpha_input is the fixed alpha value
            actor_effective_alpha = log_alpha_input

        # --- Actor Loss ---
        def actor_loss_fn(actor_params):
            logits = self.actor_model.apply({'params': actor_params}, data['observations'])
            action_probs = nn.softmax(logits, axis=-1)
            action_log_probs = nn.log_softmax(logits, axis=-1)

            # Q-values from online Q-networks (detached for actor loss)
            qf1_all_actions = self.critic_model.apply({'params': qf1_state.params}, data['observations'])
            qf2_all_actions = self.critic_model.apply({'params': qf2_state.params}, data['observations'])
            min_qf_values = jnp.minimum(qf1_all_actions, qf2_all_actions)
            min_qf_values = jax.lax.stop_gradient(min_qf_values)

            # Expected value for actor loss: sum_a [ P(a|s) * (alpha * log P(a|s) - Q(s,a)) ]
            actor_loss_components = action_probs * (actor_effective_alpha * action_log_probs - min_qf_values)
            loss = jnp.sum(actor_loss_components, axis=1).mean()
            
            entropy = -jnp.sum(action_probs * action_log_probs, axis=1).mean()
            return loss, entropy

        (actor_loss_val, entropy_val), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_grads = jax.lax.pmean(actor_grads, axis_name='batch')
        actor_loss_val = jax.lax.pmean(actor_loss_val, axis_name='batch')
        entropy_val = jax.lax.pmean(entropy_val, axis_name='batch')
        actor_state_new = actor_state.apply_gradients(grads=actor_grads)

        # --- Alpha Loss (if autotuning) ---
        alpha_loss_val = 0.0 # Default if not autotuning
        log_alpha_state_to_return = log_alpha_input # Pass through if not updated
        current_alpha_to_return = actor_effective_alpha # Pass through if not updated

        if self.algo_config.autotune: # log_alpha_input is TrainState
            def alpha_loss_fn(log_alpha_params_dict): # log_alpha_params_dict is log_alpha_input.params
                # ... (rest of alpha loss logic from original, using detached_log_probs)
                logits_old_actor = self.actor_model.apply({'params': actor_state.params}, data['observations']) # old actor
                action_log_probs_old_actor = nn.log_softmax(logits_old_actor, axis=-1)
                action_probs_old_actor = nn.softmax(logits_old_actor, axis=-1)

                detached_log_probs = jax.lax.stop_gradient(action_log_probs_old_actor)
                detached_probs = jax.lax.stop_gradient(action_probs_old_actor)
                
                alpha_loss_components = detached_probs * \
                                        (-jnp.exp(log_alpha_params_dict['log_alpha']) * (detached_log_probs + self.target_entropy))
                loss = jnp.sum(alpha_loss_components, axis=1).mean()
                return loss

            # log_alpha_input here is the TrainState instance
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
                   log_alpha_input: Union[TrainState, flax.core.FrozenDict, jnp.ndarray], # TrainState or .params if autotune, else fixed float value
                   data: dict, # observations, actions, next_observations, rewards, dones
                   key_next_actions: jax.random.PRNGKey # Not used for discrete
                   ):

        # Determine the argument for _update_critic's alpha parameter
        # If autotuning, _update_critic expects log_alpha_state.params
        # If not autotuning, _update_critic expects the fixed alpha value
        alpha_arg_for_critic = log_alpha_input.params if self.algo_config.autotune and hasattr(log_alpha_input, 'params') else log_alpha_input
        
        qf1_state, qf2_state, critic_loss, critic_metrics = self._update_critic(
            actor_state, qf1_state, qf2_state, 
            alpha_arg_for_critic, 
            data, key_next_actions
        )
        
        # _update_actor_and_alpha takes the full log_alpha_input (TrainState or fixed value)
        actor_state, returned_log_alpha_state, returned_current_alpha, actor_loss, actor_alpha_metrics = self._update_actor_and_alpha(
            actor_state, qf1_state, qf2_state, 
            log_alpha_input, 
            data
        )
        
        all_metrics = {**critic_metrics, **actor_alpha_metrics, 'critic_loss_combined': critic_loss}
        # The returned_log_alpha_state is the updated TrainState or the passed-through fixed value.
        # The returned_current_alpha is the new jnp.exp(log_alpha) or the passed-through fixed value.
        return actor_state, qf1_state, qf2_state, returned_log_alpha_state, returned_current_alpha, all_metrics