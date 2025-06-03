import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from functools import partial

from config import AlgoConfig
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

    @partial(jax.jit, static_argnums=(0,))
    def select_action(self, actor_params: flax.core.FrozenDict, obs: jnp.ndarray, key: jax.random.PRNGKey):
        logits = self.actor_model.apply({'params': actor_params}, obs)
        # Gumbel-softmax trick for sampling is not strictly needed for SAC discrete action selection
        # Categorical sampling is fine
        actions = jax.random.categorical(key, logits, axis=-1)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(self,
                       actor_state: TrainState,
                       qf1_state: CriticTrainState,
                       qf2_state: CriticTrainState,
                       log_alpha_params_or_value: jnp.ndarray, # Can be log_alpha_state.params or fixed alpha
                       data: dict,
                       key_next_actions: jax.random.PRNGKey): # Not used for discrete SAC target Q

        if self.algo_config.autotune:
            current_alpha = jnp.exp(log_alpha_params_or_value['log_alpha'])
        else:
            current_alpha = log_alpha_params_or_value # This is just self.current_alpha (fixed value)

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
        qf1_state_new = qf1_state.apply_gradients(grads=qf1_grads)
        
        # --- QF2 Loss ---
        def qf2_loss_fn(params):
            qf2_all_actions = self.critic_model.apply({'params': params}, data['observations'])
            qf2_taken_action = jnp.take_along_axis(qf2_all_actions, data['actions'], axis=1).squeeze(-1)
            loss = ((qf2_taken_action - target_q_values) ** 2).mean()
            return loss, qf2_taken_action.mean()

        (qf2_loss_val, qf2_values_mean), qf2_grads = jax.value_and_grad(qf2_loss_fn, has_aux=True)(qf2_state.params)
        qf2_state_new = qf2_state.apply_gradients(grads=qf2_grads)

        critic_loss = (qf1_loss_val + qf2_loss_val) / 2.0
        
        return qf1_state_new, qf2_state_new, critic_loss, \
               {'qf1_loss': qf1_loss_val, 'qf2_loss': qf2_loss_val,
                'qf1_values': qf1_values_mean, 'qf2_values': qf2_values_mean}


    @partial(jax.jit, static_argnums=(0,))
    def _update_actor_and_alpha(self,
                                actor_state: TrainState,
                                qf1_state: CriticTrainState, # Pass online Q-network states
                                qf2_state: CriticTrainState,
                                log_alpha_state_or_value: TrainState, # Can be log_alpha_state or fixed alpha
                                data: dict):
        
        if self.algo_config.autotune:
            current_alpha = jnp.exp(log_alpha_state_or_value.params['log_alpha'])
        else:
            current_alpha = log_alpha_state_or_value # This is self.current_alpha (fixed value)

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
            actor_loss_components = action_probs * (current_alpha * action_log_probs - min_qf_values)
            loss = jnp.sum(actor_loss_components, axis=1).mean()
            
            entropy = -jnp.sum(action_probs * action_log_probs, axis=1).mean()
            return loss, entropy

        (actor_loss_val, entropy_val), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state_new = actor_state.apply_gradients(grads=actor_grads)

        # --- Alpha Loss (if autotuning) ---
        alpha_loss_val = 0.0
        log_alpha_state_new = log_alpha_state_or_value # Default to old state if not autotuning

        if self.algo_config.autotune:
            def alpha_loss_fn(log_alpha_params_dict):
                # Use log_probs from the *current* policy (before this actor update, or recompute)
                # The original CleanRL PyTorch uses log_pi from data.observations (current batch)
                # and actor.get_action(data.observations) which uses current actor params
                # So, we need log_probs from the newly updated actor_state_new if we want to match that
                # However, standard is to use log_probs from before actor update, or detach them.
                # Let's use the log_probs from the actor_loss_fn calculation (derived from actor_state.params, i.e. old actor)
                # This is equivalent to detaching log_pi for alpha loss.
                
                # Recalculate with old actor params (or pass log_probs as arg)
                logits_old_actor = self.actor_model.apply({'params': actor_state.params}, data['observations'])
                action_log_probs_old_actor = nn.log_softmax(logits_old_actor, axis=-1)
                action_probs_old_actor = nn.softmax(logits_old_actor, axis=-1)

                detached_log_probs = jax.lax.stop_gradient(action_log_probs_old_actor)
                detached_probs = jax.lax.stop_gradient(action_probs_old_actor)

                # Alpha loss: sum_a [ P(a|s) * (-alpha * (log P(a|s) + target_entropy)) ]
                # Note: log_alpha_params_dict is {'log_alpha': scalar_value}
                alpha_loss_components = detached_probs * \
                                        (-jnp.exp(log_alpha_params_dict['log_alpha']) * (detached_log_probs + self.target_entropy))
                loss = jnp.sum(alpha_loss_components, axis=1).mean()
                return loss

            alpha_loss_val, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha_state_or_value.params)
            log_alpha_state_new = log_alpha_state_or_value.apply_gradients(grads=alpha_grads)
            self.current_alpha = jnp.exp(log_alpha_state_new.params['log_alpha']) # Update current_alpha for next iter
        
        return actor_state_new, log_alpha_state_new, actor_loss_val, \
               {'actor_loss': actor_loss_val, 'alpha_loss': alpha_loss_val, 'alpha': self.current_alpha, 'entropy': entropy_val}

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
                   log_alpha_state_or_value: TrainState, # TrainState if autotune, else fixed float value
                   data: dict, # observations, actions, next_observations, rewards, dones
                   key_next_actions: jax.random.PRNGKey # Not used for discrete
                   ):

        # Update Critics
        qf1_state, qf2_state, critic_loss, critic_metrics = self._update_critic(
            actor_state, qf1_state, qf2_state, 
            log_alpha_state_or_value.params if self.algo_config.autotune else self.current_alpha, 
            data, key_next_actions
        )
        
        # Update Actor and Alpha
        actor_state, log_alpha_state_or_value, actor_loss, actor_alpha_metrics = self._update_actor_and_alpha(
            actor_state, qf1_state, qf2_state, 
            log_alpha_state_or_value, # Pass the state if autotune, else fixed value
            data
        )
        
        all_metrics = {**critic_metrics, **actor_alpha_metrics, 'critic_loss_combined': critic_loss}
        return actor_state, qf1_state, qf2_state, log_alpha_state_or_value, all_metrics