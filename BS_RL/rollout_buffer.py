import numpy as np
import jax.numpy as jnp

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, is_discrete_action: bool, gae_lambda: float, gamma: float):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.is_discrete_action = is_discrete_action
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.observations = np.zeros((self.num_steps, self.num_envs) + self.obs_shape, dtype=np.float32)
        if self.is_discrete_action:
            self.actions = np.zeros((self.num_steps, self.num_envs), dtype=np.int32)
        else:
            self.actions = np.zeros((self.num_steps, self.num_envs) + self.action_shape, dtype=np.float32)
        self.log_probs = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        
        self.advantages = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)

        self.step = 0
        self.full = False

    def add(self, obs, action, log_prob, reward, done, value):
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step = (self.step + 1)
        if self.step == self.num_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value, last_done):
        gae = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
        
        # Reset pointer and full flag
        self.step = 0
        self.full = False

    def get(self, num_minibatches):
        # For recurrent models, we need to group by environments to preserve trajectory structure
        assert self.num_envs % num_minibatches == 0, "num_envs must be divisible by num_minibatches"
        envs_per_batch = self.num_envs // num_minibatches
        
        # Flatten the data but keep trajectory structure: [num_steps, num_envs, ...] -> [num_envs * num_steps, ...]
        obs = self.observations.swapaxes(0, 1).reshape((self.num_envs * self.num_steps,) + self.obs_shape)
        if self.is_discrete_action:
            actions = self.actions.swapaxes(0, 1).reshape(self.num_envs * self.num_steps)
        else:
            actions = self.actions.swapaxes(0, 1).reshape((self.num_envs * self.num_steps,) + self.action_shape)
        log_probs = self.log_probs.swapaxes(0, 1).reshape(self.num_envs * self.num_steps)
        advantages = self.advantages.swapaxes(0, 1).reshape(self.num_envs * self.num_steps)
        returns = self.returns.swapaxes(0, 1).reshape(self.num_envs * self.num_steps)
        values = self.values.swapaxes(0, 1).reshape(self.num_envs * self.num_steps)

        # Create indices that preserve trajectory structure: [num_steps, num_envs]
        env_indices = np.arange(self.num_envs)
        flat_indices = np.arange(self.num_envs * self.num_steps).reshape(self.num_steps, self.num_envs)
        
        # Shuffle environments but keep trajectories intact
        np.random.shuffle(env_indices)
        
        for start_env in range(0, self.num_envs, envs_per_batch):
            end_env = start_env + envs_per_batch
            mb_env_indices = env_indices[start_env:end_env]
            # Get all timesteps for these environments
            mb_indices = flat_indices[:, mb_env_indices].ravel()
            
            yield {
                'obs': obs[mb_indices],
                'actions': actions[mb_indices],
                'log_probs': log_probs[mb_indices],
                'advantages': advantages[mb_indices],
                'returns': returns[mb_indices],
                'values': values[mb_indices],
                'env_indices': mb_env_indices,  # 返回对应的环境索引
            }