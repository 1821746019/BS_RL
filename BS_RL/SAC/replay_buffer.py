import numpy as np
from typing import Dict, Any


class RolloutBuffer:
    """
    On-policy rollout storage for vectorized environments.
    Stores exactly num_steps transitions for num_envs and exposes minibatch iterators.
    Shapes:
      - obs: [T+1, N, obs_dim]
      - actions: [T, N, *action_shape]
      - rewards, terminations, truncations: [T, N]
    """

    def __init__(self,
                 obs_dim: int,
                 action_shape: tuple,
                 is_discrete_action: bool,
                 num_envs: int,
                 num_steps: int):
        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.is_discrete_action = is_discrete_action
        self.num_envs = num_envs
        self.num_steps = num_steps
        self._allocate()

    def _allocate(self):
        T, N, obs_dim = self.num_steps, self.num_envs, self.obs_dim
        self.obs = np.zeros((T + 1, N, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N) + self.action_shape,
                                dtype=np.int32 if self.is_discrete_action else np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.terminations = np.zeros((T, N), dtype=np.float32)
        self.truncations = np.zeros((T, N), dtype=np.float32)
        self._t = 0

    def reset(self, initial_obs: np.ndarray):
        assert initial_obs.shape[-1] == self.obs_dim
        self._allocate()
        self.obs[0] = initial_obs
        self._t = 0

    def add(self,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            truncations: np.ndarray,
            next_obs: np.ndarray):
        t = self._t
        assert t < self.num_steps
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.terminations[t] = terminations
        self.truncations[t] = truncations
        self.obs[t + 1] = next_obs
        self._t += 1

    def get_batch(self) -> Dict[str, Any]:
        assert self._t == self.num_steps
        # Return batch in [B=N, L=T] layout
        o = np.transpose(self.obs, (1, 0, 2))  # [N, T+1, D]
        a = np.transpose(self.actions, (1, 0) + tuple(range(2, 2 + len(self.action_shape))))  # [N, T, *A]
        r = self.rewards.T  # [N, T]
        term = self.terminations.T
        trunc = self.truncations.T
        m = np.ones_like(r, dtype=np.float32)
        return {'o': o, 'a': a, 'r': r, 'term': term, 'trunc': trunc, 'm': m}

    def iterate_minibatches(self, num_minibatches: int, rng: np.random.RandomState):
        batch = self.get_batch()
        N = batch['o'].shape[0]
        assert N % num_minibatches == 0, "num_envs must be divisible by num_minibatches"
        mb = N // num_minibatches
        perm = rng.permutation(N)
        for i in range(num_minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            yield {k: v[idx] for k, v in batch.items()}