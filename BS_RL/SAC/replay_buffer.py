import numpy as np
from typing import Optional, Dict, Any

class RecurrentReplayBuffer:
    def __init__(self,
                 obs_dim: int,
                 action_shape: tuple,
                 is_discrete_action: bool,
                 capacity_segments: int,
                 num_envs: int,
                 seg_len: int,
                 burn_in: int = 0, min_gap: int = 60):
        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.is_discrete_action = is_discrete_action
        self.capacity_segments = capacity_segments
        self.num_envs = num_envs
        self.burn_in = int(max(burn_in, 0))
        self.seg_len = seg_len
        self.min_gap = min_gap
        # Storage for segments as multi-dimensional arrays for vectorized operations
        T = self.seg_len
        self.segments_o = np.zeros((capacity_segments, T + 1, obs_dim), dtype=np.float32)
        self.segments_a = np.zeros((capacity_segments, T) + action_shape, dtype=np.int32 if is_discrete_action else np.float32)
        self.segments_r = np.zeros((capacity_segments, T), dtype=np.float32)
        self.segments_term = np.zeros((capacity_segments, T), dtype=np.float32)
        self.segments_trunc = np.zeros((capacity_segments, T), dtype=np.float32)
        self.segments_m = np.zeros((capacity_segments, T), dtype=np.float32)
        
        self.seg_ptr = 0
        self.num_stored_segments = 0

        # Ongoing buffers for each env
        self._reset_working_buffers()

    def _reset_working_buffers(self):
        # Working buffers for accumulating num_bptt steps before storing as segment
        T = self.seg_len
        self.work_o = [np.zeros((T + 1, self.obs_dim), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_a = [np.zeros((T,) + self.action_shape, dtype=np.int32 if self.is_discrete_action else np.float32) for _ in range(self.num_envs)]
        self.work_r = [np.zeros((T,), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_term = [np.zeros((T,), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_trunc = [np.zeros((T,), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_t = np.zeros((self.num_envs,), dtype=np.int32)
        self.work_started = np.zeros((self.num_envs,), dtype=bool)
        self.work_ep_start = np.zeros((self.num_envs,), dtype=bool)  # True if this segment starts a new episode

    def add_batch(self, obs: np.ndarray, next_obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                  terminations: np.ndarray, truncations: np.ndarray):
        """
        obs: [N, obs_dim]
        next_obs: [N, obs_dim]
        actions: [N, *action_shape]
        rewards: [N]
        terminations: [N]
        truncations: [N]
        """
        N = obs.shape[0]
        assert N == self.num_envs
        for i in range(N):
            t = int(self.work_t[i])
            
            if not self.work_started[i]:
                # Initialize first observation for this segment
                self.work_o[i][0] = obs[i]
                self.work_started[i] = True
                self.work_ep_start[i] = True  # Mark as episode start

            # Store current transition
            self.work_a[i][t] = actions[i]
            self.work_r[i][t] = rewards[i]
            self.work_term[i][t] = terminations[i]
            self.work_trunc[i][t] = truncations[i]
            self.work_o[i][t + 1] = next_obs[i]
            self.work_t[i] = t + 1

            # Check if we should finalize this segment
            should_finalize = False
            if t + 1 >= self.seg_len:
                # Segment is full
                should_finalize = True
            elif terminations[i] > 0.5 or truncations[i] > 0.5:
                # Episode ended, finalize segment even if not full
                should_finalize = True

            if should_finalize:
                self._finalize_env_segment(i, next_obs[i], terminations[i] > 0.5 or truncations[i] > 0.5)

    def _finalize_env_segment(self, env_idx: int, last_next_obs: np.ndarray, episode_ended: bool):
        t = int(self.work_t[env_idx])
        if t == 0:
            # nothing collected
            return
        
        # Create segment with proper padding/masking
        length = t
        T = self.seg_len
        o = np.zeros((T + 1, self.obs_dim), dtype=np.float32)
        a = np.zeros((T,) + self.action_shape, dtype=np.int32 if self.is_discrete_action else np.float32)
        r = np.zeros((T,), dtype=np.float32)
        term = np.zeros((T,), dtype=np.float32)
        trunc = np.zeros((T,), dtype=np.float32)
        m = np.zeros((T,), dtype=np.float32)  # mask for valid steps
        
        # Copy actual data
        o[:length + 1] = self.work_o[env_idx][:length + 1]
        a[:length] = self.work_a[env_idx][:length]
        r[:length] = self.work_r[env_idx][:length]
        term[:length] = self.work_term[env_idx][:length]
        trunc[:length] = self.work_trunc[env_idx][:length]
        # mark valid training steps: exclude burn-in prefix
        if length > self.burn_in:
            m[self.burn_in:length] = 1.0
        else:
            # not enough steps to reach burn-in; keep m as zeros
            pass
        
        # For padded steps, mark as terminal to prevent bootstrap
        if length < self.seg_len:
            term[length:] = 1.0

        # Store segment in circular buffer using vectorized operations
        self.segments_o[self.seg_ptr] = o
        self.segments_a[self.seg_ptr] = a
        self.segments_r[self.seg_ptr] = r
        self.segments_term[self.seg_ptr] = term
        self.segments_trunc[self.seg_ptr] = trunc
        self.segments_m[self.seg_ptr] = m
        
        # Update pointers
        self.seg_ptr = (self.seg_ptr + 1) % self.capacity_segments
        self.num_stored_segments = min(self.num_stored_segments + 1, self.capacity_segments)

        # Reset working buffer for this env
        self.work_o[env_idx][:] = 0
        self.work_a[env_idx][:] = 0
        self.work_r[env_idx][:] = 0
        self.work_term[env_idx][:] = 0
        self.work_trunc[env_idx][:] = 0
        self.work_t[env_idx] = 0
        self.work_started[env_idx] = False
        self.work_ep_start[env_idx] = False
        
        # If episode didn't end, continue accumulating next segment
        if not episode_ended:
            self.work_o[env_idx][0] = last_next_obs
            self.work_started[env_idx] = True
            self.work_ep_start[env_idx] = False  # Not an episode start
        else:
            # Episode ended, next segment (if any) will be episode start
            self.work_ep_start[env_idx] = True

    def size(self) -> int:
        return self.num_stored_segments
    def can_sample(self, batch_size: int, num_bptt: int) -> bool:
        return self.num_stored_segments *(1+(self.seg_len - self.burn_in - num_bptt)/self.min_gap) >= batch_size
    def sample(self, batch_size: int, num_bptt: int) -> Dict[str, Any] | None:
        if not self.can_sample(batch_size, num_bptt):
            return None
        # 若有足够的segments，则直接均匀采样
        if self.num_stored_segments >= batch_size:
            # Vectorized sampling with advanced indexing.
            # Window length for training steps (L), excluding the final +1 obs.
            L = int(self.burn_in + num_bptt)
            max_start = self.seg_len - L
            if max_start < 0:
                return None

            # Sample segments (without replacement) and independent start indices (with replacement)
            idxs = np.random.choice(self.num_stored_segments, size=batch_size, replace=False)
            start_idxs = np.random.randint(0, max_start + 1, size=batch_size, dtype=np.int32)

            # Build time indices for actions/rewards/etc. (length L) and observations (length L+1)
            time_idx_a = start_idxs[:, None] + np.arange(L, dtype=np.int32)[None, :]
            time_idx_o = start_idxs[:, None] + np.arange(L + 1, dtype=np.int32)[None, :]

            # Gather with advanced indexing; trailing dims are preserved automatically
            o = self.segments_o[idxs[:, None], time_idx_o]          # [B, L+1, obs_dim]
            a = self.segments_a[idxs[:, None], time_idx_a]          # [B, L, *action_shape]
            r = self.segments_r[idxs[:, None], time_idx_a]          # [B, L]
            term = self.segments_term[idxs[:, None], time_idx_a]    # [B, L]
            trunc = self.segments_trunc[idxs[:, None], time_idx_a]  # [B, L]
            m = self.segments_m[idxs[:, None], time_idx_a]          # [B, L]

            batch = {
                'o': o,
                'a': a,
                'r': r,
                'term': term,
                'trunc': trunc,
                'm': m,
            }

        else: #但一开始是也许是不足够的，需要反复从已有的segment中采样，直到达到batch_size
            raise NotImplementedError("this case is not implemented, please make sure the buffer has enough segments")
        return batch

    def finalize_all_working(self):
        for i in range(self.num_envs):
            if self.work_started[i]:
                # Force finalize with dummy next obs and episode_ended=True
                dummy_next_obs = np.zeros(self.obs_dim, dtype=np.float32)
                self._finalize_env_segment(i, dummy_next_obs, episode_ended=True) 