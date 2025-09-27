import numpy as np
from typing import Optional, Dict, Any

class RecurrentReplayBuffer:
    def __init__(self,
                 obs_shape: tuple,
                 action_shape: tuple,
                 is_discrete_action: bool,
                 capacity_segments: int,
                 num_envs: int,
                 seg_len: int,
                 min_gap: int = 60,
                 min_valid_step_ratio: float = 0.8):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.is_discrete_action = is_discrete_action
        self.capacity_segments = capacity_segments
        self.num_envs = num_envs
        self.seg_len = seg_len
        self.min_gap = min_gap
        self.min_valid_step_ratio = min_valid_step_ratio
        # Storage for segments as multi-dimensional arrays for vectorized operations
        T = self.seg_len
        self.segments_o = np.zeros((capacity_segments, T + 1) + self.obs_shape, dtype=np.float32)
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
        self.work_o = [np.zeros((T + 1,) + self.obs_shape, dtype=np.float32) for _ in range(self.num_envs)]
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
        
        length = t
        T = self.seg_len

        if length / T >= self.min_valid_step_ratio:
            # Create segment with proper padding/masking
            o = np.zeros((T + 1,) + self.obs_shape, dtype=np.float32)
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
            m[:length] = 1.0
            
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
    def can_sample_many(self, batch_size: int, seq_len: int, num_batches: int = 1) -> bool:
        return self.num_stored_segments *(1+(self.seg_len - seq_len)/self.min_gap) >= batch_size * num_batches
    def sample(self, batch_size: int, seq_len: int) -> Dict[str, Any] | None:
        return self.sample_many(batch_size, seq_len, num_batches=1, remove_K_dim_if_one=True)

    def sample_many(self, batch_size: int, seq_len: int, num_batches: int, remove_K_dim_if_one: bool = False) -> Dict[str, Any] | None:
        """
        Vectorized sampling of multiple independent batches to feed multiple update steps per JIT call.
        Returns a dict of arrays stacked on a leading dimension K=num_batches.
        Shapes:
          - o: [K, B, L+1, obs_dim]
          - a: [K, B, L, *action_shape]
          - r, term, trunc, m: [K, B, L]
        """
        if num_batches <= 0:
            return None
        if not self.can_sample_many(batch_size, seq_len, num_batches):
            return None
        # Window length for training steps (L), excluding the final +1 obs.
        L = seq_len
        max_start = self.seg_len - L
        if max_start < 0:
            return None

        idxs = np.random.randint(0, self.num_stored_segments, size=(num_batches, batch_size))
        start_idxs = np.random.randint(0, max_start + 1, size=(num_batches, batch_size), dtype=np.int32)

        # Build time indices
        time_range_a = np.arange(L, dtype=np.int32)[None, None, :]
        time_range_o = np.arange(L + 1, dtype=np.int32)[None, None, :]
        time_idx_a = start_idxs[..., None] + time_range_a  # [K,B,L]
        time_idx_o = start_idxs[..., None] + time_range_o  # [K,B,L+1]

        # Advanced indexing with broadcasting
        o = self.segments_o[idxs[..., None], time_idx_o]          # [K,B,L+1, obs_dim]
        a = self.segments_a[idxs[..., None], time_idx_a]          # [K,B,L, *A]
        r = self.segments_r[idxs[..., None], time_idx_a]          # [K,B,L]
        term = self.segments_term[idxs[..., None], time_idx_a]    # [K,B,L]
        trunc = self.segments_trunc[idxs[..., None], time_idx_a]  # [K,B,L]
        m = self.segments_m[idxs[..., None], time_idx_a]          # [K,B,L]
        remove_K_dim = num_batches == 1 and remove_K_dim_if_one
        batch = {
            'o': o if not remove_K_dim else o[0],
            'a': a if not remove_K_dim else a[0],
            'r': r if not remove_K_dim else r[0],
            'term': term if not remove_K_dim else term[0],
            'trunc': trunc if not remove_K_dim else trunc[0],
            'm': m if not remove_K_dim else m[0],
        }
        return batch

    def finalize_all_working(self):
        for i in range(self.num_envs):
            if self.work_started[i]:
                # Force finalize with dummy next obs and episode_ended=True
                dummy_next_obs = np.zeros(self.obs_shape, dtype=np.float32)
                self._finalize_env_segment(i, dummy_next_obs, episode_ended=True) 