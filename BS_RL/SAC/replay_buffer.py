import numpy as np
from typing import Optional, Dict, Any, List, Tuple

class RecurrentReplayBuffer:
    def __init__(self,
                 obs_dim: int,
                 action_shape: tuple,
                 is_discrete_action: bool,
                 capacity_episodes: int,
                 num_envs: int,
                 max_episode_len: int,
                 num_bptt: int,
                 segment_sample: bool = True):
        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.is_discrete_action = is_discrete_action
        self.capacity_episodes = capacity_episodes
        self.num_envs = num_envs
        self.max_episode_len = max_episode_len
        self.num_bptt = num_bptt
        self.segment_sample = segment_sample

        # Storage for finalized episodes
        self.episodes = []  # list of dicts with keys: 'o', 'a', 'r', 'd'
        self.ep_ptr = 0

        # Ongoing buffers for each env
        self._reset_working_episodes()

    def _reset_working_episodes(self):
        self.work_o = [np.zeros((self.max_episode_len + 1, self.obs_dim), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_a = [np.zeros((self.max_episode_len,) + self.action_shape, dtype=np.int32 if self.is_discrete_action else np.float32) for _ in range(self.num_envs)]
        self.work_r = [np.zeros((self.max_episode_len,), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_d = [np.zeros((self.max_episode_len,), dtype=np.float32) for _ in range(self.num_envs)]
        self.work_t = np.zeros((self.num_envs,), dtype=np.int32)
        self.work_started = np.zeros((self.num_envs,), dtype=bool)

    def add_batch(self, obs: np.ndarray, next_obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                  dones: np.ndarray):
        """
        obs: [N, obs_dim]
        next_obs: [N, obs_dim]
        actions: [N, *action_shape]
        rewards: [N]
        dones: [N]
        """
        N = obs.shape[0]
        assert N == self.num_envs
        for i in range(N):
            t = int(self.work_t[i])
            if not self.work_started[i]:
                # initialize first observation for this episode
                self.work_o[i][0] = obs[i]
                self.work_started[i] = True

            if t >= self.max_episode_len:
                # force finalize if overflow
                self._finalize_env_episode(i, last_next_obs=next_obs[i])
                t = 0

            self.work_a[i][t] = actions[i]
            self.work_r[i][t] = rewards[i]
            self.work_d[i][t] = dones[i]
            self.work_o[i][t + 1] = next_obs[i]
            self.work_t[i] = t + 1

            if dones[i] > 0.5:
                self._finalize_env_episode(i, last_next_obs=next_obs[i])

    def _finalize_env_episode(self, env_idx: int, last_next_obs: Optional[np.ndarray] = None):
        t = int(self.work_t[env_idx])
        if t == 0:
            # nothing collected
            return
        o = self.work_o[env_idx][:t + 1].copy()
        a = self.work_a[env_idx][:t].copy()
        r = self.work_r[env_idx][:t].copy()
        d = self.work_d[env_idx][:t].copy()

        ep = {'o': o, 'a': a, 'r': r, 'd': d}
        if len(self.episodes) < self.capacity_episodes:
            self.episodes.append(ep)
        else:
            self.episodes[self.ep_ptr] = ep
            self.ep_ptr = (self.ep_ptr + 1) % self.capacity_episodes

        # reset this env working buffer
        self.work_o[env_idx][:] = 0
        self.work_a[env_idx][:] = 0
        self.work_r[env_idx][:] = 0
        self.work_d[env_idx][:] = 0
        self.work_t[env_idx] = 0
        self.work_started[env_idx] = False
        if last_next_obs is not None:
            # prepare first obs of next episode if env immediately continues
            self.work_o[env_idx][0] = last_next_obs
            self.work_started[env_idx] = True

    def size(self) -> int:
        return len(self.episodes)

    # ========= New: segment-based readiness and sampling =========
    def _gather_valid_segment_starts(self) -> List[Tuple[str, int, int]]:
        """Return a list of (src, idx, start) where src in {"ep", "work"}.
        For each episode or working buffer, collect all valid segment start indices of length num_bptt.
        """
        starts: List[Tuple[str, int, int]] = []
        if self.segment_sample:
            # finalized episodes
            for ei, ep in enumerate(self.episodes):
                T = ep['a'].shape[0]
                if T >= self.num_bptt:
                    for s in range(0, T - self.num_bptt + 1):
                        starts.append(("ep", ei, s))
            # ongoing episodes
            for wi in range(self.num_envs):
                T = int(self.work_t[wi])
                if T >= self.num_bptt:
                    for s in range(0, T - self.num_bptt + 1):
                        starts.append(("work", wi, s))
        return starts

    def can_sample(self, batch_size: int) -> bool:
        if not self.segment_sample:
            # fallback: require at least batch_size finalized episodes
            return len(self.episodes) >= batch_size
        return len(self._gather_valid_segment_starts()) >= batch_size

    def sample(self, batch_size: int) -> Dict[str, Any]:
        if self.segment_sample:
            starts = self._gather_valid_segment_starts()
            assert len(starts) >= batch_size, "Not enough valid segments to sample"
            choice_idx = np.random.choice(len(starts), size=batch_size, replace=False if len(starts) >= batch_size else True)
            o_list, a_list, r_list, d_list, m_list = [], [], [], [], []
            for ci in choice_idx:
                src, idx, s = starts[ci]
                if src == "ep":
                    ep = self.episodes[idx]
                    o_src, a_src, r_src, d_src = ep['o'], ep['a'], ep['r'], ep['d']
                else:
                    o_src = self.work_o[idx]
                    a_src = self.work_a[idx]
                    r_src = self.work_r[idx]
                    d_src = self.work_d[idx]
                e = s + self.num_bptt
                o_seg = o_src[s:e + 1]
                a_seg = a_src[s:e]
                r_seg = r_src[s:e]
                d_seg = d_src[s:e]
                m_seg = np.ones((self.num_bptt,), dtype=np.float32)
                o_list.append(o_seg)
                a_list.append(a_seg)
                r_list.append(r_seg)
                d_list.append(d_seg)
                m_list.append(m_seg)
            batch = {
                'o': np.stack(o_list, axis=0),
                'a': np.stack(a_list, axis=0),
                'r': np.stack(r_list, axis=0),
                'd': np.stack(d_list, axis=0),
                'm': np.stack(m_list, axis=0),
            }
            return batch
        # ---- fallback previous episode-based sampling ----
        idxs = self._sample_episode_indices(batch_size)
        o_list, a_list, r_list, d_list, m_list = [], [], [], [], []
        for idx in idxs:
            ep = self.episodes[idx]
            T = ep['a'].shape[0]
            if self.segment_sample and T > self.num_bptt:
                start = np.random.randint(0, T - self.num_bptt + 1)
                end = start + self.num_bptt
                o_seg = ep['o'][start:end + 1]
                a_seg = ep['a'][start:end]
                r_seg = ep['r'][start:end]
                d_seg = ep['d'][start:end]
                m_seg = np.ones_like(r_seg, dtype=np.float32)
            else:
                # take last num_bptt steps, pad if needed at front
                length = min(T, self.num_bptt)
                o_seg = ep['o'][T - length:T + 1]
                a_seg = ep['a'][T - length:T]
                r_seg = ep['r'][T - length:T]
                d_seg = ep['d'][T - length:T]
                pad = self.num_bptt - length
                if pad > 0:
                    o_pad = np.zeros((pad, self.obs_dim), dtype=np.float32)
                    a_pad = np.zeros((pad,) + self.action_shape, dtype=np.int32 if self.is_discrete_action else np.float32)
                    r_pad = np.zeros((pad,), dtype=np.float32)
                    d_pad = np.ones((pad,), dtype=np.float32)  # treat padded steps as terminal/masked
                    m_pad = np.zeros((pad,), dtype=np.float32)
                    o_seg = np.concatenate([o_pad, o_seg], axis=0)
                    a_seg = np.concatenate([a_pad, a_seg], axis=0)
                    r_seg = np.concatenate([r_pad, r_seg], axis=0)
                    d_seg = np.concatenate([d_pad, d_seg], axis=0)
                    m_seg = np.concatenate([m_pad, np.ones((length,), dtype=np.float32)], axis=0)
                else:
                    m_seg = np.ones((self.num_bptt,), dtype=np.float32)
            o_list.append(o_seg)
            a_list.append(a_seg)
            r_list.append(r_seg)
            d_list.append(d_seg)
            m_list.append(m_seg)
        batch = {
            'o': np.stack(o_list, axis=0),                 # [B, T+1, obs_dim]
            'a': np.stack(a_list, axis=0),                 # [B, T, *action_shape]
            'r': np.stack(r_list, axis=0),                 # [B, T]
            'd': np.stack(d_list, axis=0),                 # [B, T]
            'm': np.stack(m_list, axis=0),                 # [B, T]
        }
        return batch

    def finalize_all_working(self):
        for i in range(self.num_envs):
            self._finalize_env_episode(i) 