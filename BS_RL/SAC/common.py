import copy
from typing import Callable, Deque, List, Dict, Any, Optional
from TradingEnv import TradingEnv, TradingEnvConfig as TradingEnvConfig, DataLoader, Account
from TradingEnv import wrappers
import collections
import numpy as np
import wandb
import gymnasium

class StatsAggregator:
    """A helper class to aggregate episode statistics."""
    def __init__(self, maxlen: Optional[int] =  64):
        self.buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=maxlen)

    @staticmethod
    def _remap_and_filter(stats: dict) -> dict:
        remapped = {}
        keys_to_rename = {
            'r': 'return',
            'l': 'length',
        }
        keys_to_exclude = set(['t'])
        for metric,value in stats.items():
            if metric in keys_to_exclude:
                continue
            if metric in keys_to_rename:
                remapped[keys_to_rename[metric]] = value
            else:
                remapped[metric] = value
        return remapped

    def add(self, episode_info: dict):
        # The episode_info directly comes from the env's "episode" key
        # We handle both raw info and potentially remapped info
        if "episode" in episode_info:
            stats_to_process = episode_info["episode"]
        else:
            stats_to_process = episode_info
            
        remapped = self._remap_and_filter(stats_to_process)
        if remapped:
            self.buffer.append(remapped)

    def get_aggregated_stats(self) -> Dict[str, float]:
        if not self.buffer:
            return {}

        aggregated = collections.defaultdict(list)
        for stats in self.buffer:
            for key, value in stats.items():
                aggregated[key].append(float(value))

        results = {}
        for key, values in aggregated.items():
            if not values: continue
            results[f"{key}_mean"] = np.mean(values)
            # if len(values) > 1:
            #     results[f"{key}_std"] = np.std(values)
            results[f"{key}_max"] = np.max(values)
            results[f"{key}_min"] = np.min(values)
        return results

    def clear(self):
        self.buffer.clear()

class MetricLogger:
    """A helper class to log metrics to wandb."""
    def __init__(self, wandb_track: bool):
        self.wandb_track = wandb_track

    def log_stats(self, stats: dict, step: int, prefix: str):
        if not self.wandb_track or not stats:
            return
        
        log_dict = {f"{prefix}/{k}": v for k, v in stats.items()}
        wandb.log(log_dict, step=step)
    
    def log_env0_episode(self, episode_info: dict, step: int, prefix: str):
        if not self.wandb_track or not episode_info:
            return
        
        remapped = StatsAggregator._remap_and_filter(episode_info)
        if remapped:
            log_dict = {f"{prefix}/{k}_env0": v for k, v in remapped.items()}
            wandb.log(log_dict, step=step)

def train_env_maker(seed: int, config: TradingEnvConfig, data_loader: DataLoader, capture_video: bool=False, run_name: str=None,capture_episode_trigger: Callable[[int], bool]=None):
   
    def thunk():
        account = Account(config)
        env = TradingEnv(config, data_loader, account)
        env = wrappers.EpisodeWrapper(env)
        env = wrappers.TrainWrapper(env)
        env = wrappers.ObsWrapper(env)
        # 值爆炸时并没有触发断言，说明不是obs含inf导致的，可以注释掉了
        # env = FiniteCheck(env)
        return env

    return thunk

def eval_env_maker(seed: int, config: TradingEnvConfig, data_loader: DataLoader, capture_media: bool=True, run_name: str=None,capture_episode_trigger: Callable[[int], bool]=None):
   
    def thunk():
        account = Account(config)
        env = TradingEnv(config, data_loader, account)
        env = wrappers.EpisodeWrapper(env)
        if capture_media:
            env = wrappers.EpisodeRender(env)
        env = wrappers.ObsWrapper(env)
        # 值爆炸时并没有触发断言，说明不是obs含inf导致的，可以注释掉了
        # env = FiniteCheck(env)
        return env

    return thunk

def gym_train_env_maker(env_id: str, seed: int, capture_video: bool = False, run_name: str = None):
    """创建标准gym环境的训练环境制造函数"""
    def thunk():
        env = gymnasium.make(env_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def gym_eval_env_maker(env_id: str, seed: int, capture_video: bool = False, run_name: str = None):
    """创建标准gym环境的评估环境制造函数"""
    def thunk():
        # 如果需要录制视频，指定render_mode
        render_mode = "rgb_array" if capture_video else None
        env = gymnasium.make(env_id, render_mode=render_mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        if capture_video and run_name:
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class FiniteCheck(gymnasium.Wrapper):
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        assert np.isfinite(obs).all(), "obs contains non-finite"
        assert np.isfinite(r), "reward non-finite"
        return obs, r, term, trunc, info