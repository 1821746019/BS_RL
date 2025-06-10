import copy
from typing import Callable, Deque, List, Dict, Any, Optional
from TradingEnv import TradingEnv, wrappers, EnvConfig as TradingEnvConfig, DataLoader, Account
from gymnasium.wrappers import FlattenObservation
import collections
import numpy as np
import wandb

class StatsAggregator:
    """A helper class to aggregate episode statistics."""
    def __init__(self, maxlen: Optional[int] =  64):
        self.buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=maxlen)

    @staticmethod
    def _remap_and_filter(stats: dict) -> dict:
        remapped = {}
        if 'r' in stats: remapped['return'] = stats['r']
        if 'l' in stats: remapped['length'] = stats['l']
        if 'episode_ROI' in stats: remapped['roi'] = stats['episode_ROI']
        if 'sharpe_ratio' in stats: remapped['sharpe'] = stats['sharpe_ratio']
        if 'sortino_ratio' in stats: remapped['sortino'] = stats['sortino_ratio']
        if 'episode_MDD' in stats: remapped['mdd'] = stats['episode_MDD']
        if 'episode_MDD_24h' in stats: remapped['mdd_24h'] = stats['episode_MDD_24h']
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
            results[f"episodic_{key}_mean"] = np.mean(values)
            if len(values) > 1:
                results[f"episodic_{key}_std"] = np.std(values)
            results[f"episodic_{key}_max"] = np.max(values)
            results[f"episodic_{key}_min"] = np.min(values)
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
            log_dict = {f"{prefix}/episodic_{k}_env0": v for k, v in remapped.items()}
            wandb.log(log_dict, step=step)

def train_env_maker(seed: int, config: TradingEnvConfig, data_loader: DataLoader, capture_video: bool=False, run_name: str=None,capture_episode_trigger: Callable[[int], bool]=None):
   
    def thunk():
        account = Account(config)
        env = TradingEnv(config, data_loader, account, seed=seed)
        env = wrappers.DetailedRewardWrapper(env)
        env = wrappers.NormalizationWrapper(env)
        env = wrappers.EpisodeStats(env)
        env = FlattenObservation(env)
        return env

    return thunk

def eval_env_maker(seed: int, config: TradingEnvConfig, data_loader: DataLoader, capture_media: bool=False, run_name: str=None,capture_episode_trigger: Callable[[int], bool]=None):
   
    def thunk():
        account = Account(config)
        env = TradingEnv(config, data_loader, account, seed=seed)
        env = wrappers.DetailedRewardWrapper(env)
        env = wrappers.NormalizationWrapper(env)
        env = wrappers.EpisodeStats(env)
        if capture_media:
            env = wrappers.EpisodeRender(env)
        env = FlattenObservation(env)
        return env

    return thunk
