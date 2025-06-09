import copy
from typing import Callable
from TradingEnv import TradingEnv, wrappers, EnvConfig as TradingEnvConfig, DataLoader, Account
from gymnasium.wrappers import FlattenObservation
def train_env_maker(seed: int, config: TradingEnvConfig, data_loader: DataLoader, capture_video: bool=False, run_name: str=None,capture_episode_trigger: Callable[[int], bool]=None):
   
    def thunk():
        account = Account(config)
        env = TradingEnv(config, data_loader, account, seed=seed)
        env = wrappers.EpisodeTruncationWrapper(env)
        env = wrappers.DetailedRewardWrapper(env,config)
        env = wrappers.NormalizationWrapper(env)
        env = FlattenObservation(env)
        return env

    return thunk

def eval_env_maker(seed: int, config: TradingEnvConfig, data_loader: DataLoader, capture_video: bool=False, run_name: str=None,capture_episode_trigger: Callable[[int], bool]=None):
   
    def thunk():
        account = Account(config)
        env = TradingEnv(config, data_loader, account, seed=seed)
        env = wrappers.EpisodeTruncationWrapper(env)
        env = wrappers.DetailedRewardWrapper(env, config)
        env = wrappers.NormalizationWrapper(env)
        env = FlattenObservation(env)
        env = wrappers.EpisodeStats(env)
        env = wrappers.EpisodeRender(env)
        return env

    return thunk
