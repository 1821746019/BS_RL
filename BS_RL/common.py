from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import dataclasses
from typing import Callable, Deque, List, Dict, Any, Optional, Iterable
from TradingEnv import TradingEnv, TradingEnvConfig , DataLoader, Account, DataLoaderConfig, DataLoaderWithCache, Ticker
from TradingEnv import wrappers, StartMode
import collections
import jax
import numpy as np
import wandb
import gymnasium
from BS_RL.config import ENABLE_PROFILE, USE_JAX_PROFILER
from TradingEnv.feature import norm_OHLCV
jax_jit: Any = jax.jit # 为了让调用jitWrapped函数时IDE能提供正常的IntelliSense

try:
    import line_profiler
    profile: Callable = line_profiler.profile
except:
    profile = lambda x: x
class CallableFalse:
    def __bool__(self):
        """在布尔上下文中返回False"""
        return False
    
    def __call__(self, *args, **kwargs):
        """使对象可调用"""
        return None
class Profiler: # type: ignore
    def __init__(self):
        print("pyinstrument Profiler is disabled")
    def __getattr__(self, name: str, /, cached_ret = CallableFalse()) -> CallableFalse:
        return cached_ret
if ENABLE_PROFILE:
    from pyinstrument import Profiler as PyInstrumentProfiler
    Profiler: type[PyInstrumentProfiler] = PyInstrumentProfiler # 需要类型注解才能提供正确的IntelliSense
jax_profiler = jax.profiler
if  not USE_JAX_PROFILER:
    class JaxProfiler:
        def __init__(self):
            print("jax.profiler is disabled")
        def start_trace(self, *args, **kwargs):
            pass
        def stop_trace(self, *args, **kwargs):
            pass
    jax_profiler: type[jax.profiler] = JaxProfiler() # type: ignore
    
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

    def get_aggregated_stats(self, prefix: str = "") -> Dict[str, float]:
        if not self.buffer:
            return {}

        aggregated = collections.defaultdict(list)
        for stats in self.buffer:
            for key, value in stats.items():
                # 只跳过复杂的容器类型（如list, dict, tuple），但保留字符串和数值类型
                if isinstance(value, (list, dict, tuple, set)):
                    continue
                try:
                    # 尝试转换为float，如果失败则跳过该值
                    aggregated[key].append(float(value))
                except (ValueError, TypeError):
                    continue

        results = {}
        for key, values in aggregated.items():
            if not values: continue
            results[f"{prefix}{key}_mean"] = np.mean(values)
            # if len(values) > 1:
            #     results[f"{key}_std"] = np.std(values)
            results[f"{prefix}{key}_max"] = np.max(values)
            results[f"{prefix}{key}_min"] = np.min(values)
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

class BS_SyncVectorEnv(gymnasium.vector.SyncVectorEnv):
    def __init__(
        self,
        env_fns: Iterable[Callable[[], gymnasium.Env]],
        observation_space: gymnasium.Space = None,
        action_space: gymnasium.Space = None,
        copy: bool = True,
    ):
        self.env_fns = env_fns
        with ThreadPoolExecutor() as executor:
            self.envs = list(executor.map(lambda fn: fn(), env_fns))
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        
        # gymnasium.vector.SyncVectorEnv.__init__ calls gymnasium.vector.VectorEnv.__init__
        # We are re-implementing SyncVectorEnv.__init__ to parallelize env creation.
        # So we need to call VectorEnv.__init__ directly.
        gymnasium.vector.VectorEnv.__init__(
            self,
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = gymnasium.vector.utils.create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

def env_maker(config: TradingEnvConfig, data_loader_cfg: DataLoaderConfig, feat_getter: Callable, capture_media: bool=False, random_choose_tickers: bool=False, tickers_per_env: int=1):
    # 将env的tickers限制在tickers_per_env
    data_loader_cfg = dataclasses.replace(data_loader_cfg, tickers=data_loader_cfg.tickers[:tickers_per_env])
    # 若随机选tickers，则从Ticker枚举列表中选
    if random_choose_tickers:
        tickers_new = tuple(np.random.choice(list(Ticker), tickers_per_env, replace=False))
        data_loader_cfg = dataclasses.replace(data_loader_cfg, tickers=tickers_new)
    def thunk():
        data_loader = DataLoaderWithCache(data_loader_cfg, feat_getter)
        account = Account(config, tickers=data_loader_cfg.tickers)
        env = TradingEnv(config, data_loader, account)
        env = wrappers.LossLimit(env)
        env = wrappers.EpisodeWrapper(env)
        env = wrappers.RandomWrapper(env)
        if capture_media:
            env = wrappers.EpisodeRender(env)
        env = wrappers.ObsWrapper(env)
        env = wrappers.DiscreteAction(env)
        # 值爆炸时并没有触发断言，说明不是obs含inf导致的，可以注释掉了
        # env = FiniteCheck(env)
        return env

    return thunk

def gym_train_env_maker(env_id: str, seed: int, capture_video: bool = False, run_name: str|None = None):
    """创建标准gym环境的训练环境制造函数"""
    def thunk():
        env = gymnasium.make(env_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env) # type: ignore
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def gym_eval_env_maker(env_id: str, seed: int, capture_video: bool = False, run_name: str|None = None):
    """创建标准gym环境的评估环境制造函数"""
    def thunk():
        # 如果需要录制视频，指定render_mode
        render_mode = "rgb_array" if capture_video else None
        try:
            env = gymnasium.make(env_id, render_mode=render_mode)
        except Exception as e: # some envs don't support render_mode params
            env = gymnasium.make(env_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env) # type: ignore
        if capture_video and run_name:
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}") # type: ignore
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class FiniteCheck(gymnasium.Wrapper):
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        assert np.isfinite(obs).all(), "obs contains non-finite"
        assert np.isfinite(float(r)), "reward non-finite"
        return obs, r, term, trunc, info


def valid_step_to_gamma(valid_step: int):
    return 1-1/valid_step