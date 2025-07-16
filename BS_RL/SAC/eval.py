import copy
import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .agent import SACAgentBase
from .common import eval_env_maker, MetricLogger, StatsAggregator
from .config import EnvConfig, EvalConfig
from TradingEnv import DataLoader, DataLoaderConfig, TradingEnvConfig

class Evaluator:
    def __init__(self,
                 agent: SACAgentBase,
                 env_config: EnvConfig,
                 eval_config: EvalConfig,
                 run_name_suffix: str,
                 logger: MetricLogger = None):
        self.agent = agent
        self.env_config = env_config
        self.eval_config = eval_config
        self.run_name_suffix = run_name_suffix
        self.logger = logger
        self.eval_envs = None
        trading_env_config = copy.deepcopy(self.env_config.trading_env_config)
        trading_env_config.data_loader_config.mode = "test"
        self.data_loader = DataLoader(trading_env_config.data_loader_config)
        self.env_config.trading_env_config = trading_env_config
        if self.eval_config.cache_env:
            self.eval_envs = self._make_envs()

    def _make_envs(self):
        print("Creating evaluation environments...")
        eval_vec_env_cls = gym.vector.AsyncVectorEnv if self.eval_config.async_vector_env else gym.vector.SyncVectorEnv
        
        # Note: The `make_env` was changed to `eval_env_maker` which has a different signature.
        # We now pass the dataloader to it.
        return eval_vec_env_cls(
            [
                eval_env_maker(
                    seed=self.eval_config.seed + i,
                    config=self.env_config.trading_env_config,
                    data_loader=self.data_loader,
                    # Capture media for the first eval env if configured
                    capture_media=self.eval_config.capture_media,# 解除注释则只对env0绘制图片and i == 0,
                    run_name=f"{self.run_name_suffix}_eval", # Simplified run name
                    capture_episode_trigger=lambda e: e == 0
                )
                for i in range(self.eval_config.env_num)
            ]
        )

    def evaluate(self, actor_params_eval: flax.core.FrozenDict, current_train_step: int):
        num_episodes = self.eval_config.eval_episodes
        print(f"\nStarting evaluation for {num_episodes} episodes with seed {self.eval_config.seed} at step {current_train_step}...")

        eval_envs = self.eval_envs
        envs_were_created_here = False
        if eval_envs is None:
            eval_envs = self._make_envs()
            envs_were_created_here = True
        
        stats_aggregator = StatsAggregator(num_episodes) #防止默认大小64<num_episodes时下面的代码陷入死循环
        key_eval_actions = jax.random.PRNGKey(self.eval_config.seed)

        obs, _ = eval_envs.reset(seed=self.eval_config.seed + current_train_step)

        while len(stats_aggregator.buffer) < num_episodes:
            key_eval_actions, key_step = jax.random.split(key_eval_actions)
            
            actions_jax = self.agent.select_action(
                actor_params_eval, 
                jnp.asarray(obs), 
                key_step, 
                deterministic=self.eval_config.greedy_actions
            )
            actions_numpy = np.array(jax.device_get(actions_jax))

            next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions_numpy)
            obs = next_obs

            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if i == 0 and self.logger:
                            self.logger.log_env0_episode(info["episode"], current_train_step, prefix="eval")
                        
                        stats_aggregator.add(info["episode"])
                        episode_return = float(info["episode"]["r"])
                        episode_length = int(info["episode"]["l"])
                        print(f"Eval Episode {len(stats_aggregator.buffer)}/{num_episodes}: Return={episode_return:.2f}, Length={episode_length}")
                        if len(stats_aggregator.buffer) >= num_episodes:
                            break
        
        if envs_were_created_here:
            eval_envs.close()

        eval_metrics = stats_aggregator.get_aggregated_stats()

        mean_return = eval_metrics.get("episode_return_mean", 0.0)
        std_return = eval_metrics.get("episode_return_std", 0.0)
        print(f"Evaluation finished: Mean Return={mean_return:.2f} +/- {std_return:.2f}")

        if self.logger:
            self.logger.log_stats(eval_metrics, current_train_step, "eval_buffered")
            
        return eval_metrics

    def close(self):
        if self.eval_envs:
            self.eval_envs.close()
            self.eval_envs = None