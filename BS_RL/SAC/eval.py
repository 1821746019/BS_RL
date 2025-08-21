import copy
import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .agent import RSACAgent
from .common import eval_env_maker, MetricLogger, StatsAggregator
from .config import EnvConfig, EvalConfig
from TradingEnv import DataLoaderConfig, TradingEnvConfig

class Evaluator:
    def __init__(self,
                 agent: RSACAgent,
                 env_config: EnvConfig,
                 eval_config: EvalConfig,
                 run_name_suffix: str,
                 seed: int,
                 logger: MetricLogger):
        self.agent = agent
        self.env_config = env_config
        self.eval_config = eval_config
        self.run_name_suffix = run_name_suffix
        self.seed = seed
        self.logger = logger
        self.env_config.trading_env_config = copy.deepcopy(self.env_config.trading_env_config)
        self.env_config.trading_env_config.mode = "test"
        self.eval_envs = self._make_envs()
        

    def _make_envs(self):
        print("Creating evaluation environments...")
        eval_vec_env_cls = gym.vector.AsyncVectorEnv if self.eval_config.async_vector_env else gym.vector.SyncVectorEnv
        
        return eval_vec_env_cls(
            [
                eval_env_maker(
                    config=self.env_config.trading_env_config,
                    data_loader_cfg=self.env_config.data_loader_cfg,
                    feat_getter=self.env_config.feat_getter,
                    capture_media=self.eval_config.capture_media,
                )
                for i in range(self.eval_config.env_num)
            ]
        )

    def evaluate(self, actor_state_eval, summarizer_params_eval, current_train_step: int):
        num_episodes = self.eval_config.eval_episodes
        print(f"\nStarting evaluation for {num_episodes} episodes at step {current_train_step}...")

        eval_envs = self.eval_envs
        
        stats_aggregator = StatsAggregator(num_episodes) #防止默认大小64<num_episodes时下面的代码陷入死循环
        key_eval_actions = jax.random.PRNGKey(self.seed)

        obs, _ = eval_envs.reset(seed=self.seed + current_train_step)
        # init hidden states
        if self.agent.network_config.use_s5_summarizer:
            L = self.agent.network_config.s5_num_layers
            H = self.agent.network_config.s5_hidden_dim
        else:
            L = self.agent.network_config.lstm_num_layers
            H = self.agent.network_config.lstm_hidden_dim
        N = eval_envs.num_envs
        hidden_h = jnp.zeros((L, N, H), dtype=jnp.float32)
        hidden_c = jnp.zeros((L, N, H), dtype=jnp.float32)

        while len(stats_aggregator.buffer) < num_episodes:
            key_eval_actions, key_step = jax.random.split(key_eval_actions)
            actions_jax, new_h, new_c = self.agent.select_action(
                actor_state_eval, 
                summarizer_params_eval,
                jnp.asarray(obs), 
                hidden_h, hidden_c,
                key_step, 
                deterministic=self.eval_config.greedy_actions
            )
            actions_numpy = np.array(jax.device_get(actions_jax))
            hidden_h = new_h
            hidden_c = new_c

            next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions_numpy)
            obs = next_obs

            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if i == 0:
                            self.logger.log_env0_episode(info["episode"], current_train_step, prefix="eval")
                        
                        stats_aggregator.add(info["episode"])
                        episode_return = float(info["episode"]["r"])
                        episode_length = int(info["episode"]["l"])
                        print(f"Eval Episode {len(stats_aggregator.buffer)}/{num_episodes}: Return={episode_return:.2f}, Length={episode_length}")
                        if len(stats_aggregator.buffer) >= num_episodes:
                            break
        
        eval_metrics = stats_aggregator.get_aggregated_stats()

        mean_return = eval_metrics.get("episode_return_mean", 0.0)
        std_return = eval_metrics.get("episode_return_std", 0.0)
        print(f"Evaluation finished: Mean Return={mean_return:.2f} +/- {std_return:.2f}")
        self.logger.log_stats(eval_metrics, current_train_step, "eval_buffered")
            
        return eval_metrics

    def close(self):
        self.eval_envs.close()