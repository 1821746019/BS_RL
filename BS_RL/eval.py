import copy
import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .SACAgent import RSACAgent
from .common import env_maker, MetricLogger, StatsAggregator
from .config import EnvConfig, EvalConfig
from TradingEnv import DataLoaderConfig, TradingEnvConfig
from datetime import datetime, timedelta
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
        # Parse start/end once
        start_str, end_str = self.env_config.trading_env_config.test_timerange
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        total_days = max(0, (end_dt - start_dt).days)
        env_num = max(1, self.eval_config.env_num)
        # Use integer-day slices and ensure >= 1 day
        slice_days = max(1, total_days // env_num) if total_days > 0 else 1
        envs = []
        for i in range(env_num):
            trading_env_cfg = copy.deepcopy(self.env_config.trading_env_config)
            start_i = start_dt + timedelta(days=i * slice_days)
            end_i = start_dt + timedelta(days=(i + 1) * slice_days)
            # Stop if start exceeds or meets global end
            if start_i >= end_dt:
                break
            # Clamp end_i to global end
            if end_i > end_dt:
                end_i = end_dt
            # Ensure non-empty interval
            if start_i >= end_i:
                break
            trading_env_cfg.test_timerange = (start_i.strftime("%Y-%m-%d"), end_i.strftime("%Y-%m-%d"))
            env = env_maker(
                config=trading_env_cfg,
                data_loader_cfg=self.env_config.data_loader_cfg,
                feat_getter=self.env_config.feat_getter,
                capture_media=self.eval_config.capture_media,
                random_choose_tickers=env_num > 1,
                tickers_per_env=self.env_config.tickers_per_env,
            )
            envs.append(env)
        return eval_vec_env_cls(envs)

    def evaluate(self, actor_state_eval, encoder_params_eval, rsnorm_state_eval, current_train_step: int):
        num_episodes = self.eval_config.eval_episodes
        print(f"\nStarting evaluation for {num_episodes} episodes at step {current_train_step}...")

        eval_envs = self.eval_envs
        
        stats_aggregator = StatsAggregator(num_episodes) #防止默认大小64<num_episodes时下面的代码陷入死循环
        key_eval_actions = jax.random.PRNGKey(self.seed)

        obs, _ = eval_envs.reset(seed=self.seed + current_train_step)
        # init hidden states
        hidden_state = None

        while len(stats_aggregator.buffer) < num_episodes:
            actions_jax, new_hidden_state, key_eval_actions = self.agent.select_action_for_eval(
                actor_state_eval,
                encoder_params_eval,
                rsnorm_state_eval,
                jnp.asarray(obs),
                hidden_state,
                key_eval_actions,
                deterministic=self.eval_config.greedy_actions,
            )
            actions_numpy = np.array(jax.device_get(actions_jax))
            hidden_state = new_hidden_state

            next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions_numpy)
            obs = next_obs

            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if i == 0:
                            self.logger.log_env0_episode(info["episode"], current_train_step, prefix="eval")
                        
                        stats_aggregator.add(info["episode"])
                        r = info["episode"]["r"]
                        r = r[0] if isinstance(r, np.ndarray) and r.ndim == 1 else r
                        l = info["episode"]["l"]
                        l = l[0] if isinstance(l, np.ndarray) and l.ndim == 1 else l
                        print(f"Eval Episode {len(stats_aggregator.buffer)}/{num_episodes}: Return={r:.2f}, Length={l}")
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