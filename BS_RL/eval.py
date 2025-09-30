from flax.training import checkpoints
import os
import copy
from dataclasses import dataclass
from typing import Optional
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .SACAgent import RSACAgent
from .common import env_maker, MetricLogger, StatsAggregator
from .config import EnvConfig, EvalConfig, Args
from TradingEnv import DataLoaderConfig, TradingEnvConfig
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from .SACAgent import AgentState
@dataclass
class Evaluator:
    agent: RSACAgent
    env_config: EnvConfig
    eval_config: EvalConfig
    logger: MetricLogger
    eval_envs: Optional[AsyncVectorEnv| SyncVectorEnv] = None
    seed: int = 996007
    add_train_step_to_seed: bool = False
    "为False时，每次评估的环境初始状态相同"
    
    def __post_init__(self):
        if self.eval_envs is None:
            self.eval_envs = self._make_envs()
        self.env_config.trading_env_config = copy.deepcopy(self.env_config.trading_env_config)
        self.env_config.trading_env_config.mode = "test"
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

    def evaluate(self, actor_state_eval, encoder_params_eval, rsnorm_state_eval, curr_train_step: int):
        num_episodes = self.eval_config.eval_episodes
        print(f"\nStarting evaluation for {num_episodes} episodes at step {curr_train_step}...")

        eval_envs = self.eval_envs
        
        stats_aggregator = StatsAggregator(num_episodes) #防止默认大小64<num_episodes时下面的代码陷入死循环
        key_eval_actions = jax.random.PRNGKey(self.seed)

        obs, _ = eval_envs.reset(seed=self.seed + curr_train_step if self.add_train_step_to_seed else self.seed)
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

            if "final_info" not in infos: continue
            for i, info in enumerate(infos["final_info"]):
                if not info or "episode" not in info: continue
                if i == 0:
                    self.logger.log_env0_episode(info["episode"], curr_train_step, prefix="eval")
                
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
        self.logger.log_stats(eval_metrics, curr_train_step, "eval_buffered")
            
        return eval_metrics

    def close(self):
        self.eval_envs.close()
        
def eval(args: Args, ckpt_dir: Path):
    vec_env_cls = AsyncVectorEnv if args.eval.async_vector_env else SyncVectorEnv
    envs = vec_env_cls(
        [env_maker(
            config=args.env.trading_env_config,
            data_loader_cfg=args.env.data_loader_cfg,
            feat_getter=args.env.feat_getter,
            random_choose_tickers= args.env.env_num > 1,
            tickers_per_env=args.env.tickers_per_env
        ) for i in range(args.env.env_num)]
    )
    agent = RSACAgent(
            obs_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            network_config=args.network,
            algo_config=args.algo,
            obs_split_fn=args.env.obs_split_fn,
        )
    logger = MetricLogger()
    restore_target = {'agent_state': None}
    loaded_contents = checkpoints.restore_checkpoint(
                    ckpt_dir=ckpt_dir,
                    target=restore_target,
                    prefix="ckpt_step_"
    )
    agent_state = cast(AgentState, loaded_contents['agent_state'])
    print(f"Agent states restored from {ckpt_dir}.")
    
    evaluator = Evaluator(agent, args.env, args.eval, logger)
    evaluator.evaluate(agent_state.actor_state, agent_state.encoder_state.params, agent_state.rsnorm_state, 0)