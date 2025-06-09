import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .agent import SACAgent
from .common import eval_env_maker
from .config import EnvConfig, EvalConfig
from TradingEnv import DataLoader

class Evaluator:
    def __init__(self,
                 agent: SACAgent,
                 env_config: EnvConfig,
                 eval_config: EvalConfig,
                 data_loader: DataLoader,
                 run_name_suffix: str):
        self.agent = agent
        self.env_config = env_config
        self.eval_config = eval_config
        self.data_loader = data_loader
        self.run_name_suffix = run_name_suffix
        self.eval_envs = None

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
                    # Capture video for the first eval env if configured
                    capture_video=self.eval_config.capture_video and i == 0,
                    run_name=f"{self.run_name_suffix}_eval", # Simplified run name
                    capture_episode_trigger=lambda e: e == 0
                )
                for i in range(self.eval_config.env_num)
            ]
        )

    def evaluate(self, actor_params_eval: flax.core.FrozenDict, current_train_step: int):
        num_episodes = self.eval_config.eval_episodes
        print(f"Starting evaluation for {num_episodes} episodes with seed {self.eval_config.seed} at step {current_train_step}...")

        eval_envs = self.eval_envs
        envs_were_created_here = False
        if eval_envs is None:
            eval_envs = self._make_envs()
            envs_were_created_here = True

        episode_returns_unclipped = []
        episode_lengths_unclipped = []
        key_eval_actions = jax.random.PRNGKey(self.eval_config.seed)

        obs, _ = eval_envs.reset(seed=self.eval_config.seed + current_train_step)

        while len(episode_returns_unclipped) < num_episodes:
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
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        unclipped_return = info["episode"]["r"]
                        unclipped_length = info["episode"]["l"]
                        episode_returns_unclipped.append(unclipped_return)
                        episode_lengths_unclipped.append(unclipped_length)
                        print(f"Eval Episode {len(episode_returns_unclipped)}/{num_episodes}: Return={unclipped_return:.2f}, Length={unclipped_length}")
                        if len(episode_returns_unclipped) >= num_episodes:
                            break
        
        if envs_were_created_here:
            eval_envs.close()

        if episode_returns_unclipped:
            mean_return = np.mean(episode_returns_unclipped)
            std_return = np.std(episode_returns_unclipped)
        else:
            mean_return, std_return = 0.0, 0.0

        print(f"Evaluation finished: Mean Return={mean_return:.2f} +/- {std_return:.2f}")
        return {
            "episodic_return_mean": mean_return,
            "episodic_return_std": std_return,
            "episodic_length_mean": np.mean(episode_lengths_unclipped) if episode_lengths_unclipped else 0.0,
        }

    def close(self):
        if self.eval_envs:
            self.eval_envs.close()
            self.eval_envs = None