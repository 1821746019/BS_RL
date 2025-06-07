import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .agent import SACAgent
from .common import make_env
from .config import EnvConfig, EvalConfig


def evaluate_agent(
    agent_eval: SACAgent, # The agent instance
    actor_params_eval: flax.core.FrozenDict, # Current actor parameters for eval
    env_config: EnvConfig, # Env config for evaluation (can be different from training)
    eval_config: EvalConfig, # Eval config for evaluation (can be different from training)
    num_episodes: int,
    greedy_actions: bool,
    run_name_suffix_eval: str, # For video naming
    current_train_step: int, # For logging context
    eval_envs: gym.vector.VectorEnv = None
):
    print(f"Starting evaluation for {num_episodes} episodes with seed {eval_config.seed}...")

    envs_were_created_here = False
    if eval_envs is None:
        print("Creating evaluation environments for this run (not cached).")
        envs_were_created_here = True
        eval_vec_env_cls = gym.vector.AsyncVectorEnv if eval_config.async_vector_env else gym.vector.SyncVectorEnv
        eval_envs = eval_vec_env_cls(
            [
                make_env(
                    env_config.env_id,
                    eval_config.seed + i,
                    i,
                    # Capture video for the first eval env if configured
                    capture_video=eval_config.capture_video and i == 0,
                    run_name=f"{run_name_suffix_eval}_eval_step{current_train_step}",
                    capture_episode_trigger=lambda e:e==0
                )
                for i in range(eval_config.env_num)
            ]
        )

    episode_returns_unclipped = []
    episode_lengths_unclipped = []
    key_eval_actions = jax.random.PRNGKey(eval_config.seed) # Fresh key for eval actions

    obs, _ = eval_envs.reset(seed=eval_config.seed + current_train_step)

    while len(episode_returns_unclipped) < num_episodes:
        key_eval_actions, key_step = jax.random.split(key_eval_actions)
        jax_obs = jnp.asarray(obs) # Convert current obs (numpy) to JAX array

        actions_jax = agent_eval.select_action(actor_params_eval, jax_obs, key_step, deterministic=greedy_actions)
        actions_numpy = np.array(jax.device_get(actions_jax)) # Convert to numpy for env.step

        next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions_numpy)
        obs = next_obs

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    unclipped_return = info["episode"]["r"]
                    unclipped_length = info["episode"]["l"]
                    episode_returns_unclipped.append(unclipped_return)
                    episode_lengths_unclipped.append(unclipped_length)
                    print(f"Eval Episode {len(episode_returns_unclipped)}/{num_episodes}: Unclipped Return={unclipped_return}, Length={unclipped_length}")
                    if len(episode_returns_unclipped) >= num_episodes:
                        break # Break from inner info loop

    if envs_were_created_here:
        eval_envs.close()

    if episode_returns_unclipped: # Check if list is not empty
        mean_return = np.mean(episode_returns_unclipped)
        std_return = np.std(episode_returns_unclipped)
        min_return = np.min(episode_returns_unclipped)
        max_return = np.max(episode_returns_unclipped)
        mean_length = np.mean(episode_lengths_unclipped)
    else: # Handle case where no episodes were successfully recorded
        print("Warning: No episode statistics were recorded during evaluation. Returning zero statistics.")
        mean_return, std_return, min_return, max_return, mean_length = 0.0, 0.0, 0.0, 0.0, 0.0

    print(f"Evaluation finished: Mean Unclipped Return={mean_return:.2f}, Std Unclipped Return={std_return:.2f}")
    return {
        "episodic_return_mean": mean_return,
        "episodic_return_std": std_return,
        "episodic_return_min": min_return,
        "episodic_return_max": max_return,
        "episodic_length_mean": mean_length,
    }