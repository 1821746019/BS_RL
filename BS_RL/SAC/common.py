import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from .agent import SACAgent
from .config import EnvConfig
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv


def make_env(env_id, seed, idx, capture_video, run_name, num_envs=1, env_config: dict = None):
    """
    Creates a thunk for a Gymnasium environment with Atari preprocessing.
    If env_id is CartPole-v1, it creates a simpler environment.
    """
    def thunk():
        # Determine if this is an Atari-like environment
        is_atari_like = "NoFrameskip" in env_id
        env_wrapped_with_record_stats = False # Keep track if RecordEpisodeStatistics is applied

        # Create the base environment
        if capture_video and idx == 0: # Video recording for the first env
            env = gym.make(env_id, render_mode="rgb_array")
            # RecordVideo will be applied later, after observation wrappers
        else:
            env = gym.make(env_id)

        # Apply RecordEpisodeStatistics EARLY if we want unclipped rewards in logs for Atari
        # For non-Atari, apply it later, as ClipRewardEnv might not be used or relevant.
        if is_atari_like:
            env = gym.wrappers.RecordEpisodeStatistics(env) # Wrap early for original scores
            env_wrapped_with_record_stats = True

        # Apply Atari-specific wrappers first if it's an Atari env
        if is_atari_like:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4) # Outputs (4, 84, 84) NCHW format
        elif "CartPole-v1" == env_id:
            # For CartPole, no complex observation wrappers are typically needed before recording
            pass # Add any specific wrappers for CartPole if necessary before video recording
        else:
            # For other envs, ensure any observation modifications happen here
            pass

        # Ensure RecordEpisodeStatistics is applied if not already (e.g. for non-Atari envs)
        # This allows evaluate_agent to consistently use info['episode'] for stats.
        if not env_wrapped_with_record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)

        # Now, apply RecordVideo if enabled, after all observation modifications
        if capture_video and idx == 0:
            video_folder = f"videos/{run_name}"
            # Ensure the folder exists, RecordVideo might not create it.
            # os.makedirs(video_folder, exist_ok=True) # Handled by RecordVideo in newer gym
            env = gym.wrappers.RecordVideo(env, video_folder)

        env.action_space.seed(seed + idx)
        # env.observation_space.seed(seed + idx) # Not standard for obs space
        return env

    return thunk


def evaluate_agent(
    agent_eval: SACAgent, # The agent instance
    actor_params_eval: flax.core.FrozenDict, # Current actor parameters for eval
    eval_env_config: EnvConfig, # Env config for evaluation (can be different from training)
    num_episodes: int,
    seed: int,
    greedy_actions: bool,
    run_name_suffix_eval: str, # For video naming
    current_train_step: int # For logging context
):
    print(f"Starting evaluation for {num_episodes} episodes with seed {seed}...")
    eval_envs = gym.vector.AsyncVectorEnv([
        make_env(
            eval_env_config.env_id,
            seed + i,
            i,
            # Capture video for the first eval env if configured
            capture_video=eval_env_config.capture_video and i == 0,
            run_name=f"{run_name_suffix_eval}_eval_step{current_train_step}",
            num_envs=1 # Assuming eval runs one env at a time for simplicity in this helper
        ) for i in range(1) # Simplified to 1 eval env for now, can be >1 if needed
    ])
    # If using multiple eval_envs, aggregation logic would be similar to training log.

    episode_returns_unclipped = []
    episode_lengths_unclipped = []
    key_eval_actions = jax.random.PRNGKey(seed) # Fresh key for eval actions

    for episode_idx in range(num_episodes):
        obs, _ = eval_envs.reset(seed=seed + episode_idx) # Ensure different start for each episode
        
        done_current_episode = False
        while not done_current_episode:
            key_eval_actions, key_step = jax.random.split(key_eval_actions)
            jax_obs = jnp.asarray(obs) # Convert current obs (numpy) to JAX array

            actions_jax = agent_eval.select_action(actor_params_eval, jax_obs, key_step, deterministic=greedy_actions)
            actions_numpy = np.array(jax.device_get(actions_jax)) # Convert to numpy for env.step

            next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions_numpy)
            
            # Assuming eval_envs.num_envs is 1 as per current setup in make_env for evaluation
            obs = next_obs
            
            terminated_this_step = terminations[0]
            truncated_this_step = truncations[0]
            info_this_step = infos[0]

            if terminated_this_step or truncated_this_step:
                # Episode has ended for the first (and only) environment
                if "episode" in info_this_step:
                    unclipped_return = info_this_step["episode"]["r"]
                    unclipped_length = info_this_step["episode"]["l"]
                    episode_returns_unclipped.append(unclipped_return)
                    episode_lengths_unclipped.append(unclipped_length)
                    print(f"Eval Episode {episode_idx + 1}/{num_episodes}: Unclipped Return={unclipped_return}, Length={unclipped_length}")
                else:
                    # This case should ideally not happen if RecordEpisodeStatistics is working correctly.
                    print(f"Warning: 'episode' key not found in infos for episode {episode_idx + 1} upon termination. Stats might be incomplete.")
                done_current_episode = True # Break from while loop for this episode
        # End of while loop (single episode)
    # End of for loop (all episodes)

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

