import gymnasium as gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv
from typing import Callable


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str,capture_episode_trigger: Callable[[int], bool]=None):
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
        env.action_space.seed(seed) # 不进行随机动作采样，基本没有用
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
            env = gym.wrappers.RecordVideo(env, video_folder,episode_trigger=capture_episode_trigger)

        return env

    return thunk



