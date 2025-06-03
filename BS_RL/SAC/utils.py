import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.linen as nn
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id, seed, idx, capture_video, run_name, num_envs=1, env_config: dict = None):
    """
    Creates a thunk for a Gymnasium environment with Atari preprocessing.
    If env_id is CartPole-v1, it creates a simpler environment.
    """
    def thunk():
        if "NoFrameskip" in env_id: # Atari
            if capture_video and idx == 0: # only video record the first environment
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env) # Records episodic statistics

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
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        else:
            # Fallback for other envs, may need specific wrappers
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed + idx)
        # env.observation_space.seed(seed + idx) # Not standard for obs space
        return env

    return thunk

# For network initialization, matching PyTorch's Kaiming Normal and constant bias
def kaiming_normal_initializer(key, shape, dtype=jnp.float32):
    # Flax's KaimingNormal is actually KaimingUniform in PyTorch terms if scale=sqrt(2)
    # PyTorch KaimingNormal uses fan_in. Flax default is fan_in.
    return jax.random.normal(key, shape, dtype) * jnp.sqrt(1.0 / shape[-2]) # Simplified for Dense/Conv

def constant_initializer(value):
    def init(key, shape, dtype=jnp.float32):
        return jnp.full(shape, value, dtype)
    return init

# Polyak averaging (already available in optax.incremental_update)
# def polyak_update(params, target_params, tau):
#     return jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), params, target_params)