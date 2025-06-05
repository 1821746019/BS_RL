import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax.training import checkpoints # For loading checkpoints

from .config import EnvConfig, AlgoConfig # Re-use parts of config if useful
from .agent import SACAgent
from .networks import ActorCNN, CriticCNN, ActorMLP, CriticMLP # For model class selection
from .common import make_env

@dataclass
class InferArgs:
    ckpt_path: str
    """Path to the model checkpoint file (e.g., runs/EXP_NAME/ckpts/ckpt_step_100000)."""
    env_id: str = "CartPole-v1"
    """The id of the environment to run inference on."""
    num_episodes: int = 10
    """Number of episodes to run for inference."""
    seed: int = 42
    """Seed for the environment and JAX PRNG key."""
    greedy: bool = True
    """Whether to use greedy action selection (deterministic)."""
    capture_video: bool = True
    """Whether to capture videos of the agent's performance."""
    video_save_dir: str = "videos_infer"
    """Directory to save inference videos."""
    run_name: Optional[str] = None
    """Optional run name for video subfolder. Defaults to env_id + timestamp."""
    jax_platform_name: Optional[str] = "cpu" # Default to CPU for inference
    """The JAX platform to run on ('cpu', 'gpu', 'tpu', or None for JAX default)."""
    # For agent/network init, we might need some AlgoConfig defaults if not implicitly handled
    # For now, SACAgent init requires algo_config for alpha, tau etc., which are not used in pure actor inference
    # We can pass a minimal or default AlgoConfig.
    use_atari_cnn: Optional[bool] = None # If None, infer from env_id

def infer(args: InferArgs):
    if args.jax_platform_name:
        jax.config.update('jax_platform_name', args.jax_platform_name)
        print(f"JAX platform set to: {jax.config.jax_platform_name}")

    # Run name for video saving
    if args.run_name is None:
        current_time = int(time.time())
        infer_run_name = f"{args.env_id}__{current_time}"
    else:
        infer_run_name = args.run_name
    
    # Ensure video_save_dir exists
    Path(args.video_save_dir).mkdir(parents=True, exist_ok=True)

    # PRNG Key
    key = jax.random.PRNGKey(args.seed)
    key_agent_init, key_actions = jax.random.split(key)

    # Environment Setup
    # We use a single environment for inference.
    # make_env needs a run_name for video path: videos/{run_name}/...
    # So, the actual video path will be: args.video_save_dir / infer_run_name / *.mp4
    # To achieve this, we can tell make_env that its "base" video dir is args.video_save_dir
    # and then the run_name is the subfolder.
    # Let's adjust make_env or how we call it if necessary.
    # For now, make_env creates f"videos/{run_name}". We want f"{args.video_save_dir}/{infer_run_name}"
    # One way is to ensure make_env's video path is relative to a specific root if capture_video=True
    # Or, pass full path to RecordVideo if we modify make_env.
    # Current make_env: gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # Let's assume for now make_env will use a subfolder "videos" in the current working dir.
    # To control this, we might need to pass the video folder to make_env or make_env's behavior configurable.
    
    # For simplicity, let make_env save to its default `videos/{infer_run_name}` relative to CWD
    # and then we can inform user. If more control is needed, make_env can be refactored.
    # If `args.capture_video` is True, videos will be in `videos/{infer_run_name}`
    
    print(f"Setting up environment: {args.env_id}")
    # Create a dummy EnvConfig for make_env, as it expects it.
    # The important parts for make_env are env_id, seed, idx, capture_video, run_name.
    # num_envs in make_env is not directly used, it's more for the SyncVectorEnv context.
    envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id=args.env_id, 
            seed=args.seed, 
            idx=0, # Single env for inference
            capture_video=args.capture_video, 
            run_name=infer_run_name, # This will create videos/infer_run_name
            # num_envs=1 # Not directly used by make_env for single thunk
            # video_output_folder=args.video_save_dir # If make_env supported this
        )
    ])
    # If videos are captured, they will be in a subfolder of 'videos' directory, e.g., 'videos/CartPole-v1__1678886400'
    if args.capture_video:
        print(f"Captured videos will be saved in a subdirectory under: {Path('videos') / infer_run_name}")


    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space supported."
    action_dim = envs.single_action_space.n
    
    # Determine observation shape and network type
    obs_space_shape = envs.single_observation_space.shape
    if args.use_atari_cnn is None:
        is_atari = "NoFrameskip" in args.env_id
    else:
        is_atari = args.use_atari_cnn

    if is_atari:
        # For Atari, obs_space_shape from FrameStack is (C, H, W), e.g. (4, 84, 84)
        # Networks expect (Batch, C, H, W)
        actor_model_cls = ActorCNN
        critic_model_cls = CriticCNN # Needed for SACAgent init, though not used by actor
        print(f"Using CNN networks for Atari-like environment. Obs shape: {obs_space_shape}")
    else:
        # For MLP, obs_space_shape is (Features,), e.g. (4,) for CartPole
        # Networks expect (Batch, Features)
        actor_model_cls = ActorMLP
        critic_model_cls = CriticMLP
        print(f"Using MLP networks for environment. Obs shape: {obs_space_shape}")

    # Agent Initialization
    # SACAgent requires algo_config. We create a minimal one as Q-funcs and alpha are not used for inference.
    # The key parts of AlgoConfig for SACAgent init are:
    # - autotune (determines if log_alpha_state is created)
    # - fixed_alpha (if not autotune)
    # - target_entropy_scale (if autotune)
    # - policy_lr, q_lr, adam_eps for optimizer creation (actor only needs policy_lr if we re-create state)
    # For inference, we only load actor_state.params, so optimizer state for actor is not strictly needed
    # if we directly use the loaded params.
    # However, SACAgent constructor will initialize TrainStates for actor, q_functions, and log_alpha.
    
    # Let's use default AlgoConfig, it should be fine as we overwrite actor_state.
    dummy_algo_config = AlgoConfig()

    agent = SACAgent(
        action_dim=action_dim,
        observation_space_shape=obs_space_shape, # Pass the original shape
        key=key_agent_init,
        algo_config=dummy_algo_config, # Pass a default or minimal config
        actor_model_cls=actor_model_cls,
        critic_model_cls=critic_model_cls # Required by SACAgent constructor
    )
    print("SACAgent initialized for inference.")

    # Load Checkpoint
    # The checkpoint saved by train.py is a dictionary of TrainStates.
    # We need to restore actor_state.params into agent.actor_state.
    print(f"Attempting to load checkpoint from: {args.ckpt_path}")
    
    # Target for restoration: a dictionary matching the *saved* structure.
    # The saved structure is {'actor_state': TrainState, 'qf1_state': TrainState, ...}
    # We only care about actor_state for inference.
    # We need to provide a target pytree that matches the structure of what's in the file for actor_state.
    target_to_restore = {
        'actor_state': agent.actor_state # Provide the initialized actor_state as a template
        # We don't need to provide qf1_state etc. if we only want to restore actor_state
        # and the checkpoint tool can partially restore.
        # However, flax.training.checkpoints.restore_checkpoint expects the target to match the saved dict structure
        # OR it can restore a sub-tree if the path to the sub-tree is unique.
        # Let's assume the checkpoint file contains the full dictionary.
        # We provide a target for all keys that might exist to be safe, or ensure partial loading.
    }
    
    # If the checkpoint definitely contains actor_state, qf1_state, qf2_state, log_alpha_state:
    # Then the target should reflect that structure to allow restore_checkpoint to map them.
    # We only *use* actor_state.params after loading.
    
    # Let's try to load only the actor_state.
    # `restore_checkpoint` loads the *entire* saved structure into a target with the *same* structure.
    # So, we create a target that can receive all saved states.
    
    # Create a full target structure based on how agent is initialized
    full_target_for_restore = {
        'actor_state': agent.actor_state,
        'qf1_state': agent.qf1_state,
        'qf2_state': agent.qf2_state,
    }
    if agent.log_alpha_state is not None: # If autotune was on during agent init
        full_target_for_restore['log_alpha_state'] = agent.log_alpha_state

    try:
        # restore_checkpoint restores IN-PLACE if target is mutable (like a dict of TrainStates)
        # or returns a new pytree if target is immutable. TrainState is a class, so it's tricky.
        # It's safer to get the returned value.
        loaded_states = checkpoints.restore_checkpoint(
            ckpt_dir=args.ckpt_path, # This should be the *direct path* to the checkpoint file itself
            target=full_target_for_restore 
        )
        if loaded_states and 'actor_state' in loaded_states:
            agent.actor_state = loaded_states['actor_state']
            print(f"Successfully loaded actor_state from checkpoint: {args.ckpt_path}")
            print(f"Actor params example (first layer kernel shape): {agent.actor_state.params['Dense_0']['kernel'].shape if 'Dense_0' in agent.actor_state.params else 'N/A (check network)'}")
        else:
            raise ValueError("Checkpoint did not contain 'actor_state' or failed to load.")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Ensure `ckpt_path` is the direct path to the checkpoint file (e.g., .../ckpt_step_10000) "
              "and not just the directory.")
        envs.close()
        return

    # Inference Loop
    episode_returns = []
    episode_lengths = []

    for episode_idx in range(args.num_episodes):
        obs, _ = envs.reset() # Seed is handled by make_env and SyncVectorEnv
        terminated = False
        truncated = False
        cumulative_reward = 0
        length = 0
        
        while not (terminated or truncated):
            key_actions, _ = jax.random.split(key_actions)
            
            # Ensure obs is in the correct format (Batch, ...). SyncVectorEnv handles this.
            jax_obs = jnp.asarray(obs)

            # Get action from agent
            # We use actor_state.params as per how agent.select_action is typically called in training
            actions_jax = agent.select_action(
                agent.actor_state.params, 
                jax_obs, 
                key_actions, 
                deterministic=args.greedy
            )
            actions_np = np.array(jax.device_get(actions_jax))

            next_obs, rewards, terminations, truncations, infos = envs.step(actions_np)
            
            # Assuming single environment for inference, so rewards[0], etc.
            cumulative_reward += rewards[0]
            length += 1
            obs = next_obs
            terminated = terminations[0]
            truncated = truncations[0]

            if "final_info" in infos and infos["final_info"][0] is not None:
                 # This block is usually for when an episode ends inside the loop due to final_info structure
                 # For a simple while not (terminated or truncated) loop, this might be redundant.
                 # We will capture final return and length outside the while loop based on term/trunc.
                 pass

        episode_returns.append(cumulative_reward)
        episode_lengths.append(length)
        print(f"Episode {episode_idx + 1}/{args.num_episodes}: Return={cumulative_reward:.2f}, Length={length}")

    envs.close()

    # Aggregate and Print Results
    if episode_returns:
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        min_return = np.min(episode_returns)
        max_return = np.max(episode_returns)
        mean_length = np.mean(episode_lengths)

        print("\\n--- Inference Results ---")
        print(f"Episodes run: {len(episode_returns)}")
        print(f"Mean Episodic Return: {mean_return:.2f} +/- {std_return:.2f}")
        print(f"Min Episodic Return:  {min_return:.2f}")
        print(f"Max Episodic Return:  {max_return:.2f}")
        print(f"Mean Episodic Length: {mean_length:.2f}")
    else:
        print("No episodes were completed.")

if __name__ == "__main__":
    infer_args = tyro.cli(InferArgs)
    infer(infer_args) 