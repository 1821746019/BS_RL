import os
import random
import time
from dataclasses import asdict
from .common import evaluate_agent
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
import shutil # For directory cleaning
from pathlib import Path # For path manipulation
from flax.training import checkpoints # For robust checkpointing
import collections # Added for deque

from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig
from .common import make_env
from .networks import ActorCNN, CriticCNN, ActorMLP, CriticMLP # Import network choices
from .agent import SACAgent

def train(args: Args):
    # --- Path and Run Name Setup ---
    if args.train.save_dir:
        base_output_dir = Path(args.train.save_dir)
        # If save_dir is absolute, it's used as is. If relative, it's relative to cwd.
        # run_name for WandB and unique suffix if needed (e.g. if save_dir is a shared parent)
        run_name_suffix = f"{args.env.env_id}__{args.train.exp_name}__{args.env.seed}__{int(time.time())}"
        # WandB run name should still be unique across experiments even if save_dir is reused
        wandb_run_name = f"{base_output_dir.name}__{run_name_suffix}" if base_output_dir.name else run_name_suffix
    else:
        run_name_suffix = f"{args.env.env_id}__{args.train.exp_name}__{args.env.seed}__{int(time.time())}"
        base_output_dir = Path(f"runs/{run_name_suffix}")
        wandb_run_name = run_name_suffix

    ckpt_dir = base_output_dir / "ckpts"

    # --- Resume Logic & Directory Setup ---
    initial_global_step = 0
    restored_states = None

    if args.train.resume:
        if not base_output_dir.exists() or not ckpt_dir.exists():
            print(f"Warning: Resume requested, but directory {base_output_dir} or {ckpt_dir} not found. Starting from scratch.")
            args.train.resume = False # Effectively disable resume
        else:
            # Use flax.training.checkpoints to find the latest checkpoint
            latest_ckpt_path_str = checkpoints.latest_checkpoint(str(ckpt_dir))
            if latest_ckpt_path_str:
                print(f"Resuming training from checkpoint: {latest_ckpt_path_str}")
                try:
                    # We need to extract step from path: ckpt_dir/ckpt_step_12345.flax -> 12345
                    try:
                        initial_global_step = int(Path(latest_ckpt_path_str).stem.split("_")[-1])
                        print(f"Successfully parsed global_step {initial_global_step} from checkpoint path.")
                        # Mark that we have a valid checkpoint to restore later
                        # restored_states variable will now hold the path to the checkpoint
                        restored_states = latest_ckpt_path_str 
                    except ValueError:
                        print(f"Could not parse step from checkpoint filename: {latest_ckpt_path_str}. Cannot resume precisely.")
                        args.train.resume = False # Disable resume if step is not found

                except Exception as e:
                    print(f"Error processing checkpoint path: {e}. Starting from scratch.")
                    args.train.resume = False
                    initial_global_step = 0
                    restored_states = None # This will be the path string or None
            else:
                print(f"Warning: Resume requested, but no checkpoints found in {ckpt_dir} using flax.training.checkpoints. Starting from scratch.")
    
    if not args.train.resume: # Starting fresh or failed resume
        if args.train.save_dir: # User specified a directory
            if base_output_dir.exists():
                print(f"Starting fresh: Clearing specified save_dir subdirectory {ckpt_dir}")
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
        # Create directories if they don't exist (or were just cleared)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure an effective ckpt_save_frequency_abs_steps is set (from Args.__post_init__)
    # If still None or <=0, it implies no periodic checkpointing
    if not args.train.ckpt_save_frequency_abs_steps or args.train.ckpt_save_frequency_abs_steps <= 0:
        print("Periodic checkpointing is disabled (ckpt_save_frequency_abs_steps is not positive).")
        # Set to a very large number to effectively disable if not already handled by __post_init__
        args.train.ckpt_save_frequency_abs_steps = args.algo.total_timesteps + 1


    if args.wandb.track:
        import wandb
        wandb.init(
            project=args.wandb.project_name,
            entity=args.wandb.entity,
            sync_tensorboard=False,
            config=asdict(args), 
            name=wandb_run_name, # Use the potentially prefixed name
            monitor_gym=True, 
            save_code=True,
            resume="allow", # WandB native resume
            id=wandb.util.generate_id() if not args.train.resume else None # Generate new id if not resuming wandb run
        )
        # Log hyperparameters to wandb
        flat_args_dict = {}
        for main_key, main_value in asdict(args).items():
            if isinstance(main_value, dict):
                for sub_key, sub_value in main_value.items():
                    flat_args_dict[f"{main_key}.{sub_key}"] = sub_value
            else:
                flat_args_dict[main_key] = main_value
        
        wandb.config.update(flat_args_dict)

    # --- Buffers for smoothed episodic statistics ---
    # Buffer size is now dynamic, based on the number of environments
    episode_stats_buffer_size = args.env.num_envs 
    recent_episode_returns_buffer = collections.deque(maxlen=episode_stats_buffer_size)
    recent_episode_lengths_buffer = collections.deque(maxlen=episode_stats_buffer_size)

    # Seeding
    random.seed(args.env.seed)
    np.random.seed(args.env.seed)
    key = jax.random.PRNGKey(args.env.seed)
    # Split key for agent init, ensuring it's different if resuming to avoid reusing keys implicitly
    # Though JAX agent typically re-initializes PRNG streams internally if needed.
    # This key_agent_init_seed is for the *initial* construction.
    key_agent_init_seed = args.env.seed if not args.train.resume else args.env.seed + initial_global_step 
    key_agent = jax.random.PRNGKey(key_agent_init_seed)
    key_actions, key_rb_init = jax.random.split(jax.random.PRNGKey(args.env.seed + 1), 2) # Separate keys for action loop and RB init


    # Env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env.env_id, args.env.seed + i*1000, i, args.env.capture_video, run_name_suffix, args.env.num_envs) for i in range(args.env.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported by this SAC version"
    
    obs_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    # Determine network type based on environment
    if "NoFrameskip" in args.env.env_id: # Atari-like
        actor_model_cls = ActorCNN
        critic_model_cls = CriticCNN
    elif "CartPole-v1" == args.env.env_id: # Simple MLP env
        actor_model_cls = ActorMLP
        critic_model_cls = CriticMLP
        # Adjust target entropy for simpler envs if needed
        if args.algo.autotune and args.algo.target_entropy_scale == 0.89: # Default for Atari
            print("Warning: Using Atari target_entropy_scale for CartPole. Consider adjusting.")
            # args.algo.target_entropy_scale = 0.5 # Example for CartPole
    else:
        raise ValueError(f"Unsupported environment ID for network selection: {args.env.env_id}")


    agent = SACAgent(
        action_dim=action_dim,
        observation_space_shape=obs_shape, # Pass the original shape
        key=key_agent, # Use the potentially modified key for agent init
        algo_config=args.algo,
        actor_model_cls=actor_model_cls,
        critic_model_cls=critic_model_cls
    )
    
    # Restore states if loaded from checkpoint
    if restored_states: # restored_states is now the path to the checkpoint
        try:
            # Create a target structure for restoration. This assumes agent is initialized.
            target_restore_dict = {
                'actor_state': agent.actor_state,
                'qf1_state': agent.qf1_state,
                'qf2_state': agent.qf2_state,
                'log_alpha_state': agent.log_alpha_state if args.algo.autotune else None,
                # global_step is handled by initial_global_step from filename
            }
            # Filter out None log_alpha_state if not autotuning for restoration target
            if not args.algo.autotune:
                target_restore_dict.pop('log_alpha_state')

            # Perform the restoration using flax.training.checkpoints
            loaded_train_states = checkpoints.restore_checkpoint(
                ckpt_dir=str(ckpt_dir), # Directory containing checkpoints
                target=target_restore_dict, # Target pytree with same structure as saved
                step=initial_global_step, # Specify which step to restore (optional, can use prefix)
                prefix="ckpt_step_" # Our filename prefix
            )

            agent.actor_state = loaded_train_states['actor_state']
            agent.qf1_state = loaded_train_states['qf1_state']
            agent.qf2_state = loaded_train_states['qf2_state']
            if args.algo.autotune and 'log_alpha_state' in loaded_train_states:
                agent.log_alpha_state = loaded_train_states['log_alpha_state']
                agent.current_alpha = jnp.exp(agent.log_alpha_state.params['log_alpha'])
            
            # Ensure actor_state, qf1_state etc are updated for the main loop
            actor_state = agent.actor_state
            qf1_state = agent.qf1_state
            qf2_state = agent.qf2_state
            log_alpha_state = agent.log_alpha_state
            current_alpha_val = agent.current_alpha

            print(f"Agent states successfully restored from step {initial_global_step}. Actor LR: {agent.actor_state.tx.learning_rate}")

        except Exception as e:
            print(f"Error restoring agent states using flax.training.checkpoints: {e}. Continuing with fresh/default states.")
    
    # Replay buffer
    # For JAX, ensure data is moved to CPU for SB3 buffer, then converted to JAX arrays
    rb = ReplayBuffer(
        args.algo.buffer_size,
        envs.single_observation_space, # JAX expects NCHW for Conv, SB3 buffer might store it as is
        envs.single_action_space,
        device="cpu", # SB3 buffer on CPU
        handle_timeout_termination=False, # As in original
        n_envs=args.env.num_envs
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.env.seed + initial_global_step) # Vary reset seed if resuming

    actor_state = agent.actor_state
    qf1_state = agent.qf1_state
    qf2_state = agent.qf2_state
    log_alpha_state = agent.log_alpha_state # This is TrainState if autotune, else None
    current_alpha_val = agent.current_alpha # This is jnp.array(fixed_alpha) if not autotune

    # Adjust total_timesteps if resuming:
    # The loop runs for (total_timesteps - initial_global_step) / num_envs iterations.
    # Or, the loop condition uses initial_global_step.
    # Current loop: for global_step in range(args.algo.total_timesteps // args.env.num_envs):
    # This means global_step is an iteration counter. Actual env steps = global_step * num_envs.
    # We need to start `global_step` from `initial_global_step // args.env.num_envs`.

    start_iteration = initial_global_step // args.env.num_envs
    # Loop for the REMAINING number of iterations
    for loop_iter in range(start_iteration, args.algo.total_timesteps // args.env.num_envs):
        current_env_steps = loop_iter * args.env.num_envs
        
        # ALGO LOGIC: action selection
        if current_env_steps < args.algo.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            key_actions, _ = jax.random.split(key_actions)
            jax_obs = jnp.asarray(obs)
            # Pass deterministic=False for training exploration
            actions_jax = agent.select_action(actor_state.params, jax_obs, key_actions, deterministic=False)
            actions = np.array(jax.device_get(actions_jax))

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        current_env_steps_post_step = (loop_iter + 1) * args.env.num_envs # total steps after this env step
        
        # Handle final_observation for truncated episodes (important for ReplayBuffer)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Add to replay buffer
        # Ensure rewards and dones are float for JAX processing later
        rb.add(obs, real_next_obs, actions, rewards.astype(np.float32), terminations.astype(np.float32), infos)
        obs = next_obs

        # Log episodic returns (Aggregated)
        if "final_info" in infos:
            newly_finished_episode_returns = []
            newly_finished_episode_lengths = []

            for i, info_item in enumerate(infos["final_info"]):
                if info_item and "episode" in info_item:
                    ep_ret = info_item['episode']['r']
                    ep_len = info_item['episode']['l']
                    
                    # Ensure scalar values (RecordEpisodeStatistics often returns 1-element arrays)
                    current_return_scalar = ep_ret[0] if isinstance(ep_ret, (np.ndarray, list)) and len(ep_ret) > 0 else float(ep_ret)
                    current_length_scalar = ep_len[0] if isinstance(ep_len, (np.ndarray, list)) and len(ep_len) > 0 else int(ep_len)

                    newly_finished_episode_returns.append(current_return_scalar)
                    newly_finished_episode_lengths.append(current_length_scalar)

                    if i == 0 and args.wandb.track: # This is env 0 and it finished an episode
                        wandb.log({
                            "charts/episodic_return_env0": current_return_scalar, 
                            "charts/episodic_length_env0": current_length_scalar
                        }, step=current_env_steps)
            
            if newly_finished_episode_returns: # If any episodes ended this step
                recent_episode_returns_buffer.extend(newly_finished_episode_returns)
                recent_episode_lengths_buffer.extend(newly_finished_episode_lengths)

            # Log buffered statistics if buffers have data (which they will if an episode just ended and updated them)
            # This logging section will effectively run if newly_finished_episode_returns was non-empty.
            if recent_episode_returns_buffer and args.wandb.track: 
                returns_list = list(recent_episode_returns_buffer)
                
                mean_ret_buff = np.mean(returns_list)
                min_ret_buff = np.min(returns_list)
                max_ret_buff = np.max(returns_list)
                
                wandb_metrics = {
                    "charts/episodic_return_mean_buffered": mean_ret_buff,
                    "charts/episodic_return_min_buffered": min_ret_buff,
                    "charts/episodic_return_max_buffered": max_ret_buff
                }

                std_ret_buff = 0.0
                if len(returns_list) > 1:
                    std_ret_buff = np.std(returns_list)
                wandb_metrics["charts/episodic_return_std_buffered"] = std_ret_buff
                
                wandb.log(wandb_metrics, step=current_env_steps)
                
                # Updated console print
                print(f"step={current_env_steps}, buffered_return_mean={mean_ret_buff:.2f}, buffered_return_std={std_ret_buff:.2f} (from {len(returns_list)} ep. in buffer)")

            if recent_episode_lengths_buffer and args.wandb.track:
                lengths_list = list(recent_episode_lengths_buffer)
                mean_len_buff = np.mean(lengths_list)
                wandb.log({"charts/episodic_length_mean_buffered": mean_len_buff}, step=current_env_steps)
        
        # ALGO LOGIC: training
        if current_env_steps > args.algo.learning_starts:
            if current_env_steps % args.algo.update_frequency == 0:
                data = rb.sample(args.algo.batch_size)
                
                # Convert to JAX arrays (from NumPy arrays from SB3 buffer)
                # SB3 buffer stores observations as is, e.g. (Batch, C, H, W)
                # Actions are stored as (Batch,) or (Batch, ActionDim)
                # Rewards, Dones are (Batch,)
                
                data_jax = {
                    'observations': jnp.asarray(data.observations.numpy()), # .numpy() if they are torch tensors
                    'actions': jnp.asarray(data.actions.numpy().astype(np.int32)), # Ensure int32 for take_along_axis
                    'next_observations': jnp.asarray(data.next_observations.numpy()),
                    'rewards': jnp.asarray(data.rewards.numpy().flatten()), # Ensure (Batch,)
                    'dones': jnp.asarray(data.dones.numpy().flatten())      # Ensure (Batch,)
                }
                
                key_actions, key_update = jax.random.split(key_actions) # For any stochasticity in update if needed

                log_alpha_param_or_val = log_alpha_state if args.algo.autotune else current_alpha_val
                
                actor_state, qf1_state, qf2_state, log_alpha_state_updated, metrics = agent.update_all(
                    actor_state, qf1_state, qf2_state, log_alpha_param_or_val, data_jax, key_update
                )
                if args.algo.autotune:
                    log_alpha_state = log_alpha_state_updated
                    current_alpha_val = jnp.exp(log_alpha_state.params['log_alpha'])

                # Log SPS and other metrics less frequently
                if current_env_steps % (args.algo.update_frequency * 25) == 0 and args.wandb.track: # e.g. every 100 updates if update_freq=4
                    sps = int(current_env_steps / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
                    print(f"SPS: {sps}")
                    
                    wandb_metrics_log = {f"losses/{m_key}": jax.device_get(m_val) for m_key, m_val in metrics.items()}
                    wandb_metrics_log["charts/SPS"] = sps
                    wandb.log(wandb_metrics_log, step=current_env_steps)

            # --- Checkpointing and Evaluation ---
            if args.train.ckpt_save_frequency_abs_steps and \
               current_env_steps > 0 and \
               current_env_steps_post_step >= args.train.ckpt_save_frequency_abs_steps and \
               (current_env_steps // args.train.ckpt_save_frequency_abs_steps) < (current_env_steps_post_step // args.train.ckpt_save_frequency_abs_steps):
                
                # Ensure checkpoint and evaluation are done AT `current_env_steps` that aligns with frequency.
                # The step to checkpoint should be a multiple of `ckpt_save_frequency_abs_steps`.
                # Let actual_ckpt_step be the step at which we are checkpointing.
                # This usually means current_env_steps should be the step for which data has been collected and model updated.
                actual_ckpt_step = (current_env_steps_post_step // args.train.ckpt_save_frequency_abs_steps) * args.train.ckpt_save_frequency_abs_steps
                # Check if we've already checkpointed for this specific step to avoid re-checkpointing if loop structure allows multiple triggers
                # This is mainly a safeguard, the condition above should handle it.
                # We use current_env_steps for naming, which is loop_iter * num_envs.
                # The condition fires when current_env_steps_post_step crosses the boundary.
                # So the checkpoint is for the state *after* the updates using data up to current_env_steps.

                print(f"--- Checkpoint & Evaluation Triggered at ~step {current_env_steps} (post-step: {current_env_steps_post_step}) ---")
                # The actual step number for the checkpoint will be current_env_steps, reflecting the state *after* this iteration's update.

                # 1. Save Checkpoint
                try:
                    save_target_dict = {
                        'actor_state': actor_state,
                        'qf1_state': qf1_state,
                        'qf2_state': qf2_state,
                    }
                    if args.algo.autotune and log_alpha_state is not None:
                        save_target_dict['log_alpha_state'] = log_alpha_state
                    
                    checkpoints.save_checkpoint(
                        ckpt_dir=str(os.path.abspath(ckpt_dir)),
                        target=save_target_dict, 
                        step=current_env_steps,  # Use current_env_steps which is before this step block increments
                        prefix="ckpt_step_",
                        keep=50,                  
                        overwrite=True           
                    )
                    saved_ckpt_path = checkpoints.latest_checkpoint(str(ckpt_dir))
                    print(f"Checkpoint saved to {saved_ckpt_path} for step {current_env_steps}")

                    if args.wandb.track and wandb.run is not None and saved_ckpt_path:
                        artifact = wandb.Artifact(f"model_ckpt_{wandb_run_name}", type="model")
                        artifact.add_file(str(saved_ckpt_path))
                        wandb.log_artifact(artifact, aliases=[f"step_{current_env_steps}"])

                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

                # 2. Evaluate Model
                if args.train.eval_episodes > 0:
                    eval_env_config_copy = EnvConfig(
                        env_id=args.env.env_id, # Use same env_id or could be different if specified
                        capture_video=args.train.eval_capture_video,
                        num_envs=1, # Eval typically uses 1 env at a time in this setup
                        seed=args.env.seed + current_env_steps + 1 # Different seed for eval
                    )
                    eval_metrics = evaluate_agent(
                        agent_eval=agent, # Pass the agent instance
                        actor_params_eval=actor_state.params, # Current actor parameters
                        eval_env_config=eval_env_config_copy,
                        num_episodes=args.train.eval_episodes,
                        seed=args.env.seed + 9999 + current_env_steps, # Ensure a very different seed for eval runs
                        greedy_actions=args.train.eval_greedy_actions,
                        run_name_suffix_eval=run_name_suffix, # For video naming
                        current_train_step=current_env_steps
                    )
                    print(f"Evaluation at step {current_env_steps}: {eval_metrics}")
                    if args.wandb.track:
                        wandb.log({f"eval/{m_key}": m_val for m_key, m_val in eval_metrics.items()}, step=current_env_steps)


            # Update target networks
            if current_env_steps % args.algo.target_network_frequency == 0:
                qf1_state, qf2_state = agent.update_target_networks(qf1_state, qf2_state)

    # End of training loop
    
    # Final save, if `save_model` was true and no periodic ckpting, or as a last ckpt
    # The config.__post_init__ already warns about save_model if periodic is on.
    # We can choose to always save one final checkpoint regardless.
    final_ckpt_step = (args.algo.total_timesteps // args.env.num_envs) * args.env.num_envs
    print(f"Saving final model at step {final_ckpt_step}")
    try:
        final_save_target_dict = {
            'actor_state': actor_state,
            'qf1_state': qf1_state,
            'qf2_state': qf2_state,
        }
        if args.algo.autotune and log_alpha_state is not None:
            final_save_target_dict['log_alpha_state'] = log_alpha_state

        checkpoints.save_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=final_save_target_dict,
            step=final_ckpt_step,
            prefix="ckpt_step_",
            keep=1, # Keep only this one, or rely on the periodic one's keep policy
            overwrite=True
        )
        saved_final_ckpt_path = checkpoints.latest_checkpoint(str(ckpt_dir)) # Get path of the final saved model
        print(f"Final model saved to {saved_final_ckpt_path}")
        if args.wandb.track and wandb.run is not None and saved_final_ckpt_path:
            artifact = wandb.Artifact(f"model_ckpt_{wandb_run_name}", type="model")
            artifact.add_file(str(saved_final_ckpt_path))
            wandb.log_artifact(artifact, aliases=[f"step_{final_ckpt_step}", "final"])
    except Exception as e:
        print(f"Error saving final model: {e}")


    envs.close()
    if args.wandb.track:
        wandb.finish()

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)