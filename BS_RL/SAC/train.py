import os
import random
import time
from dataclasses import asdict
import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter # Using PyTorch's SummaryWriter

from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig
from .utils import make_env
from .networks import ActorCNN, CriticCNN, ActorMLP, CriticMLP # Import network choices
from .agent import SACAgent

def train(args: Args):
    run_name = f"{args.env.env_id}__{args.train.exp_name}__{args.env.seed}__{int(time.time())}"
    if args.wandb.track:
        import wandb
        wandb.init(
            project=args.wandb.project_name,
            entity=args.wandb.entity,
            sync_tensorboard=True,
            config=asdict(args), # Log all args
            name=run_name,
            monitor_gym=True, # Auto-log videos if RecordVideo wrapper is used
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    # Log hyperparameters as text
    # Convert args to a flat dictionary for easier processing if needed
    flat_args_dict = {}
    for main_key, main_value in asdict(args).items():
        if isinstance(main_value, dict):
            for sub_key, sub_value in main_value.items():
                flat_args_dict[f"{main_key}.{sub_key}"] = sub_value
        else:
            flat_args_dict[main_key] = main_value

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flat_args_dict.items()])),
    )


    # Seeding
    random.seed(args.env.seed)
    np.random.seed(args.env.seed)
    key = jax.random.PRNGKey(args.env.seed)
    key_agent, key_actions, key_rb_init = jax.random.split(key, 3)


    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env.env_id, args.env.seed + i, i, args.env.capture_video, run_name, args.env.num_envs) for i in range(args.env.num_envs)]
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
        key=key_agent,
        algo_config=args.algo,
        actor_model_cls=actor_model_cls,
        critic_model_cls=critic_model_cls
    )
    
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
    obs, _ = envs.reset(seed=args.env.seed) # obs shape (num_envs, C, H, W) or (num_envs, Features)

    actor_state = agent.actor_state
    qf1_state = agent.qf1_state
    qf2_state = agent.qf2_state
    log_alpha_state = agent.log_alpha_state # This is TrainState if autotune, else None
    current_alpha_val = agent.current_alpha # This is jnp.array(fixed_alpha) if not autotune

    for global_step in range(args.algo.total_timesteps // args.env.num_envs):
        # ALGO LOGIC: action selection
        if global_step * args.env.num_envs < args.algo.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            key_actions, _ = jax.random.split(key_actions)
            # Ensure obs is JAX array for agent.select_action
            # obs might be (C,H,W) from FrameStack or (Features) from MLP envs
            # agent.select_action expects (Batch, ...)
            jax_obs = jnp.asarray(obs)
            
            # If obs is (C,H,W) and num_envs=1, SyncVectorEnv might return (C,H,W) instead of (1,C,H,W)
            # This was an issue with older gym versions, gymnasium should handle it.
            # Let's assume obs is already (num_envs, C, H, W) or (num_envs, Features)
            
            actions_jax = agent.select_action(actor_state.params, jax_obs, key_actions)
            actions = np.array(jax.device_get(actions_jax))

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Handle final_observation for truncated episodes (important for ReplayBuffer)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Add to replay buffer
        # Ensure rewards and dones are float for JAX processing later
        rb.add(obs, real_next_obs, actions, rewards.astype(np.float32), terminations.astype(np.float32), infos)
        obs = next_obs

        # Log episodic returns
        if "final_info" in infos:
            for info_idx, info in enumerate(infos["final_info"]):
                if info and "episode" in info:
                    print(f"global_step={global_step * args.env.num_envs}, env_idx={info_idx}, episodic_return={info['episode']['r']}")
                    writer.add_scalar(f"charts/episodic_return_env{info_idx}", info['episode']['r'], global_step * args.env.num_envs)
                    writer.add_scalar(f"charts/episodic_length_env{info_idx}", info['episode']['l'], global_step * args.env.num_envs)
        
        # ALGO LOGIC: training
        if global_step * args.env.num_envs > args.algo.learning_starts:
            if (global_step * args.env.num_envs) % args.algo.update_frequency == 0:
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


                if (global_step * args.env.num_envs) % 100 == 0: # Log losses less frequently
                    for m_key, m_val in metrics.items():
                        writer.add_scalar(f"losses/{m_key}", jax.device_get(m_val), global_step * args.env.num_envs)
                    writer.add_scalar("charts/SPS", int((global_step * args.env.num_envs) / (time.time() - start_time)), global_step * args.env.num_envs)
                    print(f"SPS: {int((global_step * args.env.num_envs) / (time.time() - start_time))}")


            # Update target networks
            if (global_step * args.env.num_envs) % args.algo.target_network_frequency == 0:
                qf1_state, qf2_state = agent.update_target_networks(qf1_state, qf2_state)

    # TODO: Model saving
    if args.train.save_model:
        # Save actor_state.params, qf1_state.params, qf2_state.params
        # Flax serialization: flax.serialization.to_bytes / from_bytes
        model_path = f"runs/{run_name}/{args.train.exp_name}.flax_model"
        with open(model_path, "wb") as f:
            # Save a dict of parameters
            model_data = {
                'actor_params': actor_state.params,
                'qf1_params': qf1_state.params,
                'qf2_params': qf2_state.params,
                'log_alpha_params': log_alpha_state.params if args.algo.autotune else None
            }
            f.write(flax.serialization.to_bytes(model_data))
        print(f"model saved to {model_path}")
        # print("Model saving not fully implemented yet for JAX.")


    envs.close()
    writer.close()
    if args.wandb.track:
        wandb.finish()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.train.exp_name = "sac_discrete_jax_test" # Override for direct run
    train(args)