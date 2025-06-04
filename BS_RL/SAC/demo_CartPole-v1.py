import tyro
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig
from .train import train
import os

if __name__ == "__main__":
    args = Args(
        train=TrainConfig(
            exp_name="sac_discrete_jax_cartpole",
            save_model=True,
            jax_platform_name="tpu" # Or "cpu" or None
        ),
        env=EnvConfig(
            env_id="CartPole-v1",
            capture_video=False,
            seed=1,
            num_envs=1 # SAC typically uses 1 env for off-policy learning
        ),
        algo=AlgoConfig(
            total_timesteps=100_00*5, # CartPole learns faster
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=128*100,
            update_frequency=50, # Update more frequently for simpler envs
            target_network_frequency=500,
            gamma=0.99,
            tau=0.005, # Softer updates can be better for MLP envs, but 1.0 is also fine
            policy_lr=3e-4*2,
            q_lr=3e-4*2,
            autotune=True,
            target_entropy_scale=0.7, # Adjusted for CartPole (action space size 2)
                                      # Target entropy = -scale * log(1/action_dim)
                                      # For CartPole (2 actions): -0.7 * log(0.5) approx 0.48
            adam_eps=1e-4
        ),
        wandb=WandbConfig(
            track=False, # Set to True to use WandB
            project_name="cleanrl-jax-sac-discrete",
            entity=None # Your WandB entity
        )
    )
    # Set XLA memory fraction if desired
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

    print("--- NOTE FOR CartPole-v1 ---")
    print("This demo uses MLP networks defined in networks.py.")
    print("Hyperparameters have been adjusted for CartPole-v1.")
    print("-----------------------------")
    
    train(args)