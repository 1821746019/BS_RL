import tyro
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig
from .train import train
import os

if __name__ == "__main__":
    batch_size = 512
    env_id = "CartPole-v1"
    env_num = 16
    eval_env_num = 16
    args = Args(
        train=TrainConfig(
            exp_name=env_id,
            save_model=True,
            ckpt_save_frequency=0.01,
            resume=True,
            save_dir=f"runs/{env_id}",
        ),
        eval=EvalConfig(
            eval_frequency=0.01,
            eval_episodes=16,
            greedy_actions=True,
            capture_video=True,
            env_num=eval_env_num,
        ),
        env=EnvConfig(
            env_id=env_id,
            seed=1,
            env_num=env_num, # SAC typically uses 1 env for off-policy learning
        ),
        algo=AlgoConfig(
            total_timesteps=int(1e6), # CartPole learns faster
            buffer_size=int(1e5),
            learning_starts=int(1e3),
            batch_size=batch_size,
            update_frequency=4, # Update more frequently for simpler envs
            target_network_frequency=int(8e3),
            gamma=0.99,
            tau=1, # Softer updates can be better for MLP envs, but 1.0 is also fine
            policy_lr=3e-4*(batch_size/64)**0.5,
            q_lr=3e-4*(batch_size/64)**0.5,
            autotune=True,
            target_entropy_scale=0.89, # Adjusted for CartPole (action space size 2)
                                      # Target entropy = -scale * log(1/action_dim)
                                      # For CartPole (2 actions): -0.7 * log(0.5) approx 0.48
            adam_eps=1e-4
        ),
        wandb=WandbConfig(
            project_name="SAC-Discrete",
            entity=None # Your WandB entity
        )
    )
    print("--- NOTE FOR CartPole-v1 ---")
    print("This demo uses MLP networks defined in networks.py.")
    print("Hyperparameters have been adjusted for CartPole-v1.")
    print("-----------------------------")
    
    train(args)