import tyro
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from .train import train
import os
from TradingEnv import EnvConfig as TradingEnvConfig, get_discrete_action_space_size

if __name__ == "__main__":
    batch_size = 5120
    env_id = "TradingEnv"
    env_num = 96
    eval_env_num = 16
    trading_env_config = TradingEnvConfig(data_path="/root/project/processed_data/")

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
            env_num=eval_env_num,
            async_vector_env=True, # 利用CPU多核
        ),
        env=EnvConfig(
            trading_env_config=trading_env_config,
            seed=1,
            env_num=env_num, # SAC typically uses 1 env for off-policy learning
            async_vector_env=True, # 利用CPU多核
        ),
        network=NetworkConfig(
            shape_1m=(trading_env_config.window_size_1m, trading_env_config.kline_dim_1m),
            shape_5m=(trading_env_config.window_size_5m, trading_env_config.kline_dim_5m),
            # encoder_type 默认为 "convnext". 若要使用 transformer, 设置: encoder_type="transformer"
        ),
        algo=AlgoConfig(
            total_timesteps=int(200e6), # 200M步
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
            project_name="SAC-Discrete_TradingEnv",
            entity=None # Your WandB entity
        )
    )
    
    train(args)