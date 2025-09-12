from TradingEnv.Config import DataLoaderConfig
import numpy as np
import tyro
import gymnasium as gym
from BS_RL.config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from BS_RL.train import train
import os
from TradingEnv import TradingEnvConfig


if __name__ == "__main__":
    env_id = "TradingEnv"
    window_size = 1 # 防止过拟合，用1
    use_SGD = False
    total_timesteps = int(10e6) 
    env_num = 32
    num_steps = 128
    num_minibatches = 4
    async_vector_env = True if env_num > 1 else False
    eval_env_num = 16
    async_vector_env_eval = True if eval_env_num > 1 else False
    exp_name = f"env_num({env_num})_window({window_size})_{env_id}_PPO_{'SGD' if use_SGD else 'AdamW'}"
    eval_episodes = 1
    is_test = total_timesteps == int(1e6)
    ckpt_save_frequency = 0.1 if is_test else 0.1
    eval_frequency = 0.1 if is_test else 0.1
    
    # 针对TradingEnv优化的网络配置
    
    args = Args(
        train=TrainConfig(
            exp_name=exp_name,
            ckpt_save_frequency=ckpt_save_frequency,
            resume=False,  # 首次运行设为False
            save_dir=f"runs/PPO_{env_id}",
            async_vector_env=async_vector_env,
        ),
        eval=EvalConfig(
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            greedy_actions=True,  # 评估时使用确定性动作
            env_num=eval_env_num,
            async_vector_env=async_vector_env_eval,
            capture_media=True,
        ),
        env=EnvConfig(
            trading_env_config=TradingEnvConfig(),
            env_num=env_num,
        ),
        network=NetworkConfig(
            actor_net_arch=[128, 128, 128],
            critic_net_arch=[128, 128, 128],
            s5_hidden_dim=256,
            s5_num_layers=3,
            # 不需要时间序列编码器，直接用MLP
            shape_tickers_positions=(0, 0),  # 不使用
            encoder_type="none",  # 标记为不使用编码器
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            num_steps=num_steps,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=num_minibatches,
            update_epochs=10,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            adam_eps=1e-5,
            use_SGD=use_SGD,
        ),
        wandb=WandbConfig(
            track=True,
            project_name=f"PPO_{env_id}",
            entity=None
        )
    )
    
    print(f"总步数: {total_timesteps:,}")
    print(f"使用SGD: {args.algo.use_SGD}")
    train(args)