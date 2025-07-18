from TradingEnv.Config import DataLoaderConfig
import numpy as np
import tyro
import gymnasium as gym
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from .nn.ResMLP import ResMLPConfig, ResidualStrategy
from .train import train
from .common import gym_train_env_maker, gym_eval_env_maker
import os
from TradingEnv import TradingEnvConfig


if __name__ == "__main__":
    env_id = "TradingEnv"
    window_size = 1
    exp_name = f"window_{window_size}_{env_id}_SAC"
    total_timesteps = int(200e6) 
    batch_size = 256
    env_num = 1  # SAC通常使用单环境
    eval_env_num = 8
    eval_episodes = 8
    
    is_test = total_timesteps == int(1e6)
    learning_starts = 10000 if not is_test else 1000
    ckpt_save_frequency = 0.01 if not is_test else 0.1 
    eval_frequency = 0.01 if not is_test else 0.1  
    
    # 针对TradingEnv优化的网络配置
    
    args = Args(
        train=TrainConfig(
            exp_name=exp_name,
            save_model=True,
            ckpt_save_frequency=ckpt_save_frequency,
            resume=True,  # 首次运行设为False
            save_dir=f"runs/SAC_{env_id}",
            async_vector_env=False,  # 单环境无需异步
        ),
        eval=EvalConfig(
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            greedy_actions=True,  # 评估时使用确定性动作
            env_num=eval_env_num,
            async_vector_env=False,
            capture_media=True,  # 记录视频
        ),
        env=EnvConfig(
            trading_env_config=TradingEnvConfig(data_loader_config=DataLoaderConfig(data_path="../../processed_data")),
            env_num=env_num,
        ),
        network=NetworkConfig(
            # 不需要时间序列编码器，直接用MLP
            shape_tickers_positions=(0, 0),  # 不使用
            encoder_type="none",  # 标记为不使用编码器
            
            # 为128维观察空间设计的ResMLP配置
            ResMLP_final=ResMLPConfig(
                hidden_dims=[768,768,768,768,768],  # 适中的网络深度
                add_initial_embedding_layer=True,
                residual_strategy=ResidualStrategy.PROJECTION,
                dropout_rate=0.0,  # LunarLander通常不需要dropout
                use_highway=False,
                name="ResMLP_final",
                description="特征处理网络"
            ),
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            buffer_size=int(1e6),  # 大缓冲区有助于稳定训练
            learning_starts=learning_starts,
            batch_size=batch_size,
            update_frequency=1,  # 每步都更新
            target_network_frequency=1,  # 软更新，每步更新
            gamma=0.99,
            tau=0.005,  # 连续动作通常用软更新
            policy_lr=3e-4,
            q_lr=3e-4,
            autotune=True,  # 自动调节熵系数
            target_entropy_scale=1.0,  # 连续动作的标准设置
            adam_eps=1e-4
        ),
        wandb=WandbConfig(
            track=True,
            project_name=f"{env_id}",
            entity=None
        )
    )
    
    print(f"总步数: {total_timesteps:,}")
    print(f"批次大小: {batch_size}")
    print(f"学习开始步数: {learning_starts:,}")
    print(f"使用SGD: {args.algo.use_SGD}")
    train(args)