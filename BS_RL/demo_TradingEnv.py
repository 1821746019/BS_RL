from TradingEnv.Config import DataLoaderConfig
import numpy as np
import tyro
import gymnasium as gym
from BS_RL.config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from BS_RL.nn.ResMLP import ResMLPConfig, ResidualStrategy
from BS_RL.train import train
from TradingEnv import TradingEnvConfig

DAY_MINUTES = 1440
if __name__ == "__main__":
    env_id = "TradingEnv"
    use_SGD = False
    total_timesteps = int(10e6) 
    env_num = 32
    batch_size = 1
    updates_per_call = 32
    async_vector_env = False #True if env_num > 1 else False
    eval_env_num = 32
    async_vector_env_eval = False #if eval_env_num > 1 else False
    exp_name = f"env_num({env_num})_{env_id}_SAC_{'SGD' if use_SGD else 'AdamW'}"
    eval_episodes = eval_env_num #每个环境评估一次
    timeframe_m = 1
    trading_timeframe_m = 60
    burn_in = DAY_MINUTES // trading_timeframe_m
    train_unroll_steps = DAY_MINUTES * 7  // trading_timeframe_m 
    rb_seg_len = DAY_MINUTES * 15 // trading_timeframe_m
    is_test = total_timesteps == int(1e6)
    learning_starts = 1000 if is_test else 30000
    ckpt_save_frequency = 0.1 if is_test else 0.1
    eval_frequency = 0.1 if is_test else 0.1
    
    # 针对TradingEnv优化的网络配置
    
    args = Args(
        train=TrainConfig(
            exp_name=exp_name,
            ckpt_save_frequency=ckpt_save_frequency,
            resume=False,  # 首次运行设为False
            save_dir=f"runs/SAC_{env_id}",
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
            data_loader_cfg=DataLoaderConfig(timeframe_m=timeframe_m),
            trading_env_config=TradingEnvConfig(cooldown_days=0, loss_aversion=1, test_episode_days=14),
            env_num=env_num,
        ),
        network=NetworkConfig(
            use_modern_tcn_encoder=True,
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
            buffer_size=int(10e6),  # 大缓冲区有助于稳定训练
            learning_starts=learning_starts,
            batch_size=batch_size,
            update_frequency=1,  # 每步都更新
            target_network_frequency=1,  # 软更新，每步更新
            gamma=0.99,
            tau=0.005,  # 连续动作通常用软更新
            policy_lr=3e-4,
            q_lr=3e-4,
            autotune=True,  # 自动调节熵系数
            adam_eps=1e-4,
            use_SGD=use_SGD,
            updates_per_call=updates_per_call,
            train_unroll_steps=train_unroll_steps,
            burn_in=burn_in,
            rb_seg_len=rb_seg_len,
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