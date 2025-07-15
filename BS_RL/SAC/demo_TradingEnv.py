import numpy as np
import tyro
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from .train import train
import os
from TradingEnv import TradingEnvConfig as TradingEnvConfig, RewardSchema
def valid_step_to_gamma(valid_step: int):
    return 1-1/valid_step

if __name__ == "__main__":
    trading_timeframe = "5m"
    valid_step = 2*60/int(trading_timeframe[:-1]) #让agent只关注未来2h的reward
    reward_schema = RewardSchema.exp_baseline
    encoder="kline"
    total_timesteps = int(200e6)
    resume = True
    batch_size = 256
    env_id = f"TradingEnv{trading_timeframe}"
    env_num = 96
    eval_env_num = 12 # 从48改为12减少评估耗时，若能实现异步评估就更好了
    eval_episodes = eval_env_num
    trading_env_config = TradingEnvConfig(
        data_path="/root/project/processed_data/",
        window_size_5m=int(4*60/5),
        timeframe_minutes=trading_timeframe,
        reward_schema=reward_schema
    )
    is_test = total_timesteps!=int(200e6)
    learning_starts = int(batch_size) if is_test else int(2e4)
    ckpt_save_frequency = None if is_test else 0.01
    eval_frequency = None if is_test else 0.01
    async_vector_env = True # 开启以利用CPU多核。v3-8的CPU主频似乎v4-8低很多，实测SPS会低近1半(900-->500)
    args = Args(
        train=TrainConfig(
            exp_name=env_id,
            save_model=True,
            ckpt_save_frequency=ckpt_save_frequency,
            resume=resume,
            save_dir=f"runs/{env_id}_{encoder}",
            async_vector_env=async_vector_env,
        ),
        eval=EvalConfig(
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            greedy_actions=True,
            env_num=eval_env_num,
            async_vector_env=async_vector_env,
        ),
        env=EnvConfig(
            trading_env_config=trading_env_config,
            seed=1,
            env_num=env_num, # SAC typically uses 1 env for off-policy learning
        ),
        network=NetworkConfig(
            encoder_type=encoder,
            shape_1m=(trading_env_config.window_size_1m, trading_env_config.kline_dim_1m),
            shape_5m=(trading_env_config.window_size_5m, trading_env_config.kline_dim_5m),
            # encoder_type 默认为 "convnext". 若要使用 transformer, 设置: encoder_type="transformer"
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            buffer_size=int(1e5),
            learning_starts=learning_starts, 
            batch_size=batch_size,
            update_frequency=4, # Update more frequently for simpler envs
            target_network_frequency=int(8e3),
            gamma=valid_step_to_gamma(valid_step), 
            tau=1, # Softer updates can be better for MLP envs, but 1.0 is also fine
            policy_lr=3e-4*np.log1p(batch_size/64), 
            q_lr=3e-4*np.log1p(batch_size/64),
            autotune=True,
            target_entropy_scale=0.89*(6/11), # 无操作(1)、多空减减仓(4)、其它的权重视为(1)
            adam_eps=1e-4
        ),
        wandb=WandbConfig(
            project_name="SAC-Discrete_TradingEnv",
            entity=None # Your WandB entity
        )
    )
    
    train(args)