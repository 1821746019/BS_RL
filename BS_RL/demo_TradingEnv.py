from BS_RL.common import valid_step_to_gamma
import numpy as np
import tyro
from BS_RL.config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from BS_RL.train import train
import os
from TradingEnv import TradingEnvConfig as TradingEnvConfig, RewardSchema
if __name__ == "__main__":
    trading_timeframe = "5m"
    valid_step = 100 #2*60/int(trading_timeframe[:-1]) #让agent只关注未来2h的reward
    reward_schema = RewardSchema.exp_baseline
    encoder="none"
    total_timesteps = int(200e6)
    resume = True
    env_num = 96
    num_steps = 2048
    batch_size = int(num_steps * env_num)
    num_minibatches = 32
    update_epochs = 4

    env_id = f"TradingEnv{trading_timeframe}"
    eval_env_num = 12 # 从48改为12减少评估耗时，若能实现异步评估就更好了
    eval_episodes = eval_env_num
    trading_env_config = TradingEnvConfig(
        data_path="/root/project/processed_data/",
        timeframe_minutes=trading_timeframe,
        reward_schema=reward_schema
    )
    is_test = total_timesteps!=int(200e6)
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
            env_num=env_num, # SAC typically uses 1 env for off-policy learning
        ),
        network=NetworkConfig(
            encoder_type=encoder,
            shape_tickers_positions=(trading_env_config.window_size, trading_env_config.kline_dim_5m),
            # encoder_type 默认为 "convnext". 若要使用 transformer, 设置: encoder_type="transformer"
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            num_steps=num_steps,
            gamma=valid_step_to_gamma(valid_step), 
            gae_lambda=0.95,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            adam_eps=1e-5
        ),
        wandb=WandbConfig(
            project_name="PPO_TradingEnv",
            entity=None # Your WandB entity
        )
    )
    
    train(args)