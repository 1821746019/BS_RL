import numpy as np
import tyro
import gymnasium as gym
from BS_RL.SAC.config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from BS_RL.SAC.nn.ResMLP import ResMLPConfig, ResidualStrategy
from BS_RL.SAC.train import Trainer
from BS_RL.SAC.common import gym_train_env_maker, gym_eval_env_maker
import os
from TradingEnv import TradingEnvConfig

class GymTrainer(Trainer):
    """适配标准gym环境的训练器"""
    
    def __init__(self, args: Args, env_id: str = "LunarLanderContinuous-v2"):
        super().__init__(args)
        self.env_id = env_id
    
    def _setup_data_loader(self):
        # gym环境不需要DataLoader
        self.data_loader = None
        print("Gym环境模式：跳过DataLoader设置")
    
    def _setup_environments(self):
        print(f"创建{self.env_id}训练环境...")
        vec_env_cls = gym.vector.AsyncVectorEnv if self.args.train.async_vector_env else gym.vector.SyncVectorEnv
        self.envs = vec_env_cls([
            gym_train_env_maker(
                env_id=self.env_id,
                seed=self.args.train.seed + i
            ) for i in range(self.args.env.env_num)
        ])
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        action_type = "离散" if self.is_discrete else "连续"
        print(f"检测到{action_type}动作空间")
    
    def _setup_evaluator(self):
        if self.args.eval.eval_episodes <= 0:
            return
        print("设置评估器（gym模式）...")
        
        from BS_RL.SAC.eval import Evaluator
        
        class GymEvaluator(Evaluator):
            def __init__(self, agent, env_config, eval_config, run_name_suffix, logger, env_id, seed):
                self.agent = agent
                self.env_config = env_config
                self.eval_config = eval_config
                self.run_name_suffix = run_name_suffix
                self.logger = logger
                self.env_id = env_id
                self.seed = seed
                self.eval_envs = None
                self.data_loader = None  # gym环境不需要DataLoader
                
                if self.eval_config.cache_env:
                    self.eval_envs = self._make_envs()
            
            def _make_envs(self):
                print("创建评估环境...")
                eval_vec_env_cls = gym.vector.AsyncVectorEnv if self.eval_config.async_vector_env else gym.vector.SyncVectorEnv
                
                return eval_vec_env_cls([
                    gym_eval_env_maker(
                        env_id=self.env_id,
                        seed=self.seed + i,
                        capture_video=self.eval_config.capture_media and i == 0,
                        run_name=f"{self.run_name_suffix}_eval"
                    ) for i in range(self.eval_config.env_num)
                ])
        
        self.evaluator = GymEvaluator(
            agent=self.agent,
            env_config=self.args.env,
            eval_config=self.args.eval,
            run_name_suffix=self.run_name_suffix,
            logger=self.logger,
            env_id=self.env_id,
            seed=self.args.train.seed + 1
        )

if __name__ == "__main__":
    # LunarLanderContinuous参数配置
    total_timesteps = int(3.5e4)  # 100万步，足够测试收敛性
    batch_size = int(256)
    env_num = 1  # SAC通常使用单环境
    eval_env_num = 10
    eval_episodes = 10
    num_bptt = 64
    is_test = total_timesteps != int(1e6)
    learning_starts = 28400 if is_test else 10000 # 28400用于测试rb性能是否会随segment_len的增大而下降
    ckpt_save_frequency = 0
    eval_frequency = 0 if is_test else 0.1
    
    # 针对LunarLanderContinuous优化的网络配置
    # 观察空间: 8维向量 (位置、速度、角度、角速度、腿接触等)
    # 动作空间: 2维连续 (主引擎推力 + 侧向引擎推力)
    
    args = Args(
        train=TrainConfig(
            jax_platform_name="",
            exp_name="LunarLanderContinuous-RSAC",
            ckpt_save_frequency=ckpt_save_frequency,
            resume=False,
            save_dir=f"runs/LunarLanderContinuous_RSAC",
            async_vector_env=False,
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
            trading_env_config=TradingEnvConfig(),  # 提供默认配置，但不会使用
            env_num=env_num,
        ),
        network=NetworkConfig(
            # summarizer输入就是原观察（本demo使用向量），不使用额外编码器
            shape_tickers_positions=(0, 0),
            encoder_type="none",
            actor_net_arch=[64, 64],
            critic_net_arch=[64, 64],
            actor_dropout_rate=0.0,
            critic_dropout_rate=0.0,
            # RSAC-share
            market_feature_dim=8,  # LunarLander obs dim
            agent_feature_dim=0,
            lstm_hidden_dim=128,
            lstm_num_layers=1,
            use_pretrained_summarizer_path=None,
            train_summarizer=True,
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            buffer_size=int(1e6),
            learning_starts=learning_starts,
            batch_size=batch_size,
            update_frequency=1,  # 每步都更新
            target_network_frequency=1,  # 软更新，每步更新
            gamma=0.99,
            tau=0.005,  # 连续动作用软更新，官方推荐的超参
            policy_lr=3e-4,
            q_lr=3e-4,
            autotune=True,  # 自动调节熵系数
            target_entropy_scale=1.0,  # 连续动作的标准设置
            adam_eps=1e-4,
            num_bptt=num_bptt,
        ),
        wandb=WandbConfig(
            track=os.getenv("USE_WANDB", "true").lower() == "true",
            project_name="RSAC-Continuous_LunarLander",
            entity=None
        )
    )
    
    print("开始训练RSAC-share在LunarLanderContinuous环境...")
    print(f"总步数: {total_timesteps:,}")
    print(f"批次大小: {batch_size}")
    print(f"学习开始步数: {learning_starts:,}")
    print(f"预期奖励: > 200 (成功着陆)")
    
    # 使用专门的gym训练器
    trainer = GymTrainer(args, env_id="LunarLanderContinuous-v2")
    trainer.setup()
    trainer.train() 