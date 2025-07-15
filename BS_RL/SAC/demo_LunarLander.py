import numpy as np
import tyro
import gymnasium as gym
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from .nn.ResMLP import ResMLPConfig, ResidualStrategy
from .train import Trainer
from .common import gym_train_env_maker, gym_eval_env_maker
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
                seed=self.args.env.seed + i
            ) for i in range(self.args.env.env_num)
        ])
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        action_type = "离散" if self.is_discrete else "连续"
        print(f"检测到{action_type}动作空间")
    
    def _setup_evaluator(self):
        if self.args.eval.eval_episodes <= 0:
            return
        print("设置评估器（gym模式）...")
        
        from .eval import Evaluator
        
        # 创建一个修改版的评估器
        class GymEvaluator(Evaluator):
            def __init__(self, agent, env_config, eval_config, run_name_suffix, logger, env_id):
                self.agent = agent
                self.env_config = env_config
                self.eval_config = eval_config
                self.run_name_suffix = run_name_suffix
                self.logger = logger
                self.env_id = env_id
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
                        seed=self.eval_config.seed + i,
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
            env_id=self.env_id
        )

if __name__ == "__main__":
    # LunarLanderContinuous参数配置
    total_timesteps = int(1e6)  # 100万步，足够测试收敛性
    batch_size = 256
    env_num = 1  # SAC通常使用单环境
    eval_env_num = 10
    eval_episodes = 10
    
    is_test = total_timesteps != int(1e6)
    learning_starts = 10000 if not is_test else 1000
    ckpt_save_frequency = 0.1 if not is_test else None  # 每10%保存一次
    eval_frequency = 0.05 if not is_test else None  # 每5%评估一次
    
    # 针对LunarLanderContinuous优化的网络配置
    # 观察空间: 8维向量 (位置、速度、角度、角速度、腿接触等)
    # 动作空间: 2维连续 (主引擎推力 + 侧向引擎推力)
    
    args = Args(
        train=TrainConfig(
            exp_name="LunarLanderContinuous-SAC",
            save_model=True,
            ckpt_save_frequency=ckpt_save_frequency,
            resume=True,  # 首次运行设为False
            save_dir=f"runs/LunarLanderContinuous_SAC",
            async_vector_env=False,  # 单环境无需异步
        ),
        eval=EvalConfig(
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            greedy_actions=True,  # 评估时使用确定性动作
            env_num=eval_env_num,
            async_vector_env=False,
            capture_media=True,  # 记录视频
            data_path="",  # gym环境不需要data_path
        ),
        env=EnvConfig(
            trading_env_config=TradingEnvConfig(),  # 提供默认配置，但不会使用
            seed=42,
            env_num=env_num,
        ),
        network=NetworkConfig(
            # LunarLander不需要时间序列编码器，直接用MLP
            shape_1m=(0, 0),  # 不使用
            shape_5m=(0, 0),  # 不使用
            encoder_type="none",  # 标记为不使用编码器
            
            # 为8维观察空间设计的ResMLP配置
            ResMLP_final=ResMLPConfig(
                hidden_dims=[256, 256, 256],  # 适中的网络深度
                add_initial_embedding_layer=True,
                residual_strategy=ResidualStrategy.PROJECTION,
                dropout_rate=0.0,  # LunarLander通常不需要dropout
                use_highway=False,
                name="lunar_lander_mlp",
                description="LunarLanderContinuous特征处理网络"
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
            project_name="SAC-Continuous_LunarLander",
            entity=None
        )
    )
    
    print("开始训练SAC在LunarLanderContinuous环境...")
    print(f"总步数: {total_timesteps:,}")
    print(f"批次大小: {batch_size}")
    print(f"学习开始步数: {learning_starts:,}")
    print(f"预期奖励: > 200 (成功着陆)")
    
    # 使用专门的gym训练器
    trainer = GymTrainer(args, env_id="LunarLanderContinuous-v2")
    trainer.setup()
    trainer.train() 