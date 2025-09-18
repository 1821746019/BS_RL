import numpy as np
import tyro
import gymnasium as gym
from BS_RL.config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from BS_RL.nn.ResMLP import ResMLPConfig, ResidualStrategy
from BS_RL.train import Trainer
from BS_RL.common import gym_train_env_maker, gym_eval_env_maker
import os
from TradingEnv import TradingEnvConfig
import popgym

class GymTrainer(Trainer):
    """适配标准gym环境的训练器"""
    
    def __init__(self, args: Args, env_id: str):
        super().__init__(args)
        self.env_id = env_id
    
    def _setup_data_loader(self):
        # gym环境不需要DataLoader
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
        obs_shape = self.envs.single_observation_space.shape
        self.obs_dim = 1 if len(obs_shape) == 0 else obs_shape[-1]
    
    def _setup_evaluator(self):
        if self.args.eval.eval_episodes <= 0:
            return
        print("设置评估器（gym模式）...")
        
        from BS_RL.eval import Evaluator
        
        class GymEvaluator(Evaluator):
            def __init__(self, agent, env_config, eval_config, run_name_suffix, logger, env_id, seed):
                self.agent = agent
                self.env_config = env_config
                self.eval_config = eval_config
                self.run_name_suffix = run_name_suffix
                self.logger = logger
                self.env_id = env_id
                self.seed = seed
                self.data_loader = None  # gym环境不需要DataLoader
                
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
    # POPGym RepeatPreviousHard-v0 参数配置
    env_id = "popgym-RepeatPreviousHard-v0" 
    # env_id="CartPole-v1") # 完全可观测环境能解决
    total_timesteps = int(1e6)
    env_num = 16
    eval_env_num = 16
    eval_episodes = 16
    rb_seg_len = 155 # 最长回合长度是155步
    batch_size = 8 #round(256 / rb_seg_len) *
    burn_in = 64 # 开始的64步无论做什么动作reward都是0，到第65步，agent必须准确地“说出”它在64步之前看到的那个数字
    train_unroll_steps = rb_seg_len - burn_in # 对于RepeatPreviousHard，理论上反向传播64步就够了
    learning_starts = 10000
    is_test = False
    ckpt_save_frequency = 0
    eval_frequency = 0 if is_test else 0.1
    updates_per_call = 16
    # 针对POPGym优化的网络配置
    # 观察空间: Box(0, 1, (1,), float32)
    # 动作空间: Discrete(3)
    
    args = Args(
        train=TrainConfig(
            jax_platform_name="",
            exp_name=f"RSAC_{env_id}",
            ckpt_save_frequency=ckpt_save_frequency,
            resume=False,
            save_dir=f"runs/RSAC_{env_id}",
            async_vector_env=False,
        ),
        eval=EvalConfig(
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            greedy_actions=True,  # 评估时使用确定性动作
            env_num=eval_env_num,
            async_vector_env=False,
            capture_media=False,  # POPGym环境通常非可视化
        ),
        env=EnvConfig(
            trading_env_config=TradingEnvConfig(),  # 提供默认配置，但不会使用
            env_num=env_num,
        ),
        network=NetworkConfig(
            # summarizer输入就是原观察（本demo使用向量），不使用额外编码器
            encoder_type="none",
            actor_net_arch=[64, 64],
            critic_net_arch=[64, 64],
            actor_dropout_rate=0.0,
            critic_dropout_rate=0.0,
            # RSAC-share
            market_feature_dim=1,  # POPGym obs dim
            agent_feature_dim=0,
            use_pretrained_summarizer_path=None,
            use_s5_summarizer=True,
            train_summarizer=True,
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            buffer_size=int(1e6),
            learning_starts=learning_starts,
            batch_size=batch_size,
            update_frequency=1,  # 每步都更新
            target_network_frequency=1000,  # 硬更新频率
            gamma=0.99,
            tau=1.0,  # 离散动作用硬更新
            summarizer_lr=3e-4,
            policy_lr=3e-4,
            q_lr=3e-4,
            autotune=True,  # 自动调节熵系数
            adam_eps=1e-4,
            rb_seg_len=rb_seg_len,
            train_unroll_steps=train_unroll_steps,
            burn_in=burn_in,
            rb_min_gap=1,
            updates_per_call=updates_per_call,
        ),
        wandb=WandbConfig(
            track=os.getenv("USE_WANDB", "true").lower() == "true",
            project_name=f"RSAC_{env_id}",
            entity=None
        )
    )
    
    print(f"开始训练RSAC-share在{env_id}环境...")
    print(f"总步数: {total_timesteps:,}")
    print(f"批次大小: {batch_size}")
    print(f"学习开始步数: {learning_starts:,}")
    print(f"预期成功率: > 0.8")
    
    # 使用专门的gym训练器
    trainer = GymTrainer(args, env_id=env_id) 
    trainer.setup()
    trainer.train() 
