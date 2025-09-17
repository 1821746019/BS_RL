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
    total_timesteps = int(1e6)
    env_num = 16
    eval_env_num = 16
    eval_episodes = 16
    rb_seg_len = 10 # 最长回合长度似乎是155步
    batch_size = 8 #round(256 / rb_seg_len) *
    burn_in = 5 # 智能体必须准确地“说出”它在5步之前看到的那个数字，开始5步agent是不知道答案的
    train_unroll_steps = 5
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
            exp_name="POPGym-RepeatPreviousHard-RSAC",
            ckpt_save_frequency=ckpt_save_frequency,
            resume=False,
            save_dir=f"runs/POPGym_RepeatPreviousHard_RSAC",
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
            project_name="RSAC-POPGym-RepeatPreviousHard",
            entity=None
        )
    )
    
    print("开始训练RSAC-share在POPGym RepeatPreviousHard-v0环境...")
    print(f"总步数: {total_timesteps:,}")
    print(f"批次大小: {batch_size}")
    print(f"学习开始步数: {learning_starts:,}")
    print(f"预期成功率: > 0.8")
    
    # 使用专门的gym训练器
    trainer = GymTrainer(args, env_id="popgym-RepeatPreviousHard-v0")
    # trainer = GymTrainer(args, env_id="CartPole-v1")
    trainer.setup()
    trainer.train() 

"""
popgym-RepeatPreviousHard-v0 这个“游戏”的玩法和其背后的核心挑战。

这并非一个传统意义上给人玩的游戏，而是一个为训练人工智能（特别是强化学习智能体）设计的“环境”或“任务”。它的核心是考验智能体的短期记忆能力。

游戏规则
可以把这个游戏想象成一个简单的记忆力测试：

输入（Observation）: 系统会一步一步地给智能体展示一个随机数字（或符号）。

目标（Objective）: 在当前的这一步，智能体必须准确地“说出”它在5步之前看到的那个数字。

奖励（Reward）: 如果智能体回答正确，它会得到奖励；如果回答错误，则没有奖励。

一个具体的例子：
假设游戏按顺序给智能体展示了以下数字序列：

[9, 2, 5, 1, 7, 4, 6, ...]

第1步到第5步: 智能体看到了 9, 2, 5, 1, 7。在这些步骤里，因为它没有“5步前”的记忆，所以无论它输出什么都无法获得奖励。它的任务是默默记住这些数字。

第6步: 系统展示了数字 4。此时，智能体必须回忆起5步前（即第1步）看到的数字，也就是 9。如果它输出了 9，就得分。

第7步: 系统展示了数字 6。智能体必须回忆起5步前（即第2步）看到的数字，也就是 2。如果它输出了 2，就得分。

...以此类推，直到整个回合（总共200步）结束。

核心挑战：为什么叫 "Hard" 并且属于 "Popgym"？
这个任务的难点在于环境的部分可观测性 (Partially Observable)，这也是 Popgym 库所有环境的共同特点。

视野局限: 智能体在任何一步都只能看到当前的数字。它无法像我们玩游戏时一样看到屏幕上完整的历史记录。

内在记忆: 为了完成任务，智能体不能是一个简单的“反应机器”（看到X就做Y）。它必须在内部建立并维持一个记忆机制，例如一个长度为5的队列（Queue）。每看到一个新数字，就将其存入记忆，并挤掉最旧的那个。

持续运作: 这个“存入-挤出-回忆”的过程必须在整个回合中持续、准确地进行，任何一步的记忆错乱都可能导致后续的连续失败。

本质剖析
这个任务的本质是从一个持续的数据流中，分离出信号和噪声，并根据一个固定的时间延迟（time-delay）进行信息召回。它迫使智能体学习一种非常基础但至关重要的能力：状态的表征与维持。智能体必须学会，当前所见的 4 并不是它行动的依据，而仅仅是一个更新其内部“世界模型”（在这里即为短期记忆）的信号。真正的行动依据，是它自己维持的、关于5步之前的那个内部记忆。

简单来说，这个游戏就是在强迫AI学会“记事儿”，而且是记清楚“什么时候、发生了什么事”。
"""