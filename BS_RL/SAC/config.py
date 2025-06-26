import os
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List
from TradingEnv import EnvConfig as TradingEnvConfig
from .nn.ResMLP import ResMLPConfig, ResidualStrategy, ActivationPosition, ResMLPPresets
@dataclass
class EnvConfig:
    trading_env_config: TradingEnvConfig = field(default_factory=TradingEnvConfig)
    env_num: int = 1 # sac_atari.py uses 1 env
    """the number of parallel game environments"""
    seed: int = 1
    """seed of the experiment"""


@dataclass
class AlgoConfig:
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0  # Original SAC discrete paper and CleanRL use 1.0 for hard updates for discrete
    """target smoothing coefficient (default: 1.0 for hard update)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 20000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates in environment steps"""
    target_network_frequency: int = 8000 # In environment steps
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89 # From CleanRL's sac_atari.py
    """coefficient for scaling the autotune entropy target (e.g., 0.89 for Atari)"""
    # JAX-specific Adam epsilon, matching PyTorch default for fair comparison
    adam_eps: float = 1e-4 # CleanRL used 1e-4 for PyTorch Adam, default optax Adam is 1e-8.

@dataclass
class WandbConfig:
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    project_name: str = "cleanrl-jax-sac-discrete"
    """the wandb's project name"""
    entity: Optional[str] = None
    """the entity (team) of wandb's project"""

@dataclass
class TrainConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] # Adjusted in train.py
    """the name of this experiment"""
    save_model: bool = False # This will be implicitly True if checkpointing is frequent
    """whether to save model into the `runs/{run_name}` folder (deprecated by save_dir and ckpt logic)"""
    # JAX specific
    jax_platform_name: Optional[str] = "tpu" # "cpu", "gpu", "tpu". None means JAX default.
    """The platform to run JAX on"""

    # parameters for save directories, resume, checkpointing, and evaluation
    save_dir: Optional[str] = None
    """Base directory to save all outputs (logs, checkpoints). If None, defaults to runs/{run_name}."""
    resume: bool = False
    """Whether to resume training from the latest checkpoint in save_dir/ckpts/."""
    
    ckpt_save_frequency: Union[float, int] = 0.01
    """Frequency to save a checkpoint. If > 1, it's absolute steps. If (0, 1], it's fraction of total_timesteps."""
    ckpt_save_frequency_abs_steps: Optional[int] = None # Will be populated by Args.__post_init__
    """Absolute step frequency for saving checkpoints, resolved from ckpt_save_frequency."""
    upload_model: bool = False
    """Whether to upload the model checkpoint to wandb."""
    async_vector_env: bool = False
    """whether to use async vector env"""
@dataclass
class EvalConfig:
    eval_frequency: Union[float, int] = 0.01
    """Frequency to run evaluation. If > 1, it's absolute steps. If (0, 1], it's fraction of total_timesteps."""
    eval_frequency_abs_steps: Optional[int] = None # Will be populated by Args.__post_init__
    """Absolute step frequency for running evaluation, resolved from eval_frequency."""
    cache_env: bool = True
    """Whether to cache the evaluation environment."""
    eval_episodes: int = 16
    """Number of episodes to run for evaluation during checkpointing."""
    greedy_actions: bool = True
    """Whether to use greedy actions during evaluation."""
    capture_media: bool = True
    """Whether to capture video/image during evaluation (for the first eval environment)."""
    env_num: int = 16
    """the number of parallel game environments for evaluation"""
    async_vector_env: bool = False
    """whether to use async vector env for evaluation"""
    seed: int = 996
    """the seed for the evaluation environment"""
    data_path: str = "/root/project/processed_data/test_dataset/"
    """path to the evaluation dataset"""

@dataclass
class ResNet1DConfig:
    """Configuration for a 1D ResNet encoder."""
    stage_sizes: List[int]
    num_filters: List[int]
    stem_features: int = 64

@dataclass
class ConvNextConfig:
    """Configuration for a ConvNeXt encoder block."""
    num_layers: int
    embed_dim: int
    ffn_dim_multiplier: int = 4
    drop_path_rate: float = 0.1
    depthwise_kernel_size: int = 7

@dataclass
class Cnn1DConfig:
    """Configuration for a 1D CNN encoder."""
    num_layers: int
    embed_dim: int
    kernel_size: int = 3
    dropout_rate: float = 0.1

@dataclass
class KLineEncoderConfig:
    """Configuration for a KLine encoder."""
    block_features: List[int]
    kernel_sizes: List[int]


@dataclass
class NetworkConfig:
    shape_1m: Tuple[int, int]
    shape_5m: Tuple[int, int]
    encoder_type: str = "resnet1d"  # "convnext", "cnn1d", "resnet1d", "kline"
    
    # Encoder configs
    convnext_layers_1m: ConvNextConfig = field(default_factory=lambda: ConvNextConfig(num_layers=8, embed_dim=16))
    convnext_layers_5m: ConvNextConfig = field(default_factory=lambda: ConvNextConfig(num_layers=8, embed_dim=16)) # 48=128*0.375
    cnn1d_layers_1m: Cnn1DConfig = field(default_factory=lambda: Cnn1DConfig(num_layers=8, embed_dim=16))
    cnn1d_layers_5m: Cnn1DConfig = field(default_factory=lambda: Cnn1DConfig(num_layers=8, embed_dim=16))
    resnet1d_layers_1m: ResNet1DConfig = field(default_factory=lambda: ResNet1DConfig(stage_sizes=[2, 2], num_filters=[64, 128]))
    resnet1d_layers_5m: ResNet1DConfig = field(default_factory=lambda: ResNet1DConfig(stage_sizes=[2, 2], num_filters=[32, 64]))
    # 针对1m数据，shape为(30, 14)（序列短，噪声多）的配置
    kline_encoder_1m: KLineEncoderConfig = field(default_factory=lambda: KLineEncoderConfig(
        block_features=[64, 128, 128, 256, 256, 256],
        kernel_sizes=[7, 5, 5, 3, 3, 3]
    ))
    # 针对5m数据, shape为(48, 18)（序列长，趋势更明显）的配置
    kline_encoder_5m: KLineEncoderConfig = field(default_factory=lambda: KLineEncoderConfig(
        block_features=[64, 128, 256, 256, 512, 512, 512, 512],
        kernel_sizes=[9, 7, 5, 5, 3, 3, 3, 3]
    ))

    # # MLP configs
    # MLP_layers_rest: List[int] = field(default_factory=lambda: [32, 32]) # restet有40维
    # MLP_layers_final: List[int] = field(default_factory=lambda: [128, 128, 128])
    MLP_type: str = "ResMLP" # "MLP" or "ResMLP"
    activation:str = "gelu"
    # CartPole的观察是4维向量，网络用的(64,64)，那么通道的扩展倍数是64/4=16。若要借鉴的话：40*16=640
    # 但是data_reset主要是回合+账户仓位+时间特征信息，并不能对盈利起决定性作用故不应用太多维度和层数都不应过大？
    # 卷积投影适合有空间结构的数据(相邻特征有时/空关系)，在这里用不合适
    ResMLP_rest: ResMLPConfig = field(default_factory=lambda: ResMLPConfig(
            hidden_dims=[32, 32],
            skip_initial_ln=True,
            residual_strategy=ResidualStrategy.PROJECTION,
            dropout_rate=0.1,
            name="account_state",
            description="账户状态数据处理配置"
        ))
    ResMLP_final: ResMLPConfig = field(default_factory=lambda: ResMLPConfig(
            hidden_dims=[512, 512, 512, 512, 512],
            residual_strategy=ResidualStrategy.PROJECTION,  # 线性投影，适合特征融合
            use_highway=True,                               # 门控机制，动态选择特征
            dropout_rate=0.1,
            name="time_series+account_state_fusion",
            description="时间序列+账户状态特征融合配置"
    ))
@dataclass
class Args:
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    def __post_init__(self):
        # Resolve ckpt_save_frequency
        if self.train.ckpt_save_frequency is not None:
            if 0 < self.train.ckpt_save_frequency and isinstance(self.train.ckpt_save_frequency, float):
                if self.algo.total_timesteps > 0:
                    self.train.ckpt_save_frequency_abs_steps = int(self.train.ckpt_save_frequency * self.algo.total_timesteps)
                else: # Should not happen if total_timesteps is properly set
                    self.train.ckpt_save_frequency_abs_steps = None # Or raise error
            elif self.train.ckpt_save_frequency > 1 and isinstance(self.train.ckpt_save_frequency, int):
                self.train.ckpt_save_frequency_abs_steps = int(self.train.ckpt_save_frequency)
            else: # 0 or negative, effectively disabling scheduled ckpting based on this param
                self.train.ckpt_save_frequency_abs_steps = None
        
        # Ensure a very large number if None, to effectively disable if not set through percentage or direct steps
        if self.train.ckpt_save_frequency_abs_steps is None or self.train.ckpt_save_frequency_abs_steps <= 0:
            self.train.ckpt_save_frequency_abs_steps = self.algo.total_timesteps + 1 # Effectively disable

        # Resolve eval_frequency
        if self.eval.eval_frequency is not None:
            if 0 < self.eval.eval_frequency and isinstance(self.eval.eval_frequency, float):
                if self.algo.total_timesteps > 0:
                    self.eval.eval_frequency_abs_steps = int(self.eval.eval_frequency * self.algo.total_timesteps)
                else:
                    self.eval.eval_frequency_abs_steps = None
            elif self.eval.eval_frequency > 1 and isinstance(self.eval.eval_frequency, int):
                self.eval.eval_frequency_abs_steps = int(self.eval.eval_frequency)
            else:
                self.eval.eval_frequency_abs_steps = None
        
        if self.eval.eval_frequency_abs_steps is None or self.eval.eval_frequency_abs_steps <= 0:
            self.eval.eval_frequency_abs_steps = self.algo.total_timesteps + 1
        
        # Deprecate save_model if periodic checkpointing is active
        if self.train.save_model and self.train.ckpt_save_frequency_abs_steps is not None and self.train.ckpt_save_frequency_abs_steps <= self.algo.total_timesteps:
            print("Warning: `train.save_model` is True, but periodic checkpointing is active. Only periodic checkpoints will be saved. The final model will be one of these periodic checkpoints.")
        # self.train.save_model = False # Optionally force it False
        
        # If save_dir is not set, construct the default one here or in train.py.
        # For now, train.py will handle the default path construction if save_dir is None.
        pass