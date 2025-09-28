import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache" # 设置缓存目录才会启用编译缓存，需在导入jax前设置
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List, Callable, Literal
from TradingEnv import TradingEnvConfig as TradingEnvConfig, DataLoaderConfig
from TradingEnv.feature import norm_OHLCV
from .nn.ResMLP import ResMLPConfig, ResidualStrategy, ActivationPosition, ResMLPPresets
import numpy as np
import jax
ENABLE_PROFILE = __name__.split(".")[0] in os.getenv("PROFILE_PACKAGES", "").split(",") # 如果PROFILE_PACKAGES中包含当前包名，则进行profile
USE_JAX_PROFILER = os.getenv("USE_JAX_PROFILER", "false").lower() == "true"

def obs_as(obs: np.ndarray, obs_as_type: Literal['auto', "obs_cnn_mem", "obs_mem", "obs_instant"] = "auto") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    as_obs_cnn_mem = lambda: (obs, None, None)
    as_obs_mem = lambda: (None, obs, None)
    as_obs_instant = lambda: (None, None, obs)
    auto_as = lambda : as_obs_cnn_mem() if obs.ndim >= 3 else as_obs_mem()

    return {
        "obs_cnn_mem": as_obs_cnn_mem,
        "obs_mem": as_obs_mem,
        "obs_instant": as_obs_instant,
        "auto": auto_as,
    }[obs_as_type]()

@dataclass
class EnvConfig:
    trading_env_config: TradingEnvConfig = field(default_factory=TradingEnvConfig)
    data_loader_cfg: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    tickers_per_env: int = 1
    feat_getter: Callable = field(default_factory=lambda: norm_OHLCV.features_getter) 
    obs_split_fn: Callable[[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = field(default_factory=lambda: obs_as)
    env_num: int = 1 # sac_atari.py uses 1 env
    """the number of parallel game environments"""

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
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 20000
    """timestep to start learning"""
    summarizer_lr: float = 3e-4 #1e-5
    """the learning rate of the summarizer network optimizer"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates in environment steps"""
    target_network_frequency: int = 8000 # In environment steps
    """the frequency of updates for the target networks"""
    updates_per_call: int = 1
    """number of gradient updates to perform inside a single JIT call to reduce host-device overhead"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    # Added alpha bounds to stabilize autotuning
    alpha_min: float = 0 # 探索温度下限，设为0等于不设置。1e-6
    """Minimum alpha when autotune is enabled (clamped)."""
    alpha_max: float = 1.0 # 1 can avoid alpha value explosion for CartPole-v1
    """Maximum alpha when autotune is enabled (clamped)."""
    target_entropy_scale_for_disc: float = 0.89 # From CleanRL's sac_atari.py
    target_entropy_scale_for_cont: float = 1 # 连续动作的标准设置，官方推荐将target_entropy设为-action_dim，所以设为1即不缩放
    """coefficient for scaling the autotune entropy target (e.g., 0.89 for Atari)"""
    # JAX-specific Adam epsilon, matching PyTorch default for fair comparison
    adam_eps: float = 1e-4 # CleanRL used 1e-4 for PyTorch Adam, default optax Adam is 1e-8.
    use_SGD: bool = False
    """whether to use SGD instead of AdamW"""
    n_critics: int = 2
    """number of critics"""
    # RSAC-share specific
    rb_seg_len: int = int(1440*7)
    """Segment length for replay buffer."""
    train_unroll_steps: int = 1440
    """sequence length for calc loss"""
    burn_in: int = 1440 # 1440m = 1d
    """Number of prefix steps used only to roll LSTM hidden state (excluded from loss)."""
    rb_min_gap: int = 1
    """Minimum gap between sequences in the same segment."""
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
    # JAX specific
    jax_platform_name: Optional[str] = None # "cpu", "cuda", "tpu". None means JAX default. 填gpu似乎会被jax识别为要用amd的gpu，用cuda才能正确用nvidia的gpu
    """The platform to run JAX on"""

    # parameters for save directories, resume, checkpointing, and evaluation
    save_dir: Optional[str] = None
    """Base directory to save all outputs (logs, checkpoints). If None, defaults to runs/{run_name}."""
    resume: bool = False
    """Whether to resume training from the latest checkpoint in save_dir/ckpts/."""
    log_freq: int = 100
    """Frequency to log metrics to terminal and wandb"""
    ckpt_save_frequency: Union[float, int] = 0.01
    """Frequency to save a checkpoint. If > 1, it's absolute steps. If (0, 1], it's fraction of total_timesteps."""
    ckpt_save_frequency_abs_steps: Optional[int] = None # Will be populated by Args.__post_init__
    """Absolute step frequency for saving checkpoints, resolved from ckpt_save_frequency."""
    upload_model: bool = False
    """Whether to upload the model checkpoint to wandb."""
    async_vector_env: bool = False
    """whether to use async vector env"""
    seed: int = 996
    np_rng: np.random.Generator = field(init=False)
    """无需外部传参初始化, 在post_init中用seed初始化"""
    def __post_init__(self):
        if self.jax_platform_name is None:
            # 自动选择可用后端
            self.jax_platform_name = ""
        jax.config.update('jax_platforms', self.jax_platform_name)
        # 重置PRNG状态
        self.np_rng = np.random.default_rng(self.seed)
@dataclass
class EvalConfig:
    eval_frequency: Union[float, int] = 0.01
    """Frequency to run evaluation. If > 1, it's absolute steps. If (0, 1], it's fraction of total_timesteps."""
    eval_frequency_abs_steps: Optional[int] = None # Will be populated by Args.__post_init__
    """Absolute step frequency for running evaluation, resolved from eval_frequency."""
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


@dataclass
class NetworkConfig:
    shape_tickers_positions: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    actor_net_arch: List[int] = field(default_factory=lambda: [512, 512, 512])
    critic_net_arch: List[int] = field(default_factory=lambda: [512, 512, 512])
    actor_dropout_rate: float = 0
    critic_dropout_rate: float = 0
    encoder_type: Literal["none", "tcn", "resnet"] = "none"  # "none"
    activation:str = "gelu"
    tcn_patch_size: int = 4
    tcn_patch_stride: int = 2 
    tcn_dims: List[int] = field(default_factory=lambda: [64, 128]) 
    tcn_num_blocks: List[int] = field(default_factory=lambda: [2, 2])
    tcn_large_kernel_sizes: List[int] = field(default_factory=lambda: [7, 9])
    tcn_small_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    tcn_downsample_ratio: int = 2
    # RSAC-share specific
    use_pretrained_summarizer_path: Optional[str] = None
    """Path to a pretrained summarizer params (Flax serialization). If None, train from scratch."""
    train_summarizer: bool = False
    """Whether to update summarizer during training. If False and pretrained path is provided, summarizer is frozen."""
    lstm_hidden_dim: int = 256
    """Hidden size of LSTM summarizer."""
    lstm_num_layers: int = 1
    """Number of stacked LSTM layers."""
    # S5 options
    use_s5_summarizer: bool = True
    """Whether to use S5 summarizer instead of LSTM (default True)."""
    s5_hidden_dim: int = 256
    """Hidden size of S5 summarizer (defaults to LSTM hidden size for compatibility)."""
    s5_num_layers: int = 1
    """Number of stacked S5 layers."""
    s5_delta_min: float = 0.001
    s5_delta_max: float = 0.1
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
            else:  # 0 or negative, effectively disabling scheduled ckpting based on this param
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
        pass