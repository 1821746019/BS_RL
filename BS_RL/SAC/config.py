import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvConfig:
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_envs: int = 1 # sac_atari.py uses 1 env
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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    project_name: str = "cleanrl-jax-sac-discrete"
    """the wandb's project name"""
    entity: Optional[str] = None
    """the entity (team) of wandb's project"""

@dataclass
class TrainConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] # Adjusted in train.py
    """the name of this experiment"""
    torch_deterministic: bool = True # For SB3 buffer, not JAX directly
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    # JAX specific
    jax_platform_name: Optional[str] = "tpu" # "cpu", "gpu", "tpu". None means JAX default.
    """The platform to run JAX on"""

@dataclass
class Args:
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)