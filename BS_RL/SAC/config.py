import os
from dataclasses import dataclass, field
from typing import Optional, Union

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

    eval_episodes: int = 16
    """Number of episodes to run for evaluation during checkpointing."""
    eval_greedy_actions: bool = True
    """Whether to use greedy actions during evaluation."""
    eval_capture_video: bool = True
    """Whether to capture video during evaluation (for the first eval environment)."""

@dataclass
class Args:
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        # Resolve ckpt_save_frequency
        if self.train.ckpt_save_frequency is not None:
            if 0 < self.train.ckpt_save_frequency <= 1.0:
                if self.algo.total_timesteps > 0:
                    self.train.ckpt_save_frequency_abs_steps = int(self.train.ckpt_save_frequency * self.algo.total_timesteps)
                else: # Should not happen if total_timesteps is properly set
                    self.train.ckpt_save_frequency_abs_steps = None # Or raise error
            elif self.train.ckpt_save_frequency > 1.0:
                self.train.ckpt_save_frequency_abs_steps = int(self.train.ckpt_save_frequency)
            else: # 0 or negative, effectively disabling scheduled ckpting based on this param
                self.train.ckpt_save_frequency_abs_steps = None
        
        # Ensure a very large number if None, to effectively disable if not set through percentage or direct steps
        if self.train.ckpt_save_frequency_abs_steps is None or self.train.ckpt_save_frequency_abs_steps <= 0:
             # If user wants to disable, they can set frequency to total_timesteps + 1 or similar large number
             # Setting to a very large number if not specified or invalid.
             # Or, could default to something like algo.total_timesteps to save only at the end if save_model was True.
             # For now, let's assume if it's None or <=0, it means no periodic checkpointing.
             # The save_model flag can be used for a final save, or we make ckpting more explicit.
             # User wants periodic ckpt, so if frequency isn't positive, it's an issue or means no ckpt.
             # Let's make it so if abs_steps is not positive, no ckpting happens via this mechanism.
             if self.train.ckpt_save_frequency_abs_steps is not None and self.train.ckpt_save_frequency_abs_steps <= 0:
                 print(f"Warning: ckpt_save_frequency resolved to {self.train.ckpt_save_frequency_abs_steps}, disabling periodic checkpointing.")
                 self.train.ckpt_save_frequency_abs_steps = self.algo.total_timesteps + 1 # Effectively disable

        # Deprecate save_model if periodic checkpointing is active
        if self.train.save_model and self.train.ckpt_save_frequency_abs_steps is not None and self.train.ckpt_save_frequency_abs_steps <= self.algo.total_timesteps:
            print("Warning: `train.save_model` is True, but periodic checkpointing is active. Only periodic checkpoints will be saved. The final model will be one of these periodic checkpoints.")
            # self.train.save_model = False # Optionally force it False
            
        # If save_dir is not set, construct the default one here or in train.py.
        # For now, train.py will handle the default path construction if save_dir is None.
        pass