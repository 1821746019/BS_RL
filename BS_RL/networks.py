import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable, Sequence, Optional, Tuple
from .config import NetworkConfig
from .nn.ResNetNDEncoder import ResNet8, ResNetConfig, ResNetNDEncoder
from .nn.Simba import SimbaMLPResidualBlock, RSNorm, SimbaMLP
from .nn.ModernTCN import ModernTCNEncoder
from .nn.s5 import S5Summarizer
from .nn.lstm import LSTMSummarizer
from .jax_utils import concat_valid
LOG_STD_MAX = 2
LOG_STD_MIN = -20
    
class SharedEncoder(nn.Module):
    """
    一个统一的编码器模块，可以选择性地组合ModernTCNEncoder和循环摘要器(S5/LSTM)。
    """
    network_cfg: NetworkConfig
    
    def setup(self):
        """在setup方法中创建子模块"""
        if self.network_cfg.use_s5_summarizer:
            self.summarizer = S5Summarizer(
                hidden_dim=self.network_cfg.s5_hidden_dim,
                num_layers=self.network_cfg.s5_num_layers,
                delta_min=self.network_cfg.s5_delta_min,
                delta_max=self.network_cfg.s5_delta_max,
            )
        else:
            self.summarizer = LSTMSummarizer(
                hidden_dim=self.network_cfg.lstm_hidden_dim,
                num_layers=self.network_cfg.lstm_num_layers
            )
        
        # 创建ModernTCN编码器（如果需要）
        if self.network_cfg.encoder_type == "tcn":
            self.pre_encoder = ModernTCNEncoder(
                patch_size=self.network_cfg.tcn_patch_size,
                patch_stride=self.network_cfg.tcn_patch_stride,
                dims=self.network_cfg.tcn_dims,
                num_blocks=self.network_cfg.tcn_num_blocks,
                large_kernel_sizes=self.network_cfg.tcn_large_kernel_sizes,
                small_kernel_sizes=self.network_cfg.tcn_small_kernel_sizes,
                downsample_ratio=self.network_cfg.tcn_downsample_ratio,
                post_proc="global_max_pool",
                name='tcn_encoder'
            )
        elif self.network_cfg.encoder_type == "resnet":
            self.pre_encoder = ResNet8(dropout_rate=0.1)
        else:
            self.pre_encoder = None

    def __call__(self, obs_mem: Optional[jnp.ndarray], obs_cnn_mem: Optional[jnp.ndarray], hidden_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]], training: bool):
        # obs_mem shape: (B, L, M_mem)
        # obs_cnn_mem shape: (B, L, M_cnn)
        
        # 当没有任何记忆输入时，直接返回
        if obs_mem is None and obs_cnn_mem is None:
            return None, hidden_state

        cnn_features = None
        # 1. (可选) 特征提取
        if self.pre_encoder is not None:
            if obs_cnn_mem is None:
                raise ValueError("obs_cnn_mem must be provided when using pre_encoder")
            B, L, *M_cnn = obs_cnn_mem.shape
            obs_cnn_mem = obs_cnn_mem.reshape(B*L, *M_cnn)
            features = self.pre_encoder(obs_cnn_mem, training=training)
            cnn_features = features.reshape(B, L, -1)

        x = concat_valid([obs_mem, cnn_features], axis=-1)

        # 2. 序列摘要
        outputs, new_hidden_state = self.summarizer(x, hidden_state)
            
        return outputs, new_hidden_state

def get_activation(name: str) -> Callable:
    if name == "relu":
        return nn.relu
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    else:
        raise ValueError(f"Unknown activation: {name}")

class FeatExtractor(nn.Module):
    net_arch: List[int]
    dropout_rate: float = 0
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        x = SimbaMLP(net_arch=self.net_arch, dropout_rate=self.dropout_rate)(x, deterministic=deterministic)
        return x

class Actor(nn.Module):
    network_config: NetworkConfig
    action_dim: int
    is_discrete: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        # x is [B, H+agent_feat]
        features = FeatExtractor(net_arch=self.network_config.actor_net_arch, dropout_rate=self.network_config.actor_dropout_rate)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        activation_fn = get_activation(self.network_config.activation)
        features = activation_fn(features)

        if self.is_discrete:
            logits = nn.Dense(self.action_dim)(features)
            return logits
        else:
            mean = nn.Dense(self.action_dim, name="mean")(features)
            log_std = nn.Dense(self.action_dim, name="log_std")(features)
            log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std

class Critic(nn.Module):
    network_config: NetworkConfig
    action_dim: int
    is_discrete: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool, action: Optional[jnp.ndarray] = None):
        # x is [B, H+agent_feat]
        features = FeatExtractor(net_arch=self.network_config.critic_net_arch, dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        activation_fn = get_activation(self.network_config.activation)
        features = activation_fn(features)
        
        if self.is_discrete:
            q_values = nn.Dense(self.action_dim)(features)
            return q_values
        else:
            assert action is not None, "Action must be provided for continuous critic"
            # The continuous critic in original code has different structure
            x = jnp.concatenate([features, action], axis=-1)
            x = SimbaMLP(net_arch=[self.network_config.critic_net_arch[0]], dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
            x = nn.LayerNorm()(x)
            x = activation_fn(x)
            q_value = nn.Dense(1)(x).squeeze(-1)
            return q_value

class VectorCritic(nn.Module):
    network_config: NetworkConfig
    action_dim: int
    is_discrete: bool
    n_critics: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool, action: Optional[jnp.ndarray] = None):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            network_config=self.network_config,
            action_dim=self.action_dim,
            is_discrete=self.is_discrete,
            name="SingleCritic"
        )(x, deterministic, action)
        return q_values
