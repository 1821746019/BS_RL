import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable, Sequence, Optional
from .config import NetworkConfig
from .nn.ResNet1DEncoder import ResNet1DEncoder
from .nn.Simba import SimbaMLPResidualBlock, RSNorm, SimbaMLP

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
    dropout_rate: float = 0.1
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        x = SimbaMLP(net_arch=self.net_arch, dropout_rate=self.dropout_rate)(x, deterministic=deterministic)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -20
    
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