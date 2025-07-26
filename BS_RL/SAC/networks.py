import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable, Sequence
from .config import NetworkConfig, ConvNextConfig, Cnn1DConfig, ResNet1DConfig
from .nn.ResMLP import UnifiedResMLP, ResMLPConfig
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
        # 对x(obs)应用RSNorm
        x = RSNorm(name="rs_norm")(obs=x, use_running_average=deterministic)
        x = SimbaMLP(net_arch=self.net_arch, dropout_rate=self.dropout_rate)(x, deterministic=deterministic)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -20
    
class TradingActorContinuous(nn.Module):
    network_config: NetworkConfig
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        x = FeatExtractor(net_arch=self.network_config.actor_net_arch, dropout_rate=self.network_config.actor_dropout_rate)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        mean = nn.Dense(self.action_dim, name="mean")(x)
        log_std = nn.Dense(self.action_dim, name="log_std")(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

class TradingCriticContinuous(nn.Module):
    network_config: NetworkConfig
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, deterministic: bool):
        x = FeatExtractor(net_arch=self.network_config.actor_net_arch, dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = jnp.concatenate([x, action], axis=-1)
        # 将市场账户仓位特征和action拼接后，多用一层处理，
        x = SimbaMLP(net_arch=[self.network_config.actor_net_arch[0]], dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        q_value = nn.Dense(1)(x).squeeze(-1)
        
        return q_value

class TradingActorDiscrete(nn.Module):
    network_config: NetworkConfig
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = FeatExtractor(net_arch=self.network_config.actor_net_arch, dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        features = activation_fn(features)
        logits = nn.Dense(self.action_dim)(features)
        return logits

class TradingCriticDiscrete(nn.Module):
    network_config: NetworkConfig
    action_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = FeatExtractor(net_arch=self.network_config.actor_net_arch, dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        features = activation_fn(features)
        q_values = nn.Dense(self.action_dim)(features)
        return q_values