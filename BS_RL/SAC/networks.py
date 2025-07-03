import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable, Sequence
from .config import NetworkConfig, ConvNextConfig, Cnn1DConfig, ResNet1DConfig
from .nn.KlineEncoder import KLineEncoder
from .nn.ResMLP import UnifiedResMLP, ResMLPConfig
from .nn.ResNet1DEncoder import ResNet1DEncoder
def get_activation(name: str) -> Callable:
    if name == "relu":
        return nn.relu
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    else:
        raise ValueError(f"Unknown activation: {name}")

class DropPath(nn.Module):
    drop_prob: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        if self.drop_prob == 0.0 or deterministic:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        rng = self.make_rng('dropout')
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
        
        return (x / keep_prob) * random_tensor

class ConvNeXtBlock(nn.Module):
    config: ConvNextConfig
    activation: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        residual = x
        
        # 完整的预激活：Norm -> Activation -> Conv
        y = nn.LayerNorm()(x)
        y = self.activation(y)  # 添加激活函数
        y = nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(self.config.depthwise_kernel_size,),
            strides=(1,),
            padding='SAME',
            feature_group_count=self.config.embed_dim
        )(y)

        # MLP部分也改为预激活
        mlp_dim = self.config.embed_dim * self.config.ffn_dim_multiplier
        y = nn.LayerNorm()(y)  # 第二个LayerNorm
        y = self.activation(y)
        y = nn.Dense(features=mlp_dim)(y)
        
        y = nn.LayerNorm()(y)  # 第三个LayerNorm
        y = self.activation(y)
        y = nn.Dense(features=self.config.embed_dim)(y)
        
        if self.config.drop_path_rate > 0.0:
            y = DropPath(drop_prob=self.config.drop_path_rate)(y, deterministic=deterministic)
            
        x = residual + y
        return x

class Cnn1DEncoderBlock(nn.Module):
    config: Cnn1DConfig
    activation: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        residual = x
        
        y = nn.LayerNorm()(x)
        y = self.activation(y)
        y = nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(self.config.kernel_size,),
            padding='SAME'
        )(y)

        y = nn.Dropout(rate=self.config.dropout_rate)(y, deterministic=deterministic)

        return residual + y

class TradingNetwork(nn.Module):
    network_config: NetworkConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        
        # Split and reshape
        dim_1m = self.network_config.shape_1m[0] * self.network_config.shape_1m[1]
        dim_5m = self.network_config.shape_5m[0] * self.network_config.shape_5m[1]
        
        x_1m = x[:, :dim_1m].reshape((-1, *self.network_config.shape_1m))
        x_5m = x[:, dim_1m:dim_1m+dim_5m].reshape((-1, *self.network_config.shape_5m))
        x_rest = x[:, dim_1m+dim_5m:]

        if self.network_config.encoder_type == 'convnext':
            x_1m_config = self.network_config.convnext_layers_1m
            x_5m_config = self.network_config.convnext_layers_5m
            EncoderBlock = ConvNeXtBlock
        elif self.network_config.encoder_type == 'cnn1d':
            x_1m_config = self.network_config.cnn1d_layers_1m
            x_5m_config = self.network_config.cnn1d_layers_5m
            EncoderBlock = Cnn1DEncoderBlock
        elif self.network_config.encoder_type == 'resnet1d':
            x_1m_config = self.network_config.resnet1d_layers_1m
            x_5m_config = self.network_config.resnet1d_layers_5m
            # ResNet1DEncoder is a full encoder, not a block
        elif self.network_config.encoder_type == 'kline':
            x_1m_config = self.network_config.kline_encoder_1m
            x_5m_config = self.network_config.kline_encoder_5m
            EncoderBlock = KLineEncoder
        else:
            raise ValueError(f"Unknown encoder type: {self.network_config.encoder_type}")

        # Process 1m and 5m data
        if self.network_config.encoder_type == 'resnet1d':
            x_1m = ResNet1DEncoder(
                config=x_1m_config,
                activation=activation_fn,
                name='resnet1d_1m'
            )(x_1m, deterministic=deterministic)
            x_5m = ResNet1DEncoder(
                config=x_5m_config,
                activation=activation_fn,
                name='resnet1d_5m'
            )(x_5m, deterministic=deterministic)
        elif self.network_config.encoder_type == 'kline':
            x_1m = KLineEncoder(
                block_features=x_1m_config.block_features,
                kernel_sizes=x_1m_config.kernel_sizes,
                activation=activation_fn,
                name='kline_1m'
            )(x_1m)
            x_5m = KLineEncoder(
                block_features=x_5m_config.block_features,
                kernel_sizes=x_5m_config.kernel_sizes,
                activation=activation_fn,
                name='kline_5m'
            )(x_5m)
        else:
            # Project to embed_dim before encoders
            x_1m = nn.Dense(x_1m_config.embed_dim, name='projection_1m')(x_1m)
            x_5m = nn.Dense(x_5m_config.embed_dim, name='projection_5m')(x_5m)

            # Process 1m data
            for i in range(x_1m_config.num_layers):
                x_1m = EncoderBlock(
                    config=x_1m_config,
                    activation=activation_fn,
                    name=f'encoder_1m_block_{i}'
                )(x_1m, deterministic=deterministic)

            # Process 5m data
            for i in range(x_5m_config.num_layers):
                x_5m = EncoderBlock(
                    config=x_5m_config,
                    activation=activation_fn,
                    name=f'encoder_5m_block_{i}'
                )(x_5m, deterministic=deterministic)

            x_1m = x_1m.reshape((x_1m.shape[0], -1)) # Flatten
            x_5m = x_5m.reshape((x_5m.shape[0], -1)) # Flatten
        
        # Process rest of data
        x_rest = UnifiedResMLP(config=self.network_config.ResMLP_rest, activation=activation_fn)(x_rest)

        # Concatenate and final MLP
        concatenated = jnp.concatenate([x_1m, x_5m, x_rest], axis=-1)
        
        final_features = UnifiedResMLP(config=self.network_config.ResMLP_final, activation=activation_fn)(concatenated)
        
        return final_features

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class TradingActorContinuous(nn.Module):
    network_config: NetworkConfig
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = TradingNetwork(network_config=self.network_config)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        features = activation_fn(features)
        
        mean = nn.Dense(self.action_dim, name="mean")(features)
        log_std = nn.Dense(self.action_dim, name="log_std")(features)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

class TradingCriticContinuous(nn.Module):
    network_config: NetworkConfig
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = TradingNetwork(network_config=self.network_config)(x, deterministic=deterministic)
        
        combined = jnp.concatenate([features, action], axis=-1)
        
        # Simple MLP for Q-value
        q = nn.Dense(256)(combined)
        q = activation_fn(q)
        q = nn.LayerNorm()(q)
        q = nn.Dense(256)(q)
        q = activation_fn(q)
        q = nn.LayerNorm()(q)
        q_value = nn.Dense(1)(q).squeeze(-1)
        
        return q_value

class TradingActorDiscrete(nn.Module):
    network_config: NetworkConfig
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = TradingNetwork(network_config=self.network_config)(x, deterministic=deterministic)
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
        features = TradingNetwork(network_config=self.network_config)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        features = activation_fn(features)
        q_values = nn.Dense(self.action_dim)(features)
        return q_values