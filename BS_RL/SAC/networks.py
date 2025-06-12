import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable
from .config import NetworkConfig, TransformerConfig, ConvNextConfig, Cnn1DConfig

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
    def __call__(self, x, deterministic: bool):
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
    def __call__(self, x, deterministic: bool):
        residual = x
        
        x = nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(self.config.depthwise_kernel_size,),
            strides=(1,),
            padding='SAME',
            feature_group_count=self.config.embed_dim
        )(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        mlp_dim = self.config.embed_dim * self.config.ffn_dim_multiplier
        x = nn.Dense(features=mlp_dim)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.config.embed_dim)(x)
        
        if self.config.drop_path_rate > 0.0:
            x = DropPath(drop_prob=self.config.drop_path_rate)(x, deterministic=deterministic)
            
        x = residual + x
        return x

class Cnn1DEncoderBlock(nn.Module):
    config: Cnn1DConfig
    activation: Callable

    @nn.compact
    def __call__(self, x, deterministic: bool):
        residual = x
        y = nn.LayerNorm()(x)
        y = nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(self.config.kernel_size,),
            padding='SAME'
        )(y)
        y = self.activation(y)
        y = nn.Dropout(rate=self.config.dropout_rate)(y, deterministic=deterministic)

        return residual + y

class ResMLPBlock(nn.Module):
    output_dim: int
    activation: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        residual = x
        
        y = nn.Dense(self.output_dim)(x)
        y = self.activation(y)
        y = nn.LayerNorm()(y)

        if residual.shape[-1] != self.output_dim:
            residual = nn.Dense(self.output_dim, name="projection")(residual)
        
        return residual + y
class MLP(nn.Module):
    hidden_dims: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.activation(x)
        return x

class MLPWithLayerNorm(nn.Module):
    hidden_dims: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
        return x

# class ResMLP(nn.Module):
#     hidden_dims: List[int]
#     activation: Callable

#     @nn.compact
#     def __call__(self, x):
#         for i, dim in enumerate(self.hidden_dims):
#             x = ResMLPBlock(output_dim=dim, activation=self.activation, name=f"res_mlp_block_{i}")(x)
#         return x

class ResMLP(nn.Module):
    hidden_dims: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_dims):
            y = nn.Dense(dim)(x)
            y = self.activation(y)
            y = nn.LayerNorm()(y)
            x = x + y if x.shape[-1] == y.shape[-1] else y
        return x
    
class SelfAttention(nn.Module):
    config: TransformerConfig
    
    @nn.compact
    def __call__(self, x, deterministic: bool):
        attention_layer = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            dropout_rate=self.config.dropout_rate,
        )
        x_norm = nn.LayerNorm()(x)
        x_attn = attention_layer(x_norm, x_norm, deterministic=deterministic)
        x = x + x_attn
        return x

class TransformerEncoderBlock(nn.Module):
    config: TransformerConfig
    activation: Callable

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = SelfAttention(config=self.config)(x, deterministic=deterministic)
        
        mlp_dim = self.config.embed_dim * self.config.ffn_dim_multiplier
        y = nn.LayerNorm()(x)
        y = nn.Dense(mlp_dim)(y)
        y = self.activation(y)
        y = nn.Dense(self.config.embed_dim)(y)
        
        x = x + y
        return x

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

        if self.network_config.encoder_type == 'transformer':
            x_1m_config = self.network_config.transformer_layers_1m
            x_5m_config = self.network_config.transformer_layers_5m
        elif self.network_config.encoder_type == 'convnext':
            x_1m_config = self.network_config.convnext_layers_1m
            x_5m_config = self.network_config.convnext_layers_5m
        elif self.network_config.encoder_type == 'cnn1d':
            x_1m_config = self.network_config.cnn1d_layers_1m
            x_5m_config = self.network_config.cnn1d_layers_5m
        else:
            raise ValueError(f"Unknown encoder type: {self.network_config.encoder_type}")

        # Project to embed_dim before encoders
        x_1m = nn.Dense(x_1m_config.embed_dim)(x_1m)
        x_5m = nn.Dense(x_5m_config.embed_dim)(x_5m)
        
        if self.network_config.encoder_type == 'transformer':
            # Process 1m data
            for _ in range(x_1m_config.num_layers):
                x_1m = TransformerEncoderBlock(
                    config=x_1m_config,
                    activation=activation_fn
                )(x_1m, deterministic=deterministic)

            # Process 5m data
            for _ in range(x_5m_config.num_layers):
                x_5m = TransformerEncoderBlock(
                    config=x_5m_config,
                    activation=activation_fn
                )(x_5m, deterministic=deterministic)
            
        elif self.network_config.encoder_type == 'convnext':
             # Process 1m data
            for _ in range(x_1m_config.num_layers):
                x_1m = ConvNeXtBlock(
                    config=x_1m_config,
                    activation=activation_fn
                )(x_1m, deterministic=deterministic)

            # Process 5m data
            for _ in range(x_5m_config.num_layers):
                x_5m = ConvNeXtBlock(
                    config=x_5m_config,
                    activation=activation_fn
                )(x_5m, deterministic=deterministic)

        elif self.network_config.encoder_type == 'cnn1d':
             # Process 1m data
            for _ in range(x_1m_config.num_layers):
                x_1m = Cnn1DEncoderBlock(
                    config=x_1m_config,
                    activation=activation_fn
                )(x_1m, deterministic=deterministic)

            # Process 5m data
            for _ in range(x_5m_config.num_layers):
                x_5m = Cnn1DEncoderBlock(
                    config=x_5m_config,
                    activation=activation_fn
                )(x_5m, deterministic=deterministic)

        x_1m = x_1m.reshape((x_1m.shape[0], -1)) # Flatten
        x_5m = x_5m.reshape((x_5m.shape[0], -1)) # Flatten

        if self.network_config.MLP_type == "ResMLP":
            MLP_module = ResMLP
        elif self.network_config.MLP_type == "MLP_with_LayerNorm":
            MLP_module = MLPWithLayerNorm
        elif self.network_config.MLP_type == "MLP":
            MLP_module = MLP
        else:
            raise ValueError(f"Unknown MLP type: {self.network_config.MLP_type}")

        # Process rest of data
        x_rest = MLP_module(
            hidden_dims=self.network_config.MLP_layers_rest,
            activation=activation_fn
        )(x_rest)

        # Concatenate and final MLP
        concatenated = jnp.concatenate([x_1m, x_5m, x_rest], axis=-1)
        
        final_features = MLP_module(
            hidden_dims=self.network_config.MLP_layers_final,
            activation=activation_fn
        )(concatenated)
        
        return final_features

class TradingActor(nn.Module):
    network_config: NetworkConfig
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        features = TradingNetwork(network_config=self.network_config)(x, deterministic=deterministic)
        logits = nn.Dense(self.action_dim)(features)
        return logits

class TradingCritic(nn.Module):
    network_config: NetworkConfig
    action_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        features = TradingNetwork(network_config=self.network_config)(x, deterministic=deterministic)
        q_values = nn.Dense(self.action_dim)(features)
        return q_values