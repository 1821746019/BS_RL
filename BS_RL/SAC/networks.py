import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable
from .config import NetworkConfig, TransformerConfig

def get_activation(name: str) -> Callable:
    if name == "relu":
        return nn.relu
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    else:
        raise ValueError(f"Unknown activation: {name}")

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

        # Project to embed_dim before transformer blocks
        x_1m = nn.Dense(self.network_config.transformer_layers_1m.embed_dim)(x_1m)
        x_5m = nn.Dense(self.network_config.transformer_layers_5m.embed_dim)(x_5m)
        
        # Process 1m data
        for _ in range(self.network_config.transformer_layers_1m.num_layers):
            x_1m = TransformerEncoderBlock(
                config=self.network_config.transformer_layers_1m,
                activation=activation_fn
            )(x_1m, deterministic=deterministic)
        x_1m = x_1m.reshape((x_1m.shape[0], -1)) # Flatten

        # Process 5m data
        for _ in range(self.network_config.transformer_layers_5m.num_layers):
            x_5m = TransformerEncoderBlock(
                config=self.network_config.transformer_layers_5m,
                activation=activation_fn
            )(x_5m, deterministic=deterministic)
        x_5m = x_5m.reshape((x_5m.shape[0], -1)) # Flatten

        # Process rest of data
        x_rest = ResMLP(
            hidden_dims=self.network_config.resMLP_layers_rest,
            activation=activation_fn
        )(x_rest)

        # Concatenate and final MLP
        concatenated = jnp.concatenate([x_1m, x_5m, x_rest], axis=-1)
        
        final_features = ResMLP(
            hidden_dims=self.network_config.resMLP_layers_final,
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