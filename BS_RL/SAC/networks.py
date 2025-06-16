import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable, Sequence
from .config import NetworkConfig, ConvNextConfig, Cnn1DConfig, ResNet1DConfig

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
    def __call__(self, x, deterministic: bool):
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

class ResidualBlock1D(nn.Module):
    """1D Residual Block for Pre-activation (ResNetV2)."""
    features: int
    strides: int
    activation: Callable
    
    @nn.compact
    def __call__(self, x, deterministic: bool):
        residual = x
        
        # Pre-activation (ResNetV2 style)
        y = nn.LayerNorm(name='norm1')(x)
        y = self.activation(y)
        y = nn.Conv(
            features=self.features,
            kernel_size=(3,),
            strides=(self.strides,),
            padding='SAME',
            name='conv1'
        )(y)
        
        y = nn.LayerNorm(name='norm2')(y)
        y = self.activation(y)
        y = nn.Conv(
            features=self.features,
            kernel_size=(3,),
            padding='SAME',
            name='conv2'
        )(y)

        # Shortcut connection
        if residual.shape != y.shape:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1,),
                strides=(self.strides,),
                name='shortcut_conv'
            )(x)
            
        return residual + y

class ResNet1DEncoder(nn.Module):
    """1D ResNet Encoder."""
    config: ResNet1DConfig
    activation: Callable

    @nn.compact
    def __call__(self, x, deterministic: bool):

        # Stem
        x = nn.Conv(
            features=self.config.stem_features,
            kernel_size=(7,),
            strides=(2,),
            padding='SAME',
            name='stem_conv'
        )(x)
        x = nn.LayerNorm(name='stem_norm')(x)
        x = self.activation(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(2,), padding='SAME')

        # Stages
        for i, (num_blocks, features) in enumerate(zip(self.config.stage_sizes, self.config.num_filters)):
            for j in range(num_blocks):
                strides = 2 if j == 0 and i > 0 else 1
                x = ResidualBlock1D(
                    features=features,
                    strides=strides,
                    activation=self.activation,
                    name=f'stage_{i+1}_block_{j+1}'
                )(x, deterministic=deterministic)
        
        # In pre-activation mode, a final normalization and activation is applied before pooling.
        x = nn.LayerNorm(name='final_encoder_norm')(x)
        x = self.activation(x)
        
        # Global average pooling
        return jnp.mean(x, axis=1)

class MLP(nn.Module):
    hidden_dims: List[int]
    activation: Callable
    skip_initial_ln: bool = False

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_dims):
            y = x
            if not (self.skip_initial_ln and i == 0):
                y = nn.LayerNorm()(y)
            y = self.activation(y)
            x = nn.Dense(dim)(y)
        return x

class ResMLP(nn.Module):
    hidden_dims: List[int]
    activation: Callable
    skip_initial_ln: bool = False

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_dims):
            if x.shape[-1] != dim:
                # Project input to the hidden dimension if necessary for the residual connection
                shortcut = nn.Dense(dim, name=f"shortcut_projection_{i}")(x)
            else:
                shortcut = x

            # Pre-activation
            y = x
            if not (self.skip_initial_ln and i == 0):
                y = nn.LayerNorm(name=f"norm_{i}")(y)
            y = self.activation(y)
            y = nn.Dense(dim, name=f"dense_{i}")(y)
            
            x = shortcut + y
            
        return x

class ResMLPNoProjection(nn.Module):
    hidden_dims: List[int]
    activation: Callable
    skip_initial_ln: bool = False

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_dims):
            shortcut = x

            # Pre-activation
            y = x
            if not (self.skip_initial_ln and i == 0):
                y = nn.LayerNorm(name=f"norm_{i}")(y)
            y = self.activation(y)
            y = nn.Dense(dim, name=f"dense_{i}")(y)
            
            if shortcut.shape[-1] == y.shape[-1]:
                x = shortcut + y
            else:
                # Drop residual connection if dimensions mis-match
                x = y
            
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

        # Initial normalization
        x_1m = nn.LayerNorm(name="initial_norm_1m")(x_1m)
        x_5m = nn.LayerNorm(name="initial_norm_5m")(x_5m)
        x_rest = nn.LayerNorm(name="initial_norm_rest")(x_rest)

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
        
        MLP_type_map = {
            "MLP": MLP,
            "ResMLP": ResMLP,
            "ResMLPNoProjection": ResMLPNoProjection,
        }
        if self.network_config.MLP_type not in MLP_type_map:
            raise ValueError(f"Unknown MLP type: {self.network_config.MLP_type}")
        MLP_module = MLP_type_map[self.network_config.MLP_type]
        
        # Process rest of data
        x_rest = MLP_module(
            hidden_dims=self.network_config.MLP_layers_rest,
            activation=activation_fn,
            skip_initial_ln=True
        )(x_rest)

        # Concatenate and final MLP
        concatenated = jnp.concatenate([x_1m, x_5m, x_rest], axis=-1)
        
        final_features = MLP_module(
            hidden_dims=self.network_config.MLP_layers_final,
            activation=activation_fn
        )(concatenated)
        
        # With pre-activation, the final output comes from a linear layer and is not normalized.
        # This can lead to very large feature values, causing instability in the actor's output logits.
        # We add a final normalization and activation step to stabilize the output.
        # This mirrors the behavior of post-activation, where the final operation in a block is normalization.
        # but after applying LN to each parts of the initial input x, it seems that no need to do this anymore
        # final_features = nn.LayerNorm(name="final_norm")(final_features)
        # final_features = activation_fn(final_features)
        
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