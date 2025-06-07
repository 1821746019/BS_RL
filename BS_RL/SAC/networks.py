import flax.linen as nn
import jax.numpy as jnp

# Common CNN Encoder for Atari
class AtariEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Input x is (Batch, C, H, W) from FrameStack
        # Transpose to (Batch, H, W, C) for Flax Conv
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0  # Normalize
        
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                    kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                    kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                    kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        return x

# Actor Network for Atari (CNN based)
class ActorCNN(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # x is raw observations (Batch, C, H, W)
        encoded_x = AtariEncoder(name="encoder")(x)
        
        x = nn.Dense(features=512, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(encoded_x)
        x = nn.gelu(x)
        logits = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        return logits

# Critic Network for Atari (CNN based) - SoftQNetwork
class CriticCNN(nn.Module):
    action_dim: int # Though output is num_actions, good to have for consistency

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # x is raw observations (Batch, C, H, W)
        encoded_x = AtariEncoder(name="encoder")(x)
        
        x = nn.Dense(features=512, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(encoded_x)
        x = nn.gelu(x)
        q_values = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        return q_values

# --- For CartPole (MLP based) ---
class ActorMLP(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(features=64, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        x = nn.Dense(features=64, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        logits = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        return logits

class CriticMLP(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(features=64, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        x = nn.Dense(features=64, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        x = nn.gelu(x)
        q_values = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.constant(0.0))(x)
        return q_values