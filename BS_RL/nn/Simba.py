import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, List

class SimbaMLPResidualBlock(nn.Module):
    scale_factor: int = 4
    hidden_dim: int = 512
    activation_fn: Callable = nn.gelu
    dropout_rate: float = 0.1
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True):
        """通过lunarLander实验，我的写法(LN -> GELU -> Dense -> GELU -> Dense -> +)
        才是最优的，性能提升最快。中间的Dense后跟LN的排名第二
        比gemini推荐的transformerFFN写法更优
        注意，用同一套参数，用tpu和cpu训练的结果是不同的!需控制变量，经过实验
        在应用RSNorm后情况发生了变化，中间有LN的性能提升更快，无LN的性能提升缓慢。
        还是保持用我最开始的想法：一致地使用LN，参考ResNetV2的写法，
        ... -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add -> ...
        """
        residual = x
        x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.scale_factor * self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.hidden_dim)(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = residual + x
        return x
    # gemini推荐的、参考transformer的更合理的写法(倒置瓶颈中间无LN)
    # @nn.compact
    # def __call__(self, x: jnp.ndarray, deterministic: bool = True):
    #     residual = x

    #     # 1. Pre-Normalization
    #     x = nn.LayerNorm()(x)

    #     # 2. MLP: Linear -> Activation -> Linear
    #     x = nn.Dense(self.scale_factor * self.hidden_dim)(x)
    #     x = self.activation_fn(x)
    #     x = nn.Dense(self.hidden_dim)(x)

    #     # 3. Dropout for regularization
    #     # x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

    #     # 4. Residual Connection
    #     output = residual + x

    #     return output
class SimbaMLP(nn.Module):
    net_arch: List[int]
    dropout_rate: float = 0.1
    activation_fn: Callable = nn.gelu
    first_projection: bool = True
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True):
        if self.first_projection:
            x = nn.Dense(self.net_arch[0])(x)
        for i, hidden_dim in enumerate(self.net_arch):
            x = SimbaMLPResidualBlock(scale_factor=4, hidden_dim=hidden_dim, dropout_rate=self.dropout_rate)(x, deterministic=deterministic)
        return x
class RSNorm(nn.Module):
    epsilon: float = 1e-8
    
    @nn.compact
    def __call__(self, obs, update_stats: bool = False):
        """
        Args:
            obs: The input observation batch.
            update_stats: If True, update running statistics.
        """
        # Define state variables to track and put them in the 'batch_stats' collection
        running_mean = self.variable('batch_stats', 'mean', lambda: jnp.zeros(obs.shape[-1], dtype=jnp.float32))
        running_var = self.variable('batch_stats', 'var', lambda: jnp.ones(obs.shape[-1], dtype=jnp.float32))
        count = self.variable('batch_stats', 'count', lambda: jnp.array(0, dtype=jnp.float32))

        # Update statistics in training mode
        if update_stats:
            # The shape of `obs` can be 2D or 3D.
            # - 2D: [num_envs, obs_dim] during live agent-environment interaction. Used to update the statistics.
            # - 3D: [batch_size, seq_len, obs_dim] from the replay buffer during training updates.
            # We must handle the 3D case here because of JAX's tracing mechanism for jax.lax.cond,
            # even though stats are not updated with 3D data in practice.
            # The goal is to calculate statistics per feature, so we reduce over batch and time axes.
            axis = (0, 1) if obs.ndim > 2 else 0
            batch_mean = jnp.mean(obs, axis=axis, dtype=jnp.float32)
            batch_var = jnp.var(obs, axis=axis)
            batch_count = obs.shape[0] * obs.shape[1] if obs.ndim > 2 else obs.shape[0]
            
            delta = batch_mean - running_mean.value
            tot_count = count.value + batch_count

            # Update mean and variance
            new_mean = running_mean.value + delta * batch_count / tot_count
            
            M2_a = running_var.value * count.value
            M2_b = batch_var * batch_count
            M2 = M2_a + M2_b + jnp.square(delta) * count.value * batch_count / tot_count
            new_var = M2 / tot_count

            # In Flax, update state by assigning to .value
            running_mean.value = new_mean
            running_var.value = new_var
            count.value = tot_count

        # Normalize using the statistics
        normalized_obs = (obs - running_mean.value) / jnp.sqrt(running_var.value + self.epsilon)
        return normalized_obs 