import jax
import jax.numpy as jnp

# For network initialization, matching PyTorch's Kaiming Normal and constant bias
def kaiming_normal_initializer(key, shape, dtype=jnp.float32):
    # Flax's KaimingNormal is actually KaimingUniform in PyTorch terms if scale=sqrt(2)
    # PyTorch KaimingNormal uses fan_in. Flax default is fan_in.
    return jax.random.normal(key, shape, dtype) * jnp.sqrt(1.0 / shape[-2]) # Simplified for Dense/Conv

def constant_initializer(value):
    def init(key, shape, dtype=jnp.float32):
        return jnp.full(shape, value, dtype)
    return init

# Polyak averaging (already available in optax.incremental_update)
# def polyak_update(params, target_params, tau):
#     return jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), params, target_params)