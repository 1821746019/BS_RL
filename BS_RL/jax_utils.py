import jax.numpy as jnp
import jax

def concat_valid(arrays: list[None|jnp.ndarray], axis: int | None = 0,  dtype: jax.typing.DTypeLike | None = None):
    filtered = [ arr for arr in arrays if arr is not None and arr.size > 0 ]
    return jnp.concatenate(filtered, axis=axis, dtype=dtype)