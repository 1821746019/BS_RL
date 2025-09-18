from flax import linen as nn
import jax.numpy as jnp
from typing import Optional, Tuple

class _LSTMStackStep(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x_t: jnp.ndarray):
        # carry: (h_layers, c_layers) with shape [L,B,H]
        h_layers, c_layers = carry
        cur = x_t  # [B,H]
        new_h_layers = []
        new_c_layers = []
        for i in range(self.num_layers):
            cell = nn.OptimizedLSTMCell(features=self.hidden_dim, name=f"lstm_{i}")
            (h_new, c_new), cur = cell((h_layers[i], c_layers[i]), cur)
            new_h_layers.append(h_new)
            new_c_layers.append(c_new)
        new_h = jnp.stack(new_h_layers, axis=0)
        new_c = jnp.stack(new_c_layers, axis=0)
        summary_t = new_h_layers[-1]  # [B,H]
        return (new_h, new_c), summary_t

class LSTMSummarizer(nn.Module):
    hidden_dim: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        """
        x: [B, T, D]
        Returns:
          outputs: [B, T, H] 
          final_state: (h, c) with shapes [L,B,H]
        """
        B, T, _ = x.shape
        H = self.hidden_dim
        L = self.num_layers
        x_proj = nn.Dense(H, name="in_proj")(x)  # [B,T,H]

        def init_state(bs):
            h0 = jnp.zeros((L, bs, H))
            c0 = jnp.zeros((L, bs, H))
            return (h0, c0)

        if initial_state is None:
            h, c = init_state(B)
        else:
            h, c = initial_state

        # Create a scanned module over time
        Scanned = nn.scan(
            _LSTMStackStep,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        time_scan = Scanned(hidden_dim=H, num_layers=L, name="time_scan")
        (final_h, final_c), ys = time_scan((h, c), x_proj)  # ys: [B,T,H]
        outputs = ys  # [B,T,H]
        return outputs, (final_h, final_c)
