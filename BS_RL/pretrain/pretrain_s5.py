from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from typing import Optional
from tqdm.auto import tqdm
from BS_RL.nn.s5 import S5Summarizer
from flax.training import checkpoints
from flax.training.train_state import TrainState
from BS_RL.common import jax_jit
import os
from dataclasses import dataclass, field
from BS_RL.jax_utils import count_params
from functools import partial
@dataclass
class AutoencoderConfig:
    channel_dim: int = 5 # HLCV+Delta
    latent_dim: int = 36
    encoder_hidden_dim: int = 1024
    encoder_num_layers: int = 3
    decoder_hidden_dim: int = 1536
    decoder_num_layers: int = 4
@dataclass
class PretrainConfig:
    autoencoder_cfg: AutoencoderConfig = field(default_factory=lambda: AutoencoderConfig())
    noise_std: float = 0.01
    mask_prob: float = 0.01
    steps: int = 100000
    batch_size: int = 8 # 128会耗187G内存，改用8，消耗的内存约为12G
    seq_len_m: int = 1440*3 # 3 days
    lr: float = 1e-3
    save_freq_pct: float = 0.1
    run_name: str = "S5SDAE"
    seed: int = 996
    
    def __post_init__(self):
        self.save_freq = int(self.steps * self.save_freq_pct)
        self.save_dir = f"runs/{self.run_name}"
class Encoder(nn.Module):
    num_layers: int
    hidden_dim: int
    latent_dim: int
    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_hidden: Optional[jnp.ndarray]=None):
        """
        outputs: [B, T, latent_dim]. T时刻的latent应能用于准确reconstruct过去min(T, seq_len)个时刻的输入
        hidden: [L, B, H]
        """
        latent_outputs, hidden = S5Summarizer(self.hidden_dim, self.num_layers)(x, initial_hidden)
        latent_outputs = nn.Dense(self.latent_dim)(latent_outputs)
        # 应用LN让latent的尺度保持稳定
        latent_outputs = nn.LayerNorm()(latent_outputs)
        
        return latent_outputs, hidden
class Decoder(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    @nn.compact
    def __call__(self, latent:jnp.ndarray, seq_len:int):
        """
        latent: [B, T, latent_dim]
        recon: [B, seq_len, out_dim]
        """
        # 只用最后一个时刻的latent来reconstruct过去seq_len个时刻的输入
        last_latent = latent[:, -1, :]
        # latent转为hidden
        batch_size = latent.shape[0]
        hidden = nn.Dense(self.num_layers * self.hidden_dim)(last_latent) # [B, H * L]
        hidden = hidden.reshape(batch_size, self.num_layers, self.hidden_dim) # [B, L, H]
        hidden = jnp.transpose(hidden, (1, 0, 2)) # [L, B, H]
        # 用零输入展开seq_len步
        zeros = jnp.zeros((batch_size, seq_len, self.hidden_dim))
        outputs, _ = S5Summarizer(self.hidden_dim, self.num_layers)(zeros, hidden) # [B, seq_len, H]
        recon = nn.Dense(self.out_dim)(outputs) # [B, seq_len, out_dim]
        return recon
class Autoencoder(nn.Module):
    cfg: AutoencoderConfig
    def setup(self):
        self.encoder = Encoder(self.cfg.encoder_num_layers, self.cfg.encoder_hidden_dim, self.cfg.latent_dim)
        self.decoder = Decoder(self.cfg.decoder_num_layers, self.cfg.decoder_hidden_dim, self.cfg.channel_dim)
    @nn.compact
    def __call__(self, x:jnp.ndarray, seq_len:Optional[int]=None):
        if seq_len is None:
            seq_len = x.shape[1]
        latent, _ = self.encoder(x)  # encoder返回(latent, hidden)，只需要latent
        recon = self.decoder(latent, seq_len)
        return recon, latent

def add_noise(batch:jnp.ndarray, noise_std: float, mask_prob: float, key:jnp.ndarray):
    noise = noise_std * jax.random.normal(key, batch.shape)
    noisy = batch + noise
    if mask_prob > 0.0:
        k1, _ = jax.random.split(key)
        mask = jax.random.bernoulli(k1, p=mask_prob, shape=batch.shape)
        noisy = jnp.where(mask, 0.0, noisy)
    return noisy
@partial(jax_jit, static_argnames=("seq_len",))
def train_step(state:TrainState, batch:jnp.ndarray, seq_len: Optional[int]=None):
    def loss_fn(params):
        recon, latent = state.apply_fn({"params": params}, batch, seq_len)
        # recon形状: [B, min(T, seq_len), C]，只和batch的最后几个时刻比较
        recon_len = recon.shape[1]
        batch_truncated = batch[:, -recon_len:, :]
        loss = jnp.mean(jnp.square(recon - batch_truncated))
        return loss, (recon, latent)
    (loss, (recon, latent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state_new = state.apply_gradients(grads=grads)
    return state_new, loss
@partial(jax_jit, static_argnames=("seq_len", "noise_std", "mask_prob"))
def train_step_denoise(state:TrainState, batch:jnp.ndarray, seq_len: Optional[int], key:jnp.ndarray, noise_std: float, mask_prob: float):
    key, key_new = jax.random.split(key)
    batch = add_noise(batch, noise_std=noise_std, mask_prob=mask_prob, key=key)
    state_new, loss = train_step(state, batch, seq_len)
    return state_new, loss, key_new
@partial(jax_jit, static_argnames=("seq_len",))
def eval_step(state:TrainState, batch:jnp.ndarray, seq_len: Optional[int]=None):
    recon, latent = state.apply_fn({"params": state.params}, batch, seq_len)
    # recon形状: [B, min(T, seq_len), C]，只和batch的最后几个时刻比较
    recon_len = recon.shape[1]
    batch_truncated = batch[:, -recon_len:, :]
    loss = jnp.mean(jnp.square(recon - batch_truncated))
    return loss, recon, latent
def pretrain_s5(args: PretrainConfig):
    model = Autoencoder(args.autoencoder_cfg)
    key = jax.random.PRNGKey(args.seed)
    variables = model.init(key, jnp.zeros((1, 1, args.autoencoder_cfg.channel_dim)))
    tx = optax.adamw(learning_rate=args.lr)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    encoder_p_count = count_params(state.params['encoder'])
    decoder_p_count = count_params(state.params['decoder'])
    total_p_count = count_params(state.params)
    print(f"编码器参数量: {encoder_p_count / 1e6:.2f}M")
    print(f"解码器参数量: {decoder_p_count / 1e6:.2f}M")
    print(f"总参数量: {total_p_count / 1e6:.2f}M")
    for step in tqdm(range(args.steps)):
        batch = jax.random.normal(key, (args.batch_size, args.seq_len_m, args.autoencoder_cfg.channel_dim))
        state, loss, key = train_step_denoise(state, batch, seq_len=args.seq_len_m, key=key, noise_std=args.noise_std, mask_prob=args.mask_prob)
        if step % args.save_freq == 0:
            print(f"Step {step}, Loss: {loss}")
            wandb.log({"loss": loss})
            checkpoints.save_checkpoint(os.path.abspath(args.save_dir), state, step=step, keep=10, overwrite=True)
    return state
if __name__ == "__main__":
    args = PretrainConfig()
    wandb.init(project="pretrain", name=args.run_name)
    state = pretrain_s5(args)