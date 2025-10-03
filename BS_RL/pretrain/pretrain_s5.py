import time
from dataclasses import dataclass
from TradingEnv.Config import TradingEnvConfig
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from typing import Optional, Callable
from tqdm.auto import tqdm
from BS_RL.nn.s5 import S5Summarizer
from flax.training import checkpoints
from flax.training.train_state import TrainState
from BS_RL.common import jax_jit
import os
from dataclasses import dataclass, field
from BS_RL.jax_utils import count_params
from functools import partial
from TradingEnv.DataLoader import DataLoader, DataLoaderConfig
from TradingEnv.feature.norm_OHLCV import feat_getter_with_norm
@dataclass
class AutoencoderConfig:
    channel_dim: int = 5 # HLCV+Delta
    latent_dim: int = 36
    encoder_hidden_dim: int = 1024
    encoder_num_layers: int = 1
    decoder_hidden_dim: int = 1536
    decoder_num_layers: int = 1
@dataclass
class PretrainConfig:
    autoencoder_cfg: AutoencoderConfig = field(default_factory=lambda: AutoencoderConfig())
    noise_std: float = 0.01
    mask_prob: float = 0.01
    steps: int = 100000
    batch_size: int = 8 # 128会耗187G内存，改用8，消耗的内存约为12G
    seq_len_m: int = 1440*3 # 3 days
    num_buckets: int = 5 # 不同的seq_len会触发重编译，需将batch填充到最近的bucket
    train_data_loader_cfg: DataLoaderConfig = field(default_factory=lambda: DataLoaderConfig())
    eval_data_loader_cfg: DataLoaderConfig = field(default_factory=lambda: DataLoaderConfig())
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
    def __call__(self, last_latent:jnp.ndarray, seq_len:int):
        """
        last_latent: [B, latent_dim]
        recon: [B, seq_len, out_dim]
        """
        # latent转为hidden
        batch_size = last_latent.shape[0]
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
    def __call__(self, x:jnp.ndarray, seq_len:int, hidden:Optional[jnp.ndarray]=None):
        latent, hidden = self.encoder(x, hidden)  # encoder返回(latent, hidden)，只需要latent
        # 只用最后一个时刻的latent来reconstruct过去seq_len个时刻的输入
        last_latent = latent[:, -1, :]
        recon = self.decoder(last_latent, seq_len)
        return recon, latent, hidden

def add_noise(batch:jnp.ndarray, noise_std: float, mask_prob: float, key:jnp.ndarray):
    k1, k2 = jax.random.split(key)
    noise = noise_std * jax.random.normal(k1, batch.shape)
    noisy = batch + noise
    if mask_prob > 0.0:
        mask = jax.random.bernoulli(k2, p=mask_prob, shape=batch.shape)
        noisy = jnp.where(mask, 0.0, noisy)
    return noisy


def autoencoder_forward(params:dict, apply_fn:Callable, batch:jnp.ndarray, seq_len: int, hidden:Optional[jnp.ndarray]=None):
    recon, latent, hidden = apply_fn({"params": params}, batch, seq_len, hidden)
    # recon形状: [B, seq_len, C]，batch形状: [B, T, C]
    # 只比较最后min(T, seq_len)个时刻
    recon_len = min(batch.shape[1], seq_len)
    batch_truncated = batch[:, -recon_len:, :]
    recon_truncated = recon[:, -recon_len:, :]
    loss = jnp.mean(jnp.square(recon_truncated - batch_truncated))
    return loss, (recon, latent, hidden)

@partial(jax_jit, static_argnames=("seq_len",))
def train_step(state:TrainState, batch:jnp.ndarray, seq_len: int, hidden:Optional[jnp.ndarray]=None):
    def loss_fn(params, hidden):
        loss, (recon, latent, hidden) = autoencoder_forward(params, state.apply_fn, batch, seq_len, hidden)
        return loss, (recon, latent, hidden)
    (loss, (recon, latent, hidden)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, hidden)
    state_new = state.apply_gradients(grads=grads)
    return state_new, loss, hidden

@partial(jax_jit, static_argnames=("seq_len", "mask_prob"))
def train_step_denoise(state:TrainState, batch:jnp.ndarray, seq_len: int, key:jnp.ndarray, noise_std: float, mask_prob: float, hidden:Optional[jnp.ndarray]=None):
    key, key_new = jax.random.split(key)
    batch = add_noise(batch, noise_std=noise_std, mask_prob=mask_prob, key=key)
    state_new, loss, hidden = train_step(state, batch, seq_len, hidden)
    return state_new, loss, key_new, hidden

@partial(jax_jit, static_argnames=("seq_len",))
def eval_step(state:TrainState, batch:jnp.ndarray, seq_len: int, hidden:Optional[jnp.ndarray]=None):
    loss, (recon, latent, hidden) = autoencoder_forward(state.params, state.apply_fn, batch, seq_len, hidden)
    return loss, (recon, latent, hidden)

def data_gen(batch_size: int, seq_len_upper: int, data_loader: DataLoader):
    """
    生成训练batch。每个batch随机采样多个序列段。
    
    假设tickers_features.shape = (length, num_tickers, features_dim)
    输出batch.shape = (batch_size, seq_len, features_dim)
    """
    while True:
        seq_len = np.random.randint(1, seq_len_upper + 1)
        
        # 向量化采样：一次性为所有batch样本生成索引
        ticker_idxs = np.random.randint(0, data_loader.num_tickers, size=batch_size)
        start_idxs = np.random.randint(0, data_loader.max_idx - seq_len + 1, size=batch_size)
        
        # 使用高级索引提取所有序列
        # 构建索引数组：对每个样本生成 [start:start+seq_len] 的索引
        time_idxs = start_idxs[:, None] + np.arange(seq_len)[None, :]  # [batch_size, seq_len]
        
        # 一次性提取所有样本: [batch_size, seq_len, features_dim]
        batch = data_loader.tickers_features[time_idxs, ticker_idxs[:, None], :]
        
        yield batch
def sim_data_gen(seq_len: int, data_loader: DataLoader):
    """
    输出batch.shape = (batch_size, seq_len, features_dim)
    """
    batch_size = data_loader.num_tickers
    while True:
        
        # 向量化采样：一次性为所有batch样本生成索引
        ticker_idxs = np.arange(batch_size)
        start_idxs = np.random.randint(0, data_loader.max_idx - seq_len + 1, size=batch_size)
        
        # 使用高级索引提取所有序列
        # 构建索引数组：对每个样本生成 [start:start+seq_len] 的索引
        time_idxs = start_idxs[:, None] + np.arange(seq_len)[None, :]  # [batch_size, seq_len]
        
        # 一次性提取所有样本: [batch_size, seq_len, features_dim]
        batch = data_loader.tickers_features[time_idxs, ticker_idxs[:, None], :]
        
        yield batch
def setup_data_gen(data_loader_cfg: DataLoaderConfig, timerange: tuple[str,str], seq_len: int):
    data_loader = DataLoader(data_loader_cfg, feat_getter_with_norm)
    data_loader.setup(timerange)
    return sim_data_gen(seq_len=seq_len, data_loader=data_loader)
def test_compile(state:TrainState, key:jnp.ndarray, args: PretrainConfig):
    # 测试不同 seq_len 的编译时间
    print("Testing compilation time for different seq_len values...")
    test_seq_lens = np.array([1, 2, 3 , 256, 257, 258, 512 ])  # 使用指数增长的测试点
    # test_seq_lens = [2]
    
    hidden = None
    for batch_seq_len in test_seq_lens:
        if batch_seq_len > 1024:
            break
        batch = jax.random.normal(key, (args.batch_size, batch_seq_len, args.autoencoder_cfg.channel_dim))
        start_time = time.time()
        state, loss, key, hidden = train_step_denoise(state, batch, batch_seq_len, key=key, noise_std=args.noise_std, mask_prob=args.mask_prob, hidden=hidden)
        loss.block_until_ready()  # 等待计算完成
        end_time = time.time()
        print(f"seq_len: {batch_seq_len}, Loss: {float(loss):.6f}, Compile+Run Time: {end_time - start_time:.4f}s")
def pretrain_s5(args: PretrainConfig):
    model = Autoencoder(args.autoencoder_cfg)
    key = jax.random.PRNGKey(args.seed)
    variables = model.init(key, jnp.zeros((args.batch_size, args.seq_len_m, args.autoencoder_cfg.channel_dim)), 1)
    tx = optax.adamw(learning_rate=args.lr)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    encoder_p_count = count_params(state.params['encoder'])
    decoder_p_count = count_params(state.params['decoder'])
    total_p_count = count_params(state.params)
    print(f"编码器参数量: {encoder_p_count / 1e6:.2f}M")
    print(f"解码器参数量: {decoder_p_count / 1e6:.2f}M")
    print(f"总参数量: {total_p_count / 1e6:.2f}M")
    data_gen = setup_data_gen(DataLoaderConfig(), TradingEnvConfig.train_timerange, args.seq_len_m)
    # test_compile(state, key, args)
    print("\nStarting normal training...")
    hidden = None
    for step in tqdm(range(args.steps)):
        batch = next(data_gen)
        start_time = time.time()
        state, loss, key, hidden = train_step_denoise(state, batch, seq_len=args.seq_len_m, key=key, noise_std=args.noise_std, mask_prob=args.mask_prob, hidden=hidden)
        end_time = time.time()
        wandb.log({"loss": loss})
        if step % args.save_freq == 0:
            print(f"Step {step}, Loss: {loss}, time: {end_time - start_time:.1f}s")
            checkpoints.save_checkpoint(os.path.abspath(args.save_dir), state, step=step, keep=10, overwrite=True)
    return state
if __name__ == "__main__":
    args = PretrainConfig()
    wandb.init(project="pretrain", name=args.run_name)
    state = pretrain_s5(args)