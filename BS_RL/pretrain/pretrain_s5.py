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
    num_epochs: int = 10
    batch_size: int = 8 # 128会耗187G内存，改用8，消耗的内存约为12G
    seq_len_m: int = 1440*3 # 3 days
    num_buckets: int = 5 # 不同的seq_len会触发重编译，需将batch填充到最近的bucket
    train_data_loader_cfg: DataLoaderConfig = field(default_factory=lambda: DataLoaderConfig())
    eval_data_loader_cfg: DataLoaderConfig = field(default_factory=lambda: DataLoaderConfig())
    lr: float = 1e-4
    save_freq_pct: float = 0.1
    run_name: str = "S5SDAE"
    seed: int = 996
    
    def __post_init__(self):
        self.save_freq = int(self.num_epochs * self.save_freq_pct)
        self.save_dir = f"runs/{self.run_name}"
    @property
    def initial_encoder_hidden(self):
        return jnp.zeros((self.autoencoder_cfg.encoder_num_layers, self.batch_size, self.autoencoder_cfg.encoder_hidden_dim)) # (L, B, H)
class Encoder(nn.Module):
    num_layers: int
    hidden_dim: int
    latent_dim: int
    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_hidden: jnp.ndarray):
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
        # last_latent作为每个时间步的输入：将last_latent重复到每个时间步: [B, latent_dim] -> [B, seq_len, latent_dim]
        last_latent = jnp.tile(last_latent[:, None, :], (1, seq_len, 1))
   
        hidden = jnp.zeros((self.num_layers, last_latent.shape[0], self.hidden_dim))
        outputs, _ = S5Summarizer(self.hidden_dim, self.num_layers)(last_latent, hidden) # [B, seq_len, H]
        recon = nn.Dense(self.out_dim)(outputs) # [B, seq_len, out_dim]
        return recon
class Autoencoder(nn.Module):
    cfg: AutoencoderConfig
    def setup(self):
        self.encoder = Encoder(self.cfg.encoder_num_layers, self.cfg.encoder_hidden_dim, self.cfg.latent_dim)
        self.decoder = Decoder(self.cfg.decoder_num_layers, self.cfg.decoder_hidden_dim, self.cfg.channel_dim)
    @nn.compact
    def __call__(self, x:jnp.ndarray, seq_len:int, hidden:jnp.ndarray):
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


def autoencoder_forward(params:dict, apply_fn:Callable, batch:jnp.ndarray, seq_len: int, hidden:jnp.ndarray):
    recon, latent, hidden = apply_fn({"params": params}, batch, seq_len, hidden)
    # 反转 recon：decoder 输出的是 t[-1] -> t[-2] -> ... -> t[0]
    recon = recon[:, ::-1, :]  # 沿时间维度反转
    
    # recon形状: [B, seq_len, C]，batch形状: [B, T, C]
    # 只比较最后min(T, seq_len)个时刻
    recon_len = min(batch.shape[1], seq_len)
    batch_truncated = batch[:, -recon_len:, :]
    recon_truncated = recon[:, -recon_len:, :]
    return recon, latent, hidden
def mse_loss(batch:jnp.ndarray, recon:jnp.ndarray):
    return jnp.mean(jnp.square(recon - batch))
@partial(jax_jit, static_argnames=("seq_len",))
def train_step(state:TrainState, batch:jnp.ndarray, seq_len: int, hidden:jnp.ndarray):
    def loss_fn(params, hidden):
        recon, latent, hidden = autoencoder_forward(params, state.apply_fn, batch, seq_len, hidden)
        return mse_loss(batch, recon), (recon, latent, hidden)
    (loss, (recon, latent, hidden)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, hidden)
    state_new = state.apply_gradients(grads=grads)
    return state_new, loss, hidden

@partial(jax_jit, static_argnames=("seq_len", "mask_prob"))
def train_step_denoise(state:TrainState, batch:jnp.ndarray, seq_len: int, key:jnp.ndarray, noise_std: float, mask_prob: float, hidden:jnp.ndarray):
    key, key_new = jax.random.split(key)
    batch_noisy = add_noise(batch, noise_std=noise_std, mask_prob=mask_prob, key=key)
    def loss_fn(params, hidden):
        # 传入batch_noisy
        recon, latent, hidden = autoencoder_forward(params, state.apply_fn, batch_noisy, seq_len, hidden)
        return mse_loss(batch, recon), (recon, latent, hidden) # recon应和干净batch计算mse
    (loss, (recon, latent, hidden)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, hidden)
    state_new = state.apply_gradients(grads=grads)
    return state_new, loss, key_new, hidden

@partial(jax_jit, static_argnames=("seq_len",))
def eval_step(state:TrainState, batch:jnp.ndarray, seq_len: int, hidden:jnp.ndarray):
    recon, latent, hidden = autoencoder_forward(state.params, state.apply_fn, batch, seq_len, hidden)
    return mse_loss(batch, recon), (recon, latent, hidden)

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
        start_idxs = np.random.randint(data_loader.min_idx, data_loader.max_idx - seq_len + 1, size=batch_size)
        
        # 使用高级索引提取所有序列
        # 构建索引数组：对每个样本生成 [start:start+seq_len] 的索引
        time_idxs = start_idxs[:, None] + np.arange(seq_len)[None, :]  # [batch_size, seq_len]
        
        # 一次性提取所有样本: [batch_size, seq_len, features_dim]
        batch = data_loader.tickers_features[time_idxs, ticker_idxs[:, None], :]
        
        yield batch
@dataclass
class DataGen:
    data_loader: DataLoader
    max_seq_len: int
    batch_size: int
    curr_idx: int = field(init=False)
    def random_idx(self, size: int):
        return np.random.randint(self.data_loader.min_idx, self.data_loader.max_idx - self.max_seq_len + 1, size=size)
    def random_ticker_idx(self, size: int):
        return np.random.randint(0, self.data_loader.num_tickers, size=size)
    def __post_init__(self):
        self.curr_idx = self.random_idx(self.batch_size)
        self.ticker_idxs = self.random_ticker_idx(self.batch_size)
    def __call__(self, seq_len: int):
        reset = np.where(self.curr_idx > self.data_loader.max_idx - self.max_seq_len, True, False)
        self.curr_idx[reset] = self.random_idx(reset.sum())
        self.ticker_idxs[reset] = self.random_ticker_idx(reset.sum())
        # 向量化采样：一次性为所有batch样本生成索引
        # 使用高级索引提取所有序列
        # 构建索引数组：对每个样本生成 [start:start+seq_len] 的索引
        time_idxs = self.curr_idx[:, None] + np.arange(seq_len)[None, :]  # [batch_size, seq_len]
        
        # 一次性提取所有样本: [batch_size, seq_len, features_dim]
        batch = self.data_loader.tickers_features[time_idxs, self.ticker_idxs[:, None], :]
        self.curr_idx += seq_len
        return batch, reset

def test_compile(state:TrainState, key:jnp.ndarray, args: PretrainConfig):
    # 测试不同 seq_len 的编译时间
    print("Testing compilation time for different seq_len values...")
    test_seq_lens = np.array([1, 2, 3 , 256, 257, 258, 512 ])  # 使用指数增长的测试点
    # test_seq_lens = [2]
    
    hidden = args.initial_encoder_hidden
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
    variables = model.init(key, jnp.zeros((args.batch_size, args.seq_len_m, args.autoencoder_cfg.channel_dim)), args.seq_len_m, args.initial_encoder_hidden)
    tx = optax.adamw(learning_rate=args.lr)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    encoder_p_count = count_params(state.params['encoder'])
    decoder_p_count = count_params(state.params['decoder'])
    total_p_count = count_params(state.params)
    print(f"编码器参数量: {encoder_p_count / 1e6:.2f}M")
    print(f"解码器参数量: {decoder_p_count / 1e6:.2f}M")
    print(f"总参数量: {total_p_count / 1e6:.2f}M")
    data_loader = DataLoader(args.train_data_loader_cfg, feat_getter_with_norm)
    data_loader.setup(TradingEnvConfig.train_timerange)
    data_gen = DataGen(data_loader, args.seq_len_m, args.batch_size)
    # test_compile(state, key, args)
    print("\nStarting normal training...")
    hidden = args.initial_encoder_hidden 
    for epoch in tqdm(range(1, args.num_epochs + 1), desc="epoch"):
        num_batches = 4*365*1440//args.seq_len_m # 4年的数据，每个batch会用3天
        for iter in tqdm(range(num_batches), desc=f"Epoch{epoch}-Iter", leave=False): 
            batch, reset = data_gen(args.seq_len_m)
            # 对于reset的样本，将其hidden状态清零。reset: (B,), hidden: (L, B, H)
            hidden = jnp.where(reset.reshape(1, -1, 1), 0.0, hidden)
            start_time = time.time()
            state, loss, key, hidden = train_step_denoise(state, batch, seq_len=args.seq_len_m, key=key, noise_std=args.noise_std, mask_prob=args.mask_prob, hidden=hidden)
            end_time = time.time()
            loss_value = float(loss)
            wandb.log({"loss": loss_value})
            if epoch == 1 and iter == 0:
                print(f"首次Compile+Run耗时: {end_time - start_time:.1f}s")
           
        print(f"\nEpoch {epoch}, Loss: {loss_value:.6f}")
        checkpoints.save_checkpoint(os.path.abspath(args.save_dir), state, step=epoch, keep=50, overwrite=True)
    return state
if __name__ == "__main__":
    args = PretrainConfig()
    wandb.init(project="pretrain", name=args.run_name)
    state = pretrain_s5(args)