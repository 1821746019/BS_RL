import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable, Sequence, Optional, Tuple
from .config import NetworkConfig
from .nn.Simba import RSNorm, SimbaMLP
from .nn.s5 import S5Summarizer
from .nn.lstm import LSTMSummarizer
import distrax

def get_activation(name: str) -> Callable:
    if name == "relu":
        return nn.relu
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    else:
        raise ValueError(f"Unknown activation: {name}")

class FeatExtractor(nn.Module):
    net_arch: List[int]
    dropout_rate: float = 0.1
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        x = SimbaMLP(net_arch=self.net_arch, dropout_rate=self.dropout_rate)(x, deterministic=deterministic)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    action_dim: int
    network_config: NetworkConfig
    is_continuous: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = FeatExtractor(net_arch=self.network_config.actor_net_arch, dropout_rate=self.network_config.actor_dropout_rate)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        features = activation_fn(features)
        
        if self.is_continuous:
            mean = nn.Dense(self.action_dim, name="mean")(features)
            log_std = nn.Dense(self.action_dim, name="log_std")(features)
            log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std
        else:
            logits = nn.Dense(self.action_dim)(features)
            return logits

class Critic(nn.Module):
    network_config: NetworkConfig
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool):
        activation_fn = get_activation(self.network_config.activation)
        features = FeatExtractor(net_arch=self.network_config.critic_net_arch, dropout_rate=self.network_config.critic_dropout_rate)(x, deterministic=deterministic)
        features = nn.LayerNorm(name="final_norm")(features)
        features = activation_fn(features)
        value = nn.Dense(1)(features).squeeze(-1)
        return value

class ActorCritic(nn.Module):
    action_dim: int
    network_config: NetworkConfig
    is_continuous: bool

    def setup(self):
        self.pre_norm = RSNorm(name="RSNorm_before_summarizer")
        if self.network_config.use_s5_summarizer:
            self.summarizer = S5Summarizer(
                hidden_dim=self.network_config.s5_hidden_dim,
                num_layers=self.network_config.s5_num_layers,
                delta_min=self.network_config.s5_delta_min,
                delta_max=self.network_config.s5_delta_max,
            )
            self.summarizer_hidden_dim = self.network_config.s5_hidden_dim
        else:
            self.summarizer = LSTMSummarizer(
                hidden_dim=self.network_config.lstm_hidden_dim, 
                num_layers=self.network_config.lstm_num_layers
            )
            self.summarizer_hidden_dim = self.network_config.lstm_hidden_dim

        self.actor = Actor(action_dim=self.action_dim, network_config=self.network_config, is_continuous=self.is_continuous)
        self.critic = Critic(network_config=self.network_config)

    def __call__(self, 
                 hidden_state: Tuple[jnp.ndarray, jnp.ndarray], 
                 obs: jnp.ndarray,
                 deterministic: bool) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], distrax.Distribution, jnp.ndarray]:
        
        # Apply RSNorm to obs before summarizer
        normed_obs = self.pre_norm(obs=obs, use_running_average=deterministic)
        
        market_feature_dim = self.network_config.market_feature_dim if self.network_config.market_feature_dim > 0 else normed_obs.shape[-1] - self.network_config.agent_feature_dim
        agent_feature_dim = self.network_config.agent_feature_dim
        
        market_features = normed_obs[..., :market_feature_dim]
        agent_features = normed_obs[..., market_feature_dim:] if agent_feature_dim > 0 else jnp.zeros(normed_obs.shape[:-1] + (0,))
        
        # Summarizer expects sequence input: [B, T, D]
        # For single-step inference, we add a time dimension
        is_batched_sequence = market_features.ndim == 3
        if not is_batched_sequence:
            market_features = market_features[:, None, :] # (B, D) -> (B, 1, D)

        summarizer_output, new_hidden_state = self.summarizer(market_features, initial_state=hidden_state)
        
        if not is_batched_sequence:
            summarizer_output = summarizer_output[:, -1, :] # (B, 1, H) -> (B, H)
        else:
            # For sequence input, summarizer returns [B, T+1, H], we need [B, T, H] to match agent_features
            # Skip the first timestep (initial state) and take the rest
            summarizer_output = summarizer_output[:, 1:, :] # (B, T+1, H) -> (B, T, H)

        # Actor and Critic features
        features = jnp.concatenate([summarizer_output, agent_features], axis=-1)
        
        # Potentially stop gradient for summarizer from actor
        if not self.network_config.summarizer_grad_from_actor:
            actor_features = jax.lax.stop_gradient(features)
        else:
            actor_features = features
        
        # --- Actor ---
        if self.is_continuous:
            mean, log_std = self.actor(actor_features, deterministic=deterministic)
            pi = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        else:
            logits = self.actor(actor_features, deterministic=deterministic)
            pi = distrax.Categorical(logits=logits)

        # --- Critic ---
        value = self.critic(features, deterministic=deterministic)
        
        return new_hidden_state, pi, value

    def get_initial_state(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.network_config.use_s5_summarizer:
            L = self.network_config.s5_num_layers
            H = self.network_config.s5_hidden_dim
        else:
            L = self.network_config.lstm_num_layers
            H = self.network_config.lstm_hidden_dim
        
        h0 = jnp.zeros((L, batch_size, H))
        c0 = jnp.zeros((L, batch_size, H))
        return (h0, c0)