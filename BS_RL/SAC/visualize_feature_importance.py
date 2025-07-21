import os
import jax
import jax.numpy as jnp
import numpy as np
import tyro
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from flax.training import checkpoints
import gymnasium as gym

from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig, EvalConfig, NetworkConfig
from .nn.ResMLP import ResMLPConfig, ResidualStrategy
from TradingEnv import TradingEnvConfig, TradingEnv
from TradingEnv.Config import DataLoaderConfig
import warnings
from TradingEnv import DataLoader
from .common import train_env_maker
from .agent import SACAgentContinuous
from .networks import TradingActorContinuous, TradingCriticContinuous

warnings.filterwarnings("ignore", category=UserWarning, module="absl")
def get_feature_labels(env: gym.Env, agg:bool=False):
    # The environment is wrapped, so we need to access the underlying env
    # After ObsWrapper, there is RandomWrapper, then EpisodeWrapper, then the base TradingEnv
    trading_env: TradingEnv = env.unwrapped
    loader = trading_env.loader
    cfg = trading_env.cfg
    
    labels = []
    
    # Part 1: tickers_positions features
    # This part is flattened from a (num_tickers, num_features) array
    ticker_feature_names = loader.feature_names
    position_feature_names = ["marginXdirection_norm", "roi_on_margin_unleveraged"]
    # Corrected Label Logic
    pattern = "T({t})_{feature_name}" if agg else "{feature_name}"
    for i in range(loader.num_tickers):
        # For each ticker, we have window_size * feature_dim features + 2 position features
        
        # Market features for ticker i
        # The reshape in DataLoader is `(tickers, window_size, features_dim)` -> `(tickers, window_size * features_dim)`.
        # With 'C' order, this is equivalent to flattening the last two dimensions.
        for t in range(-cfg.window_size+1, 0+1):
            for feature_name in ticker_feature_names:
                labels.append(pattern.format(t=t, feature_name=feature_name))
        
        # Position features for ticker i
        for feature_name in position_feature_names:
            labels.append(f"{feature_name}")

    # Part 2. Extra features (stacked)
    # loader.extra_features is (length, extra_features_dim)
    # sliced is (window_size, extra_features_dim), then reshaped to (window_size * extra_features_dim,)
    # The flatten order is also row-major.
    # To make labels more intuitive, let's group by feature then time.
    extra_feature_names = [f"Extra_{j}" for j in range(loader.extra_features_dim)]
    for t in range(-cfg.window_size+1, 0+1):
        for feature_name in extra_feature_names:
            labels.append(pattern.format(t=t, feature_name=feature_name))
            
    # Part 3. Extra features (not stacked)
    time_feature_names = cfg.data_loader_config.time_features_to_add
    for feature_name in time_feature_names:
        labels.append(f"Time_{feature_name}_sin")
        labels.append(f"Time_{feature_name}_cos")
        
    # Part 4. Final account/episode features
    labels.append("leverage_norm")
    labels.append("dis_to_baseline_ROI")
    
    return labels
def main(args: Args):
    # This script will be filled in later
    env_id = "TradingEnv"
    window_size = 3
    use_SGD = False
    total_timesteps = int(100e6)
    batch_size = 256
    env_num = 1  # SAC通常使用单环境
    eval_env_num = 1
    exp_name = f"env_num({env_num})_window({window_size})_{env_id}_SAC_{'SGD' if use_SGD else 'AdamW'}"
    eval_episodes = 1
    
    is_test = total_timesteps == int(1e6)
    learning_starts = 10000 if not is_test else 1000
    ckpt_save_frequency = 0.01 if not is_test else 0.1
    eval_frequency = 0.01 if not is_test else 0.1
    
    # 针对TradingEnv优化的网络配置
    
    args = Args(
        train=TrainConfig(
            exp_name=exp_name,
            save_model=True,
            ckpt_save_frequency=ckpt_save_frequency,
            resume=False,  # 首次运行设为False
            save_dir=f"runs/SAC_{env_id}",
            async_vector_env=False,  # 单环境无需异步
        ),
        eval=EvalConfig(
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            greedy_actions=True,  # 评估时使用确定性动作
            env_num=eval_env_num,
            async_vector_env=False,
            capture_media=False, # We don't need to capture video for this
        ),
        env=EnvConfig(
            trading_env_config=TradingEnvConfig(data_loader_config=DataLoaderConfig(data_path="../../processed_data")),
            env_num=env_num,
        ),
        network=NetworkConfig(
            actor_net_arch=[512, 512, 512],
            critic_net_arch=[512, 512, 512, 512, 512],
            # 不需要时间序列编码器，直接用MLP
            shape_tickers_positions=(0, 0),  # 不使用
            encoder_type="none",  # 标记为不使用编码器
        ),
        algo=AlgoConfig(
            total_timesteps=total_timesteps,
            buffer_size=int(1e6),  # 大缓冲区有助于稳定训练
            learning_starts=learning_starts,
            batch_size=batch_size,
            update_frequency=1,  # 每步都更新
            target_network_frequency=1,  # 软更新，每步更新
            gamma=0.99,
            tau=0.005,  # 连续动作通常用软更新
            policy_lr=3e-4,
            q_lr=3e-4,
            autotune=True,  # 自动调节熵系数
            target_entropy_scale=1.0,  # 连续动作的标准设置
            adam_eps=1e-4,
            use_SGD=use_SGD
        ),
        wandb=WandbConfig(
            track=False, # Visualization doesn't need to be tracked
            project_name=f"{env_id}",
            entity=None
        )
    )

    ckpt_dir = Path(args.train.save_dir) / "ckpts"
    print(f"Loading checkpoints from: {ckpt_dir}")

    # Find the latest checkpoint
    latest_ckpt_path_str = checkpoints.latest_checkpoint(os.path.abspath(ckpt_dir), prefix="ckpt_step")
    if not latest_ckpt_path_str:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. Please train a model first.")
    
    print(f"Found latest checkpoint: {latest_ckpt_path_str}")

    # Setup environment to get observation and action specs
    data_loader = DataLoader(args.env.trading_env_config.data_loader_config)
    env = train_env_maker(
        seed=args.env.seed,
        config=args.env.trading_env_config,
        data_loader=data_loader
    )()

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    # Initialize agent
    key = jax.random.PRNGKey(args.env.seed)
    key_agent, key = jax.random.split(key)
    agent = SACAgentContinuous(
        action_dim=action_dim,
        observation_space_shape=obs_shape,
        key=key_agent,
        network_config=args.network,
        algo_config=args.algo,
        actor_model_cls=TradingActorContinuous,
        critic_model_cls=TradingCriticContinuous # Provide the critic model
    )

    # Restore actor state
    actor_state = agent.actor_state
    
    # Define the structure for restoration
    restore_target = {
        'actor_params': actor_state.params,
    }

    loaded_contents = checkpoints.restore_checkpoint(
        ckpt_dir=latest_ckpt_path_str,
        target=restore_target
    )

    actor_state = actor_state.replace(
        params=loaded_contents['actor_params']
    )
    
    print("Actor state restored successfully.")

    # --- Feature Importance Calculation ---
    
    # 1. Sample a batch of observations from the environment
    num_samples = 288*7
    observations = []
    obs, _ = env.reset()
    for _ in range(num_samples):
        observations.append(obs)
        action = env.action_space.sample() # Use random actions to gather diverse states
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    observations = jnp.array(observations)
    print(f"Sampled {len(observations)} observations.")

    # 2. Define the function to get gradients
    @jax.jit
    def get_action_gradients(params, single_obs):
        def get_action_sum(obs_input):
            # We need a scalar output for jacrev, so we sum the actions.
            # The gradient of the sum is the sum of the gradients.
            # The correct attribute is agent.actor_model, not agent.actor
            mean, _ = agent.actor_model.apply({'params': params}, obs_input, deterministic=True)
            return mean.sum()

        # Use jax.grad to get the gradient of the action output with respect to the input observation
        grad_fn = jax.grad(get_action_sum)
        return grad_fn(single_obs)

    # 3. Calculate gradients for all sampled observations
    # Use vmap to efficiently apply the gradient function across the batch of observations
    batched_get_gradients = jax.vmap(get_action_gradients, in_axes=(None, 0))
    gradients = batched_get_gradients(actor_state.params, observations)
    
    # 4. Aggregate feature importances
    # Take the absolute value of gradients and average over the samples
    feature_importance = jnp.mean(jnp.abs(gradients), axis=0)
    
    print("Feature importance calculated.")
    
    # 5. Visualize the feature importance
    
    # Create meaningful labels for each feature in the flattened observation space
    feature_labels = get_feature_labels(env)

    # Ensure the number of labels matches the number of features
    if len(feature_labels) != feature_importance.shape[0]:
        print(f"FATAL: Mismatch between number of labels ({len(feature_labels)}) and feature importance dimension ({feature_importance.shape[0]})")
        print("Cannot proceed with visualization.")
        # Print first 10 labels and expected shape for debugging
        print("First 10 generated labels:", feature_labels[:10])
        return # Stop execution
        
    # Aggregate feature importances
    agg_importance, agg_labels = feature_importance, get_feature_labels(env, agg=True)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(25, max(12, len(feature_labels) // 5)))
    axes0, axes1 = axes
    axes0: plt.Axes
    axes1: plt.Axes
    # --- Plot 1: Aggregated Feature Importance ---
    sns.barplot(x=np.array(agg_importance), y=agg_labels, orient='h', ax=axes0)
    axes0.set_xlabel("Average Gradient Magnitude (Aggregated)")
    axes0.set_ylabel("Features")
    axes0.set_title("Aggregated Feature Importance")
    axes0.tick_params(axis='y', labelsize=8) # Smaller font for y-axis
    
    # --- Plot 2: Detailed Feature Importance ---
    sns.barplot(x=np.array(feature_importance), y=feature_labels, orient='h', ax=axes1)
    axes1.set_xlabel("Average Gradient Magnitude (Detailed)")
    axes1.set_ylabel("Features (with Time Steps)")
    axes1.set_title("Detailed Feature Importance for SAC Policy Network")
    axes1.tick_params(axis='y', labelsize=6) # Even smaller font for detailed view
    
    plt.tight_layout()
    
    save_path = "insights/feature_importance_combined.png"
    plt.savefig(save_path)
    print(f"Combined feature importance chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args) 
