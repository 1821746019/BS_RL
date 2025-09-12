from functools import partial
from BS_RL.common import Profiler
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
import copy
import random
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from pathlib import Path
import shutil
import pickle
import collections
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax.training import checkpoints
from tqdm.auto import tqdm
import joblib
from BS_RL.config import Args
from BS_RL.common import profile, train_env_maker, MetricLogger, StatsAggregator, jax_profiler
from BS_RL.networks import ActorCritic
from BS_RL.PPOAgent import PPOAgent, TrainStateWithBatchStats
from BS_RL.eval import Evaluator
from TradingEnv import DataLoader, DataLoaderConfig
from BS_RL.rollout_buffer import RolloutBuffer
import wandb
import optax
from typing import Optional, Tuple

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

class Trainer:
    def __init__(self, args: Args):
        self.args = args
        self.num_devices = 1
        self.run_name_suffix: str
        self.wandb_run_name: str
        self.base_output_dir: Path
        self.ckpt_dir: Path
        self.initial_global_step = 0
        self.jax_key: jax.Array
        self.envs: AsyncVectorEnv| SyncVectorEnv
        self.agent : PPOAgent
        self.rb: RolloutBuffer
        self.train_state: TrainStateWithBatchStats
        self.data_loader: DataLoader
        self.evaluator: Evaluator
        self.logger: MetricLogger
        self.is_discrete: bool
        self.hidden_state: Tuple[jnp.ndarray, jnp.ndarray]

    def setup(self):
        self._setup_paths_and_run_name()
        self._setup_jax_devices()
        self._handle_resume_and_directory_setup()
        self._setup_wandb()
        self._setup_seeds_and_keys()
        self._setup_environments()
        self._setup_agent()
        self._setup_replay_buffer()
        self._setup_evaluator()
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)
    def _setup_paths_and_run_name(self):
        run_name_suffix = f"{self.args.train.exp_name}__{self.args.train.seed}__{datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')}"
        if self.args.train.save_dir:
            self.base_output_dir = Path(self.args.train.save_dir)
            self.wandb_run_name = f"{run_name_suffix}__{self.base_output_dir.name}__{os.environ.get('TUNNEL_NAME', 'unknownDevice')}" if self.base_output_dir.name else run_name_suffix
        else:
            self.base_output_dir = Path(f"runs/{run_name_suffix}")
            self.wandb_run_name = run_name_suffix
        
        self.run_name_suffix = run_name_suffix
        self.ckpt_dir = self.base_output_dir / "ckpts"
        print(f"Run name: {self.run_name_suffix}")
        print(f"Checkpoint directory: {self.ckpt_dir}")

    def _setup_jax_devices(self):
        print(f"JAX running on: {jax.devices()}")
        # self.batch_size_per_device = self.args.algo.batch_size
        # print(f"Using global batch size: {self.args.algo.batch_size}")

    def _handle_resume_and_directory_setup(self):
        restored_ckpt_path = None
        if self.args.train.resume:
            latest_ckpt_path_str = checkpoints.latest_checkpoint(os.path.abspath(self.ckpt_dir), prefix="ckpt_step")
            if latest_ckpt_path_str:
                print(f"Found latest checkpoint: {latest_ckpt_path_str}")
                try:
                    self.initial_global_step = int(Path(latest_ckpt_path_str).name.split("_")[-1])
                    print(f"Resuming from global_step {self.initial_global_step}")
                    restored_ckpt_path = latest_ckpt_path_str
                except (ValueError, IndexError):
                    print(f"Could not parse step from checkpoint directory name: {latest_ckpt_path_str}. Starting fresh.")
            else:
                print(f"Warning: Resume requested, but no checkpoints found in {self.ckpt_dir}. Starting fresh.")

        if not restored_ckpt_path:
            self.initial_global_step = 0
            if self.args.train.save_dir and self.base_output_dir.exists():
                print(f"Starting fresh: Clearing specified save_dir subdirectory {self.ckpt_dir}")
                if self.ckpt_dir.exists():
                    shutil.rmtree(self.ckpt_dir)
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.restored_ckpt_path = restored_ckpt_path

    def _setup_wandb(self):
        if not self.args.wandb.track:
            self.logger = MetricLogger(wandb_track=False)
            return
        wandb.init(
            project=self.args.wandb.project_name,
            entity=self.args.wandb.entity,
            sync_tensorboard=False,
            config=asdict(self.args),
            name=self.wandb_run_name,
            monitor_gym=True,
            save_code=True,
            resume="allow" if self.args.train.resume else None,
        )
        self.logger = MetricLogger(wandb_track=True)
        flat_args_dict = {}
        for main_key, main_value in asdict(self.args).items():
            if isinstance(main_value, dict):
                for sub_key, sub_value in main_value.items():
                    flat_args_dict[f"{main_key}.{sub_key}"] = sub_value
            else:
                flat_args_dict[main_key] = main_value
        wandb.config.update(flat_args_dict)

    def _setup_seeds_and_keys(self):
        prng_restored = False
        if self.restored_ckpt_path:
            prng_path = os.path.join(self.restored_ckpt_path, "prng_states.pkl")
            if os.path.exists(prng_path):
                try:
                    print(f"Loading PRNG states from {prng_path}...")
                    with open(prng_path, 'rb') as f:
                        prng_states = pickle.load(f)
                    random.setstate(prng_states['random_state'])
                    np.random.set_state(prng_states['np_random_state'])
                    self.jax_key = prng_states['jax_key']
                    print("PRNG states successfully restored.")
                    prng_restored = True
                except Exception as e:
                    print(f"Could not load PRNG states due to {e}. Re-initializing.")
        
        if not prng_restored:
            print("Initializing new PRNG states.")
            random.seed(self.args.train.seed)
            np.random.seed(self.args.train.seed)
            self.jax_key = jax.random.PRNGKey(self.args.train.seed)

    def _setup_environments(self):
        print("Creating training environments...")
        vec_env_cls = AsyncVectorEnv if self.args.train.async_vector_env else SyncVectorEnv
        self.envs = vec_env_cls(
            [train_env_maker(
                config=self.args.env.trading_env_config,
                data_loader_cfg=self.args.env.data_loader_cfg,
                feat_getter=self.args.env.feat_getter,
                random_choose_tickers= self.args.env.env_num > 1
            ) for i in range(self.args.env.env_num)]
        )
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        if not self.is_discrete:
            print("Continuous action space detected.")

    def _setup_agent(self):
        obs_shape = self.envs.single_observation_space.shape
        if self.is_discrete:
            action_dim = self.envs.single_action_space.n # type: ignore
            print(f"Using PPO Discrete with action dim: {action_dim}")
        else:
            action_dim = self.envs.single_action_space.shape[0] # type: ignore
            print(f"Using PPO Continuous with action dim: {action_dim}")

        key_agent, self.jax_key = jax.random.split(self.jax_key)
        self.agent = PPOAgent(
            action_dim=action_dim,
            observation_space_shape=obs_shape,
            key=key_agent,
            network_config=self.args.network,
            algo_config=self.args.algo,
            is_continuous=not self.is_discrete,
        )
        self.train_state = self.agent.train_state
        
        # initialize per-env hidden states
        self.hidden_state = self.agent.ac_model.get_initial_state(self.args.env.env_num)

        # restore if checkpoint exists
        self._initialize_or_restore_agent_states()

        agent_p_count = count_params(self.train_state.params) / 1e6
        print(f"Agent params: {agent_p_count:.2f}M")

        if self.args.wandb.track:
            wandb.summary['agent_params_m'] = agent_p_count

    def _initialize_or_restore_agent_states(self):
        if self.restored_ckpt_path:
            try:
                # A single train_state now holds everything
                loaded_contents = checkpoints.restore_checkpoint(
                    ckpt_dir=self.restored_ckpt_path,
                    target=self.train_state
                )
                self.train_state = loaded_contents
                print(f"Agent state restored from step {self.initial_global_step}.")
            except Exception as e:
                print(f"Error restoring agent state: {e}. Starting with fresh state.")
                self.initial_global_step = 0

    def _setup_replay_buffer(self):
        print("Creating rollout buffer.")
        obs_shape = self.envs.single_observation_space.shape
        if self.is_discrete:
            action_shape = ()
        else:
            action_shape = self.envs.single_action_space.shape

        self.rb = RolloutBuffer(
            num_steps=self.args.algo.num_steps,
            num_envs=self.args.env.env_num,
            obs_shape=obs_shape,
            action_shape=action_shape,
            is_discrete_action=self.is_discrete,
            gae_lambda=self.args.algo.gae_lambda,
            gamma=self.args.algo.gamma
        )

    def _setup_evaluator(self):
        if self.args.eval.eval_episodes <= 0:
            return
        print("Setting up evaluator...")

        self.evaluator = Evaluator(
            agent=self.agent,
            env_config=self.args.env,
            eval_config=self.args.eval,
            run_name_suffix=self.run_name_suffix,
            logger=self.logger,
            seed=self.args.train.seed + self.args.env.env_num + 1 # 훈련VectorEnv中env的seed依次是arange(seed, seed + i)
        )

    def train(self):
        obs, _ = self.envs.reset(seed=self.args.train.seed + self.initial_global_step)
        train_stats_aggregator = StatsAggregator()
        pbar_postfix = collections.OrderedDict()

        num_updates = self.args.algo.total_timesteps // (self.args.algo.num_steps * self.args.env.env_num)
        start_update = self.initial_global_step // (self.args.algo.num_steps * self.args.env.env_num)
        
        profiler = Profiler()
        jax_profiler.start_trace(self.base_output_dir / "trace")
        
        with tqdm(initial=self.initial_global_step, total=self.args.algo.total_timesteps, desc="Training") as pbar:
            for update in range(start_update, num_updates):
                if update == 1 and not profiler.is_running:
                    profiler.start()

                # Collect rollouts
                for step in range(self.args.algo.num_steps):
                    current_step = (update * self.args.algo.num_steps + step) * self.args.env.env_num
                    
                    self.jax_key, action_key = jax.random.split(self.jax_key)
                    actions_jax, log_probs_jax, values_jax, new_hidden_state = self.agent.get_action_and_value(
                        self.train_state, self.hidden_state, obs, action_key
                    )
                    self.hidden_state = new_hidden_state
                    
                    actions, log_probs, values = jax.device_get((actions_jax, log_probs_jax, values_jax))

                    next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                    
                    self.rb.add(obs, actions, log_probs, rewards, terminations, values)
                    obs = next_obs
                    
                    # Reset hidden state for done environments
                    done_mask = (terminations | truncations).astype(bool)
                    if done_mask.any():
                        h, c = self.hidden_state
                        h = h.at[:, done_mask, :].set(0.0)
                        c = c.at[:, done_mask, :].set(0.0)
                        self.hidden_state = (h, c)

                    # Logging
                    if "final_info" in infos:
                        for i, info_item in enumerate(infos["final_info"]):
                            if info_item and "episode" in info_item:
                                train_stats_aggregator.add(info_item)
                                if i == 0:
                                    self.logger.log_env0_episode(info_item['episode'], current_step, prefix="train")
                
                # Bootstrap value and compute advantages
                self.jax_key, value_key = jax.random.split(self.jax_key)
                last_value_jax = self.agent.get_value(self.train_state, self.hidden_state, next_obs)
                last_value = jax.device_get(last_value_jax)
                self.rb.compute_returns_and_advantages(last_value, terminations | truncations)
                
                # Update agent
                for epoch in range(self.args.algo.update_epochs):
                    for batch in self.rb.get(self.args.algo.num_minibatches):
                        self.jax_key, update_key = jax.random.split(self.jax_key)
                        
                        # For recurrent PPO, this is an approximation. A better way would be to store hidden states per minibatch.
                        # For now, we pass a zero initial state for each minibatch update.
                        initial_hidden_state_for_update = self.agent.ac_model.get_initial_state(batch['obs'].shape[0])

                        self.train_state, metrics = self.agent.update(
                            self.train_state,
                            initial_hidden_state_for_update, 
                            batch,
                            update_key
                        )

                # Logging
                current_step = (update + 1) * self.args.algo.num_steps * self.args.env.env_num
                if metrics and current_step % (self.args.train.log_freq) == 0:
                    sps = int(pbar.format_dict['rate'])
                    pbar_postfix["SPS"] = sps
                    log_data = {}
                    metrics["SPS"] = sps
                    for k, v in metrics.items():
                        log_data[f"metrics/{k}"] = float(v)
                    buffered_stats = train_stats_aggregator.get_aggregated_stats()
                    if buffered_stats:
                        for k, v in buffered_stats.items():
                            log_data[f"train_buffered/{k}"] = v
                        if 'return_mean' in buffered_stats:
                            pbar_postfix["return_mean"] = f"{buffered_stats['return_mean']:.2f}"
                    if self.args.wandb.track:
                        wandb.log(log_data, step=current_step)

                next_step = current_step
                self._run_evaluation(current_step, next_step)
                self._save_checkpoint(current_step, next_step)

                pbar.set_postfix(pbar_postfix, refresh=False)
                pbar.update(self.args.algo.num_steps * self.args.env.env_num)
        
        profiler.stop()
        profiler.print()
        jax_profiler.stop_trace()
        self.cleanup()

    def _run_evaluation(self, current_step, next_step):
        if not self.evaluator: return
        
        eval_freq = getattr(self.args.eval, 'eval_frequency_abs_steps', 0)
        if not eval_freq and hasattr(self.args.eval, 'eval_frequency') and self.args.eval.eval_frequency > 0:
            eval_freq = int(self.args.algo.total_timesteps * self.args.eval.eval_frequency)

        if not eval_freq or next_step < eval_freq or (current_step // eval_freq) >= (next_step // eval_freq):
            return
        
        eval_trigger_step = (next_step // eval_freq) * eval_freq
        tqdm.write(f"--- Evaluation Triggered at step {current_step} (effective eval step: {eval_trigger_step}) ---")
        
        eval_metrics = self.evaluator.evaluate(
            # Pass what's needed for evaluation. Now it's just the train_state.
            train_state_eval=self.train_state,
            current_train_step=current_step
        )
        
        tqdm.write(f"Evaluation at step {current_step}: {eval_metrics}")

    def _save_checkpoint(self, current_step, next_step, is_final=False):
        ckpt_freq = self.args.train.ckpt_save_frequency_abs_steps
        if not ckpt_freq or next_step < ckpt_freq or (current_step // ckpt_freq) >= (next_step // ckpt_freq):
            return
        step_for_ckpt = current_step
        print(f"--- Saving checkpoint at step {step_for_ckpt} ---")

        try:
            # Now we only need to save the single train_state
            save_target = self.train_state
            
            checkpoints.save_checkpoint(
                ckpt_dir=os.path.abspath(self.ckpt_dir), target=save_target, step=step_for_ckpt,
                prefix="ckpt_step_", keep=50, overwrite=True
            )
            saved_path = checkpoints.latest_checkpoint(os.path.abspath(self.ckpt_dir), prefix="ckpt_step")
            
            if saved_path:
                print(f"Checkpoint saved to {saved_path}")

                if self.args.wandb.track and wandb.run and self.args.train.upload_model:
                    artifact = wandb.Artifact(f"model_ckpt_{self.wandb_run_name}", type="model")
                    artifact.add_dir(str(saved_path))
                    aliases = [f"step_{step_for_ckpt}"]
                    if is_final:
                        aliases.append("final")
                    wandb.log_artifact(artifact, aliases=aliases)
                
                # No need to save replay buffer for PPO, it's transient.
                # But we should save PRNG state.
                prng_states = {
                    'random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'jax_key': self.jax_key,
                }
                with open(os.path.join(saved_path, "prng_states.pkl"), 'wb') as f:
                    pickle.dump(prng_states, f)

            else:
                print(f"Warning: Checkpoint for step {step_for_ckpt} could not be found after saving.")
        except Exception as e:
            print(f"Error saving checkpoint at step {step_for_ckpt}: {e}")
            
    def cleanup(self):
        self.envs.close()
        if self.evaluator:
            self.evaluator.close()
        if self.args.wandb.track and wandb.run:
            wandb.finish()


def train(args: Args):
    if args.train.jax_platform_name:
        jax.config.update('jax_platform_name', args.train.jax_platform_name)
    
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
