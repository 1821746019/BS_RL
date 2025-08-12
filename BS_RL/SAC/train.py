from BS_RL.SAC.common import Profiler

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
import copy
import random
import time
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
from BS_RL.SAC.config import Args
from BS_RL.SAC.common import profile, train_env_maker, MetricLogger, StatsAggregator, jax_profiler
from BS_RL.SAC.networks import TradingActorDiscrete, TradingCriticDiscrete, TradingActorContinuous, TradingCriticContinuous
from BS_RL.SAC.agent import RSACAgentDiscrete, RSACAgentContinuous, TrainStateWithBatchStats, CriticTrainState, SimpleTrainState
from BS_RL.SAC.eval import Evaluator
from TradingEnv import DataLoader
from BS_RL.SAC.replay_buffer import RecurrentReplayBuffer
import wandb
import optax

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

class Trainer:
    def __init__(self, args: Args):
        self.args = args
        self.num_devices = 1

        self.run_name_suffix = None
        self.wandb_run_name = None
        self.base_output_dir = None
        self.ckpt_dir = None
        self.initial_global_step = 0
        self.key = None
        self.key_actions_base = None
        self.key_update_base = None
        self.envs: AsyncVectorEnv| SyncVectorEnv
        self.agent = None
        self.rb: RecurrentReplayBuffer = None
        self.actor_state: TrainStateWithBatchStats = None
        self.qf1_state: CriticTrainState = None
        self.qf2_state: CriticTrainState = None
        self.summarizer_state: SimpleTrainState = None
        self.summarizer_target_params = None
        self.log_alpha_state = None
        self.current_alpha = None
        self.data_loader = None
        self.evaluator = None
        self.logger = None
        self.is_discrete: bool
        self.hidden_h = None  # [L, N, H]
        self.hidden_c = None  # [L, N, H]

    def setup(self):
        self._setup_paths_and_run_name()
        self._setup_jax_devices()
        self._handle_resume_and_directory_setup()
        self._setup_wandb()
        self._setup_seeds_and_keys()
        self._setup_data_loader()
        self._setup_environments()
        self._setup_agent()
        self._setup_replay_buffer()
        self._setup_evaluator()
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)

    def _setup_paths_and_run_name(self):
        run_name_suffix = f"{self.args.train.exp_name}__{self.args.env.seed}__{int(time.time())}"
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
        self.batch_size_per_device = self.args.algo.batch_size
        print(f"Using global batch size: {self.args.algo.batch_size}")

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
            id=wandb.util.generate_id() if not self.restored_ckpt_path else None
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
                    self.key = prng_states['key']
                    self.key_actions_base = prng_states['key_actions_base']
                    self.key_update_base = prng_states['key_update_base']
                    print("PRNG states successfully restored.")
                    prng_restored = True
                except Exception as e:
                    print(f"Could not load PRNG states due to {e}. Re-initializing.")
        
        if not prng_restored:
            print("Initializing new PRNG states.")
            random.seed(self.args.env.seed)
            np.random.seed(self.args.env.seed)
            base_key = jax.random.PRNGKey(self.args.env.seed)
            self.key, key_for_loop = jax.random.split(base_key)
            self.key_actions_base, self.key_update_base = jax.random.split(key_for_loop)

    def _setup_data_loader(self):
        print("Initializing Trainer's DataLoader...")
        self.data_loader = DataLoader(self.args.env.trading_env_config.data_loader_config)
        print("DataLoader initialized.")

    def _setup_environments(self):
        print("Creating training environments...")
        vec_env_cls = AsyncVectorEnv if self.args.train.async_vector_env else SyncVectorEnv
        self.envs = vec_env_cls(
            [train_env_maker(
                seed=self.args.env.seed + i,
                config=self.args.env.trading_env_config,
                data_loader=self.data_loader
            ) for i in range(self.args.env.env_num)]
        )
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        if not self.is_discrete:
            print("Continuous action space detected.")

    def _setup_agent(self):
        obs_shape = self.envs.single_observation_space.shape
        if self.is_discrete:
            action_dim = self.envs.single_action_space.n
            actor_model_cls, critic_model_cls = TradingActorDiscrete, TradingCriticDiscrete
            agent_cls = RSACAgentDiscrete
            print(f"Using RSAC-Share Discrete with action dim: {action_dim}")
        else:
            action_dim = self.envs.single_action_space.shape[0]
            actor_model_cls, critic_model_cls = TradingActorContinuous, TradingCriticContinuous
            agent_cls = RSACAgentContinuous
            print(f"Using RSAC-Share Continuous with action dim: {action_dim}")

        key_agent, self.key = jax.random.split(self.key)
        self.agent = agent_cls(
            action_dim=action_dim,
            observation_space_shape=obs_shape,
            key=key_agent,
            network_config=self.args.network,
            algo_config=self.args.algo,
            actor_model_cls=actor_model_cls,
            critic_model_cls=critic_model_cls
        )
        self.actor_state = self.agent.actor_state
        self.qf1_state = self.agent.qf1_state
        self.qf2_state = self.agent.qf2_state
        self.summarizer_state = self.agent.summarizer_state
        self.summarizer_target_params = self.agent.summarizer_target_params
        self.log_alpha_state = self.agent.log_alpha_state if self.args.algo.autotune else None
        self.current_alpha = jnp.exp(self.log_alpha_state.params['log_alpha']) if self.args.algo.autotune and self.log_alpha_state else jnp.array(self.args.algo.alpha)

        # initialize per-env hidden states
        L = self.args.network.lstm_num_layers
        H = self.args.network.lstm_hidden_dim
        N = self.envs.num_envs
        self.hidden_h = jnp.zeros((L, N, H), dtype=jnp.float32)
        self.hidden_c = jnp.zeros((L, N, H), dtype=jnp.float32)

        # restore if checkpoint exists
        self._initialize_or_restore_agent_states()

        actor_p_count = count_params(self.actor_state.params) / 1e6
        critic_p_count = count_params(self.qf1_state.params) / 1e6
        summarizer_p_count = count_params(self.summarizer_state.params) / 1e6
        print(f"Actor params: {actor_p_count:.2f}M")
        print(f"Critic params: {critic_p_count:.2f}M (x2 networks)")
        print(f"Summarizer params: {summarizer_p_count:.2f}M")
        if self.args.wandb.track:
            wandb.summary['actor_params_m'] = actor_p_count
            wandb.summary['critic_params_m'] = critic_p_count
            wandb.summary['summarizer_params_m'] = summarizer_p_count

    def _initialize_or_restore_agent_states(self):
        if self.restored_ckpt_path:
            try:
                restore_target = {
                    'actor_params': self.actor_state.params,
                    'actor_opt_state': self.actor_state.opt_state,
                    'actor_batch_stats': self.actor_state.batch_stats,
                    'qf1_params': self.qf1_state.params,
                    'qf1_opt_state': self.qf1_state.opt_state,
                    'qf1_batch_stats': self.qf1_state.batch_stats,
                    'qf1_target_params': self.qf1_state.target_params,
                    'qf1_target_batch_stats': self.qf1_state.target_batch_stats,
                    'qf2_params': self.qf2_state.params,
                    'qf2_opt_state': self.qf2_state.opt_state,
                    'qf2_batch_stats': self.qf2_state.batch_stats,
                    'qf2_target_params': self.qf2_state.target_params,
                    'qf2_target_batch_stats': self.qf2_state.target_batch_stats,
                    'summarizer_params': self.summarizer_state.params,
                    'summarizer_opt_state': self.summarizer_state.opt_state,
                    'summarizer_target_params': self.summarizer_target_params,
                }
                if self.args.algo.autotune:
                    restore_target['log_alpha_params'] = self.log_alpha_state.params
                    restore_target['log_alpha_opt_state'] = self.log_alpha_state.opt_state

                loaded_contents = checkpoints.restore_checkpoint(
                    ckpt_dir=self.restored_ckpt_path,
                    target=restore_target
                )

                self.actor_state = self.actor_state.replace(
                    params=loaded_contents['actor_params'],
                    opt_state=loaded_contents['actor_opt_state'],
                    batch_stats=loaded_contents['actor_batch_stats']
                )
                self.qf1_state = self.qf1_state.replace(
                    params=loaded_contents['qf1_params'],
                    opt_state=loaded_contents['qf1_opt_state'],
                    batch_stats=loaded_contents['qf1_batch_stats'],
                    target_params=loaded_contents['qf1_target_params'],
                    target_batch_stats=loaded_contents['qf1_target_batch_stats']
                )
                self.qf2_state = self.qf2_state.replace(
                    params=loaded_contents['qf2_params'],
                    opt_state=loaded_contents['qf2_opt_state'],
                    batch_stats=loaded_contents['qf2_batch_stats'],
                    target_params=loaded_contents['qf2_target_params'],
                    target_batch_stats=loaded_contents['qf2_target_batch_stats']
                )
                self.summarizer_state = self.summarizer_state.replace(
                    params=loaded_contents['summarizer_params'],
                    opt_state=loaded_contents['summarizer_opt_state']
                )
                self.summarizer_target_params = loaded_contents['summarizer_target_params']
                if self.args.algo.autotune and 'log_alpha_params' in loaded_contents:
                    self.log_alpha_state = self.log_alpha_state.replace(
                        params=loaded_contents['log_alpha_params'],
                        opt_state=loaded_contents['log_alpha_opt_state']
                    )
                print(f"Agent states restored from step {self.initial_global_step}.")
            except Exception as e:
                print(f"Error restoring agent states: {e}. Starting with fresh states.")
                self.initial_global_step = 0

    def _setup_replay_buffer(self):
        print("Creating recurrent replay buffer.")
        obs_dim = int(np.prod(self.envs.single_observation_space.shape)) if len(self.envs.single_observation_space.shape) == 1 else self.envs.single_observation_space.shape[-1]
        if self.is_discrete:
            action_shape = ()
        else:
            action_shape = self.envs.single_action_space.shape
        self.rb = RecurrentReplayBuffer(
            obs_dim=obs_dim,
            action_shape=action_shape,
            is_discrete_action=self.is_discrete,
            capacity_episodes=max(self.args.algo.buffer_size // max(self.args.env.env_num,1), 1),
            num_envs=self.args.env.env_num,
            max_episode_len=self.args.algo.max_episode_len,
            num_bptt=self.args.algo.num_bptt,
            segment_sample=self.args.algo.segment_sample
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
        )

    def train(self):
        obs, _ = self.envs.reset(seed=self.args.env.seed + self.initial_global_step)
        # initialize first obs in rb working buffers
        # environment step loop will populate
        train_stats_aggregator = StatsAggregator()
        pbar_postfix = collections.OrderedDict()

        total_iterations = self.args.algo.total_timesteps // self.args.env.env_num
        start_iteration = self.initial_global_step // self.args.env.env_num
        profiler = Profiler()
        jax_profiler.start_trace(self.base_output_dir / "trace", True, True)
        update_cnt = 0
        with tqdm(initial=start_iteration, total=total_iterations, desc="Training") as pbar:
            for loop_iter in range(start_iteration, total_iterations):
                if update_cnt == 1:
                    profiler.start() # 首次更新，tpu预热完毕，开始profile
                current_step = loop_iter * self.args.env.env_num
                
                obs, infos = self._environment_step(obs, current_step)
                
                if "final_info" in infos and self.logger:
                    for i, info_item in enumerate(infos["final_info"]):
                        if info_item and "episode" in info_item:
                            train_stats_aggregator.add(info_item)
                            if i == 0:
                                self.logger.log_env0_episode(info_item['episode'], current_step, prefix="train")

                if current_step > self.args.algo.learning_starts:
                    if current_step % self.args.algo.update_frequency == 0:
                        update_cnt += 1
                        metrics_from_update = self._agent_update(current_step)
                        if metrics_from_update:
                            sps = int(pbar.format_dict['rate'] * self.args.env.env_num) # iter/s * env_num = step/s
                            pbar_postfix["SPS"] = sps
                            log_data = {}
                            metrics_from_update["SPS"] = sps
                            for k, v in metrics_from_update.items():
                                log_data[f"metrics/{k}"] = v
                            buffered_stats = train_stats_aggregator.get_aggregated_stats()
                            if buffered_stats:
                                for k, v in buffered_stats.items():
                                    log_data[f"train_buffered/{k}"] = v
                                if 'return_mean' in buffered_stats:
                                    pbar_postfix["return_mean"] = f"{buffered_stats['return_mean']:.2f}"
                            if self.args.wandb.track:
                                wandb.log(log_data, step=current_step)

                next_step = (loop_iter + 1) * self.args.env.env_num
                self._run_evaluation(current_step, next_step)
                self._save_checkpoint(current_step, next_step)

                pbar.set_postfix(pbar_postfix, refresh=False) #不立即刷新提升性能 4.73/37.15
                pbar.update(1)
        profiler.stop()
        profiler.print()
        jax_profiler.stop_trace()
        self.cleanup()

    def _environment_step(self, obs, current_step):
        self.key_actions_base, key_actions_step = jax.random.split(self.key_actions_base)
        if current_step < self.args.algo.learning_starts:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            # keep hidden states unchanged
        else:
            jax_obs = jnp.asarray(obs)
            actions_jax, new_h, new_c = self.agent.select_action(self.actor_state, self.summarizer_state.params, jax_obs, self.hidden_h, self.hidden_c, key_actions_step, deterministic=False)
            actions = np.array(jax.device_get(actions_jax))
            self.hidden_h = jax.device_get(new_h)
            self.hidden_c = jax.device_get(new_c)

        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos and infos["final_observation"][idx] is not None:
                real_next_obs[idx] = infos["final_observation"][idx]
        # add to recurrent buffer
        self.rb.add_batch(obs.astype(np.float32), real_next_obs.astype(np.float32), actions.astype(np.int32 if self.is_discrete else np.float32), rewards.astype(np.float32), (terminations | truncations).astype(np.float32))

        # reset hidden states on done envs
        done_mask = (terminations | truncations).astype(bool)
        if done_mask.any():
            L, N, H = self.hidden_h.shape
            hh = np.array(self.hidden_h)
            hc = np.array(self.hidden_c)
            hh[:, done_mask, :] = 0.0
            hc[:, done_mask, :] = 0.0
            self.hidden_h = jnp.asarray(hh)
            self.hidden_c = jnp.asarray(hc)
        return next_obs, infos

    @profile
    def _agent_update(self, current_step):
        self.key_update_base, key_update_step = jax.random.split(self.key_update_base)
        # allow early start when enough segments exist
        if not self.rb.can_sample(self.args.algo.batch_size):
            return None
        batch_np = self.rb.sample(self.args.algo.batch_size)
        # pack into jnp
        data = {
            'o': jnp.asarray(batch_np['o']),
            'a': jnp.asarray(batch_np['a']) if self.is_discrete else jnp.asarray(batch_np['a']).astype(jnp.float32),
            'r': jnp.asarray(batch_np['r']),
            'd': jnp.asarray(batch_np['d']),
            'm': jnp.asarray(batch_np['m']),
        }
        log_alpha_arg = self.log_alpha_state if self.args.algo.autotune else None

        # one update step
        actor_state_new, qf1_state_new, qf2_state_new, summarizer_state_new, returned_log_alpha, self.current_alpha, metrics = self.agent._update(
            self.actor_state, self.qf1_state, self.qf2_state, self.summarizer_state, self.summarizer_target_params, log_alpha_arg, data, key_update_step
        )

        self.actor_state = actor_state_new
        self.qf1_state = qf1_state_new
        self.qf2_state = qf2_state_new
        self.summarizer_state = summarizer_state_new
        if self.args.algo.autotune:
            self.log_alpha_state = returned_log_alpha

        # target network update (soft/hard)
        do_target_update = (current_step // self.args.algo.update_frequency) % max(self.args.algo.target_network_frequency // max(self.args.algo.update_frequency,1), 1) == 0
        if do_target_update:
            tau = self.args.algo.tau
            self.qf1_state = self.qf1_state.replace(
                target_params=optax.incremental_update(self.qf1_state.params, self.qf1_state.target_params, tau),
                target_batch_stats=optax.incremental_update(self.qf1_state.batch_stats, self.qf1_state.target_batch_stats, tau) if self.qf1_state.batch_stats is not None else self.qf1_state.target_batch_stats
            )
            self.qf2_state = self.qf2_state.replace(
                target_params=optax.incremental_update(self.qf2_state.params, self.qf2_state.target_params, tau),
                target_batch_stats=optax.incremental_update(self.qf2_state.batch_stats, self.qf2_state.target_batch_stats, tau) if self.qf2_state.batch_stats is not None else self.qf2_state.target_batch_stats
            )
            # summarizer target uses same tau
            self.summarizer_target_params = optax.incremental_update(self.summarizer_state.params, self.summarizer_target_params, tau)

        if current_step % (self.args.algo.update_frequency * 100) == 0:
            return {f"{k}": v for k, v in metrics.items()}
        return None

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
            actor_state_eval=self.actor_state,
            summarizer_params_eval=self.summarizer_state.params,
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
            save_target = {
                'actor_params': self.actor_state.params,
                'actor_opt_state': self.actor_state.opt_state,
                'actor_batch_stats': self.actor_state.batch_stats,
                'qf1_params': self.qf1_state.params,
                'qf1_opt_state': self.qf1_state.opt_state,
                'qf1_batch_stats': self.qf1_state.batch_stats,
                'qf1_target_params': self.qf1_state.target_params,
                'qf1_target_batch_stats': self.qf1_state.target_batch_stats,
                'qf2_params': self.qf2_state.params,
                'qf2_opt_state': self.qf2_state.opt_state,
                'qf2_batch_stats': self.qf2_state.batch_stats,
                'qf2_target_params': self.qf2_state.target_params,
                'qf2_target_batch_stats': self.qf2_state.target_batch_stats,
                'summarizer_params': self.summarizer_state.params,
                'summarizer_opt_state': self.summarizer_state.opt_state,
                'summarizer_target_params': self.summarizer_target_params,
            }
            if self.args.algo.autotune and self.log_alpha_state:
                save_target['log_alpha_params'] = self.log_alpha_state.params
                save_target['log_alpha_opt_state'] = self.log_alpha_state.opt_state

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
                
                rb_path = os.path.join(saved_path, "replay_buffer.joblib.gz")
                joblib.dump(self.rb, rb_path, compress='gzip')

                prng_states = {
                    'random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'key': self.key,
                    'key_actions_base': self.key_actions_base,
                    'key_update_base': self.key_update_base,
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
