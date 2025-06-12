import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache" # 设置缓存目录才会启用编译缓存
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
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from flax.training import checkpoints
import flax.jax_utils
from tqdm.auto import tqdm
import joblib

from .config import Args
from .common import train_env_maker, MetricLogger, StatsAggregator
from .networks import TradingActor, TradingCritic
from .agent import SACAgent
from .eval import Evaluator
from TradingEnv import DataLoader
import wandb

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

class Trainer:
    def __init__(self, args: Args):
        self.args = args
        self.num_devices = jax.local_device_count()

        self.run_name_suffix = None
        self.wandb_run_name = None
        self.base_output_dir = None
        self.ckpt_dir = None
        self.initial_global_step = 0
        self.key = None
        self.key_actions_base = None
        self.key_update_base = None
        self.envs = None
        self.agent = None
        self.rb = None
        self.actor_state = None
        self.qf1_state = None
        self.qf2_state = None
        self.log_alpha_state = None
        self.current_alpha = None
        self.p_update_all = None
        self.p_update_target_networks = None
        self.data_loader = None
        self.evaluator = None
        self.logger = None

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

    def _setup_paths_and_run_name(self):
        run_name_suffix = f"{self.args.train.exp_name}__{self.args.env.seed}__{int(time.time())}"
        if self.args.train.save_dir:
            self.base_output_dir = Path(self.args.train.save_dir)
            self.wandb_run_name = f"{self.base_output_dir.name}__{run_name_suffix}" if self.base_output_dir.name else run_name_suffix
        else:
            self.base_output_dir = Path(f"runs/{run_name_suffix}")
            self.wandb_run_name = run_name_suffix
        
        self.run_name_suffix = run_name_suffix
        self.ckpt_dir = self.base_output_dir / "ckpts"
        print(f"Run name: {self.run_name_suffix}")
        print(f"Checkpoint directory: {self.ckpt_dir}")

    def _setup_jax_devices(self):
        print(f"Found {self.num_devices} JAX devices: {jax.devices()}")
        if self.args.algo.batch_size % self.num_devices != 0:
            raise ValueError(f"Global batch size {self.args.algo.batch_size} must be divisible by number of devices {self.num_devices}")
        self.batch_size_per_device = self.args.algo.batch_size // self.num_devices
        print(f"Using global batch size: {self.args.algo.batch_size}, per-device batch size: {self.batch_size_per_device}")

    def _handle_resume_and_directory_setup(self):
        restored_ckpt_path = None
        if self.args.train.resume:
            # The ckpt_dir here should be the parent directory containing all checkpoint folders.
            latest_ckpt_path_str = checkpoints.latest_checkpoint(os.path.abspath(self.ckpt_dir), prefix="ckpt_step")
            if latest_ckpt_path_str:
                print(f"Found latest checkpoint: {latest_ckpt_path_str}")
                try:
                    # The step is parsed from the directory name itself.
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
        
        # This will be used to load states later in _setup_agent
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
        self.data_loader = DataLoader(self.args.env.trading_env_config)
        print("DataLoader initialized.")

    def _setup_environments(self):
        print("Creating training environments...")
        vec_env_cls = gym.vector.AsyncVectorEnv if self.args.env.async_vector_env else gym.vector.SyncVectorEnv
        self.envs = vec_env_cls(
            [train_env_maker(
                seed=self.args.env.seed + i,
                config=self.args.env.trading_env_config,
                data_loader=self.data_loader
            ) for i in range(self.args.env.env_num)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported by this SAC version"

    def _setup_agent(self):
        obs_shape = self.envs.single_observation_space.shape
        action_dim = self.envs.single_action_space.n

        actor_model_cls, critic_model_cls = TradingActor, TradingCritic
        
        key_agent, self.key = jax.random.split(self.key)
        self.agent = SACAgent(
            action_dim=action_dim,
            observation_space_shape=obs_shape,
            key=key_agent,
            network_config=self.args.network,
            algo_config=self.args.algo,
            actor_model_cls=actor_model_cls,
            critic_model_cls=critic_model_cls
        )
        
        self._initialize_or_restore_agent_states()

        actor_p_count = count_params(self.actor_state_single.params) / 1e6
        critic_p_count = count_params(self.qf1_state_single.params) / 1e6
        print(f"Actor params: {actor_p_count:.2f}M")
        print(f"Critic params: {critic_p_count:.2f}M (x2 networks)")
        if self.args.wandb.track:
            wandb.summary['actor_params_m'] = actor_p_count
            wandb.summary['critic_params_m'] = critic_p_count

        self._replicate_states()

        self.p_update_all = jax.pmap(self.agent.update_all, axis_name='batch')
        self.p_update_target_networks = jax.pmap(self.agent.update_target_networks)

    def _initialize_or_restore_agent_states(self):
        actor_state_single = self.agent.actor_state
        qf1_state_single = self.agent.qf1_state
        qf2_state_single = self.agent.qf2_state
        log_alpha_state_single = self.agent.log_alpha_state if self.args.algo.autotune else None
        
        if self.restored_ckpt_path:
            try:
                # Define the structure for restoration using placeholder values from the initial states.
                # This structure must match what was saved in _save_checkpoint.
                restore_target = {
                    'actor_params': actor_state_single.params,
                    'actor_opt_state': actor_state_single.opt_state,
                    'qf1_params': qf1_state_single.params,
                    'qf1_opt_state': qf1_state_single.opt_state,
                    'qf1_target_params': qf1_state_single.target_params,
                    'qf2_params': qf2_state_single.params,
                    'qf2_opt_state': qf2_state_single.opt_state,
                    'qf2_target_params': qf2_state_single.target_params,
                }
                if self.args.algo.autotune:
                    restore_target['log_alpha_params'] = log_alpha_state_single.params
                    restore_target['log_alpha_opt_state'] = log_alpha_state_single.opt_state

                # Restore the raw arrays and optimizer states.
                # latest_checkpoint provides the full path to the specific checkpoint directory.
                loaded_contents = checkpoints.restore_checkpoint(
                    ckpt_dir=self.restored_ckpt_path,
                    target=restore_target
                )
                
                # Manually update the TrainState objects with the loaded contents.
                actor_state_single = actor_state_single.replace(
                    params=loaded_contents['actor_params'],
                    opt_state=loaded_contents['actor_opt_state']
                )
                qf1_state_single = qf1_state_single.replace(
                    params=loaded_contents['qf1_params'],
                    opt_state=loaded_contents['qf1_opt_state'],
                    target_params=loaded_contents['qf1_target_params']
                )
                qf2_state_single = qf2_state_single.replace(
                    params=loaded_contents['qf2_params'],
                    opt_state=loaded_contents['qf2_opt_state'],
                    target_params=loaded_contents['qf2_target_params']
                )
                if self.args.algo.autotune and 'log_alpha_params' in loaded_contents:
                    log_alpha_state_single = log_alpha_state_single.replace(
                        params=loaded_contents['log_alpha_params'],
                        opt_state=loaded_contents['log_alpha_opt_state']
                    )
                print(f"Agent states, including optimizer states, successfully restored from step {self.initial_global_step}.")
            except Exception as e:
                print(f"Error restoring agent states: {e}. Starting with fresh states.")
                self.initial_global_step = 0
        
        self.actor_state_single = actor_state_single
        self.qf1_state_single = qf1_state_single
        self.qf2_state_single = qf2_state_single
        self.log_alpha_state_single = log_alpha_state_single
        self.current_alpha_single = jnp.exp(log_alpha_state_single.params['log_alpha']) if self.args.algo.autotune and log_alpha_state_single else jnp.array(self.args.algo.alpha)

        self.restored_ckpt_path = self.restored_ckpt_path

    def _replicate_states(self):
        self.actor_state = flax.jax_utils.replicate(self.actor_state_single)
        self.qf1_state = flax.jax_utils.replicate(self.qf1_state_single)
        self.qf2_state = flax.jax_utils.replicate(self.qf2_state_single)
        if self.args.algo.autotune:
            self.log_alpha_state = flax.jax_utils.replicate(self.log_alpha_state_single)
        else:
            self.log_alpha_state = None
        self.current_alpha = flax.jax_utils.replicate(self.current_alpha_single)

    def _setup_replay_buffer(self):
        loaded_rb = False
        if self.restored_ckpt_path:
            rb_path = os.path.join(self.restored_ckpt_path, "replay_buffer.joblib.gz")
            if os.path.exists(rb_path):
                try:
                    print(f"Loading replay buffer from {rb_path}...")
                    self.rb:ReplayBuffer = joblib.load(rb_path)
                    print(f"Replay buffer loaded. Current size: {self.rb.size()}, full: {self.rb.full}")
                    assert self.rb.buffer_size == max(self.args.algo.buffer_size//self.args.env.env_num, 1) # 和SB3的ReplayBuffer的内部逻辑保持一致
                    assert self.rb.n_envs == self.args.env.env_num
                    loaded_rb = True
                except Exception as e:
                    print(f"Could not load replay buffer due to {e}. A new one will be created.")

        if not loaded_rb:
            print("Creating new replay buffer.")
            self.rb = ReplayBuffer(
                self.args.algo.buffer_size,
                self.envs.single_observation_space,
                self.envs.single_action_space,
                device="cpu",
                handle_timeout_termination=False,
                n_envs=self.args.env.env_num
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
        start_time = time.time()
        obs, _ = self.envs.reset(seed=self.args.env.seed + self.initial_global_step)
        
        
        train_stats_aggregator = StatsAggregator()
        pbar_postfix = collections.OrderedDict()

        total_iterations = self.args.algo.total_timesteps // self.args.env.env_num
        start_iteration = self.initial_global_step // self.args.env.env_num
        
        with tqdm(initial=start_iteration, total=total_iterations, desc="Training") as pbar:
            for loop_iter in range(start_iteration, total_iterations):
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
                        metrics_from_update = self._agent_update(current_step)
                        if metrics_from_update:
                            sps = int((current_step - self.initial_global_step) / (time.time() - start_time + 1e-9))
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

                    if current_step % self.args.algo.target_network_frequency == 0:
                        self._update_target_networks()
                
                next_step = (loop_iter + 1) * self.args.env.env_num
                self._run_evaluation(current_step, next_step)
                self._save_checkpoint(current_step, next_step)

                pbar.set_postfix(pbar_postfix)
                pbar.update(1)
        
        self._save_final_model()
        self.cleanup()

    def _environment_step(self, obs, current_step):
        self.key_actions_base, key_actions_step = jax.random.split(self.key_actions_base)
        if current_step < self.args.algo.learning_starts:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            jax_obs = jnp.asarray(obs)
            unreplicated_actor_params = flax.jax_utils.unreplicate(self.actor_state.params)
            actions_jax = self.agent.select_action(unreplicated_actor_params, jax_obs, key_actions_step, deterministic=False)
            actions = np.array(jax.device_get(actions_jax))

        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos and infos["final_observation"][idx] is not None:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        self.rb.add(obs, real_next_obs, actions, rewards.astype(np.float32), terminations.astype(np.float32), infos)
        return next_obs, infos

    def _agent_update(self, current_step):
        self.key_update_base, key_update_step = jax.random.split(self.key_update_base)
        data = self.rb.sample(self.args.algo.batch_size)
        data_numpy = {
            'observations': data.observations.numpy(), 'actions': data.actions.numpy().astype(np.int32),
            'next_observations': data.next_observations.numpy(), 'rewards': data.rewards.numpy().flatten(),
            'dones': data.dones.numpy().flatten()
        }
        sharded_data = {k: v.reshape(self.num_devices, self.batch_size_per_device, *v.shape[1:]) for k, v in data_numpy.items()}
        sharded_key = jax.random.split(key_update_step, self.num_devices)
        
        log_alpha_arg = self.log_alpha_state if self.args.algo.autotune else self.current_alpha

        self.actor_state, self.qf1_state, self.qf2_state, returned_log_alpha, self.current_alpha, metrics_sharded = self.p_update_all(
            self.actor_state, self.qf1_state, self.qf2_state, log_alpha_arg, sharded_data, sharded_key
        )
        if self.args.algo.autotune:
            self.log_alpha_state = returned_log_alpha
        
        if current_step % (self.args.algo.update_frequency * 100) == 0:
            metrics = flax.jax_utils.unreplicate(metrics_sharded)
            return {f"{k}": v for k, v in metrics.items()}
        return None

    def _update_target_networks(self):
        self.qf1_state, self.qf2_state = self.p_update_target_networks(self.qf1_state, self.qf2_state)

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
            actor_params_eval=flax.jax_utils.unreplicate(self.actor_state.params),
            current_train_step=current_step
        )
        
        tqdm.write(f"Evaluation at step {current_step}: {eval_metrics}")

    def _save_checkpoint(self, current_step, next_step, is_final=False):
        step_for_ckpt: int
        if is_final:
            step_for_ckpt = current_step
            print(f"--- Saving final model at step {step_for_ckpt} ---")
        else:
            ckpt_freq = self.args.train.ckpt_save_frequency_abs_steps
            if not ckpt_freq or next_step < ckpt_freq or (current_step // ckpt_freq) >= (next_step // ckpt_freq):
                return
            step_for_ckpt = current_step
            print(f"--- Saving checkpoint at step {step_for_ckpt} ---")

        try:
            save_target = {
                'actor_params': flax.jax_utils.unreplicate(self.actor_state.params),
                'actor_opt_state': flax.jax_utils.unreplicate(self.actor_state.opt_state),
                'qf1_params': flax.jax_utils.unreplicate(self.qf1_state.params),
                'qf1_opt_state': flax.jax_utils.unreplicate(self.qf1_state.opt_state),
                'qf1_target_params': flax.jax_utils.unreplicate(self.qf1_state.target_params),
                'qf2_params': flax.jax_utils.unreplicate(self.qf2_state.params),
                'qf2_opt_state': flax.jax_utils.unreplicate(self.qf2_state.opt_state),
                'qf2_target_params': flax.jax_utils.unreplicate(self.qf2_state.target_params),
            }
            if self.args.algo.autotune and self.log_alpha_state:
                save_target['log_alpha_params'] = flax.jax_utils.unreplicate(self.log_alpha_state.params)
                save_target['log_alpha_opt_state'] = flax.jax_utils.unreplicate(self.log_alpha_state.opt_state)

            checkpoints.save_checkpoint(
                ckpt_dir=os.path.abspath(self.ckpt_dir), target=save_target, step=step_for_ckpt,
                prefix="ckpt_step_", keep=50, overwrite=True
            )
            saved_path = checkpoints.latest_checkpoint(os.path.abspath(self.ckpt_dir), prefix="ckpt_step")
            
            if saved_path:
                print(f"Checkpoint saved to {saved_path}")

                # Log the model artifact to wandb before adding other files to the checkpoint directory.
                # This ensures that only the model is part of the artifact.
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
            
    def _save_final_model(self):
        final_step = (self.args.algo.total_timesteps // self.args.env.env_num) * self.args.env.env_num
        self._save_checkpoint(final_step, final_step + 1, is_final=True)

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
