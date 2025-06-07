import os
import random
import time
from dataclasses import asdict
from pathlib import Path
import shutil

import collections
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from flax.training import checkpoints # checkpoints只接受绝对路径
import flax.jax_utils

from .config import Args
from .common import make_env
from .networks import ActorCNN, CriticCNN, ActorMLP, CriticMLP
from .agent import SACAgent
from .eval import evaluate_agent
import wandb
class Trainer:
    def __init__(self, args: Args):
        self.args = args
        self.num_devices = jax.local_device_count()

        # To be initialized in setup
        self.run_name_suffix = None
        self.wandb_run_name = None
        self.base_output_dir = None
        self.ckpt_dir = None
        self.initial_global_step = 0
        self.key = None
        self.envs = None
        self.eval_envs = None
        self.agent = None
        self.rb = None
        self.actor_state = None
        self.qf1_state = None
        self.qf2_state = None
        self.log_alpha_state = None
        self.current_alpha = None
        self.p_update_all = None
        self.p_update_target_networks = None

    def setup(self):
        self._setup_paths_and_run_name()
        self._setup_jax_devices()
        self._handle_resume_and_directory_setup()
        self._setup_wandb()
        self._setup_seeds_and_keys()
        self._setup_environments()
        self._setup_agent()
        self._setup_replay_buffer()

    def _setup_paths_and_run_name(self):
        if self.args.train.save_dir:
            self.base_output_dir = Path(self.args.train.save_dir)
            run_name_suffix = f"{self.args.env.env_id}__{self.args.train.exp_name}__{self.args.env.seed}__{int(time.time())}"
            self.wandb_run_name = f"{self.base_output_dir.name}__{run_name_suffix}" if self.base_output_dir.name else run_name_suffix
        else:
            run_name_suffix = f"{self.args.env.env_id}__{self.args.train.exp_name}__{self.args.env.seed}__{int(time.time())}"
            self.base_output_dir = Path(f"runs/{run_name_suffix}")
            self.wandb_run_name = run_name_suffix
        
        self.run_name_suffix = run_name_suffix
        self.ckpt_dir = self.base_output_dir / "ckpts"
        print(f"Run name suffix: {self.run_name_suffix}")
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
            latest_ckpt_path_str = checkpoints.latest_checkpoint(os.path.abspath(self.ckpt_dir))
            if latest_ckpt_path_str:
                print(f"Found latest checkpoint: {latest_ckpt_path_str}")
                try:
                    self.initial_global_step = int(Path(latest_ckpt_path_str).stem.split("_")[-1])
                    print(f"Resuming from global_step {self.initial_global_step}")
                    restored_ckpt_path = latest_ckpt_path_str
                except (ValueError, IndexError):
                    print(f"Could not parse step from checkpoint filename: {latest_ckpt_path_str}. Starting fresh.")
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
        flat_args_dict = {}
        for main_key, main_value in asdict(self.args).items():
            if isinstance(main_value, dict):
                for sub_key, sub_value in main_value.items():
                    flat_args_dict[f"{main_key}.{sub_key}"] = sub_value
            else:
                flat_args_dict[main_key] = main_value
        wandb.config.update(flat_args_dict)

    def _setup_seeds_and_keys(self):
        random.seed(self.args.env.seed)
        np.random.seed(self.args.env.seed)
        base_key_seed = self.args.env.seed if not self.restored_ckpt_path else self.args.env.seed + self.initial_global_step
        self.key = jax.random.PRNGKey(base_key_seed)

    def _setup_environments(self):
        vec_env_cls = gym.vector.AsyncVectorEnv if self.args.env.async_vector_env else gym.vector.SyncVectorEnv
        self.envs = vec_env_cls(
            [make_env(self.args.env.env_id, self.args.env.seed + i, i, False, self.run_name_suffix, self.args.env.env_num) for i in range(self.args.env.env_num)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported by this SAC version"

        if self.args.eval.cache_env and not self.args.eval.capture_video and self.args.eval.eval_episodes > 0:
            print("Caching evaluation environment.")
            eval_vec_env_cls = gym.vector.AsyncVectorEnv if self.args.eval.async_vector_env else gym.vector.SyncVectorEnv
            self.eval_envs = eval_vec_env_cls([
                make_env(
                    self.args.env.env_id,
                    self.args.eval.seed + i,
                    i,
                    capture_video=False,
                    run_name=f"{self.run_name_suffix}_eval_cached",
                    num_envs=self.args.eval.env_num
                ) for i in range(self.args.eval.env_num)
            ])

    def _setup_agent(self):
        obs_shape = self.envs.single_observation_space.shape
        action_dim = self.envs.single_action_space.n

        if "NoFrameskip" in self.args.env.env_id:
            actor_model_cls, critic_model_cls = ActorCNN, CriticCNN
        elif "CartPole-v1" == self.args.env.env_id:
            actor_model_cls, critic_model_cls = ActorMLP, CriticMLP
        else:
            raise ValueError(f"Unsupported environment ID for network selection: {self.args.env.env_id}")
        
        key_agent, self.key = jax.random.split(self.key)
        self.agent = SACAgent(
            action_dim=action_dim,
            observation_space_shape=obs_shape,
            key=key_agent,
            algo_config=self.args.algo,
            actor_model_cls=actor_model_cls,
            critic_model_cls=critic_model_cls
        )
        
        self._initialize_or_restore_agent_states()
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
                target_restore_dict = {'actor_state': actor_state_single, 'qf1_state': qf1_state_single, 'qf2_state': qf2_state_single}
                if self.args.algo.autotune:
                    target_restore_dict['log_alpha_state'] = log_alpha_state_single
                
                loaded_states = checkpoints.restore_checkpoint(
                    ckpt_dir=os.path.abspath(self.ckpt_dir), target=target_restore_dict, step=self.initial_global_step, prefix="ckpt_step_"
                )
                actor_state_single = loaded_states['actor_state']
                qf1_state_single = loaded_states['qf1_state']
                qf2_state_single = loaded_states['qf2_state']
                if self.args.algo.autotune and 'log_alpha_state' in loaded_states:
                    log_alpha_state_single = loaded_states['log_alpha_state']
                print(f"Agent states successfully restored from step {self.initial_global_step}.")
            except Exception as e:
                print(f"Error restoring agent states: {e}. Starting with fresh states.")
                self.initial_global_step = 0
        
        self.actor_state_single = actor_state_single
        self.qf1_state_single = qf1_state_single
        self.qf2_state_single = qf2_state_single
        self.log_alpha_state_single = log_alpha_state_single
        self.current_alpha_single = jnp.exp(log_alpha_state_single.params['log_alpha']) if self.args.algo.autotune else jnp.array(self.args.algo.alpha)

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
        self.rb = ReplayBuffer(
            self.args.algo.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            device="cpu",
            handle_timeout_termination=False,
            n_envs=self.args.env.env_num
        )

    def train(self):
        start_time = time.time()
        obs, _ = self.envs.reset(seed=self.args.env.seed + self.initial_global_step)
        
        # Buffers for smoothed episodic statistics
        ep_stats_buffer_size = self.args.env.env_num
        returns_buffer = collections.deque(maxlen=ep_stats_buffer_size)
        lengths_buffer = collections.deque(maxlen=ep_stats_buffer_size)

        key_actions_base, key_update_base = jax.random.split(self.key)
        start_iteration = self.initial_global_step // self.args.env.env_num

        for loop_iter in range(start_iteration, self.args.algo.total_timesteps // self.args.env.env_num):
            current_step = loop_iter * self.args.env.env_num
            key_actions_base, key_actions_step = jax.random.split(key_actions_base)
            
            obs, infos = self._environment_step(obs, key_actions_step, current_step)
            
            self._log_episode_stats(infos, current_step, returns_buffer, lengths_buffer)

            if current_step > self.args.algo.learning_starts:
                if current_step % self.args.algo.update_frequency == 0:
                    key_update_base, key_update_step = jax.random.split(key_update_base)
                    self._agent_update(key_update_step, current_step, start_time)

                if current_step % self.args.algo.target_network_frequency == 0:
                    self._update_target_networks()
            
            next_step = (loop_iter + 1) * self.args.env.env_num
            self._run_evaluation(current_step, next_step)
            self._save_checkpoint(current_step, next_step)
        
        self._save_final_model()
        self.cleanup()

    def _environment_step(self, obs, key_actions, current_step):
        if current_step < self.args.algo.learning_starts:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            jax_obs = jnp.asarray(obs)
            unreplicated_actor_params = flax.jax_utils.unreplicate(self.actor_state.params)
            actions_jax = self.agent.select_action(unreplicated_actor_params, jax_obs, key_actions, deterministic=False)
            actions = np.array(jax.device_get(actions_jax))

        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos and infos["final_observation"][idx] is not None:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        self.rb.add(obs, real_next_obs, actions, rewards.astype(np.float32), terminations.astype(np.float32), infos)
        return next_obs, infos

    def _log_episode_stats(self, infos, current_step, returns_buffer, lengths_buffer):
        if "final_info" not in infos or not self.args.wandb.track:
            return
            
        for i, info_item in enumerate(infos["final_info"]):
            if info_item and "episode" in info_item:
                returns_buffer.append(info_item['episode']['r'][0])
                lengths_buffer.append(info_item['episode']['l'][0])
                if i == 0:
                    wandb.log({"charts/episodic_return_env0": info_item['episode']['r'][0], 
                               "charts/episodic_length_env0": info_item['episode']['l'][0]}, step=current_step)

        if returns_buffer:
            returns_list = list(returns_buffer)
            mean_ret, min_ret, max_ret = np.mean(returns_list), np.min(returns_list), np.max(returns_list)
            std_ret = np.std(returns_list) if len(returns_list) > 1 else 0.0
            wandb.log({
                "charts/episodic_return_mean_buffered": mean_ret,
                "charts/episodic_return_std_buffered": std_ret
            }, step=current_step)
            print(f"step={current_step}, buffered_return_mean={mean_ret:.2f}, buffered_return_std={std_ret:.2f} (from {len(returns_list)} ep.)")
        if lengths_buffer:
            wandb.log({"charts/episodic_length_mean_buffered": np.mean(list(lengths_buffer))}, step=current_step)

    def _agent_update(self, key_update, current_step, start_time):
        data = self.rb.sample(self.args.algo.batch_size)
        data_numpy = {
            'observations': data.observations.numpy(), 'actions': data.actions.numpy().astype(np.int32),
            'next_observations': data.next_observations.numpy(), 'rewards': data.rewards.numpy().flatten(),
            'dones': data.dones.numpy().flatten()
        }
        sharded_data = {k: v.reshape(self.num_devices, self.batch_size_per_device, *v.shape[1:]) for k, v in data_numpy.items()}
        sharded_key = jax.random.split(key_update, self.num_devices)
        
        log_alpha_arg = self.log_alpha_state if self.args.algo.autotune else self.current_alpha

        self.actor_state, self.qf1_state, self.qf2_state, returned_log_alpha, self.current_alpha, metrics = self.p_update_all(
            self.actor_state, self.qf1_state, self.qf2_state, log_alpha_arg, sharded_data, sharded_key
        )
        if self.args.algo.autotune:
            self.log_alpha_state = returned_log_alpha
            
        if current_step % (self.args.algo.update_frequency * 25) == 0 and self.args.wandb.track:
            self._log_training_metrics(metrics, current_step, start_time)

    def _log_training_metrics(self, metrics_sharded, current_step, start_time):
        metrics = flax.jax_utils.unreplicate(metrics_sharded)
        sps = int(current_step / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
        wandb_metrics = {f"losses/{k}": v for k, v in metrics.items()}
        wandb_metrics.update({
            "charts/SPS": sps,
            "losses/alpha_current_val": jax.device_get(self.current_alpha[0])
        })
        wandb.log(wandb_metrics, step=current_step)
        print(f"SPS: {sps}")
    
    def _update_target_networks(self):
        self.qf1_state, self.qf2_state = self.p_update_target_networks(self.qf1_state, self.qf2_state)

    def _run_evaluation(self, current_step, next_step):
        eval_freq = self.args.eval.eval_frequency_abs_steps
        if self.args.eval.eval_episodes <= 0 or not eval_freq or next_step < eval_freq:
            return
        
        if (current_step // eval_freq) < (next_step // eval_freq):
            eval_trigger_step = (next_step // eval_freq) * eval_freq
            print(f"--- Evaluation Triggered at step {current_step} (effective eval step: {eval_trigger_step}) ---")
            
            eval_metrics = evaluate_agent(
                agent_eval=self.agent,
                actor_params_eval=flax.jax_utils.unreplicate(self.actor_state.params),
                env_config=self.args.env,
                eval_config=self.args.eval,
                num_episodes=self.args.eval.eval_episodes,
                greedy_actions=self.args.eval.greedy_actions,
                run_name_suffix_eval=self.run_name_suffix,
                current_train_step=current_step,
                eval_envs=self.eval_envs
            )
            print(f"Evaluation at step {current_step}: {eval_metrics}")
            if self.args.wandb.track:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=current_step)

    def _save_checkpoint(self, current_step, next_step, is_final=False):
        if is_final:
            step_to_save = (self.args.algo.total_timesteps // self.args.env.env_num) * self.args.env.env_num
            print(f"--- Saving final model at step {step_to_save} ---")
        else:
            ckpt_freq = self.args.train.ckpt_save_frequency_abs_steps
            if not ckpt_freq or next_step < ckpt_freq or (current_step // ckpt_freq) >= (next_step // ckpt_freq):
                return
            step_to_save = (next_step // ckpt_freq) * ckpt_freq
            print(f"--- Checkpoint Triggered at step {current_step} (effective ckpt step: {step_to_save}) ---")

        try:
            save_target = {
                'actor_state': flax.jax_utils.unreplicate(self.actor_state),
                'qf1_state': flax.jax_utils.unreplicate(self.qf1_state),
                'qf2_state': flax.jax_utils.unreplicate(self.qf2_state),
            }
            if self.args.algo.autotune and self.log_alpha_state:
                save_target['log_alpha_state'] = flax.jax_utils.unreplicate(self.log_alpha_state)

            checkpoints.save_checkpoint(
                ckpt_dir=os.path.abspath(self.ckpt_dir), target=save_target, step=step_to_save,
                prefix="ckpt_step_", keep=50, overwrite=True
            )
            saved_path = checkpoints.latest_checkpoint(os.path.abspath(self.ckpt_dir))
            print(f"Checkpoint saved to {saved_path}")
            if self.args.wandb.track and wandb.run and saved_path:
                artifact = wandb.Artifact(f"model_ckpt_{self.wandb_run_name}", type="model")
                artifact.add_file(str(saved_path))
                aliases = [f"step_{step_to_save}"]
                if is_final:
                    aliases.append("final")
                wandb.log_artifact(artifact, aliases=aliases)
        except Exception as e:
            print(f"Error saving checkpoint at step {step_to_save}: {e}")
            
    def _save_final_model(self):
        self._save_checkpoint(self.args.algo.total_timesteps, self.args.algo.total_timesteps, is_final=True)

    def cleanup(self):
        if self.eval_envs:
            self.eval_envs.close()
        self.envs.close()
        if self.args.wandb.track and wandb.run:
            wandb.finish()


def train(args: Args):
    # JAX platform selection
    if args.train.jax_platform_name:
        try:
            print(f"Attempting to set JAX platform to: {args.train.jax_platform_name}")
            jax.config.update('jax_platform_name', args.train.jax_platform_name)
            print(f"JAX platform successfully set to: {jax.devices()[0].platform.upper() if jax.devices() else 'Unknown'}")
        except Exception as e:
            print(f"Could not set JAX platform name: {e}")
    
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
