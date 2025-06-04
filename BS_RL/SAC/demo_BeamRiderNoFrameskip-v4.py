import tyro
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig
from .train import train
import os

if __name__ == "__main__":
    args = Args(
        train=TrainConfig(
            exp_name="sac_discrete_jax_beamrider",
            save_model=True,
            jax_platform_name="tpu" # Or "cpu" or None
        ),
        env=EnvConfig(
            env_id="BeamRiderNoFrameskip-v4",
            capture_video=False, # Set to True to record videos
            seed=1,
            num_envs=96 # As in original sac_atari.py
        ),
        algo=AlgoConfig(
            total_timesteps=int(1e6*200), #训练200M次，原本是5_000_000
            buffer_size=1_000_000,
            learning_starts=20_000, # Default from sac_atari.py
            batch_size=64,         # Default from sac_atari.py
            update_frequency=4,    # Default from sac_atari.py
            target_network_frequency=8000, # Default from sac_atari.py
            gamma=0.99,
            tau=1.0, # Hard update for discrete SAC target Qs
            policy_lr=3e-4,
            q_lr=3e-4,
            autotune=True,
            target_entropy_scale=0.89, # Matches sac_atari.py
            adam_eps=1e-4 # Matches PyTorch Adam default used in sac_atari.py
        ),
        wandb=WandbConfig(
            track=False, # Set to True to use WandB
            project_name="cleanrl-jax-sac-discrete",
            entity=None # Your WandB entity
        )
    )
    # Set XLA memory fraction if desired (example, adjust as needed)
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
    
    train(args)