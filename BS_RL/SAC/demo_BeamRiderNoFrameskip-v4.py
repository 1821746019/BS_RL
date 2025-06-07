import tyro
from .config import Args, EnvConfig, AlgoConfig, WandbConfig, TrainConfig
from .train import train
import os

if __name__ == "__main__":
    batch_size = 5120
    args = Args(
        train=TrainConfig(
            exp_name="beamrider",
            save_model=True,
            save_dir="runs/beamrider"
        ),
        env=EnvConfig(
            env_id="BeamRiderNoFrameskip-v4",
            seed=1,
            env_num=96 # As in original sac_atari.py
        ),
        algo=AlgoConfig(
            total_timesteps=int(1e6*200), #训练200M次，原本是5_000_000
            buffer_size=1_000_000,
            learning_starts=20_000, # Default from sac_atari.py
            batch_size=batch_size,         # Default from sac_atari.py
            update_frequency=4,    # Default from sac_atari.py
            target_network_frequency=8000, # Default from sac_atari.py
            gamma=0.99,
            tau=1.0, # Hard update for discrete SAC target Qs
            policy_lr=3e-4*(batch_size/64)**0.5,
            q_lr=3e-4*(batch_size/64)**0.5,
            autotune=True,
            target_entropy_scale=0.89, # Matches sac_atari.py
            adam_eps=1e-4 # Matches PyTorch Adam default used in sac_atari.py
        ),
        wandb=WandbConfig(
            track=True, # Set to True to use WandB
            project_name="cleanrl-jax-sac-discrete",
            entity=None # Your WandB entity
        )
    )
    
    train(args)