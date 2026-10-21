"""Training script for controlled reward shaping experiments on LunarLander-v3.

Supports three experimental conditions:
  - 'baseline': Original Gymnasium reward
  - 'naive': Persistent +1 per frame when hovering upright and slow
  - 'corrected': Potential-Based Reward Shaping (PBRS) F(s, s') = gamma * Phi(s') - Phi(s)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import make_lunar_lander_env
from src.reward_wrappers import ShapedRewardWrapper
from src.train_baseline import check_tensorboard_available, load_config, set_global_seeds


def create_shaped_vectorized_env(
    env_id: str,
    log_dir: str,
    seed: int,
    condition: str = "baseline",
    beta: float = 1.0,
    gamma: float = 0.99,
    n_envs: int = 1,
) -> DummyVecEnv:
    """Create vectorized environment wrapped with ShapedRewardWrapper and Monitor.

    Args:
        env_id: Gymnasium environment ID.
        log_dir: Directory where Monitor CSV logs should be stored.
        seed: Random seed for environment initialization.
        condition: 'baseline', 'naive', or 'corrected'.
        beta: Potential-based shaping coefficient.
        gamma: Discount factor for potential difference.
        n_envs: Number of parallel environments.

    Returns:
        DummyVecEnv: Vectorized environment.
    """
    os.makedirs(log_dir, exist_ok=True)

    def _make_env(rank: int):
        def _init() -> gym.Env:
            base_env = make_lunar_lander_env(
                env_id=env_id,
                render_mode=None,
                seed=seed + rank * 100,
            )
            # Wrap with reward shaping
            shaped_env = ShapedRewardWrapper(
                base_env,
                condition=condition,
                beta=beta,
                gamma=gamma,
            )
            # Wrap with Monitor for episode return & length logging
            target_name = "monitor" if n_envs == 1 else f"monitor_{rank}"
            m_path = os.path.join(log_dir, target_name)
            return Monitor(shaped_env, filename=m_path)
        return _init

    return DummyVecEnv([_make_env(i) for i in range(n_envs)])


def run_reward_experiment(
    condition: str,
    timesteps: int = 100000,
    seed: int = 42,
    beta: float = 1.0,
    gamma: float = 0.99,
    output_dir: str = "models/experiments",
    log_dir: str = "logs/experiments",
    resume_from: Optional[str] = None,
    n_envs: int = 4,
    config_path: str = "configs/default.yaml",
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute a controlled reward shaping training run.

    Args:
        condition: 'baseline', 'naive', or 'corrected'.
        timesteps: Environment timesteps to train.
        seed: Random seed.
        beta: Scaling coefficient for potential-based shaping.
        gamma: Discount factor.
        output_dir: Destination folder for checkpoints.
        log_dir: Destination folder for logs.
        resume_from: Optional checkpoint path to resume from.
        n_envs: Number of parallel environments.
        config_path: Master configuration file path.
        force_overwrite: Whether to overwrite existing checkpoints.

    Returns:
        Dict[str, Any]: Experiment run metadata.
    """
    config = load_config(config_path)
    env_id = config["environment"].get("env_id", "LunarLander-v3")

    # Generate isolated directories and filenames
    experiment_tag = f"{condition}_seed{seed}"
    exp_log_dir = os.path.join(log_dir, experiment_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)

    model_filename = f"ppo_shaping_{experiment_tag}.zip"
    save_path = os.path.join(output_dir, model_filename)

    if os.path.exists(save_path) and not force_overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at '{save_path}'. "
            "Use a different seed/condition or enable force_overwrite."
        )

    set_global_seeds(seed)

    print("=" * 75)
    print("        REWARD SHAPING CONTROLLED EXPERIMENT")
    print("=" * 75)
    print(f"Condition:            {condition.upper()}")
    print(f"Environment ID:       {env_id}")
    print(f"Training Timesteps:   {timesteps:,}")
    print(f"Random Seed:          {seed}")
    print(f"Shaping Beta:         {beta if condition == 'corrected' else 'N/A'}")
    print(f"Discount Gamma:       {gamma}")
    print(f"Model Save Path:      {save_path}")
    print(f"Log Save Path:        {exp_log_dir}")
    print(f"Parallel Envs:        {n_envs}")

    # Build shaped vectorized environment
    vec_env = create_shaped_vectorized_env(
        env_id=env_id,
        log_dir=exp_log_dir,
        seed=seed,
        condition=condition,
        beta=beta,
        gamma=gamma,
        n_envs=n_envs,
    )

    tb_available = check_tensorboard_available()
    tb_log = exp_log_dir if tb_available else None

    # Save exact experiment configuration snapshot
    exp_config = {
        "condition": condition,
        "timesteps": timesteps,
        "seed": seed,
        "beta": beta,
        "gamma": gamma,
        "resumed_from": resume_from,
        "n_envs": n_envs,
        "master_config": config,
    }
    with open(os.path.join(exp_log_dir, "experiment_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(exp_config, f, default_flow_style=False)

    ppo_params = {
        "learning_rate": float(config["training"].get("learning_rate", 3e-4)),
        "n_steps": int(config["training"].get("n_steps", 2048)),
        "batch_size": int(config["training"].get("batch_size", 64)),
        "n_epochs": int(config["training"].get("n_epochs", 10)),
        "gamma": gamma,
        "gae_lambda": float(config["training"].get("gae_lambda", 0.95)),
        "clip_range": float(config["training"].get("clip_range", 0.2)),
        "ent_coef": float(config["training"].get("ent_coef", 0.01)),
        "vf_coef": float(config["training"].get("vf_coef", 0.5)),
        "max_grad_norm": float(config["training"].get("max_grad_norm", 0.5)),
    }

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Checkpoint for resume not found at: {resume_from}")
        print(f"\n[RESUME MODE] Initializing from base checkpoint: {resume_from}")
        model = PPO.load(
            resume_from,
            env=vec_env,
            tensorboard_log=tb_log,
        )
        reset_num_timesteps = False
    else:
        print("\n[FRESH START] Initializing brand new PPO agent.")
        net_arch = config["training"].get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
        policy_kwargs = dict(
            net_arch=dict(
                pi=net_arch.get("pi", [64, 64]),
                vf=net_arch.get("vf", [64, 64]),
            )
        )
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log=tb_log,
            **ppo_params,
        )
        reset_num_timesteps = True

    print("\nStarting reward experiment training loop...")
    start_time = time.perf_counter()

    try:
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    finally:
        vec_env.close()

    elapsed = time.perf_counter() - start_time
    model.save(save_path)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "condition": condition,
        "seed": seed,
        "timesteps": timesteps,
        "cum_env_timesteps": model.num_timesteps,
        "elapsed_seconds": round(elapsed, 2),
        "beta": beta if condition == "corrected" else None,
        "gamma": gamma,
        "resumed_from": resume_from,
        "saved_model_path": save_path,
        "log_dir": exp_log_dir,
    }
    meta_path = os.path.join(output_dir, f"ppo_shaping_{experiment_tag}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 75)
    print("               EXPERIMENT COMPLETE")
    print("=" * 75)
    print(f"Condition:             {condition.upper()}")
    print(f"Training Duration:     {int(elapsed // 60)}m {elapsed % 60:.2f}s ({elapsed:.2f}s)")
    print(f"Saved Checkpoint:      {save_path}")
    print(f"Saved Metadata:        {meta_path}")
    print(f"Logs:                  {exp_log_dir}")
    print("=" * 75)

    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reward shaping experiment."""
    parser = argparse.ArgumentParser(
        description="Run controlled reward shaping experiments on LunarLander-v3."
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["baseline", "naive", "corrected"],
        help="Experimental condition: baseline, naive, or corrected.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Training timesteps budget (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Potential-based shaping scaling factor beta (default: 1.0).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/experiments",
        help="Directory to store experimental checkpoints (default: models/experiments).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/experiments",
        help="Directory to store experimental logs (default: logs/experiments).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional base checkpoint to resume from for all conditions.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_reward_experiment(
        condition=args.condition,
        timesteps=args.timesteps,
        seed=args.seed,
        beta=args.beta,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        n_envs=args.n_envs,
        force_overwrite=args.force,
    )
