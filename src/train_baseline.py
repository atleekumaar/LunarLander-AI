"""PPO Baseline Training Script for LunarLander-v3.

Uses Stable-Baselines3 PPO with an MLP policy (two 64-unit hidden layers)
to train an agent to land the LunarLander safely between the flags.
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

# Add project root to sys.path so scripts can be invoked directly from any working directory
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


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def check_tensorboard_available() -> bool:
    """Check if TensorBoard SummaryWriter is functional without import errors."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True
    except Exception:
        return False


def set_global_seeds(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_vectorized_env(
    env_id: str,
    log_dir: str,
    seed: int,
    monitor_filename: Optional[str] = None,
    n_envs: int = 1,
) -> DummyVecEnv:
    """Create a single or multi-env vectorized environment with SB3 Monitor wrapper.

    Args:
        env_id: Gymnasium environment name.
        log_dir: Directory where Monitor CSV logs should be stored.
        seed: Random seed for environment initialization.
        monitor_filename: Optional custom file prefix for the monitor log.
        n_envs: Number of parallel environments (default: 1).

    Returns:
        DummyVecEnv: Vectorized environment wrapped for SB3 logging.
    """
    os.makedirs(log_dir, exist_ok=True)
    base_name = monitor_filename or "monitor"

    def _make_env(rank: int):
        def _init() -> gym.Env:
            env = make_lunar_lander_env(env_id=env_id, render_mode=None, seed=seed + rank * 100)
            target_name = base_name if n_envs == 1 else f"{base_name}_{rank}"
            m_path = os.path.join(log_dir, target_name)
            return Monitor(env, filename=m_path)
        return _init

    return DummyVecEnv([_make_env(i) for i in range(n_envs)])


def train_baseline_agent(
    config_path: str = "configs/default.yaml",
    timesteps_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    resume_from: Optional[str] = None,
    model_dir_override: Optional[str] = None,
    log_dir_override: Optional[str] = None,
    model_filename_override: Optional[str] = None,
    n_envs_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Train or resume a PPO baseline agent on LunarLander-v3.

    Args:
        config_path: Path to configuration YAML.
        timesteps_override: Optional CLI override for training timesteps.
        seed_override: Optional CLI override for random seed.
        resume_from: Optional path to a saved model checkpoint to resume.
        model_dir_override: Optional override for saving model artifacts.
        log_dir_override: Optional override for saving training logs.
        model_filename_override: Optional override for model checkpoint filename.
        n_envs_override: Optional override for number of parallel environments.

    Returns:
        Dict[str, Any]: Training run metadata and artifact locations.
    """
    config = load_config(config_path)

    # Resolve parameters with CLI overrides
    seed = seed_override if seed_override is not None else config["training"].get("seed", 42)
    total_timesteps = (
        timesteps_override
        if timesteps_override is not None
        else config["training"].get("total_timesteps", 100000)
    )
    env_id = config["environment"].get("env_id", "LunarLander-v3")
    model_dir = model_dir_override or config["paths"].get("model_dir", "models")
    log_dir = log_dir_override or config["paths"].get("log_dir", "logs")
    model_filename = (
        model_filename_override
        or config["paths"].get("model_filename", "ppo_lunarlander_baseline.zip")
    )
    n_envs = (
        n_envs_override
        if n_envs_override is not None
        else config["training"].get("n_envs", 1)
    )

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save a configuration snapshot in log_dir for complete provenance
    snapshot_path = os.path.join(log_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    set_global_seeds(seed)

    print("=" * 70)
    print("           LUNAR LANDER v3 - PPO BASELINE TRAINING")
    print("=" * 70)
    print(f"Environment ID:       {env_id}")
    print(f"Training Timesteps:   {total_timesteps:,}")
    print(f"Random Seed:          {seed}")
    print(f"Model Save Directory: {model_dir}")
    print(f"Log Save Directory:   {log_dir}")

    # Build vectorized environment
    vec_env = create_vectorized_env(env_id=env_id, log_dir=log_dir, seed=seed, n_envs=n_envs)

    # Setup TensorBoard if installed
    tb_available = check_tensorboard_available()
    tb_log = log_dir if tb_available else None
    if tb_available:
        print(f"TensorBoard logging:  ENABLED (writing to {log_dir})")
    else:
        print("TensorBoard logging:  DISABLED (tensorboard package not found)")

    # Define policy network architecture (two 64-unit hidden layers for actor & critic)
    net_arch = config["training"].get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
    policy_kwargs = dict(
        net_arch=dict(
            pi=net_arch.get("pi", [64, 64]),
            vf=net_arch.get("vf", [64, 64]),
        )
    )

    ppo_params = {
        "learning_rate": float(config["training"].get("learning_rate", 3e-4)),
        "n_steps": int(config["training"].get("n_steps", 2048)),
        "batch_size": int(config["training"].get("batch_size", 64)),
        "n_epochs": int(config["training"].get("n_epochs", 10)),
        "gamma": float(config["training"].get("gamma", 0.99)),
        "gae_lambda": float(config["training"].get("gae_lambda", 0.95)),
        "clip_range": float(config["training"].get("clip_range", 0.2)),
        "ent_coef": float(config["training"].get("ent_coef", 0.01)),
        "vf_coef": float(config["training"].get("vf_coef", 0.5)),
        "max_grad_norm": float(config["training"].get("max_grad_norm", 0.5)),
    }

    print("\nPPO Hyperparameters:")
    for k, v in ppo_params.items():
        print(f"  {k:<16}: {v}")
    print(f"  Parallel Envs:    {n_envs}")
    print(f"  Policy Network:   MLP [Input(8) -> Dense(64) -> Dense(64) -> Actor(4)/Critic(1)]")

    # Explicit differentiation between fresh training and resuming
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Checkpoint for resume not found at: {resume_from}")
        print(f"\n[RESUME MODE] Loading existing model checkpoint from: {resume_from}")
        model = PPO.load(
            resume_from,
            env=vec_env,
            tensorboard_log=tb_log,
        )
        starting_timesteps = model.num_timesteps
        optimizer_restored = hasattr(model.policy, "optimizer") and model.policy.optimizer is not None
        opt_keys_count = len(model.policy.optimizer.state_dict().get("state", {})) if optimizer_restored else 0
        print(f"  [CHECK] Restored starting environment timesteps: {starting_timesteps:,}")
        print(f"  [CHECK] Optimizer state restored: {optimizer_restored} ({opt_keys_count} parameter state entries)")
        reset_num_timesteps = False
    else:
        print("\n[FRESH START] Initializing brand new PPO agent with random weights.")
        model = PPO(
            policy=config["training"].get("policy", "MlpPolicy"),
            env=vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log=tb_log,
            **ppo_params,
        )
        reset_num_timesteps = True

    print("\nStarting PPO training loop...")
    start_time = time.perf_counter()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    finally:
        vec_env.close()

    elapsed_time = time.perf_counter() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    # Save trained model
    save_path = os.path.join(model_dir, model_filename)
    model.save(save_path)

    # Save run metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "cum_env_timesteps": model.num_timesteps,
        "elapsed_seconds": round(elapsed_time, 2),
        "saved_model_path": save_path,
        "hyperparameters": ppo_params,
        "net_arch": net_arch,
        "n_envs": n_envs,
        "resumed_from": resume_from,
    }
    stem = Path(model_filename).stem
    meta_path = os.path.join(model_dir, f"{stem}_meta.json")
    last_meta_path = os.path.join(model_dir, "last_run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(last_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE")
    print("=" * 70)
    print(f"Training Duration:     {minutes}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    print(f"Run Timesteps Added:   {total_timesteps:,}")
    print(f"Total Cumulative Steps:{model.num_timesteps:,}")
    print(f"Saved Model File:      {save_path}")
    print(f"Saved Run Metadata:    {meta_path}")
    print(f"Log Directory:         {log_dir}")
    if tb_available:
        print(f"TensorBoard Event Dir: {log_dir}")
        print("  View logs with: tensorboard --logdir logs")
    print("=" * 70)

    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PPO baseline training."""
    parser = argparse.ArgumentParser(
        description="Train Stable-Baselines3 PPO baseline agent on LunarLander-v3."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config, default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for training reproducibility (overrides config, default: 42).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Explicit path to a saved .zip model checkpoint to resume training from.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: models).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save Monitor and TensorBoard logs (default: logs).",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default=None,
        help="Filename for the saved model checkpoint (e.g. ppo_lunarlander_320k_resumed.zip).",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=None,
        help="Number of parallel environments for rollout collection (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline_agent(
        config_path=args.config,
        timesteps_override=args.timesteps,
        seed_override=args.seed,
        resume_from=args.resume_from,
        model_dir_override=args.model_dir,
        log_dir_override=args.log_dir,
        model_filename_override=args.model_filename,
        n_envs_override=args.n_envs,
    )
