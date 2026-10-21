"""Tests for configuration parsing, schema validation, and hyperparameter ranges."""

from __future__ import annotations

import os
import pytest
from src.train_baseline import load_config


def test_config_file_exists_and_loads():
    """Verify default.yaml exists and parses into a valid dictionary."""
    config_path = "configs/default.yaml"
    assert os.path.exists(config_path), f"Configuration file not found at {config_path}."

    config = load_config(config_path)
    assert isinstance(config, dict), "Configuration root must be a dictionary."

    required_sections = ["environment", "training", "evaluation", "paths", "video"]
    for section in required_sections:
        assert section in config, f"Missing required configuration section: '{section}'."


def test_environment_config_values():
    """Verify environment configuration settings."""
    config = load_config("configs/default.yaml")
    env_cfg = config["environment"]

    assert env_cfg.get("env_id") == "LunarLander-v3", "Expected env_id to be 'LunarLander-v3'."
    assert isinstance(env_cfg.get("seed"), int), "Environment seed must be an integer."


def test_training_hyperparameters_ranges():
    """Verify that training hyperparameters lie within theoretically valid RL bounds."""
    config = load_config("configs/default.yaml")
    train_cfg = config["training"]

    # Timestep & batch constraints
    assert isinstance(train_cfg["total_timesteps"], int) and train_cfg["total_timesteps"] > 0
    assert isinstance(train_cfg["n_steps"], int) and train_cfg["n_steps"] > 0
    assert isinstance(train_cfg["batch_size"], int) and train_cfg["batch_size"] > 0
    assert train_cfg["batch_size"] <= train_cfg["n_steps"], "Batch size cannot exceed rollout n_steps."

    # Mathematical RL parameter ranges
    assert 0.0 < float(train_cfg["learning_rate"]) < 1.0, "Learning rate must be in (0, 1)."
    assert 0.8 <= float(train_cfg["gamma"]) <= 1.0, "Discount factor gamma should be in [0.8, 1.0]."
    assert 0.0 <= float(train_cfg["gae_lambda"]) <= 1.0, "GAE lambda must be in [0, 1]."
    assert 0.0 < float(train_cfg["clip_range"]) < 1.0, "PPO clip range should be in (0, 1)."
    assert 0.0 <= float(train_cfg["ent_coef"]) <= 0.1, "Entropy coefficient should be non-negative and small."
    assert float(train_cfg["vf_coef"]) > 0.0, "Value function coefficient must be positive."

    # Neural network architecture validation
    net_arch = train_cfg["net_arch"]
    assert "pi" in net_arch and "vf" in net_arch, "net_arch must contain 'pi' and 'vf' definitions."
    assert isinstance(net_arch["pi"], list) and len(net_arch["pi"]) == 2
    assert isinstance(net_arch["vf"], list) and len(net_arch["vf"]) == 2
    assert net_arch["pi"] == [64, 64], "Expected policy network architecture [64, 64]."
    assert net_arch["vf"] == [64, 64], "Expected value network architecture [64, 64]."


def test_paths_and_evaluation_config():
    """Verify evaluation and storage path settings."""
    config = load_config("configs/default.yaml")
    eval_cfg = config["evaluation"]
    paths_cfg = config["paths"]

    assert isinstance(eval_cfg["eval_episodes"], int) and eval_cfg["eval_episodes"] > 0
    assert isinstance(eval_cfg["deterministic"], bool)

    for path_key in ["model_dir", "log_dir", "results_dir"]:
        assert path_key in paths_cfg, f"Missing path configuration key: {path_key}"
        assert isinstance(paths_cfg[path_key], str) and len(paths_cfg[path_key]) > 0
