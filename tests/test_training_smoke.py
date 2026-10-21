"""Smoke tests for PPO model instantiation, short training cycle, model serialization, inference, and evaluation."""

from __future__ import annotations

import os
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import make_lunar_lander_env
from src.evaluate import evaluate_ppo_agent


def test_ppo_model_instantiation():
    """Requirement 7: Verify PPO model can be instantiated with custom 64x64 net_arch."""
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        seed=42,
        verbose=0,
    )
    assert model is not None
    assert model.observation_space.shape == (8,)
    assert model.action_space.n == 4
    env.close()


def test_ppo_short_smoke_training_and_saving(tmp_path):
    """Requirement 8: Verify a minimal smoke training run (128 steps) completes and saves model checkpoint."""
    save_dir = tmp_path / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "smoke_model.zip")

    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        seed=42,
        verbose=0,
    )

    # Minimal smoke training
    model.learn(total_timesteps=128)
    model.save(save_path)
    env.close()

    assert os.path.exists(save_path), f"Saved smoke model checkpoint not found at: {save_path}"
    assert os.path.getsize(save_path) > 1000, "Saved model file is unexpectedly small or empty."


def test_saved_model_loading_and_inference(tmp_path):
    """Requirement 9: Verify saved model can be loaded back into memory and generates valid actions."""
    save_path = str(tmp_path / "smoke_model.zip")

    # Create and save a quick model
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        n_steps=64,
        batch_size=32,
        verbose=0,
    )
    model.learn(total_timesteps=64)
    model.save(save_path)
    env.close()

    # Load model back
    loaded_model = PPO.load(save_path)
    assert loaded_model is not None, "Failed to load saved PPO model."

    # Test inference on test observation
    test_obs = np.array([0.0, 1.4, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Test deterministic action
    action_det, _ = loaded_model.predict(test_obs, deterministic=True)
    assert action_det.item() in [0, 1, 2, 3], f"Invalid deterministic action: {action_det}"

    # Test stochastic action
    action_stoch, _ = loaded_model.predict(test_obs, deterministic=False)
    assert action_stoch.item() in [0, 1, 2, 3], f"Invalid stochastic action: {action_stoch}"


def test_evaluation_produces_expected_metrics_structure(tmp_path):
    """Requirement 10: Verify evaluate_ppo_agent produces the comprehensive metrics dictionary."""
    model_path = str(tmp_path / "smoke_eval_model.zip")
    json_path = str(tmp_path / "eval_output.json")

    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        n_steps=64,
        batch_size=32,
        verbose=0,
    )
    model.learn(total_timesteps=64)
    model.save(model_path)
    env.close()

    # Run evaluation for 2 episodes
    results = evaluate_ppo_agent(
        model_path=model_path,
        env_id="LunarLander-v3",
        num_episodes=2,
        render=False,
        deterministic=True,
        seed=42,
        output_json=json_path,
    )

    required_keys = [
        "model_path",
        "num_episodes",
        "deterministic",
        "seed",
        "mean_return",
        "std_return",
        "median_return",
        "min_return",
        "max_return",
        "mean_length",
        "successful_landings",
        "crashes",
        "truncations",
        "success_rate",
        "crash_rate",
        "truncation_rate",
        "is_solved",
        "episodes",
    ]
    for key in required_keys:
        assert key in results, f"Missing expected key in evaluation results: '{key}'"

    assert len(results["episodes"]) == 2
    assert results["successful_landings"] + results["crashes"] + results["truncations"] == 2
    assert os.path.exists(json_path), "Failed to save evaluation JSON output."
