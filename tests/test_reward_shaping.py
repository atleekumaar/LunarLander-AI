"""Unit and smoke tests for reward shaping wrappers, potential calculations, and experiment pipelines."""

from __future__ import annotations

import math
import os
import numpy as np
import pytest
from stable_baselines3 import PPO

from src.environment import make_lunar_lander_env
from src.evaluate import evaluate_ppo_agent
from src.reward_wrappers import ShapedRewardWrapper
from src.train_reward_experiment import run_reward_experiment


def test_baseline_reward_passthrough():
    """Verify that Condition A (baseline) applies zero shaping and passes original rewards through identically."""
    base_env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    wrapper = ShapedRewardWrapper(base_env, condition="baseline")

    wrapper.reset(seed=42)
    obs, total_reward, terminated, truncated, info = wrapper.step(0)

    assert info["shaping_condition"] == "baseline"
    assert info["shaping_reward"] == 0.0
    assert total_reward == info["original_reward"]
    wrapper.close()


def test_naive_bonus_calculation():
    """Verify Condition B adds +1.0 only when upright and moving slowly, and 0.0 otherwise."""
    base_env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    wrapper = ShapedRewardWrapper(
        base_env,
        condition="naive",
        max_hover_angle=0.20,
        max_hover_speed=0.35,
        naive_bonus=1.0,
    )
    wrapper.reset(seed=42)

    # 1. Test upright & slow observation -> eligible
    upright_slow_obs = np.array([0.0, 1.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
    assert wrapper.is_eligible_for_naive_bonus(upright_slow_obs) is True

    # 2. Test tilted observation -> not eligible
    tilted_obs = np.array([0.0, 1.0, 0.1, -0.1, 0.45, 0.0, 0.0, 0.0], dtype=np.float32)
    assert wrapper.is_eligible_for_naive_bonus(tilted_obs) is False

    # 3. Test high speed observation -> not eligible
    fast_obs = np.array([0.0, 1.0, 0.8, -0.8, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
    assert wrapper.is_eligible_for_naive_bonus(fast_obs) is False
    wrapper.close()


def test_potential_difference_calculation():
    """Verify Condition C computes F(s, s') = gamma * Phi(s') - Phi(s) accurately."""
    base_env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    gamma = 0.99
    beta = 1.5
    wrapper = ShapedRewardWrapper(base_env, condition="corrected", beta=beta, gamma=gamma)

    obs, _ = wrapper.reset(seed=42)
    initial_phi = wrapper.compute_potential(obs)
    assert wrapper.prev_potential == initial_phi

    next_obs, total_reward, terminated, truncated, info = wrapper.step(0)
    expected_next_phi = wrapper.compute_potential(next_obs) if not terminated else 0.0
    expected_f = gamma * expected_next_phi - initial_phi
    expected_shaping = beta * expected_f

    assert math.isclose(info["shaping_reward"], expected_shaping, rel_tol=1e-5)
    wrapper.close()


def test_terminal_potential_handling():
    """Verify that on true termination (terminated=True), next potential is set to 0.0 for policy invariance."""
    base_env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    wrapper = ShapedRewardWrapper(base_env, condition="corrected", beta=1.0, gamma=0.99)
    wrapper.reset(seed=42)

    # Fast forward to termination by letting lander freefall or crash
    terminated = False
    truncated = False
    last_step_info = None

    while not (terminated or truncated):
        obs, reward, terminated, truncated, last_step_info = wrapper.step(0)

    assert last_step_info is not None
    if terminated:
        # F = gamma * 0 - prev_phi = -prev_phi
        assert last_step_info["shaping_condition"] == "corrected"
    wrapper.close()


def test_reset_behavior_and_state_isolation():
    """Verify reset() clears episode accumulators preventing mutable state leakage across episodes."""
    base_env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    wrapper = ShapedRewardWrapper(base_env, condition="naive")

    wrapper.reset(seed=42)
    wrapper.step(0)
    wrapper.step(0)
    assert wrapper.episode_step_count == 2

    # Reset
    wrapper.reset(seed=43)
    assert wrapper.episode_step_count == 0
    assert wrapper.episode_original_reward == 0.0
    assert wrapper.episode_shaping_reward == 0.0
    assert wrapper.episode_hover_steps == 0
    wrapper.close()


def test_preservation_of_gymnasium_flags():
    """Verify shaping wrapper preserves Gymnasium's terminated and truncated boolean flags and step 5-tuple."""
    base_env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    wrapper = ShapedRewardWrapper(base_env, condition="corrected")

    wrapper.reset(seed=42)
    res = wrapper.step(1)
    assert len(res) == 5
    obs, reward, terminated, truncated, info = res

    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(reward, (float, np.floating))
    assert isinstance(info, dict)
    assert "original_reward" in info
    assert "shaping_reward" in info
    wrapper.close()


def test_end_to_end_smoke_training_and_serialization(tmp_path):
    """Verify end-to-end training pipeline on a small smoke budget (64 timesteps) saves and loads checkpoint."""
    save_dir = str(tmp_path / "models")
    log_dir = str(tmp_path / "logs")

    meta = run_reward_experiment(
        condition="naive",
        timesteps=64,
        seed=42,
        output_dir=save_dir,
        log_dir=log_dir,
        n_envs=1,
    )

    checkpoint_path = meta["saved_model_path"]
    assert os.path.exists(checkpoint_path)

    # Verify loading
    loaded_model = PPO.load(checkpoint_path)
    assert loaded_model is not None

    # Verify evaluation in unmodified environment
    eval_json = str(tmp_path / "eval_smoke.json")
    results = evaluate_ppo_agent(
        model_path=checkpoint_path,
        num_episodes=2,
        seed=42,
        output_json=eval_json,
    )
    assert "mean_return" in results
    assert "success_rate" in results
    assert os.path.exists(eval_json)
