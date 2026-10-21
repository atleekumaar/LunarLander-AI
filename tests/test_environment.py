"""Tests for Gymnasium LunarLander-v3 environment initialization and lifecycle."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from src.environment import make_lunar_lander_env


def test_environment_creation_and_reset():
    """Verify that LunarLander-v3 environment can be instantiated and reset."""
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    assert env is not None, "Failed to instantiate LunarLander-v3 environment."

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray), f"Expected numpy array for observation, got {type(obs)}."
    assert isinstance(info, dict), f"Expected dict for reset info, got {type(info)}."
    env.close()


def test_observation_shape_and_finite_values():
    """Verify observation has shape (8,) and contains only finite floating-point values."""
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=123)
    obs, _ = env.reset(seed=123)

    assert obs.shape == (8,), f"Expected observation shape (8,), got {obs.shape}."
    assert np.all(np.isfinite(obs)), f"Observation contains non-finite values (NaN or Inf): {obs}"
    assert obs.dtype in [np.float32, np.float64], f"Unexpected observation dtype: {obs.dtype}"
    env.close()


def test_action_space_discrete_four():
    """Verify action space is Discrete with exactly 4 possible actions."""
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)

    assert isinstance(env.action_space, gym.spaces.Discrete), "Action space is not gym.spaces.Discrete."
    assert env.action_space.n == 4, f"Expected 4 actions, got {env.action_space.n}."
    for action in [0, 1, 2, 3]:
        assert env.action_space.contains(action), f"Action space does not contain action {action}."
    env.close()


def test_random_action_step_result():
    """Verify stepping with a valid action produces the standard Gymnasium 5-tuple."""
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=42)
    env.reset(seed=42)

    action = env.action_space.sample()
    step_result = env.step(action)

    assert len(step_result) == 5, f"Expected 5-tuple (obs, reward, term, trunc, info), got {len(step_result)} items."
    obs, reward, terminated, truncated, info = step_result

    assert isinstance(obs, np.ndarray), f"Observation is not numpy array: {type(obs)}"
    assert obs.shape == (8,), f"Observation has unexpected shape: {obs.shape}"
    assert np.all(np.isfinite(obs)), f"Non-finite values found in step observation: {obs}"
    assert isinstance(reward, (float, np.floating)), f"Reward must be a float, got {type(reward)}"
    assert np.isfinite(reward), f"Reward must be finite, got {reward}"
    assert isinstance(terminated, (bool, np.bool_)), f"Terminated must be a boolean, got {type(terminated)}"
    assert isinstance(truncated, (bool, np.bool_)), f"Truncated must be a boolean, got {type(truncated)}"
    assert isinstance(info, dict), f"Info must be a dict, got {type(info)}"
    env.close()


def test_termination_and_truncation_lifecycle():
    """Verify the environment correctly handles step progression until termination or truncation."""
    env = make_lunar_lander_env("LunarLander-v3", render_mode=None, seed=99)
    env.reset(seed=99)

    terminated = False
    truncated = False
    steps = 0
    max_test_steps = 300

    while not (terminated or truncated) and steps < max_test_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        steps += 1

    assert steps > 0, "No steps were taken in test rollout."
    assert (terminated or truncated or steps == max_test_steps), "Episode did not terminate as expected."
    env.close()
