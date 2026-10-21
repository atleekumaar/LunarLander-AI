"""Environment module for LunarLander-v3.

Provides functions to create, configure, inspect, and demonstrate the
LunarLander-v3 environment from Gymnasium with proper lifecycle management.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np


OBSERVATION_DESCRIPTIONS = [
    ("Horizontal Position (x)", "Position along x-axis; landing pad is centered at 0.0."),
    ("Vertical Position (y)", "Position along y-axis; 0.0 is roughly ground/pad level."),
    ("Horizontal Velocity (vx)", "Linear velocity along the horizontal axis."),
    ("Vertical Velocity (vy)", "Linear velocity along the vertical axis (negative means falling)."),
    ("Angular Position (angle)", "Lander tilt angle in radians (0 is upright)."),
    ("Angular Velocity (omega)", "Rotational speed in radians per second."),
    ("Left Leg Contact", "Boolean indicator: 1.0 if left leg is touching ground, else 0.0."),
    ("Right Leg Contact", "Boolean indicator: 1.0 if right leg is touching ground, else 0.0."),
]

ACTION_DESCRIPTIONS = {
    0: ("Do Nothing", "All thrusters idle; vehicle free-falls under lunar gravity."),
    1: ("Fire Left Engine", "Orientation thruster on the left fires, rotating lander clockwise (right)."),
    2: ("Fire Main Engine", "Main central thruster fires, pushing lander upward against gravity."),
    3: ("Fire Right Engine", "Orientation thruster on the right fires, rotating lander counter-clockwise (left)."),
}


def make_lunar_lander_env(
    env_id: str = "LunarLander-v3",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """Instantiate and optionally seed a LunarLander environment.

    Args:
        env_id: Gymnasium environment ID, defaults to 'LunarLander-v3'.
        render_mode: 'human' for GUI display, 'rgb_array' for frame capture, or None for headless.
        seed: Optional integer random seed for reproducibility.

    Returns:
        gym.Env: Instantiated gymnasium environment.
    """
    valid_modes = [None, "human", "rgb_array"]
    if render_mode not in valid_modes:
        raise ValueError(f"Invalid render_mode '{render_mode}'. Must be one of {valid_modes}.")

    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def print_environment_specs(env: gym.Env) -> None:
    """Print detailed specification of the environment observation and action spaces."""
    print("=" * 70)
    print("         LUNAR LANDER v3 — ENVIRONMENT SPECIFICATIONS")
    print("=" * 70)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space:      {env.action_space}")
    print("\nObservation Vector Details (8 continuous state dimensions):")
    for idx, (name, desc) in enumerate(OBSERVATION_DESCRIPTIONS):
        low = env.observation_space.low[idx] if hasattr(env.observation_space, "low") else -np.inf
        high = env.observation_space.high[idx] if hasattr(env.observation_space, "high") else np.inf
        print(f"  [{idx}] {name:<26} Low: {low:>8.2f} | High: {high:>8.2f}")
        print(f"      -> {desc}")

    print("\nAction Space Details (4 discrete actions):")
    for action_idx, (name, desc) in ACTION_DESCRIPTIONS.items():
        print(f"  Action {action_idx}: {name:<18} -> {desc}")
    print("=" * 70)


def run_random_demonstration(
    env_id: str = "LunarLander-v3",
    render_mode: Optional[str] = None,
    num_episodes: int = 1,
    max_steps: int = 500,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Execute a random policy demonstration to inspect environment dynamics.

    Args:
        env_id: The environment ID.
        render_mode: Render mode ('human', 'rgb_array', or None).
        num_episodes: Number of demonstration episodes to run.
        max_steps: Maximum timesteps per episode before manual truncation.
        seed: Random seed for repeatability.

    Returns:
        Dict[str, Any]: Demonstration summary statistics.
    """
    env = make_lunar_lander_env(env_id=env_id, render_mode=render_mode, seed=seed)
    print_environment_specs(env)

    print("\n" + "-" * 70)
    print(f"RUNNING RANDOM POLICY DEMONSTRATION ({num_episodes} episode(s))")
    print("-" * 70)

    episode_summaries = []

    try:
        for ep in range(1, num_episodes + 1):
            ep_seed = seed + ep if seed is not None else None
            obs, info = env.reset(seed=ep_seed)

            print(f"\n--- Episode {ep}/{num_episodes} Initial State (Seed: {ep_seed}) ---")
            print(f"Observation raw array: {np.array2string(obs, precision=4, suppress_small=True)}")
            print("Decoded Initial State:")
            for idx, (name, _) in enumerate(OBSERVATION_DESCRIPTIONS):
                print(f"  {name:<26}: {obs[idx]:.4f}")

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated) and steps < max_steps:
                action = env.action_space.sample()  # Random uniform action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            left_contact = bool(obs[6])
            right_contact = bool(obs[7])

            ep_summary = {
                "episode": ep,
                "steps": steps,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "left_leg_contact": left_contact,
                "right_leg_contact": right_contact,
                "final_position": (float(obs[0]), float(obs[1])),
            }
            episode_summaries.append(ep_summary)

            print(f"\nEpisode {ep} Outcome:")
            print(f"  Total Steps:      {steps}")
            print(f"  Cumulative Reward: {total_reward:.2f}")
            print(f"  Terminated:       {terminated} (Crash or Landing reached)")
            print(f"  Truncated:        {truncated} (Time limit reached)")
            print(f"  Leg Contact:      Left={left_contact}, Right={right_contact}")
            print(f"  Final Position:   x={obs[0]:.3f}, y={obs[1]:.3f}")

    finally:
        env.close()
        print("\nEnvironment successfully closed.")

    return {
        "num_episodes": num_episodes,
        "episodes": episode_summaries,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inspecting the LunarLander environment."""
    parser = argparse.ArgumentParser(
        description="Inspect and demonstrate the LunarLander-v3 Gymnasium environment."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human visualization window (requires GUI display).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of random-policy episodes to run (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for demonstration reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_mode = "human" if args.render else None
    run_random_demonstration(
        render_mode=render_mode,
        num_episodes=args.episodes,
        seed=args.seed,
    )
