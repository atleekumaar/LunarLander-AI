"""Evaluation and validation module for trained LunarLander-v3 PPO agents.

Measures agent performance across multiple episodes with deterministic inference,
reporting returns, episode lengths, and verified physical landing criteria
(distinguishing actual successful landings, crashes, and time-limit truncations).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from src.environment import make_lunar_lander_env


def classify_episode_outcome(
    terminated: bool,
    truncated: bool,
    total_reward: float,
    last_reward: float,
    final_obs: np.ndarray,
) -> str:
    """Classify episode outcome based on physical landing indicators and termination.

    In LunarLander-v3:
      - A crash ends the episode with `terminated=True` and a -100 penalty.
      - A successful landing ends with `terminated=True`, +100 bonus for coming to rest,
        both legs firmly touching the ground (final_obs[6] == 1.0 and final_obs[7] == 1.0),
        and small linear and angular velocities.
      - A truncation happens when the step limit (typically 1,000 steps) is reached
        without the lander coming to rest or crashing.

    Important: A positive cumulative reward alone is NEVER treated as proof of landing.
    Hovering or drifting towards the center can yield positive reward without landing.

    Returns:
        One of: 'SUCCESSFUL_LANDING', 'CRASH', 'TRUNCATED'.
    """
    left_leg_contact = bool(final_obs[6] >= 0.9)
    right_leg_contact = bool(final_obs[7] >= 0.9)
    both_legs_touching = left_leg_contact and right_leg_contact

    if truncated:
        return "TRUNCATED"

    if terminated:
        # A true landing requires both legs touching, centered near pad, and terminal bonus
        # or overall high positive reward. In Gym Box2D LunarLander, terminal landing reward adds +100.
        if both_legs_touching and (last_reward >= 50.0 or total_reward >= 150.0):
            return "SUCCESSFUL_LANDING"
        else:
            return "CRASH"

    # Fallback if neither terminated nor truncated flag triggered
    return "TRUNCATED"


def evaluate_ppo_agent(
    model_path: str = "models/ppo_lunarlander_baseline.zip",
    env_id: str = "LunarLander-v3",
    num_episodes: int = 20,
    render: bool = False,
    deterministic: bool = True,
    seed: int = 42,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained PPO model over a specified number of episodes.

    Args:
        model_path: Path to the saved PPO model (.zip).
        env_id: Gymnasium environment ID.
        num_episodes: Number of episodes to evaluate.
        render: Whether to display a GUI window during evaluation.
        deterministic: Use deterministic actions (mean of Gaussian / argmax of categorical).
        seed: Base random seed for evaluation reproducibility.
        output_json: Optional path to save evaluation results as JSON.

    Returns:
        Dict[str, Any]: Comprehensive evaluation report dictionary.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print("=" * 70)
    print("           LUNAR LANDER v3 - PPO MODEL EVALUATION")
    print("=" * 70)
    print(f"Loading Model:      {model_path}")
    print(f"Evaluation Episodes:{num_episodes}")
    print(f"Deterministic:      {deterministic}")
    print(f"Base Seed:          {seed}")
    print(f"Render Mode:        {'human' if render else 'headless'}")

    model = PPO.load(model_path)
    render_mode = "human" if render else None
    env = make_lunar_lander_env(env_id=env_id, render_mode=render_mode, seed=seed)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    outcomes: List[str] = []
    episode_details: List[Dict[str, Any]] = []

    try:
        for ep in range(1, num_episodes + 1):
            ep_seed = seed + ep * 1000
            obs, info = env.reset(seed=ep_seed)
            ep_reward = 0.0
            ep_length = 0
            terminated = False
            truncated = False
            last_reward = 0.0

            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                last_reward = float(reward)
                ep_length += 1

            outcome = classify_episode_outcome(
                terminated=terminated,
                truncated=truncated,
                total_reward=ep_reward,
                last_reward=last_reward,
                final_obs=obs,
            )

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            outcomes.append(outcome)

            ep_data = {
                "episode": ep,
                "seed": ep_seed,
                "return": round(ep_reward, 2),
                "length": ep_length,
                "outcome": outcome,
                "final_pos_x": round(float(obs[0]), 3),
                "final_pos_y": round(float(obs[1]), 3),
                "left_leg_contact": bool(obs[6]),
                "right_leg_contact": bool(obs[7]),
            }
            episode_details.append(ep_data)

    finally:
        env.close()

    # Calculate summary statistics
    mean_return = float(np.mean(episode_rewards))
    std_return = float(np.std(episode_rewards))
    median_return = float(np.median(episode_rewards))
    min_return = float(np.min(episode_rewards))
    max_return = float(np.max(episode_rewards))
    mean_length = float(np.mean(episode_lengths))

    successful_landings = outcomes.count("SUCCESSFUL_LANDING")
    crashes = outcomes.count("CRASH")
    truncations = outcomes.count("TRUNCATED")

    success_rate = (successful_landings / num_episodes) * 100.0
    crash_rate = (crashes / num_episodes) * 100.0
    truncation_rate = (truncations / num_episodes) * 100.0

    print("\n" + "-" * 70)
    print(f"EVALUATION RESULTS ({num_episodes} EPISODES)")
    print("-" * 70)
    print(f"Mean Return:         {mean_return:>8.2f} +/- {std_return:.2f}")
    print(f"Median Return:       {median_return:>8.2f}")
    print(f"Min / Max Return:    {min_return:>8.2f} / {max_return:.2f}")
    print(f"Mean Episode Length: {mean_length:>8.1f} steps")
    print(f"Successful Landings: {successful_landings:>4}/{num_episodes} ({success_rate:.1f}%)")
    print(f"Crashes:             {crashes:>4}/{num_episodes} ({crash_rate:.1f}%)")
    print(f"Truncations (timeout):{truncations:>3}/{num_episodes} ({truncation_rate:.1f}%)")

    # Documented success criteria clarification
    is_solved = mean_return >= 200.0 and success_rate >= 80.0
    print("-" * 70)
    print(f"Status vs 200+ Benchmark: {'SOLVED (Consistently >= 200)' if is_solved else 'NOT SOLVED (Typical for early/short baseline)'}")
    print("=" * 70)

    results = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "seed": seed,
        "mean_return": round(mean_return, 2),
        "std_return": round(std_return, 2),
        "median_return": round(median_return, 2),
        "min_return": round(min_return, 2),
        "max_return": round(max_return, 2),
        "mean_length": round(mean_length, 1),
        "successful_landings": successful_landings,
        "crashes": crashes,
        "truncations": truncations,
        "success_rate": round(success_rate, 2),
        "crash_rate": round(crash_rate, 2),
        "truncation_rate": round(truncation_rate, 2),
        "is_solved": is_solved,
        "episodes": episode_details,
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation report saved to: {output_json}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PPO evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Stable-Baselines3 PPO agent on LunarLander-v3."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ppo_lunarlander_baseline.zip",
        help="Path to the saved model checkpoint (default: models/ppo_lunarlander_baseline.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes to run (default: 20).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment window during evaluation (default: headless).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions sampled from policy instead of deterministic argmax.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for evaluation (default: 42).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to output JSON results file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_ppo_agent(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        seed=args.seed,
        output_json=args.output_json,
    )
