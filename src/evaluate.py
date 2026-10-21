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
import math

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
    lander_awake: Optional[bool] = None,
    game_over: Optional[bool] = None,
) -> str:
    """Classify episode outcome based on physical landing indicators and termination.

    In Gymnasium LunarLander-v3:
      - A crash ends the episode with `game_over=True`, `terminated=True`, and a -100 penalty.
      - A successful landing ends with `lander.awake=False` (came to rest safely on ground),
        `terminated=True`, +100 terminal bonus, and both legs in contact (final_obs[6]==1 and final_obs[7]==1).
      - A truncation happens when the step limit (1,000 steps) is reached before crashing or resting.

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
        # Check ground-truth physics engine state if available
        if lander_awake is False and (game_over is False or game_over is None):
            return "SUCCESSFUL_LANDING"
        if game_over is True:
            return "CRASH"

        # Observation and reward telemetry fallback
        # Landing requires both legs touching, upright orientation, and positive terminal arrival
        if both_legs_touching and (last_reward >= 50.0 or total_reward >= 150.0):
            return "SUCCESSFUL_LANDING"
        else:
            return "CRASH"

    return "TRUNCATED"


def evaluate_ppo_agent(
    model_path: str = "models/ppo_lunarlander_baseline.zip",
    env_id: str = "LunarLander-v3",
    num_episodes: int = 20,
    render: bool = False,
    deterministic: bool = True,
    seed: int = 42,
    output_json: Optional[str] = None,
    shaping_condition: Optional[str] = None,
    beta: float = 1.0,
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
        shaping_condition: Optional shaping wrapper ('baseline', 'naive', 'corrected', or None).
        beta: Potential-based shaping factor if shaping_condition == 'corrected'.

    Returns:
        Dict[str, Any]: Comprehensive evaluation report dictionary.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    from src.reward_wrappers import ShapedRewardWrapper

    print("=" * 70)
    print("           LUNAR LANDER v3 - PPO MODEL EVALUATION")
    print("=" * 70)
    print(f"Loading Model:      {model_path}")
    print(f"Evaluation Episodes:{num_episodes}")
    print(f"Deterministic:      {deterministic}")
    print(f"Base Seed:          {seed}")
    print(f"Render Mode:        {'human' if render else 'headless'}")
    print(f"Shaping Wrapper:    {shaping_condition or 'None (Original Gymnasium)'}")

    model = PPO.load(model_path)
    render_mode = "human" if render else None
    base_env = make_lunar_lander_env(env_id=env_id, render_mode=render_mode, seed=seed)

    if shaping_condition:
        env = ShapedRewardWrapper(base_env, condition=shaping_condition, beta=beta)
    else:
        env = base_env

    orig_rewards: List[float] = []
    shaped_rewards: List[float] = []
    shaping_bonuses: List[float] = []
    episode_lengths: List[int] = []
    hover_percentages: List[float] = []
    main_engine_percentages: List[float] = []
    side_engine_percentages: List[float] = []
    outcomes: List[str] = []
    episode_details: List[Dict[str, Any]] = []

    try:
        for ep in range(1, num_episodes + 1):
            ep_seed = seed + ep * 1000
            obs, info = env.reset(seed=ep_seed)
            ep_orig_reward = 0.0
            ep_shaped_reward = 0.0
            ep_shaping_bonus = 0.0
            ep_length = 0
            ep_hover_steps = 0
            ep_main_engine_steps = 0
            ep_side_engine_steps = 0
            terminated = False
            truncated = False
            last_orig_reward = 0.0

            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=deterministic)
                act_int = int(action.item() if hasattr(action, "item") else action)

                if act_int == 2:
                    ep_main_engine_steps += 1
                elif act_int in [1, 3]:
                    ep_side_engine_steps += 1

                # Check hover condition
                is_upright = abs(float(obs[4])) <= 0.20
                speed = math.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2)
                if is_upright and speed <= 0.35:
                    ep_hover_steps += 1

                obs, step_reward, terminated, truncated, step_info = env.step(action)

                # Extract separated reward components
                r_orig = step_info.get("original_reward", step_reward)
                r_shape = step_info.get("shaping_reward", 0.0)

                ep_orig_reward += float(r_orig)
                ep_shaped_reward += float(step_reward)
                ep_shaping_bonus += float(r_shape)
                last_orig_reward = float(r_orig)
                ep_length += 1

            unwrapped = getattr(env, "unwrapped", env)
            lander = getattr(unwrapped, "lander", None)
            lander_awake = getattr(lander, "awake", None)
            game_over = getattr(unwrapped, "game_over", None)

            outcome = classify_episode_outcome(
                terminated=terminated,
                truncated=truncated,
                total_reward=ep_orig_reward,
                last_reward=last_orig_reward,
                final_obs=obs,
                lander_awake=lander_awake,
                game_over=game_over,
            )

            hover_pct = (ep_hover_steps / max(1, ep_length)) * 100.0
            main_pct = (ep_main_engine_steps / max(1, ep_length)) * 100.0
            side_pct = (ep_side_engine_steps / max(1, ep_length)) * 100.0

            orig_rewards.append(ep_orig_reward)
            shaped_rewards.append(ep_shaped_reward)
            shaping_bonuses.append(ep_shaping_bonus)
            episode_lengths.append(ep_length)
            hover_percentages.append(hover_pct)
            main_engine_percentages.append(main_pct)
            side_engine_percentages.append(side_pct)
            outcomes.append(outcome)

            ep_data = {
                "episode": ep,
                "seed": ep_seed,
                "original_return": round(ep_orig_reward, 2),
                "shaped_return": round(ep_shaped_reward, 2),
                "shaping_bonus": round(ep_shaping_bonus, 2),
                "length": ep_length,
                "outcome": outcome,
                "hover_pct": round(hover_pct, 1),
                "main_engine_pct": round(main_pct, 1),
                "side_engine_pct": round(side_pct, 1),
                "final_pos_x": round(float(obs[0]), 3),
                "final_pos_y": round(float(obs[1]), 3),
                "left_leg_contact": bool(obs[6]),
                "right_leg_contact": bool(obs[7]),
            }
            episode_details.append(ep_data)

    finally:
        env.close()

    # Calculate summary statistics (Primary: Original Gymnasium Return)
    mean_orig_return = float(np.mean(orig_rewards))
    std_orig_return = float(np.std(orig_rewards))
    median_orig_return = float(np.median(orig_rewards))
    min_orig_return = float(np.min(orig_rewards))
    max_orig_return = float(np.max(orig_rewards))

    mean_shaped_return = float(np.mean(shaped_rewards))
    mean_shaping_bonus = float(np.mean(shaping_bonuses))

    mean_length = float(np.mean(episode_lengths))
    mean_hover_pct = float(np.mean(hover_percentages))
    mean_main_pct = float(np.mean(main_engine_percentages))
    mean_side_pct = float(np.mean(side_engine_percentages))

    successful_landings = outcomes.count("SUCCESSFUL_LANDING")
    crashes = outcomes.count("CRASH")
    truncations = outcomes.count("TRUNCATED")

    success_rate = (successful_landings / num_episodes) * 100.0
    crash_rate = (crashes / num_episodes) * 100.0
    truncation_rate = (truncations / num_episodes) * 100.0

    print("\n" + "-" * 70)
    print(f"EVALUATION RESULTS ({num_episodes} EPISODES)")
    print("-" * 70)
    print(f"Original Return (Mean): {mean_orig_return:>8.2f} +/- {std_orig_return:.2f}")
    print(f"Original Return (Median): {median_orig_return:>6.2f}")
    print(f"Original Return (Range):  {min_orig_return:>6.2f} to {max_orig_return:.2f}")
    if shaping_condition:
        print(f"Shaped Return (Mean):   {mean_shaped_return:>8.2f}")
        print(f"Shaping Bonus (Mean):  {mean_shaping_bonus:>8.2f}")
    print(f"Mean Episode Length:    {mean_length:>8.1f} steps")
    print(f"Mean Hovering Time:     {mean_hover_pct:>8.1f}% of episode")
    print(f"Main Thruster Usage:    {mean_main_pct:>8.1f}% of frames")
    print(f"Side Thruster Usage:    {mean_side_pct:>8.1f}% of frames")
    print(f"Successful Landings:    {successful_landings:>4}/{num_episodes} ({success_rate:.1f}%)")
    print(f"Crashes:                {crashes:>4}/{num_episodes} ({crash_rate:.1f}%)")
    print(f"Time-limit Truncations: {truncations:>4}/{num_episodes} ({truncation_rate:.1f}%)")
    print("=" * 70)

    # Documented success criteria clarification (Unbiased: Original Return >= 200)
    is_solved = mean_orig_return >= 200.0 and success_rate >= 80.0
    print("-" * 70)
    print(f"Status vs 200+ Benchmark: {'SOLVED (Consistently >= 200)' if is_solved else 'NOT SOLVED (Typical for early/short baseline)'}")
    print("=" * 70)

    results = {
        "model_path": model_path,
        "shaping_condition": shaping_condition,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "seed": seed,
        "mean_return": round(mean_orig_return, 2),
        "std_return": round(std_orig_return, 2),
        "median_return": round(median_orig_return, 2),
        "min_return": round(min_orig_return, 2),
        "max_return": round(max_orig_return, 2),
        "mean_shaped_return": round(mean_shaped_return, 2) if shaping_condition else round(mean_orig_return, 2),
        "mean_shaping_bonus": round(mean_shaping_bonus, 2) if shaping_condition else 0.0,
        "mean_length": round(mean_length, 1),
        "mean_hover_pct": round(mean_hover_pct, 1),
        "mean_main_engine_pct": round(mean_main_pct, 1),
        "mean_side_engine_pct": round(mean_side_pct, 1),
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
    parser.add_argument(
        "--shaping_condition",
        type=str,
        default=None,
        choices=["baseline", "naive", "corrected"],
        help="Optional shaping wrapper for diagnostic evaluation (default: None for unmodified Gymnasium).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Scaling factor beta for potential-based shaping (default: 1.0).",
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
        shaping_condition=args.shaping_condition,
        beta=args.beta,
    )
