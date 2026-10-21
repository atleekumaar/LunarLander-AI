"""Training curve and reward analysis module for Stable-Baselines3 Monitor logs.

Extracts reward and episode length logs from SB3 Monitor CSV files,
generates publication-quality smoothed visualization curves, computes rigorous
trend metrics with documented decision rules, and explains fundamental RL return concepts.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RL_CONCEPTS_EXPLANATION = """
================================================================================
           REINFORCEMENT LEARNING CONCEPTS: REWARD TAXONOMY
================================================================================
1. Instantaneous Reward (r_t):
   The immediate scalar feedback emitted by the environment at time step t
   after executing action a_t in state s_t.

2. Episode Return (Undiscounted G):
   The raw sum of instantaneous rewards from the start of an episode until
   termination/truncation: G = sum_{t=0}^{T-1} r_t.

3. Mean Episode Return:
   The empirical sample average of total episode returns over N observed
   episodes: (1 / N) * sum_{i=1}^{N} G_i. This is the primary scalar benchmark.

4. Discounted Return (G_t^gamma):
   The geometrically discounted cumulative future reward from step t onwards:
   G_t^gamma = sum_{k=0}^{infinity} (gamma^k * r_{t+k+1}), where gamma in [0, 1).
   Discounting ensures mathematical convergence in infinite horizons and models
   time-preference / uncertainty about distant futures.

5. Advantage Estimate (A(s_t, a_t)):
   The difference between the expected return of taking specific action a_t
   in state s_t, and the baseline expected value of simply being in state s_t:
   A(s, a) = Q(s, a) - V(s).
   In PPO, Generalized Advantage Estimation (GAE) computes A_hat to tell the
   policy whether an action was better (A > 0) or worse (A < 0) than average.

6. Built-in LunarLander-v3 Reward Structure:
   - Moving from the top of the screen to the landing pad: +100 to +140 points.
   - Moving away from the landing pad: Negative reward.
   - Lander coming to rest safely on the pad: +100 points bonus.
   - Lander crashing: -100 points penalty.
   - Each ground leg touching the surface: +10 points.
   - Firing main engine thruster: -0.3 points per frame (fuel cost).
   - Firing side orientation thrusters: -0.03 points per frame.
   - Solved threshold: An average return of +200 over 100 consecutive episodes.
================================================================================
"""


def load_monitor_data(log_path: str) -> Optional[pd.DataFrame]:
    """Load and parse Monitor CSV file produced by Stable-Baselines3.

    Args:
        log_path: Path to monitor CSV file or directory containing it.

    Returns:
        Optional[pd.DataFrame]: Parsed dataframe with columns ['r', 'l', 't'],
        or None if no valid monitor data is found.
    """
    csv_file = None
    if os.path.isdir(log_path):
        candidates = glob.glob(os.path.join(log_path, "*.monitor.csv")) + glob.glob(
            os.path.join(log_path, "monitor.csv")
        )
        if candidates:
            csv_file = candidates[0]
    elif os.path.isfile(log_path):
        csv_file = log_path

    if not csv_file or not os.path.exists(csv_file):
        return None

    try:
        # SB3 Monitor CSV has 1 line of JSON comment header followed by header row: r,l,t
        df = pd.read_csv(csv_file, skiprows=1)
        if "r" not in df.columns or "l" not in df.columns:
            return None
        # Clean potential empty or non-numeric rows
        df["r"] = pd.to_numeric(df["r"], errors="coerce")
        df["l"] = pd.to_numeric(df["l"], errors="coerce")
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        if len(df) == 0:
            return None
        # Add cumulative timesteps
        df["timesteps"] = df["l"].cumsum()
        df["episode"] = np.arange(1, len(df) + 1)
        return df
    except Exception as e:
        print(f"Error reading monitor file {csv_file}: {e}")
        return None


def evaluate_learning_trend(returns: np.ndarray) -> Tuple[str, str]:
    """Assess whether the observed return trajectory is improving, flat, or inconclusive.

    Documented Decision Rule:
      1. If fewer than 10 episodes are logged, classify as 'Inconclusive' (insufficient data).
      2. Compute linear regression slope (points per episode) and compare early quartile
         (first 25% of episodes) vs late quartile (last 25% of episodes).
      3. Classification criteria:
         - 'Improving': Linear slope is positive (> 0.05) AND late quartile mean exceeds
           early quartile mean by at least +15.0 points.
         - 'Flat': Absolute change between late and early quartile means is within [-15.0, +15.0]
           and the magnitude of the slope is <= 0.05.
         - 'Inconclusive': High variance without monotonic direction, negative trend,
           or conflicting slope and quartile signs.

    Args:
        returns: Array of episode returns in chronological order.

    Returns:
        Tuple[str, str]: (trend_label, explanation_string)
    """
    n = len(returns)
    if n < 10:
        return (
            "Inconclusive",
            f"Only {n} episode(s) recorded. A minimum of 10 episodes is required for statistical trend testing.",
        )

    episodes = np.arange(n)
    slope, intercept = np.polyfit(episodes, returns, 1)

    q_size = max(2, n // 4)
    early_mean = float(np.mean(returns[:q_size]))
    late_mean = float(np.mean(returns[-q_size:]))
    delta_mean = late_mean - early_mean

    if slope > 0.05 and delta_mean >= 15.0:
        trend = "Improving"
        reason = (
            f"Strong positive trajectory detected: Linear slope is +{slope:.3f} points/episode, "
            f"and final quartile mean ({late_mean:.1f}) gained {delta_mean:+.1f} points over initial quartile ({early_mean:.1f})."
        )
    elif abs(delta_mean) <= 15.0 and abs(slope) <= 0.05:
        trend = "Flat"
        reason = (
            f"Stationary trajectory detected: Linear slope is {slope:+.3f} points/episode, "
            f"and quartile mean change is minimal ({delta_mean:+.1f} points)."
        )
    else:
        trend = "Inconclusive"
        reason = (
            f"Fluctuating or noisy trajectory: Linear slope is {slope:+.3f} points/episode "
            f"with quartile difference of {delta_mean:+.1f} points (early: {early_mean:.1f}, late: {late_mean:.1f})."
        )

    return trend, reason


def plot_training_curves(
    df: pd.DataFrame,
    output_image_path: str = "results/training_curves.png",
    smoothing_window: Optional[int] = None,
) -> None:
    """Generate two-panel training performance plots (returns and episode lengths).

    Args:
        df: DataFrame containing parsed monitor data ('r', 'l', 'timesteps', 'episode').
        output_image_path: Destination path for the saved plot image.
        smoothing_window: Moving average window length (defaults to max(5, n // 15)).
    """
    n = len(df)
    if smoothing_window is None:
        smoothing_window = max(5, min(50, n // 10))

    smoothed_returns = df["r"].rolling(window=smoothing_window, min_periods=1).mean()
    smoothed_lengths = df["l"].rolling(window=smoothing_window, min_periods=1).mean()

    os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Top Panel: Episode Returns
    ax1.scatter(
        df["timesteps"],
        df["r"],
        color="#3b82f6",
        alpha=0.25,
        s=18,
        label="Raw Episode Return",
    )
    ax1.plot(
        df["timesteps"],
        smoothed_returns,
        color="#1d4ed8",
        linewidth=2.2,
        label=f"Rolling Mean (Window={smoothing_window})",
    )
    ax1.axhline(200.0, color="#10b981", linestyle="--", linewidth=1.5, label="Solved Threshold (+200)")
    ax1.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
    ax1.set_ylabel("Episode Return", fontsize=11, fontweight="bold")
    ax1.set_title("PPO Training Progress on LunarLander-v3", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Bottom Panel: Episode Lengths
    ax2.scatter(
        df["timesteps"],
        df["l"],
        color="#f97316",
        alpha=0.25,
        s=18,
        label="Raw Episode Length",
    )
    ax2.plot(
        df["timesteps"],
        smoothed_lengths,
        color="#c2410c",
        linewidth=2.2,
        label=f"Rolling Mean (Window={smoothing_window})",
    )
    ax2.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Episode Length (Steps)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] -> {output_image_path}")


def analyze_training_logs(
    log_dir: str = "logs",
    results_dir: str = "results",
) -> Optional[Dict[str, Any]]:
    """Analyze SB3 training logs and output trend evaluation and plots.

    Args:
        log_dir: Directory containing monitor CSV files.
        results_dir: Directory to save generated plot artifacts.

    Returns:
        Optional[Dict[str, Any]]: Analysis summary dictionary or None if logs missing.
    """
    print(RL_CONCEPTS_EXPLANATION)

    df = load_monitor_data(log_dir)
    if df is None or len(df) == 0:
        print(f"Notice: No completed training logs found in '{log_dir}'.")
        print("To generate training logs, run:")
        print("  python src/train_baseline.py --timesteps 100000")
        return None

    returns = df["r"].to_numpy()
    lengths = df["l"].to_numpy()
    timesteps = df["timesteps"].to_numpy()

    trend, reason = evaluate_learning_trend(returns)

    print("=" * 70)
    print("                 TRAINING LOG ANALYSIS REPORT")
    print("=" * 70)
    print(f"Total Episodes Logged:    {len(df)}")
    print(f"Total Timesteps Logged:   {int(timesteps[-1]):,}")
    print(f"Overall Mean Return:      {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Initial 10-Ep Mean:       {np.mean(returns[:10]):.2f}")
    print(f"Latest 10-Ep Mean:        {np.mean(returns[-10:]):.2f}")
    print(f"Max Return Achieved:      {np.max(returns):.2f}")
    print("-" * 70)
    print(f"TRAINING TREND STATUS:    [{trend.upper()}]")
    print(f"Decision Rule Rationale:  {reason}")
    print("=" * 70)

    plot_file = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(df, output_image_path=plot_file)

    return {
        "total_episodes": len(df),
        "total_timesteps": int(timesteps[-1]),
        "mean_return": round(float(np.mean(returns)), 2),
        "latest_10_mean": round(float(np.mean(returns[-10:])), 2),
        "max_return": round(float(np.max(returns)), 2),
        "trend": trend,
        "reason": reason,
        "plot_file": plot_file,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Stable-Baselines3 Monitor CSV logs and plot learning curves."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory containing Monitor CSV logs (default: logs).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where plots will be saved (default: results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_training_logs(log_dir=args.log_dir, results_dir=args.results_dir)
