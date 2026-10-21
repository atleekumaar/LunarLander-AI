"""Comparison module for evaluating and benchmarking multiple LunarLander-v3 models.

Reads evaluation JSON reports, generates comparison tables, calculates deltas,
and exports structured comparison JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def compare_models(
    baseline_json: str = "results/eval_20k_baseline.json",
    stronger_json: str = "results/eval_320k_resumed.json",
    output_json: str = "results/model_comparison.json",
) -> Dict[str, Any]:
    """Compare evaluation results of two models and produce structured reports.

    Args:
        baseline_json: Path to baseline model evaluation JSON.
        stronger_json: Path to stronger model evaluation JSON.
        output_json: Destination for output comparison JSON.

    Returns:
        Dict[str, Any]: Complete model comparison data.
    """
    if not os.path.exists(baseline_json):
        raise FileNotFoundError(f"Baseline evaluation file not found: {baseline_json}")
    if not os.path.exists(stronger_json):
        raise FileNotFoundError(f"Stronger model evaluation file not found: {stronger_json}")

    with open(baseline_json, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(stronger_json, "r", encoding="utf-8") as f:
        strong = json.load(f)

    # Compute comparison metrics
    delta_mean_return = strong["mean_return"] - base["mean_return"]
    delta_median_return = strong["median_return"] - base["median_return"]
    delta_success_rate = strong["success_rate"] - base["success_rate"]
    delta_crash_rate = strong["crash_rate"] - base["crash_rate"]
    delta_mean_length = strong["mean_length"] - base["mean_length"]

    comparison_data = {
        "evaluation_protocol": {
            "num_episodes": base["num_episodes"],
            "deterministic": base["deterministic"],
            "base_seed": base["seed"],
        },
        "models": {
            "baseline_20k": {
                "checkpoint": base["model_path"],
                "mean_return": base["mean_return"],
                "std_return": base["std_return"],
                "median_return": base["median_return"],
                "min_return": base["min_return"],
                "max_return": base["max_return"],
                "mean_length": base["mean_length"],
                "successful_landings": base["successful_landings"],
                "crashes": base["crashes"],
                "truncations": base["truncations"],
                "success_rate_pct": base["success_rate"],
                "crash_rate_pct": base["crash_rate"],
                "truncation_rate_pct": base["truncation_rate"],
                "is_solved": base["is_solved"],
            },
            "stronger_320k": {
                "checkpoint": strong["model_path"],
                "mean_return": strong["mean_return"],
                "std_return": strong["std_return"],
                "median_return": strong["median_return"],
                "min_return": strong["min_return"],
                "max_return": strong["max_return"],
                "mean_length": strong["mean_length"],
                "successful_landings": strong["successful_landings"],
                "crashes": strong["crashes"],
                "truncations": strong["truncations"],
                "success_rate_pct": strong["success_rate"],
                "crash_rate_pct": strong["crash_rate"],
                "truncation_rate_pct": strong["truncation_rate"],
                "is_solved": strong["is_solved"],
            },
        },
        "deltas": {
            "mean_return_delta": round(delta_mean_return, 2),
            "median_return_delta": round(delta_median_return, 2),
            "success_rate_delta_pct": round(delta_success_rate, 2),
            "crash_rate_delta_pct": round(delta_crash_rate, 2),
            "mean_length_delta": round(delta_mean_length, 1),
        },
    }

    # Print formatted comparison table
    print("=" * 80)
    print("             MODEL BENCHMARK COMPARISON TABLE (20 EPISODES)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Baseline (20k)':<18} | {'Stronger (320k)':<18} | {'Delta':<12}")
    print("-" * 80)
    print(f"{'Mean Return':<28} | {base['mean_return']:>14.2f}     | {strong['mean_return']:>14.2f}     | {delta_mean_return:>+10.2f}")
    print(f"{'Std Return':<28} | {base['std_return']:>14.2f}     | {strong['std_return']:>14.2f}     | {strong['std_return'] - base['std_return']:>+10.2f}")
    print(f"{'Median Return':<28} | {base['median_return']:>14.2f}     | {strong['median_return']:>14.2f}     | {delta_median_return:>+10.2f}")
    print(f"{'Min Return':<28} | {base['min_return']:>14.2f}     | {strong['min_return']:>14.2f}     | {strong['min_return'] - base['min_return']:>+10.2f}")
    print(f"{'Max Return':<28} | {base['max_return']:>14.2f}     | {strong['max_return']:>14.2f}     | {strong['max_return'] - base['max_return']:>+10.2f}")
    print(f"{'Mean Episode Length':<28} | {base['mean_length']:>14.1f}     | {strong['mean_length']:>14.1f}     | {delta_mean_length:>+10.1f}")
    print(f"{'Landing Success Rate':<28} | {base['success_rate']:>13.1f}%     | {strong['success_rate']:>13.1f}%     | {delta_success_rate:>+9.1f}%")
    print(f"{'Crash Rate':<28} | {base['crash_rate']:>13.1f}%     | {strong['crash_rate']:>13.1f}%     | {delta_crash_rate:>+9.1f}%")
    print(f"{'Truncation Rate':<28} | {base['truncation_rate']:>13.1f}%     | {strong['truncation_rate']:>13.1f}%     | {strong['truncation_rate'] - base['truncation_rate']:>+9.1f}%")
    print(f"{'Solved Benchmark (>= 200)':<28} | {'No':<18} | {'No (Partial)':<18} | {'In progress':<12}")
    print("=" * 80)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n[Comparison Saved] -> {output_json}")

    return comparison_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results of baseline vs stronger LunarLander models."
    )
    parser.add_argument(
        "--baseline_json",
        type=str,
        default="results/eval_20k_baseline.json",
        help="Path to baseline evaluation JSON (default: results/eval_20k_baseline.json).",
    )
    parser.add_argument(
        "--stronger_json",
        type=str,
        default="results/eval_320k_resumed.json",
        help="Path to stronger model evaluation JSON (default: results/eval_320k_resumed.json).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/model_comparison.json",
        help="Destination path for comparison JSON (default: results/model_comparison.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_models(
        baseline_json=args.baseline_json,
        stronger_json=args.stronger_json,
        output_json=args.output_json,
    )
