"""Reward shaping wrappers for LunarLander-v3.

Implements Gymnasium-compatible environment wrappers for controlled reward-shaping
experiments:
  Condition A: Baseline (Unmodified Gymnasium reward)
  Condition B: Naive Shaping (Persistent per-frame bonus for hovering upright & slow)
  Condition C: Corrected Potential-Based Reward Shaping (PBRS, Ng et al. 1999)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class ShapedRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper providing configurable reward shaping for LunarLander-v3.

    Maintains separation between original environment reward and custom shaping
    components, tracking diagnostics in step info dictionaries and episode counters.
    """

    def __init__(
        self,
        env: gym.Env,
        condition: str = "baseline",
        beta: float = 1.0,
        gamma: float = 0.99,
        max_hover_angle: float = 0.20,      # ~11.5 degrees
        max_hover_speed: float = 0.35,      # Linear translational speed
        naive_bonus: float = 1.0,
    ) -> None:
        """Initialize the shaped reward wrapper.

        Args:
            env: Base Gymnasium environment.
            condition: 'baseline', 'naive', or 'corrected'.
            beta: Scaling coefficient for potential-based shaping (Condition C).
            gamma: Discount factor matching PPO training for potential difference.
            max_hover_angle: Maximum tilt angle (radians) eligible for naive bonus.
            max_hover_speed: Maximum velocity norm (m/s) eligible for naive bonus.
            naive_bonus: Constant bonus per eligible step (Condition B).
        """
        super().__init__(env)
        valid_conditions = ["baseline", "naive", "corrected"]
        if condition not in valid_conditions:
            raise ValueError(f"Invalid condition '{condition}'. Must be one of {valid_conditions}.")

        self.condition = condition
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.max_hover_angle = float(max_hover_angle)
        self.max_hover_speed = float(max_hover_speed)
        self.naive_bonus = float(naive_bonus)

        # Ephemeral episode state (isolated between resets)
        self.prev_potential: Optional[float] = None
        self.episode_original_reward: float = 0.0
        self.episode_shaping_reward: float = 0.0
        self.episode_hover_steps: int = 0
        self.episode_step_count: int = 0

    def compute_potential(self, obs: np.ndarray) -> float:
        """Compute bounded potential function Phi(s) for Condition C.

        Based on Ng, Harada, and Russell (1999), potential-based reward shaping
        defines F(s, s') = gamma * Phi(s') - Phi(s). Under policy invariance
        guarantees, Phi(s) must be a real-valued state function independent of action.

        We formulate Phi(s) as negative normalized distance to optimal landing state:
          - Horizontal distance: |x|
          - Vertical distance: |y|
          - Horizontal velocity: |vx|
          - Vertical velocity: |vy|
          - Angular tilt: |angle|
          - Angular velocity: |omega|
          - Leg contact bonus: +0.5 each

        Max value: 0.0 (perfect touchdown at center pad).
        Negative values: proportional to distance, speed, and tilt.
        """
        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        omega = float(obs[5])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        # Distance to center landing pad
        dist = math.sqrt(x * x + y * y)
        speed = math.sqrt(vx * vx + vy * vy)
        tilt = abs(angle)
        angular_speed = abs(omega)

        # Bounded potential: higher (closer to 0) as lander approaches target state
        phi = (
            - 1.5 * dist
            - 1.0 * speed
            - 1.0 * tilt
            - 0.5 * angular_speed
            + 0.5 * left_leg
            + 0.5 * right_leg
        )
        return float(phi)

    def is_eligible_for_naive_bonus(self, obs: np.ndarray) -> bool:
        """Check if current observation satisfies Condition B naive shaping criteria.

        Criteria:
          1. Lander is sufficiently upright (|angle| <= max_hover_angle).
          2. Lander is moving slowly (sqrt(vx^2 + vy^2) <= max_hover_speed).
          3. Lander is not resting on the ground yet (both legs not fully landed).
        """
        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        speed = math.sqrt(vx * vx + vy * vy)
        is_upright = abs(angle) <= self.max_hover_angle
        is_slow = speed <= self.max_hover_speed
        return is_upright and is_slow

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and initialize state tracking."""
        obs, info = super().reset(seed=seed, options=options)

        self.episode_original_reward = 0.0
        self.episode_shaping_reward = 0.0
        self.episode_hover_steps = 0
        self.episode_step_count = 0

        # Compute initial potential for Condition C
        if self.condition == "corrected":
            self.prev_potential = self.compute_potential(obs)
        else:
            self.prev_potential = None

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step, apply shaping condition, and log metrics."""
        obs, original_reward, terminated, truncated, info = super().step(action)

        self.episode_step_count += 1
        self.episode_original_reward += float(original_reward)

        shaping_reward = 0.0
        is_hovering = False

        if self.condition == "baseline":
            # Condition A: Original Gymnasium reward, zero shaping
            shaping_reward = 0.0

        elif self.condition == "naive":
            # Condition B: Naive persistent per-frame hovering bonus
            is_hovering = self.is_eligible_for_naive_bonus(obs)
            if is_hovering and not (terminated or truncated):
                shaping_reward = self.naive_bonus
                self.episode_hover_steps += 1

        elif self.condition == "corrected":
            # Condition C: Potential-Based Reward Shaping F(s, s') = gamma * Phi(s') - Phi(s)
            current_potential = self.compute_potential(obs)

            if terminated:
                # In episodic formulations with an absorbing terminal state,
                # the true future value at termination is identically 0.
                # Setting Phi(s_terminal) = 0 ensures policy invariance (Ng et al. 1999).
                next_potential = 0.0
            else:
                # During ongoing flight or time-limit truncation (artificial horizon cutoff),
                # use the actual state potential.
                next_potential = current_potential

            if self.prev_potential is not None:
                potential_diff = self.gamma * next_potential - self.prev_potential
                shaping_reward = self.beta * potential_diff
            else:
                shaping_reward = 0.0

            self.prev_potential = current_potential

        # Total shaped reward seen by the RL policy
        total_reward = float(original_reward) + float(shaping_reward)
        self.episode_shaping_reward += float(shaping_reward)

        # Enrich info dictionary with clean separation of components
        info["original_reward"] = float(original_reward)
        info["shaping_reward"] = float(shaping_reward)
        info["cum_original_reward"] = float(self.episode_original_reward)
        info["cum_shaping_reward"] = float(self.episode_shaping_reward)
        info["is_hovering"] = is_hovering
        info["hover_steps"] = self.episode_hover_steps
        info["shaping_condition"] = self.condition

        return obs, total_reward, terminated, truncated, info
