# LunarLander AI: PPO Baseline & Reinforcement Learning Internals

A modular, reproducible, and educational reinforcement learning repository implementing **Phase 1 (Environment Setup & PPO Baseline)** and **Phase 2 (PPO Internals & Diagnostics)** for Gymnasium's `LunarLander-v3`.

---

## 1. Project Objective

The goal of this project is to build a rock-solid, ground-up understanding of deep reinforcement learning using **Proximal Policy Optimization (PPO)**. Rather than treating reinforcement learning as a black box, this codebase explains:
- How physics observations translate through neural network layers into action probabilities and state-value predictions.
- How Generalized Advantage Estimation (GAE) and clipped surrogate objectives stabilize policy updates.
- How to evaluate agents using verified physical landing criteria rather than raw score alone.
- How to inspect training curves and interpret critical training diagnostics.

---

## 2. Core Reinforcement Learning Concepts (Simplified)

```
   ┌────────────────────────────────────────────────────────┐
   │                       ENVIRONMENT                      │
   │                    (LunarLander-v3)                    │
   └───────────────▲────────────────────────┬───────────────┘
                   │ Action (a_t)           │
                   │ [0, 1, 2, 3]           │ Observation (s_{t+1}) [8 dims]
                   │                        │ Instantaneous Reward (r_t)
   ┌───────────────┴────────────────────────▼───────────────┐
   │                          AGENT                         │
   │   Actor (Policy Head)  ──► Selects optimal engine      │
   │   Critic (Value Head)  ──► Predicts expected return    │
   └────────────────────────────────────────────────────────┘
```

- **State / Observation ($s_t$):** An 8-dimensional telemetry snapshot:
  `[x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]`.
- **Action ($a_t$):** Four discrete choices:
  - `0`: Coast (no engines).
  - `1`: Fire left orientation thruster (turns lander right).
  - `2`: Fire main thruster (upward push).
  - `3`: Fire right orientation thruster (turns lander left).
- **Instantaneous Reward ($r_t$):** Immediate scalar feedback (fuel cost, altitude descent toward landing pad, touchdown bonus, or crash penalty).
- **Episode Return ($G$):** Total undiscounted cumulative reward from spawn until crash, landing, or timeout.
- **Actor:** Neural policy $\pi(a|s)$ outputting action probabilities.
- **Critic:** Neural value function $V(s)$ estimating expected future returns.

---

## 3. Project Directory Structure

```
lunarlander-rl/
├── README.md                      # Comprehensive user and architectural guide
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git rules (excludes heavy checkpoints & logs)
├── configs/
│   └── default.yaml               # Centralized hyperparameters & configuration
├── src/
│   ├── __init__.py
│   ├── environment.py             # Env initialization, telemetry & random demo
│   ├── train_baseline.py          # SB3 PPO trainer with resume protection
│   ├── evaluate.py                # Rigorous physical landing evaluation
│   ├── record_video.py            # MP4 and GIF rollout recorder
│   └── analyze_training.py        # Log parser, smoothed plots & trend metrics
├── docs/
│   └── ppo_explained.md           # Deep dive into network math, GAE & diagnostics
├── tests/
│   ├── test_environment.py        # Tests 1-5: Gymnasium lifecycle & shapes
│   ├── test_config.py             # Test 6: Configuration & bounds validation
│   └── test_training_smoke.py     # Tests 7-10: PPO smoke training & inference
├── models/                        # Saved model checkpoints (.zip)
├── logs/                          # Monitor CSVs & TensorBoard event files
└── results/                       # Generated plots, evaluation JSONs & videos
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.12 recommended; wheels are readily available for Box2D and PyTorch).

### Option A: Windows PowerShell Setup
```powershell
# 1. Create a virtual environment using Python 3.12
py -3.12 -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Linux / macOS Setup
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Usage Commands

### 1. Environment Inspection & Random Policy Demo
Inspect state space details and run an unguided demonstration:
```bash
# Headless inspection
python src/environment.py --episodes 1 --seed 42

# Visual GUI inspection (requires local display)
python src/environment.py --render --episodes 2
```

### 2. Train the Baseline PPO Agent
Train the baseline PPO model with two 64-unit hidden layers:
```bash
# Smoke/Baseline training (100,000 timesteps default)
python src/train_baseline.py --timesteps 100000 --seed 42

# Longer training run aimed at solving the environment (500,000 timesteps)
python src/train_baseline.py --timesteps 500000 --seed 42

# Explicitly resume from an existing checkpoint
python src/train_baseline.py --timesteps 100000 --resume_from models/ppo_lunarlander_baseline.zip
```

> **Resume Safety:** By default, running `python src/train_baseline.py` will always initialize a fresh model with random weights. Resuming will **only** occur if the `--resume_from` flag is explicitly provided.

### 3. Evaluate the Trained Model
Evaluate policy performance over multiple episodes with deterministic inference:
```bash
# Headless evaluation (20 episodes default)
python src/evaluate.py --model_path models/ppo_lunarlander_baseline.zip --episodes 20

# Visual evaluation with rendering window
python src/evaluate.py --model_path models/ppo_lunarlander_baseline.zip --render --episodes 5

# Save structured results to JSON
python src/evaluate.py --model_path models/ppo_lunarlander_baseline.zip --output_json results/eval_report.json
```

#### Documented Success Criteria:
A positive cumulative reward alone is **never** treated as proof of a successful landing. The evaluation module requires:
1. `terminated == True` (episode concluded naturally, not by reaching the 1,000-step time limit).
2. `left_leg_contact == 1.0` and `right_leg_contact == 1.0` (both legs physically resting on the terrain).
3. Terminal bonus or score verifying rest position without body impact.

### 4. Record Videos & Animated GIFs
Capture high-definition rollouts of your trained agent:
```bash
# Record MP4 video (default)
python src/record_video.py --model_path models/ppo_lunarlander_baseline.zip --episodes 2

# Record animated GIF for documentation or README embedding
python src/record_video.py --model_path models/ppo_lunarlander_baseline.zip --gif --episodes 1
```
Output media will be saved under `results/`.

### 5. Plot Training Curves & Analyze Logs
Parse Stable-Baselines3 `Monitor` CSV files and generate two-panel smoothed plots:
```bash
python src/analyze_training.py --log_dir logs --results_dir results
```
Generates `results/training_curves.png` showing:
- Raw episode returns and rolling smoothed mean curve.
- Episode lengths over timesteps.
- Statistical trend assessment (`Improving`, `Flat`, or `Inconclusive`) based on documented regression slopes and quartile deltas.

### 6. Interactive TensorBoard Diagnostics
If `tensorboard` is available in your environment:
```bash
tensorboard --logdir logs
```
Open your browser at `http://localhost:6006` to inspect `rollout/ep_rew_mean`, `train/approx_kl`, `train/clip_fraction`, `train/entropy_loss`, and `train/value_loss`.

---

## 6. Running the Test Suite

Run all 10 verified requirements using `pytest`:
```bash
pytest tests/ -v
```
The test suite validates:
1. Environment instantiation and reset.
2. Observation dimensions $(8,)$ and finite value bounds.
3. Action space specification ($\text{Discrete}(4)$).
4. Step transition 5-tuple consistency.
5. Termination (`terminated`) and truncation (`truncated`) handling.
6. Schema and range validation of `configs/default.yaml`.
7. PPO actor-critic network instantiation.
8. Rapid smoke-training cycle and checkpoint serialization.
9. Saved model loading and deterministic/stochastic action inference.
10. Evaluation metrics structure integrity.

---

## 7. Troubleshooting Guide

### 1. `box2d` or `swig` installation errors
- **Issue:** On Windows, installing `box2d-py` or `gymnasium[box2d]` might fail if C++ build tools are missing.
- **Solution:** Use Python 3.12 (or 3.11). Pre-compiled binary wheels are available (`box2d-2.3.10-cp312-win_amd64.whl`). Simply run:
  ```bash
  pip install "gymnasium[box2d]"
  ```

### 2. Missing display / GUI rendering on headless servers
- **Issue:** `pygame.error: No available video device` when passing `--render`.
- **Solution:** Remote environments, CI containers, or WSL without an X-server cannot open a native window. Use the default headless mode for evaluation (`python src/evaluate.py`), or use `python src/record_video.py` which records frames in `rgb_array` mode without opening an OS window.

### 3. Environment ID deprecation warnings
- **Issue:** Gymnasium deprecated `LunarLander-v2`.
- **Solution:** Always use `LunarLander-v3`, which is the current standard in Gymnasium $\ge 1.0.0$.

---

## 8. Expected Artifacts & Interpretation

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `models/ppo_lunarlander_baseline.zip` | Zip archive | PyTorch state dictionary and SB3 algorithm metadata. |
| `models/last_run_meta.json` | JSON | Exact hyperparameters, seed, elapsed wall-clock training time, and timesteps. |
| `logs/monitor.csv` | CSV | SB3 Monitor output with episode return (`r`), length (`l`), and elapsed time (`t`). |
| `results/training_curves.png` | PNG image | High-resolution publication-quality plot of returns and episode durations. |
| `results/eval_report.json` | JSON | Detailed episode-by-episode breakdown of returns, seeds, and landing outcomes. |
| `results/*.mp4` / `*.gif` | Video / GIF | Visual playback of lander behavior during evaluation episodes. |

> **Distinction Between Expectations and Measured Results:**
> An initial baseline trained for 100,000 timesteps will demonstrate hovering and throttle control, but may still crash or drift occasionally (mean return typically between $-80$ and $+80$). Consistently achieving $+200$ points with $>80\%$ landing success typically requires $500,000$ to $1,000,000$ timesteps.
