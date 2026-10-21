# Deep Reinforcement Learning with PPO: LunarLander-v3 Internals

A comprehensive, ground-up guide to understanding Proximal Policy Optimization (PPO), neural actor-critic architectures, generalized advantage estimation, and training diagnostics.

---

## Table of Contents
1. [Core Reinforcement Learning Terminology](#1-core-reinforcement-learning-terminology)
2. [Actor-Critic Architecture in Stable-Baselines3](#2-actor-critic-architecture-in-stable-baselines3)
3. [ASCII Architecture Diagram](#3-ascii-architecture-diagram)
4. [Step-by-Step Numerical Walkthrough (Hypothetical Values)](#4-step-by-step-numerical-walkthrough-hypothetical-values)
5. [How Discrete Actions and State Values are Produced](#5-how-discrete-actions-and-state-values-are-produced)
6. [Why Initial Policies Exhibit Erratic Behavior](#6-why-initial-policies-exhibit-erratic-behavior)
7. [The PPO Training Loop and Hyperparameters](#7-the-ppo-training-loop-and-hyperparameters)
8. [Generalized Advantage Estimation (GAE)](#8-generalized-advantage-estimation-gae)
9. [The Clipped PPO Surrogate Objective](#9-the-clipped-ppo-surrogate-objective)
10. [Critic Loss and Policy Entropy](#10-critic-loss-and-policy-entropy)
11. [Training vs. Inference](#11-training-vs-inference)
12. [Reproducibility and Random Seeds Across Hardware](#12-reproducibility-and-random-seeds-across-hardware)
13. [TensorBoard Diagnostics and Interpretation](#13-tensorboard-diagnostics-and-interpretation)

---

## 1. Core Reinforcement Learning Terminology

Reinforcement Learning (RL) models sequential decision-making under uncertainty, formalised as a Markov Decision Process (MDP): $(S, A, P, R, \gamma)$.

- **Observation / State ($s_t \in \mathcal{S}$):**  
  A quantitative vector describing the lander at timestep $t$. In LunarLander-v3, this is an 8-dimensional continuous vector: $[x, y, v_x, v_y, \theta, \omega, \text{left\_leg}, \text{right\_leg}]$.
- **Action ($a_t \in \mathcal{A}$):**  
  A decision chosen by the agent. For LunarLander-v3, this is a discrete set of 4 choices:
  - $0$: Do nothing (coasting under gravity).
  - $1$: Fire left thruster (rotates lander clockwise / right).
  - $2$: Fire main thruster (provides upward vertical thrust).
  - $3$: Fire right thruster (rotates lander counter-clockwise / left).
- **Instantaneous Reward ($r_t \in \mathbb{R}$):**  
  The scalar feedback emitted by the environment immediately following action $a_t$. Reflects progress towards the landing pad, fuel penalties, or contact bonuses.
- **Episode:**  
  A complete trajectory from initial spawn ($t=0$) until termination (crash or landed) or truncation (time limit of 1,000 steps reached).
- **Policy ($\pi_\theta(a|s)$):**  
  The decision-making function parameterized by neural network weights $\theta$. Given state $s$, it assigns a probability distribution over all possible actions: $\sum_{a \in \mathcal{A}} \pi_\theta(a|s) = 1$.
- **Value Function ($V_\phi(s)$):**  
  The critic function parameterized by weights $\phi$. It estimates the *expected cumulative discounted return* starting from state $s$ under policy $\pi$:
  $$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \;\middle|\; s_t = s \right]$$

---

## 2. Actor-Critic Architecture in Stable-Baselines3

In Stable-Baselines3 (SB3), when configuring `PPO("MlpPolicy", ..., policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))`:

1. **The Actor (Policy Head):**  
   Takes the 8-dimensional observation vector, passes it through two feedforward linear layers of 64 units each with non-linear activation (tanh), and terminates in a linear layer with **4 outputs** (one logit per action).
2. **The Critic (Value Head):**  
   Takes the same 8-dimensional observation vector, passes it through its own path of two feedforward linear layers of 64 units each, and terminates in a linear layer with **1 scalar output** estimating $V(s)$.

In SB3's default discrete PPO architecture, the actor and critic maintain separate hidden feature layers (`pi=[64, 64]` and `vf=[64, 64]`), avoiding interference between policy gradients and value regression gradients.

---

## 3. ASCII Architecture Diagram

```
                 OBSERVATION VECTOR (8 dimensions)
       [ x,  y,  vx,  vy,  angle,  omega,  leg_L,  leg_R ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │  ACTOR (Policy)   │       │  CRITIC (Value)   │
    │  Input Layer: 8   │       │  Input Layer: 8   │
    └─────────┬─────────┘       └─────────┬─────────┘
              │ W_pi_1 [64x8]             │ W_vf_1 [64x8]
              ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │  Hidden Layer 1   │       │  Hidden Layer 1   │
    │  64 units (Tanh)  │       │  64 units (Tanh)  │
    └─────────┬─────────┘       └─────────┬─────────┘
              │ W_pi_2 [64x64]            │ W_vf_2 [64x64]
              ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │  Hidden Layer 2   │       │  Hidden Layer 2   │
    │  64 units (Tanh)  │       │  64 units (Tanh)  │
    └─────────┬─────────┘       └─────────┬─────────┘
              │ W_pi_out [4x64]           │ W_vf_out [1x64]
              ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │  Action Logits    │       │   State Value     │
    │  z = [z0,z1,z2,z3]│       │      V(s)         │
    └─────────┬─────────┘       └───────────────────┘
              │ Softmax
              ▼
    ┌───────────────────┐
    │ Action Probs:     │
    │ [p0, p1, p2, p3]  │
    └───────────────────┘
```

---

## 4. Step-by-Step Numerical Walkthrough (Hypothetical Values)

> **Important:** All numerical quantities in this section are **hypothetical illustrative values** designed to demonstrate the arithmetic operations clearly. They do not represent real model outputs.

### Input Observation:
Suppose at step $t$, the lander is slightly to the right ($x = +0.20$), falling ($y = 0.85$, $v_y = -0.40$), with no leg contact:
$$\mathbf{s} = [0.20, \; 0.85, \; -0.05, \; -0.40, \; 0.02, \; 0.01, \; 0.0, \; 0.0]^T$$

### Layer 1 Computation:
Linear projection followed by hyperbolic tangent activation:
$$\mathbf{h}_1 = \tanh(\mathbf{W}_1 \mathbf{s} + \mathbf{b}_1)$$
where $\mathbf{W}_1 \in \mathbb{R}^{64 \times 8}$ and $\mathbf{b}_1 \in \mathbb{R}^{64}$. Each of the 64 units computes a weighted dot product of the 8 inputs.

### Layer 2 Computation:
$$\mathbf{h}_2 = \tanh(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
where $\mathbf{W}_2 \in \mathbb{R}^{64 \times 64}$.

### Output Heads:

#### 1. Policy Head (Actor):
Produces raw unnormalized logits $\mathbf{z} \in \mathbb{R}^4$:
$$\mathbf{z} = \mathbf{W}_{\pi} \mathbf{h}_2 + \mathbf{b}_{\pi}$$
Suppose hypothetical linear projection yields:
$$\mathbf{z} = [0.45, \; 0.10, \; 1.80, \; 0.15]$$

To convert logits into a valid probability distribution, we apply the **Softmax** function:
$$p_i = \pi(a_i | s) = \frac{e^{z_i}}{\sum_{j=0}^3 e^{z_j}}$$

Calculating exponentials:
- $e^{0.45} \approx 1.568$
- $e^{0.10} \approx 1.105$
- $e^{1.80} \approx 6.050$
- $e^{0.15} \approx 1.162$
- Sum of exponentials: $\sum = 1.568 + 1.105 + 6.050 + 1.162 = 9.885$

Normalizing:
- $p_0 (\text{Do nothing}) = 1.568 / 9.885 \approx \mathbf{0.159}$ ($15.9\%$)
- $p_1 (\text{Fire left}) = 1.105 / 9.885 \approx \mathbf{0.112}$ ($11.2\%$)
- $p_2 (\text{Fire main}) = 6.050 / 9.885 \approx \mathbf{0.612}$ ($61.2\%$)
- $p_3 (\text{Fire right}) = 1.162 / 9.885 \approx \mathbf{0.117}$ ($11.7\%$)

During **training**, the agent samples action $a \sim \text{Categorical}(\mathbf{p})$. Here, action 2 (Fire Main Engine) has the highest probability ($61.2\%$).  
During **deterministic evaluation**, the agent takes the argmax: $\arg\max_i(p_i) = 2$.

#### 2. Value Head (Critic):
Produces a single scalar representing the expected return from this state:
$$V(s) = \mathbf{W}_{vf} \mathbf{h}_{2, vf} + b_{vf}$$
Suppose hypothetical linear projection produces:
$$V(s) = \mathbf{+12.40}$$
This critic prediction asserts that starting from state $s$ under the current policy, the agent expects to accumulate approximately $+12.40$ cumulative reward.

---

## 5. How Discrete Actions and State Values are Produced

- **Action Selection via Categorical Distribution:**  
  Unlike continuous environments where a Gaussian mean and log-std are parameterized, discrete environments map network logits directly to a multinomial (Categorical) probability distribution using Softmax.
- **Critic Value Regression:**  
  The critic is a linear regressor without an activation function on its final layer. This allows it to output arbitrarily large positive numbers (for smooth landings) or negative numbers (for impending crashes).

---

## 6. Why Initial Policies Exhibit Erratic Behavior

Before training begins:
1. Neural weights are initialized using orthogonal random initialization: $\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$.
2. Initial logits for all four actions are near zero, resulting in a near-uniform probability distribution:
   $$\pi(a=0) \approx \pi(a=1) \approx \pi(a=2) \approx \pi(a=3) \approx 0.25$$
3. The lander fires random thrusters at each frame, tumbling and quickly crashing.
4. Learning occurs when positive advantages reinforce thruster sequences that stabilize altitude and angle, while negative advantages suppress actions that cause crashes.

---

## 7. The PPO Training Loop and Hyperparameters

PPO is an **on-policy** actor-critic algorithm that alternates between two phases:
1. **Rollout Collection:** The agent interacts with the environment for $N_{\text{steps}}$ (e.g., 2,048 timesteps), storing tuples $(s_t, a_t, r_t, s_{t+1}, \log \pi_{\text{old}}(a_t|s_t), V_{\text{old}}(s_t))$.
2. **Optimization Epochs:** The stored buffer of 2,048 transitions is shuffled and divided into mini-batches of size `batch_size` (e.g., 64). The network is trained for `n_epochs` (e.g., 10 passes) using the clipped PPO objective with Adam optimizer.

### Hyperparameters Explained:
- `n_steps` ($2048$): Number of interaction steps collected per rollout cycle before updating weights.
- `batch_size` ($64$): Number of samples fed into the neural network per gradient descent mini-batch update ($2048 / 64 = 32$ updates per epoch).
- `n_epochs` ($10$): How many times the optimizer iterates over the collected rollout buffer.
- `gamma` ($\gamma = 0.99$): Discount factor. Prioritizes future rewards ($\gamma \approx 1$ values long-term stability over short-term fuel usage).
- `gae_lambda` ($\lambda = 0.95$): Generalized Advantage Estimation smoothing factor balancing bias vs. variance.
- `learning_rate` ($\alpha = 0.0003$): Step size taken by the Adam optimizer along the gradient direction.
- `clip_range` ($\epsilon = 0.2$): Clipping boundary for policy updates ($[1 - \epsilon, 1 + \epsilon] = [0.8, 1.2]$).
- `ent_coef` ($c_2 = 0.01$): Weight of the policy entropy regularization bonus, incentivizing exploration.
- `vf_coef` ($c_1 = 0.5$): Coefficient scaling the critic's mean squared error loss relative to the policy loss.
- `max_grad_norm` ($0.5$): Maximum allowable gradient norm, preventing destabilizing gradient explosions.

---

## 8. Generalized Advantage Estimation (GAE)

To train the policy, we need to know whether action $a_t$ was better or worse than the average action in state $s_t$.

The 1-step temporal difference (TD) residual is:
$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- If $\delta_t^V > 0$, the transition was better than the critic predicted.
- If $\delta_t^V < 0$, the transition was worse than expected.

GAE aggregates multi-step TD residuals exponentially discounted by $\lambda \in [0, 1]$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

- **When $\lambda = 0$:** $\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ (lowest variance, highest bias).
- **When $\lambda = 1$:** $\hat{A}_t = \sum_{k=0}^\infty \gamma^k r_{t+k} - V(s_t)$ (empirical Monte Carlo return minus baseline; lowest bias, highest variance).
- $\lambda = 0.95$ provides the optimal trade-off for continuous physics environments like LunarLander.

---

## 9. The Clipped PPO Surrogate Objective

Standard policy gradient algorithms compute the policy gradient using the likelihood ratio:
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

If $r_t(\theta)$ is updated without constraints, an aggressive gradient step can drastically change the policy, destroying previously learned behaviors (policy collapse).

PPO prevents this with the **clipped surrogate objective**:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

### Intuition:
- **Case 1: Positive Advantage ($\hat{A}_t > 0$):**  
  The action was favorable. We want to increase $\pi_\theta(a_t|s_t)$, so $r_t(\theta) > 1$. However, the objective clips at $1 + \epsilon = 1.20$. Once the new policy is $20\%$ more likely to select this action, further gradient reward is clipped to zero.
- **Case 2: Negative Advantage ($\hat{A}_t < 0$):**  
  The action was unfavorable. We want to decrease $\pi_\theta(a_t|s_t)$, so $r_t(\theta) < 1$. The objective clips at $1 - \epsilon = 0.80$. The optimizer is prevented from excessively penalizing a policy in a single rollout.

---

## 10. Critic Loss and Policy Entropy

The full objective maximized by PPO combines policy improvement, value function fitting, and entropy regularization:
$$L_t(\theta) = \hat{\mathbb{E}}_t \left[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 \mathcal{S}[\pi_\theta](s_t) \right]$$

### 1. Value Function Loss ($L^{\text{VF}}$):
The critic is trained via mean squared error regression between predicted value and empirical target returns $R_t = \hat{A}_t + V_{\text{old}}(s_t)$:
$$L^{\text{VF}}(\theta) = \left( V_\theta(s_t) - R_t \right)^2$$

### 2. Policy Entropy Bonus ($\mathcal{S}[\pi_\theta]$):
Entropy measures the unpredictability of the action distribution:
$$\mathcal{S}[\pi_\theta](s_t) = -\sum_{a \in \mathcal{A}} \pi_\theta(a | s_t) \log \pi_\theta(a | s_t)$$
- High entropy means exploration (balanced action probabilities).
- Low entropy means exploitation (deterministic decisions).
- Maximizing entropy with coefficient $c_2 = 0.01$ prevents the policy from prematurely converging to suboptimal local minima (e.g., shutting down all engines to avoid fuel penalties).

---

## 11. Training vs. Inference

| Attribute | Training Mode | Inference / Evaluation Mode |
| :--- | :--- | :--- |
| **Action Selection** | Stochastic sampling: $a \sim \text{Categorical}(\mathbf{p})$ | Deterministic: $a = \arg\max_i(p_i)$ |
| **Exploration** | Active (guided by policy entropy) | Disabled (exploits learned knowledge) |
| **Gradient Tracking** | Enabled (`torch.enable_grad()`) | Disabled (`torch.no_grad()`) |
| **Environment Wrapper** | Monitor wrapper with logging | Clean evaluation environment |
| **Performance** | High variance due to random exploration | Consistent, deterministic trajectory |

---

## 12. Reproducibility and Random Seeds Across Hardware

Setting a seed (e.g., `seed = 42`) in Python, NumPy, PyTorch, and Gymnasium standardizes:
1. Initial neural network weight initialization.
2. Initial lander position, velocity, and terrain generation.
3. Random action sampling during rollouts.

### Why Identical Seeds Can Produce Divergent Results Across Machines:
- **Non-associative Floating-Point Arithmetic:** CPU instructions (x86 SSE/AVX vs ARM NEON) and GPU CUDA thread scheduling aggregate floating-point additions in different non-deterministic orders: $(a + b) + c \neq a + (b + c)$.
- **Box2D Physics Engine Solvers:** Gymnasium's Box2D physics solver relies on iterative constraint solvers that are sensitive to small rounding discrepancies. Over a 500-step trajectory, tiny floating-point differences compound exponentially (the "butterfly effect").
- **PyTorch Non-deterministic Kernels:** Certain atomic operations in CUDA/cuDNN sacrifice bitwise determinism for high computational parallelism.

---

## 13. TensorBoard Diagnostics and Interpretation

To launch the interactive TensorBoard dashboard:
```bash
tensorboard --logdir logs
```
Then navigate to `http://localhost:6006` in your browser.

### Key Diagnostics Guide:

| Metric | What It Measures | Healthy Target Trend |
| :--- | :--- | :--- |
| **`rollout/ep_rew_mean`** | Rolling average of cumulative episode return. | Should steadily rise from $\approx -150$ toward $+200$ as learning progresses. |
| **`rollout/ep_len_mean`** | Rolling average episode duration in steps. | Initially short (quick crashes), rises as lander learns to hover, then stabilizes when it lands efficiently. |
| **`train/approx_kl`** | Approximate Kullback-Leibler divergence between old and updated policy. | Should stay small (typically between $0.005$ and $0.02$). Spikes ($> 0.05$) indicate excessively large policy shifts. |
| **`train/clip_fraction`** | Fraction of updates where $r_t(\theta)$ exceeded $[1-\epsilon, 1+\epsilon]$. | Typically between $0.05$ and $0.25$. If it reaches $0.50+$, the policy is changing too abruptly. |
| **`train/entropy_loss`** | Negative entropy of the policy distribution ($-\mathcal{S}$). | Gradually increases from $\approx -1.38$ ($\ln 4$, maximum uniform entropy) toward $\approx -0.2$ as policy becomes decisive. |
| **`train/value_loss`** | Mean squared error of the critic's state value predictions. | Increases initially as returns vary wildly, then steadily decreases as the critic accurately predicts returns. |
| **`train/explained_variance`** | Fraction of return variance explained by critic: $1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$. | Should climb from near $0.0$ toward $0.7 - 0.95$. Negative values indicate critic predictions are worse than guessing the mean. |
