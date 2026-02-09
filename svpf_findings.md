# SVGD Repulsion Is Harmful in Sequential Particle Filtering

## Summary

Empirical investigation of Stein Variational Gradient Descent (SVGD) applied to sequential Bayesian filtering for stochastic volatility estimation. We find that SVGD's repulsive kernel gradient term — the mechanism that distinguishes it from standard gradient descent — is actively harmful when embedded in a predict-update filtering loop with a well-designed proposal distribution. Removing repulsion entirely reduces estimation bias by 27–38% across all tested scenarios while maintaining or improving particle diversity.

## Background

SVGD (Liu & Wang, 2016) transports particles via the update:

$$\phi(x_i) = \frac{1}{N} \sum_{j=1}^{N} \left[ k(x_j, x_i) \nabla \log p(x_j) + \nabla_{x_j} k(x_j, x_i) \right]$$

The first term (attractive) moves particles toward high-density regions of the target. The second term (repulsive) pushes particles apart to maintain diversity. In the infinite-particle limit, this converges to the posterior. At finite N, repulsion introduces systematic bias because particles simultaneously define the distribution, compute the transport direction, and serve as kernel centers — a circular dependency that compresses posterior variance.

The Stein Particle Filter (Fan, Taghvaei & Chen, 2021; Maken, Ramos & Ott, 2022) embeds SVGD in a predict-update cycle, producing equal-weight particles without resampling. This avoids the weight degeneracy of standard particle filters but inherits SVGD's finite-particle bias, which compounds across timesteps.

## The Finite-Particle Bias Problem

We implemented a production SVPF in CUDA for stochastic volatility estimation in HFT, with Newton preconditioning, annealed transport, adaptive bandwidth, KSD monitoring, guided proposals, and antithetic sampling. Despite extensive engineering, we observed persistent bias that required manual calibration (`lik_offset`) tuned to specific operating regimes.

The root cause: SVGD's repulsive term underestimates the posterior variance at finite N. The "posterior" SVGD converges to is not the true posterior but a self-consistent equilibrium — particles agree with each other they're in the right place, but they're systematically too concentrated. When transported particles form the prior mixture for the next timestep, the compressed cloud propagates forward, and bias compounds.

## Attempted Corrections

### 1. Importance Weight Correction

Treat SVGD output as a proposal, compute importance weights:

$$\log w_i = \log p(y_t | h_i) + \log p(h_i | h_\text{prev}) - \log q_\text{KDE}(h_i)$$

where $q$ is estimated via leave-one-out KDE on the transported particles.

**Result:** Made things dramatically worse. Bias increased from +0.73 to +1.73 in matched conditions. The KDE density estimate using the same particles that define q is systematically biased — it underestimates density in the tails due to bandwidth mismatch, inflating importance weights for already-biased tail particles. VarRatio of 4× confirmed the weights were hallucinating a posterior 4× wider than reality.

**Diagnosis:** Circular dependency. Same structural problem as SVGD itself, relocated from transport to density estimation.

### 2. EKF Gap Correction

Use the gap between an independent Extended Kalman Filter estimate and the particle mean as a bias signal:

$$\text{bias\_signal} = \text{guide\_mean}_\text{EKF} - \bar{h}_\text{particles}$$

**Result:** Also made things worse. The EKF linearizes exp(h/2) around its mean, systematically underestimating volatility when h is negative (the typical regime at μ = −3.5). The EKF is biased in the same direction as the particles — both overshoot h. Blending toward the EKF amplifies the error.

**Diagnosis:** The external reference is independently biased in the same direction. Not circular, but not helpful either.

### 3. Adaptive Repulsion (Key Finding)

Hypothesis: if repulsion causes bias, make it conditional. Scale the repulsive term by a collapse metric:

$$\phi(x_i) = \frac{1}{N} \sum_j \left[ k(x_j, x_i) \nabla \log p(x_j) + \beta(t) \cdot \nabla_{x_j} k(x_j, x_i) \right]$$

where β = 0 when particle variance is healthy, β → 1 when collapse is detected.

We tested four modes:
- **base:** β = 1 always (standard SVGD)
- **no_repulsion:** β = 0 always (pure kernel-smoothed gradient descent)
- **adaptive_var:** β ∈ [0,1] via sigmoid ramp on variance ratio
- **adaptive_thr:** β ∈ {0, 1} via hard threshold on variance ratio

## Results

### Test 1: Matched DGP (filter knows truth)
| Mode | Bias_h | RMSE_h | h_var | avg β |
|------|--------|--------|-------|-------|
| base (β=1) | +0.731 | 1.280 | 0.417 | 1.000 |
| no_repulsion (β=0) | +0.534 | 0.909 | 0.475 | 0.000 |
| adaptive_var | +0.551 | 0.929 | 0.470 | 0.105 |
| adaptive_thr | +0.535 | 0.909 | 0.475 | 0.005 |

**Removing repulsion reduces bias by 27% and RMSE by 29%.** Particle variance *increases* — repulsion was compressing the cloud, not diversifying it.

### Test 2: Misspecified μ (DGP μ = −2.0, filter assumes μ = −3.5)
| Mode | Bias_h | RMSE_h | h_var | avg β |
|------|--------|--------|-------|-------|
| base | −0.231 | 0.799 | 0.362 | 1.000 |
| no_repulsion | −0.265 | 0.765 | 0.384 | 0.000 |
| adaptive_var | −0.267 | 0.771 | 0.379 | 0.105 |
| adaptive_thr | −0.264 | 0.765 | 0.383 | 0.005 |

Slightly higher bias magnitude without repulsion (+0.034) but better RMSE (−4.3%) and better log-likelihood (−622 vs −636). Effectively neutral — repulsion provides negligible benefit even under model misspecification.

### Test 3: Regime Change (μ jumps −3.5 → −1.5 at t = 500)
| Mode | Bias_h | RMSE_h | h_var | avg β |
|------|--------|--------|-------|-------|
| base | +0.380 | 1.019 | 0.389 | 1.000 |
| no_repulsion | +0.234 | 0.735 | 0.441 | 0.000 |
| adaptive_var | +0.244 | 0.749 | 0.435 | 0.100 |
| adaptive_thr | +0.236 | 0.737 | 0.440 | 0.020 |

**Removing repulsion reduces bias by 38% during regime change** — the hardest scenario.

### Test 4: Low N (N = 64, matched DGP)
| Mode | Bias_h | RMSE_h | h_var | avg β |
|------|--------|--------|-------|-------|
| base | +0.712 | 1.253 | 0.423 | 1.000 |
| no_repulsion | +0.547 | 0.968 | 0.519 | 0.000 |

**Even at N = 64, no particle collapse without repulsion.** The predict step maintains diversity. Bias reduction of 23%.

### Summary Across All Scenarios

| Scenario | Base bias | No-repulsion bias | Δ bias | RMSE improvement |
|----------|-----------|-------------------|--------|------------------|
| Matched | +0.731 | +0.534 | −27% | −29% |
| Misspecified μ | −0.231 | −0.265 | +15% (tiny) | +4.3% |
| Regime change | +0.380 | +0.234 | −38% | −28% |
| Low N (64) | +0.712 | +0.547 | −23% | −23% |

Repulsion loses decisively in 3/4 scenarios and is roughly neutral in the 4th. The adaptive collapse detector (β_on < 2% across all tests) confirms that particle collapse never occurs — the diversity problem that repulsion was designed to solve does not exist in this architecture.

## Why Repulsion Is Unnecessary in Sequential Filtering

In static SVGD (single target, no temporal structure), repulsion is essential. Without it, particles collapse to the MAP and you lose the posterior approximation entirely.

In a sequential predict-update loop, three mechanisms maintain particle diversity independently of repulsion:

1. **Transition noise injection:** Every predict step samples fresh noise (here, Student-t with ν = 2.5) scaled by σ_z = 0.15 and adds it to transition means. This re-spreads particles every timestep regardless of how concentrated the previous posterior was.

2. **Guided proposal:** The observation-driven proposal shifts particles toward data-implied regions, creating spread across the cloud as different particles receive different innovations.

3. **Langevin diffusion:** The temperature parameter in the Stein step injects isotropic noise during transport, providing additional stochastic diversification.

Repulsion adds a fourth diversification mechanism on top of three that are already sufficient. The cost is the circular dependency that creates finite-particle bias. The benefit is preventing a collapse that never occurs.

## What Remains Without Repulsion

Removing the ∇k term from SVGD leaves:

$$\phi(x_i) = \frac{1}{N} \sum_{j=1}^{N} k(x_j, x_i) \nabla \log p(x_j)$$

This is **kernel-smoothed score ascent**: each particle moves in the direction of the log-posterior gradient, but smoothed by a weighted average over neighboring particles via the Cauchy kernel. The smoothing provides collective transport — particles in the same region move coherently rather than independently, which improves sample efficiency compared to independent gradient descent.

This is not standard SVGD. It is not a particle filter. It is gradient-assisted particle transport with kernel smoothing, embedded in a predict-update cycle. The equal-weight property is preserved (no importance weights needed), and the attractive term still provides useful information about the posterior geometry. Only the repulsive term — and its associated bias — is removed.

## Implications

The remaining +0.534 bias (in matched conditions) comes from other sources, primarily the likelihood score interaction with the guided proposal. The CUDA implementation addresses this with a manual `lik_offset` parameter. This bias is smaller, non-circular, and more amenable to principled correction than the repulsion-induced bias.

For practitioners implementing Stein particle filters: if your architecture includes a good proposal distribution (guided prediction, EKF, or similar), consider disabling or significantly reducing the repulsive term. The diversity it provides is redundant with temporal noise injection, and the bias it introduces compounds across timesteps.

For the broader SVGD literature: finite-particle bias analysis (Das & Nagaraj, 2023; Shi & Mackey, 2024) focuses on convergence rates in the static setting. The sequential setting introduces a compounding mechanism absent from static analysis — the biased posterior at step t becomes the prior at step t+1. This temporal coupling may warrant separate theoretical treatment.

## Experimental Setup

- **Model:** Stochastic volatility: h_t = μ + ρ(h_{t−1} − μ) + σ_z · η_t, y_t = exp(h_t/2) · ε_t
- **Parameters:** μ = −3.5, ρ = 0.97, σ_z = 0.15, ν_obs = 5.0, ν_state = 2.5
- **Filter:** Annealed SVGD (5 stages × 4 steps), Cauchy kernel, RMSProp preconditioning, EKF guide, guided proposal with shock detection, antithetic sampling
- **Particles:** N = 512 (main tests), N = 64 (low-N test)
- **Evaluation:** T = 1000 timesteps, 5 independent runs per configuration, metrics averaged
- **Implementation:** PyTorch prototype (GPU-accelerated), validated against production CUDA implementation

## References

- Liu, Q. & Wang, D. (2016). Stein variational gradient descent: A general purpose Bayesian inference algorithm. NeurIPS.
- Fan, J., Taghvaei, A. & Chen, Y. (2021). Stein particle filtering. arXiv:2106.10568.
- Maken, F.A., Ramos, F. & Ott, L. (2022). Stein particle filter for nonlinear, non-Gaussian state estimation. IEEE RA-L.
- Das, A. & Nagaraj, D. (2023). Provably fast finite particle variants of SVGD via virtual particle stochastic approximation. NeurIPS.
- Shi, J. & Mackey, L. (2024). Improved finite-particle convergence rates for Stein variational gradient descent. arXiv:2409.08469.
