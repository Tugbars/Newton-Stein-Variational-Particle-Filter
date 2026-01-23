# SVPF 2D Parameter Learning: Post-Mortem

## Overview

**Date:** January 2025  
**Objective:** Online learning of σ_z (vol-of-vol) via 2D state augmentation  
**Result:** Failed - accuracy degraded significantly  
**Decision:** Removed from codebase, reverted to fixed-parameter 1D SVPF

---

## The Idea

### Motivation
The standard SVPF assumes fixed model parameters (ρ, σ_z, μ, γ). In reality, σ_z (volatility-of-volatility) varies over market regimes:
- Calm markets: σ_z ≈ 0.10
- Stress periods: σ_z ≈ 0.30+

We attempted to learn σ_z online by augmenting the state space from 1D to 2D:
- **h**: log-volatility (as before)
- **λ = log(σ_z)**: log vol-of-vol (new)

### Theoretical Approach
Extend SVGD to jointly track (h, λ) with:

1. **2D Predict**: Jitter λ with small random walk, use particle-specific σ_z = exp(λ_i)
2. **2D Gradient**: Compute ∇_h and ∇_λ of log-posterior
3. **2D Stein**: Product kernel K(h,λ) = K_h × K_λ with separate bandwidths
4. **2D Transport**: Move particles in (h, λ) space

The gradient w.r.t. λ comes from the transition density:
```
∂/∂λ log p(h_t | h_{t-1}, λ) = ∂/∂λ [-λ - ε²/(2e^{2λ})]
                              = -1 + ε²/σ²
```
Where ε = h_t - μ - ρ(h_{t-1} - μ) is the innovation.

---

## Implementation

### Files Created
- `svpf_2d_kernels.cu`: 5 fused kernels (~500 lines)
  - `svpf_predict_2d_kernel`
  - `svpf_fused_gradient_2d_kernel`
  - `svpf_fused_stein_transport_2d_kernel`
  - `svpf_fused_bandwidth_2d_kernel`
  - `svpf_fused_outputs_2d_kernel`

### Key Design Decisions
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| λ = log(σ_z) | Unconstrained | Avoids σ > 0 constraint |
| λ clamp | [-4.6, -0.69] | σ ∈ [0.01, 0.5] |
| λ jitter | 0.01-0.08 | Random walk exploration |
| λ step scale | 0.01-0.5× h | Slower parameter learning |
| λ prior std | 0.2-0.5 | Regularization strength |
| Product kernel | K_h × K_λ | Separable bandwidths |

---

## Results

### Baseline (1D, fixed σ_z = 0.15)
```
Scenario                   RMSE
Slow Drift               0.7792
Stress Ramp              0.8628
OU-Matched               0.4159
Intermediate Band        0.8505
Spike+Recovery           0.5095
Wrong-Model              0.7281
AVERAGE                  0.6910
```

### With 2D Parameter Learning
```
Scenario                   RMSE      Delta
Slow Drift               0.9874     +27%
Stress Ramp              1.0986     +27%
OU-Matched               0.4688     +13%
Intermediate Band        1.0997     +29%
Spike+Recovery           0.6142     +21%
Wrong-Model              0.9356     +28%
AVERAGE                  0.8674     +25%
```

**Consistent negative bias increase** across all scenarios.

---

## Failure Analysis

### The Fundamental Problem

SVGD is designed to maintain **particle diversity** through kernel repulsion. But for parameter learning, we need particles to **converge** to a single value.

| Dimension | Goal | SVGD Behavior | Conflict? |
|-----------|------|---------------|-----------|
| h (state) | Diversity | Repulsion spreads particles | ✓ Good |
| λ (param) | Consensus | Repulsion spreads particles | ✗ Bad |

The repulsive term pushes λ particles apart, preventing them from agreeing on the true σ_z.

### The Feedback Loop (Supervisor's Analysis)

1. **λ drifts high** → σ_z increases
2. **Transition prior flattens** → p(h_t | h_{t-1}) becomes uninformative
3. **Likelihood dominates** → Filter trusts observations too much
4. **On calm days** → Low y_t pulls h down aggressively
5. **Result** → Systematic underestimation (negative bias)

### Tuning Attempts

We tried multiple fixes per supervisor recommendations:

| Fix | Effect |
|-----|--------|
| Sign correction (+ for repulsion) | Made it worse initially |
| Tighter λ clamp (σ_max = 0.5) | Minor improvement |
| Slower step scale (0.01×) | Still degraded |
| Tighter prior (std = 0.2) | Still degraded |
| Less jitter (0.01) | Still degraded |

None of these addressed the fundamental problem.

---

## Why SVGD Doesn't Work for Joint State-Parameter Inference

### Mathematical Insight

SVGD approximates the posterior with particles that minimize KL divergence:
```
min KL(q || p) where q = (1/N) Σ δ(x - x_i)
```

For **state inference** (h), we want q to match the full posterior distribution → diversity is correct.

For **parameter inference** (λ), the posterior concentrates as T → ∞ → particles should collapse to a point mass.

SVGD's repulsive kernel **prevents** this collapse, keeping particles spread even when they should converge.

### What Would Work Instead

1. **Sufficient Statistics (Storvik Filter)**
   - Track E[λ] and Var[λ] analytically
   - Already implemented in HCRBPF, works well

2. **SMC² (Nested Particle Filters)**
   - Outer particles for parameters
   - Inner particles for states
   - Resample parameters based on marginal likelihood

3. **Point Estimate + Gradient Descent**
   - Treat σ_z as optimization variable, not random
   - Update via gradient of marginal likelihood
   - No diversity needed

4. **Innovation-Based Estimator**
   - σ̂_z² = EMA of (h_t - ρh_{t-1} - μ(1-ρ))²
   - Simple, no particles needed
   - Already have infrastructure for this

---

## Lessons Learned

1. **Not all inference problems suit SVGD**: State inference (tracking distributions) ≠ parameter inference (finding point estimates)

2. **Diversity is not always good**: SVGD's repulsion is a feature for distributions, a bug for parameters

3. **Joint inference is hard**: State and parameters have different posterior geometries

4. **Simple methods often win**: The 1D fixed-parameter SVPF with well-tuned σ_z outperforms adaptive approaches

5. **Test incrementally**: Should have tested 2D gradient alone, then 2D Stein alone, to isolate failure mode

---

## Code Removed

```
svpf_2d_kernels.cu          (deleted)
SVPFState.d_lambda          (removed)
SVPFState.d_lambda_prev     (removed)  
SVPFState.d_grad_lambda     (removed)
SVPFState.d_v_lambda        (removed)
SVPFState.use_param_learning (removed)
SVPFState.lambda_*          (removed)
SVPFState.sigma_z_estimate  (removed)
SVPFOptimizedState.d_bw_lambda* (removed)
SVPFOptimizedState.d_sigma_mean (removed)
svpf_graph_capture_2d_internal() (removed)
All 2D kernel declarations   (removed)
```

---

## Future Directions

If online σ_z learning is needed:

1. **Innovation variance estimator** (simplest)
   ```cpp
   float innovation = h_mean - (mu + rho * (h_prev_mean - mu));
   sigma_z_ema = alpha * innovation² + (1-alpha) * sigma_z_ema;
   sigma_z_estimate = sqrt(sigma_z_ema);
   ```

2. **Marginal likelihood optimization** (more principled)
   - Compute ∂loglik/∂σ_z via finite differences
   - Update σ_z with gradient descent
   - Invalidate graph on parameter change

3. **Regime detection + lookup table** (practical)
   - Detect high/low vol regimes from return variance
   - Use pre-calibrated σ_z for each regime
   - Already have return EMA/variance tracking

---

## References

- Liu & Wang (2016): "Stein Variational Gradient Descent"
- Fan et al. (2021): "Stein Particle Filtering" (arXiv:2106.10568)
- Storvik (2002): "Particle filters for state-space models with the presence of unknown static parameters"
- Chopin et al. (2013): "SMC²: An efficient algorithm for sequential analysis of state space models"
