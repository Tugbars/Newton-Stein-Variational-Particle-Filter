# SVPF Optimization Summary

## What Worked

### Antithetic Sampling ✅ (+3% RMSE improvement)
- Pairs particles `(i, i+n/2)` with noise `(+z, -z)` in predict step
- Halves variance of transition distribution expectations
- Zero computational overhead (actually slightly less - half the thread launches)
- **Enabled by default**: `state->use_antithetic = 1`

### Newton Preconditioning ✅ (significant improvement)
- Both regular Newton and Full Newton modes help substantially
- Hessian-weighted Stein operator adapts to local curvature
- Already part of production config

### Heun KSD Bug Fix ✅ (correctness)
- Original kernel computed KSD on `h_orig` (pre-transport particles)
- Fixed: KSD now computed on updated particles via separate `svpf_ksd_kernel`
- Required splitting fused kernel to avoid race conditions

---

## What Didn't Help

### Two-Factor Volatility Model ❌
- **Concept**: Decompose `h = h_fast + h_slow` with different persistence
- **Result**: +21% on spike-heavy scenarios, -12% to -35% elsewhere
- **Why it failed**: From one observation `y_t`, cannot identify two latent states. Likelihood only sees their sum. Coordinate-wise SVGD fights the joint structure.
- **Decision**: Reverted all code

### Heun's Method (2nd Order Integrator) ❌
- **Concept**: Predictor-corrector scheme, average φ₁ and φ₂
- **Result**: 2× gradient evaluations, ~0.001 RMSE difference (negligible)
- **Why it failed**: Stein transport is already stable; higher-order integration doesn't help when the bottleneck is observation noise, not integration error.

### LCV Bandwidth Selection ❌
- **Concept**: Leave-one-out cross-validation to select optimal bandwidth
- **Result**: No improvement over Silverman's rule
- **Why it failed**: LCV optimizes for density estimation, not transport. IMQ kernel is already robust. 1D simplicity makes adaptive bandwidth unnecessary.

### Dual-Bandwidth (Attraction vs Repulsion) ❌
- **Concept**: Wider repulsion bandwidth maintains diversity, tighter attraction tracks precisely
- **Result**: No improvement
- **Why it failed**: 
  1. In 1D with 512 particles, collapse isn't a problem
  2. Creates theoretical inconsistency (KSD measures wrong objective)
  3. Newton preconditioning becomes inconsistent (Hessian applied asymmetrically)
- **Recommendation**: Set `repel_bw_ratio = 1.0` to disable

### More Stein Steps ❌
- **Result**: Diminishing returns after 8-12 steps
- **Why**: Particles converge quickly; extra steps just add compute

---

## Key Insight

The 1F SVPF with existing features (guided prediction, EKF guide, adaptive σ, Student-t likelihood, Newton preconditioning) is already near its **practical accuracy ceiling**.

The bottleneck is **fundamental information limit**: one noisy observation per timestep cannot support more precise inference. Improving the transport mechanism doesn't help when the limiting factor is observation noise.

**What actually helped throughout the project**:
- Guided prediction (APF-style lookahead)
- EKF guide with preserving mode
- Adaptive σ boosting for shocks
- Student-t likelihood (robustness)
- KSD early stopping (efficiency)
- Newton preconditioning (curvature adaptation)

These all work because they help the filter **respond faster to observations**, not because they make Stein transport "better".

---

## Current Production Config

```c
state->use_antithetic = 1;      // ON - variance reduction
state->use_newton = 1;          // ON - curvature adaptation  
state->use_guided = 1;          // ON - APF-style lookahead
state->use_guide = 1;           // ON - EKF guidance
state->repel_bw_ratio = 1.0f;   // OFF - dual bandwidth disabled
state->use_heun = 0;            // OFF - not worth 2× cost
```

## Baseline Performance (512 particles, 7 scenarios)

| Scenario | RMSE | Notes |
|----------|------|-------|
| Calm | 0.35 | Near floor |
| Crisis | 0.62 | High absolute, but relatively good |
| Fat tails | 0.46 | Student-t helps |
| Intermediate | 0.49 | After antithetic |

**0.35 RMSE is approximately the achievable floor** given particle count and single noisy observation per step.
