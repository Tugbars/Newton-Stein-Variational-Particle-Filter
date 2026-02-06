# The Stein Transport Bias Problem in SVPF

## Summary

SVPF (Stein Variational Particle Filter) uses SVGD (Stein Variational Gradient Descent) to transport particles toward the filtering posterior at each timestep. We discovered that this transport introduces **systematic positive bias** in log-volatility estimates of approximately +0.33, which was previously masked by a hand-tuned `lik_offset` heuristic. The bias is intrinsic to finite-particle Stein transport and cannot be resolved by changing the observation likelihood model alone.

---

## How the Problem Was Discovered

### The OCSN Experiment

We replaced the Student-t observation likelihood with the OCSN (Omori-Chib-Shephard-Nakajima) 10+1 component Gaussian mixture — the standard approximation to log-χ²(1) used in stochastic volatility literature. OCSN had shown 5% improvement in our HCRBPF (Hierarchical Conditionally Rao-Blackwellized Particle Filter), so we expected similar gains in SVPF.

The motivation was to eliminate the `lik_offset` calibration heuristic. The Student-t likelihood gradient has a nonzero mean at the true parameter value:

```
∇_h log p(y|h) = -1/2 + (1/2)(ν+1)·y²·exp(-h) / (ν + y²·exp(-h))
```

At h = true value, E[∇] ≈ -0.27 (not zero). The `lik_offset` was tuned to cancel this, but required manual calibration per (ν, data regime).

OCSN operates in log-squared space where the observation model is linear: `y* = log(y²) = h + d`, where `d ~ mixture`. The mixture mean is exactly known (-1.2704), so no offset heuristic should be needed.

### What Actually Happened

| Metric | Student-t (with lik_offset) | OCSN (no offset) |
|--------|----------------------------|-------------------|
| avg RMSE | **0.5275** | 0.6433 |
| avg Bias | **-0.05** | +0.33 |
| avg var_ll | 20.4 | **1.19** |
| avg stages | 4.5 | 4.0 |

OCSN performed worse across every scenario, with a persistent +0.33 bias.

### The Diagnostic Insight

The key diagnostic was `var_ll` — the variance of log-importance-weights across particles. This dropped **17×** from 20.4 (Student-t) to 1.19 (OCSN).

**Why:** Student-t weights include `-0.5·h` from the Jacobian of the return-space density `exp(-h/2)`. This creates strong h-dependent weight contrast — particles at different h values get very different weights. OCSN operates in log-squared space where the observation is *linear* in h (`y* = h + d`), producing a much flatter weight landscape.

Low `var_ll` means:
- Resampling is nearly uniform → bad particles aren't culled
- The gradient signal from the mixture components is weaker (linear vs exponential)
- The same number of Stein transport steps can't fully converge

### The Real Discovery

Adding `lik_offset` back to OCSN partially fixed the bias — proving that the offset wasn't compensating for a likelihood modeling error. **It was compensating for Stein transport bias.** The Student-t likelihood's strong weight contrast happened to mask this through aggressive resampling.

---

## Root Cause Analysis

### What SVGD Actually Optimizes

SVGD with N particles minimizes KL(q_N || p) where q_N is the empirical particle distribution. But with finite particles, the repulsive kernel term `∇_x k(x_i, x_j)` creates a **puffier** distribution than the true posterior — it pushes particles apart beyond what the target density warrants.

In 1D (our case — scalar log-volatility h), this manifests as:
- Systematic overestimation of posterior variance
- Directional bias when the posterior is asymmetric (which it always is for SV models due to the exponential transform)

### Why It's Always Positive

The SV posterior for h is left-skewed (heavier left tail from the exp(h) likelihood term). The repulsive force pushes the particle cloud's center of mass toward the heavier tail — but since we're in log-space and the likelihood pulls harder from above (large returns → large h), the net effect is a **positive bias**.

### Why lik_offset "Works"

The offset subtracts ~0.27 from the likelihood gradient at every particle. This is equivalent to shifting the effective target distribution downward by a constant. It accidentally cancels the transport bias because:
1. Transport pushes particles ~+0.33 too high
2. Offset pulls the gradient target ~-0.27 down
3. Net bias ≈ -0.05 (close to zero)

This is a coincidence of the Student-t parameterization, not a principled correction.

---

## Approaches Tried

### 1. OCSN Likelihood (Failed)

**Idea:** Replace Student-t with OCSN mixture in log-squared space. Eliminate lik_offset.

**Result:** Exposed the transport bias (+0.33) because OCSN lacks the strong weight contrast that masked it.

**Lesson:** The observation model isn't the problem. Any likelihood that produces weaker weight contrast will expose the bias.

### 2. Contaminated Normal in Return Space (Not Tested, Rejected)

**Idea:** `p(y|h) = (1-α)·N(0, exp(h)) + α·N(0, κ²·exp(h))` — keeps the strong exp(h) nonlinearity while adding outlier robustness.

**Rejected because:** It would re-mask the bias (same mechanism as Student-t) rather than fix it. The fundamental transport problem remains.

### 3. Importance-Weighted Outputs (Failed)

**Idea:** Instead of 1/N arithmetic mean of particle positions, use importance weights to compute E[h]:
```
ĥ = Σ w_i·h_i / Σ w_i,  where w_i = exp(log_w_i)
```

**Theory:** Even if transport pushes particles too high, the likelihood evaluates those particles as having lower probability. Weighting downvotes the biased particles.

**Result:** Made accuracy significantly worse (RMSE 0.57 → 0.84).

**Why it failed:** The importance weights `log_w` are computed at particle positions *before* the final transport step. After transport moves the particles, the weights are stale. Weighting by stale weights actively penalizes particles that moved to good positions.

**Possible fix:** Recompute weights after final transport (extra gradient kernel call with no transport step). Not yet tested.

### 4. Annealed Repulsion (Not Yet Tested in Isolation)

**Idea:** Scale the repulsive kernel term by a factor that decays from 1.0 → 0.0 over Stein iterations:
```
repulsion_scale = 1.0 - current_iter / (total_iters - 1)
```

**Theory:** Early steps use full SVGD (exploration + diversity). Late steps transition to pure gradient ascent (unbiased convergence to posterior mode). Particles end up where the likelihood+prior dictates, not where the repulsive kernel pushes them.

**Status:** Implemented but tested simultaneously with stale weighted outputs, which corrupted results. Needs isolated testing.

---

## Architecture Comparison: Why OCSN Works in HCRBPF but Not SVPF

| Aspect | HCRBPF | SVPF |
|--------|--------|------|
| **State** | h/2 (half log-vol) | h (full log-vol) |
| **Update mechanism** | Kalman filter per mixture component | Stein transport on full particle cloud |
| **OCSN benefit** | Exact posterior per component | Gradient field for transport |
| **Weight contrast** | Not needed (exact updates) | Critical for resampling |
| **Bias source** | None (Kalman is unbiased) | Repulsive kernel + finite particles |

HCRBPF gets exact Rao-Blackwellized posteriors per OCSN component. The mixture structure is exploited optimally. SVPF only uses the mixture to compute a gradient — it never "sees inside" individual components. The gradient from 10 components pulling in different directions at crossover regions is actually noisier than a single Student-t gradient.

---

## Open Questions

### Can Annealed Repulsion Fix the Bias?

The theory is sound: if repulsion → 0 in the final steps, the transport converges to pure gradient ascent, which targets the true posterior mode. But:
- Does the diversity from early repulsion survive long enough?
- Does the RMSprop momentum carry repulsion-induced bias into later steps even after the force is removed?
- What's the optimal decay schedule? Linear? Cosine? Step function?

### Can Fresh Weights Fix Weighted Outputs?

The stale-weight problem has a clean solution: run one additional gradient kernel call after the final transport step, with no transport. This gives `log_w` evaluated at the *final* particle positions. Then importance-weighted outputs should work correctly.

Cost: one extra kernel launch per timestep. For 128 particles, this is ~2μs — negligible.

### Is the Bias Actually Harmful in Production?

The Student-t + lik_offset combination produces bias of only -0.05, which is excellent for practical purposes. The "fix" is already in place, even if it's not principled. The question is whether pursuing a principled solution is worth the complexity.

### Should SVPF Use a Different Transport Entirely?

- **Stein variational Newton** (SVN): Uses full Hessian information, may converge with less bias
- **Wasserstein gradient flow**: Different kernel, different bias properties
- **Amortized SVGD**: Learn the transport map, potentially debiased
- **No transport — pure SIS/SIR**: Let importance sampling + resampling do all the work. Use the EKF guide as the proposal. This is essentially what HCRBPF does.

---

## Current State of the Code

### What's Implemented

1. **OCSN likelihood** — `svpf_ocsn.cuh` (header), `svpf_ocsn_defs.cu` (constants), `ocsn_likelihood()` device function. Toggle: `filter->use_ocsn = 1`. Works but exposes transport bias.

2. **Annealed repulsion** — `repulsion_scale` parameter in both transport kernels, linear decay 1.0 → 0.0 over all Stein iterations. Always active in current code.

3. **Importance-weighted outputs** — Reverted back to 1/N averaging. The weighted version exists in git history but needs fresh-weight recomputation to work.

### What's Reverted / Inactive

- Weighted outputs (stale weights made it worse)
- OCSN is togglable (`use_ocsn` flag, defaults to 0)

### Baseline Performance (Student-t + lik_offset, no OCSN)

| Scenario | RMSE | MAE | Bias |
|----------|------|-----|------|
| Slow Drift | 0.5717 | 0.4520 | -0.0577 |
| Stress Ramp | 0.5985 | 0.4736 | -0.0530 |
| OU-Matched | 0.3748 | 0.2976 | -0.0188 |
| Intermediate Band | 0.6225 | 0.4914 | -0.0383 |
| Spike+Recovery | 0.4469 | 0.3551 | -0.0501 |
| Wrong-Model | 0.5507 | 0.4390 | -0.1026 |
| **Average** | **0.5275** | | |

---

## Recommended Next Steps (Priority Order)

1. **Test annealed repulsion in isolation** — Revert OCSN (`use_ocsn=0`), keep Student-t + lik_offset, run benchmark. Does annealed repulsion improve or degrade? If it helps, the lik_offset value might need re-tuning since the transport dynamics changed.

2. **Test fresh-weight outputs** — Add one extra gradient-only kernel call after final transport. Then enable importance-weighted averaging. This is the principled version of the supervisor's insight.

3. **If both work** — Try removing lik_offset entirely. With annealed repulsion (fixing the source) + fresh-weight outputs (fixing the estimate), the offset might become unnecessary.

4. **If nothing works** — Accept that Student-t + lik_offset is the production configuration. Document the offset as "empirical transport bias correction" rather than "likelihood centering." The -0.05 bias is already excellent.
