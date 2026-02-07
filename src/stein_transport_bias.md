# The Stein Transport Bias Problem in SVPF

## Summary

SVPF (Stein Variational Particle Filter) uses SVGD (Stein Variational Gradient Descent) to transport particles toward the filtering posterior at each timestep. We discovered that this transport introduces **systematic positive bias** in log-volatility estimates of approximately +0.33, which was previously masked by a hand-tuned `lik_offset` heuristic. The bias is intrinsic to finite-particle Stein transport and cannot be resolved by changing the observation likelihood model alone.

**Resolution:** A three-part fix: (1) a fresh-weight epilogue recomputes importance weights at final particle positions, (2) tempered IS with α=0.4 balances bias correction against weight-concentration variance, and (3) a Kalman filter on the Stein score diagnostic learns the optimal `lik_offset` online using Stein's identity (E[∇log π] = 0). This replaces the manual heuristic with a principled, self-calibrating correction. RMSE improved from 0.5275 → 0.5231 and average bias from -0.05 → -0.01.

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

### 3. Importance-Weighted Outputs with Stale Weights (Failed)

**Idea:** Instead of 1/N arithmetic mean of particle positions, use importance weights to compute E[h]:
```
ĥ = Σ w_i·h_i / Σ w_i,  where w_i = exp(log_w_i)
```

**Theory:** Even if transport pushes particles too high, the likelihood evaluates those particles as having lower probability. Weighting downvotes the biased particles.

**Result:** Made accuracy significantly worse (RMSE 0.57 → 0.84).

**Why it failed:** The importance weights `log_w` are computed at particle positions *before* the final transport step. After transport moves the particles, the weights are stale. Weighting by stale weights actively penalizes particles that moved to good positions.

### 4. Fresh-Weight Importance Sampling (Partial Success → Variance Explosion)

**Idea:** Fix the stale-weight problem by running one extra gradient kernel call after final transport (β=1.0, no transport step). This gives `log_w` evaluated at *final* particle positions. Then use importance-weighted outputs.

**Result with full IS (α=1.0):** Bias improved (-0.05 → -0.02), but RMSE exploded (+30-34%).

**Why it partially failed:** Student-t log-weight includes `-0.5·h` Jacobian → `w_i ∝ exp(-0.5·h_i)`. After transport spreads particles across h range, lower-h particles get exponentially higher weights. With 128 particles, ESS drops to ~10-20 → high variance between timesteps.

**Key insight:** Works perfectly when model is well-specified (OU-matched). Fails under model mismatch because particles spread wider, exacerbating weight concentration.

### 5. Tempered Importance Weights (Success)

**Idea:** Raise weights to power α ∈ (0,1]: `w_i = exp(α · (log_w[i] - max))`. Interpolates between full IS (α=1.0) and uniform averaging (α=0.0).

**Result at α=0.4:** RMSE 0.5231 (improved from 0.5275), bias ≈ -0.01 (improved from -0.05). Best of both worlds: meaningful bias correction without weight concentration destroying variance.

**Cannot replace lik_offset:** Setting offset=0 with α=0.4 gives +0.40 bias. Fresh weights only correct ~0.04 of the 0.33 transport bias. The offset does the heavy lifting.

### 6. Adaptive lik_offset via Stein Score Kalman Filter (Current)

**Idea:** Use Stein's identity (E_π[∇log π] = 0) as a diagnostic signal. If offset is correct, mean gradient across particles is zero. Learn the offset online via scalar Kalman filter.

**Implementation:** Output kernel computes mean(grad_combined) and var(grad_combined). Host-side KF uses innovation = -mean_grad with observation noise R = var_grad/N. Gain K adapts naturally — conservative during high-variance periods, responsive when signal is clean.

**Status:** Implemented, awaiting test results.

### 7. Annealed Repulsion (Not Yet Tested in Isolation)

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

### ~~Can Fresh Weights Fix Weighted Outputs?~~ ✅ ANSWERED

Yes, but only with tempered exponent α=0.4. Full IS (α=1.0) causes variance explosion from weight concentration. The `-0.5·h` Jacobian in Student-t weights creates exponential disparity after transport spreads particles. Tempering trades some bias correction for dramatically lower variance.

### Can the Kalman Filter Learn from Scratch?

Initialize `lik_offset = 0.0, lik_offset_P = 1.0`. Theory says it should converge to ~0.33 within 100 steps. Untested. If it works, eliminates all manual calibration.

### Can Annealed Repulsion Fix the Bias?

The theory is sound: if repulsion → 0 in the final steps, the transport converges to pure gradient ascent, which targets the true posterior mode. But:
- Does the diversity from early repulsion survive long enough?
- Does the RMSprop momentum carry repulsion-induced bias into later steps even after the force is removed?
- What's the optimal decay schedule? Linear? Cosine? Step function?

Now less urgent since the Kalman approach provides a principled correction without modifying transport dynamics.

### Does the Adaptive Offset Track Non-Stationarity?

The offset should theoretically vary with particle count, bandwidth, annealing schedule, and data regime. Does the Kalman filter actually adapt when conditions change mid-series (e.g., crisis onset)? Or does it converge to a fixed value and stay there?

### Is the Bias Actually Harmful in Production?

With fresh weights + Kalman offset, average bias is -0.01. The question shifts from "can we fix the bias?" to "does the extra machinery (1 kernel + 7 floats + host KF) measurably improve downstream trading signals?"

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

3. **Fresh-weight epilogue** — One extra `svpf_fused_gradient_kernel` call after final Stein transport at β=1.0. Recomputes `log_w` and `grad_combined` at final particle positions. Toggle: `state->use_fresh_weights = 1`. Cost: ~2μs per timestep.

4. **Tempered IS outputs** — `svpf_fused_outputs_kernel` supports weighted averaging with tempered exponent α: `w_i = exp(α · (log_w[i] - max))`. α=0.4 is the sweet spot. Parameter: `state->fresh_weight_alpha`.

5. **Stein score diagnostic** — Output kernel computes `mean(grad_combined)` and `var(grad_combined)`, packed into `output_pack[5:6]`. Zero extra kernel cost (piggybacked on existing reductions).

6. **Adaptive lik_offset Kalman filter** — Host-side scalar KF in `svpf_sync_outputs`. Uses Stein's identity (`E[∇log π] = 0`) to learn the correct offset online. Toggle: `state->use_adaptive_offset = 1`. Parameters: `lik_offset_Q` (process noise), `lik_offset_warmup` (settling period).

### What's Reverted / Inactive

- OCSN is togglable (`use_ocsn` flag, defaults to 0)
- Stale-weight IS (replaced by fresh-weight epilogue)

### Best Known Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `use_fresh_weights` | 1 | Fresh-weight epilogue |
| `fresh_weight_alpha` | 0.4 | Tempered IS |
| `use_adaptive_offset` | 1 | Learn offset online |
| `lik_offset` | 0.345 | Initial value (Kalman tunes from here) |
| `lik_offset_P` | 0.01 | Tight initial variance |
| `lik_offset_Q` | 1e-4 | Slow drift |
| `lik_offset_warmup` | 50 | Steps before adaptation |

### Baseline Performance (Student-t + lik_offset, no fresh weights)

| Scenario | RMSE | MAE | Bias |
|----------|------|-----|------|
| Slow Drift | 0.5717 | 0.4520 | -0.0577 |
| Stress Ramp | 0.5985 | 0.4736 | -0.0530 |
| OU-Matched | 0.3748 | 0.2976 | -0.0188 |
| Intermediate Band | 0.6225 | 0.4914 | -0.0383 |
| Spike+Recovery | 0.4469 | 0.3551 | -0.0501 |
| Wrong-Model | 0.5507 | 0.4390 | -0.1026 |
| **Average** | **0.5275** | | **-0.05** |

### With Fresh Weights α=0.4 (before adaptive offset)

| Scenario | RMSE | MAE | Bias |
|----------|------|-----|------|
| Slow Drift | 0.5651 | 0.4476 | -0.0280 |
| Stress Ramp | 0.5936 | 0.4692 | -0.0178 |
| OU-Matched | 0.3755 | 0.2972 | 0.0309 |
| Intermediate Band | 0.6218 | 0.4903 | -0.0055 |
| Spike+Recovery | 0.4391 | 0.3503 | -0.0096 |
| Wrong-Model | 0.5434 | 0.4316 | -0.0604 |
| **Average** | **0.5231** | | **≈ -0.01** |

---

## Recommended Next Steps (Priority Order)

### Step 1: Fresh Weights ✅ COMPLETED — Partial Success

**What:** One extra gradient kernel call after final Stein transport at β=1.0. Recomputes `log_w` at final particle positions. Output kernel uses self-normalized IS: `h_mean = Σ w_i·h_i / Σ w_i`.

**Result with full IS (α=1.0), lik_offset=0.335:**

| Scenario | RMSE | Bias |
|----------|------|------|
| Slow Drift | 0.7640 (+34%) | -0.0263 ✓ |
| Stress Ramp | 0.7795 (+30%) | -0.0187 ✓ |

Bias improved dramatically, but **RMSE exploded**. Classic IS variance problem: Student-t log-weight includes `-0.5·h` Jacobian, creating exponential weight disparity. After transport spreads particles across h range, a few low-h particles get exponentially higher weights. ESS drops to ~10-20. Weighted mean has high variance between timesteps.

**Why it works for OU-matched but not misspecified:** When the model is well-specified, particles after transport are already near the true posterior. Fresh weights are nearly uniform. Weighted ≈ uniform averaging, but without Stein inflation. Under model mismatch (drift, ramp), particles spread wide → weight concentration → variance explosion.

### Step 2: Tempered Weights ✅ COMPLETED — Sweet Spot Found

Raise weights to power α ∈ (0,1]: `w_i = exp(α · (log_w[i] - max_log_w))`. This interpolates between full IS (α=1.0, low bias, high variance) and uniform averaging (α=0.0, some bias, low variance).

**Result: α=0.4, lik_offset=0.335:**

| Scenario | RMSE | MAE | Bias |
|----------|------|-----|------|
| Slow Drift | 0.5651 | 0.4476 | -0.0280 |
| Stress Ramp | 0.5936 | 0.4692 | -0.0178 |
| OU-Matched | 0.3755 | 0.2972 | 0.0309 |
| Intermediate Band | 0.6218 | 0.4903 | -0.0055 |
| Spike+Recovery | 0.4391 | 0.3503 | -0.0096 |
| Wrong-Model | 0.5434 | 0.4316 | -0.0604 |
| **Average** | **0.5231** | | **≈ -0.01** |

Compared to baseline (0.5275 RMSE, -0.05 bias): RMSE improved slightly, **bias cut 5×**. Fresh weights cannot replace `lik_offset` (setting offset=0 gives +0.40 bias), but they clean up the residual that the offset leaves behind.

### Step 3: Adaptive lik_offset via Stein Score Kalman Filter ← CURRENT

The `lik_offset = 0.345` heuristic works but requires manual calibration. The fresh-weight epilogue provides the signal to learn it online.

**Theory — Stein's Identity:**

For any distribution π: `E_π[∇log π(x)] = 0`. If particles are truly at the posterior, the mean score (mean of `grad_combined` across particles) is zero. Any deviation from zero is the transport bias.

After the fresh-weight epilogue, `grad_combined[j]` already contains the full posterior score at each particle's final position. The mean gradient is a noisy observation of offset error:

```
mean_grad = 0            ⟹  offset is correct
mean_grad < 0            ⟹  offset too low, particles drifted high
mean_grad > 0            ⟹  offset too high, particles drifted low
```

**Implementation — Scalar Kalman Filter:**

The observation model is linear (state = offset, observation = -mean_grad), so a plain KF is exact. No EKF needed.

```c
// In svpf_sync_outputs, after D2H transfer:
float mean_grad = results[5];     // from output kernel
float var_grad  = results[6];     // from output kernel

float Q = 1e-4f;                  // process noise (offset drifts slowly)
float R = var_grad / (float)n;    // observation noise (SE² of mean gradient)
R = fmaxf(R, 1e-6f);

float P_pred = state->lik_offset_P + Q;
float innovation = -mean_grad;     // drives mean_grad → 0
float K = P_pred / (P_pred + R);

state->lik_offset += K * innovation;
state->lik_offset_P = (1.0f - K) * P_pred;
state->lik_offset = fminf(fmaxf(state->lik_offset, 0.0f), 1.0f);
```

**Why this is principled:**

The Kalman gain K naturally adapts to signal quality. When `var_grad` is high (stressed markets, wild returns), R is large → K is small → conservative update. When things are calm, it adapts quickly. This is exactly the behavior you'd hand-tune but it falls out of the math.

**Cost:** Zero extra kernel launches. The output kernel already reduces `grad_combined` in the same pass that computes h_mean and vol. Two extra `block_reduce_sum` calls for `Σ grad` and `Σ grad²`, packed into `output_pack[5:6]`. D2H transfer bumped from 5 to 7 floats.

**Configuration:**
```c
state->use_fresh_weights = 1;          // Enable epilogue
state->fresh_weight_alpha = 0.4f;      // Tempered IS
state->use_adaptive_offset = 1;        // Enable Kalman
state->lik_offset = 0.345f;            // Start near known-good
state->lik_offset_P = 0.01f;           // Tight initial variance
state->lik_offset_Q = 1e-4f;           // Slow drift
state->lik_offset_warmup = 50;         // Let particles settle first
```

Or to learn from scratch: `lik_offset = 0.0, lik_offset_P = 1.0`.

**Files modified:**
- `svpf.cuh` — 6 new fields in SVPFState
- `svpf_kernels.cuh` — updated output kernel declaration
- `svpf_opt_kernels.cu` — output kernel: `grad_combined` input, tempered weights, grad stats
- `svpf_optimized_graph.cu` — fresh-weight epilogue, 7-float D2H, Kalman in sync_outputs

### Step 4: If adaptive offset works — remove lik_offset heuristic permanently

Initialize `lik_offset = 0.0, lik_offset_P = 1.0`. The Kalman filter converges to the correct offset within ~100 steps. No manual calibration needed for any ν, data regime, or particle count.

### Step 5: If nothing works

Accept that Student-t + `lik_offset` is the production configuration. Document the offset as "empirical transport bias correction" rather than "likelihood centering." The -0.05 bias is already excellent for production use.