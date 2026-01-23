# SVPF Adaptive Methods: Implementation & Results

## Overview

This document describes the adaptive methods implemented in the Stein Variational Particle Filter (SVPF) for stochastic volatility tracking, along with measured accuracy improvements on a standardized benchmark.

**Baseline:** SVPF with fixed parameters  
**Benchmark:** 6 synthetic scenarios from HCRBPF DGP (Slow Drift, Stress Ramp, OU-Matched, Intermediate Band, Spike+Recovery, Wrong-Model)  
**Metric:** RMSE of log-volatility estimate vs true log-volatility

---

## Results Summary

| Method | RMSE | Δ from Baseline | Cumulative Δ |
|--------|------|-----------------|--------------|
| Baseline (fixed params) | 0.6910 | — | — |
| + Adaptive μ | 0.6217 | -10.0% | -10.0% |
| + Adaptive Guide | 0.6074 | -2.3% | -12.1% |
| + lik_offset Correction | 0.5625 | -7.4% | -18.6% |
| + nu=30 + Adaptive σ_z | 0.5538 | -1.5% | -19.9% |
| + IMQ Kernel | **0.5507** | -0.6% | **-20.3%** |

**Final bias:** Near-zero (range: -0.07 to +0.07)

---

## Method 1: Adaptive μ (Mean Level Tracking)

### Problem
The SV model assumes a fixed mean level μ for log-volatility:
```
h_t = μ + ρ(h_{t-1} - μ) + σ_z·ε_t
```
But real markets have regime-dependent mean levels (μ ∈ [-4.5, -1.0]).

### Solution: 1D Kalman Filter on μ

**Key Insight:** Use the filter's own confidence (particle spread) to gate learning.

```
Signal:      h_mean (particle posterior mean, already filtered)
Noise:       R = scale × bandwidth² (particle uncertainty)
```

- **Calm market (low bandwidth):** High confidence → fast adaptation
- **Crisis (high bandwidth):** Low confidence → freeze μ

### Kalman Equations
```
Predict:  μ_pred = μ,  P_pred = P + Q
Update:   K = P_pred / (P_pred + R)
          μ = μ_pred + K × (h_mean - μ_pred)
          P = (1 - K) × P_pred
```

### Configuration
```cpp
filter->use_adaptive_mu = 1;
filter->mu_process_var = 0.001f;    // Q: how fast μ can drift
filter->mu_obs_var_scale = 11.5f;   // R = scale × bandwidth²
filter->mu_min = -4.0f;
filter->mu_max = -1.0f;
```

### Impact
| Scenario | Before | After | Δ |
|----------|--------|-------|---|
| Slow Drift | 0.7792 | 0.6735 | -14% |
| Stress Ramp | 0.8628 | 0.6965 | -19% |
| Intermediate Band | 0.8505 | 0.7217 | -15% |

**Why it works:** Drift/ramp scenarios have shifting mean levels. Fixed μ causes persistent bias. Adaptive μ tracks the drift.

---

## Method 2: Adaptive Guide Strength (Innovation-Gated Nudging)

### Problem
When the model is "surprised" (large innovation), particles lag behind reality. The EKF guide knows where to go (it "cheats" by using y_t directly), but base guide_strength is too weak to overcome prior inertia.

### Solution: Asymmetric Boost on Upward Surprises

**Key Insight:** Only boost when implied volatility > estimated volatility.

- Upward surprise (price crash): **Information** → boost guide
- Downward surprise (calm when vol expected high): **Noise** → ignore

### Implementation
```cpp
// Compute on CPU before graph launch (zero latency)
float vol_est = state->vol_prev;  // Use previous step's estimate
float return_z = fabsf(y_t) / vol_est;
float implied_h = logf(y_t * y_t + 1e-8f) + 1.27f;
float h_est = logf(vol_est * vol_est + 1e-8f);
float h_innovation = implied_h - h_est;

// ASYMMETRIC: only boost on upward vol surprises
if (h_innovation > 0.0f && return_z > threshold) {
    float severity = fminf((return_z - threshold) / 3.0f, 1.0f);
    guide_strength = base + (max - base) * severity;
}
```

### Configuration
```cpp
filter->use_adaptive_guide = 1;
filter->guide_strength_base = 0.05f;       // When model fits
filter->guide_strength_max = 0.30f;        // During surprises
filter->guide_innovation_threshold = 1.0f; // Z-score to start boost
```

### Impact
| Scenario | Before | After | Δ |
|----------|--------|-------|---|
| Spike+Recovery | 0.5341 | 0.5213 | -2.4% |
| All scenarios | — | — | ~-2.3% avg |

**Why it works:** Spikes need instant response. Boosting guide strength "teleports" particles to the EKF estimate in 1 step instead of 5.

---

## Method 3: Likelihood Offset Correction

### Problem
The likelihood gradient was using the wrong center point:

```cpp
// OLD (biased)
float grad_lik = (log_y2 - h_j + 1.0f/nu) / R_noise;  // offset ≈ 0.2
```

For Student-t(ν=5), `1/ν = 0.2`. But the true expected value relationship is:
```
E[log(y²) | h] = h + offset
```
Where offset depends on the observation distribution. The approximation was ~1.0 too low, causing systematic underestimation (bias ≈ -0.3).

### Solution: Tunable Likelihood Offset

```cpp
// NEW (configurable)
float grad_lik = (log_y2 - h_j + lik_offset) / R_noise;
```

### Calibration
| lik_offset | Resulting Bias |
|------------|----------------|
| 0.20 (1/ν) | -0.30 |
| 0.70 | **~0.00** |
| 1.27 (Gaussian) | +0.30 |

**Sweet spot:** `lik_offset = 0.70`

### Configuration
```cpp
filter->lik_offset = 0.70f;  // Tuned for minimal bias
```

### Impact
| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Average RMSE | 0.6074 | 0.5625 | -7.4% |
| Average Bias | -0.25 | ~0.00 | **Eliminated** |

**Why it works:** The gradient is the "steering wheel" for particles. Wrong offset = particles consistently pulled in wrong direction. Correct offset = unbiased tracking.

---

## Method 4: Distribution Fix (nu = 30)

### Problem
The DGP generates Gaussian returns, but the filter assumed fat-tailed Student-t(ν=5).

- **Gaussian model:** A 3σ move screams "Volatility is up!"
- **Student-t(5) model:** A 3σ move shrugs "Eh, just an outlier."

This mismatch causes the filter to underreact to genuine vol changes.

### Solution: Quasi-Gaussian Observation Model

Set ν = 30 (close to Gaussian but retains slight robustness):

```cpp
float nu = 30.0f;  // Was 5.0f
```

### Why 30?
- ν = 5: Fat tails, tolerates outliers, underreacts to vol changes
- ν = 30: Nearly Gaussian, sensitive to shocks, proper tracking
- ν = ∞: Pure Gaussian (no robustness to real-world anomalies)

---

## Method 5: Adaptive σ_z (Breathing Filter)

### Problem
The true vol-of-vol (σ_z) varies by regime:
- Calm market: σ_z ≈ 0.08
- Stress ramp: σ_z ≈ 0.42

A fixed σ_z = 0.15 is too rigid — particles can't spread fast enough during stress.

### Solution: Innovation-Gated Vol-of-Vol

**Key Insight:** When innovation is consistently high, the filter is too rigid. Boost σ_z to let particles explore.

```cpp
float vol_est = fmaxf(state->vol_prev, 1e-4f);
float return_z = fabsf(y_t) / vol_est;

// Boost sigma_z when innovation exceeds threshold
float sigma_boost = 1.0f;
if (return_z > threshold) {
    float severity = fminf((return_z - threshold) / 3.0f, 1.0f);
    sigma_boost = 1.0f + (max_boost - 1.0f) * severity;
}

effective_sigma_z = params->sigma_z * sigma_boost;
```

### Configuration
```cpp
filter->use_adaptive_sigma = 1;
filter->sigma_boost_threshold = 1.0f;  // Start boosting when |z| > 1
filter->sigma_boost_max = 3.0f;        // Max 3x boost
```

### Behavior
| Market State | return_z | σ_z Boost | Effective σ_z |
|--------------|----------|-----------|---------------|
| Calm | < 1.0 | 1.0x | 0.15 |
| Moderate surprise | 2.0 | 1.67x | 0.25 |
| Large shock | 4.0+ | 3.0x | 0.45 |

### Impact (Combined with nu=30)
| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Average RMSE | 0.5625 | 0.5538 | -1.5% |

**Why it works:** Matches the HCRBPF DGP behavior where σ(z) varies with regime. Particles can now "breathe" — spreading during stress, tightening during calm.

---

## Method 6: IMQ Kernel (Inverse Multiquadric)

### Problem
The Gaussian RBF kernel decays exponentially:
```
K_gaussian = exp(-r²)
```

When a particle explores the high-vol region during a crash (3-5σ from the group), the repulsive force drops to nearly zero. The particle gets "stranded" and stops contributing useful gradient information.

### Solution: Inverse Multiquadric Kernel

IMQ has polynomial (heavy-tailed) decay:
```
K_imq = 1 / (1 + r²)
```

This ensures particles "talk" to each other even across large distances.

### Repulsion Comparison

| Distance | Gaussian | IMQ |
|----------|----------|-----|
| 1σ | 0.61 | 0.50 |
| 2σ | 0.14 | 0.20 |
| 3σ | 0.01 | **0.10** |
| 5σ | 0.00 | **0.04** |

At 3σ distance, IMQ maintains 10x more repulsion than Gaussian.

### Implementation

```cpp
// OLD: Gaussian RBF
float K = __expf(-diff * diff * inv_2bw_sq);
gk_sum -= K * diff * inv_bw_sq;

// NEW: IMQ (Rational Quadratic)
float dist_sq = diff * diff * inv_bw_sq;
float base = 1.0f + dist_sq;
float K = 1.0f / base;
float K_sq = K * K;

k_sum += K * sh_grad[j];
gk_sum -= 2.0f * diff * inv_bw_sq * K_sq;  // Gradient: dK/dx = -2x/(1+x²)²
```

### Impact

| Scenario | Before | After | Δ |
|----------|--------|-------|---|
| Slow Drift | 0.5987 | 0.5942 | -0.8% |
| Stress Ramp | 0.6268 | 0.6233 | -0.6% |
| OU-Matched | 0.4126 | 0.4105 | -0.5% |
| Intermediate Band | 0.6459 | 0.6390 | **-1.1%** |
| Spike+Recovery | 0.4773 | 0.4762 | -0.2% |
| Wrong-Model | 0.5615 | 0.5611 | -0.1% |
| **Average** | 0.5538 | **0.5507** | **-0.6%** |

**Why it works:** Intermediate Band scenario improved most — that's where particles were getting stranded exploring regime boundaries. IMQ keeps them connected to the group.

---

## Methods That Did NOT Work

### Elastic ρ (Surprise-Based Amnesia)
**Idea:** Drop ρ during shocks to allow faster mean reversion.

**Result:** No improvement. The asymmetric rho (rho_up/rho_down) already handles directional persistence. Additional elasticity either hurt drift/ramp scenarios or had no effect.

**Lesson:** Don't over-engineer. Simple asymmetric ρ is sufficient.

---

## Why Adaptive Methods Work So Well with SVPF

### The Key Difference: Particles Move

**Bootstrap PF:**
- Particles are points with weights
- Can only die and resample
- Need pre-positioned filters at all regimes

**SVPF:**
- Particles move along gradients
- Stein transport: `φ(x) = E[K∇log p + ∇K]`
- Single adaptive filter can "teleport" to any regime

### Gradients as Communication Channel

Every adaptive parameter directly modulates the gradient:

| Parameter | Gradient Effect | Particle Response |
|-----------|-----------------|-------------------|
| ↑ σ_z | Prior weakens | Spread out, explore |
| ↓ σ_z | Prior strengthens | Concentrate |
| Shift μ | Prior mean shifts | Drift toward new μ |
| ↑ guide_strength | Likelihood pull ↑ | Snap to observation |
| Fix lik_offset | Correct gradient direction | Unbiased flow |
| ↑ nu | Sharper likelihood | More sensitive to shocks |
| IMQ kernel | Polynomial repulsion decay | Particles stay connected at distance |

When you change a parameter, you're not hoping particles randomly land correctly — you're **telling them where to go**.

---

## Final Configuration

```cpp
// Student-t degrees of freedom (quasi-Gaussian)
float nu = 30.0f;

// Stein kernel: IMQ (Inverse Multiquadric) - hardcoded in svpf_opt_kernels.cu
// K = 1/(1 + r²) instead of exp(-r²)
// Provides polynomial decay for better particle communication at distance

// Asymmetric persistence (vol spikes fast, decays slow)
filter->use_asymmetric_rho = 1;
filter->rho_up = 0.99f;
filter->rho_down = 0.92f;

// Guided Prediction
filter->use_guided = 1;
filter->guided_alpha_base = 0.0f;
filter->guided_alpha_shock = 0.5f;
filter->guided_innovation_threshold = 1.5f;

// EKF Guide density
filter->use_guide = 1;
filter->use_guide_preserving = 1;
filter->guide_strength = 0.05f;

// Adaptive μ (Kalman filter on mean level)
filter->use_adaptive_mu = 1;
filter->mu_process_var = 0.001f;
filter->mu_obs_var_scale = 11.5f;
filter->mu_min = -4.0f;
filter->mu_max = -1.0f;

// Adaptive Guide Strength (asymmetric innovation gating)
filter->use_adaptive_guide = 1;
filter->guide_strength_base = 0.05f;
filter->guide_strength_max = 0.30f;
filter->guide_innovation_threshold = 1.0f;

// Likelihood offset (bias correction)
filter->lik_offset = 0.70f;

// Adaptive σ_z (breathing filter)
filter->use_adaptive_sigma = 1;
filter->sigma_boost_threshold = 1.0f;
filter->sigma_boost_max = 3.0f;
```

---

## Final Results by Scenario

```
Scenario                   RMSE        MAE       Bias
────────────────────────────────────────────────────
Slow Drift               0.5942     0.4726    -0.0209
Stress Ramp              0.6233     0.4966    -0.0136
OU-Matched               0.4105     0.3245     0.0742
Intermediate Band        0.6390     0.5058    -0.0232
Spike+Recovery           0.4762     0.3791     0.0179
Wrong-Model              0.5611     0.4477    -0.0665
────────────────────────────────────────────────────
AVERAGE RMSE             0.5507
```

---

## Backlog (Future Optimization)

### Quick Wins (Fine-tuning)
1. **Fine-tune lik_offset** in 0.65-0.75 range
2. **Fine-tune nu** in 20-50 range
3. **Fine-tune sigma_boost_max** in 2.0-4.0 range

### Production Refactoring
4. **Zero-Copy Oracle Params for σ_z** - Currently adaptive σ_z triggers graph recapture when it drifts > 0.05. For HFT, refactor kernels to read σ_z from device pointer (like we did for guide_strength) to eliminate recapture latency. Makes sense, try later when integrating with PMMH Oracle.

5. **Guided Variance Inflation ("Searchlight")** - When adaptive guide activates during a spike, inflate proposal variance proportional to guide strength. Wide beam when moving fast, narrow beam when stationary. Makes sense, try later.

### Architecture
6. **PMMH Oracle Integration** - Learn nu, rho_up, rho_down, sigma_z, mu, lik_offset via parallel PMMH (~100μs per iteration). See SVPF_PMMH_ORACLE.md.

---

## References

- Liu & Wang (2016): "Stein Variational Gradient Descent"
- Fan et al. (2021): "Stein Particle Filtering" (arXiv:2106.10568)
- Internal: SVPF_2D_PARAM_LEARNING_POSTMORTEM.md (failed 2D approach)
- Internal: SVPF_PMMH_ORACLE.md (parameter learning architecture)