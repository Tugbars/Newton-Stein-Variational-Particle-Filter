# SVPF ν Gradient Diagnostic: Findings & Conclusions

## Overview

This document summarizes the results of Step 0 gradient diagnostic for the self-tuning SVPF parameter learning system. The goal was to verify whether the ν (degrees of freedom) parameter could be learned online via gradient descent.

**Conclusion: ν learning is not practical for a well-tuned SV filter. The gradient is mathematically correct but rarely informative.**

---

## Experimental Setup

### Tests Performed

1. **Synthetic DGP** - HCRBPF-style data generating process with:
   - Time-varying parameters: θ(z), μ(z), σ(z)
   - True observation noise: Student-t with ν = 5
   - Intentional model mismatch (SVPF uses fixed params)

2. **Real Market Data** - Historical equity returns:
   - SPY 2007-2024 (4,528 observations)
   - SPY 2008 Financial Crisis (375 obs)
   - SPY 2020 COVID Crash (123 obs)
   - TSLA high-volatility (2,514 obs)

### Filter Configuration

- Particles: 4,096
- Stein steps: 5
- μ: Auto-calibrated via moment matching
- ρ: 0.97-0.98 (fixed)
- σ_z: 0.15 (fixed)

### Gradient Formula

The ν gradient for Student-t observation likelihood:

```
∂/∂ν log p(y|h,ν) = 0.5 * [ψ((ν+1)/2) - ψ(ν/2) - 1/ν - log(1 + z²/ν) + (ν+1)z²/(ν²(1 + z²/ν))]
```

where z = y / exp(h/2) is the standardized residual.

---

## Results

### Synthetic DGP (True ν = 5)

| ν_filter | Mean Gradient | z²_mean | Crashes (z²>9) | Direction |
|----------|---------------|---------|----------------|-----------|
| 3.0 | +0.024 | 1.27 | 17 | ↑ increase |
| 5.0 | +0.008 | 1.09 | 1 | ↑ increase |
| 10.0 | +0.001 | 0.97 | 0 | ≈ equilibrium |
| 30.0 | +0.000 | 0.90 | 0 | ≈ equilibrium |

**Equilibrium ν > 30** (gradient never crosses zero)

### Real Market Data (SPY Full)

| ν_filter | Mean Gradient | z²_mean | Crashes (z²>9) | Direction |
|----------|---------------|---------|----------------|-----------|
| 3.0 | +0.024 | 1.28 | 14 | ↑ increase |
| 5.0 | +0.008 | 1.11 | 0 | ↑ increase |
| 10.0 | +0.001 | 0.99 | 0 | ↑ increase |
| 50.0 | +0.000 | 0.90 | 0 | ≈ equilibrium |

**Equilibrium ν > 50** (essentially Gaussian)

### Crash Gradient Behavior

When crashes ARE detected (z² > 9):
- Gradient is **negative** (correct direction: wants lower ν)
- Magnitude is large: -0.13 to -0.19

But crashes are rarely detected because the filter adapts vol fast enough.

---

## Analysis

### Why Equilibrium ν >> True ν

The SV model explains fat tails through **time-varying volatility**, not fat-tailed observation noise.

```
Raw returns:           Fat-tailed (kurtosis > 3)
Standardized residuals: z = return / vol_estimate ≈ Gaussian
```

When the filter tracks vol well:
1. Large returns coincide with large vol estimates
2. Standardized residuals z remain small
3. No "crashes" (z² > 9) detected
4. ν gradient gets no informative signal
5. Equilibrium ν → ∞ (Gaussian)

This is the SV model **working correctly**.

### Model Mismatch Doesn't Help

Even with intentional model mismatch (DGP has time-varying params, filter assumes fixed):
- RMSE(h) ≈ 0.8-0.9 (significant tracking error)
- But equilibrium ν still > 30

The vol tracking is "good enough" to absorb apparent fat tails.

### When ν Learning Would Help

ν < ∞ matters for:
1. **Jumps** - Discrete events not explained by vol (earnings surprises, flash crashes)
2. **Microstructure noise** - Bid-ask bounce, tick size effects
3. **Cold start** - Before filter has adapted to data scale
4. **Severe model mismatch** - If filter fundamentally can't track vol

For typical equity index filtering with a well-tuned filter, none of these apply strongly enough to make ν learning worthwhile.

---

## Recommendations

### For Production

**Fix ν = 5-10** for mild robustness against outliers. Don't attempt online learning.

```c
// Reasonable default - provides outlier robustness without affecting normal operation
const float nu = 7.0f;
```

### For Self-Tuning SVPF

Pivot to learning parameters with identifiable signal:

| Parameter | Affects | Signal Availability |
|-----------|---------|---------------------|
| ν | Observation tails | Rare (only during crashes) |
| **σ_z** | Vol-of-vol | **Always** (every timestep) |
| **ρ** | Vol persistence | **Always** (every timestep) |
| μ | Vol level | Moderate (long-term drift) |

**σ_z is the best candidate for online learning:**
- Gradient available at every timestep
- Directly controls how fast vol can change
- Misspecification causes persistent tracking error

---

## Next Steps

### Step 1: σ Gradient Diagnostic

Implement and verify σ_z gradient:

```
∂/∂σ log p(h_t | h_{t-1}, θ) = (h_t - μ - ρ(h_{t-1} - μ))² / σ³ - 1/σ
```

Expected behavior:
- σ too high → gradient negative (reduce σ)
- σ too low → gradient positive (increase σ)
- Signal available **every timestep**, not just during crashes

### Step 2: σ Learning with Breathing

Combine learned σ with existing breathing mechanism:
```
σ_effective = σ_learned × breathing_factor(z_t)
```

### Step 3: ρ Gradient (Optional)

If σ learning works well, consider ρ:
- More sensitive to misspecification
- May require careful regularization

---

## Files

| File | Purpose |
|------|---------|
| `src/svpf_gradient_diagnostic.cu` | ν gradient kernel implementation |
| `test/test_svpf_gradient_dgp.cu` | Synthetic DGP verification |
| `test/test_svpf_gradient_market.cu` | Real market data tests |
| `include/market_data_loader.h` | Binary/CSV data loader |
| `scripts/fetch_market_data.py` | Download market data from yfinance |

---

## References

- Kim, Shephard & Chib (1998) - Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models
- Jacquier, Polson & Rossi (2004) - Bayesian Analysis of Stochastic Volatility Models with Fat-Tails and Correlated Errors
