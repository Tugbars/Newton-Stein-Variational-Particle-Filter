# SVPF Gradient Diagnostic: Findings & Conclusions

## Overview

This document summarizes the results of gradient diagnostics for the self-tuning SVPF parameter learning system. We tested both ν (observation tail thickness) and σ (vol-of-vol) parameters.

**Conclusion: Neither ν nor σ require online learning. The filter works well with fixed, calibrated values.**

---

## ν (Degrees of Freedom) - DO NOT LEARN

### Finding

Equilibrium ν > 50 (nearly Gaussian) on all datasets.

### Why

The SV model explains fat tails through **time-varying volatility**, not fat-tailed observation noise. When the filter tracks vol well:

```
z_t = return_t / vol_estimate ≈ N(0,1)
```

The ν gradient is only informative during rare crash events (z² > 9), which almost never happen when the filter is working correctly.

### Recommendation

Fix ν = 5-10 for mild robustness against outliers. Don't attempt online learning.

---

## σ (Vol-of-Vol) - DO NOT LEARN

### Finding

Equilibrium σ ≈ 0.09-0.10 on all datasets (SPY, 2008 crisis, COVID, TSLA).

| σ_filter | ε²/σ² mean | Direction |
|----------|------------|-----------|
| 0.05 | 0.40 | ↓ decrease |
| 0.08 | 0.78 | ↓ decrease |
| **0.10** | **1.14** | ≈ equilibrium |
| 0.15 | 2.33 | ↑ increase |

### Why

1. **Breathing already handles dynamics** - The filter scales σ_eff = σ_base × boost during crises
2. **Equilibrium is stable** - Same value across all datasets
3. **Complexity not justified** - Online learning adds risk for minimal benefit

### Recommendation

Set default σ = 0.10 (calibrated value). Let breathing handle crisis scaling.

```c
state->sigma_z_effective = 0.10f;  // Calibrated via gradient diagnostic
```

---

## Summary Table

| Parameter | Worth Learning? | Equilibrium | Action |
|-----------|-----------------|-------------|--------|
| ν | No | > 50 (Gaussian) | Fix at 5-10 |
| σ | No | 0.09-0.10 | Fix at 0.10, breathing handles dynamics |
| μ | Maybe | Data-dependent | Auto-calibrate or adaptive Kalman |
| ρ | Possibly | ~0.98 | Could vary across regimes |

---

## Key Insight

**The filter is working very well.** Both gradient diagnostics revealed that:
- Fat tails are explained by volatility clustering, not observation noise (ν)
- Vol-of-vol is stable across market conditions (σ)

This is the SV model succeeding, not failing. The gradients told us the correct values; they don't need to be continuously learned.

---

## Files

| File | Purpose |
|------|---------|
| `src/svpf_gradient_diagnostic.cu` | ν and σ gradient kernels |
| `test/test_svpf_gradient_dgp.cu` | Synthetic DGP verification |
| `test/test_svpf_gradient_market.cu` | Real market data (ν) |
| `test/test_svpf_gradient_sigma.cu` | Real market data (σ) |
| `scripts/fetch_market_data.py` | Download market data |
