# SVPF: Stein Variational Particle Filter for Stochastic Volatility

A high-performance CUDA implementation of the Stein Variational Particle Filter for real-time volatility tracking.

[![CUDA](https://img.shields.io/badge/CUDA-12+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---
<img width="4170" height="2370" alt="529603395-23dc2d9c-f92a-4b3c-8fe0-73b165dde93d" src="https://github.com/user-attachments/assets/3490999d-1356-4cea-badb-bf3bb4c7a67c" />

## Performance

| Metric | Value |
|--------|-------|
| **Latency** | ~172 μs/step |
| **Particles** | 512 |

```
Scenario            RMSE (log-vol)
────────────────────────────────────
calm                0.3468
crisis              0.6639
spike               0.5677
regime_shift        0.3468
asymmetric          0.4067
jumps               0.4108
fat_tails           0.3899
────────────────────────────────────
```

---

## The Problem

Volatility is **latent**—we never observe it directly:

```
Hidden:     h_t = μ + ρ(h_{t-1} - μ) + σ_z·ε     (log-volatility)
Observed:   y_t = exp(h_t/2)·η                   (returns)
```

The challenge: infer `h_t` from `y_t` in real-time, handling regime changes and fat tails.

---

## Why Stein Variational?

**Bootstrap particle filters** rely on resampling—particles die and clone. When the target moves to a new region, you need particles already waiting there.

**SVPF** replaces resampling with **gradient transport**. Particles flow toward high-probability regions:

```
φ(x) = E[K·∇log p] + E[∇K]
       ───────────   ─────
       ATTRACTION    REPULSION
       (to mode)     (diversity)
```

When a regime shifts, the gradient points there—particles **flow** instead of hoping to randomly land correctly.

---

## Algorithm

```
┌──────────────────────────────────────────────────────────────────┐
│                     SVPF STEP (per tick)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PREDICT        h ~ p(h_t | h_{t-1})                         │
│     ├── MIM: Mixture transition for jump handling               │
│     └── Asymmetric ρ: fast spike, slow decay                    │
│                          ↓                                       │
│  2. GUIDE          Nudge toward EKF estimate                    │
│                          ↓                                       │
│  3. STEIN LOOP     For β in [0.3 → 1.0]:                        │
│     │   • Compute gradient ∇log p(h|y)                          │
│     │   • Newton precondition: H⁻¹·∇                            │
│     │   • Kernel transport with IMQ                             │
│     │   • Add Langevin noise (SVLD)                             │
│     └── KSD early stopping                                       │
│                          ↓                                       │
│  4. OUTPUT         vol = mean(exp(h/2))                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Newton-Stein Preconditioning
Standard SVGD uses raw gradients. We precondition with the Hessian:
```
Standard:  h += ε · ∇log p           (same step everywhere)
Newton:    h += ε · H⁻¹ · ∇log p     (adapts to curvature)
```
Converges in 4-8 steps vs 15-20 for standard SVGD.

### Annealed Stein Updates
Sharp likelihoods create local traps. We start with tempered likelihood (β=0.3), let particles explore, then gradually commit (β→1.0). Particles get a "global map" before finding specific modes.

### EKF Guide Density
Stein gradients are local—if particles start far from the mode, gradients are flat. A parallel EKF finds the approximate mode and nudges particles to the right neighborhood. EKF is wrong (assumes Gaussian), but gets particles close; Stein then refines the shape.

### SVLD (Langevin Noise)
Deterministic SVGD suffers variance collapse in the finite-particle regime. Adding `√(2εT)·noise` maintains particle spread and proper uncertainty quantification.

### KSD-Adaptive Tempering
When particles disagree (high KSD), the likelihood is uninformative at boundaries. We reduce β to trust the prior more—prevents overfitting to noise.

### IMQ Kernel
Inverse Multiquadric `K = 1/(1+r²)` has polynomial decay vs Gaussian's exponential. Outlier particles stay connected during volatility spikes—critical for regime tracking.

---

## Literature & Extensions

Our implementation extends Fan et al. (2021) with techniques from the particle filtering and SVGD literature:

| Feature | Source | Description |
|---------|--------|-------------|
| Stein transport | Fan 2021 | Core SVGD-based particle filter |
| Newton preconditioning | Detommaso 2018 | Hessian-scaled Stein steps |
| SVLD noise | Ba 2021 | Langevin diffusion prevents variance collapse |
| Annealing | D'Angelo 2021 | Tempered likelihood for exploration |
| KSD-adaptive β | Maken 2022 | Trust prior when particles disagree |
| EKF guide | Van Der Merwe 2000 | Gaussian proposal for fast response |
| MIM predict | Gordon 1993 | Mixture transition for jumps |
| Asymmetric ρ | Nelson 1991 | Leverage effect (fast spike, slow decay) |
| Adaptive μ | Chopin 2013 | Online mean-level learning (SMC²) |
| Exact Student-t | Geweke 1993 | Consistent heavy-tailed likelihood |

### Key References

- Fan et al. (2021) "Stein Particle Filter" — Core algorithm
- Detommaso et al. (2018) "Stein Variational Newton" — Newton preconditioning
- Maken et al. (2022) "Stein Particle Filter for Nonlinear Systems" — KSD-adaptive tempering
- Liu & Wang (2016) "Stein Variational Gradient Descent" — Foundational SVGD

---

## Why Not Bootstrap PF?

| Aspect | Bootstrap PF | SVPF |
|--------|--------------|------|
| Particle update | Kill/clone (resampling) | Flow (gradient transport) |
| Regime tracking | Needs pre-positioned particles | Single adaptive filter |
| Parameter changes | Affect future births only | Instant redirect via gradient |
| Mode collapse | Risk with few particles | Kernel repulsion prevents |

The fundamental difference: Bootstrap particles are **passive** (wait to be selected). SVPF particles are **active** (move toward the target).

---

## Usage

```c
SVPFState* filter = svpf_create(400, 8, 30.0f, stream);
svpf_initialize(filter, &params, seed);

for (int t = 0; t < n_ticks; t++) {
    svpf_step_graph(filter, returns[t], returns[t-1], &params,
                    &loglik, &vol, &h_mean);
}

svpf_destroy(filter);
```

---

## License

MIT
