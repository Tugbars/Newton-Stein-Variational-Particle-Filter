# SVPF: Stein Variational Particle Filter for Stochastic Volatility

A high-performance CUDA implementation of the Stein Variational Particle Filter for real-time volatility tracking in financial applications.

[![CUDA](https://img.shields.io/badge/CUDA-13.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 0.5507 (log-vol) |
| **Latency** | ~10 μs/step |
| **Particles** | 400 |
| **Bias** | ~0.00 |

```
Scenario                   RMSE        Bias
────────────────────────────────────────────
Slow Drift               0.5942     -0.02
Stress Ramp              0.6233     -0.01
OU-Matched               0.4105     +0.07
Intermediate Band        0.6390     -0.02
Spike+Recovery           0.4762     +0.02
Wrong-Model              0.5611     -0.07
────────────────────────────────────────────
AVERAGE                  0.5507     ~0.00
```

---

## The Problem: Tracking Hidden Volatility

Financial volatility is **latent** — we never observe it directly. We only see noisy price returns:

```
Hidden:     h_t = μ + ρ(h_{t-1} - μ) + σ_z·ε     (log-volatility)
Observed:   y_t = exp(h_t/2)·η                   (returns)
```

The challenge: infer the hidden state `h_t` from observations `y_t` in real-time.

---

## Why Stein Variational?

### The Bootstrap Particle Filter Problem

Traditional particle filters represent the posterior as weighted samples. When the target moves, particles must **die and resample**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BOOTSTRAP PARTICLE FILTER                        │
│                                                                     │
│     Particles are POINTS with WEIGHTS                               │
│                  ↓                                                   │
│     Low-weight particles DIE (resampling)                           │
│                  ↓                                                   │
│     Particles can only be where they were BORN                      │
│                  ↓                                                   │
│     If target moves to new region → DEGENERACY                      │
│     (no particles there to catch it)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

This is why regime-switching filters need **pre-positioned filter banks** — particles waiting at each possible regime.

### The Stein Solution: Particles That Move

SVPF replaces resampling with **gradient transport**. Particles flow toward high-probability regions:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STEIN VARIATIONAL FILTER                         │
│                                                                     │
│     Particles MOVE along gradients                                  │
│                  ↓                                                   │
│     φ(x) = E[K·∇log p] + E[∇K]                                      │
│             ─────────    ─────                                       │
│             ATTRACTION   REPULSION                                   │
│             (to mode)    (diversity)                                 │
│                  ↓                                                   │
│     If target moves → gradient points there → particles FLOW        │
│     (single adaptive filter can "teleport" to any regime)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: When you change a model parameter, you're not hoping particles randomly land correctly — you're **telling them where to go** via the gradient.

---

## The Stein Operator

At the heart of SVPF is the **Stein Variational Gradient Descent** update:

```
φ*(x) = E_{x'~q} [ k(x', x) ∇log p(x') + ∇_{x'} k(x', x) ]
        ───────────────────────────────   ─────────────────
                   DRIFT                      REPULSION
           (toward high density)        (maintain diversity)
```

Where:
- `p(x)` is the target posterior
- `q(x)` is the current particle distribution  
- `k(x, x')` is a kernel measuring particle similarity

The drift term pulls particles toward the posterior mode. The repulsion term prevents collapse — particles push each other apart, maintaining coverage of the distribution.

### Why This Works for Filtering

In sequential filtering, the posterior changes every timestep. Bootstrap PF handles this by **killing and birthing** particles. SVPF handles it by **moving** particles.

```
Time t:    Posterior at region A    →  Particles cluster at A
Time t+1:  Posterior shifts to B    →  Gradient points to B
                                    →  Particles FLOW from A to B
```

No particles need to die. No lucky particles need to exist at B beforehand. The gradient is the **communication channel** between the model and the particles.

---

## Algorithm Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     SVPF STEP (per tick)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PREDICT        h_i ~ p(h_t | h_{t-1})                       │
│     ├── Asymmetric ρ (vol spikes fast, decays slow)             │
│     ├── Mixture Innovation (5% scouts for jumps)                │
│     └── Guided proposal (peek at y_t when surprised)            │
│                          ↓                                       │
│  2. GUIDE          h_i += λ(m_EKF - h_i)                        │
│     └── EKF posterior as "compass" to right neighborhood        │
│                          ↓                                       │
│  3. STEIN          For β in [0.3, 0.65, 1.0]:  (annealing)     │
│     │                                                            │
│     │   ┌─────────────────────────────────────┐                 │
│     │   │  GRADIENT    ∇log p(y|h) + ∇log p(h)│                 │
│     │   │      ↓                              │                 │
│     │   │  IMQ KERNEL  K = 1/(1 + r²)        │ ← Heavy tails    │
│     │   │      ↓                              │                 │
│     │   │  NEWTON      H⁻¹ preconditioning   │ ← Fast converge  │
│     │   │      ↓                              │                 │
│     │   │  LANGEVIN    √(2εT)·noise          │ ← Exploration    │
│     │   └─────────────────────────────────────┘                 │
│     │                    ↓                                       │
│     └── Transport: h_i += ε·φ(h_i)                              │
│                          ↓                                       │
│  4. OUTPUT         vol = mean(exp(h/2))                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. EKF Guide Density

**Problem**: Stein gradients are local. If particles start far from the posterior mode, gradients are flat — particles don't know where to go.

**Solution**: Run a cheap Extended Kalman Filter to find the approximate mode, then "teleport" particles to that neighborhood before running Stein updates.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PARTICLES │     │    EKF      │     │   STEIN     │
│  (scattered)│ ──→ │  (compass)  │ ──→ │  (refine)   │
│             │     │  finds mode │     │  exact shape│
└─────────────┘     └─────────────┘     └─────────────┘
     Lost            Points to A         Sculpts detail
```

The EKF is wrong (it assumes Gaussian), but it gets particles to the **right neighborhood**. Stein then corrects the shape.

### 2. Annealed Updates

**Problem**: Sharp likelihoods create local traps. Particles converge to the nearest mode, not the global mode.

**Solution**: Start with a "melted" (flat) likelihood, let particles spread globally, then gradually "freeze" to the true shape.

```
Temperature:   HIGH ──────────────────────→ LOW
Landscape:     Flat (explore everywhere)    Sharp (commit to modes)
Particles:     Spread out globally          Concentrate on peaks
```

This gives particles a "global map" before asking them to find a specific address.

### 3. SVLD (Langevin Noise)

**Problem**: In high dimensions, kernel repulsion weakens. Particles collapse to a point (variance collapse).

**Solution**: Add controlled noise to maintain exploration:

```
Standard SVGD:  h += ε·φ(h)                    ← Deterministic, can collapse
SVLD:           h += ε·φ(h) + √(2εT)·noise    ← Stochastic, maintains spread
```

The noise acts as "insurance" against variance collapse.

### 4. Newton-Stein (Hessian Preconditioning)

**Problem**: Standard gradient descent is slow in "stiff" posteriors (long narrow valleys). Particles zig-zag instead of going straight.

**Solution**: Use curvature information to take smarter steps:

```
Standard:  h += ε · ∇log p           ← Same step size everywhere
Newton:    h += ε · H⁻¹ · ∇log p     ← Adapts to local geometry
```

Newton converges in 2-3 steps where standard SVGD needs 10-20.

### 5. IMQ Kernel

**Problem**: Gaussian kernel `K = exp(-r²)` decays exponentially. Distant particles stop "talking" to each other.

**Solution**: Use Inverse Multiquadric kernel with polynomial decay:

```
Gaussian:  K = exp(-r²)      →  At 3σ: K = 0.01  (particle stranded)
IMQ:       K = 1/(1 + r²)    →  At 3σ: K = 0.10  (still connected)
```

During volatility spikes, outlier particles exploring the high-vol region stay connected to the group.

### 6. Mixture Innovation Model

**Problem**: Standard Gaussian proposals are "tunnel-visioned". If volatility jumps 10x, no particles land there.

**Solution**: 5% of particles are "scouts" with 5x wider proposals:

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   95% SHEEP: σ = 0.15    ████████████                         │
│                          (accurate if nothing crazy)           │
│                                                                │
│   5% SCOUTS: σ = 0.75         ▪   ▪       ▪    ▪              │
│                          (insurance against jumps)             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

During calm markets, scouts get killed by reweighting. During crashes, scouts survive and take over — the particle cloud "teleports" to the new regime.

---

## Adaptive Mechanisms

The gradient is a **communication channel**. Adaptive parameters modulate what the gradient says:

| Parameter | Adaptation | Gradient Effect |
|-----------|------------|-----------------|
| **μ** | Kalman filter on particle mean | Shifts prior center → particles drift |
| **σ_z** | Innovation-gated boost | Weakens prior → particles spread |
| **guide_strength** | Boost on upward vol surprises | Stronger likelihood pull → particles snap |
| **ρ** | Direction-dependent (up/down) | Changes mean-reversion speed |

```
┌─────────────────────────────────────────────────────────────┐
│              ADAPTIVE PARAMETER EFFECTS                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ↑ σ_z    →  Prior gradient WEAKENS  →  Particles SPREAD  │
│   ↓ σ_z    →  Prior gradient STRENGTHENS → Particles FOCUS │
│   Shift μ  →  Prior mean MOVES        →  Particles DRIFT   │
│   ↑ guide  →  Likelihood pull UP      →  Particles SNAP    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This is why SVPF doesn't need filter banks for regime switching — parameter changes **instantly** redirect all particles via the gradient.

---

## Why Not Just Use Bootstrap PF?

| Aspect | Bootstrap PF | SVPF |
|--------|--------------|------|
| **Particle update** | Kill/clone (resampling) | Flow (gradient transport) |
| **Regime tracking** | Need pre-positioned filters | Single adaptive filter |
| **Parameter changes** | Affect future births only | Instant effect on all particles |
| **Mode collapse** | Risk with few particles | Kernel repulsion prevents it |
| **Computational** | O(N) but degeneracy risk | O(N²) kernel but stable |

**The fundamental difference**: Bootstrap PF particles are **passive** (they wait to be selected). SVPF particles are **active** (they move toward the target).

---

## Theoretical Foundation

SVPF minimizes the KL divergence between the particle distribution q and the target posterior p:

```
min KL(q || p) = min E_q[log q - log p]
```

The Stein operator provides the direction of steepest descent in the space of distributions:

```
φ* = argmax_{φ ∈ H} { -d/dε KL(T_ε q || p) |_{ε=0} }
```

Where `T_ε` is the transport map `x → x + ε·φ(x)`. The solution is:

```
φ*(x) = E_{x'~q}[ k(x',x) ∇log p(x') + ∇_{x'} k(x',x) ]
```

This is computable from samples — no need to know the normalizing constant of p.

---

## Limitations

- **Cramér-Rao Bound**: RMSE ~0.55 is approaching theoretical minimum for this observation model
- **O(N²) Kernel**: Each particle interacts with all others (mitigated by GPU parallelism)
- **Graph Recapture**: Adaptive parameters can trigger CUDA graph recapture (~5μs overhead)

---

## Future Work

- **PMMH Oracle**: Parallel parameter learning for ν, ρ, σ_z, μ (~100μs per iteration)
- **Zero-Copy Updates**: Eliminate graph recapture for adaptive parameters
- **Multi-Asset**: Correlated volatility tracking across instruments

---

## References

- Liu & Wang (2016): "Stein Variational Gradient Descent" — *The foundational SVGD paper*
- Detommaso et al. (2018): "Stein Variational Newton" — *Hessian preconditioning*
- D'Angelo & Fortuin (2021): "Annealed SVGD" — *Temperature scheduling for multimodal targets*
- Ba et al. (2021): "Understanding Variance Collapse of SVGD" — *Why SVLD is needed*
- Fan et al. (2021): "Stein Particle Filter" — *SVPF for sequential inference*

---

## License

MIT

---

*Built for HFT volatility tracking. Optimized for NVIDIA GPUs.*
