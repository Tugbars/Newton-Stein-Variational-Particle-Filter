# Joint State-Parameter SVPF

## The Injection Problem

### The Fundamental Issue

Traditional approaches to adaptive filtering separate **state estimation** (tracking h) from **parameter learning** (finding θ). This creates an "injection" problem:

```
                    EXTERNAL SYSTEM                    
                    (SMC², PMMH, etc.)                 
                          │                            
                          │ θ_learned                  
                          ▼                            
    ┌─────────────────────────────────────────────────┐
    │                    SVPF                         │
    │                                                 │
    │   Particles: [h₁, h₂, ..., hₙ]                 │
    │   Parameters: θ ← INJECTED FROM OUTSIDE        │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

**The Problem:** When you inject θ, you're forcing a belief onto the particle system that may be immediately wrong.

### Why This Is Catastrophic During Regime Changes

Consider a flash crash scenario:

```
t = 0..10000: Calm market
              SMC²/PMMH learns θ_calm = {ρ=0.97, σ_z=0.12, μ=-3.5}
              
t = 10001:    Flash crash begins
              SVPF struggles with current θ
              Requests parameter update
              
t = 10002:    θ_calm injected into SVPF
              
              DISASTER:
              - θ_calm says "σ_z is small, jumps are unlikely"
              - But crash is happening RIGHT NOW
              - Gradient computed with wrong θ
              - Particles pushed AWAY from truth
              - Complete collapse
```

The learned parameters are **always from the past**. They cannot anticipate regime changes.

### Why SVPF Is Especially Vulnerable

Standard particle filters with wrong θ:
- Particles get wrong weights
- Resampling eventually corrects (slowly)

SVPF with wrong θ:
- Gradients computed with wrong θ
- Particles **actively pushed in wrong direction**
- Every Stein step makes it worse
- Positive feedback loop → collapse

---

## The Solution: Joint State-Parameter Estimation

### Core Insight

**Don't inject parameters. Make them part of the particle state.**

```
                    NO EXTERNAL INJECTION                 
                                                          
    ┌─────────────────────────────────────────────────────┐
    │                  JOINT SVPF                         │
    │                                                     │
    │   Particle i = [h_i, μ̃_i, ρ̃_i, σ̃_i]              │
    │                  │     └───────┴───────┘            │
    │                  │           │                      │
    │               state      parameters                 │
    │                  │           │                      │
    │                  └─────┬─────┘                      │
    │                        │                            │
    │              SINGLE GRADIENT SYSTEM                 │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

Each particle carries its own parameter estimates. Parameters evolve through the **same gradient mechanism** as the state.

### Why This Fixes the Flash Crash

```
t = 10000 (calm):
    All particles: h ≈ -3, σ_z ≈ 0.12
    Gradients ≈ 0 (equilibrium)
    
t = 10001 (CRASH):
    Observation y is huge
    
    For each particle:
        h_pred = μ + ρ(h_prev - μ)
        diff = h - h_pred          ← HUGE (crash wasn't predicted)
        z² = diff² / σ²            ← HUGE (σ is too small)
        
    Gradients:
        ∇_h     >> 0   "h is too low, push up"
        ∇_σ = z²-1 >> 0   "σ is too small, push up"
        
    RESULT: σ_z increases AUTOMATICALLY to explain the crash
    
t = 10010:
    σ_z has drifted to 0.25
    Particles now EXPECT large jumps
    System adapted without external injection
    
t = 10500 (calm returns):
    diff is small → z² < 1
    ∇_σ < 0 → "σ is too big, push down"
    σ_z slowly returns to baseline
```

**No injection. No gating. Parameters adapt through gradients.**

---

## Mathematical Framework

### Particle State

Each particle carries a 4-dimensional state (extensible to more parameters):

| Component | Symbol | Constraint | Transform | Description |
|-----------|--------|------------|-----------|-------------|
| Log-volatility | h | ℝ | identity | The state we're tracking |
| Mean level | μ̃ | ℝ | μ̃ = μ | Long-run mean |
| Persistence | ρ̃ | ℝ | ρ = sigmoid(ρ̃) | Mean-reversion speed |
| Vol-of-vol | σ̃ | ℝ | σ = exp(σ̃) | Innovation noise |

**Why transforms?** Gradient descent works best on unconstrained spaces. The transforms map bounded parameters to ℝ.

### The Joint Log-Posterior

We want to sample from:

```
p(h, θ | y) ∝ p(y | h) · p(h | h_prev, θ) · p(θ)
```

Taking logs:

```
log p(h, θ | y) = log p(y|h) + log p(h|h_prev,θ) + log p(θ) + const
                  └────┬────┘   └───────┬───────┘   └──┬──┘
                  likelihood    transition prior    param prior
```

### Gradient Derivations

#### Likelihood Term (affects h only)

Using exact Student-t:

```
∇_h log p(y|h) = -0.5 + 0.5·(ν+1)·A/(1+A) - offset

where A = y²/(ν·exp(h))
```

The likelihood doesn't depend on θ directly, so ∇_θ = 0 for this term.

#### Transition Term (affects all)

The transition density is:

```
p(h | h_prev, θ) = N(h; μ + ρ(h_prev - μ), σ²)
```

Define:
```
h_pred = μ + ρ(h_prev - μ)
diff = h - h_pred
z² = diff² / σ²
```

**Gradients:**

```
∇_h log p = -diff / σ²
            "Pull h toward prediction"

∇_μ log p = (diff / σ²) · (1 - ρ)
            "If h > prediction, increase μ"

∇_ρ̃ log p = (diff / σ²) · (h_prev - μ) · ρ(1-ρ)
            "Chain rule through sigmoid"

∇_σ̃ log p = z² - 1
            "If z² > 1: σ too small, increase it"
            "If z² < 1: σ too big, decrease it"
```

The **key insight** for crashes: when `diff` is huge, `z²` is huge, so `∇_σ̃ >> 0`. The gradient automatically pushes σ up to explain the unexpected jump.

#### Prior Term (optional regularization)

Weak Gaussian priors prevent parameters from drifting to extreme values:

```
∇_μ̃ log p(θ) = -(μ̃ - μ̃_prior) / var_prior
∇_ρ̃ log p(θ) = -(ρ̃ - ρ̃_prior) / var_prior
∇_σ̃ log p(θ) = -(σ̃ - σ̃_prior) / var_prior
```

Use large prior variances (weak regularization) so data dominates.

---

## The Diagonal Kernel

### The Scale Problem

Parameters and state have very different scales:

```
h     changes by ~1.0 per timestep
ρ̃     changes by ~0.01 per timestep
σ̃     changes by ~0.1 per timestep
```

A single bandwidth would be dominated by h, making the kernel blind to parameter differences.

### Solution: Per-Dimension Bandwidths

Instead of isotropic RBF:

```
k(x, x') = exp(-||x - x'||² / h²)
```

Use diagonal (anisotropic) kernel:

```
k(x, x') = exp(-(h-h')²/bw_h² - (μ̃-μ̃')²/bw_μ² - (ρ̃-ρ̃')²/bw_ρ² - (σ̃-σ̃')²/bw_σ²)
```

Each bandwidth is computed via the median heuristic on that dimension:

```
bw_h = median(|h_i - h_j|) / √(2 log N)
bw_μ = median(|μ̃_i - μ̃_j|) / √(2 log N)
...
```

### Kernel Gradient

For the Stein repulsive term:

```
∇_{h_i} k(x_i, x_j) = k · 2(h_i - h_j) / bw_h²
∇_{μ̃_i} k(x_i, x_j) = k · 2(μ̃_i - μ̃_j) / bw_μ²
...
```

---

## Learning Rate Separation

### Fast State vs Slow Parameters

The state h changes every tick. Parameters are "structural" - they should change slowly.

| Component | Learning Rate | Rationale |
|-----------|---------------|-----------|
| h | 0.10 | Fast tracking (main purpose) |
| μ̃ | 0.01 | Mean level drifts slowly |
| ρ̃ | 0.005 | Persistence is very stable |
| σ̃ | 0.01 | Vol-of-vol changes moderately |

**Alternative:** Use RMSProp, which naturally adapts learning rates per dimension based on gradient magnitudes.

### Why This Matters

Without separation:
- Parameters jitter on every observation
- Noise in parameter estimates
- Unstable filtering

With separation:
- h tracks quickly (low latency)
- θ adapts slowly (stability)
- Best of both worlds

---

## Implementation Architecture

### Data Structures

```cpp
struct SVPFJointState {
    // State arrays (SoA layout)
    float* d_h;              // [N] log-volatility
    float* d_h_prev;         // [N] previous h (for transition gradient)
    float* d_mu_tilde;       // [N] mean level (unconstrained)
    float* d_rho_tilde;      // [N] persistence (logit-transformed)
    float* d_sigma_tilde;    // [N] vol-of-vol (log-transformed)
    
    // Gradient arrays
    float* d_grad_h;
    float* d_grad_mu;
    float* d_grad_rho;
    float* d_grad_sigma;
    
    // Bandwidths (one per dimension)
    float bw_h, bw_mu, bw_rho, bw_sigma;
    
    // Learning rates
    float step_h, step_mu, step_rho, step_sigma;
    
    // Prior hyperparameters
    float mu_prior_mean, mu_prior_var;
    float rho_prior_mean, rho_prior_var;    // In unconstrained space
    float sigma_prior_mean, sigma_prior_var; // In log space
    
    // ... RNG, diagnostics, etc.
};
```

### Kernel Pipeline

```
For each observation y_t:

┌─────────────────────────────────────────────────────────────────┐
│ 1. PREDICT                                                      │
│    h_new = μ_i + ρ_i(h_prev - μ_i) + σ_i·noise                 │
│    (Each particle uses its OWN parameters)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. COMPUTE BANDWIDTHS                                           │
│    bw_h = median_heuristic(d_h)                                 │
│    bw_μ = median_heuristic(d_mu_tilde)                          │
│    bw_ρ = median_heuristic(d_rho_tilde)                         │
│    bw_σ = median_heuristic(d_sigma_tilde)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. STEIN ITERATIONS (repeat K times)                            │
│                                                                 │
│    3a. GRADIENT KERNEL                                          │
│        Compute: ∇_h, ∇_μ, ∇_ρ, ∇_σ for each particle           │
│        (Likelihood + Transition + Prior terms)                  │
│                                                                 │
│    3b. STEIN TRANSPORT KERNEL                                   │
│        φ_i = (1/N) Σ_j [k(x_j,x_i)·∇_j + ∇_j k(x_j,x_i)]       │
│        x_i += step · φ_i  (per-dimension steps)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. EXTRACT ESTIMATES                                            │
│    h_mean = mean(d_h)                                           │
│    μ_mean = mean(d_mu_tilde)                                    │
│    ρ_mean = mean(sigmoid(d_rho_tilde))                          │
│    σ_mean = mean(exp(d_sigma_tilde))                            │
│    (Also compute std for uncertainty quantification)            │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Requirements

Per particle (4D state + gradients + misc):

```
State:     4 floats × 4 bytes = 16 bytes
Gradients: 4 floats × 4 bytes = 16 bytes
Previous:  1 float × 4 bytes  =  4 bytes
Weights:   1 float × 4 bytes  =  4 bytes
─────────────────────────────────────────
Total:     ~40 bytes per particle
```

For N = 512 particles: **~20 KB** (trivial)

For N = 4096 particles: **~160 KB** (still trivial)

---

## Comparison with Alternative Approaches

| Approach | θ Source | During Crash | Correctness | Speed |
|----------|----------|--------------|-------------|-------|
| Hard injection | External | Wrong θ injected, collapse | ❌ | Fast |
| Soft injection | External (gated) | Delayed response | ⚠️ | Fast |
| Crisis gating | External (frozen) | No adaptation | ⚠️ | Fast |
| SMC² + RBPF | Particle cloud | Slow adaptation | ✅ | Slow |
| **Joint SVPF** | Gradient flow | Instant adaptation | ✅ | Medium |

**Key advantage of Joint SVPF:** Parameters adapt at the speed of gradients, not the speed of resampling.

---

## Theoretical Grounding

### Connection to Generalized Variational Inference (GVI)

The Joint SVPF can be viewed as minimizing:

```
L(q) = E_q[-log p(y|h)] + D_KL(q(h,θ) || p(h,θ|h_prev))
```

The Stein gradient is the functional derivative of this objective. By transporting particles along this gradient, we're doing variational inference on the joint space.

### Connection to Liu & West (2001)

Liu & West introduced parameters as particles with kernel smoothing. Joint SVPF replaces the Gaussian kernel perturbation with **gradient-driven Stein transport** - no Gaussian assumptions required.

### Connection to Feedback Particle Filter (FPF)

FPF treats filtering as optimal control. The Stein update can be viewed as a specific gain function that minimizes KL divergence. Joint SVPF extends this to the parameter space.

---

## Tuning Guide

### Recommended Defaults

```cpp
// Learning rates
step_h     = 0.10f;   // Fast state tracking
step_mu    = 0.01f;   // Slow mean adaptation
step_rho   = 0.005f;  // Very slow persistence
step_sigma = 0.01f;   // Moderate vol-of-vol

// Prior regularization (weak)
mu_prior_mean = -3.5f;  mu_prior_var = 10.0f;    // Allows μ ∈ [-10, 3]
rho_prior_mean = 2.0f;  rho_prior_var = 5.0f;    // Centers on ρ ≈ 0.88
sigma_prior_mean = -2.0f; sigma_prior_var = 5.0f; // Centers on σ ≈ 0.14

// Stein iterations
n_stein_steps = 5;  // More may be needed initially

// Particles
n_particles = 512;  // Or 1024 for higher dimensions
```

### Monitoring

Track these diagnostics:

1. **Parameter spread:** `std(ρ_i)`, `std(σ_i)` should be non-zero (diversity)
2. **ESS:** Effective sample size based on log-weights
3. **Gradient magnitudes:** Large gradients indicate surprise
4. **Parameter drift:** Moving average of μ, ρ, σ estimates

### Warning Signs

- **σ exploding:** Prior too weak, or extreme outliers in data
- **ρ → 1:** Filter losing mean-reversion, might need stronger prior
- **All particles same θ:** Loss of diversity, need more repulsion
- **Gradients stuck at clip limits:** Step sizes too large

---

## Future Extensions

### More Parameters

Easily extensible to include:

- **ν** (Student-t degrees of freedom)
- **γ** (leverage coefficient)  
- **lik_offset** (bias correction)

Each additional parameter adds one dimension to the particle state.

### Regime-Specific Parameters

Could maintain separate parameter sets for different regimes:

```
Particle i = [h_i, μ̃_calm_i, μ̃_crisis_i, ...]
```

With soft switching based on regime indicator.

### Batched Joint SVPF for SMC²

Run M independent Joint SVPFs in parallel:

```
θ-particle 1: [h¹₁..h¹ₙ, μ̃¹₁..μ̃¹ₙ, ...]
θ-particle 2: [h²₁..h²ₙ, μ̃²₁..μ̃²ₙ, ...]
...
θ-particle M: [hᴹ₁..hᴹₙ, μ̃ᴹ₁..μ̃ᴹₙ, ...]
```

This combines the theoretical correctness of SMC² with the adaptivity of Joint SVPF.

---

## Summary

**The Problem:** Injecting externally-learned parameters into SVPF causes catastrophic failures during regime changes.

**The Solution:** Make parameters part of the particle state. They evolve through the same gradient mechanism as the volatility state.

**Key Insight:** When a crash happens, the gradient ∇_σ = z² - 1 automatically pushes σ upward because z² >> 1 (the unexpected jump has a huge z-score under the current σ). No external intervention needed.

**Implementation:** 
- 4D particles: `[h, μ̃, ρ̃, σ̃]`
- Diagonal kernel with per-dimension bandwidths
- Separate learning rates (fast h, slow θ)
- Transform constrained params to ℝ

**Result:** A theoretically grounded, self-adapting filter that handles regime changes gracefully.

---

*Document version: 1.0*  
*Date: January 2026*
