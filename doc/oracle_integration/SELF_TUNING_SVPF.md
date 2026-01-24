# Self-Tuning SVPF: Development Plan

## Executive Summary

A closed-loop system where:
- **SVPF** tracks volatility h (fast, robust, handles everything)
- **Natural gradient feedback** slowly adjusts θ = (μ, ρ, σ, ν) to match what SVPF is doing

No oracle. No batch inference. No memory. Just negative feedback like an AGC, with Fisher-preconditioned updates for automatic parameter scaling.

---

## 1. The Core Insight

### 1.1 Why Batch Learners Fail

Traditional approaches (SMC², CPMMH, forgetting factors) try to **infer θ from data history**:

```
Batch learner: Data → Accumulate → Infer θ → Inject into filter

Problems:
  • Always lags behind reality
  • Needs arbitrary amnesia mechanisms (Q, λ, L)
  • Complex (SMC², CPMMH, replay buffers)
  • Asks "What was θ?" not "What is θ now?"
```

### 1.2 The AGC Alternative

SVPF particles **already know** where h is. They moved there via Stein transport + likelihood. θ should just follow.

```
AGC approach: SVPF particles → Where are they? → Adjust θ to match

Benefits:
  • No lag (SVPF already moved to the right place)
  • No memory needed (reading current state, not history)
  • Simple (just gradient feedback)
  • Asks "What is θ now?" directly
```

### 1.3 The Analogy

```
Audio Amplifier with AGC:

  Input → [Amplifier] → Output
               ↑           │
               │           │ "Is output too loud/quiet?"
               │           ▼
               └─── [Feedback] ←── "Adjust gain"
               
  • Amplifier handles the signal
  • Feedback adjusts gain to keep output in range
  • No memory of what input "should" be
  • Just: "output too high → reduce gain"

Our System:

  Observations → [SVPF] → h particles
                   ↑           │
                   │           │ "Are particles where θ predicts?"
                   │           ▼
                   └─── [Gradient] ←── "Adjust θ"
                   
  • SVPF handles tracking (fast, robust)
  • Gradient adjusts θ to reduce prediction error
  • No memory of what θ "should" be
  • Just: "particles above prediction → raise μ"
```

---

## 2. Why SVPF Doesn't Need an Oracle

### 2.1 Passive vs Active Particles

```
BOOTSTRAP PF:
                                                    
  Particles are PASSIVE                             
  They sample blindly from p(h_t | h_{t-1}, θ)     
                                                    
  If θ is wrong:                                    
    → Particles are in wrong place                  
    → They stay there (no mechanism to move)        
    → Likelihood weights can't save you (degeneracy)
    → You NEED correct θ from outside              
                                                    
  Oracle makes sense: particles can't help themselves


SVPF:
                                                    
  Particles are ACTIVE                              
  Stein transport moves them toward likelihood      
                                                    
  If θ is wrong:                                    
    → Particles start in wrong place                
    → Likelihood gradient pulls them               
    → Stein transport moves them                    
    → They end up in the RIGHT place anyway        
                                                    
  Oracle is redundant: particles already found it   
  θ just needs to follow along                      
```

**The roles are reversed:**

```
BOOTSTRAP PF + ORACLE:
                    
  Oracle: "Here's the right θ"          (LEADER)
     │
     ▼
  Particles: "Okay, we'll sample from that"   (FOLLOWER)


SVPF + GRADIENT:
                    
  Particles: "We moved to where h is"   (LEADER)
     │
     ▼
  Gradient: "Okay, I'll adjust θ to match"    (FOLLOWER)
```

### 2.2 The Resampling Collapse Problem

Bootstrap PF has a fatal flaw: **resampling is based on prior parameters**.

```
BOOTSTRAP PF COLLAPSE MECHANISM:

Step 1: PROPOSE from prior p(h_t | h_{t-1}, θ)

  True h:                        ●
  θ thinks h should be here:          ○ ○ ○ ○ ○ (particles)
                                      ↑
                                 Wrong location (θ is wrong)

Step 2: WEIGHT by likelihood p(y_t | h_t)

  Particle weights:    0.001  0.002  0.0001  0.0005  0.003
                       ↑
                  All garbage (none near true h)

Step 3: RESAMPLE based on weights

  You duplicate the "least bad" particle
  But least bad is still bad
  
  After resample:  ○ ○ ○ ○ ○  (all copies of one bad particle)
                       ↑
                  Diversity dead. Filter collapsed.

Step 4: REPEAT

  Each step makes it worse
  Degeneracy compounds
  ESS → 1
  Game over
```

**The prior proposal determines where particles go. Weights can't fix particles that are in the wrong place. Resampling just selects among bad options.**

### 2.3 SVPF Breaks This Chain

```
SVPF ESCAPE MECHANISM:

Step 1: PROPOSE from prior (same as bootstrap)

  True h:                        ●
  θ thinks h should be here:          ○ ○ ○ ○ ○ (particles)
                                      ↑
                                 Wrong location (θ is wrong)

Step 2: STEIN TRANSPORT (the key difference)

  Likelihood gradient: "h should be over THERE →"
  Repulsion: "spread out, don't collapse"
  
  Particles MOVE:                ● ○ ○ ○ ○ ○
                                 ↑
                            Found it!

Step 3: WEIGHT (now meaningful)

  Particles are in the right place
  Weights are reasonable: 0.18  0.22  0.15  0.25  0.20
  
Step 4: RESAMPLE (if needed, but often not)

  ESS is healthy (particles already good)
  Resampling doesn't destroy diversity
  Filter continues fine
```

**Stein transport corrects the prior's mistakes BEFORE weighting/resampling.**

### 2.4 The Fundamental Difference

```
BOOTSTRAP PF:
  
  Prior θ → Proposal → Weights → Resample
       ↑                            │
       │         (no feedback)      │
       └────────────────────────────┘
  
  If θ wrong at start, error propagates forever
  Weights can't move particles, only select among them
  Resampling selects among bad options = still bad


SVPF:
  
  Prior θ → Proposal → STEIN TRANSPORT → Weights → Resample
       ↑                     │                        │
       │                     │ (corrects proposal)    │
       │                     ▼                        │
       │              Particles now good              │
       │                                              │
       └──────────── Gradient feedback ───────────────┘
  
  θ wrong? Stein fixes it.
  Particles end up correct regardless.
  Gradient slowly aligns θ to match.
```

### 2.5 Summary: Why Different Approaches for Different Filters

| Aspect | Bootstrap PF | SVPF |
|--------|--------------|------|
| Wrong θ | **Fatal** (collapse) | **Survivable** (Stein corrects) |
| Particles | Passive (stuck where proposed) | Active (move toward likelihood) |
| Resampling | Selecting among bad options | Selecting among good options |
| Needs oracle? | **Yes** (survival depends on it) | **No** (feedback is enough) |

**Bootstrap PF needs an oracle because particles can't save themselves.**

**SVPF doesn't need an oracle because Stein transport is the escape hatch.**

The gradient feedback isn't rescuing the filter—it's just keeping θ aligned with what SVPF already figured out on its own.

---

## 3. What This Is (And Is Not)

### 3.1 What It Is

A **closed-loop adaptive system** in the "online maximum likelihood / recursive ML" family:

```
Inner loop (fast): SVPF produces approximate filtering posterior over h_t
                   (robust during shocks via transport + Student-t)

Outer loop (slow): Use particles to estimate score ∇θ log p(y|θ)
                   Update θ conservatively, with shock gating
```

This corresponds to classical **dual estimation / adaptive filtering**: fast state estimator + slower parameter adaptation.

**Literature connections:**
- Particle methods for parameter estimation: Kantas et al. (2015)
- Particle approximations of score: Poyiadjis et al. (2011)
- Online EM: Le Corff et al.
- Particle learning: Carvalho et al.

### 3.2 What It Is Not

```
✗ Full Bayesian parameter inference (no posterior uncertainty over θ)
✗ Regime identification (unless you add mixture over θ hypotheses)
✗ Guaranteed unbiased learning (SVPF is approximate, so gradient is 
  "variational / approximate score")
```

**This is acceptable** if your goal is self-calibration rather than asymptotically exact parameter inference.

### 3.3 When You Still Need an Oracle (SMC²/PMCMC)

Use heavy inference machinery when you need:
- Decision-grade uncertainty over θ
- Explicit regime probability (mixture over θ)
- Model comparison
- Auditable inference (not just "self-tuning control")

Otherwise, self-tuning SVPF is the sweet spot.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    SELF-TUNING SVPF                             │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │                                                     │     │
│    │   SVPF (fast loop, ~100μs)                         │     │
│    │                                                     │     │
│    │   • Student-t likelihood (ν controls tail weight)  │     │
│    │   • Stein transport (diversity, fast adaptation)   │     │
│    │   • Particles embody "where h actually is"         │     │
│    │   • Handles 99% of the work                        │     │
│    │   • Doesn't care if θ is slightly wrong            │     │
│    │                                                     │     │
│    └──────────────────────┬──────────────────────────────┘     │
│                           │                                     │
│                           │ h_t, h_prev, ancestors, weights     │
│                           │                                     │
│                           ▼                                     │
│    ┌─────────────────────────────────────────────────────┐     │
│    │                                                     │     │
│    │   Natural Gradient Feedback (slow loop)            │     │
│    │                                                     │     │
│    │   • Measures: particles - prediction               │     │
│    │   • Computes: ∇θ log p (transition + observation)  │     │
│    │   • Fisher matrix F = E[g gᵀ] (4×4)               │     │
│    │   • Updates: θ += lr · F⁻¹ · gradient             │     │
│    │   • State machine: SHOCK → RECOVERY → CALM         │     │
│    │                                                     │     │
│    └──────────────────────┬──────────────────────────────┘     │
│                           │                                     │
│                           │ θ = (μ, ρ, σ, ν)                   │
│                           ▼                                     │
│                      back to SVPF                               │
│                                                                 │
│    ─────────────────────────────────────────────────────────   │
│    Closed loop. No external oracle. No memory.                 │
│    θ tracks SVPF. SVPF tracks reality. System self-aligns.    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 The Four Parameters

| Parameter | Symbol | Unconstrained | Range | What It Controls |
|-----------|--------|---------------|-------|------------------|
| Mean level | μ | μ directly | ℝ | Long-run volatility level |
| Persistence | ρ | η = atanh(ρ) | (-1, 1) | Mean reversion speed |
| Vol-of-vol | σ | κ = log(σ) | (0, ∞) | Transition noise |
| Tail weight | ν | κ_ν = log(ν-2) | (2, ∞) | Crash robustness |

**Why ν matters:**
```
ν = 3:   Very heavy tails → survives extreme crashes, inefficient in calm
ν = 5:   Heavy tails → good default
ν = 10:  Moderate tails
ν = 30:  Almost Gaussian → efficient but fragile

Market regimes want different ν. Learning it automatically adapts.
```

---

## 5. What The Gradient Measures

### 5.1 The Model

```
Transition:    h_t = μ + ρ·(h_{t-1} - μ) + σ·noise
Observation:   y_t | h_t ~ Student-t(0, exp(h_t/2), ν)
```

Total log-likelihood gradient:
```
∇θ log p(y,h|θ) = ∇θ log p(h_t|h_{t-1}, μ,ρ,σ)  +  ∇θ log p(y_t|h_t, ν)
                  └───────── transition ─────────┘   └─── observation ───┘
```

### 5.2 Transition Gradients (μ, ρ, σ)

Prediction error:
```
ε = h_t - μ - ρ·(h_{t-1} - μ)
```

**μ (mean level):**
```
"Are particles systematically above or below prediction?"

ε > 0 consistently → raise μ
ε < 0 consistently → lower μ

Gradient: ∂/∂μ = ε·(1-ρ) / σ²
```

**ρ (persistence):**
```
"Do particles mean-revert faster or slower than predicted?"

If h_{t-1} was high and h_t didn't drop as much as predicted:
→ ρ is too low, raise it

Gradient: ∂/∂ρ = ε·(h_{t-1} - μ) / σ²
```

**σ (volatility):**
```
"Is actual scatter bigger or smaller than predicted?"

|ε|² > σ² consistently → raise σ
|ε|² < σ² consistently → lower σ

Gradient: ∂/∂σ = -1/σ + ε²/σ³
```

### 5.3 Observation Gradient (ν)

Student-t log-likelihood:
```
log p(y|h,ν) = log Γ((ν+1)/2) - log Γ(ν/2) - ½log(ν) - h/2 
             - ((ν+1)/2) · log(1 + z²/ν)

where z = y / exp(h/2)
```

**ν (tail weight):**
```
"How heavy should the tails be?"

Large |z| (crash) with high ν → low likelihood → "ν too high"
Large |z| (crash) with low ν  → okay likelihood → "ν is right"

Gradient: ∂/∂ν = ½[ψ((ν+1)/2) - ψ(ν/2) - 1/ν - log(1+z²/ν) + (ν+1)z²/(ν(ν+z²))]

where ψ = digamma function
```

**Key insight:** ν gradient is only informative during crashes (large |z|). In calm markets, all ν give similar likelihood. The Fisher matrix captures this: F_νν is large during crashes, small during calm.

### 5.4 Direct Measurement, Not Inference

```
BATCH LEARNER:
  "Given 10,000 observations, what θ maximizes likelihood?"
  → Historical inference, lags behind
  
AGC GRADIENT:
  "Right now, are particles where θ predicts?"
  "Right now, does ν explain the observation?"
  → Direct measurement, no lag
```

---

## 6. Critical Implementation Details

### 6.1 Model Consistency Warning

**Critical:** Your actual SVPF is NOT vanilla AR(1). It has:
- Leverage γ
- Asymmetric ρ (up/down)  
- MIM jump mixture
- Particle-local adjustments (Δρ, Δσ)
- Guided proposal + guide-preserving nudges
- Breathing σ_z
- Adaptive guide strength

If you compute gradient as if transition were `N(h_t; μ + ρ(h_{t-1} - μ), σ²)` while simulator uses all these extras, **you are not recovering "the" transition parameters**—you are fitting an **effective baseline**.

**This can still work** (and often does), but treat it as **control tuning**, not parameter recovery.

**Recommendation:** Learn only the baseline you want as a slowly varying operating point:
- Learn μ (you already do via Kalman update)
- Optionally learn single ρ and σ_z as "calm-regime baseline"
- Keep MIM, asymmetry, guides as fast robustness machinery (not learned)

### 6.2 Ancestor Pairing (Only If You Resample)

**Important caveat:** Your current SVPF does NOT resample inside the main loop.

```
If NO resampling:  Ancestor bookkeeping unnecessary
                   Use (h_prev[i], h[i]) directly
                   Parent i → child i

If you ADD resampling later:  Must store ancestor indices
                              Use (h_prev[ancestor[i]], h[i])
```

The full ancestor tracking is only needed if resampling scrambles particle identities:

```
Before resample (t-1):        After resample + propagate (t):
                              
   Particle 0: h=-2.0            Particle 0: h=-1.5  ← child of particle 2
   Particle 1: h=-1.0            Particle 1: h=-1.6  ← child of particle 2  
   Particle 2: h=-1.5  ───┬───►  Particle 2: h=-1.4  ← child of particle 2
   Particle 3: h=+0.5     │      Particle 3: h=-1.7  ← child of particle 2
                          │
             Particle 2 duplicated, others killed
```

**WRONG:**
```cpp
eps = h_t[i] - predict(h_prev[i])  // Comparing to wrong parent!
```

**CORRECT:**
```cpp
int a = ancestor[i];
eps = h_t[i] - predict(h_prev[a])  // Comparing to actual parent
```

### 6.3 Posterior Weighting

Not all particles are equally trustworthy. Weight by posterior:

```cpp
gradient = Σ W[i] * grad[i]    // Weighted by posterior
```

If SVPF resamples every step (weights become uniform), unweighted is okay.
If SVPF resamples only when ESS low, must use weights.

### 6.4 Stable Parameterization

Direct ρ, σ, ν have boundary issues (clamp artifacts). Use unconstrained:

```
ρ = tanh(η)         η ∈ (-∞, +∞)  →  ρ ∈ (-1, 1)
σ = exp(κ)          κ ∈ (-∞, +∞)  →  σ ∈ (0, +∞)
ν = 2 + exp(κ_ν)    κ_ν ∈ (-∞, +∞) →  ν ∈ (2, +∞)
```

Update η, κ, κ_ν instead of ρ, σ, ν:
```cpp
// Chain rule
d_eta = d_rho * (1 - rho*rho)     // tanh derivative
d_kappa = d_sigma * sigma          // exp derivative  
d_kappa_nu = d_nu * (nu - 2)       // exp derivative (shifted)
```

No boundaries. Smooth everywhere.

### 6.5 Natural Gradient (Fisher Preconditioning)

**Problem with plain gradient:**
```
∇μ = 0.5      (μ is around -2)
∇ρ = 0.001    (ρ is around 0.95)  
∇σ = 2.0      (σ is around 0.15)
∇ν = 0.01     (ν is around 5)

With same learning rate: parameters fight each other.
Need 4 separate learning rates? No - use Natural Gradient.
```

**Solution:** Fisher Information Matrix
```
F = E[(∇log p)(∇log p)ᵀ]    (4×4 matrix)

F⁻¹∇ = gradient in "natural coordinates"
     = automatically scaled by curvature
```

**Why it works:**
- Steep direction (high F) → small step (already sensitive)
- Flat direction (low F) → large step (need more movement)
- Correlations handled (μ-ρ interaction, σ-ν interaction)
- **Single learning rate for all parameters**

**Implementation:**
```cpp
// Accumulate Fisher from particles (outer product of gradients)
float F[4][4] = {0};
for (int i = 0; i < N; i++) {
    float g[4] = {grad_mu[i], grad_eta[i], grad_kappa[i], grad_kappa_nu[i]};
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++)
            F[j][k] += W[i] * g[j] * g[k];
}

// Regularize (prevent singularity)
for (int j = 0; j < 4; j++) F[j][j] += 1e-4f;

// Invert 4×4 (~100 FLOPs, negligible)
float F_inv[4][4];
invert_4x4(F, F_inv);

// Natural gradient
float nat_grad[4] = {0};
for (int j = 0; j < 4; j++)
    for (int k = 0; k < 4; k++)
        nat_grad[j] += F_inv[j][k] * mean_grad[k];

// Single learning rate works for all!
theta += lr * nat_grad;
```

**Cost:** ~100 FLOPs for 4×4 inverse. Negligible.

**Fisher correlations that matter:**
```
F_σν:  σ and ν interact (both explain spread)
       High σ + high ν ≈ Low σ + low ν
       Natural gradient handles this automatically

F_μρ:  μ and ρ interact (both affect mean behavior)
       Natural gradient prevents them fighting
```

### 6.7 Crash Handling: State Machine

During crashes, gradients are garbage (SVPF is stressed, particles scattered).

**Key principle:** Stress triggers should mostly **disable** parameter learning, not accelerate it.

**Three phases:**

```
        │ SHOCK │ RECOVERY │ CALM │
────────┼───────┼──────────┼──────┼─────────────────►
        │       │          │      │              time
Crash   │       │          │      │
detected│       │          │      │
        │       │          │      │
LR:     │  0    │   3×     │  1×  │
        │       │          │      │
Action: │Freeze │Catch up  │Normal│
```

**SHOCK** (~20 ticks): Gradients are garbage. Don't touch θ. Let SVPF handle chaos.

**RECOVERY** (~50 ticks): SVPF has re-centered. Boost LR to catch up quickly.

**CALM**: Normal operation. Steady learning.

**Robust gating condition** (combine multiple signals):
- Standardized return surprise: |y_t| / vol_{t-1}
- Bandwidth (particle spread)
- ESS (if you resample)
- Guide strength (if actively forcing state)

### 6.8 CUDA Graph Compatibility

**Problem:** Your SVPF uses CUDA graph capture. You recapture when μ drifts >0.1 or σ_z >0.05. A θ-tuner updating every tick will constantly invalidate the graph.

**Solution:** Move learnable parameters into **device memory scalars** (like you already did for guide_strength, guide_mean). Kernels read from device memory, not captured constants.

```cpp
// Instead of:
kernel<<<...>>>(mu, rho, sigma, nu);  // Captured at graph time = BAD

// Do:
__device__ float d_mu, d_rho, d_sigma, d_nu;  // Device globals

kernel<<<...>>>();  // Reads d_mu etc. inside = GOOD
// Update d_mu from host without recapture
```

This is the **single biggest integration issue** for online θ adaptation with CUDA graphs.

### 6.9 Regularization (Soft Prior / "Soft Oracle")

To prevent θ from wandering and improve identifiability, add a prior term:

```cpp
// Add to gradient (pulls toward baseline)
grad_mu       += prior_weight * (mu_prior - theta.mu);
grad_eta      += prior_weight * (eta_prior - theta.eta);  
grad_kappa    += prior_weight * (kappa_prior - theta.kappa);
grad_kappa_nu += prior_weight * (kappa_nu_prior - theta.kappa_nu);
```

This is equivalent to:
- Weight decay toward a baseline
- "How strongly do we trust the offline-calibrated values"
- Soft oracle influence without hard injection

Especially helpful when multiple θ configurations explain similar h-behavior.

---

## 7. Complete Algorithm

```cpp
// ═══════════════════════════════════════════════════════════════
//  DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════

// Unconstrained parameterization (4 parameters)
struct Theta {
    float mu;       // μ directly (unbounded)
    float eta;      // ρ = tanh(η)
    float kappa;    // σ = exp(κ)
    float kappa_nu; // ν = 2 + exp(κ_ν)
    
    float rho() const { return tanhf(eta); }
    float sigma() const { return expf(kappa); }
    float nu() const { return 2.0f + expf(kappa_nu); }
};

enum State { SHOCK, RECOVERY, CALM };

struct NaturalGradientTuner {
    Theta theta;
    State state = CALM;
    int ticks_in_state = 0;
    
    // Fisher matrix (EMA smoothed)
    float F[4][4] = {0};
    float F_ema_decay = 0.95f;
    float F_reg = 1e-4f;  // Regularization for invertibility
    
    float base_lr = 0.01f;  // Can be larger with natural gradient!
};

// ═══════════════════════════════════════════════════════════════
//  HELPER: 4x4 MATRIX INVERSE (Cramer's rule or LU, ~100 FLOPs)
// ═══════════════════════════════════════════════════════════════

void invert_4x4(const float A[4][4], float A_inv[4][4]);  // Implementation omitted

// ═══════════════════════════════════════════════════════════════
//  HELPER: DIGAMMA FUNCTION (for ν gradient)
// ═══════════════════════════════════════════════════════════════

__host__ __device__ float digamma(float x) {
    // Approximation valid for x > 0
    float result = 0.0f;
    while (x < 6.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    result += logf(x) - 0.5f/x - 1.0f/(12.0f*x*x);
    return result;
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP (per tick)
// ═══════════════════════════════════════════════════════════════

void tick(
    SVPF* svpf,
    NaturalGradientTuner* tuner,
    float y_t
) {
    // ─────────────────────────────────────────────────────────
    // 1. SNAPSHOT BEFORE SVPF STEP
    // ─────────────────────────────────────────────────────────
    float h_prev[N];
    memcpy(h_prev, svpf->h, N * sizeof(float));
    
    // ─────────────────────────────────────────────────────────
    // 2. SVPF STEP (unchanged, does all the real work)
    // ─────────────────────────────────────────────────────────
    svpf_step(svpf, y_t, tuner->theta.mu, 
              tuner->theta.rho(), tuner->theta.sigma(), tuner->theta.nu());
    
    // svpf now contains:
    //   h[N]         - current particles
    //   ancestor[N]  - parent indices (if resampling)
    //   W[N]         - normalized weights
    
    // ─────────────────────────────────────────────────────────
    // 3. COMPUTE SURPRISE (for state machine)
    // ─────────────────────────────────────────────────────────
    float h_mean = weighted_mean(svpf->h, svpf->W, N);
    float vol = expf(h_mean * 0.5f);
    float z_sq = (y_t * y_t) / (vol * vol);
    
    // ─────────────────────────────────────────────────────────
    // 4. UPDATE STATE MACHINE
    // ─────────────────────────────────────────────────────────
    tuner->ticks_in_state++;
    
    switch (tuner->state) {
        case CALM:
            if (z_sq > 9.0f) {
                tuner->state = SHOCK;
                tuner->ticks_in_state = 0;
            }
            break;
            
        case SHOCK:
            if (tuner->ticks_in_state > 20) {
                tuner->state = RECOVERY;
                tuner->ticks_in_state = 0;
            }
            break;
            
        case RECOVERY:
            if (tuner->ticks_in_state > 50 && z_sq < 4.0f) {
                tuner->state = CALM;
            }
            break;
    }
    
    // ─────────────────────────────────────────────────────────
    // 5. SKIP GRADIENT DURING SHOCK
    // ─────────────────────────────────────────────────────────
    if (tuner->state == SHOCK) return;
    
    // ─────────────────────────────────────────────────────────
    // 6. COMPUTE GRADIENTS (4 parameters)
    // ─────────────────────────────────────────────────────────
    float mu = tuner->theta.mu;
    float rho = tuner->theta.rho();
    float sigma = tuner->theta.sigma();
    float nu = tuner->theta.nu();
    float inv_sig2 = 1.0f / (sigma * sigma);
    
    // Accumulators for mean gradient and Fisher matrix
    float mean_grad[4] = {0};
    float F_new[4][4] = {0};
    
    for (int i = 0; i < N; i++) {
        int a = svpf->ancestor ? svpf->ancestor[i] : i;  // Parent index
        float hp = h_prev[a];                             // Parent's h
        float ht = svpf->h[i];                            // Child's h
        float wi = svpf->W[i];                            // Posterior weight
        
        // === TRANSITION GRADIENTS (μ, ρ, σ) ===
        float eps = ht - mu - rho * (hp - mu);
        
        float d_mu = eps * (1.0f - rho) * inv_sig2;
        float d_rho = eps * (hp - mu) * inv_sig2;
        float d_sigma = -1.0f/sigma + eps*eps/(sigma*sigma*sigma);
        
        // === OBSERVATION GRADIENT (ν) ===
        float vol_i = expf(ht * 0.5f);
        float z_i = y_t / vol_i;
        float z_sq_i = z_i * z_i;
        float A = 1.0f + z_sq_i / nu;
        
        float d_nu = 0.5f * (digamma((nu+1.0f)*0.5f) - digamma(nu*0.5f) 
                     - 1.0f/nu - logf(A) 
                     + (nu + 1.0f) * z_sq_i / (nu * nu * A));
        
        // === CHAIN RULE FOR UNCONSTRAINED PARAMS ===
        float g[4];
        g[0] = d_mu;                              // μ directly
        g[1] = d_rho * (1.0f - rho * rho);        // η via tanh
        g[2] = d_sigma * sigma;                   // κ via exp
        g[3] = d_nu * (nu - 2.0f);                // κ_ν via exp (shifted)
        
        // === ACCUMULATE MEAN GRADIENT ===
        for (int j = 0; j < 4; j++) {
            mean_grad[j] += wi * g[j];
        }
        
        // === ACCUMULATE FISHER MATRIX (outer product) ===
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                F_new[j][k] += wi * g[j] * g[k];
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────
    // 7. EMA SMOOTH FISHER MATRIX
    // ─────────────────────────────────────────────────────────
    float alpha = tuner->F_ema_decay;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            tuner->F[j][k] = alpha * tuner->F[j][k] + (1.0f - alpha) * F_new[j][k];
        }
    }
    
    // Regularize diagonal for invertibility
    for (int j = 0; j < 4; j++) {
        tuner->F[j][j] += tuner->F_reg;
    }
    
    // ─────────────────────────────────────────────────────────
    // 8. INVERT FISHER → NATURAL GRADIENT
    // ─────────────────────────────────────────────────────────
    float F_inv[4][4];
    invert_4x4(tuner->F, F_inv);
    
    float nat_grad[4] = {0};
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            nat_grad[j] += F_inv[j][k] * mean_grad[k];
        }
    }
    
    // ─────────────────────────────────────────────────────────
    // 9. GRADIENT CLIPPING (still useful for safety)
    // ─────────────────────────────────────────────────────────
    float clip = 1.0f;
    for (int j = 0; j < 4; j++) {
        nat_grad[j] = fmaxf(fminf(nat_grad[j], clip), -clip);
    }
    
    // ─────────────────────────────────────────────────────────
    // 10. UPDATE PARAMETERS
    // ─────────────────────────────────────────────────────────
    float lr = tuner->base_lr;
    if (tuner->state == RECOVERY) lr *= 2.0f;  // Smaller boost (natural grad already adaptive)
    
    tuner->theta.mu       += lr * nat_grad[0];
    tuner->theta.eta      += lr * nat_grad[1];
    tuner->theta.kappa    += lr * nat_grad[2];
    tuner->theta.kappa_nu += lr * nat_grad[3];
}
```

---

## 8. SVPF Modifications Required

SVPF needs to expose:

```cpp
struct SVPF {
    float* h;           // [N] particles
    float* W;           // [N] normalized weights  
    int* ancestor;      // [N] ancestor indices from last resample
    
    // ... rest unchanged
};
```

**Changes needed:**

1. **Store h_prev before resample** (or let caller snapshot)
2. **Expose ancestor indices** after resampling
3. **Expose normalized weights** (or guarantee uniform after resample)

Everything else unchanged. SVPF still does Stein transport, Student-t likelihood, etc.

---

## 9. Alternative: Online EM

For AR(1) Gaussian, closed-form updates exist (no learning rate):

```cpp
// Accumulate sufficient statistics (exponential moving average)
S1 = ema(E[h_t])
S2 = ema(E[h_{t-1}])
S3 = ema(E[h_t · h_{t-1}])
S4 = ema(E[h_{t-1}²])
S5 = ema(E[h_t²])

// Closed-form M-step
rho = (S3 - S1*S2) / (S4 - S2*S2)
mu = (S1 - rho*S2) / (1 - rho)
sigma_sq = S5 - 2*rho*S3 + rho*rho*S4 - (1-rho)*(1-rho)*mu*mu
```

**Pros:** No learning rate, no gradient clipping, naturally stable
**Cons:** Still need ancestor pairing, EMA decay is a hyperparameter

Consider if gradient approach is finicky.

---

## 10. Development Phases (Recommended Path)

### Phase A: Minimal Risk - Unify Existing Machinery

**Don't add new learning yet. Clean up what you have.**

| Task | Description |
|------|-------------|
| A.1 | Keep existing adaptive μ Kalman update (it's already a good outer loop) |
| A.2 | Do NOT learn ρ, σ, ν yet |
| A.3 | Create unified shock state machine (SHOCK/RECOVERY/CALM) |
| A.4 | Apply state machine consistently to: adaptive μ gain, σ_z boosting, guide strength |

**Goal:** One coherent regime controller rather than several semi-independent heuristics.

**Deliverable:** Cleaner existing system with unified shock handling.

### Phase B: Add Natural Gradient Infrastructure

| Task | Description |
|------|-------------|
| B.1 | Implement 4×4 Fisher matrix accumulation from particles |
| B.2 | Implement 4×4 matrix inverse (Cramer's rule or LU) |
| B.3 | EMA smoothing of Fisher matrix |
| B.4 | Test on synthetic data: verify gradients and Fisher are correct |

**Goal:** Natural gradient infrastructure ready for use.

**Deliverable:** Working Fisher-preconditioned update (can test with fixed SVPF).

### Phase C: Learn ν (Tail Weight) First

| Task | Description |
|------|-------------|
| C.1 | Add ν gradient (digamma function, observation likelihood) |
| C.2 | Learn ν only, keep μ/ρ/σ fixed |
| C.3 | Test on crash scenarios: does ν adapt appropriately? |
| C.4 | Verify: ν drops during crisis, rises during calm |

**Goal:** ν is the safest parameter to learn (only affects observation, not transition).

**Deliverable:** Self-tuning tail weight.

### Phase D: Learn σ Baseline

| Task | Description |
|------|-------------|
| D.1 | Add σ to learning (transition gradient) |
| D.2 | Keep "breathing filter" as fast multiplicative scaling |
| D.3 | Watch for σ-ν interaction (Fisher handles correlation) |
| D.4 | Test: does σ baseline converge? Does breathing still work? |

**Goal:** Separate slow baseline learning from fast crisis response.

**Deliverable:** σ that self-calibrates while preserving crisis machinery.

### Phase E: Learn ρ (If Needed)

| Task | Description |
|------|-------------|
| E.1 | Add ρ to learning (only in CALM/RECOVERY, not SHOCK) |
| E.2 | Strong regularization to keep ρ in plausible band |
| E.3 | Keep asymmetric ρ (up/down) as fast machinery, not learned |
| E.4 | Test: does ρ learning help or just add instability? |

**Goal:** Only add if clearly beneficial. ρ is often stable enough to fix.

**Deliverable:** Optionally adaptive ρ with guard rails.

### Phase F: Full 4-Parameter Learning

| Task | Description |
|------|-------------|
| F.1 | Enable all 4 parameters: μ, ρ, σ, ν |
| F.2 | Natural gradient handles correlations (σ-ν, μ-ρ) |
| F.3 | Move all θ to device memory for CUDA graph compatibility |
| F.4 | Benchmark latency overhead |

**Goal:** Complete self-tuning system.

**Deliverable:** Production-ready 4-parameter natural gradient learning.

### Phase G: Offline + Online Split

| Task | Description |
|------|-------------|
| G.1 | Overnight calibration: maximize predictive log-likelihood on dataset |
| G.2 | Output strong initialization + regularization targets |
| G.3 | Online: small updates to compensate for drift |
| G.4 | Use offline values as prior/regularization anchor |

**Goal:** Offline gives good operating point; online keeps it aligned.

**Deliverable:** Production split with overnight recalibration capability.

---

## 11. Tuning Guide

### 11.1 Learning Rate (Natural Gradient)

```
base_lr = 0.01 (can be larger than plain SGD due to natural gradient!)

With natural gradient:
  • Fisher matrix normalizes scales automatically
  • Single LR works for all 4 parameters
  • Can typically use 10× larger LR than plain SGD

Too high: θ oscillates
Too low:  θ adapts too slowly after regime change

Symptoms:
  • All params jump together → reduce LR
  • Params stuck after regime change → increase LR
```

### 11.2 Fisher Matrix EMA

```
F_ema_decay = 0.95 (typical)

Higher decay (0.99): More stable Fisher, slower adaptation
Lower decay (0.90):  More responsive, noisier

F_reg = 1e-4 (regularization for invertibility)
  • Too small: Singular matrix, NaN gradients
  • Too large: Reverts toward plain gradient
```

### 11.3 State Machine Thresholds

```
Surprise threshold (entering SHOCK): 9.0 (3σ event)
SHOCK duration: 20 ticks
RECOVERY duration: 50 ticks
Exit RECOVERY threshold: surprise < 4.0

Adjust based on your data frequency:
  • HFT (ms ticks): shorter durations
  • Daily data: longer durations
```

### 11.4 Gradient Clipping

```
clip = 1.0 (per-parameter, applied to natural gradient)

Still useful as safety net even with natural gradient.
Prevents explosion if Fisher becomes ill-conditioned.

If gradients regularly hit clip, consider:
  • Increasing F_reg (more regularization)
  • Reducing base_lr
  • Lengthening SHOCK duration
```

### 11.5 Parameter-Specific Notes

```
ν (tail weight):
  • Only learns from crashes (large |z|)
  • In calm markets, gradient ≈ 0 (uninformative)
  • Fisher F_νν captures this: small in calm, large in crisis
  • Natural gradient automatically learns ν faster during crashes
  
σ-ν interaction:
  • Both explain spread: high σ + high ν ≈ low σ + low ν
  • Fisher captures this correlation (F_σν)
  • Natural gradient prevents them fighting
  
μ-ρ interaction:  
  • Both affect mean behavior
  • Fisher F_μρ handles this
```

---

## 12. What This System Achieves

```
┌─────────────────────────────────────────────────────────────────┐
│  CAPABILITIES                                                   │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Robust h tracking (SVPF handles crashes)                    │
│  ✓ Automatic θ = (μ,ρ,σ,ν) adaptation                          │
│  ✓ Tail weight ν adapts to market regime                       │
│  ✓ Natural gradient: one LR for all parameters                 │
│  ✓ No lag (θ reads from current particles, not history)        │
│  ✓ No memory management (no circular buffers, checkpoints)     │
│  ✓ Simple (gradient feedback, not SMC²/CPMMH)                  │
│  ✓ Fast (O(N) gradient + O(1) 4×4 inverse)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  LIMITATIONS                                                    │
├─────────────────────────────────────────────────────────────────┤
│  ✗ No θ uncertainty (point estimate only)                      │
│  ✗ No posterior over θ                                         │
│  ✗ Not Bayesian                                                 │
│  ✗ Learning rate is a hyperparameter                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Comparison With Alternatives

| Aspect | Fixed θ | SMC² + CPMMH | RMSProp | **Natural Gradient** |
|--------|---------|--------------|---------|---------------------|
| Complexity | None | Very High | Low | Low |
| θ adaptation | None | Full posterior | Point estimate | Point estimate |
| Parameters | Manual | All | Needs per-param LR | **Single LR** |
| Correlations | N/A | Captured | Ignored | **Captured (Fisher)** |
| Lag | N/A | Yes (batch) | No | No |
| Memory | O(N) | O(N × M × L) | O(N) | O(N) + O(16) |
| Crash robust | Yes (SVPF) | Yes | Yes | Yes |
| Arbitrary knobs | Just θ | Q, λ, L, liu-west | LR per param | **Just 1 LR** |

**Natural Gradient Self-Tuning SVPF is the sweet spot:**
- Principled (Fisher-preconditioned)
- One learning rate for 4 parameters
- Handles correlations (σ-ν, μ-ρ)
- Cheap (4×4 matrix inverse ≈ 100 FLOPs)

---

## 14. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE OLD WAY (Oracle + Worker):                                │
│                                                                 │
│    Oracle infers θ from data history                           │
│    → Lag, amnesia problem, arbitrary fixes (Q, λ)              │
│    → Complex (SMC², CPMMH, replay)                             │
│                                                                 │
│  THE NEW WAY (Self-Tuning with Natural Gradient):              │
│                                                                 │
│    SVPF particles already know where h is                      │
│    Natural gradient: "adjust θ, properly scaled"               │
│    Fisher matrix captures parameter correlations               │
│    → No lag, no memory, principled                             │
│                                                                 │
│  ─────────────────────────────────────────────────────────     │
│                                                                 │
│    θ = (μ, ρ, σ, ν) — 4 parameters, 1 learning rate           │
│                                                                 │
│    SVPF is the sensor.                                         │
│    Natural gradient is the actuator.                           │
│    Fisher provides impedance matching.                         │
│    The system self-aligns.                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document version: 3.0*
*Created: January 2026*
*Paradigm: AGC-style feedback with Natural Gradient (Fisher-preconditioned)*
*Parameters: θ = (μ, ρ, σ, ν) — mean, persistence, vol-of-vol, tail weight*