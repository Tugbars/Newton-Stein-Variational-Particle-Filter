# Self-Tuning SVPF: Development Plan

## Executive Summary

A closed-loop system where:
- **SVPF** tracks volatility h (fast, robust, handles everything)
- **Gradient feedback** slowly adjusts θ = (μ, ρ, σ) to match what SVPF is doing

No oracle. No batch inference. No memory. Just negative feedback like an AGC.

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
│    │   • Student-t likelihood (crash robust)            │     │
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
│    │   Gradient Feedback (slow loop)                    │     │
│    │                                                     │     │
│    │   • Measures: particles - prediction               │     │
│    │   • Computes: ∇θ log p(h_t | h_{t-1}, θ)          │     │
│    │   • Updates: θ += lr · gradient                    │     │
│    │   • State machine: SHOCK → RECOVERY → CALM         │     │
│    │                                                     │     │
│    └──────────────────────┬──────────────────────────────┘     │
│                           │                                     │
│                           │ θ = (μ, ρ, σ)                      │
│                           ▼                                     │
│                      back to SVPF                               │
│                                                                 │
│    ─────────────────────────────────────────────────────────   │
│    Closed loop. No external oracle. No memory.                 │
│    θ tracks SVPF. SVPF tracks reality. System self-aligns.    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. What The Gradient Measures

### 5.1 The Transition Model

```
h_t = μ + ρ·(h_{t-1} - μ) + σ·noise
      ├────────────────────┤
            prediction
```

Prediction error:
```
ε = h_t - prediction = h_t - μ - ρ·(h_{t-1} - μ)
```

### 5.2 Blame Assignment

The gradient asks: "If ε is big, which parameter is wrong?"

```
μ (mean level):
  "Are particles systematically above or below prediction?"
  
  Particles here:    ●●●●●
  Prediction here:         ○
  
  ε > 0 consistently → raise μ
  ε < 0 consistently → lower μ
  
  Gradient: ∂/∂μ = ε·(1-ρ) / σ²


ρ (persistence):  
  "Do particles mean-revert faster or slower than predicted?"
  
  If h_{t-1} was high and h_t didn't drop as much as predicted:
  → ρ is too low, raise it
  
  Gradient: ∂/∂ρ = ε·(h_{t-1} - μ) / σ²


σ (volatility):
  "Is actual scatter bigger or smaller than predicted?"
  
  |ε|² > σ² consistently → raise σ (more noise than expected)
  |ε|² < σ² consistently → lower σ (less noise than expected)
  
  Gradient: ∂/∂σ = -1/σ + ε²/σ³
```

### 5.3 Direct Measurement, Not Inference

```
BATCH LEARNER:
  "Given 10,000 observations, what θ maximizes likelihood?"
  → Historical inference, lags behind
  
AGC GRADIENT:
  "Right now, are particles where θ predicts?"
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

Direct ρ, σ have boundary issues (clamp artifacts). Use unconstrained:

```
ρ = tanh(η)     η ∈ (-∞, +∞)  →  ρ ∈ (-1, 1)
σ = exp(κ)      κ ∈ (-∞, +∞)  →  σ ∈ (0, +∞)
```

Update η, κ instead of ρ, σ:
```cpp
// Chain rule
d_eta = d_rho * (1 - rho*rho)   // tanh derivative
d_kappa = d_sigma * sigma        // exp derivative
```

No boundaries. Smooth everywhere.

### 6.5 Crash Handling: State Machine

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

### 6.6 CUDA Graph Compatibility

**Problem:** Your SVPF uses CUDA graph capture. You recapture when μ drifts >0.1 or σ_z >0.05. A θ-tuner updating every tick will constantly invalidate the graph.

**Solution:** Move learnable parameters into **device memory scalars** (like you already did for guide_strength, guide_mean). Kernels read from device memory, not captured constants.

```cpp
// Instead of:
kernel<<<...>>>(mu, rho, sigma);  // Captured at graph time = BAD

// Do:
__device__ float d_mu, d_rho, d_sigma;  // Device globals

kernel<<<...>>>();  // Reads d_mu etc. inside = GOOD
// Update d_mu from host without recapture
```

This is the **single biggest integration issue** for online θ adaptation with CUDA graphs.

### 6.7 Regularization (Soft Prior / "Soft Oracle")

To prevent θ from wandering and improve identifiability, add a prior term:

```cpp
// Add to gradient (pulls toward baseline)
grad_mu    += prior_weight * (mu_prior - theta.mu);
grad_eta   += prior_weight * (eta_prior - theta.eta);  
grad_kappa += prior_weight * (kappa_prior - theta.kappa);
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

// Unconstrained parameterization
struct Theta {
    float mu;      // Mean level (unbounded)
    float eta;     // ρ = tanh(η)
    float kappa;   // σ = exp(κ)
    
    float rho() const { return tanhf(eta); }
    float sigma() const { return expf(kappa); }
};

enum State { SHOCK, RECOVERY, CALM };

struct GradientTuner {
    Theta theta;
    State state = CALM;
    int ticks_in_state = 0;
    
    // RMSProp
    float v_mu = 0, v_eta = 0, v_kappa = 0;
    float decay = 0.99f;
    float eps = 1e-6f;
    float base_lr = 0.001f;
};

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP (per tick)
// ═══════════════════════════════════════════════════════════════

void tick(
    SVPF* svpf,
    GradientTuner* tuner,
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
              tuner->theta.rho(), tuner->theta.sigma());
    
    // svpf now contains:
    //   h[N]         - current particles
    //   ancestor[N]  - parent indices
    //   W[N]         - normalized weights
    
    // ─────────────────────────────────────────────────────────
    // 3. COMPUTE SURPRISE (for state machine)
    // ─────────────────────────────────────────────────────────
    float h_mean = weighted_mean(svpf->h, svpf->W, N);
    float vol = expf(h_mean * 0.5f);
    float surprise = (y_t * y_t) / (vol * vol);
    
    // ─────────────────────────────────────────────────────────
    // 4. UPDATE STATE MACHINE
    // ─────────────────────────────────────────────────────────
    tuner->ticks_in_state++;
    
    switch (tuner->state) {
        case CALM:
            if (surprise > 9.0f) {
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
            if (tuner->ticks_in_state > 50 && surprise < 4.0f) {
                tuner->state = CALM;
            }
            break;
    }
    
    // ─────────────────────────────────────────────────────────
    // 5. SKIP GRADIENT DURING SHOCK
    // ─────────────────────────────────────────────────────────
    if (tuner->state == SHOCK) return;
    
    // ─────────────────────────────────────────────────────────
    // 6. COMPUTE GRADIENTS
    // ─────────────────────────────────────────────────────────
    float mu = tuner->theta.mu;
    float rho = tuner->theta.rho();
    float sigma = tuner->theta.sigma();
    float inv_sig2 = 1.0f / (sigma * sigma);
    
    float g_mu = 0, g_eta = 0, g_kappa = 0;
    
    for (int i = 0; i < N; i++) {
        int a = svpf->ancestor[i];          // CRITICAL: actual parent
        float hp = h_prev[a];               // Parent's h
        float ht = svpf->h[i];              // Child's h
        float wi = svpf->W[i];              // Posterior weight
        
        // Prediction error
        float eps = ht - mu - rho * (hp - mu);
        
        // Gradients w.r.t. original params
        float d_mu = eps * (1.0f - rho) * inv_sig2;
        float d_rho = eps * (hp - mu) * inv_sig2;
        float d_sigma = -1.0f/sigma + eps*eps/(sigma*sigma*sigma);
        
        // Chain rule for unconstrained params
        float d_eta = d_rho * (1.0f - rho * rho);
        float d_kappa = d_sigma * sigma;
        
        // Accumulate weighted
        g_mu += wi * d_mu;
        g_eta += wi * d_eta;
        g_kappa += wi * d_kappa;
    }
    
    // ─────────────────────────────────────────────────────────
    // 7. GRADIENT CLIPPING
    // ─────────────────────────────────────────────────────────
    float clip = 1.0f;
    g_mu = fmaxf(fminf(g_mu, clip), -clip);
    g_eta = fmaxf(fminf(g_eta, clip), -clip);
    g_kappa = fmaxf(fminf(g_kappa, clip), -clip);
    
    // ─────────────────────────────────────────────────────────
    // 8. RMSPROP UPDATE
    // ─────────────────────────────────────────────────────────
    float d = tuner->decay;
    tuner->v_mu = d * tuner->v_mu + (1-d) * g_mu * g_mu;
    tuner->v_eta = d * tuner->v_eta + (1-d) * g_eta * g_eta;
    tuner->v_kappa = d * tuner->v_kappa + (1-d) * g_kappa * g_kappa;
    
    float lr = tuner->base_lr;
    if (tuner->state == RECOVERY) lr *= 3.0f;
    
    tuner->theta.mu += lr * g_mu / (sqrtf(tuner->v_mu) + tuner->eps);
    tuner->theta.eta += lr * g_eta / (sqrtf(tuner->v_eta) + tuner->eps);
    tuner->theta.kappa += lr * g_kappa / (sqrtf(tuner->v_kappa) + tuner->eps);
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
| A.2 | Do NOT learn ρ and σ_z yet |
| A.3 | Create unified shock state machine (SHOCK/RECOVERY/CALM) |
| A.4 | Apply state machine consistently to: adaptive μ gain, σ_z boosting, guide strength |

**Goal:** One coherent regime controller rather than several semi-independent heuristics.

**Deliverable:** Cleaner existing system with unified shock handling.

### Phase B: Learn σ_z Baseline

| Task | Description |
|------|-------------|
| B.1 | Learn σ_z baseline only (outer loop, slow) |
| B.2 | Keep "breathing filter" as fast multiplicative scaling around baseline |
| B.3 | `σ_z_effective(t) = σ_z_baseline(t) · boost(t)` |
| B.4 | Test: does σ_z baseline converge? Does breathing still work? |

**Goal:** Separate slow baseline learning from fast crisis response.

**Deliverable:** σ_z that self-calibrates while preserving crisis machinery.

### Phase C: Learn ρ (If Needed)

| Task | Description |
|------|-------------|
| C.1 | Learn ρ baseline (only in CALM/RECOVERY, not SHOCK) |
| C.2 | Strong regularization to keep ρ in plausible band |
| C.3 | Keep asymmetric ρ (up/down) as fast machinery, not learned |
| C.4 | Test: does ρ learning help or just add instability? |

**Goal:** Only add if clearly beneficial. ρ is often stable enough to fix.

**Deliverable:** Optionally adaptive ρ with guard rails.

### Phase D: Full Score-Based Learning (Optional)

| Task | Description |
|------|-------------|
| D.1 | Implement proper particle score kernel (Fisher identity) |
| D.2 | Per-particle gradient contributions, reduced to update |
| D.3 | Move all θ to device memory for CUDA graph compatibility |
| D.4 | Benchmark latency overhead |

**Goal:** Most "principled" route without SMC²/PMCMC.

**Deliverable:** Full gradient-based learning if phases A-C insufficient.

### Phase E: Offline + Online Split

| Task | Description |
|------|-------------|
| E.1 | Overnight calibration: maximize predictive log-likelihood on dataset |
| E.2 | Output strong initialization + regularization targets |
| E.3 | Online: small updates to compensate for drift |
| E.4 | Use offline values as prior/regularization anchor |

**Goal:** Offline gives good operating point; online keeps it aligned.

**Deliverable:** Production split with overnight recalibration capability.

---

## 11. Tuning Guide

### 11.1 Learning Rate

```
base_lr = 0.001 (conservative start)

Too high: θ oscillates, chases noise
Too low:  θ adapts too slowly after regime change

Symptoms of wrong LR:
  • θ jumps around → reduce LR
  • θ stuck after regime change → increase LR or check state machine
```

### 11.2 State Machine Thresholds

```
Surprise threshold (entering SHOCK): 9.0 (3σ event)
SHOCK duration: 20 ticks
RECOVERY duration: 50 ticks
Exit RECOVERY threshold: surprise < 4.0

Adjust based on your data frequency:
  • HFT (ms ticks): shorter durations
  • Daily data: longer durations
```

### 11.3 RMSProp

```
decay = 0.99 (standard)
eps = 1e-6 (numerical stability)

Higher decay = more smoothing, slower adaptation
Lower decay = more responsive, noisier
```

### 11.4 Gradient Clipping

```
clip = 1.0 (per-parameter)

Prevents explosion during volatile periods.
If gradients regularly hit clip, consider:
  • Reducing base_lr
  • Lengthening SHOCK duration
```

---

## 12. What This System Achieves

```
┌─────────────────────────────────────────────────────────────────┐
│  CAPABILITIES                                                   │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Robust h tracking (SVPF handles crashes)                    │
│  ✓ Automatic θ adaptation (no manual recalibration)            │
│  ✓ No lag (θ reads from current particles, not history)        │
│  ✓ No memory management (no circular buffers, checkpoints)     │
│  ✓ Simple (gradient feedback, not SMC²/CPMMH)                  │
│  ✓ Fast (O(N) gradient, not O(N²) or O(N×T))                  │
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

| Aspect | Fixed θ | SMC² + CPMMH | Self-Tuning SVPF |
|--------|---------|--------------|------------------|
| Complexity | None | Very High | Low |
| θ adaptation | None | Full posterior | Point estimate |
| Lag | N/A | Yes (batch) | No (feedback) |
| Memory | O(N) | O(N × M × L) | O(N) |
| Crash robust | Yes (SVPF) | Yes | Yes |
| Arbitrary knobs | Just θ | Q, λ, L, liu-west | Just LR |

**Self-Tuning SVPF is the sweet spot for most applications.**

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
│  THE NEW WAY (Self-Tuning):                                    │
│                                                                 │
│    SVPF particles already know where h is                      │
│    Gradient measures: "Is θ consistent with particles?"        │
│    θ adjusts to match → Closed-loop negative feedback          │
│    → No lag, no memory, simple                                 │
│                                                                 │
│  ─────────────────────────────────────────────────────────     │
│                                                                 │
│    SVPF is the sensor.                                         │
│    Gradient is the actuator.                                   │
│    The system self-aligns.                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document version: 2.0*
*Created: January 2026*
*Paradigm: AGC-style feedback, not batch inference*