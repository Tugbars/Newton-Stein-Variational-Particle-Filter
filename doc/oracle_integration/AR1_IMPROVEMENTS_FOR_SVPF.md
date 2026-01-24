# AR(1) Improvements for SVPF

## Executive Summary

The AR(1) transition in SVPF is **not a volatility model**. It's a **proposal geometry** that Stein transport corrects. Improvements must preserve Stein's authority, not compete with it.

**Current extensions (keep):** Asymmetric ρ, MIM jumps
**Recommended addition:** State-dependent σ_z
**Probably skip:** Weak anchoring to particle mean

---

## 1. What AR(1) Actually Does in SVPF

### 1.1 The Transition

```
h_t = μ + ρ·(h_{t-1} - μ) + σ_z·ε_t
```

This is **not** trying to be a perfect model of volatility dynamics.

### 1.2 The Three Structural Roles

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. LOCAL CONTINUITY PRIOR                                     │
│     Prevents particles from teleporting arbitrarily            │
│     "h_t should be somewhere near h_{t-1}"                     │
│                                                                 │
│  2. MEMORY TIMESCALE                                           │
│     Encodes how fast volatility forgets the past               │
│     ρ close to 1 = slow decay, long memory                     │
│     ρ close to 0 = fast decay, short memory                    │
│                                                                 │
│  3. PROPOSAL GEOMETRY                                          │
│     Provides a reference direction for Stein to correct        │
│     "Start here, then Stein will fix it"                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 The Crucial Insight

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE AR(1) IS ALLOWED TO BE WRONG.                             │
│                                                                 │
│  Stein transport + likelihood correction exist                  │
│  BECAUSE it will be wrong.                                     │
│                                                                 │
│  "Improving AR(1)" does NOT mean                               │
│  "making it more realistic at all costs."                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. What "Improvement" Actually Means

### 2.1 Requirements for Valid Improvements

An improvement must satisfy **ALL** of these:

| Requirement | Why |
|-------------|-----|
| Does not reduce Stein's corrective authority | Stein is the real worker |
| Does not introduce hard regime commitments | Regimes collapse particles |
| Does not increase parameter identifiability problems | Unlearnable = useless |
| Does not create collapse modes | Filter must survive anything |
| Preserves fast recovery from misspecification | Wrong params must be survivable |

### 2.2 What This Rules Out

```
❌ Explicit regime-switching states
   → Particles commit to regimes, collapse when wrong
   
❌ GARCH-style h_t depends on y_{t-1}
   → Creates feedback loops, identifiability nightmare
   
❌ Multi-factor models (2D, 3D state)
   → Curse of dimensionality, Stein struggles
   
❌ Long-memory / fractional models
   → Requires storing long history, complex
   
❌ Anything that "knows better" than Stein
   → If AR(1) is too smart, Stein has nothing to correct
```

---

## 3. Improvements That Make Sense

### 3.1 State-Dependent Noise (Heteroskedastic Transition)

**Status: ➕ RECOMMENDED ADDITION**

Instead of constant σ_z:
```
σ_z = constant  ← Current
```

Use state-dependent:
```
σ_z(h_{t-1}) = σ_base · (1 + α·|h_{t-1} - μ|)
```

**Why this works:**

```
When h is far from μ (extreme volatility):
  → We're uncertain where it goes next
  → Proposal should be WIDER
  → Gives Stein more room to explore
  → Doesn't commit to a direction, just admits uncertainty

When h is near μ (normal volatility):
  → More predictable dynamics  
  → Tighter proposal is fine
  → Stein can still correct if needed
```

**Interpretation:** "When volatility is already high, allow bigger exploratory moves."

This is a **proposal improvement**, not a regime model.

**Implementation:**

```cpp
// In transition step
float h_dev = fabsf(h_prev[i] - mu);
float sigma_z_local = sigma_z_base * (1.0f + alpha * fminf(h_dev, cap));

h[i] = mu + rho * (h_prev[i] - mu) + sigma_z_local * noise[i];
```

**Parameters:**
```
alpha = 0.2   (20% wider per unit deviation from mean)
cap = 3.0     (maximum multiplier = 1 + 0.2*3 = 1.6×)
```

**Why the cap:** Prevents explosion when h is extreme. Without cap, 5σ deviation → 2× width, which may be too much.

**Analogy:** This is what good MCMC proposals do — adapt width to local uncertainty.

---

### 3.2 Asymmetric Persistence

**Status: ✅ ALREADY HAVE — KEEP**

```
ρ_up   for h_{t-1} < h_t  (volatility rising)
ρ_down for h_{t-1} > h_t  (volatility falling)
```

Typically ρ_down > ρ_up: volatility drops slower than it spikes (leverage effect).

**Why this is excellent:**

| Benefit | Explanation |
|---------|-------------|
| Captures leverage/panic asymmetry | Market reality |
| Keeps model 1-dimensional | No state explosion |
| Minimal parameter cost | Just 2 instead of 1 |
| Strong empirical payoff | Well-documented effect |
| Stein-compatible | Still just a proposal, Stein corrects |

**Implementation:**

```cpp
float rho_effective = (h_prev[i] < h_target) ? rho_up : rho_down;
h[i] = mu + rho_effective * (h_prev[i] - mu) + sigma_z * noise[i];
```

This is **strictly better** than many GARCH-like extensions in SVPF context.

---

### 3.3 Innovation Mixture (MIM / Jump-Augmented)

**Status: ✅ ALREADY HAVE — KEEP**

```
ε_t ~ (1-p)·N(0, σ²) + p·N(0, (k·σ)²)

Where:
  p = jump probability (~0.05)
  k = jump multiplier (~3-5)
```

**Why this is exactly right:**

| Benefit | Explanation |
|---------|-------------|
| Captures rare jumps | Without explicit jump states |
| No persistent regime variable | Nothing to collapse on |
| Stein handles localization | Jumps are just wider noise |
| Simple to implement | Just mixture of Gaussians |

**Implementation:**

```cpp
float u = uniform_rng();
float sigma_effective = (u < p_jump) ? (k_jump * sigma_z) : sigma_z;
h[i] = mu + rho * (h_prev[i] - mu) + sigma_effective * normal_rng();
```

This is **vastly safer** than explicit jump states or regime-switching.

---

### 3.4 Weak Mean Reversion Anchoring

**Status: ❓ HESITANT — TEST BEFORE ADOPTING**

```
h_t = μ + ρ(h_{t-1} - μ) - λ(h_{t-1} - h̄_{t-1}) + σε_t
                          ↑
              Weak pull toward particle mean
```

Where:
- h̄_{t-1} = particle mean from previous step
- λ ≪ 1 (very weak, ~0.01)

**Claimed benefits:**
- Prevents long-run drift
- Keeps particles coherent during extended chaos
- Does not override likelihood

**My concerns:**

```
┌─────────────────────────────────────────────────────────────────┐
│  CONCERN 1: ISN'T THIS STEIN'S JOB?                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stein transport already provides:                              │
│    • Likelihood attraction (toward good regions)               │
│    • Repulsion (stay spread)                                   │
│                                                                 │
│  Anchoring does cohesion in the PROPOSAL.                      │
│  But cohesion should emerge from LIKELIHOOD + STEIN.           │
│                                                                 │
│  Adding it to proposal is redundant at best,                   │
│  counterproductive at worst.                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  CONCERN 2: COULD HIDE REAL UNCERTAINTY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario:                                                     │
│    True posterior is bimodal (crash vs recovery paths)         │
│    Particles start exploring both modes                        │
│    Anchoring pulls them back to unimodal mean                  │
│    → Lost the bimodality signal                                │
│                                                                 │
│  During chaos, particles SHOULD scatter if there's             │
│  genuine uncertainty about the path forward.                   │
│                                                                 │
│  Artificially pulling them together might look stable          │
│  but be LYING about uncertainty.                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  CONCERN 3: CIRCULAR DEPENDENCY                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  If using current particle mean h̄_t:                          │
│    h_t depends on h̄_t                                         │
│    h̄_t depends on h_t                                         │
│    → Circular                                                  │
│                                                                 │
│  Must use h̄_{t-1} (previous step's mean) to avoid this.       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**When it might help:**

```
Extended chaos (100+ ticks of crisis):
  - Particles drift apart due to accumulated noise
  - Not because of real bimodality
  - Just random walk divergence
  - Weak anchoring prevents this drift

But: If particles drift apart, doesn't Stein repulsion 
     already handle this? The repulsion kernel keeps 
     particles from collapsing, not from drifting.
     
     Actually... drift might be the one thing Stein 
     doesn't prevent. Repulsion prevents collapse,
     likelihood prevents wrong location, but neither
     prevents slow drift if likelihood is flat.
```

**Verdict:** 

Try it with λ = 0.01 (very weak) and monitor:
- Does it improve RMSE? (good)
- Does it reduce particle diversity? (bad)
- Does it hide real bimodality? (bad)
- Does it help during extended crises? (good)

If marginal or unclear benefit, don't add the complexity.

---

## 4. Summary: What To Do

### 4.1 Recommendation Table

| Extension | Status | Action | Risk | Benefit |
|-----------|--------|--------|------|---------|
| Asymmetric ρ | ✅ Have | Keep | None | High |
| MIM jumps | ✅ Have | Keep | None | High |
| State-dependent σ_z | ❌ Don't have | **Add** | Low | Medium |
| Weak anchoring | ❌ Don't have | Test, probably skip | Medium | Unclear |

### 4.2 What To Avoid

| Extension | Why Avoid |
|-----------|-----------|
| Explicit regime states | Collapse risk, Stein can't fix wrong commitment |
| GARCH feedback | Identifiability nightmare, circular dependencies |
| Multi-factor (2D+) | Curse of dimensionality |
| Long-memory models | Complexity without clear SVPF benefit |
| Anything smarter than Stein | If proposal is "right", Stein has nothing to do |

### 4.3 The Guiding Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  AR(1) should be a HUMBLE PROPOSAL.                            │
│                                                                 │
│  It says: "I think h_t is roughly here, with this spread."    │
│                                                                 │
│  Stein says: "Thanks, I'll take it from here."                 │
│              *moves particles to where they should be*         │
│                                                                 │
│  If AR(1) tries to be too smart, it fights Stein.             │
│  If AR(1) is too dumb, Stein works harder but still wins.     │
│                                                                 │
│  Err on the side of humble.                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. State-Dependent σ_z: Detailed Design

### 5.1 The Formula

```
σ_z(h) = σ_base · (1 + α · min(|h - μ|, cap))
```

### 5.2 Parameters

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| σ_base | 0.15 | 0.05-0.3 | Base transition noise |
| α | 0.2 | 0.1-0.4 | Scaling factor |
| cap | 3.0 | 2.0-5.0 | Maximum deviation to consider |

### 5.3 Behavior

```
|h - μ| = 0   →  σ_z = σ_base · 1.0  = σ_base
|h - μ| = 1   →  σ_z = σ_base · 1.2  = 1.2 × σ_base
|h - μ| = 2   →  σ_z = σ_base · 1.4  = 1.4 × σ_base
|h - μ| = 3+  →  σ_z = σ_base · 1.6  = 1.6 × σ_base (capped)
```

### 5.4 Implementation

```cpp
__device__ float compute_adaptive_sigma_z(
    float h_prev,
    float mu,
    float sigma_z_base,
    float alpha,
    float cap
) {
    float h_dev = fabsf(h_prev - mu);
    float h_dev_capped = fminf(h_dev, cap);
    return sigma_z_base * (1.0f + alpha * h_dev_capped);
}

// In transition kernel:
float sigma_z_local = compute_adaptive_sigma_z(h_prev[i], mu, sigma_z_base, alpha, cap);
float h_pred = mu + rho * (h_prev[i] - mu);
h[i] = h_pred + sigma_z_local * noise[i];
```

### 5.5 Interaction with Existing Features

```
┌─────────────────────────────────────────────────────────────────┐
│  BREATHING FILTER (existing):                                   │
│    σ_z_effective = σ_z_base · boost(t)                         │
│    boost(t) based on recent surprise                           │
│                                                                 │
│  STATE-DEPENDENT (new):                                        │
│    σ_z_effective = σ_z_base · (1 + α|h-μ|)                    │
│    Based on current particle position                          │
│                                                                 │
│  COMBINED:                                                     │
│    σ_z_effective = σ_z_base · boost(t) · (1 + α|h-μ|)        │
│                    ↑          ↑           ↑                    │
│                 baseline   temporal    spatial                 │
│                                                                 │
│  Both are multiplicative, both are "proposal width" adjustments│
│  They compose naturally.                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 Should We Learn α?

**Option A: Fixed α = 0.2**
- Simple
- One less parameter to identify
- Probably good enough

**Option B: Learn α online**
- Add to θ: (μ, ρ, σ_base, α, ν) → 5 parameters
- Fisher matrix becomes 5×5 (still cheap)
- Risk: α and σ_base might fight (both control spread)

**Recommendation:** Start with fixed α. Only learn if:
- Clear evidence it matters
- σ_base and α don't create identifiability issues

---

## 6. Relationship to Self-Tuning SVPF

### 6.1 What Gets Learned

```
Self-Tuning SVPF learns: θ = (μ, ρ, σ_base, ν)

AR(1) improvements are FIXED STRUCTURE:
  - Asymmetric ρ: ρ_up, ρ_down (could learn, currently fixed)
  - MIM: p_jump, k_jump (fixed)
  - State-dependent σ_z: α, cap (fixed)
```

### 6.2 The Separation

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LEARNED (slow adaptation):                                    │
│    μ     - mean level                                          │
│    ρ     - persistence (or ρ_up, ρ_down)                      │
│    σ_base - base transition noise                              │
│    ν     - observation tail weight                             │
│                                                                 │
│  FIXED (structural choices):                                   │
│    α, cap        - state-dependent noise shape                 │
│    p_jump, k_jump - MIM parameters                             │
│    Stein steps   - transport iterations                        │
│    Kernel bandwidth - repulsion strength                       │
│                                                                 │
│  The learned parameters adapt to market regime.                │
│  The fixed structure defines "how SVPF works."                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Why Not Learn Everything?

```
More parameters = more identifiability problems

σ_base and α both control spread:
  High σ_base, low α  ≈  Low σ_base, high α
  
Learning both → they fight, convergence issues

Same for:
  ρ and μ (both affect mean behavior)
  σ and ν (both explain observation spread)
  
Natural gradient helps with correlations,
but can't fix fundamental unidentifiability.

Rule: Learn the minimum set that captures regime variation.
      Fix the rest at sensible defaults.
```

---

## 7. Testing Plan

### 7.1 State-Dependent σ_z

| Test | Metric | Expected |
|------|--------|----------|
| Synthetic calm | RMSE | Same or slightly better |
| Synthetic crash | RMSE | Better (wider proposal helps) |
| Real data (calm period) | NLL | Same |
| Real data (crisis) | NLL | Better |
| Particle diversity | Bandwidth | Same or higher |

### 7.2 Weak Anchoring (If Testing)

| Test | Metric | Expected | Red Flag |
|------|--------|----------|----------|
| Extended chaos (100+ ticks) | RMSE | Better? | Diversity drops |
| Bimodal synthetic | Mode capture | Same | Collapses to unimodal |
| Real data | NLL | Same or better | Worse = bad |

---

## 8. Final Recommendations

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IMMEDIATE (Phase A):                                          │
│    • Add state-dependent σ_z with α=0.2, cap=3.0              │
│    • Keep existing asymmetric ρ, MIM                           │
│    • Test on synthetic + real data                             │
│                                                                 │
│  LATER (if needed):                                            │
│    • Test weak anchoring (λ=0.01) on extended chaos           │
│    • Only adopt if clear benefit without diversity loss        │
│                                                                 │
│  AVOID:                                                        │
│    • Regime states                                             │
│    • GARCH feedback                                            │
│    • Multi-factor models                                       │
│    • Anything that competes with Stein                         │
│                                                                 │
│  GUIDING PRINCIPLE:                                            │
│    AR(1) is a humble proposal.                                 │
│    Stein is the real worker.                                   │
│    Don't let the proposal get too clever.                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document version: 1.0*
*Created: January 2026*
*Related: SELF_TUNING_SVPF.md*
