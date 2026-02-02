# Adaptive Step Size Methods for Stein Transport

## The Step Size Problem

In Stein variational methods, the update is:

```
h_new = h + ε · φ(h)
```

Fixed ε is fragile:
- **Too large** → overshoot, oscillation, particle collapse
- **Too small** → slow convergence, wasted compute

The question: how to adapt ε automatically?

---

## Method Comparison

| Method | Cost | Latency Impact | When It Helps |
|--------|------|----------------|---------------|
| Fixed ε | O(1) | Baseline | Never adapts |
| Newton (ε≈1) | O(n·d³) | +10-20% | Built-in curvature scaling |
| Barzilai-Borwein | O(d) | ~0% | Transitions |
| ESS-based | O(n) | ~0% | Particle health monitoring |
| KSD line search | O(n²) per attempt | **+30%** | Overkill |

---

## 1. Newton Preconditioning = Implicit Adaptivity

If you're using Newton preconditioning:

```
φ_newton(h) = H⁻¹(h) · φ(h)
```

Then **ε ≈ 1 is already adaptive**. The Hessian inverse absorbs the curvature:
- High curvature region → H large → H⁻¹ small → effective step shrinks
- Low curvature region → H small → H⁻¹ large → effective step grows

This is the Detommaso (2018) argument. Newton makes ε=1 a natural choice.

**Verdict:** If you have Newton, fixed ε=1 is fine. The "adaptivity" is in the preconditioner.

---

## 2. Barzilai-Borwein (Cheap History-Based)

Uses previous step to estimate local curvature:

```
Δh = h_t - h_{t-1}
Δφ = φ_t - φ_{t-1}

ε_t = <Δh, Δφ> / <Δφ, Δφ>   // "short" step
// or
ε_t = <Δh, Δh> / <Δh, Δφ>   // "long" step
```

**Cost:** Two dot products. Essentially free.

**Pros:**
- Nearly zero overhead
- Adapts to local curvature without Hessians
- Works well for smooth objectives

**Cons:**
- Needs 1-step history (can't use on first iteration)
- Can be unstable - often needs safeguards (ε_min, ε_max)
- Doesn't help if curvature changes faster than step-to-step

**Implementation:**
```c
if (step > 0) {
    float dh = h - h_prev;
    float dphi = phi - phi_prev;
    float dh_dphi = dh * dphi;
    float dphi_dphi = dphi * dphi;
    
    if (dphi_dphi > 1e-10f) {
        epsilon = fminf(fmaxf(dh_dphi / dphi_dphi, 0.01f), 2.0f);
    }
}
```

---

## 3. ESS-Based Scaling

Monitor particle health via Effective Sample Size:

```
ESS = (Σ wᵢ)² / Σ wᵢ²
```

If ESS drops → step was too aggressive:

```c
float ess_ratio = ESS / N;

if (ess_ratio < 0.3f) {
    epsilon *= 0.7f;  // back off significantly
} else if (ess_ratio < 0.5f) {
    epsilon *= 0.9f;  // back off slightly  
} else if (ess_ratio > 0.8f) {
    epsilon *= 1.05f; // can be slightly more aggressive
}

epsilon = clamp(epsilon, 0.1f, 2.0f);
```

**Cost:** O(n) - you're probably computing ESS anyway for resampling decisions.

**Pros:**
- Direct measure of particle health
- Zero extra kernel evaluations
- Catches catastrophic steps

**Cons:**
- Reactive, not predictive (damage already done when ESS drops)
- ESS can drop for reasons other than step size (likelihood shock)

---

## 4. KSD-Based Line Search (Expensive)

Since KSD measures distance to target, use it for line search:

```c
float epsilon_try = epsilon_base;

for (int attempt = 0; attempt < 3; attempt++) {
    // Trial step
    h_candidate = h + epsilon_try * phi;
    
    // Compute KSD at candidate (expensive!)
    float ksd_candidate = compute_ksd(h_candidate);
    
    if (ksd_candidate < ksd_current) {
        accept(h_candidate);
        break;
    }
    epsilon_try *= 0.5f;  // backtrack
}
```

**Cost:** O(n²) per line search attempt. With 2-3 attempts, this **multiplies latency by 30%+**.

**Verdict:** Overkill for HFT. The information gain doesn't justify the cost.

---

## 5. Fixed Steps with Adaptive Annealing (Current Approach)

Instead of adapting ε, adapt β (likelihood temperature):

```
h_new = h + ε · φ_β(h)
```

where φ_β uses annealed likelihood ∝ p(r|h)^β.

- Calm market: β = 1 (full likelihood)
- Flash crash: β < 1 (tempered likelihood, gentler gradients)

**This is what your adaptive annealing already does.** The "step size" is effectively controlled by tempering the likelihood rather than scaling ε directly.

---

## Recommendation for HFT SVPF

**Keep it simple:**

1. **Newton preconditioning** → ε ≈ 1 is natural
2. **Adaptive annealing** → handles crashes via β, not ε
3. **Fixed Stein steps** (your current 8) → predictable latency

The 30% latency hit from KSD line search isn't worth it. You're already handling the hard cases (crashes) through annealing, and Newton makes ε=1 well-scaled for normal conditions.

If you want cheap insurance, add **Barzilai-Borwein with clamps** - it's nearly free and can help during regime transitions. But it's probably not necessary given your current stack.

---

## Summary

```
                    Adaptive Step Size Methods
                    
    Cost:     Free ----+----+----+----+----+---- Expensive
                       |    |    |    |    |
              Newton   BB   ESS  |    |    KSD
              (ε=1)              |    |    Line Search
                                 |    |    (+30% latency)
                                 |    |
                        [Your current approach]
                        Fixed ε + Adaptive Annealing
                        
    Recommendation: Stay where you are. Annealing handles crashes,
                    Newton handles curvature. No need for KSD.
```

---

## References

- Barzilai, J. & Borwein, J. (1988). Two-Point Step Size Gradient Methods. IMA J. Numerical Analysis.
- Detommaso, G. et al. (2018). A Stein Variational Newton Method. NeurIPS.
- Liu, Q. (2016). Stein Variational Gradient Descent. NeurIPS.
