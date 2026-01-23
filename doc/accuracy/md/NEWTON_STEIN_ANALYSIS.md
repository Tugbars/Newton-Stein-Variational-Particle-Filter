# Newton-Stein Implementation Analysis

## Reference Paper
**"A Stein Variational Newton Method"** - Detommaso et al. (2018)

---

## The Newton Direction

### Paper's Formulation (Algorithm 2, Eq. 18)

The Newton direction `z` for particle `i` solves:

```
Σⱼ Ĥ(xⱼ, xᵢ) · zᵢ = Σⱼ [K(xⱼ, xᵢ)·∇log π(xⱼ) + ∇K(xⱼ, xᵢ)]
```

This is an `(nd) × (nd)` coupled linear system. The paper uses **block-diagonal approximation** ("mass lumping") to decouple into `n` independent `d × d` systems:

```
[Σⱼ Ĥ(xⱼ, xᵢ)] · zᵢ = Σⱼ [K(xⱼ, xᵢ)·∇log π(xⱼ) + ∇K(xⱼ, xᵢ)]
```

Where the effective Hessian Ĥ (Equation 17):

```
Ĥ(x, z) = Nπ(x)·K(x, z) + Nk(x, z)
```

- `Nπ(x)` = Gauss-Newton approximation of target Hessian (≈ Fisher information)
- `Nk(x, z)` = Gauss-Newton approximation of kernel Hessian
- `K(x, z)` = kernel value

---

## Comparison: Paper vs Our Implementation

### What the Paper Does

For each particle `i`, compute:

```
         Σⱼ [K(xⱼ, xᵢ)·∇log π(xⱼ) + ∇K(xⱼ, xᵢ)]
zᵢ = ───────────────────────────────────────────────
              Σⱼ [Nπ(xⱼ)·K(xⱼ, xᵢ) + Nk(xⱼ, xᵢ)]
```

The denominator is a **kernel-weighted average** of Hessians from ALL particles.

### What We Currently Do

```cpp
// Per-particle local Hessian (O(N) computation)
float hess_lik = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
float hess_prior = -inv_sigma_sq;
float curvature = -(hess_lik + hess_prior);
float inv_H = 1.0f / curvature;

// Precondition with LOCAL Hessian only
precond_grad[j] = 0.7f * g * inv_H;
inv_hessian[j] = inv_H;

// In Stein kernel:
k_sum += K * sh_precond_grad[j];           // K · (H⁻¹ · ∇log π)
gk_sum -= 2.0f * diff * inv_bw_sq * K_sq * sh_inv_hess[j];  // ∇K · H⁻¹
```

We use `H(xⱼ)` directly, not `Σⱼ H(xⱼ)·K(xⱼ, xᵢ)`.

---

## Gap Analysis

| Component | Paper | Ours | Status |
|-----------|-------|------|--------|
| Preconditioned gradient | `H⁻¹·∇log π` | ✅ Same | ✅ Correct |
| Preconditioned repulsion | `∇K·H⁻¹` | ✅ Same | ✅ Correct |
| Kernel-weighted Hessian | `Σⱼ Nπ(xⱼ)·K(xⱼ,xᵢ)` | ❌ Local H only | ⚠️ **Missing** |
| Kernel Hessian Nk | `Σⱼ Nk(xⱼ,xᵢ)` | ❌ Not included | ⚠️ **Missing** |
| Block-diagonal approx | Yes | Yes | ✅ Correct |

---

## What's Missing

### 1. Kernel-Weighted Hessian Averaging (Major)

**Paper:**
```
Ĥᵢ = Σⱼ Nπ(xⱼ)·K(xⱼ, xᵢ)
```

Each particle `i` gets a Hessian that's a weighted average over ALL particles, where weights are kernel values `K(xⱼ, xᵢ)`.

**Why it matters:**
- When particles are spread across regions with different curvatures, the kernel-weighted average smooths the Hessian landscape
- Particles in transition regions get a blended Hessian from their neighbors
- Prevents individual particles from taking overly aggressive steps based on local curvature alone

**Current impact:**
- In 1D with smooth posteriors: Minor (local curvature is a reasonable approximation)
- In multimodal/high-curvature regions: Could cause instability

### 2. Kernel Hessian Nk (Minor)

**Paper:**
```
Nk(x, z) = ∇²k(x, z)
```

For Gaussian kernel: `Nk = (2/h²)·K·(1 - 2·dist²/h²)`
For IMQ kernel: `Nk = (2/h²)·K²·(3·dist²/h² - 1)`

**Why it matters:**
- Accounts for curvature introduced by the kernel itself
- More important when bandwidth is small (sharp kernel)

**Current impact:**
- For typical bandwidths: ~5-10% contribution to total Hessian
- Omitting it makes steps slightly more aggressive

---

## Proposed Fix

### New Kernel: `svpf_newton_hessian_kernel`

```cpp
// Phase 1: Compute kernel-weighted Hessian for each particle i
// O(N²) but parallelizable - each particle computes its own sum

__global__ void svpf_compute_weighted_hessian_kernel(
    const float* __restrict__ h,              // [N] particle positions
    const float* __restrict__ local_hessian,  // [N] Nπ(xⱼ) for each j
    float* __restrict__ weighted_hessian,     // [N] output: Σⱼ Nπ(xⱼ)·K(xⱼ,xᵢ)
    const float* __restrict__ d_bandwidth,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float inv_bw_sq = 1.0f / (bw * bw);
    
    float H_sum = 0.0f;
    float K_sum = 0.0f;
    
    for (int j = 0; j < n; j++) {
        float diff = h_i - h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        
        // IMQ kernel
        float K = 1.0f / (1.0f + dist_sq);
        
        // Kernel Hessian (IMQ): Nk = (2/h²)·K²·(3·dist² - 1)
        // Simplified for 1D
        float Nk = 2.0f * inv_bw_sq * K * K * (3.0f * dist_sq - 1.0f);
        
        // Accumulate weighted Hessian
        H_sum += local_hessian[j] * K + Nk;
        K_sum += K;
    }
    
    // Normalize by kernel sum (optional, for stability)
    weighted_hessian[i] = H_sum / fmaxf(K_sum, 1e-6f);
}
```

### Updated Newton Transport

```cpp
// Phase 2: Use weighted Hessian in Stein operator
// Now each particle i uses its kernel-averaged Hessian

__global__ void svpf_newton_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,                // ∇log π(xⱼ)
    const float* __restrict__ weighted_hessian,    // Σⱼ H(xⱼ)·K(xⱼ,xᵢ)
    ...
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float H_i = weighted_hessian[i];
    float inv_H_i = 1.0f / fmaxf(H_i, 0.1f);
    
    // Stein operator with kernel-weighted preconditioning
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = 0; j < n; j++) {
        float diff = h_i - h[j];
        float K = ...;  // IMQ kernel
        
        // Preconditioned gradient uses particle i's weighted Hessian
        k_sum += K * grad[j] * inv_H_i;
        
        // Preconditioned repulsion
        gk_sum -= dK * inv_H_i;
    }
    
    float phi_i = (k_sum + gk_sum) / n;
    
    // Transport
    h[i] += step_size * phi_i;
}
```

---

## Computational Cost

| Operation | Current | With Fix |
|-----------|---------|----------|
| Hessian computation | O(N) | O(N²) |
| Stein transport | O(N²) | O(N²) |
| **Total per Stein step** | O(N²) | O(N²) |

The fix adds one O(N²) pass for Hessian averaging. Since we already do O(N²) for Stein, this roughly **doubles** the Stein step cost.

For N=400, 8 Stein steps:
- Current: 8 × O(N²) = 8 × 160K = 1.28M operations
- With fix: 8 × 2 × O(N²) = 2.56M operations

At ~10μs current latency, expect ~15-20μs with the fix.

---

## Implementation Plan

### Step 1: Add Hessian Buffer
```cpp
// In SVPFOptimizedState
float* d_weighted_hessian;  // [N] kernel-weighted Hessian per particle
```

### Step 2: New Kernel
- `svpf_compute_weighted_hessian_kernel` - O(N²) weighted average

### Step 3: Update Newton Stein Kernel
- Use `weighted_hessian[i]` instead of `local_hessian[j]`

### Step 4: Graph Integration
- Add new kernel to CUDA graph sequence
- Between gradient computation and Stein transport

---

## Expected Impact

| Scenario | Current RMSE | Expected After |
|----------|--------------|----------------|
| Slow Drift | 0.5942 | ~0.58 |
| Stress Ramp | 0.6233 | ~0.60 |
| Spike+Recovery | 0.4762 | ~0.46 |
| **Average** | **0.5507** | **~0.53** |

Conservative estimate: **3-5% RMSE reduction** from proper Newton preconditioning.

The benefit will be most visible in:
- Multi-regime scenarios (different curvatures)
- Transition periods (particles spread across curvature boundaries)
- High vol-of-vol settings (rapidly changing Hessian)

---

## Alternative: Scaled Hessian Kernel (Section 4 of Paper)

The paper also proposes a **geometry-aware kernel** using the average Hessian:

```
Mπ = E_{x~π}[Nπ(x)] ≈ (1/N) Σᵢ Nπ(xᵢ)
```

Then use anisotropic kernel:
```
k(x, x') = exp(-½ ||x - x'||²_M / d)
```

This is a simpler alternative that:
- Uses a single averaged Hessian for the kernel (not per-particle)
- Affects the kernel bandwidth, not the Newton direction
- Cheaper: O(N) to compute M, then standard O(N²) Stein

**For 1D:** This reduces to just scaling the bandwidth by average curvature, which we partially do with adaptive bandwidth. The full Newton fix is more impactful.

---

## Summary

**Current state:** Approximate Newton-Stein with local Hessians.

**Missing:** Kernel-weighted Hessian averaging (the "mass lumping" in Eq. 18).

**Fix complexity:** One additional O(N²) kernel per Stein step.

**Expected gain:** 3-5% RMSE reduction, better stability in multi-regime scenarios.

**Priority:** Medium-high. Worth implementing after PMMH Oracle integration.
