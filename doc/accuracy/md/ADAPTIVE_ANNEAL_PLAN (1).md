# Adaptive Annealing Implementation Plan

## Overview

Replace fixed beta annealing schedule (0.3 → 0.65 → 1.0 in 5 stages) with KL-divergence constrained adaptive annealing. This allows the filter to automatically use more stages during challenging market conditions (spikes, flash crashes) and fewer stages during calm periods.

## Motivation

- **Current approach**: Fixed 5 stages with heuristic beta values
- **Problem**: During crypto flash crashes, fixed schedule can't adapt to extreme conditions
- **Solution**: KL-based beta stepping with safety interlocks

## The Algorithm

```
beta = 0.0
while beta < 1.0 and stages < MAX_STAGES:
    
    1. Run gradient kernel at current beta → get log_w, grad for all particles
    
    2. Compute stats (one reduction kernel):
       - var_ll = Var(log_w)           # Particle disagreement
       - mean_grad = Mean(|grad|)       # Force magnitude
       - h_std = Std(h)                 # Spatial diversity
    
    3. Compute delta_beta (CPU, safety interlock):
       - d_beta_kl = sqrt(2 * kl_threshold / var_ll)      # KL constraint
       - d_beta_grad = GRAD_SCALE / (mean_grad + eps)     # Gradient constraint
       - d_beta_spatial = 0.02 if h_std < 0.05 else 0.25  # Collapse protection
       - d_beta = min(d_beta_kl, d_beta_grad, d_beta_spatial)
       - d_beta = clamp(d_beta, 0.01, 0.25)
    
    4. Update beta:
       beta = min(beta + d_beta, 1.0)
    
    5. Run Stein transport steps (configurable: 1-3 per beta)
    
    stages++
```

## Three Safety Interlocks

| Metric | What it measures | Failure it prevents |
|--------|------------------|---------------------|
| **Var(log_lik)** | Particle disagreement on likelihood | Distribution collapse |
| **Gradient norm** | Force magnitude | Position explosion |
| **Particle spread** | Spatial diversity (std of h) | False confidence / mode collapse |

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_adaptive_anneal` | 1 | Enable adaptive (1) or fixed (0) |
| `anneal_kl_threshold` | 0.5 | KL constraint (lower = more conservative) |
| `anneal_steps_per_beta` | 2 | Stein steps per beta update |
| `anneal_max_stages` | 50 | Safety cap on total stages |

## Files to Modify

### 1. svpf.cuh (DONE)

Added to `SVPFState`:
```cpp
int use_adaptive_anneal;
float anneal_kl_threshold;
int anneal_steps_per_beta;
int anneal_max_stages;
int anneal_stages_used;       // Diagnostic
float anneal_final_var_ll;    // Diagnostic
float anneal_final_h_std;     // Diagnostic
```

Added to `SVPFOptimizedState`:
```cpp
float* d_anneal_stats;         // [4]: mean_ll, var_ll, mean_grad, h_std
float* h_anneal_stats_pinned;  // Pinned host for fast D2H
```

### 2. svpf_adaptive_anneal.cuh (DONE)

Contains:
- `svpf_anneal_stats_kernel` - Computes all stats in one pass
- `svpf_anneal_stats_robust_kernel` - Two-pass with outlier clamping
- Host-side helper functions for delta_beta computation

### 3. svpf_optimized_graph.cu (TODO)

Changes needed:

**A. Include header:**
```cpp
#include "svpf_adaptive_anneal.cuh"
```

**B. In svpf_create() - Set defaults:**
```cpp
state->use_adaptive_anneal = 1;
state->anneal_kl_threshold = 0.5f;
state->anneal_steps_per_beta = 2;
state->anneal_max_stages = 50;
```

**C. In svpf_optimized_init() - Allocate buffers:**
```cpp
cudaMalloc(&opt->d_anneal_stats, 4 * sizeof(float));
cudaMallocHost(&opt->h_anneal_stats_pinned, 4 * sizeof(float));
```

**D. In svpf_optimized_cleanup() - Free buffers:**
```cpp
if (opt->d_anneal_stats) cudaFree(opt->d_anneal_stats);
if (opt->h_anneal_stats_pinned) cudaFreeHost(opt->h_anneal_stats_pinned);
```

**E. In svpf_step_async() - Replace Stein loop:**

Replace the current fixed loop:
```cpp
for (int ai = 0; ai < n_anneal; ai++) {
    float beta = compute_adaptive_beta(...);
    // ... stein steps ...
}
```

With adaptive loop:
```cpp
if (state->use_adaptive_anneal) {
    // New adaptive annealing implementation
    float beta = 0.0f;
    int stage = 0;
    
    while (beta < 1.0f && stage < state->anneal_max_stages) {
        // 1. Compute gradient at current beta
        svpf_fused_gradient_kernel<<<...>>>(... beta ...);
        
        // 2. Compute stats
        svpf_anneal_stats_kernel<<<1, 256, 0, cs>>>(
            state->log_weights, state->grad_log_p, state->h,
            opt->d_anneal_stats, n
        );
        
        // 3. D2H transfer (sync point)
        cudaMemcpyAsync(opt->h_anneal_stats_pinned, opt->d_anneal_stats,
                        4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        
        // 4. CPU decision
        float var_ll = opt->h_anneal_stats_pinned[1];
        float mean_grad = opt->h_anneal_stats_pinned[2];
        float h_std = opt->h_anneal_stats_pinned[3];
        
        float d_beta = compute_delta_beta(var_ll, mean_grad, h_std, 
                                          state->anneal_kl_threshold);
        beta = fminf(beta + d_beta, 1.0f);
        
        // 5. Stein transport at this beta
        float beta_factor = sqrtf(beta);
        for (int s = 0; s < state->anneal_steps_per_beta; s++) {
            total_steps++;
            bool is_last = (beta >= 1.0f) && (s == state->anneal_steps_per_beta - 1);
            
            // Run appropriate Stein kernel (Full Newton, Newton, or basic)
            // Same kernel selection logic as current code
        }
        
        stage++;
    }
    
    // Store diagnostics
    state->anneal_stages_used = stage;
    state->anneal_final_var_ll = opt->h_anneal_stats_pinned[1];
    state->anneal_final_h_std = opt->h_anneal_stats_pinned[3];
    
} else {
    // Keep existing fixed annealing code as fallback
}
```

## Robust Variance (Outlier Clamping)

To handle the "one bad apple" problem (single outlier particle with log_lik = -10000):

1. First pass: Compute mean and std of log_w
2. Clamp values outside [mean - 3σ, mean + 3σ]
3. Recompute variance on clamped values

This is O(N) vs O(N log N) for median, giving 90% robustness at 1% cost.

## Expected Behavior

| Market Condition | var_ll | mean_grad | Stages | d_beta |
|-----------------|--------|-----------|--------|--------|
| Calm | Low | Low | ~3-5 | 0.20-0.25 |
| Moderate vol | Medium | Medium | ~8-12 | 0.08-0.15 |
| Flash crash | High | High | ~20-40 | 0.02-0.05 |

## Testing Plan

1. **Accuracy test**: Run on challenging DGP with fixed vs adaptive
   - Compare RMSE, MAE, Bias
   - Expect adaptive to improve on spikes

2. **Latency test**: Measure overhead per timestep
   - Expected: ~10-20μs per stage for stats kernel + D2H
   - Acceptable for crypto tick rates

3. **Edge cases**:
   - All particles at same location (collapsed) → should force tiny steps
   - One outlier particle → robust variance should handle it
   - Extremely sharp likelihood → gradient constraint kicks in

## Open Questions

1. Should `steps_per_beta` be fixed or also adaptive?
2. Should we add early exit if var_ll drops below threshold?
3. Should we log diagnostics (stages_used) for post-hoc analysis?

## References

- KL-based annealing: SMC literature (Chopin, Del Moral)
- Variance approximation: Taylor expansion of KL divergence
- Safety interlocks: Supervisor's guidance on failure modes
