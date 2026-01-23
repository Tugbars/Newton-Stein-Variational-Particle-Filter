# SVPF Parameter Learning via PMMH Oracle

## Overview

SVPF uses an Oracle-Worker architecture where:
- **Worker**: Fast online filtering (~10μs per step)
- **Oracle**: Parallel PMMH learning parameters (~100μs per iteration)

This decouples fast inference from slow calibration, allowing real-time HFT while continuously adapting to market regimes.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PARALLEL EXECUTION                         │
│                                                               │
│   ┌─────────────────┐          ┌─────────────────────────┐   │
│   │   SVPF Worker   │          │      PMMH Oracle        │   │
│   │                 │          │                         │   │
│   │ • Uses θ_current│  ←────── │ • Proposes θ'           │   │
│   │ • Filters y_t   │          │ • Runs test filter      │   │
│   │ • Returns vol_t │          │ • Computes p(y|θ')      │   │
│   │ • ~10μs/step    │          │ • MH accept/reject      │   │
│   │                 │          │ • Pushes θ* to Worker   │   │
│   │                 │          │ • ~100μs/iteration      │   │
│   └─────────────────┘          └─────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Parameters to Learn

### Tier 1: Core Model Parameters (Recommended)

| Parameter | Range | Why Learn It |
|-----------|-------|--------------|
| `nu` | [5, 100] | Tail behavior; Gaussian (30+) vs fat-tailed (5-10) |
| `rho_up` | [0.90, 0.999] | Persistence when vol rising |
| `rho_down` | [0.85, 0.98] | Mean-reversion speed during recovery |
| `sigma_z` | [0.05, 0.5] | Base vol-of-vol (before adaptive boost) |
| `mu` | [-5, -1] | Base mean level (starting point for adaptive μ) |
| `lik_offset` | [0.5, 1.0] | Likelihood bias correction; regime-specific |

### Tier 2: Optional

| Parameter | Range | Why Maybe |
|-----------|-------|-----------|
| `gamma` | [-0.5, 0.5] | Leverage effect; often small but can matter |
| `guide_strength_base` | [0.02, 0.15] | Base EKF coupling; less direct likelihood impact |

### Don't Learn (Already Adaptive or Tuning)

| Parameter | Reason |
|-----------|--------|
| `mu_process_var`, `mu_obs_var_scale` | Kalman tuning, not model |
| `sigma_boost_*` | Already adapts online via innovation |
| `guide_strength_max`, thresholds | Filter tuning parameters |
| `temperature`, `rmsprop_*` | Optimizer internals |

---

## Oracle Parameter Struct

```cpp
struct SVPFOracleParams {
    float nu;           // [5, 100]     - Tail thickness
    float rho_up;       // [0.90, 0.999] - Spike persistence
    float rho_down;     // [0.85, 0.98]  - Recovery speed
    float sigma_z;      // [0.05, 0.5]   - Vol-of-vol
    float mu;           // [-5, -1]      - Mean log-vol
    float lik_offset;   // [0.5, 1.0]    - Bias correction
    float gamma;        // [-0.5, 0.5]   - Leverage (optional)
    float _pad;         // Alignment
};
```

---

## PMMH Implementation Sketch

```cpp
// Oracle thread - runs parallel to Worker
void pmmh_oracle_thread(
    SVPFState* worker_filter,      // Live filter (read/write θ)
    const float* recent_data,      // Rolling window of returns
    int window_size,               // e.g., 500-1000 ticks
    cudaStream_t oracle_stream
) {
    SVPFOracleParams theta_current = get_current_params(worker_filter);
    float log_lik_current = -INFINITY;
    
    // Proposal std devs (tune these)
    SVPFOracleParams proposal_std = {
        .nu = 5.0f,
        .rho_up = 0.01f,
        .rho_down = 0.02f,
        .sigma_z = 0.03f,
        .mu = 0.2f,
        .lik_offset = 0.05f,
        .gamma = 0.05f
    };
    
    while (running) {
        // 1. PROPOSE: Random walk in parameter space
        SVPFOracleParams theta_proposed = propose(theta_current, proposal_std);
        clamp_params(&theta_proposed);
        
        // 2. EVALUATE: Run test filter on recent window
        SVPFState* test_filter = svpf_create(N_PARTICLES, N_STEIN, 
                                              theta_proposed.nu, oracle_stream);
        apply_params(test_filter, &theta_proposed);
        
        float log_lik_proposed = 0.0f;
        for (int t = 0; t < window_size; t++) {
            SVPFResult result;
            svpf_step(test_filter, recent_data[t], &params, &result);
            log_lik_proposed += result.log_lik_increment;
        }
        svpf_destroy(test_filter);
        
        // 3. ACCEPT/REJECT: Metropolis-Hastings
        float log_alpha = log_lik_proposed - log_lik_current;
        
        if (log(uniform_rng()) < log_alpha) {
            // Accept
            theta_current = theta_proposed;
            log_lik_current = log_lik_proposed;
            
            // 4. PUSH TO WORKER (atomic update)
            push_params_to_worker(worker_filter, &theta_current);
        }
        
        // Optional: Sleep or rate-limit
    }
}
```

---

## Pushing Parameters to Worker

The Worker filter needs a way to receive updated parameters without blocking:

```cpp
void push_params_to_worker(SVPFState* worker, const SVPFOracleParams* theta) {
    // Update model params
    worker->nu = theta->nu;
    worker->student_t_const = lgammaf((theta->nu + 1.0f) / 2.0f)
                            - lgammaf(theta->nu / 2.0f)
                            - 0.5f * logf(M_PI * theta->nu);
    
    worker->rho_up = theta->rho_up;
    worker->rho_down = theta->rho_down;
    worker->lik_offset = theta->lik_offset;
    
    // Update base values for adaptive methods
    if (worker->use_adaptive_mu) {
        // Optionally reset mu_state to new base, or let Kalman adapt
        // worker->mu_state = theta->mu;  
    }
    if (worker->use_adaptive_sigma) {
        worker->sigma_z_effective = theta->sigma_z;
    }
    
    // CRITICAL: Invalidate graph (params burned in at capture time)
    svpf_graph_invalidate(worker);
}

// Helper to update nu specifically
void svpf_update_nu(SVPFState* state, float nu) {
    state->nu = nu;
    state->student_t_const = lgammaf((nu + 1.0f) / 2.0f)
                           - lgammaf(nu / 2.0f)
                           - 0.5f * logf(M_PI * nu);
    svpf_graph_invalidate(state);
}
```

---

## Why Not Learn Online?

We tried 2D SVGD (learning σ_z alongside h) - it failed because:

1. **SVGD repulsion fights parameter convergence** - particles need diversity for state inference but consensus for parameter inference
2. **Observability** - some params (like ρ) are unidentifiable in calm markets
3. **Latency** - parameter learning adds O(N²) cost per step

PMMH Oracle solves this:
- Parameter particles don't repel (it's MCMC, not SVGD)
- Runs on historical window (observability over time)
- Parallel execution (no added latency to Worker)

---

## Tuning the Oracle

| Setting | Recommendation |
|---------|----------------|
| Window size | 500-1000 ticks (balance recency vs stability) |
| PMMH iterations/sec | 10-100 (100μs × 10-100 = 1-10ms budget) |
| Proposal std | Start large, shrink as chain converges |
| Burn-in | First 100-500 iterations before pushing to Worker |
| Update frequency | Push to Worker every N accepts, not every iteration |

---

## Integration with Existing HCRBPF Oracle

If you already have HCRBPF Oracle running, SVPF slots in naturally:

```cpp
// Same Oracle can learn params for both filters
struct UnifiedOracleParams {
    // Shared
    float nu, rho, sigma_z, mu, gamma;
    
    // SVPF-specific
    float lik_offset;
    float rho_up, rho_down;  // (HCRBPF may use single rho)
};

// Oracle evaluates both, uses combined likelihood
float log_lik = alpha * eval_svpf(theta) + (1-alpha) * eval_hcrbpf(theta);
```

Or run separate Oracles if filters need different regimes.

---

## Summary

| Component | Role | Latency |
|-----------|------|---------|
| SVPF Worker | Online filtering | ~10μs/step |
| PMMH Oracle | Parameter learning | ~100μs/iter |
| Graph invalidation | Sync params | ~5μs (next capture) |

The Oracle-Worker pattern lets SVPF stay fast while continuously adapting to changing market regimes.
