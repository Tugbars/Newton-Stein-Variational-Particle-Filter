# EKF Guide Removal Summary

## What Was Removed

### 1. Configuration in `svpf_create()` (lines 133-139)
```cpp
// REMOVED:
state->use_guide = 1;
state->use_guide_preserving = 1;
state->guide_strength = 0.05f;
state->guide_mean = 0.0f;
state->guide_var = 0.0f;
state->guide_K = 0.0f;
state->guide_initialized = 1;
```

### 2. Initialization in `svpf_initialize()` (lines 298-300)
```cpp
// REMOVED:
state->guide_initialized = 0;
state->guide_mean = params->mu;
state->guide_var = stationary_var;
```

### 3. Device Memory Allocation in `svpf_optimized_init()` (lines 359-364)
```cpp
// REMOVED:
cudaMalloc(&opt->d_guide_mean, sizeof(float));
cudaMemcpy(opt->d_guide_mean, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc(&opt->d_guide_strength, sizeof(float));
float init_guide_strength = 0.05f;
cudaMemcpy(opt->d_guide_strength, &init_guide_strength, sizeof(float), cudaMemcpyHostToDevice);
```

### 4. Device Memory Cleanup in `svpf_optimized_cleanup()` (lines 400-401)
```cpp
// REMOVED:
cudaFree(opt->d_guide_mean);
cudaFree(opt->d_guide_strength);
```

### 5. Entire Guide Update Section in `svpf_step_async()` (lines 550-580)
```cpp
// REMOVED:
// =========================================================================
// GUIDE (Variance-preserving)
// =========================================================================
float current_guide_strength = state->guide_strength_base;

if (state->use_adaptive_guide && state->timestep > 0) {
    float vol_est = fmaxf(state->vol_prev, 1e-4f);
    float return_z = fabsf(y_t) / vol_est;
    
    float implied_h = logf(y_t * y_t + 1e-8f) + state->student_t_implied_offset;
    float h_est = logf(vol_est * vol_est + 1e-8f);
    float h_innovation = implied_h - h_est;
    
    if (h_innovation > 0.0f && return_z > state->guide_innovation_threshold) {
        float severity = fminf((return_z - state->guide_innovation_threshold) / 3.0f, 1.0f);
        float boost = (state->guide_strength_max - state->guide_strength_base) * severity;
        current_guide_strength = state->guide_strength_base + boost;
    }
}

{
    SVPFParams guide_params = *params;
    if (state->use_adaptive_mu) {
        guide_params.mu = effective_mu;
    }
    svpf_ekf_update(state, y_t, &guide_params);
    
    svpf_apply_guide_preserving_kernel<<<nb, SVPF_BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_h_mean_prev, state->guide_mean, current_guide_strength, n
    );
}
```

### 6. Guide Reference in Step Size Calculation (line 562)
```cpp
// BEFORE:
float base_step = SVPF_STEIN_STEP_SIZE * (state->use_guide ? 0.5f : 1.0f);

// AFTER:
float base_step = SVPF_STEIN_STEP_SIZE;
```

## What Was NOT Removed (Important!)

These features remain ACTIVE:

### 1. Adaptive Guide (Different from EKF Guide)
```cpp
state->use_adaptive_guide = 1;         // Still present
state->guide_strength_base = 0.05f;
state->guide_strength_max = 0.30f;
state->guide_innovation_threshold = 1.0f;
```
**Note**: This modulates guide strength, but since the EKF guide is removed, this has no effect currently.

### 2. Guided Prediction (Different from EKF Guide)
```cpp
state->use_guided = 1;                 // Still present
state->guided_alpha_base = 0.0f;
state->guided_alpha_shock = 0.40f;
state->guided_innovation_threshold = 1.5f;
```
**Note**: This is in the PREDICT step, blends model prediction with observation-implied volatility.

## Why EKF Guide Was Removed

**Diagnostic test results:**
- With guide: ratio = 0.651 (bias compounds)
- Without guide: ratio = 1.788 (self-correcting)

**Root cause:**
The EKF guide created a feedback loop:
1. Temperature noise → particles drift
2. Biased particles → EKF learns wrong guide mean/var
3. Guide pulls particles toward wrong target
4. More drift → worse bias → compounds

**The other adaptive systems (mu, sigma, guided prediction) create SELF-CORRECTING feedback loops.**

## Expected Behavior After Removal

With EKF guide removed + decorrelation enabled:
- ✓ Bias self-corrects over time (ratio > 1.0)
- ✓ No accumulation across timesteps
- ✓ Maintains robustness through other adaptive features
- ✓ Improved accuracy under misspecification
