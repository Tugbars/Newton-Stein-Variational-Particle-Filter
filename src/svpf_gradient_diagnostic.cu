/**
 * @file svpf_gradient_diagnostic.cu
 * @brief Diagnostic instrumentation for ν gradient (Step 0 of self-tuning SVPF)
 * 
 * PURPOSE:
 *   Compute and log the ν gradient WITHOUT updating anything.
 *   This allows verification that:
 *   1. Gradient math is correct (synthetic data test)
 *   2. Gradient behaves sensibly (negative during crashes, ~0 during calm)
 *   3. Gradient noise level is acceptable
 * 
 * USAGE:
 *   After each svpf_step_graph() call:
 *     float nu_grad, z_sq;
 *     svpf_compute_nu_diagnostic(state, y_t, &nu_grad, &z_sq);
 *     // Log nu_grad, z_sq for analysis
 * 
 * NEXT STEP:
 *   Once gradients verified, proceed to Step 1 (enable ν learning)
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// =============================================================================
// Device Helper: Digamma Function
// =============================================================================

__device__ float device_digamma(float x) {
    // Asymptotic expansion for x >= 6
    // For smaller x, use recurrence relation
    float result = 0.0f;
    while (x < 6.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    // Asymptotic: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - ...
    float inv_x = 1.0f / x;
    float inv_x2 = inv_x * inv_x;
    result += __logf(x) - 0.5f * inv_x - inv_x2 / 12.0f + inv_x2 * inv_x2 / 120.0f;
    return result;
}

// =============================================================================
// ν Gradient Kernel (Observation Likelihood Only)
// =============================================================================
// 
// Computes ∂/∂ν log p(y_t | h, ν) for Student-t observation model:
//
//   log p(y|h,ν) = log Γ((ν+1)/2) - log Γ(ν/2) - ½log(ν) - h/2
//                  - ((ν+1)/2) · log(1 + z²/ν)
//   where z = y / exp(h/2)
//
// Gradient:
//   ∂/∂ν = ½[ψ((ν+1)/2) - ψ(ν/2) - 1/ν - log(1 + z²/ν) + (ν+1)·z²/(ν²·(1 + z²/ν))]
//
// Returns weighted mean over particles (using log_weights for posterior weighting)

__global__ void svpf_nu_gradient_kernel(
    const float* __restrict__ h,          // [n] Current particles
    const float* __restrict__ log_w,      // [n] Log weights
    float y_t,                            // Current observation
    float nu,                             // Current ν value
    float* __restrict__ d_nu_grad,        // Output: mean ν gradient
    float* __restrict__ d_z_sq_mean,      // Output: mean z² (diagnostic)
    int n
) {
    // Shared memory for reductions
    extern __shared__ float smem[];
    float* s_max = smem;                    // [blockDim.x / 32] for warp maxes
    float* s_grad_sum = &smem[8];           // Single float
    float* s_weight_sum = &smem[9];
    float* s_z_sq_sum = &smem[10];
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Initialize shared memory
    if (tid == 0) {
        *s_grad_sum = 0.0f;
        *s_weight_sum = 0.0f;
        *s_z_sq_sum = 0.0f;
    }
    if (tid < 8) {
        s_max[tid] = -1e30f;
    }
    __syncthreads();
    
    // === Pass 1: Find max log weight for numerical stability ===
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        float lw = log_w[i];
        if (!isnan(lw) && !isinf(lw)) {
            local_max = fmaxf(local_max, lw);
        }
    }
    
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    
    // Lane 0 of each warp writes to shared
    if (lane_id == 0 && warp_id < 8) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Thread 0 finds global max
    float max_log_w = -1e30f;
    if (tid == 0) {
        for (int i = 0; i < 8; i++) {
            max_log_w = fmaxf(max_log_w, s_max[i]);
        }
        s_max[0] = max_log_w;  // Store result
    }
    __syncthreads();
    max_log_w = s_max[0];
    
    // === Pass 2: Compute weighted gradient ===
    // Precompute digamma terms (same for all particles)
    float psi_nu_plus_1_half = device_digamma((nu + 1.0f) * 0.5f);
    float psi_nu_half = device_digamma(nu * 0.5f);
    float inv_nu = 1.0f / nu;
    float nu_plus_1 = nu + 1.0f;
    
    float local_grad = 0.0f;
    float local_weight = 0.0f;
    float local_z_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // Convert log weight to weight (numerically stable)
        float w_i = __expf(log_w[i] - max_log_w);
        
        // Compute standardized residual
        float h_i = h[i];
        float vol_i = __expf(h_i * 0.5f);
        float z_i = y_t / (vol_i + 1e-8f);
        float z_sq = z_i * z_i;
        
        // ν gradient: ∂/∂ν log p(y|h,ν)
        // = 0.5 * [ψ((ν+1)/2) - ψ(ν/2) - 1/ν - log(1 + z²/ν) + (ν+1)·z²/(ν²·(1 + z²/ν))]
        float A = 1.0f + z_sq * inv_nu;
        float log_A = __logf(A);
        
        float grad_nu = 0.5f * (
            psi_nu_plus_1_half 
            - psi_nu_half
            - inv_nu
            - log_A
            + nu_plus_1 * z_sq / (nu * nu * A)
        );
        
        // Accumulate weighted sums
        local_grad += w_i * grad_nu;
        local_weight += w_i;
        local_z_sq += w_i * z_sq;
    }
    
    // Block reduction via atomics (simple, sufficient for single block)
    atomicAdd(s_grad_sum, local_grad);
    atomicAdd(s_weight_sum, local_weight);
    atomicAdd(s_z_sq_sum, local_z_sq);
    __syncthreads();
    
    // Thread 0 writes final output
    if (threadIdx.x == 0) {
        float inv_weight = 1.0f / (*s_weight_sum + 1e-8f);
        *d_nu_grad = *s_grad_sum * inv_weight;
        *d_z_sq_mean = *s_z_sq_sum * inv_weight;
    }
}

// =============================================================================
// Host API: Initialize Diagnostics
// =============================================================================

SVPFGradientDiagnostics* svpf_gradient_diagnostic_create(bool enable_logging, const char* log_path) {
    SVPFGradientDiagnostics* diag = (SVPFGradientDiagnostics*)malloc(sizeof(SVPFGradientDiagnostics));
    if (!diag) return nullptr;
    
    cudaMalloc(&diag->d_nu_grad, sizeof(float));
    cudaMalloc(&diag->d_z_sq_mean, sizeof(float));
    
    // Future: allocate transition gradient buffers
    diag->d_mu_grad = nullptr;
    diag->d_rho_grad = nullptr;
    diag->d_sigma_grad = nullptr;
    diag->d_fisher = nullptr;
    diag->d_fisher_inv = nullptr;
    
    diag->nu_gradient_ema = 0.0f;
    diag->z_sq_ema = 1.0f;  // Initialize to expected value under correct model
    diag->mu_gradient_ema = 0.0f;
    diag->rho_gradient_ema = 0.0f;
    diag->sigma_gradient_ema = 0.0f;
    
    // Shock state machine defaults
    diag->shock_state = 0;  // CALM
    diag->ticks_in_state = 0;
    diag->shock_threshold = 9.0f;      // 3σ
    diag->shock_duration = 20;
    diag->recovery_duration = 50;
    diag->recovery_exit_threshold = 4.0f;  // 2σ
    
    diag->enable_logging = enable_logging;
    diag->log_file = nullptr;
    
    if (enable_logging && log_path) {
        diag->log_file = fopen(log_path, "w");
        if (diag->log_file) {
            fprintf(diag->log_file, "t,y_t,vol,z_sq,nu_grad,nu_grad_ema,z_sq_ema,shock_state\n");
        }
    }
    
    diag->initialized = true;
    return diag;
}

void svpf_gradient_diagnostic_destroy(SVPFGradientDiagnostics* diag) {
    if (!diag) return;
    
    cudaFree(diag->d_nu_grad);
    cudaFree(diag->d_z_sq_mean);
    
    if (diag->d_mu_grad) cudaFree(diag->d_mu_grad);
    if (diag->d_rho_grad) cudaFree(diag->d_rho_grad);
    if (diag->d_sigma_grad) cudaFree(diag->d_sigma_grad);
    if (diag->d_fisher) cudaFree(diag->d_fisher);
    if (diag->d_fisher_inv) cudaFree(diag->d_fisher_inv);
    
    if (diag->log_file) {
        fclose(diag->log_file);
    }
    
    free(diag);
}

// =============================================================================
// Host API: Compute ν Gradient (No Update, Just Observe)
// =============================================================================

void svpf_compute_nu_diagnostic(
    SVPFState* state,
    SVPFGradientDiagnostics* diag,
    float y_t,
    int timestep,
    float* nu_grad_out,     // Optional: return raw gradient
    float* z_sq_mean_out    // Optional: return mean z²
) {
    if (!diag || !diag->initialized) return;
    
    // Launch kernel (single block sufficient for reduction)
    // Shared memory: 8 floats for warp maxes + 3 for accumulators = 44 bytes
    svpf_nu_gradient_kernel<<<1, 256, 48, state->stream>>>(
        state->h,
        state->log_weights,
        y_t,
        state->nu,
        diag->d_nu_grad,
        diag->d_z_sq_mean,
        state->n_particles
    );
    
    // Copy results to host
    float nu_grad, z_sq_mean;
    cudaMemcpy(&nu_grad, diag->d_nu_grad, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&z_sq_mean, diag->d_z_sq_mean, sizeof(float), cudaMemcpyDeviceToHost);
    
    // EMA smoothing for stability monitoring
    const float ema_alpha = 0.1f;
    diag->nu_gradient_ema = (1.0f - ema_alpha) * diag->nu_gradient_ema + ema_alpha * nu_grad;
    diag->z_sq_ema = (1.0f - ema_alpha) * diag->z_sq_ema + ema_alpha * z_sq_mean;
    
    // Log if enabled
    if (diag->enable_logging && diag->log_file) {
        float vol = state->vol_prev;
        fprintf(diag->log_file, "%d,%f,%f,%f,%f,%f,%f,%d\n",
                timestep, y_t, vol, z_sq_mean, nu_grad, 
                diag->nu_gradient_ema, diag->z_sq_ema, diag->shock_state);
    }
    
    // Return values if requested
    if (nu_grad_out) *nu_grad_out = nu_grad;
    if (z_sq_mean_out) *z_sq_mean_out = z_sq_mean;
}

// =============================================================================
// Host API: Convenience wrapper (no separate diagnostic state needed)
// =============================================================================

void svpf_compute_nu_diagnostic_simple(
    SVPFState* state,
    float y_t,
    float* nu_grad_out,
    float* z_sq_mean_out
) {
    // Allocate temporary device buffers
    float* d_nu_grad;
    float* d_z_sq_mean;
    cudaMalloc(&d_nu_grad, sizeof(float));
    cudaMalloc(&d_z_sq_mean, sizeof(float));
    
    // Launch kernel (shared memory: 8 floats for warp maxes + 3 for accumulators)
    svpf_nu_gradient_kernel<<<1, 256, 48, state->stream>>>(
        state->h,
        state->log_weights,
        y_t,
        state->nu,
        d_nu_grad,
        d_z_sq_mean,
        state->n_particles
    );
    
    // Copy results
    float nu_grad, z_sq_mean;
    cudaMemcpy(&nu_grad, d_nu_grad, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&z_sq_mean, d_z_sq_mean, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_nu_grad);
    cudaFree(d_z_sq_mean);
    
    if (nu_grad_out) *nu_grad_out = nu_grad;
    if (z_sq_mean_out) *z_sq_mean_out = z_sq_mean;
}

// =============================================================================
// Shock State Machine
// =============================================================================

void svpf_update_shock_state(SVPFGradientDiagnostics* diag, float z_sq) {
    if (!diag) return;
    
    diag->ticks_in_state++;
    
    switch (diag->shock_state) {
        case 0:  // CALM
            if (z_sq > diag->shock_threshold) {
                diag->shock_state = 1;  // → SHOCK
                diag->ticks_in_state = 0;
            }
            break;
            
        case 1:  // SHOCK
            if (diag->ticks_in_state >= diag->shock_duration) {
                diag->shock_state = 2;  // → RECOVERY
                diag->ticks_in_state = 0;
            }
            break;
            
        case 2:  // RECOVERY
            if (z_sq > diag->shock_threshold) {
                // Back to SHOCK if another spike
                diag->shock_state = 1;
                diag->ticks_in_state = 0;
            } else if (diag->ticks_in_state >= diag->recovery_duration &&
                       z_sq < diag->recovery_exit_threshold) {
                // Back to CALM
                diag->shock_state = 0;
                diag->ticks_in_state = 0;
            }
            break;
    }
}

int svpf_get_shock_state(const SVPFGradientDiagnostics* diag) {
    return diag ? diag->shock_state : 0;
}

bool svpf_should_learn(const SVPFGradientDiagnostics* diag) {
    if (!diag) return false;
    return diag->shock_state != 1;  // Learn in CALM or RECOVERY, freeze in SHOCK
}

float svpf_get_lr_multiplier(const SVPFGradientDiagnostics* diag, float lr_shock_mult) {
    if (!diag) return 1.0f;
    
    switch (diag->shock_state) {
        case 0: return 1.0f;          // CALM: normal LR
        case 1: return 0.0f;          // SHOCK: freeze
        case 2: return lr_shock_mult; // RECOVERY: boost LR
        default: return 1.0f;
    }
}

// =============================================================================
// Natural Gradient Tuner (Stub - Future Implementation)
// =============================================================================

// =============================================================================
// σ Gradient Kernel (Transition Likelihood)
// =============================================================================
// 
// Computes ∂/∂σ log p(h_t | h_{t-1}, θ) for Gaussian transition:
//
//   log p(h_t|h_{t-1}) = -½log(2πσ²) - ε²/(2σ²)
//   where ε = h_t - μ - ρ(h_{t-1} - μ)
//
// Gradient:
//   ∂/∂σ = -1/σ + ε²/σ³ = (ε²/σ² - 1) / σ
//
// CRITICAL: Must use h_pred (before Stein transport), NOT h (after Stein).
//
//   h_pred = μ + ρ(h_prev - μ) + σ*ε_random   ← correct (random innovation)
//   h      = h_pred + Stein_push              ← wrong (includes deterministic push)
//
//   Using post-Stein h gives ε²/σ² >> 1 because Stein_push >> σ*ε_random.
//
// Expected behavior (with h_pred):
//   - σ too HIGH → ε²/σ² < 1 → gradient NEGATIVE (decrease σ)
//   - σ too LOW  → ε²/σ² > 1 → gradient POSITIVE (increase σ)
//   - σ correct  → ε²/σ² ≈ 1 → gradient ≈ 0
//
// Key advantage over ν: Signal available EVERY timestep, not just crashes.

__global__ void svpf_sigma_gradient_kernel(
    const float* __restrict__ h,           // [n] Current particles h_t
    const float* __restrict__ h_prev,      // [n] Previous particles h_{t-1}
    const float* __restrict__ log_w,       // [n] Log weights
    float mu,                              // Mean level
    float rho,                             // Persistence
    float sigma,                           // Current σ value
    float* __restrict__ d_sigma_grad,      // Output: weighted mean σ gradient
    float* __restrict__ d_eps_sq_mean,     // Output: mean ε²/σ² (diagnostic)
    int n
) {
    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_grad_sum = &smem[8];
    float* s_weight_sum = &smem[9];
    float* s_eps_sq_sum = &smem[10];
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (tid == 0) {
        *s_grad_sum = 0.0f;
        *s_weight_sum = 0.0f;
        *s_eps_sq_sum = 0.0f;
    }
    if (tid < 8) {
        s_max[tid] = -1e30f;
    }
    __syncthreads();
    
    // === Pass 1: Find max log weight ===
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        float lw = log_w[i];
        if (!isnan(lw) && !isinf(lw)) {
            local_max = fmaxf(local_max, lw);
        }
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    
    if (lane_id == 0 && warp_id < 8) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    float max_log_w = -1e30f;
    if (tid == 0) {
        for (int i = 0; i < 8; i++) {
            max_log_w = fmaxf(max_log_w, s_max[i]);
        }
        s_max[0] = max_log_w;
    }
    __syncthreads();
    max_log_w = s_max[0];
    
    // === Pass 2: Compute weighted gradient ===
    float inv_sigma = 1.0f / sigma;
    float inv_sigma_sq = inv_sigma * inv_sigma;
    
    float local_grad = 0.0f;
    float local_weight = 0.0f;
    float local_eps_sq_norm = 0.0f;
    
    for (int i = tid; i < n; i += blockDim.x) {
        float w_i = __expf(log_w[i] - max_log_w);
        
        // Transition innovation: ε = h_t - μ - ρ(h_{t-1} - μ)
        float h_i = h[i];
        float h_prev_i = h_prev[i];
        float eps = h_i - mu - rho * (h_prev_i - mu);
        float eps_sq = eps * eps;
        
        // Normalized squared innovation: ε²/σ²
        float eps_sq_norm = eps_sq * inv_sigma_sq;
        
        // σ gradient: (ε²/σ² - 1) / σ
        float grad_sigma = (eps_sq_norm - 1.0f) * inv_sigma;
        
        local_grad += w_i * grad_sigma;
        local_weight += w_i;
        local_eps_sq_norm += w_i * eps_sq_norm;
    }
    
    atomicAdd(s_grad_sum, local_grad);
    atomicAdd(s_weight_sum, local_weight);
    atomicAdd(s_eps_sq_sum, local_eps_sq_norm);
    __syncthreads();
    
    if (tid == 0) {
        float inv_weight = 1.0f / (*s_weight_sum + 1e-8f);
        *d_sigma_grad = *s_grad_sum * inv_weight;
        *d_eps_sq_mean = *s_eps_sq_sum * inv_weight;  // ε²/σ² mean
    }
}

// =============================================================================
// Host API: Compute σ Gradient (Simple Version)
// =============================================================================

/**
 * @brief Compute σ gradient for diagnostic purposes
 * 
 * IMPORTANT: Uses h_pred (before Stein) not h (after Stein).
 * Stein transport is deterministic and much larger than σ,
 * so using post-Stein h gives meaningless ε²/σ² >> 1.
 * 
 * @param state         SVPF state (has h_pred, h_prev, log_weights)
 * @param params        SV parameters (μ, ρ, σ)
 * @param sigma_grad_out    Output: σ gradient (positive → increase σ)
 * @param eps_sq_norm_out   Output: mean ε²/σ² (should be ~1.0 if σ correct)
 */
void svpf_compute_sigma_diagnostic_simple(
    SVPFState* state,
    const SVPFParams* params,
    float* sigma_grad_out,
    float* eps_sq_norm_out
) {
    float* d_sigma_grad;
    float* d_eps_sq_norm;
    cudaMalloc(&d_sigma_grad, sizeof(float));
    cudaMalloc(&d_eps_sq_norm, sizeof(float));
    
    // Use h_pred (before Stein transport), NOT h (after Stein)
    // Stein push is deterministic and >> σ, so using h gives ε²/σ² >> 1
    svpf_sigma_gradient_kernel<<<1, 256, 48, state->stream>>>(
        state->h_pred,    // <-- FIXED: use h_pred, not h
        state->h_prev,
        state->log_weights,
        params->mu,
        params->rho,
        params->sigma_z,
        d_sigma_grad,
        d_eps_sq_norm,
        state->n_particles
    );
    
    float sigma_grad, eps_sq_norm;
    cudaMemcpy(&sigma_grad, d_sigma_grad, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&eps_sq_norm, d_eps_sq_norm, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_sigma_grad);
    cudaFree(d_eps_sq_norm);
    
    if (sigma_grad_out) *sigma_grad_out = sigma_grad;
    if (eps_sq_norm_out) *eps_sq_norm_out = eps_sq_norm;
}

// =============================================================================
// Host API: Snapshot h for σ gradient computation
// =============================================================================

/**
 * @brief Copy current particles to a buffer for later σ gradient computation
 * 
 * Call BEFORE svpf_step_graph() if you need h_{t-1} for gradient.
 * (Note: state->h_prev should already have this after a step)
 */
void svpf_snapshot_particles(
    SVPFState* state,
    float* d_h_buffer
) {
    cudaMemcpyAsync(
        d_h_buffer,
        state->h,
        state->n_particles * sizeof(float),
        cudaMemcpyDeviceToDevice,
        state->stream
    );
}

// =============================================================================
// Natural Gradient Tuner Implementation
// =============================================================================

SVPFNaturalGradientTuner* svpf_tuner_create(
    const SVPFParams* params,
    float nu,
    float base_lr,
    float prior_weight
) {
    SVPFNaturalGradientTuner* tuner = (SVPFNaturalGradientTuner*)calloc(1, sizeof(SVPFNaturalGradientTuner));
    if (!tuner) return nullptr;
    
    // Initialize unconstrained parameters
    tuner->theta.mu = params->mu;
    tuner->theta.eta = svpf_unconstrain_rho(params->rho);
    tuner->theta.kappa = svpf_unconstrain_sigma(params->sigma_z);
    tuner->theta.kappa_nu = svpf_unconstrain_nu(nu);
    
    // Initialize Fisher as identity
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tuner->F[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    tuner->F_ema_decay = 0.95f;
    tuner->F_reg = 1e-4f;
    tuner->base_lr = base_lr;
    tuner->lr_shock_mult = 2.0f;
    tuner->grad_clip = 1.0f;
    tuner->prior_weight = prior_weight;
    
    // Store prior (baseline from offline calibration)
    tuner->theta_prior = tuner->theta;
    
    tuner->warmup_ticks = 100;
    tuner->learning_enabled = false;  // Disabled until explicitly enabled
    
    return tuner;
}

void svpf_tuner_destroy(SVPFNaturalGradientTuner* tuner) {
    free(tuner);
}

void svpf_tuner_update(
    SVPFNaturalGradientTuner* tuner,
    const SVPFGradientDiagnostics* diag,
    SVPFParams* params_out,
    float* nu_out
) {
    // TODO: Implement natural gradient update
    // For now, just return current params
    svpf_tuner_get_params(tuner, params_out, nu_out);
}

void svpf_tuner_get_params(
    const SVPFNaturalGradientTuner* tuner,
    SVPFParams* params_out,
    float* nu_out
) {
    if (!tuner) return;
    
    if (params_out) {
        params_out->mu = tuner->theta.mu;
        params_out->rho = svpf_constrain_rho(tuner->theta.eta);
        params_out->sigma_z = svpf_constrain_sigma(tuner->theta.kappa);
        params_out->gamma = 0.0f;  // Not learned
    }
    
    if (nu_out) {
        *nu_out = svpf_constrain_nu(tuner->theta.kappa_nu);
    }
}

// =============================================================================
// Synthetic Test: Verify Gradient Direction
// =============================================================================

void svpf_test_nu_gradient_synthetic(int n_particles, int n_stein_steps) {
    printf("=== ν Gradient Synthetic Test ===\n\n");
    
    // True model parameters
    const float true_mu = -3.5f;
    const float true_rho = 0.95f;
    const float true_sigma = 0.15f;
    const float true_nu = 5.0f;
    const int T = 500;
    
    SVPFParams params = {true_rho, true_sigma, true_mu, 0.0f};
    
    // Generate synthetic data from true model
    printf("Generating %d observations from SV model with ν=%.1f...\n", T, true_nu);
    
    float* y = (float*)malloc(T * sizeof(float));
    float h = true_mu;
    
    // Simple synthetic data generation (host-side)
    srand(12345);
    for (int t = 0; t < T; t++) {
        // Transition
        float eps_h = ((float)rand() / RAND_MAX - 0.5f) * 3.46f;  // ~N(0,1) approx
        h = true_mu + true_rho * (h - true_mu) + true_sigma * eps_h;
        
        // Observation (Student-t approximation via mixture)
        float u = (float)rand() / RAND_MAX;
        float eps_y = ((float)rand() / RAND_MAX - 0.5f) * 3.46f;
        // Add tail: with prob 0.1, scale noise by 3
        if (u < 0.1f) eps_y *= 3.0f;
        
        float vol = expf(h * 0.5f);
        y[t] = vol * eps_y;
    }
    
    // Test 1: ν too HIGH (ν=30, almost Gaussian)
    printf("\nTest 1: ν=30 (too high, should want to DECREASE)...\n");
    {
        SVPFState* state = svpf_create(n_particles, n_stein_steps, 30.0f, nullptr);
        svpf_initialize(state, &params, 12345);
        
        float grad_sum = 0.0f;
        float y_prev = 0.0f;
        for (int t = 0; t < T; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(state, y[t], y_prev, &params, &loglik, &vol, &h_mean);
            
            float nu_grad;
            svpf_compute_nu_diagnostic_simple(state, y[t], &nu_grad, nullptr);
            grad_sum += nu_grad;
            y_prev = y[t];
        }
        
        float mean_grad = grad_sum / T;
        printf("  Mean ν gradient: %+.6f\n", mean_grad);
        printf("  Expected: NEGATIVE (want to decrease ν toward 5)\n");
        printf("  Result: %s\n", mean_grad < -0.001f ? "PASS" : "FAIL");
        
        svpf_destroy(state);
    }
    
    // Test 2: ν too LOW (ν=3)
    printf("\nTest 2: ν=3 (too low, should want to INCREASE)...\n");
    {
        SVPFState* state = svpf_create(n_particles, n_stein_steps, 3.0f, nullptr);
        svpf_initialize(state, &params, 12345);
        
        float grad_sum = 0.0f;
        float y_prev = 0.0f;
        for (int t = 0; t < T; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(state, y[t], y_prev, &params, &loglik, &vol, &h_mean);
            
            float nu_grad;
            svpf_compute_nu_diagnostic_simple(state, y[t], &nu_grad, nullptr);
            grad_sum += nu_grad;
            y_prev = y[t];
        }
        
        float mean_grad = grad_sum / T;
        printf("  Mean ν gradient: %+.6f\n", mean_grad);
        printf("  Expected: POSITIVE (want to increase ν toward 5)\n");
        printf("  Result: %s\n", mean_grad > 0.001f ? "PASS" : "FAIL");
        
        svpf_destroy(state);
    }
    
    // Test 3: ν CORRECT (ν=5)
    printf("\nTest 3: ν=5 (correct, gradient should be ~0)...\n");
    {
        SVPFState* state = svpf_create(n_particles, n_stein_steps, 5.0f, nullptr);
        svpf_initialize(state, &params, 12345);
        
        float grad_sum = 0.0f;
        float y_prev = 0.0f;
        for (int t = 0; t < T; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(state, y[t], y_prev, &params, &loglik, &vol, &h_mean);
            
            float nu_grad;
            svpf_compute_nu_diagnostic_simple(state, y[t], &nu_grad, nullptr);
            grad_sum += nu_grad;
            y_prev = y[t];
        }
        
        float mean_grad = grad_sum / T;
        printf("  Mean ν gradient: %+.6f\n", mean_grad);
        printf("  Expected: ~0 (at equilibrium)\n");
        printf("  Result: %s\n", fabsf(mean_grad) < 0.01f ? "PASS" : "MARGINAL");
        
        svpf_destroy(state);
    }
    
    free(y);
    printf("\n=== Test Complete ===\n");
}

// =============================================================================
// Main (for standalone testing)
// =============================================================================

#ifdef SVPF_GRADIENT_DIAGNOSTIC_MAIN

int main() {
    // Run synthetic verification
    svpf_test_nu_gradient_synthetic(4096, 5);
    return 0;
}

#endif
