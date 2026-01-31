/**
 * @file svpf_two_factor.cu
 * @brief CUDA kernel implementations for Two-Factor SVPF
 * 
 * Coordinate-wise SVGD: Reuses existing 1D Stein transport kernel.
 * Just call it twice - once for h_fast, once for h_slow.
 */

#include "svpf_two_factor.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define SVPF_H_MIN -15.0f
#define SVPF_H_MAX 5.0f

__device__ __forceinline__ float clamp_h_2f(float h) {
    return fminf(fmaxf(h, SVPF_H_MIN), SVPF_H_MAX);
}

// =============================================================================
// INITIALIZATION KERNEL
// =============================================================================

__global__ void svpf_init_two_factor_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Stationary std: σ / sqrt(1 - ρ²)
    float std_fast = sigma_fast / sqrtf(1.0f - rho_fast * rho_fast + 1e-8f);
    float std_slow = sigma_slow / sqrtf(1.0f - rho_slow * rho_slow + 1e-8f);
    
    float z_fast = curand_normal(&rng[i]);
    float z_slow = curand_normal(&rng[i]);
    
    h_fast[i] = clamp_h_2f(std_fast * z_fast);
    h_slow[i] = clamp_h_2f(std_slow * z_slow);
}

// =============================================================================
// PREDICT KERNEL (Two-Factor)
// =============================================================================
// Saves current state to prev, then samples new state

__global__ void svpf_predict_two_factor_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    float* __restrict__ h_fast_prev,
    float* __restrict__ h_slow_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    int use_mim, float mim_jump_prob, float mim_jump_scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Save current to prev
    float hf = h_fast[i];
    float hs = h_slow[i];
    h_fast_prev[i] = hf;
    h_slow_prev[i] = hs;
    
    // Sample innovations
    float z_fast = curand_normal(&rng[i]);
    float z_slow = curand_normal(&rng[i]);
    
    // MIM jump for fast component only
    if (use_mim) {
        float u = curand_uniform(&rng[i]);
        if (u < mim_jump_prob) {
            z_fast *= mim_jump_scale;
        }
    }
    
    // AR(1) updates (mean-zero processes)
    h_fast[i] = clamp_h_2f(rho_fast * hf + sigma_fast * z_fast);
    h_slow[i] = clamp_h_2f(rho_slow * hs + sigma_slow * z_slow);
}

// =============================================================================
// GUIDED PREDICT KERNEL (Two-Factor)
// =============================================================================
// APF-style lookahead: bias toward implied h when surprise is high

__global__ void svpf_predict_two_factor_guided_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    float* __restrict__ h_fast_prev,
    float* __restrict__ h_slow_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    float mu,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    int use_mim, float mim_jump_prob, float mim_jump_scale,
    float alpha_base, float alpha_shock, float innovation_threshold,
    float implied_offset,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Save current to prev
    float hf = h_fast[i];
    float hs = h_slow[i];
    h_fast_prev[i] = hf;
    h_slow_prev[i] = hs;
    
    // Compute implied h from observation
    float y = *d_y;
    float log_y2 = logf(y * y + 1e-8f);
    float implied_h = log_y2 + implied_offset;
    
    // Current h (before update)
    float h_combined = mu + hf + hs;
    
    // Surprise magnitude
    float innovation = implied_h - h_combined;
    float abs_innovation = fabsf(innovation);
    
    // Adaptive alpha: ramp up with surprise
    float alpha = alpha_base;
    if (abs_innovation > innovation_threshold) {
        float excess = (abs_innovation - innovation_threshold) / innovation_threshold;
        alpha = fminf(alpha_base + alpha_shock * excess, 0.8f);
    }
    
    // Sample innovations
    float z_fast = curand_normal(&rng[i]);
    float z_slow = curand_normal(&rng[i]);
    
    // MIM for fast component
    if (use_mim) {
        float u = curand_uniform(&rng[i]);
        if (u < mim_jump_prob) {
            z_fast *= mim_jump_scale;
        }
    }
    
    // Standard AR(1) prediction
    float hf_pred = rho_fast * hf + sigma_fast * z_fast;
    float hs_pred = rho_slow * hs + sigma_slow * z_slow;
    
    // Apply guided correction to FAST component only
    // (Fast component responds to surprises, slow component tracks trend)
    float target_hf = implied_h - mu - hs_pred;  // What h_fast "should" be
    hf_pred = (1.0f - alpha) * hf_pred + alpha * target_hf;
    
    h_fast[i] = clamp_h_2f(hf_pred);
    h_slow[i] = clamp_h_2f(hs_pred);
}

// =============================================================================
// GRADIENT KERNEL (Two-Factor)
// =============================================================================
// Key insight: Likelihood gradient is SHARED, prior gradients are SEPARATE

__global__ void svpf_gradient_two_factor_kernel(
    const float* __restrict__ h_fast,
    const float* __restrict__ h_slow,
    const float* __restrict__ h_fast_prev,
    const float* __restrict__ h_slow_prev,
    float* __restrict__ grad_fast,
    float* __restrict__ grad_slow,
    float* __restrict__ log_w,
    float y_t, float mu,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    float beta, float nu, float student_t_const,
    int use_exact_gradient,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float hf = h_fast[i];
    float hs = h_slow[i];
    float hf_prev = h_fast_prev[i];
    float hs_prev = h_slow_prev[i];
    
    // Combined h for likelihood
    float h_combined = mu + hf + hs;
    
    // =========================================================================
    // LIKELIHOOD (Student-t observation model)
    // =========================================================================
    float exp_h = expf(h_combined);
    float y_sq = y_t * y_t;
    float scale_sq = nu * exp_h;
    float ratio = y_sq / (scale_sq + 1e-8f);
    
    // Log-likelihood for this particle
    float log_lik = student_t_const - 0.5f * h_combined 
                  - 0.5f * (nu + 1.0f) * logf(1.0f + ratio);
    log_w[i] = log_lik;
    
    // Likelihood gradient: d/dh log p(y|h)
    // = -0.5 + 0.5*(nu+1)*ratio/(1+ratio)
    float lik_grad;
    if (use_exact_gradient) {
        lik_grad = -0.5f + 0.5f * (nu + 1.0f) * ratio / (1.0f + ratio + 1e-8f);
    } else {
        // Approximation for stability
        float implied_h = logf(y_sq + 1e-8f);
        lik_grad = 0.5f * (implied_h - h_combined);
    }
    
    // =========================================================================
    // PRIOR GRADIENTS (AR(1) dynamics, mean-zero processes)
    // =========================================================================
    // p(h_t | h_{t-1}) = N(ρ * h_{t-1}, σ²)
    // d/dh log p(h) = -(h - ρ * h_prev) / σ²
    
    float sigma_fast_sq = sigma_fast * sigma_fast + 1e-8f;
    float sigma_slow_sq = sigma_slow * sigma_slow + 1e-8f;
    
    float prior_grad_fast = -(hf - rho_fast * hf_prev) / sigma_fast_sq;
    float prior_grad_slow = -(hs - rho_slow * hs_prev) / sigma_slow_sq;
    
    // =========================================================================
    // VARIANCE-WEIGHTED LIKELIHOOD SPLIT (for identifiability)
    // =========================================================================
    // Use STATIONARY variances (not innovation variances) for balanced split.
    // Stationary var = σ² / (1 - ρ²)
    // This accounts for persistence: slow component has high stationary var
    // even though its innovation var is small.
    float one_minus_rho_fast_sq = 1.0f - rho_fast * rho_fast + 1e-6f;
    float one_minus_rho_slow_sq = 1.0f - rho_slow * rho_slow + 1e-6f;
    float var_fast_stat = sigma_fast_sq / one_minus_rho_fast_sq;
    float var_slow_stat = sigma_slow_sq / one_minus_rho_slow_sq;
    
    float w_fast = var_fast_stat / (var_fast_stat + var_slow_stat);
    float w_slow = 1.0f - w_fast;
    
    grad_fast[i] = prior_grad_fast + beta * w_fast * lik_grad;
    grad_slow[i] = prior_grad_slow + beta * w_slow * lik_grad;
}

// =============================================================================
// BANDWIDTH KERNEL (Two-Factor)
// =============================================================================
// Separate bandwidth for each component with different floors

__global__ void svpf_bandwidth_two_factor_kernel(
    const float* __restrict__ h_fast,
    const float* __restrict__ h_slow,
    float* __restrict__ d_bandwidth_fast,
    float* __restrict__ d_bandwidth_slow,
    float bw_min_fast, float bw_min_slow,
    int n
) {
    __shared__ float s_sum_fast, s_sum_sq_fast;
    __shared__ float s_sum_slow, s_sum_sq_slow;
    
    if (threadIdx.x == 0) {
        s_sum_fast = 0.0f;
        s_sum_sq_fast = 0.0f;
        s_sum_slow = 0.0f;
        s_sum_sq_slow = 0.0f;
    }
    __syncthreads();
    
    // Accumulate per-thread sums
    float local_sum_fast = 0.0f, local_sum_sq_fast = 0.0f;
    float local_sum_slow = 0.0f, local_sum_sq_slow = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float hf = h_fast[i];
        float hs = h_slow[i];
        
        local_sum_fast += hf;
        local_sum_sq_fast += hf * hf;
        local_sum_slow += hs;
        local_sum_sq_slow += hs * hs;
    }
    
    atomicAdd(&s_sum_fast, local_sum_fast);
    atomicAdd(&s_sum_sq_fast, local_sum_sq_fast);
    atomicAdd(&s_sum_slow, local_sum_slow);
    atomicAdd(&s_sum_sq_slow, local_sum_sq_slow);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        float n_factor = powf((float)n, -0.2f);  // Silverman exponent
        
        // Fast component
        float mean_fast = s_sum_fast * inv_n;
        float var_fast = s_sum_sq_fast * inv_n - mean_fast * mean_fast;
        float std_fast = sqrtf(fmaxf(var_fast, 1e-8f));
        float bw_fast = 1.06f * std_fast * n_factor;
        *d_bandwidth_fast = fmaxf(bw_fast, bw_min_fast);
        
        // Slow component (higher floor to prevent collapse)
        float mean_slow = s_sum_slow * inv_n;
        float var_slow = s_sum_sq_slow * inv_n - mean_slow * mean_slow;
        float std_slow = sqrtf(fmaxf(var_slow, 1e-8f));
        float bw_slow = 1.06f * std_slow * n_factor;
        *d_bandwidth_slow = fmaxf(bw_slow, bw_min_slow);
    }
}

// =============================================================================
// APPLY GUIDE KERNEL (Two-Factor) - PRESERVING VERSION
// =============================================================================
// Shifts mean toward guide while preserving deviations (particle diversity)

__global__ void svpf_apply_guide_two_factor_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    float guide_mean_fast, float guide_mean_slow,
    float current_mean_fast, float current_mean_slow,  // Current particle means
    float guide_strength,
    int use_preserving,  // If 1, preserve deviations; if 0, simple blend
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float gs = guide_strength;
    float hf = h_fast[i];
    float hs = h_slow[i];
    
    float hf_new, hs_new;
    
    if (use_preserving) {
        // Preserving: shift mean while keeping deviations
        float dev_fast = hf - current_mean_fast;
        float dev_slow = hs - current_mean_slow;
        
        float new_mean_fast = (1.0f - gs) * current_mean_fast + gs * guide_mean_fast;
        float new_mean_slow = (1.0f - gs) * current_mean_slow + gs * guide_mean_slow;
        
        hf_new = new_mean_fast + dev_fast;
        hs_new = new_mean_slow + dev_slow;
    } else {
        // Simple blend (collapses diversity)
        hf_new = (1.0f - gs) * hf + gs * guide_mean_fast;
        hs_new = (1.0f - gs) * hs + gs * guide_mean_slow;
    }
    
    h_fast[i] = clamp_h_2f(hf_new);
    h_slow[i] = clamp_h_2f(hs_new);
}

// =============================================================================
// DEVICE HELPER: Safe block reduce max for floats
// =============================================================================
__device__ float block_reduce_max_2f(float val) {
    __shared__ float shared[32]; // Shared mem for warp leaders
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // Write warp leaders to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory and reduce
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e30f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

// =============================================================================
// OUTPUT KERNEL (Two-Factor) - FIXED Jensen bias
// =============================================================================
// Computes vol = mean(exp(h/2)) NOT exp(mean(h)/2) to avoid Jensen bias

__global__ void svpf_outputs_two_factor_kernel(
    const float* __restrict__ h_fast,
    const float* __restrict__ h_slow,
    const float* __restrict__ log_w,
    float mu,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    float* __restrict__ d_h_fast_mean,
    float* __restrict__ d_h_slow_mean,
    int n
) {
    __shared__ float s_sum_fast;
    __shared__ float s_sum_slow;
    __shared__ float s_sum_vol;  // Jensen-correct vol accumulator
    __shared__ float s_max_w;
    __shared__ float s_sum_exp;
    
    if (threadIdx.x == 0) {
        s_sum_fast = 0.0f;
        s_sum_slow = 0.0f;
        s_sum_vol = 0.0f;
        s_sum_exp = 0.0f;
    }
    __syncthreads();
    
    // Pass 1: Sums and find max log-weight
    float local_sum_fast = 0.0f;
    float local_sum_slow = 0.0f;
    float local_sum_vol = 0.0f;
    float local_max = -1e30f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float hf = h_fast[i];
        float hs = h_slow[i];
        float h_total = mu + hf + hs;
        
        local_sum_fast += hf;
        local_sum_slow += hs;
        local_sum_vol += expf(0.5f * h_total);  // Jensen-correct: mean(exp(h/2))
        local_max = fmaxf(local_max, log_w[i]);
    }
    
    atomicAdd(&s_sum_fast, local_sum_fast);
    atomicAdd(&s_sum_slow, local_sum_slow);
    atomicAdd(&s_sum_vol, local_sum_vol);
    
    // Safe block reduce max (NO atomicMax on floats!)
    float block_max = block_reduce_max_2f(local_max);
    if (threadIdx.x == 0) {
        s_max_w = block_max;
    }
    __syncthreads();
    
    // Pass 2: Compute sum of exp(log_w - max) for log-sum-exp
    float max_w = s_max_w;
    float local_sum_exp = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum_exp += expf(log_w[i] - max_w);
    }
    atomicAdd(&s_sum_exp, local_sum_exp);
    __syncthreads();
    
    // Write outputs
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        
        float h_fast_mean = s_sum_fast * inv_n;
        float h_slow_mean = s_sum_slow * inv_n;
        float h_mean = mu + h_fast_mean + h_slow_mean;
        
        *d_h_mean = h_mean;
        *d_vol = s_sum_vol * inv_n;  // Jensen-correct: mean(exp(h/2))
        *d_loglik = max_w + logf(fmaxf(s_sum_exp * inv_n, 1e-10f));
        
        if (d_h_fast_mean) *d_h_fast_mean = h_fast_mean;
        if (d_h_slow_mean) *d_h_slow_mean = h_slow_mean;
    }
}

// =============================================================================
// CONVENIENCE: Initialize RMSProp states
// =============================================================================

__global__ void svpf_init_rmsprop_two_factor_kernel(
    float* __restrict__ d_grad_v_fast,
    float* __restrict__ d_grad_v_slow,
    float init_val,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_grad_v_fast[i] = init_val;
        d_grad_v_slow[i] = init_val;
    }
}
