/**
 * @file svpf_opt_kernels.cu
 * @brief CUDA kernel definitions for optimized SVPF paths
 * 
 * All __global__ kernel implementations for svpf_optimized.cu and svpf_optimized_graph.cu
 * 
 * NEW: Adaptive scout kernels that read scale from device memory
 */

#include "svpf_kernels.cuh"
#include <stdio.h>

// =============================================================================
// Device Helpers (file-local)
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 20.0f));
}

// -----------------------------------------------------------------------------
// Student-t Random Number Generator
// 
// Generates samples from Student-t distribution with nu degrees of freedom.
// Heavy tails naturally produce occasional large jumps ("Levy flights").
// No mixture model needed - every particle is a potential scout.
//
// For nu=3: ~1% chance of |x| > 4σ (vs ~0.006% for Gaussian)
// For nu=5: ~0.3% chance of |x| > 4σ
// -----------------------------------------------------------------------------

__device__ __forceinline__ float curand_student_t(
    curandStatePhilox4_32_10_t* state,
    float nu
) {
    // T = Z * sqrt(nu / V) where Z ~ N(0,1), V ~ Chi-squared(nu)
    // Chi-squared(nu) = sum of nu squared standard normals
    
    float z = curand_normal(state);
    
    // For nu <= 4, generate exact chi-squared
    // For nu > 4, use normal approximation (chi-sq concentrates)
    float chi_sq;
    if (nu <= 2.0f) {
        float v1 = curand_normal(state);
        float v2 = curand_normal(state);
        chi_sq = v1*v1 + v2*v2;
    } else if (nu <= 4.0f) {
        float v1 = curand_normal(state);
        float v2 = curand_normal(state);
        float v3 = curand_normal(state);
        chi_sq = v1*v1 + v2*v2 + v3*v3;
        if (nu > 3.0f) {
            float v4 = curand_normal(state);
            chi_sq += v4*v4;
        }
    } else {
        // For larger nu, use 5 samples (good enough, tails still heavier than Gaussian)
        float v1 = curand_normal(state);
        float v2 = curand_normal(state);
        float v3 = curand_normal(state);
        float v4 = curand_normal(state);
        float v5 = curand_normal(state);
        chi_sq = v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5;
    }
    
    // Clamp to avoid division by near-zero (extreme tail events)
    chi_sq = fmaxf(chi_sq, 0.01f);
    
    return z * sqrtf(nu / chi_sq);
}

// Convenience version for nu=3 (very heavy tails, good default)
__device__ __forceinline__ float curand_student_t_nu3(
    curandStatePhilox4_32_10_t* state
) {
    float z = curand_normal(state);
    float v1 = curand_normal(state);
    float v2 = curand_normal(state);
    float v3 = curand_normal(state);
    float chi_sq = fmaxf(v1*v1 + v2*v2 + v3*v3, 0.01f);
    return z * sqrtf(3.0f / chi_sq);
}

// =============================================================================
// IMQ (Inverse Multiquadric) Kernel Helpers
//
// k(x,y) = (c² + ||x-y||²)^(-β)  where typically β = 0.5
//
// Unlike RBF (Gaussian kernel) which decays exponentially:
//   RBF: k(x,y) = exp(-||x-y||² / 2h²)  → vanishes for distant particles
//   IMQ: k(x,y) = (1 + ||x-y||²/h²)^(-0.5) → polynomial decay, "infinite vision"
//
// This means particles always feel some attraction, even from far away.
// Solves the "vanishing gradient trap" without needing guide densities.
// =============================================================================

__device__ __forceinline__ float imq_kernel(float diff, float bw_sq) {
    // k(x,y) = (1 + (x-y)² / bw²)^(-0.5) = 1 / sqrt(1 + diff²/bw²)
    float base = 1.0f + (diff * diff) / bw_sq;
    return rsqrtf(base);
}

__device__ __forceinline__ float imq_kernel_grad(float diff, float bw_sq, float K) {
    // grad_x k(x,y) = -0.5 * (1 + diff²/bw²)^(-1.5) * 2*diff/bw²
    //              = -K³ * diff / bw²
    return -K * K * K * diff / bw_sq;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_min(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_min(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : 1e10f;
    if (wid == 0) val = warp_reduce_min(val);
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : -1e10f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

// =============================================================================
// Kernel 1: Predict (Standard)
// =============================================================================

__global__ void svpf_predict_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    int t,
    float rho, float sigma_z, float mu, float gamma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    h_prev[i] = h_i;
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    
    float noise = curand_normal(&rng[i]);
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    h[i] = clamp_logvol(mu + rho * (h_i - mu) + sigma_z * noise + leverage);
}

// =============================================================================
// Kernel 1-HeavyTail: Predict with Student-t Noise
// 
// REPLACES all MIM/Scout/Adaptive logic with principled heavy-tailed noise.
// 
// Key insight: Gaussian noise effectively never explores beyond 3σ.
// Student-t with ν=3-5 naturally generates occasional large jumps ("Levy flights")
// making every particle a potential scout without mixture heuristics.
//
// Parameters eliminated:
//   - mim_jump_prob, mim_jump_scale (no mixture)
//   - use_adaptive_scouts, phi_ema, etc. (not needed)
//   - elite_guard_* (not needed)
//
// One parameter: predict_nu (degrees of freedom)
//   - ν=3: Very heavy tails, ~1% of |x|>4σ
//   - ν=5: Moderate tails, ~0.3% of |x|>4σ
// =============================================================================

__global__ void svpf_predict_student_t_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    int t,
    float rho, float sigma_z, float mu, float gamma,
    float predict_nu,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    h_prev[i] = h_i;
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    // Heavy-tailed noise: natural exploration without heuristics
    float noise = curand_student_t(&rng[i], predict_nu);
    
    float prior_mean = mu + rho * (h_i - mu) + leverage;
    h[i] = clamp_logvol(prior_mean + sigma_z * noise);
}

// =============================================================================
// Kernel 1-HeavyTail-Asymmetric: Student-t with asymmetric rho
// =============================================================================

__global__ void svpf_predict_student_t_asymmetric_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    int t,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float predict_nu,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    // Asymmetric persistence
    float rho = (h_i > h_prev_i) ? rho_up : rho_down;
    
    // Heavy-tailed noise
    float noise = curand_student_t(&rng[i], predict_nu);
    
    float prior_mean = mu + rho * (h_i - mu) + leverage;
    h[i] = clamp_logvol(prior_mean + sigma_z * noise);
}

// =============================================================================
// Kernel 1b: Predict with MIM + Asymmetric ρ (Fixed scale)
// =============================================================================

__global__ void svpf_predict_mim_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,
    int t,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float jump_prob, float jump_scale,
    float delta_rho, float delta_sigma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;

    float h_bar = *d_h_mean;
    float dev = h_i - h_bar;
    float abs_dev = fabsf(dev);
    float tanh_dev = tanhf(dev);

    float rho_adjust = delta_rho * tanh_dev;
    float sigma_scale = 1.0f + delta_sigma * abs_dev;

    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);

    float scale = (selector < jump_prob) ? jump_scale : 1.0f;
    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    float sigma_local = sigma_z * sigma_scale;

    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);

    float prior_mean = mu + rho * (h_i - mu) + leverage;
    h[i] = clamp_logvol(prior_mean + sigma_local * scale * noise);
}

// =============================================================================
// Kernel 1b-Adaptive: Predict with MIM + Elite Guard Strategy
// 
// Split scouts into two tiers:
//   - 2% Elite Guard: Fixed scale=5.0 (always watching, no latency)
//   - 3% Reserves: Adaptive scale (reactive reinforcements)
//   - 95% Sheep: scale=1.0 (normal particles)
// 
// This addresses two problems with pure adaptive scouts:
//   1. Latency: Elite guard catches jumps immediately (0-frame response)
//   2. Vanishing gradient: Elite guard provides gradient signal even when
//      clustered particles can't "see" the target
// =============================================================================

__global__ void svpf_predict_mim_adaptive_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,
    const float* __restrict__ d_adaptive_scale,  // Reactive scale for reserves
    int t,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float jump_prob,  // Total scout probability (e.g., 0.05 = 5%)
    float delta_rho, float delta_sigma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Read adaptive scale from device memory (graph-compatible)
    float adaptive_scale = *d_adaptive_scale;
    
    // Elite Guard parameters (hardcoded for robustness)
    const float elite_prob = 0.02f;      // 2% always-on elite guard
    const float elite_scale = 5.0f;      // Fixed high scale for elite
    // Remaining (jump_prob - elite_prob) are adaptive reserves

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;

    float h_bar = *d_h_mean;
    float dev = h_i - h_bar;
    float abs_dev = fabsf(dev);
    float tanh_dev = tanhf(dev);

    float rho_adjust = delta_rho * tanh_dev;
    float sigma_scale = 1.0f + delta_sigma * abs_dev;

    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);

    // Elite Guard Strategy:
    // - [0, 0.02): Elite Guard - fixed scale=5.0 (always proactive)
    // - [0.02, 0.05): Reserves - adaptive scale (reactive)
    // - [0.05, 1.0): Sheep - scale=1.0
    float scale = 1.0f;
    if (selector < elite_prob) {
        // Elite Guard: always watching
        scale = elite_scale;
    } else if (selector < jump_prob) {
        // Reserves: reactive reinforcements
        scale = adaptive_scale;
    }
    // else: Sheep (scale = 1.0)

    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    float sigma_local = sigma_z * sigma_scale;

    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);

    float prior_mean = mu + rho * (h_i - mu) + leverage;
    h[i] = clamp_logvol(prior_mean + sigma_local * scale * noise);
}

// =============================================================================
// Kernel 1c: Guided Predict with Lookahead (Fixed scale)
// =============================================================================

__global__ void svpf_predict_guided_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,
    int t,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float jump_prob, float jump_scale,
    float delta_rho, float delta_sigma,
    float alpha_base, float alpha_shock,
    float innovation_threshold,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;

    float h_bar = *d_h_mean;
    float dev = h_i - h_bar;
    float abs_dev = fabsf(dev);
    float tanh_dev = tanhf(dev);

    float rho_adjust = delta_rho * tanh_dev;
    float sigma_scale = 1.0f + delta_sigma * abs_dev;

    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);
    float scale = (selector < jump_prob) ? jump_scale : 1.0f;

    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    float sigma_local = sigma_z * sigma_scale;

    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    float mean_prior = mu + rho * (h_i - mu) + leverage;

    float y_curr = d_y[t];
    float log_y2 = logf(y_curr * y_curr + 1e-10f);
    float mean_implied = fmaxf(log_y2 + 1.27f, -5.0f);

    float innovation = mean_implied - mean_prior;
    float total_std = 2.5f;
    float z_score = innovation / total_std;

    float activation = 0.0f;
    if (z_score > innovation_threshold) {
        activation = tanhf(z_score - innovation_threshold);
    }

    float guided_alpha = alpha_base + (alpha_shock - alpha_base) * activation;
    float mean_proposal = (1.0f - guided_alpha) * mean_prior + guided_alpha * mean_implied;
    
    h[i] = clamp_logvol(mean_proposal + sigma_local * scale * noise);
}

// =============================================================================
// Kernel 1c-Adaptive: Guided Predict with Elite Guard Strategy
// =============================================================================

__global__ void svpf_predict_guided_adaptive_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,
    const float* __restrict__ d_adaptive_scale,  // Reactive scale for reserves
    int t,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float jump_prob,  // Total scout probability
    float delta_rho, float delta_sigma,
    float alpha_base, float alpha_shock,
    float innovation_threshold,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Read adaptive scale from device memory
    float adaptive_scale = *d_adaptive_scale;
    
    // Elite Guard parameters
    const float elite_prob = 0.02f;
    const float elite_scale = 5.0f;

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;

    float h_bar = *d_h_mean;
    float dev = h_i - h_bar;
    float abs_dev = fabsf(dev);
    float tanh_dev = tanhf(dev);

    float rho_adjust = delta_rho * tanh_dev;
    float sigma_scale = 1.0f + delta_sigma * abs_dev;

    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);
    
    // Elite Guard Strategy
    float scale = 1.0f;
    if (selector < elite_prob) {
        scale = elite_scale;
    } else if (selector < jump_prob) {
        scale = adaptive_scale;
    }

    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    float sigma_local = sigma_z * sigma_scale;

    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    float mean_prior = mu + rho * (h_i - mu) + leverage;

    float y_curr = d_y[t];
    float log_y2 = logf(y_curr * y_curr + 1e-10f);
    float mean_implied = fmaxf(log_y2 + 1.27f, -5.0f);

    float innovation = mean_implied - mean_prior;
    float total_std = 2.5f;
    float z_score = innovation / total_std;

    float activation = 0.0f;
    if (z_score > innovation_threshold) {
        activation = tanhf(z_score - innovation_threshold);
    }

    float guided_alpha = alpha_base + (alpha_shock - alpha_base) * activation;
    float mean_proposal = (1.0f - guided_alpha) * mean_prior + guided_alpha * mean_implied;
    
    h[i] = clamp_logvol(mean_proposal + sigma_local * scale * noise);
}

// =============================================================================
// Kernel 2b: Mixture Prior Gradient (O(N²))
// =============================================================================

__global__ void svpf_mixture_prior_grad_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior,
    float rho, float sigma_z, float mu,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma_z_sq);
    
    float log_r_max = -1e10f;
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float log_r_i = -diff * diff * inv_2sigma_sq;
        log_r_max = fmaxf(log_r_max, log_r_i);
    }
    
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float log_r_i = -diff * diff * inv_2sigma_sq;
        float r_i = expf(log_r_i - log_r_max);
        
        sum_r += r_i;
        weighted_grad += r_i * (-diff / sigma_z_sq);
    }
    
    grad_prior[j] = weighted_grad / (sum_r + 1e-8f);
}

// Tiled version for large N
__global__ void svpf_mixture_prior_grad_tiled_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior,
    float rho, float sigma_z, float mu,
    int n
) {
    __shared__ float sh_h_prev[TILE_J];
    __shared__ float sh_mu_i[TILE_J];
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma_z_sq);
    
    float log_r_max = -1e10f;
    
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();
        
        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            log_r_max = fmaxf(log_r_max, log_r_i);
        }
    }
    
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();
        
        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            float r_i = expf(log_r_i - log_r_max);
            
            sum_r += r_i;
            weighted_grad += r_i * (-diff / sigma_z_sq);
        }
    }
    
    grad_prior[j] = weighted_grad / (sum_r + 1e-8f);
}

// =============================================================================
// Kernel 2b-Student-t: Mixture Prior Gradient (Heavy-tailed)
// 
// CRITICAL: Must be used with Student-t noise to maintain consistency.
// 
// The Gaussian gradient has linear "spring" force: F ∝ -x
// The Student-t gradient has "redescending" force: F ∝ -x / (1 + x²)
// 
// This means:
//   - Small deviations: Similar to Gaussian (pulls back)
//   - Large deviations: Force saturates and decreases (accepts outliers)
// 
// Without this, Student-t noise + Gaussian gradient = negative bias
// (particles jump far, then get crushed back by linear gradient)
// =============================================================================

__global__ void svpf_mixture_prior_grad_student_t_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior,
    float rho, float sigma_z, float mu,
    float nu,  // Degrees of freedom (match predict_nu)
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    float h_j = h[j];
    float sigma_sq = sigma_z * sigma_z + 1e-8f;
    float nu_sigma_sq = nu * sigma_sq;
    
    // Gradient prefactor: -(nu+1) / (nu * sigma^2)
    float grad_factor = -(nu + 1.0f) / nu_sigma_sq;

    float log_p_max = -1e10f;
    
    // First pass: find max log_p for numerical stability
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float diff_sq = diff * diff;
        
        // Log-PDF of Student-t (up to constant):
        // log p = -0.5*(nu+1) * log(1 + diff^2 / (nu*sigma^2))
        float term = 1.0f + diff_sq / nu_sigma_sq;
        float log_p = -0.5f * (nu + 1.0f) * logf(term);
        
        log_p_max = fmaxf(log_p_max, log_p);
    }

    float sum_p = 0.0f;
    float weighted_grad = 0.0f;

    // Second pass: compute weighted gradient
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float diff_sq = diff * diff;
        
        float term = 1.0f + diff_sq / nu_sigma_sq;
        float log_p = -0.5f * (nu + 1.0f) * logf(term);
        
        float p_i = expf(log_p - log_p_max);
        
        // Student-t score function: grad = factor * diff / term
        // Note: term in denominator makes this "redescending"
        float grad_i = grad_factor * diff / term;

        sum_p += p_i;
        weighted_grad += p_i * grad_i;
    }

    grad_prior[j] = weighted_grad / (sum_p + 1e-10f);
}

// Tiled version for large N
__global__ void svpf_mixture_prior_grad_student_t_tiled_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior,
    float rho, float sigma_z, float mu,
    float nu,
    int n
) {
    __shared__ float sh_h_prev[TILE_J];
    __shared__ float sh_mu_i[TILE_J];
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    float h_j = h[j];
    float sigma_sq = sigma_z * sigma_z + 1e-8f;
    float nu_sigma_sq = nu * sigma_sq;
    float grad_factor = -(nu + 1.0f) / nu_sigma_sq;

    float log_p_max = -1e10f;

    // First pass: find max
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();

        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float diff_sq = diff * diff;
            float term = 1.0f + diff_sq / nu_sigma_sq;
            float log_p = -0.5f * (nu + 1.0f) * logf(term);
            log_p_max = fmaxf(log_p_max, log_p);
        }
    }

    float sum_p = 0.0f;
    float weighted_grad = 0.0f;

    // Second pass: compute weighted gradient
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();

        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float diff_sq = diff * diff;
            float term = 1.0f + diff_sq / nu_sigma_sq;
            float log_p = -0.5f * (nu + 1.0f) * logf(term);
            float p_i = expf(log_p - log_p_max);
            float grad_i = grad_factor * diff / term;
            sum_p += p_i;
            weighted_grad += p_i * grad_i;
        }
    }

    grad_prior[j] = weighted_grad / (sum_p + 1e-10f);
}

// =============================================================================
// Kernel 2c: Likelihood-Only Gradient
// =============================================================================

__global__ void svpf_likelihood_only_kernel(
    const float* __restrict__ h,
    float* __restrict__ grad_lik,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    int t,
    float nu, float student_t_const,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float y_t = d_y[t];
    
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    float log_y2 = logf(y_t * y_t + 1e-10f);
    float offset = -1.0f / nu;
    float R_noise = 1.4f;
    
    grad_lik[i] = (log_y2 - h_i - offset) / R_noise;
}

// =============================================================================
// Kernel 2d: Combine Gradients
// =============================================================================

__global__ void svpf_combine_gradients_kernel(
    const float* __restrict__ grad_prior,
    const float* __restrict__ grad_lik,
    float* __restrict__ grad,
    float beta,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float g = grad_prior[i] + beta * grad_lik[i];
    grad[i] = fminf(fmaxf(g, -10.0f), 10.0f);
}

// =============================================================================
// Kernel 2e: Hessian Preconditioning
// =============================================================================

__global__ void svpf_hessian_precond_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad_combined,
    float* __restrict__ precond_grad,
    float* __restrict__ inv_hessian,
    const float* __restrict__ d_y,
    int t,
    float nu, float sigma_z,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float y = d_y[t];
    float vol = safe_exp(h_i);
    float y_sq = y * y;
    
    float A = y_sq / (nu * vol + 1e-8f);
    float one_plus_A = 1.0f + A;
    float hess_lik = -0.5f * (nu + 1.0f) * (A / (one_plus_A * one_plus_A));
    
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float hess_prior = -1.0f / sigma_z_sq;
    
    float total_hessian = hess_lik + hess_prior;
    float curvature = -total_hessian;
    curvature = fmaxf(curvature, 0.1f);
    curvature = fminf(curvature, 100.0f);
    
    float inv_H = 1.0f / curvature;
    float grad_i = grad_combined[i];
    float damping = 0.7f;
    
    precond_grad[i] = damping * grad_i * inv_H;
    inv_hessian[i] = inv_H;
}

// =============================================================================
// Kernel 3: Log-Sum-Exp
// =============================================================================

__global__ void svpf_logsumexp_kernel(
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_max_log_w,
    int t,
    int n
) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    
    float local_max = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, log_w[i]);
    }
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();
    
    local_max = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : -1e10f;
    if (wid == 0) local_max = warp_reduce_max(local_max);
    
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    local_max = s_max;
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf(log_w[i] - local_max);
    }
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        d_loglik[t] = local_max + logf(local_sum / (float)n + 1e-10f);
        *d_max_log_w = local_max;
    }
}

// =============================================================================
// Kernel 4: Bandwidth
// =============================================================================

__global__ void svpf_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float alpha,
    int n
) {
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    local_sum = block_reduce_sum(local_sum);
    __syncthreads();
    local_sum_sq = block_reduce_sum(local_sum_sq);
    
    if (threadIdx.x == 0) {
        float mean = local_sum / (float)n;
        float variance = local_sum_sq / (float)n - mean * mean;
        
        float bw_sq_new = 2.0f * variance / logf((float)n + 1.0f);
        bw_sq_new = fmaxf(bw_sq_new, 1e-6f);
        
        float bw_sq_prev = *d_bandwidth_sq;
        float bw_sq = (bw_sq_prev > 0.0f) 
                    ? alpha * bw_sq_new + (1.0f - alpha) * bw_sq_prev 
                    : bw_sq_new;
        
        float bw = sqrtf(bw_sq);
        bw = fmaxf(fminf(bw, 2.0f), 0.01f);
        
        *d_bandwidth_sq = bw_sq;
        *d_bandwidth = bw;
    }
}

// =============================================================================
// Kernel 5a: 2D Tiled Stein Kernel
// =============================================================================

__global__ void svpf_stein_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    int i = blockIdx.x;
    int tile = blockIdx.y;
    int tile_start = tile * TILE_J;
    int tile_end = min(tile_start + TILE_J, n);
    int tile_size = tile_end - tile_start;
    
    if (i >= n || tile_start >= n) return;
    
    __shared__ float sh_h[TILE_J];
    __shared__ float sh_grad[TILE_J];
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        sh_h[j] = h[tile_start + j];
        sh_grad[j] = grad[tile_start + j];
    }
    __syncthreads();
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        float diff = h_i - sh_h[j];
        float K = expf(-diff * diff / (2.0f * bw_sq));
        k_sum += K * sh_grad[j];
        gk_sum += -K * diff / bw_sq;
    }
    
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5-IMQ: 2D Tiled Stein Kernel with IMQ (Inverse Multiquadric)
// 
// Replaces Gaussian RBF kernel with IMQ for "infinite vision":
//   RBF: k(x,y) = exp(-||x-y||²/2h²)     - exponential decay, vanishes at distance
//   IMQ: k(x,y) = (1 + ||x-y||²/h²)^(-β) - polynomial decay, always interacts
//
// With IMQ, distant particles still feel attraction. Solves vanishing gradient
// problem without needing guide densities or teleportation heuristics.
// =============================================================================

__global__ void svpf_stein_imq_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    int i = blockIdx.x;
    int tile = blockIdx.y;
    int tile_start = tile * TILE_J;
    int tile_end = min(tile_start + TILE_J, n);
    int tile_size = tile_end - tile_start;
    
    if (i >= n || tile_start >= n) return;
    
    __shared__ float sh_h[TILE_J];
    __shared__ float sh_grad[TILE_J];
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        sh_h[j] = h[tile_start + j];
        sh_grad[j] = grad[tile_start + j];
    }
    __syncthreads();
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        float diff = h_i - sh_h[j];
        float diff_sq = diff * diff;
        
        // IMQ kernel: k(x,y) = (1 + diff²/bw²)^(-0.5) = 1/sqrt(base)
        float base = 1.0f + diff_sq / bw_sq;
        float K = rsqrtf(base);
        
        // Gradient: ∇_x k = -0.5 * base^(-1.5) * 2*diff/bw² = -K³ * diff/bw²
        float grad_K = -K * K * K * diff / bw_sq;
        
        k_sum += K * sh_grad[j];
        gk_sum += grad_K;
    }
    
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5a-Newton: 2D Tiled Newton-Stein Kernel
// =============================================================================

__global__ void svpf_stein_newton_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    int i = blockIdx.x;
    int tile = blockIdx.y;
    int tile_start = tile * TILE_J;
    int tile_end = min(tile_start + TILE_J, n);
    int tile_size = tile_end - tile_start;
    
    if (i >= n || tile_start >= n) return;
    
    __shared__ float sh_h[TILE_J];
    __shared__ float sh_precond_grad[TILE_J];
    __shared__ float sh_inv_hess[TILE_J];
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        sh_h[j] = h[tile_start + j];
        sh_precond_grad[j] = precond_grad[tile_start + j];
        sh_inv_hess[j] = inv_hessian[tile_start + j];
    }
    __syncthreads();
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        float diff = h_i - sh_h[j];
        float K = expf(-diff * diff / (2.0f * bw_sq));
        float inv_H_j = sh_inv_hess[j];
        
        k_sum += K * sh_precond_grad[j];
        gk_sum += (-K * diff / bw_sq) * inv_H_j;
    }
    
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5a-Newton-IMQ: 2D Tiled Newton-Stein with IMQ
// =============================================================================

__global__ void svpf_stein_newton_imq_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    int i = blockIdx.x;
    int tile = blockIdx.y;
    int tile_start = tile * TILE_J;
    int tile_end = min(tile_start + TILE_J, n);
    int tile_size = tile_end - tile_start;
    
    if (i >= n || tile_start >= n) return;
    
    __shared__ float sh_h[TILE_J];
    __shared__ float sh_precond_grad[TILE_J];
    __shared__ float sh_inv_hess[TILE_J];
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        sh_h[j] = h[tile_start + j];
        sh_precond_grad[j] = precond_grad[tile_start + j];
        sh_inv_hess[j] = inv_hessian[tile_start + j];
    }
    __syncthreads();
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        float diff = h_i - sh_h[j];
        float diff_sq = diff * diff;
        
        // IMQ kernel
        float base = 1.0f + diff_sq / bw_sq;
        float K = rsqrtf(base);
        float grad_K = -K * K * K * diff / bw_sq;
        
        float inv_H_j = sh_inv_hess[j];
        
        k_sum += K * sh_precond_grad[j];
        gk_sum += grad_K * inv_H_j;
    }
    
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5b: Persistent CTA Stein Kernel
// =============================================================================

__global__ void svpf_stein_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    float* sh_reduce = smem + 2 * n;
    
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_grad[j] = grad[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float K = expf(-diff * diff / (2.0f * bw_sq));
            k_sum += K * sh_grad[j];
            gk_sum += -K * diff / bw_sq;
        }
        
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();
    }
}

// =============================================================================
// Kernel 5b-IMQ: Persistent CTA Stein Kernel with IMQ
// =============================================================================

__global__ void svpf_stein_imq_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    float* sh_reduce = smem + 2 * n;
    
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_grad[j] = grad[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float diff_sq = diff * diff;
            
            // IMQ kernel
            float base = 1.0f + diff_sq / bw_sq;
            float K = rsqrtf(base);
            float grad_K = -K * K * K * diff / bw_sq;
            
            k_sum += K * sh_grad[j];
            gk_sum += grad_K;
        }
        
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();
    }
}

// =============================================================================
// Kernel 5b-Newton: Newton-Stein Persistent Kernel
// =============================================================================

__global__ void svpf_stein_newton_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_precond_grad = smem + n;
    float* sh_inv_hess = smem + 2 * n;
    float* sh_reduce = smem + 3 * n;
    
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_precond_grad[j] = precond_grad[j];
        sh_inv_hess[j] = inv_hessian[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float K = expf(-diff * diff / (2.0f * bw_sq));
            float inv_H_j = sh_inv_hess[j];
            
            k_sum += K * sh_precond_grad[j];
            gk_sum += (-K * diff / bw_sq) * inv_H_j;
        }
        
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();
    }
}

// =============================================================================
// Kernel 5b-Newton-IMQ: Newton-Stein Persistent Kernel with IMQ
// =============================================================================

__global__ void svpf_stein_newton_imq_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_precond_grad = smem + n;
    float* sh_inv_hess = smem + 2 * n;
    float* sh_reduce = smem + 3 * n;
    
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_precond_grad[j] = precond_grad[j];
        sh_inv_hess[j] = inv_hessian[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float diff_sq = diff * diff;
            
            // IMQ kernel
            float base = 1.0f + diff_sq / bw_sq;
            float K = rsqrtf(base);
            float grad_K = -K * K * K * diff / bw_sq;
            
            float inv_H_j = sh_inv_hess[j];
            
            k_sum += K * sh_precond_grad[j];
            gk_sum += grad_K * inv_H_j;
        }
        
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();
    }
}

// =============================================================================
// Transport Kernel: Fused SVLD + RMSProp
// =============================================================================

__global__ void svpf_apply_transport_svld_kernel(
    float* __restrict__ h,
    const float* __restrict__ phi,
    float* __restrict__ v,
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    float base_step_size,
    float beta_anneal_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float phi_i = phi[i];
    float v_prev = v[i];

    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v[i] = v_new;

    float effective_step = base_step_size * beta_anneal_factor;
    float preconditioner = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_i * preconditioner;

    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng_states[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }

    h[i] = clamp_logvol(h[i] + drift + diffusion);
}

// =============================================================================
// Guide Kernels
// =============================================================================

__global__ void svpf_apply_guide_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float delta = guide_mean - h[i];
    h[i] = clamp_logvol(h[i] + guide_strength * delta);
}

__global__ void svpf_apply_guide_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_guide_mean,
    float guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float guide_mean = *d_guide_mean;
    float delta = guide_mean - h[i];
    h[i] = clamp_logvol(h[i] + guide_strength * delta);
}

__global__ void svpf_apply_guide_preserving_kernel(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    float guide_mean,
    float guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float current_mean = *d_h_mean;
    float h_val = h[i];
    
    float deviation = h_val - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    
    h[i] = clamp_logvol(new_mean + deviation);
}

__global__ void svpf_apply_guide_preserving_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    const float* __restrict__ d_guide_mean,
    float guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float current_mean = *d_h_mean;
    float guide_mean = *d_guide_mean;
    float h_val = h[i];
    
    float deviation = h_val - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    
    h[i] = clamp_logvol(new_mean + deviation);
}

// =============================================================================
// Adaptive Bandwidth Kernels
// =============================================================================

__global__ void svpf_adaptive_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    float new_return,
    float ema_alpha,
    int n
) {
    float local_min = 1e10f, local_max = -1e10f;
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        local_sum += val;
    }
    
    local_min = block_reduce_min(local_min);
    __syncthreads();
    local_max = block_reduce_max(local_max);
    __syncthreads();
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        float spread = local_max - local_min;
        
        float abs_ret = fabsf(new_return);
        float ret_ema = *d_return_ema;
        float ret_var = *d_return_var;
        
        ret_ema = (ret_ema > 0.0f) 
                ? ema_alpha * abs_ret + (1.0f - ema_alpha) * ret_ema 
                : abs_ret;
        ret_var = (ret_var > 0.0f)
                ? ema_alpha * abs_ret * abs_ret + (1.0f - ema_alpha) * ret_var
                : abs_ret * abs_ret;
        
        *d_return_ema = ret_ema;
        *d_return_var = ret_var;
        
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        float spread_factor = fminf(spread / 2.0f, 2.0f);
        float combined_signal = fmaxf(vol_ratio, spread_factor);
        
        float alpha = 1.0f - 0.25f * fminf(combined_signal - 1.0f, 2.0f);
        alpha = fmaxf(fminf(alpha, 1.0f), 0.5f);
        
        float bw = *d_bandwidth;
        *d_bandwidth = bw * alpha;
    }
}

__global__ void svpf_adaptive_bandwidth_kernel_graph(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    const float* __restrict__ d_y,
    int y_idx,
    float ema_alpha,
    int n
) {
    float local_min = 1e10f, local_max = -1e10f;
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        local_sum += val;
    }
    
    local_min = block_reduce_min(local_min);
    __syncthreads();
    local_max = block_reduce_max(local_max);
    __syncthreads();
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        float spread = local_max - local_min;
        float new_return = d_y[y_idx];
        
        float abs_ret = fabsf(new_return);
        float ret_ema = *d_return_ema;
        float ret_var = *d_return_var;
        
        ret_ema = (ret_ema > 0.0f) 
                ? ema_alpha * abs_ret + (1.0f - ema_alpha) * ret_ema 
                : abs_ret;
        ret_var = (ret_var > 0.0f)
                ? ema_alpha * abs_ret * abs_ret + (1.0f - ema_alpha) * ret_var
                : abs_ret * abs_ret;
        
        *d_return_ema = ret_ema;
        *d_return_var = ret_var;
        
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        float spread_factor = fminf(spread / 2.0f, 2.0f);
        float combined_signal = fmaxf(vol_ratio, spread_factor);
        
        float alpha = 1.0f - 0.25f * fminf(combined_signal - 1.0f, 2.0f);
        alpha = fmaxf(fminf(alpha, 1.0f), 0.5f);
        
        float bw = *d_bandwidth;
        *d_bandwidth = bw * alpha;
    }
}

// =============================================================================
// Output Kernels
// =============================================================================

__global__ void svpf_vol_mean_opt_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_vol,
    int t,
    int n
) {
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += safe_exp(h[i] / 2.0f);
    }
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        d_vol[t] = local_sum / (float)n;
    }
}

__global__ void svpf_store_h_mean_kernel(
    const float* __restrict__ d_sum,
    float* __restrict__ d_h_mean_prev,
    int n
) {
    if (threadIdx.x == 0) {
        *d_h_mean_prev = *d_sum / (float)n;
    }
}

// =============================================================================
// Graph-Compatible Utility Kernels
// =============================================================================

__global__ void svpf_memset_kernel(float* __restrict__ data, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

__global__ void svpf_h_mean_reduce_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_partial_sums,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        sum += h[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void svpf_h_mean_finalize_kernel(
    const float* __restrict__ d_partial_sums,
    float* __restrict__ d_h_mean,
    int n_blocks,
    int n_particles
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = tid; i < n_blocks; i += blockDim.x) {
        sum += d_partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *d_h_mean = sdata[0] / (float)n_particles;
    }
}

// =============================================================================
// NEW: Phi Stress Computation Kernel (for Adaptive Scouts)
// Uses atomicAdd for multi-block safety. Caller must memset d_phi_stress to 0
// before launch and divide by N after.
// =============================================================================

__global__ void svpf_compute_phi_stress_kernel(
    const float* __restrict__ phi,
    float* __restrict__ d_phi_stress,
    int n
) {
    float local_sum = 0.0f;
    
    // Grid-stride loop for robustness with any launch config
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum += fabsf(phi[i]);
    }
    
    // Block reduction
    local_sum = block_reduce_sum(local_sum);
    
    // Atomic accumulation (safe across multiple blocks)
    if (threadIdx.x == 0) {
        atomicAdd(d_phi_stress, local_sum);
    }
}

// =============================================================================
// Host Function: Update Adaptive Scout Scale
// =============================================================================

void svpf_update_adaptive_scouts(SVPFState* state, float phi_stress) {
    if (!state->use_adaptive_scouts) {
        state->adaptive_scout_scale = state->mim_jump_scale;
        return;
    }
    
    // EMA update
    float alpha = state->phi_ema_alpha;
    if (state->phi_ema > 0.0f) {
        state->phi_ema = alpha * phi_stress + (1.0f - alpha) * state->phi_ema;
    } else {
        state->phi_ema = phi_stress;
    }
    
    // Sigmoid activation: smoothly ramps from min_scale to max_scale
    float threshold = state->phi_stress_threshold;
    float softness = state->phi_stress_softness;
    float min_scale = state->min_scout_scale;
    float max_scale = state->max_scout_scale;
    
    float x = (state->phi_ema - threshold) / softness;
    float activation = 1.0f / (1.0f + expf(-x));  // sigmoid in [0, 1]
    
    state->adaptive_scout_scale = min_scale + (max_scale - min_scale) * activation;
}
