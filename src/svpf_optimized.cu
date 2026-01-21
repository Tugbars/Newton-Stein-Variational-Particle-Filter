/**
 * @file svpf_optimized.cu
 * @brief High-performance SVPF with full SM utilization
 * 
 * FIXES from review:
 * - Predict kernel runs ONCE (samples noise), separate from likelihood/grad
 * - NO device→host copies in timestep loop - all scalars on device
 * - Persistent CTA for small N path (true "read once" behavior)
 * - Bandwidth uses variance-based heuristic with EMA smoothing
 * 
 * Key optimizations:
 * 1. 2D tiled grid for O(N²) Stein kernel - guarantees SM saturation
 * 2. Shared memory tiling - bandwidth efficient
 * 3. CUB reductions for log-sum-exp
 * 4. Small N path: Persistent CTA, all data in SMEM, no atomics
 * 5. CUDA Graph compatible (no mid-loop synchronization)
 * 
 * References:
 * - Liu & Wang (2016): SVGD algorithm
 * - Fan et al. (2021): Stein Particle Filtering
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Configuration
// =============================================================================

#define TILE_J 256                   // Match BLOCK_SIZE for full thread utilization
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SMALL_N_THRESHOLD 4096
#define BANDWIDTH_UPDATE_INTERVAL 5
#define MAX_T_SIZE 10000             // Max sequence length for staging buffer

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 20.0f));
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

// Block reduction - result valid in thread 0 only
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
// Kernel 0: Increment timestep (for CUDA Graph path)
// =============================================================================

__global__ void svpf_increment_timestep_kernel(int* d_timestep) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_timestep)++;
    }
}

// =============================================================================
// Kernel 1: Predict (Run ONCE per timestep - samples process noise)
// =============================================================================

// Version with explicit t (for non-graph path)
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
// Kernel 1b: Predict with Mixture Innovation Model (MIM) + Asymmetric ρ
//
// Innovation is a mixture of Gaussians:
//   (1-p) * N(0, σ²) + p * N(0, (jump_scale*σ)²)
//
// Asymmetric persistence:
//   ρ_eff = ρ_up   if h > h_prev  (vol increasing - persists longer)
//   ρ_eff = ρ_down if h ≤ h_prev  (vol decreasing - mean-reverts faster)
//
// Captures empirical fact: volatility spikes fast, decays slow
// =============================================================================

__global__ void svpf_predict_mim_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    int t,
    float rho_up,         // Persistence when vol increasing (e.g., 0.98)
    float rho_down,       // Persistence when vol decreasing (e.g., 0.93)
    float sigma_z, float mu, float gamma,
    float jump_prob,      // e.g., 0.05 (5% of particles get large innovation)
    float jump_scale,     // e.g., 5.0 (5x std dev for jump component)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;  // Store for next step

    // Sample innovation noise
    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);

    // Branchless mixture selection (compiles to CMOV, no divergence)
    float scale = (selector < jump_prob) ? jump_scale : 1.0f;

    // Asymmetric persistence: vol spikes fast, decays slow
    // Compare current h to previous h (not h_prev which we just overwrote)
    float rho = (h_i > h_prev_i) ? rho_up : rho_down;

    // Leverage effect
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);

    // Evolve state with scaled innovation
    float prior_mean = mu + rho * (h_i - mu) + leverage;
    h[i] = clamp_logvol(prior_mean + sigma_z * scale * noise);
}

// Version reading t from device memory (for CUDA Graph path)
__global__ void svpf_predict_kernel_graph(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const int* __restrict__ d_timestep,
    float rho, float sigma_z, float mu, float gamma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int t = *d_timestep;
    
    float h_i = h[i];
    h_prev[i] = h_i;
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    
    float noise = curand_normal(&rng[i]);
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    h[i] = clamp_logvol(mu + rho * (h_i - mu) + sigma_z * noise + leverage);
}

// =============================================================================
// Kernel 2: Likelihood + Gradient (Deterministic - runs after predict and 
//           after each Stein step)
// =============================================================================

__global__ void svpf_likelihood_grad_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    int t,
    float rho, float sigma_z, float mu, float nu, float student_t_const,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float h_prev_i = h_prev[i];
    float y_t = d_y[t];
    
    // Likelihood (Student-t)
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    // Gradient
    float mu_prior = mu + rho * (h_prev_i - mu);
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
    float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
    
    grad[i] = fminf(fmaxf(grad_prior + grad_lik, -10.0f), 10.0f);
}

// Graph version - reads t from device memory
__global__ void svpf_likelihood_grad_kernel_graph(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    const int* __restrict__ d_timestep,
    float rho, float sigma_z, float mu, float nu, float student_t_const,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int t = *d_timestep;
    float h_i = h[i];
    float h_prev_i = h_prev[i];
    float y_t = d_y[t];
    
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    float mu_prior = mu + rho * (h_prev_i - mu);
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
    float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
    
    grad[i] = fminf(fmaxf(grad_prior + grad_lik, -10.0f), 10.0f);
}

// =============================================================================
// Kernel 2b: Mixture Prior Gradient (O(N²) - CORRECT per paper Eq. 6)
//
// The paper specifies the prior as a Gaussian mixture:
//   p(h_{t+1} | Z_t) = (1/n) * Σᵢ p(h_{t+1} | h_t^i)
//
// where each component is: p(h | h_prev^i) = N(μᵢ, σ_z²)
// with μᵢ = μ + ρ(h_prev^i - μ)
//
// The gradient is:
//   ∇_h log p_prior(h) = Σᵢ rᵢ(h) * (-(h - μᵢ)/σ_z²)
//
// where rᵢ(h) = p(h|h_prev^i) / Σⱼ p(h|h_prev^j) is the responsibility
//
// This replaces the WRONG per-particle prior with the CORRECT mixture prior.
// =============================================================================

// Small N version: each thread handles one particle j, loops over all i
__global__ void svpf_mixture_prior_grad_kernel(
    const float* __restrict__ h,           // Current particles [N]
    const float* __restrict__ h_prev,      // Previous particles [N]
    float* __restrict__ grad_prior,        // Output: mixture prior gradient [N]
    float rho,
    float sigma_z,
    float mu,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma_z_sq);
    
    // First pass: find max log-responsibility for numerical stability
    float log_r_max = -1e10f;
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float log_r_i = -diff * diff * inv_2sigma_sq;
        log_r_max = fmaxf(log_r_max, log_r_i);
    }
    
    // Second pass: compute weighted gradient with log-sum-exp stability
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float log_r_i = -diff * diff * inv_2sigma_sq;
        float r_i = expf(log_r_i - log_r_max);  // Numerically stable
        
        sum_r += r_i;
        weighted_grad += r_i * (-diff / sigma_z_sq);
    }
    
    // Normalize
    grad_prior[j] = weighted_grad / (sum_r + 1e-8f);
}

// Tiled version for large N: uses shared memory for h_prev tiles
__global__ void svpf_mixture_prior_grad_tiled_kernel(
    const float* __restrict__ h,           // Current particles [N]
    const float* __restrict__ h_prev,      // Previous particles [N]
    float* __restrict__ grad_prior,        // Output: mixture prior gradient [N]
    float rho,
    float sigma_z,
    float mu,
    int n
) {
    __shared__ float sh_h_prev[TILE_J];
    __shared__ float sh_mu_i[TILE_J];
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma_z_sq);
    
    // First pass: find max log-responsibility across all tiles
    float log_r_max = -1e10f;
    
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        // Cooperative load of h_prev tile
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();
        
        // Find max in this tile
        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            log_r_max = fmaxf(log_r_max, log_r_i);
        }
    }
    
    // Second pass: compute weighted gradient
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        // Cooperative load (already in smem from first pass if single tile,
        // but need to reload for multi-tile case)
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();
        
        // Accumulate weighted gradient
        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            float r_i = expf(log_r_i - log_r_max);
            
            sum_r += r_i;
            weighted_grad += r_i * (-diff / sigma_z_sq);
        }
    }
    
    // Normalize
    grad_prior[j] = weighted_grad / (sum_r + 1e-8f);
}

// =============================================================================
// Kernel 2c: Likelihood-Only Gradient (O(N) - just the observation term)
//
// Computes only the likelihood gradient, to be combined with mixture prior.
// Also computes log_weights for importance sampling.
// =============================================================================

__global__ void svpf_likelihood_only_kernel(
    const float* __restrict__ h,
    float* __restrict__ grad_lik,          // Output: likelihood gradient only [N]
    float* __restrict__ log_w,             // Output: log weights [N]
    const float* __restrict__ d_y,         // Observations array
    int t,                                 // Current timestep
    float nu,
    float student_t_const,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float y_t = d_y[t];
    
    // Likelihood (Student-t)
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    // Likelihood gradient only
    grad_lik[i] = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
}

// Graph version - reads timestep from device memory
__global__ void svpf_likelihood_only_kernel_graph(
    const float* __restrict__ h,
    float* __restrict__ grad_lik,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    const int* __restrict__ d_timestep,
    float nu,
    float student_t_const,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int t = *d_timestep;
    float h_i = h[i];
    float y_t = d_y[t];
    
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    grad_lik[i] = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
}

// =============================================================================
// Kernel 2d: Combine Gradients (prior + likelihood)
// =============================================================================

__global__ void svpf_combine_gradients_kernel(
    const float* __restrict__ grad_prior,  // Mixture prior gradient [N]
    const float* __restrict__ grad_lik,    // Likelihood gradient [N]
    float* __restrict__ grad,              // Output: combined gradient [N]
    float beta,                            // Annealing factor (1.0 = full likelihood)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float g = grad_prior[i] + beta * grad_lik[i];
    grad[i] = fminf(fmaxf(g, -10.0f), 10.0f);
}

// =============================================================================
// Kernel 3: Log-Sum-Exp (computes loglik, all on device)
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

// Graph version - reads t from device memory
__global__ void svpf_logsumexp_kernel_graph(
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_max_log_w,
    const int* __restrict__ d_timestep,
    int n
) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    
    int t = *d_timestep;
    
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

// Multi-block version for large N using CUB
__global__ void svpf_exp_shifted_kernel(
    const float* __restrict__ log_w,
    const float* __restrict__ d_max_log_w,  // Read from device
    float* __restrict__ exp_w,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    exp_w[i] = expf(log_w[i] - *d_max_log_w);
}

__global__ void svpf_finalize_loglik_kernel(
    const float* __restrict__ d_max_log_w,
    const float* __restrict__ d_sum_exp,
    float* __restrict__ d_loglik,
    int t,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_loglik[t] = *d_max_log_w + logf(*d_sum_exp / (float)n + 1e-10f);
    }
}

// =============================================================================
// Kernel 4: Bandwidth (variance-based, all on device, with EMA)
// =============================================================================

__global__ void svpf_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,      // In/out: current bandwidth (for EMA)
    float* __restrict__ d_bandwidth_sq,   // In/out: EMA of bandwidth squared
    float alpha,                          // EMA factor
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
        
        // Bandwidth squared: h² = 2 * var / log(n+1)
        float bw_sq_new = 2.0f * variance / logf((float)n + 1.0f);
        bw_sq_new = fmaxf(bw_sq_new, 1e-6f);
        
        // EMA on bandwidth squared
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
// Kernel 5a: 2D Tiled Stein Kernel (Large N path)
// =============================================================================

__global__ void svpf_stein_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,              // Output: transport direction
    const float* __restrict__ d_bandwidth, // Read from device
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
    
    // Cooperative load
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
    
    // Block reduction
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5b: Persistent CTA Stein Kernel (Small N path)
// =============================================================================

/**
 * Persistent CTA approach:
 * - Launch one block per SM (or small multiple)
 * - Each block loads ALL data once into SMEM
 * - Grid-stride over i: block b handles particles b, b+gridDim.x, b+2*gridDim.x...
 * - Within each i: threads cooperate on j summation
 * 
 * CRITICAL: Writes to phi[], NOT h[] - avoids race condition.
 * Caller must apply transport separately with svpf_apply_transport_kernel.
 */
__global__ void svpf_stein_persistent_kernel(
    const float* __restrict__ h,      // READ-ONLY
    const float* __restrict__ grad,
    float* __restrict__ phi,          // OUTPUT: transport direction
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    float* sh_reduce = smem + 2 * n;  // Workspace for reduction
    
    // Load ALL data once (cooperative across threads)
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_grad[j] = grad[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    // Grid-stride loop over particles
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        // Threads cooperate on j summation
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float K = expf(-diff * diff / (2.0f * bw_sq));
            k_sum += K * sh_grad[j];
            gk_sum += -K * diff / bw_sq;
        }
        
        // Block reduction for k_sum
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        // Block reduction for gk_sum
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        // Thread 0 writes transport direction to phi (NOT h)
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();  // Ensure write completes before next iteration
    }
}

// =============================================================================
// Kernel 6: Apply Transport (for 2D kernel path) - Basic version
// =============================================================================

__global__ void svpf_apply_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ phi,
    float step_size,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = clamp_logvol(h[i] + step_size * phi[i]);
}

// =============================================================================
// ADAPTIVE SVPF KERNELS
// =============================================================================

// -----------------------------------------------------------------------------
// Kernel A1: Fused SVLD + RMSProp Transport (Preconditioned SVLD)
//
// Implements Preconditioned Stein Variational Langevin Descent:
//   1. RMSProp: v = ρ*v + (1-ρ)*φ²  (adaptive learning rate)
//   2. Drift:   h += ε*β*φ/√(v+eps)  (preconditioned gradient)
//   3. Diffusion: h += √(2*ε*β*T)*η  (Langevin noise for diversity)
//
// temperature = 0 → deterministic SVGD (no noise)
// temperature = 1 → full SVLD (theoretically correct)
// temperature > 1 → extra exploration (useful early, anneal down)
// -----------------------------------------------------------------------------

__global__ void svpf_apply_transport_svld_kernel(
    float* __restrict__ h,
    const float* __restrict__ phi,
    float* __restrict__ v,                              // RMSProp state [N]
    curandStatePhilox4_32_10_t* __restrict__ rng_states,// RNG states [N]
    float base_step_size,
    float beta_anneal_factor,    // Scales step during annealing (sqrt(beta) or beta)
    float temperature,           // 0.0 = SVGD, 1.0 = SVLD, >1 = extra exploration
    float rho_rmsprop,           // RMSProp decay (0.9-0.99)
    float epsilon,               // Stability (1e-6)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float phi_i = phi[i];
    float v_prev = v[i];

    // 1. Update RMSProp (Second Moment)
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v[i] = v_new;

    // 2. Calculate Deterministic Update (Drift)
    float effective_step = base_step_size * beta_anneal_factor;
    float preconditioner = rsqrtf(v_new + epsilon);  // 1/sqrt(v)
    float drift = effective_step * phi_i * preconditioner;

    // 3. Calculate Stochastic Update (Diffusion) - Langevin noise
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng_states[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }

    // 4. Apply & Clamp
    h[i] = clamp_logvol(h[i] + drift + diffusion);
}

// -----------------------------------------------------------------------------
// Kernel A1b: Apply Guide Density (Affine Re-centering)
//
// Shifts particles toward EKF guide mean without destroying diversity.
// Called once per timestep, right after predict, before Stein iterations.
//
// h[i] = h[i] + strength * (guide_mean - h[i])
//      = (1 - strength) * h[i] + strength * guide_mean
// -----------------------------------------------------------------------------

__global__ void svpf_apply_guide_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_strength,   // 0.1-0.3 recommended
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float delta = guide_mean - h[i];
    h[i] = clamp_logvol(h[i] + guide_strength * delta);
}

// -----------------------------------------------------------------------------
// Kernel A2: Adaptive Bandwidth Scaling (Improvement #1)
//
// Detects high-vol regime via particle spread and scales bandwidth α:
//   - High spread (particles dispersed) → α = 0.5-0.6 (tighter kernel)
//   - Low spread (particles clustered) → α = 1.0 (standard kernel)
//
// This is called AFTER standard bandwidth computation to scale it.
// -----------------------------------------------------------------------------

__global__ void svpf_adaptive_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,      // Modified in-place
    float* __restrict__ d_return_ema,     // EMA of |return|
    float* __restrict__ d_return_var,     // EMA of return variance
    float new_return,                     // Current observation
    float ema_alpha,                      // EMA smoothing (0.05)
    int n
) {
    // Compute particle spread (single block)
    float local_min = 1e10f, local_max = -1e10f;
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        local_sum += val;
    }
    
    // Reduce to get min/max/mean
    local_min = block_reduce_min(local_min);
    __syncthreads();
    local_max = block_reduce_max(local_max);
    __syncthreads();
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        float spread = local_max - local_min;
        float mean_h = local_sum / (float)n;
        
        // Update return EMA for regime detection
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
        
        // Compute adaptive alpha based on regime
        // High return magnitude relative to EMA → high vol regime → lower alpha
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        
        // Also consider particle spread - high spread means uncertainty
        // Typical spread in calm: 0.5-1.0, in crisis: 2.0-4.0
        float spread_factor = fminf(spread / 2.0f, 2.0f);  // Normalized
        
        // Combine signals: higher vol_ratio or spread → lower alpha
        float combined_signal = fmaxf(vol_ratio, spread_factor);
        
        // Map to alpha: signal=1 → α=1.0, signal=3+ → α=0.5
        float alpha = 1.0f - 0.25f * fminf(combined_signal - 1.0f, 2.0f);
        alpha = fmaxf(fminf(alpha, 1.0f), 0.5f);
        
        // Scale bandwidth
        float bw = *d_bandwidth;
        *d_bandwidth = bw * alpha;
    }
}

// -----------------------------------------------------------------------------
// Kernel A3: Annealed Gradient Computation (Improvement #2)
//
// Computes gradient with annealing factor β:
//   grad = β * grad_likelihood + grad_prior
//
// Called instead of standard gradient kernel during annealing passes.
// -----------------------------------------------------------------------------

__global__ void svpf_gradient_annealed_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_log_p,
    float y,
    float rho,
    float sigma_z,
    float mu,
    float gamma,
    float nu,
    float student_t_const,
    float y_prev,
    float beta,                           // Annealing factor [0, 1]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float h_prev_i = h_prev[i];
    
    // Prior gradient: d/dh log p(h|h_prev)
    // p(h|h_prev) ~ N(mu + rho*(h_prev - mu) + gamma*leverage, sigma_z²)
    float prior_mean = mu + rho * (h_prev_i - mu);
    if (fabsf(y_prev) > 1e-8f) {
        float vol_prev = safe_exp(h_prev_i / 2.0f);
        prior_mean += gamma * y_prev / vol_prev;
    }
    float grad_prior = -(h_i - prior_mean) / (sigma_z * sigma_z);
    
    // Likelihood gradient: d/dh log p(y|h)
    // Student-t: log p(y|h) = const - (nu+1)/2 * log(1 + (y/vol)²/nu) - h/2
    float vol = safe_exp(h_i / 2.0f);
    float z = y / vol;
    float z_sq_over_nu = (z * z) / nu;
    
    // d/dh log p(y|h) = -1/2 + (nu+1)/2 * (z²/nu) / (1 + z²/nu) * (-1)
    //                 = -1/2 + (nu+1)/2 * z² / (nu + z²) * (-1)
    // Simplified: grad = -0.5 + (nu+1) * z² / (2*(nu + z²))
    float grad_lik = -0.5f + ((nu + 1.0f) * z * z) / (2.0f * (nu + z * z));
    
    // Annealed gradient: β * likelihood + prior
    grad_log_p[i] = beta * grad_lik + grad_prior;
}

// =============================================================================
// Kernel 7: Volatility Mean
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

// Graph version - reads t from device memory
__global__ void svpf_vol_mean_opt_kernel_graph(
    const float* __restrict__ h,
    float* __restrict__ d_vol,
    const int* __restrict__ d_timestep,
    int n
) {
    int t = *d_timestep;
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += safe_exp(h[i] / 2.0f);
    }
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        d_vol[t] = local_sum / (float)n;
    }
}

// =============================================================================
// Host State - Uses SVPFOptimizedState from svpf.cuh (embedded in SVPFState)
// =============================================================================

// Helper to get optimized backend from state
static inline SVPFOptimizedState* get_opt(SVPFState* state) {
    return &state->opt_backend;
}

// Forward declaration
void svpf_optimized_cleanup(SVPFOptimizedState* opt);

void svpf_optimized_init(SVPFOptimizedState* opt, int n) {
    // Handle resize: if N increased, re-allocate
    if (opt->initialized && n > opt->allocated_n) {
        svpf_optimized_cleanup(opt);
    }
    if (opt->initialized) return;
    
    // Query CUB temp storage size
    float* d_dummy_in;
    float* d_dummy_out;
    cudaMalloc(&d_dummy_in, n * sizeof(float));
    cudaMalloc(&d_dummy_out, sizeof(float));
    
    opt->temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, opt->temp_storage_bytes, d_dummy_in, d_dummy_out, n);
    size_t sum_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, sum_bytes, d_dummy_in, d_dummy_out, n);
    opt->temp_storage_bytes = max(opt->temp_storage_bytes, sum_bytes);
    
    cudaMalloc(&opt->d_temp_storage, opt->temp_storage_bytes);
    cudaFree(d_dummy_in);
    cudaFree(d_dummy_out);
    
    // Device scalars
    cudaMalloc(&opt->d_max_log_w, sizeof(float));
    cudaMalloc(&opt->d_sum_exp, sizeof(float));
    cudaMalloc(&opt->d_bandwidth, sizeof(float));
    cudaMalloc(&opt->d_bandwidth_sq, sizeof(float));
    
    // Device timestep counter
    cudaMalloc(&opt->d_timestep, sizeof(int));
    
    // Initialize bandwidth_sq to 0 (triggers fresh computation on first step)
    float zero = 0.0f;
    cudaMemcpy(opt->d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Buffers
    cudaMalloc(&opt->d_exp_w, n * sizeof(float));
    cudaMalloc(&opt->d_phi, n * sizeof(float));
    cudaMalloc(&opt->d_grad_lik, n * sizeof(float));  // For mixture prior fix
    
    // Staging buffers for CUDA Graph
    cudaMalloc(&opt->d_obs_staging, MAX_T_SIZE * sizeof(float));
    cudaMalloc(&opt->d_loglik_staging, MAX_T_SIZE * sizeof(float));
    cudaMalloc(&opt->d_vol_staging, MAX_T_SIZE * sizeof(float));
    opt->staging_T = MAX_T_SIZE;
    
    // Single-step API buffers (avoid malloc in hot loop)
    cudaMalloc(&opt->d_y_single, 2 * sizeof(float));
    cudaMalloc(&opt->d_loglik_single, sizeof(float));
    cudaMalloc(&opt->d_vol_single, sizeof(float));
    
    // Graph not yet captured
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    // Create dedicated stream for graph capture (non-blocking to avoid sync with default stream)
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    
    opt->allocated_n = n;
    opt->initialized = true;
}

void svpf_optimized_cleanup(SVPFOptimizedState* opt) {
    if (!opt->initialized) return;
    
    cudaFree(opt->d_temp_storage);
    cudaFree(opt->d_max_log_w);
    cudaFree(opt->d_sum_exp);
    cudaFree(opt->d_bandwidth);
    cudaFree(opt->d_bandwidth_sq);
    cudaFree(opt->d_timestep);
    cudaFree(opt->d_exp_w);
    cudaFree(opt->d_phi);
    cudaFree(opt->d_grad_lik);
    cudaFree(opt->d_obs_staging);
    cudaFree(opt->d_loglik_staging);
    cudaFree(opt->d_vol_staging);
    cudaFree(opt->d_y_single);
    cudaFree(opt->d_loglik_single);
    cudaFree(opt->d_vol_single);
    
    // Destroy CUDA Graph if captured
    if (opt->graph_captured) {
        cudaGraphExecDestroy(opt->graphExec);
        cudaGraphDestroy(opt->graph);
        opt->graph_captured = false;
    }
    
    // Destroy dedicated stream
    if (opt->graph_stream) {
        cudaStreamDestroy(opt->graph_stream);
        opt->graph_stream = NULL;
    }
    
    opt->allocated_n = 0;
    opt->initialized = false;
}

// Wrapper for SVPFState cleanup (called by svpf_destroy)
void svpf_optimized_cleanup_state(SVPFState* state) {
    if (state) {
        svpf_optimized_cleanup(&state->opt_backend);
    }
}

// =============================================================================
// Host API: Optimized Sequence Runner (NO host sync in loop)
// =============================================================================

void svpf_run_sequence_optimized(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    int n_stein = state->n_stein_steps;
    cudaStream_t stream = state->stream;
    
    svpf_optimized_init(opt, n);
    
    // Pre-compute constant
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    // Kernel configs
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_tiles = (n + TILE_J - 1) / TILE_J;
    dim3 grid_2d(n, num_tiles);
    
    bool use_small_n = (n <= SMALL_N_THRESHOLD);
    
    // Persistent CTA config: ~1 block per SM (query at runtime)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    
    int persistent_blocks = min(num_sms, n);
    // SMEM: h[n] + grad[n] + reduce[BLOCK_SIZE/WARP_SIZE]
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    // Check if SMEM fits; if not, fall back to 2D tiled path
    if (persistent_smem > prop.sharedMemPerBlockOptin) {
        use_small_n = false;
    }
    
    // Set max dynamic shared memory for persistent kernel
    if (use_small_n) {
        cudaFuncSetAttribute(svpf_stein_persistent_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            persistent_smem);
    }
    
    float bw_alpha = 0.3f;
    
    // Main loop - NO cudaStreamSynchronize inside
    for (int t = 0; t < T; t++) {
        
        // === 1. Predict (samples noise ONCE) ===
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            d_observations, t,
            params->rho, params->sigma_z, params->mu, params->gamma,
            n
        );
        
        // === 2. Mixture Prior Gradient (O(N²) - CORRECT per paper Eq. 6) ===
        // This is the key fix: prior is a Gaussian mixture over ALL h_prev particles
        if (n <= SMALL_N_THRESHOLD) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                params->rho, params->sigma_z, params->mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                params->rho, params->sigma_z, params->mu, n
            );
        }
        
        // === 3. Likelihood Gradient + Log Weights ===
        svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            d_observations, t, state->nu, student_t_const, n
        );
        
        // === 4. Combine Gradients: grad = grad_prior + grad_lik ===
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
            1.0f,  // beta = 1.0 (no annealing in this path)
            n
        );
        
        // === 5. Log-Sum-Exp (all on device) ===
        if (n <= 4096) {
            // Single block for small N
            svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                state->log_weights, d_loglik_out, opt->d_max_log_w, t, n
            );
        } else {
            // Multi-block with CUB
            cub::DeviceReduce::Max(opt->d_temp_storage, opt->temp_storage_bytes,
                                   state->log_weights, opt->d_max_log_w, n, stream);
            
            svpf_exp_shifted_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->log_weights, opt->d_max_log_w, opt->d_exp_w, n
            );
            
            cub::DeviceReduce::Sum(opt->d_temp_storage, opt->temp_storage_bytes,
                                   opt->d_exp_w, opt->d_sum_exp, n, stream);
            
            svpf_finalize_loglik_kernel<<<1, 1, 0, stream>>>(
                opt->d_max_log_w, opt->d_sum_exp, d_loglik_out, t, n
            );
        }
        
        // === 6. Bandwidth (every few steps for stability) ===
        if (t % BANDWIDTH_UPDATE_INTERVAL == 0) {
            svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                state->h, opt->d_bandwidth, opt->d_bandwidth_sq, bw_alpha, n
            );
        }
        
        // === 7. Stein Transport Iterations ===
        for (int s = 0; s < n_stein; s++) {
            if (use_small_n) {
                // Persistent CTA path - writes to phi, then apply transport
                cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
                
                svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                    state->h, state->grad_log_p,
                    opt->d_phi, opt->d_bandwidth, n
                );
                
                // Apply transport (same as large N path)
                svpf_apply_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_phi, SVPF_STEIN_STEP_SIZE, n
                );
            } else {
                // 2D tiled path
                cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
                
                svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                    state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                );
                
                svpf_apply_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_phi, SVPF_STEIN_STEP_SIZE, n
                );
            }
            
            // Re-compute gradients for next Stein iteration (deterministic, no noise)
            if (s < n_stein - 1) {
                // Recompute mixture prior (particles moved)
                if (n <= SMALL_N_THRESHOLD) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        params->rho, params->sigma_z, params->mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        params->rho, params->sigma_z, params->mu, n
                    );
                }
                
                // Recompute likelihood gradient
                svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    d_observations, t, state->nu, student_t_const, n
                );
                
                // Combine
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
                    1.0f, n
                );
            }
        }
        
        // === 8. Volatility Mean ===
        if (d_vol_out) {
            svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                state->h, d_vol_out, t, n
            );
        }
    }
    
    // Single sync at end
    cudaStreamSynchronize(stream);
}

// =============================================================================
// Host API: CUDA Graph Sequence Runner (Fastest - minimal launch overhead)
// =============================================================================

/**
 * CUDA Graph-accelerated sequence runner.
 * 
 * Captures one timestep as a graph, then replays T times.
 * Uses staging buffers so graph captures fixed addresses.
 * 
 * Key fixes:
 * - Passthrough: If stream is already capturing, fall back to optimized path
 * - Strict sync before capture
 * - Check n_stein in invalidation
 * - Emergency fallback on capture failure
 */
void svpf_run_sequence_graph(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    int n_stein = state->n_stein_steps;
    
    // Sanity check T
    if (T > MAX_T_SIZE) {
        svpf_run_sequence_optimized(state, d_observations, T, params, d_loglik_out, d_vol_out);
        return;
    }
    
    // === CHECK STATE->STREAM CAPTURE STATUS FIRST (before any sync) ===
    // If user's stream is capturing, we cannot interfere - fall back to optimized path
    cudaStreamCaptureStatus captureStatus;
    cudaStreamIsCapturing(state->stream, &captureStatus);
    if (captureStatus != cudaStreamCaptureStatusNone) {
        svpf_run_sequence_optimized(state, d_observations, T, params, d_loglik_out, d_vol_out);
        return;
    }
    
    // Now safe to initialize (may create our dedicated stream)
    svpf_optimized_init(opt, n);
    
    // Use our dedicated graph_stream (default/NULL stream doesn't support graph capture)
    cudaStream_t stream = opt->graph_stream;
    
    // Sync with state->stream to ensure observations and particles are ready
    cudaStreamSynchronize(state->stream);
    
    // === STAGING SETUP ===
    // Copy observations to fixed staging address that graph will use
    cudaMemcpyAsync(opt->d_obs_staging, d_observations, T * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    
    // Reset counters
    int zero = 0;
    float zero_f = 0.0f;
    cudaMemcpyAsync(opt->d_timestep, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(opt->d_bandwidth_sq, &zero_f, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Pre-compute constant
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    // Kernel configs
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    bool use_small_n = (n <= SMALL_N_THRESHOLD);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    
    int persistent_blocks = min(num_sms, n);
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    if (persistent_smem > prop.sharedMemPerBlockOptin) {
        use_small_n = false;
    }
    
    if (use_small_n) {
        cudaFuncSetAttribute(svpf_stein_persistent_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            persistent_smem);
    }
    
    float bw_alpha = 0.3f;
    
    // === GRAPH CAPTURE LOGIC ===
    // Invalidation: Check n AND n_stein
    bool need_capture = !opt->graph_captured || 
                        (opt->graph_n != n) || 
                        (opt->graph_n_stein != n_stein);
    
    if (need_capture) {
        // Cleanup old graph
        if (opt->graph_captured) {
            cudaGraphExecDestroy(opt->graphExec);
            cudaGraphDestroy(opt->graph);
            opt->graph_captured = false;
        }
        
        // STRICT SYNC: Stream must be idle before capture
        cudaStreamSynchronize(stream);
        
        // Begin capture
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            svpf_run_sequence_optimized(state, d_observations, T, params, d_loglik_out, d_vol_out);
            return;
        }
        
        // === KERNEL RECORDING START ===
        
        // 1. Predict - use staging buffer
        svpf_predict_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_obs_staging, opt->d_timestep,
            params->rho, params->sigma_z, params->mu, params->gamma,
            n
        );
        
        // 2. Mixture Prior Gradient (O(N²) - CORRECT per paper Eq. 6)
        // Note: This kernel doesn't need timestep - uses h and h_prev directly
        if (n <= SMALL_N_THRESHOLD) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                params->rho, params->sigma_z, params->mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                params->rho, params->sigma_z, params->mu, n
            );
        }
        
        // 3. Likelihood Gradient + Log Weights - use staging buffer
        svpf_likelihood_only_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            opt->d_obs_staging, opt->d_timestep,
            state->nu, student_t_const, n
        );
        
        // 4. Combine Gradients: grad = grad_prior + grad_lik
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
            1.0f, n  // beta = 1.0 in graph path (no annealing)
        );
        
        // 5. Log-Sum-Exp - output to staging
        svpf_logsumexp_kernel_graph<<<1, BLOCK_SIZE, 0, stream>>>(
            state->log_weights, opt->d_loglik_staging, opt->d_max_log_w, 
            opt->d_timestep, n
        );
        
        // 6. Bandwidth
        svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_bandwidth, opt->d_bandwidth_sq, bw_alpha, n
        );
        
        // 7. Stein iterations
        for (int s = 0; s < n_stein; s++) {
            if (use_small_n) {
                // Persistent CTA path - writes to phi, then apply transport
                cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
                
                svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                    state->h, state->grad_log_p,
                    opt->d_phi, opt->d_bandwidth, n
                );
                
                svpf_apply_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_phi, SVPF_STEIN_STEP_SIZE, n
                );
            } else {
                cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
                
                int num_tiles = (n + TILE_J - 1) / TILE_J;
                dim3 grid_2d(n, num_tiles);
                svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                    state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                );
                
                svpf_apply_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_phi, SVPF_STEIN_STEP_SIZE, n
                );
            }
            
            if (s < n_stein - 1) {
                // Recompute mixture prior (particles moved)
                if (n <= SMALL_N_THRESHOLD) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        params->rho, params->sigma_z, params->mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        params->rho, params->sigma_z, params->mu, n
                    );
                }
                
                // Recompute likelihood gradient
                svpf_likelihood_only_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    opt->d_obs_staging, opt->d_timestep,
                    state->nu, student_t_const, n
                );
                
                // Combine
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
                    1.0f, n
                );
            }
        }
        
        // 8. Volatility mean - output to staging
        svpf_vol_mean_opt_kernel_graph<<<1, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_vol_staging, opt->d_timestep, n
        );
        
        // 9. Increment timestep counter
        svpf_increment_timestep_kernel<<<1, 1, 0, stream>>>(opt->d_timestep);
        
        // === KERNEL RECORDING END ===
        
        // End capture
        err = cudaStreamEndCapture(stream, &opt->graph);
        if (err != cudaSuccess) {
            opt->graph = NULL;
            svpf_run_sequence_optimized(state, d_observations, T, params, d_loglik_out, d_vol_out);
            return;
        }
        
        // Validation: Verify we captured something meaningful
        size_t numNodes = 0;
        cudaGraphGetNodes(opt->graph, NULL, &numNodes);
        
        if (numNodes < 5) {
            cudaGraphDestroy(opt->graph);
            opt->graph = NULL;
            svpf_run_sequence_optimized(state, d_observations, T, params, d_loglik_out, d_vol_out);
            return;
        }
        
        err = cudaGraphInstantiate(&opt->graphExec, opt->graph, NULL, NULL, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(opt->graph);
            opt->graph = NULL;
            svpf_run_sequence_optimized(state, d_observations, T, params, d_loglik_out, d_vol_out);
            return;
        }
        
        opt->graph_captured = true;
        opt->graph_n = n;
        opt->graph_n_stein = n_stein;
    }
    
    // === LAUNCH ===
    for (int t = 0; t < T; t++) {
        cudaGraphLaunch(opt->graphExec, stream);
    }
    
    // === COPY OUTPUTS ===
    cudaMemcpyAsync(d_loglik_out, opt->d_loglik_staging, T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    if (d_vol_out) {
        cudaMemcpyAsync(d_vol_out, opt->d_vol_staging, T * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    
    cudaStreamSynchronize(stream);
}

// =============================================================================
// EKF Guide Update (Host-side, ~20 FLOPs)
//
// Uses LOG-SQUARED observation model for stability:
//   State:   h_t = μ + ρ(h_{t-1} - μ) + σ_z ε
//   Obs:     log(y_t²) ≈ h_t + log(η²)
//
// This linearizes the observation (H=1) and handles negative returns.
// E[log(η²)] ≈ -1.27 (for χ²₁), Var ≈ 4.93
// =============================================================================

static void svpf_ekf_update(
    SVPFState* state,
    float y_t,
    const SVPFParams* p
) {
    // Initialize on first call
    if (!state->guide_initialized) {
        state->guide_mean = p->mu;
        // Stationary variance: σ² / (1 - ρ²)
        state->guide_var = p->sigma_z * p->sigma_z / (1.0f - p->rho * p->rho);
        state->guide_initialized = 1;
    }
    
    // Predict
    float m_pred = p->mu + p->rho * (state->guide_mean - p->mu);
    float P_pred = p->rho * p->rho * state->guide_var + p->sigma_z * p->sigma_z;
    
    // Log-squared observation model (linearizes the problem!)
    // Model: log(y²) = h + log(η²)
    // E[log(η²)] ≈ -1.27 for χ²₁, Var[log(η²)] ≈ 4.93
    // For Student-t, inflate variance slightly
    float log_y2 = logf(y_t * y_t + 1e-8f);
    float obs_offset = -1.27f;
    float obs_var = 4.93f + 2.0f;  // Inflated for Student-t robustness
    
    // With log-squared transform: H = 1 (linear relationship!)
    float H = 1.0f;
    float R = obs_var;
    
    // Kalman gain
    float S = H * H * P_pred + R;
    float K = P_pred * H / (S + 1e-8f);
    
    // Innovation: log(y²) - predicted log(y²)
    float y_pred = m_pred + obs_offset;
    float innovation = log_y2 - y_pred;
    
    // Update
    state->guide_mean = m_pred + K * innovation;
    state->guide_var = (1.0f - K * H) * P_pred;
    state->guide_K = K;
}

// =============================================================================
// ADAPTIVE SVPF STEP - All improvements combined
// 
// Improvements:
//   1. Mixture Innovation Model (MIM) - fat-tailed predict
//   2. Asymmetric ρ - vol spikes fast, decays slow
//   3. EKF Guide Density - coarse positioning before Stein
//   4. Adaptive bandwidth α scaling based on regime detection
//   5. Annealed Stein updates (β schedule: 0.3 → 0.65 → 1.0)
//   6. Fused RMSProp + Langevin (SVLD) for stable transport
// =============================================================================

void svpf_step_adaptive(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,    // Host pointer (copied at end)
    float* h_vol_out,       // Host pointer (copied at end)
    float* h_mean_out       // Host pointer for mean h (optional)
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t stream = state->stream;
    int t = state->timestep;
    
    svpf_optimized_init(opt, n);
    
    // Use pre-allocated buffers
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    SVPFParams p = *params;
    
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool use_small_n = (n <= SMALL_N_THRESHOLD);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int persistent_blocks = min(prop.multiProcessorCount, n);
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    if (persistent_smem > prop.sharedMemPerBlockOptin) {
        use_small_n = false;
    }
    
    // =========================================================================
    // PREDICT (with optional Mixture Innovation Model + Asymmetric ρ)
    // =========================================================================
    if (state->use_mim) {
        // MIM kernel supports asymmetric rho directly
        float rho_up = state->use_asymmetric_rho ? state->rho_up : p.rho;
        float rho_down = state->use_asymmetric_rho ? state->rho_down : p.rho;
        
        svpf_predict_mim_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, 
            rho_up, rho_down,  // Asymmetric persistence
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale, n
        );
    } else {
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, p.rho, p.sigma_z, p.mu, p.gamma, n
        );
    }
    
    // =========================================================================
    // EKF GUIDE DENSITY (coarse positioning before Stein refinement)
    // =========================================================================
    if (state->use_guide) {
        // Update EKF (host-side, ~20 FLOPs)
        svpf_ekf_update(state, y_t, &p);
        
        // Pull particles toward guide mean
        svpf_apply_guide_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->guide_mean, state->guide_strength, n
        );
    }
    
    // =========================================================================
    // BANDWIDTH with adaptive scaling (Improvement #1)
    // =========================================================================
    svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_bandwidth, opt->d_bandwidth_sq, 0.3f, n
    );
    
    // Apply adaptive scaling based on regime
    svpf_adaptive_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_bandwidth, 
        state->d_return_ema, state->d_return_var,
        y_t, 0.05f, n
    );
    
    // =========================================================================
    // ANNEALED STEIN UPDATES (Improvement #2)
    //
    // β schedule: 0.3 → 0.65 → 1.0 (configurable via state)
    // This allows particles to explore before committing to likelihood
    //
    // CORRECTED: Now uses mixture prior gradient (O(N²)) per paper Eq. 6
    // =========================================================================
    
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};  // Default schedule
    
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing 
                   ? beta_schedule[anneal_idx % 3]
                   : 1.0f;
        
        // === CORRECTED GRADIENT COMPUTATION ===
        // Step 1: Mixture prior gradient (O(N²)) - does NOT depend on β
        if (n <= SMALL_N_THRESHOLD) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        }
        
        // Step 2: Likelihood gradient (O(N))
        svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            opt->d_y_single, 1, state->nu, student_t_const, n
        );
        
        // Step 3: Combine with annealing: grad = grad_prior + β * grad_lik
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
            beta, n
        );
        
        // Stein iterations at this β level
        int stein_iters = state->n_stein_steps / n_anneal;
        if (anneal_idx == n_anneal - 1) {
            // Last annealing level gets remaining iterations
            stein_iters = state->n_stein_steps - stein_iters * (n_anneal - 1);
        }
        
        for (int s = 0; s < stein_iters; s++) {
            // Compute Stein direction
            if (use_small_n) {
                cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
                svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                    state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                );
            } else {
                cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
                int num_tiles = (n + TILE_J - 1) / TILE_J;
                dim3 grid_2d(n, num_tiles);
                svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                    state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                );
            }
            
            // Apply transport with fused SVLD (RMSProp + Langevin noise)
            // Temperature control: 0 = SVGD (deterministic), 1 = SVLD (diffusion)
            float temp = state->use_svld ? state->temperature : 0.0f;
            float beta_factor = sqrtf(beta);  // Scale step by sqrt(beta) during annealing
            
            // Reduce step size when guide handles transport (supervisor recommendation)
            float step_size = SVPF_STEIN_STEP_SIZE;
            if (state->use_guide) {
                step_size *= 0.5f;  // Guide handles transport, Stein handles geometry
            }
            
            svpf_apply_transport_svld_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, opt->d_phi,
                state->d_grad_v,             // RMSProp state
                state->rng_states,           // For Langevin noise
                step_size,
                beta_factor,                 // Annealing factor
                temp,                        // Temperature (0=SVGD, 1=SVLD)
                state->rmsprop_rho,
                state->rmsprop_eps,
                n
            );
            
            // Recompute gradient if more iterations remain
            if (s < stein_iters - 1) {
                // Mixture prior (particles moved)
                if (n <= SMALL_N_THRESHOLD) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                }
                
                // Likelihood gradient
                svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    opt->d_y_single, 1, state->nu, student_t_const, n
                );
                
                // Combine with annealing
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
                    beta, n
                );
            }
        }
    }
    
    // =========================================================================
    // FINAL LIKELIHOOD (for output)
    // =========================================================================
    svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->log_weights, opt->d_loglik_single, opt->d_max_log_w, 0, n
    );
    
    // =========================================================================
    // VOL MEAN + H MEAN
    // =========================================================================
    svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_vol_single, 0, n
    );
    
    // H mean (simple block reduce)
    // We reuse the logsumexp infrastructure to get h mean
    float* d_h_mean = state->d_result_h_mean;
    {
        // Quick inline h-mean computation
        float h_sum = 0.0f;
        cudaMemcpyAsync(&h_sum, state->d_scalar_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        // Actually, let's do this properly with a kernel call - use CUB
        cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes, 
                               state->h, state->d_scalar_sum, n, stream);
    }
    
    // Single sync + copy results
    cudaStreamSynchronize(stream);
    
    // Copy results
    float h_sum_host;
    cudaMemcpy(&h_sum_host, state->d_scalar_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (h_loglik_out) cudaMemcpy(h_loglik_out, opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_vol_out) cudaMemcpy(h_vol_out, opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_mean_out) *h_mean_out = h_sum_host / (float)n;
    
    state->timestep++;
}
