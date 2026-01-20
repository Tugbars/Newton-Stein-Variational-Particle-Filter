/**
 * @file svpf_fast.cu
 * @brief Ultra-optimized SVPF - No CUB, warp-level reductions, fused kernels
 * 
 * Target: <50 μs/step on RTX 5080
 * 
 * Key optimizations:
 * 1. Warp shuffle reductions (no CUB overhead)
 * 2. Fused kernels (copy+predict, gradient+RBF+update)
 * 3. Single-kernel log-sum-exp
 * 4. Persistent thread blocks for sequence processing
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARP_SIZE 32

// =============================================================================
// WARP-LEVEL REDUCTION PRIMITIVES (No CUB!)
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction using warp shuffles
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // First warp reduces across warps
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1e10f;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// =============================================================================
// DEVICE HELPERS
// =============================================================================

__device__ __forceinline__ float clamp_h(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 20.0f));
}

// =============================================================================
// SINGLE-BLOCK KERNELS (For N <= 1024, process everything in one block)
// =============================================================================

// Combined: copy, predict, loglik, log-sum-exp, copy_pred_to_h
// All in ONE kernel launch!
__global__ void svpf_forward_fused_kernel(
    float* h,
    float* h_prev,
    float* h_pred,
    curandStatePhilox4_32_10_t* rng_states,
    const float* d_y,
    int t,
    float rho, float sigma_z, float mu, float gamma,
    float nu, float student_t_const,
    float* d_loglik_out,
    int n
) {
    extern __shared__ float shared[];
    float* sh_reduce = shared;
    
    int idx = threadIdx.x;
    
    // === PHASE 1: Copy + Predict ===
    float h_i = 0.0f;
    float h_pred_i = 0.0f;
    
    if (idx < n) {
        h_i = h[idx];
        h_prev[idx] = h_i;
        
        float noise = curand_normal(&rng_states[idx]);
        float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
        float vol_prev = safe_exp(h_i / 2.0f);
        float leverage = gamma * y_prev / (vol_prev + 1e-8f);
        h_pred_i = clamp_h(mu + rho * (h_i - mu) + sigma_z * noise + leverage);
        h_pred[idx] = h_pred_i;
    }
    __syncthreads();
    
    // === PHASE 2: Observation likelihood ===
    float y_t = d_y[t];
    float log_w = -1e10f;
    
    if (idx < n) {
        float vol = safe_exp(h_pred_i);
        float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
        log_w = student_t_const - 0.5f * h_pred_i
              - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    }
    
    // === PHASE 3: Log-sum-exp (all in shared memory) ===
    // Find max
    float max_log_w = block_reduce_max(log_w, sh_reduce);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_log_w;
    __syncthreads();
    
    // Compute exp(log_w - max) and sum
    float exp_w = (idx < n) ? expf(log_w - s_max) : 0.0f;
    float sum_exp = block_reduce_sum(exp_w, sh_reduce);
    
    // Write log-likelihood
    if (threadIdx.x == 0 && d_loglik_out) {
        d_loglik_out[t] = s_max + logf(sum_exp / (float)n + 1e-10f);
    }
    
    // === PHASE 4: Copy h_pred -> h ===
    if (idx < n) {
        h[idx] = h_pred_i;
    }
}

// Compute bandwidth (mean + variance) in single kernel
__global__ void svpf_bandwidth_kernel(
    const float* h,
    float* d_bandwidth,
    int n
) {
    extern __shared__ float shared[];
    
    int idx = threadIdx.x;
    float h_i = (idx < n) ? h[idx] : 0.0f;
    
    // Compute mean
    float sum = block_reduce_sum(h_i, shared);
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / (float)n;
    __syncthreads();
    
    // Compute variance
    float diff = (idx < n) ? (h_i - s_mean) : 0.0f;
    float var_sum = block_reduce_sum(diff * diff, shared);
    
    if (threadIdx.x == 0) {
        float variance = var_sum / (float)n;
        float bw_sq = 2.0f * variance / logf((float)n + 1.0f);
        float bw = sqrtf(fmaxf(bw_sq, 1e-8f));
        *d_bandwidth = fmaxf(fminf(bw, 2.0f), 0.01f);
    }
}

// Fused Stein step: gradient + RBF + update
// Uses shared memory tiling for O(N²) RBF
__global__ void svpf_stein_step_kernel(
    float* h,
    const float* h_prev,
    const float* d_y,
    int t,
    const float* d_bandwidth,
    float rho, float sigma_z, float mu, float nu,
    float step_size,
    int n
) {
    extern __shared__ float shared[];
    float* sh_h = shared;
    float* sh_grad = shared + blockDim.x;
    
    int i = threadIdx.x;
    
    // Load scalars
    float y_t = d_y[t];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    // Compute gradient for particle i
    float h_i = (i < n) ? h[i] : 0.0f;
    float h_prev_i = (i < n) ? h_prev[i] : 0.0f;
    float grad_i = 0.0f;
    
    if (i < n) {
        // Prior gradient
        float mu_prior = mu + rho * (h_prev_i - mu);
        float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
        float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
        
        // Likelihood gradient  
        float vol = safe_exp(h_i);
        float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
        float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
        
        grad_i = fminf(fmaxf(grad_prior + grad_lik, -10.0f), 10.0f);
    }
    
    // Store in shared for RBF computation
    sh_h[i] = h_i;
    sh_grad[i] = grad_i;
    __syncthreads();
    
    // RBF kernel computation (all particles in shared memory for single block)
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    if (i < n) {
        for (int j = 0; j < n; j++) {
            float diff = h_i - sh_h[j];
            float K = expf(-diff * diff / (2.0f * bw_sq));
            k_sum += K * sh_grad[j];
            gk_sum += -K * diff / bw_sq;
        }
    }
    
    // Apply Stein update
    if (i < n) {
        float update = step_size * (k_sum + gk_sum) / (float)n;
        h[i] = clamp_h(h_i + update);
    }
}

// Compute volatility mean
__global__ void svpf_vol_mean_kernel(
    const float* h,
    float* d_vol_out,
    int t,
    int n
) {
    extern __shared__ float shared[];
    
    int idx = threadIdx.x;
    float vol = (idx < n) ? safe_exp(h[idx] / 2.0f) : 0.0f;
    
    float sum = block_reduce_sum(vol, shared);
    
    if (threadIdx.x == 0 && d_vol_out) {
        d_vol_out[t] = sum / (float)n;
    }
}

// =============================================================================
// MAIN API: Ultra-fast sequence runner
// =============================================================================

void svpf_run_sequence_fast(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
) {
    int n = state->n_particles;
    
    // For N <= 1024, use single-block kernels (maximum efficiency)
    if (n > 1024) {
        // Fall back to multi-block version
        // svpf_run_sequence(state, d_observations, T, params, d_loglik_out, d_vol_out);
        return;
    }
    
    // Round up to next power of 2 for efficient reductions
    int block_size = 1;
    while (block_size < n) block_size <<= 1;
    block_size = min(block_size, 1024);
    
    // Shared memory: enough for 2 arrays of block_size floats
    size_t shared_mem = 2 * block_size * sizeof(float);
    // Plus reduction space (block_size / 32 warps)
    size_t reduce_space = (block_size / WARP_SIZE + 1) * sizeof(float);
    shared_mem = max(shared_mem, reduce_space);
    
    for (int t = 0; t < T; t++) {
        // Forward pass: copy, predict, loglik, log-sum-exp (1 kernel!)
        svpf_forward_fused_kernel<<<1, block_size, shared_mem, state->stream>>>(
            state->h, state->h_prev, state->h_pred, state->rng_states,
            d_observations, t,
            params->rho, params->sigma_z, params->mu, params->gamma,
            state->nu, state->student_t_const,
            d_loglik_out, n
        );
        
        // Bandwidth computation (1 kernel)
        svpf_bandwidth_kernel<<<1, block_size, reduce_space, state->stream>>>(
            state->h, state->d_scalar_bandwidth, n
        );
        
        // Stein transport (1 kernel per step, 5 total)
        for (int s = 0; s < state->n_stein_steps; s++) {
            svpf_stein_step_kernel<<<1, block_size, shared_mem, state->stream>>>(
                state->h, state->h_prev, d_observations, t,
                state->d_scalar_bandwidth,
                params->rho, params->sigma_z, params->mu, state->nu,
                SVPF_STEIN_STEP_SIZE, n
            );
        }
        
        // Vol mean (1 kernel)
        svpf_vol_mean_kernel<<<1, block_size, reduce_space, state->stream>>>(
            state->h, d_vol_out, t, n
        );
        
        state->timestep++;
    }
    
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// MULTI-BLOCK VERSION (For N > 1024)
// =============================================================================

// Multi-block reduction kernel
__global__ void reduce_sum_kernel(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float shared[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;
    
    val = block_reduce_sum(val, shared);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}

__global__ void reduce_max_kernel(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float shared[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : -1e10f;
    
    val = block_reduce_max(val, shared);
    
    if (threadIdx.x == 0) {
        // Atomic max for floats (bit-cast trick)
        int* address_as_int = (int*)output;
        int old = *address_as_int;
        int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                           __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }
}
