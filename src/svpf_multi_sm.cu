/**
 * @file svpf_multi_sm.cu
 * @brief Multi-SM SVPF - Parallelizes O(N²) Stein kernel across ALL SMs
 * 
 * Key insight: The Stein RBF computation is O(N²) and was running on 1 SM.
 * This version uses N blocks (one per particle) to utilize all SMs.
 * 
 * Expected: 10-50x speedup on RTX 5080 (84 SMs)
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARP_SIZE 32
#define TILE_SIZE 128  // Particles per tile for RBF computation

// =============================================================================
// WARP-LEVEL REDUCTION PRIMITIVES
// =============================================================================

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

__device__ float block_reduce_sum_fast(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__device__ float block_reduce_max_fast(float val) {
    __shared__ float shared[32];
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
// MULTI-SM STEIN KERNEL: Each block handles ONE particle's full RBF sum
// This is the key optimization - N blocks = utilize all SMs
// =============================================================================

__global__ void svpf_stein_multi_sm_kernel(
    float* h,                    // [N] particles (read + write)
    const float* h_prev,         // [N] previous particles
    const float* d_y,            // [T] observations
    int t,                       // current timestep
    float bandwidth,             // Pre-computed bandwidth (host value)
    float rho, float sigma_z, float mu, float nu,
    float step_size,
    float* grad_out,             // [N] pre-computed gradients (input)
    int n
) {
    // Each block handles ONE particle (blockIdx.x = particle index i)
    // Threads within block parallelize the j-loop
    int i = blockIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float bw_sq = bandwidth * bandwidth;
    
    // Parallel reduction over j using all threads in block
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    // Each thread handles multiple j values
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float h_j = h[j];
        float grad_j = grad_out[j];
        
        float diff = h_i - h_j;
        float K = expf(-diff * diff / (2.0f * bw_sq));
        
        k_sum += K * grad_j;
        gk_sum += -K * diff / bw_sq;
    }
    
    // Reduce across threads in block
    k_sum = block_reduce_sum_fast(k_sum);
    gk_sum = block_reduce_sum_fast(gk_sum);
    
    // Thread 0 applies update
    if (threadIdx.x == 0) {
        float update = step_size * (k_sum + gk_sum) / (float)n;
        h[i] = clamp_h(h_i + update);
    }
}

// Device-pointer version - reads bandwidth from GPU memory (no sync needed)
__global__ void svpf_stein_multi_sm_device_kernel(
    float* h,
    const float* h_prev,
    const float* d_y,
    int t,
    const float* d_bandwidth,    // Device pointer!
    float rho, float sigma_z, float mu, float nu,
    float step_size,
    float* grad_out,
    int n
) {
    int i = blockIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float bandwidth = *d_bandwidth;  // Read from device memory
    float bw_sq = bandwidth * bandwidth;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float h_j = h[j];
        float grad_j = grad_out[j];
        
        float diff = h_i - h_j;
        float K = expf(-diff * diff / (2.0f * bw_sq));
        
        k_sum += K * grad_j;
        gk_sum += -K * diff / bw_sq;
    }
    
    k_sum = block_reduce_sum_fast(k_sum);
    gk_sum = block_reduce_sum_fast(gk_sum);
    
    if (threadIdx.x == 0) {
        float update = step_size * (k_sum + gk_sum) / (float)n;
        h[i] = clamp_h(h_i + update);
    }
}

// =============================================================================
// GRADIENT KERNEL: Compute gradients for all particles (parallel, no deps)
// =============================================================================

__global__ void svpf_gradient_kernel(
    const float* h,
    const float* h_prev,
    float* grad_out,
    float y_t,
    float rho, float sigma_z, float mu, float nu,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float h_i = h[idx];
    float h_prev_i = h_prev[idx];
    
    // Prior gradient
    float mu_prior = mu + rho * (h_prev_i - mu);
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
    
    // Likelihood gradient
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
    
    grad_out[idx] = fminf(fmaxf(grad_prior + grad_lik, -10.0f), 10.0f);
}

// Batch version - reads y[t] from device array
__global__ void svpf_gradient_batch_kernel(
    const float* h,
    const float* h_prev,
    float* grad_out,
    const float* d_y,
    int t,
    float rho, float sigma_z, float mu, float nu,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float y_t = d_y[t];
    float h_i = h[idx];
    float h_prev_i = h_prev[idx];
    
    float mu_prior = mu + rho * (h_prev_i - mu);
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
    
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
    
    grad_out[idx] = fminf(fmaxf(grad_prior + grad_lik, -10.0f), 10.0f);
}

// =============================================================================
// FUSED COPY + PREDICT KERNEL
// =============================================================================

__global__ void svpf_copy_predict_kernel(
    float* h,
    float* h_prev,
    float* h_pred,
    curandStatePhilox4_32_10_t* rng_states,
    const float* d_y,
    int t,
    float rho, float sigma_z, float mu, float gamma,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float h_i = h[idx];
    h_prev[idx] = h_i;
    
    float noise = curand_normal(&rng_states[idx]);
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    h_pred[idx] = clamp_h(mu + rho * (h_i - mu) + sigma_z * noise + leverage);
}

// =============================================================================
// LOG-LIKELIHOOD KERNELS
// =============================================================================

__global__ void svpf_loglik_kernel(
    const float* h_pred,
    float* log_weights,
    float y_t,
    float nu,
    float student_t_const,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float h_i = h_pred[idx];
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    
    log_weights[idx] = student_t_const - 0.5f * h_i
                     - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
}

// Batch version - reads y[t] from device array
__global__ void svpf_loglik_batch_kernel(
    const float* h_pred,
    float* log_weights,
    const float* d_y,
    int t,
    float nu,
    float student_t_const,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float y_t = d_y[t];
    float h_i = h_pred[idx];
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    
    log_weights[idx] = student_t_const - 0.5f * h_i
                     - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
}

// Max reduction kernel
__global__ void svpf_reduce_max_kernel(
    const float* input,
    float* output,
    int n
) {
    float val = -1e10f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        val = fmaxf(val, input[i]);
    }
    
    val = block_reduce_max_fast(val);
    
    if (threadIdx.x == 0) {
        atomicMax((int*)output, __float_as_int(val));
    }
}

// Exp and sum kernel
__global__ void svpf_exp_sum_kernel(
    const float* log_weights,
    float* exp_weights,
    const float* d_max,
    float* d_sum,
    int n
) {
    float max_val = *d_max;
    float local_sum = 0.0f;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float exp_w = expf(log_weights[i] - max_val);
        exp_weights[i] = exp_w;
        local_sum += exp_w;
    }
    
    local_sum = block_reduce_sum_fast(local_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(d_sum, local_sum);
    }
}

// Bandwidth kernel
__global__ void svpf_bandwidth_multi_kernel(
    const float* h,
    float* d_mean,
    float* d_var,
    float* d_bandwidth,
    int n
) {
    // First pass: compute sum for mean
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum += h[i];
    }
    local_sum = block_reduce_sum_fast(local_sum);
    if (threadIdx.x == 0) atomicAdd(d_mean, local_sum);
    
    __threadfence();
    __syncthreads();
    
    // Compute mean (one thread)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_mean = *d_mean / (float)n;
    }
    __threadfence();
    __syncthreads();
    
    float mean = *d_mean;
    
    // Second pass: compute variance
    float local_var = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float diff = h[i] - mean;
        local_var += diff * diff;
    }
    local_var = block_reduce_sum_fast(local_var);
    if (threadIdx.x == 0) atomicAdd(d_var, local_var);
    
    __threadfence();
    __syncthreads();
    
    // Compute bandwidth (one thread)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float variance = *d_var / (float)n;
        float bw_sq = 2.0f * variance / logf((float)n + 1.0f);
        float bw = sqrtf(fmaxf(bw_sq, 1e-8f));
        *d_bandwidth = fmaxf(fminf(bw, 2.0f), 0.01f);
    }
}

// Copy h_pred -> h (renamed to avoid conflict with svpf_kernels.cu)
__global__ void svpf_copy_msm_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// Vol mean kernel (renamed)
__global__ void svpf_vol_mean_msm_kernel(
    const float* h,
    float* d_vol_sum,
    int n
) {
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum += safe_exp(h[i] / 2.0f);
    }
    
    local_sum = block_reduce_sum_fast(local_sum);
    if (threadIdx.x == 0) atomicAdd(d_vol_sum, local_sum);
}

// Record results
__global__ void svpf_record_kernel(
    const float* d_max,
    const float* d_sum,
    const float* d_vol_sum,
    float* d_loglik_out,
    float* d_vol_out,
    int t,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (d_loglik_out) {
            d_loglik_out[t] = *d_max + logf(*d_sum / (float)n + 1e-10f);
        }
        if (d_vol_out) {
            d_vol_out[t] = *d_vol_sum / (float)n;
        }
    }
}

// =============================================================================
// MAIN API: Multi-SM sequence runner (ZERO mid-sequence syncs)
// =============================================================================

void svpf_run_sequence_multi_sm(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
) {
    int n = state->n_particles;
    
    // Grid/block configuration
    int threads_per_block = 256;
    int blocks_for_n = (n + threads_per_block - 1) / threads_per_block;
    
    // For Stein kernel: N blocks, each with enough threads to cover N
    int stein_threads = min(256, n);
    
    // Temp buffers for reductions (allocated once, reused)
    float *d_max, *d_sum, *d_mean, *d_var, *d_vol_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_var, sizeof(float));
    cudaMalloc(&d_vol_sum, sizeof(float));
    
    // Gradient buffer
    float* d_grad;
    cudaMalloc(&d_grad, n * sizeof(float));
    
    for (int t = 0; t < T; t++) {
        // NO SYNC - all kernels read from device memory
        
        // 1. Copy + Predict (reads y[t-1] from d_observations directly)
        svpf_copy_predict_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
            state->h, state->h_prev, state->h_pred, state->rng_states,
            d_observations, t,
            params->rho, params->sigma_z, params->mu, params->gamma, n
        );
        
        // 2. Log-likelihood (reads y[t] from d_observations directly)
        svpf_loglik_batch_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
            state->h_pred, state->log_weights, d_observations, t,
            state->nu, state->student_t_const, n
        );
        
        // 3. Log-sum-exp
        cudaMemsetAsync(d_max, 0xFF, sizeof(float), state->stream);  // -inf
        cudaMemsetAsync(d_sum, 0, sizeof(float), state->stream);
        
        svpf_reduce_max_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
            state->log_weights, d_max, n
        );
        svpf_exp_sum_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
            state->log_weights, state->d_temp, d_max, d_sum, n
        );
        
        // 4. Copy h_pred -> h
        svpf_copy_msm_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
            state->h_pred, state->h, n
        );
        
        // 5. Bandwidth (computed on GPU, stored in d_scalar_bandwidth)
        cudaMemsetAsync(d_mean, 0, sizeof(float), state->stream);
        cudaMemsetAsync(d_var, 0, sizeof(float), state->stream);
        svpf_bandwidth_multi_kernel<<<1, threads_per_block, 0, state->stream>>>(
            state->h, d_mean, d_var, state->d_scalar_bandwidth, n
        );
        
        // 6. Stein iterations (reads bandwidth from d_scalar_bandwidth)
        for (int s = 0; s < state->n_stein_steps; s++) {
            // Compute gradients (reads y[t] from d_observations)
            svpf_gradient_batch_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
                state->h, state->h_prev, d_grad, d_observations, t,
                params->rho, params->sigma_z, params->mu, state->nu, n
            );
            
            // RBF + Update (reads bandwidth from d_scalar_bandwidth)
            svpf_stein_multi_sm_device_kernel<<<n, stein_threads, 0, state->stream>>>(
                state->h, state->h_prev, d_observations, t,
                state->d_scalar_bandwidth,  // Device pointer!
                params->rho, params->sigma_z, params->mu, state->nu,
                SVPF_STEIN_STEP_SIZE, d_grad, n
            );
        }
        
        // 7. Vol mean
        cudaMemsetAsync(d_vol_sum, 0, sizeof(float), state->stream);
        svpf_vol_mean_msm_kernel<<<blocks_for_n, threads_per_block, 0, state->stream>>>(
            state->h, d_vol_sum, n
        );
        
        // 8. Record results
        svpf_record_kernel<<<1, 1, 0, state->stream>>>(
            d_max, d_sum, d_vol_sum, d_loglik_out, d_vol_out, t, n
        );
        
        state->timestep++;
    }
    
    // SINGLE sync at end
    cudaStreamSynchronize(state->stream);
    
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_var);
    cudaFree(d_vol_sum);
    cudaFree(d_grad);
}
