/**
 * @file svpf_kernels_async.cu
 * @brief Fully Asynchronous SVPF - Zero PCIe roundtrips during step execution
 * 
 * Key optimization: All intermediate scalars (max, sum, bandwidth) stay in
 * device memory. Kernels read/write via pointers. Only sync when user needs
 * results on CPU.
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ float clamp_h(float h) {
    return fminf(fmaxf(h, SVPF_H_MIN), SVPF_H_MAX);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 20.0f));
}

__device__ __forceinline__ float log1pf_safe(float x) {
    return log1pf(fmaxf(x, -0.999f));
}

// =============================================================================
// GLUE KERNELS - Scalar computations that stay on GPU
// =============================================================================

// Computes: result = max + log(sum / N + eps)
__global__ void glue_compute_loglik_kernel(
    const float* d_max,
    const float* d_sum,
    float* d_result,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_result = *d_max + logf(*d_sum / (float)n + 1e-10f);
    }
}

// Computes: mean = sum / N
__global__ void glue_compute_mean_kernel(
    const float* d_sum,
    float* d_mean,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_mean = *d_sum / (float)n;
    }
}

// Computes: bandwidth = sqrt(2 * (var_sum / N) / log(N + 1))
__global__ void glue_compute_bandwidth_kernel(
    const float* d_var_sum,
    float* d_bandwidth,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float variance = *d_var_sum / (float)n;
        float bw_sq = 2.0f * variance / logf((float)n + 1.0f);
        float bw = sqrtf(fmaxf(bw_sq, 1e-8f));
        *d_bandwidth = fmaxf(fminf(bw, SVPF_BANDWIDTH_MAX), SVPF_BANDWIDTH_MIN);
    }
}

// =============================================================================
// KERNEL: Initialize RNG States (Philox)
// =============================================================================

__global__ void svpf_init_rng_kernel(
    curandStatePhilox4_32_10_t* states,
    int n,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// =============================================================================
// KERNEL: Initialize Particles
// =============================================================================

__global__ void svpf_init_particles_kernel(
    float* h,
    curandStatePhilox4_32_10_t* rng_states,
    float mu,
    float stationary_std,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float z = curand_normal(&rng_states[idx]);
        h[idx] = clamp_h(mu + stationary_std * z);
    }
}

// =============================================================================
// KERNEL: AR(1) Prediction with Leverage
// =============================================================================

__global__ void svpf_predict_kernel(
    const float* h_prev,
    float* h_pred,
    curandStatePhilox4_32_10_t* rng_states,
    float rho,
    float sigma_z,
    float mu,
    float gamma,
    float y_prev,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float noise = curand_normal(&rng_states[idx]);
        float h_prev_i = h_prev[idx];
        
        float vol_prev = safe_exp(h_prev_i / 2.0f);
        float leverage = gamma * y_prev / (vol_prev + 1e-8f);
        
        float h_new = mu + rho * (h_prev_i - mu) + sigma_z * noise + leverage;
        h_pred[idx] = clamp_h(h_new);
    }
}

// Seeded version for CRN
__global__ void svpf_predict_seeded_kernel(
    const float* h_prev,
    float* h_pred,
    float rho,
    float sigma_z,
    float mu,
    float gamma,
    float y_prev,
    unsigned long long seed,
    int timestep,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandStatePhilox4_32_10_t local_state;
        curand_init(seed, idx, timestep, &local_state);
        float noise = curand_normal(&local_state);
        
        float h_prev_i = h_prev[idx];
        float vol_prev = safe_exp(h_prev_i / 2.0f);
        float leverage = gamma * y_prev / (vol_prev + 1e-8f);
        
        float h_new = mu + rho * (h_prev_i - mu) + sigma_z * noise + leverage;
        h_pred[idx] = clamp_h(h_new);
    }
}

// =============================================================================
// KERNEL: Observation Log-Likelihood (Student-t)
// =============================================================================

__global__ void svpf_obs_loglik_kernel(
    const float* h,
    float* log_weights,
    float y_t,
    float nu,
    float student_t_const,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h_i = h[idx];
        float vol = safe_exp(h_i);
        float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
        
        float log_w = student_t_const - 0.5f * h_i
                    - (nu + 1.0f) / 2.0f * log1pf_safe(scaled_y_sq / nu);
        
        log_weights[idx] = log_w;
    }
}

// =============================================================================
// KERNEL: Exp(log_w - max) - accepts device pointer for max
// =============================================================================

__global__ void svpf_exp_weights_kernel(
    const float* log_weights,
    float* exp_weights,
    const float* d_max,  // Device pointer!
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        exp_weights[idx] = expf(log_weights[idx] - *d_max);
    }
}

// =============================================================================
// KERNEL: Gradient of Log Posterior
// =============================================================================

__global__ void svpf_grad_log_posterior_kernel(
    const float* h,
    const float* h_prev,
    float* grad_log_p,
    float y_t,
    float rho,
    float sigma_z,
    float mu,
    float nu,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h_i = h[idx];
        float h_prev_i = h_prev[idx];
        
        float mu_prior = mu + rho * (h_prev_i - mu);
        float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
        float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
        
        float vol = safe_exp(h_i);
        float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
        float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
        
        float grad = grad_prior + grad_lik;
        grad_log_p[idx] = fminf(fmaxf(grad, -10.0f), 10.0f);
    }
}

// =============================================================================
// KERNEL: Center particles for variance - accepts device pointer for mean
// =============================================================================

__global__ void svpf_center_kernel(
    const float* h,
    float* h_centered_sq,
    const float* d_mean,  // Device pointer!
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = h[idx] - *d_mean;
        h_centered_sq[idx] = diff * diff;
    }
}

// =============================================================================
// KERNEL: RBF Kernel with shared memory - accepts device pointer for bandwidth
// =============================================================================

__global__ void svpf_rbf_kernel_shared(
    const float* h,
    const float* grad_log_p,
    float* kernel_sum,
    float* grad_kernel_sum,
    const float* d_bandwidth,  // Device pointer!
    int n
) {
    extern __shared__ float shared[];
    float* sh_h = shared;
    float* sh_grad = shared + blockDim.x;
    
    // Load bandwidth into shared memory once per block
    __shared__ float bw_sq;
    if (threadIdx.x == 0) {
        float bw = *d_bandwidth;
        bw_sq = bw * bw;
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float h_i = (i < n) ? h[i] : 0.0f;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    // Tiled computation
    for (int tile = 0; tile < n; tile += blockDim.x) {
        int j = tile + threadIdx.x;
        sh_h[threadIdx.x] = (j < n) ? h[j] : 0.0f;
        sh_grad[threadIdx.x] = (j < n) ? grad_log_p[j] : 0.0f;
        __syncthreads();
        
        if (i < n) {
            int tile_end = min((int)blockDim.x, n - tile);
            for (int k = 0; k < tile_end; k++) {
                float diff = h_i - sh_h[k];
                float K = expf(-diff * diff / (2.0f * bw_sq));
                k_sum += K * sh_grad[k];
                gk_sum += -K * diff / bw_sq;
            }
        }
        __syncthreads();
    }
    
    if (i < n) {
        kernel_sum[i] = k_sum;
        grad_kernel_sum[i] = gk_sum;
    }
}

// =============================================================================
// KERNEL: Stein Update
// =============================================================================

__global__ void svpf_stein_update_kernel(
    float* h,
    const float* kernel_sum,
    const float* grad_kernel_sum,
    float step_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float update = step_size * (kernel_sum[idx] + grad_kernel_sum[idx]) / (float)n;
        h[idx] = clamp_h(h[idx] + update);
    }
}

// =============================================================================
// KERNEL: Copy
// =============================================================================

__global__ void svpf_copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// KERNEL: Compute vol = exp(h/2) for statistics
// =============================================================================

__global__ void svpf_compute_vol_kernel(const float* h, float* vol, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vol[idx] = safe_exp(h[idx] / 2.0f);
    }
}

// =============================================================================
// HOST API: Create/Destroy
// =============================================================================

SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream) {
    SVPFState* state = (SVPFState*)malloc(sizeof(SVPFState));
    if (!state) return NULL;
    
    state->n_particles = n_particles;
    state->n_stein_steps = n_stein_steps;
    state->nu = nu;
    state->stream = stream ? stream : 0;
    state->timestep = 0;
    state->y_prev = 0.0f;
    
    state->student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                           - lgammaf(nu / 2.0f) 
                           - 0.5f * logf((float)M_PI * nu);
    
    int n = n_particles;
    
    // Particle arrays
    cudaMalloc(&state->h, n * sizeof(float));
    cudaMalloc(&state->h_prev, n * sizeof(float));
    cudaMalloc(&state->h_pred, n * sizeof(float));
    cudaMalloc(&state->grad_log_p, n * sizeof(float));
    cudaMalloc(&state->kernel_sum, n * sizeof(float));
    cudaMalloc(&state->grad_kernel_sum, n * sizeof(float));
    cudaMalloc(&state->log_weights, n * sizeof(float));
    cudaMalloc(&state->d_h_centered, n * sizeof(float));
    cudaMalloc(&state->rng_states, n * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&state->d_reduce_buf, n * sizeof(float));
    cudaMalloc(&state->d_temp, n * sizeof(float));
    
    // Device scalars (the key to async execution)
    cudaMalloc(&state->d_scalar_max, sizeof(float));
    cudaMalloc(&state->d_scalar_sum, sizeof(float));
    cudaMalloc(&state->d_scalar_mean, sizeof(float));
    cudaMalloc(&state->d_scalar_bandwidth, sizeof(float));
    cudaMalloc(&state->d_result_loglik, sizeof(float));
    cudaMalloc(&state->d_result_vol_mean, sizeof(float));
    cudaMalloc(&state->d_result_h_mean, sizeof(float));
    
    // CUB temp storage
    state->cub_temp_bytes = 0;
    cub::DeviceReduce::Sum(NULL, state->cub_temp_bytes, state->h, state->d_scalar_sum, n);
    state->cub_temp_bytes += 1024;
    cudaMalloc(&state->d_cub_temp, state->cub_temp_bytes);
    
    return state;
}

void svpf_destroy(SVPFState* state) {
    if (!state) return;
    
    cudaFree(state->h);
    cudaFree(state->h_prev);
    cudaFree(state->h_pred);
    cudaFree(state->grad_log_p);
    cudaFree(state->kernel_sum);
    cudaFree(state->grad_kernel_sum);
    cudaFree(state->log_weights);
    cudaFree(state->d_h_centered);
    cudaFree(state->rng_states);
    cudaFree(state->d_reduce_buf);
    cudaFree(state->d_temp);
    cudaFree(state->d_cub_temp);
    
    cudaFree(state->d_scalar_max);
    cudaFree(state->d_scalar_sum);
    cudaFree(state->d_scalar_mean);
    cudaFree(state->d_scalar_bandwidth);
    cudaFree(state->d_result_loglik);
    cudaFree(state->d_result_vol_mean);
    cudaFree(state->d_result_h_mean);
    
    free(state);
}

// =============================================================================
// HOST API: Initialize
// =============================================================================

void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    state->timestep = 0;
    state->y_prev = 0.0f;
    
    svpf_init_rng_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->rng_states, n, seed
    );
    
    float rho = params->rho;
    float sigma_z = params->sigma_z;
    float stationary_var = (sigma_z * sigma_z) / (1.0f - rho * rho + 1e-6f);
    float stationary_std = sqrtf(stationary_var);
    
    svpf_init_particles_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->rng_states, params->mu, stationary_std, n
    );
    
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// HOST API: Step (Fully Asynchronous - NO sync until svpf_get_result)
// =============================================================================

void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    size_t shared_mem = 2 * SVPF_BLOCK_SIZE * sizeof(float);
    
    // 1. Copy h -> h_prev
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    // 2. Predict with leverage
    svpf_predict_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_prev, state->h_pred, state->rng_states,
        params->rho, params->sigma_z, params->mu,
        params->gamma, state->y_prev, n
    );
    
    // 3. Observation likelihood
    svpf_obs_loglik_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->log_weights, y_t,
        state->nu, state->student_t_const, n
    );
    
    // 4. Log-sum-exp (FULLY ON GPU - no D2H transfers)
    // 4a. Max -> d_scalar_max
    cub::DeviceReduce::Max(state->d_cub_temp, state->cub_temp_bytes,
        state->log_weights, state->d_scalar_max, n, state->stream);
    
    // 4b. Exp(w - max) using device pointer
    svpf_exp_weights_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->log_weights, state->d_temp, state->d_scalar_max, n
    );
    
    // 4c. Sum -> d_scalar_sum
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->d_temp, state->d_scalar_sum, n, state->stream);
    
    // 4d. Compute final log-lik (on GPU)
    glue_compute_loglik_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_max, state->d_scalar_sum, state->d_result_loglik, n
    );
    
    // 5. Copy predicted -> current
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->h, n
    );
    
    // 6. Stein transport (bandwidth computed ONCE, not per iteration)
    // 6a. Compute bandwidth
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->h, state->d_scalar_sum, n, state->stream);
    glue_compute_mean_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_scalar_mean, n
    );
    svpf_center_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->d_h_centered, state->d_scalar_mean, n
    );
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->d_h_centered, state->d_scalar_sum, n, state->stream);
    glue_compute_bandwidth_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_scalar_bandwidth, n
    );
    
    // 6b. Stein iterations (use pre-computed bandwidth)
    for (int s = 0; s < state->n_stein_steps; s++) {
        svpf_grad_log_posterior_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->h_prev, state->grad_log_p,
            y_t, params->rho, params->sigma_z, params->mu, state->nu, n
        );
        
        svpf_rbf_kernel_shared<<<grid, SVPF_BLOCK_SIZE, shared_mem, state->stream>>>(
            state->h, state->grad_log_p,
            state->kernel_sum, state->grad_kernel_sum,
            state->d_scalar_bandwidth, n
        );
        
        svpf_stein_update_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->kernel_sum, state->grad_kernel_sum,
            SVPF_STEIN_STEP_SIZE, n
        );
    }
    
    // 7. Compute output statistics (on GPU)
    svpf_compute_vol_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->d_temp, n
    );
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->d_temp, state->d_scalar_sum, n, state->stream);
    glue_compute_mean_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_result_vol_mean, n
    );
    
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->h, state->d_scalar_sum, n, state->stream);
    glue_compute_mean_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_result_h_mean, n
    );
    
    // 8. Update state
    state->y_prev = y_t;
    state->timestep++;
    
    // =========================================================================
    // SYNC AND COPY RESULTS BACK TO HOST
    // For maximum throughput, you could skip this and batch-retrieve later.
    // But for API compatibility, we sync here.
    // =========================================================================
    cudaStreamSynchronize(state->stream);
    
    cudaMemcpy(&result->log_lik_increment, state->d_result_loglik, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result->vol_mean, state->d_result_vol_mean, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result->h_mean, state->d_result_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
    result->vol_std = 0.0f;
}

// =============================================================================
// HOST API: Step Seeded (for CRN/CPMMH)
// =============================================================================

void svpf_step_seeded(SVPFState* state, float y_t, const SVPFParams* params,
                      unsigned long long rng_seed, SVPFResult* result) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    size_t shared_mem = 2 * SVPF_BLOCK_SIZE * sizeof(float);
    
    // 1. Copy h -> h_prev
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    // 2. Predict with SEEDED RNG
    svpf_predict_seeded_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_prev, state->h_pred,
        params->rho, params->sigma_z, params->mu,
        params->gamma, state->y_prev,
        rng_seed, state->timestep, n
    );
    
    // 3-8: Same as svpf_step (copy-paste for now, could factor out)
    svpf_obs_loglik_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->log_weights, y_t,
        state->nu, state->student_t_const, n
    );
    
    cub::DeviceReduce::Max(state->d_cub_temp, state->cub_temp_bytes,
        state->log_weights, state->d_scalar_max, n, state->stream);
    svpf_exp_weights_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->log_weights, state->d_temp, state->d_scalar_max, n
    );
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->d_temp, state->d_scalar_sum, n, state->stream);
    glue_compute_loglik_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_max, state->d_scalar_sum, state->d_result_loglik, n
    );
    
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->h, n
    );
    
    // Bandwidth
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->h, state->d_scalar_sum, n, state->stream);
    glue_compute_mean_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_scalar_mean, n
    );
    svpf_center_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->d_h_centered, state->d_scalar_mean, n
    );
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->d_h_centered, state->d_scalar_sum, n, state->stream);
    glue_compute_bandwidth_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_scalar_bandwidth, n
    );
    
    for (int s = 0; s < state->n_stein_steps; s++) {
        svpf_grad_log_posterior_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->h_prev, state->grad_log_p,
            y_t, params->rho, params->sigma_z, params->mu, state->nu, n
        );
        svpf_rbf_kernel_shared<<<grid, SVPF_BLOCK_SIZE, shared_mem, state->stream>>>(
            state->h, state->grad_log_p,
            state->kernel_sum, state->grad_kernel_sum,
            state->d_scalar_bandwidth, n
        );
        svpf_stein_update_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->kernel_sum, state->grad_kernel_sum,
            SVPF_STEIN_STEP_SIZE, n
        );
    }
    
    svpf_compute_vol_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->d_temp, n
    );
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes,
        state->d_temp, state->d_scalar_sum, n, state->stream);
    glue_compute_mean_kernel<<<1, 1, 0, state->stream>>>(
        state->d_scalar_sum, state->d_result_vol_mean, n
    );
    
    state->y_prev = y_t;
    state->timestep++;
    
    cudaStreamSynchronize(state->stream);
    cudaMemcpy(&result->log_lik_increment, state->d_result_loglik, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result->vol_mean, state->d_result_vol_mean, sizeof(float), cudaMemcpyDeviceToHost);
    result->h_mean = 0.0f;
    result->vol_std = 0.0f;
}

// =============================================================================
// HOST API: Get Particles (diagnostic)
// =============================================================================

void svpf_get_particles(const SVPFState* state, float* h_out) {
    cudaMemcpy(h_out, state->h, state->n_particles * sizeof(float), cudaMemcpyDeviceToHost);
}

void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std) {
    int n = state->n_particles;
    float* h_host = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_host, state->h, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += h_host[i];
    *h_mean = sum / (float)n;
    
    float sq_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = h_host[i] - *h_mean;
        sq_sum += diff * diff;
    }
    *h_std = sqrtf(sq_sum / (float)n);
    
    free(h_host);
}
