/**
 * @file svpf_kernels.cu
 * @brief CUDA kernels for Stein Variational Particle Filter
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ float clamp_h(float h) {
    return fminf(fmaxf(h, SVPF_H_MIN), SVPF_H_MAX);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 20.0f));  // Prevent overflow
}

__device__ __forceinline__ float log1pf_safe(float x) {
    return log1pf(fmaxf(x, -0.999f));  // Prevent log(0)
}

// =============================================================================
// KERNEL: Initialize RNG States
// =============================================================================

__global__ void svpf_init_rng_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// =============================================================================
// KERNEL: Initialize Particles from Stationary Distribution
// =============================================================================

__global__ void svpf_init_particles_kernel(
    float* h,
    curandState* rng_states,
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
// KERNEL: AR(1) Prediction Step
// =============================================================================

__global__ void svpf_predict_kernel(
    const float* h_prev,
    float* h_pred,
    curandState* rng_states,
    float rho,
    float sigma_z,
    float mu,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float noise = curand_normal(&rng_states[idx]);
        float h_new = mu + rho * (h_prev[idx] - mu) + sigma_z * noise;
        h_pred[idx] = clamp_h(h_new);
    }
}

// Seeded version for CRN (Common Random Numbers)
__global__ void svpf_predict_seeded_kernel(
    const float* h_prev,
    float* h_pred,
    float rho,
    float sigma_z,
    float mu,
    unsigned long long seed,
    int t,  // timestep (for unique sequence)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Deterministic noise from seed + timestep + particle index
        curandState local_state;
        curand_init(seed + t * 1000000ULL, idx, 0, &local_state);
        float noise = curand_normal(&local_state);
        
        float h_new = mu + rho * (h_prev[idx] - mu) + sigma_z * noise;
        h_pred[idx] = clamp_h(h_new);
    }
}

// =============================================================================
// KERNEL: Compute Observation Log-Likelihood (Student-t)
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
        
        // Student-t log-likelihood
        // const - 0.5*h - (nu+1)/2 * log(1 + y²/(nu*vol))
        float log_w = student_t_const 
                    - 0.5f * h_i 
                    - (nu + 1.0f) / 2.0f * log1pf_safe(scaled_y_sq / nu);
        
        log_weights[idx] = log_w;
    }
}

// =============================================================================
// KERNEL: Compute Gradient of Log Posterior
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
        
        // Prior gradient: d/dh log N(h | mu_prior, sigma_z²)
        float mu_prior = mu + rho * (h_prev_i - mu);
        float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
        float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
        
        // Likelihood gradient: d/dh log p(y | h) for Student-t
        float vol = safe_exp(h_i);
        float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
        float grad_lik = 0.5f * ((nu + 1.0f) * scaled_y_sq / (nu + scaled_y_sq + 1e-8f) - 1.0f);
        
        // Total gradient (clamped for stability)
        float grad = grad_prior + grad_lik;
        grad_log_p[idx] = fminf(fmaxf(grad, -10.0f), 10.0f);
    }
}

// =============================================================================
// KERNEL: Compute Pairwise Distances for Bandwidth
// =============================================================================

__global__ void svpf_pairwise_dist_kernel(
    const float* h,
    float* distances,  // [N*N] output
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if (idx < total) {
        int i = idx / n;
        int j = idx % n;
        distances[idx] = fabsf(h[i] - h[j]);
    }
}

// =============================================================================
// KERNEL: Compute RBF Kernel and Its Gradient
// =============================================================================

__global__ void svpf_rbf_kernel_kernel(
    const float* h,
    const float* grad_log_p,
    float* kernel_sum,       // [N] sum_j K(h_i, h_j) * grad_log_p[j]
    float* grad_kernel_sum,  // [N] sum_j dK/dh_i
    float bandwidth,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    float bandwidth_sq = bandwidth * bandwidth;
    
    // Sum over all j
    for (int j = 0; j < n; j++) {
        float h_j = h[j];
        float diff = h_i - h_j;
        
        // K(h_i, h_j) = exp(-diff² / (2*bandwidth²))
        float K_ij = expf(-diff * diff / (2.0f * bandwidth_sq));
        
        // Attraction term: K * grad_log_p[j]
        k_sum += K_ij * grad_log_p[j];
        
        // Repulsion term: dK/dh_i = -K * diff / bandwidth²
        gk_sum += -K_ij * diff / bandwidth_sq;
    }
    
    kernel_sum[i] = k_sum;
    grad_kernel_sum[i] = gk_sum;
}

// Shared memory optimized version for smaller N
__global__ void svpf_rbf_kernel_shared_kernel(
    const float* h,
    const float* grad_log_p,
    float* kernel_sum,
    float* grad_kernel_sum,
    float bandwidth,
    int n
) {
    extern __shared__ float shared[];
    float* sh_h = shared;
    float* sh_grad = shared + blockDim.x;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float h_i = (i < n) ? h[i] : 0.0f;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    float bandwidth_sq = bandwidth * bandwidth;
    
    // Process in tiles
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int j = tile_start + threadIdx.x;
        
        // Load tile into shared memory
        sh_h[threadIdx.x] = (j < n) ? h[j] : 0.0f;
        sh_grad[threadIdx.x] = (j < n) ? grad_log_p[j] : 0.0f;
        __syncthreads();
        
        // Compute contributions from this tile
        if (i < n) {
            int tile_size = min(blockDim.x, n - tile_start);
            for (int k = 0; k < tile_size; k++) {
                float diff = h_i - sh_h[k];
                float K = expf(-diff * diff / (2.0f * bandwidth_sq));
                k_sum += K * sh_grad[k];
                gk_sum += -K * diff / bandwidth_sq;
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
        float n_inv = 1.0f / (float)n;
        
        // Stein update: h += step_size * (attraction + repulsion) / N
        float attraction = kernel_sum[idx] * n_inv;
        float repulsion = grad_kernel_sum[idx] * n_inv;
        
        float update = step_size * (attraction + repulsion);
        update = fminf(fmaxf(update, -1.0f), 1.0f);  // Clamp update
        
        h[idx] = clamp_h(h[idx] + update);
    }
}

// =============================================================================
// KERNEL: Copy h to h_prev
// =============================================================================

__global__ void svpf_copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// KERNEL: Compute Statistics (mean, variance of exp(h/2))
// =============================================================================

__global__ void svpf_compute_vol_stats_kernel(
    const float* h,
    float* vol_out,  // [N] exp(h/2) for reduction
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vol_out[idx] = safe_exp(h[idx] / 2.0f);
    }
}

// =============================================================================
// HOST: Parallel Reduction Helpers
// =============================================================================

// Sum reduction using CUB
static float reduce_sum(float* d_data, float* d_temp, int n, cudaStream_t stream) {
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(NULL, temp_bytes, d_data, d_temp, n, stream);
    
    void* d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_bytes);
    
    cub::DeviceReduce::Sum(d_temp_storage, temp_bytes, d_data, d_temp, n, stream);
    
    float result;
    cudaMemcpyAsync(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaFree(d_temp_storage);
    return result;
}

// Max reduction using CUB
static float reduce_max(float* d_data, float* d_temp, int n, cudaStream_t stream) {
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(NULL, temp_bytes, d_data, d_temp, n, stream);
    
    void* d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_bytes);
    
    cub::DeviceReduce::Max(d_temp_storage, temp_bytes, d_data, d_temp, n, stream);
    
    float result;
    cudaMemcpyAsync(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaFree(d_temp_storage);
    return result;
}

// Median via sorting (approximate - use 50th percentile)
static float reduce_median_nonzero(float* d_data, float* d_temp, int n, cudaStream_t stream) {
    // Sort the data
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(NULL, temp_bytes, d_data, d_temp, n, 0, 32, stream);
    
    void* d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_bytes);
    
    float* d_sorted;
    cudaMalloc(&d_sorted, n * sizeof(float));
    
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_bytes, d_data, d_sorted, n, 0, 32, stream);
    
    // Get approximate median (75th percentile to skip zeros on diagonal)
    int median_idx = (int)(0.75f * n);
    float result;
    cudaMemcpyAsync(&result, d_sorted + median_idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaFree(d_temp_storage);
    cudaFree(d_sorted);
    return result;
}

// =============================================================================
// HOST: Compute Bandwidth via Median Heuristic
// =============================================================================

static float compute_bandwidth(SVPFState* state) {
    int n = state->n_particles;
    int grid = (n * n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    // Allocate distance matrix
    float* d_distances;
    cudaMalloc(&d_distances, n * n * sizeof(float));
    
    // Compute pairwise distances
    svpf_pairwise_dist_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, d_distances, n
    );
    
    // Get median of non-zero distances
    float median = reduce_median_nonzero(d_distances, state->d_reduce_buf, n * n, state->stream);
    
    cudaFree(d_distances);
    
    // Apply bandwidth heuristic
    float bandwidth = median / logf((float)n + 1.0f);
    bandwidth = fmaxf(fminf(bandwidth, SVPF_BANDWIDTH_MAX), SVPF_BANDWIDTH_MIN);
    
    return bandwidth;
}

// =============================================================================
// HOST API IMPLEMENTATION
// =============================================================================

SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream) {
    SVPFState* state = (SVPFState*)malloc(sizeof(SVPFState));
    if (!state) return NULL;
    
    state->n_particles = n_particles;
    state->n_stein_steps = n_stein_steps;
    state->nu = nu;
    state->stream = stream ? stream : 0;
    
    // Pre-compute Student-t constant: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*nu)
    state->student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                           - lgammaf(nu / 2.0f) 
                           - 0.5f * logf(M_PI * nu);
    
    // Allocate device arrays
    int n = n_particles;
    cudaMalloc(&state->h, n * sizeof(float));
    cudaMalloc(&state->h_prev, n * sizeof(float));
    cudaMalloc(&state->h_pred, n * sizeof(float));
    cudaMalloc(&state->grad_log_p, n * sizeof(float));
    cudaMalloc(&state->kernel_sum, n * sizeof(float));
    cudaMalloc(&state->grad_kernel_sum, n * sizeof(float));
    cudaMalloc(&state->log_weights, n * sizeof(float));
    cudaMalloc(&state->rng_states, n * sizeof(curandState));
    cudaMalloc(&state->d_reduce_buf, n * sizeof(float));
    cudaMalloc(&state->d_temp, n * sizeof(float));  // Temp for reductions
    
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
    cudaFree(state->rng_states);
    cudaFree(state->d_reduce_buf);
    cudaFree(state->d_temp);
    
    free(state);
}

void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    // Initialize RNG
    svpf_init_rng_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->rng_states, n, seed
    );
    
    // Compute stationary std
    float rho = params->rho;
    float sigma_z = params->sigma_z;
    float stationary_var = (sigma_z * sigma_z) / (1.0f - rho * rho + 1e-6f);
    float stationary_std = sqrtf(stationary_var);
    
    // Initialize particles
    svpf_init_particles_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->rng_states, params->mu, stationary_std, n
    );
    
    // Copy to h_prev
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    cudaStreamSynchronize(state->stream);
}

void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    // 1. Save current particles as h_prev
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    // 2. AR(1) Prediction
    svpf_predict_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_prev, state->h_pred, state->rng_states,
        params->rho, params->sigma_z, params->mu, n
    );
    
    // 3. Compute observation log-likelihood (BEFORE Stein update)
    svpf_obs_loglik_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->log_weights, y_t,
        state->nu, state->student_t_const, n
    );
    
    // 4. Compute log-likelihood increment: log(mean(exp(log_w)))
    //    = max(log_w) + log(mean(exp(log_w - max)))
    float max_log_w = reduce_max(state->log_weights, state->d_reduce_buf, n, state->stream);
    
    // Subtract max and exp
    // (Inline kernel for brevity - in production, use separate kernel)
    float* h_log_weights = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_log_weights, state->log_weights, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(h_log_weights[i] - max_log_w);
    }
    float log_lik_incr = max_log_w + logf(sum_exp / (float)n + 1e-10f);
    
    free(h_log_weights);
    
    // 5. Copy predicted to current particles
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->h, n
    );
    
    // 6. Stein transport iterations
    for (int s = 0; s < state->n_stein_steps; s++) {
        // Compute gradient of log posterior
        svpf_grad_log_posterior_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->h_prev, state->grad_log_p,
            y_t, params->rho, params->sigma_z, params->mu, state->nu, n
        );
        
        // Compute bandwidth via median heuristic
        float bandwidth = compute_bandwidth(state);
        
        // Compute kernel sums (using shared memory version if small enough)
        size_t shared_mem = 2 * SVPF_BLOCK_SIZE * sizeof(float);
        svpf_rbf_kernel_shared_kernel<<<grid, SVPF_BLOCK_SIZE, shared_mem, state->stream>>>(
            state->h, state->grad_log_p,
            state->kernel_sum, state->grad_kernel_sum,
            bandwidth, n
        );
        
        // Apply Stein update
        svpf_stein_update_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->kernel_sum, state->grad_kernel_sum,
            SVPF_STEIN_STEP_SIZE, n
        );
    }
    
    // 7. Compute output statistics
    svpf_compute_vol_stats_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->d_reduce_buf, n
    );
    
    float vol_sum = reduce_sum(state->d_reduce_buf, state->d_temp, n, state->stream);
    float vol_mean = vol_sum / (float)n;
    
    // Also get h mean
    float h_sum = reduce_sum(state->h, state->d_temp, n, state->stream);
    float h_mean = h_sum / (float)n;
    
    // Fill result
    result->log_lik_increment = log_lik_incr;
    result->vol_mean = vol_mean;
    result->vol_std = 0.0f;  // TODO: compute if needed
    result->h_mean = h_mean;
    
    cudaStreamSynchronize(state->stream);
}

void svpf_step_seeded(SVPFState* state, float y_t, const SVPFParams* params,
                      unsigned long long rng_seed, SVPFResult* result) {
    // Same as svpf_step but with seeded prediction kernel
    // This enables Common Random Numbers for finite differences / CPMMH
    
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    static int timestep = 0;  // Track timestep for seed uniqueness
    
    // 1. Save current particles as h_prev
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    // 2. AR(1) Prediction with SEEDED noise (CRN)
    svpf_predict_seeded_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_prev, state->h_pred,
        params->rho, params->sigma_z, params->mu,
        rng_seed, timestep++, n
    );
    
    // Rest is same as svpf_step...
    // (In production, factor out common code)
    
    // 3. Compute observation log-likelihood
    svpf_obs_loglik_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->log_weights, y_t,
        state->nu, state->student_t_const, n
    );
    
    // 4. Compute log-likelihood increment
    float max_log_w = reduce_max(state->log_weights, state->d_reduce_buf, n, state->stream);
    
    float* h_log_weights = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_log_weights, state->log_weights, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(h_log_weights[i] - max_log_w);
    }
    float log_lik_incr = max_log_w + logf(sum_exp / (float)n + 1e-10f);
    
    free(h_log_weights);
    
    // 5. Copy predicted to current
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h_pred, state->h, n
    );
    
    // 6. Stein transport
    for (int s = 0; s < state->n_stein_steps; s++) {
        svpf_grad_log_posterior_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->h_prev, state->grad_log_p,
            y_t, params->rho, params->sigma_z, params->mu, state->nu, n
        );
        
        float bandwidth = compute_bandwidth(state);
        
        size_t shared_mem = 2 * SVPF_BLOCK_SIZE * sizeof(float);
        svpf_rbf_kernel_shared_kernel<<<grid, SVPF_BLOCK_SIZE, shared_mem, state->stream>>>(
            state->h, state->grad_log_p,
            state->kernel_sum, state->grad_kernel_sum,
            bandwidth, n
        );
        
        svpf_stein_update_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            state->h, state->kernel_sum, state->grad_kernel_sum,
            SVPF_STEIN_STEP_SIZE, n
        );
    }
    
    // 7. Output statistics
    svpf_compute_vol_stats_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->d_reduce_buf, n
    );
    
    float vol_sum = reduce_sum(state->d_reduce_buf, state->d_temp, n, state->stream);
    
    result->log_lik_increment = log_lik_incr;
    result->vol_mean = vol_sum / (float)n;
    result->vol_std = 0.0f;
    result->h_mean = 0.0f;
    
    cudaStreamSynchronize(state->stream);
}

void svpf_get_particles(const SVPFState* state, float* h_out) {
    cudaMemcpy(h_out, state->h, state->n_particles * sizeof(float), cudaMemcpyDeviceToHost);
}

void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std) {
    int n = state->n_particles;
    
    // Copy particles to host
    float* h_host = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_host, state->h, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += h_host[i];
    }
    *h_mean = sum / (float)n;
    
    // Compute std
    float sq_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = h_host[i] - *h_mean;
        sq_sum += diff * diff;
    }
    *h_std = sqrtf(sq_sum / (float)n);
    
    free(h_host);
}
