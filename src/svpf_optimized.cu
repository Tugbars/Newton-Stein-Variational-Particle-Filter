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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Configuration
// =============================================================================

#define TILE_J 64
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SMALL_N_THRESHOLD 4096
#define BANDWIDTH_UPDATE_INTERVAL 5

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

// =============================================================================
// Kernel 1: Predict (Run ONCE per timestep - samples process noise)
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
    h_prev[i] = h_i;  // Store for gradient computation
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    
    // Sample noise ONCE here - Stein iterations are deterministic after this
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

// =============================================================================
// Kernel 3: Log-Sum-Exp (computes loglik, all on device)
// =============================================================================

__global__ void svpf_logsumexp_kernel(
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,       // Single scalar output
    float* __restrict__ d_max_log_w,    // Stores max for exp kernel
    int t,                               // Timestep (for indexing d_loglik)
    int n
) {
    // Single block kernel for moderate N
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    
    // Pass 1: Find max
    float local_max = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, log_w[i]);
    }
    
    // Warp reduction for max
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
    
    // Pass 2: Sum exp(log_w - max)
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
 */
__global__ void svpf_stein_persistent_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ d_bandwidth,
    float step_size,
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
        
        // Thread 0 writes the update
        if (threadIdx.x == 0) {
            h[i] = clamp_logvol(h_i + step_size * (k_sum + gk_sum) / (float)n);
        }
        __syncthreads();  // Ensure write completes before next iteration
    }
}

// =============================================================================
// Kernel 6: Apply Transport (for 2D kernel path)
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

// =============================================================================
// Host State (CUB temp storage, device scalars)
// =============================================================================

struct SVPFOptimizedState {
    // CUB temporary storage
    void* d_temp_storage;
    size_t temp_storage_bytes;
    
    // Device scalars (NO host copies needed)
    float* d_max_log_w;
    float* d_sum_exp;
    float* d_bandwidth;
    float* d_bandwidth_sq;
    
    // Buffers
    float* d_exp_w;
    float* d_phi;
    
    bool initialized;
};

static SVPFOptimizedState g_opt = {0};

void svpf_optimized_init(int n) {
    if (g_opt.initialized) return;
    
    // Query CUB temp storage size
    float* d_dummy_in;
    float* d_dummy_out;
    cudaMalloc(&d_dummy_in, n * sizeof(float));
    cudaMalloc(&d_dummy_out, sizeof(float));
    
    g_opt.temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, g_opt.temp_storage_bytes, d_dummy_in, d_dummy_out, n);
    size_t sum_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, sum_bytes, d_dummy_in, d_dummy_out, n);
    g_opt.temp_storage_bytes = max(g_opt.temp_storage_bytes, sum_bytes);
    
    cudaMalloc(&g_opt.d_temp_storage, g_opt.temp_storage_bytes);
    cudaFree(d_dummy_in);
    cudaFree(d_dummy_out);
    
    // Device scalars
    cudaMalloc(&g_opt.d_max_log_w, sizeof(float));
    cudaMalloc(&g_opt.d_sum_exp, sizeof(float));
    cudaMalloc(&g_opt.d_bandwidth, sizeof(float));
    cudaMalloc(&g_opt.d_bandwidth_sq, sizeof(float));
    
    // Initialize bandwidth_sq to 0 (triggers fresh computation on first step)
    float zero = 0.0f;
    cudaMemcpy(g_opt.d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Buffers
    cudaMalloc(&g_opt.d_exp_w, n * sizeof(float));
    cudaMalloc(&g_opt.d_phi, n * sizeof(float));
    
    g_opt.initialized = true;
}

void svpf_optimized_cleanup(void) {
    if (!g_opt.initialized) return;
    
    cudaFree(g_opt.d_temp_storage);
    cudaFree(g_opt.d_max_log_w);
    cudaFree(g_opt.d_sum_exp);
    cudaFree(g_opt.d_bandwidth);
    cudaFree(g_opt.d_bandwidth_sq);
    cudaFree(g_opt.d_exp_w);
    cudaFree(g_opt.d_phi);
    
    g_opt.initialized = false;
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
    int n = state->n_particles;
    int n_stein = state->n_stein_steps;
    cudaStream_t stream = state->stream;
    
    svpf_optimized_init(n);
    
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
        
        // === 2. Likelihood + Gradient ===
        svpf_likelihood_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->grad_log_p, state->log_weights,
            d_observations, t,
            params->rho, params->sigma_z, params->mu, state->nu, student_t_const,
            n
        );
        
        // === 3. Log-Sum-Exp (all on device) ===
        if (n <= 4096) {
            // Single block for small N
            svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                state->log_weights, d_loglik_out, g_opt.d_max_log_w, t, n
            );
        } else {
            // Multi-block with CUB
            cub::DeviceReduce::Max(g_opt.d_temp_storage, g_opt.temp_storage_bytes,
                                   state->log_weights, g_opt.d_max_log_w, n, stream);
            
            svpf_exp_shifted_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->log_weights, g_opt.d_max_log_w, g_opt.d_exp_w, n
            );
            
            cub::DeviceReduce::Sum(g_opt.d_temp_storage, g_opt.temp_storage_bytes,
                                   g_opt.d_exp_w, g_opt.d_sum_exp, n, stream);
            
            svpf_finalize_loglik_kernel<<<1, 1, 0, stream>>>(
                g_opt.d_max_log_w, g_opt.d_sum_exp, d_loglik_out, t, n
            );
        }
        
        // === 4. Bandwidth (every few steps for stability) ===
        if (t % BANDWIDTH_UPDATE_INTERVAL == 0) {
            svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                state->h, g_opt.d_bandwidth, g_opt.d_bandwidth_sq, bw_alpha, n
            );
        }
        
        // === 5. Stein Transport Iterations ===
        for (int s = 0; s < n_stein; s++) {
            if (use_small_n) {
                // Persistent CTA path
                svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                    state->h, state->grad_log_p,
                    g_opt.d_bandwidth, SVPF_STEIN_STEP_SIZE, n
                );
            } else {
                // 2D tiled path
                cudaMemsetAsync(g_opt.d_phi, 0, n * sizeof(float), stream);
                
                svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                    state->h, state->grad_log_p, g_opt.d_phi, g_opt.d_bandwidth, n
                );
                
                svpf_apply_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, g_opt.d_phi, SVPF_STEIN_STEP_SIZE, n
                );
            }
            
            // Re-compute gradients for next Stein iteration (deterministic, no noise)
            if (s < n_stein - 1) {
                svpf_likelihood_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, state->h_prev, state->grad_log_p, state->log_weights,
                    d_observations, t,
                    params->rho, params->sigma_z, params->mu, state->nu, student_t_const,
                    n
                );
            }
        }
        
        // === 6. Volatility Mean ===
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
// Host API: Single Step (for real-time, still no mid-step sync)
// =============================================================================

void svpf_step_optimized(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,    // Host pointer (copied at end)
    float* h_vol_out        // Host pointer (copied at end)
) {
    // For single-step API, we need the observation on device
    // This is a limitation - caller should use sequence API for best perf
    
    int n = state->n_particles;
    cudaStream_t stream = state->stream;
    
    svpf_optimized_init(n);
    
    // Copy single observation to device (unavoidable for single-step API)
    float* d_y_single;
    cudaMalloc(&d_y_single, 2 * sizeof(float));
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpy(d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    float* d_loglik_single;
    float* d_vol_single;
    cudaMalloc(&d_loglik_single, sizeof(float));
    cudaMalloc(&d_vol_single, sizeof(float));
    
    // Run sequence of length 1 at t=1 (so y_prev is at index 0)
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
    
    // Check SMEM capacity
    if (persistent_smem > prop.sharedMemPerBlockOptin) {
        use_small_n = false;
    }
    
    // Predict
    svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
        state->h, state->h_prev, state->rng_states,
        d_y_single, 1, p.rho, p.sigma_z, p.mu, p.gamma, n
    );
    
    // Likelihood + Gradient
    svpf_likelihood_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
        state->h, state->h_prev, state->grad_log_p, state->log_weights,
        d_y_single, 1, p.rho, p.sigma_z, p.mu, state->nu, student_t_const, n
    );
    
    // Log-Sum-Exp
    svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->log_weights, d_loglik_single, g_opt.d_max_log_w, 0, n
    );
    
    // Bandwidth
    svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, g_opt.d_bandwidth, g_opt.d_bandwidth_sq, 0.3f, n
    );
    
    // Stein iterations
    for (int s = 0; s < state->n_stein_steps; s++) {
        if (use_small_n) {
            svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                state->h, state->grad_log_p, g_opt.d_bandwidth, SVPF_STEIN_STEP_SIZE, n
            );
        } else {
            cudaMemsetAsync(g_opt.d_phi, 0, n * sizeof(float), stream);
            int num_tiles = (n + TILE_J - 1) / TILE_J;
            dim3 grid_2d(n, num_tiles);
            svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->grad_log_p, g_opt.d_phi, g_opt.d_bandwidth, n
            );
            svpf_apply_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, g_opt.d_phi, SVPF_STEIN_STEP_SIZE, n
            );
        }
        
        if (s < state->n_stein_steps - 1) {
            svpf_likelihood_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p, state->log_weights,
                d_y_single, 1, p.rho, p.sigma_z, p.mu, state->nu, student_t_const, n
            );
        }
    }
    
    // Vol mean
    svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, d_vol_single, 0, n
    );
    
    // Single sync + copy results
    cudaStreamSynchronize(stream);
    
    if (h_loglik_out) cudaMemcpy(h_loglik_out, d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_vol_out) cudaMemcpy(h_vol_out, d_vol_single, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_y_single);
    cudaFree(d_loglik_single);
    cudaFree(d_vol_single);
}
