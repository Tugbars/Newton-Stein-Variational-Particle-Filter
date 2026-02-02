/**
 * @file svpf_heun_kernels.cuh
 * @brief Heun's method (improved Euler) kernels for SVPF
 *
 * Heun's method is a predictor-corrector scheme achieving 2nd-order accuracy:
 *   1. Predictor: h̃ = h + ε·φ(h)              [Euler step]
 *   2. Corrector: h = h + (ε/2)·(φ(h) + φ(h̃)) [Trapezoidal average]
 *
 * Cost: 2× gradient/Stein evaluations per step
 * Benefit: Same iterations → 2× accuracy, or half iterations → same accuracy
 *
 * Usage:
 *   1. Compute gradient at h_orig → grad_orig
 *   2. Compute Stein operator → phi_orig (svpf_stein_operator_*)
 *   3. Predictor step → h_pred (svpf_heun_predictor_kernel)
 *   4. Compute gradient at h_pred → grad_pred
 *   5. Compute Stein operator at h_pred → phi_pred
 *   6. Corrector step → h_final (svpf_heun_corrector_*)
 */

#ifndef SVPF_HEUN_KERNELS_CUH
#define SVPF_HEUN_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Assumes these are defined in main kernel file or common header
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// =============================================================================
// Device Helpers (duplicated for standalone compilation)
// =============================================================================

__device__ __forceinline__ float heun_clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float heun_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float heun_block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = heun_warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = heun_warp_reduce_sum(val);
    return val;
}

// =============================================================================
// STEIN OPERATOR KERNELS (Compute Only, No Transport)
// =============================================================================

/**
 * @brief Compute Stein operator φ(h) without applying transport.
 * @param[in]  h           Current particle positions [n]
 * @param[in]  grad        Score function ∇log π(h) [n]
 * @param[out] phi_out     Stein operator output [n]
 * @param[in]  d_bandwidth Kernel bandwidth (device pointer)
 * @param[in]  stein_sign_mode 0=legacy (subtract), 1=paper (add)
 * @param[in]  n           Number of particles
 *
 * Shared memory: 2n floats
 */
__global__ void svpf_stein_operator_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi_out,
    const float* __restrict__ d_bandwidth,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        k_sum += K * sh_grad[j];
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
    }
    
    phi_out[i] = (k_sum + gk_sum) * inv_n;
}

/**
 * @brief Compute Stein operator with Full Newton preconditioning.
 * @param[in]  h             Current particle positions [n]
 * @param[in]  grad          Score function ∇log π(h) [n]
 * @param[in]  local_hessian Local Hessian estimates [n]
 * @param[out] phi_out       Stein operator output [n]
 * @param[in]  d_bandwidth   Kernel bandwidth (device pointer)
 * @param[in]  stein_sign_mode 0=legacy (subtract), 1=paper (add)
 * @param[in]  n             Number of particles
 *
 * Shared memory: 3n floats
 */
__global__ void svpf_stein_operator_full_newton_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ phi_out,
    const float* __restrict__ d_bandwidth,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    float* sh_hess = smem + 2 * n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
        sh_hess[k] = local_hessian[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;
    float k_grad_sum = 0.0f;
    float gk_sum = 0.0f;
    
    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        H_weighted += sh_hess[j] * K;
        float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        H_weighted += Nk;
        K_sum_norm += K;
        
        k_grad_sum += K * sh_grad[j];
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
    }
    
    H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
    H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);
    float inv_H_i = 1.0f / H_weighted;
    
    phi_out[i] = (k_grad_sum + gk_sum) * inv_n * inv_H_i * 0.7f;
}

// =============================================================================
// HEUN PREDICTOR KERNEL
// =============================================================================

/**
 * @brief Heun predictor step (Euler, no noise).
 *
 * Applies h̃ = h_orig + ε·φ with RMSProp preconditioning.
 * NO SVLD noise - that's only added in the corrector.
 *
 * @param[out] h         Output predicted positions [n]
 * @param[in]  h_orig    Original positions [n]
 * @param[in]  phi       Stein operator at h_orig [n]
 * @param[in]  v_rmsprop RMSProp variance estimates (read-only) [n]
 * @param[in]  step_size Base step size ε
 * @param[in]  beta_factor Annealing factor (multiply with step_size)
 * @param[in]  epsilon   RMSProp epsilon for numerical stability
 * @param[in]  n         Number of particles
 */
__global__ void svpf_heun_predictor_kernel(
    float* __restrict__ h,
    const float* __restrict__ h_orig,
    const float* __restrict__ phi,
    const float* __restrict__ v_rmsprop,
    float step_size,
    float beta_factor,
    float epsilon,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float phi_i = phi[i];
    float v_i = v_rmsprop[i];
    
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_i + epsilon);
    float drift = effective_step * phi_i * precond;
    
    // Predictor: NO noise, just drift
    h[i] = heun_clamp_logvol(h_orig[i] + drift);
}

// =============================================================================
// HEUN CORRECTOR KERNELS
// =============================================================================

/**
 * @brief Heun corrector step with RMSProp update and SVLD noise.
 *
 * Applies h = h_orig + (ε/2)·(φ_orig + φ_pred) using trapezoidal average.
 * SVLD noise is added here (not in predictor).
 *
 * @param[out]    h           Final output positions [n]
 * @param[in]     h_orig      Original positions (before predictor) [n]
 * @param[in]     phi_orig    Stein operator at h_orig [n]
 * @param[in]     phi_pred    Stein operator at h_pred [n]
 * @param[in,out] v_rmsprop   RMSProp variance estimates [n]
 * @param[in,out] rng         CURAND states [n]
 * @param[in]     step_size   Base step size ε
 * @param[in]     beta_factor Annealing factor
 * @param[in]     temperature SVLD temperature (0 = deterministic)
 * @param[in]     rho_rmsprop RMSProp decay factor
 * @param[in]     epsilon     RMSProp epsilon
 * @param[in]     n           Number of particles
 */
__global__ void svpf_heun_corrector_kernel(
    float* __restrict__ h,
    const float* __restrict__ h_orig,
    const float* __restrict__ phi_orig,
    const float* __restrict__ phi_pred,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Average the two Stein operators (trapezoidal rule)
    float phi_avg = 0.5f * (phi_orig[i] + phi_pred[i]);
    
    // RMSProp update on averaged phi
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_avg * phi_avg;
    v_rmsprop[i] = v_new;
    
    // Transport from ORIGINAL position (not predicted)
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_avg * precond;
    
    // SVLD noise (only in corrector)
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }
    
    h[i] = heun_clamp_logvol(h_orig[i] + drift + diffusion);
}

/**
 * @brief Heun corrector with KSD computation for adaptive stepping.
 *
 * Same as svpf_heun_corrector_kernel but also computes KSD partial sums
 * for convergence monitoring.
 *
 * @param[out]    h              Final output positions [n]
 * @param[in]     h_orig         Original positions [n]
 * @param[in]     phi_orig       Stein operator at h_orig [n]
 * @param[in]     phi_pred       Stein operator at h_pred [n]
 * @param[in]     grad           Score function for KSD [n]
 * @param[in,out] v_rmsprop      RMSProp variance estimates [n]
 * @param[in,out] rng            CURAND states [n]
 * @param[in]     d_bandwidth    Kernel bandwidth (device pointer)
 * @param[out]    d_ksd_partial  KSD partial sums [n]
 * @param[in]     step_size      Base step size ε
 * @param[in]     beta_factor    Annealing factor
 * @param[in]     temperature    SVLD temperature
 * @param[in]     rho_rmsprop    RMSProp decay factor
 * @param[in]     epsilon        RMSProp epsilon
 * @param[in]     n              Number of particles
 *
 * Shared memory: 2n floats
 */
__global__ void svpf_heun_corrector_ksd_kernel(
    float* __restrict__ h,
    const float* __restrict__ h_orig,
    const float* __restrict__ phi_orig,
    const float* __restrict__ phi_pred,
    const float* __restrict__ grad,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float* __restrict__ d_ksd_partial,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    
    // Load h_orig and grad for KSD computation
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h_orig[k];
        sh_grad[k] = grad[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // ----- Heun correction -----
    float phi_avg = 0.5f * (phi_orig[i] + phi_pred[i]);
    
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_avg * phi_avg;
    v_rmsprop[i] = v_new;
    
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_avg * precond;
    
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }
    
    h[i] = heun_clamp_logvol(h_orig[i] + drift + diffusion);
    
    // ----- KSD computation -----
    float h_i = sh_h[i];
    float s_i = sh_grad[i];
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    
    float ksd_sum = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float h_j = sh_h[j];
        float s_j = sh_grad[j];
        float diff = h_i - h_j;
        float diff_sq = diff * diff;
        float dist_sq = diff_sq * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_y_k = -grad_x_k;
        float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
        
        float u_ij = K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
        ksd_sum += u_ij;
    }
    
    d_ksd_partial[i] = ksd_sum;
}

// =============================================================================
// HOST-SIDE HELPERS
// =============================================================================

/**
 * @brief Compute shared memory size for Heun kernels.
 * @param n Number of particles
 * @param kernel_type 0=stein_op, 1=stein_op_newton, 2=corrector_ksd
 */
inline size_t heun_shared_mem_size(int n, int kernel_type) {
    switch (kernel_type) {
        case 0: return 2 * n * sizeof(float);  // stein_operator
        case 1: return 3 * n * sizeof(float);  // stein_operator_full_newton
        case 2: return 2 * n * sizeof(float);  // corrector_ksd
        default: return 0;
    }
}

#endif // SVPF_HEUN_KERNELS_CUH
