/**
 * @file svpf_opt_kernels.cu
 * @brief CUDA kernel definitions for optimized SVPF paths
 * 
 * Fused kernels for low-latency execution:
 * - svpf_fused_gradient_kernel (prior + likelihood + combine + hessian)
 * - svpf_fused_stein_transport_kernel (stein + transport)
 * - svpf_fused_bandwidth_kernel (bandwidth + adaptive)
 * - svpf_fused_outputs_kernel (logsumexp + vol + h_mean)
 */

#include "svpf_kernels.cuh"
#include <stdio.h>

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float safe_exp(float x) {
    return __expf(fminf(x, 20.0f));
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
// Predict Kernels
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
    float log_y2 = __logf(y_curr * y_curr + 1e-10f);
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

// Graph-compatible version: reads guide_mean AND guide_strength from device
__global__ void svpf_apply_guide_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_guide_mean,
    const float* __restrict__ d_guide_strength,  // NEW: read from device
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float guide_mean = *d_guide_mean;
    float guide_strength = *d_guide_strength;  // Read adaptive strength
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

// Graph-compatible version: reads guide_mean AND guide_strength from device
__global__ void svpf_apply_guide_preserving_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    const float* __restrict__ d_guide_mean,
    const float* __restrict__ d_guide_strength,  // NEW: read from device
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float current_mean = *d_h_mean;
    float guide_mean = *d_guide_mean;
    float guide_strength = *d_guide_strength;  // Read adaptive strength
    float h_val = h[i];
    
    float deviation = h_val - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    
    h[i] = clamp_logvol(new_mean + deviation);
}

// =============================================================================
// FUSED: Gradient Pipeline
// Combines: mixture_prior_grad + likelihood_only + combine + hessian_precond
// =============================================================================

__global__ void svpf_fused_gradient_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_combined,
    float* __restrict__ log_w,
    float* __restrict__ precond_grad,
    float* __restrict__ inv_hessian,
    const float* __restrict__ d_y,
    int y_idx,
    float rho,
    float sigma_z,
    float mu,
    float beta,
    float nu,
    float student_t_const,
    bool use_newton,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h_prev = smem;
    float* sh_mu_i = smem + n;
    
    // Cooperative load: h_prev + precompute prior means
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float hp = h_prev[k];
        sh_h_prev[k] = hp;
        sh_mu_i[k] = mu + rho * (hp - mu);
    }
    __syncthreads();
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float y_t = d_y[y_idx];
    
    // ===== PRIOR GRADIENT (mixture of Gaussians) =====
    float sigma_z_sq = sigma_z * sigma_z;
    float inv_2sigma_sq = 0.5f / sigma_z_sq;
    float inv_sigma_sq = 1.0f / sigma_z_sq;
    
    // Pass 1: numerical stability - find max log weight
    float log_r_max = -1e10f;
    for (int i = 0; i < n; i++) {
        float diff = h_j - sh_mu_i[i];
        float log_r_i = -diff * diff * inv_2sigma_sq;
        log_r_max = fmaxf(log_r_max, log_r_i);
    }
    
    // Pass 2: weighted gradient
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = h_j - sh_mu_i[i];
        float log_r_i = -diff * diff * inv_2sigma_sq;
        float r_i = __expf(log_r_i - log_r_max);
        sum_r += r_i;
        weighted_grad -= r_i * diff * inv_sigma_sq;
    }
    float grad_prior = weighted_grad / (sum_r + 1e-8f);
    
    // ===== LIKELIHOOD GRADIENT (Student-t) =====
    float vol = safe_exp(h_j);
    float y_sq = y_t * y_t;
    float scaled_y_sq = y_sq / (vol + 1e-8f);
    
    // Log-weight for ESS / resampling
    log_w[j] = student_t_const - 0.5f * h_j
             - (nu + 1.0f) * 0.5f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    // Observation pull gradient
    float log_y2 = __logf(y_sq + 1e-10f);
    float R_noise = 1.4f;
    float grad_lik = (log_y2 - h_j + 1.0f / nu) / R_noise;
    
    // ===== COMBINE (annealed) =====
    float g = grad_prior + beta * grad_lik;
    g = fminf(fmaxf(g, -10.0f), 10.0f);
    grad_combined[j] = g;
    
    // ===== HESSIAN PRECONDITIONING (Newton) =====
    if (use_newton && precond_grad != nullptr) {
        float A = scaled_y_sq / nu;
        float one_plus_A = 1.0f + A;
        float hess_lik = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
        float hess_prior = -inv_sigma_sq;
        
        float curvature = -(hess_lik + hess_prior);
        curvature = fminf(fmaxf(curvature, 0.1f), 100.0f);
        
        float inv_H = 1.0f / curvature;
        precond_grad[j] = 0.7f * g * inv_H;
        inv_hessian[j] = inv_H;
    }
}

// =============================================================================
// FUSED: Stein + Transport (Standard)
// Combines: memset(phi) + stein_kernel + transport
// =============================================================================

__global__ void svpf_fused_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
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
    
    // Load all particles into shared memory
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float bandwidth = *d_bandwidth;
    float bw_sq = bandwidth * bandwidth;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_2bw_sq = 0.5f * inv_bw_sq;
    float inv_n = 1.0f / (float)n;
    
    // ===== STEIN OPERATOR (inline) =====
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float K = __expf(-diff * diff * inv_2bw_sq);
        k_sum += K * sh_grad[j];
        gk_sum -= K * diff * inv_bw_sq;  // Original sign
    }
    
    float phi_i = (k_sum + gk_sum) * inv_n;
    
    // ===== RMSPROP =====
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
    // ===== TRANSPORT =====
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_i * precond;
    
    // SVLD diffusion
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }
    
    h[i] = clamp_logvol(h_i + drift + diffusion);
}

// =============================================================================
// FUSED: Stein + Transport (Newton)
// =============================================================================

__global__ void svpf_fused_stein_transport_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_precond_grad = smem + n;
    float* sh_inv_hess = smem + 2 * n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_precond_grad[k] = precond_grad[k];
        sh_inv_hess[k] = inv_hessian[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float bandwidth = *d_bandwidth;
    float bw_sq = bandwidth * bandwidth;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_2bw_sq = 0.5f * inv_bw_sq;
    float inv_n = 1.0f / (float)n;
    
    // ===== NEWTON-STEIN OPERATOR =====
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float K = __expf(-diff * diff * inv_2bw_sq);
        k_sum += K * sh_precond_grad[j];
        gk_sum -= K * diff * inv_bw_sq * sh_inv_hess[j];  // Original sign
    }
    
    float phi_i = (k_sum + gk_sum) * inv_n;
    
    // ===== RMSPROP =====
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
    // ===== TRANSPORT =====
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_i * precond;
    
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }
    
    h[i] = clamp_logvol(h_i + drift + diffusion);
}

// =============================================================================
// FUSED: Outputs (logsumexp + vol_mean + h_mean)
// =============================================================================

__global__ void svpf_fused_outputs_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    int t_out,
    int n
) {
    __shared__ float s_max;
    
    // ===== PASS 1: Find max log-weight =====
    float local_max = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, log_w[i]);
    }
    local_max = block_reduce_max(local_max);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_log_w = s_max;
    
    // ===== PASS 2: Sum exp(log_w - max), sum vol, sum h =====
    float local_sum_exp = 0.0f;
    float local_sum_vol = 0.0f;
    float local_sum_h = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum_exp += __expf(log_w[i] - max_log_w);
        float h_i = h[i];
        local_sum_vol += safe_exp(h_i * 0.5f);
        local_sum_h += h_i;
    }
    
    local_sum_exp = block_reduce_sum(local_sum_exp);
    __syncthreads();
    local_sum_vol = block_reduce_sum(local_sum_vol);
    __syncthreads();
    local_sum_h = block_reduce_sum(local_sum_h);
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        d_loglik[t_out] = max_log_w + __logf(local_sum_exp * inv_n + 1e-10f);
        d_vol[t_out] = local_sum_vol * inv_n;
        *d_h_mean = local_sum_h * inv_n;
    }
}

// =============================================================================
// FUSED: Bandwidth with Adaptive Scaling
// =============================================================================

__global__ void svpf_fused_bandwidth_kernel(
    const float* __restrict__ h,
    const float* __restrict__ d_y,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    int y_idx,
    float alpha_bw,
    float alpha_ret,
    int n
) {
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    float local_min = 1e10f;
    float local_max = -1e10f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_sum += val;
        local_sum_sq += val * val;
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }
    
    local_sum = block_reduce_sum(local_sum);
    __syncthreads();
    local_sum_sq = block_reduce_sum(local_sum_sq);
    __syncthreads();
    local_min = block_reduce_min(local_min);
    __syncthreads();
    local_max = block_reduce_max(local_max);
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        float mean = local_sum * inv_n;
        float variance = local_sum_sq * inv_n - mean * mean;
        float spread = local_max - local_min;
        
        // Silverman's rule with EMA smoothing
        float bw_sq_new = 2.0f * variance / __logf((float)n + 1.0f);
        bw_sq_new = fmaxf(bw_sq_new, 1e-6f);
        
        float bw_sq_prev = *d_bandwidth_sq;
        float bw_sq = (bw_sq_prev > 0.0f)
                    ? alpha_bw * bw_sq_new + (1.0f - alpha_bw) * bw_sq_prev
                    : bw_sq_new;
        
        // Adaptive scaling based on return volatility
        float new_return = d_y[y_idx];
        float abs_ret = fabsf(new_return);
        float ret_ema = *d_return_ema;
        float ret_var = *d_return_var;
        
        ret_ema = (ret_ema > 0.0f)
                ? alpha_ret * abs_ret + (1.0f - alpha_ret) * ret_ema
                : abs_ret;
        ret_var = (ret_var > 0.0f)
                ? alpha_ret * abs_ret * abs_ret + (1.0f - alpha_ret) * ret_var
                : abs_ret * abs_ret;
        
        *d_return_ema = ret_ema;
        *d_return_var = ret_var;
        
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        float spread_factor = fminf(spread * 0.5f, 2.0f);
        float combined = fmaxf(vol_ratio, spread_factor);
        
        float scale = 1.0f - 0.25f * fminf(combined - 1.0f, 2.0f);
        scale = fmaxf(fminf(scale, 1.0f), 0.5f);
        
        bw_sq *= scale;
        float bw = sqrtf(bw_sq);
        bw = fmaxf(fminf(bw, 2.0f), 0.01f);
        
        *d_bandwidth_sq = bw_sq;
        *d_bandwidth = bw;
    }
}
