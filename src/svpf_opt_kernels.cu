/**
 * @file svpf_opt_kernels.cu
 * @brief CUDA kernel definitions for optimized SVPF paths
 * 
 * Fused kernels for low-latency execution:
 * - svpf_fused_gradient_kernel (prior + likelihood + combine + hessian)
 * - svpf_fused_stein_transport_kernel (stein + transport)
 * - svpf_fused_stein_transport_ksd_kernel (stein + transport + KSD)
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

// Sample from Student-t distribution via ratio of Gaussian to sqrt(Chi-squared/nu)
// For small nu (5-7), this is ~5-7 extra curand_normal calls per particle - negligible
// Note: This implementation assumes integer nu. For nu > 30, Student-t ≈ Gaussian.
__device__ __forceinline__ float sample_student_t(curandStatePhilox4_32_10_t* rng, float nu) {
    // Large nu: Student-t converges to Gaussian
    if (nu > 30.0f) {
        return curand_normal(rng);
    }
    
    float z = curand_normal(rng);
    
    // Chi-squared(nu) via sum of nu squared standard normals
    // Rounded to nearest integer - use integer nu for correctness
    int nu_int = (int)(nu + 0.5f);
    nu_int = max(nu_int, 3);  // Safety floor
    
    float chi2 = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < nu_int; i++) {
        float u = curand_normal(rng);
        chi2 += u * u;
    }
    
    // t = z / sqrt(chi2 / nu) = z * sqrt(nu / chi2)
    return z * sqrtf((float)nu_int / (chi2 + 1e-8f));
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
    int use_student_t_state, float nu_state,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    h_prev[i] = h_i;
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    
    // Sample innovation: Gaussian or Student-t
    float noise;
    if (use_student_t_state) {
        noise = sample_student_t(&rng[i], nu_state);
    } else {
        noise = curand_normal(&rng[i]);
    }
    
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
    int use_student_t_state, float nu_state,
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

    // Sample innovation: Gaussian or Student-t
    float noise;
    if (use_student_t_state) {
        noise = sample_student_t(&rng[i], nu_state);
    } else {
        noise = curand_normal(&rng[i]);
    }
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
    float implied_offset,
    int use_student_t_state, float nu_state,
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

    // Sample innovation: Gaussian or Student-t
    float noise;
    if (use_student_t_state) {
        noise = sample_student_t(&rng[i], nu_state);
    } else {
        noise = curand_normal(&rng[i]);
    }
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
    float mean_implied = fmaxf(log_y2 + implied_offset, -5.0f);

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
// ANTITHETIC SAMPLING VERSION
// =============================================================================
// Each thread handles TWO particles: i and i + n/2
// They share the same z, but particle i+n/2 uses -z
// This halves variance of expectations over the transition distribution.
// Launch with n/2 threads!

__global__ void svpf_predict_guided_antithetic_kernel(
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
    float implied_offset,
    int use_student_t_state, float nu_state,
    int n  // FULL n, not n/2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_n = n / 2;
    if (i >= half_n) return;
    
    int j = i + half_n;  // Antithetic partner
    
    // Load both particles
    float h_i = h[i];
    float h_j = h[j];
    float h_prev_i = h_prev[i];
    float h_prev_j = h_prev[j];
    
    // Save to h_prev
    h_prev[i] = h_i;
    h_prev[j] = h_j;
    
    float h_bar = *d_h_mean;
    
    // Generate ONE random sample, use +z and -z
    float z;
    if (use_student_t_state) {
        z = sample_student_t(&rng[i], nu_state);
    } else {
        z = curand_normal(&rng[i]);
    }
    
    // MIM jump: same decision for both (they'll go opposite directions)
    float selector = curand_uniform(&rng[i]);
    float scale = (selector < jump_prob) ? jump_scale : 1.0f;
    
    // Process particle i (with +z)
    {
        float dev = h_i - h_bar;
        float rho_adjust = delta_rho * tanhf(dev);
        float sigma_scale = 1.0f + delta_sigma * fabsf(dev);
        
        float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
        float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
        float sigma_local = sigma_z * sigma_scale;
        
        float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
        float vol_prev = safe_exp(h_i / 2.0f);
        float leverage = gamma * y_prev / (vol_prev + 1e-8f);
        float mean_prior = mu + rho * (h_i - mu) + leverage;
        
        float y_curr = d_y[t];
        float log_y2 = __logf(y_curr * y_curr + 1e-10f);
        float mean_implied = fmaxf(log_y2 + implied_offset, -5.0f);
        
        float innovation = mean_implied - mean_prior;
        float z_score = innovation / 2.5f;
        
        float activation = 0.0f;
        if (z_score > innovation_threshold) {
            activation = tanhf(z_score - innovation_threshold);
        }
        
        float guided_alpha = alpha_base + (alpha_shock - alpha_base) * activation;
        float mean_proposal = (1.0f - guided_alpha) * mean_prior + guided_alpha * mean_implied;
        
        h[i] = clamp_logvol(mean_proposal + sigma_local * scale * z);  // +z
    }
    
    // Process particle j (with -z)
    {
        float dev = h_j - h_bar;
        float rho_adjust = delta_rho * tanhf(dev);
        float sigma_scale = 1.0f + delta_sigma * fabsf(dev);
        
        float base_rho = (h_j > h_prev_j) ? rho_up : rho_down;
        float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
        float sigma_local = sigma_z * sigma_scale;
        
        float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
        float vol_prev = safe_exp(h_j / 2.0f);
        float leverage = gamma * y_prev / (vol_prev + 1e-8f);
        float mean_prior = mu + rho * (h_j - mu) + leverage;
        
        float y_curr = d_y[t];
        float log_y2 = __logf(y_curr * y_curr + 1e-10f);
        float mean_implied = fmaxf(log_y2 + implied_offset, -5.0f);
        
        float innovation = mean_implied - mean_prior;
        float z_score = innovation / 2.5f;
        
        float activation = 0.0f;
        if (z_score > innovation_threshold) {
            activation = tanhf(z_score - innovation_threshold);
        }
        
        float guided_alpha = alpha_base + (alpha_shock - alpha_base) * activation;
        float mean_proposal = (1.0f - guided_alpha) * mean_prior + guided_alpha * mean_implied;
        
        h[j] = clamp_logvol(mean_proposal + sigma_local * scale * (-z));  // -z
    }
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
    const float* __restrict__ d_guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float guide_mean = *d_guide_mean;
    float guide_strength = *d_guide_strength;
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
    const float* __restrict__ d_guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float current_mean = *d_h_mean;
    float guide_mean = *d_guide_mean;
    float guide_strength = *d_guide_strength;
    float h_val = h[i];
    
    float deviation = h_val - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    
    h[i] = clamp_logvol(new_mean + deviation);
}

// =============================================================================
// FUSED: Gradient Pipeline
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
    float lik_offset,
    float gamma,
    bool use_exact_gradient,
    bool use_newton,
    bool use_fan_mode,
    int use_student_t_state,
    float nu_state,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h_prev = smem;
    float* sh_mu_i = smem + n;
    
    float y_prev = (y_idx > 0) ? d_y[y_idx - 1] : 0.0f;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float hp = h_prev[k];
        sh_h_prev[k] = hp;
        
        float vol_prev_k = __expf(hp * 0.5f);
        float leverage_k = gamma * y_prev / (vol_prev_k + 1e-8f);
        
        sh_mu_i[k] = mu + rho * (hp - mu) + leverage_k;
    }
    __syncthreads();
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float y_t = d_y[y_idx];
    
    // ===== PRIOR GRADIENT =====
    float sigma_z_sq = sigma_z * sigma_z;
    float grad_prior;
    float hess_prior;
    
    if (use_student_t_state) {
        // Student-t prior: bounded gradient
        // log p(h|mu) = const - ((nu_state+1)/2) * log(1 + (h-mu)²/(nu_state*sigma²))
        // grad = -(nu_state+1) * (h-mu) / (nu_state*sigma² + (h-mu)²)
        float nu_sigma_sq = nu_state * sigma_z_sq;
        float nu_plus_1 = nu_state + 1.0f;
        float half_nu_plus_1 = 0.5f * nu_plus_1;
        
        float log_r_max = -1e10f;
        #pragma unroll 8
        for (int i = 0; i < n; i++) {
            float diff = h_j - sh_mu_i[i];
            float diff_sq = diff * diff;
            // log r_i = -((nu+1)/2) * log(1 + diff²/(nu*sigma²))
            float log_r_i = -half_nu_plus_1 * __logf(1.0f + diff_sq / nu_sigma_sq);
            log_r_max = fmaxf(log_r_max, log_r_i);
        }
        
        float sum_r = 0.0f;
        float weighted_grad = 0.0f;
        float weighted_hess = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < n; i++) {
            float diff = h_j - sh_mu_i[i];
            float diff_sq = diff * diff;
            float denom = nu_sigma_sq + diff_sq;
            
            float log_r_i = -half_nu_plus_1 * __logf(1.0f + diff_sq / nu_sigma_sq);
            float r_i = __expf(log_r_i - log_r_max);
            sum_r += r_i;
            
            // Bounded gradient: -(nu+1) * diff / (nu*sigma² + diff²)
            weighted_grad -= r_i * nu_plus_1 * diff / denom;
            
            // Hessian for Student-t: d²/dh² log p
            // = -(nu+1) * (nu*sigma² - diff²) / (nu*sigma² + diff²)²
            float hess_i = -nu_plus_1 * (nu_sigma_sq - diff_sq) / (denom * denom);
            weighted_hess += r_i * hess_i;
        }
        grad_prior = weighted_grad / (sum_r + 1e-8f);
        hess_prior = weighted_hess / (sum_r + 1e-8f);
        
    } else {
        // Gaussian prior: original unbounded gradient
        float inv_2sigma_sq = 0.5f / sigma_z_sq;
        float inv_sigma_sq = 1.0f / sigma_z_sq;
        
        float log_r_max = -1e10f;
        #pragma unroll 8
        for (int i = 0; i < n; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            log_r_max = fmaxf(log_r_max, log_r_i);
        }
        
        float sum_r = 0.0f;
        float weighted_grad = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < n; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            float r_i = __expf(log_r_i - log_r_max);
            sum_r += r_i;
            weighted_grad -= r_i * diff * inv_sigma_sq;
        }
        grad_prior = weighted_grad / (sum_r + 1e-8f);
        hess_prior = -inv_sigma_sq;
    }
    
    // ===== LIKELIHOOD GRADIENT =====
    float vol = safe_exp(h_j);
    float y_sq = y_t * y_t;
    float scaled_y_sq = y_sq / (vol + 1e-8f);
    float A = scaled_y_sq / nu;
    float one_plus_A = 1.0f + A;
    
    // Fan mode: uniform weights (log_w = 0)
    // Hybrid mode: importance weights from likelihood
    if (use_fan_mode) {
        log_w[j] = 0.0f;
    } else {
        log_w[j] = student_t_const - 0.5f * h_j
                 - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));
    }
    
    float grad_lik;
    if (use_exact_gradient) {
        float raw_grad = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A;
        grad_lik = raw_grad - lik_offset;
    } else {
        float log_y2 = __logf(y_sq + 1e-10f);
        float R_noise = 1.4f;
        grad_lik = (log_y2 - h_j + lik_offset) / R_noise;
    }
    
    // ===== COMBINE =====
    // Fan mode: full likelihood (beta effectively 1.0)
    // Hybrid mode: annealed likelihood
    float effective_beta = use_fan_mode ? 1.0f : beta;
    float g = grad_prior + effective_beta * grad_lik;
    g = fminf(fmaxf(g, -10.0f), 10.0f);
    grad_combined[j] = g;
    
    // ===== HESSIAN =====
    if (use_newton && precond_grad != nullptr) {
        float hess_lik = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
        
        float curvature = -(hess_lik + hess_prior);
        curvature = fminf(fmaxf(curvature, 0.1f), 100.0f);
        
        inv_hessian[j] = curvature;
        
        float inv_H = 1.0f / curvature;
        precond_grad[j] = 0.7f * g * inv_H;
    }
}

// =============================================================================
// FUSED: Stein + Transport (Standard, no KSD)
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
    
    // ===== STEIN OPERATOR with IMQ Kernel =====
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
// FUSED: Stein + Transport + KSD (computes KSD in same O(N²) pass)
// =============================================================================
// 
// KSD (Kernel Stein Discrepancy) measures how far particles are from target.
// KSD² = (1/N²) Σᵢ Σⱼ u_p(xᵢ, xⱼ)
// where u_p is the Stein kernel:
//   u_p(x,y) = k(x,y)·s(x)·s(y) + s(x)·∇ₓk + s(y)·∇ᵧk + ∇ₓ∇ᵧk
//
// For IMQ kernel k(x,y) = (1 + ||x-y||²/h²)^(-1):
//   ∇ₓk = -2(x-y)/h² · k²
//   ∇ᵧk = +2(x-y)/h² · k²
//   ∇ₓ∇ᵧk = 2k²/h² · (4(x-y)²k/h² - 1)
//
// This can be computed in the SAME O(N²) loop as Stein operator at zero cost.
// =============================================================================

__global__ void svpf_fused_stein_transport_ksd_kernel(
    float* __restrict__ h,
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
    float s_i = sh_grad[i];
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    // ===== FUSED: Stein operator + KSD =====
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
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
        
        k_sum += K * s_j;
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        
        float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_y_k = -grad_x_k;
        float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
        
        float u_ij = K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
        ksd_sum += u_ij;
    }
    
    float phi_i = (k_sum + gk_sum) * inv_n;
    d_ksd_partial[i] = ksd_sum;
    
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
// KSD Reduction Kernel
// =============================================================================
// Reduces partial sums to final KSD² value
// KSD² = (1/N²) * Σᵢ partial[i]

__global__ void svpf_ksd_reduce_kernel(
    const float* __restrict__ d_ksd_partial,
    float* __restrict__ d_ksd,
    int n
) {
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += d_ksd_partial[i];
    }
    
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        float inv_n_sq = 1.0f / ((float)n * (float)n);
        float ksd_sq = local_sum * inv_n_sq;
        // Return sqrt(KSD²) = KSD for easier interpretation
        *d_ksd = sqrtf(fmaxf(ksd_sq, 0.0f));
    }
}

// =============================================================================
// FUSED: Stein + Transport (Newton, no KSD)
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
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_precond_grad = smem + n;
    float* sh_inv_hess = smem + 2 * n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_precond_grad[k] = precond_grad[k];
        sh_inv_hess[k] = 1.0f / inv_hessian[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    
    // Sign multiplier: -1 for legacy (attraction), +1 for paper (repulsion)
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
        
        k_sum += K * sh_precond_grad[j];
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq * sh_inv_hess[j];
    }
    
    float phi_i = (k_sum + gk_sum) * inv_n;
    
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
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
// FUSED: Stein + Transport + KSD (Newton variant)
// =============================================================================

__global__ void svpf_fused_stein_transport_newton_ksd_kernel(
    float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float* __restrict__ d_ksd_partial,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_precond_grad = smem + n;
    float* sh_inv_hess = smem + 2 * n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_precond_grad[k] = precond_grad[k];
        sh_inv_hess[k] = 1.0f / inv_hessian[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float s_i = sh_precond_grad[i];
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    
    // Sign multiplier: -1 for legacy (attraction), +1 for paper (repulsion)
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    float ksd_sum = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float h_j = sh_h[j];
        float s_j = sh_precond_grad[j];
        float diff = h_i - h_j;
        float diff_sq = diff * diff;
        float dist_sq = diff_sq * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        k_sum += K * s_j;
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq * sh_inv_hess[j];
        
        // KSD
        float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_y_k = -grad_x_k;
        float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
        float u_ij = K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
        ksd_sum += u_ij;
    }
    
    float phi_i = (k_sum + gk_sum) * inv_n;
    d_ksd_partial[i] = ksd_sum;
    
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
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
// FUSED: Stein + Transport (Full Newton)
// =============================================================================

__global__ void svpf_fused_stein_transport_full_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
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
    
    // Sign multiplier: -1 for legacy (attraction), +1 for paper (repulsion)
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
    
    float phi_i = (k_grad_sum + gk_sum) * inv_n * inv_H_i * 0.7f;
    
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
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
// FUSED: Stein + Transport (Full Newton with KSD)
// =============================================================================
// Same as full_newton but also computes KSD for adaptive stepping

__global__ void svpf_fused_stein_transport_full_newton_ksd_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float* __restrict__ d_ksd_partial,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
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
    float s_i = sh_grad[i];  // Raw score for KSD
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    
    // Sign multiplier: -1 for legacy (attraction), +1 for paper (repulsion)
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;
    float k_grad_sum = 0.0f;
    float gk_sum = 0.0f;
    float ksd_sum = 0.0f;
    
    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        float h_j = sh_h[j];
        float s_j = sh_grad[j];
        float diff = h_i - h_j;
        float diff_sq = diff * diff;
        float dist_sq = diff_sq * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        // ----- Full Newton: Hessian weighting -----
        H_weighted += sh_hess[j] * K;
        float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        H_weighted += Nk;
        K_sum_norm += K;
        
        k_grad_sum += K * s_j;
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        
        // ----- KSD Stein kernel (uses raw gradients) -----
        float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_y_k = -grad_x_k;
        float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
        float u_ij = K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
        ksd_sum += u_ij;
    }
    
    // Store partial KSD sum
    d_ksd_partial[i] = ksd_sum;
    
    // Full Newton preconditioning
    H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
    H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);
    float inv_H_i = 1.0f / H_weighted;
    
    float phi_i = (k_grad_sum + gk_sum) * inv_n * inv_H_i * 0.7f;
    
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
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
// FUSED: Outputs
// =============================================================================

__global__ void svpf_fused_outputs_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    const float* __restrict__ d_bandwidth_in,  // Read bandwidth for packing
    const float* __restrict__ d_ksd_in,        // Read KSD for packing
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    float* __restrict__ d_output_pack,         // Packed output [5 floats]
    int t_out,
    int n
) {
    __shared__ float s_max;
    
    float local_max = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, log_w[i]);
    }
    local_max = block_reduce_max(local_max);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_log_w = s_max;
    
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
        float safe_sum = fmaxf(local_sum_exp * inv_n, 1e-10f);
        float loglik = max_log_w + __logf(safe_sum);
        float vol = local_sum_vol * inv_n;
        float h_mean = local_sum_h * inv_n;
        
        // Write to legacy outputs (backward compat)
        d_loglik[t_out] = loglik;
        d_vol[t_out] = vol;
        *d_h_mean = h_mean;
        
        // Pack all outputs for single D2H transfer
        d_output_pack[0] = loglik;
        d_output_pack[1] = vol;
        d_output_pack[2] = h_mean;
        d_output_pack[3] = *d_bandwidth_in;
        d_output_pack[4] = *d_ksd_in;
    }
}

// =============================================================================
// FUSED: Bandwidth
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
        
        float bw_sq_new = 2.0f * variance / __logf((float)n + 1.0f);
        bw_sq_new = fmaxf(bw_sq_new, 1e-6f);
        
        float bw_sq_prev = *d_bandwidth_sq;
        float bw_sq = (bw_sq_prev > 0.0f)
                    ? alpha_bw * bw_sq_new + (1.0f - alpha_bw) * bw_sq_prev
                    : bw_sq_new;
        
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

// =============================================================================
// PARTIAL REJUVENATION KERNEL (Maken et al. 2022)
// =============================================================================
// When particles are stuck (high KSD after Stein), randomly select some
// and nudge them toward the EKF guide prediction.
//
// For each particle with probability rejuv_prob:
//   h_new = (1 - blend) * h_old + blend * (guide_mean + guide_std * noise)
//
// This helps escape local modes at reflecting boundaries.

__global__ void svpf_partial_rejuvenation_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_std,
    float rejuv_prob,
    float blend_factor,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Draw uniform random to decide if this particle gets rejuvenated
    float u = curand_uniform(&rng[i]);
    
    if (u < rejuv_prob) {
        // This particle will be nudged toward guide
        float z = curand_normal(&rng[i]);
        float guide_sample = guide_mean + guide_std * z;
        
        // Blend current position with guide sample
        float h_old = h[i];
        float h_new = (1.0f - blend_factor) * h_old + blend_factor * guide_sample;
        
        // Clamp to valid range
        h[i] = clamp_logvol(h_new);
    }
}

// =============================================================================
// HEUN'S METHOD KERNELS
// =============================================================================
// Heun's method (improved Euler) is a predictor-corrector scheme:
//   1. Predictor: h̃ = h + ε·φ(h)         [Euler step]
//   2. Corrector: h = h + (ε/2)·(φ(h) + φ(h̃))
//
// Achieves second-order accuracy vs first-order for Euler.
// Cost: 2× gradient/Stein evaluations per step.
// Benefit: Can use half iterations for same accuracy, or same for 2× better.

// -----------------------------------------------------------------------------
// STEIN OPERATOR KERNEL (Compute Only, No Transport)
// -----------------------------------------------------------------------------
// Computes φ(h) and stores to output buffer. Does NOT apply transport.

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

// -----------------------------------------------------------------------------
// STEIN OPERATOR KERNEL (Full Newton variant)
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// HEUN PREDICTOR KERNEL (Euler step, no noise)
// -----------------------------------------------------------------------------
// Applies h̃ = h + ε·φ with RMSProp preconditioning, NO SVLD noise yet

__global__ void svpf_heun_predictor_kernel(
    float* __restrict__ h,
    const float* __restrict__ h_orig,
    const float* __restrict__ phi,
    const float* __restrict__ v_rmsprop,  // Read-only for predictor
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
    h[i] = clamp_logvol(h_orig[i] + drift);
}

// -----------------------------------------------------------------------------
// HEUN CORRECTOR KERNEL
// -----------------------------------------------------------------------------
// Applies h = h_orig + (ε/2)·(φ₁ + φ₂) with RMSProp update and SVLD noise

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
    
    // Average the two Stein operators
    float phi_avg = 0.5f * (phi_orig[i] + phi_pred[i]);
    
    // RMSProp update on averaged phi
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_avg * phi_avg;
    v_rmsprop[i] = v_new;
    
    // Transport from ORIGINAL position
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_avg * precond;
    
    // SVLD noise (only in corrector)
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }
    
    h[i] = clamp_logvol(h_orig[i] + drift + diffusion);
}

// -----------------------------------------------------------------------------
// HEUN CORRECTOR WITH KSD KERNEL
// -----------------------------------------------------------------------------
// Same as corrector but also computes KSD for adaptive stepping

__global__ void svpf_heun_corrector_ksd_kernel(
    float* __restrict__ h,
    const float* __restrict__ h_orig,
    const float* __restrict__ phi_orig,
    const float* __restrict__ phi_pred,
    const float* __restrict__ grad,  // For KSD computation
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
    
    h[i] = clamp_logvol(h_orig[i] + drift + diffusion);
    
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
// MEGA-FUSED: Gradient + Stein + Transport (No Newton, No KSD)
// =============================================================================
// 
// Fuses svpf_fused_gradient_kernel + svpf_fused_stein_transport_kernel
// into a single kernel launch, halving launch overhead per Stein iteration.
// 
// This is for non-final iterations (no KSD computation needed).
// 
// Shared memory layout (reused across phases):
//   Phase 1-2: sh_A[n] = h_prev, sh_B[n] = mu_i (per-particle AR(1) mean)
//   Phase 3-4: sh_A[n] = h,      sh_B[n] = grad
// 
// Total shared: 2n floats (same as individual kernels)
// =============================================================================

__global__ void svpf_fused_gradient_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ log_w,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_bandwidth,
    int y_idx,
    float rho,
    float sigma_z,
    float mu,
    float beta,
    float nu,
    float student_t_const,
    float lik_offset,
    float gamma,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int stein_sign_mode,
    bool use_exact_gradient,
    bool use_fan_mode,
    int use_student_t_state,
    float nu_state,
    int n
) {
    extern __shared__ float smem[];
    float* sh_A = smem;       // Phase 1-2: h_prev, Phase 3-4: h
    float* sh_B = smem + n;   // Phase 1-2: mu_i,   Phase 3-4: grad
    
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    
    float y_prev = (y_idx > 0) ? d_y[y_idx - 1] : 0.0f;
    float y_t = d_y[y_idx];
    
    // =========================================================================
    // PHASE 1: Load h_prev, compute mu_i (AR(1) conditional mean per particle)
    // =========================================================================
    for (int k = tid; k < n; k += blockDim.x) {
        float hp = h_prev[k];
        sh_A[k] = hp;  // h_prev
        
        float vol_prev_k = __expf(hp * 0.5f);
        float leverage_k = gamma * y_prev / (vol_prev_k + 1e-8f);
        sh_B[k] = mu + rho * (hp - mu) + leverage_k;  // mu_i
    }
    __syncthreads();
    
    // =========================================================================
    // PHASE 2: Compute gradient for particle j
    // =========================================================================
    float grad_j = 0.0f;
    float h_j = 0.0f;
    
    if (j < n) {
        h_j = h[j];
        
        // ===== PRIOR GRADIENT =====
        float sigma_z_sq = sigma_z * sigma_z;
        float grad_prior;
        
        if (use_student_t_state) {
            // Student-t prior: bounded gradient
            float nu_sigma_sq = nu_state * sigma_z_sq;
            float nu_plus_1 = nu_state + 1.0f;
            float half_nu_plus_1 = 0.5f * nu_plus_1;
            
            float log_r_max = -1e10f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_B[i];  // h_j - mu_i
                float diff_sq = diff * diff;
                float log_r_i = -half_nu_plus_1 * __logf(1.0f + diff_sq / nu_sigma_sq);
                log_r_max = fmaxf(log_r_max, log_r_i);
            }
            
            float sum_r = 0.0f;
            float weighted_grad = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_B[i];
                float diff_sq = diff * diff;
                float denom = nu_sigma_sq + diff_sq;
                
                float log_r_i = -half_nu_plus_1 * __logf(1.0f + diff_sq / nu_sigma_sq);
                float r_i = __expf(log_r_i - log_r_max);
                sum_r += r_i;
                
                weighted_grad -= r_i * nu_plus_1 * diff / denom;
            }
            grad_prior = weighted_grad / (sum_r + 1e-8f);
            
        } else {
            // Gaussian prior
            float inv_2sigma_sq = 0.5f / sigma_z_sq;
            float inv_sigma_sq = 1.0f / sigma_z_sq;
            
            float log_r_max = -1e10f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_B[i];
                float log_r_i = -diff * diff * inv_2sigma_sq;
                log_r_max = fmaxf(log_r_max, log_r_i);
            }
            
            float sum_r = 0.0f;
            float weighted_grad = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_B[i];
                float log_r_i = -diff * diff * inv_2sigma_sq;
                float r_i = __expf(log_r_i - log_r_max);
                sum_r += r_i;
                weighted_grad -= r_i * diff * inv_sigma_sq;
            }
            grad_prior = weighted_grad / (sum_r + 1e-8f);
        }
        
        // ===== LIKELIHOOD GRADIENT =====
        float vol = safe_exp(h_j);
        float y_sq = y_t * y_t;
        float scaled_y_sq = y_sq / (vol + 1e-8f);
        float A = scaled_y_sq / nu;
        float one_plus_A = 1.0f + A;
        
        // Log weights
        if (use_fan_mode) {
            log_w[j] = 0.0f;
        } else {
            log_w[j] = student_t_const - 0.5f * h_j
                     - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));
        }
        
        float grad_lik;
        if (use_exact_gradient) {
            float raw_grad = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A;
            grad_lik = raw_grad - lik_offset;
        } else {
            float log_y2 = __logf(y_sq + 1e-10f);
            float R_noise = 1.4f;
            grad_lik = (log_y2 - h_j + lik_offset) / R_noise;
        }
        
        // ===== COMBINE =====
        float effective_beta = use_fan_mode ? 1.0f : beta;
        grad_j = grad_prior + effective_beta * grad_lik;
        grad_j = fminf(fmaxf(grad_j, -10.0f), 10.0f);
    }
    __syncthreads();
    
    // =========================================================================
    // PHASE 3: Reload shared memory with h and grad for Stein operator
    // =========================================================================
    // Each thread writes its own gradient to shared (no race - different indices)
    // But we need to load ALL h values, so do cooperative load first
    
    for (int k = tid; k < n; k += blockDim.x) {
        sh_A[k] = h[k];  // Current particle positions
    }
    __syncthreads();
    
    // Now each valid thread writes its gradient
    if (j < n) {
        sh_B[j] = grad_j;
    }
    __syncthreads();
    
    // =========================================================================
    // PHASE 4: Stein operator + Transport
    // =========================================================================
    if (j >= n) return;
    
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n = 1.0f / (float)n;
    
    // Sign multiplier: -1 for legacy (attraction), +1 for paper (repulsion)
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    // ===== STEIN OPERATOR with IMQ Kernel =====
    float h_i = sh_A[j];  // Note: j is our particle index
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    #pragma unroll 8
    for (int k = 0; k < n; k++) {
        float diff = h_i - sh_A[k];
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        k_sum += K * sh_B[k];  // K * grad[k]
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
    }
    
    float phi_i = (k_sum + gk_sum) * inv_n;
    
    // ===== RMSPROP =====
    float v_prev = v_rmsprop[j];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[j] = v_new;
    
    // ===== TRANSPORT =====
    float effective_step = step_size * beta_factor;
    float precond = rsqrtf(v_new + epsilon);
    float drift = effective_step * phi_i * precond;
    
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[j]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }
    
    h[j] = clamp_logvol(h_i + drift + diffusion);
}

// =============================================================================
// PARALLEL STEIN KERNELS - STREAMING + WARP REDUCTION
// =============================================================================
// Optimized for N >= 1024. Uses Warp Shuffles for reduction.
// CRITICAL FIX: Inverts Hessian (1/H) to match Sequential behavior.

#define STEIN_PARALLEL_BLOCK_SIZE 256

// Warp Reduction Helpers
__device__ __forceinline__ void warp_reduce_2(float* v1, float* v2) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        *v1 += __shfl_down_sync(0xffffffff, *v1, offset);
        *v2 += __shfl_down_sync(0xffffffff, *v2, offset);
    }
}

__device__ __forceinline__ void warp_reduce_3(float* v1, float* v2, float* v3) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        *v1 += __shfl_down_sync(0xffffffff, *v1, offset);
        *v2 += __shfl_down_sync(0xffffffff, *v2, offset);
        *v3 += __shfl_down_sync(0xffffffff, *v3, offset);
    }
}

__device__ __forceinline__ void warp_reduce_4(float* v1, float* v2, float* v3, float* v4) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        *v1 += __shfl_down_sync(0xffffffff, *v1, offset);
        *v2 += __shfl_down_sync(0xffffffff, *v2, offset);
        *v3 += __shfl_down_sync(0xffffffff, *v3, offset);
        *v4 += __shfl_down_sync(0xffffffff, *v4, offset);
    }
}

// -----------------------------------------------------------------------------
// Parallel Standard Stein Operator (No Newton, No KSD)
// FIXED: Fused accumulation to match sequential floating-point behavior
// -----------------------------------------------------------------------------
__global__ void svpf_stein_operator_parallel_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi_out,
    const float* __restrict__ d_bandwidth,
    int stein_sign_mode,
    int n
) {
    int i = blockIdx.x;
    if (i >= n) return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float h_i = h[i];
    float bw = *d_bandwidth;
    float inv_bw_sq = 1.0f / (bw * bw);
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    float inv_n = 1.0f / (float)n;

    // FUSED: Single accumulator to prevent catastrophic cancellation
    float combined_sum = 0.0f;

    for (int j = tid; j < n; j += blockDim.x) {
        float h_j = h[j];
        float grad_j = grad[j];
        
        float diff = h_i - h_j;
        float dist_sq = diff * diff * inv_bw_sq;
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;

        float grad_term = K * grad_j;
        float repulsive_term = sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        
        combined_sum += (grad_term + repulsive_term);
    }

    __syncthreads();

    // Warp reduction
    warp_reduce_2(&combined_sum, &combined_sum); // Just reduce same var twice (hack for size 1)

    // Explicit Zero-Init for safety
    __shared__ float s_combined[8];
    if (tid < 8) s_combined[tid] = 0.0f;
    __syncthreads();

    if (lane == 0) { s_combined[wid] = combined_sum; }
    __syncthreads();

    if (tid < 8) combined_sum = s_combined[tid];
    else combined_sum = 0.0f;

    if (wid == 0) warp_reduce_2(&combined_sum, &combined_sum);

    if (tid == 0) {
        phi_out[i] = combined_sum * inv_n;
    }
}

// -----------------------------------------------------------------------------
// Parallel Standard Stein Operator + KSD (No Newton)
// FIXED: Fused phi accumulation + KSD
// -----------------------------------------------------------------------------
__global__ void svpf_stein_operator_parallel_ksd_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi_out,
    float* __restrict__ ksd_partial,
    const float* __restrict__ d_bandwidth,
    int stein_sign_mode,
    int n
) {
    int i = blockIdx.x;
    if (i >= n) return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float h_i = h[i];
    float grad_i = grad[i];
    float bw = *d_bandwidth;
    float inv_bw_sq = 1.0f / (bw * bw);
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    float inv_n = 1.0f / (float)n;

    float combined_sum = 0.0f;
    float ksd_sum = 0.0f;

    for (int j = tid; j < n; j += blockDim.x) {
        float h_j = h[j];
        float grad_j = grad[j];
        
        float diff = h_i - h_j;
        float dist_sq = diff * diff * inv_bw_sq;
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        float K_cube = K_sq * K;

        float grad_term = K * grad_j;
        float repulsive_term = sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        combined_sum += (grad_term + repulsive_term);

        // KSD
        float grad_k_i = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_k_j = -grad_k_i;
        float grad2_k = -2.0f * inv_bw_sq * K_sq + 8.0f * dist_sq * inv_bw_sq * K_cube;
        ksd_sum += K * grad_i * grad_j + grad_i * grad_k_j + grad_j * grad_k_i + grad2_k;
    }

    __syncthreads();

    warp_reduce_2(&combined_sum, &ksd_sum);

    __shared__ float s_combined[8], s_ksd[8];
    if (tid < 8) { s_combined[tid] = 0.0f; s_ksd[tid] = 0.0f; }
    __syncthreads();

    if (lane == 0) { s_combined[wid] = combined_sum; s_ksd[wid] = ksd_sum; }
    __syncthreads();

    if (tid < 8) { combined_sum = s_combined[tid]; ksd_sum = s_ksd[tid]; }
    else { combined_sum = ksd_sum = 0.0f; }

    if (wid == 0) warp_reduce_2(&combined_sum, &ksd_sum);

    if (tid == 0) {
        phi_out[i] = combined_sum * inv_n;
        ksd_partial[i] = ksd_sum;
    }
}

// -----------------------------------------------------------------------------
// Parallel Full Newton Stein Operator (No KSD)
// FIXED: Fused accumulation and zero-init shared memory
// -----------------------------------------------------------------------------
__global__ void svpf_stein_operator_parallel_full_newton_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ phi_out,
    const float* __restrict__ d_bandwidth,
    int stein_sign_mode,
    int n
) {
    int i = blockIdx.x;
    if (i >= n) return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float h_i = h[i];
    float bw = *d_bandwidth;
    float inv_bw_sq = 1.0f / (bw * bw);
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    float inv_n = 1.0f / (float)n;

    float combined_sum = 0.0f;
    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;

    for (int j = tid; j < n; j += blockDim.x) {
        float h_j = h[j];
        float grad_j = grad[j];
        float hess_j = local_hessian[j];

        float diff = h_i - h_j;
        float dist_sq = diff * diff * inv_bw_sq;
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;

        // Newton weighting
        float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        H_weighted += hess_j * K + Nk;
        K_sum_norm += K;

        // FUSED phi terms
        float grad_term = K * grad_j;
        float repulsive_term = sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        combined_sum += (grad_term + repulsive_term);
    }

    __syncthreads();

    warp_reduce_3(&combined_sum, &H_weighted, &K_sum_norm);

    __shared__ float s_combined[8], s_h[8], s_kn[8];
    if (tid < 8) { 
        s_combined[tid] = 0.0f; 
        s_h[tid] = 0.0f; 
        s_kn[tid] = 0.0f; 
    }
    __syncthreads();

    if (lane == 0) {
        s_combined[wid] = combined_sum;
        s_h[wid] = H_weighted; 
        s_kn[wid] = K_sum_norm;
    }
    __syncthreads();

    // Deterministic summation in thread 0 (no warp shuffle tree)
    // Fixes numerical instability from reduction order changes
    if (tid == 0) {
        float sum_c = 0.0f;
        float sum_h = 0.0f;
        float sum_k = 0.0f;

        #pragma unroll
        for (int w = 0; w < (STEIN_PARALLEL_BLOCK_SIZE / 32); ++w) {
            sum_c += s_combined[w];
            sum_h += s_h[w];
            sum_k += s_kn[w];
        }

        float H_val = sum_h / fmaxf(sum_k, 1e-6f);
        H_val = fminf(fmaxf(H_val, 0.1f), 100.0f);
        float inv_H = 1.0f / H_val;

        phi_out[i] = sum_c * (1.0f / (float)n) * inv_H * 0.7f;
    }
}

// -----------------------------------------------------------------------------
// Parallel Full Newton Stein Operator + KSD
// FIXED: Zero-init shared memory and correct reduction
// -----------------------------------------------------------------------------
__global__ void svpf_stein_operator_parallel_full_newton_ksd_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ phi_out,
    float* __restrict__ ksd_partial,
    const float* __restrict__ d_bandwidth,
    int stein_sign_mode,
    int n
) {
    int i = blockIdx.x;
    if (i >= n) return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float h_i = h[i];
    float grad_i = grad[i];
    float bw = *d_bandwidth;
    float inv_bw_sq = 1.0f / (bw * bw);
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    float inv_n = 1.0f / (float)n;

    float combined_sum = 0.0f;
    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;
    float ksd_sum = 0.0f;

    for (int j = tid; j < n; j += blockDim.x) {
        float h_j = h[j];
        float grad_j = grad[j];
        float hess_j = local_hessian[j];

        float diff = h_i - h_j;
        float dist_sq = diff * diff * inv_bw_sq;
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;

        // Newton weighting
        float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        H_weighted += hess_j * K + Nk;
        K_sum_norm += K;

        // FUSED phi terms
        float grad_term = K * grad_j;
        float repulsive_term = sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        combined_sum += (grad_term + repulsive_term);

        // KSD
        float K_cube = K_sq * K;
        float grad_k_i = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_k_j = -grad_k_i;
        float grad2_k = -2.0f * inv_bw_sq * K_sq + 8.0f * dist_sq * inv_bw_sq * K_cube;
        ksd_sum += K * grad_i * grad_j + grad_i * grad_k_j + grad_j * grad_k_i + grad2_k;
    }

    __syncthreads();

    warp_reduce_4(&combined_sum, &H_weighted, &K_sum_norm, &ksd_sum);

    __shared__ float s_combined[8], s_h[8], s_kn[8], s_ksd[8];
    if (tid < 8) {
        s_combined[tid] = 0.0f;
        s_h[tid] = 0.0f;
        s_kn[tid] = 0.0f;
        s_ksd[tid] = 0.0f;
    }
    __syncthreads();

    if (lane == 0) {
        s_combined[wid] = combined_sum;
        s_h[wid] = H_weighted; 
        s_kn[wid] = K_sum_norm; 
        s_ksd[wid] = ksd_sum;
    }
    __syncthreads();

    // Deterministic summation in thread 0 (no warp shuffle tree)
    if (tid == 0) {
        float sum_c = 0.0f;
        float sum_h = 0.0f;
        float sum_k = 0.0f;
        float sum_ksd = 0.0f;

        #pragma unroll
        for (int w = 0; w < (STEIN_PARALLEL_BLOCK_SIZE / 32); ++w) {
            sum_c += s_combined[w];
            sum_h += s_h[w];
            sum_k += s_kn[w];
            sum_ksd += s_ksd[w];
        }

        float H_val = sum_h / fmaxf(sum_k, 1e-6f);
        H_val = fminf(fmaxf(H_val, 0.1f), 100.0f);
        float inv_H = 1.0f / H_val;

        phi_out[i] = sum_c * (1.0f / (float)n) * inv_H * 0.7f;
        ksd_partial[i] = sum_ksd;
    }
}

// -----------------------------------------------------------------------------
// Separate Transport Kernel
// -----------------------------------------------------------------------------
__global__ void svpf_transport_separate_kernel(
    float* __restrict__ h,
    const float* __restrict__ phi,
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
    
    float h_i = h[i];
    float phi_i = phi[i];
    
    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;
    
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

// -----------------------------------------------------------------------------
// KSD Final Reduction
// -----------------------------------------------------------------------------
__global__ void svpf_ksd_reduce_parallel_kernel(
    const float* __restrict__ ksd_partial,
    float* __restrict__ ksd_out,
    int n
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = tid; i < n; i += blockDim.x) {
        sum += ksd_partial[i];
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
        float inv_n_sq = 1.0f / ((float)n * (float)n);
        *ksd_out = sqrtf(fabsf(sdata[0] * inv_n_sq));
    }
}
