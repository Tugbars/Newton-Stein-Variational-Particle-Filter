/**
 * @file svpf_opt_kernels.cu
 * @brief CUDA kernel definitions for optimized SVPF paths
 * 
 * Fused kernels for low-latency execution:
 * - svpf_fused_gradient_kernel (prior + likelihood + combine + hessian)
 * - svpf_fused_stein_transport_full_newton_kernel (stein + transport)
 * - svpf_fused_stein_transport_full_newton_ksd_kernel (stein + transport + KSD)
 * - svpf_fused_bandwidth_kernel (bandwidth + adaptive)
 * - svpf_fused_outputs_kernel (logsumexp + vol + h_mean)
 * 
 * CHANGE: stein_sign_mode == 0 disables repulsive kernel gradient entirely.
 *         Only kernel-smoothed score ascent (attractive term) is computed.
 *         See: SVPF v6 experiments — repulsion is harmful in sequential filtering.
 */

#include "svpf_kernels.cuh"
#include "svpf_heun_kernels.cuh"
#include <cuda_pipeline.h>
#include <stdio.h>

// =============================================================================
// Basic Utility Kernels (definitions)
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
        h[idx] = clamp_logvol(mu + stationary_std * z);
    }
}

__global__ void svpf_copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
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
    int t,
    float rho,
    float sigma_z, float mu, float gamma,
    float jump_prob, float jump_scale,
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
        
        h[i] = clamp_logvol(mean_proposal + sigma_z * scale * z);  // +z
    }
    
    // Process particle j (with -z)
    {
        
        
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
        
        h[j] = clamp_logvol(mean_proposal + sigma_z * scale * (-z));  // -z
    }
}

// =============================================================================
// Guide Kernels
// =============================================================================

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
        float nu_sigma_sq = nu_state * sigma_z_sq;
        float nu_plus_1 = nu_state + 1.0f;
        float half_nu_plus_1 = 0.5f * nu_plus_1;
        
        float log_r_max = -1e10f;
        #pragma unroll 8
        for (int i = 0; i < n; i++) {
            float diff = h_j - sh_mu_i[i];
            float diff_sq = diff * diff;
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
            
            weighted_grad -= r_i * nu_plus_1 * diff / denom;
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
// FUSED: Stein + Transport (Full Newton)
// =============================================================================
// CHANGE: stein_sign_mode == 0 disables repulsive term (gk_sum).
//         Only kernel-smoothed score ascent is computed.

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
    
    // stein_sign_mode: 0 = no repulsion, 1 = positive (default SVGD), -1 = negative
    const bool use_repulsion = (stein_sign_mode != 0);
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
        
        // Kernel-smoothed target Hessian (Nk kernel geometry term removed —
        // it inflates curvature near clusters via inter-particle distances,
        // creating circular dependency similar to repulsion)
        H_weighted += sh_hess[j] * K;
        K_sum_norm += K;
        
        // Attractive: kernel-smoothed score (always on)
        k_grad_sum += K * sh_grad[j];
        
        // Repulsive: kernel gradient (disabled when stein_sign_mode == 0)
        if (use_repulsion) {
            gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        }
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
// CHANGE: stein_sign_mode == 0 disables repulsive term (gk_sum).
//         KSD diagnostic is unaffected — it uses the Stein kernel independently.

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
    
    // stein_sign_mode: 0 = no repulsion, 1 = positive (default SVGD), -1 = negative
    const bool use_repulsion = (stein_sign_mode != 0);
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
        
        // ----- Kernel-smoothed target Hessian (Nk removed) -----
        H_weighted += sh_hess[j] * K;
        K_sum_norm += K;
        
        // Attractive: kernel-smoothed score (always on)
        k_grad_sum += K * s_j;
        
        // Repulsive: kernel gradient (disabled when stein_sign_mode == 0)
        if (use_repulsion) {
            gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        }
        
        // ----- KSD Stein kernel (independent of repulsion setting) -----
        float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_y_k = -grad_x_k;
        float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
        float u_ij = K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
        ksd_sum += u_ij;
    }
    
    // Store partial KSD sum
    d_ksd_partial[i] = ksd_sum;
    
    // Kernel-smoothed target Hessian preconditioning
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
// KSD Reduction Kernel
// =============================================================================

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
        *d_ksd = sqrtf(fmaxf(ksd_sq, 0.0f));
    }
}

// =============================================================================
// FUSED: Outputs
// =============================================================================

__global__ void svpf_fused_outputs_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    const float* __restrict__ d_bandwidth_in,
    const float* __restrict__ d_ksd_in,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    float* __restrict__ d_output_pack,
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
        
        d_loglik[t_out] = loglik;
        d_vol[t_out] = vol;
        *d_h_mean = h_mean;
        
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
    
    float u = curand_uniform(&rng[i]);
    
    if (u < rejuv_prob) {
        float z = curand_normal(&rng[i]);
        float guide_sample = guide_mean + guide_std * z;
        
        float h_old = h[i];
        float h_new = (1.0f - blend_factor) * h_old + blend_factor * guide_sample;
        
        h[i] = clamp_logvol(h_new);
    }
}
