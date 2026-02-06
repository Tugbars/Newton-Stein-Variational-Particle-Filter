/**
 * @file svpf_opt_kernels.cu
 * @brief CUDA kernel definitions for SVPF
 *
 * Production kernel set (full-Newton, adaptive annealing, antithetic, Student-t):
 * - svpf_init_rng_kernel / svpf_init_particles_kernel / svpf_copy_kernel
 * - svpf_predict_guided_antithetic_kernel
 * - svpf_apply_guide_preserving_kernel
 * - svpf_fused_gradient_kernel
 * - svpf_fused_stein_transport_full_newton_kernel      (non-last iterations)
 * - svpf_fused_stein_transport_full_newton_ksd_kernel   (last iteration)
 * - svpf_ksd_reduce_kernel
 * - svpf_fused_bandwidth_kernel
 * - svpf_fused_outputs_kernel
 * - svpf_partial_rejuvenation_kernel
 */

#include "svpf_kernels.cuh"
#include <cuda_pipeline.h>
#include <stdio.h>

// =============================================================================
// Utility Kernels
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
// ANTITHETIC GUIDED PREDICTION
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
    float nu_state,
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

    // Generate ONE random sample (Student-t), use +z and -z
    float z = sample_student_t(&rng[i], nu_state);

    // MIM jump: same decision for both
    float selector = curand_uniform(&rng[i]);
    float scale = (selector < jump_prob) ? jump_scale : 1.0f;

    float y_curr = d_y[t];
    float y_prev_val = (t > 0) ? d_y[t - 1] : 0.0f;
    float log_y2 = __logf(y_curr * y_curr + 1e-10f);

    // --- Helper lambda-style macro for both particles ---
    #define PROCESS_PARTICLE(h_val, h_prev_val, sign_z, out_idx)              \
    {                                                                          \
        float dev = h_val - h_bar;                                             \
        float rho_adjust = delta_rho * tanhf(dev);                             \
        float sigma_scale = 1.0f + delta_sigma * fabsf(dev);                  \
                                                                               \
        float base_rho = (h_val > h_prev_val) ? rho_up : rho_down;            \
        float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);        \
        float sigma_local = sigma_z * sigma_scale;                             \
                                                                               \
        float vol_prev = safe_exp(h_val / 2.0f);                              \
        float leverage = gamma * y_prev_val / (vol_prev + 1e-8f);             \
        float mean_prior = mu + rho * (h_val - mu) + leverage;                \
                                                                               \
        float mean_implied = fmaxf(log_y2 + implied_offset, -5.0f);           \
        float innovation = mean_implied - mean_prior;                          \
        float z_score = innovation / 2.5f;                                     \
                                                                               \
        float activation = 0.0f;                                               \
        if (z_score > innovation_threshold)                                    \
            activation = tanhf(z_score - innovation_threshold);                \
                                                                               \
        float guided_alpha = alpha_base                                        \
                           + (alpha_shock - alpha_base) * activation;          \
        float mean_proposal = (1.0f - guided_alpha) * mean_prior              \
                            + guided_alpha * mean_implied;                     \
                                                                               \
        h[out_idx] = clamp_logvol(                                             \
            mean_proposal + sigma_local * scale * (sign_z));                   \
    }

    PROCESS_PARTICLE(h_i, h_prev_i,  z, i)
    PROCESS_PARTICLE(h_j, h_prev_j, -z, j)

    #undef PROCESS_PARTICLE
}

// =============================================================================
// GUIDE: Variance-preserving shift toward EKF prediction
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
    float deviation = h[i] - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean
                   + guide_strength * guide_mean;

    h[i] = clamp_logvol(new_mean + deviation);
}

// =============================================================================
// FUSED: Gradient Pipeline (Student-t prior + exact likelihood + full Newton)
// =============================================================================

__global__ void svpf_fused_gradient_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_combined,
    float* __restrict__ log_w,
    float* __restrict__ precond_grad,   // Always non-null (full Newton)
    float* __restrict__ inv_hessian,    // Always non-null (full Newton)
    const float* __restrict__ d_y,
    int y_idx,
    float rho,
    float sigma_z,
    float mu,
    float beta,
    float nu,           // Likelihood degrees of freedom
    float student_t_const,
    float lik_offset,
    float gamma,
    float nu_state,     // Prior degrees of freedom
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

    // ===== PRIOR GRADIENT (Student-t) =====
    float sigma_z_sq = sigma_z * sigma_z;
    float nu_sigma_sq = nu_state * sigma_z_sq;
    float nu_plus_1 = nu_state + 1.0f;
    float half_nu_plus_1 = 0.5f * nu_plus_1;

    float log_r_max = -1e10f;
    #pragma unroll 8
    for (int i = 0; i < n; i++) {
        float diff = h_j - sh_mu_i[i];
        float log_r_i = -half_nu_plus_1 * __logf(1.0f + diff * diff / nu_sigma_sq);
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
        weighted_hess += r_i * (-nu_plus_1 * (nu_sigma_sq - diff_sq) / (denom * denom));
    }
    float grad_prior = weighted_grad / (sum_r + 1e-8f);
    float hess_prior = weighted_hess / (sum_r + 1e-8f);

    // ===== LIKELIHOOD GRADIENT (exact, Student-t observation) =====
    float vol = safe_exp(h_j);
    float y_sq = y_t * y_t;
    float scaled_y_sq = y_sq / (vol + 1e-8f);
    float A = scaled_y_sq / nu;
    float one_plus_A = 1.0f + A;

    // Importance weights (hybrid mode)
    log_w[j] = student_t_const - 0.5f * h_j
             - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));

    float grad_lik = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A - lik_offset;

    // ===== COMBINE =====
    float g = grad_prior + beta * grad_lik;
    g = fminf(fmaxf(g, -10.0f), 10.0f);
    grad_combined[j] = g;

    // ===== HESSIAN (always computed for full Newton) =====
    float hess_lik = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
    float curvature = -(hess_lik + hess_prior);
    curvature = fminf(fmaxf(curvature, 0.1f), 100.0f);

    inv_hessian[j] = curvature;
    precond_grad[j] = 0.7f * g / curvature;
}

// =============================================================================
// FUSED: Full Newton Stein Transport (no KSD — non-last iterations)
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
        H_weighted += 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        K_sum_norm += K;

        k_grad_sum += K * sh_grad[j];
        gk_sum += -2.0f * diff * inv_bw_sq * K_sq;  // Legacy sign (attraction)
    }

    H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
    H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);

    float phi_i = (k_grad_sum + gk_sum) * inv_n * (0.7f / H_weighted);

    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;

    float effective_step = step_size * beta_factor;
    float drift = effective_step * phi_i * rsqrtf(v_new + epsilon);

    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }

    h[i] = clamp_logvol(h_i + drift + diffusion);
}

// =============================================================================
// FUSED: Full Newton Stein Transport + KSD (last iteration only)
// =============================================================================

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

        // Full Newton: Hessian weighting
        H_weighted += sh_hess[j] * K;
        H_weighted += 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        K_sum_norm += K;

        k_grad_sum += K * s_j;
        gk_sum += -2.0f * diff * inv_bw_sq * K_sq;  // Legacy sign

        // KSD Stein kernel
        float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_y_k = -grad_x_k;
        float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
        ksd_sum += K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
    }

    d_ksd_partial[i] = ksd_sum;

    H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
    H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);

    float phi_i = (k_grad_sum + gk_sum) * inv_n * (0.7f / H_weighted);

    float v_prev = v_rmsprop[i];
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;

    float effective_step = step_size * beta_factor;
    float drift = effective_step * phi_i * rsqrtf(v_new + epsilon);

    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }

    h[i] = clamp_logvol(h_i + drift + diffusion);
}

// =============================================================================
// KSD Reduction
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
// FUSED: Bandwidth (adaptive with return-volatility scaling)
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
// FUSED: Outputs (logsumexp + vol + h_mean, packed for single D2H)
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
// PARTIAL REJUVENATION (Maken et al. 2022)
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
        h[i] = clamp_logvol((1.0f - blend_factor) * h_old
                           + blend_factor * guide_sample);
    }
}
