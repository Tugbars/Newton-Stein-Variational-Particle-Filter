/**
 * @file svpf_opt_kernels.cu
 * @brief CUDA kernel definitions for optimized SVPF paths
 *
 * Fused kernels for low-latency execution:
 * - svpf_fused_gradient_kernel (prior + likelihood + combine + hessian)
 * - svpf_stein_transport_kernel<MODE,KSD> (unified template, replaces 8 variants)
 * - svpf_fused_bandwidth_kernel (bandwidth + adaptive)
 * - svpf_fused_outputs_kernel (logsumexp + vol + h_mean)
 *
 * Optimization changelog (v2):
 * 1. Unified Stein template: 8 kernels → 1 template × 6 instantiations
 *    - SteinMode::STANDARD / NEWTON / FULL_NEWTON × ComputeKSD true/false
 * 2. Inner loop CSE: ~35% fewer FLOPs in hot O(N²) loop
 *    - Hoisted hc = 2/bw² · K², reused for repulsion, Nk, hess_xy
 *    - Factored repulse = hc·diff, shared between Stein operator and KSD
 *    - KSD: (s_i - s_j)·repulse replaces separate grad_x_k/grad_y_k terms
 * 3. Gradient kernel: removed dead sh_h_prev (saves N×4 bytes smem)
 * 4. Uniform __pipeline_memcpy_async across all Stein modes
 * 5. Antithetic kernel: extracted shared __device__ helper, eliminated duplication
 * 6. Backward-compatible wrapper aliases at bottom of file
 */

#include "svpf_kernels.cuh"
#include "svpf_heun_kernels.cuh"
#include <cuda_pipeline.h>
#include <stdio.h>

// =============================================================================
// Stein Mode Enum (compile-time dispatch)
// =============================================================================

enum class SteinMode {
    STANDARD,      // Uses raw grad for score, no Hessian weighting
    NEWTON,        // Uses precond_grad for score, inv_hessian weights repulsion
    FULL_NEWTON    // Uses raw grad for score, kernel-weighted Hessian preconditioner
};

// =============================================================================
// Basic Utility Kernels
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

// NOTE: Consider replacing with cudaMemcpyAsync on the host side.
__global__ void svpf_copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// Guided Prediction: shared device helper (eliminates antithetic duplication)
// =============================================================================

__device__ __forceinline__ float predict_guided_particle(
    float h_i,
    float h_prev_i,
    float h_bar,
    float y_prev,
    float y_curr,
    float noise,
    float selector,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float jump_prob, float jump_scale,
    float delta_rho, float delta_sigma,
    float alpha_base, float alpha_shock,
    float innovation_threshold,
    float implied_offset
) {
    float dev = h_i - h_bar;
    float rho_adjust = delta_rho * tanhf(dev);
    float sigma_scale = 1.0f + delta_sigma * fabsf(dev);

    float scale = (selector < jump_prob) ? jump_scale : 1.0f;

    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    float sigma_local = sigma_z * sigma_scale;

    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    float mean_prior = mu + rho * (h_i - mu) + leverage;

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

    return clamp_logvol(mean_proposal + sigma_local * scale * noise);
}

// =============================================================================
// Predict Kernels
// =============================================================================

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

    float noise;
    if (use_student_t_state) {
        noise = sample_student_t(&rng[i], nu_state);
    } else {
        noise = curand_normal(&rng[i]);
    }
    float selector = curand_uniform(&rng[i]);

    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float y_curr = d_y[t];
    float h_bar = *d_h_mean;

    h[i] = predict_guided_particle(
        h_i, h_prev_i, h_bar, y_prev, y_curr, noise, selector,
        rho_up, rho_down, sigma_z, mu, gamma,
        jump_prob, jump_scale, delta_rho, delta_sigma,
        alpha_base, alpha_shock, innovation_threshold, implied_offset
    );
}

// =============================================================================
// ANTITHETIC SAMPLING VERSION
// =============================================================================
// Each thread handles TWO particles: i and i + n/2
// They share the same z, but particle i+n/2 uses -z
// Launch with n/2 threads!
// v2: Uses shared __device__ helper — eliminated ~80 lines of duplication.

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

    int j = i + half_n;

    float h_i = h[i];
    float h_j = h[j];
    float h_prev_i = h_prev[i];
    float h_prev_j = h_prev[j];

    h_prev[i] = h_i;
    h_prev[j] = h_j;

    // Generate ONE random sample
    float z;
    if (use_student_t_state) {
        z = sample_student_t(&rng[i], nu_state);
    } else {
        z = curand_normal(&rng[i]);
    }
    float selector = curand_uniform(&rng[i]);

    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float y_curr = d_y[t];
    float h_bar = *d_h_mean;

    // Particle i with +z, particle j with -z
    h[i] = predict_guided_particle(
        h_i, h_prev_i, h_bar, y_prev, y_curr, z, selector,
        rho_up, rho_down, sigma_z, mu, gamma,
        jump_prob, jump_scale, delta_rho, delta_sigma,
        alpha_base, alpha_shock, innovation_threshold, implied_offset
    );

    h[j] = predict_guided_particle(
        h_j, h_prev_j, h_bar, y_prev, y_curr, -z, selector,
        rho_up, rho_down, sigma_z, mu, gamma,
        jump_prob, jump_scale, delta_rho, delta_sigma,
        alpha_base, alpha_shock, innovation_threshold, implied_offset
    );
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
    float deviation = h[i] - current_mean;
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
    float deviation = h[i] - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    h[i] = clamp_logvol(new_mean + deviation);
}

// =============================================================================
// FUSED: Gradient Pipeline
// =============================================================================
// v2: Removed dead sh_h_prev array. Was loaded to smem but never read after —
//     only sh_mu_i was accessed in inner loops. Saves N×4 bytes of shared memory.
//     Also hoisted inv_nu_sigma_sq for Student-t branch.

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
    // v2: Only N floats of smem needed (was 2N — sh_h_prev was dead)
    extern __shared__ float smem[];
    float* sh_mu_i = smem;

    float y_prev = (y_idx > 0) ? d_y[y_idx - 1] : 0.0f;

    // Compute transition means directly — no intermediate sh_h_prev needed
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float hp = h_prev[k];
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
        float inv_nu_sigma_sq = 1.0f / nu_sigma_sq;  // v2: hoisted

        float log_r_max = -1e10f;
        #pragma unroll 8
        for (int i = 0; i < n; i++) {
            float diff = h_j - sh_mu_i[i];
            float ratio = diff * diff * inv_nu_sigma_sq;
            float log_r_i = -half_nu_plus_1 * __logf(1.0f + ratio);
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

            float ratio = diff_sq * inv_nu_sigma_sq;
            float log_r_i = -half_nu_plus_1 * __logf(1.0f + ratio);
            float r_i = __expf(log_r_i - log_r_max);
            sum_r += r_i;

            weighted_grad -= r_i * nu_plus_1 * diff / denom;
            float hess_i = -nu_plus_1 * (nu_sigma_sq - diff_sq) / (denom * denom);
            weighted_hess += r_i * hess_i;
        }
        grad_prior = weighted_grad / (sum_r + 1e-8f);
        hess_prior = weighted_hess / (sum_r + 1e-8f);

    } else {
        // Gaussian prior
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
// UNIFIED STEIN + TRANSPORT TEMPLATE
// =============================================================================
//
// Replaces 8 separate kernels:
//   svpf_fused_stein_transport_kernel              → <STANDARD, false>
//   svpf_fused_stein_transport_ksd_kernel           → <STANDARD, true>
//   svpf_fused_stein_transport_newton_kernel         → <NEWTON, false>
//   svpf_fused_stein_transport_newton_ksd_kernel     → <NEWTON, true>
//   svpf_fused_stein_transport_full_newton_kernel    → <FULL_NEWTON, false>
//   svpf_fused_stein_transport_full_newton_ksd_kernel → <FULL_NEWTON, true>
//
// Inner loop CSE savings (~35% fewer FLOPs for FULL_NEWTON+KSD):
//   - hc = 2·inv_bw_sq · K²  (master coefficient, computed once, used 3×)
//   - repulse = hc · diff     (computed once, shared by gk_sum and KSD)
//   - KSD: (s_i - s_j)·repulse replaces 2 separate grad_x/grad_y products
//
// Shared memory layout:
//   STANDARD:    sh_h[n], sh_score[n]             → 2n floats
//   NEWTON:      sh_h[n], sh_score[n], sh_aux[n]  → 3n floats (aux = 1/curvature)
//   FULL_NEWTON: sh_h[n], sh_score[n], sh_aux[n]  → 3n floats (aux = hessian)
//
// Launch config:
//   grid:  (n + BLOCK - 1) / BLOCK
//   block: min(n, 256)
//   smem:  (MODE == STANDARD ? 2 : 3) * n * sizeof(float)

template <SteinMode MODE, bool COMPUTE_KSD>
__global__ void svpf_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ score,       // grad (STD/FN) or precond_grad (NEWTON)
    const float* __restrict__ curvature,   // NULL (STD), inv_hessian (N), local_hessian (FN)
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float* __restrict__ d_ksd_partial,     // NULL if !COMPUTE_KSD
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h     = smem;
    float* sh_score = smem + n;
    float* sh_aux   = smem + 2 * n;  // Only used for NEWTON/FULL_NEWTON

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Load shared memory with async pipeline ---
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        __pipeline_memcpy_async(&sh_h[k], &h[k], sizeof(float));
        __pipeline_memcpy_async(&sh_score[k], &score[k], sizeof(float));
    }
    if constexpr (MODE == SteinMode::FULL_NEWTON) {
        // Full Newton: load hessian directly
        for (int k = threadIdx.x; k < n; k += blockDim.x) {
            __pipeline_memcpy_async(&sh_aux[k], &curvature[k], sizeof(float));
        }
    }
    __pipeline_commit();

    // --- Independent work while loads are in flight ---
    float global_bw = *d_bandwidth;
    float bw_sq = global_bw * global_bw;
    float inv_bw_sq = 1.0f / bw_sq;
    float two_inv_bw_sq = 2.0f * inv_bw_sq;  // v2: hoisted loop-invariant
    float inv_n = 1.0f / (float)n;
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    float effective_step = step_size * beta_factor;
    float v_prev = (i < n) ? v_rmsprop[i] : 0.0f;

    // --- Wait for async loads ---
    __pipeline_wait_prior(0);
    __syncthreads();

    // --- Newton: compute reciprocal curvature (can't async a division) ---
    if constexpr (MODE == SteinMode::NEWTON) {
        for (int k = threadIdx.x; k < n; k += blockDim.x) {
            sh_aux[k] = 1.0f / curvature[k];
        }
        __syncthreads();
    }

    if (i >= n) return;

    float h_i = sh_h[i];

    // s_i only needed for KSD computation
    float s_i = 0.0f;
    if constexpr (COMPUTE_KSD) {
        s_i = sh_score[i];
    }

    // --- Accumulators ---
    float k_sum = 0.0f;      // Score attraction: Σ K(x_i,x_j) · s(x_j)
    float gk_sum = 0.0f;     // Repulsion: Σ ∇_x K(x_i,x_j) [× aux_j for NEWTON]

    // FULL_NEWTON only: kernel-weighted curvature
    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;

    // KSD only: Stein kernel sum for convergence diagnostic
    float ksd_sum = 0.0f;

    // =================================================================
    // HOT INNER LOOP — O(N²), CSE-optimized
    //
    // Per-iteration FLOP comparison (FULL_NEWTON + KSD):
    //   Original: ~35 FLOPs    Optimized: ~23 FLOPs (~35% reduction)
    //
    // Key CSE:
    //   hc = 2/(bw²)·K²  → reused for repulse, Nk, hess_xy (was 3× recomputed)
    //   repulse = hc·diff → shared by gk_sum and KSD       (was 2× recomputed)
    //   (s_i-s_j)·repulse → replaces s_i·grad_y + s_j·grad_x (saves 1 mul+add)
    // =================================================================
    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        float h_j = sh_h[j];
        float s_j = sh_score[j];
        float diff = h_i - h_j;

        // --- Core IMQ kernel: K(x,y) = (1 + ||x-y||²/bw²)^(-1) ---
        float d_bw = diff * inv_bw_sq;          // diff / bw²
        float dist_sq = diff * d_bw;             // diff² / bw²
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;

        // v2: Master coefficient — computed once, drives 3 downstream terms
        float hc = two_inv_bw_sq * K_sq;         // 2/(bw²) · K²

        // v2: Repulsion vector — computed once, used by Stein + KSD
        float repulse = hc * diff;                // 2·diff/(bw²) · K²

        // --- Score attraction (identical all modes) ---
        k_sum += K * s_j;

        // --- Repulsion (mode-dependent weighting) ---
        if constexpr (MODE == SteinMode::NEWTON) {
            gk_sum += sign_mult * repulse * sh_aux[j];  // weighted by H⁻¹_j
        } else {
            gk_sum += sign_mult * repulse;
        }

        // --- Full Newton: accumulate kernel-weighted curvature ---
        if constexpr (MODE == SteinMode::FULL_NEWTON) {
            // Nk = hc · |3·dist_sq - 1|  (kernel Hessian trace contribution)
            H_weighted += sh_aux[j] * K + hc * fabsf(3.0f * dist_sq - 1.0f);
            K_sum_norm += K;
        }

        // --- KSD Stein kernel u_p(x_i, x_j) ---
        if constexpr (COMPUTE_KSD) {
            // v2: Algebraic simplification
            //   Original: s_i·grad_y_k + s_j·grad_x_k
            //           = s_i·(+repulse) + s_j·(-repulse)
            //           = (s_i - s_j) · repulse
            //   Saves: 1 multiply + 1 add
            float hess_xy = hc * (4.0f * dist_sq * K - 1.0f);
            ksd_sum += K * s_i * s_j + (s_i - s_j) * repulse + hess_xy;
        }
    }
    // =================================================================
    // END HOT INNER LOOP
    // =================================================================

    // --- Compute Stein operator phi(x_i) ---
    float phi_i;
    if constexpr (MODE == SteinMode::FULL_NEWTON) {
        // Detommaso et al. (2018): kernel-weighted Newton preconditioning
        H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
        H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);
        float inv_H_i = 1.0f / H_weighted;
        phi_i = (k_sum + gk_sum) * inv_n * inv_H_i * 0.7f;
    } else {
        phi_i = (k_sum + gk_sum) * inv_n;
    }

    // --- Store KSD partial sum for reduction ---
    if constexpr (COMPUTE_KSD) {
        d_ksd_partial[i] = ksd_sum;
    }

    // --- RMSProp adaptive learning rate ---
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;

    // --- SVGD Transport step ---
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
// Explicit Template Instantiations (6 combinations)
// =============================================================================
// These generate actual symbol definitions so the linker can find them.
// The compiler optimizes away dead branches in each instantiation.

// __restrict__ qualifiers must match the template definition exactly.
// nvcc treats float* __restrict__ as a distinct type from float*.

#define SVPF_STEIN_INSTANTIATE(MODE, KSD) \
    template __global__ void svpf_stein_transport_kernel<MODE, KSD>( \
        float* __restrict__, const float* __restrict__, const float* __restrict__, \
        float* __restrict__, curandStatePhilox4_32_10_t* __restrict__, \
        const float* __restrict__, float* __restrict__, \
        float, float, float, float, float, int, int)

SVPF_STEIN_INSTANTIATE(SteinMode::STANDARD,    false);
SVPF_STEIN_INSTANTIATE(SteinMode::STANDARD,    true);
SVPF_STEIN_INSTANTIATE(SteinMode::NEWTON,      false);
SVPF_STEIN_INSTANTIATE(SteinMode::NEWTON,      true);
SVPF_STEIN_INSTANTIATE(SteinMode::FULL_NEWTON, false);
SVPF_STEIN_INSTANTIATE(SteinMode::FULL_NEWTON, true);

#undef SVPF_STEIN_INSTANTIATE

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

// =============================================================================
// HOST-SIDE DISPATCH GUIDE
// =============================================================================
//
// Migration from old kernel names to unified template:
//
// OLD KERNEL                                       → NEW TEMPLATE INSTANTIATION
// ─────────────────────────────────────────────────────────────────────────────
// svpf_fused_stein_transport_kernel                → <STANDARD, false>
// svpf_fused_stein_transport_ksd_kernel            → <STANDARD, true>
// svpf_fused_stein_transport_newton_kernel          → <NEWTON, false>
// svpf_fused_stein_transport_newton_ksd_kernel      → <NEWTON, true>
// svpf_fused_stein_transport_full_newton_kernel     → <FULL_NEWTON, false>
// svpf_fused_stein_transport_full_newton_ksd_kernel → <FULL_NEWTON, true>
//
// Argument mapping (unified signature):
//   score     = grad_combined (STD/FN) or precond_grad (NEWTON)
//   curvature = nullptr (STD), inv_hessian (NEWTON), local_hessian (FULL_NEWTON)
//   d_ksd_partial = actual buffer (KSD=true) or nullptr (KSD=false)
//
// Shared memory:
//   STANDARD:    2 * n * sizeof(float)
//   NEWTON:      3 * n * sizeof(float)
//   FULL_NEWTON: 3 * n * sizeof(float)
//
// Example host dispatch (production config: FULL_NEWTON + KSD):
//
//   constexpr auto kernel_fn = svpf_stein_transport_kernel<SteinMode::FULL_NEWTON, true>;
//   int smem_bytes = 3 * n * sizeof(float);
//   kernel_fn<<<grid, block, smem_bytes, stream>>>(
//       d_h, d_grad_combined, d_local_hessian,
//       d_v_rmsprop, d_rng, d_bandwidth, d_ksd_partial,
//       step_size, beta_factor, temperature,
//       rho_rmsprop, epsilon, stein_sign_mode, n);
//
// For CUDA graph capture, the template instantiation has a stable function
// pointer. Cache it once:
//   void* kernel_ptr = (void*)svpf_stein_transport_kernel<SteinMode::FULL_NEWTON, true>;
