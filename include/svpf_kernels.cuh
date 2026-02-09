/**
 * @file svpf_kernels.cuh
 * @brief CUDA kernel declarations for SVPF
 * 
 * Declarations only. Definitions in svpf_opt_kernels.cu
 * 
 * Production kernel set (all dead paths removed):
 *   Predict:   svpf_predict_guided_antithetic_kernel
 *   Guide:     svpf_apply_guide_preserving_kernel
 *   Gradient:  svpf_fused_gradient_kernel
 *   Stein:     svpf_fused_stein_transport_full_newton_kernel
 *              svpf_fused_stein_transport_full_newton_ksd_kernel
 *   KSD:       svpf_ksd_reduce_kernel
 *   Rejuv:     svpf_partial_rejuvenation_kernel
 *   Output:    svpf_fused_bandwidth_kernel, svpf_fused_outputs_kernel
 */

#ifndef SVPF_KERNELS_CUH
#define SVPF_KERNELS_CUH

#include "svpf.cuh"
#include "svpf_common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Configuration Constants
// =============================================================================

#define TILE_J 256
#define SMALL_N_THRESHOLD 4096
#define BANDWIDTH_UPDATE_INTERVAL 5
#define MAX_T_SIZE 10000

// =============================================================================
// Predict Kernel
// =============================================================================

/** @brief Antithetic guided prediction with innovation gating. Launch with n/2 threads. */
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
    int n
);

// =============================================================================
// Guide Kernel
// =============================================================================

/** @brief Variance-preserving guide (shifts mean toward EKF, keeps spread). */
__global__ void svpf_apply_guide_preserving_kernel(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    float guide_mean,
    float guide_strength,
    int n
);

// =============================================================================
// Fused Gradient Kernel
// =============================================================================

/** @brief Fused O(N²) mixture prior + likelihood gradient + optional Hessian. */
__global__ void svpf_fused_gradient_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_combined,
    float* __restrict__ log_w,
    float* __restrict__ precond_grad,
    float* __restrict__ inv_hessian,
    const float* __restrict__ d_y,
    int y_idx,
    float rho, float sigma_z, float mu,
    float beta, float nu, float student_t_const,
    float lik_offset,
    float gamma,
    bool use_exact_gradient,
    bool use_newton,
    bool use_fan_mode,
    int use_student_t_state,
    float nu_state,
    int n
);


// ============================================================================
// DESIGN NOTES — Split-Batch SVGD
// ============================================================================
//
// Problem (Ba et al. ICLR 2022):
//   Standard SVGD's S1 term uses the SAME particles to evaluate the score
//   and to define the transport field. This creates deterministic bias --
//   each particle influences its own update through the kernel-weighted average.
//   With repulsion off, our update is PURE S1, so this is the dominant bias source.
//
// Solution:
//   Even/odd split -- particle i only sees opposite-parity particles as references.
//   No particle ever contributes to its own gradient field. Complete decoupling
//   of "who defines the field" from "who gets updated by it."
//
// Implementation:
//   Non-KSD kernel: inner loop strides by 2, starting at opposite parity.
//     Cost: O(N * N/2) -- half the original inner loop.
//   KSD kernel: single loop over all N, with is_ref branch for transport.
//     KSD diagnostic must see all pairs for accurate measurement.
//     Cost: O(N^2) same as before (KSD only runs on last iteration).
//
// Normalization:
//   phi_i scaled by 1/n_ref (N/2) not 1/N, since reference set is half-sized.
//   Kernel-weighted Hessian average uses K_sum_norm from reference set only.
//
// Config:
//   state->use_split_batch = 1;  // Enable (default)
//   state->use_split_batch = 0;  // Disable (revert to standard SVGD)

// =============================================================================
// Fused Stein + Transport Kernels (Full Newton only)
// =============================================================================

/** @brief Full Newton Stein transport (Detommaso 2018). Production inner loop. */
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
    int use_split_batch,   // NEW: 1 = even/odd split, 0 = all particles
    int n
);

/** @brief Full Newton Stein transport + KSD computation. Final iteration only. */
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
    int use_split_batch,   // NEW: 1 = even/odd split, 0 = all particles
    int n
);

// =============================================================================
// KSD + Rejuvenation
// =============================================================================

/** @brief Reduce partial KSD sums to final scalar KSD value. */
__global__ void svpf_ksd_reduce_kernel(
    const float* __restrict__ d_ksd_partial,
    float* __restrict__ d_ksd,
    int n
);

/** @brief Partial rejuvenation (Maken 2022). Nudges stuck particles toward guide. */
__global__ void svpf_partial_rejuvenation_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_std,
    float rejuv_prob,
    float blend_factor,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    int n
);

// =============================================================================
// Output Kernels
// =============================================================================

/** @brief Fused bandwidth computation with adaptive EMA scaling. */
__global__ void svpf_fused_bandwidth_kernel(
    const float* __restrict__ h,
    const float* __restrict__ d_y,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    int y_idx, float alpha_bw, float alpha_ret, int n
);

/** @brief Fused output: logsumexp weights + vol mean + h mean + D2H packing. */
__global__ void svpf_fused_outputs_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    const float* __restrict__ d_bandwidth_in,
    const float* __restrict__ d_ksd_in,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    float* __restrict__ d_output_pack,
    int t_out, int n
);

// =============================================================================
// Host-side EKF Helper
// =============================================================================

/** @brief Lightweight EKF update for guide density. Called on host before guide kernel. */
static inline void svpf_ekf_update(
    SVPFState* state,
    float y_t,
    const SVPFParams* p
) {
    if (!state->guide_initialized) {
        state->guide_mean = p->mu;
        state->guide_var = p->sigma_z * p->sigma_z / (1.0f - p->rho * p->rho);
        state->guide_initialized = 1;
    }

    float m_pred = p->mu + p->rho * (state->guide_mean - p->mu);
    float P_pred = p->rho * p->rho * state->guide_var + p->sigma_z * p->sigma_z;

    float log_y2 = logf(y_t * y_t + 1e-8f);
    float obs_offset = -state->student_t_implied_offset;
    float obs_var = 4.93f + 2.0f;  // pi²/2 + measurement slack

    float S = P_pred + obs_var;
    float K = P_pred / (S + 1e-8f);

    float y_pred = m_pred + obs_offset;
    float innovation = log_y2 - y_pred;

    state->guide_mean = m_pred + K * innovation;
    state->guide_var = (1.0f - K) * P_pred;
    state->guide_K = K;
}

// =============================================================================
// Host-side Backward Smoothing (RTS-style)
// =============================================================================

#ifndef SVPF_SMOOTH_MAX_LAG
#define SVPF_SMOOTH_MAX_LAG 8
#endif

/** @brief Lightweight RTS backward pass over sliding window. Called after each sync. */
static inline void svpf_smooth_backward(
    SVPFState* state,
    float h_mean_new,
    float h_var_new,
    float y_t,
    const SVPFParams* params
) {
    if (!state->use_smoothing) return;

    int k = state->smooth_lag;
    if (k > SVPF_SMOOTH_MAX_LAG) k = SVPF_SMOOTH_MAX_LAG;
    if (k < 1) k = 1;

    int head = state->smooth_head;
    state->smooth_h_mean[head] = h_mean_new;
    state->smooth_h_var[head] = h_var_new;
    state->smooth_y[head] = y_t;

    state->smooth_head = (head + 1) % k;

    if (state->timestep < k) return;

    float rho = params->rho;
    float mu = state->use_adaptive_mu ? state->mu_state : params->mu;
    float sigma_z_sq = params->sigma_z * params->sigma_z;

    for (int lag = 1; lag < k; lag++) {
        int idx_curr = (state->smooth_head - lag - 1 + k) % k;
        int idx_next = (state->smooth_head - lag + k) % k;

        float h_curr = state->smooth_h_mean[idx_curr];
        float h_next = state->smooth_h_mean[idx_next];
        float var_curr = state->smooth_h_var[idx_curr];

        float h_pred = mu + rho * (h_curr - mu);
        float pred_var = rho * rho * var_curr + sigma_z_sq;
        float innovation = h_next - h_pred;
        float J = rho * var_curr / (pred_var + 1e-8f);

        state->smooth_h_mean[idx_curr] = h_curr + J * innovation;
        state->smooth_h_var[idx_curr] = var_curr * (1.0f - J * rho);
    }
}

/** @brief Return smoothed h_mean with configured output lag. */
static inline float svpf_get_smoothed_output(SVPFState* state, float h_mean_raw) {
    if (!state->use_smoothing) return h_mean_raw;

    int k = state->smooth_lag;
    if (k > SVPF_SMOOTH_MAX_LAG) k = SVPF_SMOOTH_MAX_LAG;

    if (state->timestep < k) return h_mean_raw;

    int output_lag = state->smooth_output_lag;
    if (output_lag <= 0) return h_mean_raw;
    if (output_lag >= k) output_lag = k - 1;

    int idx = (state->smooth_head - output_lag - 1 + k) % k;
    return state->smooth_h_mean[idx];
}

// =============================================================================
// Optimized Backend Init (svpf_optimized_graph.cu)
// =============================================================================

/** @brief Lazy-init for SVPFOptimizedState buffers. Idempotent. */
void svpf_optimized_init(SVPFOptimizedState* opt, int n);

// =============================================================================
// Persistent Kernel API (svpf_persistent.cu)
// =============================================================================

/** @brief Async 2-kernel step: predict → probe → persistent stein. */
void svpf_persistent_step_async(
    SVPFState* state, float y_t, float y_prev, const SVPFParams* params);

/** @brief Sync outputs after persistent_step_async. Includes smoothing + adaptive mu. */
void svpf_persistent_sync_outputs(
    SVPFState* state, float* h_loglik_out, float* h_vol_out, float* h_mean_out);

/** @brief Synchronous convenience: async + sync in one call. */
void svpf_persistent_step(
    SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
    float* h_loglik_out, float* h_vol_out, float* h_mean_out);

#endif // SVPF_KERNELS_CUH
