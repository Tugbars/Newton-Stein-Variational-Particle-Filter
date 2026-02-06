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
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,
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
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,
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

#endif // SVPF_KERNELS_CUH
