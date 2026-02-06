/**
 * @file svpf_kernels.cuh
 * @brief CUDA kernel declarations for SVPF
 *
 * Declarations only. Definitions in svpf_opt_kernels.cu
 *
 * Production kernel set:
 *   Predict:   antithetic guided (Student-t state)
 *   Guide:     variance-preserving
 *   Gradient:  fused prior+likelihood+Hessian (Student-t prior, exact likelihood)
 *   Stein:     full Newton (with/without KSD)
 *   Output:    bandwidth, fused outputs, KSD reduce, rejuvenation
 */

#ifndef SVPF_KERNELS_CUH
#define SVPF_KERNELS_CUH

#include "svpf.cuh"
#include "svpf_common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Utility Kernels
// =============================================================================

__global__ void svpf_init_rng_kernel(
    curandStatePhilox4_32_10_t* states, int n, unsigned long long seed
);

__global__ void svpf_init_particles_kernel(
    float* h, curandStatePhilox4_32_10_t* rng_states,
    float mu, float stationary_std, int n
);

__global__ void svpf_copy_kernel(const float* src, float* dst, int n);

// =============================================================================
// Predict: Antithetic Guided (launch with n/2 threads)
// =============================================================================

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
    int n  // Full N, kernel pairs i and i+n/2
);

// =============================================================================
// Guide: Variance-Preserving
// =============================================================================

__global__ void svpf_apply_guide_preserving_kernel(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    float guide_mean,
    float guide_strength,
    int n
);

// =============================================================================
// Fused Gradient (Student-t prior + exact likelihood + full Hessian)
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
    float rho, float sigma_z, float mu,
    float beta, float nu, float student_t_const,
    float lik_offset, float gamma,
    float nu_state,
    int n
);

// =============================================================================
// Full Newton Stein Transport (Detommaso 2018)
// =============================================================================

/** @brief Non-last iterations (no KSD). */
__global__ void svpf_fused_stein_transport_full_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int n
);

/** @brief Last iteration (computes KSD in same O(N²) pass). */
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
    int n
);

// =============================================================================
// KSD + Rejuvenation
// =============================================================================

__global__ void svpf_ksd_reduce_kernel(
    const float* __restrict__ d_ksd_partial,
    float* __restrict__ d_ksd,
    int n
);

__global__ void svpf_partial_rejuvenation_kernel(
    float* __restrict__ h,
    float guide_mean, float guide_std,
    float rejuv_prob, float blend_factor,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    int n
);

// =============================================================================
// Bandwidth + Outputs
// =============================================================================

__global__ void svpf_fused_bandwidth_kernel(
    const float* __restrict__ h,
    const float* __restrict__ d_y,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    int y_idx, float alpha_bw, float alpha_ret, int n
);

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
    float obs_var = 4.93f + 2.0f;

    float S = P_pred + obs_var;
    float K = P_pred / (S + 1e-8f);

    float innovation = log_y2 - (m_pred + obs_offset);

    state->guide_mean = m_pred + K * innovation;
    state->guide_var = (1.0f - K) * P_pred;
    state->guide_K = K;
}

#endif // SVPF_KERNELS_CUH
