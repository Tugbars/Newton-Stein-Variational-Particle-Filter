/**
 * @file svpf_fully_fused_v2.cuh
 * @brief Fully Fused Single-Kernel SVPF Step V2 - Complete Feature Set (FIXED)
 * 
 * Changes from original:
 * - Added student_t_implied_offset parameter (was hardcoded, now passed)
 * - Renamed guided_innovation_threshold to guided_innovation_thresh_predict
 *   to avoid confusion with guide_innovation_threshold
 */

#ifndef SVPF_FULLY_FUSED_V2_CUH
#define SVPF_FULLY_FUSED_V2_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if particle count is supported
 */
bool svpf_fully_fused_v2_supported(int n_particles);

/**
 * @brief Fully fused SVPF step V2 - complete feature set
 * 
 * Features:
 * - EKF guide (computed internally)
 * - Guided predict with innovation-based alpha
 * - Guide-preserving nudge
 * - Adaptive guide strength
 * - Newton / Full Newton preconditioning
 * - Local params (delta_rho, delta_sigma)
 * - MIM jumps
 * - Partial rejuvenation
 * - Adaptive μ update
 * - Adaptive σ scaling
 * 
 * Limitation: n ≤ 1024
 */
cudaError_t svpf_fully_fused_step_v2(
    // Arrays [n]
    float* h,
    float* h_prev,
    float* grad_log_p,
    float* log_weights,
    float* d_grad_v,
    float* d_inv_hessian,
    curandStatePhilox4_32_10_t* rng,
    // Scalar inputs
    float y_t,
    float y_prev,
    float h_mean_prev,
    float vol_prev,
    float ksd_prev,
    // Scalar outputs (device pointers)
    float* d_h_mean,
    float* d_vol,
    float* d_loglik,
    float* d_bandwidth,
    float* d_ksd,
    float* d_guide_mean,  // EKF guide mean (read/write, persistent)
    float* d_guide_var,   // EKF guide variance (read/write, persistent)
    // Model parameters
    float rho,
    float sigma_z,
    float mu,
    float nu,
    float lik_offset,
    float student_t_const,
    float student_t_implied_offset,  // NEW: pass from state->student_t_implied_offset
    float gamma,  // Leverage effect
    // Local params
    float delta_rho,
    float delta_sigma,
    // MIM
    float mim_jump_prob,
    float mim_jump_scale,
    // Guide parameters
    float guide_strength_base,
    float guide_strength_max,
    float guide_innovation_threshold,    // For adaptive guide strength
    float guided_alpha_base,
    float guided_alpha_shock,
    float guided_innovation_thresh_predict,  // For guided predict (separate threshold)
    int use_guide,
    int use_guided_predict,
    int use_guide_preserving,
    // Newton
    int use_newton,
    int use_full_newton,
    // Rejuvenation
    int use_rejuvenation,
    float rejuv_ksd_threshold,
    float rejuv_prob,
    float rejuv_blend,
    // Adaptive mu
    int use_adaptive_mu,
    float* d_mu_state,
    float mu_ema_alpha,
    // Adaptive sigma
    int use_adaptive_sigma,
    float sigma_boost_threshold,
    float sigma_boost_max,
    // Stein parameters
    float step_size,
    float temperature,
    float rmsprop_rho,
    float rmsprop_eps,
    int n_stein_steps,
    int n_anneal_steps,
    int stein_sign_mode,
    // Control
    int n,
    int timestep,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // SVPF_FULLY_FUSED_V2_CUH
