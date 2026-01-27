/**
 * @file svpf_kernels.cuh
 * @brief CUDA kernel declarations for SVPF
 * 
 * This header contains kernel DECLARATIONS only.
 * Definitions are in svpf_kernels.cu, svpf_opt_kernels.cu
 */

#ifndef SVPF_KERNELS_CUH
#define SVPF_KERNELS_CUH

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Configuration Constants
// =============================================================================

#define TILE_J 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SMALL_N_THRESHOLD 4096
#define BANDWIDTH_UPDATE_INTERVAL 5
#define MAX_T_SIZE 10000

// =============================================================================
// Kernel Declarations - Predict
// =============================================================================

__global__ void svpf_predict_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    int t,
    float rho, float sigma_z, float mu, float gamma,
    int n
);

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
);

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
    float implied_offset,  // Student-t implied h offset (replaces hardcoded 1.27)
    int n
);

// =============================================================================
// Kernel Declarations - Gradient (Legacy, kept for reference)
// =============================================================================

__global__ void svpf_mixture_prior_grad_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior,
    float rho, float sigma_z, float mu,
    int n
);

__global__ void svpf_mixture_prior_grad_tiled_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior,
    float rho, float sigma_z, float mu,
    int n
);

__global__ void svpf_likelihood_only_kernel(
    const float* __restrict__ h,
    float* __restrict__ grad_lik,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    int t,
    float nu, float student_t_const,
    int n
);

__global__ void svpf_combine_gradients_kernel(
    const float* __restrict__ grad_prior,
    const float* __restrict__ grad_lik,
    float* __restrict__ grad,
    float beta,
    int n
);

__global__ void svpf_hessian_precond_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad_combined,
    float* __restrict__ precond_grad,
    float* __restrict__ inv_hessian,
    const float* __restrict__ d_y,
    int t,
    float nu, float sigma_z,
    int n
);

// =============================================================================
// Kernel Declarations - Reduction (Legacy)
// =============================================================================

__global__ void svpf_logsumexp_kernel(
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_max_log_w,
    int t,
    int n
);

__global__ void svpf_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float alpha,
    int n
);

// =============================================================================
// Kernel Declarations - Stein (Legacy)
// =============================================================================

__global__ void svpf_stein_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
);

__global__ void svpf_stein_newton_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
);

// =============================================================================
// Kernel Declarations - Transport (Legacy)
// =============================================================================

__global__ void svpf_apply_transport_svld_kernel(
    float* __restrict__ h,
    const float* __restrict__ phi,
    float* __restrict__ v,
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    float base_step_size,
    float beta_anneal_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int n
);

// =============================================================================
// Kernel Declarations - Guide
// =============================================================================

__global__ void svpf_apply_guide_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_strength,
    int n
);

__global__ void svpf_apply_guide_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_guide_mean,
    const float* __restrict__ d_guide_strength,  // Adaptive strength from device
    int n
);

__global__ void svpf_apply_guide_preserving_kernel(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    float guide_mean,
    float guide_strength,
    int n
);

__global__ void svpf_apply_guide_preserving_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    const float* __restrict__ d_guide_mean,
    const float* __restrict__ d_guide_strength,  // Adaptive strength from device
    int n
);

// =============================================================================
// Kernel Declarations - Adaptive Bandwidth (Legacy)
// =============================================================================

__global__ void svpf_adaptive_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    float new_return,
    float ema_alpha,
    int n
);

__global__ void svpf_adaptive_bandwidth_kernel_graph(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    const float* __restrict__ d_y,
    int y_idx,
    float ema_alpha,
    int n
);

// =============================================================================
// Kernel Declarations - Output (Legacy)
// =============================================================================

__global__ void svpf_vol_mean_opt_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_vol,
    int t,
    int n
);

__global__ void svpf_store_h_mean_kernel(
    const float* __restrict__ d_sum,
    float* __restrict__ d_h_mean_prev,
    int n
);

__global__ void svpf_memset_kernel(float* __restrict__ data, float val, int n);

__global__ void svpf_h_mean_reduce_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_partial_sums,
    int n
);

__global__ void svpf_h_mean_finalize_kernel(
    const float* __restrict__ d_partial_sums,
    float* __restrict__ d_h_mean,
    int n_blocks,
    int n_particles
);

// =============================================================================
// FUSED KERNEL DECLARATIONS (svpf_opt_kernels.cu)
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
    float rho, float sigma_z, float mu,
    float beta, float nu, float student_t_const,
    float lik_offset,  // Likelihood center offset (only used if !use_exact_gradient)
    float gamma,       // Leverage coefficient
    bool use_exact_gradient,  // true = exact Student-t, false = log-squared surrogate
    bool use_newton,
    bool use_fan_mode,  // Fan mode: uniform weights, no annealing
    int n
);

__global__ void svpf_fused_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,  // 0=legacy(subtract), 1=paper(add)
    int n
);

// Stein + Transport + KSD (computes KSD in same O(N²) pass, zero extra cost)
__global__ void svpf_fused_stein_transport_ksd_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float* __restrict__ d_ksd_partial,  // Output: partial KSD sums [n floats]
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,  // 0=legacy(subtract), 1=paper(add)
    int n
);

__global__ void svpf_fused_stein_transport_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,  // 0=legacy(subtract), 1=paper(add)
    int n
);

// Newton + KSD variant
__global__ void svpf_fused_stein_transport_newton_ksd_kernel(
    float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float* __restrict__ d_ksd_partial,  // Output: partial KSD sums [n floats]
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,  // 0=legacy(subtract), 1=paper(add)
    int n
);

// Full Newton with kernel-weighted Hessian (Detommaso et al. 2018)
__global__ void svpf_fused_stein_transport_full_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,           // Raw combined gradient
    const float* __restrict__ local_hessian,  // Local curvature (NOT inverted)
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon,
    int stein_sign_mode,  // 0=legacy(subtract), 1=paper(add)
    int n
);

__global__ void svpf_fused_outputs_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    int t_out, int n
);

__global__ void svpf_fused_bandwidth_kernel(
    const float* __restrict__ h,
    const float* __restrict__ d_y,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    int y_idx, float alpha_bw, float alpha_ret, int n
);

// =============================================================================
// KSD REDUCTION KERNEL
// =============================================================================
// Reduces partial KSD sums to final KSD value
// KSD = sqrt((1/N²) * Σᵢ partial[i])

__global__ void svpf_ksd_reduce_kernel(
    const float* __restrict__ d_ksd_partial,
    float* __restrict__ d_ksd,
    int n
);

// =============================================================================
// PARTIAL REJUVENATION KERNEL (Maken et al. 2022)
// =============================================================================
// When KSD stays high (particles stuck at boundary), nudge a fraction toward
// the EKF guide prediction. This helps particles escape local modes.

__global__ void svpf_partial_rejuvenation_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_std,
    float rejuv_prob,       // Probability of rejuvenating each particle (e.g., 0.3)
    float blend_factor,     // How much to blend toward guide (e.g., 0.3)
    curandStatePhilox4_32_10_t* __restrict__ rng,
    int n
);

// =============================================================================
// Host-side Helper (inline - safe in header)
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
    // Observation model: log(y²) = h + E[log(ε²)]
    // For Student-t: E[log(y²)|h] = h - student_t_implied_offset
    float obs_offset = -state->student_t_implied_offset;
    float obs_var = 4.93f + 2.0f;
    
    float H = 1.0f;
    float R = obs_var;
    
    float S = H * H * P_pred + R;
    float K = P_pred * H / (S + 1e-8f);
    
    float y_pred = m_pred + obs_offset;
    float innovation = log_y2 - y_pred;
    
    state->guide_mean = m_pred + K * innovation;
    state->guide_var = (1.0f - K * H) * P_pred;
    state->guide_K = K;
}

#endif // SVPF_KERNELS_CUH
