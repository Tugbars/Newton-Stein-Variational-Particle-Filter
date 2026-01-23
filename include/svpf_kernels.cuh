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
    float guide_strength,
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
    float guide_strength,
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
    bool use_newton, int n
);

__global__ void svpf_fused_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int n
);

__global__ void svpf_fused_stein_transport_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int n
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
    float obs_offset = -1.27f;
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
