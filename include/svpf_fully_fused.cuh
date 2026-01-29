/**
 * @file svpf_fully_fused.cuh
 * @brief Fully Fused Single-Kernel SVPF Step Header
 */

#ifndef SVPF_FULLY_FUSED_CUH
#define SVPF_FULLY_FUSED_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if particle count is supported
 */
bool svpf_fully_fused_supported(int n_particles);

/**
 * @brief Fully fused SVPF step - everything in one kernel
 * 
 * Fuses: predict + guide + bandwidth + stein loop + outputs
 * Expected latency: 40-60μs (vs 250μs with separate kernels)
 * 
 * Limitation: n_particles ≤ 1024
 */
cudaError_t svpf_fully_fused_step(
    // Particle state [n]
    float* h,
    float* h_prev,
    float* grad_log_p,
    float* log_weights,
    float* d_grad_v,
    curandStatePhilox4_32_10_t* rng,
    // Inputs
    float y_t,
    float y_prev,
    float h_mean_prev,
    // Outputs
    float* d_h_mean,
    float* d_vol,
    float* d_loglik,
    float* d_bandwidth,
    float* d_ksd,
    // Model parameters
    float rho,
    float sigma_z,
    float mu,
    float nu,
    float lik_offset,
    float student_t_const,
    // MIM
    float mim_jump_prob,
    float mim_jump_scale,
    // Guide
    float guide_strength,
    float guide_mean,
    int use_guide,
    // Stein
    float step_size,
    float temperature,
    float rmsprop_rho,
    float rmsprop_eps,
    int n_stein_steps,
    int n_anneal_steps,
    int stein_sign_mode,
    // Control
    int n,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // SVPF_FULLY_FUSED_CUH
