/**
 * @file svpf_persistent_kernel_v2.cuh
 * @brief Fixed Single-Block Persistent Stein Kernel
 * 
 * Key improvements over v1:
 * - Single block execution (N ≤ 1024) - fixes shared memory scope bug
 * - Proper block_reduce_max - fixes log-sum-exp stability
 * - Configurable bandwidth update interval
 * - No cooperative groups needed
 */

#ifndef SVPF_PERSISTENT_KERNEL_V2_CUH
#define SVPF_PERSISTENT_KERNEL_V2_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if particle count is supported by v2 persistent kernel
 * @param n_particles Number of particles
 * @return true if n_particles <= 1024
 */
bool svpf_persistent_v2_supported(int n_particles);

/**
 * @brief Launch the single-block persistent Stein kernel
 * 
 * This kernel fuses the entire Stein loop into a single kernel launch,
 * eliminating per-iteration launch overhead (~200μs → ~50μs expected).
 * 
 * Limitation: n_particles must be ≤ 1024 (max threads per block)
 * 
 * @param h Particle states [n]
 * @param h_prev Previous states [n]
 * @param grad_log_p Gradient buffer [n]
 * @param log_weights Log importance weights [n]
 * @param d_grad_v RMSprop velocity [n]
 * @param d_inv_hessian Hessian diagonal [n]
 * @param rng CURAND states [n]
 * @param d_y Observations [y_prev, y_t]
 * @param d_bandwidth Bandwidth scalar
 * @param d_ksd_partial KSD partial sums [n]
 * @param d_ksd Final KSD output
 * @param d_h_mean Output: posterior mean
 * @param d_vol Output: volatility
 * @param d_loglik Output: log-likelihood
 * @param rho AR(1) persistence
 * @param sigma_z State noise std
 * @param mu Long-run mean
 * @param nu Student-t DOF
 * @param lik_offset Likelihood offset
 * @param student_t_const Precomputed Student-t normalization constant
 * @param step_size Stein step size
 * @param temperature SVLD temperature
 * @param rmsprop_rho RMSprop decay
 * @param rmsprop_eps RMSprop epsilon
 * @param d_beta_schedule Annealing schedule [n_anneal_steps] (device pointer)
 * @param n_anneal_steps Number of annealing stages
 * @param n_stein_steps Total Stein iterations
 * @param use_newton Enable Newton preconditioning
 * @param use_full_newton Enable full Newton (kernel-weighted Hessian)
 * @param stein_sign_mode 0=legacy, 1=paper
 * @param bandwidth_update_interval Recompute bandwidth every K iterations (0=never)
 * @param n Number of particles
 * @param stream CUDA stream
 * @return cudaError_t
 */
cudaError_t svpf_launch_persistent_stein_v2(
    float* h, float* h_prev, float* grad_log_p, float* log_weights,
    float* d_grad_v, float* d_inv_hessian, curandStatePhilox4_32_10_t* rng,
    const float* d_y, float* d_bandwidth, float* d_ksd_partial,
    float* d_ksd, float* d_h_mean, float* d_vol, float* d_loglik,
    float rho, float sigma_z, float mu, float nu, float lik_offset, float student_t_const,
    float step_size, float temperature, float rmsprop_rho, float rmsprop_eps,
    const float* d_beta_schedule, int n_anneal_steps,
    int n_stein_steps, int use_newton, int use_full_newton, int stein_sign_mode,
    int bandwidth_update_interval,
    int n,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // SVPF_PERSISTENT_KERNEL_V2_CUH
