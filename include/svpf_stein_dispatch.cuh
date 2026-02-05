/**
 * @file svpf_stein_dispatch.cuh
 * @brief SteinMode enum, template declaration, and dispatch helper
 * 
 * Include this from any .cu file that needs to launch Stein transport kernels.
 * The template definition + explicit instantiations live in svpf_opt_kernels.cu.
 */

#ifndef SVPF_STEIN_DISPATCH_CUH
#define SVPF_STEIN_DISPATCH_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// =============================================================================
// Stein Mode Enum (must match svpf_opt_kernels.cu)
// =============================================================================

enum class SteinMode {
    STANDARD,      // Raw grad for score, no Hessian weighting
    NEWTON,        // precond_grad for score, inv_hessian weights repulsion
    FULL_NEWTON    // Raw grad for score, kernel-weighted Hessian preconditioner
};

// =============================================================================
// Template Declaration (defined + explicitly instantiated in svpf_opt_kernels.cu)
// =============================================================================

template <SteinMode MODE, bool COMPUTE_KSD>
__global__ void svpf_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ score,
    const float* __restrict__ curvature,
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
    int n
);

// =============================================================================
// Inline Dispatch Helper
// =============================================================================
// Replaces 6-way if/else blocks at every call site.
// Selects score/curvature pointers and template instantiation based on flags.
//
// Argument mapping:
//   FULL_NEWTON: score = grad_combined,  curvature = inv_hessian (stores |H|)
//   NEWTON:      score = precond_grad,   curvature = inv_hessian
//   STANDARD:    score = grad_combined,  curvature = nullptr

template <bool COMPUTE_KSD>
static inline void launch_stein_dispatch(
    bool use_full_newton,
    bool use_newton,
    // Device pointers
    float* d_h,
    float* d_grad_combined,
    float* d_precond_grad,
    float* d_inv_hessian,
    float* d_v_rmsprop,
    curandStatePhilox4_32_10_t* d_rng,
    const float* d_bandwidth,
    float* d_ksd_partial,
    // Scalar params
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int stein_sign_mode,
    int n, int block_size, size_t stein_smem, cudaStream_t stream
) {
    int grid = (n + block_size - 1) / block_size;

    if (use_full_newton) {
        svpf_stein_transport_kernel<SteinMode::FULL_NEWTON, COMPUTE_KSD>
            <<<grid, block_size, stein_smem, stream>>>(
                d_h,
                d_grad_combined,     // score = raw gradient
                d_inv_hessian,       // curvature = local hessian values
                d_v_rmsprop, d_rng, d_bandwidth,
                COMPUTE_KSD ? d_ksd_partial : nullptr,
                step_size, beta_factor, temperature,
                rho_rmsprop, epsilon, stein_sign_mode, n);
    } else if (use_newton) {
        svpf_stein_transport_kernel<SteinMode::NEWTON, COMPUTE_KSD>
            <<<grid, block_size, stein_smem, stream>>>(
                d_h,
                d_precond_grad,      // score = preconditioned gradient
                d_inv_hessian,       // curvature = curvature values
                d_v_rmsprop, d_rng, d_bandwidth,
                COMPUTE_KSD ? d_ksd_partial : nullptr,
                step_size, beta_factor, temperature,
                rho_rmsprop, epsilon, stein_sign_mode, n);
    } else {
        svpf_stein_transport_kernel<SteinMode::STANDARD, COMPUTE_KSD>
            <<<grid, block_size, stein_smem, stream>>>(
                d_h,
                d_grad_combined,     // score = raw gradient
                (const float*)nullptr,  // no curvature
                d_v_rmsprop, d_rng, d_bandwidth,
                COMPUTE_KSD ? d_ksd_partial : nullptr,
                step_size, beta_factor, temperature,
                rho_rmsprop, epsilon, stein_sign_mode, n);
    }
}

#endif // SVPF_STEIN_DISPATCH_CUH
