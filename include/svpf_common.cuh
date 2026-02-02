/**
 * @file svpf_common.cuh
 * @brief Common device helpers and utility kernels for SVPF
 *
 * Shared primitives used across all SVPF kernel files:
 * - Math helpers (clamp, safe_exp)
 * - Student-t sampling
 * - Warp/block reductions
 * - Basic utility kernels (RNG init, particle init, copy)
 */

#ifndef SVPF_COMMON_CUH
#define SVPF_COMMON_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// =============================================================================
// Constants
// =============================================================================

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef SVPF_H_MIN
#define SVPF_H_MIN -15.0f
#endif

#ifndef SVPF_H_MAX
#define SVPF_H_MAX 5.0f
#endif

// =============================================================================
// Math Helpers
// =============================================================================

/** @brief Clamp log-volatility to valid range [-15, 5]. */
__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, SVPF_H_MIN), SVPF_H_MAX);
}

/** @brief Safe exponential, capped to prevent overflow. */
__device__ __forceinline__ float safe_exp(float x) {
    return __expf(fminf(x, 20.0f));
}

// =============================================================================
// Student-t Sampling
// =============================================================================

/**
 * @brief Sample from Student-t distribution.
 *
 * Uses ratio method: t = z / sqrt(chi2/nu) where z ~ N(0,1), chi2 ~ χ²(nu).
 * For nu > 30, returns Gaussian (Student-t ≈ N(0,1) in this regime).
 *
 * @param rng Pointer to curand state
 * @param nu  Degrees of freedom (recommend 5-7 for fat tails)
 * @return Sample from Student-t(nu)
 *
 * @note Cost: ~nu extra curand_normal calls. For nu=5, negligible overhead.
 */
__device__ __forceinline__ float sample_student_t(curandStatePhilox4_32_10_t* rng, float nu) {
    if (nu > 30.0f) {
        return curand_normal(rng);
    }
    
    float z = curand_normal(rng);
    
    int nu_int = (int)(nu + 0.5f);
    nu_int = max(nu_int, 3);
    
    float chi2 = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < nu_int; i++) {
        float u = curand_normal(rng);
        chi2 += u * u;
    }
    
    return z * sqrtf((float)nu_int / (chi2 + 1e-8f));
}

// =============================================================================
// Warp Reductions
// =============================================================================

/** @brief Warp-level sum reduction. Returns result in lane 0. */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/** @brief Warp-level max reduction. Returns result in lane 0. */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/** @brief Warp-level min reduction. Returns result in lane 0. */
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// =============================================================================
// Block Reductions
// =============================================================================

/** @brief Block-level sum reduction. Returns result in thread 0. */
__device__ inline float block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

/** @brief Block-level min reduction. Returns result in thread 0. */
__device__ inline float block_reduce_min(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_min(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : 1e10f;
    if (wid == 0) val = warp_reduce_min(val);
    return val;
}

/** @brief Block-level max reduction. Returns result in thread 0. */
__device__ inline float block_reduce_max(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : -1e10f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

// =============================================================================
// Basic Utility Kernels (declarations only - define in ONE .cu file)
// =============================================================================

/** @brief Initialize curand RNG states. */
__global__ void svpf_init_rng_kernel(
    curandStatePhilox4_32_10_t* states,
    int n,
    unsigned long long seed
);

/** @brief Initialize particles from stationary distribution. */
__global__ void svpf_init_particles_kernel(
    float* h,
    curandStatePhilox4_32_10_t* rng_states,
    float mu,
    float stationary_std,
    int n
);

/** @brief Simple device-to-device copy. */
__global__ void svpf_copy_kernel(const float* src, float* dst, int n);

#endif // SVPF_COMMON_CUH
