/**
 * @file svpf_fused_gradient_stats.cuh
 * @brief Fused gradient + annealing stats kernel (v2)
 * 
 * Combines svpf_fused_gradient_kernel + annealing stats into one launch.
 * Stats are reduced in-register using warp shuffles, then atomic-added to global.
 * 
 * Saves: 1 kernel launch + 1 global memory round-trip per timestep
 *
 * v2 changes:
 *   1. Removed dead sh_h_prev array — matches gradient kernel v2.
 *      Shared memory: 2*n*sizeof(float) → 1*n*sizeof(float)
 *   2. Added 5th stat channel (sum_h_diff) so host can compute proper Var(h).
 *      Old: d_stats[4] with sqrt(E[h²]) ≠ std(h) when mean(h) ≠ 0
 *      New: d_stats[5] with E[h-c] and E[(h-c)²] → Var = E[X²] - E[X]²
 *   3. Hoisted inv_nu_sigma_sq in Student-t branch (minor FLOP save)
 *   4. Warp reduction expanded to 5-wide
 *
 * d_stats layout [5 floats, must be zeroed before launch]:
 *   [0] sum of (log_w_clamped - CENTER_LL)          → for mean_ll_diff
 *   [1] sum of (log_w_clamped - CENTER_LL)²          → for var_ll
 *   [2] sum of |grad|                                → for mean_grad
 *   [3] sum of (h - CENTER_H)²                       → for var_h (with [4])
 *   [4] sum of (h - CENTER_H)                         → for var_h (mean correction)
 *
 * Host decode (after kernel, divide by n):
 *   float inv_n = 1.0f / (float)n;
 *   float mean_ll_diff = d_stats[0] * inv_n;
 *   float var_ll = d_stats[1] * inv_n - mean_ll_diff * mean_ll_diff;
 *   float mean_grad = d_stats[2] * inv_n;
 *   float mean_h_diff = d_stats[4] * inv_n;
 *   float h_std = sqrtf(d_stats[3] * inv_n - mean_h_diff * mean_h_diff);
 */

#ifndef SVPF_FUSED_GRADIENT_STATS_CUH
#define SVPF_FUSED_GRADIENT_STATS_CUH

#include <cuda_runtime.h>

// ============================================================================
// Warp reduction for 5 values simultaneously
// ============================================================================
__device__ __forceinline__ void warp_reduce_5(
    float& v1, float& v2, float& v3, float& v4, float& v5
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v1 += __shfl_down_sync(0xffffffff, v1, offset);
        v2 += __shfl_down_sync(0xffffffff, v2, offset);
        v3 += __shfl_down_sync(0xffffffff, v3, offset);
        v4 += __shfl_down_sync(0xffffffff, v4, offset);
        v5 += __shfl_down_sync(0xffffffff, v5, offset);
    }
}

// Safe exp with clamping
__device__ __forceinline__ float safe_exp_fused(float x) {
    x = fminf(fmaxf(x, -20.0f), 20.0f);
    return __expf(x);
}

/**
 * Fused gradient + stats kernel (v2)
 * 
 * Shared memory requirement: n * sizeof(float)  (sh_mu_i only)
 * d_stats requirement: 5 * sizeof(float), must be zeroed before launch
 */
__global__ void svpf_fused_gradient_stats_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_combined,
    float* __restrict__ log_w,
    float* __restrict__ precond_grad,
    float* __restrict__ inv_hessian,
    const float* __restrict__ d_y,
    float* __restrict__ d_stats,      // [5] output (must be zeroed)
    int y_idx,
    float rho,
    float sigma_z,
    float mu,
    float beta,
    float nu,
    float student_t_const,
    float lik_offset,
    float gamma,
    bool use_exact_gradient,
    bool use_newton,
    bool use_fan_mode,
    int use_student_t_state,
    float nu_state,
    int n
) {
    // v2: Only sh_mu_i needed — sh_h_prev was dead (loaded but never read)
    extern __shared__ float smem[];
    float* sh_mu_i = smem;

    float y_prev = (y_idx > 0) ? d_y[y_idx - 1] : 0.0f;

    // Compute transition means directly from h_prev (no intermediate copy)
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float hp = h_prev[k];
        float vol_prev_k = __expf(hp * 0.5f);
        float leverage_k = gamma * y_prev / (vol_prev_k + 1e-8f);
        sh_mu_i[k] = mu + rho * (hp - mu) + leverage_k;
    }
    __syncthreads();

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize all 5 stat accumulators
    float r_ll_diff    = 0.0f;   // [0] sum of (log_w - CENTER_LL)
    float r_ll_diff_sq = 0.0f;   // [1] sum of (log_w - CENTER_LL)²
    float r_grad_abs   = 0.0f;   // [2] sum of |grad|
    float r_h_diff_sq  = 0.0f;   // [3] sum of (h - CENTER_H)²
    float r_h_diff     = 0.0f;   // [4] sum of (h - CENTER_H)     ← v2 NEW

    float h_j = 0.0f;
    float g = 0.0f;
    float log_w_j = 0.0f;

    if (j < n) {
        h_j = h[j];
        float y_t = d_y[y_idx];

        // ===== PRIOR GRADIENT =====
        float sigma_z_sq = sigma_z * sigma_z;
        float grad_prior;
        float hess_prior;

        if (use_student_t_state) {
            float nu_sigma_sq = nu_state * sigma_z_sq;
            float nu_plus_1 = nu_state + 1.0f;
            float half_nu_plus_1 = 0.5f * nu_plus_1;
            float inv_nu_sigma_sq = 1.0f / nu_sigma_sq;  // v2: hoisted

            float log_r_max = -1e10f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_mu_i[i];
                float ratio = diff * diff * inv_nu_sigma_sq;
                float log_r_i = -half_nu_plus_1 * __logf(1.0f + ratio);
                log_r_max = fmaxf(log_r_max, log_r_i);
            }

            float sum_r = 0.0f;
            float weighted_grad = 0.0f;
            float weighted_hess = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_mu_i[i];
                float diff_sq = diff * diff;
                float denom = nu_sigma_sq + diff_sq;

                float ratio = diff_sq * inv_nu_sigma_sq;
                float log_r_i = -half_nu_plus_1 * __logf(1.0f + ratio);
                float r_i = __expf(log_r_i - log_r_max);
                sum_r += r_i;

                weighted_grad -= r_i * nu_plus_1 * diff / denom;
                float hess_i = -nu_plus_1 * (nu_sigma_sq - diff_sq) / (denom * denom);
                weighted_hess += r_i * hess_i;
            }
            grad_prior = weighted_grad / (sum_r + 1e-8f);
            hess_prior = weighted_hess / (sum_r + 1e-8f);

        } else {
            float inv_2sigma_sq = 0.5f / sigma_z_sq;
            float inv_sigma_sq = 1.0f / sigma_z_sq;

            float log_r_max = -1e10f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_mu_i[i];
                float log_r_i = -diff * diff * inv_2sigma_sq;
                log_r_max = fmaxf(log_r_max, log_r_i);
            }

            float sum_r = 0.0f;
            float weighted_grad = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < n; i++) {
                float diff = h_j - sh_mu_i[i];
                float log_r_i = -diff * diff * inv_2sigma_sq;
                float r_i = __expf(log_r_i - log_r_max);
                sum_r += r_i;
                weighted_grad -= r_i * diff * inv_sigma_sq;
            }
            grad_prior = weighted_grad / (sum_r + 1e-8f);
            hess_prior = -inv_sigma_sq;
        }

        // ===== LIKELIHOOD GRADIENT =====
        float vol = safe_exp_fused(h_j);
        float y_sq = y_t * y_t;
        float scaled_y_sq = y_sq / (vol + 1e-8f);
        float A = scaled_y_sq / nu;
        float one_plus_A = 1.0f + A;

        if (use_fan_mode) {
            log_w_j = 0.0f;
        } else {
            log_w_j = student_t_const - 0.5f * h_j
                    - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));
        }
        log_w[j] = log_w_j;

        float grad_lik;
        if (use_exact_gradient) {
            float raw_grad = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A;
            grad_lik = raw_grad - lik_offset;
        } else {
            float log_y2 = __logf(y_sq + 1e-10f);
            float R_noise = 1.4f;
            grad_lik = (log_y2 - h_j + lik_offset) / R_noise;
        }

        // ===== COMBINE =====
        float effective_beta = use_fan_mode ? 1.0f : beta;
        g = grad_prior + effective_beta * grad_lik;
        g = fminf(fmaxf(g, -10.0f), 10.0f);
        grad_combined[j] = g;

        // ===== HESSIAN =====
        if (use_newton && precond_grad != nullptr) {
            float hess_lik = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
            float curvature = -(hess_lik + hess_prior);
            curvature = fminf(fmaxf(curvature, 0.1f), 100.0f);
            inv_hessian[j] = curvature;
            float inv_H = 1.0f / curvature;
            precond_grad[j] = 0.7f * g * inv_H;
        }

        // ===== ACCUMULATE STATS (in registers) =====
        // Centering constants for numerical stability of variance computation
        const float CENTER_LL = -50.0f;
        const float CENTER_H = 0.0f;

        // Clamp log_w for outlier rejection
        float ll_clamped = fmaxf(fminf(log_w_j, -1.0f), -100.0f);
        float ll_diff = ll_clamped - CENTER_LL;

        r_ll_diff    = ll_diff;
        r_ll_diff_sq = ll_diff * ll_diff;
        r_grad_abs   = fabsf(g);

        float h_diff = h_j - CENTER_H;
        r_h_diff_sq  = h_diff * h_diff;
        r_h_diff     = h_diff;              // v2: NEW — enables Var(h) on host
    }

    // =========================================================================
    // PHASE 2: Warp-level reduction (5 channels)
    // =========================================================================
    warp_reduce_5(r_ll_diff, r_ll_diff_sq, r_grad_abs, r_h_diff_sq, r_h_diff);

    // =========================================================================
    // PHASE 3: Block-level reduction via shared memory
    // =========================================================================
    // Reuse shared memory — we're done with sh_mu_i at this point.
    // Need 5 * num_warps floats. With blockDim ≤ 1024, that's ≤ 160 floats,
    // well within the n floats we have (n ≥ 128 in any practical config).
    float* s_warp_stats = smem;  // [5 * num_warps]

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        s_warp_stats[wid * 5 + 0] = r_ll_diff;
        s_warp_stats[wid * 5 + 1] = r_ll_diff_sq;
        s_warp_stats[wid * 5 + 2] = r_grad_abs;
        s_warp_stats[wid * 5 + 3] = r_h_diff_sq;
        s_warp_stats[wid * 5 + 4] = r_h_diff;     // v2: 5th channel
    }
    __syncthreads();

    // First warp reduces all warp leaders
    if (wid == 0) {
        r_ll_diff    = (lane < num_warps) ? s_warp_stats[lane * 5 + 0] : 0.0f;
        r_ll_diff_sq = (lane < num_warps) ? s_warp_stats[lane * 5 + 1] : 0.0f;
        r_grad_abs   = (lane < num_warps) ? s_warp_stats[lane * 5 + 2] : 0.0f;
        r_h_diff_sq  = (lane < num_warps) ? s_warp_stats[lane * 5 + 3] : 0.0f;
        r_h_diff     = (lane < num_warps) ? s_warp_stats[lane * 5 + 4] : 0.0f;

        warp_reduce_5(r_ll_diff, r_ll_diff_sq, r_grad_abs, r_h_diff_sq, r_h_diff);

        // Thread 0 atomically adds to global stats
        if (lane == 0) {
            atomicAdd(&d_stats[0], r_ll_diff);
            atomicAdd(&d_stats[1], r_ll_diff_sq);
            atomicAdd(&d_stats[2], r_grad_abs);
            atomicAdd(&d_stats[3], r_h_diff_sq);
            atomicAdd(&d_stats[4], r_h_diff);      // v2: 5th channel
        }
    }
}

#endif // SVPF_FUSED_GRADIENT_STATS_CUH
