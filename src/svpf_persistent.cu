/**
 * @file svpf_persistent.cu
 * @brief 2-Kernel Persistent SVPF Architecture for N=1024
 *
 * Replaces the 30+ kernel launch pipeline with:
 *   Kernel 1 (PROBE):           bandwidth + gradient_stats at β=0
 *   [D2H sync — CPU computes annealing schedule]
 *   Kernel 2 (PERSISTENT STEIN): all annealing stages × steps in ONE launch
 *                                 + rejuvenation + output reduction + KSD
 *
 * Design targets (RTX 5080, N=1024):
 *   - 1 block × 1024 threads (1 thread per particle)
 *   - Shared memory: 4×N floats = 16KB peak (well under 128KB limit)
 *   - Register budget: ~40 regs/thread with __noinline__ phase isolation
 *   - Launch overhead: 2×5μs + 1 sync ≈ 60μs (was 150-200μs)
 *
 * Prerequisites:
 *   - Predict + guide kernels run BEFORE probe (unchanged, O(N) each)
 *   - Include after svpf.cuh / svpf_kernels.cuh
 *
 * Compile with: --ptxas-options=-v to verify register counts per phase.
 */

#include "svpf.cuh"
#include "svpf_kernels.cuh"    // Declares existing kernels + svpf_ekf_update
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#ifndef SVPF_PERSISTENT_BLOCK_SIZE
#define SVPF_PERSISTENT_BLOCK_SIZE 1024
#endif

// Max annealing stages (for static beta schedule array)
#define SVPF_MAX_ANNEAL_STAGES 64

// =============================================================================
// INLINE DEVICE HELPERS
// =============================================================================
// Self-contained: no dependency on svpf_common.cuh

__device__ __forceinline__ float p_clamp_logvol(float h) {
    return fminf(fmaxf(h, SVPF_H_MIN), SVPF_H_MAX);
}

__device__ __forceinline__ float p_safe_exp(float x) {
    x = fminf(fmaxf(x, -20.0f), 20.0f);
    return __expf(x);
}

// Warp-level reduction (full warp of 32 threads)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Block-level reduction for 1024 threads (32 warps)
// Uses the last 32 floats of shared memory as scratch
__device__ float block_reduce_sum_1024(float val, float* scratch) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) scratch[wid] = val;
    __syncthreads();

    // First warp reduces 32 partial sums
    val = (threadIdx.x < 32) ? scratch[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;  // Valid in thread 0
}

__device__ float block_reduce_max_1024(float val, float* scratch) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) scratch[wid] = val;
    __syncthreads();

    val = (threadIdx.x < 32) ? scratch[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

__device__ float block_reduce_min_1024(float val, float* scratch) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_min(val);
    if (lane == 0) scratch[wid] = val;
    __syncthreads();

    val = (threadIdx.x < 32) ? scratch[threadIdx.x] : FLT_MAX;
    if (wid == 0) val = warp_reduce_min(val);
    return val;
}

// =============================================================================
// PARAMETER STRUCT (keeps kernel signatures clean)
// =============================================================================

struct SVPFPersistentParams {
    // SV model
    float rho;
    float sigma_z;
    float mu;
    float gamma;

    // Likelihood
    float nu;               // Observation Student-t df
    float student_t_const;
    float lik_offset;

    // Prior
    float nu_state;         // State Student-t df

    // Stein transport
    float step_size;
    float temperature;
    float rho_rmsprop;
    float epsilon;

    // Rejuvenation
    float guide_mean;
    float guide_std;
    float rejuv_prob;
    float rejuv_blend;
    int   do_rejuvenation;

    // Observation index in d_y[]
    int   y_idx;

    // Annealing schedule (computed on CPU after PROBE)
    int   n_stages;
    int   steps_per_beta;
};

// =============================================================================
// PHASE 1: GRADIENT (Student-t prior + exact likelihood + full Hessian)
// =============================================================================
// Register isolation: __noinline__ ensures gradient registers are freed
// before transport phase begins. Expected: ~32 live registers.
//
// Reads:  sh_mu_i[N] from shared (persistent, computed once)
//         h[i] from global (updated by previous transport)
// Writes: sh_grad[i], sh_hess[i] to shared
//         log_w[i] to global (for final output reduction)

__device__ __noinline__ void persistent_gradient_phase(
    const float* __restrict__ sh_mu_i,      // [N] transition means (shared, persistent)
    float*       __restrict__ sh_grad,       // [N] output: combined gradient (shared)
    float*       __restrict__ sh_hess,       // [N] output: curvature (shared)
    const float* __restrict__ h,             // [N] current particles (global)
    float*       __restrict__ log_w,         // [N] log importance weights (global)
    float        y_t,                        // Current observation
    float        beta,                       // Annealing temperature
    float        nu,                         // Likelihood df
    float        student_t_const,
    float        lik_offset,
    float        nu_state,                   // Prior df
    float        sigma_z,
    int          n
) {
    int i = threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];

    // ===== PRIOR GRADIENT (Student-t mixture over N transition kernels) =====
    float sigma_z_sq   = sigma_z * sigma_z;
    float nu_sigma_sq  = nu_state * sigma_z_sq;
    float nu_plus_1    = nu_state + 1.0f;
    float half_np1     = 0.5f * nu_plus_1;

    // Pass 1: find log-max for numerical stability
    float log_r_max = -1e10f;
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_mu_i[j];
        float lr   = -half_np1 * __logf(1.0f + diff * diff / nu_sigma_sq);
        log_r_max  = fmaxf(log_r_max, lr);
    }

    // Pass 2: weighted gradient + Hessian
    float sum_r        = 0.0f;
    float weighted_grad = 0.0f;
    float weighted_hess = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff    = h_i - sh_mu_i[j];
        float diff_sq = diff * diff;
        float denom   = nu_sigma_sq + diff_sq;

        float lr  = -half_np1 * __logf(1.0f + diff_sq / nu_sigma_sq);
        float r_j = __expf(lr - log_r_max);
        sum_r += r_j;

        weighted_grad -= r_j * nu_plus_1 * diff / denom;
        weighted_hess += r_j * (-nu_plus_1 * (nu_sigma_sq - diff_sq) / (denom * denom));
    }
    float grad_prior = weighted_grad / (sum_r + 1e-8f);
    float hess_prior = weighted_hess / (sum_r + 1e-8f);

    // ===== LIKELIHOOD GRADIENT (exact, Student-t observation) =====
    float vol       = p_safe_exp(h_i);
    float y_sq      = y_t * y_t;
    float A         = y_sq / ((vol + 1e-8f) * nu);
    float one_plus_A = 1.0f + A;

    // Importance weight
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));

    float grad_lik = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A - lik_offset;

    // ===== COMBINE =====
    float g = grad_prior + beta * grad_lik;
    g = fminf(fmaxf(g, -10.0f), 10.0f);

    // ===== HESSIAN (full Newton preconditioning) =====
    float hess_lik  = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
    float curvature = -(hess_lik + hess_prior);
    curvature = fminf(fmaxf(curvature, 0.1f), 100.0f);

    // Write to shared (ready for transport phase after syncthreads)
    sh_grad[i] = g;
    sh_hess[i] = curvature;
}

// =============================================================================
// PHASE 2: STEIN TRANSPORT (Full Newton, RMSProp, Cauchy kernel)
// =============================================================================
// Register isolation: ~28 live registers without KSD, ~33 with KSD.
//
// Reads:  sh_h[N], sh_grad[N], sh_hess[N] from shared
// Writes: h[i] to global (particle update)
//         v_rmsprop[i] to global
//         d_ksd_partial[i] to global (last iteration only)

__device__ __noinline__ void persistent_transport_phase(
    const float* __restrict__ sh_h,          // [N] particle positions (shared)
    const float* __restrict__ sh_grad,       // [N] gradients (shared)
    const float* __restrict__ sh_hess,       // [N] curvatures (shared)
    float*       __restrict__ h,             // [N] output: updated particles (global)
    float*       __restrict__ v_rmsprop,     // [N] RMSProp state (global)
    curandStatePhilox4_32_10_t* __restrict__ rng,  // [N] RNG states (global)
    float        bandwidth,
    float        step_size,
    float        beta_factor,
    float        temperature,
    float        rho_rmsprop,
    float        epsilon,
    bool         compute_ksd,
    float*       __restrict__ d_ksd_partial, // [N] KSD partial sums (global, may be NULL)
    int          n
) {
    int i = threadIdx.x;
    if (i >= n) return;

    float h_i       = sh_h[i];
    float bw_sq     = bandwidth * bandwidth;
    float inv_bw_sq = 1.0f / bw_sq;
    float inv_n     = 1.0f / (float)n;

    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;
    float k_grad_sum = 0.0f;
    float gk_sum     = 0.0f;
    float ksd_sum    = 0.0f;

    // KSD needs raw score for particle i
    float s_i = compute_ksd ? sh_grad[i] : 0.0f;

    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        float h_j    = sh_h[j];
        float diff   = h_i - h_j;
        float diff_sq = diff * diff;
        float dist_sq = diff_sq * inv_bw_sq;

        float base = 1.0f + dist_sq;
        float K    = 1.0f / base;
        float K_sq = K * K;

        // Full Newton: Hessian-weighted kernel
        H_weighted += sh_hess[j] * K;
        H_weighted += 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        K_sum_norm += K;

        // Stein operator components
        k_grad_sum += K * sh_grad[j];
        gk_sum     += -2.0f * diff * inv_bw_sq * K_sq;  // Legacy sign (attraction)

        // KSD Stein kernel (last iteration only)
        if (compute_ksd) {
            float s_j       = sh_grad[j];
            float grad_x_k  = -2.0f * diff * inv_bw_sq * K_sq;
            float grad_y_k  = -grad_x_k;
            float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
            ksd_sum += K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
        }
    }

    if (compute_ksd) {
        d_ksd_partial[i] = ksd_sum;
    }

    // Normalize Hessian weighting
    H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
    H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);

    // Stein variational gradient with Newton preconditioning
    float phi_i = (k_grad_sum + gk_sum) * inv_n * (0.7f / H_weighted);

    // RMSProp adaptive step size
    float v_prev = v_rmsprop[i];
    float v_new  = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v_rmsprop[i] = v_new;

    float effective_step = step_size * beta_factor;
    float drift = effective_step * phi_i * rsqrtf(v_new + epsilon);

    // Langevin diffusion
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }

    h[i] = p_clamp_logvol(h_i + drift + diffusion);
}

// =============================================================================
// KERNEL 1: PROBE
// =============================================================================
// Computes bandwidth + initial gradient stats at β=0 for annealing schedule.
// Launched as <<<1, 1024, smem>>> — single block, 1 thread per particle.
//
// Shared memory layout:
//   [0, N)      : sh_mu_i  — transition means (persistent within kernel)
//   [N, 2N)     : sh_grad  — gradient at β=0
//   [2N, 3N)    : sh_hess  — curvature at β=0
//   [3N, 3N+32) : scratch  — warp reduction scratch (32 floats)
//
// Outputs:
//   d_bandwidth, d_bandwidth_sq     — bandwidth for Stein kernel
//   d_anneal_stats[4]               — {Σ ll_diff, Σ ll_diff², Σ|grad|, Σ(h-μ)²}

__global__ __launch_bounds__(1024, 1)
void svpf_probe_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float*       __restrict__ log_w,
    const float* __restrict__ d_y,
    // Bandwidth state (read-modify-write)
    float*       __restrict__ d_bandwidth,
    float*       __restrict__ d_bandwidth_sq,
    float*       __restrict__ d_return_ema,
    float*       __restrict__ d_return_var,
    // Anneal stats output
    float*       __restrict__ d_anneal_stats,
    // Params
    int   y_idx,
    float rho,
    float sigma_z,
    float mu,
    float gamma,
    float nu,
    float student_t_const,
    float lik_offset,
    float nu_state,
    float alpha_bw,
    float alpha_ret,
    int   n
) {
    extern __shared__ float smem[];
    float* sh_mu_i  = smem;                // [N]
    float* sh_grad  = smem + n;             // [N]
    float* sh_hess  = smem + 2 * n;         // [N]
    float* scratch  = smem + 3 * n;         // [32] reduction scratch

    int i = threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];

    // =====================================================================
    // PART 1: BANDWIDTH (block reduction over h[])
    // =====================================================================

    float local_sum    = h_i;
    float local_sum_sq = h_i * h_i;
    float local_min    = h_i;
    float local_max    = h_i;

    float sum_val  = block_reduce_sum_1024(local_sum, scratch);
    __syncthreads();
    float sum_sq   = block_reduce_sum_1024(local_sum_sq, scratch);
    __syncthreads();
    float min_val  = block_reduce_min_1024(local_min, scratch);
    __syncthreads();
    float max_val  = block_reduce_max_1024(local_max, scratch);
    __syncthreads();

    // Thread 0: compute and write bandwidth
    __shared__ float s_bandwidth;
    __shared__ float s_h_mean;  // Needed later for anneal stats

    if (threadIdx.x == 0) {
        float inv_n    = 1.0f / (float)n;
        float mean     = sum_val * inv_n;
        float variance = sum_sq * inv_n - mean * mean;
        float spread   = max_val - min_val;

        float bw_sq_new = 2.0f * variance / __logf((float)n + 1.0f);
        bw_sq_new = fmaxf(bw_sq_new, 1e-6f);

        float bw_sq_prev = *d_bandwidth_sq;
        float bw_sq = (bw_sq_prev > 0.0f)
                     ? alpha_bw * bw_sq_new + (1.0f - alpha_bw) * bw_sq_prev
                     : bw_sq_new;

        // Return-volatility adaptive scaling
        float new_return = d_y[y_idx];
        float abs_ret    = fabsf(new_return);
        float ret_ema    = *d_return_ema;
        float ret_var    = *d_return_var;

        ret_ema = (ret_ema > 0.0f)
                ? alpha_ret * abs_ret + (1.0f - alpha_ret) * ret_ema
                : abs_ret;
        ret_var = (ret_var > 0.0f)
                ? alpha_ret * abs_ret * abs_ret + (1.0f - alpha_ret) * ret_var
                : abs_ret * abs_ret;

        *d_return_ema = ret_ema;
        *d_return_var = ret_var;

        float vol_ratio     = abs_ret / fmaxf(ret_ema, 1e-8f);
        float spread_factor = fminf(spread * 0.5f, 2.0f);
        float combined      = fmaxf(vol_ratio, spread_factor);

        float scale = 1.0f - 0.25f * fminf(combined - 1.0f, 2.0f);
        scale = fmaxf(fminf(scale, 1.0f), 0.5f);

        bw_sq *= scale;
        float bw = sqrtf(bw_sq);
        bw = fmaxf(fminf(bw, 2.0f), 0.01f);

        *d_bandwidth_sq = bw_sq;
        *d_bandwidth    = bw;
        s_bandwidth     = bw;
        s_h_mean        = mean;
    }
    __syncthreads();

    // =====================================================================
    // PART 2: COMPUTE TRANSITION MEANS (persistent in shared memory)
    // =====================================================================

    float y_prev = (y_idx > 0) ? d_y[y_idx - 1] : 0.0f;
    {
        float hp          = h_prev[i];
        float vol_prev_k  = __expf(hp * 0.5f);
        float leverage_k  = gamma * y_prev / (vol_prev_k + 1e-8f);
        sh_mu_i[i]        = mu + rho * (hp - mu) + leverage_k;
    }
    __syncthreads();

    // =====================================================================
    // PART 3: GRADIENT + STATS AT β=0
    // =====================================================================
    // Compute gradient at beta=0 to measure likelihood variance for
    // adaptive annealing schedule.

    persistent_gradient_phase(
        sh_mu_i, sh_grad, sh_hess,
        h, log_w,
        d_y[y_idx],
        0.0f,  // beta = 0
        nu, student_t_const, lik_offset,
        nu_state, sigma_z,
        n
    );
    __syncthreads();

    // =====================================================================
    // PART 4: ACCUMULATE ANNEALING STATS
    // =====================================================================
    // Stats needed by CPU to compute beta schedule:
    //   [0] Σ log_w[i]          (mean log-likelihood contribution)
    //   [1] Σ log_w[i]²         (variance of log-likelihood)
    //   [2] Σ |grad[i]|         (gradient magnitude)
    //   [3] Σ (h[i] - h_mean)²  (spatial spread)

    float lw_i    = log_w[i];
    float grad_i  = sh_grad[i];
    float h_dev   = h_i - s_h_mean;

    float s0 = block_reduce_sum_1024(lw_i, scratch);
    __syncthreads();
    float s1 = block_reduce_sum_1024(lw_i * lw_i, scratch);
    __syncthreads();
    float s2 = block_reduce_sum_1024(fabsf(grad_i), scratch);
    __syncthreads();
    float s3 = block_reduce_sum_1024(h_dev * h_dev, scratch);

    if (threadIdx.x == 0) {
        d_anneal_stats[0] = s0;
        d_anneal_stats[1] = s1;
        d_anneal_stats[2] = s2;
        d_anneal_stats[3] = s3;
    }
}

// =============================================================================
// KERNEL 2: PERSISTENT STEIN
// =============================================================================
// Single-launch mega-kernel for the entire annealing loop.
//
// Shared memory layout (4×N + 32 scratch = 16.125 KB for N=1024):
//   [0,    N)     : sh_mu_i  — transition means (loaded once, persistent)
//   [N,   2N)     : sh_grad  — written by gradient phase, read by transport
//   [2N,  3N)     : sh_hess  — written by gradient phase, read by transport
//   [3N,  4N)     : sh_h     — h snapshot for transport O(N²) inner loop
//   [4N, 4N+32)   : scratch  — warp reduction scratch
//
// Flow per iteration:
//   1. Gradient phase: each thread reads h[i] from global + sh_mu_i[j] from
//      shared, writes sh_grad[i] and sh_hess[i]
//   2. __syncthreads
//   3. Transport phase: load sh_h[i]=h[i], syncthreads, each thread reads
//      sh_h[j]/sh_grad[j]/sh_hess[j], computes update, writes h[i] to global
//   4. __syncthreads (ensures all h[] writes visible for next iteration)

__global__ __launch_bounds__(1024, 1)
void svpf_persistent_stein_kernel(
    float*       __restrict__ h,             // [N] particles (read/write)
    const float* __restrict__ h_prev,        // [N] previous step (read-only during loop)
    float*       __restrict__ log_w,         // [N] log importance weights
    float*       __restrict__ v_rmsprop,     // [N] RMSProp state
    curandStatePhilox4_32_10_t* __restrict__ rng,  // [N] RNG states
    const float* __restrict__ d_bandwidth,   // [1] from probe kernel
    float*       __restrict__ d_ksd_partial, // [N] KSD partial sums
    float*       __restrict__ d_ksd,         // [1] final KSD
    float*       __restrict__ d_output_pack, // [8] packed outputs
    float*       __restrict__ d_loglik,      // [1] log-likelihood
    float*       __restrict__ d_vol,         // [1] vol estimate
    float*       __restrict__ d_h_mean,      // [1] h mean (for next predict)
    const float* __restrict__ d_y,           // Observations
    const SVPFPersistentParams params,
    int          n
) {
    extern __shared__ float smem[];
    float* sh_mu_i  = smem;                 // [N]   persistent
    float* sh_grad  = smem + n;              // [N]   per-iteration
    float* sh_hess  = smem + 2 * n;          // [N]   per-iteration
    float* sh_h     = smem + 3 * n;          // [N]   transport snapshot
    float* scratch  = smem + 4 * n;          // [32]  reduction

    int i = threadIdx.x;
    if (i >= n) return;

    // =====================================================================
    // SETUP: Compute transition means (once, persistent in shared)
    // =====================================================================
    float y_prev_val = (params.y_idx > 0) ? d_y[params.y_idx - 1] : 0.0f;
    {
        float hp         = h_prev[i];
        float vol_prev_k = __expf(hp * 0.5f);
        float leverage_k = params.gamma * y_prev_val / (vol_prev_k + 1e-8f);
        sh_mu_i[i]       = params.mu + params.rho * (hp - params.mu) + leverage_k;
    }
    __syncthreads();

    float y_t      = d_y[params.y_idx];
    float bandwidth = *d_bandwidth;

    int n_stages       = params.n_stages;
    int steps_per_beta = params.steps_per_beta;
    int total_steps    = 0;

    // =====================================================================
    // ANNEALING LOOP: stages × steps_per_beta iterations
    // =====================================================================
    for (int stage = 0; stage < n_stages; stage++) {
        float beta        = fminf((float)(stage + 1) / (float)n_stages, 1.0f);
        float beta_factor = sqrtf(beta);

        for (int s = 0; s < steps_per_beta; s++) {
            total_steps++;
            bool is_last = (stage == n_stages - 1)
                        && (s == steps_per_beta - 1);

            // ----- GRADIENT PHASE -----
            persistent_gradient_phase(
                sh_mu_i, sh_grad, sh_hess,
                h, log_w,
                y_t, beta,
                params.nu, params.student_t_const, params.lik_offset,
                params.nu_state, params.sigma_z,
                n
            );
            __syncthreads();

            // ----- TRANSPORT PHASE: snapshot h into shared -----
            sh_h[i] = h[i];
            __syncthreads();

            persistent_transport_phase(
                sh_h, sh_grad, sh_hess,
                h, v_rmsprop, rng,
                bandwidth,
                params.step_size, beta_factor,
                params.temperature, params.rho_rmsprop, params.epsilon,
                is_last, d_ksd_partial,
                n
            );
            __syncthreads();
        }
    }

    // =====================================================================
    // KSD REDUCTION (thread 0 reduces partial sums)
    // =====================================================================
    {
        float ksd_local = d_ksd_partial[i];
        float ksd_sum   = block_reduce_sum_1024(ksd_local, scratch);
        if (threadIdx.x == 0) {
            float inv_n_sq = 1.0f / ((float)n * (float)n);
            float ksd_sq   = ksd_sum * inv_n_sq;
            *d_ksd = sqrtf(fmaxf(ksd_sq, 0.0f));
        }
    }
    __syncthreads();

    // =====================================================================
    // PARTIAL REJUVENATION (Maken et al. 2022)
    // =====================================================================
    if (params.do_rejuvenation) {
        float u = curand_uniform(&rng[i]);
        if (u < params.rejuv_prob) {
            float z = curand_normal(&rng[i]);
            float guide_sample = params.guide_mean + params.guide_std * z;
            float h_old = h[i];
            h[i] = p_clamp_logvol(
                (1.0f - params.rejuv_blend) * h_old
                + params.rejuv_blend * guide_sample
            );
        }
    }
    __syncthreads();

    // =====================================================================
    // OUTPUT REDUCTION: logsumexp + vol mean + h mean
    // =====================================================================
    {
        float h_i_final = h[i];
        float lw_i      = log_w[i];

        // --- logsumexp ---
        __shared__ float s_max_lw;
        float max_lw = block_reduce_max_1024(lw_i, scratch);
        if (threadIdx.x == 0) s_max_lw = max_lw;
        __syncthreads();

        float exp_lw   = __expf(lw_i - s_max_lw);
        float vol_i    = p_safe_exp(h_i_final * 0.5f);

        float sum_exp  = block_reduce_sum_1024(exp_lw, scratch);
        __syncthreads();
        float sum_vol  = block_reduce_sum_1024(vol_i, scratch);
        __syncthreads();
        float sum_h    = block_reduce_sum_1024(h_i_final, scratch);

        if (threadIdx.x == 0) {
            float inv_n   = 1.0f / (float)n;
            float safe_sum = fmaxf(sum_exp * inv_n, 1e-10f);
            float loglik  = s_max_lw + __logf(safe_sum);
            float vol     = sum_vol * inv_n;
            float h_mean  = sum_h * inv_n;

            d_loglik[0]  = loglik;
            d_vol[0]     = vol;
            *d_h_mean    = h_mean;

            // Packed output for single D2H transfer
            d_output_pack[0] = loglik;
            d_output_pack[1] = vol;
            d_output_pack[2] = h_mean;
            d_output_pack[3] = *d_bandwidth;
            d_output_pack[4] = *d_ksd;
        }
    }
}

// =============================================================================
// HOST ORCHESTRATOR: svpf_persistent_step_async
// =============================================================================

void svpf_persistent_step_async(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params
) {
    SVPFOptimizedState* opt = &state->opt_backend;
    int n = state->n_particles;
    cudaStream_t cs = state->stream;

    // Ensure backend is initialized
    // (reuses existing svpf_optimized_init logic — caller must guarantee)

    // --- Effective parameters (adaptive sigma boost) ---
    float effective_mu      = state->mu_state;
    float effective_sigma_z = params->sigma_z;

    if (state->timestep > 0) {
        float vol_est  = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;

        if (return_z > state->sigma_boost_threshold) {
            float severity    = fminf((return_z - state->sigma_boost_threshold) / 3.0f, 1.0f);
            float sigma_boost = 1.0f + (state->sigma_boost_max - 1.0f) * severity;
            effective_sigma_z = params->sigma_z * sigma_boost;
        }
        state->sigma_z_effective = effective_sigma_z;
    }

    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);

    // --- Upload y values ---
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float),
                    cudaMemcpyHostToDevice, cs);

    // =========================================================================
    // 1. PREDICT (antithetic, guided, Student-t) — existing kernel
    // =========================================================================
    {
        int nb_half = ((n / 2) + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;

        svpf_predict_guided_antithetic_kernel<<<nb_half, SVPF_BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev, 1,
            state->rho_up, state->rho_down,
            effective_sigma_z, effective_mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            state->delta_rho, state->delta_sigma,
            state->guided_alpha_base, state->guided_alpha_shock,
            state->guided_innovation_threshold,
            state->student_t_implied_offset,
            state->nu_state,
            n
        );
    }

    // =========================================================================
    // 2. EKF GUIDE (adaptive strength, variance-preserving) — existing kernel
    // =========================================================================
    float current_guide_strength = state->guide_strength_base;
    if (state->timestep > 0) {
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;
        float implied_h = logf(y_t * y_t + 1e-8f) + state->student_t_implied_offset;
        float h_est = logf(vol_est * vol_est + 1e-8f);
        float h_innovation = implied_h - h_est;

        if (h_innovation > 0.0f && return_z > state->guide_innovation_threshold) {
            float severity = fminf((return_z - state->guide_innovation_threshold) / 3.0f, 1.0f);
            current_guide_strength = state->guide_strength_base
                + (state->guide_strength_max - state->guide_strength_base) * severity;
        }
    }

    SVPFParams guide_params = *params;
    guide_params.mu = effective_mu;
    svpf_ekf_update(state, y_t, &guide_params);

    {
        int nb = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;

        svpf_apply_guide_preserving_kernel<<<nb, SVPF_BLOCK_SIZE, 0, cs>>>(
            state->h, opt->d_h_mean_prev, state->guide_mean,
            current_guide_strength, n
        );
    }

    // =========================================================================
    // 3. PROBE KERNEL — bandwidth + gradient stats at β=0
    // =========================================================================
    size_t probe_smem = (3 * n + 32) * sizeof(float);
    cudaMemsetAsync(opt->d_anneal_stats, 0, 4 * sizeof(float), cs);

    svpf_probe_kernel<<<1, n, probe_smem, cs>>>(
        state->h, state->h_prev, state->log_weights,
        opt->d_y_single,
        opt->d_bandwidth, opt->d_bandwidth_sq,
        state->d_return_ema, state->d_return_var,
        opt->d_anneal_stats,
        1,  // y_idx (y_t is at index 1 in d_y_single)
        params->rho, effective_sigma_z, effective_mu,
        params->gamma,
        state->nu, student_t_const, state->lik_offset,
        state->nu_state,
        0.3f,   // alpha_bw
        0.05f,  // alpha_ret
        n
    );

    // =========================================================================
    // 4. SINGLE D2H SYNC — read anneal stats, compute schedule on CPU
    // =========================================================================
    cudaMemcpyAsync(opt->h_anneal_stats_pinned, opt->d_anneal_stats,
                    4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaStreamSynchronize(cs);

    float inv_n = 1.0f / (float)n;
    float mean_ll_diff    = opt->h_anneal_stats_pinned[0] * inv_n;
    float mean_ll_diff_sq = opt->h_anneal_stats_pinned[1] * inv_n;
    float var_ll = fmaxf(mean_ll_diff_sq - mean_ll_diff * mean_ll_diff, 1e-6f);
    float mean_grad = opt->h_anneal_stats_pinned[2] * inv_n;
    float h_std = sqrtf(fmaxf(opt->h_anneal_stats_pinned[3] * inv_n, 1e-8f));

    float d_beta_kl      = sqrtf(2.0f * state->anneal_kl_threshold / (var_ll + 1e-6f));
    float d_beta_grad    = 2.0f / (mean_grad + 1e-6f);
    float d_beta_spatial = (h_std < 0.05f) ? 0.02f : 0.25f;

    float d_beta = fminf(d_beta_kl, fminf(d_beta_grad, d_beta_spatial));
    d_beta = fmaxf(d_beta, 0.05f);
    d_beta = fminf(d_beta, 0.35f);

    int n_stages = (int)ceilf(1.0f / d_beta);
    n_stages = (n_stages < 2) ? 2 : (n_stages > state->anneal_max_stages)
             ? state->anneal_max_stages : n_stages;

    // =========================================================================
    // 5. PERSISTENT STEIN KERNEL — entire annealing loop in one launch
    // =========================================================================
    float base_step = SVPF_STEIN_STEP_SIZE * 0.5f;  // Halved (guide is always on)

    SVPFPersistentParams pparams;
    pparams.rho             = params->rho;
    pparams.sigma_z         = effective_sigma_z;
    pparams.mu              = effective_mu;
    pparams.gamma           = params->gamma;
    pparams.nu              = state->nu;
    pparams.student_t_const = student_t_const;
    pparams.lik_offset      = state->lik_offset;
    pparams.nu_state        = state->nu_state;
    pparams.step_size       = base_step;
    pparams.temperature     = state->temperature;
    pparams.rho_rmsprop     = state->rmsprop_rho;
    pparams.epsilon         = state->rmsprop_eps;
    pparams.guide_mean      = state->guide_mean;
    pparams.guide_std       = sqrtf(fmaxf(state->guide_var, 1e-6f));
    pparams.rejuv_prob      = state->rejuv_prob;
    pparams.rejuv_blend     = state->rejuv_blend;
    pparams.do_rejuvenation = (state->timestep > 10
                            && state->ksd_prev > state->rejuv_ksd_threshold) ? 1 : 0;
    pparams.y_idx           = 1;
    pparams.n_stages        = n_stages;
    pparams.steps_per_beta  = state->anneal_steps_per_beta;

    size_t stein_smem = (4 * n + 32) * sizeof(float);

    svpf_persistent_stein_kernel<<<1, n, stein_smem, cs>>>(
        state->h, state->h_prev, state->log_weights,
        state->d_grad_v, state->rng_states,
        opt->d_bandwidth, opt->d_ksd_partial, opt->d_ksd,
        opt->d_output_pack,
        opt->d_loglik_single, opt->d_vol_single, opt->d_h_mean_prev,
        opt->d_y_single,
        pparams,
        n
    );

    // =========================================================================
    // 6. ASYNC D2H — output pack
    // =========================================================================
    cudaMemcpyAsync(opt->h_output_pinned, opt->d_output_pack,
                    5 * sizeof(float), cudaMemcpyDeviceToHost, cs);

    // Store state for sync phase
    state->stein_steps_used     = n_stages * state->anneal_steps_per_beta;
    state->anneal_stages_used   = n_stages;
    state->anneal_final_var_ll  = var_ll;
    state->anneal_final_h_std   = h_std;
    opt->pending_y_t            = y_t;
    opt->pending_params         = (const void*)params;
}

// =============================================================================
// SYNC OUTPUTS — identical to original svpf_sync_outputs
// =============================================================================
// Can reuse existing svpf_sync_outputs() directly; included here for
// self-contained compilation.

void svpf_persistent_sync_outputs(
    SVPFState* state,
    float* h_loglik_out,
    float* h_vol_out,
    float* h_mean_out
) {
    SVPFOptimizedState* opt = &state->opt_backend;
    cudaStreamSynchronize(state->stream);

    float* r = opt->h_output_pinned;
    float h_mean_local    = r[2];
    float bandwidth_local = r[3];
    float vol_local       = r[1];
    float ksd_local       = r[4];

    if (h_loglik_out) *h_loglik_out = r[0];
    if (h_vol_out)    *h_vol_out    = vol_local;
    if (h_mean_out)   *h_mean_out   = h_mean_local;

    state->vol_prev  = vol_local;
    state->ksd_prev  = ksd_local;

    if (state->timestep > 10) {
        // Adaptive mu (Kalman update)
        float P_pred = state->mu_var + state->mu_process_var;
        float R      = state->mu_obs_var_scale * bandwidth_local * bandwidth_local;
        float K      = P_pred / (P_pred + R + 1e-8f);
        float mu_new = state->mu_state + K * (h_mean_local - state->mu_state);
        state->mu_state = fminf(fmaxf(mu_new, state->mu_min), state->mu_max);
        state->mu_var   = (1.0f - K) * P_pred;
    }

    state->timestep++;
}

// =============================================================================
// SYNCHRONOUS CONVENIENCE
// =============================================================================

void svpf_persistent_step(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out,
    float* h_mean_out
) {
    svpf_persistent_step_async(state, y_t, y_prev, params);
    svpf_persistent_sync_outputs(state, h_loglik_out, h_vol_out, h_mean_out);
}
