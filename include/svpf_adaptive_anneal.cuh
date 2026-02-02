/**
 * @file svpf_adaptive_anneal.cuh
 * @brief Adaptive Beta Annealing for SVPF Stein Transport
 * 
 * Uses KL-divergence constraint approximation with safety interlocks:
 * 1. KL-based step: Var(log_lik) controls distribution stability
 * 2. Gradient-based step: |grad| controls physical stability  
 * 3. Spatial diversity: std(h) prevents false confidence from mode collapse
 * 
 * The algorithm runs variable number of stages (not fixed 5), with
 * each stage doing configurable Stein steps before re-evaluating beta.
 */

#ifndef SVPF_ADAPTIVE_ANNEAL_CUH
#define SVPF_ADAPTIVE_ANNEAL_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// =============================================================================
// CONFIGURATION
// =============================================================================

// KL threshold (0.5 ≈ ESS drops to N/2, standard SMC choice)
#define ANNEAL_KL_THRESHOLD_DEFAULT  0.5f

// Beta step clamps
#define ANNEAL_DELTA_BETA_MIN  0.01f   // Anti-stall: always make progress
#define ANNEAL_DELTA_BETA_MAX  0.25f   // Anti-collapse: never jump too far

// Gradient-based step: d_beta = GRAD_SCALE / (max_grad + eps)
#define ANNEAL_GRAD_SCALE  2.0f

// Spatial diversity threshold (if std(h) < this, particles collapsed)
#define ANNEAL_H_STD_COLLAPSE  0.05f
#define ANNEAL_DELTA_BETA_COLLAPSED  0.02f  // Force tiny steps if collapsed

// Variance clamping (outlier rejection)
#define ANNEAL_VARIANCE_CLAMP_SIGMA  3.0f  // Clamp values > 3σ from mean

// Safety limits
#define ANNEAL_MAX_STAGES  50   // Never exceed this many stages
#define ANNEAL_STEPS_PER_BETA_DEFAULT  1  // Stein steps per beta update

// =============================================================================
// ADAPTIVE ANNEAL STATE
// =============================================================================

typedef struct {
    // Configuration
    float kl_threshold;
    int steps_per_beta;
    int max_stages;
    
    // Runtime state
    float beta;
    int stage_count;
    int total_stein_steps;
    
    // Diagnostics (host-side, updated after each stage)
    float last_var_ll;
    float last_mean_grad;
    float last_h_std;
    float last_delta_beta;
    
    // Device buffers (allocated externally, just pointers here)
    float* d_stats;     // [4]: mean_ll, var_ll, mean_grad, h_std
    float* d_temp;      // [n]: temporary for reductions
    
    // Pinned host memory for fast D2H
    float* h_stats_pinned;  // [4]
    
    bool initialized;
} AdaptiveAnnealState;

// =============================================================================
// REDUCTION KERNEL: Compute All Stats in One Pass
// =============================================================================
/**
 * Computes in parallel:
 *   - mean(log_w), var(log_w) with outlier clamping
 *   - mean(|grad|)
 *   - std(h)
 * 
 * Output: d_stats[4] = {mean_ll, var_ll, mean_abs_grad, h_std}
 */
__global__ void svpf_anneal_stats_kernel(
    const float* __restrict__ log_w,
    const float* __restrict__ grad,
    const float* __restrict__ h,
    float* __restrict__ d_stats,  // Output [4]
    int n
) {
    __shared__ float s_sum_ll_diff[256];
    __shared__ float s_sum_ll_diff_sq[256];
    __shared__ float s_sum_grad[256];
    __shared__ float s_sum_h_diff[256];
    __shared__ float s_sum_h_diff_sq[256];
    __shared__ int s_count[256];
    
    // Centering values for numerical stability
    __shared__ float s_center_ll;
    __shared__ float s_center_h;
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread 0 loads centering values (first element of each array)
    if (tid == 0) {
        s_center_ll = log_w[0];
        s_center_h = h[0];
    }
    __syncthreads();
    
    float center_ll = s_center_ll;
    float center_h = s_center_h;
    
    // Initialize accumulators
    float sum_ll_diff = 0.0f;
    float sum_ll_diff_sq = 0.0f;
    float sum_grad = 0.0f;
    float sum_h_diff = 0.0f;
    float sum_h_diff_sq = 0.0f;
    int count = 0;
    
    // Grid-stride loop with centering
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        // Log-likelihood: centered and clamped
        float ll = log_w[idx];
        ll = fmaxf(fminf(ll, -1.0f), -100.0f);  // Clamp outliers
        float ll_diff = ll - center_ll;
        
        // Gradient: absolute value
        float g = fabsf(grad[idx]);
        
        // Particle position: centered
        float hv = h[idx];
        float h_diff = hv - center_h;
        
        sum_ll_diff += ll_diff;
        sum_ll_diff_sq += ll_diff * ll_diff;
        sum_grad += g;
        sum_h_diff += h_diff;
        sum_h_diff_sq += h_diff * h_diff;
        count++;
    }
    
    // Store to shared memory
    s_sum_ll_diff[tid] = sum_ll_diff;
    s_sum_ll_diff_sq[tid] = sum_ll_diff_sq;
    s_sum_grad[tid] = sum_grad;
    s_sum_h_diff[tid] = sum_h_diff;
    s_sum_h_diff_sq[tid] = sum_h_diff_sq;
    s_count[tid] = count;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_ll_diff[tid] += s_sum_ll_diff[tid + s];
            s_sum_ll_diff_sq[tid] += s_sum_ll_diff_sq[tid + s];
            s_sum_grad[tid] += s_sum_grad[tid + s];
            s_sum_h_diff[tid] += s_sum_h_diff[tid + s];
            s_sum_h_diff_sq[tid] += s_sum_h_diff_sq[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes final stats
    if (tid == 0 && blockIdx.x == 0) {
        float total = (float)s_count[0];
        if (total < 1.0f) total = 1.0f;
        
        // Log-likelihood: mean and variance (shift-invariant)
        float mean_ll_diff = s_sum_ll_diff[0] / total;
        float mean_ll_diff_sq = s_sum_ll_diff_sq[0] / total;
        float var_ll = mean_ll_diff_sq - (mean_ll_diff * mean_ll_diff);
        var_ll = fmaxf(var_ll, 1e-6f);  // Numerical stability
        
        // Mean log-likelihood (add center back)
        float mean_ll = mean_ll_diff + center_ll;
        
        // Mean absolute gradient
        float mean_grad = s_sum_grad[0] / total;
        
        // Particle spread: std(h) (shift-invariant)
        float mean_h_diff = s_sum_h_diff[0] / total;
        float mean_h_diff_sq = s_sum_h_diff_sq[0] / total;
        float var_h = mean_h_diff_sq - (mean_h_diff * mean_h_diff);
        float std_h = sqrtf(fmaxf(var_h, 1e-8f));
        
        d_stats[0] = mean_ll;
        d_stats[1] = var_ll;
        d_stats[2] = mean_grad;
        d_stats[3] = std_h;
    }
}

// =============================================================================
// CLAMPED VARIANCE KERNEL (More Robust)
// =============================================================================
/**
 * Two-pass approach for robust variance:
 * Pass 1: Compute mean and std of log_w
 * Pass 2: Recompute variance excluding values > 3σ from mean
 */
__global__ void svpf_anneal_stats_robust_kernel(
    const float* __restrict__ log_w,
    const float* __restrict__ grad,
    const float* __restrict__ h,
    float* __restrict__ d_stats,  // Output [4]
    float mean_ll_prior,          // From first pass
    float std_ll_prior,           // From first pass
    float clamp_sigma,            // How many σ to clamp (e.g., 3.0)
    int n
) {
    __shared__ float s_sum_ll[256];
    __shared__ float s_sum_ll_sq[256];
    __shared__ float s_sum_grad[256];
    __shared__ float s_sum_h[256];
    __shared__ float s_sum_h_sq[256];
    __shared__ int s_valid_count[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float ll_lo = mean_ll_prior - clamp_sigma * std_ll_prior;
    float ll_hi = mean_ll_prior + clamp_sigma * std_ll_prior;
    
    float sum_ll = 0.0f;
    float sum_ll_sq = 0.0f;
    float sum_grad = 0.0f;
    float sum_h = 0.0f;
    float sum_h_sq = 0.0f;
    int valid_count = 0;
    
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        float ll = log_w[idx];
        float g = fabsf(grad[idx]);
        float hv = h[idx];
        
        // Clamp log-likelihood for variance computation
        float ll_clamped = fminf(fmaxf(ll, ll_lo), ll_hi);
        
        // Only count non-clamped values for variance
        if (ll >= ll_lo && ll <= ll_hi) {
            sum_ll += ll_clamped;
            sum_ll_sq += ll_clamped * ll_clamped;
            valid_count++;
        }
        
        sum_grad += g;
        sum_h += hv;
        sum_h_sq += hv * hv;
    }
    
    s_sum_ll[tid] = sum_ll;
    s_sum_ll_sq[tid] = sum_ll_sq;
    s_sum_grad[tid] = sum_grad;
    s_sum_h[tid] = sum_h;
    s_sum_h_sq[tid] = sum_h_sq;
    s_valid_count[tid] = valid_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_ll[tid] += s_sum_ll[tid + s];
            s_sum_ll_sq[tid] += s_sum_ll_sq[tid + s];
            s_sum_grad[tid] += s_sum_grad[tid + s];
            s_sum_h[tid] += s_sum_h[tid + s];
            s_sum_h_sq[tid] += s_sum_h_sq[tid + s];
            s_valid_count[tid] += s_valid_count[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x == 0) {
        float valid = (float)s_valid_count[0];
        if (valid < 1.0f) valid = 1.0f;
        float total = (float)n;
        
        // Robust variance (only from non-outliers)
        float mean_ll = s_sum_ll[0] / valid;
        float mean_ll_sq = s_sum_ll_sq[0] / valid;
        float var_ll = mean_ll_sq - mean_ll * mean_ll;
        var_ll = fmaxf(var_ll, 1e-6f);
        
        // Mean gradient (all particles)
        float mean_grad = s_sum_grad[0] / total;
        
        // Particle spread (all particles)
        float mean_h = s_sum_h[0] / total;
        float mean_h_sq = s_sum_h_sq[0] / total;
        float var_h = mean_h_sq - mean_h * mean_h;
        float std_h = sqrtf(fmaxf(var_h, 1e-8f));
        
        d_stats[0] = mean_ll;
        d_stats[1] = var_ll;
        d_stats[2] = mean_grad;
        d_stats[3] = std_h;
    }
}

// =============================================================================
// HOST-SIDE FUNCTIONS
// =============================================================================

/**
 * Initialize adaptive anneal state
 */
static inline void adaptive_anneal_init(
    AdaptiveAnnealState* state,
    float kl_threshold,
    int steps_per_beta,
    int max_stages
) {
    state->kl_threshold = kl_threshold > 0 ? kl_threshold : ANNEAL_KL_THRESHOLD_DEFAULT;
    state->steps_per_beta = steps_per_beta > 0 ? steps_per_beta : ANNEAL_STEPS_PER_BETA_DEFAULT;
    state->max_stages = max_stages > 0 ? max_stages : ANNEAL_MAX_STAGES;
    
    state->beta = 0.0f;
    state->stage_count = 0;
    state->total_stein_steps = 0;
    
    state->last_var_ll = 0.0f;
    state->last_mean_grad = 0.0f;
    state->last_h_std = 0.0f;
    state->last_delta_beta = 0.0f;
    
    state->initialized = true;
}

/**
 * Reset state for new timestep
 */
static inline void adaptive_anneal_reset(AdaptiveAnnealState* state) {
    state->beta = 0.0f;
    state->stage_count = 0;
    state->total_stein_steps = 0;
}

/**
 * Compute delta_beta from stats (CPU-side decision)
 * 
 * Returns the next beta value (clamped to 1.0)
 */
static inline float adaptive_anneal_compute_delta_beta(
    AdaptiveAnnealState* state,
    float var_ll,
    float mean_grad,
    float h_std
) {
    // Store for diagnostics
    state->last_var_ll = var_ll;
    state->last_mean_grad = mean_grad;
    state->last_h_std = h_std;
    
    // === 1. KL-based step (distribution stability) ===
    // d_beta = sqrt(2 * threshold / variance)
    float d_beta_kl = sqrtf(2.0f * state->kl_threshold / (var_ll + 1e-6f));
    
    // === 2. Gradient-based step (physical stability) ===
    // d_beta = scale / (max_grad + eps)
    float d_beta_grad = ANNEAL_GRAD_SCALE / (mean_grad + 1e-6f);
    
    // === 3. Spatial diversity check (anti-collapse) ===
    float d_beta_spatial = ANNEAL_DELTA_BETA_MAX;  // No constraint by default
    if (h_std < ANNEAL_H_STD_COLLAPSE) {
        // Particles collapsed! Force tiny steps
        d_beta_spatial = ANNEAL_DELTA_BETA_COLLAPSED;
    }
    
    // === Take minimum (most conservative) ===
    float d_beta = fminf(d_beta_kl, fminf(d_beta_grad, d_beta_spatial));
    
    // === Clamps ===
    d_beta = fmaxf(d_beta, ANNEAL_DELTA_BETA_MIN);  // Anti-stall
    d_beta = fminf(d_beta, ANNEAL_DELTA_BETA_MAX);  // Anti-collapse
    
    state->last_delta_beta = d_beta;
    
    return d_beta;
}

/**
 * Check if annealing is complete
 */
static inline bool adaptive_anneal_done(const AdaptiveAnnealState* state) {
    return (state->beta >= 1.0f) || (state->stage_count >= state->max_stages);
}

/**
 * Print diagnostic info (for debugging)
 */
static inline void adaptive_anneal_print_diagnostics(const AdaptiveAnnealState* state) {
    printf("[AdaptiveAnneal] beta=%.3f, stages=%d, steps=%d | "
           "var_ll=%.4f, mean_grad=%.4f, h_std=%.4f, d_beta=%.4f\n",
           state->beta, state->stage_count, state->total_stein_steps,
           state->last_var_ll, state->last_mean_grad, 
           state->last_h_std, state->last_delta_beta);
}

#endif // SVPF_ADAPTIVE_ANNEAL_CUH
