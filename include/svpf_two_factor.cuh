/**
 * @file svpf_two_factor.cuh
 * @brief Two-Factor Stochastic Volatility for SVPF
 * 
 * Model:
 *   h_fast,t = ρ_fast · h_fast,t-1 + σ_fast · ε_fast,t
 *   h_slow,t = ρ_slow · h_slow,t-1 + σ_slow · ε_slow,t
 *   h_t = μ + h_fast,t + h_slow,t
 *   y_t = exp(h_t/2) · z_t,   z_t ~ Student-t(ν)
 * 
 * Fast component (ρ ≈ 0.90): Captures spikes, intraday moves
 * Slow component (ρ ≈ 0.99): Captures regime, trend
 * 
 * Key insight: Coordinate-wise SVGD (not 2D) avoids "fighting" between
 * components along the unidentified h_fast ↔ h_slow tradeoff direction.
 */

#ifndef SVPF_TWO_FACTOR_CUH
#define SVPF_TWO_FACTOR_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// =============================================================================
// ADDITIONAL STRUCT FIELDS
// =============================================================================
// Add these to your existing SVPFState struct

/*
// === Two-Factor Volatility ===
int use_two_factor;          // 0 = single-factor (default), 1 = two-factor

// Particle arrays (device)
float* h_fast;               // [N] fast component
float* h_slow;               // [N] slow component
float* h_fast_prev;          // [N] previous fast
float* h_slow_prev;          // [N] previous slow
float* grad_fast;            // [N] gradient for fast
float* grad_slow;            // [N] gradient for slow
float* d_grad_v_fast;        // [N] RMSProp state for fast
float* d_grad_v_slow;        // [N] RMSProp state for slow

// EKF guide state (2D)
float guide_mean_fast;       // EKF estimate of h_fast
float guide_mean_slow;       // EKF estimate of h_slow
float guide_var_fast;        // EKF variance of h_fast
float guide_var_slow;        // EKF variance of h_slow

// Bandwidth (separate for each component)
float* d_bandwidth_fast;     // [1]
float* d_bandwidth_slow;     // [1]
*/

// =============================================================================
// ADDITIONAL PARAMS FIELDS
// =============================================================================
// Add these to your existing SVPFParams struct

/*
// === Two-Factor Parameters ===
float rho_fast;              // Fast persistence (default: 0.90)
float rho_slow;              // Slow persistence (default: 0.99)
float sigma_fast;            // Fast vol-of-vol (default: 0.15)
float sigma_slow;            // Slow vol-of-vol (default: 0.05)
*/

// =============================================================================
// DEFAULT PARAMETER VALUES
// =============================================================================

#define SVPF_TWO_FACTOR_RHO_FAST_DEFAULT     0.90f
#define SVPF_TWO_FACTOR_RHO_SLOW_DEFAULT     0.99f
#define SVPF_TWO_FACTOR_SIGMA_FAST_DEFAULT   0.15f
#define SVPF_TWO_FACTOR_SIGMA_SLOW_DEFAULT   0.05f

// Bandwidth floors (slow component clusters tightly)
#define SVPF_TWO_FACTOR_BW_FAST_MIN          0.01f
#define SVPF_TWO_FACTOR_BW_SLOW_MIN          0.10f

// =============================================================================
// KERNEL DECLARATIONS
// =============================================================================

/**
 * @brief Initialize two-factor particles
 * 
 * Samples from stationary distribution of each component:
 *   h_fast ~ N(0, σ_fast² / (1 - ρ_fast²))
 *   h_slow ~ N(0, σ_slow² / (1 - ρ_slow²))
 */
__global__ void svpf_init_two_factor_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    int n
);

/**
 * @brief Predict step for two-factor model
 * 
 * Propagates both components according to AR(1) dynamics.
 * MIM jumps applied to h_fast only (spikes go to fast component).
 */
__global__ void svpf_predict_two_factor_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    float* __restrict__ h_fast_prev,
    float* __restrict__ h_slow_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    int use_mim, float mim_jump_prob, float mim_jump_scale,
    int n
);

/**
 * @brief Guided predict for two-factor model
 * 
 * Same as predict but with APF-style lookahead bias toward observation.
 * Guide applied to h_fast (faster response to surprises).
 */
__global__ void svpf_predict_two_factor_guided_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    float* __restrict__ h_fast_prev,
    float* __restrict__ h_slow_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    float mu,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    int use_mim, float mim_jump_prob, float mim_jump_scale,
    float alpha_base, float alpha_shock, float innovation_threshold,
    float implied_offset,
    int n
);

/**
 * @brief Fused gradient computation for two-factor model
 * 
 * Computes:
 *   grad_fast[i] = prior_grad_fast + β * lik_grad
 *   grad_slow[i] = prior_grad_slow + β * lik_grad
 * 
 * Likelihood gradient is SHARED (both components affect observed y).
 * Prior gradients are SEPARATE (each component has own dynamics).
 */
__global__ void svpf_gradient_two_factor_kernel(
    const float* __restrict__ h_fast,
    const float* __restrict__ h_slow,
    const float* __restrict__ h_fast_prev,
    const float* __restrict__ h_slow_prev,
    float* __restrict__ grad_fast,
    float* __restrict__ grad_slow,
    float* __restrict__ log_w,
    float y_t, float mu,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    float beta, float nu, float student_t_const,
    int use_exact_gradient,
    int n
);

/**
 * @brief EKF guide update for two-factor model (Innovation Split)
 * 
 * CRITICAL: Do NOT run two independent EKFs!
 * This splits the single innovation based on relative uncertainty:
 *   K_fast = P_fast / (P_fast + P_slow + R)
 *   K_slow = P_slow / (P_fast + P_slow + R)
 * 
 * Natural allocation: uncertain component gets most of the correction.
 */
__global__ void svpf_ekf_two_factor_kernel(
    float* __restrict__ d_guide_mean_fast,
    float* __restrict__ d_guide_mean_slow,
    float* __restrict__ d_guide_var_fast,
    float* __restrict__ d_guide_var_slow,
    float y_t, float mu,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    float obs_var, float implied_offset
);

/**
 * @brief Apply EKF guide to two-factor particles
 * 
 * Nudges particles toward EKF estimate with variance-weighted split.
 */
__global__ void svpf_apply_guide_two_factor_kernel(
    float* __restrict__ h_fast,
    float* __restrict__ h_slow,
    float guide_mean_fast, float guide_mean_slow,
    float guide_strength,
    int n
);

/**
 * @brief Bandwidth computation for two-factor (Silverman with floors)
 */
__global__ void svpf_bandwidth_two_factor_kernel(
    const float* __restrict__ h_fast,
    const float* __restrict__ h_slow,
    float* __restrict__ d_bandwidth_fast,
    float* __restrict__ d_bandwidth_slow,
    float bw_min_fast, float bw_min_slow,
    int n
);

/**
 * @brief Compute outputs for two-factor model
 * 
 * h_mean = μ + mean(h_fast) + mean(h_slow)
 * vol = exp(h_mean / 2)
 */
__global__ void svpf_outputs_two_factor_kernel(
    const float* __restrict__ h_fast,
    const float* __restrict__ h_slow,
    const float* __restrict__ log_w,
    float mu,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    float* __restrict__ d_h_fast_mean,  // Optional: for diagnostics
    float* __restrict__ d_h_slow_mean,  // Optional: for diagnostics
    int n
);

// =============================================================================
// HOST-SIDE HELPERS
// =============================================================================

/**
 * @brief Host-side EKF update for two-factor model
 * 
 * Implements the Innovation Split algorithm:
 *   innovation = implied_h - (μ + h_fast_pred + h_slow_pred)
 *   K_fast = P_fast / (P_fast + P_slow + R)
 *   K_slow = P_slow / (P_fast + P_slow + R)
 */
static inline void svpf_ekf_two_factor_update(
    float* guide_mean_fast,
    float* guide_mean_slow,
    float* guide_var_fast,
    float* guide_var_slow,
    float y_t, float mu,
    float rho_fast, float sigma_fast,
    float rho_slow, float sigma_slow,
    float obs_var, float implied_offset,
    int* initialized
) {
    // Initialize at stationary variance if first call
    if (!(*initialized)) {
        *guide_mean_fast = 0.0f;
        *guide_mean_slow = 0.0f;
        *guide_var_fast = (sigma_fast * sigma_fast) / (1.0f - rho_fast * rho_fast);
        *guide_var_slow = (sigma_slow * sigma_slow) / (1.0f - rho_slow * rho_slow);
        *initialized = 1;
    }
    
    // Predict step (AR(1) dynamics, mean-zero components)
    float h_fast_pred = rho_fast * (*guide_mean_fast);
    float h_slow_pred = rho_slow * (*guide_mean_slow);
    
    float P_fast_pred = rho_fast * rho_fast * (*guide_var_fast) + sigma_fast * sigma_fast;
    float P_slow_pred = rho_slow * rho_slow * (*guide_var_slow) + sigma_slow * sigma_slow;
    
    // Combined prediction
    float h_total_pred = mu + h_fast_pred + h_slow_pred;
    
    // Observation: log(y²) ≈ h + offset
    float log_y2 = logf(y_t * y_t + 1e-8f);
    float implied_h = log_y2 + implied_offset;
    
    // Innovation
    float innovation = implied_h - h_total_pred;
    
    // Total uncertainty (Innovation Split)
    float S = P_fast_pred + P_slow_pred + obs_var;
    
    // Kalman gains (weighted by relative uncertainty)
    float K_fast = P_fast_pred / (S + 1e-8f);
    float K_slow = P_slow_pred / (S + 1e-8f);
    
    // Update means
    *guide_mean_fast = h_fast_pred + K_fast * innovation;
    *guide_mean_slow = h_slow_pred + K_slow * innovation;
    
    // Update variances (Joseph form for numerical stability)
    *guide_var_fast = (1.0f - K_fast) * P_fast_pred;
    *guide_var_slow = (1.0f - K_slow) * P_slow_pred;
    
    // Clamp variances to prevent collapse
    *guide_var_fast = fmaxf(*guide_var_fast, 1e-6f);
    *guide_var_slow = fmaxf(*guide_var_slow, 1e-6f);
}

/**
 * @brief Set default two-factor parameters
 */
static inline void svpf_two_factor_set_defaults(
    float* rho_fast, float* sigma_fast,
    float* rho_slow, float* sigma_slow
) {
    *rho_fast = SVPF_TWO_FACTOR_RHO_FAST_DEFAULT;
    *rho_slow = SVPF_TWO_FACTOR_RHO_SLOW_DEFAULT;
    *sigma_fast = SVPF_TWO_FACTOR_SIGMA_FAST_DEFAULT;
    *sigma_slow = SVPF_TWO_FACTOR_SIGMA_SLOW_DEFAULT;
}

#endif // SVPF_TWO_FACTOR_CUH
