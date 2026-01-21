/**
 * @file svpf.cuh
 * @brief Stein Variational Particle Filter for Stochastic Volatility
 * 
 * Purpose: Real-time volatility tracking with crash robustness.
 * 
 * Key features:
 * - Stein transport prevents particle degeneracy
 * - Student-t likelihood handles tail events
 * - Fewer particles needed vs bootstrap PF (500 vs 3000)
 * 
 * Algorithm (Fan et al. 2021, arXiv:2106.10568):
 * - Prior is Gaussian MIXTURE over all particles: p(h_t|Z_{t-1}) = (1/N) Σ_i p(h_t|h_{t-1}^i)
 * - This requires O(N²) gradient computation per Stein iteration
 * - Weights use exact Student-t likelihood for unbiased importance sampling
 * - Gradients use log-squared approximation for robust linear transport
 * 
 * Usage (single-step, real-time):
 *   SVPFState* filter = svpf_create(1024, 10, 5.0f, stream);
 *   svpf_initialize(filter, &params, seed);
 *   for each observation y_t:
 *       svpf_step_adaptive(filter, y_t, y_prev, &params, &loglik, &vol, &h_mean);
 *   svpf_destroy(filter);
 * 
 * Memory Layout: Structure of Arrays (SoA) for coalesced GPU access
 * 
 * References:
 * - Liu & Wang (2016): SVGD algorithm
 * - Fan et al. (2021): Stein Particle Filtering (arXiv:2106.10568)
 */

#ifndef SVPF_CUH
#define SVPF_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CONFIGURATION
// =============================================================================

#define SVPF_DEFAULT_PARTICLES     512
#define SVPF_DEFAULT_STEIN_STEPS   5
#define SVPF_DEFAULT_NU            5.0f
#define SVPF_STEIN_STEP_SIZE       0.1f
#define SVPF_BANDWIDTH_MIN         0.01f
#define SVPF_BANDWIDTH_MAX         10.0f
#define SVPF_H_MIN                 -15.0f
#define SVPF_H_MAX                 5.0f
#define SVPF_BLOCK_SIZE            256

// Threshold for small N optimizations (persistent CTA path)
#define SVPF_SMALL_N_THRESHOLD     4096

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * @brief Optimized backend state for batch/graph processing
 * 
 * Embedded in SVPFState for thread safety - each filter instance
 * has its own optimization buffers.
 */
typedef struct {
    // CUB temporary storage
    void* d_temp_storage;
    size_t temp_storage_bytes;
    
    // Device scalars
    float* d_max_log_w;
    float* d_sum_exp;
    float* d_bandwidth;
    float* d_bandwidth_sq;
    
    // Stein computation buffers
    float* d_exp_w;
    float* d_phi;
    
    // Mixture prior fix: separate likelihood gradient buffer
    // Required for correct O(N²) mixture prior + O(N) likelihood decomposition
    float* d_grad_lik;
    
    // Newton-Stein buffers (Hessian preconditioning)
    float* d_precond_grad;    // H^{-1} * grad (preconditioned gradient)
    float* d_inv_hessian;     // H^{-1} (inverse Hessian per particle)
    
    // Particle-local parameters: h_mean from previous step
    float* d_h_mean_prev;
    
    // Single-step API buffers (avoid malloc in hot loop)
    float* d_y_single;
    float* d_loglik_single;
    float* d_vol_single;
    
    // Capacity tracking
    int allocated_n;
    bool initialized;
} SVPFOptimizedState;

/**
 * @brief SV model parameters
 * 
 * AR(1) log-volatility with leverage effect:
 *   h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps_t + gamma*y_{t-1}/exp(h_{t-1}/2)
 * 
 * Observation: y_t = exp(h_t/2) * eta_t, eta_t ~ Student-t(nu)
 */
typedef struct {
    float rho;      // Persistence (0 < rho < 1, typically 0.9-0.99)
    float sigma_z;  // Vol-of-vol (typically 0.1-0.3)
    float mu;       // Long-run mean log-volatility (typically -5 to -3)
    float gamma;    // Leverage effect (typically -0.5 to 0 for equities)
} SVPFParams;

/**
 * @brief SVPF filter state (SoA layout for GPU)
 * 
 * Key implementation notes:
 * 
 * MIXTURE PRIOR (Fan et al. 2021, Eq. 6):
 *   The prior at time t is a Gaussian mixture:
 *     p(h_t | Z_{t-1}) = (1/N) Σ_i N(h_t; μ_i, σ_z²)
 *   where μ_i = μ + ρ(h_{t-1}^i - μ)
 * 
 *   This means each particle j feels attraction from ALL prior means,
 *   not just its own. The gradient is:
 *     ∇_h log p_prior(h_j) = Σ_i r_i(h_j) * (-(h_j - μ_i)/σ_z²)
 *   where r_i is the responsibility (soft assignment to component i).
 * 
 *   This requires O(N²) computation but is essential for correct filtering.
 * 
 * HYBRID LIKELIHOOD STRATEGY:
 *   - Weights: Exact Student-t (unbiased importance sampling)
 *   - Gradients: Log-squared approximation (robust linear transport)
 *   
 *   The log-squared gradient provides a "parabolic bowl" that always
 *   pulls particles toward the observation, even from far away.
 *   The Student-t gradient creates a "volcano" that saturates for
 *   large deviations, causing particles to get stuck.
 */
typedef struct {
    // Particle states
    float* h;           // [N] Current log-volatility particles
    float* h_prev;      // [N] Previous step (for AR(1) prior)
    float* h_pred;      // [N] Predicted particles (before Stein)
    
    // Stein computation workspace
    float* grad_log_p;  // [N] Gradient of log posterior (prior + likelihood)
    float* kernel_sum;  // [N] Sum of kernel weights (attraction)
    float* grad_kernel_sum; // [N] Sum of kernel gradients (repulsion)
    
    // Likelihood computation
    float* log_weights; // [N] Log importance weights
    
    // Bandwidth computation
    float* d_h_centered;  // [N] Centered particles for variance computation
    
    // === ADAPTIVE SVPF ADDITIONS ===
    // Per-particle RMSProp momentum (for SVLD)
    float* d_grad_v;       // [N] Second moment (uncentered variance)
    
    // Regime detection for bandwidth scaling
    float* d_return_ema;   // Scalar: EMA of |returns|
    float* d_return_var;   // Scalar: EMA of return variance
    float* d_bw_alpha;     // Scalar: adaptive bandwidth alpha
    // ===============================
    
    // RNG states
    curandStatePhilox4_32_10_t* rng_states;  // [N] CURAND Philox states
    
    // Reduction workspace
    float* d_reduce_buf;
    float* d_temp;
    void* d_cub_temp;
    size_t cub_temp_bytes;
    
    // Device scalars
    float* d_scalar_max;
    float* d_scalar_sum;
    float* d_scalar_mean;
    float* d_scalar_bandwidth;
    float* d_y_prev;
    
    // Result buffer
    float* d_result_loglik;
    float* d_result_vol_mean;
    float* d_result_h_mean;
    
    // Configuration
    int n_particles;
    int n_stein_steps;
    float nu;               // Student-t degrees of freedom
    float student_t_const;  // Precomputed: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*nu)
    int timestep;
    float y_prev;
    cudaStream_t stream;
    
    // Adaptive SVPF config
    int use_svld;           // Enable SVLD (Langevin noise) - 0=SVGD, 1=SVLD
    int use_annealing;      // Enable annealed Stein
    int n_anneal_steps;     // Number of annealing steps (2-3)
    float temperature;      // Langevin temperature: 0=SVGD, 1=SVLD, >1=exploration
    float rmsprop_rho;      // RMSProp decay (0.9-0.99)
    float rmsprop_eps;      // RMSProp epsilon (1e-6)
    
    // Mixture Innovation Model (MIM) config
    int use_mim;            // Enable MIM predict (vs standard Gaussian)
    float mim_jump_prob;    // Probability of jump component (e.g., 0.05)
    float mim_jump_scale;   // Scale factor for jump component std (e.g., 5.0)
    
    // Particle-local parameters config
    // Key insight: DGP has θ(z), σ(z) — params depend on latent z
    // We use h deviation from mean as proxy: high h → likely high z
    int use_local_params;   // Enable particle-local ρ and σ
    float delta_rho;        // Rho sensitivity to h deviation (e.g., 0.02)
    float delta_sigma;      // Sigma sensitivity to |h deviation| (e.g., 0.1)
    
    // Asymmetric persistence config
    int use_asymmetric_rho; // Enable asymmetric rho (vol spikes fast, decays slow)
    float rho_up;           // Persistence when vol increasing (e.g., 0.98)
    float rho_down;         // Persistence when vol decreasing (e.g., 0.93)
    
    // Newton-Stein config (Hessian preconditioning)
    // Moves particles along H^{-1} * grad instead of grad
    // Benefits: adaptive step size based on local curvature
    int use_newton;         // Enable Newton-Stein (0=standard SVGD, 1=Newton)
    
    // Guided Prediction config (Lookahead / APF-style)
    // Standard predict is REACTIVE: scatters blindly, then corrects.
    // Guided predict is PROACTIVE: peeks at y_t to know where to go.
    // Proposal: h ~ N((1-α)μ_prior + α·μ_implied, σ²)
    // where μ_implied = log(y_t²) + 1.27 (instantaneous implied vol)
    //
    // INNOVATION GATING: Only activate when model is SURPRISED
    // - Model fits well: innovation low → α ≈ 0 → trust prior
    // - Model lags: innovation high → α > 0 → use guidance
    int use_guided;              // Enable guided predict (0=standard, 1=lookahead)
    float guided_alpha_base;     // Alpha when model fits (e.g., 0.0 - trust prior)
    float guided_alpha_shock;    // Alpha when model fails (e.g., 0.5 - trust observation)
    float guided_innovation_threshold; // z-score threshold for "surprise" (e.g., 1.5)
    
    // Guide density (EKF) config
    int use_guide;          // Enable EKF guide density
    int use_guide_preserving; // Use variance-preserving guide (vs contraction)
    float guide_strength;   // How much to pull particles toward guide (0.1-0.3)
    float guide_mean;       // EKF posterior mean (m_t)
    float guide_var;        // EKF posterior variance (P_t)
    float guide_K;          // Kalman gain (for debugging)
    int guide_initialized;  // Whether guide has been initialized
    
    // Optimized backend (embedded for thread safety)
    SVPFOptimizedState opt_backend;
    
} SVPFState;

/**
 * @brief Result of one SVPF filtering step
 */
typedef struct {
    float log_lik_increment;  // log p(y_t | y_{1:t-1}, theta)
    float vol_mean;           // E[exp(h/2)]
    float vol_std;            // Std[exp(h/2)]
    float h_mean;             // E[h]
} SVPFResult;

// =============================================================================
// API: Core Filter Functions (svpf_kernels.cu)
// =============================================================================

/**
 * @brief Create SVPF filter
 * 
 * @param n_particles Number of particles (recommend 256-1024, power of 2)
 * @param n_stein_steps Stein iterations per timestep (recommend 3-10)
 * @param nu Student-t degrees of freedom (recommend 5.0 for fat tails)
 * @param stream CUDA stream (NULL for default stream)
 * @return Initialized state, or NULL on error
 */
SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream);

/**
 * @brief Free SVPF filter
 */
void svpf_destroy(SVPFState* state);

/**
 * @brief Initialize particles from stationary distribution
 * 
 * @param state SVPF state
 * @param params Model parameters
 * @param seed Random seed for reproducibility
 */
void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed);

/**
 * @brief Process one observation (main filtering step)
 * 
 * @param state SVPF state
 * @param y_t Observation (return) at time t
 * @param params Model parameters
 * @param result Output: volatility estimate and likelihood
 */
void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result);

/**
 * @brief Process observation with seeded RNG (for SMC²/CPMMH)
 */
void svpf_step_seeded(SVPFState* state, float y_t, const SVPFParams* params,
                      unsigned long long rng_seed, SVPFResult* result);

// =============================================================================
// API: Batch Processing (svpf_kernels.cu)
// =============================================================================

/**
 * @brief Process entire observation sequence
 * 
 * @param state SVPF state
 * @param h_observations Host array [T] of observations
 * @param T Number of observations
 * @param params Model parameters
 * @param h_loglik_out Host array [T] for log-likelihood outputs
 * @param h_vol_out Host array [T] for volatility outputs (can be NULL)
 */
void svpf_run_sequence(
    SVPFState* state,
    const float* h_observations,
    int T,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out
);

/**
 * @brief Process sequence with data already on GPU
 */
void svpf_run_sequence_device(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
);

// =============================================================================
// API: Optimized (svpf_optimized.cu) - PRODUCTION USE
// =============================================================================

/**
 * @brief OPTIMIZED: Single step (for real-time/HFT usage)
 * 
 * Uses pre-allocated buffers (zero malloc in hot loop).
 * 
 * @param state SVPF state
 * @param y_t Current observation
 * @param y_prev Previous observation (for leverage)
 * @param params Model parameters
 * @param h_loglik_out Host pointer for log-likelihood output (can be NULL)
 * @param h_vol_out Host pointer for volatility output (can be NULL)
 */
void svpf_step_optimized(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out
);

/**
 * @brief ADAPTIVE SVPF: Single step with all improvements
 * 
 * Implements Preconditioned Stein Variational Langevin Descent with
 * CORRECT mixture prior (O(N²)) per Fan et al. 2021:
 * 
 *   1. Mixture Prior Gradient (O(N²)) - each particle feels pull from all prior means
 *   2. Log-Squared Likelihood Gradient - robust linear transport (no volcano collapse)
 *   3. Mixture Innovation Model (MIM) - fat-tailed predict for scout particles
 *   4. Asymmetric ρ - vol spikes fast (ρ_up), decays slow (ρ_down)
 *   5. Adaptive bandwidth α scaling (tighter kernel during high vol)
 *   6. Annealed Stein updates (β schedule: 0.3 → 0.65 → 1.0)
 *   7. Fused RMSProp + Langevin diffusion (SVLD for diversity)
 * 
 * Configure via SVPFState fields:
 *   - use_mim: Enable Mixture Innovation (default: 1)
 *   - mim_jump_prob: Probability of jump component (default: 0.05)
 *   - mim_jump_scale: Scale factor for jump std (default: 5.0)
 *   - use_asymmetric_rho: Enable asymmetric persistence (default: 1)
 *   - rho_up: Persistence when vol increasing (default: 0.98)
 *   - rho_down: Persistence when vol decreasing (default: 0.93)
 *   - use_svld: Enable SVLD noise (default: 1)
 *   - use_annealing: Enable/disable annealing (default: 1)
 *   - temperature: 0=SVGD, 1=SVLD, >1=extra exploration
 * 
 * @param state SVPF state
 * @param y_t Current observation
 * @param y_prev Previous observation
 * @param params Model parameters
 * @param h_loglik_out Host pointer for log-likelihood output (can be NULL)
 * @param h_vol_out Host pointer for volatility output (can be NULL)
 * @param h_mean_out Host pointer for mean log-vol output (can be NULL)
 */
void svpf_step_adaptive(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out,
    float* h_mean_out
);

/**
 * @brief Internal: Initialize optimized backend (called by svpf_create)
 */
void svpf_optimized_init(SVPFState* state);

/**
 * @brief Internal: Cleanup optimized backend (called by svpf_destroy)
 */
void svpf_optimized_cleanup_state(SVPFState* state);

// =============================================================================
// API: Diagnostics
// =============================================================================

/**
 * @brief Copy particles to host (for diagnostics/plotting)
 */
void svpf_get_particles(const SVPFState* state, float* h_out);

/**
 * @brief Get current particle statistics
 */
void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std);

/**
 * @brief Get effective sample size (ESS) of current particle weights
 * 
 * ESS = 1 / Σ w_i² where w_i are normalized weights.
 * ESS close to N means particles are well-distributed.
 * ESS close to 1 means particle degeneracy (one particle dominates).
 * 
 * Note: SVPF with Stein transport should maintain high ESS without resampling.
 */
float svpf_get_ess(const SVPFState* state);

#ifdef __cplusplus
}
#endif

#endif // SVPF_CUH
