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
#include <stdio.h>

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

// Number of floats in graph parameter staging buffer
#define SVPF_GRAPH_PARAMS_SIZE     32

// === Backward Smoothing (Fan et al. 2021 lightweight) ===
#define SVPF_SMOOTH_MAX_LAG 8

// =============================================================================
// STEIN SIGN MODE CONFIGURATION
// =============================================================================
//
// The Stein operator has two terms:
//   φ(x_i) = 1/n Σⱼ [k(xⱼ, xᵢ)·∇log q(xⱼ) + ∇_{xⱼ} k(xⱼ, xᵢ)]
//                    └─── attraction ────┘   └─── repulsion ───┘
//
// For IMQ kernel with diff = x_i - x_j:
//   ∇_{xⱼ} k(xⱼ, xᵢ) = +2·diff/h²·k²
//
// STEIN_SIGN_LEGACY (0): Subtracts kernel gradient (particles attract)
//   - Empirically tuned with MIM/SVLD/guide compensating for lack of repulsion
//   - Production-tested configuration
//
// STEIN_SIGN_PAPER (1): Adds kernel gradient (particles repel)
//   - Mathematically correct per Fan et al. 2021
//   - May require retuning other parameters
//
#define SVPF_STEIN_SIGN_LEGACY  0
#define SVPF_STEIN_SIGN_PAPER   1

#ifndef SVPF_STEIN_SIGN_DEFAULT
#define SVPF_STEIN_SIGN_DEFAULT SVPF_STEIN_SIGN_LEGACY
#endif

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * @brief Device-side parameter staging for CUDA graph execution
 * 
 * Layout of d_params_staging buffer (32 floats):
 * [0]  y_prev           - Previous observation
 * [1]  y_t              - Current observation
 * [2]  guide_mean       - EKF guide mean
 * [3]  beta             - Current annealing factor
 * [4]  step_size        - Stein step size
 * [5]  temp             - SVLD temperature
 * [6]  rho              - AR persistence
 * [7]  sigma_z          - Innovation std
 * [8]  mu               - Mean level
 * [9]  gamma            - Leverage coefficient
 * [10] nu               - Student-t degrees of freedom
 * [11] student_t_const  - Precomputed Student-t normalizing constant
 * [12] rho_up           - Asymmetric rho (up)
 * [13] rho_down         - Asymmetric rho (down)
 * [14] delta_rho        - Particle-local rho sensitivity
 * [15] delta_sigma      - Particle-local sigma sensitivity
 * [16] alpha_base       - Guided alpha (base)
 * [17] alpha_shock      - Guided alpha (shock)
 * [18] innovation_thresh - Guided innovation threshold
 * [19] jump_prob        - MIM jump probability
 * [20] jump_scale       - MIM jump scale
 * [21] guide_strength   - Guide density strength
 * [22] rmsprop_rho      - RMSProp decay
 * [23] rmsprop_eps      - RMSProp epsilon
 * [24-31] reserved      - Future use
 */
typedef struct {
    float y_prev;
    float y_t;
    float guide_mean;
    float beta;
    float step_size;
    float temp;
    float rho;
    float sigma_z;
    float mu;
    float gamma;
    float nu;
    float student_t_const;
    float rho_up;
    float rho_down;
    float delta_rho;
    float delta_sigma;
    float alpha_base;
    float alpha_shock;
    float innovation_thresh;
    float jump_prob;
    float jump_scale;
    float guide_strength;
    float rmsprop_rho;
    float rmsprop_eps;
    float reserved[8];
} SVPFGraphParams;

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
    
    // Guide mean and strength (device-side for graph compatibility)
    float* d_guide_mean;
    float* d_guide_strength;   // Adaptive guide strength
    
    // Single-step API buffers (avoid malloc in hot loop)
    float* d_y_single;
    float* d_loglik_single;
    float* d_vol_single;
    
    // =========================================================================
    // CUDA GRAPH SUPPORT
    // For HFT: captures kernel sequence, replays with ~5μs overhead vs ~100μs+
    // =========================================================================
    
    // Graph handles
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t graph_stream;
    bool graph_captured;
    int graph_n;              // N at capture time (recapture if changed)
    int graph_n_stein;        // Stein steps at capture time
    
    // Device-side parameter staging (kernels read from here)
    // Updated via cudaMemcpyAsync before graph replay
    float* d_params_staging;  // Packed: [y_prev, y_t, guide_mean, beta, step_size, temp, ...]
    
    // Burned-in mu at capture time (for graph invalidation check)
    float mu_captured;
    float sigma_z_captured;
    
    // Pinned host memory for fast D2H transfers
    // Layout: [loglik, vol, h_mean, bandwidth]
    float* h_results_pinned;
    
    // Capacity tracking
    int allocated_n;
    bool initialized;

    float *d_ksd_partial; // Partial sums for KSD reduction [n floats]
    float *d_ksd;         // Final KSD value [1 float]

    float *d_beta_schedule; // [8] Beta schedule for persistent kernel

    // === Heun's Method Buffers ===
    float *d_phi_orig; // Stein operator at original h
    float *d_phi_pred; // Stein operator at predicted h̃
    float *d_h_orig;   // Original h before predictor step
    
    // === Consolidated D2H Output Pack ===
    // Single 32-byte transfer for all outputs (5 floats + padding)
    float* d_output_pack;      // Device: [loglik, vol, h_mean, bandwidth, ksd, pad, pad, pad]
    float* h_output_pinned;    // Pinned host (same layout)
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
    float student_t_implied_offset;  // Precomputed offset for implied h from observation:
                                     //   For Student-t: h_implied = log(y²) - E[log(t²_ν)]
                                     //   E[log(t²_ν)] = log(ν) + ψ(1/2) - ψ(ν/2)
                                     //   We store the negated value so: h_implied = log(y²) + offset
                                     //   For ν→∞ (Gaussian), this approaches 1.27
    int timestep;
    float y_prev;
    cudaStream_t stream;
    
    // Likelihood gradient config
    // lik_offset serves different purposes depending on use_exact_gradient:
    //   SURROGATE (use_exact_gradient=0):
    //     - grad = (log(y²) - h + lik_offset) / R_noise
    //     - Tuned value: ~0.70 for minimal bias
    //   EXACT (use_exact_gradient=1):
    //     - grad = exact_student_t_grad - lik_offset
    //     - Tuned value: ~0.25-0.30 to correct for equilibrium bias
    float lik_offset;       // Bias correction (0.70 for surrogate, 0.27 for exact)
    
    // Exact vs Surrogate likelihood gradient
    // - Exact: d/dh log p(y|h) = -0.5 + 0.5*(nu+1)*A/(1+A) - lik_offset
    //   Mathematically consistent with log_w and Hessian. Saturates at ±nu/2.
    // - Surrogate: (log(y²) - h + lik_offset) / R_noise
    //   Linear (no saturation), but inconsistent with Student-t weights/Hessian.
    int use_exact_gradient; // 0 = surrogate (legacy), 1 = exact Student-t (recommended with nu>=30)
    
    // Adaptive SVPF config
    int use_svld;           // Enable SVLD (Langevin noise) - 0=SVGD, 1=SVLD
    int use_annealing;      // Enable annealed Stein
    int use_adaptive_beta;  // KSD-adaptive beta (Maken 2022): trust prior when particles disagree
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
    
    // Full Newton-Stein (Detommaso et al. 2018)
    // When enabled, uses kernel-weighted Hessian averaging:
    //   Ĥᵢ = Σⱼ [Nπ(xⱼ)·K(xⱼ,xᵢ) + Nk(xⱼ,xᵢ)]
    // Instead of just local Hessian Nπ(xᵢ).
    // Cost: ~1.2x per Stein step (fused into O(N²) loop)
    // Benefit: Better preconditioning when particles span different curvature regions
    int use_full_newton;    // 0 = local Hessian (fast), 1 = kernel-weighted (accurate)
    
    // Guided Prediction config (Lookahead / APF-style)
    // Standard predict is REACTIVE: scatters blindly, then corrects.
    // Guided predict is PROACTIVE: peeks at y_t to know where to go.
    // Proposal: h ~ N((1-α)μ_prior + α·μ_implied, σ²)
    // where μ_implied = log(y_t²) + student_t_implied_offset
    //
    // INNOVATION GATING: Only activate when model is SURPRISED
    // - Model fits well: innovation low → α ≈ 0 → trust prior
    // - Model lags: innovation high → α > 0 → use guidance
    int use_guided;              // Enable guided predict (0=standard, 1=lookahead)
    float guided_alpha_base;     // Alpha when model fits (e.g., 0.0 - trust prior)
    float guided_alpha_shock;    // Alpha when model fails (e.g., 0.5 - trust observation)
    float guided_innovation_threshold; // z-score threshold for "surprise" (e.g., 1.5)
    
    // =========================================================================
    // PARTIAL REJUVENATION (Maken et al. 2022)
    // =========================================================================
    // When KSD stays high after Stein (particles stuck), nudge a fraction
    // toward the EKF guide prediction to help escape local modes.
    int use_rejuvenation;           // Enable partial rejuvenation (default: 1)
    float rejuv_ksd_threshold;      // KSD threshold to trigger (e.g., 0.3)
    float rejuv_prob;               // Fraction of particles to nudge (e.g., 0.3)
    float rejuv_blend;              // How much to blend toward guide (e.g., 0.3)
    
    // Guide density (EKF) config
    int use_guide;          // Enable EKF guide density
    int use_guide_preserving; // Use variance-preserving guide (vs contraction)
    float guide_strength;   // Base guide strength (0.05 default)
    float guide_mean;       // EKF posterior mean (m_t)
    float guide_var;        // EKF posterior variance (P_t)
    float guide_K;          // Kalman gain (for debugging)
    int guide_initialized;  // Whether guide has been initialized
    
    // =========================================================================
    // ADAPTIVE GUIDE STRENGTH: Innovation-gated nudging
    // =========================================================================
    // When innovation is high (model surprised), boost guide strength to
    // "teleport" particles toward EKF estimate. When innovation is low,
    // use base strength to avoid over-correction.
    int use_adaptive_guide;         // Enable adaptive guide strength (default: 0)
    float guide_strength_base;      // Base strength when model fits (e.g., 0.05)
    float guide_strength_max;       // Max strength during surprises (e.g., 0.30)
    float guide_innovation_threshold; // Z-score threshold for boost (e.g., 1.0)
    float vol_prev;                 // Previous vol estimate (for innovation calc)
    
    // =========================================================================
    // ADAPTIVE MU: 1D Kalman Filter on mean level
    // =========================================================================
    // Uses particle confidence (inverse bandwidth) to gate learning rate.
    // - Calm market (low bandwidth): adapt mu quickly to track drift
    // - Crisis (high bandwidth): freeze mu to ignore transient spikes
    int use_adaptive_mu;        // Enable adaptive mu learning (default: 0)
    float mu_state;             // Current mu estimate (Kalman state)
    float mu_var;               // Current mu variance (Kalman P)
    float mu_process_var;       // Process noise Q (how fast mu can drift)
    float mu_obs_var_scale;     // Scale factor for measurement noise R = scale * bandwidth²
    float mu_min;               // Lower bound for mu (e.g., -6.0)
    float mu_max;               // Upper bound for mu (e.g., -1.0)
    
    // =========================================================================
    // ADAPTIVE SIGMA_Z: Innovation-gated vol-of-vol ("Breathing Filter")
    // =========================================================================
    // When innovation is high, particles need to spread faster to catch up.
    // Boost sigma_z proportional to innovation magnitude.
    int use_adaptive_sigma;         // Enable adaptive sigma_z (default: 0)
    float sigma_boost_threshold;    // Z-score threshold to start boosting (e.g., 1.0)
    float sigma_boost_max;          // Maximum boost multiplier (e.g., 3.0 = 3x base)
    float sigma_z_effective;        // Current effective sigma_z (for graph capture)
    
    // =========================================================================
    // STEIN OPERATOR SIGN MODE
    // =========================================================================
    // Controls whether the kernel gradient term adds (repulsion, per paper)
    // or subtracts (attraction, legacy empirical tuning).
    //
    // 0 = SVPF_STEIN_SIGN_LEGACY: gk_sum -= 2*diff*inv_bw_sq*K_sq (attraction)
    // 1 = SVPF_STEIN_SIGN_PAPER:  gk_sum += 2*diff*inv_bw_sq*K_sq (repulsion)
    //
    // Legacy mode works with current MIM/SVLD/guide tuning.
    // Paper mode is mathematically correct but may need parameter retuning.
    int stein_repulsive_sign;
    
    // =========================================================================
    // FAN MODE (Fan et al. 2021 - Weightless SVGD)
    // =========================================================================
    // Pure Stein-based particle filter without importance weights.
    // Key changes when enabled:
    //   1. Uniform weights: log_w = 0 for all particles (no weighting)
    //   2. No annealing: beta = 1.0 always (full likelihood from start)
    //   3. No resampling: Stein repulsion maintains diversity
    //   4. Paper sign: uses correct repulsive sign (+) automatically
    //
    // Theoretical benefit: avoids weight degeneracy that causes variance collapse.
    // Trade-off: relies entirely on Stein operator for posterior approximation.
    int use_fan_mode;  // 0 = hybrid (default), 1 = weightless SVGD
    
    // Optimized backend (embedded for thread safety)
    SVPFOptimizedState opt_backend;

    // === KSD-based Adaptive Stein Steps ===
    int stein_min_steps;             // Minimum Stein iterations (default: 4)
    int stein_max_steps;             // Maximum Stein iterations (default: 12)
    float ksd_improvement_threshold; // Stop if relative improvement < this
                                     // (default: 0.05)
    float ksd_prev;                  // KSD from previous iteration (internal)
    int stein_steps_used; // Diagnostic: how many steps were actually used

    int use_student_t_state; // 0 = Gaussian, 1 = Student-t
    float nu_state;          // Degrees of freedom (recommended: 5-7)

    int use_smoothing;     // 0 = off, 1 = on
    int smooth_lag;        // Window size (1-5 recommended)
    int smooth_output_lag; // Output delay: 0=raw, 1=h[t-1], etc.
    int smooth_head;       // Circular buffer index
    float smooth_h_mean[SVPF_SMOOTH_MAX_LAG]; // Stored h estimates
    float smooth_h_var[SVPF_SMOOTH_MAX_LAG];  // Stored uncertainties
    float smooth_y[SVPF_SMOOTH_MAX_LAG];      // Stored observations

    // === Persistent Kernel Mode ===
    int use_persistent_kernel;       // 0 = standard, 1 = persistent
    int persistent_kernel_supported; // Set at creation

    int use_heun;  // 0 = Euler (default), 1 = Heun's method

    int use_antithetic;

} SVPFState;

/**
 * @brief Result of one SVPF filtering step
 */
typedef struct {
    float log_lik_increment;  // log p(y_t | y_{1:t-1}, theta)
    float vol_mean;           // E[exp(h/2)]
    float vol_std;            // Std[exp(h/2)]
    float h_mean;             // E[h]
    float mu_estimate;        // Current adaptive mu (if enabled)
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
// API: Stein Sign Mode Configuration
// =============================================================================

/**
 * @brief Set Stein operator repulsive sign mode
 * 
 * @param state   Filter state
 * @param mode    SVPF_STEIN_SIGN_LEGACY (0) or SVPF_STEIN_SIGN_PAPER (1)
 * 
 * LEGACY (0): Kernel gradient subtracts (particles attract)
 *   - Empirically tuned with MIM/SVLD/guide providing diversity
 *   - Production-tested, stable configuration
 * 
 * PAPER (1): Kernel gradient adds (particles repel)
 *   - Mathematically correct per Fan et al. 2021
 *   - May allow reducing MIM/SVLD/guide strength
 *   - Requires validation and possible retuning
 * 
 * Call svpf_graph_invalidate() after changing if using CUDA graphs.
 */
static inline void svpf_set_stein_sign_mode(SVPFState* state, int mode) {
    state->stein_repulsive_sign = (mode == SVPF_STEIN_SIGN_PAPER) ? 1 : 0;
}

/**
 * @brief Get current Stein operator sign mode
 * @return SVPF_STEIN_SIGN_LEGACY (0) or SVPF_STEIN_SIGN_PAPER (1)
 */
static inline int svpf_get_stein_sign_mode(const SVPFState* state) {
    return state->stein_repulsive_sign;
}

// =============================================================================
// API: Fan Mode (Weightless SVGD - Fan et al. 2021)
// =============================================================================

/**
 * @brief Enable/disable Fan mode (weightless SVGD)
 * 
 * Fan mode implements pure Stein variational inference without importance weights:
 *   - All particles have uniform weight (log_w = 0)
 *   - No likelihood annealing (beta = 1.0)
 *   - No resampling (Stein repulsion maintains diversity)
 *   - Automatically uses correct repulsive sign
 * 
 * Benefits: Avoids weight degeneracy that causes variance collapse
 * Trade-offs: Relies entirely on Stein operator for posterior approximation
 * 
 * @param state SVPF state
 * @param enable 1 to enable Fan mode, 0 for hybrid mode (default)
 * 
 * Call svpf_graph_invalidate() after changing if using CUDA graphs.
 */
static inline void svpf_set_fan_mode(SVPFState* state, int enable) {
    state->use_fan_mode = enable ? 1 : 0;
    // Fan mode implies paper sign (repulsive)
    if (enable) {
        state->stein_repulsive_sign = SVPF_STEIN_SIGN_PAPER;
    }
}

/**
 * @brief Get current Fan mode status
 * @return 1 if Fan mode enabled, 0 otherwise
 */
static inline int svpf_get_fan_mode(const SVPFState* state) {
    return state->use_fan_mode;
}

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

// =============================================================================
// CUDA GRAPH API (Low-latency for HFT)
// =============================================================================

/**
 * @brief Graph-accelerated SVPF step
 * 
 * Captures the kernel sequence on first call, then replays with minimal
 * CPU overhead (~5μs vs ~100μs+ for regular kernel launches).
 * 
 * Automatic recapture on:
 *   - First call
 *   - Particle count change
 *   - Stein step count change
 * 
 * IMPORTANT: Model parameters (rho, sigma_z, mu, gamma) are "burned in" at
 * capture time. If you change SVPFParams, you MUST call svpf_graph_invalidate()
 * before the next step, or the graph will run with stale parameters!
 * 
 * Filter configuration flags (use_guided, use_newton, use_param_learning, etc.)
 * are also burned in. Call svpf_graph_invalidate() after changing any configuration.
 * 
 * @param state     Filter state
 * @param y_t       Current observation (staged to device before graph launch)
 * @param y_prev    Previous observation (staged to device before graph launch)
 * @param params    Model parameters (BURNED IN at capture - invalidate if changed!)
 * @param h_loglik_out  Output: log-likelihood (optional, can be NULL)
 * @param h_vol_out     Output: volatility estimate (optional, can be NULL)
 * @param h_mean_out    Output: h mean (optional, can be NULL)
 */
void svpf_step_graph(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out,
    float* h_mean_out
);

/**
 * @brief Check if graph is currently captured
 */
bool svpf_graph_is_captured(SVPFState* state);

/**
 * @brief Force graph recapture on next step
 * 
 * Call after changing:
 *   - SVPFParams (rho, sigma_z, mu, gamma)
 *   - Filter configuration (use_guided, use_newton, use_mim, etc.)
 *   - Stein sign mode (stein_repulsive_sign)
 *   - Any other filter settings
 * 
 * The next svpf_step_graph() call will recapture with new parameters.
 */
void svpf_graph_invalidate(SVPFState* state);

// =============================================================================
// GRADIENT DIAGNOSTICS & SELF-TUNING (svpf_gradient_diagnostic.cu)
// =============================================================================
// 
// AGC-style parameter adaptation: SVPF particles find h, then θ follows.
// No oracle, no batch inference, no memory. Just negative feedback.
//
// Development phases:
//   Step 0: Observe gradients (no learning) - verify correctness
//   Step 1: Learn ν only (safest - observation likelihood only)
//   Step 2: Add σ learning (transition, with breathing overlay)
//   Step 3: Add ρ learning (if needed, often stable enough to fix)
//   Step 4: Full 4-param natural gradient (μ, ρ, σ, ν)
//
// Current implementation: Step 0 (diagnostic only)
// =============================================================================

/**
 * @brief Gradient diagnostic state for parameter learning
 * 
 * Tracks gradients and statistics for self-tuning development.
 * Start with diagnostic-only mode to verify gradient correctness
 * before enabling any parameter updates.
 */
typedef struct {
    // Device buffers for gradient computation
    float* d_nu_grad;           // Weighted mean ν gradient
    float* d_z_sq_mean;         // Mean standardized residual² (diagnostic)
    
    // Future: transition gradients (μ, ρ, σ)
    float* d_mu_grad;           // μ gradient (transition)
    float* d_rho_grad;          // ρ gradient (transition, unconstrained η)
    float* d_sigma_grad;        // σ gradient (transition, unconstrained κ)
    
    // Future: Fisher matrix for natural gradient (4x4, row-major)
    float* d_fisher;            // [16] Fisher matrix elements
    float* d_fisher_inv;        // [16] Inverse Fisher (for natural gradient)
    
    // Host-side EMA smoothing
    float nu_gradient_ema;      // Smoothed ν gradient
    float z_sq_ema;             // Smoothed z² (should be ~1 at equilibrium)
    float mu_gradient_ema;      // Smoothed μ gradient
    float rho_gradient_ema;     // Smoothed ρ gradient (unconstrained)
    float sigma_gradient_ema;   // Smoothed σ gradient (unconstrained)
    
    // Shock state machine (SHOCK → RECOVERY → CALM)
    int shock_state;            // 0=CALM, 1=SHOCK, 2=RECOVERY
    int ticks_in_state;         // Ticks since last state transition
    float shock_threshold;      // z² threshold to enter SHOCK (e.g., 9.0 = 3σ)
    int shock_duration;         // Ticks to stay in SHOCK (e.g., 20)
    int recovery_duration;      // Ticks in RECOVERY before CALM (e.g., 50)
    float recovery_exit_threshold; // z² threshold to exit RECOVERY (e.g., 4.0)
    
    // Logging
    bool enable_logging;
    FILE* log_file;
    
    bool initialized;
} SVPFGradientDiagnostics;

/**
 * @brief Unconstrained parameter representation for gradient descent
 * 
 * Maps constrained parameters to unconstrained space:
 *   μ = μ directly (unbounded)
 *   ρ = tanh(η)        → η ∈ (-∞, +∞), ρ ∈ (-1, 1)
 *   σ = exp(κ)         → κ ∈ (-∞, +∞), σ ∈ (0, +∞)
 *   ν = 2 + exp(κ_ν)   → κ_ν ∈ (-∞, +∞), ν ∈ (2, +∞)
 * 
 * Gradients are computed in unconstrained space for smooth optimization.
 */
typedef struct {
    float mu;       // Mean level (unbounded)
    float eta;      // Unconstrained persistence: ρ = tanh(η)
    float kappa;    // Unconstrained vol-of-vol: σ = exp(κ)
    float kappa_nu; // Unconstrained tail weight: ν = 2 + exp(κ_ν)
} SVPFThetaUnconstrained;

/**
 * @brief Natural gradient tuner state
 * 
 * Implements Fisher-preconditioned parameter updates:
 *   θ += lr · F⁻¹ · ∇θ
 * 
 * Fisher matrix captures parameter correlations (σ-ν, μ-ρ)
 * allowing a single learning rate for all parameters.
 */
typedef struct {
    SVPFThetaUnconstrained theta;
    
    // Fisher matrix (4x4, EMA smoothed)
    float F[4][4];              // Fisher information matrix
    float F_ema_decay;          // EMA decay for Fisher (e.g., 0.95)
    float F_reg;                // Regularization for invertibility (e.g., 1e-4)
    
    // Learning rate
    float base_lr;              // Base learning rate (e.g., 0.01)
    float lr_shock_mult;        // LR multiplier in RECOVERY (e.g., 2.0)
    
    // Gradient clipping
    float grad_clip;            // Max |gradient| per parameter (e.g., 1.0)
    
    // Regularization toward prior (soft oracle)
    float prior_weight;         // Pull toward offline-calibrated baseline
    SVPFThetaUnconstrained theta_prior;  // Offline baseline
    
    // State
    int warmup_ticks;           // Ticks before learning starts
    bool learning_enabled;      // Master switch
    
} SVPFNaturalGradientTuner;

// -----------------------------------------------------------------------------
// Gradient Diagnostic API
// -----------------------------------------------------------------------------

/**
 * @brief Create gradient diagnostic state
 * 
 * @param enable_logging  Write CSV log file
 * @param log_path        Path to log file (NULL for no logging)
 * @return Diagnostic state, or NULL on error
 */
SVPFGradientDiagnostics* svpf_gradient_diagnostic_create(bool enable_logging, const char* log_path);

/**
 * @brief Destroy gradient diagnostic state
 */
void svpf_gradient_diagnostic_destroy(SVPFGradientDiagnostics* diag);

/**
 * @brief Compute ν gradient (observation likelihood only)
 * 
 * Call AFTER svpf_step_graph() to compute the gradient of log p(y|h,ν)
 * with respect to ν, averaged over particles weighted by posterior.
 * 
 * Expected behavior:
 *   - ν too high → gradient NEGATIVE (wants heavier tails)
 *   - ν too low → gradient POSITIVE (wants lighter tails)
 *   - ν correct → gradient ≈ 0
 * 
 * @param state         SVPF state (after step)
 * @param diag          Diagnostic state
 * @param y_t           Current observation
 * @param timestep      Current timestep (for logging)
 * @param nu_grad_out   Optional: raw ν gradient
 * @param z_sq_mean_out Optional: mean standardized residual²
 */
void svpf_compute_nu_diagnostic(
    SVPFState* state,
    SVPFGradientDiagnostics* diag,
    float y_t,
    int timestep,
    float* nu_grad_out,
    float* z_sq_mean_out
);

/**
 * @brief Simplified ν gradient (no diagnostic state needed)
 * 
 * Allocates temporary buffers, computes gradient, frees buffers.
 * Less efficient but convenient for quick tests.
 */
void svpf_compute_nu_diagnostic_simple(
    SVPFState* state,
    float y_t,
    float* nu_grad_out,
    float* z_sq_mean_out
);

/**
 * @brief Compute σ gradient for diagnostic purposes
 * 
 * From transition likelihood: log p(h_t|h_{t-1}) = -½log(2πσ²) - ε²/(2σ²)
 * Gradient: ∂/∂σ = (ε²/σ² - 1) / σ
 * 
 * IMPORTANT: Uses h_pred (before Stein transport), not h (after Stein).
 * Stein transport is deterministic and >> σ, so post-Stein h gives garbage.
 * 
 * Expected behavior:
 *   - σ too HIGH → ε²/σ² < 1 → gradient NEGATIVE (decrease σ)
 *   - σ too LOW  → ε²/σ² > 1 → gradient POSITIVE (increase σ)
 *   - σ correct  → ε²/σ² ≈ 1 → gradient ≈ 0
 * 
 * Key advantage over ν: Signal available EVERY timestep, not just crashes.
 * 
 * @param state           SVPF state (uses h_pred, h_prev, log_weights)
 * @param params          SV parameters (μ, ρ, σ)
 * @param sigma_grad_out  Output: σ gradient (positive → increase σ)
 * @param eps_sq_norm_out Output: mean ε²/σ² (should be ~1.0 if σ correct)
 */
void svpf_compute_sigma_diagnostic_simple(
    SVPFState* state,
    const SVPFParams* params,
    float* sigma_grad_out,
    float* eps_sq_norm_out
);

/**
 * @brief Snapshot current particles to a buffer
 * 
 * Use when you need h_{t-1} explicitly (state->h_prev should already have it).
 */
void svpf_snapshot_particles(
    SVPFState* state,
    float* d_h_buffer
);

/**
 * @brief Compute all 4 parameter gradients (μ, ρ, σ, ν)
 * 
 * Requires h_prev snapshot from BEFORE svpf_step_graph().
 * 
 * For transition gradients (μ, ρ, σ):
 *   ε = h_t - μ - ρ·(h_{t-1} - μ)
 *   ∂/∂μ = ε·(1-ρ) / σ²
 *   ∂/∂ρ = ε·(h_{t-1} - μ) / σ²
 *   ∂/∂σ = -1/σ + ε²/σ³
 * 
 * For observation gradient (ν):
 *   ∂/∂ν = ½[ψ((ν+1)/2) - ψ(ν/2) - 1/ν - log(1+z²/ν) + (ν+1)z²/(ν²(1+z²/ν))]
 * 
 * Gradients are in UNCONSTRAINED space (η, κ, κ_ν).
 * 
 * @param state         SVPF state (after step)
 * @param diag          Diagnostic state
 * @param h_prev_snap   Snapshot of h_prev from before step [n_particles]
 * @param y_t           Current observation
 * @param params        Model parameters (for ρ, σ, μ values)
 * @param timestep      Current timestep (for logging)
 */
void svpf_compute_all_gradients(
    SVPFState* state,
    SVPFGradientDiagnostics* diag,
    const float* h_prev_snap,
    float y_t,
    const SVPFParams* params,
    int timestep
);

// -----------------------------------------------------------------------------
// Shock State Machine API
// -----------------------------------------------------------------------------

/**
 * @brief Update shock state machine
 * 
 * Transitions: CALM → SHOCK (on surprise) → RECOVERY → CALM
 * During SHOCK: freeze all learning (gradients are garbage)
 * During RECOVERY: boost learning rate to catch up
 * 
 * @param diag      Diagnostic state
 * @param z_sq      Current standardized residual² (from diagnostic)
 */
void svpf_update_shock_state(SVPFGradientDiagnostics* diag, float z_sq);

/**
 * @brief Get current shock state
 * @return 0=CALM, 1=SHOCK, 2=RECOVERY
 */
int svpf_get_shock_state(const SVPFGradientDiagnostics* diag);

/**
 * @brief Check if learning should be active
 * @return true if CALM or RECOVERY, false if SHOCK
 */
bool svpf_should_learn(const SVPFGradientDiagnostics* diag);

/**
 * @brief Get learning rate multiplier based on shock state
 * @return 0.0 (SHOCK), lr_shock_mult (RECOVERY), 1.0 (CALM)
 */
float svpf_get_lr_multiplier(const SVPFGradientDiagnostics* diag, float lr_shock_mult);

// -----------------------------------------------------------------------------
// Natural Gradient Tuner API (Future - Step 4)
// -----------------------------------------------------------------------------

/**
 * @brief Create natural gradient tuner
 * 
 * @param params        Initial parameters (from offline calibration)
 * @param base_lr       Base learning rate (e.g., 0.01)
 * @param prior_weight  Regularization toward initial params (e.g., 0.001)
 * @return Tuner state, or NULL on error
 */
SVPFNaturalGradientTuner* svpf_tuner_create(
    const SVPFParams* params,
    float nu,
    float base_lr,
    float prior_weight
);

/**
 * @brief Destroy natural gradient tuner
 */
void svpf_tuner_destroy(SVPFNaturalGradientTuner* tuner);

/**
 * @brief Update parameters with natural gradient
 * 
 * Computes: θ += lr · F⁻¹ · ∇θ
 * 
 * Call after svpf_compute_all_gradients() to update parameters.
 * Respects shock state (no update during SHOCK).
 * 
 * @param tuner         Tuner state
 * @param diag          Diagnostic state (for gradients and shock state)
 * @param params_out    Updated SVPFParams (constrained space)
 * @param nu_out        Updated ν value
 */
void svpf_tuner_update(
    SVPFNaturalGradientTuner* tuner,
    const SVPFGradientDiagnostics* diag,
    SVPFParams* params_out,
    float* nu_out
);

/**
 * @brief Get current parameters in constrained space
 */
void svpf_tuner_get_params(
    const SVPFNaturalGradientTuner* tuner,
    SVPFParams* params_out,
    float* nu_out
);

// -----------------------------------------------------------------------------
// Synthetic Verification
// -----------------------------------------------------------------------------

/**
 * @brief Run synthetic gradient verification test
 * 
 * Generates data from known model, then verifies:
 *   - ν too high → gradient negative
 *   - ν too low → gradient positive
 *   - ν correct → gradient ≈ 0
 * 
 * @param n_particles   Particles to use
 * @param n_stein_steps Stein steps per observation
 */
void svpf_test_nu_gradient_synthetic(int n_particles, int n_stein_steps);

// -----------------------------------------------------------------------------
// Utility: Convert Between Constrained/Unconstrained
// -----------------------------------------------------------------------------

static inline float svpf_constrain_rho(float eta) {
    return tanhf(eta);
}

static inline float svpf_unconstrain_rho(float rho) {
    // atanh(rho) = 0.5 * log((1+rho)/(1-rho))
    rho = fminf(fmaxf(rho, -0.999f), 0.999f);  // Clamp to avoid inf
    return 0.5f * logf((1.0f + rho) / (1.0f - rho));
}

static inline float svpf_constrain_sigma(float kappa) {
    return expf(kappa);
}

static inline float svpf_unconstrain_sigma(float sigma) {
    return logf(fmaxf(sigma, 1e-8f));
}

static inline float svpf_constrain_nu(float kappa_nu) {
    return 2.0f + expf(kappa_nu);
}

static inline float svpf_unconstrain_nu(float nu) {
    return logf(fmaxf(nu - 2.0f, 1e-8f));
}

// Chain rule factors for unconstrained gradients
static inline float svpf_drho_deta(float rho) {
    return 1.0f - rho * rho;  // sech²(η) = 1 - tanh²(η)
}

static inline float svpf_dsigma_dkappa(float sigma) {
    return sigma;  // d/dκ exp(κ) = exp(κ) = σ
}

static inline float svpf_dnu_dkappa_nu(float nu) {
    return nu - 2.0f;  // d/dκ_ν (2 + exp(κ_ν)) = exp(κ_ν) = ν - 2
}

#ifdef __cplusplus
}
#endif

#endif // SVPF_CUH
