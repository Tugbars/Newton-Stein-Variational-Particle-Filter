/**
 * @file svpf.cuh
 * @brief Stein Variational Particle Filter for Stochastic Volatility
 * 
 * Real-time volatility tracking with crash robustness.
 * Algorithm: Fan et al. 2021 (arXiv:2106.10568) with extensions:
 *   - Full Newton Stein (Detommaso 2018)
 *   - Adaptive annealing (KL-constrained beta stepping)
 *   - Student-t state dynamics for bounded gradients
 *   - Antithetic sampling for variance reduction
 *   - EKF guide density with innovation gating
 *   - RTS backward smoothing
 * 
 * Usage:
 *   SVPFState* f = svpf_create(512, 8, 5.0f, stream);
 *   svpf_initialize(f, &params, seed);
 *   for each y_t:
 *       svpf_step_async(f, y_t, y_prev, &params);
 *       svpf_sync_outputs(f, &loglik, &vol, &h_mean);
 *   svpf_destroy(f);
 * 
 * Memory Layout: Structure of Arrays (SoA) for coalesced GPU access
 * 
 * References:
 * - Liu & Wang (2016): SVGD algorithm
 * - Fan et al. (2021): Stein Particle Filtering (arXiv:2106.10568)
 * - Detommaso et al. (2018): Full Newton Stein
 * - Maken et al. (2022): Partial rejuvenation, KSD-adaptive beta
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
#define SVPF_DEFAULT_STEIN_STEPS   8
#define SVPF_DEFAULT_NU            5.0f
#define SVPF_STEIN_STEP_SIZE       0.1f
#define SVPF_BANDWIDTH_MIN         0.01f
#define SVPF_BANDWIDTH_MAX         10.0f
#define SVPF_H_MIN                 -15.0f
#define SVPF_H_MAX                 5.0f
#define SVPF_BLOCK_SIZE            256
#define SVPF_SMALL_N_THRESHOLD     4096   // Threshold for persistent CTA path
#define SVPF_GRAPH_PARAMS_SIZE     32     // Floats in graph parameter staging buffer
#define SVPF_SMOOTH_MAX_LAG        8      // Max backward smoothing window

// Stein sign: 0=legacy(attraction), 1=paper(repulsion, Fan et al. 2021)
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
 * Layout: 24 named floats + 8 reserved = 32 floats total.
 * Updated via cudaMemcpyAsync before graph replay.
 */
typedef struct {
    float y_prev;            // Previous observation
    float y_t;               // Current observation
    float guide_mean;        // EKF guide mean
    float beta;              // Current annealing factor
    float step_size;         // Stein step size
    float temp;              // SVLD temperature
    float rho;               // AR persistence
    float sigma_z;           // Innovation std
    float mu;                // Mean level
    float gamma;             // Leverage coefficient
    float nu;                // Student-t degrees of freedom
    float student_t_const;   // Precomputed Student-t normalizing constant
    float rho_up;            // Asymmetric rho (up moves)
    float rho_down;          // Asymmetric rho (down moves)
    float delta_rho;         // Particle-local rho sensitivity
    float delta_sigma;       // Particle-local sigma sensitivity
    float alpha_base;        // Guided alpha (base)
    float alpha_shock;       // Guided alpha (shock)
    float innovation_thresh; // Guided innovation threshold
    float jump_prob;         // MIM jump probability
    float jump_scale;        // MIM jump scale
    float guide_strength;    // Guide density strength
    float rmsprop_rho;       // RMSProp decay
    float rmsprop_eps;       // RMSProp epsilon
    float reserved[8];       // Future use
} SVPFGraphParams;

/**
 * @brief Optimized backend state for batch/graph processing
 * 
 * Embedded in SVPFState for thread safety — each filter instance
 * has its own optimization buffers.
 */
typedef struct {
    // CUB temporary storage
    void* d_temp_storage;          // CUB reduction workspace
    size_t temp_storage_bytes;     // Size of CUB workspace

    // Device scalars
    float* d_max_log_w;            // Max log-weight (for logsumexp)
    float* d_sum_exp;              // Sum of exp(log_w - max)
    float* d_bandwidth;            // Current kernel bandwidth
    float* d_bandwidth_sq;         // Bandwidth squared (EMA state)

    // Stein computation buffers
    float* d_exp_w;                // [N] Exponentiated weights
    float* d_phi;                  // [N] Stein operator output
    float* d_grad_lik;             // [N] Likelihood gradient (separate from prior)
    float* d_precond_grad;         // [N] H^{-1} * grad (Newton preconditioned)
    float* d_inv_hessian;          // [N] Curvature per particle

    // State from previous step
    float* d_h_mean_prev;          // Scalar: h mean from last step

    // Guide (device-side for graph compatibility)
    float* d_guide_mean;           // Scalar: EKF guide mean on device
    float* d_guide_strength;       // Scalar: adaptive guide strength on device

    // Single-step API buffers (avoid malloc in hot loop)
    float* d_y_single;             // [2] y_prev and y_t staging
    float* d_loglik_single;        // Scalar: log-likelihood output
    float* d_vol_single;           // Scalar: volatility output

    // CUDA Graph handles
    cudaGraph_t graph;             // Captured graph
    cudaGraphExec_t graph_exec;    // Executable graph instance
    cudaStream_t graph_stream;     // Stream for graph capture
    bool graph_captured;           // Whether graph is currently valid
    int graph_n;                   // N at capture time
    int graph_n_stein;             // Stein steps at capture time
    float* d_params_staging;       // [32] Packed parameters for graph replay
    float mu_captured;             // Mu burned into graph
    float sigma_z_captured;        // Sigma_z burned into graph
    float* h_results_pinned;       // [4] Pinned host: [loglik, vol, h_mean, bw]

    // KSD buffers
    float* d_ksd_partial;          // [N] Partial KSD sums per particle
    float* d_ksd;                  // Scalar: final KSD value

    // Consolidated D2H output pack (single transfer)
    float* d_output_pack;          // [8] Device: [loglik, vol, h_mean, bw, ksd, pad...]
    float* h_output_pinned;        // [8] Pinned host mirror

    // Async state (between step_async and sync_outputs)
    float pending_y_t;             // Stored y_t for post-sync processing
    const void* pending_params;    // Stored SVPFParams* for post-sync

    // Adaptive annealing buffers
    float* d_anneal_stats;         // [4] mean_ll, var_ll, mean_grad, h_std
    float* h_anneal_stats_pinned;  // [4] Pinned host mirror

    // Capacity
    int allocated_n;               // Allocated particle count
    bool initialized;              // Whether backend is initialized
} SVPFOptimizedState;

/**
 * @brief SV model parameters
 * 
 * AR(1) log-volatility with leverage:
 *   h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps_t + gamma*y_{t-1}/exp(h_{t-1}/2)
 *   y_t = exp(h_t/2) * eta_t,  eta_t ~ Student-t(nu)
 */
typedef struct {
    float rho;      // Persistence (0 < rho < 1, typically 0.9-0.99)
    float sigma_z;  // Vol-of-vol (typically 0.1-0.3)
    float mu;       // Long-run mean log-volatility (typically -5 to -3)
    float gamma;    // Leverage effect (typically -0.5 to 0 for equities)
} SVPFParams;

/**
 * @brief SVPF filter state (SoA layout for GPU)
 */
typedef struct {
    // --- Particle arrays [N] ---
    float* h;                  // Current log-volatility particles
    float* h_prev;             // Previous step (for AR(1) prior)
    float* h_pred;             // Predicted particles (before Stein)
    float* grad_log_p;         // Gradient of log posterior
    float* kernel_sum;         // Sum of kernel weights (attraction)
    float* grad_kernel_sum;    // Sum of kernel gradients (repulsion)
    float* log_weights;        // Log importance weights
    float* d_h_centered;       // Centered particles for variance
    float* d_grad_v;           // RMSProp second moment [N]

    // --- Regime detection for bandwidth scaling ---
    float* d_return_ema;       // Scalar: EMA of |returns|
    float* d_return_var;       // Scalar: EMA of return variance
    float* d_bw_alpha;         // Scalar: adaptive bandwidth alpha

    // --- RNG ---
    curandStatePhilox4_32_10_t* rng_states;  // [N] Philox RNG states

    // --- Reduction workspace ---
    float* d_reduce_buf;       // Reduction scratch [N]
    float* d_temp;             // Temp scratch [N]
    void* d_cub_temp;          // CUB temp storage
    size_t cub_temp_bytes;     // CUB temp size

    // --- Device scalars ---
    float* d_scalar_max;       // Reduction: max
    float* d_scalar_sum;       // Reduction: sum
    float* d_scalar_mean;      // Reduction: mean
    float* d_scalar_bandwidth; // Current bandwidth
    float* d_y_prev;           // Previous observation on device
    float* d_result_loglik;    // Output: log-likelihood
    float* d_result_vol_mean;  // Output: volatility mean
    float* d_result_h_mean;    // Output: h mean

    // --- Core configuration ---
    int n_particles;           // Number of particles
    int n_stein_steps;         // Stein iterations per timestep
    float nu;                  // Student-t degrees of freedom
    float student_t_const;     // Precomputed: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*nu)
    float student_t_implied_offset;  // Precomputed: -E[log(t²_ν)] for implied h
    int timestep;              // Current timestep
    float y_prev;              // Previous observation (host-side)
    cudaStream_t stream;       // CUDA stream for this filter

    // --- Likelihood gradient config ---
    int use_exact_gradient;    // 0=surrogate (log-squared), 1=exact Student-t
    float lik_offset;          // Bias correction (0.345 for exact, 0.70 for surrogate)

    // --- Stein transport config ---
    int use_svld;              // Enable SVLD (Langevin noise)
    int use_annealing;         // Enable annealed Stein
    int use_adaptive_beta;     // KSD-adaptive beta (Maken 2022)
    int n_anneal_steps;        // Number of annealing stages
    float temperature;         // Langevin temperature (0=SVGD, >0=SVLD)
    float rmsprop_rho;         // RMSProp decay (0.7)
    float rmsprop_eps;         // RMSProp epsilon (1e-6)

    // --- MIM predict ---
    int use_mim;               // Enable mixture innovation model
    float mim_jump_prob;       // Jump component probability
    float mim_jump_scale;      // Jump component scale factor

    // --- Particle-local parameters ---
    int use_local_params;      // Enable particle-local rho and sigma
    float delta_rho;           // Rho sensitivity to h deviation
    float delta_sigma;         // Sigma sensitivity to |h deviation|

    // --- Asymmetric persistence ---
    int use_asymmetric_rho;    // Enable asymmetric rho
    float rho_up;              // Persistence when vol increasing
    float rho_down;            // Persistence when vol decreasing

    // --- Newton-Stein (Hessian preconditioning) ---
    int use_newton;            // Enable Newton-Stein
    int use_full_newton;       // Detommaso 2018 kernel-weighted Hessian

    // --- Guided prediction (innovation-gated lookahead) ---
    int use_guided;            // Enable guided predict
    float guided_alpha_base;   // Alpha when model fits (0 = trust prior)
    float guided_alpha_shock;  // Alpha when model fails (0.4 = trust observation)
    float guided_innovation_threshold;  // Z-score threshold for "surprise"

    // --- Partial rejuvenation (Maken 2022) ---
    int use_rejuvenation;      // Nudge stuck particles toward guide
    float rejuv_ksd_threshold; // KSD threshold to trigger
    float rejuv_prob;          // Fraction of particles to nudge
    float rejuv_blend;         // Blend factor toward guide

    // --- EKF guide density ---
    int use_guide;             // Enable EKF guide
    int use_guide_preserving;  // Variance-preserving shift (vs contraction)
    float guide_strength;      // Base guide strength
    float guide_mean;          // EKF posterior mean (m_t)
    float guide_var;           // EKF posterior variance (P_t)
    float guide_K;             // Kalman gain
    int guide_initialized;     // Whether guide has been initialized

    // --- Adaptive guide strength (innovation-gated) ---
    int use_adaptive_guide;    // Enable adaptive guide strength
    float guide_strength_base; // Base strength when model fits
    float guide_strength_max;  // Max strength during surprises
    float guide_innovation_threshold;  // Z-score threshold for boost
    float vol_prev;            // Previous vol estimate

    // --- Adaptive mu (1D Kalman filter on mean level) ---
    int use_adaptive_mu;       // Enable adaptive mu learning
    float mu_state;            // Current mu estimate (Kalman state)
    float mu_var;              // Current mu variance (Kalman P)
    float mu_process_var;      // Process noise Q (mu drift rate)
    float mu_obs_var_scale;    // Measurement noise R = scale * bw²
    float mu_min;              // Lower bound for mu
    float mu_max;              // Upper bound for mu

    // --- Adaptive sigma_z (innovation-gated vol-of-vol boost) ---
    int use_adaptive_sigma;    // Enable adaptive sigma_z
    float sigma_boost_threshold;  // Z-score threshold to start boosting
    float sigma_boost_max;     // Maximum boost multiplier
    float sigma_z_effective;   // Current effective sigma_z

    // --- Stein operator sign mode ---
    int stein_repulsive_sign;  // 0=legacy(attract), 1=paper(repel)

    // --- Fan mode (weightless SVGD, Fan et al. 2021) ---
    int use_fan_mode;          // 0=hybrid, 1=uniform weights + no annealing

    // --- Student-t state dynamics ---
    int use_student_t_state;   // 0=Gaussian AR(1), 1=Student-t AR(1)
    float nu_state;            // State DoF (5-7 recommended, clamped >= 2.5)

    // --- KSD-based adaptive Stein steps ---
    int stein_min_steps;       // Minimum Stein iterations
    int stein_max_steps;       // Maximum Stein iterations
    float ksd_improvement_threshold;  // Stop if relative improvement < this
    float ksd_prev;            // KSD from previous timestep
    int stein_steps_used;      // Diagnostic: actual steps used

    // --- Antithetic sampling ---
    int use_antithetic;        // Pair particles with (+z, -z) noise

    // --- Adaptive annealing (KL-constrained beta stepping) ---
    int use_adaptive_anneal;   // 0=fixed stages, 1=KL-adaptive
    float anneal_kl_threshold; // KL constraint per stage
    int anneal_steps_per_beta; // Stein steps per beta update
    int anneal_max_stages;     // Safety cap
    int anneal_stages_used;    // Diagnostic: stages used this step
    float anneal_final_var_ll; // Diagnostic: final variance of log-lik
    float anneal_final_h_std;  // Diagnostic: final particle spread

    // --- Backward smoothing (lightweight RTS) ---
    int use_smoothing;         // 0=off, 1=on
    int smooth_lag;            // Window size (1-5 recommended)
    int smooth_output_lag;     // Output delay: 0=raw, 1=h[t-1], etc.
    int smooth_head;           // Circular buffer write index
    float smooth_h_mean[SVPF_SMOOTH_MAX_LAG];  // Stored h estimates
    float smooth_h_var[SVPF_SMOOTH_MAX_LAG];   // Stored uncertainties
    float smooth_y[SVPF_SMOOTH_MAX_LAG];       // Stored observations

    // --- Persistent kernel mode ---
    int use_persistent_kernel;        // 0=standard, 1=persistent CTA
    int persistent_kernel_supported;  // Set at creation based on GPU

    // --- Heun's method (reserved, currently disabled) ---
    int use_heun;              // 0=Euler, 1=Heun (2nd order, 2x grad evals)

    // --- Optimized backend (embedded for thread safety) ---
    SVPFOptimizedState opt_backend;

    int use_split_batch;    // 1 = even/odd split-batch SVGD, 0 = standard (all 

} SVPFState;

/**
 * @brief Result of one SVPF filtering step
 */
typedef struct {
    float log_lik_increment;   // log p(y_t | y_{1:t-1}, theta)
    float vol_mean;            // E[exp(h/2)]
    float vol_std;             // Std[exp(h/2)]
    float h_mean;              // E[h]
    float mu_estimate;         // Current adaptive mu (if enabled)
} SVPFResult;

// =============================================================================
// API: Core Filter Functions
// =============================================================================

/** Create SVPF filter. n_particles should be power of 2 (256-1024). */
SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream);

/** Free all GPU and host memory. */
void svpf_destroy(SVPFState* state);

/** Initialize particles from stationary distribution. */
void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed);

/** Process one observation (synchronous). */
void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result);

/** Process one observation with explicit RNG seed (for SMC²/CPMMH). */
void svpf_step_seeded(SVPFState* state, float y_t, const SVPFParams* params,
                      unsigned long long rng_seed, SVPFResult* result);

// =============================================================================
// API: Optimized Step (production use)
// =============================================================================

/** Optimized single step with pre-allocated buffers (zero malloc in hot loop). */
void svpf_step_optimized(SVPFState* state, float y_t, float y_prev,
                         const SVPFParams* params, float* h_loglik_out, float* h_vol_out);

/**
 * Graph-accelerated step. Captures kernel sequence on first call, replays
 * with ~5us overhead. Call svpf_graph_invalidate() after changing params.
 */
void svpf_step_graph(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                     float* h_loglik_out, float* h_vol_out, float* h_mean_out);

/** Alias for svpf_step_graph (adaptive annealing + full Newton). */
void svpf_step_adaptive(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                        float* h_loglik_out, float* h_vol_out, float* h_mean_out);

/** Check if CUDA graph is currently captured. */
bool svpf_graph_is_captured(SVPFState* state);

/** Force graph recapture on next step (call after changing any config). */
void svpf_graph_invalidate(SVPFState* state);

// =============================================================================
// API: Async (multi-stream parallel execution)
// =============================================================================
//
// Usage for N filters in parallel:
//   for (i) svpf_step_async(filters[i], y_t[i], y_prev[i], &params[i]);
//   for (i) svpf_sync_outputs(filters[i], &loglik[i], &vol[i], &h_mean[i]);

/** Launch all GPU work non-blocking. Call sync_outputs() to get results. */
void svpf_step_async(SVPFState* state, float y_t, float y_prev, const SVPFParams* params);

/** Wait for GPU, read outputs, update state. Must follow step_async(). */
void svpf_sync_outputs(SVPFState* state, float* h_loglik_out, float* h_vol_out, float* h_mean_out);

// =============================================================================
// API: Batch Processing
// =============================================================================

/** Process entire observation sequence (host arrays). */
void svpf_run_sequence(SVPFState* state, const float* h_observations, int T,
                       const SVPFParams* params, float* h_loglik_out, float* h_vol_out);

/** Process sequence with data already on GPU. */
void svpf_run_sequence_device(SVPFState* state, const float* d_observations, int T,
                              const SVPFParams* params, float* d_loglik_out, float* d_vol_out);

// =============================================================================
// API: Diagnostics
// =============================================================================

/** Copy particles to host (for plotting/debugging). */
void svpf_get_particles(const SVPFState* state, float* h_out);

/** Get particle mean and std on host. */
void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std);

/** Get ESS. Close to N = healthy, close to 1 = degeneracy. */
float svpf_get_ess(const SVPFState* state);

/** Get KSD and number of Stein steps used last timestep. */
void svpf_get_ksd_stats(const SVPFState* state, float* ksd_out, int* steps_used_out);

// =============================================================================
// API: Internal (called by create/destroy)
// =============================================================================

void svpf_optimized_init(SVPFState* state);
void svpf_optimized_cleanup_state(SVPFState* state);

// =============================================================================
// API: Configuration Helpers
// =============================================================================

/** Set Stein sign: SVPF_STEIN_SIGN_LEGACY(0) or SVPF_STEIN_SIGN_PAPER(1). */
static inline void svpf_set_stein_sign_mode(SVPFState* state, int mode) {
    state->stein_repulsive_sign = (mode == SVPF_STEIN_SIGN_PAPER) ? 1 : 0;
}

static inline int svpf_get_stein_sign_mode(const SVPFState* state) {
    return state->stein_repulsive_sign;
}

/** Enable Fan mode (weightless SVGD). Implies paper sign. */
static inline void svpf_set_fan_mode(SVPFState* state, int enable) {
    state->use_fan_mode = enable ? 1 : 0;
    if (enable) state->stein_repulsive_sign = SVPF_STEIN_SIGN_PAPER;
}

static inline int svpf_get_fan_mode(const SVPFState* state) {
    return state->use_fan_mode;
}

// =============================================================================
// GRADIENT DIAGNOSTICS & SELF-TUNING (svpf_gradient_diagnostic.cu)
// =============================================================================

/**
 * @brief Gradient diagnostic state for parameter learning
 */
typedef struct {
    // Device buffers
    float* d_nu_grad;          // Weighted mean nu gradient
    float* d_z_sq_mean;        // Mean standardized residual squared
    float* d_mu_grad;          // mu gradient (transition)
    float* d_rho_grad;         // rho gradient (unconstrained eta)
    float* d_sigma_grad;       // sigma gradient (unconstrained kappa)
    float* d_fisher;           // [16] Fisher matrix (4x4, row-major)
    float* d_fisher_inv;       // [16] Inverse Fisher

    // Host-side EMA smoothing
    float nu_gradient_ema;     // Smoothed nu gradient
    float z_sq_ema;            // Smoothed z squared (should be ~1 at equilibrium)
    float mu_gradient_ema;     // Smoothed mu gradient
    float rho_gradient_ema;    // Smoothed rho gradient (unconstrained)
    float sigma_gradient_ema;  // Smoothed sigma gradient (unconstrained)

    // Shock state machine (CALM -> SHOCK -> RECOVERY -> CALM)
    int shock_state;           // 0=CALM, 1=SHOCK, 2=RECOVERY
    int ticks_in_state;        // Ticks since last state transition
    float shock_threshold;     // z squared threshold to enter SHOCK
    int shock_duration;        // Ticks to stay in SHOCK
    int recovery_duration;     // Ticks in RECOVERY before CALM
    float recovery_exit_threshold;  // z squared threshold to exit RECOVERY

    // Logging
    bool enable_logging;       // Write CSV log file
    FILE* log_file;            // Log file handle

    bool initialized;          // Whether buffers are allocated
} SVPFGradientDiagnostics;

/**
 * @brief Unconstrained parameter space for gradient descent
 * 
 * Maps: rho=tanh(eta), sigma=exp(kappa), nu=2+exp(kappa_nu)
 */
typedef struct {
    float mu;       // Mean level (unbounded)
    float eta;      // Unconstrained persistence: rho = tanh(eta)
    float kappa;    // Unconstrained vol-of-vol: sigma = exp(kappa)
    float kappa_nu; // Unconstrained tail weight: nu = 2 + exp(kappa_nu)
} SVPFThetaUnconstrained;

/**
 * @brief Natural gradient tuner (Fisher-preconditioned updates)
 */
typedef struct {
    SVPFThetaUnconstrained theta;  // Current unconstrained params

    float F[4][4];             // Fisher information matrix (EMA smoothed)
    float F_ema_decay;         // EMA decay for Fisher
    float F_reg;               // Regularization for invertibility

    float base_lr;             // Base learning rate
    float lr_shock_mult;       // LR multiplier in RECOVERY
    float grad_clip;           // Max |gradient| per parameter

    float prior_weight;        // Pull toward offline baseline
    SVPFThetaUnconstrained theta_prior;  // Offline baseline

    int warmup_ticks;          // Ticks before learning starts
    bool learning_enabled;     // Master switch
} SVPFNaturalGradientTuner;

// --- Gradient Diagnostic API ---

SVPFGradientDiagnostics* svpf_gradient_diagnostic_create(bool enable_logging, const char* log_path);
void svpf_gradient_diagnostic_destroy(SVPFGradientDiagnostics* diag);

/** Compute nu gradient from observation likelihood. Call AFTER svpf_step_graph(). */
void svpf_compute_nu_diagnostic(SVPFState* state, SVPFGradientDiagnostics* diag,
                                float y_t, int timestep,
                                float* nu_grad_out, float* z_sq_mean_out);

/** Simplified nu gradient (allocates temp buffers internally). */
void svpf_compute_nu_diagnostic_simple(SVPFState* state, float y_t,
                                       float* nu_grad_out, float* z_sq_mean_out);

/** Compute sigma gradient from transition likelihood. Uses h_pred (before Stein). */
void svpf_compute_sigma_diagnostic_simple(SVPFState* state, const SVPFParams* params,
                                          float* sigma_grad_out, float* eps_sq_norm_out);

/** Snapshot current particles to a device buffer. */
void svpf_snapshot_particles(SVPFState* state, float* d_h_buffer);

/** Compute all 4 parameter gradients (mu, rho, sigma, nu) in unconstrained space. */
void svpf_compute_all_gradients(SVPFState* state, SVPFGradientDiagnostics* diag,
                                const float* h_prev_snap, float y_t,
                                const SVPFParams* params, int timestep);

// --- Shock State Machine API ---

void svpf_update_shock_state(SVPFGradientDiagnostics* diag, float z_sq);
int svpf_get_shock_state(const SVPFGradientDiagnostics* diag);
bool svpf_should_learn(const SVPFGradientDiagnostics* diag);
float svpf_get_lr_multiplier(const SVPFGradientDiagnostics* diag, float lr_shock_mult);

// --- Natural Gradient Tuner API ---

SVPFNaturalGradientTuner* svpf_tuner_create(const SVPFParams* params, float nu,
                                             float base_lr, float prior_weight);
void svpf_tuner_destroy(SVPFNaturalGradientTuner* tuner);
void svpf_tuner_update(SVPFNaturalGradientTuner* tuner, const SVPFGradientDiagnostics* diag,
                       SVPFParams* params_out, float* nu_out);
void svpf_tuner_get_params(const SVPFNaturalGradientTuner* tuner,
                           SVPFParams* params_out, float* nu_out);

// --- Synthetic Test ---

void svpf_test_nu_gradient_synthetic(int n_particles, int n_stein_steps);

// --- Constrained/Unconstrained Conversions ---

static inline float svpf_constrain_rho(float eta) { return tanhf(eta); }
static inline float svpf_unconstrain_rho(float rho) {
    rho = fminf(fmaxf(rho, -0.999f), 0.999f);
    return 0.5f * logf((1.0f + rho) / (1.0f - rho));
}
static inline float svpf_constrain_sigma(float kappa) { return expf(kappa); }
static inline float svpf_unconstrain_sigma(float sigma) { return logf(fmaxf(sigma, 1e-8f)); }
static inline float svpf_constrain_nu(float kappa_nu) { return 2.0f + expf(kappa_nu); }
static inline float svpf_unconstrain_nu(float nu) { return logf(fmaxf(nu - 2.0f, 1e-8f)); }

// Chain rule factors: d(constrained)/d(unconstrained)
static inline float svpf_drho_deta(float rho) { return 1.0f - rho * rho; }
static inline float svpf_dsigma_dkappa(float sigma) { return sigma; }
static inline float svpf_dnu_dkappa_nu(float nu) { return nu - 2.0f; }

// =============================================================================
// Persistent Kernel API (svpf_persistent.cu)
// =============================================================================

/** @brief Async 2-kernel step: predict → probe → persistent stein. */
void svpf_persistent_step_async(
    SVPFState* state, float y_t, float y_prev, const SVPFParams* params);

/** @brief Sync outputs after persistent_step_async. Includes smoothing + adaptive mu. */
void svpf_persistent_sync_outputs(
    SVPFState* state, float* h_loglik_out, float* h_vol_out, float* h_mean_out);

/** @brief Synchronous convenience: async + sync in one call. */
void svpf_persistent_step(
    SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
    float* h_loglik_out, float* h_vol_out, float* h_mean_out);


#ifdef __cplusplus
}
#endif

#endif // SVPF_CUH
