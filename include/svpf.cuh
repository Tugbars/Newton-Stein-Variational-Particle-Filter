/**
 * @file svpf.cuh
 * @brief Stein Variational Particle Filter for Stochastic Volatility
 *
 * Real-time volatility tracking with crash robustness.
 *
 * Production configuration (all always-on):
 *   Full Newton Stein | Adaptive annealing | Student-t state & likelihood
 *   Antithetic sampling | EKF guide (variance-preserving, adaptive strength)
 *   Guided prediction with innovation gating | Adaptive mu/sigma
 *   Backward smoothing | Partial rejuvenation | Asymmetric rho
 *
 * Algorithm: Fan et al. 2021 (arXiv:2106.10568) + Detommaso et al. 2018
 *
 * Usage:
 *   SVPFState* f = svpf_create(512, 8, 5.0f, stream);
 *   svpf_initialize(f, &params, seed);
 *   for each y_t:
 *       svpf_step_graph(f, y_t, y_prev, &params, &loglik, &vol, &h_mean);
 *   svpf_destroy(f);
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
#define SVPF_SMOOTH_MAX_LAG        8

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * @brief SV model parameters
 *
 * AR(1) log-volatility with leverage:
 *   h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps_t + gamma*y_{t-1}/exp(h_{t-1}/2)
 * Observation: y_t = exp(h_t/2) * eta_t,  eta_t ~ Student-t(nu)
 */
typedef struct {
    float rho;      // Persistence (typically 0.9-0.99)
    float sigma_z;  // Vol-of-vol (typically 0.1-0.3)
    float mu;       // Long-run mean log-volatility (typically -5 to -3)
    float gamma;    // Leverage effect (typically -0.5 to 0)
} SVPFParams;

/**
 * @brief Result of one SVPF filtering step
 */
typedef struct {
    float log_lik_increment;  // log p(y_t | y_{1:t-1}, theta)
    float vol_mean;           // E[exp(h/2)]
    float vol_std;            // Std[exp(h/2)]
    float h_mean;             // E[h]
    float mu_estimate;        // Current adaptive mu estimate
} SVPFResult;

/**
 * @brief Optimized backend buffers (embedded in SVPFState)
 */
typedef struct {
    // Bandwidth
    float* d_bandwidth;
    float* d_bandwidth_sq;

    // Full Newton buffers
    float* d_precond_grad;     // H^{-1} * grad
    float* d_inv_hessian;      // Diagonal Hessian per particle

    // Previous h_mean (device scalar for predict/guide kernels)
    float* d_h_mean_prev;

    // Single-step I/O buffers
    float* d_y_single;         // [2]: {y_prev, y_t}
    float* d_loglik_single;
    float* d_vol_single;

    // KSD computation
    float* d_ksd_partial;      // [N] partial sums
    float* d_ksd;              // [1] final KSD

    // Consolidated D2H output: [loglik, vol, h_mean, bandwidth, ksd]
    float* d_output_pack;      // Device (8 floats, 32B aligned)
    float* h_output_pinned;    // Pinned host mirror

    // Adaptive annealing stats: [sum_ll_diff, sum_ll_diff_sq, sum_grad, sum_h_diff_sq]
    float* d_anneal_stats;
    float* h_anneal_stats_pinned;

    // Async state (between step_async and sync_outputs)
    float pending_y_t;
    const void* pending_params;

    int allocated_n;
    bool initialized;
} SVPFOptimizedState;

/**
 * @brief SVPF filter state
 *
 * All features are always-on in the production build.
 * Tunable parameters are exposed; architectural switches are removed.
 */
typedef struct {
    // ===================== GPU Particle Arrays =====================
    float* h;                   // [N] Current log-volatility
    float* h_prev;              // [N] Previous step (AR(1) prior)
    float* grad_log_p;          // [N] Combined gradient
    float* log_weights;         // [N] Log importance weights
    curandStatePhilox4_32_10_t* rng_states;  // [N] RNG

    // RMSProp momentum for SVLD
    float* d_grad_v;            // [N] Second moment

    // Bandwidth regime detection
    float* d_return_ema;        // Scalar: EMA of |returns|
    float* d_return_var;        // Scalar: EMA of return²

    // Output scalar
    float* d_result_h_mean;

    // ===================== Filter Dimensions =====================
    int n_particles;
    int n_stein_steps;
    float nu;                   // Likelihood Student-t df
    float student_t_const;      // Precomputed normalizing constant
    float student_t_implied_offset;  // E[log(t²_ν)] correction
    int timestep;
    float y_prev;
    cudaStream_t stream;

    // ===================== Likelihood =====================
    float lik_offset;           // Exact gradient bias correction (~0.345)

    // ===================== Stein Transport =====================
    float temperature;          // SVLD temperature (e.g. 0.45)
    float rmsprop_rho;          // RMSProp decay (e.g. 0.7)
    float rmsprop_eps;          // RMSProp epsilon (1e-6)

    // ===================== MIM Predict =====================
    float mim_jump_prob;        // Jump probability (0 = disabled)
    float mim_jump_scale;       // Jump scale factor

    // ===================== Asymmetric Rho =====================
    float rho_up;               // Persistence when vol increasing
    float rho_down;             // Persistence when vol decreasing

    // ===================== Local Parameters =====================
    float delta_rho;            // Rho sensitivity to h deviation
    float delta_sigma;          // Sigma sensitivity to |h deviation|

    // ===================== Guided Prediction =====================
    // Innovation-gated: α≈0 when model fits, α→shock when surprised
    float guided_alpha_base;    // Alpha when model fits (e.g. 0.0)
    float guided_alpha_shock;   // Alpha during shock (e.g. 0.40)
    float guided_innovation_threshold;  // z-score for "surprise" (e.g. 1.5)

    // ===================== EKF Guide Density =====================
    float guide_strength_base;  // Base strength (e.g. 0.05)
    float guide_strength_max;   // Max during surprises (e.g. 0.30)
    float guide_innovation_threshold;  // z-score to start boosting
    float guide_mean;           // EKF posterior mean
    float guide_var;            // EKF posterior variance
    float guide_K;              // Kalman gain (diagnostic)
    int guide_initialized;
    float vol_prev;             // Previous vol (for innovation calc)

    // ===================== Adaptive Mu (Kalman) =====================
    float mu_state;             // Current mu estimate
    float mu_var;               // Kalman P
    float mu_process_var;       // Q: how fast mu drifts
    float mu_obs_var_scale;     // R = scale * bw²
    float mu_min;
    float mu_max;

    // ===================== Adaptive Sigma ("Breathing") =====================
    float sigma_boost_threshold;  // z-score to start boosting
    float sigma_boost_max;        // Max multiplier (e.g. 3.2x)
    float sigma_z_effective;      // Current effective sigma_z

    // ===================== Student-t State Dynamics =====================
    float nu_state;             // Prior df (5-7 recommended, clamped ≥2.5)

    // ===================== Partial Rejuvenation (Maken 2022) =====================
    float rejuv_ksd_threshold;  // KSD threshold to trigger
    float rejuv_prob;           // Fraction of particles to nudge
    float rejuv_blend;          // Blend factor toward guide

    // ===================== Adaptive Annealing =====================
    float anneal_kl_threshold;  // KL constraint per stage
    int anneal_steps_per_beta;  // Stein steps per beta level
    int anneal_max_stages;      // Safety cap

    // Diagnostics (updated each timestep)
    int anneal_stages_used;
    float anneal_final_var_ll;
    float anneal_final_h_std;

    // ===================== KSD Tracking =====================
    float ksd_prev;             // KSD from last timestep
    int stein_steps_used;       // Total Stein steps last timestep

    // ===================== Backward Smoothing (RTS) =====================
    int smooth_lag;             // Window size (1-5)
    int smooth_output_lag;      // Output delay (0=raw, 1=h[t-1])
    int smooth_head;            // Circular buffer index
    float smooth_h_mean[SVPF_SMOOTH_MAX_LAG];
    float smooth_h_var[SVPF_SMOOTH_MAX_LAG];
    float smooth_y[SVPF_SMOOTH_MAX_LAG];

    // ===================== Backend =====================
    SVPFOptimizedState opt_backend;

} SVPFState;

// =============================================================================
// API: Lifecycle
// =============================================================================

/** Create SVPF filter. stream can be NULL for default. */
SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream);

/** Free all GPU and host memory. */
void svpf_destroy(SVPFState* state);

/** Initialize particles from stationary distribution. */
void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed);

// =============================================================================
// API: Stepping
// =============================================================================

/** Synchronous step (calls async + sync internally). */
void svpf_step_graph(
    SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
    float* h_loglik_out, float* h_vol_out, float* h_mean_out
);

/** Convenience: auto-tracks y_prev internally. */
void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result);

/** Process entire observation sequence (host array). */
void svpf_run_sequence(
    SVPFState* state, const float* h_observations, int T,
    const SVPFParams* params, float* h_loglik_out, float* h_vol_out
);

// =============================================================================
// API: Async (multi-filter parallelism)
// =============================================================================
//
// Usage with N filters on separate streams:
//   for (i) svpf_step_async(filters[i], y[i], y_prev[i], &params[i]);
//   for (i) svpf_sync_outputs(filters[i], &ll[i], &vol[i], &hm[i]);

/** Launch GPU work (non-blocking). */
void svpf_step_async(
    SVPFState* state, float y_t, float y_prev, const SVPFParams* params
);

/** Wait for GPU, read outputs, update internal state. */
void svpf_sync_outputs(
    SVPFState* state, float* h_loglik_out, float* h_vol_out, float* h_mean_out
);

// =============================================================================
// API: Diagnostics
// =============================================================================

/** Copy particles to host. */
void svpf_get_particles(const SVPFState* state, float* h_out);

/** Get particle mean and std. */
void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std);

/** Get effective sample size. */
float svpf_get_ess(const SVPFState* state);

/** Get KSD and Stein step count from last timestep. */
void svpf_get_ksd_stats(const SVPFState* state, float* ksd_out, int* steps_used_out);

/** Cleanup optimized backend (called by svpf_destroy). */
void svpf_optimized_cleanup_state(SVPFState* state);

// =============================================================================
// GRADIENT DIAGNOSTICS & SELF-TUNING
// =============================================================================

typedef struct {
    float* d_nu_grad;
    float* d_z_sq_mean;
    float* d_mu_grad;
    float* d_rho_grad;
    float* d_sigma_grad;
    float* d_fisher;        // [16] Fisher matrix
    float* d_fisher_inv;    // [16] Inverse Fisher

    // Host EMA
    float nu_gradient_ema;
    float z_sq_ema;
    float mu_gradient_ema;
    float rho_gradient_ema;
    float sigma_gradient_ema;

    // Shock state machine
    int shock_state;            // 0=CALM, 1=SHOCK, 2=RECOVERY
    int ticks_in_state;
    float shock_threshold;
    int shock_duration;
    int recovery_duration;
    float recovery_exit_threshold;

    bool enable_logging;
    FILE* log_file;
    bool initialized;
} SVPFGradientDiagnostics;

typedef struct {
    float mu;
    float eta;      // rho = tanh(eta)
    float kappa;    // sigma = exp(kappa)
    float kappa_nu; // nu = 2 + exp(kappa_nu)
} SVPFThetaUnconstrained;

typedef struct {
    SVPFThetaUnconstrained theta;
    float F[4][4];
    float F_ema_decay;
    float F_reg;
    float base_lr;
    float lr_shock_mult;
    float grad_clip;
    float prior_weight;
    SVPFThetaUnconstrained theta_prior;
    int warmup_ticks;
    bool learning_enabled;
} SVPFNaturalGradientTuner;

// --- Gradient Diagnostic API ---

SVPFGradientDiagnostics* svpf_gradient_diagnostic_create(bool enable_logging, const char* log_path);
void svpf_gradient_diagnostic_destroy(SVPFGradientDiagnostics* diag);

void svpf_compute_nu_diagnostic(
    SVPFState* state, SVPFGradientDiagnostics* diag, float y_t, int timestep,
    float* nu_grad_out, float* z_sq_mean_out
);

void svpf_compute_nu_diagnostic_simple(
    SVPFState* state, float y_t, float* nu_grad_out, float* z_sq_mean_out
);

void svpf_compute_sigma_diagnostic_simple(
    SVPFState* state, const SVPFParams* params,
    float* sigma_grad_out, float* eps_sq_norm_out
);

void svpf_snapshot_particles(SVPFState* state, float* d_h_buffer);

void svpf_compute_all_gradients(
    SVPFState* state, SVPFGradientDiagnostics* diag, const float* h_prev_snap,
    float y_t, const SVPFParams* params, int timestep
);

// --- Shock State Machine ---

void svpf_update_shock_state(SVPFGradientDiagnostics* diag, float z_sq);
int svpf_get_shock_state(const SVPFGradientDiagnostics* diag);
bool svpf_should_learn(const SVPFGradientDiagnostics* diag);
float svpf_get_lr_multiplier(const SVPFGradientDiagnostics* diag, float lr_shock_mult);

// --- Natural Gradient Tuner ---

SVPFNaturalGradientTuner* svpf_tuner_create(
    const SVPFParams* params, float nu, float base_lr, float prior_weight
);
void svpf_tuner_destroy(SVPFNaturalGradientTuner* tuner);
void svpf_tuner_update(
    SVPFNaturalGradientTuner* tuner, const SVPFGradientDiagnostics* diag,
    SVPFParams* params_out, float* nu_out
);
void svpf_tuner_get_params(
    const SVPFNaturalGradientTuner* tuner, SVPFParams* params_out, float* nu_out
);
void svpf_test_nu_gradient_synthetic(int n_particles, int n_stein_steps);

// --- Constrained/Unconstrained Transforms ---

static inline float svpf_constrain_rho(float eta) { return tanhf(eta); }
static inline float svpf_unconstrain_rho(float rho) {
    rho = fminf(fmaxf(rho, -0.999f), 0.999f);
    return 0.5f * logf((1.0f + rho) / (1.0f - rho));
}
static inline float svpf_constrain_sigma(float kappa) { return expf(kappa); }
static inline float svpf_unconstrain_sigma(float sigma) { return logf(fmaxf(sigma, 1e-8f)); }
static inline float svpf_constrain_nu(float kappa_nu) { return 2.0f + expf(kappa_nu); }
static inline float svpf_unconstrain_nu(float nu) { return logf(fmaxf(nu - 2.0f, 1e-8f)); }
static inline float svpf_drho_deta(float rho) { return 1.0f - rho * rho; }
static inline float svpf_dsigma_dkappa(float sigma) { return sigma; }
static inline float svpf_dnu_dkappa_nu(float nu) { return nu - 2.0f; }

#ifdef __cplusplus
}
#endif

#endif // SVPF_CUH
