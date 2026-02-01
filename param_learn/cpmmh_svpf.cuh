/**
 * @file cpmmh_svpf.cuh
 * @brief Correlated Pseudo-Marginal Metropolis-Hastings for SVPF Parameter Learning
 * 
 * Learns SV model parameters θ = (ρ, μ, σ_z) via MCMC with:
 * - Correlated proposals (Deligiannidis et al. 2018): ε* = ρ·ε + √(1-ρ²)·ε_fresh
 * - Adaptive Metropolis (Haario et al. 2001): proposal covariance from chain history
 * - SVPF inner filter for unbiased likelihood estimation
 * 
 * Structure designed for later SMC² extension:
 * - cpmmh_step() IS the SMC² rejuvenation kernel
 * - Just parallelize across N_theta particles + add outer resampling
 * 
 * References:
 * - Deligiannidis, Doucet, Pitt (2018) "The Correlated Pseudo-Marginal Method"
 * - Andrieu, Doucet, Holenstein (2010) "Particle MCMC Methods"
 * - Haario, Saksman, Tamminen (2001) "Adaptive Metropolis Algorithm"
 */

#ifndef CPMMH_SVPF_CUH
#define CPMMH_SVPF_CUH

#include "svpf_replay_mode.cuh"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * PRIOR SPECIFICATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Prior distributions for SV parameters.
 * 
 * Crypto-calibrated defaults:
 * - ρ:    Beta-like on [0.85, 0.995], mode ~0.94 (faster mean-reversion than equities)
 * - μ:    Normal(-3.0, 1.0), log-vol around -3 → ~5% daily vol
 * - σ_z:  HalfNormal(0.20), higher vol-of-vol for crypto regime changes
 */
typedef struct {
    /* ρ prior: transformed Beta on [rho_min, rho_max] */
    float rho_min;              /* Lower bound (0.85) */
    float rho_max;              /* Upper bound (0.995) */
    float rho_alpha;            /* Beta shape α (18) */
    float rho_beta;             /* Beta shape β (2) */
    
    /* μ prior: Normal(mu_mean, mu_std²) */
    float mu_mean;              /* Prior mean (-3.0) */
    float mu_std;               /* Prior std (1.0) */
    float mu_min;               /* Hard bound (-8.0) */
    float mu_max;               /* Hard bound (2.0) */
    
    /* σ_z prior: HalfNormal(0, sigma_z_scale²) */
    float sigma_z_scale;        /* Scale parameter (0.20) */
    float sigma_z_min;          /* Hard bound (0.01) */
    float sigma_z_max;          /* Hard bound (1.0) */
} CPMMHPrior;

/**
 * Default crypto-calibrated prior.
 */
static inline CPMMHPrior cpmmh_prior_crypto_default(void) {
    CPMMHPrior p;
    
    /* ρ: Beta(18, 2) on [0.85, 0.995] → mode ≈ 0.94 */
    p.rho_min = 0.85f;
    p.rho_max = 0.995f;
    p.rho_alpha = 18.0f;
    p.rho_beta = 2.0f;
    
    /* μ: N(-3.0, 1.0) with bounds */
    p.mu_mean = -3.0f;
    p.mu_std = 1.0f;
    p.mu_min = -8.0f;
    p.mu_max = 2.0f;
    
    /* σ_z: HalfNormal(0.20) */
    p.sigma_z_scale = 0.20f;
    p.sigma_z_min = 0.01f;
    p.sigma_z_max = 1.0f;
    
    return p;
}

/**
 * Evaluate log prior density.
 * Returns -INFINITY if out of bounds.
 */
float cpmmh_log_prior(const SVPFReplayParams* theta, const CPMMHPrior* prior);

/*═══════════════════════════════════════════════════════════════════════════════
 * ADAPTIVE PROPOSAL
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Adaptive proposal state (Haario et al. 2001).
 * 
 * Maintains running mean and covariance of accepted samples.
 * Proposal: θ* ~ N(θ, (2.38²/d)·Σ) where d=3.
 * 
 * Mixture: 95% adaptive, 5% fixed (for ergodicity).
 */
typedef struct {
    /* Running statistics */
    float mean[3];              /* Running mean of chain */
    float cov[9];               /* Running covariance (3×3, row-major) */
    float chol[9];              /* Cholesky factor of scaled cov */
    int   n_samples;            /* Number of samples in statistics */
    
    /* Fixed proposal (fallback / mixture component) */
    float fixed_std[3];         /* Independent proposal stds */
    
    /* Adaptation parameters */
    int   adapt_start;          /* Start adapting after this many samples */
    int   adapt_interval;       /* Update Cholesky every N samples */
    float scale_factor;         /* 2.38² / d = 1.89 for d=3 */
    float mixture_prob;         /* Probability of using adaptive (0.95) */
    
    /* Regularization */
    float cov_regularization;   /* Small value added to diagonal (1e-6) */
} CPMMHProposal;

/**
 * Initialize proposal with sensible defaults.
 */
void cpmmh_proposal_init(CPMMHProposal* prop);

/**
 * Update proposal statistics with accepted sample.
 */
void cpmmh_proposal_update(CPMMHProposal* prop, const float* theta);

/**
 * Generate proposal θ* given current θ.
 * Returns proposed theta in theta_out.
 * Uses host RNG (not CUDA).
 */
void cpmmh_proposal_sample(
    const CPMMHProposal* prop,
    const float* theta_current,
    float* theta_out,
    uint64_t* rng_state         /* Host xorshift state */
);

/*═══════════════════════════════════════════════════════════════════════════════
 * CPMMH STATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Main CPMMH state for single-chain inference.
 */
typedef struct {
    /*─────────────────────────────────────────────────────────────────────────
     * Current chain state
     *─────────────────────────────────────────────────────────────────────────*/
    float theta[3];             /* Current parameters (ρ, μ, σ_z) */
    float log_likelihood;       /* Current log p̂(y|θ, ε) */
    float log_prior;            /* Current log p(θ) */
    float log_posterior;        /* log_likelihood + log_prior */
    
    /*─────────────────────────────────────────────────────────────────────────
     * Noise buffers (ping-pong for correlation)
     *─────────────────────────────────────────────────────────────────────────*/
    SVPFNoiseBuffer* noise[2];  /* Two buffers for current/proposed */
    int active_buf;             /* Which buffer is current (0 or 1) */
    float cpmmh_rho;            /* Correlation coefficient (0.99) */
    
    /*─────────────────────────────────────────────────────────────────────────
     * SVPF replay state
     *─────────────────────────────────────────────────────────────────────────*/
    SVPFReplayState* replay;    /* Replay filter state */
    SVPFReplayConfig replay_config;
    
    /*─────────────────────────────────────────────────────────────────────────
     * Observations (device)
     *─────────────────────────────────────────────────────────────────────────*/
    float* d_y;                 /* Observations [T] */
    int T;                      /* Number of observations */
    
    /*─────────────────────────────────────────────────────────────────────────
     * Prior and proposal
     *─────────────────────────────────────────────────────────────────────────*/
    CPMMHPrior prior;
    CPMMHProposal proposal;
    
    /*─────────────────────────────────────────────────────────────────────────
     * Diagnostics
     *─────────────────────────────────────────────────────────────────────────*/
    int n_accepts;              /* Total accepts */
    int n_total;                /* Total proposals */
    int n_prior_rejects;        /* Rejected due to prior bounds */
    
    /*─────────────────────────────────────────────────────────────────────────
     * Chain history (for output / convergence diagnostics)
     *─────────────────────────────────────────────────────────────────────────*/
    float* theta_history;       /* [n_samples × 3] host buffer */
    float* log_post_history;    /* [n_samples] host buffer */
    int    history_capacity;
    int    history_len;
    
    /*─────────────────────────────────────────────────────────────────────────
     * RNG state (host)
     *─────────────────────────────────────────────────────────────────────────*/
    uint64_t host_rng_state;
    
    /*─────────────────────────────────────────────────────────────────────────
     * CUDA
     *─────────────────────────────────────────────────────────────────────────*/
    cudaStream_t stream;
    int N_particles;            /* Inner filter particles */
    
} CPMMHState;

/*═══════════════════════════════════════════════════════════════════════════════
 * CPMMH API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Create CPMMH state.
 * 
 * @param N_particles   Inner SVPF particles (256-1024 typical)
 * @param T_max         Maximum observation length
 * @param history_cap   Chain history capacity
 * @return              Allocated state (NULL on failure)
 */
CPMMHState* cpmmh_create(int N_particles, int T_max, int history_cap);

/**
 * Destroy CPMMH state.
 */
void cpmmh_destroy(CPMMHState* state);

/**
 * Set observations for inference.
 * Copies to device and resets chain.
 * 
 * @param state         CPMMH state
 * @param h_y           Host observations [T]
 * @param T             Number of observations
 */
void cpmmh_set_observations(CPMMHState* state, const float* h_y, int T);

/**
 * Initialize chain from prior or given starting point.
 * 
 * @param state         CPMMH state
 * @param theta_init    Starting θ (NULL = sample from prior)
 * @param seed          RNG seed
 */
void cpmmh_initialize(CPMMHState* state, const float* theta_init, uint64_t seed);

/**
 * Run single CPMMH iteration.
 * 
 * This is the core function that will become SMC² rejuvenation kernel.
 * 
 * Steps:
 * 1. Propose θ* from adaptive proposal
 * 2. Check prior bounds
 * 3. Correlate noise: ε* = ρ·ε + √(1-ρ²)·ε_fresh
 * 4. Compute log p̂(y|θ*, ε*) via SVPF replay
 * 5. MH accept/reject
 * 6. Update proposal statistics (on accept)
 * 
 * @param state         CPMMH state
 * @return              1 if accepted, 0 if rejected
 */
int cpmmh_step(CPMMHState* state);

/**
 * Run multiple CPMMH iterations.
 * 
 * @param state         CPMMH state
 * @param n_iterations  Number of MH steps
 * @param n_burnin      Discard first n_burnin from statistics
 * @param thin          Keep every thin-th sample in history
 */
void cpmmh_run(CPMMHState* state, int n_iterations, int n_burnin, int thin);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS AND OUTPUT
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Get current acceptance rate.
 */
static inline float cpmmh_acceptance_rate(const CPMMHState* state) {
    return (state->n_total > 0) ? (float)state->n_accepts / state->n_total : 0.0f;
}

/**
 * Get posterior mean from chain history.
 */
void cpmmh_get_posterior_mean(const CPMMHState* state, float* theta_mean);

/**
 * Get posterior std from chain history.
 */
void cpmmh_get_posterior_std(const CPMMHState* state, float* theta_std);

/**
 * Get posterior covariance from chain history.
 */
void cpmmh_get_posterior_cov(const CPMMHState* state, float* cov_out);

/**
 * Compute Gelman-Rubin R-hat from multiple chains.
 * 
 * @param states        Array of chain states
 * @param n_chains      Number of chains (recommend 4+)
 * @param rhat_out      Output R-hat for each parameter [3]
 * @return              Maximum R-hat across parameters
 */
float cpmmh_gelman_rubin(CPMMHState** states, int n_chains, float* rhat_out);

/**
 * Export chain history to host arrays.
 * 
 * @param state         CPMMH state
 * @param theta_out     Output [history_len × 3] (can be NULL)
 * @param logpost_out   Output [history_len] (can be NULL)
 * @return              Number of samples in history
 */
int cpmmh_export_history(
    const CPMMHState* state,
    float* theta_out,
    float* logpost_out
);

/**
 * Print chain summary statistics.
 */
void cpmmh_print_summary(const CPMMHState* state);

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION SETTERS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Set CPMMH correlation coefficient.
 * Higher = more correlated proposals = lower variance but slower mixing.
 * Typical: 0.99 (default), can try 0.95-0.999.
 */
void cpmmh_set_correlation(CPMMHState* state, float rho);

/**
 * Set prior specification.
 */
void cpmmh_set_prior(CPMMHState* state, const CPMMHPrior* prior);

/**
 * Set replay configuration.
 */
void cpmmh_set_replay_config(CPMMHState* state, const SVPFReplayConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CPMMH_SVPF_CUH */
