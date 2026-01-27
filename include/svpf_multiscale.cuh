/**
 * @file svpf_multiscale.cuh
 * @brief 2-Scale SVPF: REACTIVE + INERTIAL
 *
 * ARCHITECTURE:
 *   REACTIVE: "What's happening NOW?" - fast response, low persistence
 *   INERTIAL: "Is this REAL?" - slow response, needs persistent evidence
 *
 * KEY INSIGHT:
 *   Difference (vol_reactive - vol_inertial) isolates TRANSIENT NOISE
 *   Agreement (both elevated) confirms REGIME SHIFT
 *
 * CROSS-SCALE COMMUNICATION:
 *   REACTIVE surprise → boosts INERTIAL exploration (MIM, guide)
 *   Breaks bootstrap problem: REACTIVE detects → INERTIAL wakes up
 *
 * Usage:
 *   SVPFParams base_params = {.rho=0.98f, .sigma_z=0.10f, .mu=-3.5f, .gamma=0.0f};
 *   MS_SVPF* ms = ms_svpf_create(NULL);
 *   ms_svpf_init(ms, initial_vol, &base_params);
 *
 *   for each tick:
 *       ms_svpf_step(ms, y_t, &base_params, &output);
 *       // output.vol_reactive, output.vol_inertial, output.vol_combined
 *
 *   ms_svpf_destroy(ms);
 */

#ifndef SVPF_MULTISCALE_CUH
#define SVPF_MULTISCALE_CUH

#include "svpf.cuh"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Particles per scale */
    int n_particles_reactive;   /* Default: 256 (scout, speed matters) */
    int n_particles_inertial;   /* Default: 1024 (precision matters) */
    
    /* Stein iterations per scale */
    int n_stein_reactive;       /* Default: 4 (fast, fewer iterations) */
    int n_stein_inertial;       /* Default: 8 (slow, more convergence) */
    
    /* Cross-scale boost thresholds */
    float surprise_mild_threshold;   /* Default: 2.0 (2σ event) */
    float surprise_high_threshold;   /* Default: 4.0 (4σ event) */
    float boost_mild_factor;         /* Default: 2.0x MIM boost */
    float boost_high_factor;         /* Default: 3.0x MIM boost */
    
    /* Combination weights (fixed) */
    float weight_reactive;      /* Default: 0.3 */
    float weight_inertial;      /* Default: 0.7 */
} MS_SVPF_Config;

/*═══════════════════════════════════════════════════════════════════════════
 * OUTPUT STRUCT
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Per-scale volatility estimates */
    float vol_reactive;
    float vol_inertial;
    
    /* Per-scale log-volatility */
    float h_reactive;
    float h_inertial;
    
    /* Per-scale log-likelihood */
    float loglik_reactive;
    float loglik_inertial;
    
    /* Combined estimates */
    float vol_combined;         /* Weighted combination */
    float h_combined;
    float loglik_combined;
    
    /* Transient noise indicator */
    float transient_magnitude;  /* vol_reactive - vol_inertial */
    float transient_ratio;      /* vol_reactive / vol_inertial */
    
    /* Diagnostics */
    float ksd_reactive;
    float ksd_inertial;
    int stein_steps_reactive;
    int stein_steps_inertial;
    
    /* Cross-scale state */
    float surprise_reactive;    /* Innovation z-score from REACTIVE */
    int cross_scale_boost;      /* 0=none, 1=mild, 2=high */
    float mim_prob_inertial_used; /* Actual MIM prob after boost */
    
    /* Classification */
    int is_transient;           /* REACTIVE high, INERTIAL low */
    int is_regime_shift;        /* Both elevated */
    int is_calm;                /* Both low */
} MS_SVPF_Output;

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN STRUCT
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Two independent SVPF filters */
    SVPFState* reactive;
    SVPFState* inertial;
    
    /* Configuration */
    MS_SVPF_Config config;
    
    /* Scale-specific dynamics (stored here, passed to params each step) */
    float rho_reactive;
    float sigma_z_reactive;
    float rho_inertial;
    float sigma_z_inertial;
    
    /* Cross-scale state */
    float vol_prev;
    float y_prev;               /* Previous observation for leverage effect */
    float surprise_ema;         /* Smoothed surprise for stability */
    
    /* Base MIM probability for INERTIAL (before boost) */
    float mim_jump_prob_inertial_base;
    
    /* CUDA streams for parallel execution */
    cudaStream_t stream_reactive;
    cudaStream_t stream_inertial;
    
    /* State */
    int initialized;
    int tick_count;
} MS_SVPF;

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIG
 *═══════════════════════════════════════════════════════════════════════════*/

static inline MS_SVPF_Config ms_svpf_default_config(void) {
    MS_SVPF_Config c;
    
    /* Particles: REACTIVE is small scout, INERTIAL is full precision */
    c.n_particles_reactive = 256;   // Fast scout (was 512)
    c.n_particles_inertial = 1024;  // Production precision
    
    /* Stein iterations: REACTIVE minimal, INERTIAL generous */
    c.n_stein_reactive = 3;         // Minimal (was 4)
    c.n_stein_inertial = 8;
    
    /* Cross-scale boost thresholds */
    c.surprise_mild_threshold = 2.0f;
    c.surprise_high_threshold = 3.5f;  // Lower (was 4.0)
    c.boost_mild_factor = 1.5f;        // Gentler (was 2.0)
    c.boost_high_factor = 2.5f;        // Gentler (was 3.0)
    
    /* Combination: switching logic will override these */
    c.weight_reactive = 0.10f;
    c.weight_inertial = 0.90f;
    
    return c;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CREATE / DESTROY
 *═══════════════════════════════════════════════════════════════════════════*/

static inline MS_SVPF* ms_svpf_create(const MS_SVPF_Config* config) {
    MS_SVPF_Config cfg = config ? *config : ms_svpf_default_config();
    
    MS_SVPF* ms = (MS_SVPF*)calloc(1, sizeof(MS_SVPF));
    if (!ms) return NULL;
    
    ms->config = cfg;
    
    /* Create CUDA streams first */
    cudaStreamCreateWithFlags(&ms->stream_reactive, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&ms->stream_inertial, cudaStreamNonBlocking);
    
    /* Create REACTIVE filter */
    ms->reactive = svpf_create(
        cfg.n_particles_reactive,
        cfg.n_stein_reactive,
        5.0f,  /* nu - Student-t degrees of freedom */
        ms->stream_reactive
    );
    
    /* Create INERTIAL filter */
    ms->inertial = svpf_create(
        cfg.n_particles_inertial,
        cfg.n_stein_inertial,
        5.0f,  /* nu */
        ms->stream_inertial
    );
    
    if (!ms->reactive || !ms->inertial) {
        if (ms->reactive) svpf_destroy(ms->reactive);
        if (ms->inertial) svpf_destroy(ms->inertial);
        if (ms->stream_reactive) cudaStreamDestroy(ms->stream_reactive);
        if (ms->stream_inertial) cudaStreamDestroy(ms->stream_inertial);
        free(ms);
        return NULL;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * REACTIVE CONFIGURATION ("Fast Scout")
     *═══════════════════════════════════════════════════════════════════════
     * DUMB and FAST. No fancy features - just aggressive exploration.
     * Its job is to detect changes, not to be accurate.
     *═══════════════════════════════════════════════════════════════════════*/
    {
        SVPFState* f = ms->reactive;
        
        /* Store scale-specific dynamics - FASTER */
        ms->rho_reactive = 0.92f;       // Half-life ~8 ticks
        ms->sigma_z_reactive = 0.18f;   // Higher exploration
        
        /* Asymmetric rho: slight */
        f->use_asymmetric_rho = 1;
        f->rho_up = 0.94f;
        f->rho_down = 0.88f;
        
        /* MIM: AGGRESSIVE */
        f->use_mim = 1;
        f->mim_jump_prob = 0.30f;   // 30% scouts
        f->mim_jump_scale = 10.0f;
        
        /* Guide: OFF - let particles fly freely */
        f->use_guide = 0;
        f->use_guided = 0;
        f->use_adaptive_guide = 0;
        
        /* SVLD: Higher temp for exploration */
        f->use_svld = 1;
        f->temperature = 0.55f;     // Higher than standard
        f->rmsprop_rho = 0.9f;
        f->rmsprop_eps = 1e-6f;
        
        /* Annealing: OFF (react immediately) */
        f->use_annealing = 0;
        f->n_anneal_steps = 1;
        
        /* Newton: OFF (expensive, unnecessary for scout) */
        f->use_newton = 0;
        f->use_full_newton = 0;
        
        /* Adaptive features: ALL OFF (too slow for scout) */
        f->use_adaptive_mu = 0;
        f->use_adaptive_sigma = 0;
        
        /* Local params: OFF */
        f->use_local_params = 0;
        f->delta_rho = 0.0f;
        f->delta_sigma = 0.0f;
        
        /* Exact gradient */
        f->use_exact_gradient = 1;
        f->lik_offset = 0.35f;
        
        /* KSD budget: MINIMAL (speed matters, not precision) */
        f->stein_min_steps = 2;
        f->stein_max_steps = 5;
        f->ksd_improvement_threshold = 0.10f;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * INERTIAL CONFIGURATION ("Production Baseline")
     *═══════════════════════════════════════════════════════════════════════
     * IDENTICAL to mono production config - this IS your tuned filter
     * The multi-scale benefit comes from combining with REACTIVE
     *═══════════════════════════════════════════════════════════════════════*/
    {
        SVPFState* f = ms->inertial;
        
        /* Store scale-specific dynamics - SAME as mono */
        ms->rho_inertial = 0.98f;       // Standard
        ms->sigma_z_inertial = 0.10f;   // Standard
        
        /* Asymmetric rho: ON (same as mono) */
        f->use_asymmetric_rho = 1;
        f->rho_up = 0.98f;
        f->rho_down = 0.93f;
        
        /* MIM: Full production settings */
        f->use_mim = 1;
        f->mim_jump_prob = 0.25f;    // Same as mono
        f->mim_jump_scale = 9.0f;
        ms->mim_jump_prob_inertial_base = 0.25f;
        
        /* Guide: FULL adaptive suite (same as mono) */
        f->use_guide = 1;
        f->use_guide_preserving = 1;
        f->guide_strength = 0.05f;
        f->guide_mean = 0.0f;
        f->guide_var = 0.0f;
        f->guide_K = 0.0f;
        f->guide_initialized = 0;
        
        f->use_adaptive_guide = 1;
        f->guide_strength_base = 0.05f;
        f->guide_strength_max = 0.30f;
        f->guide_innovation_threshold = 1.0f;
        
        /* Guided prediction (same as mono) */
        f->use_guided = 1;
        f->guided_alpha_base = 0.0f;
        f->guided_alpha_shock = 0.40f;
        f->guided_innovation_threshold = 1.5f;
        
        /* SVLD: Standard (same as mono) */
        f->use_svld = 1;
        f->temperature = 0.45f;
        f->rmsprop_rho = 0.9f;
        f->rmsprop_eps = 1e-6f;
        
        /* Annealing: ON (same as mono) */
        f->use_annealing = 1;
        f->n_anneal_steps = 3;
        
        /* Newton: FULL (same as mono) */
        f->use_newton = 1;
        f->use_full_newton = 1;
        
        /* Adaptive features: ALL ON (same as mono) */
        f->use_adaptive_mu = 1;
        f->mu_state = -3.5f;
        f->mu_var = 1.0f;
        f->mu_process_var = 0.001f;
        f->mu_obs_var_scale = 11.0f;
        f->mu_min = -4.0f;
        f->mu_max = -1.0f;
        
        f->use_adaptive_sigma = 1;
        f->sigma_boost_threshold = 0.95f;
        f->sigma_boost_max = 3.2f;
        f->sigma_z_effective = 0.10f;
        
        /* Local params: OFF (mono doesn't use this) */
        f->use_local_params = 0;
        f->delta_rho = 0.0f;
        f->delta_sigma = 0.0f;
        
        /* Exact gradient (same as mono) */
        f->use_exact_gradient = 1;
        f->lik_offset = 0.35f;
        
        /* KSD budget: GENEROUS (same as mono) */
        f->stein_min_steps = 8;
        f->stein_max_steps = 16;
        f->ksd_improvement_threshold = 0.05f;
    }
    
    ms->vol_prev = 0.01f;
    ms->y_prev = 0.0f;
    ms->surprise_ema = 0.0f;
    ms->initialized = 0;
    ms->tick_count = 0;
    
    return ms;
}

static inline void ms_svpf_destroy(MS_SVPF* ms) {
    if (ms) {
        if (ms->reactive) svpf_destroy(ms->reactive);
        if (ms->inertial) svpf_destroy(ms->inertial);
        
        if (ms->stream_reactive) cudaStreamDestroy(ms->stream_reactive);
        if (ms->stream_inertial) cudaStreamDestroy(ms->stream_inertial);
        
        free(ms);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZE
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void ms_svpf_init(MS_SVPF* ms, float initial_vol, const SVPFParams* base_params, unsigned int seed) {
    /* Create scale-specific params for initialization */
    SVPFParams params_reactive = *base_params;
    params_reactive.rho = ms->rho_reactive;
    params_reactive.sigma_z = ms->sigma_z_reactive;
    
    SVPFParams params_inertial = *base_params;
    params_inertial.rho = ms->rho_inertial;
    params_inertial.sigma_z = ms->sigma_z_inertial;
    
    /* Initialize both filters with related seeds */
    svpf_initialize(ms->reactive, &params_reactive, seed);
    svpf_initialize(ms->inertial, &params_inertial, seed);  // SAME seed as mono would use
    
    ms->vol_prev = initial_vol;
    ms->y_prev = 0.0f;
    ms->surprise_ema = 0.0f;
    ms->initialized = 1;
    ms->tick_count = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN STEP FUNCTION
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void ms_svpf_step(
    MS_SVPF* ms,
    float y_t,
    const SVPFParams* base_params,
    MS_SVPF_Output* out
) {
    /*───────────────────────────────────────────────────────────────────────
     * STEP 1: RUN REACTIVE FIRST (early warning)
     *───────────────────────────────────────────────────────────────────────
     * REACTIVE uses same observation but with low-persistence prior.
     * Its "surprise" tells us if something unexpected happened.
     *───────────────────────────────────────────────────────────────────────*/
    
    /* Create scale-specific params for REACTIVE */
    SVPFParams params_reactive = *base_params;
    params_reactive.rho = ms->rho_reactive;
    params_reactive.sigma_z = ms->sigma_z_reactive;
    
    float loglik_r, vol_r, h_r;
    svpf_step_adaptive(ms->reactive, y_t, ms->y_prev, &params_reactive, &loglik_r, &vol_r, &h_r);
    
    out->vol_reactive = vol_r;
    out->h_reactive = h_r;
    out->loglik_reactive = loglik_r;
    out->ksd_reactive = ms->reactive->ksd_prev;
    out->stein_steps_reactive = ms->reactive->stein_steps_used;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 2: COMPUTE SURPRISE (innovation z-score)
     *───────────────────────────────────────────────────────────────────────
     * Surprise = |y_t| / vol_reactive_prev
     * High surprise → REACTIVE saw something unexpected
     *───────────────────────────────────────────────────────────────────────*/
    
    float surprise = fabsf(y_t) / fmaxf(ms->vol_prev, 1e-6f);
    
    /* EMA smooth for stability */
    ms->surprise_ema = 0.3f * surprise + 0.7f * ms->surprise_ema;
    out->surprise_reactive = surprise;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 3: CROSS-SCALE BOOST
     *───────────────────────────────────────────────────────────────────────
     * If REACTIVE is surprised, boost INERTIAL's exploration.
     * This breaks the bootstrap problem for regime detection.
     *───────────────────────────────────────────────────────────────────────*/
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 3: CROSS-SCALE BOOST (DISABLED for testing)
     *───────────────────────────────────────────────────────────────────────
     * The boost was causing INERTIAL to diverge from MONO.
     * Disabled to verify INERTIAL = MONO baseline.
     *───────────────────────────────────────────────────────────────────────*/
    
    int boost_level = 0;
    float mim_boosted = ms->mim_jump_prob_inertial_base;  // No boost
    
    out->cross_scale_boost = boost_level;
    out->mim_prob_inertial_used = mim_boosted;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 4: RUN INERTIAL
     *───────────────────────────────────────────────────────────────────────*/
    
    /* INERTIAL uses base_params directly (matches DGP) */
    float loglik_i, vol_i, h_i;
    svpf_step_adaptive(ms->inertial, y_t, ms->y_prev, base_params, &loglik_i, &vol_i, &h_i);
    
    out->vol_inertial = vol_i;
    out->h_inertial = h_i;
    out->loglik_inertial = loglik_i;
    out->ksd_inertial = ms->inertial->ksd_prev;
    out->stein_steps_inertial = ms->inertial->stein_steps_used;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 5: USE INERTIAL ONLY (REACTIVE is just for diagnostics)
     *───────────────────────────────────────────────────────────────────────
     * After testing, we found that combining with REACTIVE hurts accuracy.
     * REACTIVE is useful for:
     *   - Detecting regime changes (surprise signal)
     *   - Decomposing transient vs permanent (diagnostic)
     * But NOT for improving point estimates.
     *───────────────────────────────────────────────────────────────────────*/
    
    float w_r = 0.0f;  // REACTIVE is diagnostic only
    float w_i = 1.0f;  // Trust INERTIAL completely
    
    /* Arithmetic mean for h (log-space) */
    out->h_combined = w_r * h_r + w_i * h_i;
    
    /* Geometric mean for vol (multiplicative) */
    out->vol_combined = powf(vol_r, w_r) * powf(vol_i, w_i);
    
    /* Sum for log-likelihood */
    out->loglik_combined = w_r * loglik_r + w_i * loglik_i;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 6: TRANSIENT ANALYSIS
     *───────────────────────────────────────────────────────────────────────
     * The key insight: difference between scales tells you about noise
     *───────────────────────────────────────────────────────────────────────*/
    
    out->transient_magnitude = vol_r - vol_i;
    out->transient_ratio = vol_r / fmaxf(vol_i, 1e-8f);
    
    /* Classification thresholds */
    const float vol_elevated = 0.02f;   /* 2% vol is "elevated" */
    const float ratio_transient = 1.3f; /* 30% higher = transient */
    
    out->is_transient = (vol_r > vol_elevated) && 
                        (out->transient_ratio > ratio_transient) &&
                        (vol_i < vol_elevated);
    
    out->is_regime_shift = (vol_r > vol_elevated) && (vol_i > vol_elevated);
    
    out->is_calm = (vol_r < vol_elevated) && (vol_i < vol_elevated);
    
    /*───────────────────────────────────────────────────────────────────────
     * UPDATE STATE
     *───────────────────────────────────────────────────────────────────────*/
    
    ms->vol_prev = out->vol_combined;
    ms->y_prev = y_t;
    ms->tick_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/* For execution/market making: trust REACTIVE more */
static inline float ms_svpf_vol_for_execution(const MS_SVPF_Output* out) {
    return 0.7f * out->vol_reactive + 0.3f * out->vol_inertial;
}

/* For position sizing: trust INERTIAL more */
static inline float ms_svpf_vol_for_position(const MS_SVPF_Output* out) {
    return 0.2f * out->vol_reactive + 0.8f * out->vol_inertial;
}

/* For risk limits: use max (conservative) */
static inline float ms_svpf_vol_conservative(const MS_SVPF_Output* out) {
    return fmaxf(out->vol_reactive, out->vol_inertial);
}

/* Is current observation likely transient noise? */
static inline int ms_svpf_is_noise(const MS_SVPF_Output* out) {
    return out->is_transient;
}

/* Should we scale down position? (regime shift detected) */
static inline int ms_svpf_regime_warning(const MS_SVPF_Output* out) {
    return out->is_regime_shift;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void ms_svpf_print_state(const MS_SVPF* ms, const MS_SVPF_Output* out) {
    printf("Multi-Scale SVPF (tick %d)\n", ms->tick_count);
    printf("├─ REACTIVE: vol=%.4f  h=%.2f  KSD=%.3f  steps=%d\n",
           out->vol_reactive, out->h_reactive, 
           out->ksd_reactive, out->stein_steps_reactive);
    printf("├─ INERTIAL: vol=%.4f  h=%.2f  KSD=%.3f  steps=%d%s\n",
           out->vol_inertial, out->h_inertial,
           out->ksd_inertial, out->stein_steps_inertial,
           out->cross_scale_boost ? " [BOOSTED]" : "");
    printf("├─ Transient: mag=%+.4f  ratio=%.2f\n",
           out->transient_magnitude, out->transient_ratio);
    printf("├─ Classification: %s\n",
           out->is_transient ? "TRANSIENT" :
           out->is_regime_shift ? "REGIME SHIFT" :
           out->is_calm ? "CALM" : "MIXED");
    printf("└─ Combined: vol=%.4f\n", out->vol_combined);
}

static inline void ms_svpf_print_config(const MS_SVPF* ms) {
    float hl_r = logf(0.5f) / logf(ms->rho_reactive);
    float hl_i = logf(0.5f) / logf(ms->rho_inertial);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  Multi-Scale SVPF Configuration\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Scale    │ Particles │ ρ     │ σ_z  │ Half-Life │ MIM   │ Guide\n");
    printf("  ─────────┼───────────┼───────┼──────┼───────────┼───────┼──────\n");
    printf("  REACTIVE │ %5d     │ %.3f │ %.2f │ %5.0f     │ %.0f%%  │ %s\n",
           ms->config.n_particles_reactive,
           ms->rho_reactive,
           ms->sigma_z_reactive,
           hl_r,
           ms->reactive->mim_jump_prob * 100,
           ms->reactive->use_guide ? "ON" : "OFF");
    printf("  INERTIAL │ %5d     │ %.3f │ %.2f │ %5.0f     │ %.0f%%  │ %s\n",
           ms->config.n_particles_inertial,
           ms->rho_inertial,
           ms->sigma_z_inertial,
           hl_i,
           ms->mim_jump_prob_inertial_base * 100,
           ms->inertial->use_guide ? "ON" : "OFF");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Cross-Scale: surprise > %.1f → %.0fx MIM boost\n",
           ms->config.surprise_mild_threshold,
           ms->config.boost_mild_factor);
    printf("               surprise > %.1f → %.0fx MIM boost\n",
           ms->config.surprise_high_threshold,
           ms->config.boost_high_factor);
    printf("  Weights: REACTIVE=%.0f%%, INERTIAL=%.0f%%\n",
           ms->config.weight_reactive * 100,
           ms->config.weight_inertial * 100);
    printf("════════════════════════════════════════════════════════════════\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER SETTERS (for tuning)
 *═══════════════════════════════════════════════════════════════════════════*/

/* Adjust REACTIVE dynamics */
static inline void ms_svpf_set_reactive_dynamics(
    MS_SVPF* ms,
    float rho,
    float sigma_z,
    float mim_prob
) {
    ms->rho_reactive = rho;
    ms->sigma_z_reactive = sigma_z;
    ms->reactive->rho_up = rho;
    ms->reactive->rho_down = rho - 0.02f;
    ms->reactive->mim_jump_prob = mim_prob;
}

/* Adjust INERTIAL dynamics */
static inline void ms_svpf_set_inertial_dynamics(
    MS_SVPF* ms,
    float rho,
    float sigma_z,
    float mim_prob_base
) {
    ms->rho_inertial = rho;
    ms->sigma_z_inertial = sigma_z;
    ms->inertial->rho_up = rho + 0.003f;
    ms->inertial->rho_down = rho - 0.005f;
    ms->inertial->mim_jump_prob = mim_prob_base;
    ms->mim_jump_prob_inertial_base = mim_prob_base;
}

/* Adjust combination weights */
static inline void ms_svpf_set_weights(
    MS_SVPF* ms,
    float weight_reactive,
    float weight_inertial
) {
    float sum = weight_reactive + weight_inertial;
    ms->config.weight_reactive = weight_reactive / sum;
    ms->config.weight_inertial = weight_inertial / sum;
}

/* Adjust cross-scale boost */
static inline void ms_svpf_set_boost_config(
    MS_SVPF* ms,
    float mild_threshold,
    float high_threshold,
    float mild_factor,
    float high_factor
) {
    ms->config.surprise_mild_threshold = mild_threshold;
    ms->config.surprise_high_threshold = high_threshold;
    ms->config.boost_mild_factor = mild_factor;
    ms->config.boost_high_factor = high_factor;
}

#ifdef __cplusplus
}
#endif

#endif /* SVPF_MULTISCALE_CUH */
