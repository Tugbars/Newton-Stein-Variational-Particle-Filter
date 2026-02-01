/**
 * @file svpf_replay_mode.cuh
 * @brief SVPF Deterministic Replay Mode for CPMMH/SMC² Parameter Learning
 * 
 * Provides a deterministic likelihood evaluation p̂(y_{1:T} | θ, ε_{1:T})
 * by running SVPF with externally-provided noise instead of RNG.
 * 
 * Key modifications from tracking mode:
 * - Prediction noise ε_t read from buffer (not generated)
 * - Fixed Stein steps (no KSD-adaptive budget)
 * - Temperature = 0 (no SVLD exploration noise)
 * - No partial rejuvenation (deterministic transport only)
 * - Guide disabled or deterministic
 * 
 * This enables CPMMH: correlated proposals share 99% of noise,
 * making likelihood estimates highly correlated → low MH rejection.
 */

#ifndef SVPF_REPLAY_MODE_CUH
#define SVPF_REPLAY_MODE_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * REPLAY CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Configuration for deterministic SVPF replay.
 * 
 * In replay mode, all stochasticity is controlled externally via noise buffers.
 * This allows CPMMH to correlate proposals by sharing noise.
 */
typedef struct {
    int   n_stein_steps;        /* Fixed Stein iterations (no KSD-adaptive) */
    int   n_anneal_steps;       /* Annealing stages (1 = no annealing) */
    float temperature;          /* SVLD temperature (0 = disabled, deterministic) */
    
    /* Numerical stability */
    float h_min;                /* Clamp floor for log-vol */
    float h_max;                /* Clamp ceiling for log-vol */
    
    /* Student-t parameters (fixed during replay) */
    float nu_obs;               /* Observation df */
    float nu_state;             /* State dynamics df (0 = Gaussian) */
    int   use_student_t_state;  /* Enable Student-t state dynamics */
} SVPFReplayConfig;

/**
 * Default replay configuration for CPMMH.
 * Conservative settings prioritizing determinism over tracking quality.
 */
static inline SVPFReplayConfig svpf_replay_config_default(void) {
    SVPFReplayConfig cfg;
    cfg.n_stein_steps = 8;          /* Fixed budget, no adaptation */
    cfg.n_anneal_steps = 3;         /* Standard annealing */
    cfg.temperature = 0.0f;         /* No SVLD noise */
    cfg.h_min = -15.0f;
    cfg.h_max = 5.0f;
    cfg.nu_obs = 5.0f;              /* Heavy tails for crypto */
    cfg.nu_state = 5.0f;
    cfg.use_student_t_state = 1;    /* Bounded gradients */
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * NOISE BUFFER MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Noise buffer for CPMMH correlation.
 * 
 * Stores prediction noise ε_t for each particle at each timestep.
 * Layout: [t * N + i] = noise for particle i at time t
 * 
 * CPMMH correlates proposals via:
 *   ε* = ρ·ε + √(1-ρ²)·ε_fresh,  ρ ≈ 0.99
 */
typedef struct {
    float* d_noise;             /* Device buffer [T_capacity × N_particles] */
    int    T_capacity;          /* Maximum timesteps allocated */
    int    N_particles;         /* Particles per timestep */
    int    T_current;           /* Current filled length */
} SVPFNoiseBuffer;

/**
 * Allocate noise buffer on device.
 */
SVPFNoiseBuffer* svpf_noise_buffer_create(int T_capacity, int N_particles);

/**
 * Free noise buffer.
 */
void svpf_noise_buffer_destroy(SVPFNoiseBuffer* buf);

/**
 * Fill buffer with fresh Gaussian noise.
 * Uses cuRAND on device for efficiency.
 */
void svpf_noise_buffer_fill_gaussian(
    SVPFNoiseBuffer* buf,
    int T,
    unsigned long long seed,
    cudaStream_t stream
);

/**
 * Correlate noise: dst = ρ·src + √(1-ρ²)·fresh
 * 
 * @param src       Source noise buffer (current chain state)
 * @param dst       Destination buffer (proposed state)
 * @param rho       Correlation coefficient (0.99 typical)
 * @param T         Number of timesteps to correlate
 * @param seed      RNG seed for fresh noise
 * @param stream    CUDA stream
 */
void svpf_noise_buffer_correlate(
    const SVPFNoiseBuffer* src,
    SVPFNoiseBuffer* dst,
    float rho,
    int T,
    unsigned long long seed,
    cudaStream_t stream
);

/**
 * Copy noise buffer (on accept).
 */
void svpf_noise_buffer_copy(
    const SVPFNoiseBuffer* src,
    SVPFNoiseBuffer* dst,
    int T,
    cudaStream_t stream
);

/*═══════════════════════════════════════════════════════════════════════════════
 * REPLAY STATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * State for SVPF replay mode.
 * 
 * Lightweight compared to full SVPFState — no tracking features,
 * just what's needed for deterministic likelihood evaluation.
 */
typedef struct {
    /* Dimensions */
    int N;                      /* Number of particles */
    
    /* Particle state (device) */
    float* d_h;                 /* Current log-vol particles [N] */
    float* d_h_prev;            /* Previous step [N] */
    float* d_grad_log_p;        /* Gradient buffer [N] */
    float* d_phi;               /* Stein operator output [N] */
    float* d_grad_v;            /* RMSProp accumulator [N] */
    
    /* Reduction buffers */
    float* d_reduce_buf;        /* General reduction scratch [N] */
    float* d_bandwidth;         /* Kernel bandwidth [1] */
    float* d_log_lik_inc;       /* Per-step log-likelihood increment [1] */
    
    /* Precomputed constants */
    float student_t_const;      /* lgamma terms for Student-t */
    float student_t_offset;     /* E[log(t²)] offset */
    
    /* CUB temp storage */
    void*  d_cub_temp;
    size_t cub_temp_bytes;
    
    /* Config */
    SVPFReplayConfig config;
    
    /* Stream */
    cudaStream_t stream;
} SVPFReplayState;

/**
 * Create replay state.
 */
SVPFReplayState* svpf_replay_create(int N_particles, const SVPFReplayConfig* config);

/**
 * Destroy replay state.
 */
void svpf_replay_destroy(SVPFReplayState* state);

/**
 * Reset replay state for new sequence.
 * Initializes particles from stationary distribution of given params.
 */
void svpf_replay_reset(
    SVPFReplayState* state,
    float rho,
    float mu,
    float sigma_z,
    const float* d_init_noise   /* [N] Gaussian noise for initialization */
);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE REPLAY API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * SV model parameters for replay.
 */
typedef struct {
    float rho;                  /* Persistence ∈ [0.85, 0.995] */
    float mu;                   /* Long-run mean log-vol ∈ [-6, 0] */
    float sigma_z;              /* Vol-of-vol ∈ [0.02, 0.5] */
} SVPFReplayParams;

/**
 * Run single replay step (one observation).
 * 
 * @param state         Replay state
 * @param y_t           Current observation
 * @param y_prev        Previous observation (for return computation)
 * @param params        Model parameters θ = (ρ, μ, σ_z)
 * @param d_noise_t     Prediction noise for this timestep [N], device pointer
 * @return              Log-likelihood increment log p̂(y_t | y_{1:t-1}, θ)
 */
float svpf_replay_step(
    SVPFReplayState* state,
    float y_t,
    float y_prev,
    const SVPFReplayParams* params,
    const float* d_noise_t
);

/**
 * Replay entire sequence, return total log-likelihood.
 * 
 * This is the main entry point for CPMMH.
 * 
 * @param state         Replay state (will be reset internally)
 * @param d_y           Observations [T], device pointer
 * @param T             Number of observations
 * @param params        Model parameters θ
 * @param noise_buf     Noise buffer with T timesteps filled
 * @return              Total log-likelihood log p̂(y_{1:T} | θ, ε_{1:T})
 */
float svpf_replay_likelihood(
    SVPFReplayState* state,
    const float* d_y,
    int T,
    const SVPFReplayParams* params,
    const SVPFNoiseBuffer* noise_buf
);

/**
 * Batch replay: multiple parameter sets in parallel.
 * 
 * For SMC² rejuvenation where we need to evaluate many θ-particles.
 * Each θ gets its own noise buffer and replay state.
 * 
 * @param states        Array of replay states [N_theta]
 * @param d_y           Shared observations [T]
 * @param T             Number of observations
 * @param params        Parameter sets [N_theta]
 * @param noise_bufs    Noise buffers [N_theta]
 * @param d_log_lik_out Output log-likelihoods [N_theta], device pointer
 * @param N_theta       Number of parameter particles
 * @param stream        CUDA stream
 */
void svpf_replay_likelihood_batch(
    SVPFReplayState** states,
    const float* d_y,
    int T,
    const SVPFReplayParams* params,
    SVPFNoiseBuffer** noise_bufs,
    float* d_log_lik_out,
    int N_theta,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* SVPF_REPLAY_MODE_CUH */
