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
 * Usage:
 *   SVPFState* filter = svpf_create(512, 5, 5.0f, stream);
 *   svpf_initialize(filter, &params, seed);
 *   for each observation y_t:
 *       svpf_step(filter, y_t, &params, &result);
 *       // result.vol_mean is current volatility estimate
 *   svpf_destroy(filter);
 * 
 * Memory Layout: Structure of Arrays (SoA) for coalesced GPU access
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

#define SVPF_DEFAULT_PARTICLES     512    // Power of 2 for reductions
#define SVPF_DEFAULT_STEIN_STEPS   5
#define SVPF_DEFAULT_NU            5.0f   // Student-t degrees of freedom
#define SVPF_STEIN_STEP_SIZE       0.1f
#define SVPF_BANDWIDTH_MIN         0.01f
#define SVPF_BANDWIDTH_MAX         10.0f
#define SVPF_H_MIN                 -15.0f
#define SVPF_H_MAX                 5.0f

// Block size for kernels
#define SVPF_BLOCK_SIZE            256

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * @brief SV model parameters
 * 
 * AR(1) log-volatility with leverage effect:
 *   h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps_t + gamma*y_{t-1}/exp(h_{t-1}/2)
 * 
 * Observation: y_t = exp(h_t/2) * eta_t, eta_t ~ Student-t(nu)
 * 
 * Leverage: gamma < 0 means negative returns increase volatility (typical for equities)
 */
typedef struct {
    float rho;      // Persistence (0 < rho < 1, typically 0.9-0.99)
    float sigma_z;  // Vol-of-vol (typically 0.1-0.3)
    float mu;       // Long-run mean log-volatility (typically -5 to -3)
    float gamma;    // Leverage effect (typically -0.5 to 0 for equities, 0 for crypto)
} SVPFParams;

/**
 * @brief SVPF filter state (SoA layout for GPU)
 * 
 * All arrays have length n_particles, allocated on device.
 * 
 * OPTIMIZATION: All intermediate scalars live in device memory.
 * No D2H transfers during step execution - fully async pipeline.
 */
typedef struct {
    // Particle states
    float* h;           // [N] Current log-volatility particles
    float* h_prev;      // [N] Previous step (for AR(1) prior)
    float* h_pred;      // [N] Predicted particles (before Stein)
    
    // Stein computation workspace
    float* grad_log_p;  // [N] Gradient of log posterior
    float* kernel_sum;  // [N] Sum of kernel weights (attraction)
    float* grad_kernel_sum; // [N] Sum of kernel gradients (repulsion)
    
    // Likelihood computation
    float* log_weights; // [N] Log importance weights
    
    // Bandwidth computation (variance-based, O(N) not O(N²))
    float* d_h_centered;  // [N] Centered particles for variance computation
    
    // RNG states (Philox for fast CRN)
    curandStatePhilox4_32_10_t* rng_states;  // [N] CURAND Philox states
    
    // Reduction workspace (pre-allocated for CUB)
    float* d_reduce_buf;    // [N] Workspace for parallel reduction
    float* d_temp;          // [N] Temp buffer for intermediate results
    void* d_cub_temp;       // Pre-allocated CUB temp storage
    size_t cub_temp_bytes;  // Size of CUB temp storage
    
    // =========================================================================
    // DEVICE SCALARS - Eliminates PCIe roundtrips during step execution
    // These stay on GPU; kernels read/write via pointers, not D2H copies
    // =========================================================================
    float* d_scalar_max;       // [1] Max log-weight for log-sum-exp
    float* d_scalar_sum;       // [1] Sum for reductions  
    float* d_scalar_mean;      // [1] Mean for bandwidth computation
    float* d_scalar_bandwidth; // [1] Computed bandwidth (stays on GPU)
    float* d_y_prev;           // [1] Previous observation (for batch leverage)
    
    // Result buffer - stays on GPU until user calls svpf_get_result()
    float* d_result_loglik;    // [1] Log-likelihood increment
    float* d_result_vol_mean;  // [1] Volatility mean
    float* d_result_h_mean;    // [1] Log-vol mean
    
    // Configuration
    int n_particles;
    int n_stein_steps;
    float nu;               // Student-t degrees of freedom
    
    // Pre-computed Student-t constant
    float student_t_const;
    
    // State for seeded stepping (CRN reproducibility)
    int timestep;           // Current timestep (reset in initialize)
    float y_prev;           // Previous observation (for leverage effect)
    
    // Stream for async execution
    cudaStream_t stream;
    
} SVPFState;

/**
 * @brief Result of one SVPF filtering step
 */
typedef struct {
    float log_lik_increment;  // log p(y_t | y_{1:t-1}, theta) - for SMC²
    float vol_mean;           // E[exp(h/2)] - primary output for trading
    float vol_std;            // Std[exp(h/2)] - uncertainty estimate
    float h_mean;             // E[h] - log-volatility estimate
} SVPFResult;

// =============================================================================
// API: Core Filter Functions
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
 * Call before first observation or when resetting filter.
 * Particles drawn from N(mu, sigma_z² / (1 - rho²))
 * 
 * @param state SVPF state
 * @param params Model parameters
 * @param seed Random seed for reproducibility
 */
void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed);

/**
 * @brief Process one observation (main filtering step)
 * 
 * Pipeline:
 * 1. AR(1) prediction: h_pred = mu + rho*(h - mu) + sigma_z*noise
 * 2. Compute predictive likelihood p(y_t | h_pred)
 * 3. Stein transport: move particles toward posterior
 * 
 * @param state SVPF state
 * @param y_t Observation (return) at time t
 * @param params Model parameters
 * @param result Output: volatility estimate and likelihood
 */
void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result);

/**
 * @brief Process observation with seeded RNG (for SMC²/CPMMH)
 * 
 * Same as svpf_step but uses deterministic noise from seed.
 * Critical for correlated pseudo-marginal methods.
 * 
 * @param state SVPF state  
 * @param y_t Observation
 * @param params Model parameters
 * @param rng_seed Seed for this step's random numbers
 * @param result Output
 */
void svpf_step_seeded(SVPFState* state, float y_t, const SVPFParams* params,
                      unsigned long long rng_seed, SVPFResult* result);

// =============================================================================
// API: Batch Processing (Maximum Throughput - Single Sync for Full Sequence)
// =============================================================================

/**
 * @brief Process entire observation sequence with minimal sync overhead
 * 
 * Runs T timesteps with only ONE synchronization at the end.
 * ~10x faster than calling svpf_step T times.
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
 * 
 * For when observations are pre-loaded to device memory.
 * Avoids H2D copy of observations.
 * 
 * @param state SVPF state
 * @param d_observations Device array [T] of observations
 * @param T Number of observations
 * @param params Model parameters
 * @param d_loglik_out Device array [T] for outputs (pre-allocated)
 * @param d_vol_out Device array [T] for outputs (can be NULL, pre-allocated)
 */
void svpf_run_sequence_device(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
);

/**
 * @brief Ultra-fast sequence runner (for N <= 1024)
 * 
 * Uses single-block kernels with warp-shuffle reductions.
 * ~4x fewer kernel launches than standard version.
 * 
 * Kernel count per step:
 * - Standard: ~31 kernels
 * - Fast: 8 kernels (1 forward + 1 bandwidth + 5 Stein + 1 vol)
 * 
 * @param state SVPF state (must have n_particles <= 1024)
 * @param d_observations Device array [T]
 * @param T Number of observations  
 * @param params Model parameters
 * @param d_loglik_out Device array [T] for log-likelihoods
 * @param d_vol_out Device array [T] for volatilities (can be NULL)
 */
void svpf_run_sequence_fast(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
);

// =============================================================================
// API: Diagnostics
// =============================================================================

/**
 * @brief Copy particles to host (for diagnostics/plotting)
 * 
 * @param state SVPF state
 * @param h_out Host array of length n_particles
 */
void svpf_get_particles(const SVPFState* state, float* h_out);

/**
 * @brief Get current particle statistics
 * 
 * @param state SVPF state
 * @param h_mean Output: mean of h particles
 * @param h_std Output: std of h particles
 */
void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std);

#ifdef __cplusplus
}
#endif

#endif // SVPF_CUH
