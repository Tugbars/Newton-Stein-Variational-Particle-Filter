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
 * For batch processing (faster):
 *   svpf_optimized_init(N);
 *   svpf_run_sequence_optimized(filter, d_observations, T, &params, d_loglik, d_vol);
 *   svpf_optimized_cleanup();
 * 
 * Memory Layout: Structure of Arrays (SoA) for coalesced GPU access
 * 
 * References:
 * - Liu & Wang (2016): SVGD algorithm
 * - Fan et al. (2021): Stein Particle Filtering
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
    
    // Bandwidth computation
    float* d_h_centered;  // [N] Centered particles for variance computation
    
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
    float nu;
    float student_t_const;
    int timestep;
    float y_prev;
    cudaStream_t stream;
    
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
 * @brief Initialize optimized SVPF state (call once before using optimized API)
 */
void svpf_optimized_init(int n);

/**
 * @brief Cleanup optimized SVPF state
 */
void svpf_optimized_cleanup(void);

/**
 * @brief OPTIMIZED: Run full sequence with 2D tiled Stein kernel
 * 
 * Key optimizations:
 * - 2D tiled grid for O(N²) Stein: guarantees SM saturation for any N
 * - Shared memory tiling for bandwidth-efficient access
 * - CUB reductions for log-sum-exp
 * - Small N path (N ≤ 4096): persistent CTA, all data in SMEM
 * - NO device→host copies in loop (CUDA Graph compatible)
 * - EMA bandwidth smoothing for stability
 * 
 * @param state SVPF state
 * @param d_observations Device array [T]
 * @param T Number of observations
 * @param params Model parameters
 * @param d_loglik_out Device array [T] for log-likelihoods
 * @param d_vol_out Device array [T] for volatilities (can be NULL)
 */
void svpf_run_sequence_optimized(
    SVPFState* state,
    const float* d_observations,
    int T,
    const SVPFParams* params,
    float* d_loglik_out,
    float* d_vol_out
);

/**
 * @brief OPTIMIZED: Single step (for real-time usage)
 * 
 * Note: For best performance, use svpf_run_sequence_optimized instead.
 * Single-step API has unavoidable overhead from per-step memory allocation.
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
 * @brief CUDA GRAPH: Fastest sequence runner (minimal launch overhead)
 * 
 * Captures one timestep as a CUDA Graph, then replays it T times.
 * Reduces per-step overhead from ~65μs to ~5-10μs.
 * 
 * Note: Graph is cached and reused. First call has capture overhead.
 * 
 * @param state SVPF state
 * @param d_observations Device array [T]
 * @param T Number of observations
 * @param params Model parameters
 * @param d_loglik_out Device array [T] for log-likelihoods
 * @param d_vol_out Device array [T] for volatilities (can be NULL)
 */
void svpf_run_sequence_graph(
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
 */
void svpf_get_particles(const SVPFState* state, float* h_out);

/**
 * @brief Get current particle statistics
 */
void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std);

#ifdef __cplusplus
}
#endif

#endif // SVPF_CUH
