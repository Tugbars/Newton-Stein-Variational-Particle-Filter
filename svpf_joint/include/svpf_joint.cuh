/**
 * @file svpf_joint.cuh
 * @brief Joint State-Parameter Stein Variational Particle Filter
 * 
 * Key innovation: Parameters are part of each particle's state, not external.
 * This eliminates the "injection problem" during regime changes.
 * 
 * Each particle carries: [h, μ̃, ρ̃, σ̃]
 * - h:  log-volatility (unconstrained)
 * - μ̃:  mean level (unconstrained, μ̃ = μ)
 * - ρ̃:  persistence (logit-transformed, ρ = sigmoid(ρ̃))
 * - σ̃:  vol-of-vol (log-transformed, σ = exp(σ̃))
 * 
 * Gradients flow through BOTH state and parameters simultaneously.
 * During crashes, ∇_σ̃ = z² - 1 automatically inflates σ to explain large jumps.
 */

#ifndef SVPF_JOINT_CUH
#define SVPF_JOINT_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CONFIGURATION
// =============================================================================

#define SVPF_JOINT_DEFAULT_PARTICLES     512
#define SVPF_JOINT_DEFAULT_STEIN_STEPS   5
#define SVPF_JOINT_BLOCK_SIZE            256

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * @brief Configuration for Joint SVPF
 */
typedef struct {
    // Particle count
    int n_particles;
    int n_stein_steps;
    
    // Learning rates (per-parameter) - h is fast, θ is slow
    float step_h;           // 0.10 (fast state tracking)
    float step_mu;          // 0.01 (slow parameter learning)
    float step_rho;         // 0.005 (very slow)
    float step_sigma;       // 0.01 (slow)
    
    // Parameter diffusion rates (small random walk for well-posed filtering)
    float diffusion_mu;     // 0.01
    float diffusion_rho;    // 0.001
    float diffusion_sigma;  // 0.005
    
    // Diversity collapse thresholds (in unconstrained space)
    float collapse_thresh_mu;     // 0.05
    float collapse_thresh_rho;    // 0.02
    float collapse_thresh_sigma;  // 0.02
    
    // Prior hyperparameters (weak regularization)
    float mu_prior_mean;    // -3.5 (log-vol ~ -3.5 → vol ~ 17%)
    float mu_prior_var;     // 10.0 (wide)
    float rho_prior_mean;   // 2.0 (sigmoid(2) ≈ 0.88)
    float rho_prior_var;    // 5.0 (wide)
    float sigma_prior_mean; // -2.0 (exp(-2) ≈ 0.14)
    float sigma_prior_var;  // 5.0 (wide)
    
    // Student-t degrees of freedom
    float nu;
    
    // Precomputed constants
    float student_t_const;          // lgamma terms for likelihood
    float student_t_implied_offset; // For observation → h mapping
    
    // Gradient configuration
    float lik_offset;       // Bias correction for exact gradient (0.30)
    float prior_weight;     // Weight of prior regularization (0.01)
    
} SVPFJointConfig;

/**
 * @brief Diagnostics output from Joint SVPF
 */
typedef struct {
    // State estimate
    float h_mean;
    float vol_mean;
    
    // Parameter estimates (constrained space)
    float mu_mean;
    float rho_mean;
    float sigma_mean;
    
    // Parameter uncertainty (constrained space)
    float mu_std;
    float rho_std;
    float sigma_std;
    
    // Diversity metrics (unconstrained space) - for collapse detection
    float std_mu_tilde;
    float std_rho_tilde;
    float std_sigma_tilde;
    
    // Collapse flags (1 = collapsed)
    int mu_collapsed;
    int rho_collapsed;
    int sigma_collapsed;
    
    // ESS and likelihood
    float ess;
    float log_likelihood;
    
} SVPFJointDiagnostics;

/**
 * @brief Joint SVPF filter state
 */
typedef struct {
    // === PARTICLE ARRAYS (SoA layout) ===
    float* d_h;             // [N] log-volatility
    float* d_h_prev;        // [N] previous h (for transition gradient)
    float* d_mu_tilde;      // [N] mean level (unconstrained)
    float* d_rho_tilde;     // [N] persistence (logit-transformed)
    float* d_sigma_tilde;   // [N] vol-of-vol (log-transformed)
    
    // === GRADIENT ARRAYS ===
    float* d_grad_h;
    float* d_grad_mu;
    float* d_grad_rho;
    float* d_grad_sigma;
    
    // === WEIGHT ARRAY ===
    float* d_log_w;         // [N] log importance weights
    
    // === BANDWIDTH (computed per step) ===
    float bw_h;
    float bw_mu;
    float bw_rho;
    float bw_sigma;
    
    // === RNG ===
    curandStatePhilox4_32_10_t* d_rng;
    
    // === REDUCTION WORKSPACE ===
    float* d_temp;          // [N] scratch buffer
    float* d_reduce_out;    // [8] reduction outputs
    
    // === DIAGNOSTIC OUTPUTS (device) ===
    float* d_param_mean;    // [3] μ, ρ, σ means
    float* d_param_std;     // [3] μ, ρ, σ stds
    float* d_std_unconstrained; // [3] std in unconstrained space
    int* d_collapse_flags;  // [3] collapse indicators
    
    // === CONFIGURATION ===
    SVPFJointConfig config;
    
    // === RUNTIME STATE ===
    int timestep;
    float y_prev;
    cudaStream_t stream;
    
} SVPFJointState;

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * @brief Create default configuration
 */
SVPFJointConfig svpf_joint_default_config(void);

/**
 * @brief Allocate filter state
 */
SVPFJointState* svpf_joint_create(const SVPFJointConfig* config, cudaStream_t stream);

/**
 * @brief Initialize particles (call after create)
 * @param seed Random seed for reproducibility
 */
void svpf_joint_initialize(SVPFJointState* state, unsigned long long seed);

/**
 * @brief Process one observation
 * @param y_t Current observation
 * @param diag Optional diagnostics output (can be NULL)
 */
void svpf_joint_step(SVPFJointState* state, float y_t, SVPFJointDiagnostics* diag);

/**
 * @brief Get current volatility estimate
 */
float svpf_joint_get_vol(const SVPFJointState* state);

/**
 * @brief Get current parameter estimates
 */
void svpf_joint_get_params(const SVPFJointState* state, float* mu, float* rho, float* sigma);

/**
 * @brief Free filter state
 */
void svpf_joint_destroy(SVPFJointState* state);

// =============================================================================
// KERNEL DECLARATIONS
// =============================================================================

__global__ void svpf_joint_init_rng_kernel(
    curandStatePhilox4_32_10_t* rng,
    unsigned long long seed,
    int n
);

__global__ void svpf_joint_init_particles_kernel(
    float* d_h,
    float* d_mu_tilde,
    float* d_rho_tilde,
    float* d_sigma_tilde,
    curandStatePhilox4_32_10_t* rng,
    float mu_init, float rho_init, float sigma_init,
    float h_spread, float param_spread,
    int n
);

__global__ void svpf_joint_predict_kernel(
    float* __restrict__ d_h,
    float* __restrict__ d_h_prev,
    float* __restrict__ d_mu_tilde,
    float* __restrict__ d_rho_tilde,
    float* __restrict__ d_sigma_tilde,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float diffusion_mu,
    float diffusion_rho,
    float diffusion_sigma,
    int n
);

__global__ void svpf_joint_gradient_kernel(
    const float* __restrict__ d_h,
    const float* __restrict__ d_h_prev,
    const float* __restrict__ d_mu_tilde,
    const float* __restrict__ d_rho_tilde,
    const float* __restrict__ d_sigma_tilde,
    float* __restrict__ d_grad_h,
    float* __restrict__ d_grad_mu,
    float* __restrict__ d_grad_rho,
    float* __restrict__ d_grad_sigma,
    float* __restrict__ d_log_w,
    float y_t,
    float nu,
    float student_t_const,
    float lik_offset,
    float mu_prior_mean, float mu_prior_var,
    float rho_prior_mean, float rho_prior_var,
    float sigma_prior_mean, float sigma_prior_var,
    float prior_weight,
    int n
);

__global__ void svpf_joint_stein_kernel(
    float* __restrict__ d_h,
    float* __restrict__ d_mu_tilde,
    float* __restrict__ d_rho_tilde,
    float* __restrict__ d_sigma_tilde,
    const float* __restrict__ d_grad_h,
    const float* __restrict__ d_grad_mu,
    const float* __restrict__ d_grad_rho,
    const float* __restrict__ d_grad_sigma,
    const float* __restrict__ d_h_prev,  // For surprise detection
    float y_t,  // For surprise detection
    float bw_h, float bw_mu, float bw_rho, float bw_sigma,
    float step_h, float step_mu, float step_rho, float step_sigma,
    int n
);

__global__ void svpf_joint_extract_kernel(
    const float* __restrict__ d_h,
    const float* __restrict__ d_mu_tilde,
    const float* __restrict__ d_rho_tilde,
    const float* __restrict__ d_sigma_tilde,
    float* __restrict__ d_param_mean,
    float* __restrict__ d_param_std,
    float* __restrict__ d_std_unconstrained,
    int* __restrict__ d_collapse_flags,
    float collapse_thresh_mu,
    float collapse_thresh_rho,
    float collapse_thresh_sigma,
    int n
);

__global__ void svpf_joint_compute_bandwidth_kernel(
    const float* __restrict__ d_values,
    float* __restrict__ d_output,
    int n
);

#ifdef __cplusplus
}
#endif

#endif // SVPF_JOINT_CUH
