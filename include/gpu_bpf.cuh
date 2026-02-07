/**
 * @file gpu_bpf.cuh
 * @brief GPU Bootstrap PF + Interacting Multiple Model filter
 *
 * Layer 1: GpuBpfState  — single-model BPF, N particles, zero heuristics
 * Layer 2: GpuImmState  — K models with Markov switching, each running a BPF
 *
 * IMM per tick:
 *   1. Interaction: mix model probabilities via transition matrix
 *   2. Each BPF: propagate, weight, estimate, resample
 *   3. Update: π_k ∝ π_k × p(y_t | model_k)
 *   4. Output: h = Σ_k π_k × h_k,  vol = exp(h/2)
 */

#ifndef GPU_BPF_CUH
#define GPU_BPF_CUH

#include <curand_kernel.h>

#define IMM_MAX_MODELS 64

// =============================================================================
// Layer 1: Single-Model BPF
// =============================================================================

typedef struct {
    float h_mean;       // Posterior mean of h
    float log_lik;      // Log marginal likelihood p(y_t | y_{1:t-1}, model)
} BpfResult;

typedef struct {
    float* d_h;
    float* d_h2;
    float* d_log_w;
    float* d_w;
    float* d_cdf;
    float* d_wh;
    curandState* d_rng;
    int n_particles;
    int block;
    int grid;
    float rho, sigma_z, mu, nu_state, nu_obs;
    unsigned long long host_rng_state;
    int timestep;
} GpuBpfState;

GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed);

BpfResult gpu_bpf_step(GpuBpfState* state, float y_t);

void gpu_bpf_destroy(GpuBpfState* state);

// Batch RMSE (for testing)
double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
);

// =============================================================================
// Layer 2: IMM — Interacting Multiple Model
// =============================================================================

typedef struct {
    float rho;
    float sigma_z;
    float mu;
    float nu_state;     // 0 = Gaussian
    float nu_obs;       // 0 = Gaussian
} ImmModelParams;

typedef struct {
    float h_mean;       // Mixed posterior mean
    float vol;          // exp(h_mean / 2)
    float log_lik;      // Mixed log-likelihood
    int best_model;     // Highest probability model index
    float best_prob;    // Probability of best model
} ImmResult;

typedef struct {
    GpuBpfState** filters;          // K BPF instances
    int n_models;
    int n_particles_per_model;
    
    // Model probabilities (log-space for stability)
    double* log_pi;                 // [K] log model probabilities
    double* log_pi_pred;            // [K] after interaction step
    
    // Transition matrix (row-major, log-space)
    // T[i][j] = log P(model_j at t | model_i at t-1)
    double* log_T;                  // [K x K]
    
    int timestep;
} GpuImmState;

// Create IMM with K models. transition_matrix is row-major [K x K] probabilities
// (NOT log — will be converted internally). Pass NULL for uniform transitions.
GpuImmState* gpu_imm_create(
    const ImmModelParams* models, int n_models,
    int n_particles_per_model,
    const float* transition_matrix,     // [K x K] or NULL for uniform
    int seed
);

ImmResult gpu_imm_step(GpuImmState* state, float y_t);

// Get model probability for model k (linear scale)
float gpu_imm_get_prob(const GpuImmState* state, int k);

// Get all model probabilities
void gpu_imm_get_probs(const GpuImmState* state, float* probs_out);

// Get per-model h estimates from last step
void gpu_imm_get_model_h(const GpuImmState* state, float* h_out);

void gpu_imm_destroy(GpuImmState* state);

// =============================================================================
// Convenience: grid builder
// =============================================================================

// Build a parameter grid from arrays. Returns allocated ImmModelParams[n_rho * n_sigma * n_mu].
// Caller must free().
ImmModelParams* gpu_imm_build_grid(
    const float* rhos, int n_rho,
    const float* sigma_zs, int n_sigma,
    const float* mus, int n_mu,
    float nu_state, float nu_obs,
    int* out_n_models
);

#endif // GPU_BPF_CUH
