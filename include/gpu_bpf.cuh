/**
 * @file gpu_bpf.cuh
 * @brief GPU Bootstrap PF + IMM with CUDA stream parallelism
 *
 * Layer 1: GpuBpfState  — single-model BPF with async step on its own stream
 * Layer 2: GpuImmState  — K models running concurrently on K streams
 *
 * All K BPF steps launch asynchronously, one cudaDeviceSynchronize,
 * then results are collected. No host-sync thrust calls in the hot path.
 */

#ifndef GPU_BPF_CUH
#define GPU_BPF_CUH

#include <curand_kernel.h>

#define IMM_MAX_MODELS 64

// =============================================================================
// Layer 1: Single-Model BPF
// =============================================================================

typedef struct {
    float h_mean;
    float log_lik;
} BpfResult;

typedef struct {
    // Particle arrays
    float* d_h;
    float* d_h2;
    float* d_log_w;
    float* d_w;
    float* d_cdf;
    float* d_wh;
    curandState* d_rng;

    // Async result storage (device-side)
    float* d_scalars;       // [4]: {max_lw, sum_w, h_est, log_lik}

    // Launch config
    int n_particles;
    int block;
    int grid;
    cudaStream_t stream;

    // Model params
    float rho, sigma_z, mu, nu_state, nu_obs;

    // Host RNG for resampling uniform
    unsigned long long host_rng_state;
    int timestep;
} GpuBpfState;

// Create / destroy
GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed);
void gpu_bpf_destroy(GpuBpfState* state);

// Synchronous step (launches + waits + returns result)
BpfResult gpu_bpf_step(GpuBpfState* state, float y_t);

// Async two-phase API (for IMM parallelism)
void gpu_bpf_step_async(GpuBpfState* state, float y_t);
BpfResult gpu_bpf_get_result(GpuBpfState* state);  // call after cudaDeviceSynchronize

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
    float nu_state;
    float nu_obs;
} ImmModelParams;

typedef struct {
    float h_mean;
    float vol;
    float log_lik;
    int best_model;
    float best_prob;
} ImmResult;

typedef struct {
    GpuBpfState** filters;
    int n_models;
    int n_particles_per_model;
    double* log_pi;
    double* log_pi_pred;
    double* log_T;              // [K x K] row-major, log-space
    int timestep;
} GpuImmState;

GpuImmState* gpu_imm_create(
    const ImmModelParams* models, int n_models,
    int n_particles_per_model,
    const float* transition_matrix,     // [K x K] or NULL
    int seed
);

ImmResult gpu_imm_step(GpuImmState* state, float y_t);
float gpu_imm_get_prob(const GpuImmState* state, int k);
void gpu_imm_get_probs(const GpuImmState* state, float* probs_out);
void gpu_imm_destroy(GpuImmState* state);

// Grid builder
ImmModelParams* gpu_imm_build_grid(
    const float* rhos, int n_rho,
    const float* sigma_zs, int n_sigma,
    const float* mus, int n_mu,
    float nu_state, float nu_obs,
    int* out_n_models
);

#endif // GPU_BPF_CUH
