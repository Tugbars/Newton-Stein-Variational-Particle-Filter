/**
 * @file gpu_bpf.cuh
 * @brief GPU Bootstrap SIR Particle Filter — zero bias, zero heuristics
 *
 * 50K particles, systematic resampling, thrust reductions.
 * Provides both batch RMSE evaluation and a tick-by-tick streaming API.
 */

#ifndef GPU_BPF_CUH
#define GPU_BPF_CUH

#include <curand_kernel.h>

// ─── Batch RMSE evaluation (for testing) ─────────────────────────────────────

double gpu_bpf_run_rmse(
    const double* returns,      // observed returns [n_ticks]
    const double* true_h,       // ground truth log-vol [n_ticks]
    int n_ticks,
    int n_particles,
    float rho, float sigma_z, float mu,
    float nu_state,             // 0 = Gaussian state
    float nu_obs,               // 0 = Gaussian obs
    int seed
);

// ─── Streaming API (for production) ──────────────────────────────────────────

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

// Returns filtered h estimate (posterior mean) for this tick
float gpu_bpf_step(GpuBpfState* state, float y_t);

void gpu_bpf_destroy(GpuBpfState* state);

#endif // GPU_BPF_CUH
