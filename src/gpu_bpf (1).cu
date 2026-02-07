/**
 * @file gpu_bpf.cu
 * @brief GPU Bootstrap PF + IMM implementation
 */

#include "gpu_bpf.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

// =============================================================================
// Device helpers
// =============================================================================

__device__ static float bpf_sample_t(curandState* s, float nu) {
    if (nu <= 0.0f || nu > 100.0f) return curand_normal(s);
    float z = curand_normal(s);
    float chi2 = 0.0f;
    for (int k = 0; k < (int)nu; k++) {
        float g = curand_normal(s);
        chi2 += g * g;
    }
    return z * rsqrtf(chi2 / nu);
}

__device__ static float bpf_log_t_pdf(float x, float nu) {
    return lgammaf((nu + 1.0f) / 2.0f) - lgammaf(nu / 2.0f)
         - 0.5f * logf(nu * 3.14159265f)
         - (nu + 1.0f) / 2.0f * logf(1.0f + x * x / nu);
}

// =============================================================================
// BPF Kernels
// =============================================================================

__global__ void bpf_init_rng(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

__global__ void bpf_init_particles(float* h, curandState* states,
                                    float mu, float std_stat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = mu + std_stat * curand_normal(&states[i]);
}

__global__ void bpf_propagate_weight(
    float* h, float* log_w, curandState* states,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, float y_t,
    int n, int do_propagate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    if (do_propagate) {
        float eps = bpf_sample_t(&states[i], nu_state);
        h[i] = mu + rho * (h[i] - mu) + sigma_z * eps;
    }
    
    float h_i = h[i];
    float eta = y_t * __expf(-h_i * 0.5f);
    log_w[i] = (nu_obs > 0.0f)
        ? bpf_log_t_pdf(eta, nu_obs) - h_i * 0.5f
        : -0.9189385f - 0.5f * eta * eta - h_i * 0.5f;
}

__global__ void bpf_exp_sub(float* w, const float* log_w, float max_lw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __expf(log_w[i] - max_lw);
}

__global__ void bpf_scale_and_wh(float* w, float* wh, const float* h,
                                  float inv_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] *= inv_sum;
        wh[i] = w[i] * h[i];
    }
}

__global__ void bpf_resample(float* h_out, const float* h_in,
                              const float* cdf, float u_base, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float target = u_base + (float)i / (float)n;
    if (target >= 1.0f) target -= 1.0f;
    
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    h_out[i] = h_in[lo];
}

// =============================================================================
// Host-side PCG32
// =============================================================================

static inline unsigned int bpf_pcg32(unsigned long long* state) {
    unsigned long long old = *state;
    *state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xor_shifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

static inline float bpf_pcg32_float(unsigned long long* state) {
    return (float)(bpf_pcg32(state) >> 9) * (1.0f / 8388608.0f);
}

// =============================================================================
// BPF Streaming API
// =============================================================================

GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed) {
    GpuBpfState* s = (GpuBpfState*)calloc(1, sizeof(GpuBpfState));
    s->n_particles = n_particles;
    s->rho = rho;
    s->sigma_z = sigma_z;
    s->mu = mu;
    s->nu_state = nu_state;
    s->nu_obs = nu_obs;
    s->block = 256;
    s->grid = (n_particles + s->block - 1) / s->block;
    s->host_rng_state = (unsigned long long)seed * 67890ULL + 12345ULL;
    s->timestep = 0;
    
    cudaMalloc(&s->d_h,     n_particles * sizeof(float));
    cudaMalloc(&s->d_h2,    n_particles * sizeof(float));
    cudaMalloc(&s->d_log_w, n_particles * sizeof(float));
    cudaMalloc(&s->d_w,     n_particles * sizeof(float));
    cudaMalloc(&s->d_cdf,   n_particles * sizeof(float));
    cudaMalloc(&s->d_wh,    n_particles * sizeof(float));
    cudaMalloc(&s->d_rng,   n_particles * sizeof(curandState));
    
    bpf_init_rng<<<s->grid, s->block>>>(s->d_rng, (unsigned long long)seed, n_particles);
    
    float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
    bpf_init_particles<<<s->grid, s->block>>>(s->d_h, s->d_rng, mu, std_stat, n_particles);
    cudaDeviceSynchronize();
    
    return s;
}

BpfResult gpu_bpf_step(GpuBpfState* s, float y_t) {
    int n = s->n_particles;
    
    bpf_propagate_weight<<<s->grid, s->block>>>(
        s->d_h, s->d_log_w, s->d_rng,
        s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t,
        n, (s->timestep > 0) ? 1 : 0);
    
    thrust::device_ptr<float> t_log_w(s->d_log_w);
    thrust::device_ptr<float> t_w(s->d_w);
    thrust::device_ptr<float> t_wh(s->d_wh);
    thrust::device_ptr<float> t_cdf(s->d_cdf);
    
    float max_lw = *thrust::max_element(t_log_w, t_log_w + n);
    bpf_exp_sub<<<s->grid, s->block>>>(s->d_w, s->d_log_w, max_lw, n);
    float sum_w = thrust::reduce(t_w, t_w + n, 0.0f);
    bpf_scale_and_wh<<<s->grid, s->block>>>(s->d_w, s->d_wh, s->d_h, 1.0f / sum_w, n);
    float h_est = thrust::reduce(t_wh, t_wh + n, 0.0f);
    
    thrust::inclusive_scan(t_w, t_w + n, t_cdf);
    float u = bpf_pcg32_float(&s->host_rng_state) / (float)n;
    bpf_resample<<<s->grid, s->block>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
    
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;
    s->timestep++;
    
    // Log marginal likelihood: log p(y_t | y_{1:t-1}) = max_lw + log(mean(w))
    float log_lik = max_lw + logf(fmaxf(sum_w / (float)n, 1e-30f));
    
    BpfResult r;
    r.h_mean = h_est;
    r.log_lik = log_lik;
    return r;
}

void gpu_bpf_destroy(GpuBpfState* s) {
    if (!s) return;
    cudaFree(s->d_h);
    cudaFree(s->d_h2);
    cudaFree(s->d_log_w);
    cudaFree(s->d_w);
    cudaFree(s->d_cdf);
    cudaFree(s->d_wh);
    cudaFree(s->d_rng);
    free(s);
}

// =============================================================================
// BPF Batch RMSE
// =============================================================================

double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
) {
    GpuBpfState* state = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                         nu_state, nu_obs, seed);
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;
    
    for (int t = 0; t < n_ticks; t++) {
        BpfResult r = gpu_bpf_step(state, (float)returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - true_h[t];
            sum_sq += err * err;
            count++;
        }
    }
    
    cudaDeviceSynchronize();
    gpu_bpf_destroy(state);
    return sqrt(sum_sq / count);
}

// =============================================================================
// IMM: Log-Sum-Exp utility
// =============================================================================

static double log_sum_exp(const double* x, int n) {
    double mx = -1e30;
    for (int i = 0; i < n; i++)
        if (x[i] > mx) mx = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += exp(x[i] - mx);
    return mx + log(s);
}

// =============================================================================
// IMM Create
// =============================================================================

GpuImmState* gpu_imm_create(
    const ImmModelParams* models, int n_models,
    int n_particles_per_model,
    const float* transition_matrix,
    int seed
) {
    if (n_models > IMM_MAX_MODELS) {
        fprintf(stderr, "IMM: n_models=%d exceeds max=%d\n", n_models, IMM_MAX_MODELS);
        return NULL;
    }
    
    GpuImmState* s = (GpuImmState*)calloc(1, sizeof(GpuImmState));
    s->n_models = n_models;
    s->n_particles_per_model = n_particles_per_model;
    s->timestep = 0;
    
    // Allocate filters
    s->filters = (GpuBpfState**)malloc(n_models * sizeof(GpuBpfState*));
    for (int k = 0; k < n_models; k++) {
        s->filters[k] = gpu_bpf_create(
            n_particles_per_model,
            models[k].rho, models[k].sigma_z, models[k].mu,
            models[k].nu_state, models[k].nu_obs,
            seed + k * 7919  // different seed per model
        );
    }
    
    // Uniform initial model probabilities (log-space)
    s->log_pi = (double*)malloc(n_models * sizeof(double));
    s->log_pi_pred = (double*)malloc(n_models * sizeof(double));
    double log_uniform = -log((double)n_models);
    for (int k = 0; k < n_models; k++)
        s->log_pi[k] = log_uniform;
    
    // Transition matrix (log-space)
    s->log_T = (double*)malloc(n_models * n_models * sizeof(double));
    if (transition_matrix) {
        for (int i = 0; i < n_models * n_models; i++) {
            s->log_T[i] = log(fmax((double)transition_matrix[i], 1e-30));
        }
    } else {
        // Default: high self-transition, uniform off-diagonal
        // P(stay) = 0.95, P(switch to any other) = 0.05 / (K-1)
        double p_stay = 0.95;
        double p_switch = (n_models > 1) ? (1.0 - p_stay) / (n_models - 1) : 0.0;
        for (int i = 0; i < n_models; i++) {
            for (int j = 0; j < n_models; j++) {
                double p = (i == j) ? p_stay : p_switch;
                s->log_T[i * n_models + j] = log(fmax(p, 1e-30));
            }
        }
    }
    
    return s;
}

// =============================================================================
// IMM Step
// =============================================================================

ImmResult gpu_imm_step(GpuImmState* s, float y_t) {
    int K = s->n_models;
    
    // ─── 1. Interaction: π_k^- = Σ_j T(j→k) * π_j ───
    // In log-space: log π_k^- = log Σ_j exp(log T[j][k] + log π_j)
    for (int k = 0; k < K; k++) {
        double terms[IMM_MAX_MODELS];
        for (int j = 0; j < K; j++) {
            terms[j] = s->log_T[j * K + k] + s->log_pi[j];
        }
        s->log_pi_pred[k] = log_sum_exp(terms, K);
    }
    
    // ─── 2. Run each BPF, collect h_est and log-likelihood ───
    double log_liks[IMM_MAX_MODELS];
    float h_ests[IMM_MAX_MODELS];
    
    for (int k = 0; k < K; k++) {
        BpfResult r = gpu_bpf_step(s->filters[k], y_t);
        h_ests[k] = r.h_mean;
        log_liks[k] = (double)r.log_lik;
    }
    
    // ─── 3. Update: log π_k = log π_k^- + log p(y|model_k) - log Z ───
    double log_joint[IMM_MAX_MODELS];
    for (int k = 0; k < K; k++)
        log_joint[k] = s->log_pi_pred[k] + log_liks[k];
    
    double log_Z = log_sum_exp(log_joint, K);
    
    for (int k = 0; k < K; k++)
        s->log_pi[k] = log_joint[k] - log_Z;
    
    // ─── 4. Mixed output ───
    // h = Σ_k π_k * h_k (in linear probability space)
    double h_mixed = 0.0;
    int best_k = 0;
    double best_log_pi = -1e30;
    
    for (int k = 0; k < K; k++) {
        double pi_k = exp(s->log_pi[k]);
        h_mixed += pi_k * (double)h_ests[k];
        if (s->log_pi[k] > best_log_pi) {
            best_log_pi = s->log_pi[k];
            best_k = k;
        }
    }
    
    s->timestep++;
    
    ImmResult r;
    r.h_mean = (float)h_mixed;
    r.vol = expf((float)h_mixed * 0.5f);
    r.log_lik = (float)log_Z;
    r.best_model = best_k;
    r.best_prob = (float)exp(best_log_pi);
    return r;
}

// =============================================================================
// IMM Accessors
// =============================================================================

float gpu_imm_get_prob(const GpuImmState* state, int k) {
    if (k < 0 || k >= state->n_models) return 0.0f;
    return (float)exp(state->log_pi[k]);
}

void gpu_imm_get_probs(const GpuImmState* state, float* probs_out) {
    for (int k = 0; k < state->n_models; k++)
        probs_out[k] = (float)exp(state->log_pi[k]);
}

void gpu_imm_get_model_h(const GpuImmState* state, float* h_out) {
    // Would need to store last h_ests — add if needed
    (void)state; (void)h_out;
}

void gpu_imm_destroy(GpuImmState* s) {
    if (!s) return;
    for (int k = 0; k < s->n_models; k++)
        gpu_bpf_destroy(s->filters[k]);
    free(s->filters);
    free(s->log_pi);
    free(s->log_pi_pred);
    free(s->log_T);
    free(s);
}

// =============================================================================
// Grid Builder
// =============================================================================

ImmModelParams* gpu_imm_build_grid(
    const float* rhos, int n_rho,
    const float* sigma_zs, int n_sigma,
    const float* mus, int n_mu,
    float nu_state, float nu_obs,
    int* out_n_models
) {
    int total = n_rho * n_sigma * n_mu;
    ImmModelParams* grid = (ImmModelParams*)malloc(total * sizeof(ImmModelParams));
    int idx = 0;
    for (int r = 0; r < n_rho; r++) {
        for (int s = 0; s < n_sigma; s++) {
            for (int m = 0; m < n_mu; m++) {
                grid[idx].rho = rhos[r];
                grid[idx].sigma_z = sigma_zs[s];
                grid[idx].mu = mus[m];
                grid[idx].nu_state = nu_state;
                grid[idx].nu_obs = nu_obs;
                idx++;
            }
        }
    }
    *out_n_models = total;
    return grid;
}
