/**
 * @file cpmmh_svpf.cu
 * @brief CPMMH Implementation for SVPF Parameter Learning
 * 
 * Contains:
 * - Noise buffer management
 * - SVPF replay kernels (deterministic mode)
 * - CPMMH state management
 * - Core MH stepping logic
 * - Adaptive proposal updates
 */

#include "cpmmh_svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CUDA ERROR CHECKING
 *═══════════════════════════════════════════════════════════════════════════════*/

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST RNG (xorshift64*)
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline uint64_t xorshift64star(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float host_uniform(uint64_t* state) {
    return (float)(xorshift64star(state) >> 11) * (1.0f / 9007199254740992.0f);
}

static inline float host_normal(uint64_t* state) {
    /* Box-Muller transform */
    float u1 = host_uniform(state);
    float u2 = host_uniform(state);
    u1 = fmaxf(u1, 1e-10f);  /* Avoid log(0) */
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * NOISE BUFFER IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

SVPFNoiseBuffer* svpf_noise_buffer_create(int T_capacity, int N_particles) {
    SVPFNoiseBuffer* buf = (SVPFNoiseBuffer*)malloc(sizeof(SVPFNoiseBuffer));
    if (!buf) return NULL;
    
    buf->T_capacity = T_capacity;
    buf->N_particles = N_particles;
    buf->T_current = 0;
    
    size_t bytes = (size_t)T_capacity * N_particles * sizeof(float);
    CUDA_CHECK(cudaMalloc(&buf->d_noise, bytes));
    CUDA_CHECK(cudaMemset(buf->d_noise, 0, bytes));
    
    return buf;
}

void svpf_noise_buffer_destroy(SVPFNoiseBuffer* buf) {
    if (!buf) return;
    cudaFree(buf->d_noise);
    free(buf);
}

/* Kernel: Fill buffer with Gaussian noise */
__global__ void kernel_fill_gaussian_noise(
    float* d_noise,
    int T,
    int N,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * N;
    
    if (idx < total) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        d_noise[idx] = curand_normal(&rng);
    }
}

void svpf_noise_buffer_fill_gaussian(
    SVPFNoiseBuffer* buf,
    int T,
    unsigned long long seed,
    cudaStream_t stream
) {
    if (T > buf->T_capacity) {
        fprintf(stderr, "Error: T=%d exceeds buffer capacity %d\n", T, buf->T_capacity);
        return;
    }
    
    int total = T * buf->N_particles;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    kernel_fill_gaussian_noise<<<grid, block, 0, stream>>>(
        buf->d_noise, T, buf->N_particles, seed
    );
    
    buf->T_current = T;
}

/* Kernel: Correlate noise dst = rho * src + sqrt(1-rho²) * fresh */
__global__ void kernel_correlate_noise(
    const float* src,
    float* dst,
    int T,
    int N,
    float rho,
    float scale,  /* sqrt(1 - rho²) */
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * N;
    
    if (idx < total) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        
        float src_val = src[idx];
        float fresh = curand_normal(&rng);
        dst[idx] = rho * src_val + scale * fresh;
    }
}

void svpf_noise_buffer_correlate(
    const SVPFNoiseBuffer* src,
    SVPFNoiseBuffer* dst,
    float rho,
    int T,
    unsigned long long seed,
    cudaStream_t stream
) {
    float scale = sqrtf(1.0f - rho * rho);
    int total = T * src->N_particles;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    kernel_correlate_noise<<<grid, block, 0, stream>>>(
        src->d_noise, dst->d_noise, T, src->N_particles, rho, scale, seed
    );
    
    dst->T_current = T;
}

void svpf_noise_buffer_copy(
    const SVPFNoiseBuffer* src,
    SVPFNoiseBuffer* dst,
    int T,
    cudaStream_t stream
) {
    size_t bytes = (size_t)T * src->N_particles * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(dst->d_noise, src->d_noise, bytes, 
                               cudaMemcpyDeviceToDevice, stream));
    dst->T_current = T;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SVPF REPLAY STATE IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

SVPFReplayState* svpf_replay_create(int N_particles, const SVPFReplayConfig* config) {
    SVPFReplayState* state = (SVPFReplayState*)calloc(1, sizeof(SVPFReplayState));
    if (!state) return NULL;
    
    state->N = N_particles;
    state->config = config ? *config : svpf_replay_config_default();
    
    /* Allocate particle arrays */
    size_t N_bytes = N_particles * sizeof(float);
    CUDA_CHECK(cudaMalloc(&state->d_h, N_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_h_prev, N_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_grad_log_p, N_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_phi, N_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_grad_v, N_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_reduce_buf, N_bytes));
    
    /* Scalars */
    CUDA_CHECK(cudaMalloc(&state->d_bandwidth, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_log_lik_inc, sizeof(float)));
    
    /* CUB temp storage */
    float* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, N_bytes));
    state->cub_temp_bytes = 0;
    cub::DeviceReduce::Sum(NULL, state->cub_temp_bytes, d_dummy, state->d_bandwidth, N_particles);
    state->cub_temp_bytes += 1024;
    CUDA_CHECK(cudaMalloc(&state->d_cub_temp, state->cub_temp_bytes));
    cudaFree(d_dummy);
    
    /* Precompute Student-t constants */
    float nu = state->config.nu_obs;
    state->student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                           - lgammaf(nu / 2.0f) 
                           - 0.5f * logf((float)M_PI * nu);
    
    /* E[log(t²)] offset for Student-t implied vol */
    float psi_half = -1.9635100260214235f;
    float nu_half = nu / 2.0f;
    float psi_nu_half = (nu_half >= 1.0f) 
                       ? (logf(nu_half) - 1.0f/(2.0f*nu_half))
                       : (-0.5772156649f - 1.0f/nu_half);
    state->student_t_offset = -(logf(nu) + psi_half - psi_nu_half);
    
    CUDA_CHECK(cudaStreamCreate(&state->stream));
    
    return state;
}

void svpf_replay_destroy(SVPFReplayState* state) {
    if (!state) return;
    
    cudaFree(state->d_h);
    cudaFree(state->d_h_prev);
    cudaFree(state->d_grad_log_p);
    cudaFree(state->d_phi);
    cudaFree(state->d_grad_v);
    cudaFree(state->d_reduce_buf);
    cudaFree(state->d_bandwidth);
    cudaFree(state->d_log_lik_inc);
    cudaFree(state->d_cub_temp);
    cudaStreamDestroy(state->stream);
    
    free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SVPF REPLAY KERNELS
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Kernel: Initialize particles from stationary distribution */
__global__ void kernel_replay_init(
    float* d_h,
    const float* d_init_noise,
    float mu,
    float stationary_std,
    float h_min,
    float h_max,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float h = mu + stationary_std * d_init_noise[idx];
        d_h[idx] = fminf(fmaxf(h, h_min), h_max);
    }
}

void svpf_replay_reset(
    SVPFReplayState* state,
    float rho,
    float mu,
    float sigma_z,
    const float* d_init_noise
) {
    int N = state->N;
    int block = 256;
    int grid = (N + block - 1) / block;
    
    /* Compute stationary std */
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float base_var = (sigma_z * sigma_z) / one_minus_rho_sq;
    
    /* Student-t adjustment if enabled */
    float var_scale = 1.0f;
    if (state->config.use_student_t_state && state->config.nu_state > 2.0f) {
        var_scale = state->config.nu_state / (state->config.nu_state - 2.0f);
    }
    float stationary_std = sqrtf(base_var * var_scale);
    
    kernel_replay_init<<<grid, block, 0, state->stream>>>(
        state->d_h, d_init_noise, mu, stationary_std,
        state->config.h_min, state->config.h_max, N
    );
    
    /* Copy to h_prev */
    CUDA_CHECK(cudaMemcpyAsync(state->d_h_prev, state->d_h, 
                               N * sizeof(float), cudaMemcpyDeviceToDevice, state->stream));
    
    /* Reset RMSProp accumulator */
    CUDA_CHECK(cudaMemsetAsync(state->d_grad_v, 0, N * sizeof(float), state->stream));
}

/* Kernel: Predict step with external noise (deterministic) */
__global__ void kernel_replay_predict(
    float* d_h,
    const float* d_h_prev,
    const float* d_noise,   /* External noise for this timestep */
    float rho,
    float mu,
    float sigma_z,
    float h_min,
    float h_max,
    int use_student_t_state,
    float nu_state,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float h_prev = d_h_prev[idx];
        float eps = d_noise[idx];
        
        /* AR(1) mean */
        float h_mean = mu + rho * (h_prev - mu);
        
        /* Student-t state: scale noise by sqrt((nu-2)/nu) for unit variance */
        if (use_student_t_state && nu_state > 2.0f) {
            /* Note: input eps is Gaussian, but effect is similar to Student-t
             * because we're using it for correlation structure, not exact distribution */
            eps *= sqrtf((nu_state - 2.0f) / nu_state);
        }
        
        float h_new = h_mean + sigma_z * eps;
        d_h[idx] = fminf(fmaxf(h_new, h_min), h_max);
    }
}

/* Kernel: Compute gradient of log posterior */
__global__ void kernel_replay_gradient(
    const float* d_h,
    const float* d_h_prev,
    float* d_grad_log_p,
    float y_t,
    float rho,
    float sigma_z,
    float mu,
    float nu_obs,
    float student_t_const,
    int use_student_t_state,
    float nu_state,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float h = d_h[idx];
        float h_prev = d_h_prev[idx];
        
        /* Log-likelihood gradient: d/dh log p(y|h) */
        /* Student-t: log p(y|h) = const - (ν+1)/2 * log(1 + y²/(ν·exp(h))) - h/2 */
        float exp_h = expf(h);
        float y_sq = y_t * y_t;
        float z_sq = y_sq / (nu_obs * exp_h + 1e-10f);
        float denom = 1.0f + z_sq;
        
        /* d/dh log p(y|h) = -0.5 + (ν+1)/2 * z² / (1 + z²) */
        float grad_lik = -0.5f + 0.5f * (nu_obs + 1.0f) * z_sq / denom;
        
        /* Prior gradient: d/dh log p(h|h_prev) */
        /* Gaussian AR(1): -0.5 * (h - μ - ρ(h_prev - μ))² / σ² */
        float h_mean = mu + rho * (h_prev - mu);
        float residual = h - h_mean;
        float sigma_z_sq = sigma_z * sigma_z + 1e-10f;
        
        float grad_prior;
        if (use_student_t_state && nu_state > 2.5f) {
            /* Student-t state: bounded gradient */
            float scale_sq = sigma_z_sq * nu_state;
            float z_state = residual * residual / scale_sq;
            grad_prior = -(nu_state + 1.0f) * residual / (scale_sq * (1.0f + z_state));
        } else {
            /* Gaussian state */
            grad_prior = -residual / sigma_z_sq;
        }
        
        d_grad_log_p[idx] = grad_lik + grad_prior;
    }
}

/* Kernel: Compute median-based bandwidth */
__global__ void kernel_compute_bandwidth(
    const float* d_h,
    float* d_bandwidth,
    int N
) {
    /* Simple variance-based bandwidth: h = 1.06 * σ * n^(-1/5) */
    /* More stable than median for small N */
    __shared__ float s_sum;
    __shared__ float s_sum_sq;
    
    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float h = d_h[i];
        local_sum += h;
        local_sum_sq += h * h;
    }
    
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float mean = s_sum / (float)N;
        float var = s_sum_sq / (float)N - mean * mean;
        float std = sqrtf(fmaxf(var, 1e-6f));
        
        /* Silverman's rule of thumb */
        float h = 1.06f * std * powf((float)N, -0.2f);
        h = fmaxf(h, 0.1f);  /* Floor */
        h = fminf(h, 2.0f);  /* Ceiling */
        
        *d_bandwidth = h;
    }
}

/* Kernel: Stein transport operator (deterministic, no SVLD noise) */
__global__ void kernel_replay_stein_transport(
    float* d_h,
    const float* d_grad_log_p,
    float* d_grad_v,         /* RMSProp accumulator */
    float bandwidth,
    float step_size,
    float rmsprop_rho,
    float rmsprop_eps,
    float h_min,
    float h_max,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float h_i = d_h[i];
    float grad_i = d_grad_log_p[i];
    float h_sq = bandwidth * bandwidth;
    float inv_h_sq = 1.0f / (h_sq + 1e-10f);
    
    /* Compute Stein operator: φ(x_i) = (1/N) Σ_j [k(x_j, x_i) ∇log p(x_j) + ∇_j k(x_j, x_i)] */
    float phi = 0.0f;
    
    for (int j = 0; j < N; j++) {
        float h_j = d_h[j];
        float grad_j = d_grad_log_p[j];
        
        float diff = h_j - h_i;
        float dist_sq = diff * diff;
        
        /* RBF kernel: k(x,y) = exp(-||x-y||² / (2h²)) */
        float k = expf(-0.5f * dist_sq * inv_h_sq);
        
        /* ∇_x k(x,y) = k(x,y) * (y-x) / h² */
        float grad_k = k * diff * inv_h_sq;
        
        /* Stein operator (repulsive form): k·∇log p + ∇k */
        phi += k * grad_j + grad_k;
    }
    phi /= (float)N;
    
    /* RMSProp update */
    float grad_v_old = d_grad_v[i];
    float grad_v_new = rmsprop_rho * grad_v_old + (1.0f - rmsprop_rho) * phi * phi;
    d_grad_v[i] = grad_v_new;
    
    float adaptive_step = step_size / (sqrtf(grad_v_new) + rmsprop_eps);
    
    /* Update particle (no SVLD noise in replay mode) */
    float h_new = h_i + adaptive_step * phi;
    d_h[i] = fminf(fmaxf(h_new, h_min), h_max);
}

/* Kernel: Compute log-likelihood increment */
__global__ void kernel_replay_log_lik(
    const float* d_h,
    float* d_log_lik_inc,
    float y_t,
    float nu_obs,
    float student_t_const,
    int N
) {
    __shared__ float s_sum;
    
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    float y_sq = y_t * y_t;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float h = d_h[i];
        float exp_h = expf(h);
        float z_sq = y_sq / (nu_obs * exp_h + 1e-10f);
        
        /* Student-t log density */
        float log_p = student_t_const - 0.5f * h 
                    - 0.5f * (nu_obs + 1.0f) * logf(1.0f + z_sq);
        
        local_sum += log_p;
    }
    
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        /* Average log-likelihood (particle approximation) */
        *d_log_lik_inc = s_sum / (float)N;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SVPF REPLAY STEP IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

float svpf_replay_step(
    SVPFReplayState* state,
    float y_t,
    float y_prev,
    const SVPFReplayParams* params,
    const float* d_noise_t
) {
    int N = state->N;
    int block = 256;
    int grid = (N + block - 1) / block;
    cudaStream_t cs = state->stream;
    
    SVPFReplayConfig* cfg = &state->config;
    
    (void)y_prev;  /* Not used in basic replay */
    
    /* 1. Predict with external noise */
    kernel_replay_predict<<<grid, block, 0, cs>>>(
        state->d_h, state->d_h_prev, d_noise_t,
        params->rho, params->mu, params->sigma_z,
        cfg->h_min, cfg->h_max,
        cfg->use_student_t_state, cfg->nu_state, N
    );
    
    /* 2. Compute bandwidth */
    kernel_compute_bandwidth<<<1, 256, 0, cs>>>(state->d_h, state->d_bandwidth, N);
    
    /* Copy bandwidth to host for kernel launch */
    float h_bandwidth;
    CUDA_CHECK(cudaMemcpyAsync(&h_bandwidth, state->d_bandwidth, sizeof(float), 
                               cudaMemcpyDeviceToHost, cs));
    CUDA_CHECK(cudaStreamSynchronize(cs));
    
    /* 3. Stein iterations (fixed steps, no KSD early stopping) */
    float step_size = 0.1f;
    float rmsprop_rho = 0.9f;
    float rmsprop_eps = 1e-6f;
    
    for (int ai = 0; ai < cfg->n_anneal_steps; ai++) {
        /* Annealing schedule */
        float beta = (float)(ai + 1) / (float)cfg->n_anneal_steps;
        int n_steps = cfg->n_stein_steps / cfg->n_anneal_steps;
        if (ai == cfg->n_anneal_steps - 1) {
            n_steps = cfg->n_stein_steps - n_steps * (cfg->n_anneal_steps - 1);
        }
        
        for (int s = 0; s < n_steps; s++) {
            /* Gradient computation */
            kernel_replay_gradient<<<grid, block, 0, cs>>>(
                state->d_h, state->d_h_prev, state->d_grad_log_p,
                y_t, params->rho, params->sigma_z, params->mu,
                cfg->nu_obs, state->student_t_const,
                cfg->use_student_t_state, cfg->nu_state, N
            );
            
            /* Stein transport */
            kernel_replay_stein_transport<<<grid, block, 0, cs>>>(
                state->d_h, state->d_grad_log_p, state->d_grad_v,
                h_bandwidth, step_size * sqrtf(beta), rmsprop_rho, rmsprop_eps,
                cfg->h_min, cfg->h_max, N
            );
        }
    }
    
    /* 4. Compute log-likelihood increment */
    kernel_replay_log_lik<<<1, 256, 0, cs>>>(
        state->d_h, state->d_log_lik_inc, y_t,
        cfg->nu_obs, state->student_t_const, N
    );
    
    float h_log_lik_inc;
    CUDA_CHECK(cudaMemcpyAsync(&h_log_lik_inc, state->d_log_lik_inc, sizeof(float),
                               cudaMemcpyDeviceToHost, cs));
    
    /* 5. Update h_prev for next step */
    CUDA_CHECK(cudaMemcpyAsync(state->d_h_prev, state->d_h, N * sizeof(float),
                               cudaMemcpyDeviceToDevice, cs));
    
    CUDA_CHECK(cudaStreamSynchronize(cs));
    
    return h_log_lik_inc;
}

float svpf_replay_likelihood(
    SVPFReplayState* state,
    const float* d_y,
    int T,
    const SVPFReplayParams* params,
    const SVPFNoiseBuffer* noise_buf
) {
    /* Reset state with first noise slice (for initialization) */
    svpf_replay_reset(state, params->rho, params->mu, params->sigma_z, 
                      noise_buf->d_noise);  /* t=0 noise for init */
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    
    /* Accumulate log-likelihood */
    float total_log_lik = 0.0f;
    float y_prev = 0.0f;
    
    /* Copy observations to host for stepping */
    float* h_y = (float*)malloc(T * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, T * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int t = 0; t < T; t++) {
        float y_t = h_y[t];
        
        /* Get noise for this timestep (offset by 1 because t=0 used for init) */
        const float* d_noise_t = noise_buf->d_noise + (t + 1) * state->N;
        
        float log_lik_inc = svpf_replay_step(state, y_t, y_prev, params, d_noise_t);
        total_log_lik += log_lik_inc;
        
        y_prev = y_t;
    }
    
    free(h_y);
    return total_log_lik;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG PRIOR IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

float cpmmh_log_prior(const SVPFReplayParams* theta, const CPMMHPrior* prior) {
    float rho = theta->rho;
    float mu = theta->mu;
    float sigma_z = theta->sigma_z;
    
    /* Check hard bounds */
    if (rho < prior->rho_min || rho > prior->rho_max) return -INFINITY;
    if (mu < prior->mu_min || mu > prior->mu_max) return -INFINITY;
    if (sigma_z < prior->sigma_z_min || sigma_z > prior->sigma_z_max) return -INFINITY;
    
    float log_p = 0.0f;
    
    /* ρ prior: Beta on [rho_min, rho_max] */
    float rho_normalized = (rho - prior->rho_min) / (prior->rho_max - prior->rho_min);
    rho_normalized = fmaxf(fminf(rho_normalized, 1.0f - 1e-6f), 1e-6f);
    log_p += (prior->rho_alpha - 1.0f) * logf(rho_normalized)
           + (prior->rho_beta - 1.0f) * logf(1.0f - rho_normalized);
    /* Note: normalization constant cancels in MH ratio */
    
    /* μ prior: Normal */
    float d_mu = (mu - prior->mu_mean) / prior->mu_std;
    log_p += -0.5f * d_mu * d_mu;
    
    /* σ_z prior: HalfNormal */
    log_p += -0.5f * (sigma_z * sigma_z) / (prior->sigma_z_scale * prior->sigma_z_scale);
    
    return log_p;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ADAPTIVE PROPOSAL IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void cpmmh_proposal_init(CPMMHProposal* prop) {
    /* Initialize with reasonable defaults for crypto SV */
    prop->mean[0] = 0.94f;   /* rho */
    prop->mean[1] = -3.0f;   /* mu */
    prop->mean[2] = 0.15f;   /* sigma_z */
    
    /* Initial covariance (diagonal) */
    memset(prop->cov, 0, 9 * sizeof(float));
    prop->cov[0] = 0.001f;   /* var(rho) */
    prop->cov[4] = 0.25f;    /* var(mu) */
    prop->cov[8] = 0.01f;    /* var(sigma_z) */
    
    /* Cholesky of scaled covariance */
    memset(prop->chol, 0, 9 * sizeof(float));
    prop->chol[0] = sqrtf(prop->cov[0]);
    prop->chol[4] = sqrtf(prop->cov[4]);
    prop->chol[8] = sqrtf(prop->cov[8]);
    
    /* Fixed fallback proposal */
    prop->fixed_std[0] = 0.01f;   /* rho */
    prop->fixed_std[1] = 0.3f;    /* mu */
    prop->fixed_std[2] = 0.05f;   /* sigma_z */
    
    prop->n_samples = 0;
    prop->adapt_start = 100;
    prop->adapt_interval = 50;
    prop->scale_factor = 2.38f * 2.38f / 3.0f;  /* 2.38²/d for d=3 */
    prop->mixture_prob = 0.95f;
    prop->cov_regularization = 1e-6f;
}

void cpmmh_proposal_update(CPMMHProposal* prop, const float* theta) {
    prop->n_samples++;
    int n = prop->n_samples;
    
    /* Welford's online algorithm for mean and covariance */
    float delta[3];
    for (int i = 0; i < 3; i++) {
        delta[i] = theta[i] - prop->mean[i];
        prop->mean[i] += delta[i] / (float)n;
    }
    
    /* Update covariance (only after adapt_start) */
    if (n >= prop->adapt_start) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                float delta2 = theta[j] - prop->mean[j];
                float cov_update = (delta[i] * delta2 - prop->cov[i * 3 + j]) / (float)n;
                prop->cov[i * 3 + j] += cov_update;
                prop->cov[j * 3 + i] = prop->cov[i * 3 + j];  /* Symmetric */
            }
        }
        
        /* Recompute Cholesky every adapt_interval samples */
        if (n % prop->adapt_interval == 0) {
            /* Copy and scale covariance */
            float scaled_cov[9];
            for (int i = 0; i < 9; i++) {
                scaled_cov[i] = prop->scale_factor * prop->cov[i];
            }
            
            /* Add regularization */
            scaled_cov[0] += prop->cov_regularization;
            scaled_cov[4] += prop->cov_regularization;
            scaled_cov[8] += prop->cov_regularization;
            
            /* Cholesky decomposition (3x3) */
            memset(prop->chol, 0, 9 * sizeof(float));
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < j; k++) {
                        sum += prop->chol[i * 3 + k] * prop->chol[j * 3 + k];
                    }
                    
                    if (i == j) {
                        float val = scaled_cov[i * 3 + i] - sum;
                        prop->chol[i * 3 + i] = (val > 0.0f) ? sqrtf(val) : 1e-4f;
                    } else {
                        float diag = prop->chol[j * 3 + j];
                        prop->chol[i * 3 + j] = (diag > 1e-10f) 
                            ? (scaled_cov[i * 3 + j] - sum) / diag : 0.0f;
                    }
                }
            }
        }
    }
}

void cpmmh_proposal_sample(
    const CPMMHProposal* prop,
    const float* theta_current,
    float* theta_out,
    uint64_t* rng_state
) {
    float z[3];
    z[0] = host_normal(rng_state);
    z[1] = host_normal(rng_state);
    z[2] = host_normal(rng_state);
    
    /* Mixture: adaptive or fixed */
    if (host_uniform(rng_state) < prop->mixture_prob && prop->n_samples >= prop->adapt_start) {
        /* Adaptive: θ* = θ + L·z where L is Cholesky factor */
        theta_out[0] = theta_current[0] + prop->chol[0] * z[0];
        theta_out[1] = theta_current[1] + prop->chol[3] * z[0] + prop->chol[4] * z[1];
        theta_out[2] = theta_current[2] + prop->chol[6] * z[0] + prop->chol[7] * z[1] + prop->chol[8] * z[2];
    } else {
        /* Fixed independent proposal */
        theta_out[0] = theta_current[0] + prop->fixed_std[0] * z[0];
        theta_out[1] = theta_current[1] + prop->fixed_std[1] * z[1];
        theta_out[2] = theta_current[2] + prop->fixed_std[2] * z[2];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CPMMH STATE MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════════*/

CPMMHState* cpmmh_create(int N_particles, int T_max, int history_cap) {
    CPMMHState* state = (CPMMHState*)calloc(1, sizeof(CPMMHState));
    if (!state) return NULL;
    
    state->N_particles = N_particles;
    state->cpmmh_rho = 0.99f;
    
    /* Noise buffers (need T+1 for init + T steps) */
    state->noise[0] = svpf_noise_buffer_create(T_max + 1, N_particles);
    state->noise[1] = svpf_noise_buffer_create(T_max + 1, N_particles);
    state->active_buf = 0;
    
    /* Replay state */
    state->replay_config = svpf_replay_config_default();
    state->replay = svpf_replay_create(N_particles, &state->replay_config);
    
    /* Observations */
    CUDA_CHECK(cudaMalloc(&state->d_y, T_max * sizeof(float)));
    state->T = 0;
    
    /* Prior and proposal */
    state->prior = cpmmh_prior_crypto_default();
    cpmmh_proposal_init(&state->proposal);
    
    /* History */
    state->history_capacity = history_cap;
    state->history_len = 0;
    state->theta_history = (float*)malloc(history_cap * 3 * sizeof(float));
    state->log_post_history = (float*)malloc(history_cap * sizeof(float));
    
    /* RNG */
    state->host_rng_state = 0x853C49E6748FEA9BULL;
    
    CUDA_CHECK(cudaStreamCreate(&state->stream));
    
    return state;
}

void cpmmh_destroy(CPMMHState* state) {
    if (!state) return;
    
    svpf_noise_buffer_destroy(state->noise[0]);
    svpf_noise_buffer_destroy(state->noise[1]);
    svpf_replay_destroy(state->replay);
    
    cudaFree(state->d_y);
    free(state->theta_history);
    free(state->log_post_history);
    cudaStreamDestroy(state->stream);
    
    free(state);
}

void cpmmh_set_observations(CPMMHState* state, const float* h_y, int T) {
    CUDA_CHECK(cudaMemcpy(state->d_y, h_y, T * sizeof(float), cudaMemcpyHostToDevice));
    state->T = T;
}

void cpmmh_initialize(CPMMHState* state, const float* theta_init, uint64_t seed) {
    state->host_rng_state = 0x853C49E6748FEA9BULL ^ seed;
    
    /* Initialize theta (from prior or given) */
    if (theta_init) {
        memcpy(state->theta, theta_init, 3 * sizeof(float));
    } else {
        /* Sample from prior */
        float u_rho = host_uniform(&state->host_rng_state);
        float rho_norm = powf(u_rho, 1.0f / state->prior.rho_alpha) * 
                        powf(1.0f - u_rho, 1.0f / state->prior.rho_beta);
        /* Simplified: just use uniform on bounds for init */
        state->theta[0] = state->prior.rho_min + 
                         (state->prior.rho_max - state->prior.rho_min) * host_uniform(&state->host_rng_state);
        state->theta[1] = state->prior.mu_mean + state->prior.mu_std * host_normal(&state->host_rng_state);
        state->theta[2] = fabsf(state->prior.sigma_z_scale * host_normal(&state->host_rng_state));
        
        /* Clamp to bounds */
        state->theta[0] = fmaxf(fminf(state->theta[0], state->prior.rho_max), state->prior.rho_min);
        state->theta[1] = fmaxf(fminf(state->theta[1], state->prior.mu_max), state->prior.mu_min);
        state->theta[2] = fmaxf(fminf(state->theta[2], state->prior.sigma_z_max), state->prior.sigma_z_min);
    }
    
    /* Fill initial noise buffer */
    svpf_noise_buffer_fill_gaussian(state->noise[state->active_buf], 
                                    state->T + 1, seed, state->stream);
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    
    /* Compute initial likelihood */
    SVPFReplayParams params = {state->theta[0], state->theta[1], state->theta[2]};
    state->log_likelihood = svpf_replay_likelihood(
        state->replay, state->d_y, state->T, &params, state->noise[state->active_buf]
    );
    state->log_prior = cpmmh_log_prior(&params, &state->prior);
    state->log_posterior = state->log_likelihood + state->log_prior;
    
    /* Reset counters */
    state->n_accepts = 0;
    state->n_total = 0;
    state->n_prior_rejects = 0;
    state->history_len = 0;
    
    /* Reset proposal */
    cpmmh_proposal_init(&state->proposal);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CPMMH STEP - THE CORE FUNCTION
 *═══════════════════════════════════════════════════════════════════════════════*/

int cpmmh_step(CPMMHState* state) {
    /* 1. Propose θ* */
    float theta_prop[3];
    cpmmh_proposal_sample(&state->proposal, state->theta, theta_prop, &state->host_rng_state);
    
    /* 2. Check prior */
    SVPFReplayParams params_prop = {theta_prop[0], theta_prop[1], theta_prop[2]};
    float log_prior_prop = cpmmh_log_prior(&params_prop, &state->prior);
    
    if (!isfinite(log_prior_prop)) {
        state->n_total++;
        state->n_prior_rejects++;
        return 0;  /* Reject */
    }
    
    /* 3. Correlate noise */
    int other_buf = 1 - state->active_buf;
    unsigned long long noise_seed = xorshift64star(&state->host_rng_state);
    
    svpf_noise_buffer_correlate(
        state->noise[state->active_buf],
        state->noise[other_buf],
        state->cpmmh_rho,
        state->T + 1,
        noise_seed,
        state->stream
    );
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    
    /* 4. Compute log-likelihood via SVPF replay */
    float log_lik_prop = svpf_replay_likelihood(
        state->replay, state->d_y, state->T, &params_prop, state->noise[other_buf]
    );
    
    /* 5. MH accept/reject */
    float log_post_prop = log_lik_prop + log_prior_prop;
    float log_alpha = log_post_prop - state->log_posterior;
    
    float u = host_uniform(&state->host_rng_state);
    int accept = (logf(u) < log_alpha) ? 1 : 0;
    
    if (accept) {
        memcpy(state->theta, theta_prop, 3 * sizeof(float));
        state->log_likelihood = log_lik_prop;
        state->log_prior = log_prior_prop;
        state->log_posterior = log_post_prop;
        state->active_buf = other_buf;
        state->n_accepts++;
        
        /* Update adaptive proposal */
        cpmmh_proposal_update(&state->proposal, state->theta);
    }
    
    state->n_total++;
    return accept;
}

void cpmmh_run(CPMMHState* state, int n_iterations, int n_burnin, int thin) {
    for (int i = 0; i < n_iterations; i++) {
        cpmmh_step(state);
        
        /* Store in history (after burnin, with thinning) */
        if (i >= n_burnin && ((i - n_burnin) % thin == 0)) {
            if (state->history_len < state->history_capacity) {
                int idx = state->history_len;
                state->theta_history[idx * 3 + 0] = state->theta[0];
                state->theta_history[idx * 3 + 1] = state->theta[1];
                state->theta_history[idx * 3 + 2] = state->theta[2];
                state->log_post_history[idx] = state->log_posterior;
                state->history_len++;
            }
        }
        
        /* Progress */
        if ((i + 1) % 100 == 0) {
            printf("Iteration %d/%d: accept_rate=%.3f, theta=(%.4f, %.3f, %.4f), log_post=%.2f\n",
                   i + 1, n_iterations, cpmmh_acceptance_rate(state),
                   state->theta[0], state->theta[1], state->theta[2],
                   state->log_posterior);
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void cpmmh_get_posterior_mean(const CPMMHState* state, float* theta_mean) {
    theta_mean[0] = 0.0f;
    theta_mean[1] = 0.0f;
    theta_mean[2] = 0.0f;
    
    for (int i = 0; i < state->history_len; i++) {
        theta_mean[0] += state->theta_history[i * 3 + 0];
        theta_mean[1] += state->theta_history[i * 3 + 1];
        theta_mean[2] += state->theta_history[i * 3 + 2];
    }
    
    if (state->history_len > 0) {
        theta_mean[0] /= (float)state->history_len;
        theta_mean[1] /= (float)state->history_len;
        theta_mean[2] /= (float)state->history_len;
    }
}

void cpmmh_get_posterior_std(const CPMMHState* state, float* theta_std) {
    float mean[3];
    cpmmh_get_posterior_mean(state, mean);
    
    theta_std[0] = 0.0f;
    theta_std[1] = 0.0f;
    theta_std[2] = 0.0f;
    
    for (int i = 0; i < state->history_len; i++) {
        for (int j = 0; j < 3; j++) {
            float d = state->theta_history[i * 3 + j] - mean[j];
            theta_std[j] += d * d;
        }
    }
    
    if (state->history_len > 1) {
        for (int j = 0; j < 3; j++) {
            theta_std[j] = sqrtf(theta_std[j] / (float)(state->history_len - 1));
        }
    }
}

void cpmmh_print_summary(const CPMMHState* state) {
    float mean[3], std[3];
    cpmmh_get_posterior_mean(state, mean);
    cpmmh_get_posterior_std(state, std);
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("CPMMH Summary\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Iterations: %d, Accepts: %d, Accept Rate: %.3f\n",
           state->n_total, state->n_accepts, cpmmh_acceptance_rate(state));
    printf("Prior Rejects: %d\n", state->n_prior_rejects);
    printf("History Length: %d\n", state->history_len);
    printf("\nPosterior Estimates:\n");
    printf("  ρ     = %.5f ± %.5f\n", mean[0], std[0]);
    printf("  μ     = %.4f ± %.4f\n", mean[1], std[1]);
    printf("  σ_z   = %.5f ± %.5f\n", mean[2], std[2]);
    printf("═══════════════════════════════════════════════════════════\n\n");
}

int cpmmh_export_history(const CPMMHState* state, float* theta_out, float* logpost_out) {
    if (theta_out) {
        memcpy(theta_out, state->theta_history, state->history_len * 3 * sizeof(float));
    }
    if (logpost_out) {
        memcpy(logpost_out, state->log_post_history, state->history_len * sizeof(float));
    }
    return state->history_len;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION SETTERS
 *═══════════════════════════════════════════════════════════════════════════════*/

void cpmmh_set_correlation(CPMMHState* state, float rho) {
    state->cpmmh_rho = rho;
}

void cpmmh_set_prior(CPMMHState* state, const CPMMHPrior* prior) {
    state->prior = *prior;
}

void cpmmh_set_replay_config(CPMMHState* state, const SVPFReplayConfig* config) {
    state->replay_config = *config;
    /* Note: Need to recreate replay state if config changes significantly */
}
