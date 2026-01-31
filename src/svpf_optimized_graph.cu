/**
 * @file svpf_optimized_graph.cu
 * @brief Consolidated SVPF Implementation with KSD-based Adaptive Stein Steps
 * 
 * Single implementation file containing:
 * - State management (svpf_create/destroy/initialize)
 * - Basic utility kernels
 * - Stein loop with KSD-based early stopping
 * - Public API (svpf_step_graph, svpf_step_adaptive, svpf_run_sequence)
 * - Diagnostics (svpf_get_particles, svpf_get_stats, svpf_get_ess)
 * 
 * KSD (Kernel Stein Discrepancy) is computed in the same O(N²) pass as Stein
 * transport at zero extra cost. Early stopping occurs when relative KSD
 * improvement drops below threshold.
 */

#include "svpf_kernels.cuh"
#include "svpf_two_factor.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Maximum smoothing window size
#ifndef SVPF_SMOOTH_MAX_LAG
#define SVPF_SMOOTH_MAX_LAG 8
#endif

// Forward declarations
static void svpf_optimized_init(SVPFOptimizedState* opt, int n);

// =============================================================================
// HEUN'S METHOD KERNEL DECLARATIONS
// =============================================================================
// These kernels are defined in svpf_opt_kernels.cu

__global__ void svpf_stein_operator_kernel(
    const float* h, const float* grad, float* phi_out,
    const float* d_bandwidth, int stein_sign_mode, int n);

__global__ void svpf_stein_operator_full_newton_kernel(
    const float* h, const float* grad, const float* local_hessian,
    float* phi_out, const float* d_bandwidth, int stein_sign_mode, int n);

__global__ void svpf_heun_predictor_kernel(
    float* h, const float* h_orig, const float* phi, const float* v_rmsprop,
    float step_size, float beta_factor, float epsilon, int n);

__global__ void svpf_heun_corrector_kernel(
    float* h, const float* h_orig, const float* phi_orig, const float* phi_pred,
    float* v_rmsprop, curandStatePhilox4_32_10_t* rng,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int n);

__global__ void svpf_heun_corrector_ksd_kernel(
    float* h, const float* h_orig, const float* phi_orig, const float* phi_pred,
    const float* grad, float* v_rmsprop, curandStatePhilox4_32_10_t* rng,
    const float* d_bandwidth, float* d_ksd_partial,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int n);

// =============================================================================
// BASIC UTILITY KERNELS
// =============================================================================

__global__ void svpf_init_rng_kernel(
    curandStatePhilox4_32_10_t* states,
    int n,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void svpf_init_particles_kernel(
    float* h,
    curandStatePhilox4_32_10_t* rng_states,
    float mu,
    float stationary_std,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float z = curand_normal(&rng_states[idx]);
        float h_new = mu + stationary_std * z;
        h[idx] = fminf(fmaxf(h_new, SVPF_H_MIN), SVPF_H_MAX);
    }
}

__global__ void svpf_copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// STATE MANAGEMENT: Create
// =============================================================================

SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream) {
    SVPFState* state = (SVPFState*)malloc(sizeof(SVPFState));
    if (!state) return NULL;
    
    memset(&state->opt_backend, 0, sizeof(SVPFOptimizedState));
    
    state->n_particles = n_particles;
    state->n_stein_steps = n_stein_steps;
    state->nu = nu;
    state->stream = stream ? stream : 0;
    state->timestep = 0;
    state->y_prev = 0.0f;
    
    state->student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                           - lgammaf(nu / 2.0f) 
                           - 0.5f * logf((float)M_PI * nu);
    
    {
        const float psi_half = -1.9635100260214235f;
        float nu_half = nu / 2.0f;
        float psi_nu_half;
        if (nu_half >= 1.0f) {
            psi_nu_half = logf(nu_half) - 1.0f/(2.0f*nu_half) - 1.0f/(12.0f*nu_half*nu_half);
        } else {
            psi_nu_half = -0.5772156649f - 1.0f/nu_half;
        }
        float expected_log_t_sq = logf(nu) + psi_half - psi_nu_half;
        state->student_t_implied_offset = -expected_log_t_sq;
    }
    
    int n = n_particles;
    
    // Particle arrays
    cudaMalloc(&state->h, n * sizeof(float));
    cudaMalloc(&state->h_prev, n * sizeof(float));
    cudaMalloc(&state->h_pred, n * sizeof(float));
    cudaMalloc(&state->grad_log_p, n * sizeof(float));
    cudaMalloc(&state->kernel_sum, n * sizeof(float));
    cudaMalloc(&state->grad_kernel_sum, n * sizeof(float));
    cudaMalloc(&state->log_weights, n * sizeof(float));
    cudaMalloc(&state->d_h_centered, n * sizeof(float));
    cudaMalloc(&state->rng_states, n * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&state->d_reduce_buf, n * sizeof(float));
    cudaMalloc(&state->d_temp, n * sizeof(float));
    
    cudaMalloc(&state->d_grad_v, n * sizeof(float));
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));
    
    cudaMalloc(&state->d_return_ema, sizeof(float));
    cudaMalloc(&state->d_return_var, sizeof(float));
    cudaMalloc(&state->d_bw_alpha, sizeof(float));
    float init_ema = 0.0f;
    float init_alpha = 0.3f;
    cudaMemcpy(state->d_return_ema, &init_ema, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_return_var, &init_ema, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_bw_alpha, &init_alpha, sizeof(float), cudaMemcpyHostToDevice);
    
    state->lik_offset = 0.45f;
    state->use_exact_gradient = 0;
    
    state->use_svld = 1;
    state->use_annealing = 1;
    state->use_adaptive_beta = 1;  // KSD-adaptive beta (Maken 2022)
    state->n_anneal_steps = 3;
    state->temperature = 1.0f;
    state->rmsprop_rho = 0.9f;
    state->rmsprop_eps = 1e-6f;
    
    state->use_mim = 1;
    state->mim_jump_prob = 0.05f;
    state->mim_jump_scale = 5.0f;
    
    state->use_asymmetric_rho = 1;
    state->rho_up = 0.98f;
    state->rho_down = 0.93f;
    
    state->use_guide = 1;
    state->use_guide_preserving = 1;
    state->guide_strength = 0.05f;
    state->guide_mean = 0.0f;
    state->guide_var = 0.0f;
    state->guide_K = 0.0f;
    state->guide_initialized = 0;
    
    state->use_adaptive_guide = 0;
    state->guide_strength_base = 0.05f;
    state->guide_strength_max = 0.30f;
    state->guide_innovation_threshold = 1.0f;
    state->vol_prev = 0.05f;
    
    // Partial rejuvenation (Maken 2022)
    state->use_rejuvenation = 1;        // ON by default
    state->rejuv_ksd_threshold = 0.30f; // Trigger when KSD > 0.3
    state->rejuv_prob = 0.30f;          // Nudge 30% of particles
    state->rejuv_blend = 0.30f;         // 30% toward guide, 70% stay
    
    state->use_newton = 0;
    state->use_full_newton = 0;
    
    state->use_guided = 0;
    state->guided_alpha_base = 0.0f;
    state->guided_alpha_shock = 0.5f;
    state->guided_innovation_threshold = 1.5f;
    
    state->use_local_params = 0;
    state->delta_rho = 0.02f;
    state->delta_sigma = 0.1f;
    
    state->use_adaptive_mu = 0;
    state->mu_state = -3.5f;
    state->mu_var = 1.0f;
    state->mu_process_var = 0.001f;
    state->mu_obs_var_scale = 10.0f;
    state->mu_min = -6.0f;
    state->mu_max = -1.0f;
    
    state->use_adaptive_sigma = 0;
    state->sigma_boost_threshold = 1.0f;
    state->sigma_boost_max = 3.0f;
    state->sigma_z_effective = 0.10f;
    
    // === Stein operator sign mode ===
    // 0 = legacy (subtract, attraction) - production-tested with MIM/SVLD/guide
    // 1 = paper (add, repulsion) - mathematically correct per Fan et al. 2021
    state->stein_repulsive_sign = SVPF_STEIN_SIGN_DEFAULT;
    
    // === Fan mode (weightless SVGD) ===
    // 0 = hybrid (default), 1 = pure Stein without importance weights
    state->use_fan_mode = 0;
    
    // === Student-t state dynamics ===
    // 0 = Gaussian AR(1) (default), 1 = Student-t AR(1) with bounded gradients
    state->use_student_t_state = 0;
    state->nu_state = 5.0f;  // Degrees of freedom for state dynamics (recommended: 5-7, min: 3)
    // Note: nu_state is clamped to >= 2.5 in svpf_initialize to ensure finite variance
    
    // === KSD-based Adaptive Stein Steps ===
    state->stein_min_steps = 4;              // Always run at least this many
    state->stein_max_steps = 12;             // Never exceed this
    state->ksd_improvement_threshold = 0.05f; // Stop if relative improvement < 5%
    state->ksd_prev = 1e10f;                 // Initialize high
    state->stein_steps_used = n_stein_steps; // Diagnostic
    
    // === Heun's Method (Improved Euler) ===
    // 0 = Euler (default), 1 = Heun's method (2nd order, 2× gradient evals)
    state->use_heun = 0;
    
    // === Two-Factor Volatility ===
    // h_t = μ + h_fast,t + h_slow,t
    // Fast captures spikes (ρ≈0.90), slow captures regimes (ρ≈0.99)
    state->use_two_factor = 0;  // Off by default
    state->rho_fast = 0.90f;
    state->sigma_fast = 0.15f;
    state->rho_slow = 0.99f;
    state->sigma_slow = 0.05f;
    state->bw_floor_fast = 0.01f;
    state->bw_floor_slow = 0.05f;  // Lowered from 0.10 - allows h_slow transport
    state->guide_mean_fast = 0.0f;
    state->guide_mean_slow = 0.0f;
    state->guide_var_fast = 0.0f;
    state->guide_var_slow = 0.0f;
    state->guide_2f_initialized = 0;
    
    // === Backward Smoothing (Fan et al. 2021 sliding window, lightweight) ===
    // Applies RTS-style correction to past estimates using recent observations
    state->use_smoothing = 0;                // OFF by default
    state->smooth_lag = 3;                   // Window size (1-5 recommended)
    state->smooth_output_lag = 1;            // Output h[t-1] instead of h[t]
    for (int i = 0; i < SVPF_SMOOTH_MAX_LAG; i++) {
        state->smooth_h_mean[i] = 0.0f;
        state->smooth_h_var[i] = 1.0f;
        state->smooth_y[i] = 0.0f;
    }
    state->smooth_head = 0;
    
    // Device scalars
    cudaMalloc(&state->d_scalar_max, sizeof(float));
    cudaMalloc(&state->d_scalar_sum, sizeof(float));
    cudaMalloc(&state->d_scalar_mean, sizeof(float));
    cudaMalloc(&state->d_scalar_bandwidth, sizeof(float));
    cudaMalloc(&state->d_y_prev, sizeof(float));
    cudaMalloc(&state->d_result_loglik, sizeof(float));
    cudaMalloc(&state->d_result_vol_mean, sizeof(float));
    cudaMalloc(&state->d_result_h_mean, sizeof(float));
    
    // CUB temp storage
    state->cub_temp_bytes = 0;
    cub::DeviceReduce::Sum(NULL, state->cub_temp_bytes, state->h, state->d_scalar_sum, n);
    state->cub_temp_bytes += 1024;
    cudaMalloc(&state->d_cub_temp, state->cub_temp_bytes);
    
    return state;
}

// =============================================================================
// STATE MANAGEMENT: Destroy
// =============================================================================

static void svpf_optimized_cleanup(SVPFOptimizedState* opt);

void svpf_destroy(SVPFState* state) {
    if (!state) return;
    
    svpf_optimized_cleanup(&state->opt_backend);
    
    cudaFree(state->h);
    cudaFree(state->h_prev);
    cudaFree(state->h_pred);
    cudaFree(state->grad_log_p);
    cudaFree(state->kernel_sum);
    cudaFree(state->grad_kernel_sum);
    cudaFree(state->log_weights);
    cudaFree(state->d_h_centered);
    cudaFree(state->rng_states);
    cudaFree(state->d_reduce_buf);
    cudaFree(state->d_temp);
    cudaFree(state->d_cub_temp);
    
    cudaFree(state->d_grad_v);
    cudaFree(state->d_return_ema);
    cudaFree(state->d_return_var);
    cudaFree(state->d_bw_alpha);
    
    cudaFree(state->d_scalar_max);
    cudaFree(state->d_scalar_sum);
    cudaFree(state->d_scalar_mean);
    cudaFree(state->d_scalar_bandwidth);
    cudaFree(state->d_y_prev);
    cudaFree(state->d_result_loglik);
    cudaFree(state->d_result_vol_mean);
    cudaFree(state->d_result_h_mean);
    
    free(state);
}

// =============================================================================
// STATE MANAGEMENT: Initialize
// =============================================================================

void svpf_initialize(SVPFState* state, const SVPFParams* params, unsigned long long seed) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    state->timestep = 0;
    state->y_prev = 0.0f;
    
    float zero = 0.0f;
    cudaMemcpyAsync(state->d_y_prev, &zero, sizeof(float), cudaMemcpyHostToDevice, state->stream);
    
    svpf_init_rng_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->rng_states, n, seed
    );
    
    float rho = params->rho;
    float sigma_z = params->sigma_z;
    
    // Clamp nu_state to ensure finite variance (requires nu > 2)
    float nu_state_clamped = fmaxf(state->nu_state, 2.5f);
    state->nu_state = nu_state_clamped;
    
    // Compute stationary variance
    // Gaussian: sigma²/(1-rho²)
    // Student-t: (nu/(nu-2)) * sigma²/(1-rho²)  for nu > 2
    float base_var = (sigma_z * sigma_z) / (1.0f - rho * rho + 1e-6f);
    float stationary_var;
    
    if (state->use_student_t_state) {
        // Student-t has larger variance due to heavier tails
        float var_scale = nu_state_clamped / (nu_state_clamped - 2.0f);
        stationary_var = var_scale * base_var;
    } else {
        stationary_var = base_var;
    }
    
    float stationary_std = sqrtf(stationary_var);
    
    svpf_init_particles_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->rng_states, params->mu, stationary_std, n
    );
    
    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );
    
    state->guide_initialized = 0;
    state->guide_mean = params->mu;
    state->guide_var = stationary_var;
    
    if (state->use_adaptive_mu) {
        state->mu_state = params->mu;
        state->mu_var = 1.0f;
    }
    
    if (state->use_adaptive_sigma) {
        state->sigma_z_effective = params->sigma_z;
    }
    
    // Reset KSD tracking
    state->ksd_prev = 1e10f;
    state->stein_steps_used = state->n_stein_steps;
    
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));
    
    // === Two-factor initialization ===
    if (state->use_two_factor) {
        SVPFOptimizedState* opt = &state->opt_backend;
        
        // Ensure buffers are allocated
        svpf_optimized_init(opt, n);
        
        // Initialize particles from stationary distribution
        svpf_init_two_factor_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
            opt->d_h_fast, opt->d_h_slow, state->rng_states,
            state->rho_fast, state->sigma_fast,
            state->rho_slow, state->sigma_slow,
            n
        );
        
        // Copy to prev
        cudaMemcpyAsync(opt->d_h_fast_prev, opt->d_h_fast, n * sizeof(float),
                        cudaMemcpyDeviceToDevice, state->stream);
        cudaMemcpyAsync(opt->d_h_slow_prev, opt->d_h_slow, n * sizeof(float),
                        cudaMemcpyDeviceToDevice, state->stream);
        
        // Initialize RMSProp states
        float one = 1.0f;
        cudaMemsetAsync(opt->d_grad_v_fast, 0, n * sizeof(float), state->stream);
        cudaMemsetAsync(opt->d_grad_v_slow, 0, n * sizeof(float), state->stream);
        
        // Reset 2F EKF guide
        state->guide_2f_initialized = 0;
        state->guide_mean_fast = 0.0f;
        state->guide_mean_slow = 0.0f;
        state->guide_var_fast = (state->sigma_fast * state->sigma_fast) / 
                                (1.0f - state->rho_fast * state->rho_fast + 1e-6f);
        state->guide_var_slow = (state->sigma_slow * state->sigma_slow) / 
                                (1.0f - state->rho_slow * state->rho_slow + 1e-6f);
    }
    
    svpf_graph_invalidate(state);
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// OPTIMIZED BACKEND
// =============================================================================

static void svpf_optimized_init(SVPFOptimizedState* opt, int n) {
    if (opt->initialized && n > opt->allocated_n) {
        svpf_optimized_cleanup(opt);
    }
    if (opt->initialized) return;
    
    float* d_dummy_in;
    float* d_dummy_out;
    cudaMalloc(&d_dummy_in, n * sizeof(float));
    cudaMalloc(&d_dummy_out, sizeof(float));
    
    opt->temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, opt->temp_storage_bytes, d_dummy_in, d_dummy_out, n);
    size_t sum_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, sum_bytes, d_dummy_in, d_dummy_out, n);
    opt->temp_storage_bytes = max(opt->temp_storage_bytes, sum_bytes);
    
    cudaMalloc(&opt->d_temp_storage, opt->temp_storage_bytes);
    cudaFree(d_dummy_in);
    cudaFree(d_dummy_out);
    
    cudaMalloc(&opt->d_max_log_w, sizeof(float));
    cudaMalloc(&opt->d_sum_exp, sizeof(float));
    cudaMalloc(&opt->d_bandwidth, sizeof(float));
    cudaMalloc(&opt->d_bandwidth_sq, sizeof(float));
    
    float zero = 0.0f;
    cudaMemcpy(opt->d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&opt->d_exp_w, n * sizeof(float));
    cudaMalloc(&opt->d_phi, n * sizeof(float));
    cudaMalloc(&opt->d_grad_lik, n * sizeof(float));
    cudaMalloc(&opt->d_precond_grad, n * sizeof(float));
    cudaMalloc(&opt->d_inv_hessian, n * sizeof(float));
    
    cudaMalloc(&opt->d_h_mean_prev, sizeof(float));
    float init_h_mean = -3.5f;
    cudaMemcpy(opt->d_h_mean_prev, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&opt->d_guide_mean, sizeof(float));
    cudaMemcpy(opt->d_guide_mean, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&opt->d_guide_strength, sizeof(float));
    float init_guide_strength = 0.05f;
    cudaMemcpy(opt->d_guide_strength, &init_guide_strength, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&opt->d_y_single, 2 * sizeof(float));
    cudaMalloc(&opt->d_loglik_single, sizeof(float));
    cudaMalloc(&opt->d_vol_single, sizeof(float));
    
    cudaMalloc(&opt->d_params_staging, SVPF_GRAPH_PARAMS_SIZE * sizeof(float));
    
    // === KSD buffers ===
    cudaMalloc(&opt->d_ksd_partial, n * sizeof(float));
    cudaMalloc(&opt->d_ksd, sizeof(float));
    
    // === Heun's method buffers ===
    cudaMalloc(&opt->d_phi_orig, n * sizeof(float));
    cudaMalloc(&opt->d_phi_pred, n * sizeof(float));
    cudaMalloc(&opt->d_h_orig, n * sizeof(float));
    
    // === Two-factor buffers ===
    cudaMalloc(&opt->d_h_fast, n * sizeof(float));
    cudaMalloc(&opt->d_h_slow, n * sizeof(float));
    cudaMalloc(&opt->d_h_fast_prev, n * sizeof(float));
    cudaMalloc(&opt->d_h_slow_prev, n * sizeof(float));
    cudaMalloc(&opt->d_grad_fast, n * sizeof(float));
    cudaMalloc(&opt->d_grad_slow, n * sizeof(float));
    cudaMalloc(&opt->d_grad_v_fast, n * sizeof(float));
    cudaMalloc(&opt->d_grad_v_slow, n * sizeof(float));
    cudaMalloc(&opt->d_bandwidth_fast, sizeof(float));
    cudaMalloc(&opt->d_bandwidth_slow, sizeof(float));
    cudaMalloc(&opt->d_h_fast_mean, sizeof(float));
    cudaMalloc(&opt->d_h_slow_mean, sizeof(float));
    
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    cudaMallocHost(&opt->h_results_pinned, 4 * sizeof(float));
    
    opt->allocated_n = n;
    opt->initialized = true;
}

static void svpf_optimized_cleanup(SVPFOptimizedState* opt) {
    if (!opt->initialized) return;
    
    cudaFree(opt->d_temp_storage);
    cudaFree(opt->d_max_log_w);
    cudaFree(opt->d_sum_exp);
    cudaFree(opt->d_bandwidth);
    cudaFree(opt->d_bandwidth_sq);
    cudaFree(opt->d_exp_w);
    cudaFree(opt->d_phi);
    cudaFree(opt->d_grad_lik);
    cudaFree(opt->d_precond_grad);
    cudaFree(opt->d_inv_hessian);
    cudaFree(opt->d_h_mean_prev);
    cudaFree(opt->d_guide_mean);
    cudaFree(opt->d_guide_strength);
    cudaFree(opt->d_y_single);
    cudaFree(opt->d_loglik_single);
    cudaFree(opt->d_vol_single);
    cudaFree(opt->d_params_staging);
    
    // === KSD buffers ===
    cudaFree(opt->d_ksd_partial);
    cudaFree(opt->d_ksd);
    
    // === Heun's method buffers ===
    cudaFree(opt->d_phi_orig);
    cudaFree(opt->d_phi_pred);
    cudaFree(opt->d_h_orig);
    
    // === Two-factor buffers ===
    cudaFree(opt->d_h_fast);
    cudaFree(opt->d_h_slow);
    cudaFree(opt->d_h_fast_prev);
    cudaFree(opt->d_h_slow_prev);
    cudaFree(opt->d_grad_fast);
    cudaFree(opt->d_grad_slow);
    cudaFree(opt->d_grad_v_fast);
    cudaFree(opt->d_grad_v_slow);
    cudaFree(opt->d_bandwidth_fast);
    cudaFree(opt->d_bandwidth_slow);
    cudaFree(opt->d_h_fast_mean);
    cudaFree(opt->d_h_slow_mean);
    
    if (opt->h_results_pinned) {
        cudaFreeHost(opt->h_results_pinned);
        opt->h_results_pinned = nullptr;
    }
    
    if (opt->graph_captured) {
        cudaGraphExecDestroy(opt->graph_exec);
        cudaGraphDestroy(opt->graph);
        opt->graph_captured = false;
    }
    
    if (opt->graph_stream) {
        cudaStreamDestroy(opt->graph_stream);
        opt->graph_stream = nullptr;
    }
    
    opt->allocated_n = 0;
    opt->initialized = false;
}

void svpf_optimized_cleanup_state(SVPFState* state) {
    if (state) svpf_optimized_cleanup(&state->opt_backend);
}

static inline SVPFOptimizedState* get_opt(SVPFState* state) {
    return &state->opt_backend;
}

// =============================================================================
// ADAPTIVE MU: 1D Kalman Filter Update
// =============================================================================

static void svpf_adaptive_mu_update(
    SVPFState* state,
    float h_mean,
    float bandwidth
) {
    if (!state->use_adaptive_mu) return;
    
    float mu_pred = state->mu_state;
    float P_pred = state->mu_var + state->mu_process_var;
    
    float R = state->mu_obs_var_scale * bandwidth * bandwidth;
    
    float K = P_pred / (P_pred + R + 1e-8f);
    
    float innovation = h_mean - mu_pred;
    
    float mu_new = mu_pred + K * innovation;
    float P_new = (1.0f - K) * P_pred;
    
    mu_new = fminf(fmaxf(mu_new, state->mu_min), state->mu_max);
    
    state->mu_state = mu_new;
    state->mu_var = P_new;
}

// =============================================================================
// BACKWARD SMOOTHING: Lightweight RTS-style correction
// =============================================================================
// 
// Implements a simplified Rauch-Tung-Striebel smoother on stored summary stats.
// After each forward filter step, we:
// 1. Store h_mean, h_var, y in circular buffer
// 2. Run backward pass to refine past estimates using "future" observations
// 3. Output lagged smoothed estimate instead of raw filtered estimate
// 
// This is NOT true joint inference over trajectories (like full sliding window),
// but captures most of the benefit at minimal cost (O(k) scalar ops per step).
// =============================================================================

static void svpf_smooth_backward(
    SVPFState* state,
    float h_mean_new,
    float h_var_new,
    float y_t,
    const SVPFParams* params
) {
    if (!state->use_smoothing) return;
    
    int k = state->smooth_lag;
    if (k > SVPF_SMOOTH_MAX_LAG) k = SVPF_SMOOTH_MAX_LAG;
    if (k < 1) k = 1;
    
    // Store current estimate in buffer
    int head = state->smooth_head;
    state->smooth_h_mean[head] = h_mean_new;
    state->smooth_h_var[head] = h_var_new;
    state->smooth_y[head] = y_t;
    
    // Advance head (circular)
    state->smooth_head = (head + 1) % k;
    
    // Skip backward pass until buffer is full
    if (state->timestep < k) return;
    
    // Get AR(1) parameters
    float rho = params->rho;
    float mu = state->use_adaptive_mu ? state->mu_state : params->mu;
    float sigma_z_sq = params->sigma_z * params->sigma_z;
    
    // Backward pass: newest to oldest
    // At index idx_next is the "future" estimate, at idx_curr is current
    // We correct idx_curr using information from idx_next
    for (int lag = 1; lag < k; lag++) {
        int idx_curr = (state->smooth_head - lag - 1 + k) % k;
        int idx_next = (state->smooth_head - lag + k) % k;
        
        float h_curr = state->smooth_h_mean[idx_curr];
        float h_next = state->smooth_h_mean[idx_next];
        float var_curr = state->smooth_h_var[idx_curr];
        
        // What did h_curr predict for h_next?
        float h_pred = mu + rho * (h_curr - mu);
        
        // Prediction variance
        float pred_var = rho * rho * var_curr + sigma_z_sq;
        
        // Innovation: how far was the prediction off?
        float innovation = h_next - h_pred;
        
        // Backward (RTS) gain: how much should past estimate adjust?
        float J = rho * var_curr / (pred_var + 1e-8f);
        
        // Correct past estimate
        state->smooth_h_mean[idx_curr] = h_curr + J * innovation;
        
        // Reduce uncertainty (we've learned from future)
        state->smooth_h_var[idx_curr] = var_curr * (1.0f - J * rho);
    }
}

// Get smoothed output (with configured lag)
static float svpf_get_smoothed_output(SVPFState* state, float h_mean_raw) {
    if (!state->use_smoothing) return h_mean_raw;
    
    int k = state->smooth_lag;
    if (k > SVPF_SMOOTH_MAX_LAG) k = SVPF_SMOOTH_MAX_LAG;
    
    // If not enough history yet, return raw
    if (state->timestep < k) return h_mean_raw;
    
    int output_lag = state->smooth_output_lag;
    if (output_lag <= 0) return h_mean_raw;  // No lag = raw output
    if (output_lag >= k) output_lag = k - 1;  // Cap at buffer size
    
    // Get lagged smoothed estimate
    // output_lag=1 means output h[t-1] which has seen y[t]
    int idx = (state->smooth_head - output_lag - 1 + k) % k;
    return state->smooth_h_mean[idx];
}

// =============================================================================
// TWO-FACTOR STEP FUNCTION
// =============================================================================
// Coordinate-wise SVGD: reuses existing 1D Stein transport kernel twice

static void svpf_step_two_factor(
    SVPFState* state,
    SVPFOptimizedState* opt,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* loglik_out,
    float* vol_out,
    float* h_mean_out
) {
    int n = state->n_particles;
    int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t cs = state->stream;
    
    // Upload y
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, cs);
    
    // =========================================================================
    // PREDICT (Two-Factor)
    // =========================================================================
    // NOTE: Don't use guided predict for two-factor - it puts all surprise
    // into h_fast which is wrong. The EKF guide with innovation split handles
    // the guidance properly.
    svpf_predict_two_factor_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
        opt->d_h_fast, opt->d_h_slow,
        opt->d_h_fast_prev, opt->d_h_slow_prev,
        state->rng_states,
        state->rho_fast, state->sigma_fast,
        state->rho_slow, state->sigma_slow,
        state->use_mim, state->mim_jump_prob, state->mim_jump_scale,
        n
    );
    
    // =========================================================================
    // EKF GUIDE (Innovation Split)
    // =========================================================================
    if (state->use_guide) {
        svpf_ekf_two_factor_update(
            &state->guide_mean_fast, &state->guide_mean_slow,
            &state->guide_var_fast, &state->guide_var_slow,
            y_t, params->mu,
            state->rho_fast, state->sigma_fast,
            state->rho_slow, state->sigma_slow,
            4.93f + 2.0f,  // obs_var
            state->student_t_implied_offset,
            &state->guide_2f_initialized
        );
        
        svpf_apply_guide_two_factor_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            opt->d_h_fast, opt->d_h_slow,
            state->guide_mean_fast, state->guide_mean_slow,
            state->guide_strength,
            n
        );
    }
    
    // =========================================================================
    // BANDWIDTH (separate for each component)
    // =========================================================================
    svpf_bandwidth_two_factor_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        opt->d_h_fast, opt->d_h_slow,
        opt->d_bandwidth_fast, opt->d_bandwidth_slow,
        state->bw_floor_fast, state->bw_floor_slow,
        n
    );
    
    // =========================================================================
    // STEIN ITERATIONS (Coordinate-wise)
    // =========================================================================
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float base_step = 0.5f / (float)state->n_stein_steps;
    size_t stein_smem = 2 * n * sizeof(float);
    
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    for (int a = 0; a < n_anneal; a++) {
        float beta = (float)(a + 1) / (float)n_anneal;
        float temp = state->use_svld ? state->temperature : 0.0f;
        
        for (int s = 0; s < state->n_stein_steps; s++) {
            // -----------------------------------------------------------------
            // Compute gradients (shared likelihood, separate priors)
            // -----------------------------------------------------------------
            svpf_gradient_two_factor_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
                opt->d_h_fast, opt->d_h_slow,
                opt->d_h_fast_prev, opt->d_h_slow_prev,
                opt->d_grad_fast, opt->d_grad_slow,
                state->log_weights,
                y_t, params->mu,
                state->rho_fast, state->sigma_fast,
                state->rho_slow, state->sigma_slow,
                beta, state->nu, student_t_const,
                state->use_exact_gradient,
                n
            );
            
            // -----------------------------------------------------------------
            // Stein transport: h_fast (reuse existing 1D kernel)
            // -----------------------------------------------------------------
            svpf_fused_stein_transport_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                opt->d_h_fast, opt->d_grad_fast, opt->d_grad_v_fast,
                state->rng_states, opt->d_bandwidth_fast,
                base_step, beta, temp, state->rmsprop_rho, state->rmsprop_eps,
                state->stein_repulsive_sign, n
            );
            
            // -----------------------------------------------------------------
            // Stein transport: h_slow (same kernel, different buffers)
            // -----------------------------------------------------------------
            svpf_fused_stein_transport_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                opt->d_h_slow, opt->d_grad_slow, opt->d_grad_v_slow,
                state->rng_states, opt->d_bandwidth_slow,
                base_step, beta, temp, state->rmsprop_rho, state->rmsprop_eps,
                state->stein_repulsive_sign, n
            );
        }
    }
    
    // =========================================================================
    // OUTPUTS
    // =========================================================================
    svpf_outputs_two_factor_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        opt->d_h_fast, opt->d_h_slow, state->log_weights,
        params->mu,
        opt->d_loglik_single, opt->d_vol_single, opt->d_h_mean_prev,
        opt->d_h_fast_mean, opt->d_h_slow_mean,
        n
    );
    
    // Read results
    float results[3];
    cudaMemcpyAsync(&results[0], opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(&results[1], opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(&results[2], opt->d_h_mean_prev, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaStreamSynchronize(cs);
    
    if (loglik_out) *loglik_out = results[0];
    if (vol_out) *vol_out = results[1];
    if (h_mean_out) *h_mean_out = results[2];
    
    state->vol_prev = results[1];
    state->timestep++;
}

// =============================================================================
// PUBLIC API: svpf_step_graph
// =============================================================================
// 
// This is the main stepping function. It now uses:
// - Regular kernel launches (no CUDA graph for Stein loop)
// - KSD-based early stopping for Stein iterations
// 
// Structure:
// 1. Predict step
// 2. Guide step (optional)
// 3. Bandwidth computation
// 4. Stein loop with KSD early stopping
// 5. Output computation
// =============================================================================

void svpf_step_graph(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                     float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t cs = state->stream;
    
    svpf_optimized_init(opt, n);
    
    // === Two-factor branch ===
    if (state->use_two_factor) {
        svpf_step_two_factor(state, opt, y_t, y_prev, params,
                             h_loglik_out, h_vol_out, h_mean_out);
        return;
    }
    
    // === Single-factor code continues below ===
    
    // Determine effective parameters
    float effective_mu = state->use_adaptive_mu ? state->mu_state : params->mu;
    float effective_sigma_z = params->sigma_z;
    
    if (state->use_adaptive_sigma && state->timestep > 0) {
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;
        
        float sigma_boost = 1.0f;
        if (return_z > state->sigma_boost_threshold) {
            float severity = fminf((return_z - state->sigma_boost_threshold) / 3.0f, 1.0f);
            sigma_boost = 1.0f + (state->sigma_boost_max - 1.0f) * severity;
        }
        
        effective_sigma_z = params->sigma_z * sigma_boost;
        state->sigma_z_effective = effective_sigma_z;
    }
    
    // Precompute constants
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t grad_smem = 2 * n * sizeof(float);
    size_t stein_smem = state->use_newton ? 3 * n * sizeof(float) : 2 * n * sizeof(float);
    
    float rho_up = state->use_asymmetric_rho ? state->rho_up : params->rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : params->rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    // Upload y values
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, cs);
    
    // =========================================================================
    // PREDICT
    // =========================================================================
    if (state->use_guided) {
        svpf_predict_guided_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev, 1,
            rho_up, rho_down, effective_sigma_z, effective_mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            state->guided_alpha_base, state->guided_alpha_shock,
            state->guided_innovation_threshold,
            state->student_t_implied_offset,
            state->use_student_t_state, state->nu_state,
            n
        );
    } else if (state->use_mim) {
        svpf_predict_mim_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev, 1,
            rho_up, rho_down, effective_sigma_z, effective_mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            state->use_student_t_state, state->nu_state,
            n
        );
    } else {
        svpf_predict_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu, params->gamma,
            state->use_student_t_state, state->nu_state,
            n
        );
    }
    
    // =========================================================================
    // GUIDE
    // =========================================================================
    float current_guide_strength = state->guide_strength_base;
    
    if (state->use_guide && state->use_adaptive_guide && state->timestep > 0) {
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;
        
        float implied_h = logf(y_t * y_t + 1e-8f) + state->student_t_implied_offset;
        float h_est = logf(vol_est * vol_est + 1e-8f);
        float h_innovation = implied_h - h_est;
        
        if (h_innovation > 0.0f && return_z > state->guide_innovation_threshold) {
            float severity = fminf((return_z - state->guide_innovation_threshold) / 3.0f, 1.0f);
            float boost = (state->guide_strength_max - state->guide_strength_base) * severity;
            current_guide_strength = state->guide_strength_base + boost;
        }
    }
    
    if (state->use_guide) {
        SVPFParams guide_params = *params;
        if (state->use_adaptive_mu) {
            guide_params.mu = effective_mu;
        }
        svpf_ekf_update(state, y_t, &guide_params);
        
        if (state->use_guide_preserving) {
            svpf_apply_guide_preserving_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
                state->h, opt->d_h_mean_prev, state->guide_mean, current_guide_strength, n
            );
        } else {
            svpf_apply_guide_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
                state->h, state->guide_mean, current_guide_strength, n
            );
        }
    }
    
    // =========================================================================
    // BANDWIDTH
    // =========================================================================
    svpf_fused_bandwidth_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_y_single, opt->d_bandwidth, opt->d_bandwidth_sq,
        state->d_return_ema, state->d_return_var, 1, 0.3f, 0.05f, n
    );
    
    // =========================================================================
    // STEIN ITERATIONS WITH KSD-ADAPTIVE BUDGET
    // =========================================================================
    // Option B: KSD from timestep t determines step budget for timestep t+1
    // - Only 1 D2H sync per timestep (at the end)
    // - No intra-timestep early stopping overhead
    // - High KSD → more steps next timestep
    
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    
    // KSD-ADAPTIVE BETA TEMPERING (Maken et al. 2022)
    // When particles disagree (high KSD), likelihood is flat at boundaries.
    // Trust the prior more by reducing beta.
    // Thresholds:
    //   KSD > 0.50: beta = 0.30 (trust prior)
    //   KSD < 0.05: beta = 0.80 (trust likelihood)
    //   Final step: beta = 1.00 (always commit)
    auto compute_adaptive_beta = [](float ksd_prev, int anneal_idx, int n_anneal_steps) -> float {
        // Final annealing step: always full likelihood
        if (anneal_idx == n_anneal_steps - 1) {
            return 1.0f;
        }
        
        const float ksd_high = 0.50f;
        const float ksd_low = 0.05f;
        const float beta_min = 0.30f;
        const float beta_mid = 0.80f;
        
        float beta;
        if (ksd_prev > ksd_high) {
            beta = beta_min;
        } else if (ksd_prev < ksd_low) {
            beta = beta_mid;
        } else {
            float t = (ksd_high - ksd_prev) / (ksd_high - ksd_low);
            beta = beta_min + t * (beta_mid - beta_min);
        }
        
        // Scale by annealing progress
        float progress = (float)(anneal_idx + 1) / (float)n_anneal_steps;
        beta *= (0.5f + 0.5f * progress);
        
        return fminf(fmaxf(beta, 0.1f), 1.0f);
    };
    
    float base_step = SVPF_STEIN_STEP_SIZE * (state->use_guide ? 0.5f : 1.0f);
    
    // Determine step budget from PREVIOUS timestep's KSD
    int stein_budget;
    if (state->timestep < 10) {
        // Warmup: use max steps
        stein_budget = state->stein_max_steps;
    } else {
        // Map KSD to step count: high KSD → more steps
        // KSD typically ranges 0.01 (converged) to 1.0+ (far from target)
        // Linear interpolation with clamp
        float ksd_low = 0.05f;   // Below this → min steps
        float ksd_high = 0.50f;  // Above this → max steps
        float ksd_normalized = (state->ksd_prev - ksd_low) / (ksd_high - ksd_low);
        ksd_normalized = fminf(fmaxf(ksd_normalized, 0.0f), 1.0f);
        
        stein_budget = state->stein_min_steps + 
                       (int)(ksd_normalized * (state->stein_max_steps - state->stein_min_steps));
    }
    
    int total_steps = 0;
    
    // =========================================================================
    // STANDARD STEIN LOOP (per-iteration kernel launches)
    // =========================================================================
    
    for (int ai = 0; ai < n_anneal; ai++) {
        // Beta computation: adaptive (KSD-based) or fixed schedule
        float beta;
        if (!state->use_annealing) {
            beta = 1.0f;
        } else if (state->use_adaptive_beta) {
            // KSD-adaptive beta (replaces fixed schedule)
            beta = compute_adaptive_beta(state->ksd_prev, ai, n_anneal);
        } else {
            // Fallback to fixed schedule
            static const float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
            beta = beta_schedule[ai % 3];
        }
        float beta_factor = sqrtf(beta);
        float temp = state->use_svld ? state->temperature : 0.0f;
        
        // Distribute budget across annealing stages
        int si = stein_budget / n_anneal;
        if (ai == n_anneal - 1) si = stein_budget - si * (n_anneal - 1);
        
        for (int s = 0; s < si; s++) {
            total_steps++;
            bool is_last_iteration = (ai == n_anneal - 1) && (s == si - 1);
            
            // =================================================================
            // HEUN'S METHOD (Improved Euler, 2nd order)
            // =================================================================
            if (state->use_heun) {
                // --- Step 1: Save original h ---
                svpf_copy_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(state->h, opt->d_h_orig, n);
                
                // --- Step 2: Gradient at original h ---
                svpf_fused_gradient_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
                    state->h, state->h_prev, state->grad_log_p, state->log_weights,
                    state->use_newton ? opt->d_precond_grad : nullptr,
                    state->use_newton ? opt->d_inv_hessian : nullptr,
                    opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu,
                    beta, state->nu, student_t_const, state->lik_offset,
                    params->gamma, state->use_exact_gradient, state->use_newton,
                    state->use_fan_mode,
                    state->use_student_t_state, state->nu_state,
                    n
                );
                
                // --- Step 3: Compute φ(h) at original position ---
                if (state->use_full_newton) {
                    svpf_stein_operator_full_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, state->grad_log_p, opt->d_inv_hessian,
                        opt->d_phi_orig, opt->d_bandwidth,
                        state->stein_repulsive_sign, n
                    );
                } else {
                    svpf_stein_operator_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, state->grad_log_p, opt->d_phi_orig,
                        opt->d_bandwidth, state->stein_repulsive_sign, n
                    );
                }
                
                // --- Step 4: Predictor (Euler step to h̃, NO noise) ---
                svpf_heun_predictor_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
                    state->h, opt->d_h_orig, opt->d_phi_orig,
                    state->d_grad_v,  // RMSProp state (read-only here)
                    base_step, beta_factor, state->rmsprop_eps, n
                );
                
                // --- Step 5: Gradient at predicted h̃ ---
                svpf_fused_gradient_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
                    state->h, state->h_prev, state->grad_log_p, state->log_weights,
                    state->use_newton ? opt->d_precond_grad : nullptr,
                    state->use_newton ? opt->d_inv_hessian : nullptr,
                    opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu,
                    beta, state->nu, student_t_const, state->lik_offset,
                    params->gamma, state->use_exact_gradient, state->use_newton,
                    state->use_fan_mode,
                    state->use_student_t_state, state->nu_state,
                    n
                );
                
                // --- Step 6: Compute φ(h̃) at predicted position ---
                if (state->use_full_newton) {
                    svpf_stein_operator_full_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, state->grad_log_p, opt->d_inv_hessian,
                        opt->d_phi_pred, opt->d_bandwidth,
                        state->stein_repulsive_sign, n
                    );
                } else {
                    svpf_stein_operator_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, state->grad_log_p, opt->d_phi_pred,
                        opt->d_bandwidth, state->stein_repulsive_sign, n
                    );
                }
                
                // --- Step 7: Corrector (average φ, apply from h_orig, with noise) ---
                if (is_last_iteration) {
                    // Final iteration: compute KSD
                    svpf_heun_corrector_ksd_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, opt->d_h_orig, opt->d_phi_orig, opt->d_phi_pred,
                        state->grad_log_p,  // For KSD
                        state->d_grad_v, state->rng_states, opt->d_bandwidth,
                        opt->d_ksd_partial,
                        base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps, n
                    );
                    svpf_ksd_reduce_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
                        opt->d_ksd_partial, opt->d_ksd, n
                    );
                } else {
                    svpf_heun_corrector_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
                        state->h, opt->d_h_orig, opt->d_phi_orig, opt->d_phi_pred,
                        state->d_grad_v, state->rng_states,
                        base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps, n
                    );
                }
            }
            // =================================================================
            // EULER METHOD (Standard, 1st order)
            // =================================================================
            else {
                // Gradient computation
                svpf_fused_gradient_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
                    state->h, state->h_prev, state->grad_log_p, state->log_weights,
                    state->use_newton ? opt->d_precond_grad : nullptr,
                    state->use_newton ? opt->d_inv_hessian : nullptr,
                    opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu,
                    beta, state->nu, student_t_const, state->lik_offset,
                    params->gamma, state->use_exact_gradient, state->use_newton,
                    state->use_fan_mode,
                    state->use_student_t_state, state->nu_state,
                    n
                );
                
                // Stein transport (compute KSD only on LAST iteration)
                if (is_last_iteration) {
                    // Final iteration: compute KSD for next timestep's budget
                    if (state->use_newton) {
                        if (state->use_full_newton) {
                            // Full Newton with KSD
                            svpf_fused_stein_transport_full_newton_ksd_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                                state->h, state->grad_log_p, opt->d_inv_hessian,
                                state->d_grad_v, state->rng_states, opt->d_bandwidth,
                                opt->d_ksd_partial,
                                base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                                state->stein_repulsive_sign, n
                            );
                        } else {
                            // Regular Newton with KSD (uses preconditioned gradient)
                            svpf_fused_stein_transport_newton_ksd_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                                state->h, opt->d_precond_grad, opt->d_inv_hessian,
                                state->d_grad_v, state->rng_states, opt->d_bandwidth,
                                opt->d_ksd_partial,
                                base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                                state->stein_repulsive_sign, n
                            );
                        }
                    } else {
                        // No Newton with KSD
                        svpf_fused_stein_transport_ksd_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                            state->h, state->grad_log_p, state->d_grad_v,
                            state->rng_states, opt->d_bandwidth,
                            opt->d_ksd_partial,
                            base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                            state->stein_repulsive_sign, n
                        );
                    }
                    
                    // Reduce KSD (async - will sync later with outputs)
                    svpf_ksd_reduce_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
                        opt->d_ksd_partial, opt->d_ksd, n
                    );
                } else {
                    // Non-final iterations: standard transport (no KSD overhead)
                    if (state->use_newton) {
                        if (state->use_full_newton) {
                            svpf_fused_stein_transport_full_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                                state->h, state->grad_log_p, opt->d_inv_hessian,
                                state->d_grad_v, state->rng_states, opt->d_bandwidth,
                                base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                                state->stein_repulsive_sign, n
                            );
                        } else {
                            svpf_fused_stein_transport_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                                state->h, opt->d_precond_grad, opt->d_inv_hessian,
                                state->d_grad_v, state->rng_states, opt->d_bandwidth,
                                base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                                state->stein_repulsive_sign, n
                            );
                        }
                    } else {
                        svpf_fused_stein_transport_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                            state->h, state->grad_log_p, state->d_grad_v,
                            state->rng_states, opt->d_bandwidth,
                            base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                            state->stein_repulsive_sign, n
                        );
                    }
                }
            }
        }
    }
    
    // Store diagnostic
    state->stein_steps_used = total_steps;
    
    // =========================================================================
    // PARTIAL REJUVENATION (Maken et al. 2022)
    // =========================================================================
    // If KSD is still high after Stein (particles stuck), nudge some toward guide.
    // Uses ksd_prev since we haven't computed this timestep's KSD yet.
    // This helps particles escape reflecting boundaries.
    if (state->use_rejuvenation && state->use_guide && state->timestep > 10) {
        if (state->ksd_prev > state->rejuv_ksd_threshold) {
            float guide_std = sqrtf(fmaxf(state->guide_var, 1e-6f));
            svpf_partial_rejuvenation_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
                state->h,
                state->guide_mean,
                guide_std,
                state->rejuv_prob,
                state->rejuv_blend,
                state->rng_states,
                n
            );
        }
    }
    
    // =========================================================================
    // OUTPUTS
    // =========================================================================
    svpf_fused_outputs_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, state->log_weights,
        opt->d_loglik_single, opt->d_vol_single, opt->d_h_mean_prev, 0, n
    );
    
    // Read results (including KSD for next timestep's budget)
    float results[5];
    cudaMemcpyAsync(&results[0], opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(&results[1], opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(&results[2], opt->d_h_mean_prev, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(&results[3], opt->d_bandwidth, sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(&results[4], opt->d_ksd, sizeof(float), cudaMemcpyDeviceToHost, cs);
    
    cudaStreamSynchronize(cs);
    
    float h_mean_local = results[2];
    float bandwidth_local = results[3];
    float vol_local = results[1];
    float ksd_local = results[4];
    
    // =========================================================================
    // BACKWARD SMOOTHING (Optional)
    // =========================================================================
    // Estimate variance from bandwidth (heuristic: var ≈ bw²)
    float h_var_est = bandwidth_local * bandwidth_local;
    svpf_smooth_backward(state, h_mean_local, h_var_est, y_t, params);
    
    // Get smoothed output (if smoothing enabled and output_lag > 0)
    float h_mean_output = svpf_get_smoothed_output(state, h_mean_local);
    
    // Return outputs
    if (h_loglik_out) *h_loglik_out = results[0];
    if (h_vol_out) *h_vol_out = vol_local;
    if (h_mean_out) *h_mean_out = h_mean_output;  // Smoothed if enabled
    
    state->vol_prev = vol_local;
    state->ksd_prev = ksd_local;  // For next timestep's step budget
    
    if (state->use_adaptive_mu && state->timestep > 10) {
        svpf_adaptive_mu_update(state, h_mean_local, bandwidth_local);
    }
    
    state->timestep++;
}

void svpf_step_adaptive(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                        float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    svpf_step_graph(state, y_t, y_prev, params, h_loglik_out, h_vol_out, h_mean_out);
}

void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result) {
    float loglik, vol, h_mean;
    svpf_step_graph(state, y_t, state->y_prev, params, &loglik, &vol, &h_mean);
    if (result) {
        result->log_lik_increment = loglik;
        result->vol_mean = vol;
        result->h_mean = h_mean;
        result->vol_std = 0.0f;
        result->mu_estimate = state->use_adaptive_mu ? state->mu_state : params->mu;
    }
    state->y_prev = y_t;
}

void svpf_step_seeded(SVPFState* state, float y_t, const SVPFParams* params,
                      unsigned long long rng_seed, SVPFResult* result) {
    svpf_step(state, y_t, params, result);
}

void svpf_run_sequence(SVPFState* state, const float* h_obs, int T, const SVPFParams* params,
                       float* h_loglik_out, float* h_vol_out) {
    float y_prev = 0.0f;
    for (int t = 0; t < T; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, h_obs[t], y_prev, params, &loglik, &vol, &h_mean);
        if (h_loglik_out) h_loglik_out[t] = loglik;
        if (h_vol_out) h_vol_out[t] = vol;
        y_prev = h_obs[t];
    }
}

void svpf_run_sequence_device(SVPFState* state, const float* d_obs, int T, const SVPFParams* params,
                              float* d_loglik_out, float* d_vol_out) {
    float* h_obs = (float*)malloc(T * sizeof(float));
    float* h_ll = d_loglik_out ? (float*)malloc(T * sizeof(float)) : NULL;
    float* h_vol = d_vol_out ? (float*)malloc(T * sizeof(float)) : NULL;
    
    cudaMemcpy(h_obs, d_obs, T * sizeof(float), cudaMemcpyDeviceToHost);
    svpf_run_sequence(state, h_obs, T, params, h_ll, h_vol);
    
    if (h_ll) { cudaMemcpy(d_loglik_out, h_ll, T * sizeof(float), cudaMemcpyHostToDevice); free(h_ll); }
    if (h_vol) { cudaMemcpy(d_vol_out, h_vol, T * sizeof(float), cudaMemcpyHostToDevice); free(h_vol); }
    free(h_obs);
}

bool svpf_graph_is_captured(SVPFState* state) { return get_opt(state)->graph_captured; }

void svpf_graph_invalidate(SVPFState* state) {
    SVPFOptimizedState* opt = get_opt(state);
    if (opt->graph_captured) {
        cudaGraphExecDestroy(opt->graph_exec);
        cudaGraphDestroy(opt->graph);
        opt->graph_captured = false;
    }
}

void svpf_get_particles(const SVPFState* state, float* h_out) {
    cudaMemcpy(h_out, state->h, state->n_particles * sizeof(float), cudaMemcpyDeviceToHost);
}

void svpf_get_stats(const SVPFState* state, float* h_mean, float* h_std) {
    int n = state->n_particles;
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, state->h, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += h[i];
    *h_mean = sum / (float)n;
    
    float sq = 0.0f;
    for (int i = 0; i < n; i++) { float d = h[i] - *h_mean; sq += d * d; }
    *h_std = sqrtf(sq / (float)n);
    
    free(h);
}

float svpf_get_ess(const SVPFState* state) {
    int n = state->n_particles;
    float* lw = (float*)malloc(n * sizeof(float));
    cudaMemcpy(lw, state->log_weights, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float mx = lw[0];
    for (int i = 1; i < n; i++) if (lw[i] > mx) mx = lw[i];
    
    float sw = 0.0f, sw2 = 0.0f;
    for (int i = 0; i < n; i++) { float w = expf(lw[i] - mx); sw += w; sw2 += w * w; }
    
    free(lw);
    return (sw * sw) / (sw2 + 1e-10f);
}

// =============================================================================
// DIAGNOSTIC: Get KSD and Stein steps used
// =============================================================================

void svpf_get_ksd_stats(const SVPFState* state, float* ksd_out, int* steps_used_out) {
    if (ksd_out) *ksd_out = state->ksd_prev;
    if (steps_used_out) *steps_used_out = state->stein_steps_used;
}
