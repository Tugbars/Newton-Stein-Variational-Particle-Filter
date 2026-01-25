/**
 * @file svpf_optimized_graph.cu
 * @brief Consolidated SVPF Implementation
 * 
 * Single implementation file containing:
 * - State management (svpf_create/destroy/initialize)
 * - Basic utility kernels
 * - Graph capture
 * - Public API (svpf_step_graph, svpf_step_adaptive, svpf_run_sequence)
 * - Diagnostics (svpf_get_particles, svpf_get_stats, svpf_get_ess)
 * 
 * External kernel files:
 * - svpf_opt_kernels.cu: Fused kernels (gradient, Stein, bandwidth, outputs)
 */

#include "svpf_kernels.cuh"
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
    
    // Zero-initialize opt_backend (CRITICAL - malloc doesn't zero memory)
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
    
    // Compute student_t_implied_offset for observation-to-h mapping
    // For Student-t(ν), E[log(y²)|h] = h + E[log(t²_ν)]
    // where E[log(t²_ν)] = log(ν) + ψ(1/2) - ψ(ν/2)
    // We want: h_implied = log(y²) + offset, so offset = -E[log(t²_ν)]
    //
    // Digamma function approximations:
    //   ψ(1/2) = -γ - 2*ln(2) ≈ -1.9635100260214235  (exact)
    //   ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²)  for x > 1  (asymptotic)
    {
        const float psi_half = -1.9635100260214235f;  // ψ(1/2) exact
        float nu_half = nu / 2.0f;
        float psi_nu_half;
        if (nu_half >= 1.0f) {
            // Asymptotic expansion for ψ(x) when x >= 1
            psi_nu_half = logf(nu_half) - 1.0f/(2.0f*nu_half) - 1.0f/(12.0f*nu_half*nu_half);
        } else {
            // For small ν, use recurrence: ψ(x+1) = ψ(x) + 1/x
            // ψ(1) = -γ ≈ -0.5772
            psi_nu_half = -0.5772156649f - 1.0f/nu_half;  // Rough approximation for small ν
        }
        float expected_log_t_sq = logf(nu) + psi_half - psi_nu_half;
        state->student_t_implied_offset = -expected_log_t_sq;
        // For ν=30: offset ≈ 1.24
        // For ν→∞: offset → 1.27 (Gaussian limit)
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
    
    // === ADAPTIVE SVPF: Per-particle RMSProp state ===
    cudaMalloc(&state->d_grad_v, n * sizeof(float));
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));
    
    // === ADAPTIVE SVPF: Regime detection scalars ===
    cudaMalloc(&state->d_return_ema, sizeof(float));
    cudaMalloc(&state->d_return_var, sizeof(float));
    cudaMalloc(&state->d_bw_alpha, sizeof(float));
    float init_ema = 0.0f;
    float init_alpha = 0.3f;
    cudaMemcpy(state->d_return_ema, &init_ema, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_return_var, &init_ema, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_bw_alpha, &init_alpha, sizeof(float), cudaMemcpyHostToDevice);
    
    // === Likelihood gradient config ===
    // Likelihood offset/bias correction:
    //   For SURROGATE (use_exact_gradient=0):
    //     - Offset in log-squared gradient: (log(y²) - h + lik_offset) / R
    //     - Tuned value: 0.70 for minimal bias
    //   For EXACT (use_exact_gradient=1):
    //     - Bias correction subtracted from exact gradient
    //     - Raw exact gradient has ~+0.30 positive bias at equilibrium
    //     - Tuned value: ~0.25-0.30 to center gradient around zero
    state->lik_offset = 0.70f;  // For surrogate. Try 0.25-0.30 for exact gradient.
    
    // Exact vs Surrogate gradient selection:
    //   - 0 = Surrogate (log-squared), needs lik_offset tuning, no saturation
    //   - 1 = Exact Student-t, consistent with weights/Hessian, saturates at ±nu/2
    // Recommended: use_exact_gradient=1 when nu >= 30, with lik_offset ≈ 0.27
    state->use_exact_gradient = 0;  // Default to legacy for backward compatibility
    
    // === ADAPTIVE SVPF: Configuration defaults ===
    state->use_svld = 1;
    state->use_annealing = 1;
    state->n_anneal_steps = 3;
    state->temperature = 1.0f;
    state->rmsprop_rho = 0.9f;
    state->rmsprop_eps = 1e-6f;
    
    // === Mixture Innovation Model defaults ===
    state->use_mim = 1;
    state->mim_jump_prob = 0.05f;
    state->mim_jump_scale = 5.0f;
    
    // === Asymmetric persistence defaults ===
    state->use_asymmetric_rho = 1;
    state->rho_up = 0.98f;
    state->rho_down = 0.93f;
    
    // === Guide density (EKF) defaults ===
    state->use_guide = 1;
    state->use_guide_preserving = 1;
    state->guide_strength = 0.05f;  // Base strength
    state->guide_mean = 0.0f;
    state->guide_var = 0.0f;
    state->guide_K = 0.0f;
    state->guide_initialized = 0;
    
    // === Adaptive Guide Strength defaults ===
    state->use_adaptive_guide = 0;           // Disabled by default
    state->guide_strength_base = 0.05f;      // Base when model fits
    state->guide_strength_max = 0.30f;       // Max during surprises
    state->guide_innovation_threshold = 1.0f; // Z-score threshold for boost
    state->vol_prev = 0.05f;                 // Initial vol estimate
    
    // === Newton-Stein defaults ===
    state->use_newton = 0;
    state->use_full_newton = 0;  // 0=local Hessian (fast), 1=kernel-weighted (Detommaso 2018)
    
    // === Guided Prediction defaults ===
    state->use_guided = 0;
    state->guided_alpha_base = 0.0f;
    state->guided_alpha_shock = 0.5f;
    state->guided_innovation_threshold = 1.5f;
    
    // === Particle-local parameters defaults ===
    state->use_local_params = 0;
    state->delta_rho = 0.02f;
    state->delta_sigma = 0.1f;
    
    // === Adaptive Mu (1D Kalman Filter) defaults ===
    state->use_adaptive_mu = 0;          // Disabled by default
    state->mu_state = -3.5f;             // Initial mu estimate
    state->mu_var = 1.0f;                // Initial uncertainty (will shrink)
    state->mu_process_var = 0.001f;      // Q: slow drift allowed (~0.03/step std)
    state->mu_obs_var_scale = 10.0f;     // R = scale * bandwidth²
    state->mu_min = -6.0f;               // Lower bound
    state->mu_max = -1.0f;               // Upper bound
    
    // === Adaptive Sigma_Z (Breathing Filter) defaults ===
    state->use_adaptive_sigma = 0;       // Disabled by default
    state->sigma_boost_threshold = 1.0f; // Start boosting when |z| > 1
    state->sigma_boost_max = 3.0f;       // Max 3x boost
    state->sigma_z_effective = 0.10f;    // Calibrated via gradient diagnostic (was 0.15)
    
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

// Forward declaration
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
    float stationary_var = (sigma_z * sigma_z) / (1.0f - rho * rho + 1e-6f);
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
    
    // Initialize adaptive mu Kalman filter state
    if (state->use_adaptive_mu) {
        state->mu_state = params->mu;         // Start from provided mu
        state->mu_var = 1.0f;                 // High initial uncertainty
    }
    
    // Initialize adaptive sigma_z state
    if (state->use_adaptive_sigma) {
        state->sigma_z_effective = params->sigma_z;
    }
    
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));
    
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
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    // Pinned host memory for fast D2H transfers
    // Layout: [loglik, vol, h_mean, bandwidth]
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
    
    // Free pinned host memory
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
// GRAPH CAPTURE: 1D Path
// =============================================================================

static void svpf_graph_capture_internal(SVPFState* state, const SVPFParams* params) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t cs = opt->graph_stream;
    
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t grad_smem = 2 * n * sizeof(float);
    size_t stein_smem = state->use_newton ? 3 * n * sizeof(float) : 2 * n * sizeof(float);
    
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
    float base_step = SVPF_STEIN_STEP_SIZE * (state->use_guide ? 0.5f : 1.0f);
    
    float rho_up = state->use_asymmetric_rho ? state->rho_up : params->rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : params->rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);
    
    // PREDICT
    if (state->use_guided) {
        svpf_predict_guided_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev, 1,
            rho_up, rho_down, params->sigma_z, params->mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            state->guided_alpha_base, state->guided_alpha_shock,
            state->guided_innovation_threshold, 
            state->student_t_implied_offset, n
        );
    } else if (state->use_mim) {
        svpf_predict_mim_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev, 1,
            rho_up, rho_down, params->sigma_z, params->mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma, n
        );
    } else {
        svpf_predict_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, params->rho, params->sigma_z, params->mu, params->gamma, n
        );
    }
    
#ifdef SVPF_ENABLE_DIAGNOSTICS
    // Snapshot predicted particles BEFORE Stein transport (for σ gradient diagnostic)
    // h_pred = h after prediction, before any transport/guide modifications
    cudaMemcpyAsync(state->h_pred, state->h, n * sizeof(float), cudaMemcpyDeviceToDevice, cs);
#endif
    
    // GUIDE
    if (state->use_guide) {
        if (state->use_guide_preserving) {
            svpf_apply_guide_preserving_kernel_graph<<<nb, BLOCK_SIZE, 0, cs>>>(
                state->h, opt->d_h_mean_prev, opt->d_guide_mean, opt->d_guide_strength, n
            );
        } else {
            svpf_apply_guide_kernel_graph<<<nb, BLOCK_SIZE, 0, cs>>>(
                state->h, opt->d_guide_mean, opt->d_guide_strength, n
            );
        }
    }
    
    // BANDWIDTH
    svpf_fused_bandwidth_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_y_single, opt->d_bandwidth, opt->d_bandwidth_sq,
        state->d_return_ema, state->d_return_var, 1, 0.3f, 0.05f, n
    );
    
    // STEIN ITERATIONS
    for (int ai = 0; ai < n_anneal; ai++) {
        float beta = state->use_annealing ? beta_schedule[ai % 3] : 1.0f;
        float beta_factor = sqrtf(beta);
        float temp = state->use_svld ? state->temperature : 0.0f;
        
        int si = state->n_stein_steps / n_anneal;
        if (ai == n_anneal - 1) si = state->n_stein_steps - si * (n_anneal - 1);
        
        for (int s = 0; s < si; s++) {
            svpf_fused_gradient_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
                state->h, state->h_prev, state->grad_log_p, state->log_weights,
                state->use_newton ? opt->d_precond_grad : nullptr,
                state->use_newton ? opt->d_inv_hessian : nullptr,
                opt->d_y_single, 1, params->rho, params->sigma_z, params->mu,
                beta, state->nu, student_t_const, state->lik_offset,
                params->gamma,  // Leverage coefficient for prior consistency
                state->use_exact_gradient,  // Exact Student-t vs log-squared surrogate
                state->use_newton, n
            );
            
            if (state->use_newton) {
                if (state->use_full_newton) {
                    // Full Newton: kernel-weighted Hessian averaging (Detommaso 2018)
                    // Uses raw gradient and local curvature, computes weighted H in kernel
                    svpf_fused_stein_transport_full_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, state->grad_log_p, opt->d_inv_hessian,
                        state->d_grad_v, state->rng_states, opt->d_bandwidth,
                        base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps, n
                    );
                } else {
                    // Approximate Newton: local Hessian only (faster)
                    svpf_fused_stein_transport_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        state->d_grad_v, state->rng_states, opt->d_bandwidth,
                        base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps, n
                    );
                }
            } else {
                svpf_fused_stein_transport_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                    state->h, state->grad_log_p, state->d_grad_v,
                    state->rng_states, opt->d_bandwidth,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps, n
                );
            }
        }
    }
    
    // OUTPUTS
    svpf_fused_outputs_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, state->log_weights,
        opt->d_loglik_single, opt->d_vol_single, opt->d_h_mean_prev, 0, n
    );
    
    cudaStreamEndCapture(cs, &opt->graph);
    cudaGraphInstantiate(&opt->graph_exec, opt->graph, NULL, NULL, 0);
    
    opt->graph_captured = true;
    opt->graph_n = n;
    opt->graph_n_stein = state->n_stein_steps;
}

// =============================================================================
// ADAPTIVE MU: 1D Kalman Filter Update
// =============================================================================
// 
// Uses particle confidence (inverse bandwidth) to gate learning rate:
// - Calm market (low bandwidth) → high confidence → adapt mu quickly
// - Crisis (high bandwidth) → low confidence → freeze mu
//
// Kalman equations:
//   Predict: mu_pred = mu, P_pred = P + Q
//   Update:  K = P_pred / (P_pred + R)
//            mu = mu_pred + K * (h_mean - mu_pred)
//            P = (1 - K) * P_pred
//
// Where R = scale * bandwidth² (measurement noise from particle spread)

static void svpf_adaptive_mu_update(
    SVPFState* state,
    float h_mean,       // Observation: particle mean
    float bandwidth     // Particle spread → measurement noise
) {
    if (!state->use_adaptive_mu) return;
    
    // Kalman predict
    float mu_pred = state->mu_state;
    float P_pred = state->mu_var + state->mu_process_var;
    
    // Measurement noise: high bandwidth → high noise → ignore observation
    float R = state->mu_obs_var_scale * bandwidth * bandwidth;
    
    // Kalman gain
    float K = P_pred / (P_pred + R + 1e-8f);
    
    // Innovation
    float innovation = h_mean - mu_pred;
    
    // Update
    float mu_new = mu_pred + K * innovation;
    float P_new = (1.0f - K) * P_pred;
    
    // Clamp mu to valid range
    mu_new = fminf(fmaxf(mu_new, state->mu_min), state->mu_max);
    
    // Store
    state->mu_state = mu_new;
    state->mu_var = P_new;
}

// =============================================================================
// PUBLIC API
// =============================================================================

void svpf_step_graph(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                     float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    
    svpf_optimized_init(opt, n);
    
    // Determine effective mu (adaptive or fixed)
    float effective_mu = state->use_adaptive_mu ? state->mu_state : params->mu;
    
    // =========================================================================
    // ADAPTIVE SIGMA_Z: Innovation-gated vol-of-vol ("Breathing Filter")
    // =========================================================================
    // Boost sigma_z when innovation is high to allow particles to spread faster
    float effective_sigma_z = params->sigma_z;
    
    if (state->use_adaptive_sigma && state->timestep > 0) {
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;
        
        // Boost sigma_z when innovation exceeds threshold
        float sigma_boost = 1.0f;
        if (return_z > state->sigma_boost_threshold) {
            float severity = fminf((return_z - state->sigma_boost_threshold) / 3.0f, 1.0f);
            sigma_boost = 1.0f + (state->sigma_boost_max - 1.0f) * severity;
        }
        
        effective_sigma_z = params->sigma_z * sigma_boost;
        state->sigma_z_effective = effective_sigma_z;
    }
    
    // Check if graph needs recapture
    bool need_capture = !opt->graph_captured 
                     || opt->graph_n != n 
                     || opt->graph_n_stein != state->n_stein_steps;
    
    // If adaptive mu, check if mu has drifted significantly from captured value
    if (state->use_adaptive_mu && opt->graph_captured) {
        float mu_drift = fabsf(effective_mu - opt->mu_captured);
        if (mu_drift > 0.1f) {
            need_capture = true;
        }
    }
    
    // If adaptive sigma_z, check if it has drifted significantly
    if (state->use_adaptive_sigma && opt->graph_captured) {
        float sigma_drift = fabsf(effective_sigma_z - opt->sigma_z_captured);
        if (sigma_drift > 0.05f) {  // Recapture if sigma_z changed by more than 0.05
            need_capture = true;
        }
    }
    
    if (need_capture) {
        if (opt->graph_captured) {
            cudaGraphExecDestroy(opt->graph_exec);
            cudaGraphDestroy(opt->graph);
            opt->graph_captured = false;
        }
        
        // Create modified params for capture
        SVPFParams capture_params = *params;
        if (state->use_adaptive_mu) {
            capture_params.mu = effective_mu;
        }
        if (state->use_adaptive_sigma) {
            capture_params.sigma_z = effective_sigma_z;
        }
        
        svpf_graph_capture_internal(state, &capture_params);
        opt->mu_captured = effective_mu;
        opt->sigma_z_captured = effective_sigma_z;
    }
    
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, opt->graph_stream);
    
    // =========================================================================
    // ADAPTIVE GUIDE STRENGTH: Asymmetric Innovation-gated nudging
    // =========================================================================
    // Only boost when IMPLIED volatility > ESTIMATED volatility (upward surprise)
    // Downward surprises (price calm when vol expected high) are often just mean reversion
    // Upward surprises (price crash) are INFORMATION - trust the guide!
    float current_guide_strength = state->guide_strength_base;
    
    if (state->use_guide && state->use_adaptive_guide && state->timestep > 0) {
        // Use previous vol estimate as proxy for prediction
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        
        // Return z-score: magnitude of surprise
        float return_z = fabsf(y_t) / vol_est;
        
        // Implied h from observation: log(y²) + student_t_implied_offset
        float implied_h = logf(y_t * y_t + 1e-8f) + state->student_t_implied_offset;
        
        // h_prev is our current estimate (from last step's h_mean)
        float h_est = logf(vol_est * vol_est + 1e-8f);  // Convert vol back to h
        
        // Innovation in log-vol space
        float h_innovation = implied_h - h_est;
        
        // ASYMMETRIC LOGIC:
        // Only boost if implied volatility is HIGHER than current estimate (h_innovation > 0)
        // AND the move is statistically significant (return_z > threshold)
        if (h_innovation > 0.0f && return_z > state->guide_innovation_threshold) {
            // We are surprised by a spike. Trust the guide!
            float severity = fminf((return_z - state->guide_innovation_threshold) / 3.0f, 1.0f);
            float boost = (state->guide_strength_max - state->guide_strength_base) * severity;
            current_guide_strength = state->guide_strength_base + boost;
        }
        // else: downward surprise or small move -> keep base strength
    }
    
    // Upload adaptive guide strength to device
    cudaMemcpyAsync(opt->d_guide_strength, &current_guide_strength, sizeof(float), cudaMemcpyHostToDevice, opt->graph_stream);
    
    if (state->use_guide) {
        // Use effective_mu for EKF update
        SVPFParams guide_params = *params;
        if (state->use_adaptive_mu) {
            guide_params.mu = effective_mu;
        }
        svpf_ekf_update(state, y_t, &guide_params);
        cudaMemcpyAsync(opt->d_guide_mean, &state->guide_mean, sizeof(float), cudaMemcpyHostToDevice, opt->graph_stream);
    }
    
    // NOTE: No sync needed here! cudaMemcpyAsync on same stream guarantees
    // ordering - the graph will see the uploaded params.
    cudaGraphLaunch(opt->graph_exec, opt->graph_stream);
    
    // Read back results using PINNED MEMORY for faster D2H
    // NO SYNC between graph and memcpy - stream ordering handles it!
    // Layout: [0]=loglik, [1]=vol, [2]=h_mean, [3]=bandwidth
    float* results = opt->h_results_pinned;
    
    // Async copies to pinned memory - queued after graph completion
    cudaMemcpyAsync(&results[0], opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost, opt->graph_stream);
    cudaMemcpyAsync(&results[1], opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost, opt->graph_stream);
    cudaMemcpyAsync(&results[2], opt->d_h_mean_prev, sizeof(float), cudaMemcpyDeviceToHost, opt->graph_stream);
    cudaMemcpyAsync(&results[3], opt->d_bandwidth, sizeof(float), cudaMemcpyDeviceToHost, opt->graph_stream);
    
    // SINGLE sync - waits for graph AND all memcpy to complete
    cudaStreamSynchronize(opt->graph_stream);
    
    if (h_loglik_out) *h_loglik_out = results[0];
    if (h_vol_out) *h_vol_out = results[1];
    if (h_mean_out) *h_mean_out = results[2];
    
    float h_mean_local = results[2];
    float bandwidth_local = results[3];
    float vol_local = results[1];
    
    // Update state for next step
    state->vol_prev = vol_local;              // For adaptive guide and adaptive sigma
    
    // =========================================================================
    // ADAPTIVE MU: 1D Kalman Filter Update
    // =========================================================================
    // Signal: h_mean (particle posterior mean)
    // Noise: bandwidth² (particle uncertainty)
    // Effect: Fast adaptation in calm markets, freeze during crises
    if (state->use_adaptive_mu && state->timestep > 10) {  // Skip warmup
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
