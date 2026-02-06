/**
 * @file svpf_optimized_graph.cu
 * @brief Consolidated SVPF Implementation with KSD-based Adaptive Stein Steps
 * 
 * Single implementation file containing:
 * - State management (svpf_create/destroy/initialize)
 * - Basic utility kernels
 * - Stein loop with KSD-based early stopping (now via CUDA Graphs)
 * - Public API (svpf_step_graph, svpf_step_adaptive, svpf_run_sequence)
 * - Diagnostics (svpf_get_particles, svpf_get_stats, svpf_get_ess)
 * 
 * KSD (Kernel Stein Discrepancy) is computed in the same O(N²) pass as Stein
 * transport at zero extra cost. Early stopping occurs when relative KSD
 * improvement drops below threshold.
 * 
 * CUDA GRAPHS: The Stein loop is now captured as a CUDA Graph to eliminate
 * kernel launch overhead. This reduces 16-64 kernel launches to 1 graph launch.
 */

#include "svpf_kernels.cuh"
#include "svpf_fused_gradient_stats.cuh"  // Fused gradient + stats kernel

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
    
    // =========================================================================
    // Production defaults (matches test harness configuration)
    // =========================================================================
    
    state->use_exact_gradient = 1;
    state->lik_offset = 0.345f;
    
    // --- SVLD + Annealing ---
    state->use_svld = 1;
    state->use_annealing = 1;
    state->use_adaptive_beta = 1;   // KSD-adaptive beta (Maken 2022)
    state->n_anneal_steps = 5;
    state->temperature = 0.45f;
    state->rmsprop_rho = 0.7f;
    state->rmsprop_eps = 1e-6f;
    
    // --- MIM (OFF by default — guided prediction supersedes) ---
    state->use_mim = 0;
    state->mim_jump_prob = 0.25f;
    state->mim_jump_scale = 9.0f;
    
    // --- Asymmetric persistence ---
    state->use_asymmetric_rho = 0;
    state->rho_up = 0.98f;
    state->rho_down = 0.93f;
    
    // --- EKF Guide density ---
    state->use_guide = 1;
    state->use_guide_preserving = 1;  // Variance-preserving shift (not contraction)
    state->guide_strength = 0.05f;
    state->guide_mean = 0.0f;
    state->guide_var = 0.0f;
    state->guide_K = 0.0f;
    state->guide_initialized = 1;
    
    // --- Adaptive guide (innovation-gated strength) ---
    state->use_adaptive_guide = 1;
    state->guide_strength_base = 0.05f;       // Base when model fits
    state->guide_strength_max = 0.30f;        // Max during surprises
    state->guide_innovation_threshold = 1.0f; // Z-score to start boosting
    state->vol_prev = 0.05f;
    
    // --- Partial rejuvenation (Maken 2022) ---
    state->use_rejuvenation = 1;
    state->rejuv_ksd_threshold = 0.05f;  // Trigger threshold
    state->rejuv_prob = 0.30f;           // 30% of particles
    state->rejuv_blend = 0.30f;          // 30% blend factor
    
    // --- Newton-Stein (Hessian preconditioning) ---
    state->use_newton = 1;
    state->use_full_newton = 1;  // Detommaso 2018 full Newton
    
    // --- Guided Prediction with innovation gating ---
    state->use_guided = 1;
    state->guided_alpha_base = 0.0f;             // 0% when model fits
    state->guided_alpha_shock = 0.40f;            // 40% when model fails
    state->guided_innovation_threshold = 1.5f;    // 1.5σ = "surprised"
    
    // --- Local parameter perturbation ---
    state->use_local_params = 0;
    state->delta_rho = 0.02f;
    state->delta_sigma = 0.1f;
    
    // --- Adaptive mu (Kalman drift) ---
    state->use_adaptive_mu = 1;
    state->mu_state = -3.5f;
    state->mu_var = 1.0f;
    state->mu_process_var = 0.001f;   // Q: how fast can mu drift
    state->mu_obs_var_scale = 11.0f;  // R = scale * bw²
    state->mu_min = -4.0f;
    state->mu_max = -1.0f;
    
    // --- Adaptive sigma (volatility-of-volatility boost) ---
    state->use_adaptive_sigma = 1;
    state->sigma_boost_threshold = 0.95f;  // Start boosting when |z| > ~1
    state->sigma_boost_max = 3.2f;         // Max 3.2× boost
    state->sigma_z_effective = 0.10f;
    
    // === Stein operator sign mode ===
    state->stein_repulsive_sign = SVPF_STEIN_SIGN_DEFAULT;
    
    // === Fan mode (weightless SVGD) ===
    state->use_fan_mode = 0;
    
    // === Student-t state dynamics ===
    state->use_student_t_state = 1;
    state->nu_state = 5.0f;
    
    // === KSD-based Adaptive Stein Steps ===
    state->stein_min_steps = 8;
    state->stein_max_steps = 16;
    state->ksd_improvement_threshold = 0.05f;
    state->ksd_prev = 1e10f;
    state->stein_steps_used = n_stein_steps;
    
    // === Heun's Method (OFF) ===
    state->use_heun = 0;
    
    // === Antithetic Sampling ===
    state->use_antithetic = 1;
    
    // === Adaptive Annealing (KL-based beta stepping) ===
    state->use_adaptive_anneal = 1;
    state->anneal_kl_threshold = 0.9f;
    state->anneal_steps_per_beta = 3;
    state->anneal_max_stages = 50;
    state->anneal_stages_used = 0;
    state->anneal_final_var_ll = 0.0f;
    state->anneal_final_h_std = 0.0f;
    
    // === Backward Smoothing (Fan et al. 2021 sliding window) ===
    state->use_smoothing = 1;
    state->smooth_lag = 3;
    state->smooth_output_lag = 1;
    for (int i = 0; i < SVPF_SMOOTH_MAX_LAG; i++) {
        state->smooth_h_mean[i] = 0.0f;
        state->smooth_h_var[i] = 1.0f;
        state->smooth_y[i] = 0.0f;
    }
    state->smooth_head = 1;
    
    // === Persistent kernel ===
    state->use_persistent_kernel = 1;
    
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
    float base_var = (sigma_z * sigma_z) / (1.0f - rho * rho + 1e-6f);
    float stationary_var;
    
    if (state->use_student_t_state) {
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
    
    // === Consolidated output pack (single D2H transfer) ===
    cudaMalloc(&opt->d_output_pack, 8 * sizeof(float));  // 32 bytes aligned
    cudaMallocHost(&opt->h_output_pinned, 8 * sizeof(float));
    
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    cudaMallocHost(&opt->h_results_pinned, 4 * sizeof(float));
    
    // === Adaptive Annealing Buffers ===
    cudaMalloc(&opt->d_anneal_stats, 4 * sizeof(float));
    cudaMallocHost(&opt->h_anneal_stats_pinned, 4 * sizeof(float));
    
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
    
    // === Consolidated output pack ===
    cudaFree(opt->d_output_pack);
    if (opt->h_output_pinned) {
        cudaFreeHost(opt->h_output_pinned);
        opt->h_output_pinned = nullptr;
    }
    
    if (opt->h_results_pinned) {
        cudaFreeHost(opt->h_results_pinned);
        opt->h_results_pinned = nullptr;
    }
    
    // === Adaptive Annealing Buffers ===
    if (opt->d_anneal_stats) {
        cudaFree(opt->d_anneal_stats);
        opt->d_anneal_stats = nullptr;
    }
    if (opt->h_anneal_stats_pinned) {
        cudaFreeHost(opt->h_anneal_stats_pinned);
        opt->h_anneal_stats_pinned = nullptr;
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
    
    for (int lag = 1; lag < k; lag++) {
        int idx_curr = (state->smooth_head - lag - 1 + k) % k;
        int idx_next = (state->smooth_head - lag + k) % k;
        
        float h_curr = state->smooth_h_mean[idx_curr];
        float h_next = state->smooth_h_mean[idx_next];
        float var_curr = state->smooth_h_var[idx_curr];
        
        float h_pred = mu + rho * (h_curr - mu);
        float pred_var = rho * rho * var_curr + sigma_z_sq;
        float innovation = h_next - h_pred;
        float J = rho * var_curr / (pred_var + 1e-8f);
        
        state->smooth_h_mean[idx_curr] = h_curr + J * innovation;
        state->smooth_h_var[idx_curr] = var_curr * (1.0f - J * rho);
    }
}

// Get smoothed output (with configured lag)
static float svpf_get_smoothed_output(SVPFState* state, float h_mean_raw) {
    if (!state->use_smoothing) return h_mean_raw;
    
    int k = state->smooth_lag;
    if (k > SVPF_SMOOTH_MAX_LAG) k = SVPF_SMOOTH_MAX_LAG;
    
    if (state->timestep < k) return h_mean_raw;
    
    int output_lag = state->smooth_output_lag;
    if (output_lag <= 0) return h_mean_raw;
    if (output_lag >= k) output_lag = k - 1;
    
    int idx = (state->smooth_head - output_lag - 1 + k) % k;
    return state->smooth_h_mean[idx];
}

// =============================================================================
// ASYNC STEP: Launch all GPU work, return immediately
// =============================================================================

void svpf_step_async(SVPFState* state, float y_t, float y_prev, const SVPFParams* params) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t cs = state->stream;
    
    svpf_optimized_init(opt, n);
    
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
    size_t stein_smem = 3 * n * sizeof(float);  // Full Newton always uses 3× shared
    
    float rho_up = state->use_asymmetric_rho ? state->rho_up : params->rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : params->rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    // Upload y values
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, cs);
    
    // =========================================================================
    // PREDICT (Antithetic guided)
    // =========================================================================
    {
        int nb_half = ((n / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        svpf_predict_guided_antithetic_kernel<<<nb_half, BLOCK_SIZE, 0, cs>>>(
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
    }

    // =========================================================================
    // GUIDE (Variance-preserving)
    // =========================================================================
    float current_guide_strength = state->guide_strength_base;
    
    if (state->use_adaptive_guide && state->timestep > 0) {
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
    
    {
        SVPFParams guide_params = *params;
        if (state->use_adaptive_mu) {
            guide_params.mu = effective_mu;
        }
        svpf_ekf_update(state, y_t, &guide_params);
        
        svpf_apply_guide_preserving_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, opt->d_h_mean_prev, state->guide_mean, current_guide_strength, n
        );
    }
    
    // =========================================================================
    // BANDWIDTH
    // =========================================================================
    svpf_fused_bandwidth_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_y_single, opt->d_bandwidth, opt->d_bandwidth_sq,
        state->d_return_ema, state->d_return_var, 1, 0.3f, 0.05f, n
    );
    
    // =========================================================================
    // ADAPTIVE ANNEALING STEIN ITERATIONS
    // =========================================================================
    
    int total_steps = 0;
    
    float base_step = SVPF_STEIN_STEP_SIZE * (state->use_guide ? 0.5f : 1.0f);
    float temp = state->use_svld ? state->temperature : 0.0f;
    
    // -----------------------------------------------------------------
    // 1. Compute initial gradient and stats at beta=0 (FUSED)
    // -----------------------------------------------------------------
    
    cudaMemsetAsync(opt->d_anneal_stats, 0, 4 * sizeof(float), cs);
    
    svpf_fused_gradient_stats_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
        state->h, state->h_prev, state->grad_log_p, state->log_weights,
        opt->d_precond_grad, opt->d_inv_hessian,
        opt->d_y_single, 
        opt->d_anneal_stats,
        1, params->rho, effective_sigma_z, effective_mu,
        0.0f, state->nu, student_t_const, state->lik_offset,
        params->gamma, state->use_exact_gradient, state->use_newton,
        state->use_fan_mode,
        state->use_student_t_state, state->nu_state,
        n
    );
    
    // SINGLE D2H sync for the entire annealing
    cudaMemcpyAsync(opt->h_anneal_stats_pinned, opt->d_anneal_stats,
                    4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaStreamSynchronize(cs);
    
    // Convert raw sums to stats (CPU side)
    float inv_n = 1.0f / (float)n;
    
    float sum_ll_diff = opt->h_anneal_stats_pinned[0];
    float sum_ll_diff_sq = opt->h_anneal_stats_pinned[1];
    float sum_grad = opt->h_anneal_stats_pinned[2];
    float sum_h_diff_sq = opt->h_anneal_stats_pinned[3];
    
    float mean_ll_diff = sum_ll_diff * inv_n;
    float mean_ll_diff_sq = sum_ll_diff_sq * inv_n;
    float var_ll = mean_ll_diff_sq - (mean_ll_diff * mean_ll_diff);
    var_ll = fmaxf(var_ll, 1e-6f);
    
    float mean_grad = sum_grad * inv_n;
    float h_std = sqrtf(fmaxf(sum_h_diff_sq * inv_n, 1e-8f));
    
    // -----------------------------------------------------------------
    // 2. Compute delta_beta and number of stages UPFRONT
    // -----------------------------------------------------------------
    float d_beta_kl = sqrtf(2.0f * state->anneal_kl_threshold / (var_ll + 1e-6f));
    float d_beta_grad = 2.0f / (mean_grad + 1e-6f);
    float d_beta_spatial = (h_std < 0.05f) ? 0.02f : 0.25f;
    
    float d_beta = fminf(d_beta_kl, fminf(d_beta_grad, d_beta_spatial));
    d_beta = fmaxf(d_beta, 0.05f);
    d_beta = fminf(d_beta, 0.35f);
    
    int n_stages = (int)ceilf(1.0f / d_beta);
    n_stages = max(2, min(n_stages, state->anneal_max_stages));
    
    // -----------------------------------------------------------------
    // 3. Run all stages WITHOUT any more syncs
    // -----------------------------------------------------------------
    for (int stage = 0; stage < n_stages; stage++) {
        float beta = fminf((float)(stage + 1) / (float)n_stages, 1.0f);
        float beta_factor = sqrtf(beta);
        
        for (int s = 0; s < state->anneal_steps_per_beta; s++) {
            total_steps++;
            bool is_last_iteration = (stage == n_stages - 1) && (s == state->anneal_steps_per_beta - 1);
            
            // Gradient
            svpf_fused_gradient_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
                state->h, state->h_prev, state->grad_log_p, state->log_weights,
                opt->d_precond_grad, opt->d_inv_hessian,
                opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu,
                beta, state->nu, student_t_const, state->lik_offset,
                params->gamma, state->use_exact_gradient, state->use_newton,
                state->use_fan_mode,
                state->use_student_t_state, state->nu_state,
                n
            );
            
            // Stein transport (Full Newton, +KSD on last iteration)
            if (is_last_iteration) {
                svpf_fused_stein_transport_full_newton_ksd_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                    state->h, state->grad_log_p, opt->d_inv_hessian,
                    state->d_grad_v, state->rng_states, opt->d_bandwidth,
                    opt->d_ksd_partial,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                    state->stein_repulsive_sign, n
                );
                
                svpf_ksd_reduce_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
                    opt->d_ksd_partial, opt->d_ksd, n
                );
            } else {
                svpf_fused_stein_transport_full_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                    state->h, state->grad_log_p, opt->d_inv_hessian,
                    state->d_grad_v, state->rng_states, opt->d_bandwidth,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                    state->stein_repulsive_sign, n
                );
            }
        }
    }
    
    // Store diagnostics
    state->anneal_stages_used = n_stages;
    state->anneal_final_var_ll = var_ll;
    state->anneal_final_h_std = h_std;
    state->stein_steps_used = total_steps;
    
    // =========================================================================
    // PARTIAL REJUVENATION (Maken et al. 2022)
    // =========================================================================
    if (state->use_rejuvenation && state->timestep > 10) {
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
    // OUTPUTS (consolidated D2H transfer)
    // =========================================================================
    svpf_fused_outputs_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, state->log_weights,
        opt->d_bandwidth, opt->d_ksd,
        opt->d_loglik_single, opt->d_vol_single, opt->d_h_mean_prev,
        (float*)opt->d_output_pack,
        0, n
    );
    
    cudaMemcpyAsync(opt->h_output_pinned, opt->d_output_pack, 
                    5 * sizeof(float), cudaMemcpyDeviceToHost, cs);
    
    // Store params needed for post-sync processing
    opt->pending_y_t = y_t;
    opt->pending_params = (const void*)params;
}

// =============================================================================
// SYNC AND FINALIZE: Wait for GPU, read outputs, update state
// =============================================================================

void svpf_sync_outputs(SVPFState* state, 
                       float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    SVPFOptimizedState* opt = get_opt(state);
    cudaStream_t cs = state->stream;
    
    cudaStreamSynchronize(cs);
    
    float* results = opt->h_output_pinned;
    float h_mean_local = results[2];
    float bandwidth_local = results[3];
    float vol_local = results[1];
    float ksd_local = results[4];
    
    // Backward smoothing
    float h_var_est = bandwidth_local * bandwidth_local;
    const SVPFParams* params = (const SVPFParams*)opt->pending_params;
    svpf_smooth_backward(state, h_mean_local, h_var_est, opt->pending_y_t, params);
    
    float h_mean_output = svpf_get_smoothed_output(state, h_mean_local);
    
    if (h_loglik_out) *h_loglik_out = results[0];
    if (h_vol_out) *h_vol_out = vol_local;
    if (h_mean_out) *h_mean_out = h_mean_output;
    
    state->vol_prev = vol_local;
    state->ksd_prev = ksd_local;
    
    if (state->use_adaptive_mu && state->timestep > 10) {
        svpf_adaptive_mu_update(state, h_mean_local, bandwidth_local);
    }
    
    state->timestep++;
}

// =============================================================================
// SYNCHRONOUS STEP (Original API - calls async + sync)
// =============================================================================

void svpf_step_graph(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                     float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    svpf_step_async(state, y_t, y_prev, params);
    svpf_sync_outputs(state, h_loglik_out, h_vol_out, h_mean_out);
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

void svpf_get_ksd_stats(const SVPFState* state, float* ksd_out, int* steps_used_out) {
    if (ksd_out) *ksd_out = state->ksd_prev;
    if (steps_used_out) *steps_used_out = state->stein_steps_used;
}
