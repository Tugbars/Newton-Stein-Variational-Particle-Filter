/**
 * @file svpf_optimized_graph.cu
 * @brief SVPF Implementation — Production Configuration
 *
 * Hardened for the production kernel set:
 *   Full Newton | Adaptive Annealing | Student-t state | Antithetic sampling
 *   EKF guide (variance-preserving) | Adaptive mu/sigma | Smoothing
 *
 * Removed dead code paths:
 *   - Non-Newton / non-full-Newton Stein variants
 *   - Fixed annealing (legacy beta schedules)
 *   - Heun's method (2nd order integrator)
 *   - Fan mode (weightless SVGD)
 *   - Non-exact gradient
 *   - Gaussian state dynamics (always Student-t now)
 *   - Non-antithetic predict
 *   - Non-preserving guide
 *   - CUDA graph capture/replay (was unused)
 *   - Configurable stein_sign_mode (hardcoded legacy=0)
 */

#include "svpf_kernels.cuh"
#include "svpf_fused_gradient_stats.cuh"

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

#ifndef SVPF_SMOOTH_MAX_LAG
#define SVPF_SMOOTH_MAX_LAG 8
#endif

// Forward declarations
static void svpf_optimized_init(SVPFOptimizedState* opt, int n);
static void svpf_optimized_cleanup(SVPFOptimizedState* opt);

// =============================================================================
// STATE MANAGEMENT: Create
// =============================================================================

SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream) {
    SVPFState* state = (SVPFState*)calloc(1, sizeof(SVPFState));
    if (!state) return NULL;

    state->n_particles = n_particles;
    state->n_stein_steps = n_stein_steps;
    state->nu = nu;
    state->stream = stream ? stream : 0;
    state->timestep = 0;
    state->y_prev = 0.0f;

    state->student_t_const = lgammaf((nu + 1.0f) / 2.0f)
                           - lgammaf(nu / 2.0f)
                           - 0.5f * logf((float)M_PI * nu);

    // Precompute implied offset for Student-t observation model
    {
        const float psi_half = -1.9635100260214235f;
        float nu_half = nu / 2.0f;
        float psi_nu_half;
        if (nu_half >= 1.0f) {
            psi_nu_half = logf(nu_half) - 1.0f / (2.0f * nu_half)
                        - 1.0f / (12.0f * nu_half * nu_half);
        } else {
            psi_nu_half = -0.5772156649f - 1.0f / nu_half;
        }
        state->student_t_implied_offset = -(logf(nu) + psi_half - psi_nu_half);
    }

    int n = n_particles;

    // Particle arrays
    cudaMalloc(&state->h, n * sizeof(float));
    cudaMalloc(&state->h_prev, n * sizeof(float));
    cudaMalloc(&state->grad_log_p, n * sizeof(float));
    cudaMalloc(&state->log_weights, n * sizeof(float));
    cudaMalloc(&state->rng_states, n * sizeof(curandStatePhilox4_32_10_t));

    cudaMalloc(&state->d_grad_v, n * sizeof(float));
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));

    cudaMalloc(&state->d_return_ema, sizeof(float));
    cudaMalloc(&state->d_return_var, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(state->d_return_ema, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_return_var, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // --- Fixed production configuration ---
    state->lik_offset = 0.345f;
    state->temperature = 0.45f;
    state->rmsprop_rho = 0.7f;
    state->rmsprop_eps = 1e-6f;

    // MIM (off by default, but jump params stored for predict kernel)
    state->mim_jump_prob = 0.25f;
    state->mim_jump_scale = 9.0f;

    // Asymmetric rho
    state->rho_up = 0.98f;
    state->rho_down = 0.93f;

    // Guide
    state->guide_strength_base = 0.05f;
    state->guide_strength_max = 0.30f;
    state->guide_innovation_threshold = 1.0f;
    state->guide_mean = 0.0f;
    state->guide_var = 0.0f;
    state->guide_initialized = 0;
    state->vol_prev = 0.05f;

    // Guided prediction
    state->guided_alpha_base = 0.0f;
    state->guided_alpha_shock = 0.40f;
    state->guided_innovation_threshold = 1.5f;

    // Local params
    state->delta_rho = 0.02f;
    state->delta_sigma = 0.1f;

    // Adaptive mu (Kalman)
    state->mu_state = -3.5f;
    state->mu_var = 1.0f;
    state->mu_process_var = 0.001f;
    state->mu_obs_var_scale = 11.0f;
    state->mu_min = -4.0f;
    state->mu_max = -1.0f;

    // Adaptive sigma
    state->sigma_boost_threshold = 0.95f;
    state->sigma_boost_max = 3.2f;
    state->sigma_z_effective = 0.10f;

    // Student-t state dynamics
    state->nu_state = 5.0f;

    // Rejuvenation (Maken 2022)
    state->rejuv_ksd_threshold = 0.05f;
    state->rejuv_prob = 0.30f;
    state->rejuv_blend = 0.30f;

    // Adaptive annealing
    state->anneal_kl_threshold = 0.9f;
    state->anneal_steps_per_beta = 3;
    state->anneal_max_stages = 50;
    state->anneal_stages_used = 0;
    state->anneal_final_var_ll = 0.0f;
    state->anneal_final_h_std = 0.0f;

    // KSD tracking
    state->ksd_prev = 1e10f;
    state->stein_steps_used = n_stein_steps;

    // Smoothing (RTS backward)
    state->smooth_lag = 3;
    state->smooth_output_lag = 1;
    for (int i = 0; i < SVPF_SMOOTH_MAX_LAG; i++) {
        state->smooth_h_mean[i] = 0.0f;
        state->smooth_h_var[i] = 1.0f;
        state->smooth_y[i] = 0.0f;
    }
    state->smooth_head = 0;

    // Device scalars for output
    cudaMalloc(&state->d_result_h_mean, sizeof(float));

    return state;
}

// =============================================================================
// STATE MANAGEMENT: Destroy
// =============================================================================

void svpf_destroy(SVPFState* state) {
    if (!state) return;

    svpf_optimized_cleanup(&state->opt_backend);

    cudaFree(state->h);
    cudaFree(state->h_prev);
    cudaFree(state->grad_log_p);
    cudaFree(state->log_weights);
    cudaFree(state->rng_states);
    cudaFree(state->d_grad_v);
    cudaFree(state->d_return_ema);
    cudaFree(state->d_return_var);
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

    svpf_init_rng_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->rng_states, n, seed
    );

    // Clamp nu_state for finite variance (requires nu > 2)
    state->nu_state = fmaxf(state->nu_state, 2.5f);

    // Stationary variance: Student-t AR(1)
    float base_var = (params->sigma_z * params->sigma_z)
                   / (1.0f - params->rho * params->rho + 1e-6f);
    float var_scale = state->nu_state / (state->nu_state - 2.0f);
    float stationary_std = sqrtf(var_scale * base_var);

    svpf_init_particles_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->rng_states, params->mu, stationary_std, n
    );

    svpf_copy_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->h_prev, n
    );

    state->guide_initialized = 0;
    state->guide_mean = params->mu;
    state->guide_var = var_scale * base_var;

    state->mu_state = params->mu;
    state->mu_var = 1.0f;
    state->sigma_z_effective = params->sigma_z;

    state->ksd_prev = 1e10f;
    state->stein_steps_used = state->n_stein_steps;

    cudaMemset(state->d_grad_v, 0, n * sizeof(float));
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// OPTIMIZED BACKEND: Init / Cleanup
// =============================================================================

static void svpf_optimized_init(SVPFOptimizedState* opt, int n) {
    if (opt->initialized && n <= opt->allocated_n) return;
    if (opt->initialized) svpf_optimized_cleanup(opt);

    cudaMalloc(&opt->d_bandwidth, sizeof(float));
    cudaMalloc(&opt->d_bandwidth_sq, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(opt->d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&opt->d_precond_grad, n * sizeof(float));
    cudaMalloc(&opt->d_inv_hessian, n * sizeof(float));

    cudaMalloc(&opt->d_h_mean_prev, sizeof(float));
    float init_h = -3.5f;
    cudaMemcpy(opt->d_h_mean_prev, &init_h, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&opt->d_y_single, 2 * sizeof(float));
    cudaMalloc(&opt->d_loglik_single, sizeof(float));
    cudaMalloc(&opt->d_vol_single, sizeof(float));

    // KSD buffers
    cudaMalloc(&opt->d_ksd_partial, n * sizeof(float));
    cudaMalloc(&opt->d_ksd, sizeof(float));

    // Consolidated output pack (single D2H)
    cudaMalloc(&opt->d_output_pack, 8 * sizeof(float));
    cudaMallocHost(&opt->h_output_pinned, 8 * sizeof(float));

    // Adaptive annealing buffers
    cudaMalloc(&opt->d_anneal_stats, 4 * sizeof(float));
    cudaMallocHost(&opt->h_anneal_stats_pinned, 4 * sizeof(float));

    opt->allocated_n = n;
    opt->initialized = true;
}

static void svpf_optimized_cleanup(SVPFOptimizedState* opt) {
    if (!opt->initialized) return;

    cudaFree(opt->d_bandwidth);
    cudaFree(opt->d_bandwidth_sq);
    cudaFree(opt->d_precond_grad);
    cudaFree(opt->d_inv_hessian);
    cudaFree(opt->d_h_mean_prev);
    cudaFree(opt->d_y_single);
    cudaFree(opt->d_loglik_single);
    cudaFree(opt->d_vol_single);
    cudaFree(opt->d_ksd_partial);
    cudaFree(opt->d_ksd);
    cudaFree(opt->d_output_pack);

    if (opt->h_output_pinned) {
        cudaFreeHost(opt->h_output_pinned);
        opt->h_output_pinned = nullptr;
    }

    cudaFree(opt->d_anneal_stats);
    if (opt->h_anneal_stats_pinned) {
        cudaFreeHost(opt->h_anneal_stats_pinned);
        opt->h_anneal_stats_pinned = nullptr;
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

static void svpf_adaptive_mu_update(SVPFState* state, float h_mean, float bandwidth) {
    float P_pred = state->mu_var + state->mu_process_var;
    float R = state->mu_obs_var_scale * bandwidth * bandwidth;
    float K = P_pred / (P_pred + R + 1e-8f);

    float mu_new = state->mu_state + K * (h_mean - state->mu_state);
    state->mu_state = fminf(fmaxf(mu_new, state->mu_min), state->mu_max);
    state->mu_var = (1.0f - K) * P_pred;
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
    int k = state->smooth_lag;
    if (k > SVPF_SMOOTH_MAX_LAG) k = SVPF_SMOOTH_MAX_LAG;
    if (k < 1) k = 1;

    int head = state->smooth_head;
    state->smooth_h_mean[head] = h_mean_new;
    state->smooth_h_var[head] = h_var_new;
    state->smooth_y[head] = y_t;
    state->smooth_head = (head + 1) % k;

    if (state->timestep < k) return;

    float rho = params->rho;
    float mu = state->mu_state;
    float sigma_z_sq = params->sigma_z * params->sigma_z;

    for (int lag = 1; lag < k; lag++) {
        int idx_curr = (state->smooth_head - lag - 1 + k) % k;
        int idx_next = (state->smooth_head - lag + k) % k;

        float h_curr = state->smooth_h_mean[idx_curr];
        float var_curr = state->smooth_h_var[idx_curr];

        float h_pred = mu + rho * (h_curr - mu);
        float pred_var = rho * rho * var_curr + sigma_z_sq;

        float innovation = state->smooth_h_mean[idx_next] - h_pred;
        float J = rho * var_curr / (pred_var + 1e-8f);

        state->smooth_h_mean[idx_curr] = h_curr + J * innovation;
        state->smooth_h_var[idx_curr] = var_curr * (1.0f - J * rho);
    }
}

static float svpf_get_smoothed_output(SVPFState* state, float h_mean_raw) {
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

    // --- Effective parameters ---
    float effective_mu = state->mu_state;
    float effective_sigma_z = params->sigma_z;

    if (state->timestep > 0) {
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;

        if (return_z > state->sigma_boost_threshold) {
            float severity = fminf((return_z - state->sigma_boost_threshold) / 3.0f, 1.0f);
            float sigma_boost = 1.0f + (state->sigma_boost_max - 1.0f) * severity;
            effective_sigma_z = params->sigma_z * sigma_boost;
        }
        state->sigma_z_effective = effective_sigma_z;
    }

    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);

    int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t grad_smem = 2 * n * sizeof(float);
    size_t stein_smem = 3 * n * sizeof(float);  // Full Newton always needs 3×n

    // Upload y values
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, cs);

    // =========================================================================
    // 1. PREDICT (antithetic, guided, Student-t)
    // =========================================================================
    int nb_half = ((n / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    svpf_predict_guided_antithetic_kernel<<<nb_half, BLOCK_SIZE, 0, cs>>>(
        state->h, state->h_prev, state->rng_states,
        opt->d_y_single, opt->d_h_mean_prev, 1,
        state->rho_up, state->rho_down,
        effective_sigma_z, effective_mu, params->gamma,
        state->mim_jump_prob, state->mim_jump_scale,
        state->delta_rho, state->delta_sigma,
        state->guided_alpha_base, state->guided_alpha_shock,
        state->guided_innovation_threshold,
        state->student_t_implied_offset,
        state->nu_state,
        n
    );

    // =========================================================================
    // 2. EKF GUIDE (adaptive strength, variance-preserving)
    // =========================================================================
    float current_guide_strength = state->guide_strength_base;

    if (state->timestep > 0) {
        float vol_est = fmaxf(state->vol_prev, 1e-4f);
        float return_z = fabsf(y_t) / vol_est;

        float implied_h = logf(y_t * y_t + 1e-8f) + state->student_t_implied_offset;
        float h_est = logf(vol_est * vol_est + 1e-8f);
        float h_innovation = implied_h - h_est;

        if (h_innovation > 0.0f && return_z > state->guide_innovation_threshold) {
            float severity = fminf((return_z - state->guide_innovation_threshold) / 3.0f, 1.0f);
            current_guide_strength = state->guide_strength_base
                + (state->guide_strength_max - state->guide_strength_base) * severity;
        }
    }

    SVPFParams guide_params = *params;
    guide_params.mu = effective_mu;
    svpf_ekf_update(state, y_t, &guide_params);

    svpf_apply_guide_preserving_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_h_mean_prev, state->guide_mean, current_guide_strength, n
    );

    // =========================================================================
    // 3. BANDWIDTH
    // =========================================================================
    svpf_fused_bandwidth_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_y_single, opt->d_bandwidth, opt->d_bandwidth_sq,
        state->d_return_ema, state->d_return_var, 1, 0.3f, 0.05f, n
    );

    // =========================================================================
    // 4. ADAPTIVE ANNEALING + FULL NEWTON STEIN ITERATIONS
    // =========================================================================
    //
    // Single-sync strategy:
    //   1. Fused gradient+stats at beta=0 → one D2H sync
    //   2. Compute all betas upfront on CPU
    //   3. Run all stages with zero further syncs

    float base_step = SVPF_STEIN_STEP_SIZE * 0.5f;  // Halved because guide is always on
    float temp = state->temperature;
    int total_steps = 0;

    // --- 4a. Initial gradient + stats at beta=0 ---
    cudaMemsetAsync(opt->d_anneal_stats, 0, 4 * sizeof(float), cs);

    svpf_fused_gradient_stats_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
        state->h, state->h_prev, state->grad_log_p, state->log_weights,
        opt->d_precond_grad, opt->d_inv_hessian,
        opt->d_y_single,
        opt->d_anneal_stats,
        1, params->rho, effective_sigma_z, effective_mu,
        0.0f, state->nu, student_t_const, state->lik_offset,
        params->gamma,
        /*use_exact_gradient=*/true, /*use_newton=*/true, /*use_fan_mode=*/false,
        /*use_student_t_state=*/1, state->nu_state,
        n
    );

    // SINGLE D2H sync for entire annealing schedule
    cudaMemcpyAsync(opt->h_anneal_stats_pinned, opt->d_anneal_stats,
                    4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
    cudaStreamSynchronize(cs);

    // --- 4b. Compute annealing schedule on CPU ---
    float inv_n = 1.0f / (float)n;

    float mean_ll_diff = opt->h_anneal_stats_pinned[0] * inv_n;
    float mean_ll_diff_sq = opt->h_anneal_stats_pinned[1] * inv_n;
    float var_ll = fmaxf(mean_ll_diff_sq - mean_ll_diff * mean_ll_diff, 1e-6f);
    float mean_grad = opt->h_anneal_stats_pinned[2] * inv_n;
    float h_std = sqrtf(fmaxf(opt->h_anneal_stats_pinned[3] * inv_n, 1e-8f));

    float d_beta_kl = sqrtf(2.0f * state->anneal_kl_threshold / (var_ll + 1e-6f));
    float d_beta_grad = 2.0f / (mean_grad + 1e-6f);
    float d_beta_spatial = (h_std < 0.05f) ? 0.02f : 0.25f;

    float d_beta = fminf(d_beta_kl, fminf(d_beta_grad, d_beta_spatial));
    d_beta = fmaxf(d_beta, 0.05f);
    d_beta = fminf(d_beta, 0.35f);

    int n_stages = (int)ceilf(1.0f / d_beta);
    n_stages = max(2, min(n_stages, state->anneal_max_stages));

    // --- 4c. Run all stages (zero syncs) ---
    for (int stage = 0; stage < n_stages; stage++) {
        float beta = fminf((float)(stage + 1) / (float)n_stages, 1.0f);
        float beta_factor = sqrtf(beta);

        for (int s = 0; s < state->anneal_steps_per_beta; s++) {
            total_steps++;
            bool is_last = (stage == n_stages - 1) && (s == state->anneal_steps_per_beta - 1);

            // Gradient (full Newton: always writes precond_grad + inv_hessian)
            svpf_fused_gradient_kernel<<<nb, BLOCK_SIZE, grad_smem, cs>>>(
                state->h, state->h_prev, state->grad_log_p, state->log_weights,
                opt->d_precond_grad, opt->d_inv_hessian,
                opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu,
                beta, state->nu, student_t_const, state->lik_offset,
                params->gamma, state->nu_state,
                n
            );

            // Full Newton Stein transport (last iteration includes KSD)
            if (is_last) {
                svpf_fused_stein_transport_full_newton_ksd_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                    state->h, state->grad_log_p, opt->d_inv_hessian,
                    state->d_grad_v, state->rng_states, opt->d_bandwidth,
                    opt->d_ksd_partial,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                    n
                );
                svpf_ksd_reduce_kernel<<<1, BLOCK_SIZE, 0, cs>>>(
                    opt->d_ksd_partial, opt->d_ksd, n
                );
            } else {
                svpf_fused_stein_transport_full_newton_kernel<<<nb, BLOCK_SIZE, stein_smem, cs>>>(
                    state->h, state->grad_log_p, opt->d_inv_hessian,
                    state->d_grad_v, state->rng_states, opt->d_bandwidth,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                    n
                );
            }
        }
    }

    state->stein_steps_used = total_steps;
    state->anneal_stages_used = n_stages;
    state->anneal_final_var_ll = var_ll;
    state->anneal_final_h_std = h_std;

    // =========================================================================
    // 5. PARTIAL REJUVENATION (when KSD is high)
    // =========================================================================
    if (state->timestep > 10 && state->ksd_prev > state->rejuv_ksd_threshold) {
        float guide_std = sqrtf(fmaxf(state->guide_var, 1e-6f));
        svpf_partial_rejuvenation_kernel<<<nb, BLOCK_SIZE, 0, cs>>>(
            state->h, state->guide_mean, guide_std,
            state->rejuv_prob, state->rejuv_blend,
            state->rng_states, n
        );
    }

    // =========================================================================
    // 6. OUTPUTS (consolidated D2H)
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

    opt->pending_y_t = y_t;
    opt->pending_params = (const void*)params;
}

// =============================================================================
// SYNC AND FINALIZE
// =============================================================================

void svpf_sync_outputs(SVPFState* state,
                       float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    SVPFOptimizedState* opt = get_opt(state);
    cudaStreamSynchronize(state->stream);

    float* r = opt->h_output_pinned;
    float h_mean_local = r[2];
    float bandwidth_local = r[3];
    float vol_local = r[1];
    float ksd_local = r[4];

    // Backward smoothing
    const SVPFParams* params = (const SVPFParams*)opt->pending_params;
    svpf_smooth_backward(state, h_mean_local, bandwidth_local * bandwidth_local,
                         opt->pending_y_t, params);
    float h_mean_output = svpf_get_smoothed_output(state, h_mean_local);

    if (h_loglik_out) *h_loglik_out = r[0];
    if (h_vol_out) *h_vol_out = vol_local;
    if (h_mean_out) *h_mean_out = h_mean_output;

    state->vol_prev = vol_local;
    state->ksd_prev = ksd_local;

    if (state->timestep > 10) {
        svpf_adaptive_mu_update(state, h_mean_local, bandwidth_local);
    }

    state->timestep++;
}

// =============================================================================
// SYNCHRONOUS API
// =============================================================================

void svpf_step_graph(SVPFState* state, float y_t, float y_prev, const SVPFParams* params,
                     float* h_loglik_out, float* h_vol_out, float* h_mean_out) {
    svpf_step_async(state, y_t, y_prev, params);
    svpf_sync_outputs(state, h_loglik_out, h_vol_out, h_mean_out);
}

void svpf_step(SVPFState* state, float y_t, const SVPFParams* params, SVPFResult* result) {
    float loglik, vol, h_mean;
    svpf_step_graph(state, y_t, state->y_prev, params, &loglik, &vol, &h_mean);
    if (result) {
        result->log_lik_increment = loglik;
        result->vol_mean = vol;
        result->h_mean = h_mean;
        result->vol_std = 0.0f;
        result->mu_estimate = state->mu_state;
    }
    state->y_prev = y_t;
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

// =============================================================================
// DIAGNOSTICS
// =============================================================================

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
    for (int i = 0; i < n; i++) {
        float w = expf(lw[i] - mx);
        sw += w;
        sw2 += w * w;
    }

    free(lw);
    return (sw * sw) / (sw2 + 1e-10f);
}

void svpf_get_ksd_stats(const SVPFState* state, float* ksd_out, int* steps_used_out) {
    if (ksd_out) *ksd_out = state->ksd_prev;
    if (steps_used_out) *steps_used_out = state->stein_steps_used;
}
