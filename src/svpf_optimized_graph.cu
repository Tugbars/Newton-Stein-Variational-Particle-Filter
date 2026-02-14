/**
 * @file svpf_optimized_graph_with_decorrelation.cu
 * @brief SVPF with Periodic Variance Injection for Bias Decorrelation
 * 
 * MODIFICATION: Added periodic variance injection to break temporal correlation
 * of temperature-induced noise that causes bias compounding across timesteps.
 * 
 * NEW FEATURES:
 * - use_decorrelation: Enable/disable periodic variance injection
 * - decorrelation_interval: How often to inject (e.g., every 25 timesteps)
 * - decorrelation_scale: Magnitude of injection (fraction of sigma_z)
 * - svpf_decorrelation_kernel: Adds i.i.d. noise to break serial correlation
 * 
 * MECHANISM: Every K timesteps, add independent Gaussian noise to particles.
 * This breaks the chain: noise from t=0..K-1 is independent from t=K..2K-1.
 * Prevents unbounded accumulation of temperature-induced drift.
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

// Forward declarations
void svpf_optimized_init(SVPFOptimizedState* opt, int n);

// =============================================================================
// DECORRELATION KERNEL: Periodic Variance Injection
// =============================================================================

__global__ void svpf_decorrelation_kernel(
    float* __restrict__ h,
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    float decorrelation_scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Add i.i.d. Gaussian noise (independent of particle positions)
    float noise = curand_normal(&rng_states[idx]);
    h[idx] = clamp_logvol(h[idx] + decorrelation_scale * noise);
}

// =============================================================================
// STATE MANAGEMENT: Create
// =============================================================================

SVPFState* svpf_create(int n_particles, int n_stein_steps, float nu, cudaStream_t stream) {
    (void)n_stein_steps;
    
    SVPFState* state = (SVPFState*)malloc(sizeof(SVPFState));
    if (!state) return NULL;
    
    memset(&state->opt_backend, 0, sizeof(SVPFOptimizedState));
    
    state->n_particles = n_particles;
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
    float init_alpha = 0.4f;
    cudaMemcpy(state->d_return_ema, &init_ema, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_return_var, &init_ema, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_bw_alpha, &init_alpha, sizeof(float), cudaMemcpyHostToDevice);
    
    // =========================================================================
    // Production defaults
    // =========================================================================
    
    state->use_exact_gradient = 1;
    state->lik_offset = 0.065f;
    state->use_svld = 1;
    state->temperature = 0.45f;
    state->rmsprop_rho = 0.7f;
    state->rmsprop_eps = 1e-6f;
    state->anneal_n_stages_fixed = 4;
    state->anneal_steps_per_beta = 3;
    state->anneal_stages_used = 0;
    state->use_persistent_kernel = 1;
    state->use_newton = 1;
    state->use_full_newton = 1;
    state->use_mim = 0;
    state->mim_jump_prob = 0.25f;
    state->mim_jump_scale = 8.2f;
    state->use_guided = 0;
    state->guided_alpha_base = 0.0f;
    state->guided_alpha_shock = 0.40f;
    state->guided_innovation_threshold = 1.5f;
    state->use_adaptive_mu = 1;
    state->mu_state = -3.5f;
    state->mu_var = 1.0f;
    state->mu_process_var = 0.001f;
    state->mu_obs_var_scale = 11.0f;
    state->mu_min = -4.0f;
    state->mu_max = -1.0f;
    
    state->use_adaptive_guide = 0;
    state->guide_strength_base = 0.00f;
    state->guide_strength_max = 0.00f;
    state->guide_innovation_threshold = 0.00f;
    state->vol_prev = 0.00f;
    state->use_adaptive_sigma = 0;
    state->sigma_boost_threshold =0.00f; 
    state->sigma_boost_max = 0.00f;
    state->sigma_z_effective = 0.00f;
    state->use_heun = 0;
    state->use_smoothing = 0;
    state->smooth_lag = 0;
    state->smooth_output_lag = 1;
    for (int i = 0; i < SVPF_SMOOTH_MAX_LAG; i++) {
        state->smooth_h_mean[i] = 0.0f;
        state->smooth_h_var[i] = 0.0f;
        state->smooth_y[i] = 0.0f;
    }
    state->smooth_head = 0;
    state->use_student_t_state = 0;
    state->nu_state = 0.0f;

    state->stein_repulsive_sign = SVPF_STEIN_SIGN_NONE;
    state->use_fan_mode = 1;

    state->ksd_prev = 1e10f;
    state->stein_steps_used = 0;
    
    state->use_antithetic = 1;
    
    // =========================================================================
    // NEW: Decorrelation (Bias Mitigation)
    // =========================================================================
    state->use_decorrelation = 0;           // Enable by default
    state->decorrelation_interval = 25;     // Every 25 timesteps
    state->decorrelation_scale = 0.15f;     // 15% of sigma_z
    
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
    
    float nu_state_clamped = fmaxf(state->nu_state, 2.5f);
    state->nu_state = nu_state_clamped;
    
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
    
    if (state->use_adaptive_mu) {
        state->mu_state = params->mu;
        state->mu_var = 1.0f;
    }
    
    if (state->use_adaptive_sigma) {
        state->sigma_z_effective = params->sigma_z;
    }
    
    state->ksd_prev = 1e10f;
    state->stein_steps_used = 0;
    
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));
    
    svpf_graph_invalidate(state);
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// OPTIMIZED BACKEND
// =============================================================================

void svpf_optimized_init(SVPFOptimizedState* opt, int n) {
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
    
    cudaMalloc(&opt->d_y_single, 2 * sizeof(float));
    cudaMalloc(&opt->d_loglik_single, sizeof(float));
    cudaMalloc(&opt->d_vol_single, sizeof(float));
    
    cudaMalloc(&opt->d_params_staging, SVPF_GRAPH_PARAMS_SIZE * sizeof(float));
    
    cudaMalloc(&opt->d_ksd_partial, n * sizeof(float));
    cudaMalloc(&opt->d_ksd, sizeof(float));
    
    cudaMalloc(&opt->d_output_pack, 8 * sizeof(float));
    cudaMallocHost(&opt->h_output_pinned, 8 * sizeof(float));
    
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    cudaMallocHost(&opt->h_results_pinned, 4 * sizeof(float));
    
    cudaMalloc(&opt->d_anneal_stats, 4 * sizeof(float));
    cudaMallocHost(&opt->h_anneal_stats_pinned, 4 * sizeof(float));
    
    opt->allocated_n = n;
    opt->stein_block_size = svpf_optimal_stein_block_size(n);
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
    cudaFree(opt->d_y_single);
    cudaFree(opt->d_loglik_single);
    cudaFree(opt->d_vol_single);
    cudaFree(opt->d_params_staging);
    
    cudaFree(opt->d_ksd_partial);
    cudaFree(opt->d_ksd);
    
    cudaFree(opt->d_output_pack);
    if (opt->h_output_pinned) {
        cudaFreeHost(opt->h_output_pinned);
        opt->h_output_pinned = nullptr;
    }
    
    if (opt->h_results_pinned) {
        cudaFreeHost(opt->h_results_pinned);
        opt->h_results_pinned = nullptr;
    }
    
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
// ASYNC STEP: Launch all GPU work, return immediately
// =============================================================================

void svpf_step_async(SVPFState* state, float y_t, float y_prev, const SVPFParams* params) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t cs = state->stream;
    
    svpf_optimized_init(opt, n);
    
    // --- Effective parameters ---
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
    
    int nb = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    int sbs = opt->stein_block_size;
    int nb_stein = (n + sbs - 1) / sbs;
    size_t grad_smem = 2 * n * sizeof(float);
    size_t stein_smem = 3 * n * sizeof(float);
    
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, cs);
    
    // =========================================================================
    // PREDICT (Antithetic guided)
    // =========================================================================
    {
        int nb_half = ((n / 2) + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
        svpf_predict_guided_antithetic_kernel<<<nb_half, SVPF_BLOCK_SIZE, 0, cs>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1,
            params->rho,
            effective_sigma_z, effective_mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            state->guided_alpha_base, state->guided_alpha_shock,
            state->guided_innovation_threshold,
            state->student_t_implied_offset,
            state->use_student_t_state, state->nu_state,
            n
        );
    }

    // =========================================================================
    // BANDWIDTH
    // =========================================================================
    svpf_fused_bandwidth_kernel<<<1, SVPF_BLOCK_SIZE, 0, cs>>>(
        state->h, opt->d_y_single, opt->d_bandwidth, opt->d_bandwidth_sq,
        state->d_return_ema, state->d_return_var, 1, 0.3f, 0.05f, n
    );
    
    // =========================================================================
    // STEIN ITERATIONS
    // =========================================================================
    
    int n_stages = state->anneal_n_stages_fixed;
    int steps_per_beta = state->anneal_steps_per_beta;
    int total_steps = 0;
    
    float base_step = SVPF_STEIN_STEP_SIZE;
    float temp = state->use_svld ? state->temperature : 0.0f;
    
    for (int stage = 0; stage < n_stages; stage++) {
        float beta = (float)(stage + 1) / (float)n_stages;
        float beta_factor = sqrtf(beta);
        
        for (int s = 0; s < steps_per_beta; s++) {
            total_steps++;
            bool is_last = (stage == n_stages - 1) && (s == steps_per_beta - 1);
            
            svpf_fused_gradient_kernel<<<nb_stein, sbs, grad_smem, cs>>>(
                state->h, state->h_prev, state->grad_log_p, state->log_weights,
                opt->d_precond_grad, opt->d_inv_hessian,
                opt->d_y_single, 1, params->rho, effective_sigma_z, effective_mu,
                state->nu, state->lik_offset,
                params->gamma, state->use_exact_gradient, state->use_newton,
                state->use_student_t_state, state->nu_state,
                n
            );
            
            if (is_last) {
                svpf_fused_stein_transport_full_newton_ksd_kernel<<<nb_stein, sbs, stein_smem, cs>>>(
                    state->h, state->grad_log_p, opt->d_inv_hessian,
                    state->d_grad_v, state->rng_states, opt->d_bandwidth,
                    opt->d_ksd_partial,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                    state->stein_repulsive_sign, state->use_split_batch, n
                );
                
                svpf_ksd_reduce_kernel<<<1, SVPF_BLOCK_SIZE, 0, cs>>>(
                    opt->d_ksd_partial, opt->d_ksd, n
                );
            } else {
                svpf_fused_stein_transport_full_newton_kernel<<<nb_stein, sbs, stein_smem, cs>>>(
                    state->h, state->grad_log_p, opt->d_inv_hessian,
                    state->d_grad_v, state->rng_states, opt->d_bandwidth,
                    base_step, beta_factor, temp, state->rmsprop_rho, state->rmsprop_eps,
                    state->stein_repulsive_sign, state->use_split_batch, n
                );
            }
        }
    }
    
    state->anneal_stages_used = n_stages;
    state->stein_steps_used = total_steps;
    
    // =========================================================================
    // NEW: PERIODIC DECORRELATION (Bias Mitigation)
    // =========================================================================
    if (state->use_decorrelation && state->decorrelation_interval > 0) {
        if ((state->timestep > 0) && (state->timestep % state->decorrelation_interval == 0)) {
            // Inject independent variance to break temporal correlation
            float decorr_noise_scale = state->decorrelation_scale * effective_sigma_z;
            
            svpf_decorrelation_kernel<<<nb, SVPF_BLOCK_SIZE, 0, cs>>>(
                state->h, state->rng_states, decorr_noise_scale, n
            );
        }
    }
    
    // =========================================================================
    // OUTPUTS
    // =========================================================================
    svpf_fused_outputs_kernel<<<1, SVPF_BLOCK_SIZE, 0, cs>>>(
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
    cudaStream_t cs = state->stream;
    
    cudaStreamSynchronize(cs);
    
    float* results = opt->h_output_pinned;
    float h_mean_local = results[2];
    float bandwidth_local = results[3];
    float vol_local = results[1];
    float ksd_local = results[4];
    
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
// SYNCHRONOUS STEP (Original API)
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
