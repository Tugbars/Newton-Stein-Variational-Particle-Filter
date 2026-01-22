/**
 * @file svpf_optimized_graph.cu
 * @brief CUDA Graph-accelerated SVPF implementation
 * 
 * Uses fused kernels for reduced launch overhead:
 * - svpf_fused_gradient_kernel (replaces 4 kernels)
 * - svpf_fused_stein_transport_kernel (replaces 3 kernels)
 * - svpf_fused_bandwidth_kernel (replaces 2 kernels)
 * - svpf_fused_outputs_kernel (replaces 3 kernels)
 */

#include "svpf_kernels.cuh"

// =============================================================================
// Forward declarations for fused kernels (defined in svpf_opt_kernels.cu)
// =============================================================================

__global__ void svpf_fused_gradient_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_combined,
    float* __restrict__ log_w,
    float* __restrict__ precond_grad,
    float* __restrict__ inv_hessian,
    const float* __restrict__ d_y,
    int y_idx,
    float rho, float sigma_z, float mu,
    float beta, float nu, float student_t_const,
    bool use_newton, int n
);

__global__ void svpf_fused_stein_transport_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int n
);

__global__ void svpf_fused_stein_transport_newton_kernel(
    float* __restrict__ h,
    const float* __restrict__ precond_grad,
    const float* __restrict__ inv_hessian,
    float* __restrict__ v_rmsprop,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bandwidth,
    float step_size, float beta_factor, float temperature,
    float rho_rmsprop, float epsilon, int n
);

__global__ void svpf_fused_outputs_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    int t_out, int n
);

__global__ void svpf_fused_bandwidth_kernel(
    const float* __restrict__ h,
    const float* __restrict__ d_y,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_bandwidth_sq,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    int y_idx, float alpha_bw, float alpha_ret, int n
);

// =============================================================================
// Helper
// =============================================================================

static inline SVPFOptimizedState* get_opt(SVPFState* state) {
    return &state->opt_backend;
}

// Defined in svpf_optimized.cu
extern void svpf_optimized_init(SVPFOptimizedState* opt, int n);

// =============================================================================
// Graph Capture
// =============================================================================

static void svpf_graph_capture_internal(
    SVPFState* state,
    const SVPFParams* params
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t capture_stream = opt->graph_stream;
    
    // Precompute constants
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Shared memory sizes for fused kernels
    size_t grad_smem = 2 * n * sizeof(float);   // h_prev + mu_i
    size_t stein_smem = state->use_newton 
                      ? 3 * n * sizeof(float)   // h + precond_grad + inv_hess
                      : 2 * n * sizeof(float);  // h + grad
    
    // Annealing
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
    
    // Step size
    float base_step = SVPF_STEIN_STEP_SIZE;
    if (state->use_guide) base_step *= 0.5f;
    
    // Asymmetric rho
    float rho_up = state->use_asymmetric_rho ? state->rho_up : params->rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : params->rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    // =========================================================================
    // BEGIN CAPTURE
    // =========================================================================
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    
    // -------------------------------------------------------------------------
    // PREDICT
    // -------------------------------------------------------------------------
    if (state->use_guided) {
        svpf_predict_guided_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1, rho_up, rho_down,
            params->sigma_z, params->mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            state->guided_alpha_base,
            state->guided_alpha_shock,
            state->guided_innovation_threshold,
            n
        );
    } else if (state->use_mim) {
        svpf_predict_mim_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1, rho_up, rho_down,
            params->sigma_z, params->mu, params->gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma, n
        );
    } else {
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, params->rho, params->sigma_z, params->mu, params->gamma, n
        );
    }
    
    // -------------------------------------------------------------------------
    // EKF GUIDE (if enabled)
    // -------------------------------------------------------------------------
    if (state->use_guide) {
        if (state->use_guide_preserving) {
            svpf_apply_guide_preserving_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, opt->d_h_mean_prev, opt->d_guide_mean, state->guide_strength, n
            );
        } else {
            svpf_apply_guide_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, opt->d_guide_mean, state->guide_strength, n
            );
        }
    }
    
    // -------------------------------------------------------------------------
    // BANDWIDTH (fused: base + adaptive)
    // -------------------------------------------------------------------------
    svpf_fused_bandwidth_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h,
        opt->d_y_single,
        opt->d_bandwidth,
        opt->d_bandwidth_sq,
        state->d_return_ema,
        state->d_return_var,
        1,      // y_idx
        0.3f,   // alpha_bw
        0.05f,  // alpha_ret
        n
    );
    
    // -------------------------------------------------------------------------
    // ANNEALED STEIN ITERATIONS
    // -------------------------------------------------------------------------
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing ? beta_schedule[anneal_idx % 3] : 1.0f;
        float beta_factor = sqrtf(beta);
        float temp = state->use_svld ? state->temperature : 0.0f;
        
        // Distribute Stein iterations across annealing stages
        int stein_iters = state->n_stein_steps / n_anneal;
        if (anneal_idx == n_anneal - 1) {
            stein_iters = state->n_stein_steps - stein_iters * (n_anneal - 1);
        }
        
        for (int s = 0; s < stein_iters; s++) {
            // FUSED GRADIENT (was: prior_grad + likelihood + combine + hessian)
            svpf_fused_gradient_kernel<<<n_blocks_1d, BLOCK_SIZE, grad_smem, capture_stream>>>(
                state->h,
                state->h_prev,
                state->grad_log_p,
                state->log_weights,
                state->use_newton ? opt->d_precond_grad : nullptr,
                state->use_newton ? opt->d_inv_hessian : nullptr,
                opt->d_y_single,
                1,  // y_idx
                params->rho, params->sigma_z, params->mu,
                beta, state->nu, student_t_const,
                state->use_newton,
                n
            );
            
            // FUSED STEIN + TRANSPORT (was: memset + stein + transport)
            if (state->use_newton) {
                svpf_fused_stein_transport_newton_kernel<<<n_blocks_1d, BLOCK_SIZE, stein_smem, capture_stream>>>(
                    state->h,
                    opt->d_precond_grad,
                    opt->d_inv_hessian,
                    state->d_grad_v,
                    state->rng_states,
                    opt->d_bandwidth,
                    base_step, beta_factor, temp,
                    state->rmsprop_rho, state->rmsprop_eps,
                    n
                );
            } else {
                svpf_fused_stein_transport_kernel<<<n_blocks_1d, BLOCK_SIZE, stein_smem, capture_stream>>>(
                    state->h,
                    state->grad_log_p,
                    state->d_grad_v,
                    state->rng_states,
                    opt->d_bandwidth,
                    base_step, beta_factor, temp,
                    state->rmsprop_rho, state->rmsprop_eps,
                    n
                );
            }
        }
    }
    
    // -------------------------------------------------------------------------
    // OUTPUTS (fused: logsumexp + vol_mean + h_mean)
    // -------------------------------------------------------------------------
    svpf_fused_outputs_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h,
        state->log_weights,
        opt->d_loglik_single,
        opt->d_vol_single,
        opt->d_h_mean_prev,
        0,  // t_out
        n
    );
    
    // =========================================================================
    // END CAPTURE
    // =========================================================================
    cudaStreamEndCapture(capture_stream, &opt->graph);
    cudaGraphInstantiate(&opt->graph_exec, opt->graph, NULL, NULL, 0);
    
    opt->graph_captured = true;
    opt->graph_n = n;
    opt->graph_n_stein = state->n_stein_steps;
}

// =============================================================================
// PUBLIC API: Graph-accelerated step
// =============================================================================

void svpf_step_graph(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out,
    float* h_mean_out
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    
    svpf_optimized_init(opt, n);
    
    // Check if recapture needed
    bool need_capture = !opt->graph_captured 
                     || opt->graph_n != n 
                     || opt->graph_n_stein != state->n_stein_steps;
    
    if (need_capture) {
        if (opt->graph_captured) {
            cudaGraphExecDestroy(opt->graph_exec);
            cudaGraphDestroy(opt->graph);
            opt->graph_captured = false;
        }
        svpf_graph_capture_internal(state, params);
    }
    
    // PRE-GRAPH: Stage parameters
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), 
                    cudaMemcpyHostToDevice, opt->graph_stream);
    
    if (state->use_guide) {
        svpf_ekf_update(state, y_t, params);
        cudaMemcpyAsync(opt->d_guide_mean, &state->guide_mean, sizeof(float),
                        cudaMemcpyHostToDevice, opt->graph_stream);
    }
    
    cudaStreamSynchronize(opt->graph_stream);
    
    // GRAPH LAUNCH
    cudaGraphLaunch(opt->graph_exec, opt->graph_stream);
    
    // POST-GRAPH: Copy results
    cudaStreamSynchronize(opt->graph_stream);
    
    if (h_loglik_out) {
        cudaMemcpy(h_loglik_out, opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (h_vol_out) {
        cudaMemcpy(h_vol_out, opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (h_mean_out) {
        float h_mean;
        cudaMemcpy(&h_mean, opt->d_h_mean_prev, sizeof(float), cudaMemcpyDeviceToHost);
        *h_mean_out = h_mean;
    }
    
    state->timestep++;
}

// =============================================================================
// Utilities
// =============================================================================

bool svpf_graph_is_captured(SVPFState* state) {
    return get_opt(state)->graph_captured;
}

void svpf_graph_invalidate(SVPFState* state) {
    SVPFOptimizedState* opt = get_opt(state);
    if (opt->graph_captured) {
        cudaGraphExecDestroy(opt->graph_exec);
        cudaGraphDestroy(opt->graph);
        opt->graph_captured = false;
    }
}
