/**
 * @file svpf_optimized_graph.cu
 * @brief CUDA Graph-accelerated SVPF implementation
 * 
 * Contains:
 * - svpf_step_graph (graph-based step function)
 * - svpf_graph_is_captured / svpf_graph_invalidate
 * 
 * Kernels are in svpf_kernels.cu
 */

#include "svpf_kernels.cuh"

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
    
    SVPFParams p = *params;
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int persistent_blocks = min(prop.multiProcessorCount, n);
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    size_t newton_smem = (3 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    bool use_small_n = (n <= SMALL_N_THRESHOLD) && (persistent_smem <= prop.sharedMemPerBlockOptin);
    bool use_newton = state->use_newton && (newton_smem <= prop.sharedMemPerBlockOptin);
    
    float rho_up = state->use_asymmetric_rho ? state->rho_up : p.rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : p.rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
    
    // =========================================================================
    // BEGIN CAPTURE
    // =========================================================================
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    
    // PREDICT
    if (state->use_guided) {
        svpf_predict_guided_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1, rho_up, rho_down,
            p.sigma_z, p.mu, p.gamma,
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
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma, n
        );
    } else {
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, p.rho, p.sigma_z, p.mu, p.gamma, n
        );
    }
    
    // EKF GUIDE
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
    
    // BANDWIDTH
    svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_bandwidth, opt->d_bandwidth_sq, 0.3f, n
    );
    
    svpf_adaptive_bandwidth_kernel_graph<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_bandwidth,
        state->d_return_ema, state->d_return_var,
        opt->d_y_single, 1, 0.05f, n
    );
    
    // ANNEALED STEIN UPDATES
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing ? beta_schedule[anneal_idx % 3] : 1.0f;
        
        // Gradient
        if (use_small_n) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        }
        
        svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            opt->d_y_single, 1, state->nu, student_t_const, n
        );
        
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p, beta, n
        );
        
        if (use_newton) {
            svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->grad_log_p,
                opt->d_precond_grad, opt->d_inv_hessian,
                opt->d_y_single, 1, state->nu, p.sigma_z, n
            );
        }
        
        // Stein iterations
        int stein_iters = state->n_stein_steps / n_anneal;
        if (anneal_idx == n_anneal - 1) {
            stein_iters = state->n_stein_steps - stein_iters * (n_anneal - 1);
        }
        
        for (int s = 0; s < stein_iters; s++) {
            // Graph-compatible memset
            svpf_memset_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                opt->d_phi, 0.0f, n
            );
            
            if (use_newton) {
                if (use_small_n) {
                    svpf_stein_newton_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, newton_smem, capture_stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_newton_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                }
            } else {
                if (use_small_n) {
                    svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, capture_stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                }
            }
            
            // Transport
            float temp = state->use_svld ? state->temperature : 0.0f;
            float beta_factor = sqrtf(beta);
            float step_size = SVPF_STEIN_STEP_SIZE;
            if (state->use_guide) step_size *= 0.5f;
            
            svpf_apply_transport_svld_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, opt->d_phi, state->d_grad_v, state->rng_states,
                step_size, beta_factor, temp,
                state->rmsprop_rho, state->rmsprop_eps, n
            );
            
            // Recompute gradient if more iterations
            if (s < stein_iters - 1) {
                if (use_small_n) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                }
                
                svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    opt->d_y_single, 1, state->nu, student_t_const, n
                );
                
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p, beta, n
                );
                
                if (use_newton) {
                    svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->grad_log_p,
                        opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_y_single, 1, state->nu, p.sigma_z, n
                    );
                }
            }
        }
    }
    
    // FINAL OUTPUTS
    svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->log_weights, opt->d_loglik_single, opt->d_max_log_w, 0, n
    );
    
    svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_vol_single, 0, n
    );
    
    // Graph-compatible h-mean
    svpf_memset_kernel<<<1, 1, 0, capture_stream>>>(
        state->d_scalar_sum, 0.0f, 1
    );
    
    svpf_h_mean_reduce_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_phi, n
    );
    
    svpf_h_mean_finalize_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        opt->d_phi, opt->d_h_mean_prev, n_blocks_1d, n
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
