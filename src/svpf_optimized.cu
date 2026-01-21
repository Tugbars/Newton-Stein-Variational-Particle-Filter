/**
 * @file svpf_optimized.cu
 * @brief Normal/Debug SVPF implementation
 * 
 * Contains:
 * - svpf_optimized_init / svpf_optimized_cleanup
 * - svpf_step_adaptive (debug-friendly step function)
 * 
 * Kernels are in svpf_kernels.cu
 */

#include "svpf_kernels.cuh"
#include <cub/cub.cuh>

// =============================================================================
// Helper
// =============================================================================

static inline SVPFOptimizedState* get_opt(SVPFState* state) {
    return &state->opt_backend;
}

// Forward declaration
void svpf_optimized_cleanup(SVPFOptimizedState* opt);

// =============================================================================
// State Management
// =============================================================================

void svpf_optimized_init(SVPFOptimizedState* opt, int n) {
    if (opt->initialized && n > opt->allocated_n) {
        svpf_optimized_cleanup(opt);
    }
    if (opt->initialized) return;
    
    // Query CUB temp storage size
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
    
    // Device scalars
    cudaMalloc(&opt->d_max_log_w, sizeof(float));
    cudaMalloc(&opt->d_sum_exp, sizeof(float));
    cudaMalloc(&opt->d_bandwidth, sizeof(float));
    cudaMalloc(&opt->d_bandwidth_sq, sizeof(float));
    
    float zero = 0.0f;
    cudaMemcpy(opt->d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Stein computation buffers
    cudaMalloc(&opt->d_exp_w, n * sizeof(float));
    cudaMalloc(&opt->d_phi, n * sizeof(float));
    cudaMalloc(&opt->d_grad_lik, n * sizeof(float));
    
    // Newton-Stein buffers
    cudaMalloc(&opt->d_precond_grad, n * sizeof(float));
    cudaMalloc(&opt->d_inv_hessian, n * sizeof(float));
    
    // Particle-local parameters
    cudaMalloc(&opt->d_h_mean_prev, sizeof(float));
    float init_h_mean = -3.5f;
    cudaMemcpy(opt->d_h_mean_prev, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);
    
    // Guide mean
    cudaMalloc(&opt->d_guide_mean, sizeof(float));
    cudaMemcpy(opt->d_guide_mean, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);
    
    // Single-step API buffers
    cudaMalloc(&opt->d_y_single, 2 * sizeof(float));
    cudaMalloc(&opt->d_loglik_single, sizeof(float));
    cudaMalloc(&opt->d_vol_single, sizeof(float));
    
    // CUDA Graph support
    cudaMalloc(&opt->d_params_staging, SVPF_GRAPH_PARAMS_SIZE * sizeof(float));
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    opt->allocated_n = n;
    opt->initialized = true;
}

void svpf_optimized_cleanup(SVPFOptimizedState* opt) {
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
    cudaFree(opt->d_y_single);
    cudaFree(opt->d_loglik_single);
    cudaFree(opt->d_vol_single);
    cudaFree(opt->d_params_staging);
    
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
    if (state) {
        svpf_optimized_cleanup(&state->opt_backend);
    }
}

// =============================================================================
// ADAPTIVE SVPF STEP - Normal/Debug Path
// =============================================================================

void svpf_step_adaptive(
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
    cudaStream_t stream = state->stream;
    
    svpf_optimized_init(opt, n);
    
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    SVPFParams p = *params;
    
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool use_small_n = (n <= SMALL_N_THRESHOLD);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int persistent_blocks = min(prop.multiProcessorCount, n);
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    if (persistent_smem > prop.sharedMemPerBlockOptin) {
        use_small_n = false;
    }
    
    // =========================================================================
    // PREDICT
    // =========================================================================
    float rho_up = state->use_asymmetric_rho ? state->rho_up : p.rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : p.rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    if (state->use_guided) {
        svpf_predict_guided_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
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
        svpf_predict_mim_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1, rho_up, rho_down,
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma, n
        );
    } else {
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, p.rho, p.sigma_z, p.mu, p.gamma, n
        );
    }
    
    // =========================================================================
    // EKF GUIDE DENSITY
    // =========================================================================
    if (state->use_guide) {
        svpf_ekf_update(state, y_t, &p);
        
        if (state->use_guide_preserving) {
            svpf_apply_guide_preserving_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, opt->d_h_mean_prev, state->guide_mean, state->guide_strength, n
            );
        } else {
            svpf_apply_guide_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->guide_mean, state->guide_strength, n
            );
        }
    }
    
    // =========================================================================
    // BANDWIDTH
    // =========================================================================
    svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_bandwidth, opt->d_bandwidth_sq, 0.3f, n
    );
    
    svpf_adaptive_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_bandwidth, 
        state->d_return_ema, state->d_return_var,
        y_t, 0.05f, n
    );
    
    // =========================================================================
    // ANNEALED STEIN UPDATES
    // =========================================================================
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
    
    size_t newton_smem = (3 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    bool newton_fits_smem = (newton_smem <= prop.sharedMemPerBlockOptin);
    bool use_newton = state->use_newton && newton_fits_smem;
    
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing ? beta_schedule[anneal_idx % 3] : 1.0f;
        
        // Gradient computation
        if (n <= SMALL_N_THRESHOLD) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        }
        
        svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            opt->d_y_single, 1, state->nu, student_t_const, n
        );
        
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p, beta, n
        );
        
        if (use_newton) {
            svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
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
            cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
            
            if (use_newton) {
                if (use_small_n) {
                    svpf_stein_newton_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, newton_smem, stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_newton_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                }
            } else {
                if (use_small_n) {
                    svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                }
            }
            
            // Transport
            float temp = state->use_svld ? state->temperature : 0.0f;
            float beta_factor = sqrtf(beta);
            float step_size = SVPF_STEIN_STEP_SIZE;
            if (state->use_guide) step_size *= 0.5f;
            
            svpf_apply_transport_svld_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, opt->d_phi, state->d_grad_v, state->rng_states,
                step_size, beta_factor, temp,
                state->rmsprop_rho, state->rmsprop_eps, n
            );
            
            // Recompute gradient if more iterations
            if (s < stein_iters - 1) {
                if (n <= SMALL_N_THRESHOLD) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                }
                
                svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    opt->d_y_single, 1, state->nu, student_t_const, n
                );
                
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p, beta, n
                );
                
                if (use_newton) {
                    svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->grad_log_p,
                        opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_y_single, 1, state->nu, p.sigma_z, n
                    );
                }
            }
        }
    }
    
    // =========================================================================
    // FINAL OUTPUTS
    // =========================================================================
    svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->log_weights, opt->d_loglik_single, opt->d_max_log_w, 0, n
    );
    
    svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_vol_single, 0, n
    );
    
    // H mean using CUB
    cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes, 
                           state->h, state->d_scalar_sum, n, stream);
    
    svpf_store_h_mean_kernel<<<1, 1, 0, stream>>>(
        state->d_scalar_sum, opt->d_h_mean_prev, n
    );
    
    // Sync and copy results
    cudaStreamSynchronize(stream);
    
    float h_sum_host;
    cudaMemcpy(&h_sum_host, state->d_scalar_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (h_loglik_out) cudaMemcpy(h_loglik_out, opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_vol_out) cudaMemcpy(h_vol_out, opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_mean_out) *h_mean_out = h_sum_host / (float)n;
    
    state->timestep++;
}
