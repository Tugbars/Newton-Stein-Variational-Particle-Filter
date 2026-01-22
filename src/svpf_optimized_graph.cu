/**
 * @file svpf_optimized_graph.cu
 * @brief CUDA Graph-accelerated SVPF implementation with Adaptive Scouts
 * 
 * Contains:
 * - svpf_step_graph (graph-based step function)
 * - svpf_graph_is_captured / svpf_graph_invalidate
 * 
 * NEW: Adaptive scout mechanism using φ-stress feedback
 *      - Computes mean |φ| after final Stein iteration
 *      - Modulates scout variance based on swarm convergence
 *      - Scouts tighten during stable tracking, expand during jumps
 * 
 * Kernels are in svpf_opt_kernels.cu
 */

#include "svpf_kernels.cuh"
#include <cmath>

// =============================================================================
// Helper
// =============================================================================

static inline SVPFOptimizedState* get_opt(SVPFState* state) {
    return &state->opt_backend;
}

// Defined in svpf_optimized.cu
extern void svpf_optimized_init(SVPFOptimizedState* opt, int n);

// =============================================================================
// Adaptive Scout Memory Allocation
// =============================================================================

static void svpf_adaptive_scouts_init(SVPFOptimizedState* opt) {
    if (opt->d_phi_stress == nullptr) {
        cudaMalloc(&opt->d_phi_stress, sizeof(float));
        cudaMemset(opt->d_phi_stress, 0, sizeof(float));
    }
    if (opt->d_adaptive_scale == nullptr) {
        cudaMalloc(&opt->d_adaptive_scale, sizeof(float));
        float init_scale = 1.0f;
        cudaMemcpy(opt->d_adaptive_scale, &init_scale, sizeof(float), cudaMemcpyHostToDevice);
    }
}

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
    
    // Ensure adaptive scout memory is allocated (only if enabled)
    if (state->use_adaptive_scouts) {
        svpf_adaptive_scouts_init(opt);
    }
    
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
    
    // -------------------------------------------------------------------------
    // PREDICT - Choose kernel based on configuration
    // Priority: Student-t (cleanest) > Guided > MIM > Standard
    // -------------------------------------------------------------------------
    if (state->use_student_t) {
        // Heavy-tail predict: Student-t noise, no mixture heuristics
        if (state->use_asymmetric_rho) {
            svpf_predict_student_t_asymmetric_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->rng_states,
                opt->d_y_single, 1,
                rho_up, rho_down,
                p.sigma_z, p.mu, p.gamma,
                state->predict_nu,
                n
            );
        } else {
            svpf_predict_student_t_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->rng_states,
                opt->d_y_single, 1,
                p.rho, p.sigma_z, p.mu, p.gamma,
                state->predict_nu,
                n
            );
        }
    } else if (state->use_guided) {
        if (state->use_adaptive_scouts) {
            // Adaptive guided prediction - reads scale from d_adaptive_scale
            svpf_predict_guided_adaptive_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->rng_states,
                opt->d_y_single, opt->d_h_mean_prev,
                opt->d_adaptive_scale,  // NEW: device pointer
                1, rho_up, rho_down,
                p.sigma_z, p.mu, p.gamma,
                state->mim_jump_prob,
                delta_rho, delta_sigma,
                state->guided_alpha_base,
                state->guided_alpha_shock,
                state->guided_innovation_threshold,
                n
            );
        } else {
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
        }
    } else if (state->use_mim) {
        if (state->use_adaptive_scouts) {
            // Adaptive MIM prediction - reads scale from d_adaptive_scale
            svpf_predict_mim_adaptive_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->rng_states,
                opt->d_y_single, opt->d_h_mean_prev,
                opt->d_adaptive_scale,  // NEW: device pointer
                1, rho_up, rho_down,
                p.sigma_z, p.mu, p.gamma,
                state->mim_jump_prob,
                delta_rho, delta_sigma, n
            );
        } else {
            svpf_predict_mim_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->rng_states,
                opt->d_y_single, opt->d_h_mean_prev,
                1, rho_up, rho_down,
                p.sigma_z, p.mu, p.gamma,
                state->mim_jump_prob, state->mim_jump_scale,
                delta_rho, delta_sigma, n
            );
        }
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
    
    // -------------------------------------------------------------------------
    // ANNEALED STEIN UPDATES
    // -------------------------------------------------------------------------
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing ? beta_schedule[anneal_idx % 3] : 1.0f;
        
        // Gradient - use Student-t gradient when heavy-tail mode enabled
        if (state->use_student_t) {
            if (use_small_n) {
                svpf_mixture_prior_grad_student_t_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                    state->h, state->h_prev, state->grad_log_p,
                    p.rho, p.sigma_z, p.mu, state->predict_nu, n
                );
            } else {
                svpf_mixture_prior_grad_student_t_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                    state->h, state->h_prev, state->grad_log_p,
                    p.rho, p.sigma_z, p.mu, state->predict_nu, n
                );
            }
        } else if (use_small_n) {
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
            
            // Choose kernel: Newton vs Standard, IMQ vs RBF, Persistent vs 2D
            if (use_newton) {
                if (state->use_imq) {
                    // Newton + IMQ (polynomial decay - "infinite vision")
                    if (use_small_n) {
                        svpf_stein_newton_imq_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, newton_smem, capture_stream>>>(
                            state->h, opt->d_precond_grad, opt->d_inv_hessian,
                            opt->d_phi, opt->d_bandwidth, n
                        );
                    } else {
                        int num_tiles = (n + TILE_J - 1) / TILE_J;
                        dim3 grid_2d(n, num_tiles);
                        svpf_stein_newton_imq_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, capture_stream>>>(
                            state->h, opt->d_precond_grad, opt->d_inv_hessian,
                            opt->d_phi, opt->d_bandwidth, n
                        );
                    }
                } else {
                    // Newton + RBF (original)
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
                }
            } else {
                if (state->use_imq) {
                    // Standard SVGD + IMQ
                    if (use_small_n) {
                        svpf_stein_imq_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, capture_stream>>>(
                            state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                        );
                    } else {
                        int num_tiles = (n + TILE_J - 1) / TILE_J;
                        dim3 grid_2d(n, num_tiles);
                        svpf_stein_imq_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, capture_stream>>>(
                            state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                        );
                    }
                } else {
                    // Standard SVGD + RBF (original)
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
                if (state->use_student_t) {
                    if (use_small_n) {
                        svpf_mixture_prior_grad_student_t_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                            state->h, state->h_prev, state->grad_log_p,
                            p.rho, p.sigma_z, p.mu, state->predict_nu, n
                        );
                    } else {
                        svpf_mixture_prior_grad_student_t_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                            state->h, state->h_prev, state->grad_log_p,
                            p.rho, p.sigma_z, p.mu, state->predict_nu, n
                        );
                    }
                } else if (use_small_n) {
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
    
    // -------------------------------------------------------------------------
    // NEW: COMPUTE PHI STRESS (after final Stein iteration, before outputs)
    // This measures mean |φ| as a convergence diagnostic for adaptive scouts
    // -------------------------------------------------------------------------
    if (state->use_adaptive_scouts) {
        // Reset accumulator (required for atomicAdd) - use kernel for graph compatibility
        svpf_memset_kernel<<<1, 1, 0, capture_stream>>>(
            opt->d_phi_stress, 0.0f, 1
        );
        
        // Launch kernel - accumulates sum of |φ|, division by N done on host
        svpf_compute_phi_stress_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            opt->d_phi, opt->d_phi_stress, n
        );
    }
    
    // -------------------------------------------------------------------------
    // FINAL OUTPUTS
    // -------------------------------------------------------------------------
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
        state->h, opt->d_phi, n  // Reuse d_phi as partial sums buffer
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
    opt->graph_n_anneal = state->n_anneal_steps;
    opt->graph_adaptive_enabled = state->use_adaptive_scouts;
    opt->graph_use_student_t = state->use_student_t;
    opt->graph_use_imq = state->use_imq;
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
    
    // Only allocate adaptive scout memory if enabled
    if (state->use_adaptive_scouts) {
        svpf_adaptive_scouts_init(opt);
    }
    
    // Check if recapture needed (includes all kernel-affecting flags)
    bool need_capture = !opt->graph_captured 
                     || opt->graph_n != n 
                     || opt->graph_n_stein != state->n_stein_steps
                     || opt->graph_n_anneal != state->n_anneal_steps
                     || opt->graph_adaptive_enabled != state->use_adaptive_scouts
                     || opt->graph_use_student_t != state->use_student_t
                     || opt->graph_use_imq != state->use_imq;
    
    if (need_capture) {
        if (opt->graph_captured) {
            cudaGraphExecDestroy(opt->graph_exec);
            cudaGraphDestroy(opt->graph);
            opt->graph_captured = false;
        }
        svpf_graph_capture_internal(state, params);
    }
    
    // -------------------------------------------------------------------------
    // PRE-GRAPH: Stage parameters to device
    // -------------------------------------------------------------------------
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), 
                    cudaMemcpyHostToDevice, opt->graph_stream);
    
    // Stage adaptive scout scale (computed from previous timestep's phi_ema)
    if (state->use_adaptive_scouts) {
        cudaMemcpyAsync(opt->d_adaptive_scale, &state->adaptive_scout_scale, 
                        sizeof(float), cudaMemcpyHostToDevice, opt->graph_stream);
    }
    
    if (state->use_guide) {
        svpf_ekf_update(state, y_t, params);
        cudaMemcpyAsync(opt->d_guide_mean, &state->guide_mean, sizeof(float),
                        cudaMemcpyHostToDevice, opt->graph_stream);
    }
    
    cudaStreamSynchronize(opt->graph_stream);
    
    // -------------------------------------------------------------------------
    // GRAPH LAUNCH
    // -------------------------------------------------------------------------
    cudaGraphLaunch(opt->graph_exec, opt->graph_stream);
    
    // -------------------------------------------------------------------------
    // POST-GRAPH: Copy results and update adaptive scouts
    // -------------------------------------------------------------------------
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
    
    // NEW: Update adaptive scout scale based on phi stress
    if (state->use_adaptive_scouts) {
        float phi_sum;
        cudaMemcpy(&phi_sum, opt->d_phi_stress, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Kernel returns sum, normalize to mean here
        float phi_mean = phi_sum / (float)n;
        svpf_update_adaptive_scouts(state, phi_mean);
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

// =============================================================================
// Initialization helper for adaptive scout defaults
// =============================================================================

void svpf_init_adaptive_scouts(SVPFState* state) {
    state->use_adaptive_scouts = true;
    state->phi_ema = 0.0f;
    state->adaptive_scout_scale = 1.0f;  // Reserves start conservative
    
    // Tunable parameters
    // 
    // Elite Guard Strategy breakdown (with jump_prob = 0.05):
    //   - 2% Elite Guard: Fixed scale=5.0 (hardcoded in kernel)
    //   - 3% Reserves: Adaptive scale [min_scout_scale, max_scout_scale]
    //   - 95% Sheep: scale=1.0
    //
    // During stable periods: Elite guard (2%) + Reserves at min (3% @ 1.0)
    //   Effective: 2% at 5x, 98% at 1x
    //
    // During panic: Elite guard (2%) + Reserves at max (3% @ 10.0)
    //   Effective: 2% at 5x, 3% at 10x, 95% at 1x
    //
    state->phi_ema_alpha = 0.15f;        // EMA smoothing (higher = more reactive)
    state->phi_stress_threshold = 0.5f;  // "Normal" |φ| level - tune based on data
    state->phi_stress_softness = 0.2f;   // Sigmoid width
    state->min_scout_scale = 1.0f;       // Reserves at minimum (elite guard handles baseline)
    state->max_scout_scale = 10.0f;      // Reserves can go very high during panic
}

// =============================================================================
// Debug helper: Get current adaptive state
// =============================================================================

void svpf_get_adaptive_state(
    SVPFState* state,
    float* phi_ema_out,
    float* scout_scale_out
) {
    if (phi_ema_out) *phi_ema_out = state->phi_ema;
    if (scout_scale_out) *scout_scale_out = state->adaptive_scout_scale;
}

// =============================================================================
// Initialization helper for heavy-tail mode (Student-t + IMQ)
// 
// This is the recommended approach for robust vol tracking without heuristics.
// Replaces: MIM scouts, adaptive scouts, guide density
// =============================================================================

void svpf_init_heavy_tail(SVPFState* state, float nu) {
    // Enable heavy-tail physics
    state->use_student_t = 1;
    state->predict_nu = nu;  // Recommend 3.0 for very heavy tails, 5.0 for moderate
    state->use_imq = 1;
    
    // Disable superseded heuristics
    state->use_mim = 0;
    state->use_adaptive_scouts = 0;
    state->use_guided = 0;
    state->use_guide = 0;
    
    // Keep asymmetric rho if set (orthogonal feature)
    // Keep Newton-Stein if set (orthogonal feature)
}

// Convenience: Heavy-tail with defaults (nu=3.0, very heavy)
void svpf_init_heavy_tail_default(SVPFState* state) {
    svpf_init_heavy_tail(state, 3.0f);
}
