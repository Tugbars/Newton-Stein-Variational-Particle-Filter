/**
 * @file svpf_fully_fused_v2_fixed.cu
 * @brief Fully Fused Single-Kernel SVPF Step - Complete Feature Set (FIXED)
 * 
 * Fixes from original:
 * - EKF obs_offset: now uses student_t_implied_offset parameter (not hardcoded)
 * - Hessian computation: no beta scaling (only gradient is annealed)
 * - Full Newton: uses raw gradients, applies preconditioning at end
 * - Full Newton repulsion: no Hessian scaling (only regular Newton scales)
 * - KSD computation: uses raw gradients for both Newton modes
 * 
 * Features included:
 * - EKF guide computation (inside kernel)
 * - Guided predict with innovation-based alpha
 * - Guide-preserving nudge
 * - Adaptive guide strength
 * - Newton preconditioning
 * - Full Newton (kernel-weighted Hessian)
 * - Local params (delta_rho, delta_sigma)
 * - MIM jumps
 * - Partial rejuvenation
 * - Adaptive μ update
 * - Adaptive σ scaling
 * 
 * Limitation: N ≤ 1024 (single block execution)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum_v2(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;
    
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__device__ float block_reduce_max_v2(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;
    
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : -1e10f;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// =============================================================================
// FULLY FUSED SVPF STEP KERNEL V2 - FIXED VERSION
// =============================================================================

__global__ void svpf_fully_fused_step_v2_kernel(
    // Particle state [N]
    float* __restrict__ h,
    float* __restrict__ h_prev,
    float* __restrict__ grad_log_p,
    float* __restrict__ log_weights,
    float* __restrict__ d_grad_v,           // RMSprop velocity
    float* __restrict__ d_inv_hessian,      // For Newton (diagnostic output)
    curandStatePhilox4_32_10_t* __restrict__ rng,
    // Inputs
    float y_t,
    float y_prev,
    float h_mean_prev,                      // Previous posterior mean
    float vol_prev,                         // Previous volatility (for adaptive sigma)
    float ksd_prev,                         // Previous KSD (for rejuvenation trigger)
    // Outputs
    float* __restrict__ d_h_mean,
    float* __restrict__ d_vol,
    float* __restrict__ d_loglik,
    float* __restrict__ d_bandwidth_out,
    float* __restrict__ d_ksd,
    float* __restrict__ d_guide_mean,       // EKF guide mean (read/write)
    float* __restrict__ d_guide_var,        // EKF guide variance (read/write)
    // Model parameters
    float rho,
    float sigma_z,
    float mu,
    float nu,
    float lik_offset,
    float student_t_const,
    float student_t_implied_offset,         // FIX: Added parameter (was hardcoded)
    float gamma,                            // Leverage effect
    // Local params
    float delta_rho,                        // Per-particle rho variation
    float delta_sigma,                      // Per-particle sigma variation
    // MIM parameters
    float mim_jump_prob,
    float mim_jump_scale,
    // Guide parameters
    float guide_strength_base,
    float guide_strength_max,
    float guide_innovation_threshold,
    float guided_alpha_base,                // For guided predict
    float guided_alpha_shock,               // Shock-responsive alpha
    float guided_innovation_thresh_predict, // Separate threshold for guided predict
    int use_guide,
    int use_guided_predict,
    int use_guide_preserving,
    // Newton parameters
    int use_newton,
    int use_full_newton,
    // Rejuvenation parameters
    int use_rejuvenation,
    float rejuv_ksd_threshold,
    float rejuv_prob,
    float rejuv_blend,
    // Adaptive mu parameters
    int use_adaptive_mu,
    float* d_mu_state,                      // Adaptive mu estimate (read/write)
    float mu_ema_alpha,
    // Adaptive sigma parameters  
    int use_adaptive_sigma,
    float sigma_boost_threshold,
    float sigma_boost_max,
    // Stein parameters
    float step_size,
    float temperature,
    float rmsprop_rho,
    float rmsprop_eps,
    int n_stein_steps,
    int n_anneal_steps,
    int stein_sign_mode,
    // Control
    int n,
    int timestep
) {
    int i = threadIdx.x;
    if (i >= n) return;
    
    // Shared memory layout
    extern __shared__ float smem[];
    float* sh_h = smem;                     // [N]
    float* sh_grad = smem + n;              // [N]
    float* sh_mu = smem + 2 * n;            // [N] AR(1) prior means for mixture
    float* sh_hess = smem + 3 * n;          // [N] (for Newton)
    float* sh_reduce = smem + 4 * n;        // [32] for reductions
    
    // Shared scalars (computed by thread 0, used by all)
    __shared__ float sh_guide_mean;
    __shared__ float sh_guide_var;
    __shared__ float sh_guide_strength;
    __shared__ float sh_bandwidth;
    __shared__ float sh_effective_sigma;
    __shared__ float sh_effective_mu;
    __shared__ float sh_mean_h;
    
    float inv_n = 1.0f / (float)n;
    
    // ==========================================================================
    // PHASE 0: COMPUTE SHARED PARAMETERS (thread 0)
    // ==========================================================================
    
    if (i == 0) {
        // Effective mu (adaptive or fixed)
        sh_effective_mu = use_adaptive_mu ? *d_mu_state : mu;
        
        // Load EKF state from previous step
        float prev_guide_mean = *d_guide_mean;
        sh_guide_var = *d_guide_var;
        
        // Initialize on first call (guide_var == 0)
        if (sh_guide_var < 1e-6f) {
            prev_guide_mean = mu;
            sh_guide_var = sigma_z * sigma_z / (1.0f - rho * rho);
        }
        
        // Effective sigma (adaptive scaling for high-vol regimes)
        float sigma_boost = 1.0f;
        if (use_adaptive_sigma && timestep > 0) {
            float vol_est = fmaxf(vol_prev, 1e-4f);
            float return_z = fabsf(y_t) / vol_est;
            
            if (return_z > sigma_boost_threshold) {
                float severity = fminf((return_z - sigma_boost_threshold) / 3.0f, 1.0f);
                sigma_boost = 1.0f + (sigma_boost_max - 1.0f) * severity;
            }
        }
        sh_effective_sigma = sigma_z * sigma_boost;
        
        // =====================================================================
        // EKF UPDATE (FIXED: uses student_t_implied_offset parameter)
        // =====================================================================
        if (use_guide) {
            float eff_mu = sh_effective_mu;
            
            // Predict step - use previous GUIDE MEAN
            float m_pred = eff_mu + rho * (prev_guide_mean - eff_mu);
            float P_pred = rho * rho * sh_guide_var + sigma_z * sigma_z;
            
            // Observation model: log(y²) = h + E[log(ε²)]
            // For Student-t: E[log(y²)|h] = h - student_t_implied_offset
            // So obs_offset = -student_t_implied_offset
            float log_y2 = logf(y_t * y_t + 1e-8f);
            float obs_offset = -student_t_implied_offset;  // FIX: Use parameter
            float obs_var = 4.93f + 2.0f;  // Variance of log(chi²) ≈ 4.93, plus slack
            
            float H = 1.0f;
            float R = obs_var;
            
            float S = H * H * P_pred + R;
            float K = P_pred * H / (S + 1e-8f);
            
            float y_pred = m_pred + obs_offset;
            float innovation = log_y2 - y_pred;
            
            sh_guide_mean = m_pred + K * innovation;
            sh_guide_var = (1.0f - K * H) * P_pred;
            
            // Write back updated EKF state
            *d_guide_mean = sh_guide_mean;
            *d_guide_var = sh_guide_var;
            
            // Adaptive guide strength based on return magnitude
            float vol_est = fmaxf(vol_prev, 1e-4f);
            float return_z = fabsf(y_t) / vol_est;
            
            if (return_z > guide_innovation_threshold) {
                float severity = fminf((return_z - guide_innovation_threshold) / 3.0f, 1.0f);
                sh_guide_strength = guide_strength_base + 
                                   (guide_strength_max - guide_strength_base) * severity;
            } else {
                sh_guide_strength = guide_strength_base;
            }
        } else {
            sh_guide_mean = h_mean_prev;
            sh_guide_var = 1.0f;
            sh_guide_strength = 0.0f;
        }
    }
    __syncthreads();
    
    // Load shared values to registers
    float effective_mu = sh_effective_mu;
    float effective_sigma = sh_effective_sigma;
    float guide_mean = sh_guide_mean;
    float guide_strength = sh_guide_strength;
    
    float sigma_z_sq = effective_sigma * effective_sigma;
    float inv_sigma_z_sq = 1.0f / sigma_z_sq;
    float inv_2sigma_sq = 0.5f * inv_sigma_z_sq;
    
    // ==========================================================================
    // PHASE 1: PREDICT
    // ==========================================================================
    
    float h_old = h[i];  // Current posterior BEFORE predict
    
    // Compute prior means for mixture gradient using CURRENT h
    {
        float vol_i = expf(h_old * 0.5f);
        float leverage_i = gamma * y_prev / (vol_i + 1e-8f);
        sh_mu[i] = effective_mu + rho * (h_old - effective_mu) + leverage_i;
    }
    __syncthreads();
    
    // Per-particle parameter variation (local params)
    // Original uses deviation from ensemble mean for adaptive per-particle params
    float local_rho = rho;
    float local_sigma = effective_sigma;
    if (delta_rho > 0.0f || delta_sigma > 0.0f) {
        // Deviation from ensemble mean (h_mean_prev passed as parameter)
        float dev = h_old - h_mean_prev;
        float abs_dev = fabsf(dev);
        float tanh_dev = tanhf(dev);  // Bounded, sign-preserving
        
        // Particles far from mean get adjusted rho (mean-reverting pressure)
        float rho_adjust = delta_rho * tanh_dev;
        local_rho = fminf(fmaxf(rho + rho_adjust, 0.0f), 0.999f);
        
        // Particles far from mean get higher sigma (more uncertainty)
        float sigma_scale = 1.0f + delta_sigma * abs_dev;
        local_sigma = effective_sigma * sigma_scale;
    }
    
    // AR(1) prior mean WITH LEVERAGE EFFECT
    float leverage = 0.0f;
    if (fabsf(gamma) > 1e-6f) {
        float vol_old = expf(h_old * 0.5f);
        leverage = gamma * y_prev / (vol_old + 1e-8f);
    }
    float prior_mean = effective_mu + local_rho * (h_old - effective_mu) + leverage;
    
    // Sample noise
    float4 rand4 = curand_normal4(&rng[i]);
    float z_state = rand4.x;
    float z_diffusion = rand4.y;
    
    // MIM: mixture of narrow and wide components
    float is_jump = (curand_uniform(&rng[i]) < mim_jump_prob) ? 1.0f : 0.0f;
    float mim_sigma = local_sigma * (1.0f + is_jump * (mim_jump_scale - 1.0f));
    
    float h_i;
    if (use_guided_predict) {
        // Guided predict: blend prior with OBSERVATION-IMPLIED mean (not EKF guide!)
        // This pulls particles toward where the observation suggests volatility should be
        
        // Compute implied log-volatility from observation
        // log(y²) ≈ h + E[log(ε²)], so h ≈ log(y²) - E[log(ε²)]
        // implied_offset = -E[log(ε²)] for Student-t
        float log_y2 = logf(y_t * y_t + 1e-10f);
        float mean_implied = fmaxf(log_y2 + student_t_implied_offset, -5.0f);
        
        // Innovation: how far is observation-implied from our prior prediction?
        float innovation = mean_implied - prior_mean;
        float total_std = 2.5f;  // Approximate total uncertainty
        float z_score = innovation / total_std;
        
        // Activation: tanh ramp when z_score exceeds threshold
        // Only activate for POSITIVE innovations (vol spike detection)
        float activation = 0.0f;
        if (z_score > guided_innovation_thresh_predict) {
            activation = tanhf(z_score - guided_innovation_thresh_predict);
        }
        
        // Blend coefficient: ramps from alpha_base to alpha_shock
        float guided_alpha = guided_alpha_base + (guided_alpha_shock - guided_alpha_base) * activation;
        
        // Proposal mean: blend prior with observation-implied
        float mean_proposal = (1.0f - guided_alpha) * prior_mean + guided_alpha * mean_implied;
        
        h_i = mean_proposal + mim_sigma * z_state;
    } else {
        h_i = prior_mean + mim_sigma * z_state;
    }
    
    h_i = clamp_logvol(h_i);
    
    // Store old state to h_prev, write new state to h
    h_prev[i] = h_old;
    h[i] = h_i;
    
    __syncthreads();
    
    // ==========================================================================
    // PHASE 2: GUIDE APPLICATION
    // ==========================================================================
    
    if (use_guide && guide_strength > 0.0f) {
        h_i = h[i];
        
        if (use_guide_preserving) {
            // Guide-preserving: shift mean but keep deviations
            h_i = h_i + guide_strength * (guide_mean - h_mean_prev);
        } else {
            // Simple nudge toward guide
            h_i = (1.0f - guide_strength) * h_i + guide_strength * guide_mean;
        }
        
        h[i] = clamp_logvol(h_i);
    }
    
    __syncthreads();
    
    // ==========================================================================
    // PHASE 3: BANDWIDTH (Silverman's rule)
    // ==========================================================================
    
    sh_h[i] = h[i];
    __syncthreads();
    
    float sum_h = block_reduce_sum_v2(sh_h[i], sh_reduce);
    __syncthreads();
    
    if (i == 0) sh_mean_h = sum_h * inv_n;
    __syncthreads();
    
    float mean_h = sh_mean_h;
    float diff_h = sh_h[i] - mean_h;
    
    float sum_sq = block_reduce_sum_v2(diff_h * diff_h, sh_reduce);
    __syncthreads();
    
    if (i == 0) {
        float var_h = sum_sq * inv_n;
        float std_h = sqrtf(fmaxf(var_h, 1e-6f));
        float bw = 1.06f * std_h * powf((float)n, -0.2f);
        bw = fminf(fmaxf(bw, 0.05f), 2.0f);
        sh_bandwidth = bw;
        *d_bandwidth_out = bw;
    }
    __syncthreads();
    
    float bandwidth = sh_bandwidth;
    float bw_sq = bandwidth * bandwidth;
    float inv_bw_sq = 1.0f / bw_sq;
    
    // ==========================================================================
    // PHASE 4: STEIN LOOP (FIXED NEWTON HANDLING)
    // ==========================================================================
    
    int safe_anneal = (n_anneal_steps < 1) ? 1 : n_anneal_steps;
    int safe_stein = (n_stein_steps < 1) ? 1 : n_stein_steps;
    
    int steps_per_anneal = safe_stein / safe_anneal;
    int remainder = safe_stein - steps_per_anneal * safe_anneal;
    
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    float hess_prior = -inv_sigma_z_sq;  // Gaussian prior hessian (constant)
    
    int global_step = 0;
    float ksd_final = 0.0f;
    
    for (int ai = 0; ai < safe_anneal; ai++) {
        // Compute beta: ramp from ~0.3 to 1.0, always end at 1.0
        float beta;
        if (safe_anneal <= 1) {
            beta = 1.0f;
        } else if (ai == safe_anneal - 1) {
            beta = 1.0f;  // Last stage always full likelihood
        } else {
            float t = (float)ai / (float)(safe_anneal - 1);
            beta = 0.3f + t * 0.5f;
        }
        
        int si = steps_per_anneal;
        if (ai == safe_anneal - 1) si += remainder;
        
        for (int s = 0; s < si; s++) {
            global_step++;
            bool is_last = (global_step == safe_stein);
            
            h_i = h[i];
            
            // ----- Mixture Prior Gradient (O(N²)) -----
            float log_r_max = -1e10f;
            #pragma unroll 8
            for (int k = 0; k < n; k++) {
                float diff = h_i - sh_mu[k];
                float log_r = -diff * diff * inv_2sigma_sq;
                log_r_max = fmaxf(log_r_max, log_r);
            }
            
            float sum_r = 0.0f;
            float weighted_grad = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < n; k++) {
                float diff = h_i - sh_mu[k];
                float log_r = -diff * diff * inv_2sigma_sq;
                float r = expf(log_r - log_r_max);
                sum_r += r;
                weighted_grad -= r * diff * inv_sigma_z_sq;
            }
            float grad_prior_i = weighted_grad / (sum_r + 1e-8f);
            grad_prior_i = fminf(fmaxf(grad_prior_i, -10.0f), 10.0f);
            
            // ----- Likelihood Gradient -----
            float vol = expf(h_i * 0.5f);
            float y_sq = y_t * y_t;
            float scaled_y_sq = y_sq / (vol * vol + 1e-8f);
            float A = scaled_y_sq / nu;
            float one_plus_A = 1.0f + A;
            
            // Log weight (full likelihood, no beta)
            log_weights[i] = student_t_const - 0.5f * h_i 
                           - 0.5f * (nu + 1.0f) * logf(one_plus_A);
            
            // Likelihood gradient
            float raw_grad = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A;
            float grad_lik = raw_grad - lik_offset;
            
            // Combined gradient (annealed likelihood)
            float g = grad_prior_i + beta * grad_lik;
            g = fminf(fmaxf(g, -10.0f), 10.0f);
            grad_log_p[i] = g;  // Store raw gradient
            
            // ----- Hessian (FIX: NO BETA SCALING) -----
            float hess_lik = -0.5f * (nu + 1.0f) * A / (one_plus_A * one_plus_A);
            float curvature_i = -(hess_lik + hess_prior);  // FIX: No beta here
            curvature_i = fminf(fmaxf(curvature_i, 0.1f), 100.0f);
            
            float inv_H_i = 1.0f / curvature_i;
            float precond_grad_i = 0.7f * g * inv_H_i;
            
            // Store for diagnostic output
            if (use_newton || use_full_newton) {
                d_inv_hessian[i] = curvature_i;
            }
            
            __syncthreads();
            
            // ----- Load to shared (FIX: MODE-DEPENDENT) -----
            sh_h[i] = h[i];
            
            if (use_full_newton) {
                // FIX: Full Newton uses RAW gradient, stores curvature
                sh_grad[i] = g;
                sh_hess[i] = curvature_i;
            } else if (use_newton) {
                // Regular Newton: preconditioned gradient, stores inverse Hessian
                sh_grad[i] = precond_grad_i;
                sh_hess[i] = inv_H_i;
            } else {
                // Vanilla: raw gradient
                sh_grad[i] = g;
                sh_hess[i] = 1.0f;
            }
            __syncthreads();
            
            // ----- Stein Transport (FIX: MODE-DEPENDENT REPULSION) -----
            float k_sum = 0.0f;
            float gk_sum = 0.0f;
            float ksd_sum = 0.0f;
            
            // Full Newton accumulators
            float H_weighted = 0.0f;
            float K_sum_norm = 0.0f;
            
            #pragma unroll 8
            for (int j = 0; j < n; j++) {
                float h_j = sh_h[j];
                float s_j = sh_grad[j];
                float diff = h_i - h_j;
                float diff_sq = diff * diff;
                float dist_sq = diff_sq * inv_bw_sq;
                
                // IMQ kernel
                float base = 1.0f + dist_sq;
                float K = 1.0f / base;
                float K_sq = K * K;
                
                // Gradient term
                k_sum += K * s_j;
                
                // FIX: Repulsion scaling depends on mode
                if (use_newton) {
                    // Regular Newton: scale repulsion by inverse Hessian
                    gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq * sh_hess[j];
                } else {
                    // FIX: Vanilla AND Full Newton: NO Hessian scaling
                    gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
                }
                
                // Full Newton: accumulate kernel-weighted Hessian
                if (use_full_newton) {
                    H_weighted += sh_hess[j] * K;  // sh_hess is curvature for full_newton
                    float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
                    H_weighted += Nk;
                    K_sum_norm += K;
                }
                
                // KSD on last iteration (use RAW gradients for correctness)
                if (is_last) {
                    // Get raw gradients for KSD
                    float raw_s_i = g;  // We have raw gradient
                    float raw_s_j;
                    if (use_newton) {
                        // Undo preconditioning: precond = 0.7 * g * inv_H
                        // So raw = precond / (0.7 * inv_H)
                        raw_s_j = s_j / (0.7f * sh_hess[j] + 1e-8f);
                    } else {
                        // Full Newton and Vanilla already have raw gradients
                        raw_s_j = s_j;
                    }
                    
                    float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
                    float grad_y_k = -grad_x_k;
                    float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
                    ksd_sum += K * raw_s_i * raw_s_j + raw_s_i * grad_y_k 
                             + raw_s_j * grad_x_k + hess_xy_k;
                }
            }
            
            // Compute phi_i (FIX: mode-dependent)
            float phi_i;
            if (use_full_newton) {
                // FIX: Full Newton applies preconditioning HERE at the end
                H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
                H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);
                float inv_H_avg = 1.0f / H_weighted;
                phi_i = (k_sum + gk_sum) * inv_n * inv_H_avg * 0.7f;  // FIX: Apply 0.7 here
            } else {
                // Newton: already preconditioned via sh_grad
                // Vanilla: no preconditioning
                phi_i = (k_sum + gk_sum) * inv_n;
            }
            
            // RMSprop
            float v_prev = d_grad_v[i];
            float v_new = rmsprop_rho * v_prev + (1.0f - rmsprop_rho) * phi_i * phi_i;
            d_grad_v[i] = v_new;
            
            // Transport
            float effective_step = step_size * sqrtf(beta);
            float precond = rsqrtf(v_new + rmsprop_eps);
            float drift = effective_step * phi_i * precond;
            
            float diffusion = 0.0f;
            if (temperature > 1e-6f) {
                diffusion = sqrtf(2.0f * effective_step * temperature) * z_diffusion;
                z_diffusion = curand_normal(&rng[i]);  // Refresh for next iteration
            }
            
            h[i] = clamp_logvol(h_i + drift + diffusion);
            
            if (is_last) {
                ksd_final = ksd_sum;
            }
            
            __syncthreads();
        }
    }
    
    // ==========================================================================
    // PHASE 5: PARTIAL REJUVENATION
    // ==========================================================================
    
    if (use_rejuvenation && ksd_prev > rejuv_ksd_threshold && timestep > 10) {
        float u = curand_uniform(&rng[i]);
        if (u < rejuv_prob) {
            h_i = h[i];
            float guide_std = sqrtf(fmaxf(sh_guide_var, 1e-6f));
            float noise = curand_normal(&rng[i]);
            float rejuv_target = guide_mean + guide_std * noise;
            h_i = (1.0f - rejuv_blend) * h_i + rejuv_blend * rejuv_target;
            h[i] = clamp_logvol(h_i);
        }
    }
    __syncthreads();
    
    // ==========================================================================
    // PHASE 6: OUTPUTS
    // ==========================================================================
    
    // KSD reduction
    float sum_ksd = block_reduce_sum_v2(ksd_final, sh_reduce);
    __syncthreads();
    
    if (i == 0) {
        float final_ksd = sqrtf(fmaxf(sum_ksd * inv_n * inv_n, 0.0f));
        *d_ksd = final_ksd;
    }
    __syncthreads();
    
    // UNWEIGHTED h_mean and vol (matching standard svpf_fused_outputs_kernel)
    float h_final = h[i];
    float vol_i = expf(h_final * 0.5f);
    
    float sum_h_out = block_reduce_sum_v2(h_final, sh_reduce);
    __syncthreads();
    
    float sum_vol = block_reduce_sum_v2(vol_i, sh_reduce);
    __syncthreads();
    
    // Log-likelihood via log-sum-exp of importance weights
    float log_w_i = log_weights[i];
    float max_log_w = block_reduce_max_v2(log_w_i, sh_reduce);
    __syncthreads();
    
    if (i == 0) sh_reduce[0] = max_log_w;
    __syncthreads();
    max_log_w = sh_reduce[0];
    
    float w_shifted = expf(log_w_i - max_log_w);
    float sum_w = block_reduce_sum_v2(w_shifted, sh_reduce);
    __syncthreads();
    
    if (i == 0) {
        float h_mean_new = sum_h_out * inv_n;
        float vol_new = sum_vol * inv_n;
        
        *d_h_mean = h_mean_new;
        *d_vol = vol_new;
        *d_loglik = max_log_w + logf(fmaxf(sum_w, 1e-10f) * inv_n);
        
        // Adaptive mu update
        if (use_adaptive_mu && timestep > 10) {
            float mu_old = *d_mu_state;
            float mu_new = mu_ema_alpha * h_mean_new + (1.0f - mu_ema_alpha) * mu_old;
            mu_new = fminf(fmaxf(mu_new, -10.0f), 0.0f);
            *d_mu_state = mu_new;
        }
    }
}

// =============================================================================
// Wrapper
// =============================================================================

extern "C" {

cudaError_t svpf_fully_fused_step_v2(
    // Arrays
    float* h, float* h_prev, float* grad_log_p, float* log_weights,
    float* d_grad_v, float* d_inv_hessian, curandStatePhilox4_32_10_t* rng,
    // Scalar inputs
    float y_t, float y_prev, float h_mean_prev, float vol_prev, float ksd_prev,
    // Scalar outputs
    float* d_h_mean, float* d_vol, float* d_loglik, float* d_bandwidth, float* d_ksd,
    float* d_guide_mean, float* d_guide_var,
    // Model params
    float rho, float sigma_z, float mu, float nu, float lik_offset, float student_t_const,
    float student_t_implied_offset,  // FIX: Added parameter
    float gamma,
    // Local params
    float delta_rho, float delta_sigma,
    // MIM
    float mim_jump_prob, float mim_jump_scale,
    // Guide params
    float guide_strength_base, float guide_strength_max, float guide_innovation_threshold,
    float guided_alpha_base, float guided_alpha_shock, float guided_innovation_thresh_predict,
    int use_guide, int use_guided_predict, int use_guide_preserving,
    // Newton
    int use_newton, int use_full_newton,
    // Rejuvenation
    int use_rejuvenation, float rejuv_ksd_threshold, float rejuv_prob, float rejuv_blend,
    // Adaptive mu
    int use_adaptive_mu, float* d_mu_state, float mu_ema_alpha,
    // Adaptive sigma
    int use_adaptive_sigma, float sigma_boost_threshold, float sigma_boost_max,
    // Stein params
    float step_size, float temperature, float rmsprop_rho, float rmsprop_eps,
    int n_stein_steps, int n_anneal_steps, int stein_sign_mode,
    // Control
    int n, int timestep, cudaStream_t stream
) {
    if (n > 1024) {
        return cudaErrorInvalidConfiguration;
    }
    
    // Shared memory: 4 arrays + reduction buffer
    int smem_size = (4 * n + 32) * sizeof(float);
    
    svpf_fully_fused_step_v2_kernel<<<1, n, smem_size, stream>>>(
        h, h_prev, grad_log_p, log_weights, d_grad_v, d_inv_hessian, rng,
        y_t, y_prev, h_mean_prev, vol_prev, ksd_prev,
        d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd, d_guide_mean, d_guide_var,
        rho, sigma_z, mu, nu, lik_offset, student_t_const,
        student_t_implied_offset,  // FIX: Pass parameter
        gamma,
        delta_rho, delta_sigma,
        mim_jump_prob, mim_jump_scale,
        guide_strength_base, guide_strength_max, guide_innovation_threshold,
        guided_alpha_base, guided_alpha_shock, guided_innovation_thresh_predict,
        use_guide, use_guided_predict, use_guide_preserving,
        use_newton, use_full_newton,
        use_rejuvenation, rejuv_ksd_threshold, rejuv_prob, rejuv_blend,
        use_adaptive_mu, d_mu_state, mu_ema_alpha,
        use_adaptive_sigma, sigma_boost_threshold, sigma_boost_max,
        step_size, temperature, rmsprop_rho, rmsprop_eps,
        n_stein_steps, n_anneal_steps, stein_sign_mode,
        n, timestep
    );
    
    return cudaGetLastError();
}

bool svpf_fully_fused_v2_supported(int n_particles) {
    return n_particles <= 1024;
}

} // extern "C"
