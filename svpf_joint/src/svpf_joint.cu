/**
 * @file svpf_joint.cu
 * @brief Host-side implementation for Joint State-Parameter SVPF
 */

#include "svpf_joint.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Windows/MSVC doesn't define M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

SVPFJointConfig svpf_joint_default_config(void) {
    SVPFJointConfig cfg;
    
    cfg.n_particles = SVPF_JOINT_DEFAULT_PARTICLES;
    cfg.n_stein_steps = SVPF_JOINT_DEFAULT_STEIN_STEPS;
    
    // Learning rates: h is fast, parameters are slow
    // Note: These base rates are scaled by bw² (natural gradient) in the kernel
    // sigma is additionally boosted by surprise factor during crashes
    cfg.step_h = 0.10f;
    cfg.step_mu = 0.01f;
    cfg.step_rho = 0.005f;   // Let rho be stiff (dominated by prior)
    cfg.step_sigma = 0.05f;  // Higher base rate - will be boosted during crashes
    
    // Parameter diffusion (small random walk)
    cfg.diffusion_mu = 0.01f;
    cfg.diffusion_rho = 0.001f;
    cfg.diffusion_sigma = 0.005f;
    
    // Diversity collapse thresholds
    cfg.collapse_thresh_mu = 0.05f;
    cfg.collapse_thresh_rho = 0.02f;
    cfg.collapse_thresh_sigma = 0.02f;
    
    // Weak priors
    cfg.mu_prior_mean = -3.5f;
    cfg.mu_prior_var = 10.0f;
    cfg.rho_prior_mean = 2.0f;      // sigmoid(2.0) ≈ 0.88
    cfg.rho_prior_var = 5.0f;
    cfg.sigma_prior_mean = -2.0f;   // exp(-2.0) ≈ 0.14
    cfg.sigma_prior_var = 5.0f;
    
    // Student-t
    cfg.nu = 30.0f;
    
    // Precompute Student-t constants
    float nu = cfg.nu;
    cfg.student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                        - lgammaf(nu / 2.0f) 
                        - 0.5f * logf((float)M_PI * nu);
    
    // Implied offset: E[log(t²_ν)] = log(ν) + ψ(1/2) - ψ(ν/2)
    const float psi_half = -1.9635100260214235f;
    float nu_half = nu / 2.0f;
    float psi_nu_half = logf(nu_half) - 1.0f/(2.0f*nu_half) - 1.0f/(12.0f*nu_half*nu_half);
    float expected_log_t_sq = logf(nu) + psi_half - psi_nu_half;
    cfg.student_t_implied_offset = -expected_log_t_sq;
    
    // Gradient config
    cfg.lik_offset = 0.30f;
    cfg.prior_weight = 0.01f;
    
    return cfg;
}

// =============================================================================
// ALLOCATION
// =============================================================================

SVPFJointState* svpf_joint_create(const SVPFJointConfig* config, cudaStream_t stream) {
    SVPFJointState* state = (SVPFJointState*)malloc(sizeof(SVPFJointState));
    if (!state) return NULL;
    
    // Copy config
    if (config) {
        state->config = *config;
    } else {
        state->config = svpf_joint_default_config();
    }
    
    int n = state->config.n_particles;
    state->stream = stream;
    state->timestep = 0;
    state->y_prev = 0.0f;
    
    // Particle arrays
    cudaMalloc(&state->d_h, n * sizeof(float));
    cudaMalloc(&state->d_h_prev, n * sizeof(float));
    cudaMalloc(&state->d_mu_tilde, n * sizeof(float));
    cudaMalloc(&state->d_rho_tilde, n * sizeof(float));
    cudaMalloc(&state->d_sigma_tilde, n * sizeof(float));
    
    // Gradient arrays
    cudaMalloc(&state->d_grad_h, n * sizeof(float));
    cudaMalloc(&state->d_grad_mu, n * sizeof(float));
    cudaMalloc(&state->d_grad_rho, n * sizeof(float));
    cudaMalloc(&state->d_grad_sigma, n * sizeof(float));
    
    // Weights
    cudaMalloc(&state->d_log_w, n * sizeof(float));
    
    // RNG
    cudaMalloc(&state->d_rng, n * sizeof(curandStatePhilox4_32_10_t));
    
    // Workspace
    cudaMalloc(&state->d_temp, n * sizeof(float));
    cudaMalloc(&state->d_reduce_out, 8 * sizeof(float));
    
    // Diagnostic outputs
    cudaMalloc(&state->d_param_mean, 4 * sizeof(float));  // Extra slot for h_mean
    cudaMalloc(&state->d_param_std, 3 * sizeof(float));
    cudaMalloc(&state->d_std_unconstrained, 3 * sizeof(float));
    cudaMalloc(&state->d_collapse_flags, 3 * sizeof(int));
    
    // Initialize bandwidths
    state->bw_h = 1.0f;
    state->bw_mu = 0.5f;
    state->bw_rho = 0.2f;
    state->bw_sigma = 0.2f;
    
    return state;
}

// =============================================================================
// INITIALIZATION
// =============================================================================

void svpf_joint_initialize(SVPFJointState* state, unsigned long long seed) {
    int n = state->config.n_particles;
    int block = SVPF_JOINT_BLOCK_SIZE;
    int grid = (n + block - 1) / block;
    
    // Initialize RNG
    svpf_joint_init_rng_kernel<<<grid, block, 0, state->stream>>>(
        state->d_rng, seed, n
    );
    
    // Initialize particles
    // Start with parameters at prior means, h around mu
    float mu_init = state->config.mu_prior_mean;
    float rho_init = state->config.rho_prior_mean;
    float sigma_init = state->config.sigma_prior_mean;
    float h_spread = 0.5f;      // Spread for h initialization
    float param_spread = 0.2f;  // Spread for parameter initialization
    
    svpf_joint_init_particles_kernel<<<grid, block, 0, state->stream>>>(
        state->d_h,
        state->d_mu_tilde,
        state->d_rho_tilde,
        state->d_sigma_tilde,
        state->d_rng,
        mu_init, rho_init, sigma_init,
        h_spread, param_spread,
        n
    );
    
    // Initialize h_prev
    cudaMemcpyAsync(state->d_h_prev, state->d_h, n * sizeof(float), 
                    cudaMemcpyDeviceToDevice, state->stream);
    
    cudaStreamSynchronize(state->stream);
    state->timestep = 0;
}

// =============================================================================
// BANDWIDTH COMPUTATION (host helper)
// =============================================================================

static float compute_bandwidth(SVPFJointState* state, const float* d_values) {
    svpf_joint_compute_bandwidth_kernel<<<1, 256, 0, state->stream>>>(
        d_values, state->d_reduce_out, state->config.n_particles
    );
    
    float result[3];
    cudaMemcpyAsync(result, state->d_reduce_out, 3 * sizeof(float), 
                    cudaMemcpyDeviceToHost, state->stream);
    cudaStreamSynchronize(state->stream);
    
    return result[2];  // bandwidth
}

// =============================================================================
// STEP FUNCTION
// =============================================================================

void svpf_joint_step(SVPFJointState* state, float y_t, SVPFJointDiagnostics* diag) {
    const SVPFJointConfig* cfg = &state->config;
    int n = cfg->n_particles;
    int block = SVPF_JOINT_BLOCK_SIZE;
    int grid = (n + block - 1) / block;
    cudaStream_t stream = state->stream;
    
    // =========================================================================
    // 1. PREDICT (with parameter diffusion)
    // =========================================================================
    svpf_joint_predict_kernel<<<grid, block, 0, stream>>>(
        state->d_h,
        state->d_h_prev,
        state->d_mu_tilde,
        state->d_rho_tilde,
        state->d_sigma_tilde,
        state->d_rng,
        cfg->diffusion_mu,
        cfg->diffusion_rho,
        cfg->diffusion_sigma,
        n
    );
    
    // =========================================================================
    // 2. COMPUTE BANDWIDTHS
    // =========================================================================
    state->bw_h = compute_bandwidth(state, state->d_h);
    state->bw_mu = compute_bandwidth(state, state->d_mu_tilde);
    state->bw_rho = compute_bandwidth(state, state->d_rho_tilde);
    state->bw_sigma = compute_bandwidth(state, state->d_sigma_tilde);
    
    // =========================================================================
    // 3. STEIN ITERATIONS
    // =========================================================================
    size_t smem_size = 8 * n * sizeof(float);
    
    for (int s = 0; s < cfg->n_stein_steps; s++) {
        // Gradient
        svpf_joint_gradient_kernel<<<grid, block, 0, stream>>>(
            state->d_h,
            state->d_h_prev,
            state->d_mu_tilde,
            state->d_rho_tilde,
            state->d_sigma_tilde,
            state->d_grad_h,
            state->d_grad_mu,
            state->d_grad_rho,
            state->d_grad_sigma,
            state->d_log_w,
            y_t,
            cfg->nu,
            cfg->student_t_const,
            cfg->lik_offset,
            cfg->mu_prior_mean, cfg->mu_prior_var,
            cfg->rho_prior_mean, cfg->rho_prior_var,
            cfg->sigma_prior_mean, cfg->sigma_prior_var,
            cfg->prior_weight,
            n
        );
        
        // Stein transport (with surprise-boosted step size)
        svpf_joint_stein_kernel<<<1, block, smem_size, stream>>>(
            state->d_h,
            state->d_mu_tilde,
            state->d_rho_tilde,
            state->d_sigma_tilde,
            state->d_grad_h,
            state->d_grad_mu,
            state->d_grad_rho,
            state->d_grad_sigma,
            state->d_h_prev,  // For surprise detection
            y_t,              // For surprise detection
            state->bw_h, state->bw_mu, state->bw_rho, state->bw_sigma,
            cfg->step_h, cfg->step_mu, cfg->step_rho, cfg->step_sigma,
            n
        );
    }
    
    // =========================================================================
    // 4. EXTRACT DIAGNOSTICS
    // =========================================================================
    svpf_joint_extract_kernel<<<1, 256, 0, stream>>>(
        state->d_h,
        state->d_mu_tilde,
        state->d_rho_tilde,
        state->d_sigma_tilde,
        state->d_param_mean,
        state->d_param_std,
        state->d_std_unconstrained,
        state->d_collapse_flags,
        cfg->collapse_thresh_mu,
        cfg->collapse_thresh_rho,
        cfg->collapse_thresh_sigma,
        n
    );
    
    // =========================================================================
    // 5. COPY TO HOST (if diagnostics requested)
    // =========================================================================
    if (diag) {
        float param_mean[4], param_std[3], std_unc[3];
        int collapse[3];
        
        cudaMemcpyAsync(param_mean, state->d_param_mean, 4 * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(param_std, state->d_param_std, 3 * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(std_unc, state->d_std_unconstrained, 3 * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(collapse, state->d_collapse_flags, 3 * sizeof(int), 
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // h_mean is precomputed in extract kernel (slot [3])
        diag->h_mean = param_mean[3];
        diag->vol_mean = expf(diag->h_mean * 0.5f);
        
        diag->mu_mean = param_mean[0];
        diag->rho_mean = param_mean[1];
        diag->sigma_mean = param_mean[2];
        
        diag->mu_std = param_std[0];
        diag->rho_std = param_std[1];
        diag->sigma_std = param_std[2];
        
        diag->std_mu_tilde = std_unc[0];
        diag->std_rho_tilde = std_unc[1];
        diag->std_sigma_tilde = std_unc[2];
        
        diag->mu_collapsed = collapse[0];
        diag->rho_collapsed = collapse[1];
        diag->sigma_collapsed = collapse[2];
        
        // ESS computation
        float* log_w_host = (float*)malloc(n * sizeof(float));
        cudaMemcpy(log_w_host, state->d_log_w, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_log_w = log_w_host[0];
        for (int i = 1; i < n; i++) {
            if (log_w_host[i] > max_log_w) max_log_w = log_w_host[i];
        }
        
        float sum_w = 0.0f, sum_w2 = 0.0f;
        for (int i = 0; i < n; i++) {
            float w = expf(log_w_host[i] - max_log_w);
            sum_w += w;
            sum_w2 += w * w;
        }
        free(log_w_host);
        
        diag->ess = (sum_w * sum_w) / (sum_w2 + 1e-10f);
        diag->log_likelihood = max_log_w + logf(sum_w) - logf((float)n);
        
        // Collapse warning
        if (diag->mu_collapsed || diag->rho_collapsed || diag->sigma_collapsed) {
            printf("WARNING [t=%d]: Diversity collapsed - mu:%d rho:%d sigma:%d\n",
                   state->timestep, diag->mu_collapsed, diag->rho_collapsed, diag->sigma_collapsed);
        }
    }
    
    // Update state
    state->y_prev = y_t;
    state->timestep++;
}

// =============================================================================
// GETTERS
// =============================================================================

float svpf_joint_get_vol(const SVPFJointState* state) {
    // h_mean is precomputed in extract kernel (slot [3])
    float param_mean[4];
    cudaMemcpy(param_mean, state->d_param_mean, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    return expf(param_mean[3] * 0.5f);
}

void svpf_joint_get_params(const SVPFJointState* state, float* mu, float* rho, float* sigma) {
    float param_mean[4];
    cudaMemcpy(param_mean, state->d_param_mean, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (mu) *mu = param_mean[0];
    if (rho) *rho = param_mean[1];
    if (sigma) *sigma = param_mean[2];
}

// =============================================================================
// CLEANUP
// =============================================================================

void svpf_joint_destroy(SVPFJointState* state) {
    if (!state) return;
    
    cudaFree(state->d_h);
    cudaFree(state->d_h_prev);
    cudaFree(state->d_mu_tilde);
    cudaFree(state->d_rho_tilde);
    cudaFree(state->d_sigma_tilde);
    
    cudaFree(state->d_grad_h);
    cudaFree(state->d_grad_mu);
    cudaFree(state->d_grad_rho);
    cudaFree(state->d_grad_sigma);
    
    cudaFree(state->d_log_w);
    cudaFree(state->d_rng);
    cudaFree(state->d_temp);
    cudaFree(state->d_reduce_out);
    
    cudaFree(state->d_param_mean);
    cudaFree(state->d_param_std);
    cudaFree(state->d_std_unconstrained);
    cudaFree(state->d_collapse_flags);
    
    free(state);
}
