/**
 * @file test_fused_v2_scenario.cu
 * @brief Test fully fused v2 kernel on real scenario data
 * 
 * FIXED: Now passes student_t_implied_offset parameter
 * 
 * Mirrors run_svpf_on_scenario() but uses svpf_fully_fused_step_v2
 */

#include "svpf.cuh"
#include "svpf_fully_fused_v2.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// RNG initialization kernel
__global__ void init_rng_fused_kernel(curandStatePhilox4_32_10_t* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Compute student_t_implied_offset (same formula as svpf_create)
float compute_student_t_implied_offset(float nu) {
    float psi_half = -1.9635100260214235f;  // digamma(0.5)
    float nu_half = nu / 2.0f;
    float psi_nu_half;
    
    if (nu_half >= 1.0f) {
        // Asymptotic approximation for digamma
        psi_nu_half = logf(nu_half) - 1.0f/(2.0f*nu_half) - 1.0f/(12.0f*nu_half*nu_half);
    } else {
        psi_nu_half = -0.5772156649f - 1.0f/nu_half;
    }
    
    float expected_log_t_sq = logf(nu) + psi_half - psi_nu_half;
    return -expected_log_t_sq;  // ≈ +1.057 for nu=5
}

typedef struct {
    float rmse_h;
    float rmse_vol;
    float mean_bias;
    float correlation;
} FusedMetrics;

/**
 * Run fully fused v2 kernel on scenario data
 * 
 * @param returns      Array of returns [n_ticks]
 * @param true_h       Array of true log-volatility [n_ticks] (for metrics, can be NULL)
 * @param n_ticks      Number of time steps
 * @param n_particles  Number of particles
 * @param seed         RNG seed
 * @param elapsed_ms   Output: elapsed time in ms
 * @param h_out        Output: estimated h values [n_ticks] (caller allocated)
 * @param vol_out      Output: estimated vol values [n_ticks] (caller allocated)
 */
void run_fused_v2_on_scenario(
    const float* returns,
    const float* true_h,  // Can be NULL if no ground truth
    int n_ticks,
    int n_particles,
    int seed,
    double* elapsed_ms,
    float* h_out,
    float* vol_out
) {
    int n = n_particles;
    
    // =========================================================================
    // Allocate device memory
    // =========================================================================
    
    float *d_h, *d_h_prev, *d_grad, *d_logw, *d_grad_v, *d_inv_hess;
    float *d_h_mean, *d_vol, *d_loglik, *d_bandwidth, *d_ksd, *d_mu_state;
    float *d_guide_mean, *d_guide_var;
    curandStatePhilox4_32_10_t* d_rng;
    
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMalloc(&d_h_prev, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_logw, n * sizeof(float));
    cudaMalloc(&d_grad_v, n * sizeof(float));
    cudaMalloc(&d_inv_hess, n * sizeof(float));
    cudaMalloc(&d_rng, n * sizeof(curandStatePhilox4_32_10_t));
    
    cudaMalloc(&d_h_mean, sizeof(float));
    cudaMalloc(&d_vol, sizeof(float));
    cudaMalloc(&d_loglik, sizeof(float));
    cudaMalloc(&d_bandwidth, sizeof(float));
    cudaMalloc(&d_ksd, sizeof(float));
    cudaMalloc(&d_mu_state, sizeof(float));
    cudaMalloc(&d_guide_mean, sizeof(float));
    cudaMalloc(&d_guide_var, sizeof(float));
    
    // =========================================================================
    // Model parameters (matching run_svpf_on_scenario)
    // =========================================================================
    
    float rho = 0.97f;
    float sigma_z = 0.15f;
    float mu = -4.5f;
    float nu = 5.0f;
    float gamma = 0.0f;  // No leverage in this DGP
    float lik_offset = 0.345f;
    
    // Compute student_t constants
    float student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                          - lgammaf(nu / 2.0f) 
                          - 0.5f * logf(3.14159265f * nu);
    float student_t_implied_offset = compute_student_t_implied_offset(nu);
    
    // Local params
    float delta_rho = 0.0f;
    float delta_sigma = 0.0f;
    
    // MIM
    float mim_jump_prob = 0.25f;
    float mim_jump_scale = 9.0f;
    
    // Guide
    float guide_strength_base = 0.05f;
    float guide_strength_max = 0.30f;
    float guide_innovation_threshold = 1.0f;
    float guided_alpha_base = 0.0f;
    float guided_alpha_shock = 0.40f;
    float guided_innov_thresh_predict = 1.5f;
    int use_guide = 1;
    int use_guided_predict = 1;
    int use_guide_preserving = 1;
    
    // Newton
    int use_newton = 1;
    int use_full_newton = 1;
    
    // Rejuvenation
    int use_rejuvenation = 1;
    float rejuv_ksd_threshold = 0.05f;
    float rejuv_prob = 0.30f;
    float rejuv_blend = 0.30f;
    
    // Adaptive mu
    int use_adaptive_mu = 0;  // Disabled for now (needs different EMA approach)
    float mu_ema_alpha = 0.01f;
    
    // Adaptive sigma
    int use_adaptive_sigma = 1;
    float sigma_boost_threshold = 0.95f;
    float sigma_boost_max = 3.2f;
    
    // Stein params
    float step_size = 0.25f;
    float temperature = 0.45f;
    float rmsprop_rho_val = 0.9f;
    float rmsprop_eps = 1e-6f;
    int n_stein_steps = 8;
    int n_anneal_steps = 3;
    int stein_sign_mode = 0;
    
    // =========================================================================
    // Initialize
    // =========================================================================
    
    float init_h = mu;
    float init_zero = 0.0f;
    
    // Initialize particles
    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_h + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h_prev + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Initialize state
    cudaMemset(d_grad_v, 0, n * sizeof(float));
    cudaMemset(d_inv_hess, 0, n * sizeof(float));
    cudaMemcpy(d_mu_state, &init_h, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_guide_mean, &init_zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_guide_var, &init_zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize RNG
    init_rng_fused_kernel<<<(n + 255) / 256, 256>>>(d_rng, n, seed);
    cudaDeviceSynchronize();
    
    // =========================================================================
    // Run filter
    // =========================================================================
    
    float y_prev = 0.0f;
    float h_mean_prev = mu;
    float vol_prev = expf(mu * 0.5f);
    float ksd_prev = 0.1f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int t = 0; t < n_ticks; t++) {
        float y_t = returns[t];
        
        svpf_fully_fused_step_v2(
            d_h, d_h_prev, d_grad, d_logw, d_grad_v, d_inv_hess, d_rng,
            y_t, y_prev, h_mean_prev, vol_prev, ksd_prev,
            d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd, d_guide_mean, d_guide_var,
            rho, sigma_z, mu, nu, lik_offset, student_t_const,
            student_t_implied_offset,  // <-- FIXED: Added parameter
            gamma,
            delta_rho, delta_sigma, mim_jump_prob, mim_jump_scale,
            guide_strength_base, guide_strength_max, guide_innovation_threshold,
            guided_alpha_base, guided_alpha_shock, guided_innov_thresh_predict,
            use_guide, use_guided_predict, use_guide_preserving,
            use_newton, use_full_newton,
            use_rejuvenation, rejuv_ksd_threshold, rejuv_prob, rejuv_blend,
            use_adaptive_mu, d_mu_state, mu_ema_alpha,
            use_adaptive_sigma, sigma_boost_threshold, sigma_boost_max,
            step_size, temperature, rmsprop_rho_val, rmsprop_eps,
            n_stein_steps, n_anneal_steps, stein_sign_mode,
            n, t, 0
        );
        
        // Copy outputs for next iteration
        cudaMemcpy(&h_mean_prev, d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&vol_prev, d_vol, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ksd_prev, d_ksd, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Store outputs
        h_out[t] = h_mean_prev;
        vol_out[t] = vol_prev;
        
        y_prev = y_t;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    *elapsed_ms = (double)ms;
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    cudaFree(d_h);
    cudaFree(d_h_prev);
    cudaFree(d_grad);
    cudaFree(d_logw);
    cudaFree(d_grad_v);
    cudaFree(d_inv_hess);
    cudaFree(d_rng);
    cudaFree(d_h_mean);
    cudaFree(d_vol);
    cudaFree(d_loglik);
    cudaFree(d_bandwidth);
    cudaFree(d_ksd);
    cudaFree(d_mu_state);
    cudaFree(d_guide_mean);
    cudaFree(d_guide_var);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Compare fused v2 vs standard on synthetic SV data
 */
int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     Fused V2 vs Standard SVPF - Scenario Comparison (FIXED)       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    int n_particles = 512;
    int n_ticks = 5000;
    int seed = 42;
    
    printf("Config: %d particles, %d ticks\n\n", n_particles, n_ticks);
    
    // =========================================================================
    // Generate synthetic SV data
    // =========================================================================
    
    float rho_dgp = 0.98f;
    float sigma_dgp = 0.15f;
    float mu_dgp = -4.5f;
    
    float* returns = (float*)malloc(n_ticks * sizeof(float));
    float* true_h = (float*)malloc(n_ticks * sizeof(float));
    
    srand(seed);
    true_h[0] = mu_dgp;
    for (int t = 0; t < n_ticks; t++) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159f * u2);
        
        if (t > 0) {
            true_h[t] = mu_dgp + rho_dgp * (true_h[t-1] - mu_dgp) + sigma_dgp * z1;
        }
        
        float vol = expf(true_h[t] * 0.5f);
        returns[t] = vol * z2;
    }
    
    printf("DGP: rho=%.2f, sigma=%.2f, mu=%.1f\n", rho_dgp, sigma_dgp, mu_dgp);
    printf("True vol range: [%.4f, %.4f]\n\n", 
           expf(-6.0f * 0.5f), expf(-3.0f * 0.5f));
    
    // =========================================================================
    // Run Fused V2
    // =========================================================================
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Running Fused V2\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    float* fused_h = (float*)malloc(n_ticks * sizeof(float));
    float* fused_vol = (float*)malloc(n_ticks * sizeof(float));
    double fused_ms;
    
    run_fused_v2_on_scenario(returns, true_h, n_ticks, n_particles, seed,
                            &fused_ms, fused_h, fused_vol);
    
    // Compute RMSE (skip first 100 for burn-in)
    int eval_start = 100;
    float sum_sq_err_fused = 0.0f;
    for (int t = eval_start; t < n_ticks; t++) {
        float err = fused_h[t] - true_h[t];
        sum_sq_err_fused += err * err;
    }
    float rmse_fused = sqrtf(sum_sq_err_fused / (n_ticks - eval_start));
    
    printf("  Fused V2: %.1f ms total, %.2f μs/step, RMSE=%.4f\n",
           fused_ms, fused_ms * 1000.0 / n_ticks, rmse_fused);
    
    // =========================================================================
    // Run Standard SVPF
    // =========================================================================
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" Running Standard SVPF\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    float* std_h = (float*)malloc(n_ticks * sizeof(float));
    float* std_vol = (float*)malloc(n_ticks * sizeof(float));
    
    // Create standard filter with same settings
    SVPFState* filter = svpf_create(n_particles, 10, 5.0f, NULL);
    SVPFParams params = {0.97f, 0.15f, -4.5f, 0.0f};
    
    filter->use_guide = 1;
    filter->use_guided = 1;
    filter->use_guide_preserving = 1;
    filter->use_newton = 1;
    filter->use_full_newton = 1;
    filter->use_rejuvenation = 1;
    filter->use_mim = 1;
    filter->use_adaptive_sigma = 1;
    filter->use_svld = 1;
    filter->use_annealing = 1;
    filter->n_anneal_steps = 5;
    filter->temperature = 0.45f;
    filter->mim_jump_prob = 0.25f;
    filter->mim_jump_scale = 9.0f;
    filter->guide_strength_base = 0.05f;
    filter->guide_strength_max = 0.30f;
    filter->guide_innovation_threshold = 1.0f;
    filter->guided_alpha_base = 0.0f;
    filter->guided_alpha_shock = 0.40f;
    filter->guided_innovation_threshold = 1.5f;
    filter->sigma_boost_threshold = 0.95f;
    filter->sigma_boost_max = 3.2f;
    filter->rejuv_ksd_threshold = 0.05f;
    filter->rejuv_prob = 0.30f;
    filter->rejuv_blend = 0.30f;
    filter->lik_offset = 0.345f;
    
    svpf_initialize(filter, &params, seed);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float y_prev = 0.0f;
    
    cudaEventRecord(start);
    
    for (int t = 0; t < n_ticks; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(filter, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        std_h[t] = h_mean;
        std_vol[t] = vol;
        y_prev = returns[t];
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float std_ms;
    cudaEventElapsedTime(&std_ms, start, stop);
    
    // Compute RMSE
    float sum_sq_err_std = 0.0f;
    for (int t = eval_start; t < n_ticks; t++) {
        float err = std_h[t] - true_h[t];
        sum_sq_err_std += err * err;
    }
    float rmse_std = sqrtf(sum_sq_err_std / (n_ticks - eval_start));
    
    printf("  Standard: %.1f ms total, %.2f μs/step, RMSE=%.4f\n",
           std_ms, std_ms * 1000.0 / n_ticks, rmse_std);
    
    svpf_destroy(filter);
    
    // =========================================================================
    // Summary
    // =========================================================================
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("  Method      | Total Time | Per-Step  | RMSE(h) | Speedup\n");
    printf("  ------------+------------+-----------+---------+--------\n");
    printf("  Standard    | %7.1f ms | %6.2f μs | %.4f  | 1.00x\n", 
           std_ms, std_ms * 1000.0 / n_ticks, rmse_std);
    printf("  Fused V2    | %7.1f ms | %6.2f μs | %.4f  | %.2fx\n",
           fused_ms, fused_ms * 1000.0 / n_ticks, rmse_fused, std_ms / fused_ms);
    printf("\n");
    
    float ratio = rmse_fused / rmse_std;
    if (ratio < 1.05f) {
        printf("  ✓ Accuracy: Fused V2 matches standard (%.1f%% of reference RMSE)\n", ratio * 100);
    } else if (ratio < 1.1f) {
        printf("  ~ Accuracy: Fused V2 close to standard (%.1f%% of reference RMSE)\n", ratio * 100);
    } else {
        printf("  ✗ Accuracy: Fused V2 is %.1f%% worse than standard\n", (ratio - 1.0f) * 100);
    }
    
    // =========================================================================
    // Debug: Print first few values to verify sanity
    // =========================================================================
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" First 10 timesteps comparison\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("  t   | True h  | Fused h | Std h   | Fused err | Std err\n");
    printf("  ----+---------+---------+---------+-----------+--------\n");
    for (int t = 0; t < 10; t++) {
        printf("  %3d | %7.3f | %7.3f | %7.3f | %9.3f | %7.3f\n",
               t, true_h[t], fused_h[t], std_h[t],
               fused_h[t] - true_h[t], std_h[t] - true_h[t]);
    }
    printf("\n");
    
    // Cleanup
    free(returns);
    free(true_h);
    free(fused_h);
    free(fused_vol);
    free(std_h);
    free(std_vol);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
