/**
 * @file test_fully_fused_v2.cu
 * @brief Test fully fused v2 kernel - latency AND accuracy comparison
 */

#include "svpf.cuh"
#include "svpf_fully_fused_v2.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cmath>

__global__ void init_rng_v2_kernel(curandStatePhilox4_32_10_t* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SVPF Fully Fused V2 - Complete Feature Test                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    int n = 512;
    int n_steps = 2000;
    int warmup = 200;
    
    printf("Config: %d particles, %d steps\n\n", n, n_steps);
    
    // Allocate
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
    
    // Initialize
    float init_h = -4.5f;
    float init_mu = -4.5f;
    float init_zero = 0.0f;
    cudaMemset(d_grad_v, 0, n * sizeof(float));
    cudaMemset(d_inv_hess, 0, n * sizeof(float));
    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_h + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h_prev + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_mu_state, &init_mu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_guide_mean, &init_zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_guide_var, &init_zero, sizeof(float), cudaMemcpyHostToDevice);
    init_rng_v2_kernel<<<(n+255)/256, 256>>>(d_rng, n, 12345);
    cudaDeviceSynchronize();
    
    // Parameters
    float rho = 0.98f, sigma_z = 0.10f, mu = -4.5f, nu = 5.0f;
    float lik_offset = 0.345f;
    float gamma = 0.0f;  // Match standard test (no leverage in synthetic data)
    float student_t_const = lgammaf((nu + 1.0f) / 2.0f) - lgammaf(nu / 2.0f) - 0.5f * logf(3.14159f * nu);
    float delta_rho = 0.0f, delta_sigma = 0.0f;
    float mim_jump_prob = 0.25f, mim_jump_scale = 9.0f;
    float guide_strength_base = 0.05f, guide_strength_max = 0.30f, guide_innovation_threshold = 1.0f;
    float guided_alpha_base = 0.0f, guided_alpha_shock = 0.40f, guided_innov_thresh = 1.5f;
    int use_guide = 1, use_guided_predict = 1, use_guide_preserving = 1;
    int use_newton = 1, use_full_newton = 1;
    int use_rejuvenation = 1;
    float rejuv_ksd_threshold = 0.05f, rejuv_prob = 0.30f, rejuv_blend = 0.30f;
    int use_adaptive_mu = 0;
    float mu_ema_alpha = 0.01f;
    int use_adaptive_sigma = 1;
    float sigma_boost_threshold = 0.95f, sigma_boost_max = 3.2f;
    float step_size = 0.25f, temperature = 0.45f, rmsprop_rho_val = 0.9f, rmsprop_eps = 1e-6f;
    int n_stein_steps = 10, n_anneal_steps = 5, stein_sign_mode = 0;
    
    // Generate data
    std::vector<float> true_h(n_steps), returns(n_steps);
    srand(42);
    true_h[0] = mu;
    for (int t = 0; t < n_steps; t++) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159f * u2);
        if (t > 0) true_h[t] = mu + rho * (true_h[t-1] - mu) + sigma_z * z1;
        returns[t] = expf(true_h[t] * 0.5f) * z2;
    }
    
    printf("DGP: rho=%.2f, sigma=%.2f, mu=%.1f\n\n", rho, sigma_z, mu);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ==========================================================================
    // Test 1: Fused V2 Latency
    // ==========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Test 1: Fused V2 - Latency\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    float y_prev = 0.0f, h_mean_prev = mu, vol_prev = expf(mu * 0.5f), ksd_prev = 0.1f;
    
    for (int t = 0; t < warmup; t++) {
        svpf_fully_fused_step_v2(
            d_h, d_h_prev, d_grad, d_logw, d_grad_v, d_inv_hess, d_rng,
            returns[t], y_prev, h_mean_prev, vol_prev, ksd_prev,
            d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd, d_guide_mean, d_guide_var,
            rho, sigma_z, mu, nu, lik_offset, student_t_const, gamma,
            delta_rho, delta_sigma, mim_jump_prob, mim_jump_scale,
            guide_strength_base, guide_strength_max, guide_innovation_threshold,
            guided_alpha_base, guided_alpha_shock, guided_innov_thresh,
            use_guide, use_guided_predict, use_guide_preserving,
            use_newton, use_full_newton,
            use_rejuvenation, rejuv_ksd_threshold, rejuv_prob, rejuv_blend,
            use_adaptive_mu, d_mu_state, mu_ema_alpha,
            use_adaptive_sigma, sigma_boost_threshold, sigma_boost_max,
            step_size, temperature, rmsprop_rho_val, rmsprop_eps,
            n_stein_steps, n_anneal_steps, stein_sign_mode,
            n, t, 0
        );
        cudaMemcpy(&h_mean_prev, d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&vol_prev, d_vol, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ksd_prev, d_ksd, sizeof(float), cudaMemcpyDeviceToHost);
        y_prev = returns[t];
    }
    cudaDeviceSynchronize();
    
    float total_time = 0.0f;
    for (int t = warmup; t < n_steps; t++) {
        cudaEventRecord(start);
        svpf_fully_fused_step_v2(
            d_h, d_h_prev, d_grad, d_logw, d_grad_v, d_inv_hess, d_rng,
            returns[t], y_prev, h_mean_prev, vol_prev, ksd_prev,
            d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd, d_guide_mean, d_guide_var,
            rho, sigma_z, mu, nu, lik_offset, student_t_const, gamma,
            delta_rho, delta_sigma, mim_jump_prob, mim_jump_scale,
            guide_strength_base, guide_strength_max, guide_innovation_threshold,
            guided_alpha_base, guided_alpha_shock, guided_innov_thresh,
            use_guide, use_guided_predict, use_guide_preserving,
            use_newton, use_full_newton,
            use_rejuvenation, rejuv_ksd_threshold, rejuv_prob, rejuv_blend,
            use_adaptive_mu, d_mu_state, mu_ema_alpha,
            use_adaptive_sigma, sigma_boost_threshold, sigma_boost_max,
            step_size, temperature, rmsprop_rho_val, rmsprop_eps,
            n_stein_steps, n_anneal_steps, stein_sign_mode,
            n, t, 0
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        cudaMemcpy(&h_mean_prev, d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&vol_prev, d_vol, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ksd_prev, d_ksd, sizeof(float), cudaMemcpyDeviceToHost);
        y_prev = returns[t];
    }
    
    int measured = n_steps - warmup;
    float avg_fused_v2 = (total_time / measured) * 1000.0f;
    printf("  Fused V2 latency: %.1f μs/step\n\n", avg_fused_v2);
    
    // ==========================================================================
    // Test 2: Fused V2 Accuracy
    // ==========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Test 2: Fused V2 - Accuracy\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Reset
    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_h + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h_prev + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemset(d_grad_v, 0, n * sizeof(float));
    cudaMemcpy(d_mu_state, &init_mu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_guide_mean, &init_zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_guide_var, &init_zero, sizeof(float), cudaMemcpyHostToDevice);
    init_rng_v2_kernel<<<(n+255)/256, 256>>>(d_rng, n, 12345);
    cudaDeviceSynchronize();
    
    y_prev = 0.0f; h_mean_prev = mu; vol_prev = expf(mu * 0.5f); ksd_prev = 0.1f;
    std::vector<float> fused_v2_h(n_steps);
    
    for (int t = 0; t < n_steps; t++) {
        svpf_fully_fused_step_v2(
            d_h, d_h_prev, d_grad, d_logw, d_grad_v, d_inv_hess, d_rng,
            returns[t], y_prev, h_mean_prev, vol_prev, ksd_prev,
            d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd, d_guide_mean, d_guide_var,
            rho, sigma_z, mu, nu, lik_offset, student_t_const, gamma,
            delta_rho, delta_sigma, mim_jump_prob, mim_jump_scale,
            guide_strength_base, guide_strength_max, guide_innovation_threshold,
            guided_alpha_base, guided_alpha_shock, guided_innov_thresh,
            use_guide, use_guided_predict, use_guide_preserving,
            use_newton, use_full_newton,
            use_rejuvenation, rejuv_ksd_threshold, rejuv_prob, rejuv_blend,
            use_adaptive_mu, d_mu_state, mu_ema_alpha,
            use_adaptive_sigma, sigma_boost_threshold, sigma_boost_max,
            step_size, temperature, rmsprop_rho_val, rmsprop_eps,
            n_stein_steps, n_anneal_steps, stein_sign_mode,
            n, t, 0
        );
        cudaDeviceSynchronize();
        cudaMemcpy(&h_mean_prev, d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&vol_prev, d_vol, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ksd_prev, d_ksd, sizeof(float), cudaMemcpyDeviceToHost);
        fused_v2_h[t] = h_mean_prev;
        y_prev = returns[t];
    }
    
    float sum_sq_err_v2 = 0.0f;
    for (int t = 100; t < n_steps; t++) {
        float err = fused_v2_h[t] - true_h[t];
        sum_sq_err_v2 += err * err;
    }
    float rmse_v2 = sqrtf(sum_sq_err_v2 / (n_steps - 100));
    printf("  Fused V2 RMSE(h): %.4f\n", rmse_v2);
    
    // ==========================================================================
    // Test 3: Standard SVPF (configured to match fused V2 parameters)
    // ==========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Test 3: Standard SVPF - Reference\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    SVPFState* state = svpf_create(n, 8, nu, nullptr);
    SVPFParams params = {rho, sigma_z, mu, gamma};
    
    // Match feature flags
    state->use_guide = use_guide;
    state->use_guided = use_guided_predict;
    state->use_guide_preserving = use_guide_preserving;
    state->use_newton = use_newton;
    state->use_full_newton = use_full_newton;
    state->use_rejuvenation = use_rejuvenation;
    state->use_mim = 1;
    state->use_adaptive_sigma = use_adaptive_sigma;
    
    // Match numeric parameters that exist in SVPFState
    // Note: Some parameters use internal defaults - not all are exposed in struct
    state->lik_offset = lik_offset;
    state->mim_jump_prob = mim_jump_prob;
    state->mim_jump_scale = mim_jump_scale;
    state->guide_strength_base = guide_strength_base;
    state->guide_strength_max = guide_strength_max;
    state->guide_innovation_threshold = guide_innovation_threshold;
    state->guided_alpha_base = guided_alpha_base;
    state->guided_alpha_shock = guided_alpha_shock;
    state->guided_innovation_threshold = guided_innov_thresh;
    state->sigma_boost_threshold = sigma_boost_threshold;
    state->sigma_boost_max = sigma_boost_max;
    state->rejuv_ksd_threshold = rejuv_ksd_threshold;
    state->rejuv_prob = rejuv_prob;
    state->rejuv_blend = rejuv_blend;
    // Note: step_size, temperature, rmsprop_rho, n_stein_steps, n_anneal_steps
    // use internal defaults in svpf_step_graph - not exposed in SVPFState
    
    svpf_initialize(state, &params, 12345);
    
    std::vector<float> std_h(n_steps);
    y_prev = 0.0f;
    for (int t = 0; t < n_steps; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        std_h[t] = h_mean;
        y_prev = returns[t];
    }
    
    float sum_sq_err_std = 0.0f;
    for (int t = 100; t < n_steps; t++) {
        float err = std_h[t] - true_h[t];
        sum_sq_err_std += err * err;
    }
    float rmse_std = sqrtf(sum_sq_err_std / (n_steps - 100));
    printf("  Standard RMSE(h): %.4f\n", rmse_std);
    
    // Latency
    y_prev = 0.0f;
    svpf_initialize(state, &params, 12345);
    for (int t = 0; t < warmup; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        y_prev = returns[t];
    }
    cudaDeviceSynchronize();
    
    total_time = 0.0f;
    for (int t = warmup; t < n_steps; t++) {
        cudaEventRecord(start);
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        y_prev = returns[t];
    }
    float avg_std = (total_time / measured) * 1000.0f;
    printf("  Standard latency: %.1f μs/step\n", avg_std);
    
    svpf_destroy(state);
    
    // Summary
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("  Method        | Latency    | RMSE(h) | Speedup\n");
    printf("  --------------+------------+---------+--------\n");
    printf("  Standard      | %7.1f μs | %.4f  | 1.00x\n", avg_std, rmse_std);
    printf("  Fused V2      | %7.1f μs | %.4f  | %.2fx\n", avg_fused_v2, rmse_v2, avg_std / avg_fused_v2);
    printf("\n");
    
    float ratio = rmse_v2 / rmse_std;
    if (ratio < 1.1f) printf("  ✓ Accuracy matches (%.0f%% of ref)\n", ratio * 100);
    else printf("  ✗ Accuracy %.0f%% worse\n", (ratio - 1.0f) * 100);
    
    if (avg_fused_v2 < 100) printf("  ✓ SPY-ready: %.1f μs\n", avg_fused_v2);
    else if (avg_fused_v2 < 150) printf("  ~ Close: %.1f μs\n", avg_fused_v2);
    else printf("  ✗ Latency: %.1f μs\n", avg_fused_v2);
    
    // Cleanup
    cudaFree(d_h); cudaFree(d_h_prev); cudaFree(d_grad); cudaFree(d_logw);
    cudaFree(d_grad_v); cudaFree(d_inv_hess); cudaFree(d_rng);
    cudaFree(d_h_mean); cudaFree(d_vol); cudaFree(d_loglik); cudaFree(d_bandwidth);
    cudaFree(d_ksd); cudaFree(d_mu_state); cudaFree(d_guide_mean); cudaFree(d_guide_var);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
