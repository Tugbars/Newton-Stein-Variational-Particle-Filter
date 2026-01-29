/**
 * @file test_fully_fused.cu
 * @brief Test the fully fused single-kernel SVPF step
 */

#include "svpf.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Declare the fully fused function
extern "C" cudaError_t svpf_fully_fused_step(
    float* h, float* h_prev, float* grad_log_p, float* log_weights,
    float* d_grad_v, curandStatePhilox4_32_10_t* rng,
    float y_t, float y_prev, float h_mean_prev,
    float* d_h_mean, float* d_vol, float* d_loglik, float* d_bandwidth, float* d_ksd,
    float rho, float sigma_z, float mu, float nu, float lik_offset, float student_t_const,
    float mim_jump_prob, float mim_jump_scale,
    float guide_strength, float guide_mean, int use_guide,
    float step_size, float temperature, float rmsprop_rho, float rmsprop_eps,
    int n_stein_steps, int n_anneal_steps, int stein_sign_mode,
    int n, cudaStream_t stream
);

extern "C" bool svpf_fully_fused_supported(int n_particles);

// RNG init kernel
__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SVPF Fully Fused Kernel Test                                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    int n = 512;
    int n_filters = 16;
    int n_steps = 1000;
    int warmup = 100;
    
    printf("Config: %d particles, %d filters\n\n", n, n_filters);
    
    if (!svpf_fully_fused_supported(n)) {
        printf("ERROR: N=%d not supported (max 1024)\n", n);
        return 1;
    }
    
    // Allocate device memory for one filter
    float *d_h, *d_h_prev, *d_grad, *d_logw, *d_grad_v;
    float *d_h_mean, *d_vol, *d_loglik, *d_bandwidth, *d_ksd;
    curandStatePhilox4_32_10_t* d_rng;
    
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMalloc(&d_h_prev, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_logw, n * sizeof(float));
    cudaMalloc(&d_grad_v, n * sizeof(float));
    cudaMalloc(&d_rng, n * sizeof(curandStatePhilox4_32_10_t));
    
    cudaMalloc(&d_h_mean, sizeof(float));
    cudaMalloc(&d_vol, sizeof(float));
    cudaMalloc(&d_loglik, sizeof(float));
    cudaMalloc(&d_bandwidth, sizeof(float));
    cudaMalloc(&d_ksd, sizeof(float));
    
    // Initialize
    cudaMemset(d_grad_v, 0, n * sizeof(float));
    float init_h = -4.5f;
    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_h + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h_prev + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    init_rng_kernel<<<(n+255)/256, 256>>>(d_rng, n, 12345);
    cudaDeviceSynchronize();
    
    // Parameters
    float rho = 0.98f, sigma_z = 0.10f, mu = -4.5f, nu = 5.0f, lik_offset = 0.0f;
    float student_t_const = lgammaf((nu + 1.0f) / 2.0f) - lgammaf(nu / 2.0f) - 0.5f * logf(3.14159f * nu);
    float mim_jump_prob = 0.05f, mim_jump_scale = 5.0f;
    float guide_strength = 0.0f, guide_mean = -4.5f;
    int use_guide = 0;
    float step_size = 0.25f, temperature = 1.0f, rmsprop_rho_val = 0.9f, rmsprop_eps = 1e-6f;
    int n_stein_steps = 8, n_anneal_steps = 3, stein_sign_mode = 0;
    
    // Generate test returns
    float* returns = (float*)malloc(n_steps * sizeof(float));
    srand(12345);
    for (int t = 0; t < n_steps; t++) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        returns[t] = 0.02f * z;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // =========================================================================
    // Test 1: Single Filter Latency (Fully Fused)
    // =========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Test 1: Fully Fused Kernel - Single Filter\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    float y_prev = 0.0f;
    float h_mean_prev = -4.5f;
    
    // Warmup
    for (int t = 0; t < warmup; t++) {
        svpf_fully_fused_step(
            d_h, d_h_prev, d_grad, d_logw, d_grad_v, d_rng,
            returns[t], y_prev, h_mean_prev,
            d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd,
            rho, sigma_z, mu, nu, lik_offset, student_t_const,
            mim_jump_prob, mim_jump_scale,
            guide_strength, guide_mean, use_guide,
            step_size, temperature, rmsprop_rho_val, rmsprop_eps,
            n_stein_steps, n_anneal_steps, stein_sign_mode,
            n, 0
        );
        cudaMemcpy(&h_mean_prev, d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
        y_prev = returns[t];
    }
    cudaDeviceSynchronize();
    
    // Profile
    float total_time = 0.0f;
    for (int t = warmup; t < n_steps; t++) {
        cudaEventRecord(start);
        
        svpf_fully_fused_step(
            d_h, d_h_prev, d_grad, d_logw, d_grad_v, d_rng,
            returns[t], y_prev, h_mean_prev,
            d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd,
            rho, sigma_z, mu, nu, lik_offset, student_t_const,
            mim_jump_prob, mim_jump_scale,
            guide_strength, guide_mean, use_guide,
            step_size, temperature, rmsprop_rho_val, rmsprop_eps,
            n_stein_steps, n_anneal_steps, stein_sign_mode,
            n, 0
        );
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        
        cudaMemcpy(&h_mean_prev, d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
        y_prev = returns[t];
    }
    
    int measured = n_steps - warmup;
    float avg_fused = (total_time / measured) * 1000.0f;
    
    printf("  Fully fused single kernel: %.1f μs/step\n", avg_fused);
    printf("  Throughput: %.0f steps/sec\n\n", 1e6 / avg_fused);
    
    // =========================================================================
    // Test 2: Compare with Standard svpf_step_graph
    // =========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Test 2: Standard svpf_step_graph (for comparison)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    SVPFState* state = svpf_create(n, 8, nu, nullptr);
    SVPFParams params = {rho, sigma_z, mu, 0.0f};
    state->use_persistent_kernel = 0;
    svpf_initialize(state, &params, 12345);
    
    // Warmup
    y_prev = 0.0f;
    for (int t = 0; t < warmup; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        y_prev = returns[t];
    }
    cudaDeviceSynchronize();
    
    // Profile
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
    
    printf("  Standard multi-kernel: %.1f μs/step\n", avg_std);
    printf("  Speedup from fusion: %.2fx\n\n", avg_std / avg_fused);
    
    svpf_destroy(state);
    
    // =========================================================================
    // Test 3: Parallel Multi-Filter (Fully Fused)
    // =========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Test 3: Parallel Multi-Filter (Fully Fused)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Allocate for all filters
    struct FilterState {
        float *d_h, *d_h_prev, *d_grad, *d_logw, *d_grad_v;
        float *d_h_mean, *d_vol, *d_loglik, *d_bandwidth, *d_ksd;
        curandStatePhilox4_32_10_t* d_rng;
        cudaStream_t stream;
        float y_prev, h_mean_prev;
    };
    
    FilterState* filters = new FilterState[n_filters];
    
    for (int f = 0; f < n_filters; f++) {
        cudaStreamCreate(&filters[f].stream);
        
        cudaMalloc(&filters[f].d_h, n * sizeof(float));
        cudaMalloc(&filters[f].d_h_prev, n * sizeof(float));
        cudaMalloc(&filters[f].d_grad, n * sizeof(float));
        cudaMalloc(&filters[f].d_logw, n * sizeof(float));
        cudaMalloc(&filters[f].d_grad_v, n * sizeof(float));
        cudaMalloc(&filters[f].d_rng, n * sizeof(curandStatePhilox4_32_10_t));
        
        cudaMalloc(&filters[f].d_h_mean, sizeof(float));
        cudaMalloc(&filters[f].d_vol, sizeof(float));
        cudaMalloc(&filters[f].d_loglik, sizeof(float));
        cudaMalloc(&filters[f].d_bandwidth, sizeof(float));
        cudaMalloc(&filters[f].d_ksd, sizeof(float));
        
        cudaMemset(filters[f].d_grad_v, 0, n * sizeof(float));
        for (int i = 0; i < n; i++) {
            cudaMemcpy(filters[f].d_h + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(filters[f].d_h_prev + i, &init_h, sizeof(float), cudaMemcpyHostToDevice);
        }
        
        init_rng_kernel<<<(n+255)/256, 256, 0, filters[f].stream>>>(filters[f].d_rng, n, 12345 + f);
        
        filters[f].y_prev = 0.0f;
        filters[f].h_mean_prev = -4.5f;
    }
    cudaDeviceSynchronize();
    
    // Warmup
    for (int t = 0; t < warmup; t++) {
        for (int f = 0; f < n_filters; f++) {
            svpf_fully_fused_step(
                filters[f].d_h, filters[f].d_h_prev, filters[f].d_grad, filters[f].d_logw,
                filters[f].d_grad_v, filters[f].d_rng,
                returns[t], filters[f].y_prev, filters[f].h_mean_prev,
                filters[f].d_h_mean, filters[f].d_vol, filters[f].d_loglik,
                filters[f].d_bandwidth, filters[f].d_ksd,
                rho, sigma_z, mu, nu, lik_offset, student_t_const,
                mim_jump_prob, mim_jump_scale,
                guide_strength, guide_mean, use_guide,
                step_size, temperature, rmsprop_rho_val, rmsprop_eps,
                n_stein_steps, n_anneal_steps, stein_sign_mode,
                n, filters[f].stream
            );
        }
        cudaDeviceSynchronize();
        for (int f = 0; f < n_filters; f++) {
            cudaMemcpy(&filters[f].h_mean_prev, filters[f].d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
            filters[f].y_prev = returns[t];
        }
    }
    
    // Profile parallel execution
    total_time = 0.0f;
    for (int t = warmup; t < n_steps; t++) {
        cudaEventRecord(start);
        
        // Launch ALL filters in parallel (no sync between!)
        for (int f = 0; f < n_filters; f++) {
            svpf_fully_fused_step(
                filters[f].d_h, filters[f].d_h_prev, filters[f].d_grad, filters[f].d_logw,
                filters[f].d_grad_v, filters[f].d_rng,
                returns[t], filters[f].y_prev, filters[f].h_mean_prev,
                filters[f].d_h_mean, filters[f].d_vol, filters[f].d_loglik,
                filters[f].d_bandwidth, filters[f].d_ksd,
                rho, sigma_z, mu, nu, lik_offset, student_t_const,
                mim_jump_prob, mim_jump_scale,
                guide_strength, guide_mean, use_guide,
                step_size, temperature, rmsprop_rho_val, rmsprop_eps,
                n_stein_steps, n_anneal_steps, stein_sign_mode,
                n, filters[f].stream
            );
        }
        
        // Single sync for ALL filters
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        
        // Update state (would be async in real system)
        for (int f = 0; f < n_filters; f++) {
            cudaMemcpy(&filters[f].h_mean_prev, filters[f].d_h_mean, sizeof(float), cudaMemcpyDeviceToHost);
            filters[f].y_prev = returns[t];
        }
    }
    
    float avg_parallel = (total_time / measured) * 1000.0f;
    
    printf("  %d filters parallel: %.1f μs total\n", n_filters, avg_parallel);
    printf("  Effective per-filter: %.1f μs\n", avg_parallel / n_filters);
    printf("  Parallelism efficiency: %.1f%% (ideal=%.1f%%)\n", 
           (avg_fused / (avg_parallel / n_filters)) * 100.0f / n_filters,
           100.0f);
    printf("  Throughput: %.0f filter-steps/sec\n\n", n_filters * 1e6 / avg_parallel);
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("  Configuration              | Latency    | vs Standard\n");
    printf("  ---------------------------+------------+------------\n");
    printf("  Standard (multi-kernel)    | %7.1f μs | 1.00x\n", avg_std);
    printf("  Fully Fused (single kernel)| %7.1f μs | %.2fx\n", avg_fused, avg_std / avg_fused);
    printf("  %d Filters Parallel        | %7.1f μs | %.2fx (total)\n", 
           n_filters, avg_parallel, avg_std * n_filters / avg_parallel);
    printf("\n");
    
    if (avg_fused < 100.0f) {
        printf("  ✓ SPY-ready: %.1f μs < 100 μs\n", avg_fused);
    } else {
        printf("  ✗ Need more work: %.1f μs > 100 μs target\n", avg_fused);
    }
    
    // Cleanup
    for (int f = 0; f < n_filters; f++) {
        cudaFree(filters[f].d_h);
        cudaFree(filters[f].d_h_prev);
        cudaFree(filters[f].d_grad);
        cudaFree(filters[f].d_logw);
        cudaFree(filters[f].d_grad_v);
        cudaFree(filters[f].d_rng);
        cudaFree(filters[f].d_h_mean);
        cudaFree(filters[f].d_vol);
        cudaFree(filters[f].d_loglik);
        cudaFree(filters[f].d_bandwidth);
        cudaFree(filters[f].d_ksd);
        cudaStreamDestroy(filters[f].stream);
    }
    delete[] filters;
    
    cudaFree(d_h);
    cudaFree(d_h_prev);
    cudaFree(d_grad);
    cudaFree(d_logw);
    cudaFree(d_grad_v);
    cudaFree(d_rng);
    cudaFree(d_h_mean);
    cudaFree(d_vol);
    cudaFree(d_loglik);
    cudaFree(d_bandwidth);
    cudaFree(d_ksd);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(returns);
    
    return 0;
}
