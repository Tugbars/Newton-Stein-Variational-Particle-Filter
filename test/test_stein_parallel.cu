/**
 * @file test_stein_parallel.cu
 * @brief Correctness and performance test for parallel Stein kernel
 * 
 * Compares:
 *   - Original O(N) threads × O(N) serial loop
 *   - Parallel O(N) blocks × O(256) threads with reduction
 * 
 * Tests:
 *   1. Correctness: parallel output matches serial within epsilon
 *   2. KSD correctness: parallel KSD matches serial KSD
 *   3. Performance: speedup at N=512, 1024, 2048
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Include the parallel implementation
#include "svpf_stein_parallel.cuh"

// =============================================================================
// Original Serial Stein Kernel (for comparison)
// =============================================================================

__global__ void svpf_stein_operator_serial_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi_out,
    float inv_bw_sq,
    float inv_n,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    
    // Cooperative load
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        k_sum += K * sh_grad[j];
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
    }
    
    phi_out[i] = (k_sum + gk_sum) * inv_n;
}

// Serial KSD computation
__global__ void svpf_stein_operator_serial_ksd_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi_out,
    float* __restrict__ ksd_partial,
    float inv_bw_sq,
    float inv_n,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float grad_i = sh_grad[i];
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    float ksd_sum = 0.0f;
    
    for (int j = 0; j < n; j++) {
        float h_j = sh_h[j];
        float grad_j = sh_grad[j];
        float diff = h_i - h_j;
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        float K_cube = K_sq * K;
        
        k_sum += K * grad_j;
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
        
        // KSD
        float grad_k_i = -2.0f * diff * inv_bw_sq * K_sq;
        float grad_k_j = -grad_k_i;
        float grad2_k = -2.0f * inv_bw_sq * K_sq + 8.0f * dist_sq * inv_bw_sq * inv_bw_sq * K_cube;
        float u_p = K * grad_i * grad_j + grad_i * grad_k_j + grad_j * grad_k_i + grad2_k;
        ksd_sum += u_p;
    }
    
    phi_out[i] = (k_sum + gk_sum) * inv_n;
    ksd_partial[i] = ksd_sum;
}

// =============================================================================
// Test Utilities
// =============================================================================

void init_test_data(float* h_h, float* h_grad, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        // Random particles in typical log-vol range [-6, -2]
        h_h[i] = -6.0f + 4.0f * (float)rand() / RAND_MAX;
        // Random gradients
        h_grad[i] = -2.0f + 4.0f * (float)rand() / RAND_MAX;
    }
}

float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

float mean_abs_diff(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum / n;
}

// =============================================================================
// Correctness Test
// =============================================================================

bool test_correctness(int n) {
    printf("\n=== Correctness Test: N=%d ===\n", n);
    
    // Host allocations
    std::vector<float> h_h(n), h_grad(n);
    std::vector<float> h_phi_serial(n), h_phi_parallel(n);
    std::vector<float> h_ksd_partial_serial(n), h_ksd_partial_parallel(n);
    
    // Initialize test data
    init_test_data(h_h.data(), h_grad.data(), n, 42);
    
    // Device allocations
    float *d_h, *d_grad, *d_phi_serial, *d_phi_parallel;
    float *d_ksd_partial_serial, *d_ksd_partial_parallel;
    float *d_ksd_serial, *d_ksd_parallel;
    
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_phi_serial, n * sizeof(float));
    cudaMalloc(&d_phi_parallel, n * sizeof(float));
    cudaMalloc(&d_ksd_partial_serial, n * sizeof(float));
    cudaMalloc(&d_ksd_partial_parallel, n * sizeof(float));
    cudaMalloc(&d_ksd_serial, sizeof(float));
    cudaMalloc(&d_ksd_parallel, sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_h, h_h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Parameters
    float bandwidth = 0.5f;
    float inv_bw_sq = 1.0f / (bandwidth * bandwidth);
    float inv_n = 1.0f / (float)n;
    int stein_sign_mode = 1;
    
    // ===================
    // Test 1: Phi output
    // ===================
    
    // Serial kernel
    int nb_serial = (n + 255) / 256;
    size_t smem_serial = 2 * n * sizeof(float);
    svpf_stein_operator_serial_kernel<<<nb_serial, 256, smem_serial>>>(
        d_h, d_grad, d_phi_serial, inv_bw_sq, inv_n, stein_sign_mode, n
    );
    
    // Parallel kernel  
    size_t smem_parallel = (2 * n + 2 * STEIN_BLOCK_SIZE) * sizeof(float);
    svpf_stein_operator_parallel_kernel<<<n, STEIN_BLOCK_SIZE, smem_parallel>>>(
        d_h, d_grad, d_phi_parallel, inv_bw_sq, inv_n, stein_sign_mode, n
    );
    
    cudaDeviceSynchronize();
    
    // Copy back
    cudaMemcpy(h_phi_serial.data(), d_phi_serial, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phi_parallel.data(), d_phi_parallel, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_diff_phi = max_abs_diff(h_phi_serial.data(), h_phi_parallel.data(), n);
    float mean_diff_phi = mean_abs_diff(h_phi_serial.data(), h_phi_parallel.data(), n);
    
    printf("Phi output:\n");
    printf("  Max diff:  %.2e\n", max_diff_phi);
    printf("  Mean diff: %.2e\n", mean_diff_phi);
    
    bool phi_ok = max_diff_phi < 1e-4f;
    printf("  Status: %s\n", phi_ok ? "✓ PASS" : "❌ FAIL");
    
    // ===================
    // Test 2: KSD output
    // ===================
    
    // Serial with KSD
    svpf_stein_operator_serial_ksd_kernel<<<nb_serial, 256, smem_serial>>>(
        d_h, d_grad, d_phi_serial, d_ksd_partial_serial, inv_bw_sq, inv_n, stein_sign_mode, n
    );
    
    // Parallel with KSD
    size_t smem_parallel_ksd = (2 * n + 3 * STEIN_BLOCK_SIZE) * sizeof(float);
    svpf_stein_operator_parallel_ksd_kernel<<<n, STEIN_BLOCK_SIZE, smem_parallel_ksd>>>(
        d_h, d_grad, d_phi_parallel, d_ksd_partial_parallel, inv_bw_sq, inv_n, stein_sign_mode, n
    );
    
    // Reduce KSD
    svpf_ksd_reduce_parallel_kernel<<<1, 256>>>(d_ksd_partial_serial, d_ksd_serial, n);
    svpf_ksd_reduce_parallel_kernel<<<1, 256>>>(d_ksd_partial_parallel, d_ksd_parallel, n);
    
    cudaDeviceSynchronize();
    
    float h_ksd_serial, h_ksd_parallel;
    cudaMemcpy(&h_ksd_serial, d_ksd_serial, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ksd_parallel, d_ksd_parallel, sizeof(float), cudaMemcpyDeviceToHost);
    
    float ksd_diff = fabsf(h_ksd_serial - h_ksd_parallel);
    float ksd_rel_diff = ksd_diff / (fabsf(h_ksd_serial) + 1e-8f);
    
    printf("\nKSD output:\n");
    printf("  Serial:   %.6f\n", h_ksd_serial);
    printf("  Parallel: %.6f\n", h_ksd_parallel);
    printf("  Abs diff: %.2e\n", ksd_diff);
    printf("  Rel diff: %.2e\n", ksd_rel_diff);
    
    bool ksd_ok = ksd_rel_diff < 1e-3f;
    printf("  Status: %s\n", ksd_ok ? "✓ PASS" : "❌ FAIL");
    
    // Cleanup
    cudaFree(d_h);
    cudaFree(d_grad);
    cudaFree(d_phi_serial);
    cudaFree(d_phi_parallel);
    cudaFree(d_ksd_partial_serial);
    cudaFree(d_ksd_partial_parallel);
    cudaFree(d_ksd_serial);
    cudaFree(d_ksd_parallel);
    
    return phi_ok && ksd_ok;
}

// =============================================================================
// Performance Benchmark
// =============================================================================

void benchmark_stein(int n, int n_steps, int n_warmup) {
    printf("\n=== Performance Benchmark: N=%d, Steps=%d ===\n", n, n_steps);
    
    // Check shared memory
    if (!svpf_parallel_stein_fits_shared_mem(n)) {
        printf("WARNING: N=%d requires %.1f KB shared mem (may exceed 48KB limit)\n",
               n, svpf_parallel_stein_shared_mem(n, true) / 1024.0f);
    }
    
    // Host allocations
    std::vector<float> h_h(n), h_grad(n);
    init_test_data(h_h.data(), h_grad.data(), n, 123);
    
    // Device allocations
    float *d_h, *d_grad, *d_phi;
    float *d_ksd_partial, *d_ksd;
    
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_phi, n * sizeof(float));
    cudaMalloc(&d_ksd_partial, n * sizeof(float));
    cudaMalloc(&d_ksd, sizeof(float));
    
    cudaMemcpy(d_h, h_h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Parameters
    float bandwidth = 0.5f;
    float inv_bw_sq = 1.0f / (bandwidth * bandwidth);
    float inv_n = 1.0f / (float)n;
    int stein_sign_mode = 1;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ===================
    // Benchmark Serial
    // ===================
    int nb_serial = (n + 255) / 256;
    size_t smem_serial = 2 * n * sizeof(float);
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        svpf_stein_operator_serial_kernel<<<nb_serial, 256, smem_serial>>>(
            d_h, d_grad, d_phi, inv_bw_sq, inv_n, stein_sign_mode, n
        );
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < n_steps; i++) {
        svpf_stein_operator_serial_kernel<<<nb_serial, 256, smem_serial>>>(
            d_h, d_grad, d_phi, inv_bw_sq, inv_n, stein_sign_mode, n
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float serial_time_ms;
    cudaEventElapsedTime(&serial_time_ms, start, stop);
    float serial_per_step_us = serial_time_ms * 1000.0f / n_steps;
    
    // ===================
    // Benchmark Parallel
    // ===================
    size_t smem_parallel = (2 * n + 2 * STEIN_BLOCK_SIZE) * sizeof(float);
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        svpf_stein_operator_parallel_kernel<<<n, STEIN_BLOCK_SIZE, smem_parallel>>>(
            d_h, d_grad, d_phi, inv_bw_sq, inv_n, stein_sign_mode, n
        );
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < n_steps; i++) {
        svpf_stein_operator_parallel_kernel<<<n, STEIN_BLOCK_SIZE, smem_parallel>>>(
            d_h, d_grad, d_phi, inv_bw_sq, inv_n, stein_sign_mode, n
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float parallel_time_ms;
    cudaEventElapsedTime(&parallel_time_ms, start, stop);
    float parallel_per_step_us = parallel_time_ms * 1000.0f / n_steps;
    
    // ===================
    // Benchmark Parallel + KSD
    // ===================
    size_t smem_parallel_ksd = (2 * n + 3 * STEIN_BLOCK_SIZE) * sizeof(float);
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        svpf_stein_operator_parallel_ksd_kernel<<<n, STEIN_BLOCK_SIZE, smem_parallel_ksd>>>(
            d_h, d_grad, d_phi, d_ksd_partial, inv_bw_sq, inv_n, stein_sign_mode, n
        );
        svpf_ksd_reduce_parallel_kernel<<<1, 256>>>(d_ksd_partial, d_ksd, n);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < n_steps; i++) {
        svpf_stein_operator_parallel_ksd_kernel<<<n, STEIN_BLOCK_SIZE, smem_parallel_ksd>>>(
            d_h, d_grad, d_phi, d_ksd_partial, inv_bw_sq, inv_n, stein_sign_mode, n
        );
        svpf_ksd_reduce_parallel_kernel<<<1, 256>>>(d_ksd_partial, d_ksd, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float parallel_ksd_time_ms;
    cudaEventElapsedTime(&parallel_ksd_time_ms, start, stop);
    float parallel_ksd_per_step_us = parallel_ksd_time_ms * 1000.0f / n_steps;
    
    // ===================
    // Results
    // ===================
    float speedup = serial_per_step_us / parallel_per_step_us;
    float speedup_ksd = serial_per_step_us / parallel_ksd_per_step_us;
    
    printf("\nResults:\n");
    printf("  Serial:           %8.2f μs/step\n", serial_per_step_us);
    printf("  Parallel:         %8.2f μs/step  (%.1fx speedup)\n", parallel_per_step_us, speedup);
    printf("  Parallel + KSD:   %8.2f μs/step  (%.1fx speedup)\n", parallel_ksd_per_step_us, speedup_ksd);
    
    // Extrapolate to 32 steps
    printf("\nProjected 32-step Stein time:\n");
    printf("  Serial:           %8.2f μs\n", serial_per_step_us * 32);
    printf("  Parallel:         %8.2f μs\n", parallel_per_step_us * 32);
    printf("  Parallel + KSD:   %8.2f μs (last step only)\n", 
           parallel_per_step_us * 31 + parallel_ksd_per_step_us);
    
    // Cleanup
    cudaFree(d_h);
    cudaFree(d_grad);
    cudaFree(d_phi);
    cudaFree(d_ksd_partial);
    cudaFree(d_ksd);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     Parallel Stein Kernel Test & Benchmark                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("SMs: %d, Shared mem/block: %zu KB\n", 
           prop.multiProcessorCount, prop.sharedMemPerBlock / 1024);
    
    // Correctness tests
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf(" CORRECTNESS TESTS\n");
    printf("════════════════════════════════════════════════════════════════\n");
    
    bool all_pass = true;
    all_pass &= test_correctness(512);
    all_pass &= test_correctness(1024);
    all_pass &= test_correctness(2048);
    
    if (!all_pass) {
        printf("\n❌ Some correctness tests FAILED. Aborting benchmark.\n");
        return 1;
    }
    
    printf("\n✓ All correctness tests passed!\n");
    
    // Performance benchmarks
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf(" PERFORMANCE BENCHMARKS\n");
    printf("════════════════════════════════════════════════════════════════\n");
    
    int n_steps = 1000;
    int n_warmup = 100;
    
    benchmark_stein(512, n_steps, n_warmup);
    benchmark_stein(1024, n_steps, n_warmup);
    benchmark_stein(2048, n_steps, n_warmup);
    
    // Summary
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf(" SUMMARY\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("\nFor N=2048 with 32 Stein steps (crypto crisis mode):\n");
    printf("  - Serial kernel would take ~3+ ms per timestep\n");
    printf("  - Parallel kernel targets ~200 μs per timestep\n");
    printf("  - This enables 5000 ticks/sec vs 300 ticks/sec\n");
    
    return 0;
}
