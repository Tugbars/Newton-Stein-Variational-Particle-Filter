/**
 * @file test_full_newton_parallel.cu
 * @brief Correctness test for parallel FULL NEWTON Stein kernel
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define STEIN_PARALLEL_BLOCK_SIZE 256

// =============================================================================
// Serial Full Newton Stein Operator (reference)
// =============================================================================
__global__ void svpf_stein_operator_full_newton_serial(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ phi_out,
    float inv_bw_sq,
    float inv_n,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    float* sh_hess = smem + 2 * n;
    
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
        sh_hess[k] = local_hessian[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    float H_weighted = 0.0f;
    float K_sum_norm = 0.0f;
    float k_grad_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        H_weighted += sh_hess[j] * K;
        float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        H_weighted += Nk;
        K_sum_norm += K;
        
        k_grad_sum += K * sh_grad[j];
        gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
    }
    
    H_weighted = H_weighted / fmaxf(K_sum_norm, 1e-6f);
    H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);
    float inv_H_i = 1.0f / H_weighted;
    
    phi_out[i] = (k_grad_sum + gk_sum) * inv_n * inv_H_i * 0.7f;
}

// =============================================================================
// Parallel Full Newton Stein Operator (to test)
// =============================================================================
__global__ void svpf_stein_operator_full_newton_parallel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ local_hessian,
    float* __restrict__ phi_out,
    float inv_bw_sq,
    float inv_n,
    int stein_sign_mode,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = &smem[n];
    float* sh_hess = &smem[2*n];
    float* sh_k_grad_sum = &smem[3*n];
    float* sh_gk_sum = &smem[3*n + STEIN_PARALLEL_BLOCK_SIZE];
    float* sh_H_weighted = &smem[3*n + 2*STEIN_PARALLEL_BLOCK_SIZE];
    float* sh_K_sum_norm = &smem[3*n + 3*STEIN_PARALLEL_BLOCK_SIZE];
    
    int i = blockIdx.x;        // Each BLOCK handles particle i
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Cooperative load
    for (int k = tid; k < n; k += stride) {
        sh_h[k] = h[k];
        sh_grad[k] = grad[k];
        sh_hess[k] = local_hessian[k];
    }
    __syncthreads();
    
    float h_i = sh_h[i];
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    // Each thread computes partial sums
    float my_H_weighted = 0.0f;
    float my_K_sum_norm = 0.0f;
    float my_k_grad_sum = 0.0f;
    float my_gk_sum = 0.0f;
    
    for (int j = tid; j < n; j += stride) {
        float diff = h_i - sh_h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;
        float K_sq = K * K;
        
        my_H_weighted += sh_hess[j] * K;
        float Nk = 2.0f * inv_bw_sq * K_sq * fabsf(3.0f * dist_sq - 1.0f);
        my_H_weighted += Nk;
        my_K_sum_norm += K;
        
        my_k_grad_sum += K * sh_grad[j];
        my_gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
    }
    
    sh_k_grad_sum[tid] = my_k_grad_sum;
    sh_gk_sum[tid] = my_gk_sum;
    sh_H_weighted[tid] = my_H_weighted;
    sh_K_sum_norm[tid] = my_K_sum_norm;
    __syncthreads();
    
    // Parallel reduction
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_k_grad_sum[tid] += sh_k_grad_sum[tid + s];
            sh_gk_sum[tid] += sh_gk_sum[tid + s];
            sh_H_weighted[tid] += sh_H_weighted[tid + s];
            sh_K_sum_norm[tid] += sh_K_sum_norm[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float H_weighted = sh_H_weighted[0] / fmaxf(sh_K_sum_norm[0], 1e-6f);
        H_weighted = fminf(fmaxf(H_weighted, 0.1f), 100.0f);
        float inv_H_i = 1.0f / H_weighted;
        
        phi_out[i] = (sh_k_grad_sum[0] + sh_gk_sum[0]) * inv_n * inv_H_i * 0.7f;
    }
}

// =============================================================================
// Test
// =============================================================================

void init_test_data(float* h_h, float* h_grad, float* h_hess, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        h_h[i] = -6.0f + 4.0f * (float)rand() / RAND_MAX;
        h_grad[i] = -2.0f + 4.0f * (float)rand() / RAND_MAX;
        h_hess[i] = 0.5f + 2.0f * (float)rand() / RAND_MAX;  // Positive hessian
    }
}

bool test_full_newton_correctness(int n) {
    printf("\n=== Full Newton Correctness Test: N=%d ===\n", n);
    
    std::vector<float> h_h(n), h_grad(n), h_hess(n);
    std::vector<float> h_phi_serial(n), h_phi_parallel(n);
    
    init_test_data(h_h.data(), h_grad.data(), h_hess.data(), n, 42);
    
    float *d_h, *d_grad, *d_hess, *d_phi_serial, *d_phi_parallel;
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_hess, n * sizeof(float));
    cudaMalloc(&d_phi_serial, n * sizeof(float));
    cudaMalloc(&d_phi_parallel, n * sizeof(float));
    
    cudaMemcpy(d_h, h_h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hess, h_hess.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    float bandwidth = 0.5f;
    float inv_bw_sq = 1.0f / (bandwidth * bandwidth);
    float inv_n = 1.0f / (float)n;
    int stein_sign_mode = 1;
    
    // Serial
    int nb = (n + 255) / 256;
    size_t smem_serial = 3 * n * sizeof(float);
    svpf_stein_operator_full_newton_serial<<<nb, 256, smem_serial>>>(
        d_h, d_grad, d_hess, d_phi_serial, inv_bw_sq, inv_n, stein_sign_mode, n
    );
    
    // Parallel
    size_t smem_parallel = (3 * n + 4 * STEIN_PARALLEL_BLOCK_SIZE) * sizeof(float);
    svpf_stein_operator_full_newton_parallel<<<n, STEIN_PARALLEL_BLOCK_SIZE, smem_parallel>>>(
        d_h, d_grad, d_hess, d_phi_parallel, inv_bw_sq, inv_n, stein_sign_mode, n
    );
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_phi_serial.data(), d_phi_serial, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phi_parallel.data(), d_phi_parallel, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int max_diff_idx = 0;
    
    for (int i = 0; i < n; i++) {
        float diff = fabsf(h_phi_serial[i] - h_phi_parallel[i]);
        float rel_diff = diff / (fabsf(h_phi_serial[i]) + 1e-8f);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    
    printf("Phi output:\n");
    printf("  Max abs diff:  %.6e (at idx %d)\n", max_diff, max_diff_idx);
    printf("  Max rel diff:  %.6e\n", max_rel_diff);
    printf("  Serial[%d]:    %.6f\n", max_diff_idx, h_phi_serial[max_diff_idx]);
    printf("  Parallel[%d]:  %.6f\n", max_diff_idx, h_phi_parallel[max_diff_idx]);
    
    bool pass = max_diff < 1e-4f;
    printf("  Status: %s\n", pass ? "✓ PASS" : "❌ FAIL");
    
    // Print a few samples
    printf("\nSample values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  [%d] Serial=%.6f, Parallel=%.6f, Diff=%.2e\n", 
               i, h_phi_serial[i], h_phi_parallel[i], 
               fabsf(h_phi_serial[i] - h_phi_parallel[i]));
    }
    
    cudaFree(d_h);
    cudaFree(d_grad);
    cudaFree(d_hess);
    cudaFree(d_phi_serial);
    cudaFree(d_phi_parallel);
    
    return pass;
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     Full Newton Parallel Kernel Correctness Test               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    bool all_pass = true;
    all_pass &= test_full_newton_correctness(512);
    all_pass &= test_full_newton_correctness(1024);
    all_pass &= test_full_newton_correctness(2048);
    
    printf("\n");
    if (all_pass) {
        printf("✓ All tests passed!\n");
    } else {
        printf("❌ Some tests FAILED!\n");
    }
    
    return all_pass ? 0 : 1;
}
