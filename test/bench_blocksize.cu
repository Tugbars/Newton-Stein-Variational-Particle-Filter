/**
 * @file bench_blocksize.cu
 * @brief Block size sweep for O(N²) Stein kernel pattern
 *
 * Measures wall-clock time for a kernel that mimics the Stein transport
 * inner loop: each thread loads all N particles into shared memory,
 * then computes N multiply-adds. Sweeps BLOCK_SIZE from 32 to 512
 * for N = 512 and N = 1024.
 *
 * Build:
 *   nvcc -O2 bench_blocksize.cu -o bench_blocksize
 *
 * Usage:
 *   ./bench_blocksize
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

// Mimics the Stein transport O(N²) pattern:
// - Load all N particles + gradients + hessians into shared memory
// - Each thread computes over all N particles (inner loop)
// - Write back one value per thread
template<int BLOCK>
__global__ void stein_pattern_kernel(
    float* __restrict__ h,
    const float* __restrict__ grad,
    const float* __restrict__ hess,
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h    = smem;
    float* sh_grad = smem + n;
    float* sh_hess = smem + 2 * n;

    // Cooperative load: all threads load all N values
    for (int k = threadIdx.x; k < n; k += BLOCK) {
        sh_h[k]    = h[k];
        sh_grad[k] = grad[k];
        sh_hess[k] = hess[k];
    }
    __syncthreads();

    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;

    float h_i = sh_h[i];
    float bw_sq = *d_bandwidth;
    float inv_bw_sq = 1.0f / bw_sq;

    // O(N) inner loop — mirrors real Stein kernel
    float H_weighted = 0.0f;
    float K_sum = 0.0f;
    float k_grad_sum = 0.0f;

    for (int j = 0; j < n; j++) {
        float diff = h_i - sh_h[j];
        float dist_sq = diff * diff * inv_bw_sq;
        float base = 1.0f + dist_sq;
        float K = 1.0f / base;

        H_weighted += sh_hess[j] * K;
        K_sum += K;
        k_grad_sum += K * sh_grad[j];
    }

    H_weighted = H_weighted / fmaxf(K_sum, 1e-6f);
    float inv_H = 1.0f / fmaxf(H_weighted, 0.1f);
    float phi = k_grad_sum * inv_H * (1.0f / (float)n);

    h[i] = h_i + 0.01f * phi;
}

// Launch wrapper
template<int BLOCK>
float bench_one(float* d_h, float* d_grad, float* d_hess, float* d_bw, int n, int n_iters) {
    int nb = (n + BLOCK - 1) / BLOCK;
    size_t smem = 3 * n * sizeof(float);

    // Warmup
    for (int i = 0; i < 50; i++) {
        stein_pattern_kernel<BLOCK><<<nb, BLOCK, smem>>>(d_h, d_grad, d_hess, d_bw, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_iters; i++) {
        stein_pattern_kernel<BLOCK><<<nb, BLOCK, smem>>>(d_h, d_grad, d_hess, d_bw, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    const int n_iters = 10000;  // Many iterations for stable timing
    const int stein_steps = 24; // 12 iterations × 2 kernels (gradient + transport)

    int particle_counts[] = {256, 512, 1024};
    int n_particles_configs = 3;

    printf("Block Size Sweep for O(N²) Stein Pattern\n");
    printf("Each 'tick' = %d Stein kernel launches\n", stein_steps);
    printf("Iterations per measurement: %d\n\n", n_iters);

    for (int pc = 0; pc < n_particles_configs; pc++) {
        int n = particle_counts[pc];

        float* d_h, *d_grad, *d_hess, *d_bw;
        cudaMalloc(&d_h, n * sizeof(float));
        cudaMalloc(&d_grad, n * sizeof(float));
        cudaMalloc(&d_hess, n * sizeof(float));
        cudaMalloc(&d_bw, sizeof(float));

        // Initialize with reasonable values
        float* h_init = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) h_init[i] = -4.5f + 0.5f * (i / (float)n);
        cudaMemcpy(d_h, h_init, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad, h_init, n * sizeof(float), cudaMemcpyHostToDevice);

        float ones = 1.0f;
        for (int i = 0; i < n; i++) h_init[i] = 1.0f;
        cudaMemcpy(d_hess, h_init, n * sizeof(float), cudaMemcpyHostToDevice);

        float bw = 0.5f;
        cudaMemcpy(d_bw, &bw, sizeof(float), cudaMemcpyHostToDevice);

        printf("═══════════════════════════════════════════════\n");
        printf("  N = %d particles\n", n);
        printf("═══════════════════════════════════════════════\n");
        printf("  Block │ Blocks │ per-kernel │ per-tick (%d) │ Speedup\n", stein_steps);
        printf("  ──────┼────────┼────────────┼──────────────┼────────\n");

        float baseline_us = 0.0f;

        // Sweep block sizes
        int block_sizes[] = {32, 64, 128, 256, 512};
        int n_blocks = 5;

        for (int b = 0; b < n_blocks; b++) {
            int bs = block_sizes[b];
            if (bs > n) continue;  // Skip if block > particle count

            float ms = 0.0f;
            switch (bs) {
                case 32:  ms = bench_one<32> (d_h, d_grad, d_hess, d_bw, n, n_iters); break;
                case 64:  ms = bench_one<64> (d_h, d_grad, d_hess, d_bw, n, n_iters); break;
                case 128: ms = bench_one<128>(d_h, d_grad, d_hess, d_bw, n, n_iters); break;
                case 256: ms = bench_one<256>(d_h, d_grad, d_hess, d_bw, n, n_iters); break;
                case 512: ms = bench_one<512>(d_h, d_grad, d_hess, d_bw, n, n_iters); break;
            }

            float per_kernel_us = (ms * 1000.0f) / n_iters;
            float per_tick_us = per_kernel_us * stein_steps;
            int nb = (n + bs - 1) / bs;

            if (bs == 512 || (bs == n && baseline_us == 0.0f)) {
                baseline_us = per_tick_us;
            }

            float speedup = (baseline_us > 0.0f) ? baseline_us / per_tick_us : 1.0f;

            printf("  %5d │ %6d │ %7.2f μs │ %9.1f μs │ %.2fx\n",
                   bs, nb, per_kernel_us, per_tick_us, speedup);
        }

        printf("\n");

        // Also benchmark: 4 concurrent instances (simulating production)
        printf("  4 concurrent instances (separate streams):\n");
        printf("  Block │ Wall-clock per tick (4 instances)\n");
        printf("  ──────┼──────────────────────────────────\n");

        cudaStream_t streams[4];
        float* d_h4[4], *d_grad4[4], *d_hess4[4], *d_bw4[4];
        for (int s = 0; s < 4; s++) {
            cudaStreamCreate(&streams[s]);
            cudaMalloc(&d_h4[s], n * sizeof(float));
            cudaMalloc(&d_grad4[s], n * sizeof(float));
            cudaMalloc(&d_hess4[s], n * sizeof(float));
            cudaMalloc(&d_bw4[s], sizeof(float));
            cudaMemcpy(d_h4[s], d_h, n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_grad4[s], d_grad, n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_hess4[s], d_hess, n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_bw4[s], d_bw, sizeof(float), cudaMemcpyDeviceToDevice);
        }

        for (int b = 0; b < n_blocks; b++) {
            int bs = block_sizes[b];
            if (bs > n) continue;
            int nb = (n + bs - 1) / bs;
            size_t smem = 3 * n * sizeof(float);

            // Warmup
            for (int w = 0; w < 50; w++) {
                for (int s = 0; s < 4; s++) {
                    switch (bs) {
                        case 32:  stein_pattern_kernel<32> <<<nb, 32,  smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                        case 64:  stein_pattern_kernel<64> <<<nb, 64,  smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                        case 128: stein_pattern_kernel<128><<<nb, 128, smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                        case 256: stein_pattern_kernel<256><<<nb, 256, smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                        case 512: stein_pattern_kernel<512><<<nb, 512, smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                    }
                }
            }
            cudaDeviceSynchronize();

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int tick_iters = n_iters / stein_steps;  // Measure in ticks

            cudaEventRecord(start);
            for (int t = 0; t < tick_iters; t++) {
                for (int k = 0; k < stein_steps; k++) {
                    for (int s = 0; s < 4; s++) {
                        switch (bs) {
                            case 32:  stein_pattern_kernel<32> <<<nb, 32,  smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                            case 64:  stein_pattern_kernel<64> <<<nb, 64,  smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                            case 128: stein_pattern_kernel<128><<<nb, 128, smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                            case 256: stein_pattern_kernel<256><<<nb, 256, smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                            case 512: stein_pattern_kernel<512><<<nb, 512, smem, streams[s]>>>(d_h4[s], d_grad4[s], d_hess4[s], d_bw4[s], n); break;
                        }
                    }
                }
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            float per_tick_us = (ms * 1000.0f) / tick_iters;

            printf("  %5d │ %9.1f μs\n", bs, per_tick_us);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        for (int s = 0; s < 4; s++) {
            cudaFree(d_h4[s]); cudaFree(d_grad4[s]);
            cudaFree(d_hess4[s]); cudaFree(d_bw4[s]);
            cudaStreamDestroy(streams[s]);
        }

        cudaFree(d_h); cudaFree(d_grad); cudaFree(d_hess); cudaFree(d_bw);
        free(h_init);
        printf("\n");
    }

    printf("Done.\n");
    return 0;
}
