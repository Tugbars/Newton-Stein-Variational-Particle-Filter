/**
 * @file svpf_bandwidth_diagnostic.cu
 * @brief Diagnostic tool to measure per-particle local density variation
 * 
 * Purpose: Determine if global bandwidth is inappropriate by measuring
 * how much local density varies across particles. High variance suggests
 * per-particle adaptive bandwidth would help.
 * 
 * Key metric: If some particles have density 0.8 while others have 0.1,
 * the global bandwidth is squeezing explorers too hard.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cmath>
#include <vector>

// =============================================================================
// DIAGNOSTIC KERNEL
// =============================================================================

/**
 * @brief Compute per-particle local density and k-NN distance
 * 
 * For each particle i, computes:
 *   - local_density[i]: fraction of particles within bandwidth distance
 *   - knn_distance[i]: distance to k-th nearest neighbor
 * 
 * These reveal whether particles live in regions of very different density.
 */
__global__ void svpf_local_density_kernel(
    const float* __restrict__ h,          // [n] particle positions
    float* __restrict__ local_density,    // [n] output: fraction within bandwidth
    float* __restrict__ knn_distance,     // [n] output: k-th NN distance
    float bandwidth,                       // global bandwidth
    int k_neighbors,                       // k for k-NN distance
    int n
) {
    extern __shared__ float sh_h[];
    
    // Load all particles to shared memory
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        sh_h[idx] = h[idx];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    
    // Count neighbors within bandwidth
    int count_within_bw = 0;
    for (int j = 0; j < n; j++) {
        float dist = fabsf(h_i - sh_h[j]);
        if (dist < bandwidth && i != j) {
            count_within_bw++;
        }
    }
    local_density[i] = (float)count_within_bw / (float)(n - 1);
    
    // Find k-th nearest neighbor distance using partial insertion sort
    // We maintain the k smallest distances seen so far
    const int MAX_K = 64;
    float knn_dists[MAX_K];
    int k_actual = min(k_neighbors, min(n - 1, MAX_K));
    
    // Initialize with large values
    for (int kk = 0; kk < k_actual; kk++) {
        knn_dists[kk] = 1e10f;
    }
    
    // Find k smallest distances
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        float dist = fabsf(h_i - sh_h[j]);
        
        // Insert into sorted array if smaller than largest
        if (dist < knn_dists[k_actual - 1]) {
            // Find insertion point
            int insert_pos = k_actual - 1;
            while (insert_pos > 0 && dist < knn_dists[insert_pos - 1]) {
                knn_dists[insert_pos] = knn_dists[insert_pos - 1];
                insert_pos--;
            }
            knn_dists[insert_pos] = dist;
        }
    }
    
    // k-th nearest neighbor distance (0-indexed, so k_actual-1)
    knn_distance[i] = knn_dists[k_actual - 1];
}

/**
 * @brief Compute statistics (min, max, mean, std) of an array
 */
__global__ void svpf_array_stats_kernel(
    const float* __restrict__ data,
    float* __restrict__ stats,  // [4]: min, max, mean, std
    int n
) {
    __shared__ float s_min, s_max, s_sum, s_sum_sq;
    
    if (threadIdx.x == 0) {
        s_min = 1e10f;
        s_max = -1e10f;
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    float local_min = 1e10f;
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = data[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Reduce within block
    atomicMin((int*)&s_min, __float_as_int(local_min));
    atomicMax((int*)&s_max, __float_as_int(local_max));
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float mean = s_sum / n;
        float var = (s_sum_sq / n) - mean * mean;
        stats[0] = s_min;
        stats[1] = s_max;
        stats[2] = mean;
        stats[3] = sqrtf(fmaxf(var, 0.0f));
    }
}

// =============================================================================
// HOST-SIDE DIAGNOSTIC STRUCT
// =============================================================================

struct BandwidthDiagnostics {
    // Device buffers
    float* d_local_density;
    float* d_knn_distance;
    float* d_density_stats;
    float* d_knn_stats;
    
    // Host results
    float density_min, density_max, density_mean, density_std;
    float knn_min, knn_max, knn_mean, knn_std;
    float global_bandwidth;
    
    int n;
    int k_neighbors;
    bool initialized;
};

BandwidthDiagnostics* bandwidth_diag_create(int n_particles, int k_neighbors = 32) {
    BandwidthDiagnostics* diag = new BandwidthDiagnostics();
    diag->n = n_particles;
    diag->k_neighbors = k_neighbors;
    
    cudaMalloc(&diag->d_local_density, n_particles * sizeof(float));
    cudaMalloc(&diag->d_knn_distance, n_particles * sizeof(float));
    cudaMalloc(&diag->d_density_stats, 4 * sizeof(float));
    cudaMalloc(&diag->d_knn_stats, 4 * sizeof(float));
    
    diag->initialized = true;
    return diag;
}

void bandwidth_diag_destroy(BandwidthDiagnostics* diag) {
    if (!diag) return;
    if (diag->initialized) {
        cudaFree(diag->d_local_density);
        cudaFree(diag->d_knn_distance);
        cudaFree(diag->d_density_stats);
        cudaFree(diag->d_knn_stats);
    }
    delete diag;
}

void bandwidth_diag_compute(
    BandwidthDiagnostics* diag,
    const float* d_h,           // Device: particle positions
    float global_bandwidth,
    cudaStream_t stream = 0
) {
    int n = diag->n;
    int block = 256;
    int grid = (n + block - 1) / block;
    size_t smem = n * sizeof(float);
    
    diag->global_bandwidth = global_bandwidth;
    
    // Compute local density and k-NN distance
    svpf_local_density_kernel<<<grid, block, smem, stream>>>(
        d_h, diag->d_local_density, diag->d_knn_distance,
        global_bandwidth, diag->k_neighbors, n
    );
    
    // Compute statistics
    svpf_array_stats_kernel<<<1, 256, 0, stream>>>(
        diag->d_local_density, diag->d_density_stats, n
    );
    svpf_array_stats_kernel<<<1, 256, 0, stream>>>(
        diag->d_knn_distance, diag->d_knn_stats, n
    );
    
    // Copy results to host
    float density_stats[4], knn_stats[4];
    cudaMemcpyAsync(density_stats, diag->d_density_stats, 4 * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(knn_stats, diag->d_knn_stats, 4 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    diag->density_min = density_stats[0];
    diag->density_max = density_stats[1];
    diag->density_mean = density_stats[2];
    diag->density_std = density_stats[3];
    
    diag->knn_min = knn_stats[0];
    diag->knn_max = knn_stats[1];
    diag->knn_mean = knn_stats[2];
    diag->knn_std = knn_stats[3];
}

void bandwidth_diag_print(const BandwidthDiagnostics* diag, int timestep) {
    printf("t=%4d | bw=%.4f | density[%.2f, %.2f] μ=%.2f σ=%.2f | "
           "kNN[%.3f, %.3f] μ=%.3f σ=%.3f | ratio=%.2f\n",
           timestep,
           diag->global_bandwidth,
           diag->density_min, diag->density_max, 
           diag->density_mean, diag->density_std,
           diag->knn_min, diag->knn_max,
           diag->knn_mean, diag->knn_std,
           diag->knn_max / fmaxf(diag->knn_min, 1e-6f)  // Ratio of max to min k-NN dist
    );
}

// =============================================================================
// SYNTHETIC TEST: Vol spike scenario
// =============================================================================

#include <curand_kernel.h>

__global__ void init_particles_stationary(
    float* h, 
    curandState* rng,
    float mu, float sigma_z, float rho,
    unsigned long long seed,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    curand_init(seed, i, 0, &rng[i]);
    
    // Stationary distribution: h ~ N(mu, sigma_z^2 / (1 - rho^2))
    float stationary_std = sigma_z / sqrtf(1.0f - rho * rho);
    h[i] = mu + stationary_std * curand_normal(&rng[i]);
}

__global__ void simple_predict_step(
    float* h,
    float* h_prev,
    curandState* rng,
    float mu, float rho, float sigma_z,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_old = h[i];
    h_prev[i] = h_old;
    
    // AR(1) predict
    float eps = curand_normal(&rng[i]);
    h[i] = mu + rho * (h_old - mu) + sigma_z * eps;
}

__global__ void simple_likelihood_update(
    float* h,
    float y_t,
    float nu,
    float step_size,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Simple gradient step toward observation
    // grad = (log(y^2) - h) / 2  (approximate)
    float y_sq = y_t * y_t + 1e-8f;
    float h_implied = logf(y_sq);
    float grad = (h_implied - h[i]) * 0.5f;
    
    h[i] += step_size * grad;
}

__global__ void compute_median_bandwidth(
    const float* h,
    float* d_bandwidth,
    int n
) {
    // Use global memory or smaller batches
    if (threadIdx.x == 0) {
        // Simple variance-based estimate instead of true median
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < n; i++) {
            sum += h[i];
            sum_sq += h[i] * h[i];
        }
        float mean = sum / n;
        float var = sum_sq / n - mean * mean;
        *d_bandwidth = sqrtf(var) * 1.06f * powf(n, -0.2f);  // Silverman
    }
}

void run_bandwidth_diagnostic_test() {
    printf("\n");
    printf("========================================\n");
    printf("  SVPF Bandwidth Diagnostic Test\n");
    printf("========================================\n\n");
    
    // Parameters
    const int N = 256;          // Particles
    const int T = 200;          // Timesteps
    const float mu = -3.5f;
    const float rho = 0.97f;
    const float sigma_z = 0.15f;
    const float nu = 5.0f;
    
    // Vol spike schedule: calm -> spike -> calm
    // Returns: small -> large -> small
    auto get_return = [](int t) -> float {
        if (t >= 80 && t < 100) {
            // Vol spike: returns jump from ~0.01 to ~0.08
            return 0.08f * ((t % 2 == 0) ? 1.0f : -1.0f);
        }
        return 0.01f * ((t % 2 == 0) ? 1.0f : -1.0f);
    };
    
    // Allocate
    float *d_h, *d_h_prev, *d_bandwidth;
    curandState* d_rng;
    
    cudaMalloc(&d_h, N * sizeof(float));
    cudaMalloc(&d_h_prev, N * sizeof(float));
    cudaMalloc(&d_bandwidth, sizeof(float));
    cudaMalloc(&d_rng, N * sizeof(curandState));
    
    // Initialize
    int block = 256;
    int grid = (N + block - 1) / block;
    
    init_particles_stationary<<<grid, block>>>(
        d_h, d_rng, mu, sigma_z, rho, 12345ULL, N
    );
    cudaDeviceSynchronize();
    
    // Create diagnostic
    BandwidthDiagnostics* diag = bandwidth_diag_create(N, 32);
    
    printf("Config: N=%d, mu=%.2f, rho=%.2f, sigma=%.2f, nu=%.1f\n", N, mu, rho, sigma_z, nu);
    printf("Scenario: Calm (t<80) -> Vol spike (80<=t<100) -> Calm (t>=100)\n\n");
    printf("Legend:\n");
    printf("  density = fraction of particles within bandwidth distance\n");
    printf("  kNN = distance to k-th nearest neighbor (k=32)\n");
    printf("  ratio = max_kNN / min_kNN (high = particles at very different densities)\n\n");
    
    // Run filter with diagnostics
    for (int t = 0; t < T; t++) {
        float y_t = get_return(t);
        
        // Predict
        simple_predict_step<<<grid, block>>>(d_h, d_h_prev, d_rng, mu, rho, sigma_z, N);
        
        // Compute bandwidth (median heuristic)
        compute_median_bandwidth<<<1, 1>>>(d_h, d_bandwidth, N);
        
        // Simple likelihood update (mimics Stein without full complexity)
        simple_likelihood_update<<<grid, block>>>(d_h, y_t, nu, 0.1f, N);
        
        // Get bandwidth to host
        float h_bandwidth;
        cudaMemcpy(&h_bandwidth, d_bandwidth, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute and print diagnostics at key points
        bool is_key_timestep = (t < 5) ||                    // Start
                               (t >= 78 && t <= 82) ||       // Spike onset
                               (t >= 95 && t <= 105) ||      // Spike end
                               (t == T - 1) ||               // End
                               (t % 20 == 0);                // Periodic
        
        if (is_key_timestep) {
            bandwidth_diag_compute(diag, d_h, h_bandwidth);
            
            // Mark regime
            const char* regime = (t >= 80 && t < 100) ? " [SPIKE]" : "";
            printf("t=%3d%s\n", t, regime);
            bandwidth_diag_print(diag, t);
            printf("\n");
        }
    }
    
    // Summary
    printf("========================================\n");
    printf("  Interpretation Guide\n");
    printf("========================================\n");
    printf("If during vol spike you see:\n");
    printf("  - density range widens (e.g., [0.1, 0.9] vs [0.4, 0.6])\n");
    printf("  - kNN ratio increases (e.g., 5x vs 2x)\n");
    printf("Then particles are at very different local densities,\n");
    printf("and per-particle adaptive bandwidth would help.\n\n");
    
    // Cleanup
    bandwidth_diag_destroy(diag);
    cudaFree(d_h);
    cudaFree(d_h_prev);
    cudaFree(d_bandwidth);
    cudaFree(d_rng);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    // Check CUDA device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s\n", prop.name);
    
    run_bandwidth_diagnostic_test();
    
    printf("Done.\n");
    return 0;
}
