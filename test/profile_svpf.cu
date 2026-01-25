/**
 * @file profile_svpf.cu
 * @brief Profiling harness for SVPF performance analysis
 * 
 * Measures:
 *   - Total step latency (P50, P99, max)
 *   - Kernel breakdown (predict, gradient, transport, etc.)
 *   - Memory operations
 *   - Graph replay vs non-graph overhead
 * 
 * Usage:
 *   ./profile_svpf [n_particles] [n_steps] [warmup]
 *   ./profile_svpf 4096 10000 100
 */

#include "svpf.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

// =============================================================================
// Timing Utilities
// =============================================================================

struct TimingStats {
    float min_us;
    float max_us;
    float mean_us;
    float p50_us;
    float p90_us;
    float p99_us;
    float std_us;
};

static TimingStats compute_stats(std::vector<float>& times_us) {
    TimingStats s = {0};
    if (times_us.empty()) return s;
    
    std::sort(times_us.begin(), times_us.end());
    
    int n = (int)times_us.size();
    s.min_us = times_us[0];
    s.max_us = times_us[n - 1];
    s.p50_us = times_us[n / 2];
    s.p90_us = times_us[(int)(n * 0.90)];
    s.p99_us = times_us[(int)(n * 0.99)];
    
    double sum = 0;
    for (float t : times_us) sum += t;
    s.mean_us = (float)(sum / n);
    
    double var = 0;
    for (float t : times_us) var += (t - s.mean_us) * (t - s.mean_us);
    s.std_us = (float)sqrt(var / n);
    
    return s;
}

static void print_stats(const char* name, const TimingStats& s) {
    printf("  %-20s: mean=%7.1f  p50=%7.1f  p90=%7.1f  p99=%7.1f  max=%7.1f μs\n",
           name, s.mean_us, s.p50_us, s.p90_us, s.p99_us, s.max_us);
}

// =============================================================================
// Synthetic Data Generator
// =============================================================================

static void generate_sv_returns(float* returns, int n, float mu, float rho, 
                                 float sigma, unsigned int seed) {
    srand(seed);
    
    float h = mu;
    for (int t = 0; t < n; t++) {
        // Box-Muller for Gaussian
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159f * u2);
        
        // Transition
        h = mu + rho * (h - mu) + sigma * z1;
        
        // Observation
        float vol = expf(h / 2.0f);
        returns[t] = vol * z2;
    }
}

// =============================================================================
// Profile: Overall Step Latency
// =============================================================================

static void profile_step_latency(int n_particles, int n_stein, int n_steps, int warmup) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Profile: Step Latency (N=%d, Stein=%d)\n", n_particles, n_stein);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    // Generate test data
    int total_steps = warmup + n_steps;
    float* returns = (float*)malloc(total_steps * sizeof(float));
    generate_sv_returns(returns, total_steps, -4.5f, 0.98f, 0.10f, 12345);
    
    // Create filter
    SVPFState* state = svpf_create(n_particles, n_stein, 7.0f, nullptr);
    SVPFParams params = {0.98f, 0.10f, -4.5f, 0.0f};
    svpf_initialize(state, &params, 12345);
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> step_times;
    step_times.reserve(n_steps);
    
    float y_prev = 0.0f;
    
    // Warmup (also captures graph)
    printf(" Warmup: %d steps...\n", warmup);
    for (int t = 0; t < warmup; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        y_prev = returns[t];
    }
    cudaDeviceSynchronize();
    
    // Timed steps
    printf(" Profiling: %d steps...\n", n_steps);
    for (int t = warmup; t < total_steps; t++) {
        cudaEventRecord(start);
        
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        step_times.push_back(ms * 1000.0f);  // Convert to microseconds
        
        y_prev = returns[t];
    }
    
    // Stats
    TimingStats stats = compute_stats(step_times);
    printf("\n Results:\n");
    print_stats("svpf_step_graph", stats);
    printf("\n");
    printf("  Throughput: %.0f steps/sec (at mean latency)\n", 1e6f / stats.mean_us);
    printf("  Throughput: %.0f steps/sec (at P99 latency)\n", 1e6f / stats.p99_us);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    svpf_destroy(state);
    free(returns);
}

// =============================================================================
// Profile: Particle Count Scaling
// =============================================================================

static void profile_particle_scaling(int n_steps, int warmup) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Profile: Particle Count Scaling\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    // Check max shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t max_smem = prop.sharedMemPerBlock;
    printf("\n  Max shared memory per block: %zu KB\n", max_smem / 1024);
    printf("  WARNING: N > %zu will exceed shared memory!\n\n", max_smem / (2 * sizeof(float)));
    
    int particle_counts[] = {256, 384, 512, 640, 768, 1024, 2048};
    int n_counts = sizeof(particle_counts) / sizeof(particle_counts[0]);
    
    int total_steps = warmup + n_steps;
    float* returns = (float*)malloc(total_steps * sizeof(float));
    generate_sv_returns(returns, total_steps, -4.5f, 0.98f, 0.10f, 12345);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\n");
    printf("  Particles |   Mean   |   P50    |   P99    |   Max    | Throughput\n");
    printf("  ----------+----------+----------+----------+----------+-----------\n");
    
    for (int i = 0; i < n_counts; i++) {
        int n_particles = particle_counts[i];
        
        SVPFState* state = svpf_create(n_particles, 5, 7.0f, nullptr);
        SVPFParams params = {0.98f, 0.10f, -4.5f, 0.0f};
        svpf_initialize(state, &params, 12345);
        
        std::vector<float> times;
        times.reserve(n_steps);
        
        float y_prev = 0.0f;
        
        // Warmup
        for (int t = 0; t < warmup; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
            y_prev = returns[t];
        }
        cudaDeviceSynchronize();
        
        // Profile
        for (int t = warmup; t < total_steps; t++) {
            cudaEventRecord(start);
            
            float loglik, vol, h_mean;
            svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms * 1000.0f);
            
            y_prev = returns[t];
        }
        
        TimingStats s = compute_stats(times);
        printf("  %9d | %6.1f μs | %6.1f μs | %6.1f μs | %6.1f μs | %6.0f/s\n",
               n_particles, s.mean_us, s.p50_us, s.p99_us, s.max_us, 1e6f / s.mean_us);
        
        svpf_destroy(state);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(returns);
}

// =============================================================================
// Profile: Stein Steps Scaling
// =============================================================================

static void profile_stein_scaling(int n_particles, int n_steps, int warmup) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Profile: Stein Steps Scaling (N=%d)\n", n_particles);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    int stein_counts[] = {1, 2, 3, 5, 7, 10, 15, 20};
    int n_counts = sizeof(stein_counts) / sizeof(stein_counts[0]);
    
    int total_steps = warmup + n_steps;
    float* returns = (float*)malloc(total_steps * sizeof(float));
    generate_sv_returns(returns, total_steps, -4.5f, 0.98f, 0.10f, 12345);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\n");
    printf("  Stein |   Mean   |   P50    |   P99    |   Max    | Per-Stein\n");
    printf("  ------+----------+----------+----------+----------+----------\n");
    
    for (int i = 0; i < n_counts; i++) {
        int n_stein = stein_counts[i];
        
        SVPFState* state = svpf_create(n_particles, n_stein, 7.0f, nullptr);
        SVPFParams params = {0.98f, 0.10f, -4.5f, 0.0f};
        svpf_initialize(state, &params, 12345);
        
        std::vector<float> times;
        times.reserve(n_steps);
        
        float y_prev = 0.0f;
        
        // Warmup
        for (int t = 0; t < warmup; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
            y_prev = returns[t];
        }
        cudaDeviceSynchronize();
        
        // Profile
        for (int t = warmup; t < total_steps; t++) {
            cudaEventRecord(start);
            
            float loglik, vol, h_mean;
            svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms * 1000.0f);
            
            y_prev = returns[t];
        }
        
        TimingStats s = compute_stats(times);
        printf("  %5d | %6.1f μs | %6.1f μs | %6.1f μs | %6.1f μs | %5.1f μs\n",
               n_stein, s.mean_us, s.p50_us, s.p99_us, s.max_us, s.mean_us / n_stein);
        
        svpf_destroy(state);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(returns);
}

// =============================================================================
// Profile: Detailed Overhead Breakdown
// =============================================================================

static void profile_overhead_breakdown(int n_particles, int n_steps) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Profile: Overhead Breakdown (N=%d)\n", n_particles);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    // Generate test data
    float* returns = (float*)malloc((n_steps + 100) * sizeof(float));
    generate_sv_returns(returns, n_steps + 100, -4.5f, 0.98f, 0.10f, 12345);
    
    // Allocate device memory for manual timing
    float* d_y;
    float* d_result;
    cudaMalloc(&d_y, 2 * sizeof(float));
    cudaMalloc(&d_result, 4 * sizeof(float));
    
    cudaEvent_t e_start, e_memcpy_up, e_sync1, e_launch, e_sync2, e_memcpy_down, e_end;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_memcpy_up);
    cudaEventCreate(&e_sync1);
    cudaEventCreate(&e_launch);
    cudaEventCreate(&e_sync2);
    cudaEventCreate(&e_memcpy_down);
    cudaEventCreate(&e_end);
    
    // Test: Measure individual overheads
    printf("\n  Component timings (averaged over %d iterations):\n\n", n_steps);
    
    // 1. Measure cudaMemcpyAsync H2D (small - 2 floats)
    std::vector<float> t_memcpy_h2d;
    for (int i = 0; i < n_steps; i++) {
        float y_arr[2] = {returns[i], returns[i+1]};
        cudaEventRecord(e_start);
        cudaMemcpyAsync(d_y, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, 0);
        cudaEventRecord(e_end);
        cudaEventSynchronize(e_end);
        float ms;
        cudaEventElapsedTime(&ms, e_start, e_end);
        t_memcpy_h2d.push_back(ms * 1000.0f);
    }
    TimingStats s_h2d = compute_stats(t_memcpy_h2d);
    printf("    cudaMemcpyAsync H2D (8B):  mean=%5.1f μs, p99=%5.1f μs\n", s_h2d.mean_us, s_h2d.p99_us);
    
    // 2. Measure cudaStreamSynchronize (empty stream)
    std::vector<float> t_sync_empty;
    for (int i = 0; i < n_steps; i++) {
        cudaEventRecord(e_start);
        cudaStreamSynchronize(0);
        cudaEventRecord(e_end);
        cudaEventSynchronize(e_end);
        float ms;
        cudaEventElapsedTime(&ms, e_start, e_end);
        t_sync_empty.push_back(ms * 1000.0f);
    }
    TimingStats s_sync = compute_stats(t_sync_empty);
    printf("    cudaStreamSynchronize:     mean=%5.1f μs, p99=%5.1f μs\n", s_sync.mean_us, s_sync.p99_us);
    
    // 3. Measure cudaMemcpy D2H (synchronous - 4 floats)
    std::vector<float> t_memcpy_d2h;
    float result_buf[4];
    for (int i = 0; i < n_steps; i++) {
        cudaEventRecord(e_start);
        cudaMemcpy(result_buf, d_result, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(e_end);
        cudaEventSynchronize(e_end);
        float ms;
        cudaEventElapsedTime(&ms, e_start, e_end);
        t_memcpy_d2h.push_back(ms * 1000.0f);
    }
    TimingStats s_d2h = compute_stats(t_memcpy_d2h);
    printf("    cudaMemcpy D2H (16B):      mean=%5.1f μs, p99=%5.1f μs\n", s_d2h.mean_us, s_d2h.p99_us);
    
    // 4. Measure 4x separate cudaMemcpy D2H (current code does this!)
    std::vector<float> t_memcpy_d2h_4x;
    for (int i = 0; i < n_steps; i++) {
        cudaEventRecord(e_start);
        cudaMemcpy(&result_buf[0], d_result + 0, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result_buf[1], d_result + 1, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result_buf[2], d_result + 2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result_buf[3], d_result + 3, sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(e_end);
        cudaEventSynchronize(e_end);
        float ms;
        cudaEventElapsedTime(&ms, e_start, e_end);
        t_memcpy_d2h_4x.push_back(ms * 1000.0f);
    }
    TimingStats s_d2h_4x = compute_stats(t_memcpy_d2h_4x);
    printf("    4x cudaMemcpy D2H (4B ea): mean=%5.1f μs, p99=%5.1f μs  ← CURRENT CODE!\n", 
           s_d2h_4x.mean_us, s_d2h_4x.p99_us);
    
    // 5. Now profile actual svpf_step_graph
    SVPFState* state = svpf_create(n_particles, 5, 7.0f, nullptr);
    SVPFParams params = {0.98f, 0.10f, -4.5f, 0.0f};
    svpf_initialize(state, &params, 12345);
    
    // Warmup
    float y_prev = 0.0f;
    for (int t = 0; t < 100; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        y_prev = returns[t];
    }
    cudaDeviceSynchronize();
    
    std::vector<float> t_total;
    for (int t = 100; t < n_steps + 100; t++) {
        cudaEventRecord(e_start);
        float loglik, vol, h_mean;
        svpf_step_graph(state, returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        cudaEventRecord(e_end);
        cudaEventSynchronize(e_end);
        float ms;
        cudaEventElapsedTime(&ms, e_start, e_end);
        t_total.push_back(ms * 1000.0f);
        y_prev = returns[t];
    }
    TimingStats s_total = compute_stats(t_total);
    printf("\n    TOTAL svpf_step_graph:     mean=%5.1f μs, p99=%5.1f μs\n", 
           s_total.mean_us, s_total.p99_us);
    
    // Estimate breakdown
    float overhead_sync = 2 * s_sync.mean_us;  // Two syncs
    float overhead_memcpy = s_d2h_4x.mean_us + 3 * s_h2d.mean_us;  // 4 D2H + 3 H2D
    float estimated_kernels = s_total.mean_us - overhead_sync - overhead_memcpy;
    
    printf("\n  Estimated breakdown:\n");
    printf("    Sync overhead (2x):        ~%5.1f μs (%.0f%%)\n", 
           overhead_sync, 100.0f * overhead_sync / s_total.mean_us);
    printf("    Memcpy overhead:           ~%5.1f μs (%.0f%%)\n", 
           overhead_memcpy, 100.0f * overhead_memcpy / s_total.mean_us);
    printf("    Actual kernels:            ~%5.1f μs (%.0f%%)\n",
           estimated_kernels, 100.0f * estimated_kernels / s_total.mean_us);
    
    printf("\n  OPTIMIZATION OPPORTUNITY:\n");
    printf("    Batch 4 D2H memcpy → 1:    save ~%.0f μs (%.0f%%)\n",
           s_d2h_4x.mean_us - s_d2h.mean_us,
           100.0f * (s_d2h_4x.mean_us - s_d2h.mean_us) / s_total.mean_us);
    
    // Cleanup
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_memcpy_up);
    cudaEventDestroy(e_sync1);
    cudaEventDestroy(e_launch);
    cudaEventDestroy(e_sync2);
    cudaEventDestroy(e_memcpy_down);
    cudaEventDestroy(e_end);
    cudaFree(d_y);
    cudaFree(d_result);
    svpf_destroy(state);
    free(returns);
}

// =============================================================================
// Profile: Memory Bandwidth Estimate
// =============================================================================

static void profile_memory_estimate(int n_particles) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Memory Estimate (N=%d)\n", n_particles);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    // Per-step memory access estimate
    // Read: h, h_prev, rng_state
    // Write: h, h_prev, h_pred, grad_log_p, log_weights
    
    size_t bytes_per_particle = 
        4 * sizeof(float) +      // h (read + write)
        4 * sizeof(float) +      // h_prev (read + write)
        sizeof(float) +          // h_pred (write)
        sizeof(float) +          // grad_log_p
        sizeof(float) +          // log_weights
        sizeof(float) * 3;       // kernel_sum, grad_kernel_sum, etc.
    
    size_t total_bytes = n_particles * bytes_per_particle;
    float total_mb = total_bytes / (1024.0f * 1024.0f);
    
    printf("\n");
    printf("  Per-particle: ~%zu bytes\n", bytes_per_particle);
    printf("  Total per step: ~%.2f MB\n", total_mb);
    printf("\n");
    
    // Estimate bandwidth requirement
    // If step takes 50μs, need 50MB/50μs = 1 TB/s (typical GPU: 500-900 GB/s)
    printf("  At 50μs/step: need %.0f GB/s bandwidth\n", total_mb * 1000.0f / 50.0f);
    printf("  At 100μs/step: need %.0f GB/s bandwidth\n", total_mb * 1000.0f / 100.0f);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SVPF Performance Profiler                                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    // Defaults
    int n_particles = 512;
    int n_steps = 5000;
    int warmup = 100;
    
    // Parse args
    if (argc > 1) n_particles = atoi(argv[1]);
    if (argc > 2) n_steps = atoi(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    
    printf("\n Config: particles=%d, steps=%d, warmup=%d\n", n_particles, n_steps, warmup);
    
    // GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf(" GPU: %s (SM %d.%d, %d SMs)\n", 
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    
    // Run profiles
    profile_step_latency(n_particles, 5, n_steps, warmup);
    profile_particle_scaling(n_steps, warmup);
    profile_stein_scaling(n_particles, n_steps, warmup);
    profile_overhead_breakdown(n_particles, n_steps);
    profile_memory_estimate(n_particles);
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" Done. For kernel-level analysis, use:\n");
    printf("   ncu --set full ./profile_svpf %d %d %d\n", n_particles, n_steps, warmup);
    printf("   nsys profile ./profile_svpf %d %d %d\n", n_particles, n_steps, warmup);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
