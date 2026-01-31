// =============================================================================
// Multi-Stream Parallel Filter Profiler
// =============================================================================
// Tests true parallel execution of multiple SVPF filters using async API

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// Simple SV data generator
static void generate_sv_returns(float* returns, int T, float mu, float rho, float sigma, int seed) {
    srand(seed);
    float h = mu;
    for (int t = 0; t < T; t++) {
        float z1 = ((float)rand() / RAND_MAX - 0.5f) * 3.46f;
        float z2 = ((float)rand() / RAND_MAX - 0.5f) * 3.46f;
        h = mu + rho * (h - mu) + sigma * z1;
        returns[t] = expf(h * 0.5f) * z2;
    }
}

struct TimingStats {
    float mean_us, p50_us, p99_us, max_us;
};

static TimingStats compute_stats(std::vector<float>& times) {
    std::sort(times.begin(), times.end());
    int n = times.size();
    float sum = 0;
    for (float t : times) sum += t;
    
    TimingStats s;
    s.mean_us = sum / n;
    s.p50_us = times[n / 2];
    s.p99_us = times[(int)(n * 0.99)];
    s.max_us = times[n - 1];
    return s;
}

// =============================================================================
// SEQUENTIAL BASELINE (old way - for comparison)
// =============================================================================
static float profile_sequential(
    std::vector<SVPFState*>& filters,
    std::vector<float*>& returns_all,
    std::vector<float>& y_prevs,
    const SVPFParams* params,
    int t
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Sequential: each filter blocks until complete
    for (size_t f = 0; f < filters.size(); f++) {
        float loglik, vol, h_mean;
        svpf_step_graph(filters[f], returns_all[f][t], y_prevs[f], params,
                       &loglik, &vol, &h_mean);
        y_prevs[f] = returns_all[f][t];
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms * 1000.0f;
}

// =============================================================================
// PARALLEL WITH ASYNC API (new way)
// =============================================================================
static float profile_parallel(
    std::vector<SVPFState*>& filters,
    std::vector<float*>& returns_all,
    std::vector<float>& y_prevs,
    const SVPFParams* params,
    int t
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Phase 1: Launch ALL filters (non-blocking)
    for (size_t f = 0; f < filters.size(); f++) {
        svpf_step_async(filters[f], returns_all[f][t], y_prevs[f], params);
    }
    
    // Phase 2: Sync ALL and read outputs
    for (size_t f = 0; f < filters.size(); f++) {
        float loglik, vol, h_mean;
        svpf_sync_outputs(filters[f], &loglik, &vol, &h_mean);
        y_prevs[f] = returns_all[f][t];
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms * 1000.0f;
}

// =============================================================================
// MAIN PROFILER
// =============================================================================
static void profile_multi_stream(int max_filters, int n_particles, int n_steps, int warmup) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Profile: Multi-Stream Parallel Scaling (N=%d per filter)\n", n_particles);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    int total_steps = warmup + n_steps;
    
    // Generate different data for each instrument
    std::vector<float*> returns_all;
    for (int f = 0; f < max_filters; f++) {
        float* returns = (float*)malloc(total_steps * sizeof(float));
        generate_sv_returns(returns, total_steps, -4.5f + 0.1f * f, 0.98f, 0.10f, 12345 + f * 1000);
        returns_all.push_back(returns);
    }
    
    printf("\n");
    printf("  Filters | Sequential |  Parallel  | Speedup | Par/Filter\n");
    printf("  --------+------------+------------+---------+-----------\n");
    
    int filter_counts[] = {1, 2, 4, 8, 12, 16, 20, 24, 32};
    int n_counts = sizeof(filter_counts) / sizeof(filter_counts[0]);
    
    for (int c = 0; c < n_counts; c++) {
        int n_filters = filter_counts[c];
        if (n_filters > max_filters) break;
        
        // Create streams - ONE PER FILTER (critical!)
        std::vector<cudaStream_t> streams(n_filters);
        for (int f = 0; f < n_filters; f++) {
            cudaStreamCreate(&streams[f]);
        }
        
        // Create filters with their own streams
        std::vector<SVPFState*> filters;
        SVPFParams params = {0.98f, 0.10f, -4.5f, 0.0f};
        
        for (int f = 0; f < n_filters; f++) {
            SVPFState* state = svpf_create(n_particles, 8, 5.0f, streams[f]);
            svpf_initialize(state, &params, 12345 + f);
            filters.push_back(state);
        }
        
        std::vector<float> y_prevs(n_filters, 0.0f);
        
        // Warmup
        for (int t = 0; t < warmup; t++) {
            for (int f = 0; f < n_filters; f++) {
                svpf_step_async(filters[f], returns_all[f][t], y_prevs[f], &params);
            }
            for (int f = 0; f < n_filters; f++) {
                float loglik, vol, h_mean;
                svpf_sync_outputs(filters[f], &loglik, &vol, &h_mean);
                y_prevs[f] = returns_all[f][t];
            }
        }
        cudaDeviceSynchronize();
        
        // Reset y_prevs for fair comparison
        std::fill(y_prevs.begin(), y_prevs.end(), 0.0f);
        
        // Profile SEQUENTIAL
        std::vector<float> seq_times;
        seq_times.reserve(n_steps);
        for (int t = warmup; t < total_steps; t++) {
            float us = profile_sequential(filters, returns_all, y_prevs, &params, t);
            seq_times.push_back(us);
        }
        TimingStats seq_stats = compute_stats(seq_times);
        
        // Reset y_prevs
        std::fill(y_prevs.begin(), y_prevs.end(), 0.0f);
        
        // Profile PARALLEL
        std::vector<float> par_times;
        par_times.reserve(n_steps);
        for (int t = warmup; t < total_steps; t++) {
            float us = profile_parallel(filters, returns_all, y_prevs, &params, t);
            par_times.push_back(us);
        }
        TimingStats par_stats = compute_stats(par_times);
        
        float speedup = seq_stats.mean_us / par_stats.mean_us;
        float per_filter = par_stats.mean_us / n_filters;
        
        printf("  %7d | %8.1f μs | %8.1f μs | %6.2fx | %8.1f μs\n",
               n_filters, seq_stats.mean_us, par_stats.mean_us, speedup, per_filter);
        
        // Cleanup
        for (SVPFState* state : filters) {
            svpf_destroy(state);
        }
        for (cudaStream_t s : streams) {
            cudaStreamDestroy(s);
        }
    }
    
    printf("\n");
    printf("  Sequential: svpf_step_graph() called one-by-one (blocking)\n");
    printf("  Parallel:   svpf_step_async() on all, then svpf_sync_outputs() on all\n");
    printf("  Speedup > 1.0x = GPU parallelism working\n");
    printf("  Per-Filter ≈ constant = perfect scaling\n");
    
    for (float* r : returns_all) free(r);
}

int main(int argc, char** argv) {
    int max_filters = 32;
    int n_particles = 512;
    int n_steps = 1000;
    int warmup = 100;
    
    if (argc > 1) max_filters = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    
    printf("SVPF Multi-Stream Parallel Profiler\n");
    printf("Max filters: %d, Particles: %d\n", max_filters, n_particles);
    
    profile_multi_stream(max_filters, n_particles, n_steps, warmup);
    
    return 0;
}
