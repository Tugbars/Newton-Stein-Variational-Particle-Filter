/**
 * @file svpf_test.cu
 * @brief Test and benchmark SVPF CUDA implementation
 */

#include "svpf.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Windows doesn't define M_PI by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// SYNTHETIC DATA GENERATION
// =============================================================================

void generate_synthetic_data(float* y, float* h_true, int T,
                             float rho, float sigma_z, float mu, int seed) {
    srand(seed);
    
    // Box-Muller for normal random numbers
    auto randn = []() -> float {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
    };
    
    // Stationary distribution
    float stationary_var = (sigma_z * sigma_z) / (1.0f - rho * rho);
    h_true[0] = mu + sqrtf(stationary_var) * randn();
    
    // Generate latent volatility
    for (int t = 1; t < T; t++) {
        h_true[t] = mu + rho * (h_true[t-1] - mu) + sigma_z * randn();
    }
    
    // Generate observations
    for (int t = 0; t < T; t++) {
        float vol = expf(h_true[t] / 2.0f);
        y[t] = vol * randn();
    }
}

// =============================================================================
// TEST 1: Basic Functionality
// =============================================================================

int test_basic_functionality() {
    printf("\n========================================\n");
    printf("TEST 1: Basic Functionality\n");
    printf("========================================\n");
    
    // Create filter
    SVPFState* state = svpf_create(512, 5, 5.0f, NULL);
    if (!state) {
        printf("FAILED: Could not create SVPF state\n");
        return 0;
    }
    printf("✓ Created SVPF state (N=%d, Stein steps=%d)\n", 512, 5);
    
    // Initialize
    SVPFParams params = {0.95f, 0.20f, -5.0f, 0.0f};  // rho, sigma_z, mu, gamma (no leverage)
    svpf_initialize(state, &params, 42);
    printf("✓ Initialized particles\n");
    
    // Get particles and check
    float* h_check = (float*)malloc(512 * sizeof(float));
    svpf_get_particles(state, h_check);
    
    float h_mean = 0.0f;
    for (int i = 0; i < 512; i++) {
        h_mean += h_check[i];
    }
    h_mean /= 512.0f;
    
    printf("  Particle mean: %.3f (expected: ~%.1f)\n", h_mean, params.mu);
    
    if (fabsf(h_mean - params.mu) > 2.0f) {
        printf("FAILED: Particle mean too far from mu\n");
        free(h_check);
        svpf_destroy(state);
        return 0;
    }
    printf("✓ Particle initialization correct\n");
    
    // Run a few steps
    float y_test[] = {0.05f, -0.03f, 0.10f, -0.15f, 0.02f};
    SVPFResult result;
    
    for (int t = 0; t < 5; t++) {
        svpf_step(state, y_test[t], &params, &result);
        printf("  Step %d: log_lik=%.2f, vol=%.4f\n", 
               t, result.log_lik_increment, result.vol_mean);
        
        if (!isfinite(result.log_lik_increment)) {
            printf("FAILED: Non-finite log-likelihood at step %d\n", t);
            free(h_check);
            svpf_destroy(state);
            return 0;
        }
    }
    printf("✓ Filter steps completed without errors\n");
    
    free(h_check);
    svpf_destroy(state);
    printf("✓ Cleanup successful\n");
    
    return 1;
}

// =============================================================================
// TEST 2: Volatility Tracking
// =============================================================================

int test_volatility_tracking() {
    printf("\n========================================\n");
    printf("TEST 2: Volatility Tracking\n");
    printf("========================================\n");
    
    // True parameters
    float true_rho = 0.95f;
    float true_sigma = 0.20f;
    float true_mu = -5.0f;
    int T = 500;
    
    // Generate synthetic data
    float* y = (float*)malloc(T * sizeof(float));
    float* h_true = (float*)malloc(T * sizeof(float));
    generate_synthetic_data(y, h_true, T, true_rho, true_sigma, true_mu, 42);
    
    printf("Generated %d observations\n", T);
    printf("True params: ρ=%.2f, σ_z=%.2f, μ=%.1f\n", true_rho, true_sigma, true_mu);
    
    // Create and run filter
    SVPFState* state = svpf_create(512, 5, 5.0f, NULL);
    SVPFParams params = {true_rho, true_sigma, true_mu, 0.0f};  // No leverage for basic test
    svpf_initialize(state, &params, 123);
    
    float* vol_est = (float*)malloc(T * sizeof(float));
    float* vol_true = (float*)malloc(T * sizeof(float));
    
    SVPFResult result;
    float total_log_lik = 0.0f;
    
    for (int t = 0; t < T; t++) {
        svpf_step(state, y[t], &params, &result);
        vol_est[t] = result.vol_mean;
        vol_true[t] = expf(h_true[t] / 2.0f);
        total_log_lik += result.log_lik_increment;
    }
    
    // Compute RMSE
    float mse = 0.0f;
    for (int t = 0; t < T; t++) {
        float err = vol_est[t] - vol_true[t];
        mse += err * err;
    }
    float rmse = sqrtf(mse / T);
    
    printf("\nResults:\n");
    printf("  Total log-likelihood: %.2f\n", total_log_lik);
    printf("  Volatility RMSE: %.4f\n", rmse);
    printf("  Mean true vol: %.4f\n", expf(true_mu / 2.0f));
    
    // Check RMSE is reasonable (should be < 0.05 for well-specified model)
    int success = rmse < 0.05f;
    if (success) {
        printf("✓ Volatility tracking PASSED (RMSE < 0.05)\n");
    } else {
        printf("✗ Volatility tracking FAILED (RMSE = %.4f)\n", rmse);
    }
    
    free(y);
    free(h_true);
    free(vol_est);
    free(vol_true);
    svpf_destroy(state);
    
    return success;
}

// =============================================================================
// TEST 3: Benchmark
// =============================================================================

void benchmark() {
    printf("\n========================================\n");
    printf("BENCHMARK\n");
    printf("========================================\n");
    
    int particle_counts[] = {256, 512, 1024, 2048};
    int n_configs = sizeof(particle_counts) / sizeof(particle_counts[0]);
    int T = 1000;
    int n_runs = 3;
    
    // Generate test data
    float* y = (float*)malloc(T * sizeof(float));
    float* h_true = (float*)malloc(T * sizeof(float));
    generate_synthetic_data(y, h_true, T, 0.95f, 0.20f, -5.0f, 42);
    
    printf("Sequence length: T=%d\n", T);
    printf("Warmup + %d timed runs\n\n", n_runs);
    
    printf("%-12s %-15s %-15s %-12s\n", "Particles", "Total (ms)", "Per-step (μs)", "Steps/sec");
    printf("%-12s %-15s %-15s %-12s\n", "--------", "----------", "------------", "---------");
    
    for (int c = 0; c < n_configs; c++) {
        int N = particle_counts[c];
        
        SVPFState* state = svpf_create(N, 5, 5.0f, NULL);
        SVPFParams params = {0.95f, 0.20f, -5.0f, 0.0f};
        SVPFResult result;
        
        // Warmup
        svpf_initialize(state, &params, 42);
        for (int t = 0; t < T; t++) {
            svpf_step(state, y[t], &params, &result);
        }
        
        // Timed runs
        double total_time = 0.0;
        for (int run = 0; run < n_runs; run++) {
            svpf_initialize(state, &params, 42 + run);
            
            cudaDeviceSynchronize();
            clock_t start = clock();
            
            for (int t = 0; t < T; t++) {
                svpf_step(state, y[t], &params, &result);
            }
            
            cudaDeviceSynchronize();
            clock_t end = clock();
            
            total_time += (double)(end - start) / CLOCKS_PER_SEC * 1000.0;  // ms
        }
        
        double avg_time_ms = total_time / n_runs;
        double per_step_us = avg_time_ms * 1000.0 / T;
        double steps_per_sec = T / (avg_time_ms / 1000.0);
        
        printf("%-12d %-15.2f %-15.1f %-12.0f\n", 
               N, avg_time_ms, per_step_us, steps_per_sec);
        
        svpf_destroy(state);
    }
    
    free(y);
    free(h_true);
}

// =============================================================================
// TEST 4: Seeded Step (CRN verification)
// =============================================================================

int test_crn() {
    printf("\n========================================\n");
    printf("TEST 4: Common Random Numbers (CRN)\n");
    printf("========================================\n");
    
    SVPFState* state1 = svpf_create(256, 3, 5.0f, NULL);
    SVPFState* state2 = svpf_create(256, 3, 5.0f, NULL);
    
    SVPFParams params1 = {0.95f, 0.20f, -5.0f, 0.0f};
    SVPFParams params2 = {0.96f, 0.21f, -5.1f, 0.0f};  // Slightly perturbed
    
    // Initialize both with same seed
    svpf_initialize(state1, &params1, 42);
    svpf_initialize(state2, &params2, 42);
    
    // Run steps with same RNG seed
    SVPFResult result1, result2;
    unsigned long long rng_seed = 12345;
    
    float y_test = 0.05f;
    
    svpf_step_seeded(state1, y_test, &params1, rng_seed, &result1);
    svpf_step_seeded(state2, y_test, &params2, rng_seed, &result2);
    
    float ll_diff = fabsf(result1.log_lik_increment - result2.log_lik_increment);
    
    printf("Params 1: ρ=%.2f, σ=%.2f → LL=%.4f\n", 
           params1.rho, params1.sigma_z, result1.log_lik_increment);
    printf("Params 2: ρ=%.2f, σ=%.2f → LL=%.4f\n", 
           params2.rho, params2.sigma_z, result2.log_lik_increment);
    printf("LL difference: %.4f\n", ll_diff);
    
    // With CRN, small param changes should give small LL changes
    int success = ll_diff < 5.0f;  // Reasonable threshold
    if (success) {
        printf("✓ CRN appears to be working (LL diff is reasonable)\n");
    } else {
        printf("✗ CRN may not be working (LL diff too large)\n");
    }
    
    svpf_destroy(state1);
    svpf_destroy(state2);
    
    return success;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("SVPF CUDA Test Suite\n");
    printf("====================\n");
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d, Max threads/block: %d\n", 
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    
    // Run tests
    int passed = 0;
    int total = 4;
    
    passed += test_basic_functionality();
    passed += test_volatility_tracking();
    passed += test_crn();
    
    benchmark();
    
    printf("\n========================================\n");
    printf("SUMMARY: %d/%d tests passed\n", passed, total - 1);  // -1 for benchmark
    printf("========================================\n");
    
    return (passed == total - 1) ? 0 : 1;
}
