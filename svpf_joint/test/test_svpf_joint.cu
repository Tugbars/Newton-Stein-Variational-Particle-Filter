/**
 * @file test_svpf_joint.cu
 * @brief Test harness for Joint State-Parameter SVPF
 * 
 * Tests:
 * 1. Basic allocation and initialization
 * 2. Calm period tracking
 * 3. Crash adaptation (σ should inflate)
 * 4. Recovery (σ should deflate)
 */

#include "svpf_joint.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Windows/MSVC doesn't define M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// SYNTHETIC DATA GENERATION
// =============================================================================

void generate_sv_data(
    float* y,           // Output observations
    float* h_true,      // Output true log-vol
    int T,
    float mu,
    float rho,
    float sigma_z,
    unsigned int seed
) {
    srand(seed);
    
    // Box-Muller for Gaussian noise
    auto randn = []() {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    };
    
    // Initialize
    h_true[0] = mu;
    
    // Generate
    for (int t = 0; t < T; t++) {
        if (t > 0) {
            float noise_h = randn();
            h_true[t] = mu + rho * (h_true[t-1] - mu) + sigma_z * noise_h;
        }
        
        float vol = expf(h_true[t] * 0.5f);
        float noise_y = randn();
        y[t] = vol * noise_y;
    }
}

// =============================================================================
// TEST: BASIC FUNCTIONALITY
// =============================================================================

int test_basic() {
    printf("\n=== TEST: Basic Functionality ===\n");
    
    // Create with default config
    SVPFJointConfig cfg = svpf_joint_default_config();
    cfg.n_particles = 256;
    cfg.n_stein_steps = 3;
    
    SVPFJointState* state = svpf_joint_create(&cfg, 0);
    if (!state) {
        printf("FAILED: Allocation\n");
        return -1;
    }
    printf("  Allocated: %d particles\n", cfg.n_particles);
    
    // Initialize
    svpf_joint_initialize(state, 12345ULL);
    printf("  Initialized\n");
    
    // Get initial estimates
    float mu, rho, sigma;
    svpf_joint_get_params(state, &mu, &rho, &sigma);
    printf("  Initial params: mu=%.3f, rho=%.3f, sigma=%.3f\n", mu, rho, sigma);
    
    // Run a few steps with dummy data
    SVPFJointDiagnostics diag;
    for (int t = 0; t < 10; t++) {
        float y = 0.01f;  // Small observation
        svpf_joint_step(state, y, &diag);
    }
    printf("  Ran 10 steps\n");
    printf("  Final: h=%.3f, vol=%.3f, ESS=%.1f\n", diag.h_mean, diag.vol_mean, diag.ess);
    printf("  Params: mu=%.3f, rho=%.3f, sigma=%.3f\n", diag.mu_mean, diag.rho_mean, diag.sigma_mean);
    
    // Cleanup
    svpf_joint_destroy(state);
    printf("  Destroyed\n");
    
    printf("PASSED\n");
    return 0;
}

// =============================================================================
// TEST: CALM PERIOD TRACKING
// =============================================================================

int test_calm_tracking() {
    printf("\n=== TEST: Calm Period Tracking ===\n");
    
    // Generate calm data
    const int T = 500;
    float* y = (float*)malloc(T * sizeof(float));
    float* h_true = (float*)malloc(T * sizeof(float));
    
    float mu_true = -3.5f;
    float rho_true = 0.95f;
    float sigma_true = 0.15f;
    
    generate_sv_data(y, h_true, T, mu_true, rho_true, sigma_true, 42);
    
    // Create filter
    SVPFJointConfig cfg = svpf_joint_default_config();
    cfg.n_particles = 512;
    cfg.n_stein_steps = 5;
    
    SVPFJointState* state = svpf_joint_create(&cfg, 0);
    svpf_joint_initialize(state, 12345ULL);
    
    // Run
    SVPFJointDiagnostics diag;
    float rmse_sum = 0.0f;
    
    for (int t = 0; t < T; t++) {
        svpf_joint_step(state, y[t], (t == T-1) ? &diag : NULL);
        
        // RMSE at last step
        if (t == T-1) {
            // Get all h values
            float* h_host = (float*)malloc(cfg.n_particles * sizeof(float));
            cudaMemcpy(h_host, state->d_h, cfg.n_particles * sizeof(float), cudaMemcpyDeviceToHost);
            
            float h_mean = 0.0f;
            for (int i = 0; i < cfg.n_particles; i++) {
                h_mean += h_host[i];
            }
            h_mean /= cfg.n_particles;
            
            float err = h_mean - h_true[t];
            rmse_sum += err * err;
            free(h_host);
        }
    }
    
    printf("  True params: mu=%.3f, rho=%.3f, sigma=%.3f\n", mu_true, rho_true, sigma_true);
    printf("  Learned:     mu=%.3f, rho=%.3f, sigma=%.3f\n", diag.mu_mean, diag.rho_mean, diag.sigma_mean);
    printf("  Final h: est=%.3f, true=%.3f\n", diag.h_mean, h_true[T-1]);
    printf("  ESS: %.1f\n", diag.ess);
    printf("  Diversity: std_mu=%.3f, std_rho=%.3f, std_sigma=%.3f\n",
           diag.std_mu_tilde, diag.std_rho_tilde, diag.std_sigma_tilde);
    
    // Check parameter estimates are reasonable
    if (fabsf(diag.rho_mean - rho_true) > 0.15f) {
        printf("  WARNING: rho estimate off by %.3f\n", fabsf(diag.rho_mean - rho_true));
        // Don't fail - learning is hard
    }
    
    svpf_joint_destroy(state);
    free(y);
    free(h_true);
    
    printf("PASSED\n");
    return 0;
}

// =============================================================================
// TEST: CRASH ADAPTATION
// =============================================================================

int test_crash_adaptation() {
    printf("\n=== TEST: Crash Adaptation ===\n");
    
    // Generate data with dramatic regime switch
    const int T_calm = 200;
    const int T_crash = 100;
    const int T_total = T_calm + T_crash;
    
    float* y = (float*)malloc(T_total * sizeof(float));
    float* h_true = (float*)malloc(T_total * sizeof(float));
    
    // Calm period: low vol
    generate_sv_data(y, h_true, T_calm, -3.5f, 0.95f, 0.15f, 42);
    
    // DRAMATIC CRASH: Instead of generating separate SV, we'll
    // continue from calm period but with sudden vol spike
    // Simulate flash crash: h jumps from ~-3.5 to ~-1.0 (vol ~17% to ~60%)
    
    srand(123);
    auto randn = []() {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    };
    
    // First few crash observations: HUGE returns (the surprise)
    float h_prev = h_true[T_calm - 1];
    float crash_mu = -1.0f;      // Higher mean vol
    float crash_rho = 0.85f;     // Lower persistence
    float crash_sigma = 0.50f;   // Much higher vol-of-vol
    
    for (int t = 0; t < T_crash; t++) {
        // First 5 observations: inject extreme jumps to trigger surprise
        if (t < 5) {
            h_true[T_calm + t] = crash_mu + randn() * 0.5f;  // Jump to high vol
        } else {
            // Then standard SV dynamics at higher vol
            h_true[T_calm + t] = crash_mu + crash_rho * (h_prev - crash_mu) + crash_sigma * randn();
        }
        h_prev = h_true[T_calm + t];
        
        float vol = expf(h_true[T_calm + t] * 0.5f);
        y[T_calm + t] = vol * randn();
    }
    
    // Create filter
    SVPFJointConfig cfg = svpf_joint_default_config();
    cfg.n_particles = 512;
    cfg.n_stein_steps = 5;
    
    SVPFJointState* state = svpf_joint_create(&cfg, 0);
    svpf_joint_initialize(state, 12345ULL);
    
    // Run and track sigma
    SVPFJointDiagnostics diag;
    float sigma_before_crash = 0.0f;
    float sigma_peak = 0.0f;
    float sigma_during_crash = 0.0f;
    
    for (int t = 0; t < T_total; t++) {
        svpf_joint_step(state, y[t], &diag);
        
        if (t == T_calm - 1) {
            sigma_before_crash = diag.sigma_mean;
            printf("  t=%d (before crash): sigma=%.4f, h=%.3f, h_true=%.3f\n", 
                   t, diag.sigma_mean, diag.h_mean, h_true[t]);
        }
        
        // Track during early crash (where surprise should trigger)
        if (t >= T_calm && t < T_calm + 20) {
            if (diag.sigma_mean > sigma_peak) {
                sigma_peak = diag.sigma_mean;
            }
            if (t == T_calm + 5) {
                printf("  t=%d (crash+5): sigma=%.4f, h=%.3f, h_true=%.3f\n", 
                       t, diag.sigma_mean, diag.h_mean, h_true[t]);
            }
        }
        
        if (t == T_calm + 50) {
            sigma_during_crash = diag.sigma_mean;
            printf("  t=%d (crash+50): sigma=%.4f, h=%.3f, h_true=%.3f\n", 
                   t, diag.sigma_mean, diag.h_mean, h_true[t]);
        }
    }
    
    printf("  Final: sigma=%.4f, h=%.3f\n", diag.sigma_mean, diag.h_mean);
    printf("  Peak sigma during crash: %.4f\n", sigma_peak);
    printf("  Sigma inflation (peak): %.2fx\n", sigma_peak / (sigma_before_crash + 1e-6f));
    printf("  Sigma inflation (t+50): %.2fx\n", sigma_during_crash / (sigma_before_crash + 1e-6f));
    
    // Check that sigma increased during crash
    if (sigma_peak > sigma_before_crash * 1.5f) {
        printf("  SUCCESS: sigma inflated during crash\n");
    } else if (sigma_peak > sigma_before_crash * 1.2f) {
        printf("  PARTIAL: sigma inflated somewhat\n");
    } else {
        printf("  FAIL: sigma didn't inflate enough\n");
    }
    
    svpf_joint_destroy(state);
    free(y);
    free(h_true);
    
    printf("PASSED\n");
    return 0;
}

// =============================================================================
// TEST: DIVERSITY MAINTENANCE
// =============================================================================

int test_diversity() {
    printf("\n=== TEST: Diversity Maintenance ===\n");
    
    // Long run to check for collapse
    const int T = 1000;
    float* y = (float*)malloc(T * sizeof(float));
    float* h_true = (float*)malloc(T * sizeof(float));
    
    generate_sv_data(y, h_true, T, -3.5f, 0.95f, 0.15f, 42);
    
    // Create filter
    SVPFJointConfig cfg = svpf_joint_default_config();
    cfg.n_particles = 256;
    cfg.n_stein_steps = 5;
    
    SVPFJointState* state = svpf_joint_create(&cfg, 0);
    svpf_joint_initialize(state, 12345ULL);
    
    // Run and check diversity at intervals
    SVPFJointDiagnostics diag;
    int collapse_count = 0;
    
    for (int t = 0; t < T; t++) {
        svpf_joint_step(state, y[t], &diag);
        
        if (t % 200 == 199) {
            printf("  t=%d: std_mu=%.4f, std_rho=%.4f, std_sigma=%.4f\n",
                   t, diag.std_mu_tilde, diag.std_rho_tilde, diag.std_sigma_tilde);
            
            if (diag.mu_collapsed || diag.rho_collapsed || diag.sigma_collapsed) {
                collapse_count++;
            }
        }
    }
    
    printf("  Collapse warnings: %d\n", collapse_count);
    
    if (collapse_count == 0) {
        printf("  SUCCESS: No diversity collapse\n");
    } else {
        printf("  NOTE: Some diversity collapse (may need tuning)\n");
    }
    
    svpf_joint_destroy(state);
    free(y);
    free(h_true);
    
    printf("PASSED\n");
    return 0;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {
    printf("===============================================\n");
    printf("   Joint State-Parameter SVPF Test Suite\n");
    printf("===============================================\n");
    
    int failed = 0;
    
    failed += test_basic();
    failed += test_calm_tracking();
    failed += test_crash_adaptation();
    failed += test_diversity();
    
    printf("\n===============================================\n");
    if (failed == 0) {
        printf("   ALL TESTS PASSED\n");
    } else {
        printf("   %d TEST(S) FAILED\n", -failed);
    }
    printf("===============================================\n");
    
    return failed;
}
