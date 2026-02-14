/**
 * @file test_bias_compounding.cu
 * @brief Diagnostic test for temperature-induced bias compounding in SVPF
 * 
 * EXPERIMENT:
 * 1. Run SVPF from t=0 to t=100 on synthetic data (known ground truth)
 * 2. Measure bias at t=1, 10, 20, 50, 100
 * 3. At t=50, reset particles to ground truth h_50
 * 4. Run one more step to t=51
 * 5. Compare: bias(t=51 | reset) vs bias(t=51 | no reset)
 * 
 * INTERPRETATION:
 * - If bias(51|reset) ≈ bias(1): Bias is LOCAL (not compounding)
 * - If bias(51|reset) << bias(51|no_reset): Bias COMPOUNDS across timesteps
 */

#include "svpf_kernels.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Ground Truth DGP (Oracle SV Model)
// =============================================================================

struct GroundTruth {
    float rho;
    float sigma_z;
    float mu;
    float nu;
    float gamma;
    
    std::vector<float> h_true;  // True log-volatility
    std::vector<float> y_obs;   // Observations
};

// Generate synthetic data from known SV model
GroundTruth generate_synthetic_data(int T, unsigned long long seed) {
    GroundTruth gt;
    
    // DGP parameters (ground truth)
    gt.rho = 0.975f;
    gt.sigma_z = 0.15f;
    gt.mu = -3.5f;
    gt.nu = 5.0f;
    gt.gamma = 0.0f;  // No leverage for simplicity
    
    gt.h_true.resize(T + 1);
    gt.y_obs.resize(T);
    
    // Random number generator
    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    
    std::vector<float> normals(T + 1);
    curandGenerateNormal(gen, normals.data(), T + 1, 0.0f, 1.0f);
    
    // Initial state from stationary distribution
    float stationary_std = gt.sigma_z / sqrtf(1.0f - gt.rho * gt.rho);
    gt.h_true[0] = gt.mu + stationary_std * normals[0];
    
    // Generate trajectory
    for (int t = 0; t < T; t++) {
        // State transition
        gt.h_true[t + 1] = gt.mu + gt.rho * (gt.h_true[t] - gt.mu) 
                         + gt.sigma_z * normals[t + 1];
        
        // Observation (Student-t noise)
        float vol = expf(gt.h_true[t + 1] / 2.0f);
        
        // Approximate Student-t with mixture of normals for simplicity
        float scale = vol * sqrtf(gt.nu / (gt.nu - 2.0f));
        gt.y_obs[t] = scale * normals[t];
    }
    
    curandDestroyGenerator(gen);
    
    return gt;
}

// =============================================================================
// Bias Measurement
// =============================================================================

struct BiasMetrics {
    float mean_error;      // E[h_estimate - h_true]
    float abs_error;       // E[|h_estimate - h_true|]
    float rmse;            // sqrt(E[(h_estimate - h_true)^2])
    float h_true_mean;     // For reference
    float h_estimate_mean; // For reference
};

BiasMetrics measure_bias(SVPFState* state, float h_true) {
    int n = state->n_particles;
    std::vector<float> h_particles(n);
    
    cudaMemcpy(h_particles.data(), state->h, n * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    BiasMetrics metrics = {0};
    metrics.h_true_mean = h_true;
    
    float sum = 0.0f;
    float sum_abs = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float error = h_particles[i] - h_true;
        sum += error;
        sum_abs += fabsf(error);
        sum_sq += error * error;
    }
    
    metrics.mean_error = sum / n;
    metrics.abs_error = sum_abs / n;
    metrics.rmse = sqrtf(sum_sq / n);
    metrics.h_estimate_mean = metrics.h_true_mean + metrics.mean_error;
    
    return metrics;
}

// =============================================================================
// Reset Particles to Ground Truth
// =============================================================================

void reset_particles_to_truth(SVPFState* state, float h_true, float std_dev) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    // Initialize around ground truth with small noise
    svpf_init_particles_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->rng_states, h_true, std_dev, n
    );
    
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// Main Diagnostic Test
// =============================================================================

int main() {
    printf("=============================================================================\n");
    printf("SVPF BIAS COMPOUNDING DIAGNOSTIC TEST\n");
    printf("=============================================================================\n\n");
    
    // Configuration
    const int T = 100;              // Total timesteps
    const int reset_point = 50;     // When to reset particles
    const int n_particles = 256;
    const unsigned long long seed = 12345;
    
    // NOTE: To test WITHOUT decorrelation, set these to 0 after svpf_create():
    //   state->use_decorrelation = 0;
    //   state->decorrelation_interval = 0;
    //   state->decorrelation_scale = 0.0f;
    
    // Generate synthetic data
    printf("Generating synthetic data (T=%d)...\n", T);
    GroundTruth gt = generate_synthetic_data(T, seed);
    
    // SVPF parameters (match DGP)
    SVPFParams params;
    params.rho = gt.rho;
    params.sigma_z = gt.sigma_z;
    params.mu = gt.mu;
    params.gamma = 0.0f;
    
    printf("DGP: rho=%.3f, sigma_z=%.3f, mu=%.3f, nu=%.1f\n\n",
           gt.rho, gt.sigma_z, gt.mu, gt.nu);
    
    // ==========================================================================
    // EXPERIMENT A: Normal run (no reset)
    // ==========================================================================
    printf("--- EXPERIMENT A: Normal Run (No Reset) ---\n");
    
    SVPFState* state_A = svpf_create(n_particles, 12, gt.nu, 0);
    
    // Enable decorrelation (bias mitigation)
    state_A->use_decorrelation = 1;
    state_A->decorrelation_interval = 25;
    state_A->decorrelation_scale = 0.15f;
    
    printf("Decorrelation: %s (interval=%d, scale=%.2f)\n",
           state_A->use_decorrelation ? "ENABLED" : "DISABLED",
           state_A->decorrelation_interval,
           state_A->decorrelation_scale);
    
    svpf_initialize(state_A, &params, seed);
    
    std::vector<BiasMetrics> bias_A(T);
    int decorr_count = 0;
    
    for (int t = 0; t < T; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state_A, gt.y_obs[t], 
                       (t > 0) ? gt.y_obs[t-1] : 0.0f,
                       &params, &loglik, &vol, &h_mean);
        
        bias_A[t] = measure_bias(state_A, gt.h_true[t + 1]);
        
        // Track decorrelation events
        if (state_A->use_decorrelation && state_A->decorrelation_interval > 0) {
            if ((t > 0) && (t % state_A->decorrelation_interval == 0)) {
                decorr_count++;
            }
        }
        
        // Print key timesteps
        if (t == 0 || t == 9 || t == 19 || t == 49 || t == 50 || t == 99) {
            printf("t=%3d: bias=%.4f, abs_err=%.4f, RMSE=%.4f, h_true=%.3f, h_est=%.3f\n",
                   t + 1, bias_A[t].mean_error, bias_A[t].abs_error, 
                   bias_A[t].rmse, bias_A[t].h_true_mean, bias_A[t].h_estimate_mean);
        }
    }
    
    printf("Decorrelation fired %d times (expected: %d)\n", 
           decorr_count, T / state_A->decorrelation_interval);
    printf("\n");
    
    // ==========================================================================
    // EXPERIMENT B: Reset at t=50, measure t=51
    // ==========================================================================
    printf("--- EXPERIMENT B: Reset at t=%d ---\n", reset_point);
    
    SVPFState* state_B = svpf_create(n_particles, 12, gt.nu, 0);
    
    // Enable decorrelation (same settings as A)
    state_B->use_decorrelation = 1;
    state_B->decorrelation_interval = 25;
    state_B->decorrelation_scale = 0.15f;
    
    svpf_initialize(state_B, &params, seed);
    
    // Run up to reset point
    for (int t = 0; t < reset_point; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state_B, gt.y_obs[t],
                       (t > 0) ? gt.y_obs[t-1] : 0.0f,
                       &params, &loglik, &vol, &h_mean);
    }
    
    BiasMetrics bias_before_reset = measure_bias(state_B, gt.h_true[reset_point]);
    printf("Before reset (t=%d): bias=%.4f, RMSE=%.4f\n",
           reset_point, bias_before_reset.mean_error, bias_before_reset.rmse);
    
    // RESET particles to ground truth with small noise
    float reset_std = 0.05f;  // Small noise around truth
    printf("Resetting particles to h_true=%.3f ± %.3f...\n", 
           gt.h_true[reset_point], reset_std);
    reset_particles_to_truth(state_B, gt.h_true[reset_point], reset_std);
    
    BiasMetrics bias_after_reset = measure_bias(state_B, gt.h_true[reset_point]);
    printf("After reset (t=%d):  bias=%.4f, RMSE=%.4f\n",
           reset_point, bias_after_reset.mean_error, bias_after_reset.rmse);
    
    // Run ONE MORE STEP to t=51
    float loglik_51, vol_51, h_mean_51;
    svpf_step_graph(state_B, gt.y_obs[reset_point],
                   gt.y_obs[reset_point - 1],
                   &params, &loglik_51, &vol_51, &h_mean_51);
    
    BiasMetrics bias_after_step = measure_bias(state_B, gt.h_true[reset_point + 1]);
    printf("One step later (t=%d): bias=%.4f, RMSE=%.4f\n\n",
           reset_point + 1, bias_after_step.mean_error, bias_after_step.rmse);
    
    // ==========================================================================
    // ANALYSIS: Compare bias at t=51
    // ==========================================================================
    printf("=============================================================================\n");
    printf("DIAGNOSTIC RESULTS\n");
    printf("=============================================================================\n\n");
    
    BiasMetrics bias_51_no_reset = bias_A[reset_point];  // t=51 from experiment A
    BiasMetrics bias_51_with_reset = bias_after_step;    // t=51 from experiment B
    BiasMetrics bias_1 = bias_A[0];                      // t=1 baseline
    
    printf("Bias at t=1  (baseline):          %.4f (RMSE: %.4f)\n", 
           bias_1.mean_error, bias_1.rmse);
    printf("Bias at t=51 (no reset):          %.4f (RMSE: %.4f)\n",
           bias_51_no_reset.mean_error, bias_51_no_reset.rmse);
    printf("Bias at t=51 (after reset at 50): %.4f (RMSE: %.4f)\n\n",
           bias_51_with_reset.mean_error, bias_51_with_reset.rmse);
    
    // Compute diagnostic ratios
    float ratio_reset_vs_no_reset = fabsf(bias_51_with_reset.mean_error) / 
                                    fmaxf(fabsf(bias_51_no_reset.mean_error), 1e-6f);
    float ratio_reset_vs_baseline = fabsf(bias_51_with_reset.mean_error) /
                                    fmaxf(fabsf(bias_1.mean_error), 1e-6f);
    
    printf("Ratio: |bias(51|reset)| / |bias(51|no_reset)| = %.3f\n", ratio_reset_vs_no_reset);
    printf("Ratio: |bias(51|reset)| / |bias(1)|            = %.3f\n\n", ratio_reset_vs_baseline);
    
    // ==========================================================================
    // INTERPRETATION
    // ==========================================================================
    printf("INTERPRETATION:\n");
    printf("---------------\n");
    
    printf("Decorrelation was: %s\n", state_A->use_decorrelation ? "ENABLED" : "DISABLED");
    if (state_A->use_decorrelation) {
        printf("  Interval: %d, Scale: %.2f\n", 
               state_A->decorrelation_interval, state_A->decorrelation_scale);
    }
    printf("\n");
    
    if (ratio_reset_vs_no_reset < 0.7f) {
        printf("✓ BIAS COMPOUNDS: bias(51|reset) << bias(51|no_reset)\n");
        printf("  → Resetting particles significantly reduced bias\n");
        printf("  → Bias accumulates across timesteps\n");
        if (state_A->use_decorrelation) {
            printf("  → WARNING: Decorrelation is enabled but bias still compounds!\n");
            printf("  → Decorrelation may need stronger settings or alternative approach\n");
        } else {
            printf("  → Static lik_offset cannot fix this\n");
            printf("  → Need: periodic rejuvenation, drift tracking, or architectural change\n");
        }
    } else if (ratio_reset_vs_no_reset > 1.3f) {
        printf("✗ ANOMALY: bias(51|reset) > bias(51|no_reset)\n");
        printf("  → Unexpected result, may need more trials for statistical significance\n");
    } else {
        printf("✓ BIAS IS LOCAL: bias(51|reset) ≈ bias(51|no_reset)\n");
        printf("  → Reset did not significantly change bias\n");
        printf("  → Bias is per-observation, not cumulative\n");
        if (state_A->use_decorrelation) {
            printf("  → Decorrelation successfully breaks temporal correlation!\n");
        } else {
            printf("  → Static lik_offset is near-optimal\n");
            printf("  → Variation across observations is intrinsic to finite-particle SVLD\n");
        }
    }
    
    printf("\n");
    
    if (ratio_reset_vs_baseline < 1.3f && ratio_reset_vs_baseline > 0.7f) {
        printf("Additionally: bias(51|reset) ≈ bias(1)\n");
        printf("  → Single-step bias is consistent across trajectory\n");
        printf("  → Current lik_offset=%.2f is trajectory-averaged optimum\n", 
               state_A->lik_offset);
    }
    
    printf("\n=============================================================================\n");
    
    // Cleanup
    svpf_destroy(state_A);
    svpf_destroy(state_B);
    
    return 0;
}
