/**
 * @file test_offset_vs_difficulty.cu
 * @brief Test optimal lik_offset dependency on data difficulty/volatility
 * 
 * HYPOTHESIS: More challenging/volatile data requires stronger bias correction
 * 
 * TEST DESIGN:
 * - Generate 3 DGPs: EASY (σ=0.10), MEDIUM (σ=0.15), HARD (σ=0.25)
 * - For each DGP, sweep lik_offset ∈ [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
 * - Run bias compounding diagnostic
 * - Find which offset gives ratio closest to 1.0 (self-correcting, no compounding)
 * 
 * EXPECTED RESULT:
 * - Easy data:   optimal_offset ≈ 0.06-0.08
 * - Medium data: optimal_offset ≈ 0.08-0.10
 * - Hard data:   optimal_offset ≈ 0.12-0.14
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
// DGP Configuration
// =============================================================================

struct DGPConfig {
    const char* name;
    float rho;
    float sigma_z;
    float mu;
    float nu;
    float gamma;
};

// Three difficulty levels
const DGPConfig DGP_EASY = {
    "EASY (Calm)",
    0.95f,      // Lower persistence
    0.10f,      // Low volatility
    -3.5f,
    5.0f,
    0.0f
};

const DGPConfig DGP_MEDIUM = {
    "MEDIUM (Standard)",
    0.975f,
    0.15f,      // Standard volatility
    -3.5f,
    5.0f,
    0.0f
};

const DGPConfig DGP_HARD = {
    "HARD (Volatile)",
    0.98f,      // High persistence
    0.25f,      // High volatility
    -3.5f,
    5.0f,
    0.0f
};

// =============================================================================
// Ground Truth DGP Generator
// =============================================================================

struct GroundTruth {
    std::vector<float> h_true;
    std::vector<float> y_obs;
};

GroundTruth generate_synthetic_data(const DGPConfig& config, int T, unsigned long long seed) {
    GroundTruth gt;
    gt.h_true.resize(T + 1);
    gt.y_obs.resize(T);
    
    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    
    std::vector<float> normals(T + 1);
    curandGenerateNormal(gen, normals.data(), T + 1, 0.0f, 1.0f);
    
    float stationary_std = config.sigma_z / sqrtf(1.0f - config.rho * config.rho);
    gt.h_true[0] = config.mu + stationary_std * normals[0];
    
    for (int t = 0; t < T; t++) {
        gt.h_true[t + 1] = config.mu + config.rho * (gt.h_true[t] - config.mu) 
                         + config.sigma_z * normals[t + 1];
        
        float vol = expf(gt.h_true[t + 1] / 2.0f);
        float scale = vol * sqrtf(config.nu / (config.nu - 2.0f));
        gt.y_obs[t] = scale * normals[t];
    }
    
    curandDestroyGenerator(gen);
    return gt;
}

// =============================================================================
// Bias Measurement
// =============================================================================

struct BiasMetrics {
    float mean_error;
    float rmse;
};

BiasMetrics measure_bias(SVPFState* state, float h_true) {
    int n = state->n_particles;
    std::vector<float> h_particles(n);
    
    cudaMemcpy(h_particles.data(), state->h, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float error = h_particles[i] - h_true;
        sum += error;
        sum_sq += error * error;
    }
    
    BiasMetrics metrics;
    metrics.mean_error = sum / n;
    metrics.rmse = sqrtf(sum_sq / n);
    
    return metrics;
}

// =============================================================================
// Reset Particles to Ground Truth
// =============================================================================

void reset_particles_to_truth(SVPFState* state, float h_true, float std_dev) {
    int n = state->n_particles;
    int grid = (n + SVPF_BLOCK_SIZE - 1) / SVPF_BLOCK_SIZE;
    
    svpf_init_particles_kernel<<<grid, SVPF_BLOCK_SIZE, 0, state->stream>>>(
        state->h, state->rng_states, h_true, std_dev, n
    );
    
    cudaStreamSynchronize(state->stream);
}

// =============================================================================
// Single Diagnostic Run
// =============================================================================

float run_diagnostic(const DGPConfig& config, float lik_offset, 
                     int T, int reset_point, int n_particles, unsigned long long seed) {
    
    GroundTruth gt = generate_synthetic_data(config, T, seed);
    
    SVPFParams params;
    params.rho = config.rho;
    params.sigma_z = config.sigma_z;
    params.mu = config.mu;
    params.gamma = config.gamma;
    
    // ==========================================================================
    // EXPERIMENT A: Normal run (no reset)
    // ==========================================================================
    
    SVPFState* state_A = svpf_create(n_particles, 12, config.nu, 0);
    
    state_A->use_decorrelation = 1;
    state_A->decorrelation_interval = 25;
    state_A->decorrelation_scale = 0.15f;
    state_A->lik_offset = lik_offset;  // TEST PARAMETER
    
    svpf_initialize(state_A, &params, seed);
    
    for (int t = 0; t < T; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state_A, gt.y_obs[t], 
                       (t > 0) ? gt.y_obs[t-1] : 0.0f,
                       &params, &loglik, &vol, &h_mean);
    }
    
    BiasMetrics bias_no_reset = measure_bias(state_A, gt.h_true[reset_point + 1]);
    svpf_destroy(state_A);
    
    // ==========================================================================
    // EXPERIMENT B: Reset at t=50, measure t=51
    // ==========================================================================
    
    SVPFState* state_B = svpf_create(n_particles, 12, config.nu, 0);
    
    state_B->use_decorrelation = 1;
    state_B->decorrelation_interval = 25;
    state_B->decorrelation_scale = 0.15f;
    state_B->lik_offset = lik_offset;  // TEST PARAMETER
    
    svpf_initialize(state_B, &params, seed);
    
    for (int t = 0; t < reset_point; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state_B, gt.y_obs[t],
                       (t > 0) ? gt.y_obs[t-1] : 0.0f,
                       &params, &loglik, &vol, &h_mean);
    }
    
    reset_particles_to_truth(state_B, gt.h_true[reset_point], 0.05f);
    
    float loglik_51, vol_51, h_mean_51;
    svpf_step_graph(state_B, gt.y_obs[reset_point],
                   gt.y_obs[reset_point - 1],
                   &params, &loglik_51, &vol_51, &h_mean_51);
    
    BiasMetrics bias_with_reset = measure_bias(state_B, gt.h_true[reset_point + 1]);
    svpf_destroy(state_B);
    
    // ==========================================================================
    // Compute Ratio
    // ==========================================================================
    
    float ratio = fabsf(bias_with_reset.mean_error) / 
                  fmaxf(fabsf(bias_no_reset.mean_error), 1e-6f);
    
    return ratio;
}

// =============================================================================
// Main Test
// =============================================================================

int main() {
    printf("=============================================================================\n");
    printf("LIK_OFFSET DEPENDENCY ON DATA DIFFICULTY TEST\n");
    printf("=============================================================================\n\n");
    
    printf("HYPOTHESIS: More volatile data requires stronger bias correction\n\n");
    
    const int T = 100;
    const int reset_point = 50;
    const int n_particles = 256;
    const unsigned long long seed = 12345;
    
    // Offset values to test
    const float offsets[] = {0.06f, 0.08f, 0.10f, 0.12f, 0.14f, 0.16f};
    const int n_offsets = sizeof(offsets) / sizeof(offsets[0]);
    
    // DGP configurations
    const DGPConfig configs[] = {DGP_EASY, DGP_MEDIUM, DGP_HARD};
    const int n_configs = sizeof(configs) / sizeof(configs[0]);
    
    printf("Testing %d difficulty levels × %d offset values = %d combinations\n\n",
           n_configs, n_offsets, n_configs * n_offsets);
    
    // Results storage
    struct Result {
        const char* dgp_name;
        float sigma_z;
        float offset;
        float ratio;
        float distance_from_one;  // |ratio - 1.0|
    };
    
    std::vector<Result> results;
    
    // Run all combinations
    for (int cfg = 0; cfg < n_configs; cfg++) {
        const DGPConfig& config = configs[cfg];
        
        printf("--- Testing %s (σ=%.2f) ---\n", config.name, config.sigma_z);
        
        for (int off = 0; off < n_offsets; off++) {
            float offset = offsets[off];
            
            printf("  offset=%.2f ... ", offset);
            fflush(stdout);
            
            float ratio = run_diagnostic(config, offset, T, reset_point, n_particles, seed);
            
            Result r;
            r.dgp_name = config.name;
            r.sigma_z = config.sigma_z;
            r.offset = offset;
            r.ratio = ratio;
            r.distance_from_one = fabsf(ratio - 1.0f);
            
            results.push_back(r);
            
            printf("ratio=%.3f (dist=%.3f)\n", ratio, r.distance_from_one);
        }
        printf("\n");
    }
    
    // ==========================================================================
    // Analysis: Find optimal offset for each DGP
    // ==========================================================================
    
    printf("=============================================================================\n");
    printf("RESULTS: OPTIMAL LIK_OFFSET PER DIFFICULTY LEVEL\n");
    printf("=============================================================================\n\n");
    
    printf("Optimal = offset giving ratio closest to 1.0 (self-correcting, no compounding)\n\n");
    
    for (int cfg = 0; cfg < n_configs; cfg++) {
        const DGPConfig& config = configs[cfg];
        
        // Find best offset for this DGP
        float best_offset = 0.0f;
        float best_distance = 1e10f;
        float best_ratio = 0.0f;
        
        for (const auto& r : results) {
            if (r.sigma_z == config.sigma_z) {
                if (r.distance_from_one < best_distance) {
                    best_distance = r.distance_from_one;
                    best_offset = r.offset;
                    best_ratio = r.ratio;
                }
            }
        }
        
        printf("%s (σ=%.2f):\n", config.name, config.sigma_z);
        printf("  Optimal offset: %.2f\n", best_offset);
        printf("  Ratio at optimal: %.3f (distance from 1.0: %.3f)\n", best_ratio, best_distance);
        printf("\n");
    }
    
    // ==========================================================================
    // Full Results Table
    // ==========================================================================
    
    printf("=============================================================================\n");
    printf("DETAILED RESULTS TABLE\n");
    printf("=============================================================================\n\n");
    
    printf("DGP                | σ_z  | offset | ratio | distance | status\n");
    printf("-------------------|------|--------|-------|----------|------------------\n");
    
    for (const auto& r : results) {
        const char* status;
        if (r.distance_from_one < 0.2f) {
            status = "EXCELLENT ✓";
        } else if (r.distance_from_one < 0.4f) {
            status = "GOOD";
        } else if (r.ratio < 0.7f) {
            status = "COMPOUNDS";
        } else if (r.ratio > 1.3f) {
            status = "OVER-CORRECTS";
        } else {
            status = "ACCEPTABLE";
        }
        
        printf("%-18s | %.2f | %.2f   | %.3f | %.3f    | %s\n",
               r.dgp_name, r.sigma_z, r.offset, r.ratio, r.distance_from_one, status);
    }
    
    printf("\n");
    
    // ==========================================================================
    // Validation of Hypothesis
    // ==========================================================================
    
    printf("=============================================================================\n");
    printf("HYPOTHESIS VALIDATION\n");
    printf("=============================================================================\n\n");
    
    // Extract optimal offsets
    float opt_easy = 0.0f, opt_medium = 0.0f, opt_hard = 0.0f;
    
    for (int cfg = 0; cfg < n_configs; cfg++) {
        float best_offset = 0.0f;
        float best_distance = 1e10f;
        
        for (const auto& r : results) {
            if (r.sigma_z == configs[cfg].sigma_z && r.distance_from_one < best_distance) {
                best_distance = r.distance_from_one;
                best_offset = r.offset;
            }
        }
        
        if (cfg == 0) opt_easy = best_offset;
        else if (cfg == 1) opt_medium = best_offset;
        else if (cfg == 2) opt_hard = best_offset;
    }
    
    printf("Optimal offsets by difficulty:\n");
    printf("  EASY   (σ=0.10): %.2f\n", opt_easy);
    printf("  MEDIUM (σ=0.15): %.2f\n", opt_medium);
    printf("  HARD   (σ=0.25): %.2f\n\n", opt_hard);
    
    bool hypothesis_confirmed = (opt_hard >= opt_medium) && (opt_medium >= opt_easy);
    
    if (hypothesis_confirmed) {
        printf("✓ HYPOTHESIS CONFIRMED\n");
        printf("  More volatile data requires stronger bias correction\n");
        printf("  Relationship: lik_offset ∝ σ_z (data volatility)\n");
    } else {
        printf("✗ HYPOTHESIS NOT CONFIRMED\n");
        printf("  Optimal offset does not monotonically increase with volatility\n");
    }
    
    printf("\n=============================================================================\n");
    
    return 0;
}
