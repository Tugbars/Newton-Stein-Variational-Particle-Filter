/**
 * @file svpf_multiscale_test.cu
 * @brief A/B Test: Multi-Scale SVPF vs Mono SVPF
 *
 * Compares:
 *   A: Mono SVPF (single filter, 1024 particles, full adaptive suite)
 *   B: Multi-Scale SVPF (REACTIVE + INERTIAL, 512 + 1024 particles)
 *
 * Metrics:
 *   - RMSE(h): Accuracy vs ground truth
 *   - RMSE(vol): Volatility estimation accuracy
 *   - Bias: Systematic over/under estimation
 *   - Lag correlation: Tracking quality
 *   - Runtime: Computational cost
 *
 * Scenarios tested:
 *   - Calm: Baseline performance
 *   - Crisis: Persistent high vol
 *   - Spike: Sudden jump (tests REACTIVE speed)
 *   - Regime shift: Permanent change (tests INERTIAL adaptation)
 *   - Asymmetric: Leverage effect
 *   - Jumps: MIM-like DGP
 */

#include "svpf_multiscale.cuh"
#include "svpf_test_framework.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>

/*═══════════════════════════════════════════════════════════════════════════
 * TEST CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

// Quick mode (default) vs Full mode (--full flag)
static bool g_full_mode = false;

#define N_SEEDS_QUICK   10
#define N_SEEDS_FULL    50
#define N_STEPS_QUICK   2000
#define N_STEPS_FULL    5000
#define WARMUP_STEPS    100

static int get_n_seeds() { return g_full_mode ? N_SEEDS_FULL : N_SEEDS_QUICK; }
static int get_n_steps() { return g_full_mode ? N_STEPS_FULL : N_STEPS_QUICK; }

// Mono SVPF config (baseline)
#define MONO_PARTICLES      1024
#define MONO_STEIN_STEPS    8

// Multi-scale config (already in svpf_multiscale.cuh defaults)
// REACTIVE: 512 particles, 4 stein steps
// INERTIAL: 1024 particles, 8 stein steps

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS STRUCT
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    float rmse_h;
    float rmse_vol;
    float bias_h;
    float mae_h;
    float correlation;
    float runtime_ms;
    
    // Multi-scale specific
    float rmse_reactive;
    float rmse_inertial;
    float transient_detection_rate;  // % of transients correctly identified
    float regime_detection_rate;     // % of regimes correctly identified
} MSTestMetrics;

typedef struct {
    float mean;
    float std;
    float median;
    float p5;
    float p95;
} MSMetricStats;

static MSMetricStats compute_stats(const std::vector<float>& values) {
    MSMetricStats s = {};
    int n = (int)values.size();
    if (n == 0) return s;
    
    for (float v : values) s.mean += v;
    s.mean /= n;
    
    for (float v : values) s.std += (v - s.mean) * (v - s.mean);
    s.std = sqrtf(s.std / (n - 1 + 1e-10f));
    
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    
    s.median = sorted[n / 2];
    s.p5 = sorted[(int)(0.05f * n)];
    s.p95 = sorted[(int)(0.95f * n)];
    
    return s;
}

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER: Compute RMSE
 *═══════════════════════════════════════════════════════════════════════════*/

static float compute_rmse(const float* est, const float* truth, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = est[i] - truth[i];
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / n);
}

static float compute_bias(const float* est, const float* truth, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += est[i] - truth[i];
    }
    return sum / n;
}

static float compute_correlation(const float* a, const float* b, int n) {
    float mean_a = 0.0f, mean_b = 0.0f;
    for (int i = 0; i < n; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= n;
    mean_b /= n;
    
    float cov = 0.0f, var_a = 0.0f, var_b = 0.0f;
    for (int i = 0; i < n; i++) {
        float da = a[i] - mean_a;
        float db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    
    if (var_a < 1e-10f || var_b < 1e-10f) return 0.0f;
    return cov / sqrtf(var_a * var_b);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MONO SVPF CONFIGURATION (Baseline - Full Adaptive Suite)
 *═══════════════════════════════════════════════════════════════════════════*/

static void configure_mono_adaptive(void* state_ptr) {
    SVPFState* f = (SVPFState*)state_ptr;
    
    // Full adaptive suite (your production config)
    f->use_svld = 1;
    f->use_annealing = 1;
    f->n_anneal_steps = 3;
    f->temperature = 0.45f;
    f->rmsprop_rho = 0.9f;
    f->rmsprop_eps = 1e-6f;
    f->use_adaptive_beta = 1;  // ON by default, set 0 for A/B test
    
    f->use_mim = 1;
    f->mim_jump_prob = 0.25f;
    f->mim_jump_scale = 9.0f;

    f->use_newton = 1;
    f->use_full_newton = 1;

    f->use_guided = 1;
    f->guided_alpha_base = 0.0f;
    f->guided_alpha_shock = 0.40f;
    f->guided_innovation_threshold = 1.5f;
    
    f->use_guide = 1;
    f->use_guide_preserving = 1;
    f->guide_strength = 0.05f;
    f->guide_mean = 0.0f;
    f->guide_var = 0.0f;
    f->guide_K = 0.0f;
    f->guide_initialized = 0;
  
    f->use_adaptive_mu = 1;
    f->mu_state = -3.5f;
    f->mu_var = 1.0f;
    f->mu_process_var = 0.001f;
    f->mu_obs_var_scale = 11.0f;
    f->mu_min = -4.0f;
    f->mu_max = -1.0f;

    f->use_adaptive_guide = 1;
    f->guide_strength_base = 0.05f;
    f->guide_strength_max = 0.30f;
    f->guide_innovation_threshold = 1.0f;

    f->use_adaptive_sigma = 1;
    f->sigma_boost_threshold = 0.95f;
    f->sigma_boost_max = 3.2f;
    f->sigma_z_effective = 0.10f;

    f->use_exact_gradient = 1;
    f->lik_offset = 0.35f;

    f->stein_min_steps = 8;
    f->stein_max_steps = 16;
    f->ksd_improvement_threshold = 0.05f;
    
    f->use_asymmetric_rho = 1;
    f->rho_up = 0.98f;
    f->rho_down = 0.93f;
    
    f->use_local_params = 0;
    f->delta_rho = 0.0f;
    f->delta_sigma = 0.0f;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN MONO SVPF
 *═══════════════════════════════════════════════════════════════════════════*/

static MSTestMetrics run_mono_svpf(
    const float* y,
    const float* h_true,
    int n_steps,
    const SVPFParams* params,
    uint64_t seed
) {
    MSTestMetrics m = {};
    
    int eval_steps = n_steps - WARMUP_STEPS;
    float* h_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_true = (float*)malloc(n_steps * sizeof(float));
    
    for (int t = 0; t < n_steps; t++) {
        vol_true[t] = expf(h_true[t] * 0.5f);
    }
    
    // Create and configure filter
    SVPFState* state = svpf_create(MONO_PARTICLES, MONO_STEIN_STEPS, 5.0f, NULL);
    svpf_initialize(state, params, (unsigned int)seed);
    configure_mono_adaptive(state);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    float y_prev = 0.0f;
    for (int t = 0; t < n_steps; t++) {
        float loglik, vol, h_mean;
        svpf_step_adaptive(state, y[t], y_prev, params, &loglik, &vol, &h_mean);
        
        h_est[t] = h_mean;
        vol_est[t] = vol;
        y_prev = y[t];
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    m.runtime_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Compute metrics on post-warmup data
    float* h_true_eval = (float*)h_true + WARMUP_STEPS;
    float* h_est_eval = h_est + WARMUP_STEPS;
    float* vol_true_eval = vol_true + WARMUP_STEPS;
    float* vol_est_eval = vol_est + WARMUP_STEPS;
    
    m.rmse_h = compute_rmse(h_est_eval, h_true_eval, eval_steps);
    m.rmse_vol = compute_rmse(vol_est_eval, vol_true_eval, eval_steps);
    m.bias_h = compute_bias(h_est_eval, h_true_eval, eval_steps);
    m.correlation = compute_correlation(h_est_eval, h_true_eval, eval_steps);
    
    svpf_destroy(state);
    free(h_est);
    free(vol_est);
    free(vol_true);
    
    return m;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN MULTI-SCALE SVPF
 *═══════════════════════════════════════════════════════════════════════════*/

static MSTestMetrics run_multiscale_svpf(
    const float* y,
    const float* h_true,
    int n_steps,
    const SVPFParams* params,
    uint64_t seed
) {
    MSTestMetrics m = {};
    
    int eval_steps = n_steps - WARMUP_STEPS;
    float* h_est = (float*)malloc(n_steps * sizeof(float));
    float* h_reactive = (float*)malloc(n_steps * sizeof(float));
    float* h_inertial = (float*)malloc(n_steps * sizeof(float));
    float* vol_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_true = (float*)malloc(n_steps * sizeof(float));
    
    for (int t = 0; t < n_steps; t++) {
        vol_true[t] = expf(h_true[t] * 0.5f);
    }
    
    // Create multi-scale filter
    MS_SVPF* ms = ms_svpf_create(NULL);
    if (!ms) {
        fprintf(stderr, "Failed to create multi-scale SVPF\n");
        free(h_est);
        free(h_reactive);
        free(h_inertial);
        free(vol_est);
        free(vol_true);
        return m;
    }
    
    // Initialize
    float initial_vol = expf(params->mu * 0.5f);
    ms_svpf_init(ms, initial_vol, params, (unsigned int)seed);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    MS_SVPF_Output out;
    int transient_correct = 0;
    int transient_total = 0;
    int regime_correct = 0;
    int regime_total = 0;
    
    for (int t = 0; t < n_steps; t++) {
        ms_svpf_step(ms, y[t], params, &out);
        
        h_est[t] = out.h_combined;
        h_reactive[t] = out.h_reactive;
        h_inertial[t] = out.h_inertial;
        vol_est[t] = out.vol_combined;
        
        // Track transient/regime detection accuracy (post-warmup)
        if (t >= WARMUP_STEPS) {
            // Simple heuristic: if h_true changed rapidly, it's transient
            if (t > 0) {
                float h_change = fabsf(h_true[t] - h_true[t-1]);
                bool true_transient = (h_change > 0.5f);  // Large single-step change
                
                if (true_transient) {
                    transient_total++;
                    if (out.is_transient || out.cross_scale_boost > 0) {
                        transient_correct++;
                    }
                }
            }
            
            // If h_true is elevated for multiple steps, it's a regime
            if (t > 10) {
                float h_avg = 0.0f;
                for (int j = t - 10; j < t; j++) h_avg += h_true[j];
                h_avg /= 10;
                bool true_regime = (h_avg > params->mu + 1.0f);  // Persistently elevated
                
                if (true_regime) {
                    regime_total++;
                    if (out.is_regime_shift || out.vol_inertial > expf((params->mu + 0.5f) * 0.5f)) {
                        regime_correct++;
                    }
                }
            }
        }
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    m.runtime_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Compute metrics on post-warmup data
    float* h_true_eval = (float*)h_true + WARMUP_STEPS;
    float* h_est_eval = h_est + WARMUP_STEPS;
    float* h_reactive_eval = h_reactive + WARMUP_STEPS;
    float* h_inertial_eval = h_inertial + WARMUP_STEPS;
    float* vol_true_eval = vol_true + WARMUP_STEPS;
    float* vol_est_eval = vol_est + WARMUP_STEPS;
    
    m.rmse_h = compute_rmse(h_est_eval, h_true_eval, eval_steps);
    m.rmse_vol = compute_rmse(vol_est_eval, vol_true_eval, eval_steps);
    m.bias_h = compute_bias(h_est_eval, h_true_eval, eval_steps);
    m.correlation = compute_correlation(h_est_eval, h_true_eval, eval_steps);
    
    m.rmse_reactive = compute_rmse(h_reactive_eval, h_true_eval, eval_steps);
    m.rmse_inertial = compute_rmse(h_inertial_eval, h_true_eval, eval_steps);
    
    m.transient_detection_rate = (transient_total > 0) ? 
        (float)transient_correct / transient_total : 0.0f;
    m.regime_detection_rate = (regime_total > 0) ?
        (float)regime_correct / regime_total : 0.0f;
    
    ms_svpf_destroy(ms);
    free(h_est);
    free(h_reactive);
    free(h_inertial);
    free(vol_est);
    free(vol_true);
    
    return m;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN COMPARISON FOR ONE SCENARIO
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    const char* scenario_name;
    
    // Mono stats
    MSMetricStats mono_rmse_h;
    MSMetricStats mono_rmse_vol;
    MSMetricStats mono_bias;
    MSMetricStats mono_corr;
    MSMetricStats mono_runtime;
    
    // Multi-scale stats
    MSMetricStats ms_rmse_h;
    MSMetricStats ms_rmse_vol;
    MSMetricStats ms_bias;
    MSMetricStats ms_corr;
    MSMetricStats ms_runtime;
    MSMetricStats ms_rmse_reactive;
    MSMetricStats ms_rmse_inertial;
    MSMetricStats ms_transient_rate;
    MSMetricStats ms_regime_rate;
    
    // Comparison
    float rmse_diff_mean;    // MS - Mono (negative = MS better)
    float rmse_diff_pvalue;
    bool ms_significantly_better;
    
} ScenarioComparison;

static void paired_ttest(const std::vector<float>& a, const std::vector<float>& b, 
                         float* diff_mean, float* pvalue) {
    int n = (int)a.size();
    if (n < 2) {
        *diff_mean = 0;
        *pvalue = 1.0f;
        return;
    }
    
    std::vector<float> diffs(n);
    float mean_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        diffs[i] = b[i] - a[i];
        mean_diff += diffs[i];
    }
    mean_diff /= n;
    
    float var_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = diffs[i] - mean_diff;
        var_diff += d * d;
    }
    var_diff /= (n - 1);
    
    float se = sqrtf(var_diff / n);
    float t_stat = (se > 1e-10f) ? (mean_diff / se) : 0.0f;
    
    // Approximate p-value using normal CDF
    float p = 2.0f * (1.0f - 0.5f * (1.0f + erff(fabsf(t_stat) / sqrtf(2.0f))));
    
    *diff_mean = mean_diff;
    *pvalue = p;
}

static ScenarioComparison run_scenario_comparison(
    const SVPFTestScenario* scenario,
    const SVPFParams* base_params
) {
    ScenarioComparison result = {};
    result.scenario_name = scenario->name;
    
    const int n_seeds = get_n_seeds();
    const int n_steps = get_n_steps();
    
    /* Match filter params to DGP for fair comparison */
    SVPFParams params;
    params.rho = scenario->true_rho;
    params.sigma_z = scenario->true_sigma_z;
    params.mu = scenario->true_mu;
    params.gamma = 0.0f;
    
    std::vector<float> mono_rmse, mono_vol_rmse, mono_bias, mono_corr, mono_runtime;
    std::vector<float> ms_rmse, ms_vol_rmse, ms_bias, ms_corr, ms_runtime;
    std::vector<float> ms_reactive_rmse, ms_inertial_rmse;
    std::vector<float> ms_transient, ms_regime;
    
    // Allocate data arrays
    float* h_true = (float*)malloc(n_steps * sizeof(float));
    float* y = (float*)malloc(n_steps * sizeof(float));
    
    printf("  Running %d seeds for scenario: %s (rho=%.3f, sigma_z=%.2f, mu=%.1f)\n", 
           n_seeds, scenario->name, params.rho, params.sigma_z, params.mu);
    
    for (int s = 0; s < n_seeds; s++) {
        uint64_t seed = 12345 + s * 1000;
        
        // Generate data
        svpf_test_generate_data(scenario, n_steps, seed, h_true, y);
        
        // Run mono
        MSTestMetrics m_mono = run_mono_svpf(y, h_true, n_steps, &params, seed);
        
        // Run multi-scale
        MSTestMetrics m_ms = run_multiscale_svpf(y, h_true, n_steps, &params, seed);
        
        // Skip if either failed
        if (!isfinite(m_mono.rmse_h) || !isfinite(m_ms.rmse_h)) continue;
        
        mono_rmse.push_back(m_mono.rmse_h);
        mono_vol_rmse.push_back(m_mono.rmse_vol);
        mono_bias.push_back(m_mono.bias_h);
        mono_corr.push_back(m_mono.correlation);
        mono_runtime.push_back(m_mono.runtime_ms);
        
        ms_rmse.push_back(m_ms.rmse_h);
        ms_vol_rmse.push_back(m_ms.rmse_vol);
        ms_bias.push_back(m_ms.bias_h);
        ms_corr.push_back(m_ms.correlation);
        ms_runtime.push_back(m_ms.runtime_ms);
        ms_reactive_rmse.push_back(m_ms.rmse_reactive);
        ms_inertial_rmse.push_back(m_ms.rmse_inertial);
        ms_transient.push_back(m_ms.transient_detection_rate);
        ms_regime.push_back(m_ms.regime_detection_rate);
        
        if ((s + 1) % 5 == 0 || s == n_seeds - 1) {
            printf("    Seed %d/%d: Mono RMSE=%.4f, MS RMSE=%.4f\n",
                   s + 1, n_seeds, m_mono.rmse_h, m_ms.rmse_h);
        }
    }
    
    free(h_true);
    free(y);
    
    // Compute statistics
    result.mono_rmse_h = compute_stats(mono_rmse);
    result.mono_rmse_vol = compute_stats(mono_vol_rmse);
    result.mono_bias = compute_stats(mono_bias);
    result.mono_corr = compute_stats(mono_corr);
    result.mono_runtime = compute_stats(mono_runtime);
    
    result.ms_rmse_h = compute_stats(ms_rmse);
    result.ms_rmse_vol = compute_stats(ms_vol_rmse);
    result.ms_bias = compute_stats(ms_bias);
    result.ms_corr = compute_stats(ms_corr);
    result.ms_runtime = compute_stats(ms_runtime);
    result.ms_rmse_reactive = compute_stats(ms_reactive_rmse);
    result.ms_rmse_inertial = compute_stats(ms_inertial_rmse);
    result.ms_transient_rate = compute_stats(ms_transient);
    result.ms_regime_rate = compute_stats(ms_regime);
    
    // Paired t-test
    paired_ttest(mono_rmse, ms_rmse, &result.rmse_diff_mean, &result.rmse_diff_pvalue);
    result.ms_significantly_better = (result.rmse_diff_mean < 0) && (result.rmse_diff_pvalue < 0.05f);
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRINT RESULTS
 *═══════════════════════════════════════════════════════════════════════════*/

static void print_scenario_result(const ScenarioComparison* r) {
    const char* winner_rmse = (r->ms_rmse_h.mean < r->mono_rmse_h.mean) ? "MS" : "MONO";
    float pct_diff = 100.0f * (r->ms_rmse_h.mean - r->mono_rmse_h.mean) / r->mono_rmse_h.mean;
    
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Scenario: %-62s  ║\n", r->scenario_name);
    printf("╠═════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                        │     MONO SVPF      │   MULTI-SCALE SVPF        ║\n");
    printf("║  RMSE(log-vol)         │   Mean ± Std       │   Mean ± Std      │Winner ║\n");
    printf("╟────────────────────────┼────────────────────┼───────────────────┼───────╢\n");
    printf("║  Combined              │ %6.4f ± %6.4f   │ %6.4f ± %6.4f  │ %-4s  ║\n",
           r->mono_rmse_h.mean, r->mono_rmse_h.std,
           r->ms_rmse_h.mean, r->ms_rmse_h.std, winner_rmse);
    printf("║  REACTIVE only         │        -           │ %6.4f ± %6.4f  │       ║\n",
           r->ms_rmse_reactive.mean, r->ms_rmse_reactive.std);
    printf("║  INERTIAL only         │        -           │ %6.4f ± %6.4f  │       ║\n",
           r->ms_rmse_inertial.mean, r->ms_rmse_inertial.std);
    printf("╟────────────────────────┼────────────────────┼───────────────────┼───────╢\n");
    printf("║  Bias(h)               │ %+6.4f ± %6.4f  │ %+6.4f ± %6.4f │       ║\n",
           r->mono_bias.mean, r->mono_bias.std,
           r->ms_bias.mean, r->ms_bias.std);
    printf("║  Correlation           │ %6.4f ± %6.4f   │ %6.4f ± %6.4f  │       ║\n",
           r->mono_corr.mean, r->mono_corr.std,
           r->ms_corr.mean, r->ms_corr.std);
    printf("║  Runtime (ms)          │ %6.1f ± %6.1f    │ %6.1f ± %6.1f   │       ║\n",
           r->mono_runtime.mean, r->mono_runtime.std,
           r->ms_runtime.mean, r->ms_runtime.std);
    printf("╠═════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  RMSE Difference: %+.4f (%+.2f%%)                                       ║\n",
           r->rmse_diff_mean, pct_diff);
    printf("║  P-value: %.4f  →  %s      ║\n",
           r->rmse_diff_pvalue,
           r->ms_significantly_better ? "MS SIGNIFICANTLY BETTER" :
           (r->rmse_diff_pvalue < 0.05f ? "MONO SIGNIFICANTLY BETTER" : 
                                          "NO SIGNIFICANT DIFFERENCE"));
    printf("╚═════════════════════════════════════════════════════════════════════════╝\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--full") == 0) {
            g_full_mode = true;
        }
    }
    
    const int n_seeds = get_n_seeds();
    const int n_steps = get_n_steps();
    
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           SVPF A/B TEST: MULTI-SCALE vs MONO                            ║\n");
    printf("╠═════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Mode: %-6s (use --full for comprehensive test)                       ║\n",
           g_full_mode ? "FULL" : "QUICK");
    printf("║  MONO:  1024 particles, full adaptive suite                             ║\n");
    printf("║  MS:    REACTIVE (512p) + INERTIAL (1024p)                              ║\n");
    printf("║  Seeds: %d, Steps: %d, Warmup: %d                                       ║\n",
           n_seeds, n_steps, WARMUP_STEPS);
    printf("╚═════════════════════════════════════════════════════════════════════════╝\n\n");
    
    // Base params (filter assumes these, DGP may differ)
    SVPFParams params;
    params.rho = 0.98f;
    params.sigma_z = 0.10f;
    params.mu = -4.5f;
    params.gamma = 0.0f;
    
    // Get all scenarios
    SVPFTestScenario scenarios[10];
    int n_scenarios = svpf_test_get_all_scenarios(scenarios, 10);
    
    // Store results
    std::vector<ScenarioComparison> results;
    
    // Run all scenarios
    for (int i = 0; i < n_scenarios; i++) {
        ScenarioComparison r = run_scenario_comparison(&scenarios[i], &params);
        results.push_back(r);
        print_scenario_result(&r);
    }
    
    // Summary table
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    SUMMARY: RMSE(log-vol)                               ║\n");
    printf("╠═════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Scenario         │   MONO    │    MS     │   Diff   │ %% Change│ Winner ║\n");
    printf("╟───────────────────┼───────────┼───────────┼──────────┼─────────┼────────╢\n");
    
    int mono_wins = 0, ms_wins = 0, ties = 0;
    
    for (const auto& r : results) {
        float pct = 100.0f * (r.ms_rmse_h.mean - r.mono_rmse_h.mean) / r.mono_rmse_h.mean;
        const char* winner;
        if (r.rmse_diff_pvalue < 0.05f) {
            if (r.rmse_diff_mean < 0) {
                winner = "MS *";
                ms_wins++;
            } else {
                winner = "MONO *";
                mono_wins++;
            }
        } else {
            winner = "TIE";
            ties++;
        }
        
        printf("║  %-16s │  %7.4f  │  %7.4f  │ %+7.4f │ %+6.2f%% │ %-6s ║\n",
               r.scenario_name,
               r.mono_rmse_h.mean,
               r.ms_rmse_h.mean,
               r.rmse_diff_mean,
               pct,
               winner);
    }
    
    printf("╠═════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  OVERALL: MONO wins %d, MS wins %d, Ties %d                              ║\n",
           mono_wins, ms_wins, ties);
    printf("║  (* = statistically significant at p<0.05)                              ║\n");
    printf("║  Negative diff / %% change = MS is better                                ║\n");
    printf("╚═════════════════════════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}
