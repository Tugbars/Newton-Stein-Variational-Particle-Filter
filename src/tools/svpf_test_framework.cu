/**
 * @file svpf_test_framework.cu
 * @brief Implementation of SVPF testing framework
 */

#include "svpf_test_framework.h"
#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>

// =============================================================================
// Pre-defined Scenarios
// =============================================================================

SVPFTestScenario svpf_test_scenario_calm(void) {
    SVPFTestScenario s = {};
    s.name = "calm";
    s.description = "Calm market - low volatility, standard parameters";
    s.true_mu = -5.0f;          // ~0.7% daily vol
    s.true_rho = 0.98f;
    s.true_sigma_z = 0.10f;
    s.true_nu = INFINITY;       // Gaussian observations
    s.asymmetric = false;
    s.has_jumps = false;
    s.has_regime_shift = false;
    return s;
}

SVPFTestScenario svpf_test_scenario_crisis(void) {
    SVPFTestScenario s = {};
    s.name = "crisis";
    s.description = "Persistent crisis - high vol, slow mean reversion";
    s.true_mu = -3.5f;          // ~2.5% daily vol (elevated)
    s.true_rho = 0.995f;        // Very persistent
    s.true_sigma_z = 0.20f;     // Higher vol-of-vol
    s.true_nu = INFINITY;
    s.asymmetric = false;
    s.has_jumps = false;
    s.has_regime_shift = false;
    return s;
}

SVPFTestScenario svpf_test_scenario_spike(void) {
    SVPFTestScenario s = {};
    s.name = "spike";
    s.description = "Volatility spike at t=1000, tests adaptation speed";
    s.true_mu = -5.0f;
    s.true_rho = 0.97f;
    s.true_sigma_z = 0.12f;
    s.true_nu = INFINITY;
    s.asymmetric = false;
    s.has_jumps = false;
    s.has_regime_shift = true;
    s.regime_shift_time = 1000;
    s.regime_shift_mu = -2.5f;  // Jump to ~5% vol
    return s;
}

SVPFTestScenario svpf_test_scenario_regime_shift(void) {
    SVPFTestScenario s = {};
    s.name = "regime_shift";
    s.description = "Permanent regime change at t=2500";
    s.true_mu = -5.0f;
    s.true_rho = 0.98f;
    s.true_sigma_z = 0.10f;
    s.true_nu = INFINITY;
    s.asymmetric = false;
    s.has_jumps = false;
    s.has_regime_shift = true;
    s.regime_shift_time = 2500;
    s.regime_shift_mu = -3.0f;  // New normal is higher vol
    return s;
}

SVPFTestScenario svpf_test_scenario_asymmetric(void) {
    SVPFTestScenario s = {};
    s.name = "asymmetric";
    s.description = "Asymmetric dynamics - fast spike, slow recovery (leverage)";
    s.true_mu = -4.5f;
    s.true_rho = 0.98f;         // Not used directly
    s.true_sigma_z = 0.12f;
    s.true_nu = INFINITY;
    s.asymmetric = true;
    s.true_rho_up = 0.99f;      // Vol rises slowly (persistent)
    s.true_rho_down = 0.95f;    // Vol falls quickly (mean revert)
    s.has_jumps = false;
    s.has_regime_shift = false;
    return s;
}

SVPFTestScenario svpf_test_scenario_jumps(void) {
    SVPFTestScenario s = {};
    s.name = "jumps";
    s.description = "Occasional large jumps in volatility";
    s.true_mu = -4.5f;
    s.true_rho = 0.98f;
    s.true_sigma_z = 0.10f;
    s.true_nu = INFINITY;
    s.asymmetric = false;
    s.has_jumps = true;
    s.jump_prob = 0.02f;        // 2% chance per step
    s.jump_scale = 4.0f;        // 4x normal innovation
    s.has_regime_shift = false;
    return s;
}

SVPFTestScenario svpf_test_scenario_fat_tails(void) {
    SVPFTestScenario s = {};
    s.name = "fat_tails";
    s.description = "Student-t observations with df=5";
    s.true_mu = -4.5f;
    s.true_rho = 0.98f;
    s.true_sigma_z = 0.10f;
    s.true_nu = 5.0f;           // Fat tails
    s.asymmetric = false;
    s.has_jumps = false;
    s.has_regime_shift = false;
    return s;
}

int svpf_test_get_all_scenarios(SVPFTestScenario* scenarios, int max_scenarios) {
    int count = 0;
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_calm();
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_crisis();
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_spike();
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_regime_shift();
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_asymmetric();
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_jumps();
    if (count < max_scenarios) scenarios[count++] = svpf_test_scenario_fat_tails();
    return count;
}

// =============================================================================
// Random Number Generation (Host-side for data generation)
// =============================================================================

static float randn(uint64_t* state) {
    // Xorshift64 + Box-Muller
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    
    float u1 = (float)(x & 0xFFFFFFFF) / 4294967296.0f + 1e-10f;
    
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    
    float u2 = (float)(x & 0xFFFFFFFF) / 4294967296.0f;
    
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static float rand_uniform(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (float)(x & 0xFFFFFFFF) / 4294967296.0f;
}

static float rand_student_t(uint64_t* state, float nu) {
    if (nu > 1e6f) return randn(state);  // Approximate as Gaussian
    
    // Ratio of uniforms method for Student-t
    float z = randn(state);
    float chi2 = 0.0f;
    for (int i = 0; i < (int)nu; i++) {
        float g = randn(state);
        chi2 += g * g;
    }
    return z * sqrtf(nu / chi2);
}

// =============================================================================
// Data Generation
// =============================================================================

void svpf_test_generate_data(
    const SVPFTestScenario* scenario,
    int n_steps,
    uint64_t seed,
    float* h_true_out,
    float* y_out
) {
    uint64_t rng_state = seed;
    
    float mu = scenario->true_mu;
    float rho = scenario->true_rho;
    float sigma_z = scenario->true_sigma_z;
    float nu = scenario->true_nu;
    
    // Initialize h at equilibrium with some noise
    float h = mu + sigma_z * randn(&rng_state);
    float h_prev = h;
    
    for (int t = 0; t < n_steps; t++) {
        // Check for regime shift
        if (scenario->has_regime_shift && t == scenario->regime_shift_time) {
            mu = scenario->regime_shift_mu;
        }
        
        // Determine effective rho
        float rho_eff = rho;
        if (scenario->asymmetric) {
            rho_eff = (h > h_prev) ? scenario->true_rho_up : scenario->true_rho_down;
        }
        
        // Generate innovation
        float epsilon = randn(&rng_state);
        
        // Apply jump if applicable
        if (scenario->has_jumps) {
            if (rand_uniform(&rng_state) < scenario->jump_prob) {
                epsilon *= scenario->jump_scale;
            }
        }
        
        // Update h
        h_prev = h;
        h = mu + rho_eff * (h_prev - mu) + sigma_z * epsilon;
        
        // Clamp to reasonable range
        h = fmaxf(fminf(h, 2.0f), -12.0f);
        
        h_true_out[t] = h;
        
        // Generate observation
        float vol = expf(h * 0.5f);
        float z;
        if (nu > 1e6f || !isfinite(nu)) {
            z = randn(&rng_state);
        } else {
            z = rand_student_t(&rng_state, nu);
        }
        y_out[t] = vol * z;
    }
}

// =============================================================================
// Metric Computation
// =============================================================================

static float compute_rmse(const float* a, const float* b, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / n);
}

static float compute_mae(const float* a, const float* b, int n) {
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_abs += fabsf(a[i] - b[i]);
    }
    return sum_abs / n;
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

float svpf_test_compute_coverage(
    const float* h_true,
    const float* h_particles,  // Stored as [step][particle], i.e., row-major
    int n_steps,
    int n_particles,
    float level
) {
    // For each step, compute empirical quantiles and check if h_true is inside
    float alpha = (1.0f - level) / 2.0f;  // Two-tailed
    int lower_idx = (int)(alpha * n_particles);
    int upper_idx = (int)((1.0f - alpha) * n_particles);
    
    if (lower_idx < 0) lower_idx = 0;
    if (upper_idx >= n_particles) upper_idx = n_particles - 1;
    
    std::vector<float> sorted(n_particles);
    int in_ci = 0;
    
    for (int t = 0; t < n_steps; t++) {
        // Copy particles for this step
        for (int i = 0; i < n_particles; i++) {
            sorted[i] = h_particles[t * n_particles + i];
        }
        std::sort(sorted.begin(), sorted.end());
        
        float lower = sorted[lower_idx];
        float upper = sorted[upper_idx];
        
        if (h_true[t] >= lower && h_true[t] <= upper) {
            in_ci++;
        }
    }
    
    return (float)in_ci / n_steps;
}

void svpf_test_cross_correlation(
    const float* h_true,
    const float* h_est,
    int n,
    int max_lag,
    float* xcorr_out
) {
    // Compute mean
    float mean_true = 0.0f, mean_est = 0.0f;
    for (int i = 0; i < n; i++) {
        mean_true += h_true[i];
        mean_est += h_est[i];
    }
    mean_true /= n;
    mean_est /= n;
    
    // Compute variances
    float var_true = 0.0f, var_est = 0.0f;
    for (int i = 0; i < n; i++) {
        var_true += (h_true[i] - mean_true) * (h_true[i] - mean_true);
        var_est += (h_est[i] - mean_est) * (h_est[i] - mean_est);
    }
    float norm = sqrtf(var_true * var_est);
    if (norm < 1e-10f) norm = 1e-10f;
    
    // Compute cross-correlation at each lag
    for (int lag = -max_lag; lag <= max_lag; lag++) {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < n; i++) {
            int j = i + lag;
            if (j >= 0 && j < n) {
                sum += (h_true[i] - mean_true) * (h_est[j] - mean_est);
                count++;
            }
        }
        xcorr_out[lag + max_lag] = sum / norm;
    }
}

// =============================================================================
// Statistical Tests
// =============================================================================

// Approximation of Student-t CDF for p-value computation
static float t_cdf(float t, int df) {
    // Use approximation for large df
    if (df > 100) {
        // Normal approximation
        float x = t;
        return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    }
    
    // Simple approximation using incomplete beta function relation
    // For df > 2, use normal as approximation (good enough for our purposes)
    float x = t;
    return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

void svpf_test_paired_ttest(
    const float* diffs,
    int n,
    float* t_stat,
    float* p_value
) {
    // Compute mean difference
    float mean_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        mean_diff += diffs[i];
    }
    mean_diff /= n;
    
    // Compute std of differences
    float var_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = diffs[i] - mean_diff;
        var_diff += d * d;
    }
    var_diff /= (n - 1);
    float std_diff = sqrtf(var_diff);
    
    // T-statistic
    float se = std_diff / sqrtf((float)n);
    *t_stat = (se > 1e-10f) ? (mean_diff / se) : 0.0f;
    
    // P-value (two-tailed)
    float cdf = t_cdf(fabsf(*t_stat), n - 1);
    *p_value = 2.0f * (1.0f - cdf);
}

// =============================================================================
// Aggregate Statistics
// =============================================================================

static MetricStats compute_stats(const std::vector<float>& values) {
    MetricStats s = {};
    int n = (int)values.size();
    if (n == 0) return s;
    
    // Mean
    for (float v : values) s.mean += v;
    s.mean /= n;
    
    // Std
    for (float v : values) s.std += (v - s.mean) * (v - s.mean);
    s.std = sqrtf(s.std / (n - 1));
    
    // Sort for percentiles
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    
    s.min = sorted[0];
    s.max = sorted[n - 1];
    s.median = sorted[n / 2];
    s.p5 = sorted[(int)(0.05f * n)];
    s.p95 = sorted[(int)(0.95f * n)];
    
    return s;
}

// =============================================================================
// Default Configuration
// =============================================================================

SVPFTestConfig svpf_test_default_config(void) {
    SVPFTestConfig c = {};
    c.n_seeds = 100;
    c.n_steps = 5000;
    c.warmup_steps = 100;
    c.n_particles = 512;
    c.n_stein_steps = 8;
    c.coverage_levels[0] = 0.50f;
    c.coverage_levels[1] = 0.90f;
    c.coverage_levels[2] = 0.95f;
    c.n_coverage_levels = 3;
    c.verbose = false;
    c.save_traces = false;
    c.output_dir = NULL;
    return c;
}

// =============================================================================
// Single Test Run
// =============================================================================

SVPFTestMetrics svpf_test_run_single(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params_ptr,
    uint64_t seed
) {
    const SVPFParams* filter_params = (const SVPFParams*)filter_params_ptr;
    SVPFTestMetrics m = {};
    m.seed = seed;
    
    int n_steps = config->n_steps;
    int n_particles = config->n_particles;
    int warmup = config->warmup_steps;
    int eval_steps = n_steps - warmup;
    
    // Allocate data
    float* h_true = (float*)malloc(n_steps * sizeof(float));
    float* y = (float*)malloc(n_steps * sizeof(float));
    float* h_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_true = (float*)malloc(n_steps * sizeof(float));
    float* ess_trace = (float*)malloc(n_steps * sizeof(float));
    float* h_particles = (float*)malloc(n_steps * n_particles * sizeof(float));
    
    // Generate data
    svpf_test_generate_data(scenario, n_steps, seed, h_true, y);
    
    // Compute true vol
    for (int t = 0; t < n_steps; t++) {
        vol_true[t] = expf(h_true[t] * 0.5f);
    }
    
    // Create filter
    SVPFState* state = svpf_create(n_particles, config->n_stein_steps, 50.0f, NULL);
    svpf_initialize(state, filter_params, (unsigned int)seed);
    
    // Timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run filter
    float y_prev = 0.0f;
    float total_nll = 0.0f;
    float total_ess = 0.0f;
    float min_ess = 1.0f;
    int ess_below_30 = 0;
    float total_bandwidth = 0.0f;
    
    for (int t = 0; t < n_steps; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, y[t], y_prev, filter_params, &loglik, &vol, &h_mean);
        
        h_est[t] = h_mean;
        vol_est[t] = vol;
        
        // Copy particles for coverage computation
        cudaMemcpy(&h_particles[t * n_particles], state->h, 
                   n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Track NLL (skip warmup)
        if (t >= warmup) {
            total_nll -= loglik;  // Negative log-likelihood
        }
        
        // Track ESS (approximate from weights)
        // Note: This is a simplification - ideally we'd compute ESS properly
        float ess_ratio = 0.7f;  // Placeholder - would need to expose from filter
        ess_trace[t] = ess_ratio;
        if (t >= warmup) {
            total_ess += ess_ratio;
            if (ess_ratio < min_ess) min_ess = ess_ratio;
            if (ess_ratio < 0.3f) ess_below_30++;
        }
        
        y_prev = y[t];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    m.runtime_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Compute metrics (on post-warmup data)
    float* h_true_eval = h_true + warmup;
    float* h_est_eval = h_est + warmup;
    float* vol_true_eval = vol_true + warmup;
    float* vol_est_eval = vol_est + warmup;
    float* h_particles_eval = h_particles + warmup * n_particles;
    
    m.rmse_h = compute_rmse(h_est_eval, h_true_eval, eval_steps);
    m.rmse_vol = compute_rmse(vol_est_eval, vol_true_eval, eval_steps);
    m.mae_h = compute_mae(h_est_eval, h_true_eval, eval_steps);
    m.bias_h = compute_bias(h_est_eval, h_true_eval, eval_steps);
    
    m.nll = total_nll / eval_steps;
    
    m.coverage_50 = svpf_test_compute_coverage(h_true_eval, h_particles_eval, 
                                                eval_steps, n_particles, 0.50f);
    m.coverage_90 = svpf_test_compute_coverage(h_true_eval, h_particles_eval,
                                                eval_steps, n_particles, 0.90f);
    m.coverage_95 = svpf_test_compute_coverage(h_true_eval, h_particles_eval,
                                                eval_steps, n_particles, 0.95f);
    
    m.ess_mean = total_ess / eval_steps;
    m.ess_min = min_ess;
    m.ess_below_30_pct = (float)ess_below_30 / eval_steps;
    
    // Cross-correlation for lag detection
    m.lag_correlation = compute_correlation(h_est_eval, h_true_eval, eval_steps);
    
    // Compute lag-1 correlation (shift h_est by 1)
    if (eval_steps > 1) {
        m.lag_1_correlation = compute_correlation(h_est_eval + 1, h_true_eval, eval_steps - 1);
    }
    
    // Cleanup
    svpf_destroy(state);
    free(h_true);
    free(y);
    free(h_est);
    free(vol_est);
    free(vol_true);
    free(ess_trace);
    free(h_particles);
    
    return m;
}

// =============================================================================
// Full Scenario Run
// =============================================================================

SVPFTestAggregateResult svpf_test_run_scenario(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params
) {
    SVPFTestAggregateResult result = {};
    result.scenario_name = scenario->name;
    result.n_seeds = config->n_seeds;
    result.n_steps = config->n_steps;
    
    std::vector<float> rmse_h_vals, rmse_vol_vals, bias_h_vals;
    std::vector<float> nll_vals, cov50_vals, cov90_vals, cov95_vals;
    std::vector<float> ess_mean_vals, ess_min_vals, bandwidth_vals, corr_vals;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int s = 0; s < config->n_seeds; s++) {
        uint64_t seed = 12345 + s * 1000;
        
        SVPFTestMetrics m = svpf_test_run_single(scenario, config, filter_params, seed);
        
        // Check for failures
        if (m.nan_count > 0 || m.inf_count > 0 || !isfinite(m.rmse_h)) {
            result.failed_runs++;
            continue;
        }
        
        rmse_h_vals.push_back(m.rmse_h);
        rmse_vol_vals.push_back(m.rmse_vol);
        bias_h_vals.push_back(m.bias_h);
        nll_vals.push_back(m.nll);
        cov50_vals.push_back(m.coverage_50);
        cov90_vals.push_back(m.coverage_90);
        cov95_vals.push_back(m.coverage_95);
        ess_mean_vals.push_back(m.ess_mean);
        ess_min_vals.push_back(m.ess_min);
        corr_vals.push_back(m.lag_correlation);
        
        result.total_nans += m.nan_count;
        result.total_infs += m.inf_count;
        
        if (config->verbose) {
            printf("  Seed %d: RMSE=%.4f, NLL=%.4f, Cov95=%.2f%%\n",
                   s, m.rmse_h, m.nll, m.coverage_95 * 100.0f);
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_runtime_s = std::chrono::duration<float>(total_end - total_start).count();
    result.mean_runtime_ms = (result.total_runtime_s * 1000.0f) / config->n_seeds;
    
    // Aggregate
    result.rmse_h = compute_stats(rmse_h_vals);
    result.rmse_vol = compute_stats(rmse_vol_vals);
    result.bias_h = compute_stats(bias_h_vals);
    result.nll = compute_stats(nll_vals);
    result.coverage_50 = compute_stats(cov50_vals);
    result.coverage_90 = compute_stats(cov90_vals);
    result.coverage_95 = compute_stats(cov95_vals);
    result.ess_mean = compute_stats(ess_mean_vals);
    result.ess_min = compute_stats(ess_min_vals);
    result.lag_correlation = compute_stats(corr_vals);
    
    return result;
}

// =============================================================================
// A/B Comparison
// =============================================================================

SVPFTestComparison svpf_test_compare(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params_a,
    const void* filter_params_b,
    SVPFTestAggregateResult* result_a_out,
    SVPFTestAggregateResult* result_b_out
) {
    SVPFTestComparison comp = {};
    
    std::vector<float> rmse_a, rmse_b, nll_a, nll_b, cov95_a, cov95_b, ess_a, ess_b;
    
    for (int s = 0; s < config->n_seeds; s++) {
        uint64_t seed = 12345 + s * 1000;
        
        SVPFTestMetrics m_a = svpf_test_run_single(scenario, config, filter_params_a, seed);
        SVPFTestMetrics m_b = svpf_test_run_single(scenario, config, filter_params_b, seed);
        
        // Skip if either failed
        if (!isfinite(m_a.rmse_h) || !isfinite(m_b.rmse_h)) continue;
        
        rmse_a.push_back(m_a.rmse_h);
        rmse_b.push_back(m_b.rmse_h);
        nll_a.push_back(m_a.nll);
        nll_b.push_back(m_b.nll);
        cov95_a.push_back(m_a.coverage_95);
        cov95_b.push_back(m_b.coverage_95);
        ess_a.push_back(m_a.ess_mean);
        ess_b.push_back(m_b.ess_mean);
    }
    
    int n = (int)rmse_a.size();
    if (n < 2) return comp;
    
    // Compute paired differences
    std::vector<float> rmse_diff(n), nll_diff(n), cov95_diff(n), ess_diff(n);
    for (int i = 0; i < n; i++) {
        rmse_diff[i] = rmse_b[i] - rmse_a[i];
        nll_diff[i] = nll_b[i] - nll_a[i];
        cov95_diff[i] = cov95_b[i] - cov95_a[i];
        ess_diff[i] = ess_b[i] - ess_a[i];
    }
    
    // Mean differences
    comp.rmse_h_diff = 0; for (float d : rmse_diff) comp.rmse_h_diff += d; comp.rmse_h_diff /= n;
    comp.nll_diff = 0; for (float d : nll_diff) comp.nll_diff += d; comp.nll_diff /= n;
    comp.coverage_95_diff = 0; for (float d : cov95_diff) comp.coverage_95_diff += d; comp.coverage_95_diff /= n;
    comp.ess_mean_diff = 0; for (float d : ess_diff) comp.ess_mean_diff += d; comp.ess_mean_diff /= n;
    
    // T-tests
    svpf_test_paired_ttest(rmse_diff.data(), n, &comp.rmse_h_tstat, &comp.rmse_h_pvalue);
    svpf_test_paired_ttest(nll_diff.data(), n, &comp.nll_tstat, &comp.nll_pvalue);
    svpf_test_paired_ttest(cov95_diff.data(), n, &comp.coverage_95_tstat, &comp.coverage_95_pvalue);
    svpf_test_paired_ttest(ess_diff.data(), n, &comp.ess_mean_tstat, &comp.ess_mean_pvalue);
    
    // Standard errors
    float var_rmse = 0, var_nll = 0;
    for (int i = 0; i < n; i++) {
        var_rmse += (rmse_diff[i] - comp.rmse_h_diff) * (rmse_diff[i] - comp.rmse_h_diff);
        var_nll += (nll_diff[i] - comp.nll_diff) * (nll_diff[i] - comp.nll_diff);
    }
    comp.rmse_h_se = sqrtf(var_rmse / (n * (n - 1)));
    comp.nll_se = sqrtf(var_nll / (n * (n - 1)));
    
    // Cohen's d (effect size)
    float sd_rmse = sqrtf(var_rmse / (n - 1));
    float sd_nll = sqrtf(var_nll / (n - 1));
    comp.rmse_h_cohens_d = (sd_rmse > 1e-10f) ? (comp.rmse_h_diff / sd_rmse) : 0.0f;
    comp.nll_cohens_d = (sd_nll > 1e-10f) ? (comp.nll_diff / sd_nll) : 0.0f;
    
    // Verdicts
    comp.rmse_significant = comp.rmse_h_pvalue < 0.05f;
    comp.nll_significant = comp.nll_pvalue < 0.05f;
    comp.rmse_b_better = comp.rmse_h_diff < 0;
    comp.nll_b_better = comp.nll_diff < 0;
    
    // Optionally compute full aggregates
    if (result_a_out) {
        *result_a_out = svpf_test_run_scenario(scenario, config, filter_params_a);
    }
    if (result_b_out) {
        *result_b_out = svpf_test_run_scenario(scenario, config, filter_params_b);
    }
    
    return comp;
}

// =============================================================================
// Single Run with Config Callback
// =============================================================================

SVPFTestMetrics svpf_test_run_single_with_config(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params_ptr,
    SVPFConfigCallback configure,
    uint64_t seed
) {
    const SVPFParams* filter_params = (const SVPFParams*)filter_params_ptr;
    SVPFTestMetrics m = {};
    m.seed = seed;
    
    int n_steps = config->n_steps;
    int n_particles = config->n_particles;
    int warmup = config->warmup_steps;
    int eval_steps = n_steps - warmup;
    
    // Allocate data
    float* h_true = (float*)malloc(n_steps * sizeof(float));
    float* y = (float*)malloc(n_steps * sizeof(float));
    float* h_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_est = (float*)malloc(n_steps * sizeof(float));
    float* vol_true = (float*)malloc(n_steps * sizeof(float));
    float* ess_trace = (float*)malloc(n_steps * sizeof(float));
    float* h_particles = (float*)malloc(n_steps * n_particles * sizeof(float));
    
    // Generate data
    svpf_test_generate_data(scenario, n_steps, seed, h_true, y);
    
    // Compute true vol
    for (int t = 0; t < n_steps; t++) {
        vol_true[t] = expf(h_true[t] * 0.5f);
    }
    
    // Create and configure filter
    SVPFState* state = svpf_create(n_particles, config->n_stein_steps, 7.0f, NULL);
    svpf_initialize(state, filter_params, (unsigned int)seed);
    
    // Apply configuration callback (sets all the flags)
    if (configure) {
        configure(state);
    }
    
    // Timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run filter
    float y_prev = 0.0f;
    float total_nll = 0.0f;
    float total_ess = 0.0f;
    float min_ess = 1.0f;
    int ess_below_30 = 0;
    
    for (int t = 0; t < n_steps; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, y[t], y_prev, filter_params, &loglik, &vol, &h_mean);
        
        h_est[t] = h_mean;
        vol_est[t] = vol;
        
        // Copy particles for coverage computation
        cudaMemcpy(&h_particles[t * n_particles], state->h, 
                   n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Track NLL (skip warmup)
        if (t >= warmup) {
            total_nll -= loglik;
        }
        
        // Track ESS
        float ess_ratio = 0.7f;  // Placeholder
        ess_trace[t] = ess_ratio;
        if (t >= warmup) {
            total_ess += ess_ratio;
            if (ess_ratio < min_ess) min_ess = ess_ratio;
            if (ess_ratio < 0.3f) ess_below_30++;
        }
        
        y_prev = y[t];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    m.runtime_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Compute metrics (on post-warmup data)
    float* h_true_eval = h_true + warmup;
    float* h_est_eval = h_est + warmup;
    float* vol_true_eval = vol_true + warmup;
    float* vol_est_eval = vol_est + warmup;
    float* h_particles_eval = h_particles + warmup * n_particles;
    
    m.rmse_h = compute_rmse(h_est_eval, h_true_eval, eval_steps);
    m.rmse_vol = compute_rmse(vol_est_eval, vol_true_eval, eval_steps);
    m.mae_h = compute_mae(h_est_eval, h_true_eval, eval_steps);
    m.bias_h = compute_bias(h_est_eval, h_true_eval, eval_steps);
    
    m.nll = total_nll / eval_steps;
    
    m.coverage_50 = svpf_test_compute_coverage(h_true_eval, h_particles_eval, 
                                                eval_steps, n_particles, 0.50f);
    m.coverage_90 = svpf_test_compute_coverage(h_true_eval, h_particles_eval,
                                                eval_steps, n_particles, 0.90f);
    m.coverage_95 = svpf_test_compute_coverage(h_true_eval, h_particles_eval,
                                                eval_steps, n_particles, 0.95f);
    
    m.ess_mean = total_ess / eval_steps;
    m.ess_min = min_ess;
    m.ess_below_30_pct = (float)ess_below_30 / eval_steps;
    
    m.lag_correlation = compute_correlation(h_est_eval, h_true_eval, eval_steps);
    
    if (eval_steps > 1) {
        m.lag_1_correlation = compute_correlation(h_est_eval + 1, h_true_eval, eval_steps - 1);
    }
    
    // Cleanup
    svpf_destroy(state);
    free(h_true);
    free(y);
    free(h_est);
    free(vol_est);
    free(vol_true);
    free(ess_trace);
    free(h_particles);
    
    return m;
}

// =============================================================================
// A/B Comparison with Config Callbacks
// =============================================================================

SVPFTestComparison svpf_test_compare_with_config(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params_a,
    SVPFConfigCallback configure_a,
    const void* filter_params_b,
    SVPFConfigCallback configure_b,
    SVPFTestAggregateResult* result_a_out,
    SVPFTestAggregateResult* result_b_out
) {
    SVPFTestComparison comp = {};
    
    std::vector<float> rmse_a, rmse_b, nll_a, nll_b, cov95_a, cov95_b, ess_a, ess_b;
    
    for (int s = 0; s < config->n_seeds; s++) {
        uint64_t seed = 12345 + s * 1000;
        
        SVPFTestMetrics m_a = svpf_test_run_single_with_config(
            scenario, config, filter_params_a, configure_a, seed);
        SVPFTestMetrics m_b = svpf_test_run_single_with_config(
            scenario, config, filter_params_b, configure_b, seed);
        
        // Skip if either failed
        if (!isfinite(m_a.rmse_h) || !isfinite(m_b.rmse_h)) continue;
        
        rmse_a.push_back(m_a.rmse_h);
        rmse_b.push_back(m_b.rmse_h);
        nll_a.push_back(m_a.nll);
        nll_b.push_back(m_b.nll);
        cov95_a.push_back(m_a.coverage_95);
        cov95_b.push_back(m_b.coverage_95);
        ess_a.push_back(m_a.ess_mean);
        ess_b.push_back(m_b.ess_mean);
    }
    
    int n = (int)rmse_a.size();
    if (n < 2) return comp;
    
    // Compute paired differences
    std::vector<float> rmse_diff(n), nll_diff(n), cov95_diff(n), ess_diff(n);
    for (int i = 0; i < n; i++) {
        rmse_diff[i] = rmse_b[i] - rmse_a[i];
        nll_diff[i] = nll_b[i] - nll_a[i];
        cov95_diff[i] = cov95_b[i] - cov95_a[i];
        ess_diff[i] = ess_b[i] - ess_a[i];
    }
    
    // Mean differences
    comp.rmse_h_diff = 0; for (float d : rmse_diff) comp.rmse_h_diff += d; comp.rmse_h_diff /= n;
    comp.nll_diff = 0; for (float d : nll_diff) comp.nll_diff += d; comp.nll_diff /= n;
    comp.coverage_95_diff = 0; for (float d : cov95_diff) comp.coverage_95_diff += d; comp.coverage_95_diff /= n;
    comp.ess_mean_diff = 0; for (float d : ess_diff) comp.ess_mean_diff += d; comp.ess_mean_diff /= n;
    
    // T-tests
    svpf_test_paired_ttest(rmse_diff.data(), n, &comp.rmse_h_tstat, &comp.rmse_h_pvalue);
    svpf_test_paired_ttest(nll_diff.data(), n, &comp.nll_tstat, &comp.nll_pvalue);
    svpf_test_paired_ttest(cov95_diff.data(), n, &comp.coverage_95_tstat, &comp.coverage_95_pvalue);
    svpf_test_paired_ttest(ess_diff.data(), n, &comp.ess_mean_tstat, &comp.ess_mean_pvalue);
    
    // Standard errors
    float var_rmse = 0, var_nll = 0;
    for (int i = 0; i < n; i++) {
        var_rmse += (rmse_diff[i] - comp.rmse_h_diff) * (rmse_diff[i] - comp.rmse_h_diff);
        var_nll += (nll_diff[i] - comp.nll_diff) * (nll_diff[i] - comp.nll_diff);
    }
    comp.rmse_h_se = sqrtf(var_rmse / (n * (n - 1)));
    comp.nll_se = sqrtf(var_nll / (n * (n - 1)));
    
    // Cohen's d
    float sd_rmse = sqrtf(var_rmse / (n - 1));
    float sd_nll = sqrtf(var_nll / (n - 1));
    comp.rmse_h_cohens_d = (sd_rmse > 1e-10f) ? (comp.rmse_h_diff / sd_rmse) : 0.0f;
    comp.nll_cohens_d = (sd_nll > 1e-10f) ? (comp.nll_diff / sd_nll) : 0.0f;
    
    // Verdicts
    comp.rmse_significant = comp.rmse_h_pvalue < 0.05f;
    comp.nll_significant = comp.nll_pvalue < 0.05f;
    comp.rmse_b_better = comp.rmse_h_diff < 0;
    comp.nll_b_better = comp.nll_diff < 0;
    
    // Compute aggregate results if requested
    // Note: For proper aggregates, we'd need to re-run, but for efficiency
    // we construct from the collected data
    if (result_a_out) {
        result_a_out->scenario_name = scenario->name;
        result_a_out->n_seeds = n;
        result_a_out->n_steps = config->n_steps;
        result_a_out->rmse_h = compute_stats(rmse_a);
        result_a_out->nll = compute_stats(nll_a);
        result_a_out->coverage_95 = compute_stats(cov95_a);
        result_a_out->ess_mean = compute_stats(ess_a);
    }
    if (result_b_out) {
        result_b_out->scenario_name = scenario->name;
        result_b_out->n_seeds = n;
        result_b_out->n_steps = config->n_steps;
        result_b_out->rmse_h = compute_stats(rmse_b);
        result_b_out->nll = compute_stats(nll_b);
        result_b_out->coverage_95 = compute_stats(cov95_b);
        result_b_out->ess_mean = compute_stats(ess_b);
    }
    
    return comp;
}

// =============================================================================
// Output Functions
// =============================================================================

void svpf_test_print_metrics(const SVPFTestMetrics* m) {
    printf("  Seed: %llu\n", (unsigned long long)m->seed);
    printf("  RMSE(h):     %.4f\n", m->rmse_h);
    printf("  RMSE(vol):   %.4f\n", m->rmse_vol);
    printf("  Bias(h):     %.4f\n", m->bias_h);
    printf("  NLL:         %.4f\n", m->nll);
    printf("  Coverage:    50%%=%.1f%%, 90%%=%.1f%%, 95%%=%.1f%%\n",
           m->coverage_50 * 100, m->coverage_90 * 100, m->coverage_95 * 100);
    printf("  ESS:         mean=%.1f%%, min=%.1f%%\n", 
           m->ess_mean * 100, m->ess_min * 100);
    printf("  Correlation: %.4f\n", m->lag_correlation);
    printf("  Runtime:     %.1f ms\n", m->runtime_ms);
}

void svpf_test_print_aggregate(const SVPFTestAggregateResult* r) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  SVPF Test Results: %-45s  ║\n", r->scenario_name);
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Seeds: %d, Steps: %d, Failed: %d                              ║\n",
           r->n_seeds, r->n_steps, r->failed_runs);
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Metric          │   Mean   │   Std    │  Median  │  [P5,P95]     ║\n");
    printf("╟───────────────────┼──────────┼──────────┼──────────┼───────────────╢\n");
    printf("║  RMSE(h)         │ %8.4f │ %8.4f │ %8.4f │ [%.3f,%.3f]   ║\n",
           r->rmse_h.mean, r->rmse_h.std, r->rmse_h.median, r->rmse_h.p5, r->rmse_h.p95);
    printf("║  Bias(h)         │ %+7.4f │ %8.4f │ %+7.4f │ [%+.3f,%+.3f]  ║\n",
           r->bias_h.mean, r->bias_h.std, r->bias_h.median, r->bias_h.p5, r->bias_h.p95);
    printf("║  NLL             │ %8.4f │ %8.4f │ %8.4f │ [%.3f,%.3f]   ║\n",
           r->nll.mean, r->nll.std, r->nll.median, r->nll.p5, r->nll.p95);
    printf("╟───────────────────┼──────────┼──────────┼──────────┼───────────────╢\n");
    printf("║  Coverage(50%%)   │ %7.1f%% │ %7.1f%% │ %7.1f%% │ target: 50%%   ║\n",
           r->coverage_50.mean * 100, r->coverage_50.std * 100, r->coverage_50.median * 100);
    printf("║  Coverage(90%%)   │ %7.1f%% │ %7.1f%% │ %7.1f%% │ target: 90%%   ║\n",
           r->coverage_90.mean * 100, r->coverage_90.std * 100, r->coverage_90.median * 100);
    printf("║  Coverage(95%%)   │ %7.1f%% │ %7.1f%% │ %7.1f%% │ target: 95%%   ║\n",
           r->coverage_95.mean * 100, r->coverage_95.std * 100, r->coverage_95.median * 100);
    printf("╟───────────────────┼──────────┼──────────┼──────────┼───────────────╢\n");
    printf("║  Correlation     │ %8.4f │ %8.4f │ %8.4f │ target: 1.0   ║\n",
           r->lag_correlation.mean, r->lag_correlation.std, r->lag_correlation.median);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("  Total runtime: %.1f s (%.1f ms/run)\n\n", r->total_runtime_s, r->mean_runtime_ms);
}

void svpf_test_print_comparison(const SVPFTestComparison* c) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  A/B Comparison: B vs A (negative diff = B better for errors)     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Metric    │   Diff   │  T-stat  │ P-value  │  Significant?       ║\n");
    printf("╟────────────┼──────────┼──────────┼──────────┼─────────────────────╢\n");
    printf("║  RMSE(h)   │ %+7.4f │ %+7.2f │ %7.4f │ %s %s ║\n",
           c->rmse_h_diff, c->rmse_h_tstat, c->rmse_h_pvalue,
           c->rmse_significant ? "YES" : "NO ",
           c->rmse_b_better ? "(B better)" : "(A better)");
    printf("║  NLL       │ %+7.4f │ %+7.2f │ %7.4f │ %s %s ║\n",
           c->nll_diff, c->nll_tstat, c->nll_pvalue,
           c->nll_significant ? "YES" : "NO ",
           c->nll_b_better ? "(B better)" : "(A better)");
    printf("║  Cov(95%%)  │ %+7.4f │ %+7.2f │ %7.4f │                     ║\n",
           c->coverage_95_diff, c->coverage_95_tstat, c->coverage_95_pvalue);
    printf("║  ESS       │ %+7.4f │ %+7.2f │ %7.4f │                     ║\n",
           c->ess_mean_diff, c->ess_mean_tstat, c->ess_mean_pvalue);
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Effect size (Cohen's d): RMSE=%.2f, NLL=%.2f                     ║\n",
           c->rmse_h_cohens_d, c->nll_cohens_d);
    printf("║  (|d|<0.2: small, 0.2-0.8: medium, >0.8: large)                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
}

int svpf_test_to_json(const SVPFTestAggregateResult* r, char* buffer, int size) {
    return snprintf(buffer, size,
        "{"
        "\"scenario\":\"%s\","
        "\"n_seeds\":%d,"
        "\"n_steps\":%d,"
        "\"rmse_h\":{\"mean\":%.4f,\"std\":%.4f},"
        "\"nll\":{\"mean\":%.4f,\"std\":%.4f},"
        "\"coverage_95\":{\"mean\":%.4f,\"std\":%.4f},"
        "\"correlation\":{\"mean\":%.4f,\"std\":%.4f}"
        "}",
        r->scenario_name, r->n_seeds, r->n_steps,
        r->rmse_h.mean, r->rmse_h.std,
        r->nll.mean, r->nll.std,
        r->coverage_95.mean, r->coverage_95.std,
        r->lag_correlation.mean, r->lag_correlation.std
    );
}

int svpf_test_comparison_to_json(const SVPFTestComparison* c, char* buffer, int size) {
    return snprintf(buffer, size,
        "{"
        "\"rmse_h\":{\"diff\":%.4f,\"pvalue\":%.4f,\"significant\":%s,\"b_better\":%s},"
        "\"nll\":{\"diff\":%.4f,\"pvalue\":%.4f,\"significant\":%s,\"b_better\":%s}"
        "}",
        c->rmse_h_diff, c->rmse_h_pvalue,
        c->rmse_significant ? "true" : "false",
        c->rmse_b_better ? "true" : "false",
        c->nll_diff, c->nll_pvalue,
        c->nll_significant ? "true" : "false",
        c->nll_b_better ? "true" : "false"
    );
}
