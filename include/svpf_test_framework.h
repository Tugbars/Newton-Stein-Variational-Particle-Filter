/**
 * @file svpf_test_framework.h
 * @brief Rigorous testing framework for SVPF algorithm evaluation
 * 
 * Provides:
 *   - Synthetic data generation with known ground truth
 *   - Multiple test scenarios (calm, crisis, jumps, misspecified)
 *   - Comprehensive metrics (RMSE, coverage, NLL, ESS)
 *   - A/B testing with statistical significance
 *   - Multi-seed aggregation for robust conclusions
 * 
 * Usage:
 *   SVPFTestScenario scenario = svpf_test_scenario_calm();
 *   SVPFTestConfig config = {.n_seeds = 100, .n_steps = 5000};
 *   SVPFTestAggregateResult result = svpf_test_run_scenario(scenario, config, filter_params);
 *   svpf_test_print_result(&result);
 */

#ifndef SVPF_TEST_FRAMEWORK_H
#define SVPF_TEST_FRAMEWORK_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Test Scenario Definition
// =============================================================================

/**
 * @brief True data-generating process parameters
 * 
 * These are the GROUND TRUTH parameters used to simulate synthetic data.
 * The filter may use different parameters (for misspecification testing).
 */
typedef struct {
    // AR(1) log-vol dynamics
    float true_mu;          // True mean log-vol
    float true_rho;         // True persistence
    float true_sigma_z;     // True vol-of-vol
    
    // Observation model
    float true_nu;          // True Student-t df (INFINITY for Gaussian)
    
    // Asymmetry (optional)
    bool asymmetric;
    float true_rho_up;      // Persistence when vol rising
    float true_rho_down;    // Persistence when vol falling
    
    // Jumps (optional)
    bool has_jumps;
    float jump_prob;        // Probability of jump per step
    float jump_scale;       // Jump size multiplier
    
    // Regime shifts (optional)
    bool has_regime_shift;
    int regime_shift_time;  // Step at which shift occurs
    float regime_shift_mu;  // New mu after shift
    
    // Scenario metadata
    const char* name;
    const char* description;
    
} SVPFTestScenario;

// =============================================================================
// Test Configuration
// =============================================================================

typedef struct {
    // Run parameters
    int n_seeds;            // Number of random seeds (recommend: 100)
    int n_steps;            // Steps per run (recommend: 5000)
    int warmup_steps;       // Steps to discard (recommend: 100)
    
    // Filter configuration
    int n_particles;        // Particles (recommend: 512)
    int n_stein_steps;      // Stein iterations (recommend: 5)
    
    // Coverage levels to compute
    float coverage_levels[3];  // e.g., {0.50, 0.90, 0.95}
    int n_coverage_levels;
    
    // Output options
    bool verbose;           // Print per-seed results
    bool save_traces;       // Save h_true, h_est for plotting
    const char* output_dir; // Directory for traces (if save_traces)
    
} SVPFTestConfig;

// =============================================================================
// Single-Run Metrics
// =============================================================================

/**
 * @brief Metrics from a single test run (one seed)
 */
typedef struct {
    // Accuracy (vs ground truth)
    float rmse_h;           // RMSE of log-vol estimate
    float rmse_vol;         // RMSE of vol estimate (exp(h/2))
    float mae_h;            // Mean absolute error
    float bias_h;           // Mean signed error (positive = overestimate)
    
    // Probabilistic calibration
    float nll;              // Negative log-likelihood (lower = better)
    float coverage_50;      // % of true h in 50% CI
    float coverage_90;      // % of true h in 90% CI
    float coverage_95;      // % of true h in 95% CI
    
    // Filter health
    float ess_mean;         // Mean ESS ratio over run
    float ess_min;          // Minimum ESS ratio
    float ess_below_30_pct; // % of steps with ESS < 30%
    float bandwidth_mean;   // Mean kernel bandwidth
    
    // Tracking dynamics
    float lag_correlation;  // Cross-corr of h_est vs h_true at lag 0
    float lag_1_correlation;// Cross-corr at lag 1 (if high, filter is lagging)
    int adaptation_steps;   // Steps to recover from jump (if applicable)
    
    // Numerical health
    int nan_count;          // Number of NaN outputs
    int inf_count;          // Number of Inf outputs
    int resampling_count;   // Number of resampling events
    
    // Metadata
    uint64_t seed;
    float runtime_ms;
    
} SVPFTestMetrics;

// =============================================================================
// Aggregate Results (Multi-Seed)
// =============================================================================

/**
 * @brief Statistics for a single metric across seeds
 */
typedef struct {
    float mean;
    float std;
    float median;
    float p5;               // 5th percentile
    float p95;              // 95th percentile
    float min;
    float max;
} MetricStats;

/**
 * @brief Aggregated results across all seeds for a scenario
 */
typedef struct {
    // Scenario info
    const char* scenario_name;
    int n_seeds;
    int n_steps;
    
    // Aggregated metrics
    MetricStats rmse_h;
    MetricStats rmse_vol;
    MetricStats bias_h;
    MetricStats nll;
    MetricStats coverage_50;
    MetricStats coverage_90;
    MetricStats coverage_95;
    MetricStats ess_mean;
    MetricStats ess_min;
    MetricStats bandwidth_mean;
    MetricStats lag_correlation;
    
    // Overall health
    int total_nans;
    int total_infs;
    int failed_runs;        // Runs that crashed or had NaN
    
    // Timing
    float total_runtime_s;
    float mean_runtime_ms;
    
} SVPFTestAggregateResult;

// =============================================================================
// A/B Comparison Results
// =============================================================================

/**
 * @brief Paired comparison of two filter configurations
 */
typedef struct {
    // Mean differences (B - A, negative = B better for error metrics)
    float rmse_h_diff;
    float nll_diff;
    float coverage_95_diff;
    float ess_mean_diff;
    
    // Standard error of differences
    float rmse_h_se;
    float nll_se;
    float coverage_95_se;
    float ess_mean_se;
    
    // T-statistics
    float rmse_h_tstat;
    float nll_tstat;
    float coverage_95_tstat;
    float ess_mean_tstat;
    
    // P-values (two-tailed)
    float rmse_h_pvalue;
    float nll_pvalue;
    float coverage_95_pvalue;
    float ess_mean_pvalue;
    
    // Effect sizes (Cohen's d)
    float rmse_h_cohens_d;
    float nll_cohens_d;
    
    // Verdict
    bool rmse_significant;      // p < 0.05
    bool nll_significant;
    bool rmse_b_better;         // B has lower RMSE
    bool nll_b_better;          // B has lower NLL
    
} SVPFTestComparison;

// =============================================================================
// Pre-defined Scenarios
// =============================================================================

/**
 * @brief Calm market - baseline performance
 */
SVPFTestScenario svpf_test_scenario_calm(void);

/**
 * @brief Persistent high volatility (like 2008 crisis)
 */
SVPFTestScenario svpf_test_scenario_crisis(void);

/**
 * @brief Sudden volatility spike (flash crash)
 */
SVPFTestScenario svpf_test_scenario_spike(void);

/**
 * @brief Regime shift - μ changes mid-series
 */
SVPFTestScenario svpf_test_scenario_regime_shift(void);

/**
 * @brief Asymmetric dynamics (leverage effect)
 */
SVPFTestScenario svpf_test_scenario_asymmetric(void);

/**
 * @brief With jumps (MIM-like DGP)
 */
SVPFTestScenario svpf_test_scenario_jumps(void);

/**
 * @brief Fat tails (Student-t observations)
 */
SVPFTestScenario svpf_test_scenario_fat_tails(void);

/**
 * @brief Get all standard scenarios
 */
int svpf_test_get_all_scenarios(SVPFTestScenario* scenarios, int max_scenarios);

// =============================================================================
// Data Generation
// =============================================================================

/**
 * @brief Generate synthetic SV data with known ground truth
 * 
 * @param scenario      DGP parameters
 * @param n_steps       Number of time steps
 * @param seed          Random seed
 * @param h_true_out    Output: true log-vol (size n_steps)
 * @param y_out         Output: observed returns (size n_steps)
 */
void svpf_test_generate_data(
    const SVPFTestScenario* scenario,
    int n_steps,
    uint64_t seed,
    float* h_true_out,
    float* y_out
);

// =============================================================================
// Test Execution
// =============================================================================

/**
 * @brief Default test configuration
 */
SVPFTestConfig svpf_test_default_config(void);

/**
 * @brief Run single test (one seed)
 * 
 * @param scenario      DGP parameters
 * @param config        Test configuration
 * @param filter_params Filter parameters (may differ from scenario for misspec)
 * @param seed          Random seed
 * @return              Metrics from this run
 */
SVPFTestMetrics svpf_test_run_single(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params,  // SVPFParams*
    uint64_t seed
);

/**
 * @brief Run full test suite (all seeds)
 * 
 * @param scenario      DGP parameters  
 * @param config        Test configuration
 * @param filter_params Filter parameters
 * @return              Aggregated results
 */
SVPFTestAggregateResult svpf_test_run_scenario(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params
);

/**
 * @brief A/B comparison of two filter configurations
 * 
 * Uses paired t-test on same random seeds.
 * 
 * @param scenario          DGP parameters
 * @param config            Test configuration
 * @param filter_params_a   Baseline filter
 * @param filter_params_b   Modified filter
 * @param result_a_out      Aggregate results for A (optional, can be NULL)
 * @param result_b_out      Aggregate results for B (optional, can be NULL)
 * @return                  Statistical comparison
 */
SVPFTestComparison svpf_test_compare(
    const SVPFTestScenario* scenario,
    const SVPFTestConfig* config,
    const void* filter_params_a,
    const void* filter_params_b,
    SVPFTestAggregateResult* result_a_out,
    SVPFTestAggregateResult* result_b_out
);

// =============================================================================
// Output and Reporting
// =============================================================================

/**
 * @brief Print single-run metrics
 */
void svpf_test_print_metrics(const SVPFTestMetrics* metrics);

/**
 * @brief Print aggregated results
 */
void svpf_test_print_aggregate(const SVPFTestAggregateResult* result);

/**
 * @brief Print A/B comparison with significance indicators
 */
void svpf_test_print_comparison(const SVPFTestComparison* comparison);

/**
 * @brief Export results to JSON
 */
int svpf_test_to_json(const SVPFTestAggregateResult* result, char* buffer, int size);

/**
 * @brief Export comparison to JSON  
 */
int svpf_test_comparison_to_json(const SVPFTestComparison* comparison, char* buffer, int size);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute coverage: what % of true values fall within CI?
 * 
 * @param h_true        True log-vol values
 * @param h_particles   Particle values at each step (n_steps x n_particles)
 * @param n_steps       Number of time steps
 * @param n_particles   Number of particles
 * @param level         CI level (e.g., 0.95 for 95% CI)
 * @return              Coverage fraction (target: equal to level)
 */
float svpf_test_compute_coverage(
    const float* h_true,
    const float* h_particles,
    int n_steps,
    int n_particles,
    float level
);

/**
 * @brief Compute cross-correlation between h_true and h_est
 * 
 * @param h_true    True values
 * @param h_est     Estimated values
 * @param n         Length
 * @param max_lag   Maximum lag to compute
 * @param xcorr_out Output: correlation at each lag (size 2*max_lag+1)
 */
void svpf_test_cross_correlation(
    const float* h_true,
    const float* h_est,
    int n,
    int max_lag,
    float* xcorr_out
);

/**
 * @brief Paired t-test
 * 
 * @param diffs     Array of paired differences
 * @param n         Number of pairs
 * @param t_stat    Output: t-statistic
 * @param p_value   Output: two-tailed p-value
 */
void svpf_test_paired_ttest(
    const float* diffs,
    int n,
    float* t_stat,
    float* p_value
);

#ifdef __cplusplus
}
#endif

#endif // SVPF_TEST_FRAMEWORK_H
