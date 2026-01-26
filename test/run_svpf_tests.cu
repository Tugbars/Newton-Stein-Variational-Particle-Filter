/**
 * @file run_svpf_tests.cu
 * @brief SVPF Test Runner - Single command, complete evaluation
 * 
 * Just run it. No arguments needed.
 * 
 * What it does:
 *   1. Runs ALL scenarios
 *   2. ALWAYS compares baseline vs modified
 *   3. Reports statistical significance
 *   4. Gives clear verdict: "Use it" or "Don't bother"
 */

#include "svpf_test_framework.h"
#include "svpf.cuh"
#include <stdio.h>
#include <string.h>

// =============================================================================
// CONFIGURATION - Edit these to change test behavior
// =============================================================================

// Test intensity (trade-off: speed vs statistical power)
#define N_SEEDS         30      // 30 = good balance, 100 = publication quality
#define N_STEPS         3000    // 3000 = reasonable, 5000 = thorough
#define WARMUP_STEPS    100

// Filter settings
#define N_PARTICLES     512
#define N_STEIN_STEPS   5

// =============================================================================
// DEFINE YOUR MODIFICATIONS HERE
// =============================================================================

/**
 * Baseline filter parameters (control group)
 */
static SVPFParams get_baseline_params(const SVPFTestScenario* scenario) {
    SVPFParams p;
    p.rho = scenario->true_rho;
    p.sigma_z = scenario->true_sigma_z;
    p.mu = scenario->true_mu;
    p.gamma = 0.0f;
    return p;
}

/**
 * Modified filter parameters (treatment group)
 * 
 * THIS IS WHERE YOU IMPLEMENT YOUR CHANGES.
 * For state-dependent sigma_z, we'll need to modify the filter itself,
 * not just parameters. For now this is a placeholder.
 */
static SVPFParams get_modified_params(const SVPFTestScenario* scenario) {
    SVPFParams p = get_baseline_params(scenario);
    
    // Example modification: slightly different sigma_z
    // In practice, state-dependent sigma would be a filter flag, not a param
    // p.sigma_z *= 1.1f;  // Uncomment to test
    
    return p;
}

/**
 * Description of what's being compared
 */
static const char* BASELINE_NAME = "Standard SVPF";
static const char* MODIFIED_NAME = "State-Dependent σ_z";  // Change this when testing

// =============================================================================
// MAIN - Just run it
// =============================================================================

int main(int argc, char** argv) {
    (void)argc; (void)argv;  // No arguments needed
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    SVPF A/B Test Suite                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  Comparing: %s  vs  %s\n", BASELINE_NAME, MODIFIED_NAME);
    printf("\n");
    printf("  Configuration:\n");
    printf("    Seeds:     %d (for statistical significance)\n", N_SEEDS);
    printf("    Steps:     %d per run\n", N_STEPS);
    printf("    Particles: %d\n", N_PARTICLES);
    printf("    Stein:     %d iterations\n", N_STEIN_STEPS);
    printf("\n");
    
    // Setup config
    SVPFTestConfig config = svpf_test_default_config();
    config.n_seeds = N_SEEDS;
    config.n_steps = N_STEPS;
    config.warmup_steps = WARMUP_STEPS;
    config.n_particles = N_PARTICLES;
    config.n_stein_steps = N_STEIN_STEPS;
    config.verbose = false;
    
    // Get all scenarios
    SVPFTestScenario scenarios[10];
    int n_scenarios = svpf_test_get_all_scenarios(scenarios, 10);
    
    // Track overall results
    int wins_baseline = 0;
    int wins_modified = 0;
    int ties = 0;
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Running %d scenarios × %d seeds = %d total filter runs...\n",
           n_scenarios, N_SEEDS * 2, n_scenarios * N_SEEDS * 2);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Results table header
    printf("┌─────────────────┬──────────┬──────────┬──────────┬─────────────────┐\n");
    printf("│ Scenario        │ RMSE(A)  │ RMSE(B)  │ p-value  │ Winner          │\n");
    printf("├─────────────────┼──────────┼──────────┼──────────┼─────────────────┤\n");
    
    for (int s = 0; s < n_scenarios; s++) {
        SVPFParams params_a = get_baseline_params(&scenarios[s]);
        SVPFParams params_b = get_modified_params(&scenarios[s]);
        
        // Run comparison
        SVPFTestAggregateResult result_a, result_b;
        SVPFTestComparison comp = svpf_test_compare(
            &scenarios[s], &config,
            &params_a, &params_b,
            &result_a, &result_b
        );
        
        // Determine winner
        const char* winner;
        if (comp.rmse_significant) {
            if (comp.rmse_b_better) {
                winner = "MODIFIED ✓";
                wins_modified++;
            } else {
                winner = "BASELINE ✓";
                wins_baseline++;
            }
        } else {
            winner = "No difference";
            ties++;
        }
        
        printf("│ %-15s │ %8.4f │ %8.4f │ %8.4f │ %-15s │\n",
               scenarios[s].name,
               result_a.rmse_h.mean,
               result_b.rmse_h.mean,
               comp.rmse_h_pvalue,
               winner);
    }
    
    printf("└─────────────────┴──────────┴──────────┴──────────┴─────────────────┘\n\n");
    
    // ==========================================================================
    // VERDICT
    // ==========================================================================
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  VERDICT\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("  Scenarios won:\n");
    printf("    Baseline: %d\n", wins_baseline);
    printf("    Modified: %d\n", wins_modified);
    printf("    Tie:      %d\n", ties);
    printf("\n");
    
    if (wins_modified > wins_baseline && wins_modified > ties) {
        printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
        printf("  ║  RECOMMENDATION: USE THE MODIFICATION                         ║\n");
        printf("  ║  Modified filter wins %d/%d scenarios with p<0.05             ║\n",
               wins_modified, n_scenarios);
        printf("  ╚═══════════════════════════════════════════════════════════════╝\n");
    } else if (wins_baseline > wins_modified) {
        printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
        printf("  ║  RECOMMENDATION: KEEP BASELINE                                ║\n");
        printf("  ║  Modification doesn't help (baseline wins %d/%d)              ║\n",
               wins_baseline, n_scenarios);
        printf("  ╚═══════════════════════════════════════════════════════════════╝\n");
    } else {
        printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
        printf("  ║  RECOMMENDATION: NO CLEAR WINNER                              ║\n");
        printf("  ║  Prefer simpler baseline (Occam's razor)                      ║\n");
        printf("  ╚═══════════════════════════════════════════════════════════════╝\n");
    }
    
    printf("\n");
    
    // ==========================================================================
    // Detailed results (for debugging)
    // ==========================================================================
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  DETAILED METRICS (all scenarios, baseline)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Scenario        │ RMSE(h)  │ Bias     │ Cov(95%%) │ Corr     │ NLL      │\n");
    printf("├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");
    
    for (int s = 0; s < n_scenarios; s++) {
        SVPFParams params_a = get_baseline_params(&scenarios[s]);
        SVPFTestAggregateResult result = svpf_test_run_scenario(&scenarios[s], &config, &params_a);
        
        // Coverage check
        char cov_indicator = ' ';
        if (result.coverage_95.mean < 0.90f) cov_indicator = '!';  // Overconfident
        if (result.coverage_95.mean > 0.98f) cov_indicator = '?';  // Underconfident
        
        printf("│ %-15s │ %8.4f │ %+7.4f │ %6.1f%%%c │ %8.4f │ %8.4f │\n",
               scenarios[s].name,
               result.rmse_h.mean,
               result.bias_h.mean,
               result.coverage_95.mean * 100.0f,
               cov_indicator,
               result.lag_correlation.mean,
               result.nll.mean);
    }
    
    printf("└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n");
    printf("  Coverage: ! = overconfident (<90%%), ? = underconfident (>98%%)\n\n");
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Done.\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}

