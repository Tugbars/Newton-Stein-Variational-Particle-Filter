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
#define N_SEEDS         10      // 30 = good balance, 100 = publication quality
#define N_STEPS         3000    // 3000 = reasonable, 5000 = thorough
#define WARMUP_STEPS    100

// Filter settings
#define N_PARTICLES     512
#define N_STEIN_STEPS   8

// =============================================================================
// FILTER CONFIGURATION FUNCTIONS
// =============================================================================

/**
 * Get baseline SVPFParams for a scenario
 */
static SVPFParams get_params_for_scenario(const SVPFTestScenario* scenario) {
    SVPFParams p;
    p.rho = scenario->true_rho;
    p.sigma_z = scenario->true_sigma_z;
    p.mu = scenario->true_mu;
    p.gamma = 0.0f;
    return p;
}

/**
 * Configure BASELINE filter (control group)
 * 
 * Full production configuration. All features enabled and tuned.
 * Only the feature under test differs between baseline and modified.
 */
static void configure_baseline(void* state_ptr) {
    SVPFState* f = (SVPFState*)state_ptr;
    
    // Core
    f->use_svld = 1;
    f->use_annealing = 1;
    f->n_anneal_steps = 5;
    f->temperature = 0.45f;
    f->rmsprop_rho = 0.9f;
    f->rmsprop_eps = 1e-6f;
    
    // MIM
    f->use_mim = 1;
    f->mim_jump_prob = 0.25f;
    f->mim_jump_scale = 9.0f;
    
    // Asymmetric rho
    //f->use_asymmetric_rho = 0;
    //f->rho_up = 0.99f;
    //f->rho_down = 0.92f;
    
    // Newton-Stein (simplified - more robust under misspecification)
    f->use_newton = 1;
    f->use_full_newton = 1;
    
    // Guided prediction
    f->use_guided = 1;
    f->guided_alpha_base = 0.0f;
    f->guided_alpha_shock = 0.50f;
    f->guided_innovation_threshold = 1.5f;
    
    // EKF guide
    f->use_guide = 1;
    f->use_guide_preserving = 1;
    f->guide_strength = 0.05f;
    
    // Adaptive mu
    f->use_adaptive_mu = 1;
    f->mu_process_var = 0.001f;
    f->mu_obs_var_scale = 11.0f;
    f->mu_min = -4.0f;
    f->mu_max = -1.0f;
    
    // Adaptive guide
    f->use_adaptive_guide = 1;
    f->guide_strength_base = 0.05f;
    f->guide_strength_max = 0.30f;
    f->guide_innovation_threshold = 1.0f;
    
    // Adaptive sigma (breathing)
    f->use_adaptive_sigma = 1;
    f->sigma_boost_threshold = 1.0f;
    f->sigma_boost_max = 3.2f;
    
    // Likelihood / gradient
    f->use_exact_gradient = 1;
    f->lik_offset = 0.345f;

    // =========================================================================
    // FEATURE UNDER TEST - OFF in baseline
    // =========================================================================
    f->use_local_params = 0;
    f->delta_rho = 0.0f;
    f->delta_sigma = 0.0f;

    // === KSD-based Adaptive Stein Steps ===
    // Replaces fixed n_stein_steps with convergence-based early stopping
    // KSD (Kernel Stein Discrepancy) computed in same O(N²) pass - zero extra
    // cost
    f->stein_min_steps = 8;  // Always run at least 4 (RMSProp warmup)
    f->stein_max_steps = 16; // Cap at 12 (crisis budget)
    f->ksd_improvement_threshold = 0.05; // Stop if <5% relative improvement

    // Enable Student-t state dynamics
    f->use_student_t_state = 1;
    f->nu_state = 5.0f; // 5-7 recommended, lower = fatter tails

    // Enable smoothing with 1-tick output lag
    f->use_smoothing = 1;
    f->smooth_lag = 3;        // Buffer last 3 estimates
    f->smooth_output_lag = 1; // Output h[t-1] (smoothed by y[t])

    f->use_persistent_kernel = 1;

    //f->use_heun = 1;
}

/**
 * Configure MODIFIED filter (treatment group)
 * 
 * Same as baseline, but with the feature under test enabled.
 */
static void configure_modified(void* state_ptr) {
    configure_baseline(state_ptr);
    SVPFState* f = (SVPFState*)state_ptr;
    
    // =========================================================================
    // FEATURE UNDER TEST - ON in modified
    // =========================================================================
    //f->use_local_params = 1;
   // f->delta_sigma = 0.15f;  // State-dependent sigma: wider when far from mu
    // f->delta_rho = 0.04f; // Could also test state-dependent rho
    //f->use_heun = 1;
}

/**
 * Description of what's being compared
 */
static const char* BASELINE_NAME = "Standard (local_params=OFF)";
static const char* MODIFIED_NAME = "State-Dependent σ_z (local_params=ON)";

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
    printf("  Comparing:\n");
    printf("    A: %s\n", BASELINE_NAME);
    printf("    B: %s\n", MODIFIED_NAME);
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
    printf("  Running %d scenarios × %d seeds × 2 configs = %d filter runs...\n",
           n_scenarios, N_SEEDS, n_scenarios * N_SEEDS * 2);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Results table header
    printf("┌─────────────────┬──────────┬──────────┬──────────┬─────────────────┐\n");
    printf("│ Scenario        │ RMSE(A)  │ RMSE(B)  │ p-value  │ Winner          │\n");
    printf("├─────────────────┼──────────┼──────────┼──────────┼─────────────────┤\n");
    
    for (int s = 0; s < n_scenarios; s++) {
        SVPFParams params = get_params_for_scenario(&scenarios[s]);
        
        // Run comparison with proper configuration
        SVPFTestAggregateResult result_a, result_b;
        SVPFTestComparison comp = svpf_test_compare_with_config(
            &scenarios[s], &config,
            &params, configure_baseline,
            &params, configure_modified,
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
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Done.\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}

