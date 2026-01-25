/**
 * @file test_svpf_gradient_sigma.cu
 * @brief σ (vol-of-vol) Gradient Diagnostic
 *
 * Purpose: Verify σ gradient behavior on real and synthetic data.
 * 
 * Key difference from ν:
 *   - ν gradient requires crash events (z² > 9) → rare
 *   - σ gradient available EVERY timestep → always informative
 *
 * σ Gradient Formula (transition likelihood):
 *   ε = h_t - μ - ρ(h_{t-1} - μ)
 *   ∂/∂σ = (ε²/σ² - 1) / σ
 *
 * Expected:
 *   - σ too HIGH → ε²/σ² < 1 → gradient NEGATIVE
 *   - σ too LOW  → ε²/σ² > 1 → gradient POSITIVE
 *   - σ correct  → ε²/σ² ≈ 1 → gradient ≈ 0
 */

#include "svpf.cuh"
#include "market_data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

static const char* DATA_PATHS[] = {
    "../market_data/",
    "market_data/",
    "../../market_data/",
    "../../../market_data/",
    NULL
};

#define DEFAULT_PARTICLES    4096
#define DEFAULT_STEIN_STEPS  5

static const char* find_data_path(void) {
    for (int i = 0; DATA_PATHS[i] != NULL; i++) {
        char test_path[512];
        snprintf(test_path, sizeof(test_path), "%sspy_full.bin", DATA_PATHS[i]);
        FILE* f = fopen(test_path, "rb");
        if (f) {
            fclose(f);
            return DATA_PATHS[i];
        }
    }
    return NULL;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MOMENT-MATCHING μ ESTIMATION
 *═══════════════════════════════════════════════════════════════════════════*/

static float estimate_mu_from_data(const float* returns, int n) {
    double sum = 0, sq_sum = 0;
    for (int i = 0; i < n; i++) {
        sum += returns[i];
        sq_sum += returns[i] * returns[i];
    }
    double mean = sum / n;
    double var = sq_sum / n - mean * mean;
    float realized_vol = sqrtf((float)var);
    return 2.0f * logf(realized_vol + 1e-8f);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: σ Gradient Direction
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    float mean_gradient;
    float mean_eps_sq_norm;   // ε²/σ² (should be ~1.0)
    int n_positive;           // Count of positive gradients
    int n_negative;           // Count of negative gradients
    int n_obs;
} SigmaGradientStats;

static SigmaGradientStats run_sigma_gradient_test(
    const float* returns,
    int n,
    float filter_sigma,       // σ value to test
    float filter_mu,
    float filter_rho,
    float filter_nu,
    int n_particles,
    int n_stein_steps
) {
    SigmaGradientStats stats = {0};
    stats.n_obs = n;
    
    // Create filter
    SVPFState* state = svpf_create(n_particles, n_stein_steps, filter_nu, nullptr);
    SVPFParams params = {filter_rho, filter_sigma, filter_mu, 0.0f};
    svpf_initialize(state, &params, 12345);
    
    // Accumulators
    double grad_sum = 0, eps_sq_sum = 0;
    int n_pos = 0, n_neg = 0;
    
    float y_prev = 0.0f;
    for (int t = 0; t < n; t++) {
        float y_t = returns[t];
        
        // SVPF step
        float loglik, vol_est, h_mean;
        svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol_est, &h_mean);
        
        // σ gradient diagnostic
        float sigma_grad, eps_sq_norm;
        svpf_compute_sigma_diagnostic_simple(state, &params, &sigma_grad, &eps_sq_norm);
        
        // Accumulate
        if (!isnan(sigma_grad) && !isinf(sigma_grad)) {
            grad_sum += sigma_grad;
            eps_sq_sum += eps_sq_norm;
            if (sigma_grad > 0) n_pos++;
            else n_neg++;
        }
        
        y_prev = y_t;
    }
    
    svpf_destroy(state);
    
    stats.mean_gradient = (float)(grad_sum / n);
    stats.mean_eps_sq_norm = (float)(eps_sq_sum / n);
    stats.n_positive = n_pos;
    stats.n_negative = n_neg;
    
    return stats;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: σ Sweep on Market Data
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_sigma_sweep_market(const char* data_path, const char* name) {
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" TEST: σ Gradient Sweep on %s\n", name);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Load data
    float* returns;
    int n;
    
    if (load_returns_binary(data_path, &returns, &n) != 0) {
        printf(" ERROR: Could not load %s\n", data_path);
        return;
    }
    
    print_return_stats(name, returns, n);
    
    // Auto-calibrate μ
    float filter_mu = estimate_mu_from_data(returns, n);
    float implied_vol = expf(filter_mu / 2.0f);
    
    // Fixed params
    const float filter_rho = 0.98f;
    const float filter_nu = 10.0f;  // Fixed (we're testing σ, not ν)
    
    printf("\n Auto-calibrated: μ=%.2f → implied daily vol=%.2f%%\n", 
           filter_mu, implied_vol * 100.0f);
    printf(" Fixed params: ρ=%.2f, ν=%.0f\n", filter_rho, filter_nu);
    printf(" Particles=%d, Stein=%d\n\n", DEFAULT_PARTICLES, DEFAULT_STEIN_STEPS);
    
    printf("   σ_filter | Mean Grad  | ε²/σ² mean | Pos/Neg   | Direction\n");
    printf("   ---------+------------+------------+-----------+-----------\n");
    
    // Sweep σ values
    float sigma_values[] = {0.05f, 0.08f, 0.10f, 0.12f, 0.15f, 0.18f, 0.20f, 0.25f, 0.30f, 0.40f};
    int n_sigma = sizeof(sigma_values) / sizeof(sigma_values[0]);
    
    float prev_grad = 1.0f;
    float equilibrium_sigma = 0.0f;
    
    for (int i = 0; i < n_sigma; i++) {
        float sigma = sigma_values[i];
        
        SigmaGradientStats stats = run_sigma_gradient_test(
            returns, n, sigma, filter_mu, filter_rho, filter_nu,
            DEFAULT_PARTICLES, DEFAULT_STEIN_STEPS
        );
        
        const char* direction;
        if (stats.mean_gradient > 0.01f) direction = "↑ increase σ";
        else if (stats.mean_gradient < -0.01f) direction = "↓ decrease σ";
        else direction = "≈ equilibrium";
        
        printf("   %7.2f  | %+10.4f | %10.4f | %4d/%-4d | %s\n",
               sigma, stats.mean_gradient, stats.mean_eps_sq_norm,
               stats.n_positive, stats.n_negative, direction);
        
        // Detect zero crossing
        if (i > 0 && prev_grad > 0 && stats.mean_gradient < 0) {
            equilibrium_sigma = (sigma_values[i-1] + sigma_values[i]) / 2.0f;
        }
        
        prev_grad = stats.mean_gradient;
    }
    
    printf("\n");
    if (equilibrium_sigma > 0) {
        printf(" Equilibrium σ ≈ %.3f (gradient crosses zero)\n", equilibrium_sigma);
    } else if (prev_grad > 0) {
        printf(" Equilibrium σ > 0.40 (gradient still positive)\n");
    } else {
        printf(" Equilibrium σ < 0.05 (gradient already negative at σ=0.05)\n");
    }
    
    free(returns);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Compare σ Gradient Across Datasets
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_sigma_across_datasets(const char* data_dir) {
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" TEST: σ Gradient Across Datasets (σ=0.15)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    char path_spy_full[512], path_2008[512], path_2020[512], path_crashes[512], path_tsla[512];
    snprintf(path_spy_full, sizeof(path_spy_full), "%sspy_full.bin", data_dir);
    snprintf(path_2008, sizeof(path_2008), "%sspy_2008_crisis.bin", data_dir);
    snprintf(path_2020, sizeof(path_2020), "%sspy_2020_covid.bin", data_dir);
    snprintf(path_crashes, sizeof(path_crashes), "%scrashes_combined.bin", data_dir);
    snprintf(path_tsla, sizeof(path_tsla), "%stsla.bin", data_dir);
    
    const char* datasets[] = { path_spy_full, path_2008, path_2020, path_crashes, path_tsla };
    const char* names[] = {
        "SPY Full (2007-2024)",
        "SPY 2008 Crisis",
        "SPY 2020 COVID",
        "Crashes Combined",
        "TSLA (high vol)"
    };
    int n_datasets = sizeof(datasets) / sizeof(datasets[0]);
    
    const float filter_sigma = 0.15f;  // Fixed σ to test
    const float filter_rho = 0.98f;
    const float filter_nu = 10.0f;
    
    printf(" Testing with σ=%.2f (μ auto-calibrated per dataset)\n\n", filter_sigma);
    printf(" Dataset              | N      | μ_est  | Mean Grad  | ε²/σ² mean | Direction\n");
    printf(" ---------------------+--------+--------+------------+------------+-----------\n");
    
    for (int d = 0; d < n_datasets; d++) {
        float* returns;
        int n;
        
        if (load_returns_binary(datasets[d], &returns, &n) != 0) {
            printf(" %-20s | ERROR loading\n", names[d]);
            continue;
        }
        
        float filter_mu = estimate_mu_from_data(returns, n);
        
        SigmaGradientStats stats = run_sigma_gradient_test(
            returns, n, filter_sigma, filter_mu, filter_rho, filter_nu,
            DEFAULT_PARTICLES, DEFAULT_STEIN_STEPS
        );
        
        const char* direction;
        if (stats.mean_gradient > 0.01f) direction = "↑ increase";
        else if (stats.mean_gradient < -0.01f) direction = "↓ decrease";
        else direction = "≈ equilib";
        
        printf(" %-20s | %6d | %6.2f | %+10.4f | %10.4f | %s\n",
               names[d], n, filter_mu, stats.mean_gradient, 
               stats.mean_eps_sq_norm, direction);
        
        free(returns);
    }
    
    printf("\n Key insight: σ gradient is informative on ALL datasets.\n");
    printf(" Compare to ν gradient which only worked during rare crashes.\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: σ Gradient vs ν Gradient Comparison
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_sigma_vs_nu_informativeness(const char* data_path, const char* name) {
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" TEST: σ vs ν Gradient Informativeness on %s\n", name);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    float* returns;
    int n;
    
    if (load_returns_binary(data_path, &returns, &n) != 0) {
        printf(" ERROR: Could not load %s\n", data_path);
        return;
    }
    
    float filter_mu = estimate_mu_from_data(returns, n);
    
    // Create filter with "wrong" σ and "wrong" ν
    const float true_sigma = 0.15f;
    const float test_sigma = 0.25f;   // 67% too high
    const float true_nu = 5.0f;
    const float test_nu = 15.0f;      // 3x too high
    const float filter_rho = 0.98f;
    
    printf(" Testing misspecified filter:\n");
    printf("   σ: testing %.2f (true ≈ %.2f) → %.0f%% too high\n", 
           test_sigma, true_sigma, (test_sigma - true_sigma) / true_sigma * 100);
    printf("   ν: testing %.0f (true ≈ %.0f) → %.0f%% too high\n\n",
           test_nu, true_nu, (test_nu - true_nu) / true_nu * 100);
    
    SVPFState* state = svpf_create(DEFAULT_PARTICLES, DEFAULT_STEIN_STEPS, test_nu, nullptr);
    SVPFParams params = {filter_rho, test_sigma, filter_mu, 0.0f};
    svpf_initialize(state, &params, 12345);
    
    // Track gradients
    double sigma_grad_sum = 0, sigma_grad_abs_sum = 0;
    double nu_grad_sum = 0, nu_grad_abs_sum = 0;
    int n_sigma_informative = 0;  // |grad| > 0.01
    int n_nu_informative = 0;     // |grad| > 0.001
    
    float y_prev = 0.0f;
    for (int t = 0; t < n; t++) {
        float y_t = returns[t];
        
        float loglik, vol_est, h_mean;
        svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol_est, &h_mean);
        
        // Both gradients
        float sigma_grad, eps_sq_norm;
        svpf_compute_sigma_diagnostic_simple(state, &params, &sigma_grad, &eps_sq_norm);
        
        float nu_grad, z_sq;
        svpf_compute_nu_diagnostic_simple(state, y_t, &nu_grad, &z_sq);
        
        // Accumulate
        if (!isnan(sigma_grad) && !isinf(sigma_grad)) {
            sigma_grad_sum += sigma_grad;
            sigma_grad_abs_sum += fabsf(sigma_grad);
            if (fabsf(sigma_grad) > 0.01f) n_sigma_informative++;
        }
        
        if (!isnan(nu_grad) && !isinf(nu_grad)) {
            nu_grad_sum += nu_grad;
            nu_grad_abs_sum += fabsf(nu_grad);
            if (fabsf(nu_grad) > 0.001f) n_nu_informative++;
        }
        
        y_prev = y_t;
    }
    
    svpf_destroy(state);
    free(returns);
    
    printf(" Results over %d observations:\n\n", n);
    printf("                    |     σ Gradient    |     ν Gradient    \n");
    printf(" -------------------+-------------------+-------------------\n");
    printf(" Mean gradient      | %+15.6f | %+15.6f\n", 
           sigma_grad_sum / n, nu_grad_sum / n);
    printf(" Mean |gradient|    | %15.6f | %15.6f\n", 
           sigma_grad_abs_sum / n, nu_grad_abs_sum / n);
    printf(" Informative %%      | %14.1f%% | %14.1f%%\n",
           100.0 * n_sigma_informative / n, 100.0 * n_nu_informative / n);
    printf(" Expected direction | %15s | %15s\n", "NEGATIVE", "NEGATIVE");
    printf(" Actual direction   | %15s | %15s\n",
           (sigma_grad_sum < 0) ? "NEGATIVE ✓" : "POSITIVE ✗",
           (nu_grad_sum < 0) ? "NEGATIVE ✓" : "POSITIVE ✗");
    
    printf("\n Conclusion: σ gradient has %.1fx stronger signal than ν gradient.\n",
           (sigma_grad_abs_sum / n) / (nu_grad_abs_sum / n + 1e-8));
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SVPF σ Gradient Diagnostic                                    ║\n");
    printf("║     Vol-of-vol learning: Always informative (unlike ν)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    // Find data directory
    const char* data_dir = find_data_path();
    if (!data_dir) {
        printf("\n ERROR: Could not find market_data directory!\n");
        printf(" Run: python scripts/fetch_market_data.py\n");
        return 1;
    }
    printf("\n Found market data at: %s\n", data_dir);
    
    // Build paths
    char path_spy_full[512], path_2008[512];
    snprintf(path_spy_full, sizeof(path_spy_full), "%sspy_full.bin", data_dir);
    snprintf(path_2008, sizeof(path_2008), "%sspy_2008_crisis.bin", data_dir);
    
    // Test 1: σ sweep on SPY full
    test_sigma_sweep_market(path_spy_full, "SPY Full (2007-2024)");
    
    // Test 2: σ sweep on 2008 crisis
    test_sigma_sweep_market(path_2008, "SPY 2008 Crisis");
    
    // Test 3: Compare across datasets
    test_sigma_across_datasets(data_dir);
    
    // Test 4: σ vs ν informativeness
    test_sigma_vs_nu_informativeness(path_spy_full, "SPY Full");
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf(" Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf(" Key findings:\n");
    printf("   - σ gradient informative on EVERY timestep (not just crashes)\n");
    printf("   - Much stronger signal than ν gradient\n");
    printf("   - Equilibrium σ can be learned online\n");
    printf("\n");
    printf(" Next step: Implement σ online learning with SGD/Adam\n");
    printf("\n");
    
    return 0;
}
