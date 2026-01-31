/**
 * @file test_two_factor.cu
 * @brief Two-Factor SVPF Test with Correctly-Specified DGP
 * 
 * Generates data from actual two-factor SV model:
 *   h_fast,t = rho_fast * h_fast,t-1 + sigma_fast * eps_fast
 *   h_slow,t = rho_slow * h_slow,t-1 + sigma_slow * eps_slow
 *   h_t = mu + h_fast,t + h_slow,t
 *   y_t = exp(h_t/2) * z_t,   z_t ~ Student-t(nu)
 * 
 * Compares:
 *   A: Single-factor filter (misspecified)
 *   B: Two-factor filter (correctly specified)
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define N_SEEDS         10
#define N_STEPS         3000
#define WARMUP_STEPS    100
#define N_PARTICLES     512
#define N_STEIN_STEPS   8

// =============================================================================
// TWO-FACTOR DGP PARAMETERS
// =============================================================================

struct TwoFactorDGPParams {
    // Observation model
    float mu;           // Long-run mean
    float nu;           // Student-t df
    
    // Fast component (spikes)
    float rho_fast;
    float sigma_fast;
    
    // Slow component (regimes)
    float rho_slow;
    float sigma_slow;
    
    const char* name;
};

// Scenario presets
static TwoFactorDGPParams dgp_balanced() {
    TwoFactorDGPParams p;
    p.mu = -3.5f;
    p.nu = 7.0f;
    p.rho_fast = 0.90f;
    p.sigma_fast = 0.15f;
    p.rho_slow = 0.99f;
    p.sigma_slow = 0.05f;
    p.name = "balanced";
    return p;
}

static TwoFactorDGPParams dgp_fast_dominant() {
    // Fast component dominates - lots of spikes
    TwoFactorDGPParams p;
    p.mu = -3.5f;
    p.nu = 7.0f;
    p.rho_fast = 0.85f;
    p.sigma_fast = 0.25f;
    p.rho_slow = 0.995f;
    p.sigma_slow = 0.02f;
    p.name = "fast_dominant";
    return p;
}

static TwoFactorDGPParams dgp_slow_dominant() {
    // Slow component dominates - regime shifts
    TwoFactorDGPParams p;
    p.mu = -3.5f;
    p.nu = 7.0f;
    p.rho_fast = 0.95f;
    p.sigma_fast = 0.08f;
    p.rho_slow = 0.98f;
    p.sigma_slow = 0.12f;
    p.name = "slow_dominant";
    return p;
}

static TwoFactorDGPParams dgp_crisis() {
    // High persistence on both, large vol-of-vol
    TwoFactorDGPParams p;
    p.mu = -3.0f;
    p.nu = 5.0f;
    p.rho_fast = 0.92f;
    p.sigma_fast = 0.20f;
    p.rho_slow = 0.995f;
    p.sigma_slow = 0.08f;
    p.name = "crisis";
    return p;
}

// =============================================================================
// TWO-FACTOR DATA GENERATION
// =============================================================================

static void generate_two_factor_data(
    const TwoFactorDGPParams& dgp,
    int n_steps,
    float* y_out,
    float* h_true_out,
    float* h_fast_out,
    float* h_slow_out,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    // Student-t via normal/chi2
    std::chi_squared_distribution<float> chi2(dgp.nu);
    
    // Initialize from stationary distribution
    float std_fast = dgp.sigma_fast / sqrtf(1.0f - dgp.rho_fast * dgp.rho_fast);
    float std_slow = dgp.sigma_slow / sqrtf(1.0f - dgp.rho_slow * dgp.rho_slow);
    
    float h_fast = std_fast * normal(rng);
    float h_slow = std_slow * normal(rng);
    
    for (int t = 0; t < n_steps; t++) {
        // State transition
        float eps_fast = normal(rng);
        float eps_slow = normal(rng);
        
        h_fast = dgp.rho_fast * h_fast + dgp.sigma_fast * eps_fast;
        h_slow = dgp.rho_slow * h_slow + dgp.sigma_slow * eps_slow;
        
        float h = dgp.mu + h_fast + h_slow;
        
        // Observation: y = exp(h/2) * z, z ~ Student-t(nu)
        float z_normal = normal(rng);
        float v = chi2(rng);
        float z_t = z_normal * sqrtf(dgp.nu / v);  // Student-t
        
        float vol = expf(h / 2.0f);
        float y = vol * z_t;
        
        // Store
        y_out[t] = y;
        h_true_out[t] = h;
        if (h_fast_out) h_fast_out[t] = h_fast;
        if (h_slow_out) h_slow_out[t] = h_slow;
    }
}

// =============================================================================
// RMSE CALCULATION
// =============================================================================

static float compute_rmse(const float* est, const float* truth, int n, int warmup) {
    double sum_sq = 0.0;
    int count = 0;
    for (int t = warmup; t < n; t++) {
        double diff = est[t] - truth[t];
        sum_sq += diff * diff;
        count++;
    }
    return (float)sqrt(sum_sq / count);
}

static float compute_correlation(const float* est, const float* truth, int n, int warmup) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    int count = 0;
    for (int t = warmup; t < n; t++) {
        double x = est[t];
        double y = truth[t];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    double mean_x = sum_x / count;
    double mean_y = sum_y / count;
    double cov = sum_xy / count - mean_x * mean_y;
    double std_x = sqrt(sum_x2 / count - mean_x * mean_x);
    double std_y = sqrt(sum_y2 / count - mean_y * mean_y);
    return (float)(cov / (std_x * std_y + 1e-10));
}

// =============================================================================
// FILTER CONFIGURATION
// =============================================================================

static void configure_single_factor(SVPFState* f) {
    // Production single-factor config
    f->use_svld = 1;
    f->use_annealing = 1;
    f->n_anneal_steps = 5;
    f->temperature = 0.45f;
    
    f->use_mim = 1;
    f->mim_jump_prob = 0.25f;
    f->mim_jump_scale = 9.0f;
    
    f->use_newton = 1;
    f->use_full_newton = 1;
    
    f->use_guided = 1;
    f->guided_alpha_base = 0.0f;
    f->guided_alpha_shock = 0.50f;
    f->guided_innovation_threshold = 1.5f;
    
    f->use_guide = 1;
    f->use_guide_preserving = 1;
    f->guide_strength = 0.05f;
    
    f->use_adaptive_mu = 1;
    f->mu_process_var = 0.001f;
    f->mu_obs_var_scale = 11.0f;
    
    f->use_adaptive_guide = 1;
    f->guide_strength_base = 0.05f;
    f->guide_strength_max = 0.30f;
    
    f->use_adaptive_sigma = 1;
    f->sigma_boost_threshold = 1.0f;
    f->sigma_boost_max = 3.2f;
    
    f->use_exact_gradient = 1;
    f->lik_offset = 0.345f;
    f->use_student_t_state = 1;
    f->nu_state = 5.0f;
    
    f->stein_min_steps = 8;
    f->stein_max_steps = 16;
    f->ksd_improvement_threshold = 0.05f;
    
    f->use_two_factor = 0;  // OFF
}

static void configure_two_factor(SVPFState* f, const TwoFactorDGPParams& dgp) {
    // Start with baseline
    configure_single_factor(f);
    
    // Enable two-factor
    f->use_two_factor = 1;
    f->rho_fast = dgp.rho_fast;
    f->sigma_fast = dgp.sigma_fast;
    f->rho_slow = dgp.rho_slow;
    f->sigma_slow = dgp.sigma_slow;
    
    // Disable features that compete with two-factor
    f->use_adaptive_mu = 0;      // h_slow tracks long-run level
    f->use_adaptive_sigma = 0;   // We have separate sigma_fast/slow
}

// =============================================================================
// RUN SINGLE TEST
// =============================================================================

struct TestResult {
    float rmse_h;
    float corr_h;
};

static TestResult run_filter(
    SVPFState* state,
    const SVPFParams& params,
    const float* y_data,
    const float* h_true,
    int n_steps,
    int warmup
) {
    std::vector<float> h_est(n_steps);
    
    svpf_initialize(state, &params, 42);
    
    float y_prev = 0.0f;
    for (int t = 0; t < n_steps; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(state, y_data[t], y_prev, &params, &loglik, &vol, &h_mean);
        h_est[t] = h_mean;
        y_prev = y_data[t];
    }
    
    TestResult res;
    res.rmse_h = compute_rmse(h_est.data(), h_true, n_steps, warmup);
    res.corr_h = compute_correlation(h_est.data(), h_true, n_steps, warmup);
    return res;
}

// =============================================================================
// RUN COMPARISON
// =============================================================================

struct ComparisonResult {
    float rmse_1f_mean, rmse_1f_std;
    float rmse_2f_mean, rmse_2f_std;
    float corr_1f_mean, corr_2f_mean;
    float improvement_pct;
};

static ComparisonResult run_comparison(
    const TwoFactorDGPParams& dgp,
    int n_seeds,
    int n_steps,
    int warmup
) {
    std::vector<float> rmse_1f(n_seeds);
    std::vector<float> rmse_2f(n_seeds);
    std::vector<float> corr_1f(n_seeds);
    std::vector<float> corr_2f(n_seeds);
    
    // Allocate data buffers
    std::vector<float> y_data(n_steps);
    std::vector<float> h_true(n_steps);
    std::vector<float> h_fast(n_steps);
    std::vector<float> h_slow(n_steps);
    
    // Create filters once
    SVPFState* state_1f = svpf_create(N_PARTICLES, N_STEIN_STEPS, dgp.nu, nullptr);
    SVPFState* state_2f = svpf_create(N_PARTICLES, N_STEIN_STEPS, dgp.nu, nullptr);
    
    // Setup params (single-factor uses combined rho/sigma)
    SVPFParams params_1f;
    params_1f.mu = dgp.mu;
    params_1f.gamma = 0.0f;
    // Approximate combined dynamics: use average persistence, combined vol
    float var_fast = dgp.sigma_fast * dgp.sigma_fast / (1.0f - dgp.rho_fast * dgp.rho_fast);
    float var_slow = dgp.sigma_slow * dgp.sigma_slow / (1.0f - dgp.rho_slow * dgp.rho_slow);
    float var_total = var_fast + var_slow;
    // Weighted average rho
    params_1f.rho = (dgp.rho_fast * var_fast + dgp.rho_slow * var_slow) / var_total;
    // Combined sigma (approximate)
    params_1f.sigma_z = sqrtf(dgp.sigma_fast * dgp.sigma_fast + dgp.sigma_slow * dgp.sigma_slow);
    
    SVPFParams params_2f;
    params_2f.mu = dgp.mu;
    params_2f.rho = dgp.rho_slow;  // Not really used in 2F mode
    params_2f.sigma_z = dgp.sigma_slow;  // Not really used in 2F mode
    params_2f.gamma = 0.0f;
    
    for (int seed = 0; seed < n_seeds; seed++) {
        // Generate data
        generate_two_factor_data(dgp, n_steps, y_data.data(), h_true.data(),
                                 h_fast.data(), h_slow.data(), seed * 12345);
        
        // Configure and run single-factor
        configure_single_factor(state_1f);
        TestResult res_1f = run_filter(state_1f, params_1f, y_data.data(), 
                                        h_true.data(), n_steps, warmup);
        rmse_1f[seed] = res_1f.rmse_h;
        corr_1f[seed] = res_1f.corr_h;
        
        // Configure and run two-factor
        configure_two_factor(state_2f, dgp);
        TestResult res_2f = run_filter(state_2f, params_2f, y_data.data(),
                                        h_true.data(), n_steps, warmup);
        rmse_2f[seed] = res_2f.rmse_h;
        corr_2f[seed] = res_2f.corr_h;
        
        printf("  Seed %2d: 1F RMSE=%.4f, 2F RMSE=%.4f, diff=%+.4f\n",
               seed, res_1f.rmse_h, res_2f.rmse_h, res_1f.rmse_h - res_2f.rmse_h);
    }
    
    svpf_destroy(state_1f);
    svpf_destroy(state_2f);
    
    // Compute statistics
    ComparisonResult result;
    
    float sum_1f = 0, sum_2f = 0, sum_c1f = 0, sum_c2f = 0;
    for (int i = 0; i < n_seeds; i++) {
        sum_1f += rmse_1f[i];
        sum_2f += rmse_2f[i];
        sum_c1f += corr_1f[i];
        sum_c2f += corr_2f[i];
    }
    result.rmse_1f_mean = sum_1f / n_seeds;
    result.rmse_2f_mean = sum_2f / n_seeds;
    result.corr_1f_mean = sum_c1f / n_seeds;
    result.corr_2f_mean = sum_c2f / n_seeds;
    
    float var_1f = 0, var_2f = 0;
    for (int i = 0; i < n_seeds; i++) {
        var_1f += (rmse_1f[i] - result.rmse_1f_mean) * (rmse_1f[i] - result.rmse_1f_mean);
        var_2f += (rmse_2f[i] - result.rmse_2f_mean) * (rmse_2f[i] - result.rmse_2f_mean);
    }
    result.rmse_1f_std = sqrtf(var_1f / (n_seeds - 1));
    result.rmse_2f_std = sqrtf(var_2f / (n_seeds - 1));
    
    result.improvement_pct = 100.0f * (result.rmse_1f_mean - result.rmse_2f_mean) / result.rmse_1f_mean;
    
    return result;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║           Two-Factor SVPF Test (Correctly-Specified DGP)          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  Configuration:\n");
    printf("    Seeds:     %d\n", N_SEEDS);
    printf("    Steps:     %d\n", N_STEPS);
    printf("    Warmup:    %d\n", WARMUP_STEPS);
    printf("    Particles: %d\n", N_PARTICLES);
    printf("\n");
    printf("  DGP: h_t = mu + h_fast,t + h_slow,t\n");
    printf("       h_fast,t = rho_fast * h_fast,t-1 + sigma_fast * eps\n");
    printf("       h_slow,t = rho_slow * h_slow,t-1 + sigma_slow * eps\n");
    printf("\n");
    
    // Scenarios
    TwoFactorDGPParams scenarios[] = {
        dgp_balanced(),
        dgp_fast_dominant(),
        dgp_slow_dominant(),
        dgp_crisis()
    };
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Running %d scenarios × %d seeds = %d comparisons\n", 
           n_scenarios, N_SEEDS, n_scenarios * N_SEEDS);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Results table
    printf("┌────────────────┬────────────┬────────────┬────────────┬──────────┐\n");
    printf("│ Scenario       │ 1F RMSE    │ 2F RMSE    │ Improv %%   │ Winner   │\n");
    printf("├────────────────┼────────────┼────────────┼────────────┼──────────┤\n");
    
    int wins_1f = 0, wins_2f = 0;
    
    for (int s = 0; s < n_scenarios; s++) {
        const TwoFactorDGPParams& dgp = scenarios[s];
        
        printf("\nScenario: %s\n", dgp.name);
        printf("  rho_fast=%.2f, sigma_fast=%.2f\n", dgp.rho_fast, dgp.sigma_fast);
        printf("  rho_slow=%.2f, sigma_slow=%.2f\n", dgp.rho_slow, dgp.sigma_slow);
        printf("\n");
        
        ComparisonResult result = run_comparison(dgp, N_SEEDS, N_STEPS, WARMUP_STEPS);
        
        const char* winner;
        if (result.improvement_pct > 2.0f) {
            winner = "2F ✓";
            wins_2f++;
        } else if (result.improvement_pct < -2.0f) {
            winner = "1F ✓";
            wins_1f++;
        } else {
            winner = "Tie";
        }
        
        printf("│ %-14s │ %5.3f±%.3f │ %5.3f±%.3f │ %+7.1f%%   │ %-8s │\n",
               dgp.name,
               result.rmse_1f_mean, result.rmse_1f_std,
               result.rmse_2f_mean, result.rmse_2f_std,
               result.improvement_pct,
               winner);
    }
    
    printf("└────────────────┴────────────┴────────────┴────────────┴──────────┘\n\n");
    
    // Verdict
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  VERDICT: 1F wins %d, 2F wins %d\n", wins_1f, wins_2f);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    if (wins_2f > wins_1f) {
        printf("  ✓ Two-factor filter correctly outperforms on two-factor DGP\n");
    } else if (wins_1f > wins_2f) {
        printf("  ✗ Single-factor still wins - check two-factor implementation\n");
    } else {
        printf("  ~ No clear winner\n");
    }
    
    printf("\n");
    return 0;
}
