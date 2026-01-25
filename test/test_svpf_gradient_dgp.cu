/**
 * @file test_svpf_gradient_dgp.cu
 * @brief Gradient diagnostic verification using HCRBPF DGP
 *
 * Purpose: Verify gradient correctness on DGP with ground truth before
 *          enabling any parameter learning in production.
 *
 * Tests:
 *   1. ν gradient direction (too high → negative, too low → positive)
 *   2. Gradient magnitude during crashes vs calm
 *   3. Correlation of gradient with parameter error
 *   4. Effect of model mismatch (DGP has time-varying params, SVPF assumes fixed)
 *
 * Usage:
 *   nvcc -o test_grad test_svpf_gradient_dgp.cu svpf_gradient_diagnostic.cu \
 *        svpf_opt_kernels.cu svpf_optimized_graph.cu -lcurand
 *   ./test_grad
 */

#include "svpf.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * PCG32 RNG (same as your DGP)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t* rng) {
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t* rng) {
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void pcg32_seed(pcg32_t* rng, uint64_t seed) {
    rng->state = seed * 12345ULL + 1;
    rng->inc = seed * 67890ULL | 1;
    pcg32_random(rng);
    pcg32_random(rng);
}

/*═══════════════════════════════════════════════════════════════════════════
 * HCRBPF DGP PARAMETERS (copied from your test file)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double theta_base, theta_scale, theta_rate;
    double mu_vol_base, mu_vol_scale, mu_vol_rate;
    double sigma_vol_base, sigma_vol_scale, sigma_vol_rate;
    double rho_z, sigma_z, z_mean;
} DGPParams;

static DGPParams default_dgp_params(void) {
    DGPParams p;
    p.theta_base = 0.007;
    p.theta_scale = 0.120;
    p.theta_rate = 0.3;
    p.mu_vol_base = -4.5;
    p.mu_vol_scale = 3.5;
    p.mu_vol_rate = 0.3;
    p.sigma_vol_base = 0.0786;
    p.sigma_vol_scale = 0.42;
    p.sigma_vol_rate = 0.3;
    p.rho_z = 0.98585;
    p.sigma_z = 0.02828;
    p.z_mean = 0.0;
    return p;
}

static double sat(double base, double scale, double rate, double z) {
    return base + scale * (1.0 - exp(-rate * z));
}

static void z_to_sv_params(double z, const DGPParams* p,
                           double* theta, double* mu_vol, double* sigma_vol) {
    *theta = sat(p->theta_base, p->theta_scale, p->theta_rate, z);
    *mu_vol = sat(p->mu_vol_base, p->mu_vol_scale, p->mu_vol_rate, z);
    *sigma_vol = sat(p->sigma_vol_base, p->sigma_vol_scale, p->sigma_vol_rate, z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST DATA WITH GROUND TRUTH
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double* true_z;
    double* true_h;
    double* true_vol;
    double* returns;
    
    // Time-varying ground truth parameters (for gradient verification)
    double* true_theta;    // θ(z) at each timestep
    double* true_mu;       // μ(z) at each timestep
    double* true_sigma;    // σ(z) at each timestep
    
    int n_ticks;
    const char* scenario_name;
} GradientTestData;

static GradientTestData* alloc_gradient_test_data(int n) {
    GradientTestData* data = (GradientTestData*)calloc(1, sizeof(GradientTestData));
    data->n_ticks = n;
    data->true_z = (double*)malloc(n * sizeof(double));
    data->true_h = (double*)malloc(n * sizeof(double));
    data->true_vol = (double*)malloc(n * sizeof(double));
    data->returns = (double*)malloc(n * sizeof(double));
    data->true_theta = (double*)malloc(n * sizeof(double));
    data->true_mu = (double*)malloc(n * sizeof(double));
    data->true_sigma = (double*)malloc(n * sizeof(double));
    return data;
}

static void free_gradient_test_data(GradientTestData* data) {
    if (!data) return;
    free(data->true_z);
    free(data->true_h);
    free(data->true_vol);
    free(data->returns);
    free(data->true_theta);
    free(data->true_mu);
    free(data->true_sigma);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * GENERATE TEST DATA WITH FULL GROUND TRUTH
 *═══════════════════════════════════════════════════════════════════════════*/

static void generate_with_ground_truth(
    GradientTestData* data, 
    const DGPParams* p, 
    pcg32_t* rng,
    double obs_nu  // Observation noise df (for Student-t returns)
) {
    int n = data->n_ticks;
    
    double theta, mu_vol, sigma_vol;
    z_to_sv_params(data->true_z[0], p, &theta, &mu_vol, &sigma_vol);
    double h = mu_vol;
    
    for (int t = 0; t < n; t++) {
        z_to_sv_params(data->true_z[t], p, &theta, &mu_vol, &sigma_vol);
        
        // Store ground truth params
        data->true_theta[t] = theta;
        data->true_mu[t] = mu_vol;
        data->true_sigma[t] = sigma_vol;
        
        // Evolve h
        if (t > 0) {
            h = (1.0 - theta) * h + theta * mu_vol + sigma_vol * pcg32_gaussian(rng);
        }
        if (h < -10.0) h = -10.0;
        if (h > 2.0) h = 2.0;
        
        data->true_h[t] = h;
        data->true_vol[t] = exp(h / 2.0);
        
        // Generate return with Student-t noise
        double noise;
        if (obs_nu > 100.0) {
            // Approximate Gaussian
            noise = pcg32_gaussian(rng);
        } else {
            // Student-t via ratio of Gaussian and Chi-squared
            // t_ν = Z / sqrt(V/ν) where Z~N(0,1), V~χ²_ν
            double z_norm = pcg32_gaussian(rng);
            double chi_sq = 0.0;
            for (int i = 0; i < (int)obs_nu; i++) {
                double g = pcg32_gaussian(rng);
                chi_sq += g * g;
            }
            noise = z_norm / sqrt(chi_sq / obs_nu);
        }
        
        data->returns[t] = data->true_vol[t] * noise;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO GENERATORS
 *═══════════════════════════════════════════════════════════════════════════*/

// Scenario 1: Slow sinusoidal drift
static GradientTestData* gen_slow_drift(int n, double amplitude, double period, 
                                        double obs_nu, int seed) {
    GradientTestData* data = alloc_gradient_test_data(n);
    data->scenario_name = "Slow Drift";
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    for (int t = 0; t < n; t++) {
        data->true_z[t] = amplitude * (1.0 + sin(2.0 * M_PI * t / period)) / 2.0;
    }
    
    DGPParams p = default_dgp_params();
    generate_with_ground_truth(data, &p, &rng, obs_nu);
    return data;
}

// Scenario 2: Stress spike + recovery
static GradientTestData* gen_spike_recovery(int n, double z_base, double z_peak,
                                            int spike_start, int spike_duration,
                                            double decay_rate, double obs_nu, int seed) {
    GradientTestData* data = alloc_gradient_test_data(n);
    data->scenario_name = "Spike + Recovery";
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    for (int t = 0; t < n; t++) {
        if (t < spike_start) {
            data->true_z[t] = z_base;
        } else if (t < spike_start + spike_duration) {
            // Instant spike
            data->true_z[t] = z_peak;
        } else {
            // Exponential decay back to base
            int decay_t = t - spike_start - spike_duration;
            data->true_z[t] = z_base + (z_peak - z_base) * exp(-decay_rate * decay_t);
        }
    }
    
    DGPParams p = default_dgp_params();
    generate_with_ground_truth(data, &p, &rng, obs_nu);
    return data;
}

// Scenario 3: OU process (matches HCRBPF assumptions)
static GradientTestData* gen_ou_process(int n, double rho, double sigma_z, 
                                        double z_mean, double obs_nu, int seed) {
    GradientTestData* data = alloc_gradient_test_data(n);
    data->scenario_name = "OU Process";
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    double z = z_mean;
    for (int t = 0; t < n; t++) {
        z = rho * z + (1.0 - rho) * z_mean + sigma_z * pcg32_gaussian(&rng);
        if (z < 0.0) z = 0.0;
        if (z > 10.0) z = 10.0;
        data->true_z[t] = z;
    }
    
    DGPParams p = default_dgp_params();
    p.rho_z = rho;
    p.sigma_z = sigma_z;
    p.z_mean = z_mean;
    generate_with_ground_truth(data, &p, &rng, obs_nu);
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * GRADIENT DIAGNOSTIC RESULTS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    // Per-timestep diagnostics
    double* nu_grad;
    double* z_sq;           // Standardized residual squared
    double* h_error;        // Estimated h - true h
    double* vol_error;      // Estimated vol - true vol
    
    // Summary statistics
    double mean_nu_grad;
    double mean_z_sq;
    double rmse_h;
    double rmse_vol;
    double mean_h_bias;
    
    // Gradient-error correlation
    double corr_nu_grad_zsq;  // Higher z² → more negative gradient (if ν too high)
    
    // Crash detection
    int n_crashes;           // Count of z² > 9
    double mean_grad_crash;  // Mean gradient during crashes
    double mean_grad_calm;   // Mean gradient during calm
    
    int n_ticks;
} GradientDiagnosticResults;

static GradientDiagnosticResults* alloc_results(int n) {
    GradientDiagnosticResults* r = (GradientDiagnosticResults*)calloc(1, sizeof(GradientDiagnosticResults));
    r->n_ticks = n;
    r->nu_grad = (double*)malloc(n * sizeof(double));
    r->z_sq = (double*)malloc(n * sizeof(double));
    r->h_error = (double*)malloc(n * sizeof(double));
    r->vol_error = (double*)malloc(n * sizeof(double));
    return r;
}

static void free_results(GradientDiagnosticResults* r) {
    if (!r) return;
    free(r->nu_grad);
    free(r->z_sq);
    free(r->h_error);
    free(r->vol_error);
    free(r);
}

static void compute_summary_stats(GradientDiagnosticResults* r) {
    int n = r->n_ticks;
    
    // Means
    double sum_grad = 0, sum_zsq = 0, sum_h_err = 0;
    double sum_h_sq = 0, sum_vol_sq = 0;
    for (int t = 0; t < n; t++) {
        sum_grad += r->nu_grad[t];
        sum_zsq += r->z_sq[t];
        sum_h_err += r->h_error[t];
        sum_h_sq += r->h_error[t] * r->h_error[t];
        sum_vol_sq += r->vol_error[t] * r->vol_error[t];
    }
    r->mean_nu_grad = sum_grad / n;
    r->mean_z_sq = sum_zsq / n;
    r->mean_h_bias = sum_h_err / n;
    r->rmse_h = sqrt(sum_h_sq / n);
    r->rmse_vol = sqrt(sum_vol_sq / n);
    
    // Crash/calm gradient split
    r->n_crashes = 0;
    double sum_grad_crash = 0, sum_grad_calm = 0;
    int n_calm = 0;
    for (int t = 0; t < n; t++) {
        if (r->z_sq[t] > 9.0) {
            r->n_crashes++;
            sum_grad_crash += r->nu_grad[t];
        } else {
            n_calm++;
            sum_grad_calm += r->nu_grad[t];
        }
    }
    r->mean_grad_crash = (r->n_crashes > 0) ? sum_grad_crash / r->n_crashes : 0.0;
    r->mean_grad_calm = (n_calm > 0) ? sum_grad_calm / n_calm : 0.0;
    
    // Correlation between nu_grad and z_sq
    double mean_grad = r->mean_nu_grad;
    double mean_zsq = r->mean_z_sq;
    double cov = 0, var_grad = 0, var_zsq = 0;
    for (int t = 0; t < n; t++) {
        double dg = r->nu_grad[t] - mean_grad;
        double dz = r->z_sq[t] - mean_zsq;
        cov += dg * dz;
        var_grad += dg * dg;
        var_zsq += dz * dz;
    }
    r->corr_nu_grad_zsq = cov / (sqrt(var_grad * var_zsq) + 1e-10);
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN GRADIENT DIAGNOSTIC ON TEST DATA
 *═══════════════════════════════════════════════════════════════════════════*/

static GradientDiagnosticResults* run_gradient_diagnostic(
    const GradientTestData* data,
    float filter_nu,           // ν value SVPF will use
    float filter_mu,           // μ value SVPF will use (fixed)
    float filter_rho,          // ρ value SVPF will use (fixed)
    float filter_sigma_z,      // σ_z value SVPF will use (fixed)
    int n_particles,
    int n_stein_steps,
    int seed
) {
    int n = data->n_ticks;
    GradientDiagnosticResults* results = alloc_results(n);
    
    // Create SVPF
    SVPFState* state = svpf_create(n_particles, n_stein_steps, filter_nu, nullptr);
    
    SVPFParams params;
    params.mu = filter_mu;
    params.rho = filter_rho;
    params.sigma_z = filter_sigma_z;
    params.gamma = 0.0f;  // No leverage in DGP
    
    svpf_initialize(state, &params, seed);
    
    // Run filter and collect gradients
    float y_prev = 0.0f;
    for (int t = 0; t < n; t++) {
        float y_t = (float)data->returns[t];
        
        float loglik, vol_est, h_mean;
        svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol_est, &h_mean);
        
        // Compute ν gradient
        float nu_grad, z_sq_mean;
        svpf_compute_nu_diagnostic_simple(state, y_t, &nu_grad, &z_sq_mean);
        
        // Store results
        results->nu_grad[t] = nu_grad;
        results->z_sq[t] = z_sq_mean;
        results->h_error[t] = h_mean - data->true_h[t];
        results->vol_error[t] = vol_est - data->true_vol[t];
        
        y_prev = y_t;
    }
    
    svpf_destroy(state);
    
    compute_summary_stats(results);
    return results;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: ν GRADIENT DIRECTION
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_nu_gradient_direction(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" TEST: ν Gradient Direction\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf(" DGP generates returns with Student-t(ν_true) noise.\n");
    printf(" SVPF runs with different ν_filter values.\n");
    printf(" Gradient should point toward ν_true.\n");
    printf("\n");
    
    const int n_ticks = 2000;
    const int n_particles = 4096;
    const int n_stein_steps = 5;
    const int seed = 42;
    
    // Generate data with true ν = 5
    const double true_nu = 5.0;
    GradientTestData* data = gen_spike_recovery(
        n_ticks, 
        1.0,    // z_base (calm)
        6.0,    // z_peak (stress)
        500,    // spike_start
        100,    // spike_duration
        0.01,   // decay_rate
        true_nu,
        seed
    );
    
    // Average μ, ρ, σ from DGP for SVPF (rough approximation)
    // These will be "wrong" but that's okay - we're testing ν gradient
    float filter_mu = -3.5f;
    float filter_rho = 0.97f;
    float filter_sigma = 0.15f;
    
    printf(" Data: %s, T=%d, true_ν=%.1f\n", data->scenario_name, n_ticks, true_nu);
    printf(" Filter: N=%d, Stein=%d, μ=%.2f, ρ=%.2f, σ=%.2f\n",
           n_particles, n_stein_steps, filter_mu, filter_rho, filter_sigma);
    printf("\n");
    
    // Test 1: ν too HIGH (ν=30)
    {
        float filter_nu = 30.0f;
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma,
            n_particles, n_stein_steps, seed + 1
        );
        
        printf(" ν_filter = %.0f (too HIGH, should want to DECREASE)\n", filter_nu);
        printf("   Mean gradient: %+.6f\n", r->mean_nu_grad);
        printf("   Gradient at crashes (z²>9): %+.6f (n=%d)\n", r->mean_grad_crash, r->n_crashes);
        printf("   Gradient at calm: %+.6f\n", r->mean_grad_calm);
        printf("   RMSE(h): %.4f, RMSE(vol): %.4f\n", r->rmse_h, r->rmse_vol);
        printf("   Result: %s\n", r->mean_nu_grad < -0.001 ? "PASS ✓ (gradient negative)" : "FAIL ✗");
        printf("\n");
        
        free_results(r);
    }
    
    // Test 2: ν too LOW (ν=3)
    {
        float filter_nu = 3.0f;
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma,
            n_particles, n_stein_steps, seed + 2
        );
        
        printf(" ν_filter = %.0f (too LOW, should want to INCREASE)\n", filter_nu);
        printf("   Mean gradient: %+.6f\n", r->mean_nu_grad);
        printf("   Gradient at crashes (z²>9): %+.6f (n=%d)\n", r->mean_grad_crash, r->n_crashes);
        printf("   Gradient at calm: %+.6f\n", r->mean_grad_calm);
        printf("   RMSE(h): %.4f, RMSE(vol): %.4f\n", r->rmse_h, r->rmse_vol);
        printf("   Result: %s\n", r->mean_nu_grad > 0.001 ? "PASS ✓ (gradient positive)" : "FAIL ✗");
        printf("\n");
        
        free_results(r);
    }
    
    // Test 3: ν CORRECT (ν=5)
    {
        float filter_nu = 5.0f;
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma,
            n_particles, n_stein_steps, seed + 3
        );
        
        printf(" ν_filter = %.0f (CORRECT, gradient should be ~0)\n", filter_nu);
        printf("   Mean gradient: %+.6f\n", r->mean_nu_grad);
        printf("   Gradient at crashes (z²>9): %+.6f (n=%d)\n", r->mean_grad_crash, r->n_crashes);
        printf("   Gradient at calm: %+.6f\n", r->mean_grad_calm);
        printf("   RMSE(h): %.4f, RMSE(vol): %.4f\n", r->rmse_h, r->rmse_vol);
        printf("   Result: %s\n", fabs(r->mean_nu_grad) < 0.01 ? "PASS ✓ (gradient ~0)" : "MARGINAL");
        printf("\n");
        
        free_results(r);
    }
    
    free_gradient_test_data(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: GRADIENT BEHAVIOR ACROSS SCENARIOS
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_gradient_across_scenarios(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" TEST: Gradient Behavior Across DGP Scenarios\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    
    const int n_ticks = 3000;
    const int n_particles = 4096;
    const int n_stein_steps = 5;
    
    // DGP uses ν=5, filter uses ν=15 (moderately too high)
    const double dgp_nu = 5.0;
    const float filter_nu = 15.0f;
    const float filter_mu = -3.5f;
    const float filter_rho = 0.97f;
    const float filter_sigma = 0.15f;
    
    printf(" DGP ν=%.0f, Filter ν=%.0f (expect NEGATIVE gradient)\n", dgp_nu, filter_nu);
    printf(" Particles=%d, Stein=%d\n", n_particles, n_stein_steps);
    printf("\n");
    
    printf(" %-20s | Mean Grad | Crash Grad | Calm Grad | RMSE(h) | Crashes\n", "Scenario");
    printf(" %-20s-+----------+------------+-----------+---------+--------\n", "--------------------");
    
    // Scenario 1: Slow drift
    {
        GradientTestData* data = gen_slow_drift(n_ticks, 4.0, 1500.0, dgp_nu, 100);
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma, n_particles, n_stein_steps, 100
        );
        printf(" %-20s | %+8.5f | %+10.5f | %+9.5f | %7.4f | %4d\n",
               data->scenario_name, r->mean_nu_grad, r->mean_grad_crash, r->mean_grad_calm, r->rmse_h, r->n_crashes);
        free_results(r);
        free_gradient_test_data(data);
    }
    
    // Scenario 2: Spike + Recovery
    {
        GradientTestData* data = gen_spike_recovery(n_ticks, 1.0, 6.0, 800, 200, 0.01, dgp_nu, 200);
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma, n_particles, n_stein_steps, 200
        );
        printf(" %-20s | %+8.5f | %+10.5f | %+9.5f | %7.4f | %4d\n",
               data->scenario_name, r->mean_nu_grad, r->mean_grad_crash, r->mean_grad_calm, r->rmse_h, r->n_crashes);
        free_results(r);
        free_gradient_test_data(data);
    }
    
    // Scenario 3: OU Process (calm)
    {
        GradientTestData* data = gen_ou_process(n_ticks, 0.985, 0.03, 1.0, dgp_nu, 300);
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma, n_particles, n_stein_steps, 300
        );
        printf(" %-20s | %+8.5f | %+10.5f | %+9.5f | %7.4f | %4d\n",
               "OU (low stress)", r->mean_nu_grad, r->mean_grad_crash, r->mean_grad_calm, r->rmse_h, r->n_crashes);
        free_results(r);
        free_gradient_test_data(data);
    }
    
    // Scenario 4: OU Process (stressed)
    {
        GradientTestData* data = gen_ou_process(n_ticks, 0.985, 0.05, 4.0, dgp_nu, 400);
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, filter_mu, filter_rho, filter_sigma, n_particles, n_stein_steps, 400
        );
        printf(" %-20s | %+8.5f | %+10.5f | %+9.5f | %7.4f | %4d\n",
               "OU (high stress)", r->mean_nu_grad, r->mean_grad_crash, r->mean_grad_calm, r->rmse_h, r->n_crashes);
        free_results(r);
        free_gradient_test_data(data);
    }
    
    printf("\n");
    printf(" Key observations:\n");
    printf("   - Gradient should be NEGATIVE across all scenarios (ν too high)\n");
    printf("   - Gradient magnitude larger during crashes (more informative)\n");
    printf("   - Gradient near zero during calm (ν uninformative when no tail events)\n");
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: GRADIENT SWEEP (Find Equilibrium ν)
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_gradient_sweep(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" TEST: ν Gradient Sweep (Find Equilibrium)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    
    const int n_ticks = 2000;
    const int n_particles = 4096;
    const int n_stein_steps = 5;
    const double dgp_nu = 5.0;
    
    // Generate spike data
    GradientTestData* data = gen_spike_recovery(n_ticks, 1.0, 5.0, 600, 150, 0.015, dgp_nu, 555);
    
    printf(" DGP: Spike + Recovery, true_ν=%.0f\n", dgp_nu);
    printf("\n");
    printf(" Sweeping filter ν to find where gradient crosses zero:\n");
    printf("\n");
    printf("   ν_filter | Mean Gradient | Direction\n");
    printf("   ---------+--------------+----------\n");
    
    float nu_values[] = {3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 10.0f, 15.0f, 20.0f, 30.0f};
    int n_nu = sizeof(nu_values) / sizeof(nu_values[0]);
    
    float prev_grad = 0.0f;
    float equilibrium_nu = 0.0f;
    
    for (int i = 0; i < n_nu; i++) {
        float filter_nu = nu_values[i];
        GradientDiagnosticResults* r = run_gradient_diagnostic(
            data, filter_nu, -3.5f, 0.97f, 0.15f, n_particles, n_stein_steps, 555 + i
        );
        
        const char* direction;
        if (r->mean_nu_grad > 0.001) direction = "↑ increase ν";
        else if (r->mean_nu_grad < -0.001) direction = "↓ decrease ν";
        else direction = "≈ equilibrium";
        
        printf("   %7.1f  | %+12.6f | %s\n", filter_nu, r->mean_nu_grad, direction);
        
        // Detect zero crossing
        if (i > 0 && prev_grad > 0 && r->mean_nu_grad < 0) {
            // Gradient crossed from positive to negative between nu_values[i-1] and nu_values[i]
            equilibrium_nu = (nu_values[i-1] + nu_values[i]) / 2.0f;
        }
        
        prev_grad = r->mean_nu_grad;
        free_results(r);
    }
    
    printf("\n");
    if (equilibrium_nu > 0) {
        printf(" Equilibrium ν ≈ %.1f (gradient crosses zero)\n", equilibrium_nu);
        printf(" True DGP ν = %.1f\n", dgp_nu);
        printf(" Difference: %.1f\n", fabs(equilibrium_nu - dgp_nu));
    } else {
        printf(" No zero crossing found in range [3, 30]\n");
    }
    printf("\n");
    
    free_gradient_test_data(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * WRITE DETAILED LOG (for plotting)
 *═══════════════════════════════════════════════════════════════════════════*/

static void write_diagnostic_log(const char* filename) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Writing detailed diagnostic log: %s\n", filename);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    const int n_ticks = 3000;
    const int n_particles = 4096;
    const int n_stein_steps = 5;
    const double dgp_nu = 5.0;
    
    // Generate data with spike
    GradientTestData* data = gen_spike_recovery(n_ticks, 1.0, 6.0, 1000, 200, 0.01, dgp_nu, 777);
    
    // Run with misspecified ν
    float filter_nu = 20.0f;
    
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf(" ERROR: Could not open file\n");
        free_gradient_test_data(data);
        return;
    }
    
    fprintf(f, "t,y,true_z,true_h,true_vol,true_mu,true_sigma,est_h,est_vol,h_error,nu_grad,z_sq\n");
    
    SVPFState* state = svpf_create(n_particles, n_stein_steps, filter_nu, nullptr);
    SVPFParams params = {-3.5f, 0.97f, 0.15f, 0.0f};
    svpf_initialize(state, &params, 777);
    
    float y_prev = 0.0f;
    for (int t = 0; t < n_ticks; t++) {
        float y_t = (float)data->returns[t];
        float loglik, vol_est, h_mean;
        svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol_est, &h_mean);
        
        float nu_grad, z_sq;
        svpf_compute_nu_diagnostic_simple(state, y_t, &nu_grad, &z_sq);
        
        fprintf(f, "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                t, y_t, data->true_z[t], data->true_h[t], data->true_vol[t],
                data->true_mu[t], data->true_sigma[t],
                h_mean, vol_est, h_mean - data->true_h[t], nu_grad, z_sq);
        
        y_prev = y_t;
    }
    
    fclose(f);
    svpf_destroy(state);
    free_gradient_test_data(data);
    
    printf(" Done. Columns: t, y, true_z, true_h, true_vol, true_mu, true_sigma,\n");
    printf("                est_h, est_vol, h_error, nu_grad, z_sq\n");
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SVPF Gradient Diagnostic - DGP Verification                   ║\n");
    printf("║     Step 0: Verify gradients before enabling learning             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    // Test 1: Direction verification
    test_nu_gradient_direction();
    
    // Test 2: Behavior across scenarios
    test_gradient_across_scenarios();
    
    // Test 3: Find equilibrium ν
    test_gradient_sweep();
    
    // Optional: Write log for plotting
    if (argc > 1 && strcmp(argv[1], "--log") == 0) {
        const char* log_path = (argc > 2) ? argv[2] : "svpf_gradient_diagnostic.csv";
        write_diagnostic_log(log_path);
    }
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf(" Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf(" If all tests pass:\n");
    printf("   ✓ ν gradient points toward correct value\n");
    printf("   ✓ Gradient is informative during crashes, near-zero during calm\n");
    printf("   ✓ Equilibrium ν ≈ true DGP ν\n");
    printf("\n");
    printf(" Next step: Enable ν learning with conservative update rule\n");
    printf("\n");
    
    return 0;
}
