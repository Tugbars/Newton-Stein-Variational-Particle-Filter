/**
 * @file cpmmh_test.cu
 * @brief Test CPMMH Parameter Learning with Simulated SV Data
 * 
 * Workflow:
 * 1. Simulate SV time series with known θ_true = (ρ, μ, σ_z)
 * 2. Run CPMMH to estimate parameters
 * 3. Verify posterior concentrates around true values
 * 
 * Usage:
 *   ./cpmmh_test [T] [n_iter] [seed]
 *   
 * Example:
 *   ./cpmmh_test 500 2000 12345
 */

#include "cpmmh_svpf.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA SIMULATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Simulate SV time series on host.
 * 
 * Model:
 *   h_t = μ + ρ(h_{t-1} - μ) + σ_z·ε_t,  ε_t ~ N(0,1)
 *   y_t = exp(h_t/2) · ξ_t,               ξ_t ~ t(ν)
 */
void simulate_sv_data(
    float* y_out,           /* Output: observations [T] */
    float* h_out,           /* Output: latent log-vol [T] (can be NULL) */
    int T,
    float rho,
    float mu,
    float sigma_z,
    float nu,               /* Student-t df for observations */
    unsigned int seed
) {
    srand(seed);
    
    /* Helper: Box-Muller normal */
    auto randn = []() -> float {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        u1 = fmaxf(u1, 1e-10f);
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    };
    
    /* Helper: Student-t via normal/chi-squared */
    auto randt = [&](float df) -> float {
        /* t = N(0,1) / sqrt(χ²(ν)/ν) */
        float z = randn();
        float chi_sq = 0.0f;
        for (int i = 0; i < (int)df; i++) {
            float n = randn();
            chi_sq += n * n;
        }
        return z / sqrtf(chi_sq / df + 1e-10f);
    };
    
    /* Initialize from stationary distribution */
    float h_var = (sigma_z * sigma_z) / (1.0f - rho * rho + 1e-6f);
    float h_std = sqrtf(h_var);
    float h = mu + h_std * randn();
    
    for (int t = 0; t < T; t++) {
        /* State transition */
        float eps = randn();
        h = mu + rho * (h - mu) + sigma_z * eps;
        
        /* Observation */
        float vol = expf(h / 2.0f);
        float xi = randt(nu);
        y_out[t] = vol * xi;
        
        if (h_out) h_out[t] = h;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CONVERGENCE DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute effective sample size (ESS) using autocorrelation.
 * ESS = N / (1 + 2·Σ_k ρ_k) where ρ_k is lag-k autocorrelation.
 */
float compute_ess(const float* samples, int n) {
    if (n < 10) return (float)n;
    
    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += samples[i];
    mean /= (float)n;
    
    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = samples[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    
    if (var < 1e-10f) return (float)n;
    
    /* Compute autocorrelations until they become negative (Geyer's rule) */
    float sum_rho = 0.0f;
    int max_lag = n / 3;  /* Conservative max lag */
    
    for (int k = 1; k < max_lag; k++) {
        float cov = 0.0f;
        for (int i = 0; i < n - k; i++) {
            cov += (samples[i] - mean) * (samples[i + k] - mean);
        }
        cov /= (float)(n - k);
        float rho_k = cov / var;
        
        /* Geyer's initial monotone sequence estimator */
        if (rho_k < 0.0f) break;
        sum_rho += rho_k;
    }
    
    float ess = (float)n / (1.0f + 2.0f * sum_rho);
    return fmaxf(ess, 1.0f);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    /* Parse arguments */
    int T = (argc > 1) ? atoi(argv[1]) : 500;
    int n_iter = (argc > 2) ? atoi(argv[2]) : 2000;
    unsigned int seed = (argc > 3) ? (unsigned int)atoi(argv[3]) : (unsigned int)time(NULL);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("CPMMH Parameter Learning Test\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("T = %d observations\n", T);
    printf("n_iter = %d MCMC iterations\n", n_iter);
    printf("seed = %u\n", seed);
    printf("\n");
    
    /*─────────────────────────────────────────────────────────────────────────
     * TRUE PARAMETERS (Crypto-calibrated)
     *─────────────────────────────────────────────────────────────────────────*/
    float rho_true = 0.94f;
    float mu_true = -3.2f;       /* ~4% daily vol in exp(mu/2) */
    float sigma_z_true = 0.18f;  /* High vol-of-vol for crypto */
    float nu_obs = 5.0f;         /* Heavy tails */
    
    printf("True Parameters:\n");
    printf("  ρ     = %.4f\n", rho_true);
    printf("  μ     = %.4f (vol ≈ %.1f%%)\n", mu_true, 100.0f * expf(mu_true / 2.0f));
    printf("  σ_z   = %.4f\n", sigma_z_true);
    printf("  ν_obs = %.1f\n", nu_obs);
    printf("\n");
    
    /*─────────────────────────────────────────────────────────────────────────
     * SIMULATE DATA
     *─────────────────────────────────────────────────────────────────────────*/
    printf("Simulating %d observations...\n", T);
    
    float* y = (float*)malloc(T * sizeof(float));
    float* h_true = (float*)malloc(T * sizeof(float));
    
    simulate_sv_data(y, h_true, T, rho_true, mu_true, sigma_z_true, nu_obs, seed);
    
    /* Print data summary */
    float y_mean = 0.0f, y_sq = 0.0f;
    for (int t = 0; t < T; t++) {
        y_mean += y[t];
        y_sq += y[t] * y[t];
    }
    y_mean /= (float)T;
    float y_std = sqrtf(y_sq / (float)T - y_mean * y_mean);
    
    printf("Data summary: mean=%.4f, std=%.4f\n", y_mean, y_std);
    printf("First 10 returns: ");
    for (int t = 0; t < 10 && t < T; t++) printf("%.3f ", y[t]);
    printf("\n\n");
    
    /*─────────────────────────────────────────────────────────────────────────
     * CREATE CPMMH STATE
     *─────────────────────────────────────────────────────────────────────────*/
    printf("Creating CPMMH state...\n");
    
    int N_particles = 256;      /* Inner SVPF particles */
    int T_max = T + 100;        /* Buffer */
    int history_cap = n_iter;   /* Store all samples */
    
    CPMMHState* cpmmh = cpmmh_create(N_particles, T_max, history_cap);
    if (!cpmmh) {
        fprintf(stderr, "Failed to create CPMMH state\n");
        return 1;
    }
    
    /* Set observations */
    cpmmh_set_observations(cpmmh, y, T);
    
    /* Configure replay */
    SVPFReplayConfig replay_cfg = svpf_replay_config_default();
    replay_cfg.nu_obs = nu_obs;
    replay_cfg.n_stein_steps = 8;
    replay_cfg.n_anneal_steps = 3;
    cpmmh_set_replay_config(cpmmh, &replay_cfg);
    
    /* Set CPMMH correlation */
    cpmmh_set_correlation(cpmmh, 0.99f);
    
    /* Initialize chain */
    printf("Initializing from prior...\n\n");
    cpmmh_initialize(cpmmh, NULL, seed + 1);
    
    printf("Initial state:\n");
    printf("  θ = (%.4f, %.4f, %.4f)\n", 
           cpmmh->theta[0], cpmmh->theta[1], cpmmh->theta[2]);
    printf("  log_posterior = %.2f\n\n", cpmmh->log_posterior);
    
    /*─────────────────────────────────────────────────────────────────────────
     * RUN CPMMH
     *─────────────────────────────────────────────────────────────────────────*/
    printf("Running %d CPMMH iterations...\n", n_iter);
    printf("(Progress every 100 iterations)\n\n");
    
    clock_t start_time = clock();
    
    int n_burnin = n_iter / 4;   /* 25% burnin */
    int thin = 1;                /* No thinning */
    
    cpmmh_run(cpmmh, n_iter, n_burnin, thin);
    
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\nCompleted in %.2f seconds (%.1f iter/sec)\n\n", 
           elapsed, n_iter / elapsed);
    
    /*─────────────────────────────────────────────────────────────────────────
     * RESULTS
     *─────────────────────────────────────────────────────────────────────────*/
    cpmmh_print_summary(cpmmh);
    
    /* Extract samples for ESS calculation */
    float* theta_samples = (float*)malloc(cpmmh->history_len * 3 * sizeof(float));
    cpmmh_export_history(cpmmh, theta_samples, NULL);
    
    /* Compute ESS for each parameter */
    float* rho_samples = (float*)malloc(cpmmh->history_len * sizeof(float));
    float* mu_samples = (float*)malloc(cpmmh->history_len * sizeof(float));
    float* sigma_samples = (float*)malloc(cpmmh->history_len * sizeof(float));
    
    for (int i = 0; i < cpmmh->history_len; i++) {
        rho_samples[i] = theta_samples[i * 3 + 0];
        mu_samples[i] = theta_samples[i * 3 + 1];
        sigma_samples[i] = theta_samples[i * 3 + 2];
    }
    
    float ess_rho = compute_ess(rho_samples, cpmmh->history_len);
    float ess_mu = compute_ess(mu_samples, cpmmh->history_len);
    float ess_sigma = compute_ess(sigma_samples, cpmmh->history_len);
    
    printf("Effective Sample Sizes:\n");
    printf("  ESS(ρ)     = %.1f\n", ess_rho);
    printf("  ESS(μ)     = %.1f\n", ess_mu);
    printf("  ESS(σ_z)   = %.1f\n", ess_sigma);
    printf("\n");
    
    /* Check recovery */
    float mean[3], std[3];
    cpmmh_get_posterior_mean(cpmmh, mean);
    cpmmh_get_posterior_std(cpmmh, std);
    
    float z_rho = (mean[0] - rho_true) / (std[0] + 1e-6f);
    float z_mu = (mean[1] - mu_true) / (std[1] + 1e-6f);
    float z_sigma = (mean[2] - sigma_z_true) / (std[2] + 1e-6f);
    
    printf("Parameter Recovery (z-scores, |z| < 2 is good):\n");
    printf("  ρ:     true=%.4f, est=%.4f±%.4f, z=%.2f %s\n", 
           rho_true, mean[0], std[0], z_rho, 
           (fabsf(z_rho) < 2.0f) ? "✓" : "✗");
    printf("  μ:     true=%.4f, est=%.4f±%.4f, z=%.2f %s\n", 
           mu_true, mean[1], std[1], z_mu,
           (fabsf(z_mu) < 2.0f) ? "✓" : "✗");
    printf("  σ_z:   true=%.4f, est=%.4f±%.4f, z=%.2f %s\n", 
           sigma_z_true, mean[2], std[2], z_sigma,
           (fabsf(z_sigma) < 2.0f) ? "✓" : "✗");
    printf("\n");
    
    /* Overall assessment */
    int pass = (fabsf(z_rho) < 2.5f) && (fabsf(z_mu) < 2.5f) && (fabsf(z_sigma) < 2.5f);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("TEST %s\n", pass ? "PASSED ✓" : "FAILED ✗");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /*─────────────────────────────────────────────────────────────────────────
     * CLEANUP
     *─────────────────────────────────────────────────────────────────────────*/
    free(y);
    free(h_true);
    free(theta_samples);
    free(rho_samples);
    free(mu_samples);
    free(sigma_samples);
    cpmmh_destroy(cpmmh);
    
    return pass ? 0 : 1;
}
