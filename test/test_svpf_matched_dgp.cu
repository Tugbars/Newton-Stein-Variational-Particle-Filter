/**
 * @file test_svpf_matched_dgp.cu
 * @brief Perfectly OU-matched DGP test suite for SVPF bias analysis
 *
 * All scenarios generate data from the EXACT SVPF model:
 *   State:       h_t = mu + rho*(h_{t-1} - mu) + sigma_z * eps_t
 *   Observation: y_t = exp(h_t / 2) * eta_t
 *
 * Zero model mismatch — any remaining bias is purely from Stein transport.
 *
 * Build: nvcc -O3 test_svpf_matched_dgp.cu svpf_opt_kernels.cu svpf_optimized_graph.cu -o test_matched -lcurand
 */

#include "svpf.cuh"
#include "svpf_kernels.cuh"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// =============================================================================
// Timing
// =============================================================================

static double get_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1e6 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
#endif
}

// =============================================================================
// PCG32 RNG (self-contained, matches existing test infra)
// =============================================================================

typedef struct { uint64_t state; uint64_t inc; } pcg32_dgp_t;

static inline uint32_t pcg32_dgp_random(pcg32_dgp_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void pcg32_dgp_seed(pcg32_dgp_t* rng, uint64_t seed) {
    rng->state = 0;
    rng->inc = (seed * 12345ULL) | 1;
    pcg32_dgp_random(rng);
    rng->state += seed * 67890ULL;
    pcg32_dgp_random(rng);
    pcg32_dgp_random(rng);
}

static inline double pcg32_dgp_double(pcg32_dgp_t* rng) {
    return (pcg32_dgp_random(rng) >> 11) * (1.0 / 2097152.0);
}

static inline double pcg32_dgp_gaussian(pcg32_dgp_t* rng) {
    double u1, u2;
    do { u1 = pcg32_dgp_double(rng); } while (u1 < 1e-10);
    u2 = pcg32_dgp_double(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static inline double pcg32_dgp_student_t(pcg32_dgp_t* rng, double nu) {
    if (nu > 100.0) return pcg32_dgp_gaussian(rng);
    double z = pcg32_dgp_gaussian(rng);
    double chi_sq = 0.0;
    int inu = (int)nu;
    for (int i = 0; i < inu; i++) {
        double g = pcg32_dgp_gaussian(rng);
        chi_sq += g * g;
    }
    return z / sqrt(chi_sq / nu);
}

// =============================================================================
// Test Data
// =============================================================================

typedef struct {
    double* true_h;
    double* true_vol;
    double* returns;
    int n_ticks;
    int scenario_id;
    const char* scenario_name;
    const char* scenario_desc;
    double dgp_rho;
    double dgp_sigma_z;
    double dgp_mu;
    double dgp_nu_state;  // 0 = Gaussian
    double dgp_nu_obs;    // 0 = Gaussian
} MatchedTestData;

static MatchedTestData* alloc_matched_data(int n) {
    MatchedTestData* d = (MatchedTestData*)calloc(1, sizeof(MatchedTestData));
    d->n_ticks = n;
    d->true_h = (double*)malloc(n * sizeof(double));
    d->true_vol = (double*)malloc(n * sizeof(double));
    d->returns = (double*)malloc(n * sizeof(double));
    return d;
}

static void free_matched_data(MatchedTestData* d) {
    if (!d) return;
    free(d->true_h);
    free(d->true_vol);
    free(d->returns);
    free(d);
}

// =============================================================================
// Core DGP: h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps_t
//           y_t = exp(h_t/2) * eta_t
// =============================================================================

static void generate_matched_series(
    MatchedTestData* data,
    double rho, double sigma_z, double mu,
    double nu_state, double nu_obs,
    pcg32_dgp_t* rng
) {
    int n = data->n_ticks;
    data->dgp_rho = rho;
    data->dgp_sigma_z = sigma_z;
    data->dgp_mu = mu;
    data->dgp_nu_state = nu_state;
    data->dgp_nu_obs = nu_obs;
    
    double var_stat = (sigma_z * sigma_z) / fmax(1.0 - rho * rho, 1e-6);
    double h = mu + sqrt(var_stat) * pcg32_dgp_gaussian(rng);
    
    for (int t = 0; t < n; t++) {
        if (t > 0) {
            double eps = (nu_state > 0.0) 
                ? pcg32_dgp_student_t(rng, nu_state)
                : pcg32_dgp_gaussian(rng);
            h = mu + rho * (h - mu) + sigma_z * eps;
        }
        if (h < -12.0) h = -12.0;
        if (h > 4.0) h = 4.0;
        
        data->true_h[t] = h;
        data->true_vol[t] = exp(h / 2.0);
        
        double eta = (nu_obs > 0.0)
            ? pcg32_dgp_student_t(rng, nu_obs)
            : pcg32_dgp_gaussian(rng);
        data->returns[t] = data->true_vol[t] * eta;
    }
}

// =============================================================================
// Scenarios
// =============================================================================

static MatchedTestData* gen_matched_baseline(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 1;
    d->scenario_name = "Baseline OU (Gauss state)";
    d->scenario_desc = "rho=0.97, sigma_z=0.15, mu=-4.5, Gauss state, t(5) obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -4.5, 0.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_student_t(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 2;
    d->scenario_name = "Student-t Matched";
    d->scenario_desc = "rho=0.97, sigma_z=0.15, mu=-4.5, t(5) state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_high_persist(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 3;
    d->scenario_name = "High Persistence";
    d->scenario_desc = "rho=0.995, sigma_z=0.15, mu=-4.5, t(5) state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.995, 0.15, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_high_volvol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 4;
    d->scenario_name = "High Vol-of-Vol";
    d->scenario_desc = "rho=0.97, sigma_z=0.30, mu=-4.5, t(5) state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.30, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_low_vol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 5;
    d->scenario_name = "Low Vol Regime";
    d->scenario_desc = "rho=0.97, sigma_z=0.15, mu=-6.0, t(5) state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -6.0, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_high_vol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 6;
    d->scenario_name = "High Vol Regime";
    d->scenario_desc = "rho=0.97, sigma_z=0.15, mu=-2.0, t(5) state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -2.0, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_fast_revert(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 7;
    d->scenario_name = "Fast Reversion";
    d->scenario_desc = "rho=0.90, sigma_z=0.15, mu=-4.5, t(5) state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.90, 0.15, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_matched_gaussian(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id = 8;
    d->scenario_name = "Pure Gaussian";
    d->scenario_desc = "rho=0.97, sigma_z=0.15, mu=-4.5, Gauss state+obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -4.5, 0.0, 0.0, &rng);
    return d;
}

// =============================================================================
// Metrics (skip first 100 ticks for warmup)
// =============================================================================

typedef struct {
    double logvol_rmse;
    double logvol_mae;
    double logvol_bias;
} MatchedMetrics;

static MatchedMetrics compute_matched_metrics(const MatchedTestData* data,
                                               const float* est_h) {
    MatchedMetrics m = {0};
    int n = data->n_ticks;
    int skip = 100;
    int count = n - skip;
    
    double sum_sq = 0.0, sum_abs = 0.0, sum_bias = 0.0;
    for (int t = skip; t < n; t++) {
        double err = (double)est_h[t] - data->true_h[t];
        sum_sq += err * err;
        sum_abs += fabs(err);
        sum_bias += err;
    }
    m.logvol_rmse = sqrt(sum_sq / count);
    m.logvol_mae = sum_abs / count;
    m.logvol_bias = sum_bias / count;
    return m;
}

// =============================================================================
// Run SVPF on Matched Scenario
// =============================================================================

static MatchedMetrics run_matched_scenario(
    MatchedTestData* data,
    int n_particles,
    int n_stein,
    int seed,
    double* elapsed_ms_out
) {
    int n = data->n_ticks;
    
    float* h_returns = (float*)malloc(n * sizeof(float));
    float* h_loglik  = (float*)malloc(n * sizeof(float));
    float* h_vol     = (float*)malloc(n * sizeof(float));
    float* h_logvol  = (float*)malloc(n * sizeof(float));
    
    for (int t = 0; t < n; t++)
        h_returns[t] = (float)data->returns[t];
    
    // Create filter with EXACTLY matching DGP parameters
    // Use DGP's observation nu; if Gaussian (0), use very high nu to approximate
    float filter_nu_obs = (data->dgp_nu_obs > 0.0) 
        ? (float)data->dgp_nu_obs 
        : 1000.0f;
    SVPFState* filter = svpf_create(n_particles, n_stein, filter_nu_obs, NULL);
    
    SVPFParams params;
    params.rho     = (float)data->dgp_rho;
    params.sigma_z = (float)data->dgp_sigma_z;
    params.mu      = (float)data->dgp_mu;
    params.gamma   = 0.0f;
    
    // Match state dynamics exactly
    if (data->dgp_nu_state > 0.0) {
        filter->use_student_t_state = 1;
        filter->nu_state = (float)data->dgp_nu_state;
    } else {
        filter->use_student_t_state = 0;
    }
    
    // Set adaptive mu to true value so it doesn't drift
    filter->mu_state = (float)data->dgp_mu;
    
    svpf_initialize(filter, &params, seed);
    
    double t_start = get_time_us();
    float y_prev = 0.0f;
    for (int t = 0; t < n; t++) {
        float y_t = h_returns[t];
        svpf_step_graph(filter, y_t, y_prev, &params,
                       &h_loglik[t], &h_vol[t], &h_logvol[t]);
        y_prev = y_t;
    }
    double t_end = get_time_us();
    *elapsed_ms_out = (t_end - t_start) / 1000.0;
    
    MatchedMetrics m = compute_matched_metrics(data, h_logvol);
    
    svpf_destroy(filter);
    free(h_returns);
    free(h_loglik);
    free(h_vol);
    free(h_logvol);
    return m;
}

// =============================================================================
// Gold Standard: Kalman Steady-State RMSE (Analytical Lower Bound)
// =============================================================================
//
// Linearize: y* = log(y²) = h + log(eta²)
// Gaussian eta: var(log(eta²)) = π²/2 ≈ 4.9348
// Student-t(nu): var(log(eta²)) ≈ π²/2 + 2*trigamma(nu/2)
// Steady-state Riccati: a*P² + P*(Q + R*(1-a)) - Q*R = 0

static double kalman_steady_state_rmse(double rho, double sigma_z, double nu_obs) {
    double Q = sigma_z * sigma_z;
    double a = rho * rho;
    
    double R;
    if (nu_obs <= 0 || nu_obs > 100.0) {
        R = 4.9348;
    } else {
        double trigamma_half_nu;
        if      (nu_obs < 3.5) trigamma_half_nu = 1.50;
        else if (nu_obs < 4.5) trigamma_half_nu = 0.80;
        else if (nu_obs < 5.5) trigamma_half_nu = 0.49;
        else if (nu_obs < 7.5) trigamma_half_nu = 0.30;
        else                   trigamma_half_nu = 0.15;
        R = 4.9348 + 2.0 * trigamma_half_nu;
    }
    
    double b = Q + R * (1.0 - a);
    double c = -Q * R;
    double disc = b * b - 4.0 * a * c;
    double P = (-b + sqrt(disc)) / (2.0 * a);
    return sqrt(P);
}

// =============================================================================
// Gold Standard: Bootstrap SIR Particle Filter (50K particles)
// =============================================================================

static double log_student_t_pdf(double x, double nu) {
    return lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)
         - 0.5 * log(nu * 3.14159265358979323846)
         - (nu + 1.0) / 2.0 * log(1.0 + x * x / nu);
}

static double log_gaussian_pdf(double x) {
    return -0.5 * log(2.0 * 3.14159265358979323846) - 0.5 * x * x;
}

static void systematic_resample(double* particles, double* weights, 
                                double* tmp, int n, pcg32_dgp_t* rng) {
    double u = pcg32_dgp_double(rng) / n;
    double cumsum = weights[0];
    int j = 0;
    for (int i = 0; i < n; i++) {
        double target = u + (double)i / n;
        while (cumsum < target && j < n - 1) {
            j++;
            cumsum += weights[j];
        }
        tmp[i] = particles[j];
    }
    memcpy(particles, tmp, n * sizeof(double));
}

static double bpf_run_rmse(const MatchedTestData* data, int n_pf, int seed) {
    int n = data->n_ticks;
    double rho = data->dgp_rho;
    double sigma_z = data->dgp_sigma_z;
    double mu = data->dgp_mu;
    double nu_state = data->dgp_nu_state;
    double nu_obs = data->dgp_nu_obs;
    
    pcg32_dgp_t rng;
    pcg32_dgp_seed(&rng, seed);
    
    double* particles = (double*)malloc(n_pf * sizeof(double));
    double* weights   = (double*)malloc(n_pf * sizeof(double));
    double* tmp       = (double*)malloc(n_pf * sizeof(double));
    
    double var_stat = (sigma_z * sigma_z) / fmax(1.0 - rho * rho, 1e-6);
    for (int i = 0; i < n_pf; i++)
        particles[i] = mu + sqrt(var_stat) * pcg32_dgp_gaussian(&rng);
    
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;
    
    for (int t = 0; t < n; t++) {
        double y_t = data->returns[t];
        
        // Predict
        if (t > 0) {
            for (int i = 0; i < n_pf; i++) {
                double eps = (nu_state > 0.0)
                    ? pcg32_dgp_student_t(&rng, nu_state)
                    : pcg32_dgp_gaussian(&rng);
                particles[i] = mu + rho * (particles[i] - mu) + sigma_z * eps;
            }
        }
        
        // Weight: log p(y|h) = log p_eta(y*exp(-h/2)) - h/2
        double max_lw = -1e30;
        for (int i = 0; i < n_pf; i++) {
            double h_i = particles[i];
            double eta = y_t * exp(-h_i / 2.0);
            double lw = (nu_obs > 0.0)
                ? log_student_t_pdf(eta, nu_obs) - h_i / 2.0
                : log_gaussian_pdf(eta) - h_i / 2.0;
            weights[i] = lw;
            if (lw > max_lw) max_lw = lw;
        }
        
        double sum_w = 0.0;
        for (int i = 0; i < n_pf; i++) {
            weights[i] = exp(weights[i] - max_lw);
            sum_w += weights[i];
        }
        double inv_w = 1.0 / sum_w;
        for (int i = 0; i < n_pf; i++)
            weights[i] *= inv_w;
        
        // Estimate
        double h_est = 0.0;
        for (int i = 0; i < n_pf; i++)
            h_est += weights[i] * particles[i];
        
        if (t >= skip) {
            double err = h_est - data->true_h[t];
            sum_sq += err * err;
            count++;
        }
        
        // Resample
        systematic_resample(particles, weights, tmp, n_pf, &rng);
    }
    
    free(particles);
    free(weights);
    free(tmp);
    return sqrt(sum_sq / count);
}

// =============================================================================
// Classical Baselines: EWMA + GARCH(1,1)
// =============================================================================

// EWMA: sigma²_t = lambda * sigma²_{t-1} + (1-lambda) * y²_{t-1}
// Returns RMSE of log-vol estimate h_hat = log(sigma²_t) vs true h
static double ewma_rmse(const MatchedTestData* data, double lambda) {
    int n = data->n_ticks;
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;
    
    // Initialize variance at sample variance of first 20 returns
    double init_var = 0.0;
    int init_n = (n < 20) ? n : 20;
    for (int t = 0; t < init_n; t++)
        init_var += data->returns[t] * data->returns[t];
    init_var /= init_n;
    if (init_var < 1e-10) init_var = 1e-10;
    
    double var = init_var;
    for (int t = 0; t < n; t++) {
        if (t >= skip) {
            double h_hat = log(var);
            double err = h_hat - data->true_h[t];
            sum_sq += err * err;
            count++;
        }
        double y2 = data->returns[t] * data->returns[t];
        var = lambda * var + (1.0 - lambda) * y2;
        if (var < 1e-10) var = 1e-10;
    }
    return sqrt(sum_sq / count);
}

// GARCH(1,1): sigma²_t = omega + alpha * y²_{t-1} + beta * sigma²_{t-1}
// Fit via moment-matching: target unconditional var, then split alpha/beta
static double garch_rmse(const MatchedTestData* data, 
                         double omega, double alpha, double beta) {
    int n = data->n_ticks;
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;
    
    // Initialize at unconditional variance
    double persist = alpha + beta;
    double var = (persist < 0.999) ? omega / (1.0 - persist) : 0.01;
    if (var < 1e-10) var = 1e-10;
    
    for (int t = 0; t < n; t++) {
        if (t >= skip) {
            double h_hat = log(var);
            double err = h_hat - data->true_h[t];
            sum_sq += err * err;
            count++;
        }
        double y2 = data->returns[t] * data->returns[t];
        var = omega + alpha * y2 + beta * var;
        if (var < 1e-10) var = 1e-10;
    }
    return sqrt(sum_sq / count);
}

// Quick grid search for best GARCH params on this data
static double garch_best_rmse(const MatchedTestData* data) {
    double best = 1e10;
    // Grid over reasonable GARCH(1,1) params
    double alphas[] = {0.02, 0.05, 0.08, 0.10, 0.15, 0.20};
    double betas[]  = {0.75, 0.80, 0.85, 0.90, 0.93, 0.95};
    int na = sizeof(alphas) / sizeof(alphas[0]);
    int nb = sizeof(betas) / sizeof(betas[0]);
    
    for (int ia = 0; ia < na; ia++) {
        for (int ib = 0; ib < nb; ib++) {
            double a = alphas[ia], b = betas[ib];
            if (a + b >= 0.999) continue;
            // Estimate omega from sample variance
            double sv = 0.0;
            int n = data->n_ticks;
            for (int t = 0; t < n; t++)
                sv += data->returns[t] * data->returns[t];
            sv /= n;
            double omega = sv * (1.0 - a - b);
            
            double r = garch_rmse(data, omega, a, b);
            if (r < best) best = r;
        }
    }
    return best;
}

// =============================================================================
// Print Gold Standard + Classical Comparison Table
// =============================================================================

static void print_gold_standard(int n_ticks, int base_seed, int n_particles, int n_stein) {
    typedef MatchedTestData* (*GenFn)(int, int);
    struct { GenFn gen; } scenarios[] = {
        { gen_matched_baseline },
        { gen_matched_student_t },
        { gen_matched_high_persist },
        { gen_matched_high_volvol },
        { gen_matched_low_vol },
        { gen_matched_high_vol },
        { gen_matched_fast_revert },
        { gen_matched_gaussian },
    };
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);
    int bpf_n = 50000;
    
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  FULL COMPARISON: SVPF vs BPF vs EWMA vs GARCH\n");
    printf("  BPF: %d particles | EWMA: lambda=0.94 | GARCH: grid-best | SVPF: %d particles\n", bpf_n, n_particles);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-24s %8s %8s %8s %8s %8s\n", 
           "Scenario", "KF bnd", "BPF 50K", "EWMA", "GARCH", "SVPF");
    printf("  ──────────────────────── ──────── ──────── ──────── ──────── ────────\n");
    
    // We need SVPF results — re-run with same seeds as main table
    for (int i = 0; i < n_scenarios; i++) {
        MatchedTestData* data = scenarios[i].gen(n_ticks, base_seed + i);
        
        double kf  = kalman_steady_state_rmse(data->dgp_rho, data->dgp_sigma_z, data->dgp_nu_obs);
        double bpf = bpf_run_rmse(data, bpf_n, base_seed + 100 + i);
        double ew  = ewma_rmse(data, 0.94);
        double ga  = garch_best_rmse(data);
        
        // SVPF: re-run (same config as main)
        double svpf_elapsed;
        MatchedMetrics svpf_m = run_matched_scenario(data, n_particles, n_stein,
                                                      base_seed, &svpf_elapsed);
        
        printf("  %-24s %8.4f %8.4f %8.4f %8.4f %8.4f\n",
               data->scenario_name, kf, bpf, ew, ga, svpf_m.logvol_rmse);
        
        free_matched_data(data);
    }
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int n_ticks    = 5000;
    int n_particles = 512;
    int n_stein    = 8;
    int base_seed  = 42;
    
    // Parse optional overrides
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ticks") == 0 && i+1 < argc)
            n_ticks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--particles") == 0 && i+1 < argc)
            n_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stein") == 0 && i+1 < argc)
            n_stein = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            base_seed = atoi(argv[++i]);
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  SVPF MATCHED-DGP TEST SUITE\n");
    printf("  All scenarios: DGP = SVPF model (zero model mismatch)\n");
    printf("  Any remaining bias is PURELY from Stein transport\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  Particles: %d | Stein steps: %d | Obs nu: per-scenario | Ticks: %d\n\n",
           n_particles, n_stein, n_ticks);
    
    // --- Scenario table ---
    typedef MatchedTestData* (*GenFn)(int, int);
    struct { GenFn gen; } scenarios[] = {
        { gen_matched_baseline },
        { gen_matched_student_t },
        { gen_matched_high_persist },
        { gen_matched_high_volvol },
        { gen_matched_low_vol },
        { gen_matched_high_vol },
        { gen_matched_fast_revert },
        { gen_matched_gaussian },
    };
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);
    
    printf("  %-28s %8s %8s %8s %8s\n", "Scenario", "RMSE", "MAE", "Bias", "ms");
    printf("  ──────────────────────────── ──────── ──────── ──────── ────────\n");
    
    double sum_rmse = 0.0, sum_bias = 0.0;
    
    for (int i = 0; i < n_scenarios; i++) {
        MatchedTestData* data = scenarios[i].gen(n_ticks, base_seed + i);
        double elapsed;
        MatchedMetrics m = run_matched_scenario(data, n_particles, n_stein,
                                                 base_seed, &elapsed);
        
        printf("  %-28s %8.4f %8.4f %+8.4f %8.1f\n",
               data->scenario_name, m.logvol_rmse, m.logvol_mae, m.logvol_bias, elapsed);
        
        sum_rmse += m.logvol_rmse;
        sum_bias += m.logvol_bias;
        free_matched_data(data);
    }
    
    printf("  ──────────────────────────── ──────── ──────── ────────\n");
    printf("  %-28s %8.4f %8s %+8.4f\n", "AVERAGE",
           sum_rmse / n_scenarios, "", sum_bias / n_scenarios);
    
    // --- Multi-seed average (reduce sampling noise on bias estimate) ---
    printf("\n  ── Multi-Seed Average (Student-t Matched, %d seeds) ──\n", 10);
    int n_seeds = 10;
    double ms_rmse = 0, ms_mae = 0, ms_bias = 0;
    for (int s = 0; s < n_seeds; s++) {
        MatchedTestData* data = gen_matched_student_t(n_ticks, 1000 + s * 137);
        double elapsed;
        MatchedMetrics m = run_matched_scenario(data, n_particles, n_stein,
                                                 2000 + s * 73, &elapsed);
        ms_rmse += m.logvol_rmse;
        ms_mae  += m.logvol_mae;
        ms_bias += m.logvol_bias;
        free_matched_data(data);
    }
    printf("  RMSE: %.4f | MAE: %.4f | Bias: %+.4f\n",
           ms_rmse / n_seeds, ms_mae / n_seeds, ms_bias / n_seeds);
    
    // --- Gold standard comparison ---
    print_gold_standard(n_ticks, base_seed, n_particles, n_stein);
    
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    return 0;
}
