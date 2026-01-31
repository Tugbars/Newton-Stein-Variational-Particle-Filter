/**
 * @file test_svpf_hcrbpf_dgp.cu
 * @brief SVPF test on HCRBPF's DGP for fair comparison
 *
 * Purpose: Compare SVPF vs HCRBPF on the SAME data generating process.
 *
 * DGP (from HCRBPF):
 *   z_t = ρ_z * z_{t-1} + (1 - ρ_z) * z_mean + σ_z * ε_t   (stress index)
 *   θ(z) = θ_base + θ_scale * (1 - exp(-θ_rate * z))        (mean-reversion speed)
 *   μ(z) = μ_base + μ_scale * (1 - exp(-μ_rate * z))        (long-run mean)
 *   σ(z) = σ_base + σ_scale * (1 - exp(-σ_rate * z))        (vol-of-vol)
 *   h_t = (1-θ) * h_{t-1} + θ * μ + σ * η_t                 (log-vol)
 *   r_t = exp(h_t/2) * ε_t                                   (return)
 *
 * SVPF sees only r_t, estimates h_t. Compare with true h_t.
 *
 * Scenarios (matching HCRBPF test):
 *   1. Slow Drift      - Sinusoidal z
 *   2. Stress Ramp     - Linear z increase
 *   3. OU-Matched      - z follows OU (model=DGP for HCRBPF)
 *   4. Intermediate    - z stays in [2,4] band
 *   5. Spike+Recovery  - Instant spike + decay
 *   6. Wrong-Model     - Fast mean-reversion (stress test)
 */

/*═══════════════════════════════════════════════════════════════════════════
 * BUILD CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * SVPF Step Mode:
 *   0 = svpf_step_adaptive  (Regular kernel launches, ~100-150μs CPU overhead)
 *   1 = svpf_step_graph     (CUDA Graph replay, ~10-15μs CPU overhead)
 *
 * Graph mode captures the kernel sequence on first call, then replays
 * with minimal CPU overhead. Best for HFT where latency matters.
 */
#define USE_CUDA_GRAPH 1

/**
 * Latency Benchmarking:
 *   0 = Disabled (only total time reported)
 *   1 = Enabled  (per-step latency histogram + percentiles)
 */
#define BENCHMARK_LATENCY 0

/*═══════════════════════════════════════════════════════════════════════════*/

#include "svpf.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static double get_time_us(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart * 1e6;
}
#else
#include <time.h>
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * LATENCY BENCHMARK HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

#if BENCHMARK_LATENCY
// Comparison function for qsort (latency percentiles)
int compare_double_for_qsort(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * PCG32 RNG
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
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static void pcg32_seed(pcg32_t* rng, uint64_t seed) {
    rng->state = seed * 12345ULL + 1;
    rng->inc = seed * 67890ULL | 1;
    pcg32_random(rng);
    pcg32_random(rng);
}

/*═══════════════════════════════════════════════════════════════════════════
 * HCRBPF DGP PARAMETERS
 * 
 * z → SV parameter mapping (saturating functions):
 *   θ(z) = θ_base + θ_scale * (1 - exp(-θ_rate * z))
 *   μ(z) = μ_base + μ_scale * (1 - exp(-μ_rate * z))
 *   σ(z) = σ_base + σ_scale * (1 - exp(-σ_rate * z))
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* z → θ (mean-reversion speed) */
    double theta_base;
    double theta_scale;
    double theta_rate;
    
    /* z → μ (long-run mean of log-vol) */
    double mu_vol_base;
    double mu_vol_scale;
    double mu_vol_rate;
    
    /* z → σ (vol-of-vol) */
    double sigma_vol_base;
    double sigma_vol_scale;
    double sigma_vol_rate;
    
    /* z dynamics (OU process) */
    double rho_z;
    double sigma_z;
    double z_mean;
} DGPParams;

/* Default parameters - matches HCRBPF CUDA filter */
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

/* Saturating function */
static double sat(double base, double scale, double rate, double z) {
    return base + scale * (1.0 - exp(-rate * z));
}

/* z → SV parameters */
static void z_to_sv_params(double z, const DGPParams* p,
                           double* theta, double* mu_vol, double* sigma_vol) {
    *theta = sat(p->theta_base, p->theta_scale, p->theta_rate, z);
    *mu_vol = sat(p->mu_vol_base, p->mu_vol_scale, p->mu_vol_rate, z);
    *sigma_vol = sat(p->sigma_vol_base, p->sigma_vol_scale, p->sigma_vol_rate, z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST DATA STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double* true_z;        /* Ground truth stress index */
    double* true_log_vol;  /* Ground truth h */
    double* true_vol;      /* exp(h/2) for comparison with SVPF output */
    double* returns;       /* Observations */
    int n_ticks;
    
    int scenario_id;
    const char* scenario_name;
    const char* scenario_desc;
    
    double param1;
    double param2;
    double param3;
} TestData;

static TestData* alloc_test_data(int n) {
    TestData* data = (TestData*)calloc(1, sizeof(TestData));
    data->n_ticks = n;
    data->true_z = (double*)malloc(n * sizeof(double));
    data->true_log_vol = (double*)malloc(n * sizeof(double));
    data->true_vol = (double*)malloc(n * sizeof(double));
    data->returns = (double*)malloc(n * sizeof(double));
    return data;
}

static void free_test_data(TestData* data) {
    if (!data) return;
    free(data->true_z);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->returns);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * GENERATE OBSERVATIONS FROM z TRAJECTORY
 * 
 * Given true_z[t], evolve h and generate returns
 *═══════════════════════════════════════════════════════════════════════════*/

static void generate_observations(TestData* data, const DGPParams* p, pcg32_t* rng) {
    int n = data->n_ticks;
    
    /* Initialize h from first z */
    double theta, mu_vol, sigma_vol;
    z_to_sv_params(data->true_z[0], p, &theta, &mu_vol, &sigma_vol);
    double h = mu_vol;  /* Start at long-run mean */
    
    for (int t = 0; t < n; t++) {
        /* Get SV params from z */
        z_to_sv_params(data->true_z[t], p, &theta, &mu_vol, &sigma_vol);
        
        /* Evolve h: h_t = (1-θ)*h_{t-1} + θ*μ + σ*ε */
        if (t > 0) {
            h = (1.0 - theta) * h + theta * mu_vol + sigma_vol * pcg32_gaussian(rng);
        }
        
        /* Clamp to reasonable range */
        if (h < -10.0) h = -10.0;
        if (h > 2.0) h = 2.0;
        
        data->true_log_vol[t] = h;
        data->true_vol[t] = exp(h / 2.0);  /* Standard deviation, not variance */
        
        /* Generate return: r = exp(h/2) * ε */
        data->returns[t] = data->true_vol[t] * pcg32_gaussian(rng);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 1: SLOW DRIFT
 * z drifts slowly: z(t) = A * (1 + sin(2π * t / period)) / 2
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_slow_drift(int n, double amplitude, double period, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 1;
    data->scenario_name = "Slow Drift";
    data->scenario_desc = "Sinusoidal z drift - tests tracking of gradual changes";
    data->param1 = amplitude;
    data->param2 = period;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    for (int t = 0; t < n; t++) {
        data->true_z[t] = amplitude * (1.0 + sin(2.0 * 3.14159265358979 * t / period)) / 2.0;
    }
    
    DGPParams p = default_dgp_params();
    generate_observations(data, &p, &rng);
    
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 2: STRESS RAMP
 * z ramps linearly from z_start to z_end
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_stress_ramp(int n, double z_start, double z_end, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 2;
    data->scenario_name = "Stress Ramp";
    data->scenario_desc = "Linear z ramp - tests tracking bias";
    data->param1 = z_start;
    data->param2 = z_end;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    for (int t = 0; t < n; t++) {
        double frac = (double)t / (n - 1);
        data->true_z[t] = z_start + frac * (z_end - z_start);
    }
    
    DGPParams p = default_dgp_params();
    generate_observations(data, &p, &rng);
    
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 3: OU-MATCHED (Optimality test - DGP = Model for HCRBPF)
 * z follows the same OU process as HCRBPF assumes
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_ou_matched(int n, double rho, double sigma_z, 
                                     double z_mean, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 3;
    data->scenario_name = "OU-Matched";
    data->scenario_desc = "OU process for z - DGP matches HCRBPF model";
    data->param1 = rho;
    data->param2 = sigma_z;
    data->param3 = z_mean;
    
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
    generate_observations(data, &p, &rng);
    
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 4: INTERMEDIATE BAND
 * z stays in [z_center - spread, z_center + spread]
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_intermediate_band(int n, double z_center, 
                                            double z_spread, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 4;
    data->scenario_name = "Intermediate Band";
    data->scenario_desc = "z constrained to mid-range - tests non-extreme tracking";
    data->param1 = z_center;
    data->param2 = z_spread;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    double z = z_center;
    double rho = 0.98;
    double sigma = z_spread * 0.1;
    
    for (int t = 0; t < n; t++) {
        z = rho * z + (1.0 - rho) * z_center + sigma * pcg32_gaussian(&rng);
        if (z < z_center - z_spread) z = z_center - z_spread + 0.1 * pcg32_double(&rng);
        if (z > z_center + z_spread) z = z_center + z_spread - 0.1 * pcg32_double(&rng);
        data->true_z[t] = z;
    }
    
    DGPParams p = default_dgp_params();
    generate_observations(data, &p, &rng);
    
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 5: SPIKE + RECOVERY
 * z spikes instantly at spike_time, then decays exponentially
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_spike_recovery(int n, double z_spike, 
                                         double decay_rate, int spike_time, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 5;
    data->scenario_name = "Spike+Recovery";
    data->scenario_desc = "Instant z spike + exponential decay - stress test";
    data->param1 = z_spike;
    data->param2 = decay_rate;
    data->param3 = (double)spike_time;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    double z_base = 0.5;
    for (int t = 0; t < n; t++) {
        if (t < spike_time) {
            data->true_z[t] = z_base;
        } else {
            double dt = t - spike_time;
            data->true_z[t] = z_base + (z_spike - z_base) * exp(-decay_rate * dt);
        }
    }
    
    DGPParams p = default_dgp_params();
    generate_observations(data, &p, &rng);
    
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 6: WRONG-MODEL (Robustness test)
 * DGP uses faster mean-reversion than SVPF assumes
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_wrong_model(int n, double true_rho, double true_sigma_z,
                                      double z_mean, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 6;
    data->scenario_name = "Wrong-Model";
    data->scenario_desc = "Fast mean-reversion DGP - model misspecification test";
    data->param1 = true_rho;
    data->param2 = true_sigma_z;
    data->param3 = z_mean;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    double z = z_mean;
    for (int t = 0; t < n; t++) {
        z = true_rho * z + (1.0 - true_rho) * z_mean + true_sigma_z * pcg32_gaussian(&rng);
        if (z < 0.0) z = 0.0;
        if (z > 10.0) z = 10.0;
        data->true_z[t] = z;
    }
    
    DGPParams p = default_dgp_params();
    p.rho_z = true_rho;
    p.sigma_z = true_sigma_z;
    p.z_mean = z_mean;
    generate_observations(data, &p, &rng);
    
    return data;
}

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double logvol_rmse;
    double logvol_mae;
    double logvol_bias;
    double vol_rmse;
} Metrics;

static Metrics compute_metrics(const TestData* data, const float* est_logvol) {
    Metrics m = {0};
    int n = data->n_ticks;
    
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double sum_bias = 0.0;
    double sum_sq_vol = 0.0;
    
    for (int t = 0; t < n; t++) {
        double err = est_logvol[t] - data->true_log_vol[t];
        sum_sq += err * err;
        sum_abs += fabs(err);
        sum_bias += err;
        
        double est_vol = exp(est_logvol[t] / 2.0);
        double err_vol = est_vol - data->true_vol[t];
        sum_sq_vol += err_vol * err_vol;
    }
    
    m.logvol_rmse = sqrt(sum_sq / n);
    m.logvol_mae = sum_abs / n;
    m.logvol_bias = sum_bias / n;
    m.vol_rmse = sqrt(sum_sq_vol / n);
    
    return m;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN SVPF ON SCENARIO
 *═══════════════════════════════════════════════════════════════════════════*/

static Metrics run_svpf_on_scenario(
    TestData* data,
    int n_particles,
    int n_stein,
    float nu,
    int use_adaptive,
    int seed,
    double* elapsed_ms_out
) {
    int n = data->n_ticks;
    
    /* Allocate outputs */
    float* h_returns = (float*)malloc(n * sizeof(float));
    float* h_loglik = (float*)malloc(n * sizeof(float));
    float* h_vol = (float*)malloc(n * sizeof(float));
    float* h_logvol = (float*)malloc(n * sizeof(float));
    
    for (int t = 0; t < n; t++) {
        h_returns[t] = (float)data->returns[t];
    }
    
    /* Create filter */
    SVPFState* filter = svpf_create(n_particles, n_stein, nu, NULL);

    //svpf_set_stein_sign_mode(filter, SVPF_STEIN_SIGN_PAPER);  // or just: filter->stein_repulsive_sign = 1;

    
    /* 
     * SVPF parameters - note these are MISSPECIFIED for this DGP!
     * SVPF assumes fixed AR(1) parameters, but DGP has z-dependent params.
     * This is intentional - tests SVPF robustness to misspecification.
     */
    SVPFParams params;
    params.rho = 0.97f;     /* Fixed, but DGP has θ(z) ∈ [0.007, 0.127] */
    params.sigma_z = 0.15f; /* Fixed, but DGP has σ(z) ∈ [0.08, 0.50] */
    params.mu = -4.5f;      /* Fixed, but DGP has μ(z) ∈ [-4.5, -1.0] */
    params.gamma = 0.0f;    /* No leverage in this DGP */

    // IMPORTANT: Re-initialize to set up lambda particles
    svpf_initialize(filter, &params, seed);

    /* Configure adaptive settings */
    if (use_adaptive) {
        filter->use_svld = 1;
        filter->use_annealing = 1;
        filter->n_anneal_steps = 5;
        filter->temperature = 0.45f;
        filter->rmsprop_rho = 0.9f;
        filter->rmsprop_eps = 1e-6f;
        
        filter->use_mim = 1;
        filter->mim_jump_prob = 0.25f;
        filter->mim_jump_scale = 9.0f;
        filter->use_adaptive_beta = 1;  // ON by default, set 0 for A/B test

        filter->use_rejuvenation = 1;        // ON
        filter->rejuv_ksd_threshold = 0.05f; // Trigger threshold
        filter->rejuv_prob = 0.30f;          // 30% of particles
        filter->rejuv_blend = 0.30f;         // 30% blend factor

        // Newton-Stein (Hessian preconditioning)
        // Adaptive step size based on local curvature: H^{-1} * grad
        filter->use_newton = 1;
        filter->use_full_newton = 1;  // Enable Detommaso 2018 full Newton

        // Guided Prediction with INNOVATION GATING (FIXED)
        // - Bottom clamp prevents zero-return trap (log(0) → -inf)
        // - Asymmetric gating only activates on UPWARD shocks (spikes)
        filter->use_guided = 1;
        filter->guided_alpha_base = 0.0f;             // 0% when model fits
        filter->guided_alpha_shock = 0.40f;           // 40% when model fails
        filter->guided_innovation_threshold = 1.5f;   // 1.5σ = "surprised"
        
        // EKF Guide density
        filter->use_guide = 1;
        filter->use_guide_preserving = 1;  // Variance-preserving shift (not contraction)
        filter->guide_strength = 0.05f;
      
        filter->use_adaptive_mu = 1;
        filter->mu_process_var = 0.001f;  // Q: how fast can mu drift
        filter->mu_obs_var_scale = 11.0f; // R = scale * bw²
        filter->mu_min = -4.0f;
        filter->mu_max = -1.0f;

        filter->use_adaptive_guide = 1;
        filter->guide_strength_base = 0.05f;       // Base when model fits
        filter->guide_strength_max = 0.30f;        // Max during surprises
        filter->guide_innovation_threshold = 1.0f; // Z-score to start boosting

        filter->use_adaptive_sigma = 1;
        filter->sigma_boost_threshold = 0.95f; // Start boosting when |z| > 1
        filter->sigma_boost_max = 3.2f;       // Max 3x boost

        filter->use_exact_gradient = 1;
        filter->lik_offset = 0.345f;  // No correction - test if model is now consistent

         // === KSD-based Adaptive Stein Steps ===
        // Replaces fixed n_stein_steps with convergence-based early stopping
        // KSD (Kernel Stein Discrepancy) computed in same O(N²) pass - zero extra cost
        filter->stein_min_steps = 7;              // Always run at least 4 (RMSProp warmup)
        filter->stein_max_steps = 7;             // Cap at 12 (crisis budget)
        filter->ksd_improvement_threshold = 0.05; // Stop if <5% relative improvement

        // Enable Student-t state dynamics
        filter->use_student_t_state = 1;
        filter->nu_state = 5.0f; // 5-7 recommended, lower = fatter tails

        // Enable smoothing with 1-tick output lag
        filter->use_smoothing = 1;
        filter->smooth_lag = 3;        // Buffer last 3 estimates
        filter->smooth_output_lag = 1; // Output h[t-1] (smoothed by y[t])

        filter->use_persistent_kernel = 1;

        //filter->use_heun = 1;

    } else {
        filter->use_svld = 0;
        filter->use_annealing = 0;
        filter->use_mim = 0;
        filter->use_asymmetric_rho = 0;
        filter->use_local_params = 0;
        filter->use_newton = 0;
        filter->use_guided = 0;
        filter->guided_alpha_base = 0.0f;
        filter->guided_alpha_shock = 0.0f;
        filter->guided_innovation_threshold = 1.5f;
        filter->use_guide = 0;
        filter->use_guide_preserving = 0;
    }
    
    /* Run filter */
    double t_start = get_time_us();
    
#if BENCHMARK_LATENCY
    double* step_latencies = (double*)malloc(n * sizeof(double));
#endif
    
    float y_prev = 0.0f;
    for (int t = 0; t < n; t++) {
        float y_t = h_returns[t];
        
#if BENCHMARK_LATENCY
        double step_start = get_time_us();
#endif
        
        if (use_adaptive) {
#if USE_CUDA_GRAPH
            // CUDA Graph mode: captures on first call, replays thereafter
            svpf_step_graph(filter, y_t, y_prev, &params,
                           &h_loglik[t], &h_vol[t], &h_logvol[t]);
#else
            // Regular mode: launches kernels directly each step
            svpf_step_adaptive(filter, y_t, y_prev, &params,
                              &h_loglik[t], &h_vol[t], &h_logvol[t]);
#endif
        } else {
            SVPFResult result;
            svpf_step(filter, y_t, &params, &result);
            h_loglik[t] = result.log_lik_increment;
            h_vol[t] = result.vol_mean;
            h_logvol[t] = result.h_mean;
        }
        
#if BENCHMARK_LATENCY
        step_latencies[t] = get_time_us() - step_start;
#endif
        
        y_prev = y_t;
    }
    
    double t_end = get_time_us();
    *elapsed_ms_out = (t_end - t_start) / 1000.0;
    
#if BENCHMARK_LATENCY
    // Skip warmup (first 100 steps) for percentile calculation
    int warmup = 100;
    int effective_n = n - warmup;
    double* sorted_latencies = step_latencies + warmup;
    
    qsort(sorted_latencies, effective_n, sizeof(double), compare_double_for_qsort);
    
    double p50 = sorted_latencies[effective_n / 2];
    double p90 = sorted_latencies[(int)(effective_n * 0.90)];
    double p99 = sorted_latencies[(int)(effective_n * 0.99)];
    double p999 = sorted_latencies[(int)(effective_n * 0.999)];
    
    printf("  Latency (μs): P50=%.1f, P90=%.1f, P99=%.1f, P99.9=%.1f\n",
           p50, p90, p99, p999);
    
    free(step_latencies);
#endif
    
    /* Compute metrics */
    Metrics m = compute_metrics(data, h_logvol);
    
    /* Cleanup */
    svpf_destroy(filter);
    free(h_returns);
    free(h_loglik);
    free(h_vol);
    free(h_logvol);
    
    return m;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║     SVPF vs HCRBPF Fair Comparison (HCRBPF DGP)                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");
    
    /* Configuration */
    int seed = 42;
    int n_ticks = 5000;
    int n_particles = 512;
    int n_stein = 8;
    float nu = 50.0f;
    int use_adaptive = 1;
    
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ticks") == 0 && i + 1 < argc) {
            n_ticks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            n_particles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stein") == 0 && i + 1 < argc) {
            n_stein = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive") == 0) {
            use_adaptive = 1;
        } else if (strcmp(argv[i], "--vanilla") == 0) {
            use_adaptive = 0;
        }
    }
    
    printf("Configuration:\n");
    printf("  Seed:        %d\n", seed);
    printf("  Ticks:       %d\n", n_ticks);
    printf("  Particles:   %d\n", n_particles);
    printf("  Stein steps: %d\n", n_stein);
    printf("  Student-t ν: %.1f\n", nu);
    printf("  Mode:        %s\n", use_adaptive ? "ADAPTIVE" : "VANILLA");
#if USE_CUDA_GRAPH
    printf("  Execution:   CUDA GRAPH (low-latency)\n");
#else
    printf("  Execution:   STANDARD (kernel launches)\n");
#endif
#if BENCHMARK_LATENCY
    printf("  Latency:     BENCHMARKING ENABLED\n");
#endif
    printf("\n");
    
    printf("DGP: HCRBPF's z → SV parameter mapping\n");
    printf("  z_t follows OU process\n");
    printf("  θ(z), μ(z), σ(z) via saturating functions\n");
    printf("  h_t = (1-θ)*h_{t-1} + θ*μ + σ*ε\n");
    printf("  r_t = exp(h/2) * ε\n");
    printf("\n");
    printf("NOTE: SVPF model is MISSPECIFIED for this DGP!\n");
    printf("  SVPF assumes fixed (ρ, σ, μ), but DGP has z-dependent params.\n");
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * RUN ALL SCENARIOS
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  SCENARIO RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");
    
    /* Results storage */
    const int N_SCENARIOS = 6;
    const char* scenario_names[6];
    double rmse_results[6];
    double mae_results[6];
    double bias_results[6];
    double time_results[6];
    
    /* Scenario 1: Slow Drift */
    {
        TestData* data = generate_slow_drift(n_ticks, 4.0, 2000.0, seed);
        double elapsed;
        Metrics m = run_svpf_on_scenario(data, n_particles, n_stein, nu, 
                                         use_adaptive, seed, &elapsed);
        
        printf("Scenario 1: %s\n", data->scenario_name);
        printf("  %s\n", data->scenario_desc);
        printf("  Log-Vol RMSE: %.4f\n", m.logvol_rmse);
        printf("  Log-Vol MAE:  %.4f\n", m.logvol_mae);
        printf("  Log-Vol Bias: %.4f\n", m.logvol_bias);
        printf("  Time: %.2f ms\n\n", elapsed);
        
        scenario_names[0] = data->scenario_name;
        rmse_results[0] = m.logvol_rmse;
        mae_results[0] = m.logvol_mae;
        bias_results[0] = m.logvol_bias;
        time_results[0] = elapsed;
        
        free_test_data(data);
    }
    
    /* Scenario 2: Stress Ramp */
    {
        TestData* data = generate_stress_ramp(n_ticks, 0.0, 6.0, seed);
        double elapsed;
        Metrics m = run_svpf_on_scenario(data, n_particles, n_stein, nu,
                                         use_adaptive, seed, &elapsed);
        
        printf("Scenario 2: %s\n", data->scenario_name);
        printf("  %s\n", data->scenario_desc);
        printf("  Log-Vol RMSE: %.4f\n", m.logvol_rmse);
        printf("  Log-Vol MAE:  %.4f\n", m.logvol_mae);
        printf("  Log-Vol Bias: %.4f\n", m.logvol_bias);
        printf("  Time: %.2f ms\n\n", elapsed);
        
        scenario_names[1] = data->scenario_name;
        rmse_results[1] = m.logvol_rmse;
        mae_results[1] = m.logvol_mae;
        bias_results[1] = m.logvol_bias;
        time_results[1] = elapsed;
        
        free_test_data(data);
    }
    
    /* Scenario 3: OU-Matched (HCRBPF's optimal case) */
    {
        DGPParams p = default_dgp_params();
        TestData* data = generate_ou_matched(n_ticks, p.rho_z, p.sigma_z, p.z_mean, seed);
        double elapsed;
        Metrics m = run_svpf_on_scenario(data, n_particles, n_stein, nu,
                                         use_adaptive, seed, &elapsed);
        
        printf("Scenario 3: %s (HCRBPF's optimal case)\n", data->scenario_name);
        printf("  %s\n", data->scenario_desc);
        printf("  Log-Vol RMSE: %.4f\n", m.logvol_rmse);
        printf("  Log-Vol MAE:  %.4f\n", m.logvol_mae);
        printf("  Log-Vol Bias: %.4f\n", m.logvol_bias);
        printf("  Time: %.2f ms\n\n", elapsed);
        
        scenario_names[2] = data->scenario_name;
        rmse_results[2] = m.logvol_rmse;
        mae_results[2] = m.logvol_mae;
        bias_results[2] = m.logvol_bias;
        time_results[2] = elapsed;
        
        free_test_data(data);
    }
    
    /* Scenario 4: Intermediate Band */
    {
        TestData* data = generate_intermediate_band(n_ticks, 3.0, 1.0, seed);
        double elapsed;
        Metrics m = run_svpf_on_scenario(data, n_particles, n_stein, nu,
                                         use_adaptive, seed, &elapsed);
        
        printf("Scenario 4: %s\n", data->scenario_name);
        printf("  %s\n", data->scenario_desc);
        printf("  Log-Vol RMSE: %.4f\n", m.logvol_rmse);
        printf("  Log-Vol MAE:  %.4f\n", m.logvol_mae);
        printf("  Log-Vol Bias: %.4f\n", m.logvol_bias);
        printf("  Time: %.2f ms\n\n", elapsed);
        
        scenario_names[3] = data->scenario_name;
        rmse_results[3] = m.logvol_rmse;
        mae_results[3] = m.logvol_mae;
        bias_results[3] = m.logvol_bias;
        time_results[3] = elapsed;
        
        free_test_data(data);
    }
    
    /* Scenario 5: Spike + Recovery */
    {
        TestData* data = generate_spike_recovery(n_ticks, 6.0, 0.01, n_ticks/4, seed);
        double elapsed;
        Metrics m = run_svpf_on_scenario(data, n_particles, n_stein, nu,
                                         use_adaptive, seed, &elapsed);
        
        printf("Scenario 5: %s\n", data->scenario_name);
        printf("  %s\n", data->scenario_desc);
        printf("  Log-Vol RMSE: %.4f\n", m.logvol_rmse);
        printf("  Log-Vol MAE:  %.4f\n", m.logvol_mae);
        printf("  Log-Vol Bias: %.4f\n", m.logvol_bias);
        printf("  Time: %.2f ms\n\n", elapsed);
        
        scenario_names[4] = data->scenario_name;
        rmse_results[4] = m.logvol_rmse;
        mae_results[4] = m.logvol_mae;
        bias_results[4] = m.logvol_bias;
        time_results[4] = elapsed;
        
        free_test_data(data);
    }
    
    /* Scenario 6: Wrong-Model */
    {
        TestData* data = generate_wrong_model(n_ticks, 0.90, 0.10, 2.0, seed);
        double elapsed;
        Metrics m = run_svpf_on_scenario(data, n_particles, n_stein, nu,
                                         use_adaptive, seed, &elapsed);
        
        printf("Scenario 6: %s\n", data->scenario_name);
        printf("  %s\n", data->scenario_desc);
        printf("  Log-Vol RMSE: %.4f\n", m.logvol_rmse);
        printf("  Log-Vol MAE:  %.4f\n", m.logvol_mae);
        printf("  Log-Vol Bias: %.4f\n", m.logvol_bias);
        printf("  Time: %.2f ms\n\n", elapsed);
        
        scenario_names[5] = data->scenario_name;
        rmse_results[5] = m.logvol_rmse;
        mae_results[5] = m.logvol_mae;
        bias_results[5] = m.logvol_bias;
        time_results[5] = elapsed;
        
        free_test_data(data);
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * SUMMARY TABLE
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY (SVPF on HCRBPF DGP)\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");
    
    printf("  %-20s %10s %10s %10s\n", "Scenario", "RMSE", "MAE", "Bias");
    printf("  %-20s %10s %10s %10s\n", "────────", "────", "───", "────");
    
    double total_rmse = 0.0;
    for (int i = 0; i < N_SCENARIOS; i++) {
        printf("  %-20s %10.4f %10.4f %10.4f\n", 
               scenario_names[i], rmse_results[i], mae_results[i], bias_results[i]);
        total_rmse += rmse_results[i];
    }
    
    printf("  %-20s %10s %10s %10s\n", "────────", "────", "───", "────");
    printf("  %-20s %10.4f\n", "AVERAGE RMSE", total_rmse / N_SCENARIOS);
    printf("\n");
    
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  COMPARISON REFERENCE\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");
    printf("  HCRBPF on OU-Matched: ~0.41 RMSE (from your tests)\n");
    printf("  SVPF on OU-Matched:   %.4f RMSE (this test)\n", rmse_results[2]);
    printf("\n");
    printf("  Note: SVPF uses fixed params, HCRBPF knows z→param mapping.\n");
    printf("        Gap reflects model misspecification + Rao-Blackwell advantage.\n");
    printf("\n");
    
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  DONE\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    
    return 0;
}
