/**
 * @file svpf_student_t_test.cu
 * @brief A/B test for Student-t state dynamics in SVPF
 * 
 * Compares Gaussian AR(1) vs Student-t AR(1) state model when DGP
 * uses Student-t innovations (fat-tailed reality).
 * 
 * DGP: Student-t innovations in both state and observations
 * Model A: use_student_t_state = 0 (Gaussian - model mismatch)
 * Model B: use_student_t_state = 1 (Student-t - model match)
 * 
 * Scenarios (from HCRBPF test):
 *   1. Slow Drift      - Sinusoidal z
 *   2. Stress Ramp     - Linear z increase
 *   3. OU-Matched      - z follows OU
 *   4. Intermediate    - z stays in [2,4] band
 *   5. Spike+Recovery  - Instant spike + decay
 *   6. Wrong-Model     - Fast mean-reversion
 */

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

/**
 * Host-side Student-t sampler
 * Uses the standard ratio: t = Z / sqrt(V/nu) where V ~ chi²(nu)
 */
static double pcg32_student_t(pcg32_t* rng, double nu) {
    // Large nu: approximate as Gaussian
    if (nu > 30.0) {
        return pcg32_gaussian(rng);
    }
    
    double z = pcg32_gaussian(rng);
    
    // Chi-squared(nu) via sum of squared normals
    int nu_int = (int)(nu + 0.5);
    if (nu_int < 3) nu_int = 3;
    
    double chi2 = 0.0;
    for (int i = 0; i < nu_int; i++) {
        double u = pcg32_gaussian(rng);
        chi2 += u * u;
    }
    
    return z * sqrt((double)nu_int / (chi2 + 1e-10));
}

static void pcg32_seed(pcg32_t* rng, uint64_t seed) {
    rng->state = seed * 12345ULL + 1;
    rng->inc = seed * 67890ULL | 1;
    pcg32_random(rng);
    pcg32_random(rng);
}

/*═══════════════════════════════════════════════════════════════════════════
 * DGP PARAMETERS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double theta_base;
    double theta_scale;
    double theta_rate;
    double mu_vol_base;
    double mu_vol_scale;
    double mu_vol_rate;
    double sigma_vol_base;
    double sigma_vol_scale;
    double sigma_vol_rate;
    double rho_z;
    double sigma_z;
    double z_mean;
    
    // Student-t degrees of freedom for DGP
    double nu_state;   // State innovations
    double nu_obs;     // Observation innovations
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
    
    // Fat-tailed DGP
    p.nu_state = 5.0;  // State: moderately fat tails
    p.nu_obs = 5.0;    // Observations: moderately fat tails
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
 * TEST DATA STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double* true_z;
    double* true_log_vol;
    double* true_vol;
    double* returns;
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
 * GENERATE OBSERVATIONS WITH STUDENT-T INNOVATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

static void generate_observations_student_t(TestData* data, const DGPParams* p, pcg32_t* rng) {
    int n = data->n_ticks;
    
    double theta, mu_vol, sigma_vol;
    z_to_sv_params(data->true_z[0], p, &theta, &mu_vol, &sigma_vol);
    double h = mu_vol;
    
    for (int t = 0; t < n; t++) {
        z_to_sv_params(data->true_z[t], p, &theta, &mu_vol, &sigma_vol);
        
        if (t > 0) {
            // STATE TRANSITION: Student-t innovation (fat-tailed)
            double state_noise = pcg32_student_t(rng, p->nu_state);
            h = (1.0 - theta) * h + theta * mu_vol + sigma_vol * state_noise;
        }
        
        if (h < -10.0) h = -10.0;
        if (h > 2.0) h = 2.0;
        
        data->true_log_vol[t] = h;
        data->true_vol[t] = exp(h / 2.0);
        
        // OBSERVATION: Student-t innovation (fat-tailed)
        double obs_noise = pcg32_student_t(rng, p->nu_obs);
        data->returns[t] = data->true_vol[t] * obs_noise;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO GENERATORS
 *═══════════════════════════════════════════════════════════════════════════*/

static TestData* generate_slow_drift(int n, double amplitude, double period, 
                                     const DGPParams* dgp, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 1;
    data->scenario_name = "Slow Drift";
    data->scenario_desc = "Sinusoidal z drift";
    data->param1 = amplitude;
    data->param2 = period;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    for (int t = 0; t < n; t++) {
        data->true_z[t] = amplitude * (1.0 + sin(2.0 * 3.14159265358979 * t / period)) / 2.0;
    }
    
    generate_observations_student_t(data, dgp, &rng);
    return data;
}

static TestData* generate_stress_ramp(int n, double z_start, double z_end,
                                      const DGPParams* dgp, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 2;
    data->scenario_name = "Stress Ramp";
    data->scenario_desc = "Linear z ramp";
    data->param1 = z_start;
    data->param2 = z_end;
    
    pcg32_t rng;
    pcg32_seed(&rng, seed);
    
    for (int t = 0; t < n; t++) {
        double frac = (double)t / (n - 1);
        data->true_z[t] = z_start + frac * (z_end - z_start);
    }
    
    generate_observations_student_t(data, dgp, &rng);
    return data;
}

static TestData* generate_ou_matched(int n, double rho, double sigma_z, double z_mean,
                                     const DGPParams* dgp, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 3;
    data->scenario_name = "OU-Matched";
    data->scenario_desc = "OU process for z";
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
    
    DGPParams p_copy = *dgp;
    p_copy.rho_z = rho;
    p_copy.sigma_z = sigma_z;
    p_copy.z_mean = z_mean;
    generate_observations_student_t(data, &p_copy, &rng);
    return data;
}

static TestData* generate_intermediate_band(int n, double z_center, double z_spread,
                                            const DGPParams* dgp, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 4;
    data->scenario_name = "Intermediate Band";
    data->scenario_desc = "z constrained to mid-range";
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
    
    generate_observations_student_t(data, dgp, &rng);
    return data;
}

static TestData* generate_spike_recovery(int n, double z_spike, double decay_rate, 
                                         int spike_time, const DGPParams* dgp, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 5;
    data->scenario_name = "Spike+Recovery";
    data->scenario_desc = "Instant z spike + decay";
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
    
    generate_observations_student_t(data, dgp, &rng);
    return data;
}

static TestData* generate_wrong_model(int n, double true_rho, double true_sigma_z,
                                      double z_mean, const DGPParams* dgp, int seed) {
    TestData* data = alloc_test_data(n);
    data->scenario_id = 6;
    data->scenario_name = "Wrong-Model";
    data->scenario_desc = "Fast mean-reversion DGP";
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
    
    DGPParams p_copy = *dgp;
    p_copy.rho_z = true_rho;
    p_copy.sigma_z = true_sigma_z;
    p_copy.z_mean = z_mean;
    generate_observations_student_t(data, &p_copy, &rng);
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
    
    double sum_sq = 0.0, sum_abs = 0.0, sum_bias = 0.0, sum_sq_vol = 0.0;
    
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
 * CONFIGURE SVPF FILTER (Your exact settings)
 *═══════════════════════════════════════════════════════════════════════════*/

static void configure_svpf(SVPFState* filter, int use_student_t_state) {
    // SVLD + Annealing
    filter->use_svld = 1;
    filter->use_annealing = 1;
    filter->n_anneal_steps = 3;
    filter->temperature = 0.45f;
    filter->rmsprop_rho = 0.9f;
    filter->rmsprop_eps = 1e-6f;
    
    // MIM
    filter->use_mim = 0;
    filter->mim_jump_prob = 0.25f;
    filter->mim_jump_scale = 9.0f;
    filter->use_adaptive_beta = 1;
    
    // Rejuvenation
    filter->use_rejuvenation = 1;
    filter->rejuv_ksd_threshold = 0.05f;
    filter->rejuv_prob = 0.30f;
    filter->rejuv_blend = 0.30f;
    
    // Newton-Stein
    filter->use_newton = 1;
    filter->use_full_newton = 1;
    
    // Guided Prediction
    filter->use_guided = 1;
    filter->guided_alpha_base = 0.0f;
    filter->guided_alpha_shock = 0.40f;
    filter->guided_innovation_threshold = 1.5f;
    
    // EKF Guide
    filter->use_guide = 1;
    filter->use_guide_preserving = 1;
    filter->guide_strength = 0.08f;
    
    // Adaptive mu
    filter->use_adaptive_mu = 1;
    filter->mu_process_var = 0.005f;
    filter->mu_obs_var_scale = 12.0f;
    filter->mu_min = -3.0f;
    filter->mu_max =  3.0f;
    
    // Adaptive guide
    filter->use_adaptive_guide = 0;
    filter->guide_strength_base = 0.05f;
    filter->guide_strength_max = 0.30f;
    filter->guide_innovation_threshold = 1.0f;
    
    // Adaptive sigma
    filter->use_adaptive_sigma = 1;
    filter->sigma_boost_threshold = 0.95f;
    filter->sigma_boost_max = 3.2f;
    
    // Exact gradient
    filter->use_exact_gradient = 1;
    filter->lik_offset = 0.35f;
    
    // KSD-adaptive Stein steps
    filter->stein_min_steps = 8;
    filter->stein_max_steps = 16;
    filter->ksd_improvement_threshold = 0.05f;
    
    // ===== THE A/B VARIABLE =====
    filter->use_student_t_state = use_student_t_state;
    filter->nu_state = 25.0f;  // Your setting
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN SVPF ON SCENARIO
 *═══════════════════════════════════════════════════════════════════════════*/

static Metrics run_svpf_on_scenario(
    TestData* data,
    int n_particles,
    int n_stein,
    float nu,
    int use_student_t_state,
    int seed,
    double* elapsed_ms_out
) {
    int n = data->n_ticks;
    
    float* h_returns = (float*)malloc(n * sizeof(float));
    float* h_loglik = (float*)malloc(n * sizeof(float));
    float* h_vol = (float*)malloc(n * sizeof(float));
    float* h_logvol = (float*)malloc(n * sizeof(float));
    
    for (int t = 0; t < n; t++) {
        h_returns[t] = (float)data->returns[t];
    }
    
    SVPFState* filter = svpf_create(n_particles, n_stein, nu, NULL);
    
    SVPFParams params;
    params.rho = 0.97f;
    params.sigma_z = 0.15f;
    params.mu = -4.5f;
    params.gamma = 0.0f;
    
    // Apply your configuration
    configure_svpf(filter, use_student_t_state);
    
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
    
    Metrics m = compute_metrics(data, h_logvol);
    
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
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  SVPF Student-t State Dynamics A/B Test\n");
    printf("  DGP: Student-t innovations (nu_state=5, nu_obs=5)\n");
    printf("  Model A: Gaussian AR(1) state (mismatch)\n");
    printf("  Model B: Student-t AR(1) state (nu=8, match)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int n_particles = 512;
    int n_stein = 12;
    float nu_obs = 5.0f;  // Observation model nu
    int n_ticks = 5000;
    int base_seed = 42;
    
    DGPParams dgp = default_dgp_params();
    // Fat-tailed DGP
    dgp.nu_state = 5.0;
    dgp.nu_obs = 5.0;
    
    printf("Configuration:\n");
    printf("  Particles: %d\n", n_particles);
    printf("  Stein steps: %d (KSD-adaptive: 8-16)\n", n_stein);
    printf("  Observation nu: %.1f\n", nu_obs);
    printf("  DGP state nu: %.1f\n", dgp.nu_state);
    printf("  DGP obs nu: %.1f\n", dgp.nu_obs);
    printf("  Ticks: %d\n\n", n_ticks);
    
    // Scenario generators
    typedef TestData* (*ScenarioGenerator)(int, const DGPParams*, int);
    
    struct {
        const char* name;
        TestData* (*generate)(int, const DGPParams*, int);
    } scenarios_simple[] = {
        {"Slow Drift", NULL},
        {"Stress Ramp", NULL},
        {"OU-Matched", NULL},
        {"Intermediate", NULL},
        {"Spike+Recovery", NULL},
        {"Wrong-Model", NULL}
    };
    
    // Generate all scenarios
    TestData* scenarios[6];
    scenarios[0] = generate_slow_drift(n_ticks, 5.0, 1000.0, &dgp, base_seed);
    scenarios[1] = generate_stress_ramp(n_ticks, 0.5, 6.0, &dgp, base_seed + 1);
    scenarios[2] = generate_ou_matched(n_ticks, 0.98585, 0.02828, 2.0, &dgp, base_seed + 2);
    scenarios[3] = generate_intermediate_band(n_ticks, 3.0, 1.0, &dgp, base_seed + 3);
    scenarios[4] = generate_spike_recovery(n_ticks, 7.0, 0.01, 1000, &dgp, base_seed + 4);
    scenarios[5] = generate_wrong_model(n_ticks, 0.90, 0.15, 2.0, &dgp, base_seed + 5);
    
    // Results table header
    printf("┌────────────────────┬──────────────────────────────┬──────────────────────────────┬─────────────┐\n");
    printf("│ Scenario           │ Gaussian AR(1)               │ Student-t AR(1)              │ Winner      │\n");
    printf("│                    │ RMSE    MAE     Bias   ms    │ RMSE    MAE     Bias   ms    │             │\n");
    printf("├────────────────────┼──────────────────────────────┼──────────────────────────────┼─────────────┤\n");
    
    int gaussian_wins = 0;
    int student_t_wins = 0;
    
    for (int i = 0; i < 6; i++) {
        TestData* data = scenarios[i];
        double elapsed_gauss, elapsed_student;
        
        // Run with Gaussian state model
        Metrics m_gauss = run_svpf_on_scenario(
            data, n_particles, n_stein, nu_obs, 
            0,  // use_student_t_state = 0
            base_seed + 100 + i, &elapsed_gauss
        );
        
        // Run with Student-t state model
        Metrics m_student = run_svpf_on_scenario(
            data, n_particles, n_stein, nu_obs, 
            1,  // use_student_t_state = 1
            base_seed + 100 + i, &elapsed_student
        );
        
        // Determine winner (by RMSE)
        const char* winner;
        if (m_student.logvol_rmse < m_gauss.logvol_rmse * 0.99) {
            winner = "Student-t";
            student_t_wins++;
        } else if (m_gauss.logvol_rmse < m_student.logvol_rmse * 0.99) {
            winner = "Gaussian";
            gaussian_wins++;
        } else {
            winner = "Tie";
        }
        
        printf("│ %-18s │ %6.3f  %6.3f  %+6.3f %5.1f │ %6.3f  %6.3f  %+6.3f %5.1f │ %-11s │\n",
               data->scenario_name,
               m_gauss.logvol_rmse, m_gauss.logvol_mae, m_gauss.logvol_bias, elapsed_gauss,
               m_student.logvol_rmse, m_student.logvol_mae, m_student.logvol_bias, elapsed_student,
               winner);
    }
    
    printf("└────────────────────┴──────────────────────────────┴──────────────────────────────┴─────────────┘\n\n");
    
    printf("Summary: Gaussian wins %d, Student-t wins %d\n\n", gaussian_wins, student_t_wins);
    
    if (student_t_wins > gaussian_wins) {
        printf("✓ Student-t state model outperforms on fat-tailed DGP\n");
    } else if (gaussian_wins > student_t_wins) {
        printf("✗ Gaussian state model still wins (unexpected for fat-tailed DGP)\n");
    } else {
        printf("≈ Models perform similarly\n");
    }
    
    // Cleanup
    for (int i = 0; i < 6; i++) {
        free_test_data(scenarios[i]);
    }
    
    return 0;
}
