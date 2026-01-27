/**
 * @file test_svpf_scenarios.cu
 * @brief SVPF test with continuous SV DGP (favors SVPF over discrete-regime filters)
 *
 * DGP characteristics that favor SVPF:
 *   1. Continuous volatility dynamics (no discrete regime jumps)
 *   2. Smooth mean-reversion with time-varying parameters
 *   3. Student-t observations (fat tails)
 *   4. Leverage effect (asymmetric response to returns)
 *   5. Volatility clustering without hard regime boundaries
 *
 * Scenarios (8000 ticks total):
 *   1. Calm Drift (0-1499)           - Low vol, slow mean-reversion
 *   2. Building Tension (1500-2499)  - Gradual vol increase
 *   3. Volatility Storm (2500-3499)  - High vol, heavy tails active
 *   4. Whipsaw (3500-4499)           - Rapid vol oscillations
 *   5. Leverage Cascade (4500-5499)  - Negative returns drive vol up
 *   6. Calm Return (5500-6499)       - Slow decay to low vol
 *   7. Mixed Dynamics (6500-7999)    - Everything combined
 *
 * Output: CSV file for comparison with HCRBPF results
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
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/* Student-t random variable via ratio of Gaussian and Chi-squared */
static double pcg32_student_t(pcg32_t* rng, double nu) {
    double z = pcg32_gaussian(rng);
    
    /* Chi-squared with nu degrees of freedom (sum of nu standard normals squared) */
    double chi_sq = 0.0;
    int nu_int = (int)nu;
    for (int i = 0; i < nu_int; i++) {
        double g = pcg32_gaussian(rng);
        chi_sq += g * g;
    }
    
    /* t = Z / sqrt(chi_sq / nu) */
    return z / sqrt(chi_sq / nu);
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONTINUOUS SV MODEL PARAMETERS
 * 
 * Model:
 *   h_t = μ(t) + ρ * (h_{t-1} - μ(t)) + σ_h(t) * ε_t + γ * y_{t-1} / exp(h_{t-1}/2)
 *   y_t = exp(h_t/2) * η_t,  η_t ~ Student-t(ν)
 *
 * Time-varying parameters via smooth functions (no discrete jumps):
 *   μ(t)   = μ_base + μ_wave * sin(2π * t / period)
 *   σ_h(t) = σ_base + σ_extra * sigmoid(tension(t))
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Base SV parameters */
    double rho;          /* Persistence */
    double mu_base;      /* Base log-vol mean */
    double sigma_base;   /* Base vol-of-vol */
    double gamma;        /* Leverage effect */
    double nu;           /* Student-t degrees of freedom */
    
    /* Time-varying modulation */
    double mu_wave;      /* Amplitude of μ oscillation */
    double sigma_extra;  /* Extra vol-of-vol during stress */
    double period;       /* Oscillation period */
} SVParams;

/* Default parameters - tuned to create interesting dynamics */
static const SVParams DEFAULT_SV_PARAMS = {
    0.97,        /* rho */
    -4.0,        /* mu_base */
    0.15,        /* sigma_base */
    -0.3,        /* gamma (negative = leverage effect) */
    5.0,         /* nu (fat tails) */
    1.0,         /* mu_wave */
    0.25,        /* sigma_extra */
    2000.0       /* period */
};

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double* returns;
    double* true_log_vol;
    double* true_vol;
    double* true_mu_t;       /* Time-varying mean */
    double* true_sigma_t;    /* Time-varying vol-of-vol */
    int* scenario_id;
    int n_ticks;
    
    /* Scenario info */
    int scenario_starts[10];
    const char* scenario_names[10];
    int n_scenarios;
    
    /* Stats */
    int n_tail_events;       /* Count of |return| > 4*vol */
    double max_return;
    double min_return;
} SyntheticData;

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double smooth_transition(double t, double center, double width) {
    return sigmoid((t - center) / width);
}

static SyntheticData* generate_test_data(int seed) {
    SyntheticData* data = (SyntheticData*)calloc(1, sizeof(SyntheticData));
    
    int n = 8000;
    data->n_ticks = n;
    data->returns = (double*)malloc(n * sizeof(double));
    data->true_log_vol = (double*)malloc(n * sizeof(double));
    data->true_vol = (double*)malloc(n * sizeof(double));
    data->true_mu_t = (double*)malloc(n * sizeof(double));
    data->true_sigma_t = (double*)malloc(n * sizeof(double));
    data->scenario_id = (int*)calloc(n, sizeof(int));
    
    pcg32_t rng = {(uint64_t)seed * 12345ULL + 1, (uint64_t)seed * 67890ULL | 1};
    
    SVParams p = DEFAULT_SV_PARAMS;
    
    /* Initial state */
    double log_vol = p.mu_base;
    double prev_return = 0.0;
    
    data->max_return = -1e10;
    data->min_return = 1e10;
    
    for (int t = 0; t < n; t++) {
        /*═══════════════════════════════════════════════════════════════════
         * SCENARIO-DEPENDENT TIME-VARYING PARAMETERS
         * (Smooth transitions, no discrete jumps)
         *═══════════════════════════════════════════════════════════════════*/
        
        double mu_t, sigma_t;
        int scenario;
        
        if (t < 1500) {
            /* Scenario 1: Calm Drift - low vol baseline */
            scenario = 0;
            mu_t = p.mu_base - 1.0;  /* Lower mean = calmer */
            sigma_t = p.sigma_base * 0.7;
            
        } else if (t < 2500) {
            /* Scenario 2: Building Tension - gradual increase */
            scenario = 1;
            double progress = (t - 1500) / 1000.0;
            mu_t = p.mu_base - 1.0 + 2.0 * progress;  /* -5 → -3 */
            sigma_t = p.sigma_base * (0.7 + 0.8 * progress);
            
        } else if (t < 3500) {
            /* Scenario 3: Volatility Storm - high vol, heavy tails */
            scenario = 2;
            mu_t = p.mu_base + 1.5;  /* High mean vol */
            sigma_t = p.sigma_base + p.sigma_extra;
            /* Add extra randomness to vol-of-vol */
            sigma_t *= (1.0 + 0.3 * sin(2.0 * 3.14159 * t / 200.0));
            
        } else if (t < 4500) {
            /* Scenario 4: Whipsaw - rapid oscillations */
            scenario = 3;
            double osc = sin(2.0 * 3.14159 * t / 150.0);
            mu_t = p.mu_base + 1.0 * osc;
            sigma_t = p.sigma_base + 0.15 * fabs(osc);
            
        } else if (t < 5500) {
            /* Scenario 5: Leverage Cascade - vol driven by negative returns */
            scenario = 4;
            mu_t = p.mu_base;
            sigma_t = p.sigma_base;
            /* Leverage effect will dominate here */
            
        } else if (t < 6500) {
            /* Scenario 6: Calm Return - slow decay */
            scenario = 5;
            double progress = (t - 5500) / 1000.0;
            mu_t = p.mu_base + 0.5 * (1.0 - progress);
            sigma_t = p.sigma_base * (1.0 - 0.3 * progress);
            
        } else {
            /* Scenario 7: Mixed Dynamics - combine everything */
            scenario = 6;
            double wave = sin(2.0 * 3.14159 * t / p.period);
            double tension = smooth_transition((double)t, 7200.0, 200.0) 
                           - smooth_transition((double)t, 7600.0, 200.0);
            mu_t = p.mu_base + p.mu_wave * wave + tension;
            sigma_t = p.sigma_base + p.sigma_extra * sigmoid(tension * 3.0);
        }
        
        data->true_mu_t[t] = mu_t;
        data->true_sigma_t[t] = sigma_t;
        data->scenario_id[t] = scenario;
        
        /*═══════════════════════════════════════════════════════════════════
         * EVOLVE STATE (Continuous SV with leverage)
         *═══════════════════════════════════════════════════════════════════*/
        
        /* Leverage effect: negative returns increase vol */
        double leverage = 0.0;
        if (t > 0) {
            double prev_vol = exp(log_vol / 2.0);
            leverage = p.gamma * prev_return / (prev_vol + 1e-8);
        }
        
        /* Vol innovation */
        double eps = pcg32_gaussian(&rng);
        
        /* SV evolution */
        log_vol = mu_t + p.rho * (log_vol - mu_t) + sigma_t * eps + leverage;
        
        /* Clamp to reasonable range */
        if (log_vol < -10.0) log_vol = -10.0;
        if (log_vol > 2.0) log_vol = 2.0;
        
        double vol = exp(log_vol / 2.0);
        
        /* Generate return with Student-t noise */
        double eta = pcg32_student_t(&rng, p.nu);
        double ret = vol * eta;
        
        /* Store */
        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        prev_return = ret;
        
        /* Track stats */
        if (ret > data->max_return) data->max_return = ret;
        if (ret < data->min_return) data->min_return = ret;
        if (fabs(ret) > 4.0 * vol) data->n_tail_events++;
    }
    
    /* Set scenario metadata */
    data->n_scenarios = 7;
    data->scenario_starts[0] = 0;     data->scenario_names[0] = "Calm Drift";
    data->scenario_starts[1] = 1500;  data->scenario_names[1] = "Building Tension";
    data->scenario_starts[2] = 2500;  data->scenario_names[2] = "Vol Storm";
    data->scenario_starts[3] = 3500;  data->scenario_names[3] = "Whipsaw";
    data->scenario_starts[4] = 4500;  data->scenario_names[4] = "Leverage Cascade";
    data->scenario_starts[5] = 5500;  data->scenario_names[5] = "Calm Return";
    data->scenario_starts[6] = 6500;  data->scenario_names[6] = "Mixed Dynamics";
    
    return data;
}

static void free_synthetic_data(SyntheticData* data) {
    if (!data) return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->true_mu_t);
    free(data->true_sigma_t);
    free(data->scenario_id);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double logvol_rmse;
    double vol_rmse;
    double logvol_mae;
    double logvol_bias;
    double coverage_90;      /* % of true values within 90% CI */
    double avg_latency_us;
} FilterMetrics;

static void compute_metrics(
    const SyntheticData* data,
    const float* est_logvol,
    const float* est_vol,
    FilterMetrics* metrics
) {
    double sum_sq_logvol = 0.0;
    double sum_sq_vol = 0.0;
    double sum_abs_logvol = 0.0;
    double sum_bias = 0.0;
    int n = data->n_ticks;
    
    for (int t = 0; t < n; t++) {
        double err_logvol = est_logvol[t] - data->true_log_vol[t];
        double err_vol = est_vol[t] - data->true_vol[t];
        
        sum_sq_logvol += err_logvol * err_logvol;
        sum_sq_vol += err_vol * err_vol;
        sum_abs_logvol += fabs(err_logvol);
        sum_bias += err_logvol;
    }
    
    metrics->logvol_rmse = sqrt(sum_sq_logvol / n);
    metrics->vol_rmse = sqrt(sum_sq_vol / n);
    metrics->logvol_mae = sum_abs_logvol / n;
    metrics->logvol_bias = sum_bias / n;
}

static void compute_scenario_metrics(
    const SyntheticData* data,
    const float* est_logvol,
    int scenario,
    double* rmse,
    double* mae
) {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    int count = 0;
    
    for (int t = 0; t < data->n_ticks; t++) {
        if (data->scenario_id[t] == scenario) {
            double err = est_logvol[t] - data->true_log_vol[t];
            sum_sq += err * err;
            sum_abs += fabs(err);
            count++;
        }
    }
    
    if (count > 0) {
        *rmse = sqrt(sum_sq / count);
        *mae = sum_abs / count;
    } else {
        *rmse = 0.0;
        *mae = 0.0;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CSV OUTPUT
 *═══════════════════════════════════════════════════════════════════════════*/

static void write_csv(
    const char* filename,
    const SyntheticData* data,
    const float* est_logvol,
    const float* est_vol
) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "t,return,true_logvol,true_vol,est_logvol,est_vol,true_mu,true_sigma,scenario\n");
    
    for (int t = 0; t < data->n_ticks; t++) {
        fprintf(f, "%d,%.8f,%.6f,%.8f,%.6f,%.8f,%.6f,%.6f,%d\n",
                t,
                data->returns[t],
                data->true_log_vol[t],
                data->true_vol[t],
                est_logvol[t],
                est_vol[t],
                data->true_mu_t[t],
                data->true_sigma_t[t],
                data->scenario_id[t]);
    }
    
    fclose(f);
    printf("  Written: %s\n", filename);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN TEST
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║          SVPF CUDA TEST - Continuous SV (SVPF-Favorable DGP)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");
    
    int seed = 42;
    int n_particles = 512;
    int n_stein = 10;
    float nu = 50.0f;
    int use_adaptive = 1;  /* Default: use adaptive improvements */
    
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            n_particles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stein") == 0 && i + 1 < argc) {
            n_stein = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nu") == 0 && i + 1 < argc) {
            nu = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive") == 0) {
            use_adaptive = 1;
        } else if (strcmp(argv[i], "--vanilla") == 0) {
            use_adaptive = 0;
        }
    }
    
    printf("Configuration:\n");
    printf("  Seed:        %d\n", seed);
    printf("  Particles:   %d\n", n_particles);
    printf("  Stein steps: %d\n", n_stein);
    printf("  Student-t ν: %.1f\n", nu);
    printf("  Mode:        %s\n", use_adaptive ? "ADAPTIVE" : "VANILLA");
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * GENERATE DATA
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("Generating synthetic data (continuous SV model)...\n");
    SyntheticData* data = generate_test_data(seed);
    
    printf("  Ticks:       %d\n", data->n_ticks);
    printf("  Scenarios:   %d\n", data->n_scenarios);
    printf("  Tail events: %d (|r| > 4σ)\n", data->n_tail_events);
    printf("  Return range: [%.4f, %.4f]\n", data->min_return, data->max_return);
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * ALLOCATE OUTPUTS
     *═══════════════════════════════════════════════════════════════════════*/
    
    int n = data->n_ticks;
    float* h_returns = (float*)malloc(n * sizeof(float));
    float* h_loglik = (float*)malloc(n * sizeof(float));
    float* h_vol = (float*)malloc(n * sizeof(float));
    float* h_logvol = (float*)malloc(n * sizeof(float));
    
    /* Convert returns to float */
    for (int t = 0; t < n; t++) {
        h_returns[t] = (float)data->returns[t];
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * CREATE SVPF FILTER
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("Creating SVPF filter...\n");
    SVPFState* filter = svpf_create(n_particles, n_stein, nu, NULL);
    if (!filter) {
        fprintf(stderr, "Failed to create SVPF filter\n");
        return 1;
    }
    
    /* Parameters matching DGP */
    SVPFParams params;
    params.rho = 0.97f;
    params.sigma_z = 0.15f;    /* Base vol-of-vol */
    params.mu = -4.0f;         /* Base mean */
    params.gamma = -0.3f;      /* Leverage */
    
    svpf_initialize(filter, &params, seed);
    
    /* Configure adaptive SVPF settings */
     filter->use_svld = 1;
        filter->use_annealing = 1;
        filter->n_anneal_steps = 3;
        filter->temperature = 0.55f;
        filter->rmsprop_rho = 0.94f;
        filter->rmsprop_eps = 1e-6f;
        
        filter->use_mim = 1;
        filter->mim_jump_prob = 0.25f;
        filter->mim_jump_scale = 9.0f;
    
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
        filter->lik_offset = 0.35f;  // No correction - test if model is now consistent

         // === KSD-based Adaptive Stein Steps ===
        // Replaces fixed n_stein_steps with convergence-based early stopping
        // KSD (Kernel Stein Discrepancy) computed in same O(N²) pass - zero extra cost
        filter->stein_min_steps = 12;              // Always run at least 4 (RMSProp warmup)
        filter->stein_max_steps = 20;             // Cap at 12 (crisis budget)
        filter->ksd_improvement_threshold = 0.03; // Stop if <5% relative improvement
    
    printf("  Filter initialized.\n");
    printf("  Mode: %s\n", use_adaptive ? "SVLD + MIM + Asym-ρ + Guide" : "VANILLA SVGD");
    printf("  Temperature: %.2f\n", filter->temperature);
    if (filter->use_mim) {
        printf("  MIM: %.0f%% @ %.1fx scale\n", filter->mim_jump_prob * 100, filter->mim_jump_scale);
    }
    if (filter->use_asymmetric_rho) {
        printf("  Asymmetric ρ: up=%.2f, down=%.2f\n", filter->rho_up, filter->rho_down);
    }
    if (filter->use_guide) {
        printf("  EKF Guide: strength=%.2f\n", filter->guide_strength);
    }
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * RUN FILTER
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("Running SVPF on %d ticks...\n", n);
    
    double t_start = get_time_us();
    
    float y_prev = 0.0f;
    for (int t = 0; t < n; t++) {
        float y_t = h_returns[t];
        
        if (use_adaptive) {
            /* Use adaptive step with all improvements */
            svpf_step_adaptive(filter, y_t, y_prev, &params,
                              &h_loglik[t], &h_vol[t], &h_logvol[t]);
        } else {
            /* Use vanilla step */
            SVPFResult result;
            svpf_step(filter, y_t, &params, &result);
            h_loglik[t] = result.log_lik_increment;
            h_vol[t] = result.vol_mean;
            h_logvol[t] = result.h_mean;
        }
        
        y_prev = y_t;
    }
    
    double t_end = get_time_us();
    double elapsed_ms = (t_end - t_start) / 1000.0;
    double us_per_step = (t_end - t_start) / n;
    
    printf("  Elapsed:     %.2f ms\n", elapsed_ms);
    printf("  Per-step:    %.2f μs\n", us_per_step);
    printf("  Throughput:  %.0f steps/sec\n", n / (elapsed_ms / 1000.0));
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * COMPUTE METRICS
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  VOLATILITY ESTIMATION\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    
    FilterMetrics metrics;
    compute_metrics(data, h_logvol, h_vol, &metrics);
    
    printf("    Log-Vol RMSE:       %.4f\n", metrics.logvol_rmse);
    printf("    Log-Vol MAE:        %.4f\n", metrics.logvol_mae);
    printf("    Log-Vol Bias:       %.4f\n", metrics.logvol_bias);
    printf("    Vol RMSE:           %.6f\n", metrics.vol_rmse);
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * PER-SCENARIO BREAKDOWN
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    printf("  PER-SCENARIO BREAKDOWN\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-20s %10s %10s\n", "Scenario", "RMSE", "MAE");
    printf("  %-20s %10s %10s\n", "--------", "----", "---");
    
    for (int s = 0; s < data->n_scenarios; s++) {
        double rmse, mae;
        compute_scenario_metrics(data, h_logvol, s, &rmse, &mae);
        printf("  %-20s %10.4f %10.4f\n", data->scenario_names[s], rmse, mae);
    }
    printf("\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * WRITE OUTPUT CSV
     *═══════════════════════════════════════════════════════════════════════*/
    
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    printf("  OUTPUT FILES\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    
    write_csv("svpf_results.csv", data, h_logvol, h_vol);
    
    /* Also write summary */
    FILE* summary = fopen("svpf_summary.txt", "w");
    if (summary) {
        fprintf(summary, "SVPF Test Results (Continuous SV DGP)\n");
        fprintf(summary, "=====================================\n\n");
        fprintf(summary, "Configuration:\n");
        fprintf(summary, "  Particles:   %d\n", n_particles);
        fprintf(summary, "  Stein steps: %d\n", n_stein);
        fprintf(summary, "  Student-t ν: %.1f\n", nu);
        fprintf(summary, "\nOverall Metrics:\n");
        fprintf(summary, "  Log-Vol RMSE: %.4f\n", metrics.logvol_rmse);
        fprintf(summary, "  Log-Vol MAE:  %.4f\n", metrics.logvol_mae);
        fprintf(summary, "  Vol RMSE:     %.6f\n", metrics.vol_rmse);
        fprintf(summary, "\nPer-Scenario RMSE:\n");
        for (int s = 0; s < data->n_scenarios; s++) {
            double rmse, mae;
            compute_scenario_metrics(data, h_logvol, s, &rmse, &mae);
            fprintf(summary, "  %-20s %.4f\n", data->scenario_names[s], rmse);
        }
        fprintf(summary, "\nPerformance:\n");
        fprintf(summary, "  %.2f μs/step (%.0f steps/sec)\n", us_per_step, n / (elapsed_ms / 1000.0));
        fclose(summary);
        printf("  Written: svpf_summary.txt\n");
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  DONE\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    
    /*═══════════════════════════════════════════════════════════════════════
     * CLEANUP
     *═══════════════════════════════════════════════════════════════════════*/
    
    svpf_destroy(filter);
    free_synthetic_data(data);
    free(h_returns);
    free(h_loglik);
    free(h_vol);
    free(h_logvol);
    
    return 0;
}
