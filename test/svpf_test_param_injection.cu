/**
 * @file svpf_test_param_injection.cu
 * @brief Test harness for SVPF parameter injection during regime changes
 *
 * Tests whether injecting correct prior parameters mid-regime helps or hurts
 * SVPF accuracy, and whether resetting optimizer state is necessary.
 *
 * Sequence structure (3 regimes, configurable lengths):
 *
 *   |--- regime00 (matched) ---|--- regime01 (misspec) ---|--- regime02 ---|
 *   0                       T0  T0                     T1  T1           T_end
 *                                        ^
 *                                   injection point
 *                                  (mid regime01)
 *
 * Conditions tested:
 *   ORACLE    — correct params for each regime, injected at regime boundary
 *   STALE     — regime00 params held throughout (pure Stein robustness)
 *   INJECT    — regime00 params until mid-regime01, then regime01 params injected
 *   INJECT_RESET — same as INJECT but also resets RMSProp + bandwidth EMA
 *
 * Per-segment RMSE is reported:
 *   - regime00 (steady state, all conditions identical)
 *   - regime01_pre  (before injection, STALE == INJECT)
 *   - regime01_post (after injection, INJECT vs STALE diverge)
 *   - regime02       (does injection help downstream?)
 *
 * Build:
 *   nvcc -O2 svpf_test_param_injection.cu svpf_optimized_graph.cu svpf_opt_kernels.cu \
 *        svpf_test_framework.cu -lcurand -o test_param_injection
 *
 * Usage:
 *   ./test_param_injection [n_seeds]
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

// =============================================================================
// Configuration
// =============================================================================

typedef struct {
    const char* name;

    // Regime 00 (initial — filter starts matched to this)
    SVPFParams params_r00;

    // Regime 01 (regime change — filter is misspecified until injection)
    SVPFParams params_r01;

    // Regime 02 (second change — tests downstream effect)
    SVPFParams params_r02;

    // Timing
    int regime00_len;      // Ticks in regime00
    int regime01_len;      // Ticks in regime01
    int regime02_len;      // Ticks in regime02
    int warmup;            // Warmup ticks (from regime00, not scored)

    // DGP observation model
    float true_nu;         // Student-t df for observations (INFINITY = Gaussian)
} InjectionScenario;

typedef enum {
    COND_ORACLE,           // Correct params at each regime boundary
    COND_STALE,            // regime00 params forever
    COND_INJECT,           // Inject regime01 params at mid-regime01
    COND_INJECT_RESET,     // Inject + reset optimizer state
    COND_COUNT
} InjectionCondition;

static const char* condition_names[COND_COUNT] = {
    "ORACLE",
    "STALE",
    "INJECT",
    "INJECT_RESET"
};

typedef struct {
    float rmse_regime00;
    float rmse_regime01_pre;
    float rmse_regime01_post;
    float rmse_regime02;
    float rmse_total;
    float bias_regime01_post;
    float bias_regime02;
    float runtime_ms;
} InjectionMetrics;

typedef struct {
    float mean;
    float std;
    float median;
} SimpleStat;

typedef struct {
    SimpleStat rmse_regime00;
    SimpleStat rmse_regime01_pre;
    SimpleStat rmse_regime01_post;
    SimpleStat rmse_regime02;
    SimpleStat rmse_total;
    SimpleStat bias_regime01_post;
    SimpleStat bias_regime02;
    int n_seeds;
} AggregateInjectionMetrics;

// =============================================================================
// Scenarios
// =============================================================================

static InjectionScenario scenario_sigma_z_jump(void) {
    InjectionScenario s = {};
    s.name = "sigma_z_jump (0.10 -> 0.22 -> 0.12)";
    s.params_r00 = {0.98f, 0.10f, -4.5f, 0.0f};
    s.params_r01 = {0.98f, 0.22f, -4.5f, 0.0f};  // sigma_z doubles
    s.params_r02 = {0.98f, 0.12f, -4.5f, 0.0f};  // back to near-baseline
    s.regime00_len = 2000;
    s.regime01_len = 2000;
    s.regime02_len = 1000;
    s.warmup = 200;
    s.true_nu = INFINITY;
    return s;
}

static InjectionScenario scenario_mu_shift(void) {
    InjectionScenario s = {};
    s.name = "mu_shift (-4.5 -> -3.0 -> -4.0)";
    s.params_r00 = {0.98f, 0.10f, -4.5f, 0.0f};
    s.params_r01 = {0.98f, 0.10f, -3.0f, 0.0f};  // mu jumps up (higher vol)
    s.params_r02 = {0.98f, 0.10f, -4.0f, 0.0f};  // partial revert
    s.regime00_len = 2000;
    s.regime01_len = 2000;
    s.regime02_len = 1000;
    s.warmup = 200;
    s.true_nu = INFINITY;
    return s;
}

static InjectionScenario scenario_rho_change(void) {
    InjectionScenario s = {};
    s.name = "rho_change (0.98 -> 0.92 -> 0.96)";
    s.params_r00 = {0.98f, 0.10f, -4.5f, 0.0f};
    s.params_r01 = {0.92f, 0.10f, -4.5f, 0.0f};  // faster mean reversion
    s.params_r02 = {0.96f, 0.10f, -4.5f, 0.0f};  // intermediate
    s.regime00_len = 2000;
    s.regime01_len = 2000;
    s.regime02_len = 1000;
    s.warmup = 200;
    s.true_nu = INFINITY;
    return s;
}

static InjectionScenario scenario_combined(void) {
    InjectionScenario s = {};
    s.name = "combined (calm -> crisis -> moderate)";
    s.params_r00 = {0.98f, 0.10f, -5.0f, 0.0f};
    s.params_r01 = {0.995f, 0.20f, -3.5f, 0.0f};  // crisis: persistent, high vol-of-vol, high level
    s.params_r02 = {0.97f, 0.14f, -4.0f, 0.0f};   // moderate recovery
    s.regime00_len = 2000;
    s.regime01_len = 2000;
    s.regime02_len = 1000;
    s.warmup = 200;
    s.true_nu = INFINITY;
    return s;
}

static InjectionScenario scenario_fat_tails_sigma(void) {
    InjectionScenario s = {};
    s.name = "fat_tails + sigma_z_jump (nu=5)";
    s.params_r00 = {0.98f, 0.10f, -4.5f, 0.0f};
    s.params_r01 = {0.98f, 0.22f, -4.5f, 0.0f};
    s.params_r02 = {0.98f, 0.12f, -4.5f, 0.0f};
    s.regime00_len = 2000;
    s.regime01_len = 2000;
    s.regime02_len = 1000;
    s.warmup = 200;
    s.true_nu = 5.0f;
    return s;
}

// =============================================================================
// DGP: Multi-regime data generation
// =============================================================================

static float randn_host(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x;
    float u1 = (float)(x & 0xFFFFFFFF) / 4294967296.0f + 1e-10f;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x;
    float u2 = (float)(x & 0xFFFFFFFF) / 4294967296.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static float rand_student_t_host(uint64_t* state, float nu) {
    if (nu > 1e6f || !isfinite(nu)) return randn_host(state);
    float z = randn_host(state);
    float chi2 = 0.0f;
    for (int i = 0; i < (int)nu; i++) {
        float g = randn_host(state);
        chi2 += g * g;
    }
    return z * sqrtf(nu / chi2);
}

static void generate_multiregime_data(
    const InjectionScenario* sc,
    uint64_t seed,
    float* h_true,      // [T_total]
    float* y_obs        // [T_total]
) {
    uint64_t rng = seed;
    int T = sc->regime00_len + sc->regime01_len + sc->regime02_len;
    int t0 = sc->regime00_len;
    int t1 = t0 + sc->regime01_len;

    // Start at stationary distribution of regime00
    float mu = sc->params_r00.mu;
    float rho = sc->params_r00.rho;
    float sigma_z = sc->params_r00.sigma_z;
    float h = mu + sigma_z / sqrtf(1.0f - rho * rho + 1e-6f) * randn_host(&rng);

    for (int t = 0; t < T; t++) {
        // Select regime parameters
        const SVPFParams* p;
        if (t < t0)      p = &sc->params_r00;
        else if (t < t1)  p = &sc->params_r01;
        else               p = &sc->params_r02;

        float eps = randn_host(&rng);
        h = p->mu + p->rho * (h - p->mu) + p->sigma_z * eps;
        h = fmaxf(fminf(h, 2.0f), -12.0f);
        h_true[t] = h;

        float vol = expf(h * 0.5f);
        float z = rand_student_t_host(&rng, sc->true_nu);
        y_obs[t] = vol * z;
    }
}

// =============================================================================
// SVPF Reset Helpers
// =============================================================================

static void svpf_reset_optimizer(SVPFState* state) {
    int n = state->n_particles;
    cudaMemset(state->d_grad_v, 0, n * sizeof(float));

    SVPFOptimizedState* opt = &state->opt_backend;
    if (opt->initialized) {
        float zero = 0.0f;
        cudaMemcpy(opt->d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
    }
}

// =============================================================================
// Metric Helpers
// =============================================================================

static float compute_rmse(const float* est, const float* truth, int start, int end) {
    if (end <= start) return 0.0f;
    float sum_sq = 0.0f;
    for (int i = start; i < end; i++) {
        float d = est[i] - truth[i];
        sum_sq += d * d;
    }
    return sqrtf(sum_sq / (float)(end - start));
}

static float compute_bias(const float* est, const float* truth, int start, int end) {
    if (end <= start) return 0.0f;
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += est[i] - truth[i];
    }
    return sum / (float)(end - start);
}

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static SimpleStat aggregate(const float* vals, int n) {
    SimpleStat s = {};
    if (n == 0) return s;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += vals[i];
    s.mean = sum / n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) var += (vals[i] - s.mean) * (vals[i] - s.mean);
    s.std = sqrtf(var / fmaxf(n - 1.0f, 1.0f));
    float* sorted = (float*)malloc(n * sizeof(float));
    memcpy(sorted, vals, n * sizeof(float));
    qsort(sorted, n, sizeof(float), cmp_float);
    s.median = sorted[n / 2];
    free(sorted);
    return s;
}

// =============================================================================
// Single Run (one seed, one condition)
// =============================================================================

static InjectionMetrics run_single(
    const InjectionScenario* sc,
    InjectionCondition cond,
    const float* h_true,
    const float* y_obs,
    int n_particles,
    uint64_t seed
) {
    InjectionMetrics m = {};
    int T = sc->regime00_len + sc->regime01_len + sc->regime02_len;
    int t0 = sc->regime00_len;
    int t1 = t0 + sc->regime01_len;
    int t_inject = t0 + sc->regime01_len / 2;  // Mid-regime01

    float* h_est = (float*)malloc(T * sizeof(float));

    // Create filter with regime00 params (all conditions start matched)
    SVPFState* state = svpf_create(n_particles, 8, 5.0f, NULL);
    SVPFParams active_params = sc->params_r00;
    svpf_initialize(state, &active_params, (unsigned long long)seed);

    auto t_start = std::chrono::high_resolution_clock::now();

    float y_prev = 0.0f;
    for (int t = 0; t < T; t++) {

        // === Parameter injection logic ===
        switch (cond) {
            case COND_ORACLE:
                // Inject correct params at each regime boundary
                if (t == t0) active_params = sc->params_r01;
                if (t == t1) active_params = sc->params_r02;
                break;

            case COND_STALE:
                // Never update — keep regime00 params forever
                break;

            case COND_INJECT:
                // Inject regime01 params at mid-point
                if (t == t_inject) {
                    active_params = sc->params_r01;
                }
                // Inject regime02 params at regime boundary
                if (t == t1) {
                    active_params = sc->params_r02;
                }
                break;

            case COND_INJECT_RESET:
                // Same injection timing, but also reset optimizer
                if (t == t_inject) {
                    active_params = sc->params_r01;
                    svpf_reset_optimizer(state);
                }
                if (t == t1) {
                    active_params = sc->params_r02;
                    svpf_reset_optimizer(state);
                }
                break;

            default:
                break;
        }

        float loglik, vol, h_mean;
        svpf_step_graph(state, y_obs[t], y_prev, &active_params, &loglik, &vol, &h_mean);
        h_est[t] = h_mean;
        y_prev = y_obs[t];
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    m.runtime_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Per-segment metrics (skip warmup in regime00)
    int eval_start = sc->warmup;
    m.rmse_regime00     = compute_rmse(h_est, h_true, eval_start, t0);
    m.rmse_regime01_pre = compute_rmse(h_est, h_true, t0, t_inject);
    m.rmse_regime01_post = compute_rmse(h_est, h_true, t_inject, t1);
    m.rmse_regime02     = compute_rmse(h_est, h_true, t1, T);
    m.rmse_total        = compute_rmse(h_est, h_true, eval_start, T);

    m.bias_regime01_post = compute_bias(h_est, h_true, t_inject, t1);
    m.bias_regime02      = compute_bias(h_est, h_true, t1, T);

    svpf_destroy(state);
    free(h_est);
    return m;
}

// =============================================================================
// Print Results
// =============================================================================

static void print_header(const InjectionScenario* sc) {
    int T = sc->regime00_len + sc->regime01_len + sc->regime02_len;
    int t0 = sc->regime00_len;
    int t1 = t0 + sc->regime01_len;
    int t_inject = t0 + sc->regime01_len / 2;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Parameter Injection Test: %-42s ║\n", sc->name);
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Timeline: regime00[0-%d] → regime01[%d-%d] → regime02[%d-%d]       \n",
           t0 - 1, t0, t1 - 1, t1, T - 1);
    printf("║  Injection point: t=%d (mid regime01)                               \n", t_inject);
    printf("║  R00: rho=%.3f sigma=%.3f mu=%.2f                                  \n",
           sc->params_r00.rho, sc->params_r00.sigma_z, sc->params_r00.mu);
    printf("║  R01: rho=%.3f sigma=%.3f mu=%.2f                                  \n",
           sc->params_r01.rho, sc->params_r01.sigma_z, sc->params_r01.mu);
    printf("║  R02: rho=%.3f sigma=%.3f mu=%.2f                                  \n",
           sc->params_r02.rho, sc->params_r02.sigma_z, sc->params_r02.mu);
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
}

static void print_condition_results(
    const char* cond_name,
    const AggregateInjectionMetrics* agg
) {
    printf("║  %-14s │ R00     │ R01_pre │ R01_post │ R02     │ Total   ║\n", cond_name);
    printf("║  RMSE  mean    │ %7.4f │ %7.4f │ %7.4f  │ %7.4f │ %7.4f ║\n",
           agg->rmse_regime00.mean, agg->rmse_regime01_pre.mean,
           agg->rmse_regime01_post.mean, agg->rmse_regime02.mean,
           agg->rmse_total.mean);
    printf("║  RMSE  std     │ %7.4f │ %7.4f │ %7.4f  │ %7.4f │ %7.4f ║\n",
           agg->rmse_regime00.std, agg->rmse_regime01_pre.std,
           agg->rmse_regime01_post.std, agg->rmse_regime02.std,
           agg->rmse_total.std);
    printf("║  Bias(post)    │         │         │ %+7.4f  │ %+7.4f │         ║\n",
           agg->bias_regime01_post.mean, agg->bias_regime02.mean);
    printf("╟────────────────┼─────────┼─────────┼──────────┼─────────┼─────────╢\n");
}

static void print_delta_row(
    const char* label,
    const AggregateInjectionMetrics* test,
    const AggregateInjectionMetrics* base
) {
    // Δ = test - base (negative means test is better)
    printf("║  Δ %-11s │ %+6.3f  │ %+6.3f  │ %+6.3f   │ %+6.3f  │ %+6.3f  ║\n",
           label,
           test->rmse_regime00.mean - base->rmse_regime00.mean,
           test->rmse_regime01_pre.mean - base->rmse_regime01_pre.mean,
           test->rmse_regime01_post.mean - base->rmse_regime01_post.mean,
           test->rmse_regime02.mean - base->rmse_regime02.mean,
           test->rmse_total.mean - base->rmse_total.mean);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int n_seeds = 50;
    int n_particles = 512;

    if (argc > 1) n_seeds = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);

    printf("SVPF Parameter Injection Test\n");
    printf("Seeds: %d, Particles: %d\n", n_seeds, n_particles);

    // All scenarios
    InjectionScenario scenarios[] = {
        scenario_sigma_z_jump(),
        scenario_mu_shift(),
        scenario_rho_change(),
        scenario_combined(),
        scenario_fat_tails_sigma(),
    };
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    for (int sc_idx = 0; sc_idx < n_scenarios; sc_idx++) {
        const InjectionScenario* sc = &scenarios[sc_idx];
        int T = sc->regime00_len + sc->regime01_len + sc->regime02_len;

        // Allocate per-seed metric storage
        float* rmse_r00[COND_COUNT], *rmse_r01_pre[COND_COUNT];
        float* rmse_r01_post[COND_COUNT], *rmse_r02[COND_COUNT], *rmse_total[COND_COUNT];
        float* bias_r01_post[COND_COUNT], *bias_r02[COND_COUNT];
        for (int c = 0; c < COND_COUNT; c++) {
            rmse_r00[c]      = (float*)malloc(n_seeds * sizeof(float));
            rmse_r01_pre[c]  = (float*)malloc(n_seeds * sizeof(float));
            rmse_r01_post[c] = (float*)malloc(n_seeds * sizeof(float));
            rmse_r02[c]      = (float*)malloc(n_seeds * sizeof(float));
            rmse_total[c]    = (float*)malloc(n_seeds * sizeof(float));
            bias_r01_post[c] = (float*)malloc(n_seeds * sizeof(float));
            bias_r02[c]      = (float*)malloc(n_seeds * sizeof(float));
        }

        float* h_true = (float*)malloc(T * sizeof(float));
        float* y_obs  = (float*)malloc(T * sizeof(float));

        for (int s = 0; s < n_seeds; s++) {
            uint64_t seed = 42 + s * 7919;  // Different prime spacing

            // Generate data once per seed (shared across conditions)
            generate_multiregime_data(sc, seed, h_true, y_obs);

            for (int c = 0; c < COND_COUNT; c++) {
                InjectionMetrics m = run_single(
                    sc, (InjectionCondition)c, h_true, y_obs, n_particles, seed);

                rmse_r00[c][s]      = m.rmse_regime00;
                rmse_r01_pre[c][s]  = m.rmse_regime01_pre;
                rmse_r01_post[c][s] = m.rmse_regime01_post;
                rmse_r02[c][s]      = m.rmse_regime02;
                rmse_total[c][s]    = m.rmse_total;
                bias_r01_post[c][s] = m.bias_regime01_post;
                bias_r02[c][s]      = m.bias_regime02;
            }

            if ((s + 1) % 10 == 0 || s == 0) {
                printf("  [%s] seed %d/%d\n", sc->name, s + 1, n_seeds);
            }
        }

        // Aggregate
        AggregateInjectionMetrics agg[COND_COUNT];
        for (int c = 0; c < COND_COUNT; c++) {
            agg[c].rmse_regime00     = aggregate(rmse_r00[c], n_seeds);
            agg[c].rmse_regime01_pre = aggregate(rmse_r01_pre[c], n_seeds);
            agg[c].rmse_regime01_post = aggregate(rmse_r01_post[c], n_seeds);
            agg[c].rmse_regime02     = aggregate(rmse_r02[c], n_seeds);
            agg[c].rmse_total        = aggregate(rmse_total[c], n_seeds);
            agg[c].bias_regime01_post = aggregate(bias_r01_post[c], n_seeds);
            agg[c].bias_regime02     = aggregate(bias_r02[c], n_seeds);
            agg[c].n_seeds = n_seeds;
        }

        // Print
        print_header(sc);
        for (int c = 0; c < COND_COUNT; c++) {
            print_condition_results(condition_names[c], &agg[c]);
        }

        printf("║                                                                       ║\n");
        printf("║  Deltas vs STALE (negative = improvement):                            ║\n");
        printf("╟────────────────┼─────────┼─────────┼──────────┼─────────┼─────────╢\n");
        print_delta_row("ORACLE", &agg[COND_ORACLE], &agg[COND_STALE]);
        print_delta_row("INJECT", &agg[COND_INJECT], &agg[COND_STALE]);
        print_delta_row("INJ+RST", &agg[COND_INJECT_RESET], &agg[COND_STALE]);
        printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

        // Cleanup
        for (int c = 0; c < COND_COUNT; c++) {
            free(rmse_r00[c]); free(rmse_r01_pre[c]); free(rmse_r01_post[c]);
            free(rmse_r02[c]); free(rmse_total[c]);
            free(bias_r01_post[c]); free(bias_r02[c]);
        }
        free(h_true);
        free(y_obs);
    }

    printf("Done.\n");
    return 0;
}
