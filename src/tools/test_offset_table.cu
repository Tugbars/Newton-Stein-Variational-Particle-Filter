// =============================================================================
// EMPIRICAL lik_offset(ν) TABLE GENERATOR
// =============================================================================
//
// For a fixed SVPF config (N particles, K Stein steps), the total bias at
// offset=0 is deterministic for each ν. This test measures it by running
// SVPF on synthetic oracle-matched data with offset=0, then recording
// the average bias. That bias IS the required offset.
//
// Output: a C lookup table mapping ν → total_lik_offset for production use.
//
// Build: nvcc -O3 test_offset_table.cu svpf_opt_kernels.cu \
//        svpf_optimized_graph.cu -o test_offset_table -lcurand
// =============================================================================

#include "svpf.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// =============================================================================
// PRNG
// =============================================================================

static inline float randf(unsigned int* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (float)((*seed >> 16) & 0x7FFF) / 32768.0f;
}

static inline float randn(unsigned int* seed) {
    float u1 = randf(seed) + 1e-10f;
    float u2 = randf(seed);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

// =============================================================================
// Generate standard SV data (Gaussian state + Gaussian obs)
// =============================================================================

static void gen_sv_gaussian(std::vector<float>& ret, std::vector<float>& true_h,
                             int n, float rho, float sigma_z, float mu,
                             unsigned int seed) {
    float h = mu;
    for (int i = 0; i < n; i++) {
        float eps = randn(&seed);
        h = mu + rho * (h - mu) + sigma_z * eps;
        true_h.push_back(h);
        float eta = randn(&seed);
        ret.push_back(expf(h * 0.5f) * eta);
    }
}

// =============================================================================
// Measure bias for a given ν with offset=0
// =============================================================================

struct BiasResult {
    double bias;        // average (h_est - h_true)
    double rmse;
    double std_bias;    // std of per-run bias estimates
    int n_runs;
};

static BiasResult measure_bias(float nu, int n_particles, int n_stein,
                                float rho, float sigma_z, float mu,
                                int n_ticks, int n_runs) {
    double bias_sum = 0, bias_sq_sum = 0, rmse_sum = 0;

    for (int run = 0; run < n_runs; run++) {
        // Generate fresh data each run
        std::vector<float> returns, true_h;
        gen_sv_gaussian(returns, true_h, n_ticks, rho, sigma_z, mu, 42 + run * 7919);

        // Create SVPF with offset=0
        SVPFParams params;
        params.rho = rho;
        params.sigma_z = sigma_z;
        params.mu = mu;
        params.gamma = 0.0f;

        SVPFState* filter = svpf_create(n_particles, n_stein, nu, nullptr);
        svpf_initialize(filter, &params, 123 + run * 3571);

        // Force offset=0
        filter->lik_offset = 0.0f;
        filter->use_adaptive_offset = 0;

        // Run filter
        float y_prev = 0.0f;
        int skip = 50;  // warmup
        double sum_err = 0, sum_sq = 0;
        int count = 0;

        for (int t = 0; t < n_ticks; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(filter, returns[t], y_prev, &params, &loglik, &vol, &h_mean);

            if (t >= skip && !std::isnan(h_mean) && !std::isinf(h_mean)) {
                double err = (double)h_mean - (double)true_h[t];
                sum_err += err;
                sum_sq += err * err;
                count++;
            }
            y_prev = returns[t];
        }

        svpf_destroy(filter);

        if (count > 0) {
            double run_bias = sum_err / count;
            double run_rmse = sqrt(sum_sq / count);
            bias_sum += run_bias;
            bias_sq_sum += run_bias * run_bias;
            rmse_sum += run_rmse;
        }
    }

    BiasResult r;
    r.n_runs = n_runs;
    r.bias = bias_sum / n_runs;
    r.rmse = rmse_sum / n_runs;
    r.std_bias = sqrt(bias_sq_sum / n_runs - r.bias * r.bias);
    return r;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Default config — should match your production SVPF config
    int n_particles = 512;
    int n_stein     = 8;
    float rho       = 0.98f;
    float sigma_z   = 0.15f;
    float mu        = -4.5f;
    int n_ticks     = 2000;   // long series for stable estimate
    int n_runs      = 10;     // multiple runs to estimate std

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--particles") && i+1<argc) n_particles = atoi(argv[++i]);
        if (!strcmp(argv[i], "--stein")     && i+1<argc) n_stein = atoi(argv[++i]);
        if (!strcmp(argv[i], "--ticks")     && i+1<argc) n_ticks = atoi(argv[++i]);
        if (!strcmp(argv[i], "--runs")      && i+1<argc) n_runs = atoi(argv[++i]);
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  EMPIRICAL lik_offset(ν) TABLE GENERATOR\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Config: %d particles, %d Stein steps\n", n_particles, n_stein);
    printf("  DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f  (Gaussian state+obs)\n", rho, sigma_z, mu);
    printf("  Per ν: %d ticks × %d runs, offset=0, skip first 50\n", n_ticks, n_runs);
    printf("  Bias measured = required lik_offset for that ν\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    // ν grid: dense near heavy tails, sparse near Gaussian
    float nu_values[] = {
        2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 12.0f, 15.0f, 20.0f, 25.0f, 30.0f, 40.0f, 50.0f,
        75.0f, 100.0f, 200.0f, 500.0f
    };
    int n_nu = sizeof(nu_values) / sizeof(nu_values[0]);

    printf("  %6s  %10s  %10s  %10s  %10s\n",
           "ν", "bias", "±std", "offset", "RMSE");
    printf("  %6s  %10s  %10s  %10s  %10s\n",
           "──────", "──────────", "──────────", "──────────", "──────────");

    // Store results for C table generation
    float measured_offsets[256];
    float measured_nus[256];
    int n_measured = 0;

    for (int i = 0; i < n_nu; i++) {
        float nu = nu_values[i];
        BiasResult r = measure_bias(nu, n_particles, n_stein, rho, sigma_z, mu,
                                     n_ticks, n_runs);
        float offset = -(float)r.bias;  // offset cancels the bias

        printf("  %6.1f  %+10.6f  %10.6f  %+10.6f  %10.4f\n",
               nu, r.bias, r.std_bias, offset, r.rmse);

        measured_nus[n_measured] = nu;
        measured_offsets[n_measured] = offset;
        n_measured++;
    }

    // Print C lookup table
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  C LOOKUP TABLE — TOTAL lik_offset (score + transport)\n");
    printf("  Config: %d particles, %d Stein steps\n", n_particles, n_stein);
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    printf("// Total lik_offset (expected score + Stein transport) for SVPF\n");
    printf("// Config: %d particles, %d Stein steps, ρ=%.2f, σ_z=%.2f, μ=%.1f\n",
           n_particles, n_stein, rho, sigma_z, mu);
    printf("// Measured empirically on Gaussian DGP with offset=0\n");
    printf("// Usage: offset = interp(nu, NU_TABLE, OFFSET_TABLE, N_TABLE)\n");
    printf("#define OFFSET_TABLE_SIZE %d\n", n_measured);
    printf("static const float OFFSET_NU_TABLE[OFFSET_TABLE_SIZE] = {\n    ");
    for (int i = 0; i < n_measured; i++) {
        printf("%.1ff", measured_nus[i]);
        if (i < n_measured - 1) printf(", ");
        if ((i + 1) % 8 == 0 && i < n_measured - 1) printf("\n    ");
    }
    printf("\n};\n");

    printf("static const float OFFSET_VAL_TABLE[OFFSET_TABLE_SIZE] = {\n    ");
    for (int i = 0; i < n_measured; i++) {
        printf("%+.6ff", measured_offsets[i]);
        if (i < n_measured - 1) printf(", ");
        if ((i + 1) % 5 == 0 && i < n_measured - 1) printf("\n    ");
    }
    printf("\n};\n\n");

    printf("static inline float get_total_lik_offset(float nu) {\n");
    printf("    if (nu >= OFFSET_NU_TABLE[OFFSET_TABLE_SIZE-1])\n");
    printf("        return OFFSET_VAL_TABLE[OFFSET_TABLE_SIZE-1];\n");
    printf("    if (nu <= OFFSET_NU_TABLE[0])\n");
    printf("        return OFFSET_VAL_TABLE[0];\n");
    printf("    // Binary search + linear interp\n");
    printf("    int lo = 0, hi = OFFSET_TABLE_SIZE - 1;\n");
    printf("    while (hi - lo > 1) {\n");
    printf("        int mid = (lo + hi) / 2;\n");
    printf("        if (OFFSET_NU_TABLE[mid] <= nu) lo = mid; else hi = mid;\n");
    printf("    }\n");
    printf("    float frac = (nu - OFFSET_NU_TABLE[lo]) /\n");
    printf("                 (OFFSET_NU_TABLE[hi] - OFFSET_NU_TABLE[lo]);\n");
    printf("    return OFFSET_VAL_TABLE[lo] * (1.0f - frac) +\n");
    printf("           OFFSET_VAL_TABLE[hi] * frac;\n");
    printf("}\n");

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  VERIFICATION\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Known: empirical offset ≈ 0.345 was tuned for ν≈7, 512 particles, 8 Stein\n");
    printf("  Check: does measured offset at ν=7 match?\n");
    for (int i = 0; i < n_measured; i++) {
        if (measured_nus[i] == 7.0f)
            printf("  → offset(ν=7) = %+.6f  (expect ~+0.345)\n", measured_offsets[i]);
        if (measured_nus[i] == 50.0f)
            printf("  → offset(ν=50) = %+.6f  (expect ~+0.08)\n", measured_offsets[i]);
    }

    return 0;
}
