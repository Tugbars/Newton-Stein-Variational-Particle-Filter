// =============================================================================
// SVPF vs BPF vs APF — Extreme Scenario Stress Test
// =============================================================================
// Same DGP data, three filters, matched params (oracle mode).
// Tests: single spike, double spike, flash crash, sustained chaos, gradual build
// at 5σ through 50σ.
// Outputs: per-scenario RMSE comparison table + grand summary.
//
// Build: nvcc -O3 test_stress_compare.cu gpu_bpf.cu svpf_opt_kernels.cu \
//        svpf_optimized_graph.cu -o test_stress -lcurand
// =============================================================================

#include "svpf.cuh"
#include "gpu_bpf.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

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
// Simple PRNG
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
// SV DGP
// =============================================================================

struct SVDGPParams {
    float mu, rho, sigma;
};

static void gen_calm(std::vector<float>& ret, std::vector<float>& th,
                     int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        float eps = randn(seed);
        h = dgp.mu + dgp.rho * (h - dgp.mu) + dgp.sigma * eps;
        th.push_back(h);
        ret.push_back(expf(h * 0.5f) * randn(seed));
    }
}

static void gen_spike(std::vector<float>& ret, std::vector<float>& th,
                      float h_jump, float& h, float mu, unsigned int* seed) {
    float elevated = mu + h_jump;
    if (h < elevated) h = elevated;
    else h += h_jump * 0.3f;
    th.push_back(h);
    ret.push_back(expf(h * 0.5f) * randn(seed));
}

static void gen_recovery(std::vector<float>& ret, std::vector<float>& th,
                         int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        h = dgp.mu + dgp.rho * (h - dgp.mu) + dgp.sigma * randn(seed);
        th.push_back(h);
        ret.push_back(expf(h * 0.5f) * randn(seed));
    }
}

static void gen_chaos(std::vector<float>& ret, std::vector<float>& th,
                      int n, const SVDGPParams& dgp, float jmin, float jmax,
                      float& h, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        float jmag = jmin + (jmax - jmin) * randf(seed);
        float jsign = (randf(seed) > 0.5f) ? 1.0f : -1.0f;
        h += jsign * jmag * 0.3f;
        h = dgp.mu + 0.5f * (h - dgp.mu) + dgp.sigma * 2.0f * randn(seed);
        th.push_back(h);
        ret.push_back(expf(h * 0.5f) * randn(seed));
    }
}

static float sigma_to_h_jump(float sigma) { return sigma * 0.05f; }

// =============================================================================
// Scenario data container
// =============================================================================

struct ScenarioData {
    std::string name;
    float sigma;
    std::vector<float> returns;
    std::vector<float> true_h;
    int spike_t;        // first spike timestep
};

// =============================================================================
// Per-filter metrics
// =============================================================================

struct FilterMetrics {
    double rmse;
    double mae;
    double bias;
    double max_err;     // worst single-tick absolute error
    int    max_err_t;
    double spike_rmse;  // RMSE over spike+recovery window only
    double ms;          // wall time
    bool   had_nan;
};

// =============================================================================
// Run SVPF on scenario
// =============================================================================

static FilterMetrics run_svpf(const ScenarioData& sc,
                               const SVPFParams& params,
                               int n_particles, int n_stein,
                               float nu_obs, int seed) {
    FilterMetrics m = {};
    int n = (int)sc.returns.size();

    SVPFState* filter = svpf_create(n_particles, n_stein, nu_obs, nullptr);
    svpf_initialize(filter, &params, seed);

    double t0 = get_time_us();
    float y_prev = 0.0f;
    int skip = 20;
    double sum_sq = 0, sum_abs = 0, sum_bias = 0;
    double spike_sq = 0; int spike_n = 0;
    double worst = 0; int worst_t = 0;
    int count = 0;

    for (int t = 0; t < n; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(filter, sc.returns[t], y_prev, &params, &loglik, &vol, &h_mean);

        if (std::isnan(h_mean) || std::isinf(h_mean)) { m.had_nan = true; break; }

        if (t >= skip) {
            double err = (double)h_mean - (double)sc.true_h[t];
            sum_sq += err * err;
            sum_abs += fabs(err);
            sum_bias += err;
            count++;
            if (fabs(err) > worst) { worst = fabs(err); worst_t = t; }
            // Spike window: from spike_t to spike_t + 100
            if (t >= sc.spike_t && t < sc.spike_t + 100) {
                spike_sq += err * err;
                spike_n++;
            }
        }
        y_prev = sc.returns[t];
    }
    m.ms = (get_time_us() - t0) / 1000.0;
    svpf_destroy(filter);

    if (count > 0 && !m.had_nan) {
        m.rmse = sqrt(sum_sq / count);
        m.mae = sum_abs / count;
        m.bias = sum_bias / count;
        m.max_err = worst;
        m.max_err_t = worst_t;
        m.spike_rmse = (spike_n > 0) ? sqrt(spike_sq / spike_n) : 0;
    }
    return m;
}

// =============================================================================
// Run BPF on scenario
// =============================================================================

static FilterMetrics run_bpf(const ScenarioData& sc,
                              float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs,
                              int n_particles, int seed) {
    FilterMetrics m = {};
    int n = (int)sc.returns.size();

    GpuBpfState* state = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                         nu_state, nu_obs, seed);
    double t0 = get_time_us();
    int skip = 20;
    double sum_sq = 0, sum_abs = 0, sum_bias = 0;
    double spike_sq = 0; int spike_n = 0;
    double worst = 0; int worst_t = 0;
    int count = 0;

    for (int t = 0; t < n; t++) {
        BpfResult r = gpu_bpf_step(state, sc.returns[t]);

        if (std::isnan(r.h_mean) || std::isinf(r.h_mean)) { m.had_nan = true; break; }

        if (t >= skip) {
            double err = (double)r.h_mean - (double)sc.true_h[t];
            sum_sq += err * err;
            sum_abs += fabs(err);
            sum_bias += err;
            count++;
            if (fabs(err) > worst) { worst = fabs(err); worst_t = t; }
            if (t >= sc.spike_t && t < sc.spike_t + 100) {
                spike_sq += err * err;
                spike_n++;
            }
        }
    }
    m.ms = (get_time_us() - t0) / 1000.0;
    gpu_bpf_destroy(state);

    if (count > 0 && !m.had_nan) {
        m.rmse = sqrt(sum_sq / count);
        m.mae = sum_abs / count;
        m.bias = sum_bias / count;
        m.max_err = worst;
        m.max_err_t = worst_t;
        m.spike_rmse = (spike_n > 0) ? sqrt(spike_sq / spike_n) : 0;
    }
    return m;
}

// =============================================================================
// Run APF on scenario
// =============================================================================

static FilterMetrics run_apf(const ScenarioData& sc,
                              float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs,
                              int n_particles, int seed) {
    FilterMetrics m = {};
    int n = (int)sc.returns.size();

    GpuApfState* state = gpu_apf_create(n_particles, rho, sigma_z, mu,
                                         nu_state, nu_obs, seed);
    double t0 = get_time_us();
    int skip = 20;
    double sum_sq = 0, sum_abs = 0, sum_bias = 0;
    double spike_sq = 0; int spike_n = 0;
    double worst = 0; int worst_t = 0;
    int count = 0;

    for (int t = 0; t < n; t++) {
        BpfResult r = gpu_apf_step(state, sc.returns[t]);

        if (std::isnan(r.h_mean) || std::isinf(r.h_mean)) { m.had_nan = true; break; }

        if (t >= skip) {
            double err = (double)r.h_mean - (double)sc.true_h[t];
            sum_sq += err * err;
            sum_abs += fabs(err);
            sum_bias += err;
            count++;
            if (fabs(err) > worst) { worst = fabs(err); worst_t = t; }
            if (t >= sc.spike_t && t < sc.spike_t + 100) {
                spike_sq += err * err;
                spike_n++;
            }
        }
    }
    m.ms = (get_time_us() - t0) / 1000.0;
    gpu_apf_destroy(state);

    if (count > 0 && !m.had_nan) {
        m.rmse = sqrt(sum_sq / count);
        m.mae = sum_abs / count;
        m.bias = sum_bias / count;
        m.max_err = worst;
        m.max_err_t = worst_t;
        m.spike_rmse = (spike_n > 0) ? sqrt(spike_sq / spike_n) : 0;
    }
    return m;
}

// =============================================================================
// Scenario generators
// =============================================================================

static ScenarioData make_single_spike(float sigma, const SVDGPParams& dgp, unsigned int seed_val) {
    ScenarioData sc;
    sc.name = "Single Spike";
    sc.sigma = sigma;
    unsigned int seed = seed_val;
    float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size();
    gen_spike(sc.returns, sc.true_h, sigma_to_h_jump(sigma), h, dgp.mu, &seed);
    gen_recovery(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

static ScenarioData make_double_spike(float sigma, const SVDGPParams& dgp, unsigned int seed_val) {
    ScenarioData sc;
    sc.name = "Double Spike";
    sc.sigma = sigma;
    unsigned int seed = seed_val;
    float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size();
    float hj = sigma_to_h_jump(sigma);
    gen_spike(sc.returns, sc.true_h, hj, h, dgp.mu, &seed);
    gen_recovery(sc.returns, sc.true_h, 50, dgp, h, &seed);
    gen_spike(sc.returns, sc.true_h, hj, h, dgp.mu, &seed);
    gen_recovery(sc.returns, sc.true_h, 150, dgp, h, &seed);
    return sc;
}

static ScenarioData make_flash_crash(float sigma, const SVDGPParams& dgp, unsigned int seed_val) {
    ScenarioData sc;
    sc.name = "Flash Crash";
    sc.sigma = sigma;
    unsigned int seed = seed_val;
    float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size();
    float hj = sigma_to_h_jump(sigma);
    gen_spike(sc.returns, sc.true_h, hj, h, dgp.mu, &seed);
    gen_spike(sc.returns, sc.true_h, hj * 0.3f, h, dgp.mu, &seed);
    gen_spike(sc.returns, sc.true_h, hj * 0.2f, h, dgp.mu, &seed);
    gen_chaos(sc.returns, sc.true_h, 20, dgp, 0.5f, 2.0f, h, &seed);
    gen_recovery(sc.returns, sc.true_h, 150, dgp, h, &seed);
    return sc;
}

static ScenarioData make_sustained_chaos(float sigma, const SVDGPParams& dgp, unsigned int seed_val) {
    ScenarioData sc;
    sc.name = "Sustained Chaos";
    sc.sigma = sigma;
    unsigned int seed = seed_val;
    float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size();
    gen_chaos(sc.returns, sc.true_h, 50, dgp,
              sigma_to_h_jump(sigma * 0.5f), sigma_to_h_jump(sigma * 1.5f), h, &seed);
    gen_recovery(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

static ScenarioData make_gradual_build(float sigma, const SVDGPParams& dgp, unsigned int seed_val) {
    ScenarioData sc;
    sc.name = "Gradual Build";
    sc.sigma = sigma;
    unsigned int seed = seed_val;
    float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size();
    float hj = sigma_to_h_jump(sigma);
    for (int i = 0; i < 20; i++) {
        float frac = (float)i / 20.0f;
        gen_spike(sc.returns, sc.true_h, hj * frac * 0.1f, h, dgp.mu, &seed);
    }
    gen_spike(sc.returns, sc.true_h, hj * 0.3f, h, dgp.mu, &seed);
    gen_recovery(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

// =============================================================================
// Summary accumulator
// =============================================================================

struct GrandSummary {
    double svpf_rmse_sum, bpf_rmse_sum, apf_rmse_sum;
    double svpf_spike_sum, bpf_spike_sum, apf_spike_sum;
    int    svpf_nan, bpf_nan, apf_nan;
    int    count;
};

// =============================================================================
// Run one scenario at one sigma level across all filters and print row
// =============================================================================

static void run_and_print(const ScenarioData& sc,
                           const SVPFParams& svpf_params,
                           int svpf_particles, int svpf_stein, float svpf_nu,
                           float rho, float sigma_z, float mu,
                           float nu_state, float nu_obs,
                           int bpf_particles, int seed,
                           GrandSummary& gs) {
    FilterMetrics svpf = run_svpf(sc, svpf_params, svpf_particles, svpf_stein, svpf_nu, seed);
    FilterMetrics bpf  = run_bpf(sc, rho, sigma_z, mu, nu_state, nu_obs, bpf_particles, seed + 100);
    FilterMetrics apf  = run_apf(sc, rho, sigma_z, mu, nu_state, nu_obs, bpf_particles, seed + 200);

    auto fmt = [](const FilterMetrics& m, char* buf) {
        if (m.had_nan)
            snprintf(buf, 80, "    NaN!   ");
        else
            snprintf(buf, 80, "%6.4f", m.rmse);
    };

    auto fmt_spike = [](const FilterMetrics& m, char* buf) {
        if (m.had_nan)
            snprintf(buf, 80, "  NaN!");
        else
            snprintf(buf, 80, "%6.4f", m.spike_rmse);
    };

    char sb[80], bb[80], ab[80], ss[80], bs[80], as[80];
    fmt(svpf, sb); fmt(bpf, bb); fmt(apf, ab);
    fmt_spike(svpf, ss); fmt_spike(bpf, bs); fmt_spike(apf, as);

    // Winner marker
    auto winner = [](double a, double b, double c, bool an, bool bn, bool cn) -> int {
        if (an && bn && cn) return -1;
        double vals[3] = {an ? 1e9 : a, bn ? 1e9 : b, cn ? 1e9 : c};
        if (vals[0] <= vals[1] && vals[0] <= vals[2]) return 0;
        if (vals[1] <= vals[0] && vals[1] <= vals[2]) return 1;
        return 2;
    };

    int w = winner(svpf.rmse, bpf.rmse, apf.rmse,
                   svpf.had_nan, bpf.had_nan, apf.had_nan);
    const char* stars[] = {"*", "*", "*"};
    const char* blanks[] = {" ", " ", " "};
    const char** marks = (w >= 0) ? blanks : blanks;
    const char* m0 = " ", *m1 = " ", *m2 = " ";
    if (w == 0) m0 = "*"; else if (w == 1) m1 = "*"; else if (w == 2) m2 = "*";

    printf("  %-17s %3.0fσ  %s%s  %s%s  %s%s │ %s  %s  %s │ %+6.3f %+6.3f %+6.3f │ %5.1f %5.1f %5.1f\n",
           sc.name.c_str(), sc.sigma,
           sb, m0, bb, m1, ab, m2,
           ss, bs, as,
           svpf.had_nan ? 0.0 : svpf.bias,
           bpf.had_nan ? 0.0 : bpf.bias,
           apf.had_nan ? 0.0 : apf.bias,
           svpf.ms, bpf.ms, apf.ms);

    // Accumulate
    if (!svpf.had_nan) { gs.svpf_rmse_sum += svpf.rmse; gs.svpf_spike_sum += svpf.spike_rmse; }
    else gs.svpf_nan++;
    if (!bpf.had_nan)  { gs.bpf_rmse_sum += bpf.rmse; gs.bpf_spike_sum += bpf.spike_rmse; }
    else gs.bpf_nan++;
    if (!apf.had_nan)  { gs.apf_rmse_sum += apf.rmse; gs.apf_spike_sum += apf.spike_rmse; }
    else gs.apf_nan++;
    gs.count++;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // --- Config ---
    int svpf_particles = 512;
    int svpf_stein     = 8;
    float svpf_nu      = 50.0f;       // observation nu for SVPF
    int bpf_particles   = 50000;
    float bpf_nu_obs    = 50.0f;       // observation nu for BPF/APF
    int seed            = 42;

    // DGP parameters (= filter params for this oracle test)
    float rho     = 0.98f;
    float sigma_z = 0.15f;
    float mu      = -4.5f;
    float nu_state = 0.0f;   // Gaussian state noise
    SVDGPParams dgp = {mu, rho, sigma_z};

    SVPFParams svpf_params;
    svpf_params.rho     = rho;
    svpf_params.sigma_z = sigma_z;
    svpf_params.mu      = mu;
    svpf_params.gamma   = 0.0f;

    // Parse overrides
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--svpf-particles") == 0 && i+1 < argc)
            svpf_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bpf-particles") == 0 && i+1 < argc)
            bpf_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--svpf-nu") == 0 && i+1 < argc)
            svpf_nu = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--bpf-nu") == 0 && i+1 < argc)
            bpf_nu_obs = (float)atof(argv[++i]);
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  EXTREME SCENARIO STRESS TEST: BPF vs APF vs SVPF\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f  (matched params — oracle mode)\n", rho, sigma_z, mu);
    printf("  SVPF: %d particles, %d Stein steps, ν_obs=%.0f\n", svpf_particles, svpf_stein, svpf_nu);
    printf("  BPF/APF: %d particles, ν_obs=%.0f\n", bpf_particles, bpf_nu_obs);
    printf("  * = best RMSE for that row\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-17s %4s  %6s  %6s  %6s │ %6s %6s %6s │ %6s %6s %6s │ %5s %5s %5s\n",
           "Scenario", "Mag", "SVPF", "BPF", "APF", "Sp.SVP", "Sp.BPF", "Sp.APF",
           "bSVPF", "bBPF", "bAPF", "msSVP", "msBPF", "msAPF");
    printf("  ───────────────── ──── ─────── ─────── ─────── │ ────── ────── ────── │ ────── ────── ────── │ ───── ───── ─────\n");

    float sigmas[] = {5, 10, 15, 20, 25, 30, 40, 50};
    int n_sigmas = 8;

    GrandSummary gs = {};

    // --- Single Spike ---
    for (int i = 0; i < n_sigmas; i++) {
        ScenarioData sc = make_single_spike(sigmas[i], dgp, 42);
        run_and_print(sc, svpf_params, svpf_particles, svpf_stein, svpf_nu,
                      rho, sigma_z, mu, nu_state, bpf_nu_obs, bpf_particles, seed, gs);
    }
    printf("  ───────────────── ──── ─────── ─────── ─────── │ ────── ────── ────── │ ────── ────── ────── │ ───── ───── ─────\n");

    // --- Double Spike ---
    for (int i = 0; i < n_sigmas; i++) {
        ScenarioData sc = make_double_spike(sigmas[i], dgp, 43);
        run_and_print(sc, svpf_params, svpf_particles, svpf_stein, svpf_nu,
                      rho, sigma_z, mu, nu_state, bpf_nu_obs, bpf_particles, seed, gs);
    }
    printf("  ───────────────── ──── ─────── ─────── ─────── │ ────── ────── ────── │ ────── ────── ────── │ ───── ───── ─────\n");

    // --- Flash Crash ---
    for (int i = 0; i < n_sigmas; i++) {
        ScenarioData sc = make_flash_crash(sigmas[i], dgp, 44);
        run_and_print(sc, svpf_params, svpf_particles, svpf_stein, svpf_nu,
                      rho, sigma_z, mu, nu_state, bpf_nu_obs, bpf_particles, seed, gs);
    }
    printf("  ───────────────── ──── ─────── ─────── ─────── │ ────── ────── ────── │ ────── ────── ────── │ ───── ───── ─────\n");

    // --- Sustained Chaos ---
    float chaos_sigmas[] = {5, 7, 10, 12, 15, 20, 25, 30};
    for (int i = 0; i < 8; i++) {
        ScenarioData sc = make_sustained_chaos(chaos_sigmas[i], dgp, 45);
        run_and_print(sc, svpf_params, svpf_particles, svpf_stein, svpf_nu,
                      rho, sigma_z, mu, nu_state, bpf_nu_obs, bpf_particles, seed, gs);
    }
    printf("  ───────────────── ──── ─────── ─────── ─────── │ ────── ────── ────── │ ────── ────── ────── │ ───── ───── ─────\n");

    // --- Gradual Build ---
    for (int i = 0; i < n_sigmas; i++) {
        ScenarioData sc = make_gradual_build(sigmas[i], dgp, 46);
        run_and_print(sc, svpf_params, svpf_particles, svpf_stein, svpf_nu,
                      rho, sigma_z, mu, nu_state, bpf_nu_obs, bpf_particles, seed, gs);
    }

    // --- Grand summary ---
    int svpf_ok = gs.count - gs.svpf_nan;
    int bpf_ok  = gs.count - gs.bpf_nan;
    int apf_ok  = gs.count - gs.apf_nan;

    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  GRAND SUMMARY (%d scenarios)\n", gs.count);
    printf("  ─────────────────────────────────────\n");
    printf("  %-18s %8s %8s %8s\n", "", "SVPF", "BPF", "APF");
    printf("  %-18s %8.4f %8.4f %8.4f\n", "Avg RMSE",
           svpf_ok > 0 ? gs.svpf_rmse_sum / svpf_ok : 0,
           bpf_ok > 0 ? gs.bpf_rmse_sum / bpf_ok : 0,
           apf_ok > 0 ? gs.apf_rmse_sum / apf_ok : 0);
    printf("  %-18s %8.4f %8.4f %8.4f\n", "Avg Spike RMSE",
           svpf_ok > 0 ? gs.svpf_spike_sum / svpf_ok : 0,
           bpf_ok > 0 ? gs.bpf_spike_sum / bpf_ok : 0,
           apf_ok > 0 ? gs.apf_spike_sum / apf_ok : 0);
    printf("  %-18s %8d %8d %8d\n", "NaN/Inf count", gs.svpf_nan, gs.bpf_nan, gs.apf_nan);
    printf("  %-18s %8d %8d %8d\n", "Survived", svpf_ok, bpf_ok, apf_ok);
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
