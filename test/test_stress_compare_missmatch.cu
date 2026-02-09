// =============================================================================
// COMPLETE STRESS TEST: BPF vs SVPF — Oracle + Misspecified
// =============================================================================
//
// Part 1: Oracle mode (matched params) — baseline on extreme events
// Part 2: Misspecified params — the REAL test. Both filters get WRONG params.
//         Graded: oracle → mild → moderate → severe → extreme misspecification.
//         DGP includes regime teleportation, pure chaos, crypto meltdown.
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

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

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

static inline float rand_t(unsigned int* seed, float nu) {
    if (nu <= 0.0f || nu > 100.0f) return randn(seed);
    float z = randn(seed);
    float chi2 = 0.0f;
    for (int k = 0; k < (int)nu; k++) {
        float g = randn(seed);
        chi2 += g * g;
    }
    return z * sqrtf(nu / fmaxf(chi2, 1e-8f));
}

// =============================================================================
// DGP generation primitives
// =============================================================================

struct SVDGPParams { float mu, rho, sigma; };

static void gen_sv(std::vector<float>& ret, std::vector<float>& th,
                   int n, float rho, float sigma_z, float mu,
                   float nu_state, float nu_obs,
                   float& h, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        float eps = (nu_state > 0) ? rand_t(seed, nu_state) : randn(seed);
        h = mu + rho * (h - mu) + sigma_z * eps;
        th.push_back(h);
        float eta = (nu_obs > 0) ? rand_t(seed, nu_obs) : randn(seed);
        ret.push_back(expf(h * 0.5f) * eta);
    }
}

static void gen_calm(std::vector<float>& ret, std::vector<float>& th,
                     int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    gen_sv(ret, th, n, dgp.rho, dgp.sigma, dgp.mu, 0.0f, 0.0f, h, seed);
}

static void gen_spike(std::vector<float>& ret, std::vector<float>& th,
                      float h_jump, float& h, float mu, unsigned int* seed) {
    float elevated = mu + h_jump;
    if (h < elevated) h = elevated; else h += h_jump * 0.3f;
    th.push_back(h);
    ret.push_back(expf(h * 0.5f) * randn(seed));
}

static void gen_recovery(std::vector<float>& ret, std::vector<float>& th,
                         int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    gen_calm(ret, th, n, dgp, h, seed);
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
// Scenario data
// =============================================================================

struct ScenarioData {
    std::string name;
    float sigma;
    std::vector<float> returns;
    std::vector<float> true_h;
    int spike_t;
    int spike_window;
};

// =============================================================================
// Metrics
// =============================================================================

struct FilterMetrics {
    double rmse, bias, spike_rmse, max_err, ms;
    bool had_nan;
};

// =============================================================================
// Run SVPF
// =============================================================================

static FilterMetrics run_svpf(const ScenarioData& sc,
                               float f_rho, float f_sigma_z, float f_mu,
                               int n_particles, int n_stein, float nu_obs,
                               int seed) {
    FilterMetrics m = {};
    int n = (int)sc.returns.size();

    SVPFParams params;
    params.rho = f_rho; params.sigma_z = f_sigma_z; params.mu = f_mu; params.gamma = 0.0f;

    SVPFState* filter = svpf_create(n_particles, n_stein, nu_obs, nullptr);
    svpf_initialize(filter, &params, seed);

    double t0 = get_time_us();
    float y_prev = 0.0f;
    int skip = 20;
    double sum_sq = 0, sum_bias = 0, spike_sq = 0, worst = 0;
    int count = 0, spike_n = 0;

    for (int t = 0; t < n; t++) {
        float loglik, vol, h_mean;
        svpf_step_graph(filter, sc.returns[t], y_prev, &params, &loglik, &vol, &h_mean);
        if (std::isnan(h_mean) || std::isinf(h_mean)) { m.had_nan = true; break; }
        if (t >= skip) {
            double err = (double)h_mean - (double)sc.true_h[t];
            sum_sq += err * err; sum_bias += err;
            if (fabs(err) > worst) worst = fabs(err);
            count++;
            if (t >= sc.spike_t && t < sc.spike_t + sc.spike_window) {
                spike_sq += err * err; spike_n++;
            }
        }
        y_prev = sc.returns[t];
    }
    m.ms = (get_time_us() - t0) / 1000.0;
    svpf_destroy(filter);

    if (count > 0 && !m.had_nan) {
        m.rmse = sqrt(sum_sq / count); m.bias = sum_bias / count;
        m.max_err = worst;
        m.spike_rmse = (spike_n > 0) ? sqrt(spike_sq / spike_n) : 0;
    }
    return m;
}

// =============================================================================
// Run BPF
// =============================================================================

static FilterMetrics run_bpf(const ScenarioData& sc,
                              float f_rho, float f_sigma_z, float f_mu,
                              float nu_obs, int n_particles, int seed) {
    FilterMetrics m = {};
    int n = (int)sc.returns.size();

    GpuBpfState* state = gpu_bpf_create(n_particles, f_rho, f_sigma_z, f_mu,
                                         0.0f, nu_obs, seed);
    double t0 = get_time_us();
    int skip = 20;
    double sum_sq = 0, sum_bias = 0, spike_sq = 0, worst = 0;
    int count = 0, spike_n = 0;

    for (int t = 0; t < n; t++) {
        BpfResult r = gpu_bpf_step(state, sc.returns[t]);
        if (std::isnan(r.h_mean) || std::isinf(r.h_mean)) { m.had_nan = true; break; }
        if (t >= skip) {
            double err = (double)r.h_mean - (double)sc.true_h[t];
            sum_sq += err * err; sum_bias += err;
            if (fabs(err) > worst) worst = fabs(err);
            count++;
            if (t >= sc.spike_t && t < sc.spike_t + sc.spike_window) {
                spike_sq += err * err; spike_n++;
            }
        }
    }
    m.ms = (get_time_us() - t0) / 1000.0;
    gpu_bpf_destroy(state);

    if (count > 0 && !m.had_nan) {
        m.rmse = sqrt(sum_sq / count); m.bias = sum_bias / count;
        m.max_err = worst;
        m.spike_rmse = (spike_n > 0) ? sqrt(spike_sq / spike_n) : 0;
    }
    return m;
}

// =============================================================================
// Oracle scenario generators
// =============================================================================

static ScenarioData make_single_spike(float sigma, const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Single Spike"; sc.sigma = sigma;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 100;
    gen_spike(sc.returns, sc.true_h, sigma_to_h_jump(sigma), h, dgp.mu, &seed);
    gen_recovery(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

static ScenarioData make_flash_crash(float sigma, const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Flash Crash"; sc.sigma = sigma;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 100;
    float hj = sigma_to_h_jump(sigma);
    gen_spike(sc.returns, sc.true_h, hj, h, dgp.mu, &seed);
    gen_spike(sc.returns, sc.true_h, hj * 0.3f, h, dgp.mu, &seed);
    gen_spike(sc.returns, sc.true_h, hj * 0.2f, h, dgp.mu, &seed);
    gen_chaos(sc.returns, sc.true_h, 20, dgp, 0.5f, 2.0f, h, &seed);
    gen_recovery(sc.returns, sc.true_h, 150, dgp, h, &seed);
    return sc;
}

// =============================================================================
// MISSPECIFIED scenario generators
// =============================================================================

// Spike gauntlet: 20σ → 30σ → 40σ → 50σ with 50-tick recovery between
static ScenarioData make_spike_gauntlet(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Spike Gauntlet"; sc.sigma = 50;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 250;
    float jumps[] = {1.0f, 1.5f, 2.0f, 2.5f};
    for (int s = 0; s < 4; s++) {
        h = dgp.mu + jumps[s];
        sc.true_h.push_back(h);
        sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
        gen_recovery(sc.returns, sc.true_h, 50, dgp, h, &seed);
    }
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

// Regime teleportation: vol teleports between wildly different levels
static ScenarioData make_regime_teleport(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Regime Teleport"; sc.sigma = 0;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 450;
    float mus[] = {-2.0f, -7.0f, -0.5f, -4.5f};
    for (int r = 0; r < 4; r++) {
        h = mus[r];
        gen_sv(sc.returns, sc.true_h, 100, dgp.rho, dgp.sigma, mus[r],
               0.0f, 0.0f, h, &seed);
    }
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

// Pure chaos: h random walk + 10% chance ±2 jump per tick
static ScenarioData make_pure_chaos(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Pure Chaos"; sc.sigma = 0;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 200;
    for (int i = 0; i < 200; i++) {
        if (randf(&seed) < 0.10f)
            h += (randf(&seed) - 0.5f) * 4.0f;
        float eps = randn(&seed);
        h = dgp.mu + 0.5f * (h - dgp.mu) + dgp.sigma * 3.0f * eps;
        if (h > 2.0f) h = 2.0f; if (h < -10.0f) h = -10.0f;
        sc.true_h.push_back(h);
        sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
    }
    gen_recovery(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

// Crypto meltdown: 150 ticks of t(3) state+obs, 2x sigma_z
static ScenarioData make_crypto_meltdown(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Crypto Meltdown"; sc.sigma = 0;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 200;
    gen_sv(sc.returns, sc.true_h, 150, dgp.rho, dgp.sigma * 2.0f, dgp.mu,
           3.0f, 3.0f, h, &seed);
    gen_calm(sc.returns, sc.true_h, 250, dgp, h, &seed);
    return sc;
}

// Periodic regimes: 8 teleports across mu range over 1200 ticks
static ScenarioData make_periodic_regimes(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Periodic Regimes"; sc.sigma = 0;
    unsigned int seed = sv; float h = dgp.mu;
    sc.spike_t = 0; sc.spike_window = 600;
    float mus[] = {-6.0f, -3.0f, -5.0f, -1.5f, -4.5f, -2.0f, -6.5f, -3.5f};
    for (int r = 0; r < 8; r++) {
        h = mus[r];
        gen_sv(sc.returns, sc.true_h, 150, dgp.rho, dgp.sigma, mus[r],
               0.0f, 0.0f, h, &seed);
    }
    return sc;
}

// Sawtooth: vol ramps h→h+3 over 100 ticks, crashes back, 4 cycles
static ScenarioData make_sawtooth(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Sawtooth Ramp"; sc.sigma = 0;
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 400;
    for (int cyc = 0; cyc < 4; cyc++) {
        for (int i = 0; i < 100; i++) {
            float target = dgp.mu + 3.0f * (float)i / 100.0f;
            h = target + dgp.sigma * randn(&seed);
            sc.true_h.push_back(h);
            sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
        }
        h = dgp.mu;
        sc.true_h.push_back(h);
        sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
    }
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    return sc;
}

// =============================================================================
// Summary
// =============================================================================

struct GrandSummary {
    double svpf_rmse_sum, bpf_rmse_sum;
    double svpf_spike_sum, bpf_spike_sum;
    int svpf_nan, bpf_nan, svpf_wins, bpf_wins, count;
};

// =============================================================================
// Oracle mode print
// =============================================================================

static void run_oracle(const ScenarioData& sc,
                       float rho, float sigma_z, float mu,
                       int sp, int ss, float snu, float bnu, int bp, int seed,
                       GrandSummary& gs) {
    FilterMetrics svpf = run_svpf(sc, rho, sigma_z, mu, sp, ss, snu, seed);
    FilterMetrics bpf  = run_bpf(sc, rho, sigma_z, mu, bnu, bp, seed + 100);

    const char* sw = " ", *bw = " ";
    if (!svpf.had_nan && !bpf.had_nan) {
        if (svpf.rmse < bpf.rmse) { sw = "*"; gs.svpf_wins++; } else { bw = "*"; gs.bpf_wins++; }
    }
    printf("  %-17s %3.0fσ  %6.4f%s %6.4f%s │ %6.4f %6.4f │ %+6.3f %+6.3f │ %5.1f %5.1f\n",
           sc.name.c_str(), sc.sigma,
           svpf.had_nan ? 0.0 : svpf.rmse, sw, bpf.had_nan ? 0.0 : bpf.rmse, bw,
           svpf.had_nan ? 0.0 : svpf.spike_rmse, bpf.had_nan ? 0.0 : bpf.spike_rmse,
           svpf.had_nan ? 0.0 : svpf.bias, bpf.had_nan ? 0.0 : bpf.bias,
           svpf.ms, bpf.ms);
    if (!svpf.had_nan) { gs.svpf_rmse_sum += svpf.rmse; gs.svpf_spike_sum += svpf.spike_rmse; } else gs.svpf_nan++;
    if (!bpf.had_nan)  { gs.bpf_rmse_sum += bpf.rmse;   gs.bpf_spike_sum += bpf.spike_rmse; }   else gs.bpf_nan++;
    gs.count++;
}

// =============================================================================
// Misspecified mode
// =============================================================================

struct MisspecConfig { const char* label; float rho, sigma_z, mu; };

static void print_ms_hdr() {
    printf("  %-18s %-12s │ %7s  %7s  │ %7s %7s │ %7s %7s │ %6s %6s │ %5s %5s\n",
           "Scenario", "Misspec", "SVPF", "BPF", "Sp.SVP", "Sp.BPF",
           "bSVPF", "bBPF", "mxSVPF", "mxBPF", "msSVP", "msBPF");
    printf("  ────────────────── ──────────── │ ──────── ──────── │ ─────── ─────── │ ─────── ─────── │ ────── ────── │ ───── ─────\n");
}

static void run_ms(const ScenarioData& sc, const MisspecConfig& wrong,
                   int sp, int ss, float snu, float bnu, int bp, int seed,
                   GrandSummary& gs) {
    FilterMetrics svpf = run_svpf(sc, wrong.rho, wrong.sigma_z, wrong.mu, sp, ss, snu, seed);
    FilterMetrics bpf  = run_bpf(sc, wrong.rho, wrong.sigma_z, wrong.mu, bnu, bp, seed + 100);

    const char* sw = " ", *bw = " ";
    if (!svpf.had_nan && !bpf.had_nan) {
        if (svpf.rmse < bpf.rmse) { sw = "*"; gs.svpf_wins++; } else { bw = "*"; gs.bpf_wins++; }
    }

    auto fmt = [](const FilterMetrics& m, char* buf) {
        if (m.had_nan) snprintf(buf, 40, "  NaN! "); else snprintf(buf, 40, "%7.4f", m.rmse);
    };
    char sr[40], br[40]; fmt(svpf, sr); fmt(bpf, br);

    printf("  %-18s %-12s │ %s%s %s%s │ %7.4f %7.4f │ %+7.3f %+7.3f │ %6.2f %6.2f │ %5.0f %5.0f\n",
           sc.name.c_str(), wrong.label,
           sr, sw, br, bw,
           svpf.had_nan ? 0.0 : svpf.spike_rmse, bpf.had_nan ? 0.0 : bpf.spike_rmse,
           svpf.had_nan ? 0.0 : svpf.bias, bpf.had_nan ? 0.0 : bpf.bias,
           svpf.had_nan ? 0.0 : svpf.max_err, bpf.had_nan ? 0.0 : bpf.max_err,
           svpf.ms, bpf.ms);

    if (!svpf.had_nan) { gs.svpf_rmse_sum += svpf.rmse; gs.svpf_spike_sum += svpf.spike_rmse; } else gs.svpf_nan++;
    if (!bpf.had_nan)  { gs.bpf_rmse_sum += bpf.rmse;   gs.bpf_spike_sum += bpf.spike_rmse; }   else gs.bpf_nan++;
    gs.count++;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int sp = 512, ss = 32; float snu = 50.0f;   // Gaussian DGP → high nu
    int bp = 50000;        float bnu = 50.0f;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--svpf-n")  && i+1<argc) sp  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--bpf-n")   && i+1<argc) bp  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--svpf-nu") && i+1<argc) snu = (float)atof(argv[++i]);
        if (!strcmp(argv[i], "--bpf-nu")  && i+1<argc) bnu = (float)atof(argv[++i]);
    }

    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    SVDGPParams dgp = {true_mu, true_rho, true_sz};

    // =====================================================================
    // PART 1: ORACLE MODE
    // =====================================================================
    printf("\n═══════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PART 1: ORACLE MODE (matched params)\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  DGP: ρ=%.2f σ_z=%.2f μ=%.1f │ SVPF %d×%d ν=%.0f │ BPF %dK ν=%.0f\n",
           true_rho, true_sz, true_mu, sp, ss, snu, bp/1000, bnu);
    printf("  %-17s %4s  %6s  %6s  │ %6s %6s │ %6s %6s │ %5s %5s\n",
           "Scenario", "Mag", "SVPF", "BPF", "SpSVPF", "SpBPF", "bSVPF", "bBPF", "msSVP", "msBPF");
    printf("  ───────────────── ──── ─────── ─────── │ ────── ────── │ ────── ────── │ ───── ─────\n");

    GrandSummary gs1 = {};
    float sigs[] = {10, 20, 30, 50};
    for (int i = 0; i < 4; i++)
        run_oracle(make_single_spike(sigs[i], dgp, 42), true_rho, true_sz, true_mu,
                   sp, ss, snu, bnu, bp, seed, gs1);
    printf("  ───────────────── ──── ─────── ─────── │ ────── ────── │ ────── ────── │ ───── ─────\n");
    for (int i = 0; i < 4; i++)
        run_oracle(make_flash_crash(sigs[i], dgp, 44), true_rho, true_sz, true_mu,
                   sp, ss, snu, bnu, bp, seed, gs1);

    int s1ok = gs1.count - gs1.svpf_nan, b1ok = gs1.count - gs1.bpf_nan;
    printf("  ─── Oracle: SVPF avg=%.4f  BPF avg=%.4f  wins %d/%d ───\n",
           s1ok>0 ? gs1.svpf_rmse_sum/s1ok : 0, b1ok>0 ? gs1.bpf_rmse_sum/b1ok : 0,
           gs1.svpf_wins, gs1.bpf_wins);

    // =====================================================================
    // PART 2: MISSPECIFIED PARAMETERS
    // =====================================================================
    printf("\n\n═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PART 2: MISSPECIFIED PARAMETERS — both filters get WRONG params\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  True DGP: ρ=0.98  σ_z=0.15  μ=-4.5\n");
    printf("  SVPF %d×%d ν=%.0f │ BPF %dK ν=%.0f  │ * = winner\n", sp, ss, snu, bp/1000, bnu);
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");

    //                                label          ρ      σ_z    μ       true: 0.98, 0.15, -4.5
    MisspecConfig mc[] = {
        {"Oracle",    0.98f, 0.15f, -4.5f},
        {"Mild",      0.95f, 0.12f, -4.0f},
        {"Moderate",  0.90f, 0.10f, -3.5f},
        {"Severe",    0.80f, 0.05f, -3.0f},
        {"Extreme",   0.70f, 0.03f, -2.0f},
        {"Wrong μ",   0.98f, 0.15f, -6.5f},
        {"Wrong ρ",   0.80f, 0.15f, -4.5f},
        {"Wrong σ_z", 0.98f, 0.02f, -4.5f},
    };
    int n_mc = 8;
    GrandSummary gs2 = {};

    // A: Spike Gauntlet
    printf("\n  ┌── TEST A: Spike Gauntlet (20σ→50σ) ──────────────────────────────────────────────────────────────┐\n");
    print_ms_hdr();
    ScenarioData a = make_spike_gauntlet(dgp, 42);
    for (int m = 0; m < n_mc; m++) run_ms(a, mc[m], sp, ss, snu, bnu, bp, seed, gs2);

    // B: Regime Teleport
    printf("\n  ┌── TEST B: Regime Teleport (vol 3%%→78%% and back) ───────────────────────────────────────────────┐\n");
    print_ms_hdr();
    ScenarioData b = make_regime_teleport(dgp, 43);
    for (int m = 0; m < n_mc; m++) run_ms(b, mc[m], sp, ss, snu, bnu, bp, seed, gs2);

    // C: Pure Chaos
    printf("\n  ┌── TEST C: Pure Chaos (h random walk + 10%% ±2 jumps) ──────────────────────────────────────────┐\n");
    print_ms_hdr();
    ScenarioData c = make_pure_chaos(dgp, 44);
    for (int m = 0; m < n_mc; m++) run_ms(c, mc[m], sp, ss, snu, bnu, bp, seed, gs2);

    // D: Crypto Meltdown
    printf("\n  ┌── TEST D: Crypto Meltdown (t(3) state+obs, 2x σ_z) ──────────────────────────────────────────┐\n");
    print_ms_hdr();
    ScenarioData d = make_crypto_meltdown(dgp, 45);
    for (int m = 0; m < n_mc; m++) run_ms(d, mc[m], sp, ss, snu, bnu, bp, seed, gs2);

    // E: Periodic Regimes
    printf("\n  ┌── TEST E: Periodic Regimes (8 teleports, 1200 ticks) ─────────────────────────────────────────┐\n");
    print_ms_hdr();
    ScenarioData e = make_periodic_regimes(dgp, 46);
    for (int m = 0; m < n_mc; m++) run_ms(e, mc[m], sp, ss, snu, bnu, bp, seed, gs2);

    // F: Sawtooth
    printf("\n  ┌── TEST F: Sawtooth Ramp (4x ramp+crash cycles) ──────────────────────────────────────────────┐\n");
    print_ms_hdr();
    ScenarioData f = make_sawtooth(dgp, 47);
    for (int m = 0; m < n_mc; m++) run_ms(f, mc[m], sp, ss, snu, bnu, bp, seed, gs2);

    // =================================================================
    // GRAND SUMMARY
    // =================================================================
    int s2ok = gs2.count - gs2.svpf_nan;
    int b2ok = gs2.count - gs2.bpf_nan;

    printf("\n═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  GRAND SUMMARY — MISSPECIFIED (%d scenarios)\n", gs2.count);
    printf("  ─────────────────────────────────────────────\n");
    printf("  %-20s %10s %10s\n",                              "", "SVPF", "BPF");
    printf("  %-20s %10.4f %10.4f\n", "Avg RMSE",             s2ok>0 ? gs2.svpf_rmse_sum/s2ok  : 0.0, b2ok>0 ? gs2.bpf_rmse_sum/b2ok  : 0.0);
    printf("  %-20s %10.4f %10.4f\n", "Avg Spike RMSE",       s2ok>0 ? gs2.svpf_spike_sum/s2ok : 0.0, b2ok>0 ? gs2.bpf_spike_sum/b2ok : 0.0);
    printf("  %-20s %10d %10d\n",     "Wins",                  gs2.svpf_wins, gs2.bpf_wins);
    printf("  %-20s %10d %10d\n",     "NaN/Inf",               gs2.svpf_nan, gs2.bpf_nan);
    printf("  %-20s %10d %10d\n",     "Survived",              s2ok, b2ok);
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
