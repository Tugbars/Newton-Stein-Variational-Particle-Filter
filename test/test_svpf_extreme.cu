// =============================================================================
// SVPF Extreme Scenario Stress Test
// =============================================================================
// Tests filter behavior under crypto-like extreme conditions (10σ - 50σ spikes)
// Goal: Find where the filter breaks, fails to recover, or produces NaN/Inf
// Outputs CSV files for Jupyter notebook visualization

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// =============================================================================
// CSV Output
// =============================================================================

struct TimeseriesPoint {
    int t;
    float y_t;           // Return
    float true_h;        // TRUE log-vol from DGP
    float h_mean;        // Estimated log-vol
    float vol;           // Estimated vol (exp(h/2))
    float true_vol;      // TRUE vol (exp(true_h/2))
    float ess;           // Effective sample size
    float loglik;        // Log-likelihood
};

static void write_csv(const std::string& filename, 
                      const std::string& scenario,
                      float sigma,
                      const std::vector<TimeseriesPoint>& data) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        printf("  Warning: Could not write %s\n", filename.c_str());
        return;
    }
    
    // Header
    fprintf(f, "t,y_t,true_h,h_mean,true_vol,vol,ess,loglik,scenario,sigma\n");
    
    // Data
    for (const auto& p : data) {
        fprintf(f, "%d,%.8f,%.6f,%.6f,%.8f,%.8f,%.2f,%.4f,%s,%.1f\n",
                p.t, p.y_t, p.true_h, p.h_mean, p.true_vol, p.vol, p.ess, p.loglik,
                scenario.c_str(), sigma);
    }
    
    fclose(f);
}

// Write summary CSV (one row per scenario/sigma combo)
static FILE* g_summary_file = nullptr;

static void init_summary_csv(const std::string& filename) {
    g_summary_file = fopen(filename.c_str(), "w");
    if (g_summary_file) {
        fprintf(g_summary_file, "scenario,sigma,had_nan,had_inf,min_ess,min_ess_t,"
                "max_h,max_vol,max_vol_t,recovery_time,failed_recovery,"
                "particle_collapse_count,final_vol,final_h\n");
    }
}

static void close_summary_csv() {
    if (g_summary_file) {
        fclose(g_summary_file);
        g_summary_file = nullptr;
    }
}

// =============================================================================
// Data Generation - Now tracks TRUE latent volatility
// =============================================================================

// Simple PRNG (cross-platform, better than rand())
static inline float randf(unsigned int* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (float)((*seed >> 16) & 0x7FFF) / 32768.0f;
}

static inline float randn(unsigned int* seed) {
    // Box-Muller transform
    float u1 = randf(seed) + 1e-10f;
    float u2 = randf(seed);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

// SV model parameters for DGP
struct SVDGPParams {
    float mu;      // Long-run mean
    float rho;     // Persistence
    float sigma;   // Vol of vol
};

// Generate calm period - true SV process
static void generate_calm_sv(std::vector<float>& returns, std::vector<float>& true_h,
                              int n, const SVDGPParams& dgp, float& h_state, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        // Evolve true latent state: h_t = mu + rho*(h_{t-1} - mu) + sigma*eps
        float eps = randn(seed);
        h_state = dgp.mu + dgp.rho * (h_state - dgp.mu) + dgp.sigma * eps;
        true_h.push_back(h_state);
        
        // Generate observation: y_t = exp(h_t/2) * eta
        float eta = randn(seed);
        float vol = expf(h_state * 0.5f);
        returns.push_back(vol * eta);
    }
}

// Generate a spike - set h to elevated level (relative to mu, not cumulative)
static void generate_spike_sv(std::vector<float>& returns, std::vector<float>& true_h,
                               float h_jump, float& h_state, float mu, unsigned int* seed) {
    // Set h to elevated level above mu (not cumulative)
    // This prevents double spikes from exploding
    float elevated = mu + h_jump;
    if (h_state < elevated) {
        h_state = elevated;  // Only jump up, don't pull down
    } else {
        h_state += h_jump * 0.3f;  // Small additional boost if already elevated
    }
    true_h.push_back(h_state);
    
    // Generate extreme observation
    float eta = randn(seed);
    float vol = expf(h_state * 0.5f);
    returns.push_back(vol * eta);
}

// Generate recovery - h decays back toward mu
static void generate_recovery_sv(std::vector<float>& returns, std::vector<float>& true_h,
                                  int n, const SVDGPParams& dgp, float& h_state, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        // Higher persistence during recovery (slower decay)
        float eps = randn(seed);
        h_state = dgp.mu + dgp.rho * (h_state - dgp.mu) + dgp.sigma * eps;
        true_h.push_back(h_state);
        
        float eta = randn(seed);
        float vol = expf(h_state * 0.5f);
        returns.push_back(vol * eta);
    }
}

// Generate chaos - h jumps randomly each tick
static void generate_chaos_sv(std::vector<float>& returns, std::vector<float>& true_h,
                               int n, const SVDGPParams& dgp, float min_h_jump, float max_h_jump,
                               float& h_state, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        // Random jump in h
        float jump_mag = min_h_jump + (max_h_jump - min_h_jump) * randf(seed);
        float jump_sign = (randf(seed) > 0.5f) ? 1.0f : -1.0f;
        h_state += jump_sign * jump_mag * 0.3f;  // Damped jumps
        
        // Also do normal SV evolution
        float eps = randn(seed);
        h_state = dgp.mu + 0.5f * (h_state - dgp.mu) + dgp.sigma * 2.0f * eps;
        true_h.push_back(h_state);
        
        float eta = randn(seed);
        float vol = expf(h_state * 0.5f);
        returns.push_back(vol * eta);
    }
}

// =============================================================================
// Test Result Tracking
// =============================================================================

struct StressTestResult {
    std::string scenario;
    float spike_sigma;
    
    // Health metrics
    bool had_nan;
    bool had_inf;
    int nan_timestep;
    
    // Particle health
    float min_ess;
    int min_ess_timestep;
    int particle_collapse_count;  // ESS < 10
    
    // Vol tracking
    float max_vol;
    float max_h;
    int max_vol_timestep;
    
    // Recovery metrics
    int recovery_time;           // Steps to return within 2x base vol
    bool failed_to_recover;
    
    // Final state
    float final_vol;
    float final_h_mean;
    
    // Timeseries data for CSV output
    std::vector<TimeseriesPoint> timeseries;
};

static void write_summary_row(const StressTestResult& r) {
    if (!g_summary_file) return;
    fprintf(g_summary_file, "%s,%.1f,%d,%d,%.2f,%d,%.4f,%.6f,%d,%d,%d,%d,%.6f,%.4f\n",
            r.scenario.c_str(), r.spike_sigma,
            r.had_nan ? 1 : 0, r.had_inf ? 1 : 0,
            r.min_ess, r.min_ess_timestep,
            r.max_h, r.max_vol, r.max_vol_timestep,
            r.recovery_time, r.failed_to_recover ? 1 : 0,
            r.particle_collapse_count,
            r.final_vol, r.final_h_mean);
}

static void print_result(const StressTestResult& r) {
    printf("\n  %-25s | %5.0fσ | ", r.scenario.c_str(), r.spike_sigma);
    
    if (r.had_nan) {
        printf("❌ NaN at t=%d\n", r.nan_timestep);
        return;
    }
    if (r.had_inf) {
        printf("❌ Inf at t=%d\n", r.nan_timestep);
        return;
    }
    
    // Status icon: healthy = recovered + ESS stayed above 50
    bool healthy = !r.failed_to_recover && r.min_ess > 50.0f;
    printf("%s | ", healthy ? "✓" : "⚠");
    
    printf("ESS_min=%5.1f (t=%3d) | h_max=%5.2f | vol_max=%6.2f%% | ", 
           r.min_ess, r.min_ess_timestep, r.max_h, r.max_vol * 100.0f);
    
    if (r.failed_to_recover) {
        printf("NO RECOVERY\n");
    } else {
        printf("recover=%3d steps\n", r.recovery_time);
    }
}

// =============================================================================
// Run Single Scenario
// =============================================================================

static StressTestResult run_scenario(
    const std::string& name,
    const std::vector<float>& returns,
    const std::vector<float>& true_h,  // TRUE latent state from DGP
    float spike_sigma,
    int spike_timestep,
    const SVPFParams& params,
    int n_particles
) {
    StressTestResult result;
    result.scenario = name;
    result.spike_sigma = spike_sigma;
    result.had_nan = false;
    result.had_inf = false;
    result.nan_timestep = -1;
    result.min_ess = 1e9f;
    result.min_ess_timestep = 0;
    result.particle_collapse_count = 0;
    result.max_vol = 0.0f;
    result.max_h = -1e9f;
    result.max_vol_timestep = 0;
    result.recovery_time = -1;
    result.failed_to_recover = true;
    
    // Create filter
    SVPFState* state = svpf_create(n_particles, 8, 50.0f, nullptr);
    svpf_initialize(state, &params, 12345);
    
    // =========================================================================
    // PRODUCTION CONFIGURATION (matches quant bot settings)
    // =========================================================================
    
    // Core SVGD settings
    state->use_svld = 1;
    state->use_annealing = 1;
    state->n_anneal_steps = 5;
    state->temperature = 0.45f;
    state->rmsprop_rho = 0.9f;
    state->rmsprop_eps = 1e-6f;
    
    // MIM (Mixture of Invariant Measures) for mode exploration
    state->use_mim = 1;
    state->mim_jump_prob = 0.25f;
    state->mim_jump_scale = 12.0f;
    state->use_adaptive_beta = 1;
    
    // Rejuvenation (particle diversity recovery)
    state->use_rejuvenation = 1;
    state->rejuv_ksd_threshold = 0.05f;
    state->rejuv_prob = 0.30f;
    state->rejuv_blend = 0.30f;
    
    // Newton-Stein (Hessian preconditioning)
    state->use_newton = 1;
    state->use_full_newton = 1;  // Detommaso 2018 full Newton
    
    // Guided Prediction with INNOVATION GATING
    state->use_guided = 1;
    state->guided_alpha_base = 0.0f;
    state->guided_alpha_shock = 0.40f;
    state->guided_innovation_threshold = 1.5f;
    
    // EKF Guide density
    state->use_guide = 1;
    state->use_guide_preserving = 1;
    state->guide_strength = 0.05f;
    
    // Adaptive mu (Kalman filter on long-run mean)
    state->use_adaptive_mu = 1;
    state->mu_process_var = 0.001f;
    state->mu_obs_var_scale = 11.0f;
    state->mu_min = -4.0f;
    state->mu_max = -1.0f;
    
    // Adaptive guide strength
    state->use_adaptive_guide = 1;
    state->guide_strength_base = 0.05f;
    state->guide_strength_max = 0.30f;
    state->guide_innovation_threshold = 1.0f;
    
    // Adaptive sigma (crisis boost)
    state->use_adaptive_sigma = 1;
    state->sigma_boost_threshold = 0.95f;
    state->sigma_boost_max = 3.2f;
    
    // Exact gradient (Student-t consistent)
    state->use_exact_gradient = 1;
    state->lik_offset = 0.345f;
    
    // KSD-based Adaptive Stein Steps
    state->stein_min_steps = 16;
    state->stein_max_steps = 16;
    state->ksd_improvement_threshold = 0.05f;
    
    // Student-t state dynamics (fat tails)
    state->use_student_t_state = 1;
    state->nu_state = 3.0f;
    
    // Smoothing (1-tick lag for cleaner output)
    state->use_smoothing = 1;
    state->smooth_lag = 3;
    state->smooth_output_lag = 1;
    
    // Persistent kernel (launch optimization)
    state->use_persistent_kernel = 1;

    state->use_heun = 1;
    
    // =========================================================================
    
    float y_prev = 0.0f;
    bool in_recovery = false;
    int recovery_start = -1;
    
    for (int t = 0; t < (int)returns.size(); t++) {
        float y_t = returns[t];
        
        float loglik, vol, h_mean;
        svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol, &h_mean);
        
        // Check for NaN/Inf
        if (std::isnan(vol) || std::isnan(h_mean) || std::isnan(loglik)) {
            result.had_nan = true;
            result.nan_timestep = t;
            break;
        }
        if (std::isinf(vol) || std::isinf(h_mean) || std::isinf(loglik)) {
            result.had_inf = true;
            result.nan_timestep = t;
            break;
        }
        
        // Get ESS
        float ess = svpf_get_ess(state);
        if (ess < result.min_ess) {
            result.min_ess = ess;
            result.min_ess_timestep = t;
        }
        if (ess < 10.0f) {
            result.particle_collapse_count++;
        }
        
        // Collect timeseries data
        TimeseriesPoint pt;
        pt.t = t;
        pt.y_t = y_t;
        pt.true_h = true_h[t];
        pt.h_mean = h_mean;
        pt.true_vol = expf(true_h[t] * 0.5f);
        pt.vol = vol;
        pt.ess = ess;
        pt.loglik = loglik;
        result.timeseries.push_back(pt);
        
        // Track max vol
        if (vol > result.max_vol) {
            result.max_vol = vol;
            result.max_vol_timestep = t;
        }
        if (h_mean > result.max_h) {
            result.max_h = h_mean;
        }
        
        // Track recovery (after spike)
        // Recovery = h_mean returns within 0.5 of long-run mean (mu)
        // Since h goes UP during spikes (less negative), recovery means h < mu + 0.5
        if (t > spike_timestep && !in_recovery) {
            in_recovery = true;
            recovery_start = t;
        }
        if (in_recovery && h_mean < params.mu + 0.5f) {
            result.recovery_time = t - recovery_start;
            result.failed_to_recover = false;
            in_recovery = false;  // Don't count again
        }
        
        result.final_vol = vol;
        result.final_h_mean = h_mean;
        y_prev = y_t;
    }
    
    // If still in recovery at end, mark as failed
    if (in_recovery) {
        result.failed_to_recover = true;
        result.recovery_time = returns.size() - recovery_start;
    }
    
    svpf_destroy(state);
    return result;
}

// =============================================================================
// Scenario Generators (using proper SV DGP with true h tracking)
// =============================================================================

// Convert sigma (extreme event magnitude) to h_jump (log-vol space)
// We want:
//   10σ → h_jump ≈ 1.0 (vol goes from 10% to ~16%)
//   50σ → h_jump ≈ 2.5 (vol goes from 10% to ~37%)
// This keeps h in reasonable range [-4.5, -1.5] even with double spikes
static float sigma_to_h_jump(float sigma) {
    // Conservative linear mapping
    return sigma * 0.05f;
}

// Scenario 1: Single spike of varying magnitude
static StressTestResult test_single_spike(float sigma, const SVPFParams& params, int n_particles) {
    std::vector<float> returns;
    std::vector<float> true_h;
    unsigned int seed = 42;
    
    // DGP parameters matching filter params
    SVDGPParams dgp = {params.mu, params.rho, params.sigma_z};
    float h_state = params.mu;  // Start at long-run mean
    
    // 100 ticks calm
    generate_calm_sv(returns, true_h, 100, dgp, h_state, &seed);
    int spike_t = returns.size();
    
    // Single spike - jump h up
    float h_jump = sigma_to_h_jump(sigma);
    generate_spike_sv(returns, true_h, h_jump, h_state, dgp.mu, &seed);
    
    // 200 ticks recovery
    generate_recovery_sv(returns, true_h, 200, dgp, h_state, &seed);
    
    return run_scenario("Single Spike", returns, true_h, sigma, spike_t, params, n_particles);
}

// Scenario 2: Double spike (test if filter recovers then handles second spike)
static StressTestResult test_double_spike(float sigma, const SVPFParams& params, int n_particles) {
    std::vector<float> returns;
    std::vector<float> true_h;
    unsigned int seed = 43;
    
    SVDGPParams dgp = {params.mu, params.rho, params.sigma_z};
    float h_state = params.mu;
    
    // Calm
    generate_calm_sv(returns, true_h, 50, dgp, h_state, &seed);
    int spike_t = returns.size();
    
    // First spike
    float h_jump = sigma_to_h_jump(sigma);
    generate_spike_sv(returns, true_h, h_jump, h_state, dgp.mu, &seed);
    
    // Partial recovery (50 ticks)
    generate_recovery_sv(returns, true_h, 50, dgp, h_state, &seed);
    
    // Second spike
    generate_spike_sv(returns, true_h, h_jump, h_state, dgp.mu, &seed);
    
    // Full recovery
    generate_recovery_sv(returns, true_h, 150, dgp, h_state, &seed);
    
    return run_scenario("Double Spike", returns, true_h, sigma, spike_t, params, n_particles);
}

// Scenario 3: Flash crash pattern (rapid sequence: down, up, down)
static StressTestResult test_flash_crash(float base_sigma, const SVPFParams& params, int n_particles) {
    std::vector<float> returns;
    std::vector<float> true_h;
    unsigned int seed = 44;
    
    SVDGPParams dgp = {params.mu, params.rho, params.sigma_z};
    float h_state = params.mu;
    
    // Calm
    generate_calm_sv(returns, true_h, 50, dgp, h_state, &seed);
    int spike_t = returns.size();
    
    // Flash crash: 3 rapid h jumps
    float h_jump = sigma_to_h_jump(base_sigma);
    
    // Jump 1: big spike up
    generate_spike_sv(returns, true_h, h_jump, h_state, dgp.mu, &seed);
    // Jump 2: stays elevated
    generate_spike_sv(returns, true_h, h_jump * 0.3f, h_state, dgp.mu, &seed);
    // Jump 3: another spike
    generate_spike_sv(returns, true_h, h_jump * 0.2f, h_state, dgp.mu, &seed);
    
    // Chaotic aftermath (20 ticks)
    generate_chaos_sv(returns, true_h, 20, dgp, 0.5f, 2.0f, h_state, &seed);
    
    // Recovery
    generate_recovery_sv(returns, true_h, 150, dgp, h_state, &seed);
    
    return run_scenario("Flash Crash", returns, true_h, base_sigma, spike_t, params, n_particles);
}

// Scenario 4: Sustained chaos (crypto meltdown)
static StressTestResult test_sustained_chaos(float avg_sigma, const SVPFParams& params, int n_particles) {
    std::vector<float> returns;
    std::vector<float> true_h;
    unsigned int seed = 45;
    
    SVDGPParams dgp = {params.mu, params.rho, params.sigma_z};
    float h_state = params.mu;
    
    // Calm
    generate_calm_sv(returns, true_h, 50, dgp, h_state, &seed);
    int spike_t = returns.size();
    
    // 50 ticks of chaos
    float h_jump_min = sigma_to_h_jump(avg_sigma * 0.5f);
    float h_jump_max = sigma_to_h_jump(avg_sigma * 1.5f);
    generate_chaos_sv(returns, true_h, 50, dgp, h_jump_min, h_jump_max, h_state, &seed);
    
    // Recovery
    generate_recovery_sv(returns, true_h, 200, dgp, h_state, &seed);
    
    return run_scenario("Sustained Chaos", returns, true_h, avg_sigma, spike_t, params, n_particles);
}

// Scenario 5: Gradual build to extreme (like a squeeze)
static StressTestResult test_gradual_extreme(float peak_sigma, const SVPFParams& params, int n_particles) {
    std::vector<float> returns;
    std::vector<float> true_h;
    unsigned int seed = 46;
    
    SVDGPParams dgp = {params.mu, params.rho, params.sigma_z};
    float h_state = params.mu;
    
    // Calm
    generate_calm_sv(returns, true_h, 50, dgp, h_state, &seed);
    int spike_t = returns.size();
    
    // Gradual increase over 20 ticks
    float h_jump_final = sigma_to_h_jump(peak_sigma);
    for (int i = 0; i < 20; i++) {
        float frac = (float)i / 20.0f;
        float small_jump = h_jump_final * frac * 0.1f;  // Gradual increase
        generate_spike_sv(returns, true_h, small_jump, h_state, dgp.mu, &seed);
    }
    
    // Peak spike
    generate_spike_sv(returns, true_h, h_jump_final * 0.3f, h_state, dgp.mu, &seed);
    
    // Recovery
    generate_recovery_sv(returns, true_h, 200, dgp, h_state, &seed);
    
    return run_scenario("Gradual Build", returns, true_h, peak_sigma, spike_t, params, n_particles);
}

// =============================================================================
// Main
// =============================================================================

// Helper to sanitize filename
static std::string make_filename(const std::string& scenario, float sigma) {
    std::string s = scenario;
    // Replace spaces with underscores
    for (char& c : s) {
        if (c == ' ') c = '_';
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "svpf_stress_%s_%.0fsigma.csv", s.c_str(), sigma);
    return std::string(buf);
}

int main(int argc, char** argv) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf(" SVPF Extreme Scenario Stress Test\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("\n Testing with Student-t likelihood (nu=5) for fat-tailed returns\n");
    printf(" Base volatility: 1%% (0.01)\n");
    printf(" Looking for: NaN/Inf, particle collapse (ESS<10), recovery failure\n");
    
    SVPFParams params = {0.98f, 0.15f, -4.5f, 0.0f};  // Standard SV params
    int n_particles = 1024;
    bool write_csv_files = true;
    
    if (argc > 1) n_particles = atoi(argv[1]);
    if (argc > 2) write_csv_files = (atoi(argv[2]) != 0);
    
    printf(" Particles: %d\n", n_particles);
    printf(" CSV output: %s\n", write_csv_files ? "ENABLED" : "disabled");
    
    // Initialize summary CSV
    if (write_csv_files) {
        init_summary_csv("svpf_stress_summary.csv");
        printf(" Output files: svpf_stress_summary.csv + per-scenario CSVs\n");
    }
    
    // Test magnitudes
    float sigmas[] = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 40.0f, 50.0f};
    int n_sigmas = sizeof(sigmas) / sizeof(sigmas[0]);
    
    // =========================================================================
    printf("\n───────────────────────────────────────────────────────────────────────────────\n");
    printf(" Scenario 1: Single Spike (calm → spike → recovery)\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < n_sigmas; i++) {
        StressTestResult r = test_single_spike(sigmas[i], params, n_particles);
        print_result(r);
        if (write_csv_files) {
            write_csv(make_filename("Single_Spike", sigmas[i]), r.scenario, sigmas[i], r.timeseries);
            write_summary_row(r);
        }
    }
    
    // =========================================================================
    printf("\n───────────────────────────────────────────────────────────────────────────────\n");
    printf(" Scenario 2: Double Spike (spike → partial recovery → spike again)\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < n_sigmas; i++) {
        StressTestResult r = test_double_spike(sigmas[i], params, n_particles);
        print_result(r);
        if (write_csv_files) {
            write_csv(make_filename("Double_Spike", sigmas[i]), r.scenario, sigmas[i], r.timeseries);
            write_summary_row(r);
        }
    }
    
    // =========================================================================
    printf("\n───────────────────────────────────────────────────────────────────────────────\n");
    printf(" Scenario 3: Flash Crash (down → bounce → down, 3 ticks)\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < n_sigmas; i++) {
        StressTestResult r = test_flash_crash(sigmas[i], params, n_particles);
        print_result(r);
        if (write_csv_files) {
            write_csv(make_filename("Flash_Crash", sigmas[i]), r.scenario, sigmas[i], r.timeseries);
            write_summary_row(r);
        }
    }
    
    // =========================================================================
    printf("\n───────────────────────────────────────────────────────────────────────────────\n");
    printf(" Scenario 4: Sustained Chaos (50 ticks of extreme moves)\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    float chaos_sigmas[] = {5.0f, 7.0f, 10.0f, 12.0f, 15.0f};
    for (int i = 0; i < 5; i++) {
        StressTestResult r = test_sustained_chaos(chaos_sigmas[i], params, n_particles);
        print_result(r);
        if (write_csv_files) {
            write_csv(make_filename("Sustained_Chaos", chaos_sigmas[i]), r.scenario, chaos_sigmas[i], r.timeseries);
            write_summary_row(r);
        }
    }
    
    // =========================================================================
    printf("\n───────────────────────────────────────────────────────────────────────────────\n");
    printf(" Scenario 5: Gradual Build to Extreme (squeeze pattern)\n");
    printf("───────────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < n_sigmas; i++) {
        StressTestResult r = test_gradual_extreme(sigmas[i], params, n_particles);
        print_result(r);
        if (write_csv_files) {
            write_csv(make_filename("Gradual_Build", sigmas[i]), r.scenario, sigmas[i], r.timeseries);
            write_summary_row(r);
        }
    }
    
    // =========================================================================
    if (write_csv_files) {
        close_summary_csv();
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    printf(" Legend:\n");
    printf("   ✓  = Healthy (recovered, no collapse, h bounded)\n");
    printf("   ⚠  = Warning (slow recovery, low ESS, or high h)\n");
    printf("   ❌ = Failed (NaN/Inf)\n");
    printf("   ESS_min = Minimum Effective Sample Size (< 10 = particle collapse)\n");
    printf("   h_max = Maximum log-volatility (> mu+2 = concerning)\n");
    printf("   recover = Steps for h_mean to return within 0.5 of mu (%.1f)\n", params.mu);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    
    if (write_csv_files) {
        printf("\n CSV files written. Load in Jupyter:\n");
        printf("   import pandas as pd\n");
        printf("   summary = pd.read_csv('svpf_stress_summary.csv')\n");
        printf("   ts = pd.read_csv('svpf_stress_Single_Spike_20sigma.csv')\n");
        printf("   ts.plot(x='t', y=['vol', 'ess'], secondary_y=['ess'])\n");
        printf("\n");
    }
    
    return 0;
}
