/*═══════════════════════════════════════════════════════════════════════════
 * RUN SVPF ON SCENARIO (Refactored — all features always-on)
 *═══════════════════════════════════════════════════════════════════════════*/

static Metrics run_svpf_on_scenario(
    TestData* data,
    int n_particles,
    int n_stein,
    float nu,
    int use_optimized,      // 1 = svpf_step_graph (production), 0 = svpf_step (basic)
    int seed,
    double* elapsed_ms_out
) {
    int n = data->n_ticks;

    /* Allocate outputs */
    float* h_returns = (float*)malloc(n * sizeof(float));
    float* h_loglik  = (float*)malloc(n * sizeof(float));
    float* h_vol     = (float*)malloc(n * sizeof(float));
    float* h_logvol  = (float*)malloc(n * sizeof(float));

    for (int t = 0; t < n; t++) {
        h_returns[t] = (float)data->returns[t];
    }

    /*
     * SVPF parameters — note these are MISSPECIFIED for this DGP!
     * SVPF assumes fixed AR(1) parameters, but DGP has z-dependent params.
     * This is intentional — tests SVPF robustness to misspecification.
     */
    SVPFParams params;
    params.rho     = 0.97f;
    params.sigma_z = 0.15f;
    params.mu      = -4.5f;
    params.gamma   = 0.0f;

    /* Create filter — all features always-on in refactored build */
    SVPFState* filter = svpf_create(n_particles, n_stein, nu, NULL);
    svpf_initialize(filter, &params, seed);

    if (use_optimized) {
        /* ── Stein transport ── */
        filter->temperature     = 0.45f;
        filter->rmsprop_rho     = 0.7f;
        filter->rmsprop_eps     = 1e-6f;

        /* ── MIM (disabled via zero probability) ── */
        filter->mim_jump_prob   = 0.0f;
        filter->mim_jump_scale  = 9.0f;

        /* ── Rejuvenation (Maken 2022) ── */
        filter->rejuv_ksd_threshold = 0.05f;
        filter->rejuv_prob          = 0.30f;
        filter->rejuv_blend         = 0.30f;

        /* ── Guided prediction (innovation-gated) ── */
        filter->guided_alpha_base           = 0.0f;
        filter->guided_alpha_shock          = 0.40f;
        filter->guided_innovation_threshold = 1.5f;

        /* ── EKF guide density (adaptive strength) ── */
        filter->guide_strength_base       = 0.05f;
        filter->guide_strength_max        = 0.30f;
        filter->guide_innovation_threshold = 1.0f;

        /* ── Adaptive mu (Kalman) ── */
        filter->mu_process_var   = 0.001f;
        filter->mu_obs_var_scale = 11.0f;
        filter->mu_min           = -4.0f;
        filter->mu_max           = -1.0f;

        /* ── Adaptive sigma ("breathing") ── */
        filter->sigma_boost_threshold = 0.95f;
        filter->sigma_boost_max       = 3.2f;

        /* ── Likelihood ── */
        filter->lik_offset = 0.345f;

        /* ── Student-t state dynamics ── */
        filter->nu_state = 5.0f;

        /* ── Backward smoothing ── */
        filter->smooth_lag        = 3;
        filter->smooth_output_lag = 1;

        /* ── Adaptive annealing ── */
        filter->anneal_kl_threshold  = 0.9f;
        filter->anneal_steps_per_beta = 3;
    }
    /* No else branch needed — svpf_create sets sane defaults for all fields.
     * The basic path (use_optimized=0) just runs with those defaults via svpf_step(). */

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

        if (use_optimized) {
            svpf_step_graph(filter, y_t, y_prev, &params,
                            &h_loglik[t], &h_vol[t], &h_logvol[t]);
        } else {
            SVPFResult result;
            svpf_step(filter, y_t, &params, &result);
            h_loglik[t] = result.log_lik_increment;
            h_vol[t]    = result.vol_mean;
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
    /* Skip warmup (first 100 steps) for percentile calculation */
    int warmup = 100;
    int effective_n = n - warmup;
    double* sorted_latencies = step_latencies + warmup;

    qsort(sorted_latencies, effective_n, sizeof(double), compare_double_for_qsort);

    double p50  = sorted_latencies[effective_n / 2];
    double p90  = sorted_latencies[(int)(effective_n * 0.90)];
    double p99  = sorted_latencies[(int)(effective_n * 0.99)];
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
