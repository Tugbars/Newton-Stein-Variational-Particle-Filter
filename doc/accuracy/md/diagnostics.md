# SVPF Diagnostics Module

## Overview

Production systems need to detect, report, and recover from errors gracefully. The diagnostics module provides:

1. **Error detection** - CUDA errors, numerical issues (NaN/Inf), algorithmic failures
2. **Health monitoring** - ESS, particle spread, weight entropy
3. **Clear reporting** - Human-readable messages, JSON output for logging
4. **Graceful recovery** - Input validation, error clearing

## Quick Start

```c
#include "svpf.cuh"
#include "svpf_diagnostics.h"

// Initialize
SVPFState* state = svpf_create(512, 5, 7.0f, nullptr);
SVPFDiagnostics diag;
svpf_diag_init(&diag);

// Main loop
for (int t = 0; t < T; t++) {
    // 1. Validate input
    if (svpf_status_is_error(svpf_diag_check_input(y_t, &diag))) {
        printf("Bad input: %s\n", svpf_diag_message(&diag));
        continue;
    }
    
    // 2. Run filter
    svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol, &h_mean);
    
    // 3. Check health
    if (svpf_status_is_error(svpf_diag_check(state, &diag))) {
        printf("Filter error: %s\n", svpf_diag_message(&diag));
        svpf_initialize(state, &params, new_seed);  // Recover
    }
}

// Summary
svpf_diag_print_summary(&diag);
```

## Error Categories

### CUDA Errors (100-199)

| Code | Name | Cause |
|------|------|-------|
| 100 | `CUDA_INIT_FAILED` | GPU initialization failed |
| 101 | `CUDA_MALLOC_FAILED` | Out of GPU memory |
| 102 | `CUDA_MEMCPY_FAILED` | Data transfer failed |
| 103 | `CUDA_KERNEL_FAILED` | Kernel execution error |
| 104 | `CUDA_GRAPH_FAILED` | Graph capture/launch error |

### Numerical Errors (200-299)

| Code | Name | Cause |
|------|------|-------|
| 200 | `NAN_IN_PARTICLES` | NaN detected in particle array |
| 201 | `NAN_IN_WEIGHTS` | NaN detected in weights |
| 210 | `PARTICLE_COLLAPSE` | All particles at same location |
| 211 | `WEIGHT_UNDERFLOW` | All weights effectively zero |

### Input Errors (300-399)

| Code | Name | Cause |
|------|------|-------|
| 300 | `INPUT_IS_NAN` | Input return is NaN |
| 301 | `INPUT_IS_INF` | Input return is infinite |
| 302 | `INPUT_EXTREME` | Return exceeds threshold |

### Warnings (1000+)

| Code | Name | Meaning |
|------|------|---------|
| 1000 | `ESS_LOW` | Effective sample size below threshold |
| 1001 | `VOL_EXTREME` | Vol estimate outside reasonable range |
| 1002 | `PARTICLES_CLUSTERED` | Low particle diversity |

## Configuration

```c
SVPFDiagnostics diag;
svpf_diag_init(&diag);

// Customize thresholds
diag.ess_warning_threshold = 0.3f;    // Warn if ESS < 30%
diag.vol_max_threshold = 10.0f;       // Max vol (1000% annualized)
diag.vol_min_threshold = 0.0001f;     // Min vol
diag.input_max_return = 0.5f;         // Max single-period return
```

## Health Metrics

The `SVPFDiagnostics` struct tracks:

| Field | Description |
|-------|-------------|
| `ess` | Effective sample size |
| `ess_ratio` | ESS / N (target: > 0.5) |
| `vol_estimate` | Current vol estimate |
| `h_mean` | Mean log-vol |
| `steps_processed` | Total steps run |
| `error_count` | Cumulative errors |
| `warning_count` | Cumulative warnings |

## Output Formats

### Human Readable

```c
svpf_diag_print_summary(&diag);
```

Output:
```
╔═══════════════════════════════════════════════════════════════════╗
║                    SVPF Diagnostics Summary                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  Status:          OK                                              ║
║  Steps processed: 10000                                           ║
║  Errors:          0                                               ║
║  Warnings:        3                                               ║
╠═══════════════════════════════════════════════════════════════════╣
║  ESS ratio:       72.3% (threshold: 30%)                          ║
║  Vol estimate:    0.0124                                          ║
╚═══════════════════════════════════════════════════════════════════╝
```

### JSON (for logging/monitoring)

```c
char buf[512];
svpf_diag_to_json(&diag, buf, sizeof(buf));
// {"status":"OK","status_code":0,"steps":10000,"errors":0,...}
```

## Recovery Strategies

### Bad Input

```c
if (svpf_status_is_error(svpf_diag_check_input(y_t, &diag))) {
    // Option 1: Skip
    continue;
    
    // Option 2: Use last valid value
    y_t = y_prev;
    
    // Option 3: Interpolate
    y_t = (y_prev + y_next) / 2;
}
```

### Filter Error

```c
if (svpf_status_is_error(svpf_diag_check(state, &diag))) {
    // Option 1: Reinitialize
    svpf_initialize(state, &params, new_seed);
    
    // Option 2: Reset with last known good state
    // (requires checkpointing)
    
    // Option 3: Failover to backup estimate
    vol = historical_vol_estimate;
}
```

### Low ESS Warning

```c
if (diag.status == SVPF_WARN_ESS_LOW) {
    // The filter will resample automatically
    // But you might want to:
    // - Log for analysis
    // - Temporarily increase particle count
    // - Be cautious about vol estimate confidence
}
```

## Performance

The health check adds minimal overhead:
- `svpf_diag_check_input()`: ~100 ns (inline float checks)
- `svpf_diag_check()`: ~10-20 μs (launches small kernel)

For latency-critical paths, you can:
1. Check input every step (cheap)
2. Check health every N steps or on demand

```c
// Check health every 100 steps
if (t % 100 == 0) {
    svpf_diag_check(state, &diag);
}
```

## Files

| File | Description |
|------|-------------|
| `include/svpf_diagnostics.h` | Header with API and error codes |
| `src/svpf_diagnostics.cu` | Implementation |
| `examples/example_diagnostics.cu` | Usage example |
