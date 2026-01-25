/**
 * @file svpf_diagnostics.cu
 * @brief Implementation of SVPF error handling and health monitoring
 */

#include "svpf_diagnostics.h"
#include "svpf.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// =============================================================================
// Status Names - C++ compatible (no designated initializers)
// =============================================================================

const char* svpf_status_name(SVPFStatus status) {
    switch (status) {
        // Success
        case SVPF_OK: return "OK";
        
        // CUDA errors
        case SVPF_ERR_CUDA_INIT: return "CUDA_INIT_FAILED";
        case SVPF_ERR_CUDA_MALLOC: return "CUDA_MALLOC_FAILED";
        case SVPF_ERR_CUDA_MEMCPY: return "CUDA_MEMCPY_FAILED";
        case SVPF_ERR_CUDA_KERNEL: return "CUDA_KERNEL_FAILED";
        case SVPF_ERR_CUDA_GRAPH: return "CUDA_GRAPH_FAILED";
        case SVPF_ERR_CUDA_SYNC: return "CUDA_SYNC_FAILED";
        case SVPF_ERR_CUDA_DEVICE: return "CUDA_DEVICE_ERROR";
        
        // Numerical errors
        case SVPF_ERR_NAN_PARTICLES: return "NAN_IN_PARTICLES";
        case SVPF_ERR_NAN_WEIGHTS: return "NAN_IN_WEIGHTS";
        case SVPF_ERR_NAN_OUTPUT: return "NAN_IN_OUTPUT";
        case SVPF_ERR_INF_PARTICLES: return "INF_IN_PARTICLES";
        case SVPF_ERR_INF_WEIGHTS: return "INF_IN_WEIGHTS";
        case SVPF_ERR_PARTICLE_COLLAPSE: return "PARTICLE_COLLAPSE";
        case SVPF_ERR_WEIGHT_UNDERFLOW: return "WEIGHT_UNDERFLOW";
        case SVPF_ERR_BANDWIDTH_ZERO: return "BANDWIDTH_ZERO";
        
        // Input errors
        case SVPF_ERR_INPUT_NAN: return "INPUT_IS_NAN";
        case SVPF_ERR_INPUT_INF: return "INPUT_IS_INF";
        case SVPF_ERR_INPUT_EXTREME: return "INPUT_EXTREME_VALUE";
        case SVPF_ERR_PARAMS_INVALID: return "PARAMS_INVALID";
        
        // State errors
        case SVPF_ERR_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case SVPF_ERR_ALREADY_DESTROYED: return "ALREADY_DESTROYED";
        case SVPF_ERR_STATE_CORRUPTED: return "STATE_CORRUPTED";
        
        // Resource errors
        case SVPF_ERR_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case SVPF_ERR_SHARED_MEM_EXCEEDED: return "SHARED_MEM_EXCEEDED";
        
        // Warnings
        case SVPF_WARN_ESS_LOW: return "ESS_LOW";
        case SVPF_WARN_VOL_EXTREME: return "VOL_EXTREME";
        case SVPF_WARN_PARTICLES_CLUSTERED: return "PARTICLES_CLUSTERED";
        case SVPF_WARN_GRAPH_RECAPTURE: return "GRAPH_RECAPTURE";
        
        default: return "UNKNOWN_STATUS";
    }
}

// =============================================================================
// Initialization
// =============================================================================

void svpf_diag_init(SVPFDiagnostics* diag) {
    memset(diag, 0, sizeof(SVPFDiagnostics));
    
    diag->status = SVPF_OK;
    diag->ess_ratio = 1.0f;
    
    // Default thresholds
    diag->ess_warning_threshold = 0.3f;
    diag->vol_max_threshold = 10.0f;       // 1000% annualized
    diag->vol_min_threshold = 0.0001f;     // 0.01% annualized
    diag->input_max_return = 0.5f;         // 50% single-period move
}

// =============================================================================
// Input Validation
// =============================================================================

SVPFStatus svpf_diag_check_input(float y_t, SVPFDiagnostics* diag) {
    // Check NaN
    if (y_t != y_t) {
        diag->status = SVPF_ERR_INPUT_NAN;
        snprintf(diag->last_error_message, 256, 
                 "Input return is NaN");
        diag->last_error = diag->status;
        diag->error_count++;
        return diag->status;
    }
    
    // Check Inf
    if (y_t > 1e30f || y_t < -1e30f) {
        diag->status = SVPF_ERR_INPUT_INF;
        snprintf(diag->last_error_message, 256,
                 "Input return is infinite: %.2e", y_t);
        diag->last_error = diag->status;
        diag->error_count++;
        return diag->status;
    }
    
    // Check extreme values
    if (fabsf(y_t) > diag->input_max_return) {
        diag->status = SVPF_ERR_INPUT_EXTREME;
        snprintf(diag->last_error_message, 256,
                 "Input return extreme: %.4f (threshold: %.4f)",
                 y_t, diag->input_max_return);
        diag->last_error = diag->status;
        diag->error_count++;
        return diag->status;
    }
    
    return SVPF_OK;
}

// =============================================================================
// Health Check Kernel
// =============================================================================

__global__ void svpf_diag_check_kernel(
    const float* __restrict__ h,
    const float* __restrict__ log_w,
    float* __restrict__ d_results,  // [nan_count, inf_count, sum_w, sum_w2, min_h, max_h, sum_h]
    int n
) {
    __shared__ float s_nan_count;
    __shared__ float s_inf_count;
    __shared__ float s_sum_w;
    __shared__ float s_sum_w2;
    __shared__ float s_min_h;
    __shared__ float s_max_h;
    __shared__ float s_sum_h;
    
    if (threadIdx.x == 0) {
        s_nan_count = 0;
        s_inf_count = 0;
        s_sum_w = 0;
        s_sum_w2 = 0;
        s_min_h = 1e10f;
        s_max_h = -1e10f;
        s_sum_h = 0;
    }
    __syncthreads();
    
    float local_nan = 0, local_inf = 0;
    float local_sum_w = 0, local_sum_w2 = 0;
    float local_min_h = 1e10f, local_max_h = -1e10f, local_sum_h = 0;
    
    // Find max log weight for numerical stability
    float max_log_w = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float lw = log_w[i];
        if (lw == lw && lw < 1e10f) {  // valid
            max_log_w = fmaxf(max_log_w, lw);
        }
    }
    // Simple reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_log_w = fmaxf(max_log_w, __shfl_down_sync(0xffffffff, max_log_w, offset));
    }
    __shared__ float s_max_log_w;
    if (threadIdx.x == 0) s_max_log_w = max_log_w;
    __syncthreads();
    max_log_w = s_max_log_w;
    
    // Main loop
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float h_i = h[i];
        float lw_i = log_w[i];
        
        // Check NaN
        if (h_i != h_i) local_nan += 1.0f;
        if (lw_i != lw_i) local_nan += 1.0f;
        
        // Check Inf
        if (h_i > 1e10f || h_i < -1e10f) local_inf += 1.0f;
        if (lw_i > 1e10f) local_inf += 1.0f;
        
        // Particle stats
        if (h_i == h_i && h_i > -1e10f && h_i < 1e10f) {
            local_min_h = fminf(local_min_h, h_i);
            local_max_h = fmaxf(local_max_h, h_i);
            local_sum_h += h_i;
        }
        
        // Weight stats (for ESS)
        if (lw_i == lw_i && lw_i < 1e10f) {
            float w_i = __expf(lw_i - max_log_w);
            local_sum_w += w_i;
            local_sum_w2 += w_i * w_i;
        }
    }
    
    // Reduce
    atomicAdd(&s_nan_count, local_nan);
    atomicAdd(&s_inf_count, local_inf);
    atomicAdd(&s_sum_w, local_sum_w);
    atomicAdd(&s_sum_w2, local_sum_w2);
    atomicAdd(&s_sum_h, local_sum_h);
    
    // Min/max need atomicMin/Max which don't exist for float, use CAS
    // For simplicity, just use thread 0 after sync
    __syncthreads();
    
    // Final reduction for min/max (simplified - could be optimized)
    if (threadIdx.x == 0) {
        d_results[0] = s_nan_count;
        d_results[1] = s_inf_count;
        d_results[2] = s_sum_w;
        d_results[3] = s_sum_w2;
        d_results[4] = s_min_h;  // Note: not fully reduced, just from thread 0
        d_results[5] = s_max_h;
        d_results[6] = s_sum_h;
    }
}

// =============================================================================
// Main Health Check
// =============================================================================

SVPFStatus svpf_diag_check(void* state_ptr, SVPFDiagnostics* diag) {
    SVPFState* state = (SVPFState*)state_ptr;
    
    // Check state validity
    if (state == NULL) {
        diag->status = SVPF_ERR_NOT_INITIALIZED;
        snprintf(diag->last_error_message, 256, "SVPF state is NULL");
        diag->last_error = diag->status;
        diag->error_count++;
        return diag->status;
    }
    
    // Check CUDA errors
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        diag->status = SVPF_ERR_CUDA_KERNEL;
        snprintf(diag->last_error_message, 256,
                 "CUDA error: %s", cudaGetErrorString(cuda_err));
        diag->last_error = diag->status;
        diag->error_count++;
        return diag->status;
    }
    
    // Allocate results buffer (could be pre-allocated for perf)
    float* d_results;
    float h_results[7];
    cudaMalloc(&d_results, 7 * sizeof(float));
    
    // Run diagnostic kernel
    svpf_diag_check_kernel<<<1, 256>>>(
        state->h, state->log_weights, d_results, state->n_particles
    );
    
    cudaMemcpy(h_results, d_results, 7 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    
    float nan_count = h_results[0];
    float inf_count = h_results[1];
    float sum_w = h_results[2];
    float sum_w2 = h_results[3];
    float sum_h = h_results[6];
    
    // Check for NaN
    if (nan_count > 0) {
        diag->status = SVPF_ERR_NAN_PARTICLES;
        snprintf(diag->last_error_message, 256,
                 "NaN detected in %.0f particles/weights", nan_count);
        diag->last_error = diag->status;
        diag->last_error_timestep = diag->steps_processed;
        diag->error_count++;
        return diag->status;
    }
    
    // Check for Inf
    if (inf_count > 0) {
        diag->status = SVPF_ERR_INF_PARTICLES;
        snprintf(diag->last_error_message, 256,
                 "Inf detected in %.0f particles/weights", inf_count);
        diag->last_error = diag->status;
        diag->last_error_timestep = diag->steps_processed;
        diag->error_count++;
        return diag->status;
    }
    
    // Compute ESS
    if (sum_w > 1e-10f) {
        diag->ess = (sum_w * sum_w) / sum_w2;
        diag->ess_ratio = diag->ess / (float)state->n_particles;
    } else {
        diag->status = SVPF_ERR_WEIGHT_UNDERFLOW;
        snprintf(diag->last_error_message, 256,
                 "Weight underflow: sum_w = %.2e", sum_w);
        diag->last_error = diag->status;
        diag->error_count++;
        return diag->status;
    }
    
    // Store outputs
    diag->h_mean = sum_h / (float)state->n_particles;
    diag->vol_estimate = expf(diag->h_mean / 2.0f);
    
    // Warnings (don't return early, just flag)
    diag->status = SVPF_OK;
    
    if (diag->ess_ratio < diag->ess_warning_threshold) {
        diag->status = SVPF_WARN_ESS_LOW;
        snprintf(diag->last_error_message, 256,
                 "ESS low: %.1f%% (threshold: %.1f%%)",
                 diag->ess_ratio * 100.0f, diag->ess_warning_threshold * 100.0f);
        diag->warning_count++;
    }
    
    if (diag->vol_estimate > diag->vol_max_threshold) {
        diag->status = SVPF_WARN_VOL_EXTREME;
        snprintf(diag->last_error_message, 256,
                 "Vol estimate extreme high: %.4f", diag->vol_estimate);
        diag->warning_count++;
    }
    
    if (diag->vol_estimate < diag->vol_min_threshold) {
        diag->status = SVPF_WARN_VOL_EXTREME;
        snprintf(diag->last_error_message, 256,
                 "Vol estimate extreme low: %.6f", diag->vol_estimate);
        diag->warning_count++;
    }
    
    diag->steps_processed++;
    return diag->status;
}

// =============================================================================
// Utility Functions
// =============================================================================

const char* svpf_diag_message(const SVPFDiagnostics* diag) {
    return diag->last_error_message;
}

bool svpf_status_is_error(SVPFStatus status) {
    return status > 0 && status < 1000;
}

bool svpf_status_is_warning(SVPFStatus status) {
    return status >= 1000;
}

void svpf_diag_clear_error(SVPFDiagnostics* diag) {
    diag->status = SVPF_OK;
    diag->last_error_message[0] = '\0';
}

void svpf_diag_print_summary(const SVPFDiagnostics* diag) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    SVPF Diagnostics Summary                       ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Status:          %-46s ║\n", svpf_status_name(diag->status));
    printf("║  Steps processed: %-46llu ║\n", (unsigned long long)diag->steps_processed);
    printf("║  Errors:          %-46u ║\n", diag->error_count);
    printf("║  Warnings:        %-46u ║\n", diag->warning_count);
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  ESS ratio:       %-6.1f%% (threshold: %.0f%%)                       ║\n",
           diag->ess_ratio * 100.0f, diag->ess_warning_threshold * 100.0f);
    printf("║  Vol estimate:    %-46.4f ║\n", diag->vol_estimate);
    printf("║  h_mean:          %-46.4f ║\n", diag->h_mean);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    if (diag->status != SVPF_OK) {
        printf("  Last message: %s\n", diag->last_error_message);
    }
    printf("\n");
}

int svpf_diag_to_json(const SVPFDiagnostics* diag, char* buffer, int size) {
    return snprintf(buffer, size,
        "{"
        "\"status\":\"%s\","
        "\"status_code\":%d,"
        "\"steps\":%llu,"
        "\"errors\":%u,"
        "\"warnings\":%u,"
        "\"ess_ratio\":%.4f,"
        "\"vol_estimate\":%.6f,"
        "\"h_mean\":%.4f,"
        "\"message\":\"%s\""
        "}",
        svpf_status_name(diag->status),
        (int)diag->status,
        (unsigned long long)diag->steps_processed,
        diag->error_count,
        diag->warning_count,
        diag->ess_ratio,
        diag->vol_estimate,
        diag->h_mean,
        diag->status != SVPF_OK ? diag->last_error_message : ""
    );
}
