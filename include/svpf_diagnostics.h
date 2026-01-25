/**
 * @file svpf_diagnostics.h
 * @brief Error handling, health monitoring, and diagnostics for SVPF
 * 
 * Production systems need to:
 *   1. Detect errors (CUDA, numerical, algorithmic)
 *   2. Report clearly what went wrong
 *   3. Recover gracefully when possible
 *   4. Log for post-mortem analysis
 * 
 * Usage:
 *   SVPFDiagnostics diag;
 *   svpf_diag_init(&diag);
 *   
 *   // In your main loop:
 *   svpf_step_graph(state, y_t, y_prev, &params, &loglik, &vol, &h_mean);
 *   svpf_diag_check(state, &diag);
 *   
 *   if (diag.status != SVPF_OK) {
 *       printf("SVPF Error: %s\n", svpf_diag_message(&diag));
 *       // Handle error...
 *   }
 */

#ifndef SVPF_DIAGNOSTICS_H
#define SVPF_DIAGNOSTICS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Error Codes
// =============================================================================

typedef enum {
    // Success
    SVPF_OK = 0,
    
    // CUDA Errors (100-199)
    SVPF_ERR_CUDA_INIT = 100,
    SVPF_ERR_CUDA_MALLOC = 101,
    SVPF_ERR_CUDA_MEMCPY = 102,
    SVPF_ERR_CUDA_KERNEL = 103,
    SVPF_ERR_CUDA_GRAPH = 104,
    SVPF_ERR_CUDA_SYNC = 105,
    SVPF_ERR_CUDA_DEVICE = 106,
    
    // Numerical Errors (200-299)
    SVPF_ERR_NAN_PARTICLES = 200,
    SVPF_ERR_NAN_WEIGHTS = 201,
    SVPF_ERR_NAN_OUTPUT = 202,
    SVPF_ERR_INF_PARTICLES = 203,
    SVPF_ERR_INF_WEIGHTS = 204,
    SVPF_ERR_PARTICLE_COLLAPSE = 210,
    SVPF_ERR_WEIGHT_UNDERFLOW = 211,
    SVPF_ERR_BANDWIDTH_ZERO = 212,
    
    // Input Errors (300-399)
    SVPF_ERR_INPUT_NAN = 300,
    SVPF_ERR_INPUT_INF = 301,
    SVPF_ERR_INPUT_EXTREME = 302,
    SVPF_ERR_PARAMS_INVALID = 310,
    
    // State Errors (400-499)
    SVPF_ERR_NOT_INITIALIZED = 400,
    SVPF_ERR_ALREADY_DESTROYED = 401,
    SVPF_ERR_STATE_CORRUPTED = 402,
    
    // Resource Errors (500-599)
    SVPF_ERR_OUT_OF_MEMORY = 500,
    SVPF_ERR_SHARED_MEM_EXCEEDED = 501,
    
    // Warnings (1000+)
    SVPF_WARN_ESS_LOW = 1000,
    SVPF_WARN_VOL_EXTREME = 1001,
    SVPF_WARN_PARTICLES_CLUSTERED = 1002,
    SVPF_WARN_GRAPH_RECAPTURE = 1003,
    
} SVPFStatus;

// =============================================================================
// Diagnostic State
// =============================================================================

/**
 * @brief Health metrics for monitoring
 */
typedef struct {
    // Current status
    SVPFStatus status;
    uint32_t error_count;
    uint32_t warning_count;
    
    // Last error details
    SVPFStatus last_error;
    uint64_t last_error_timestep;
    char last_error_message[256];
    
    // Numerical health (updated each step)
    float ess;                  // Effective sample size
    float ess_ratio;            // ESS / N (should be > 0.5)
    float particle_spread;      // Std of particles
    float weight_entropy;       // Entropy of weights (higher = healthier)
    float max_weight_ratio;     // Max weight / mean weight
    
    // Output sanity
    float vol_estimate;
    float h_mean;
    float bandwidth;
    float loglik;
    
    // Performance metrics
    uint64_t steps_processed;
    uint64_t graph_recaptures;
    float avg_step_time_us;
    float max_step_time_us;
    
    // Thresholds (configurable)
    float ess_warning_threshold;      // Default: 0.3
    float vol_max_threshold;          // Default: 10.0 (1000% daily vol)
    float vol_min_threshold;          // Default: 0.0001 (0.01% daily vol)
    float input_max_return;           // Default: 0.5 (50% move)
    
} SVPFDiagnostics;

// =============================================================================
// API Functions
// =============================================================================

/**
 * @brief Initialize diagnostics with default thresholds
 */
void svpf_diag_init(SVPFDiagnostics* diag);

/**
 * @brief Check filter health after a step
 * 
 * Call this after svpf_step_graph(). It will:
 *   - Check for CUDA errors
 *   - Validate numerical outputs
 *   - Update health metrics
 *   - Set status and error message if something is wrong
 * 
 * @param state  SVPF state (reads particle array, weights, etc.)
 * @param diag   Diagnostics struct to update
 * @return       Current status (also stored in diag->status)
 */
SVPFStatus svpf_diag_check(void* state, SVPFDiagnostics* diag);

/**
 * @brief Validate input before feeding to filter
 * 
 * @param y_t    Current return
 * @param diag   Diagnostics struct
 * @return       SVPF_OK if input is valid
 */
SVPFStatus svpf_diag_check_input(float y_t, SVPFDiagnostics* diag);

/**
 * @brief Get human-readable error message
 */
const char* svpf_diag_message(const SVPFDiagnostics* diag);

/**
 * @brief Get status code name as string
 */
const char* svpf_status_name(SVPFStatus status);

/**
 * @brief Check if status is an error (vs OK or warning)
 */
bool svpf_status_is_error(SVPFStatus status);

/**
 * @brief Check if status is a warning
 */
bool svpf_status_is_warning(SVPFStatus status);

/**
 * @brief Reset error state (after handling)
 */
void svpf_diag_clear_error(SVPFDiagnostics* diag);

/**
 * @brief Print diagnostic summary to stdout
 */
void svpf_diag_print_summary(const SVPFDiagnostics* diag);

/**
 * @brief Get JSON-formatted status (for logging/monitoring)
 * 
 * @param diag    Diagnostics struct
 * @param buffer  Output buffer
 * @param size    Buffer size
 * @return        Number of characters written
 */
int svpf_diag_to_json(const SVPFDiagnostics* diag, char* buffer, int size);

// =============================================================================
// CUDA Error Checking Macros
// =============================================================================

#ifdef SVPF_ENABLE_CUDA_CHECKS

#define SVPF_CUDA_CHECK(call, diag) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        (diag)->status = SVPF_ERR_CUDA_KERNEL; \
        snprintf((diag)->last_error_message, 256, \
                 "CUDA error at %s:%d: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        (diag)->last_error = (diag)->status; \
        (diag)->error_count++; \
    } \
} while(0)

#define SVPF_CUDA_CHECK_LAST(diag) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        (diag)->status = SVPF_ERR_CUDA_KERNEL; \
        snprintf((diag)->last_error_message, 256, \
                 "CUDA error at %s:%d: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        (diag)->last_error = (diag)->status; \
        (diag)->error_count++; \
    } \
} while(0)

#else

#define SVPF_CUDA_CHECK(call, diag) (call)
#define SVPF_CUDA_CHECK_LAST(diag) ((void)0)

#endif

// =============================================================================
// Quick Health Check (Inline)
// =============================================================================

/**
 * @brief Fast inline check for NaN/Inf
 */
static inline bool svpf_is_healthy_float(float x) {
    // NaN: x != x
    // Inf: |x| > FLT_MAX
    return (x == x) && (x > -1e30f) && (x < 1e30f);
}

#ifdef __cplusplus
}
#endif

#endif // SVPF_DIAGNOSTICS_H
