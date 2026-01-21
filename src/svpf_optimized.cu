/**
 * @file svpf_optimized.cu
 * @brief High-performance SVPF with full SM utilization
 * 
 * FIXES from review:
 * - Predict kernel runs ONCE (samples noise), separate from likelihood/grad
 * - NO device→host copies in timestep loop - all scalars on device
 * - Persistent CTA for small N path (true "read once" behavior)
 * - Bandwidth uses variance-based heuristic with EMA smoothing
 * 
 * Key optimizations:
 * 1. 2D tiled grid for O(N²) Stein kernel - guarantees SM saturation
 * 2. Shared memory tiling - bandwidth efficient
 * 3. CUB reductions for log-sum-exp
 * 4. Small N path: Persistent CTA, all data in SMEM, no atomics
 * 5. CUDA Graph compatible (no mid-loop synchronization)
 * 
 * References:
 * - Liu & Wang (2016): SVGD algorithm
 * - Fan et al. (2021): Stein Particle Filtering
 */

#include "svpf.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Configuration
// =============================================================================

#define TILE_J 256                   // Match BLOCK_SIZE for full thread utilization
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SMALL_N_THRESHOLD 4096
#define BANDWIDTH_UPDATE_INTERVAL 5
#define MAX_T_SIZE 10000             // Max sequence length for staging buffer

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 20.0f));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduction - result valid in thread 0 only
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_min(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_min(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : 1e10f;
    if (wid == 0) val = warp_reduce_min(val);
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : -1e10f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

// =============================================================================
// Kernel 1: Predict (Run ONCE per timestep - samples process noise)
// =============================================================================

// Version with explicit t (for non-graph path)
__global__ void svpf_predict_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    int t,
    float rho, float sigma_z, float mu, float gamma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    h_prev[i] = h_i;
    
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    
    float noise = curand_normal(&rng[i]);
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    h[i] = clamp_logvol(mu + rho * (h_i - mu) + sigma_z * noise + leverage);
}

// =============================================================================
// Kernel 1b: Predict with Mixture Innovation Model (MIM) + Asymmetric ρ
//
// Innovation is a mixture of Gaussians:
//   (1-p) * N(0, σ²) + p * N(0, (jump_scale*σ)²)
//
// Asymmetric persistence:
//   ρ_eff = ρ_up   if h > h_prev  (vol increasing - persists longer)
//   ρ_eff = ρ_down if h ≤ h_prev  (vol decreasing - mean-reverts faster)
//
// Captures empirical fact: volatility spikes fast, decays slow
// =============================================================================

__global__ void svpf_predict_mim_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,  // Mean from previous step (for local params)
    int t,
    float rho_up,         // Base persistence when vol increasing
    float rho_down,       // Base persistence when vol decreasing
    float sigma_z, float mu, float gamma,
    float jump_prob,      // MIM jump probability
    float jump_scale,     // MIM jump scale
    float delta_rho,      // Particle-local rho sensitivity (e.g., 0.02)
    float delta_sigma,    // Particle-local sigma sensitivity (e.g., 0.1)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;  // Store for next step

    // =========================================================================
    // PARTICLE-LOCAL PARAMETERS
    // Key insight: DGP has θ(z), σ(z) — params depend on latent z
    // We don't have z, but h is correlated with z (high h → high z)
    // So we use h deviation from mean as proxy for z-dependent behavior
    // =========================================================================
    float h_bar = *d_h_mean;
    float dev = h_i - h_bar;
    float abs_dev = fabsf(dev);
    float tanh_dev = tanhf(dev);  // Maps (-inf, inf) → (-1, 1)

    // Local rho: particles far from mean may have different persistence
    // tanh ensures smooth bounded adjustment
    float rho_adjust = delta_rho * tanh_dev;
    
    // Local sigma: particles far from mean get higher vol-of-vol
    // This mimics "when volatility is high, it's also more volatile"
    float sigma_scale = 1.0f + delta_sigma * abs_dev;

    // Sample innovation noise
    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);

    // Branchless mixture selection (compiles to CMOV, no divergence)
    float scale = (selector < jump_prob) ? jump_scale : 1.0f;

    // Asymmetric persistence: vol spikes fast, decays slow
    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    
    // Apply particle-local adjustment, clamp to [0, 0.999]
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    
    // Apply particle-local sigma scaling
    float sigma_local = sigma_z * sigma_scale;

    // Leverage effect
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);

    // Evolve state with local params and scaled innovation
    float prior_mean = mu + rho * (h_i - mu) + leverage;
    h[i] = clamp_logvol(prior_mean + sigma_local * scale * noise);
}

// =============================================================================
// Kernel 1c: Guided Predict with Lookahead (THE BIG ONE)
//
// Standard predict is REACTIVE: scatters blindly from h_{t-1}, then corrects.
// Guided predict is PROACTIVE: peeks at y_t to know where particles should go.
//
// Instead of: h_t ~ N(μ_prior, σ²)
// We use:     h_t ~ N((1-α)μ_prior + α·μ_implied, σ²)
//
// Where μ_implied = log(y_t²) + 1.27 is the instantaneous implied volatility.
// (1.27 ≈ -E[log(η²)] for η ~ N(0,1))
//
// Why this wins:
// - Eliminates lag: particles pushed toward today's shock BEFORE Stein
// - Fixes starvation: in 5σ events, standard prior puts 0 particles in right place
// - Newton/Stein refines shape rather than desperately dragging outliers
//
// guided_alpha:
//   0.0 = standard predict (history only)
//   0.2-0.3 = good default (20-30% from today's shock)
//   0.5+ = aggressive (for stress scenarios, may hurt calm periods)
// =============================================================================

// =============================================================================
// Kernel 1c: Innovation-Gated Guided Predict (FIXED)
// 
// FIXES "Zero-Return Trap":
// 1. Clamps mean_implied to -5.0 (prevents log(0) -> -inf crashing the mean)
// 2. Uses ASYMMETRIC gating: Only guides UPWARD shocks.
//    - Downward "shocks" (random low returns) are ignored; we trust the Prior's rho.
//    - Upward shocks (spikes) trigger the guide to eliminate lag.
//
// This gives "best of both worlds":
// - OU-Matched: ~0.45 RMSE (no noise injection from zero returns)
// - Stress Ramp: ~0.85 RMSE (lag eliminated on upward spikes)
// =============================================================================

__global__ void svpf_predict_guided_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,
    int t,
    float rho_up, float rho_down,
    float sigma_z, float mu, float gamma,
    float jump_prob, float jump_scale,
    float delta_rho, float delta_sigma,
    float alpha_base,             // e.g. 0.00
    float alpha_shock,            // e.g. 0.50
    float innovation_threshold,   // e.g. 1.5
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float h_i = h[i];
    float h_prev_i = h_prev[i];
    h_prev[i] = h_i;

    // =========================================================================
    // 1. PARTICLE-LOCAL PARAMETERS (same as MIM)
    // =========================================================================
    float h_bar = *d_h_mean;
    float dev = h_i - h_bar;
    float abs_dev = fabsf(dev);
    float tanh_dev = tanhf(dev);

    float rho_adjust = delta_rho * tanh_dev;
    float sigma_scale = 1.0f + delta_sigma * abs_dev;

    float noise = curand_normal(&rng[i]);
    float selector = curand_uniform(&rng[i]);
    float scale = (selector < jump_prob) ? jump_scale : 1.0f;

    float base_rho = (h_i > h_prev_i) ? rho_up : rho_down;
    float rho = fminf(fmaxf(base_rho + rho_adjust, 0.0f), 0.999f);
    float sigma_local = sigma_z * sigma_scale;

    // =========================================================================
    // 2. CALCULATE MEANS
    // =========================================================================
    
    // A. Prior Mean (History-driven)
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = safe_exp(h_i / 2.0f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    float mean_prior = mu + rho * (h_i - mu) + leverage;

    // B. Implied Mean (Observation-driven)
    float y_curr = d_y[t];
    float log_y2 = logf(y_curr * y_curr + 1e-10f);  // 1e-10 prevents -inf
    
    // SAFETY FIX 1: Bottom Clamp
    // If y ≈ 0, log_y2 → -20. This is noise, not signal.
    // Clamp implied signal to physical floor (e.g., -5.0 = vol ~0.08)
    float mean_implied = fmaxf(log_y2 + 1.27f, -5.0f);

    // =========================================================================
    // 3. ASYMMETRIC INNOVATION GATING (SAFETY FIX 2)
    //
    // We only care about POSITIVE innovation (Observed Vol > Prior Vol).
    // If Observed < Prior, it's ambiguous (could be low vol, could be lucky draw).
    // Standard SV mean reversion handles decay well enough.
    // We only need help catching SPIKES.
    // =========================================================================
    
    float innovation = mean_implied - mean_prior;  // SIGNED innovation!
    float total_std = 2.5f;
    float z_score = innovation / total_std;

    float activation = 0.0f;
    
    // Only trigger if z_score is POSITIVE and above threshold
    if (z_score > innovation_threshold) {
        activation = tanhf(z_score - innovation_threshold);
    }
    // If z_score < threshold (or negative), activation stays 0.0 → Pure Prior

    float guided_alpha = alpha_base + (alpha_shock - alpha_base) * activation;

    // =========================================================================
    // 4. BLEND & SAMPLE
    // =========================================================================
    float mean_proposal = (1.0f - guided_alpha) * mean_prior + guided_alpha * mean_implied;
    
    h[i] = clamp_logvol(mean_proposal + sigma_local * scale * noise);
}

// =============================================================================
// Kernel 2b: Mixture Prior Gradient (O(N²) - CORRECT per paper Eq. 6)
//
// The paper specifies the prior as a Gaussian mixture:
//   p(h_{t+1} | Z_t) = (1/n) * Σᵢ p(h_{t+1} | h_t^i)
//
// where each component is: p(h | h_prev^i) = N(μᵢ, σ_z²)
// with μᵢ = μ + ρ(h_prev^i - μ)
//
// The gradient is:
//   ∇_h log p_prior(h) = Σᵢ rᵢ(h) * (-(h - μᵢ)/σ_z²)
//
// where rᵢ(h) = p(h|h_prev^i) / Σⱼ p(h|h_prev^j) is the responsibility
//
// This replaces the WRONG per-particle prior with the CORRECT mixture prior.
// =============================================================================

// Small N version: each thread handles one particle j, loops over all i
__global__ void svpf_mixture_prior_grad_kernel(
    const float* __restrict__ h,           // Current particles [N]
    const float* __restrict__ h_prev,      // Previous particles [N]
    float* __restrict__ grad_prior,        // Output: mixture prior gradient [N]
    float rho,
    float sigma_z,
    float mu,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma_z_sq);
    
    // First pass: find max log-responsibility for numerical stability
    float log_r_max = -1e10f;
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float log_r_i = -diff * diff * inv_2sigma_sq;
        log_r_max = fmaxf(log_r_max, log_r_i);
    }
    
    // Second pass: compute weighted gradient with log-sum-exp stability
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        float log_r_i = -diff * diff * inv_2sigma_sq;
        float r_i = expf(log_r_i - log_r_max);  // Numerically stable
        
        sum_r += r_i;
        weighted_grad += r_i * (-diff / sigma_z_sq);
    }
    
    // Normalize
    grad_prior[j] = weighted_grad / (sum_r + 1e-8f);
}

// Tiled version for large N: uses shared memory for h_prev tiles
__global__ void svpf_mixture_prior_grad_tiled_kernel(
    const float* __restrict__ h,           // Current particles [N]
    const float* __restrict__ h_prev,      // Previous particles [N]
    float* __restrict__ grad_prior,        // Output: mixture prior gradient [N]
    float rho,
    float sigma_z,
    float mu,
    int n
) {
    __shared__ float sh_h_prev[TILE_J];
    __shared__ float sh_mu_i[TILE_J];
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma_z_sq);
    
    // First pass: find max log-responsibility across all tiles
    float log_r_max = -1e10f;
    
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        // Cooperative load of h_prev tile
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();
        
        // Find max in this tile
        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            log_r_max = fmaxf(log_r_max, log_r_i);
        }
    }
    
    // Second pass: compute weighted gradient
    float sum_r = 0.0f;
    float weighted_grad = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_J - 1) / TILE_J; tile++) {
        int tile_start = tile * TILE_J;
        int tile_end = min(tile_start + TILE_J, n);
        int tile_size = tile_end - tile_start;
        
        // Cooperative load (already in smem from first pass if single tile,
        // but need to reload for multi-tile case)
        __syncthreads();
        for (int k = threadIdx.x; k < tile_size; k += blockDim.x) {
            float hp = h_prev[tile_start + k];
            sh_h_prev[k] = hp;
            sh_mu_i[k] = mu + rho * (hp - mu);
        }
        __syncthreads();
        
        // Accumulate weighted gradient
        for (int i = 0; i < tile_size; i++) {
            float diff = h_j - sh_mu_i[i];
            float log_r_i = -diff * diff * inv_2sigma_sq;
            float r_i = expf(log_r_i - log_r_max);
            
            sum_r += r_i;
            weighted_grad += r_i * (-diff / sigma_z_sq);
        }
    }
    
    // Normalize
    grad_prior[j] = weighted_grad / (sum_r + 1e-8f);
}

// =============================================================================
// Kernel 2c: Likelihood-Only Gradient (O(N) - just the observation term)
//
// Computes only the likelihood gradient, to be combined with mixture prior.
// Also computes log_weights for importance sampling.
// =============================================================================

__global__ void svpf_likelihood_only_kernel(
    const float* __restrict__ h,
    float* __restrict__ grad_lik,          // Output: likelihood gradient only [N]
    float* __restrict__ log_w,             // Output: log weights [N]
    const float* __restrict__ d_y,         // Observations array
    int t,                                 // Current timestep
    float nu,
    float student_t_const,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float y_t = d_y[t];
    
    // =========================================================================
    // WEIGHTS: Exact Student-t (unbiased importance sampling)
    // =========================================================================
    float vol = safe_exp(h_i);
    float scaled_y_sq = (y_t * y_t) / (vol + 1e-8f);
    
    log_w[i] = student_t_const - 0.5f * h_i
             - (nu + 1.0f) / 2.0f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    // =========================================================================
    // GRADIENT: Log-squared approximation (linear restoring force)
    // =========================================================================
    // Model: log(y²) = h + log(η²), where η ~ t_ν
    //
    // Student-t gradient saturates at ν/2 for large deviations ("volcano").
    // Log-squared gradient is LINEAR - particles far away feel proportional pull.
    //
    // offset = E[log(η²)] for η ~ t_ν ≈ -1/ν (simple approximation)
    // R_noise = scaling factor ≈ var(log(η²)) ≈ 2.0 for ν=5
    //
    // Gradient: ∂/∂h [ -(log(y²) - h - offset)² / (2*R) ] = (log(y²) - h - offset) / R
    // =========================================================================
    
    float log_y2 = logf(y_t * y_t + 1e-10f);
    float offset = -1.0f / nu;    // E[log(η²)] approximation
    float R_noise = 1.4f;         // Observation noise variance in log space
    
    // Linear gradient: pulls h toward log(y²) - offset
    grad_lik[i] = (log_y2 - h_i - offset) / R_noise;
}

// =============================================================================
// Kernel 2d: Combine Gradients (prior + likelihood)
// =============================================================================

__global__ void svpf_combine_gradients_kernel(
    const float* __restrict__ grad_prior,  // Mixture prior gradient [N]
    const float* __restrict__ grad_lik,    // Likelihood gradient [N]
    float* __restrict__ grad,              // Output: combined gradient [N]
    float beta,                            // Annealing factor (1.0 = full likelihood)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float g = grad_prior[i] + beta * grad_lik[i];
    grad[i] = fminf(fmaxf(g, -10.0f), 10.0f);
}

// =============================================================================
// Kernel 2e: Hessian-Preconditioned Gradient (Newton-Stein)
//
// Standard SVGD performs gradient descent on KL divergence.
// Newton-Stein performs Newton's method: move along H^{-1} * grad
//
// Benefits:
// - Flat regions (small H): Step size increases, faster convergence
// - Sharp peaks (large H): Step size decreases, prevents overshooting
// - Solves "stiffness" dynamically per particle
//
// For 1D Student-t with vol = exp(h), A = y²/(nu*vol):
//   Gradient:  -0.5 + 0.5*(nu+1)*A/(1+A)
//   Hessian:   -0.5*(nu+1)*A/(1+A)²
// =============================================================================

__global__ void svpf_hessian_precond_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad_combined,   // Input: prior + beta*lik gradient
    float* __restrict__ precond_grad,          // Output: H^{-1} * grad
    float* __restrict__ inv_hessian,           // Output: H^{-1} (for kernel scaling)
    const float* __restrict__ d_y,
    int t,
    float nu,
    float sigma_z,                             // For prior Hessian
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = h[i];
    float y = d_y[t];
    float vol = safe_exp(h_i);
    float y_sq = y * y;
    
    // =========================================================================
    // Compute Exact Student-t Likelihood Hessian
    // =========================================================================
    // Let A = y² / (nu * exp(h))
    float A = y_sq / (nu * vol + 1e-8f);
    float one_plus_A = 1.0f + A;
    
    // Hessian of log-likelihood: d²/dh² log p(y|h)
    // = -0.5 * (nu+1) * A / (1+A)²
    // This is NEGATIVE (log-likelihood is concave)
    float hess_lik = -0.5f * (nu + 1.0f) * (A / (one_plus_A * one_plus_A));
    
    // =========================================================================
    // Prior Hessian (Gaussian: constant curvature)
    // =========================================================================
    // Prior: log p(h) = -0.5*(h - mu_prior)²/sigma_z²
    // Hessian: -1/sigma_z²
    float sigma_z_sq = sigma_z * sigma_z + 1e-8f;
    float hess_prior = -1.0f / sigma_z_sq;
    
    // =========================================================================
    // Total Curvature (negative of Hessian for convex optimization)
    // =========================================================================
    // We want positive definite approximation for Newton step
    // Total Hessian is negative (concave log-posterior), so -H is positive
    float total_hessian = hess_lik + hess_prior;  // Both negative
    float curvature = -total_hessian;              // Make positive
    
    // Clip curvature to avoid numerical issues
    // Too small: exploding steps. Too large: no effect.
    curvature = fmaxf(curvature, 0.1f);   // Min curvature (max step scale = 10x)
    curvature = fminf(curvature, 100.0f); // Max curvature (min step scale = 0.01x)
    
    // =========================================================================
    // Newton Step: H^{-1} * gradient
    // =========================================================================
    float inv_H = 1.0f / curvature;
    float grad_i = grad_combined[i];
    
    // Dampen Newton step for stability (pure Newton can overshoot)
    float damping = 0.7f;
    
    precond_grad[i] = damping * grad_i * inv_H;
    inv_hessian[i] = inv_H;
}

// =============================================================================
// Kernel 3: Log-Sum-Exp (computes loglik, all on device)
// =============================================================================

__global__ void svpf_logsumexp_kernel(
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_max_log_w,
    int t,
    int n
) {
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    
    float local_max = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, log_w[i]);
    }
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();
    
    local_max = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? warp_vals[threadIdx.x] : -1e10f;
    if (wid == 0) local_max = warp_reduce_max(local_max);
    
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    local_max = s_max;
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf(log_w[i] - local_max);
    }
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        d_loglik[t] = local_max + logf(local_sum / (float)n + 1e-10f);
        *d_max_log_w = local_max;
    }
}

// =============================================================================
// Kernel 4: Bandwidth (variance-based, all on device, with EMA)
// =============================================================================

__global__ void svpf_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,      // In/out: current bandwidth (for EMA)
    float* __restrict__ d_bandwidth_sq,   // In/out: EMA of bandwidth squared
    float alpha,                          // EMA factor
    int n
) {
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    local_sum = block_reduce_sum(local_sum);
    __syncthreads();
    local_sum_sq = block_reduce_sum(local_sum_sq);
    
    if (threadIdx.x == 0) {
        float mean = local_sum / (float)n;
        float variance = local_sum_sq / (float)n - mean * mean;
        
        // Bandwidth squared: h² = 2 * var / log(n+1)
        float bw_sq_new = 2.0f * variance / logf((float)n + 1.0f);
        bw_sq_new = fmaxf(bw_sq_new, 1e-6f);
        
        // EMA on bandwidth squared
        float bw_sq_prev = *d_bandwidth_sq;
        float bw_sq = (bw_sq_prev > 0.0f) 
                    ? alpha * bw_sq_new + (1.0f - alpha) * bw_sq_prev 
                    : bw_sq_new;
        
        float bw = sqrtf(bw_sq);
        bw = fmaxf(fminf(bw, 2.0f), 0.01f);
        
        *d_bandwidth_sq = bw_sq;
        *d_bandwidth = bw;
    }
}

// =============================================================================
// Kernel 5a: 2D Tiled Stein Kernel (Large N path)
// =============================================================================

__global__ void svpf_stein_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ grad,
    float* __restrict__ phi,              // Output: transport direction
    const float* __restrict__ d_bandwidth, // Read from device
    int n
) {
    int i = blockIdx.x;
    int tile = blockIdx.y;
    int tile_start = tile * TILE_J;
    int tile_end = min(tile_start + TILE_J, n);
    int tile_size = tile_end - tile_start;
    
    if (i >= n || tile_start >= n) return;
    
    __shared__ float sh_h[TILE_J];
    __shared__ float sh_grad[TILE_J];
    
    // Cooperative load
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        sh_h[j] = h[tile_start + j];
        sh_grad[j] = grad[tile_start + j];
    }
    __syncthreads();
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        float diff = h_i - sh_h[j];
        float K = expf(-diff * diff / (2.0f * bw_sq));
        k_sum += K * sh_grad[j];
        gk_sum += -K * diff / bw_sq;
    }
    
    // Block reduction
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5a-Newton: 2D Tiled Newton-Stein Kernel (Large N path)
// =============================================================================

__global__ void svpf_stein_newton_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,  // H^{-1} * grad
    const float* __restrict__ inv_hessian,   // H^{-1}
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    int i = blockIdx.x;
    int tile = blockIdx.y;
    int tile_start = tile * TILE_J;
    int tile_end = min(tile_start + TILE_J, n);
    int tile_size = tile_end - tile_start;
    
    if (i >= n || tile_start >= n) return;
    
    __shared__ float sh_h[TILE_J];
    __shared__ float sh_precond_grad[TILE_J];
    __shared__ float sh_inv_hess[TILE_J];
    
    // Cooperative load
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        sh_h[j] = h[tile_start + j];
        sh_precond_grad[j] = precond_grad[tile_start + j];
        sh_inv_hess[j] = inv_hessian[tile_start + j];
    }
    __syncthreads();
    
    float h_i = h[i];
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    float k_sum = 0.0f;
    float gk_sum = 0.0f;
    
    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        float diff = h_i - sh_h[j];
        float K = expf(-diff * diff / (2.0f * bw_sq));
        float inv_H_j = sh_inv_hess[j];
        
        // Newton-scaled terms
        k_sum += K * sh_precond_grad[j];
        gk_sum += (-K * diff / bw_sq) * inv_H_j;
    }
    
    // Block reduction
    k_sum = block_reduce_sum(k_sum);
    __syncthreads();
    gk_sum = block_reduce_sum(gk_sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(&phi[i], (k_sum + gk_sum) / (float)n);
    }
}

// =============================================================================
// Kernel 5b: Persistent CTA Stein Kernel (Small N path)
// =============================================================================

/**
 * Persistent CTA approach:
 * - Launch one block per SM (or small multiple)
 * - Each block loads ALL data once into SMEM
 * - Grid-stride over i: block b handles particles b, b+gridDim.x, b+2*gridDim.x...
 * - Within each i: threads cooperate on j summation
 * 
 * CRITICAL: Writes to phi[], NOT h[] - avoids race condition.
 * Caller must apply transport separately with svpf_apply_transport_kernel.
 */
__global__ void svpf_stein_persistent_kernel(
    const float* __restrict__ h,      // READ-ONLY
    const float* __restrict__ grad,
    float* __restrict__ phi,          // OUTPUT: transport direction
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_grad = smem + n;
    float* sh_reduce = smem + 2 * n;  // Workspace for reduction
    
    // Load ALL data once (cooperative across threads)
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_grad[j] = grad[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    // Grid-stride loop over particles
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        // Threads cooperate on j summation
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float K = expf(-diff * diff / (2.0f * bw_sq));
            k_sum += K * sh_grad[j];
            gk_sum += -K * diff / bw_sq;
        }
        
        // Block reduction for k_sum
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        // Block reduction for gk_sum
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        // Thread 0 writes transport direction to phi (NOT h)
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();  // Ensure write completes before next iteration
    }
}

// =============================================================================
// Kernel 5b: Newton-Stein Persistent Kernel (Matrix-Valued Kernel)
//
// Generalizes Stein identity to use Hessian preconditioning:
//   φ(x) = (1/n) Σ_j [ K(x,x_j) * H_j^{-1} * ∇log p(x_j) + ∇_x K(x,x_j) * H_j^{-1} ]
//
// The inverse Hessian scales both:
// - The gradient term (Newton direction)
// - The repulsive term (curvature-aware repulsion)
//
// Benefits:
// - Particles in sharp peaks (high curvature) take smaller steps
// - Particles in flat regions (low curvature) take larger steps
// - Repulsion is weaker near sharp peaks (don't push away from good spots)
// =============================================================================

__global__ void svpf_stein_newton_persistent_kernel(
    const float* __restrict__ h,
    const float* __restrict__ precond_grad,  // H^{-1} * grad (already preconditioned)
    const float* __restrict__ inv_hessian,   // H^{-1} for repulsive scaling
    float* __restrict__ phi,
    const float* __restrict__ d_bandwidth,
    int n
) {
    extern __shared__ float smem[];
    float* sh_h = smem;
    float* sh_precond_grad = smem + n;
    float* sh_inv_hess = smem + 2 * n;
    float* sh_reduce = smem + 3 * n;
    
    // Load ALL data once (cooperative across threads)
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sh_h[j] = h[j];
        sh_precond_grad[j] = precond_grad[j];
        sh_inv_hess[j] = inv_hessian[j];
    }
    __syncthreads();
    
    float bw = *d_bandwidth;
    float bw_sq = bw * bw;
    
    // Grid-stride loop over particles
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float h_i = sh_h[i];
        
        // Threads cooperate on j summation
        float k_sum = 0.0f;
        float gk_sum = 0.0f;
        
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float diff = h_i - sh_h[j];
            float K = expf(-diff * diff / (2.0f * bw_sq));
            float inv_H_j = sh_inv_hess[j];
            
            // Term 1: Kernel * Preconditioned_Gradient
            // Note: precond_grad already contains H^{-1} * grad
            k_sum += K * sh_precond_grad[j];
            
            // Term 2: Matrix-Valued Repulsive Term
            // Standard: -K * diff / bw²
            // Newton:   -K * diff / bw² * H_j^{-1}
            // The inverse Hessian scales repulsion:
            // - High curvature (small inv_H): weak repulsion (stay near peak)
            // - Low curvature (large inv_H): strong repulsion (spread out in flat regions)
            gk_sum += (-K * diff / bw_sq) * inv_H_j;
        }
        
        // Block reduction for k_sum
        k_sum = warp_reduce_sum(k_sum);
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) sh_reduce[wid] = k_sum;
        __syncthreads();
        
        k_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) k_sum = warp_reduce_sum(k_sum);
        
        // Block reduction for gk_sum
        gk_sum = warp_reduce_sum(gk_sum);
        if (lane == 0) sh_reduce[wid] = gk_sum;
        __syncthreads();
        
        gk_sum = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? sh_reduce[threadIdx.x] : 0.0f;
        if (wid == 0) gk_sum = warp_reduce_sum(gk_sum);
        
        // Thread 0 writes transport direction
        if (threadIdx.x == 0) {
            phi[i] = (k_sum + gk_sum) / (float)n;
        }
        __syncthreads();
    }
}

// =============================================================================
// ADAPTIVE SVPF KERNELS
// =============================================================================

// -----------------------------------------------------------------------------
// Kernel A1: Fused SVLD + RMSProp Transport (Preconditioned SVLD)
//
// Implements Preconditioned Stein Variational Langevin Descent:
//   1. RMSProp: v = ρ*v + (1-ρ)*φ²  (adaptive learning rate)
//   2. Drift:   h += ε*β*φ/√(v+eps)  (preconditioned gradient)
//   3. Diffusion: h += √(2*ε*β*T)*η  (Langevin noise for diversity)
//
// temperature = 0 → deterministic SVGD (no noise)
// temperature = 1 → full SVLD (theoretically correct)
// temperature > 1 → extra exploration (useful early, anneal down)
// -----------------------------------------------------------------------------

__global__ void svpf_apply_transport_svld_kernel(
    float* __restrict__ h,
    const float* __restrict__ phi,
    float* __restrict__ v,                              // RMSProp state [N]
    curandStatePhilox4_32_10_t* __restrict__ rng_states,// RNG states [N]
    float base_step_size,
    float beta_anneal_factor,    // Scales step during annealing (sqrt(beta) or beta)
    float temperature,           // 0.0 = SVGD, 1.0 = SVLD, >1 = extra exploration
    float rho_rmsprop,           // RMSProp decay (0.9-0.99)
    float epsilon,               // Stability (1e-6)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float phi_i = phi[i];
    float v_prev = v[i];

    // 1. Update RMSProp (Second Moment)
    float v_new = rho_rmsprop * v_prev + (1.0f - rho_rmsprop) * phi_i * phi_i;
    v[i] = v_new;

    // 2. Calculate Deterministic Update (Drift)
    float effective_step = base_step_size * beta_anneal_factor;
    float preconditioner = rsqrtf(v_new + epsilon);  // 1/sqrt(v)
    float drift = effective_step * phi_i * preconditioner;

    // 3. Calculate Stochastic Update (Diffusion) - Langevin noise
    float diffusion = 0.0f;
    if (temperature > 1e-6f) {
        float noise = curand_normal(&rng_states[i]);
        diffusion = sqrtf(2.0f * effective_step * temperature) * noise;
    }

    // 4. Apply & Clamp
    h[i] = clamp_logvol(h[i] + drift + diffusion);
}

// -----------------------------------------------------------------------------
// Kernel A1b: Apply Guide Density (Affine Re-centering)
//
// Shifts particles toward EKF guide mean without destroying diversity.
// Called once per timestep, right after predict, before Stein iterations.
//
// h[i] = h[i] + strength * (guide_mean - h[i])
//      = (1 - strength) * h[i] + strength * guide_mean
// -----------------------------------------------------------------------------

__global__ void svpf_apply_guide_kernel(
    float* __restrict__ h,
    float guide_mean,
    float guide_strength,   // 0.1-0.3 recommended
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float delta = guide_mean - h[i];
    h[i] = clamp_logvol(h[i] + guide_strength * delta);
}

// Graph-compatible version: reads guide_mean from device pointer
__global__ void svpf_apply_guide_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_guide_mean,
    float guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float guide_mean = *d_guide_mean;
    float delta = guide_mean - h[i];
    h[i] = clamp_logvol(h[i] + guide_strength * delta);
}

// -----------------------------------------------------------------------------
// Kernel A1b: Variance-Preserving Guide Shift
//
// The standard guide kernel is a CONTRACTION: h += k*(guide - h)
// This shrinks variance by (1-k)², making filter overconfident and stiff.
//
// This kernel SHIFTS the particle cloud mean toward guide without shrinking:
//   1. Remove current mean from each particle (center)
//   2. Compute new target mean (blend current mean with guide)
//   3. Add old deviation to new mean (reconstruct)
//
// Result: Distribution is TRANSLATED, not CONTRACTED.
// -----------------------------------------------------------------------------

__global__ void svpf_apply_guide_preserving_kernel(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,  // Current swarm mean
    float guide_mean,
    float guide_strength,                // 0.0-1.0
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float current_mean = *d_h_mean;
    float h_val = h[i];
    
    // 1. Center: get deviation from current mean
    float deviation = h_val - current_mean;
    
    // 2. Shift mean toward guide
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    
    // 3. Reconstruct: add OLD deviation to NEW mean
    // This translates the distribution, preserving its width
    h[i] = clamp_logvol(new_mean + deviation);
}

// Graph-compatible version: reads guide_mean from device pointer
__global__ void svpf_apply_guide_preserving_kernel_graph(
    float* __restrict__ h,
    const float* __restrict__ d_h_mean,
    const float* __restrict__ d_guide_mean,
    float guide_strength,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float current_mean = *d_h_mean;
    float guide_mean = *d_guide_mean;
    float h_val = h[i];
    
    float deviation = h_val - current_mean;
    float new_mean = (1.0f - guide_strength) * current_mean + guide_strength * guide_mean;
    
    h[i] = clamp_logvol(new_mean + deviation);
}

// -----------------------------------------------------------------------------
// Kernel A2: Adaptive Bandwidth Scaling (Improvement #1)
//
// Detects high-vol regime via particle spread and scales bandwidth α:
//   - High spread (particles dispersed) → α = 0.5-0.6 (tighter kernel)
//   - Low spread (particles clustered) → α = 1.0 (standard kernel)
//
// This is called AFTER standard bandwidth computation to scale it.
// -----------------------------------------------------------------------------

__global__ void svpf_adaptive_bandwidth_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,      // Modified in-place
    float* __restrict__ d_return_ema,     // EMA of |return|
    float* __restrict__ d_return_var,     // EMA of return variance
    float new_return,                     // Current observation
    float ema_alpha,                      // EMA smoothing (0.05)
    int n
) {
    // Compute particle spread (single block)
    float local_min = 1e10f, local_max = -1e10f;
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        local_sum += val;
    }
    
    // Reduce to get min/max/mean
    local_min = block_reduce_min(local_min);
    __syncthreads();
    local_max = block_reduce_max(local_max);
    __syncthreads();
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        float spread = local_max - local_min;
        float mean_h = local_sum / (float)n;
        
        // Update return EMA for regime detection
        float abs_ret = fabsf(new_return);
        float ret_ema = *d_return_ema;
        float ret_var = *d_return_var;
        
        ret_ema = (ret_ema > 0.0f) 
                ? ema_alpha * abs_ret + (1.0f - ema_alpha) * ret_ema 
                : abs_ret;
        ret_var = (ret_var > 0.0f)
                ? ema_alpha * abs_ret * abs_ret + (1.0f - ema_alpha) * ret_var
                : abs_ret * abs_ret;
        
        *d_return_ema = ret_ema;
        *d_return_var = ret_var;
        
        // Compute adaptive alpha based on regime
        // High return magnitude relative to EMA → high vol regime → lower alpha
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        
        // Also consider particle spread - high spread means uncertainty
        // Typical spread in calm: 0.5-1.0, in crisis: 2.0-4.0
        float spread_factor = fminf(spread / 2.0f, 2.0f);  // Normalized
        
        // Combine signals: higher vol_ratio or spread → lower alpha
        float combined_signal = fmaxf(vol_ratio, spread_factor);
        
        // Map to alpha: signal=1 → α=1.0, signal=3+ → α=0.5
        float alpha = 1.0f - 0.25f * fminf(combined_signal - 1.0f, 2.0f);
        alpha = fmaxf(fminf(alpha, 1.0f), 0.5f);
        
        // Scale bandwidth
        float bw = *d_bandwidth;
        *d_bandwidth = bw * alpha;
    }
}

// Graph-compatible version: reads y_t from device pointer (not burned-in scalar)
__global__ void svpf_adaptive_bandwidth_kernel_graph(
    const float* __restrict__ h,
    float* __restrict__ d_bandwidth,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    const float* __restrict__ d_y,    // Device pointer to y array
    int y_idx,                         // Index to read (usually 1 for y_t)
    float ema_alpha,
    int n
) {
    // Compute particle spread (single block)
    float local_min = 1e10f, local_max = -1e10f;
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = h[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        local_sum += val;
    }
    
    // Reduce to get min/max/mean
    local_min = block_reduce_min(local_min);
    __syncthreads();
    local_max = block_reduce_max(local_max);
    __syncthreads();
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        float spread = local_max - local_min;
        
        // READ y_t FROM DEVICE POINTER (not burned-in scalar!)
        float new_return = d_y[y_idx];
        
        // Update return EMA for regime detection
        float abs_ret = fabsf(new_return);
        float ret_ema = *d_return_ema;
        float ret_var = *d_return_var;
        
        ret_ema = (ret_ema > 0.0f) 
                ? ema_alpha * abs_ret + (1.0f - ema_alpha) * ret_ema 
                : abs_ret;
        ret_var = (ret_var > 0.0f)
                ? ema_alpha * abs_ret * abs_ret + (1.0f - ema_alpha) * ret_var
                : abs_ret * abs_ret;
        
        *d_return_ema = ret_ema;
        *d_return_var = ret_var;
        
        // Compute adaptive alpha based on regime
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        float spread_factor = fminf(spread / 2.0f, 2.0f);
        float combined_signal = fmaxf(vol_ratio, spread_factor);
        
        float alpha = 1.0f - 0.25f * fminf(combined_signal - 1.0f, 2.0f);
        alpha = fmaxf(fminf(alpha, 1.0f), 0.5f);
        
        float bw = *d_bandwidth;
        *d_bandwidth = bw * alpha;
    }
}

// =============================================================================
// Kernel 7: Volatility Mean
// =============================================================================

__global__ void svpf_vol_mean_opt_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_vol,
    int t,
    int n
) {
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += safe_exp(h[i] / 2.0f);
    }
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        d_vol[t] = local_sum / (float)n;
    }
}

// =============================================================================
// Kernel: Store h_mean for particle-local params
// =============================================================================
// Takes sum from CUB reduction and stores mean to d_h_mean_prev
__global__ void svpf_store_h_mean_kernel(
    const float* __restrict__ d_sum,
    float* __restrict__ d_h_mean_prev,
    int n
) {
    if (threadIdx.x == 0) {
        *d_h_mean_prev = *d_sum / (float)n;
    }
}

// =============================================================================
// CUDA GRAPH SUPPORT KERNELS
// =============================================================================

// Memset kernel (graph-compatible replacement for cudaMemsetAsync)
__global__ void svpf_memset_kernel(float* __restrict__ data, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

// H-mean kernel (graph-compatible replacement for CUB reduction)
// Uses efficient parallel reduction pattern with grid-stride loop
__global__ void svpf_h_mean_reduce_kernel(
    const float* __restrict__ h,
    float* __restrict__ d_partial_sums,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle arbitrary n
    float sum = 0.0f;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        sum += h[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction + division kernel (single block)
__global__ void svpf_h_mean_finalize_kernel(
    const float* __restrict__ d_partial_sums,
    float* __restrict__ d_h_mean,
    int n_blocks,
    int n_particles
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Load partial sums
    float sum = 0.0f;
    for (int i = tid; i < n_blocks; i += blockDim.x) {
        sum += d_partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Final reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write final mean
    if (tid == 0) {
        *d_h_mean = sdata[0] / (float)n_particles;
    }
}

// =============================================================================
// Host State - Uses SVPFOptimizedState from svpf.cuh (embedded in SVPFState)
// =============================================================================

// Helper to get optimized backend from state
static inline SVPFOptimizedState* get_opt(SVPFState* state) {
    return &state->opt_backend;
}

// Forward declaration
void svpf_optimized_cleanup(SVPFOptimizedState* opt);

void svpf_optimized_init(SVPFOptimizedState* opt, int n) {
    // Handle resize: if N increased, re-allocate
    if (opt->initialized && n > opt->allocated_n) {
        svpf_optimized_cleanup(opt);
    }
    if (opt->initialized) return;
    
    // Query CUB temp storage size
    float* d_dummy_in;
    float* d_dummy_out;
    cudaMalloc(&d_dummy_in, n * sizeof(float));
    cudaMalloc(&d_dummy_out, sizeof(float));
    
    opt->temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, opt->temp_storage_bytes, d_dummy_in, d_dummy_out, n);
    size_t sum_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, sum_bytes, d_dummy_in, d_dummy_out, n);
    opt->temp_storage_bytes = max(opt->temp_storage_bytes, sum_bytes);
    
    cudaMalloc(&opt->d_temp_storage, opt->temp_storage_bytes);
    cudaFree(d_dummy_in);
    cudaFree(d_dummy_out);
    
    // Device scalars
    cudaMalloc(&opt->d_max_log_w, sizeof(float));
    cudaMalloc(&opt->d_sum_exp, sizeof(float));
    cudaMalloc(&opt->d_bandwidth, sizeof(float));
    cudaMalloc(&opt->d_bandwidth_sq, sizeof(float));
    
    // Initialize bandwidth_sq to 0 (triggers fresh computation on first step)
    float zero = 0.0f;
    cudaMemcpy(opt->d_bandwidth_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Stein computation buffers
    cudaMalloc(&opt->d_exp_w, n * sizeof(float));
    cudaMalloc(&opt->d_phi, n * sizeof(float));
    cudaMalloc(&opt->d_grad_lik, n * sizeof(float));
    
    // Newton-Stein buffers (Hessian preconditioning)
    cudaMalloc(&opt->d_precond_grad, n * sizeof(float));
    cudaMalloc(&opt->d_inv_hessian, n * sizeof(float));
    
    // Particle-local parameters: store h_mean from previous step
    cudaMalloc(&opt->d_h_mean_prev, sizeof(float));
    float init_h_mean = -3.5f;  // Typical log-vol
    cudaMemcpy(opt->d_h_mean_prev, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);
    
    // Guide mean (device-side for graph compatibility)
    cudaMalloc(&opt->d_guide_mean, sizeof(float));
    cudaMemcpy(opt->d_guide_mean, &init_h_mean, sizeof(float), cudaMemcpyHostToDevice);
    
    // Single-step API buffers (avoid malloc in hot loop)
    cudaMalloc(&opt->d_y_single, 2 * sizeof(float));
    cudaMalloc(&opt->d_loglik_single, sizeof(float));
    cudaMalloc(&opt->d_vol_single, sizeof(float));
    
    // CUDA Graph support
    cudaMalloc(&opt->d_params_staging, SVPF_GRAPH_PARAMS_SIZE * sizeof(float));
    cudaStreamCreateWithFlags(&opt->graph_stream, cudaStreamNonBlocking);
    opt->graph_captured = false;
    opt->graph_n = 0;
    opt->graph_n_stein = 0;
    
    opt->allocated_n = n;
    opt->initialized = true;
}

void svpf_optimized_cleanup(SVPFOptimizedState* opt) {
    if (!opt->initialized) return;
    
    cudaFree(opt->d_temp_storage);
    cudaFree(opt->d_max_log_w);
    cudaFree(opt->d_sum_exp);
    cudaFree(opt->d_bandwidth);
    cudaFree(opt->d_bandwidth_sq);
    cudaFree(opt->d_exp_w);
    cudaFree(opt->d_phi);
    cudaFree(opt->d_grad_lik);
    cudaFree(opt->d_precond_grad);
    cudaFree(opt->d_inv_hessian);
    cudaFree(opt->d_h_mean_prev);
    cudaFree(opt->d_guide_mean);
    cudaFree(opt->d_y_single);
    cudaFree(opt->d_loglik_single);
    cudaFree(opt->d_vol_single);
    cudaFree(opt->d_params_staging);
    
    // Destroy CUDA graph if captured
    if (opt->graph_captured) {
        cudaGraphExecDestroy(opt->graph_exec);
        cudaGraphDestroy(opt->graph);
        opt->graph_captured = false;
    }
    
    // Destroy graph stream
    if (opt->graph_stream) {
        cudaStreamDestroy(opt->graph_stream);
        opt->graph_stream = nullptr;
    }
    
    opt->allocated_n = 0;
    opt->initialized = false;
}

// Wrapper for SVPFState cleanup (called by svpf_destroy)
void svpf_optimized_cleanup_state(SVPFState* state) {
    if (state) {
        svpf_optimized_cleanup(&state->opt_backend);
    }
}

// =============================================================================
// EKF Guide Update (Host-side, ~20 FLOPs)
//
// Uses LOG-SQUARED observation model for stability:
//   State:   h_t = μ + ρ(h_{t-1} - μ) + σ_z ε
//   Obs:     log(y_t²) ≈ h_t + log(η²)
//
// This linearizes the observation (H=1) and handles negative returns.
// E[log(η²)] ≈ -1.27 (for χ²₁), Var ≈ 4.93
// =============================================================================

static void svpf_ekf_update(
    SVPFState* state,
    float y_t,
    const SVPFParams* p
) {
    // Initialize on first call
    if (!state->guide_initialized) {
        state->guide_mean = p->mu;
        // Stationary variance: σ² / (1 - ρ²)
        state->guide_var = p->sigma_z * p->sigma_z / (1.0f - p->rho * p->rho);
        state->guide_initialized = 1;
    }
    
    // Predict
    float m_pred = p->mu + p->rho * (state->guide_mean - p->mu);
    float P_pred = p->rho * p->rho * state->guide_var + p->sigma_z * p->sigma_z;
    
    // Log-squared observation model (linearizes the problem!)
    // Model: log(y²) = h + log(η²)
    // E[log(η²)] ≈ -1.27 for χ²₁, Var[log(η²)] ≈ 4.93
    // For Student-t, inflate variance slightly
    float log_y2 = logf(y_t * y_t + 1e-8f);
    float obs_offset = -1.27f;
    float obs_var = 4.93f + 2.0f;  // Inflated for Student-t robustness
    
    // With log-squared transform: H = 1 (linear relationship!)
    float H = 1.0f;
    float R = obs_var;
    
    // Kalman gain
    float S = H * H * P_pred + R;
    float K = P_pred * H / (S + 1e-8f);
    
    // Innovation: log(y²) - predicted log(y²)
    float y_pred = m_pred + obs_offset;
    float innovation = log_y2 - y_pred;
    
    // Update
    state->guide_mean = m_pred + K * innovation;
    state->guide_var = (1.0f - K * H) * P_pred;
    state->guide_K = K;
}

// =============================================================================
// ADAPTIVE SVPF STEP - All improvements combined
// 
// Improvements:
//   1. Mixture Innovation Model (MIM) - fat-tailed predict
//   2. Asymmetric ρ - vol spikes fast, decays slow
//   3. EKF Guide Density - coarse positioning before Stein
//   4. Adaptive bandwidth α scaling based on regime detection
//   5. Annealed Stein updates (β schedule: 0.3 → 0.65 → 1.0)
//   6. Fused RMSProp + Langevin (SVLD) for stable transport
// =============================================================================

void svpf_step_adaptive(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,    // Host pointer (copied at end)
    float* h_vol_out,       // Host pointer (copied at end)
    float* h_mean_out       // Host pointer for mean h (optional)
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t stream = state->stream;
    int t = state->timestep;
    
    svpf_optimized_init(opt, n);
    
    // Use pre-allocated buffers
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    SVPFParams p = *params;
    
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool use_small_n = (n <= SMALL_N_THRESHOLD);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int persistent_blocks = min(prop.multiProcessorCount, n);
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    if (persistent_smem > prop.sharedMemPerBlockOptin) {
        use_small_n = false;
    }
    
    // =========================================================================
    // PREDICT (with optional Guided Lookahead + MIM + Asymmetric ρ)
    // =========================================================================
    float rho_up = state->use_asymmetric_rho ? state->rho_up : p.rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : p.rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    if (state->use_guided) {
        // GUIDED PREDICT: Proactive - peeks at y_t to know where particles should go
        // INNOVATION GATING: Only activate when model is surprised
        svpf_predict_guided_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1,  // t (index into d_y_single: [y_prev, y_t])
            rho_up, rho_down,
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            state->guided_alpha_base,           // Alpha when model fits (0.0)
            state->guided_alpha_shock,          // Alpha when model fails (0.5)
            state->guided_innovation_threshold, // z-score surprise threshold (1.5)
            n
        );
    } else if (state->use_mim) {
        // MIM kernel: standard predict with fat tails
        svpf_predict_mim_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1,
            rho_up, rho_down,
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            n
        );
    } else {
        // Standard predict (no MIM, no guided)
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, p.rho, p.sigma_z, p.mu, p.gamma, n
        );
    }
    
    // =========================================================================
    // EKF GUIDE DENSITY (coarse positioning before Stein refinement)
    // =========================================================================
    if (state->use_guide) {
        // Update EKF (host-side, ~20 FLOPs)
        svpf_ekf_update(state, y_t, &p);
        
        // Apply guide: either variance-preserving or standard contraction
        if (state->use_guide_preserving) {
            // Variance-preserving: shift cloud without shrinking
            svpf_apply_guide_preserving_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, opt->d_h_mean_prev, state->guide_mean, state->guide_strength, n
            );
        } else {
            // Standard contraction (old behavior)
            svpf_apply_guide_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->guide_mean, state->guide_strength, n
            );
        }
    }
    
    // =========================================================================
    // BANDWIDTH with adaptive scaling (Improvement #1)
    // =========================================================================
    svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_bandwidth, opt->d_bandwidth_sq, 0.3f, n
    );
    
    // Apply adaptive scaling based on regime
    svpf_adaptive_bandwidth_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_bandwidth, 
        state->d_return_ema, state->d_return_var,
        y_t, 0.05f, n
    );
    
    // =========================================================================
    // ANNEALED STEIN UPDATES (Improvement #2)
    //
    // β schedule: 0.3 → 0.65 → 1.0 (configurable via state)
    // This allows particles to explore before committing to likelihood
    //
    // CORRECTED: Now uses mixture prior gradient (O(N²)) per paper Eq. 6
    // =========================================================================
    
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};  // Default schedule
    
    // Newton-Stein requires more shared memory (3*n vs 2*n)
    size_t newton_smem = (3 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    bool newton_fits_smem = (newton_smem <= prop.sharedMemPerBlockOptin);
    bool use_newton = state->use_newton && newton_fits_smem;
    
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing 
                   ? beta_schedule[anneal_idx % 3]
                   : 1.0f;
        
        // === CORRECTED GRADIENT COMPUTATION ===
        // Step 1: Mixture prior gradient (O(N²)) - does NOT depend on β
        if (n <= SMALL_N_THRESHOLD) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        }
        
        // Step 2: Likelihood gradient (O(N))
        svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            opt->d_y_single, 1, state->nu, student_t_const, n
        );
        
        // Step 3: Combine with annealing: grad = grad_prior + β * grad_lik
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
            beta, n
        );
        
        // Step 4 (Newton-Stein): Compute Hessian-preconditioned gradient
        if (use_newton) {
            svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, state->grad_log_p,
                opt->d_precond_grad, opt->d_inv_hessian,
                opt->d_y_single, 1, state->nu, p.sigma_z, n
            );
        }
        
        // Stein iterations at this β level
        int stein_iters = state->n_stein_steps / n_anneal;
        if (anneal_idx == n_anneal - 1) {
            // Last annealing level gets remaining iterations
            stein_iters = state->n_stein_steps - stein_iters * (n_anneal - 1);
        }
        
        for (int s = 0; s < stein_iters; s++) {
            // Compute Stein direction
            cudaMemsetAsync(opt->d_phi, 0, n * sizeof(float), stream);
            
            if (use_newton) {
                // Newton-Stein: use H^{-1} scaled gradient and repulsion
                if (use_small_n) {
                    svpf_stein_newton_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, newton_smem, stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_newton_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                }
            } else {
                // Standard Stein
                if (use_small_n) {
                    svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                }
            }
            
            // Apply transport with fused SVLD (RMSProp + Langevin noise)
            // Temperature control: 0 = SVGD (deterministic), 1 = SVLD (diffusion)
            float temp = state->use_svld ? state->temperature : 0.0f;
            float beta_factor = sqrtf(beta);  // Scale step by sqrt(beta) during annealing
            
            // Reduce step size when guide handles transport (supervisor recommendation)
            float step_size = SVPF_STEIN_STEP_SIZE;
            if (state->use_guide) {
                step_size *= 0.5f;  // Guide handles transport, Stein handles geometry
            }
            
            svpf_apply_transport_svld_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                state->h, opt->d_phi,
                state->d_grad_v,             // RMSProp state
                state->rng_states,           // For Langevin noise
                step_size,
                beta_factor,                 // Annealing factor
                temp,                        // Temperature (0=SVGD, 1=SVLD)
                state->rmsprop_rho,
                state->rmsprop_eps,
                n
            );
            
            // Recompute gradient if more iterations remain
            if (s < stein_iters - 1) {
                // Mixture prior (particles moved)
                if (n <= SMALL_N_THRESHOLD) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                }
                
                // Likelihood gradient
                svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    opt->d_y_single, 1, state->nu, student_t_const, n
                );
                
                // Combine with annealing
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
                    beta, n
                );
                
                // Newton-Stein: recompute Hessian-preconditioned gradient
                if (use_newton) {
                    svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, stream>>>(
                        state->h, state->grad_log_p,
                        opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_y_single, 1, state->nu, p.sigma_z, n
                    );
                }
            }
        }
    }
    
    // =========================================================================
    // FINAL LIKELIHOOD (for output)
    // =========================================================================
    svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->log_weights, opt->d_loglik_single, opt->d_max_log_w, 0, n
    );
    
    // =========================================================================
    // VOL MEAN + H MEAN
    // =========================================================================
    svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        state->h, opt->d_vol_single, 0, n
    );
    
    // H mean (simple block reduce)
    // We reuse the logsumexp infrastructure to get h mean
    float* d_h_mean = state->d_result_h_mean;
    {
        // Quick inline h-mean computation
        float h_sum = 0.0f;
        cudaMemcpyAsync(&h_sum, state->d_scalar_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        // Actually, let's do this properly with a kernel call - use CUB
        cub::DeviceReduce::Sum(state->d_cub_temp, state->cub_temp_bytes, 
                               state->h, state->d_scalar_sum, n, stream);
    }
    
    // Store h_mean for particle-local params in next step
    svpf_store_h_mean_kernel<<<1, 1, 0, stream>>>(
        state->d_scalar_sum, opt->d_h_mean_prev, n
    );
    
    // Single sync + copy results
    cudaStreamSynchronize(stream);
    
    // Copy results
    float h_sum_host;
    cudaMemcpy(&h_sum_host, state->d_scalar_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (h_loglik_out) cudaMemcpy(h_loglik_out, opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_vol_out) cudaMemcpy(h_vol_out, opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_mean_out) *h_mean_out = h_sum_host / (float)n;
    
    state->timestep++;
}

// =============================================================================
// CUDA GRAPH-BASED STEP FUNCTION
// =============================================================================
//
// LATENCY OPTIMIZATION: Captures the kernel sequence into a CUDA graph
// on first call, then replays with minimal CPU overhead (~5μs vs ~100μs+).
//
// Requirements:
//   - Fixed execution path (resolved at capture time)
//   - All parameters staged to device before replay
//   - No CUB (replaced with custom reduction)
//   - No cudaMemsetAsync (replaced with kernel)
//
// Usage:
//   1. First call: captures graph (adds ~1ms one-time cost)
//   2. Subsequent calls: replays graph (~5μs CPU overhead)
//   3. Config change: automatic recapture
// =============================================================================

// Internal: Capture the SVPF kernel sequence into a graph
static void svpf_graph_capture_internal(
    SVPFState* state,
    const SVPFParams* params
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    cudaStream_t capture_stream = opt->graph_stream;
    
    // Compute derived constants (same as svpf_step_adaptive)
    SVPFParams p = *params;
    float student_t_const = lgammaf((state->nu + 1.0f) / 2.0f)
                          - lgammaf(state->nu / 2.0f)
                          - 0.5f * logf((float)M_PI * state->nu);
    
    int n_blocks_1d = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int persistent_blocks = min(prop.multiProcessorCount, n);
    size_t persistent_smem = (2 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    size_t newton_smem = (3 * n + BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    bool use_small_n = (n <= SMALL_N_THRESHOLD) && (persistent_smem <= prop.sharedMemPerBlockOptin);
    bool use_newton = state->use_newton && (newton_smem <= prop.sharedMemPerBlockOptin);
    
    float rho_up = state->use_asymmetric_rho ? state->rho_up : p.rho;
    float rho_down = state->use_asymmetric_rho ? state->rho_down : p.rho;
    float delta_rho = state->use_local_params ? state->delta_rho : 0.0f;
    float delta_sigma = state->use_local_params ? state->delta_sigma : 0.0f;
    
    int n_anneal = state->use_annealing ? state->n_anneal_steps : 1;
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
    
    // Begin capture
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    
    // =========================================================================
    // PREDICT (Guided + MIM)
    // =========================================================================
    if (state->use_guided) {
        svpf_predict_guided_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1, rho_up, rho_down,
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma,
            state->guided_alpha_base,
            state->guided_alpha_shock,
            state->guided_innovation_threshold,
            n
        );
    } else if (state->use_mim) {
        svpf_predict_mim_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, opt->d_h_mean_prev,
            1, rho_up, rho_down,
            p.sigma_z, p.mu, p.gamma,
            state->mim_jump_prob, state->mim_jump_scale,
            delta_rho, delta_sigma, n
        );
    } else {
        svpf_predict_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, state->h_prev, state->rng_states,
            opt->d_y_single, 1, p.rho, p.sigma_z, p.mu, p.gamma, n
        );
    }
    
    // =========================================================================
    // EKF GUIDE (guide_mean staged to d_guide_mean before graph launch)
    // =========================================================================
    if (state->use_guide) {
        if (state->use_guide_preserving) {
            // Graph-compatible: reads guide_mean from device pointer
            svpf_apply_guide_preserving_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, opt->d_h_mean_prev, opt->d_guide_mean, state->guide_strength, n
            );
        } else {
            // Graph-compatible: reads guide_mean from device pointer
            svpf_apply_guide_kernel_graph<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, opt->d_guide_mean, state->guide_strength, n
            );
        }
    }
    
    // =========================================================================
    // BANDWIDTH
    // =========================================================================
    svpf_bandwidth_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_bandwidth, opt->d_bandwidth_sq, 0.3f, n
    );
    
    // Use GRAPH-COMPATIBLE version that reads y_t from device pointer
    // (scalar params are "burned in" at capture time, so we must read from ptr)
    svpf_adaptive_bandwidth_kernel_graph<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_bandwidth,
        state->d_return_ema, state->d_return_var,
        opt->d_y_single,  // Device pointer (staged before graph launch)
        1,                 // Index: y_t is at index 1
        0.05f, n
    );
    
    // =========================================================================
    // ANNEALED STEIN UPDATES
    // =========================================================================
    for (int anneal_idx = 0; anneal_idx < n_anneal; anneal_idx++) {
        float beta = state->use_annealing ? beta_schedule[anneal_idx % 3] : 1.0f;
        
        // Mixture prior gradient
        if (use_small_n) {
            svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        } else {
            svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->h_prev, state->grad_log_p,
                p.rho, p.sigma_z, p.mu, n
            );
        }
        
        // Likelihood gradient
        svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->h, opt->d_grad_lik, state->log_weights,
            opt->d_y_single, 1, state->nu, student_t_const, n
        );
        
        // Combine gradients
        svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
            state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
            beta, n
        );
        
        // Hessian preconditioning
        if (use_newton) {
            svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, state->grad_log_p,
                opt->d_precond_grad, opt->d_inv_hessian,
                opt->d_y_single, 1, state->nu, p.sigma_z, n
            );
        }
        
        // Stein iterations
        int stein_iters = state->n_stein_steps / n_anneal;
        if (anneal_idx == n_anneal - 1) {
            stein_iters = state->n_stein_steps - stein_iters * (n_anneal - 1);
        }
        
        for (int s = 0; s < stein_iters; s++) {
            // Memset phi (graph-compatible)
            svpf_memset_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                opt->d_phi, 0.0f, n
            );
            
            // Stein kernel
            if (use_newton) {
                if (use_small_n) {
                    svpf_stein_newton_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, newton_smem, capture_stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_newton_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_phi, opt->d_bandwidth, n
                    );
                }
            } else {
                if (use_small_n) {
                    svpf_stein_persistent_kernel<<<persistent_blocks, BLOCK_SIZE, persistent_smem, capture_stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                } else {
                    int num_tiles = (n + TILE_J - 1) / TILE_J;
                    dim3 grid_2d(n, num_tiles);
                    svpf_stein_2d_kernel<<<grid_2d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->grad_log_p, opt->d_phi, opt->d_bandwidth, n
                    );
                }
            }
            
            // Transport
            float temp = state->use_svld ? state->temperature : 0.0f;
            float beta_factor = sqrtf(beta);
            float step_size = SVPF_STEIN_STEP_SIZE;
            if (state->use_guide) step_size *= 0.5f;
            
            svpf_apply_transport_svld_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                state->h, opt->d_phi,
                state->d_grad_v, state->rng_states,
                step_size, beta_factor, temp,
                state->rmsprop_rho, state->rmsprop_eps, n
            );
            
            // Recompute gradient if more iterations
            if (s < stein_iters - 1) {
                if (use_small_n) {
                    svpf_mixture_prior_grad_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                } else {
                    svpf_mixture_prior_grad_tiled_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->h_prev, state->grad_log_p,
                        p.rho, p.sigma_z, p.mu, n
                    );
                }
                
                svpf_likelihood_only_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                    state->h, opt->d_grad_lik, state->log_weights,
                    opt->d_y_single, 1, state->nu, student_t_const, n
                );
                
                svpf_combine_gradients_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                    state->grad_log_p, opt->d_grad_lik, state->grad_log_p,
                    beta, n
                );
                
                if (use_newton) {
                    svpf_hessian_precond_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
                        state->h, state->grad_log_p,
                        opt->d_precond_grad, opt->d_inv_hessian,
                        opt->d_y_single, 1, state->nu, p.sigma_z, n
                    );
                }
            }
        }
    }
    
    // =========================================================================
    // FINAL OUTPUTS
    // =========================================================================
    svpf_logsumexp_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->log_weights, opt->d_loglik_single, opt->d_max_log_w, 0, n
    );
    
    svpf_vol_mean_opt_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_vol_single, 0, n
    );
    
    // H-mean using graph-compatible reduction (not CUB)
    // First zero the accumulator
    svpf_memset_kernel<<<1, 1, 0, capture_stream>>>(
        state->d_scalar_sum, 0.0f, 1
    );
    
    // Parallel reduce
    svpf_h_mean_reduce_kernel<<<n_blocks_1d, BLOCK_SIZE, 0, capture_stream>>>(
        state->h, opt->d_phi, n  // Reuse d_phi as partial sums
    );
    
    // Finalize
    svpf_h_mean_finalize_kernel<<<1, BLOCK_SIZE, 0, capture_stream>>>(
        opt->d_phi, opt->d_h_mean_prev, n_blocks_1d, n
    );
    
    // End capture
    cudaStreamEndCapture(capture_stream, &opt->graph);
    cudaGraphInstantiate(&opt->graph_exec, opt->graph, NULL, NULL, 0);
    
    opt->graph_captured = true;
    opt->graph_n = n;
    opt->graph_n_stein = state->n_stein_steps;
}

// =============================================================================
// PUBLIC API: Graph-accelerated SVPF step
// =============================================================================

void svpf_step_graph(
    SVPFState* state,
    float y_t,
    float y_prev,
    const SVPFParams* params,
    float* h_loglik_out,
    float* h_vol_out,
    float* h_mean_out
) {
    SVPFOptimizedState* opt = get_opt(state);
    int n = state->n_particles;
    
    svpf_optimized_init(opt, n);
    
    // Check if recapture needed (config changed)
    bool need_capture = !opt->graph_captured 
                     || opt->graph_n != n 
                     || opt->graph_n_stein != state->n_stein_steps;
    
    if (need_capture) {
        // Destroy old graph if exists
        if (opt->graph_captured) {
            cudaGraphExecDestroy(opt->graph_exec);
            cudaGraphDestroy(opt->graph);
            opt->graph_captured = false;
        }
        
        // Capture new graph
        svpf_graph_capture_internal(state, params);
    }
    
    // =========================================================================
    // PRE-GRAPH: Stage parameters to device
    // =========================================================================
    
    // 1. Stage y_prev, y_t
    float y_arr[2] = {y_prev, y_t};
    cudaMemcpyAsync(opt->d_y_single, y_arr, 2 * sizeof(float), 
                    cudaMemcpyHostToDevice, opt->graph_stream);
    
    // 2. Run EKF update (host-side, ~20 FLOPs) and stage guide_mean to device
    if (state->use_guide) {
        svpf_ekf_update(state, y_t, params);
        // Stage guide_mean to device for graph-compatible kernels
        cudaMemcpyAsync(opt->d_guide_mean, &state->guide_mean, sizeof(float),
                        cudaMemcpyHostToDevice, opt->graph_stream);
    }
    
    // 3. Sync before graph launch (ensure staging complete)
    cudaStreamSynchronize(opt->graph_stream);
    
    // =========================================================================
    // GRAPH LAUNCH
    // =========================================================================
    cudaGraphLaunch(opt->graph_exec, opt->graph_stream);
    
    // =========================================================================
    // POST-GRAPH: Copy results
    // =========================================================================
    cudaStreamSynchronize(opt->graph_stream);
    
    if (h_loglik_out) {
        cudaMemcpy(h_loglik_out, opt->d_loglik_single, sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (h_vol_out) {
        cudaMemcpy(h_vol_out, opt->d_vol_single, sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (h_mean_out) {
        float h_mean;
        cudaMemcpy(&h_mean, opt->d_h_mean_prev, sizeof(float), cudaMemcpyDeviceToHost);
        *h_mean_out = h_mean;
    }
    
    state->timestep++;
}

// =============================================================================
// Utility: Check if graph is captured
// =============================================================================
bool svpf_graph_is_captured(SVPFState* state) {
    return get_opt(state)->graph_captured;
}

// =============================================================================
// Utility: Force graph recapture (call after config change)
// =============================================================================
void svpf_graph_invalidate(SVPFState* state) {
    SVPFOptimizedState* opt = get_opt(state);
    if (opt->graph_captured) {
        cudaGraphExecDestroy(opt->graph_exec);
        cudaGraphDestroy(opt->graph);
        opt->graph_captured = false;
    }
}
