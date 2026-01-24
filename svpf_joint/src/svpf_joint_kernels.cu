/**
 * @file svpf_joint_kernels.cu
 * @brief CUDA kernels for Joint State-Parameter SVPF
 */

#include "svpf_joint.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// =============================================================================
// DEVICE HELPERS
// =============================================================================

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_deriv(float sig) {
    // Derivative of sigmoid: sig * (1 - sig)
    // Input is already sigmoid(x)
    return sig * (1.0f - sig);
}

__device__ __forceinline__ float clamp_h(float h) {
    return fmaxf(fminf(h, 5.0f), -15.0f);
}

__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(fmaxf(x, -20.0f), 20.0f));
}

// =============================================================================
// INITIALIZATION KERNELS
// =============================================================================

__global__ void svpf_joint_init_rng_kernel(
    curandStatePhilox4_32_10_t* rng,
    unsigned long long seed,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &rng[i]);
}

__global__ void svpf_joint_init_particles_kernel(
    float* d_h,
    float* d_mu_tilde,
    float* d_rho_tilde,
    float* d_sigma_tilde,
    curandStatePhilox4_32_10_t* rng,
    float mu_init,      // Initial μ̃ (e.g., -3.5)
    float rho_init,     // Initial ρ̃ (e.g., 2.0 → ρ ≈ 0.88)
    float sigma_init,   // Initial σ̃ (e.g., -2.0 → σ ≈ 0.14)
    float h_spread,     // Spread for h initialization
    float param_spread, // Spread for parameter initialization
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Initialize with spread around prior means
    float noise_h = curand_normal(&rng[i]);
    float noise_mu = curand_normal(&rng[i]);
    float noise_rho = curand_normal(&rng[i]);
    float noise_sigma = curand_normal(&rng[i]);
    
    d_h[i] = mu_init + h_spread * noise_h;
    d_mu_tilde[i] = mu_init + param_spread * noise_mu;
    d_rho_tilde[i] = rho_init + param_spread * noise_rho;
    d_sigma_tilde[i] = sigma_init + param_spread * noise_sigma;
}

// =============================================================================
// PREDICT KERNEL (with parameter diffusion)
// =============================================================================

__global__ void svpf_joint_predict_kernel(
    float* __restrict__ d_h,
    float* __restrict__ d_h_prev,
    float* __restrict__ d_mu_tilde,
    float* __restrict__ d_rho_tilde,
    float* __restrict__ d_sigma_tilde,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    float diffusion_mu,
    float diffusion_rho,
    float diffusion_sigma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Store previous h for gradient computation
    float h_i = d_h[i];
    d_h_prev[i] = h_i;
    
    // =========================================================================
    // PARAMETER DIFFUSION (small random walk)
    // =========================================================================
    float noise_mu = curand_normal(&rng[i]);
    float noise_rho = curand_normal(&rng[i]);
    float noise_sigma = curand_normal(&rng[i]);
    
    float mu_tilde = d_mu_tilde[i] + diffusion_mu * noise_mu;
    float rho_tilde = d_rho_tilde[i] + diffusion_rho * noise_rho;
    float sigma_tilde = d_sigma_tilde[i] + diffusion_sigma * noise_sigma;
    
    // Clamp to reasonable ranges (unconstrained space)
    mu_tilde = fmaxf(fminf(mu_tilde, 5.0f), -10.0f);
    rho_tilde = fmaxf(fminf(rho_tilde, 5.0f), -5.0f);
    sigma_tilde = fmaxf(fminf(sigma_tilde, 2.0f), -5.0f);
    
    d_mu_tilde[i] = mu_tilde;
    d_rho_tilde[i] = rho_tilde;
    d_sigma_tilde[i] = sigma_tilde;
    
    // =========================================================================
    // STATE PROPAGATION with particle-local parameters
    // =========================================================================
    float mu = mu_tilde;  // Identity transform
    float rho = sigmoid(rho_tilde);
    float sigma = safe_exp(sigma_tilde);
    sigma = fmaxf(sigma, 0.01f);
    
    // AR(1) prediction
    float noise_h = curand_normal(&rng[i]);
    float h_pred = mu + rho * (h_i - mu);
    
    d_h[i] = clamp_h(h_pred + sigma * noise_h);
}

// =============================================================================
// GRADIENT KERNEL
// =============================================================================

__global__ void svpf_joint_gradient_kernel(
    const float* __restrict__ d_h,
    const float* __restrict__ d_h_prev,
    const float* __restrict__ d_mu_tilde,
    const float* __restrict__ d_rho_tilde,
    const float* __restrict__ d_sigma_tilde,
    float* __restrict__ d_grad_h,
    float* __restrict__ d_grad_mu,
    float* __restrict__ d_grad_rho,
    float* __restrict__ d_grad_sigma,
    float* __restrict__ d_log_w,
    float y_t,
    float nu,
    float student_t_const,
    float lik_offset,
    float mu_prior_mean, float mu_prior_var,
    float rho_prior_mean, float rho_prior_var,
    float sigma_prior_mean, float sigma_prior_var,
    float prior_weight,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    // Load state
    float h_j = d_h[j];
    float h_prev_j = d_h_prev[j];
    
    // Transform parameters
    float mu_tilde = d_mu_tilde[j];
    float rho_tilde = d_rho_tilde[j];
    float sigma_tilde = d_sigma_tilde[j];
    
    float mu = mu_tilde;
    float rho = sigmoid(rho_tilde);
    float sigma = safe_exp(sigma_tilde);
    sigma = fmaxf(sigma, 0.01f);
    
    float var = sigma * sigma;
    
    // =========================================================================
    // LIKELIHOOD GRADIENT (affects h only)
    // =========================================================================
    float y_sq = y_t * y_t;
    float vol = safe_exp(h_j);
    float scaled_y_sq = y_sq / (vol + 1e-8f);
    float A = scaled_y_sq / nu;
    float one_plus_A = 1.0f + A;
    
    // Log-weight for ESS
    d_log_w[j] = student_t_const - 0.5f * h_j 
               - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));
    
    // Exact Student-t gradient with bias correction
    float grad_h_lik = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A - lik_offset;
    
    // =========================================================================
    // OBSERVATION SURPRISE DETECTION
    // =========================================================================
    float expected_y_sq_prev = safe_exp(h_prev_j);  // Use h_PREV to gauge surprise
    float obs_z_sq = y_sq / (expected_y_sq_prev + 1e-8f);
    
    // =========================================================================
    // VIRTUAL VARIANCE (Decoupling Motion from Learning)
    // =========================================================================
    // The problem: σ is small (~0.15), so the spring 1/σ² is massive (~44).
    // Even with aggressive gating, the restoring force overwhelms likelihood.
    // σ inflation is reactive - there's a lag before it catches up.
    //
    // Solution: Use DIFFERENT variances for motion vs learning.
    // - var_learning: Real σ² (small). Used for ∇_σ so it sees the huge error.
    // - var_motion: Virtual inflated σ² (smoothly scaled). Used for ∇_h.
    //
    // KEY CHANGE: Smooth inflation with higher threshold to avoid calm drift.
    
    // Variance for LEARNING (true state)
    // Must use real σ so gradient sees the huge error z² >> 1
    float var_learning = var;
    
    // Variance for MOTION (smoothly inflated based on surprise)
    // 1. Always keep a minimum floor (prevents tiny σ => iron spring)
    const float var_floor = 0.03f * 0.03f;  // σ_floor = 0.03
    
    // 2. Gradual inflation: only inflate when obs_z_sq > 4 (2-sigma)
    //    This prevents casual excursions from loosening the spring too much
    float surprise = fmaxf(obs_z_sq - 4.0f, 0.0f);  // Raised from 1.0
    float inflate = 1.0f + 0.3f * fminf(surprise, 20.0f);  // Reduced gain from 0.5
    
    // 3. Apply inflation to real variance, then floor
    float var_motion = fmaxf(var * inflate, var_floor);
    
    // 4. Strong panic floor for extreme events
    bool is_panic = (obs_z_sq > 9.0f);
    if (is_panic) {
        var_motion = fmaxf(var_motion, 1.0f);
    }
    
    // =========================================================================
    // TRANSITION GRADIENT (with smoothly inflated variance)
    // =========================================================================
    float h_pred = mu + rho * (h_prev_j - mu);
    float diff = h_j - h_pred;
    
    // Gradient for h: Use MOTION variance (looser spring allows tracking)
    float grad_h_trans = -diff / var_motion;
    
    // Gradients for mu, rho: Also use motion variance for consistency
    float grad_mu_trans = (diff / var_motion) * (1.0f - rho);
    float grad_rho_trans = (diff / var_motion) * (h_prev_j - mu) * sigmoid_deriv(rho);
    
    // Gradient for sigma: Use LEARNING variance (the truth!)
    // The error z² will be massive (e.g., 100+) because we use real small σ
    // This drives σ up hard to catch up with the virtual variance
    float z_sq_learning = (diff * diff) / var_learning;
    float grad_sigma_trans = z_sq_learning - 1.0f;
    
    // =========================================================================
    // OBSERVATION-DRIVEN FORCE FOR SIGMA (bidirectional)
    // =========================================================================
    // This supplements the natural z²-1 signal
    // 
    // Threshold at 9 (3-sigma) for crash detection
    // ASYMMETRIC: Strong upward during crash, moderate downward in calm
    float obs_force_sigma = 0.0f;
    if (obs_z_sq > 9.0f) {
        obs_force_sigma = fminf((obs_z_sq - 9.0f) * 0.5f, 5.0f);
    } else if (obs_z_sq < 1.5f) {
        // When observations are consistent with current h, push σ down
        // Stronger than before to counteract upward drift
        obs_force_sigma = -1.0f * (1.5f - obs_z_sq);  // Max -1.5 when z²=0
    }
    
    // Mean reversion: pull sigma_tilde toward prior mean
    // This prevents runaway growth in σ - increased strength
    const float sigma_mean_reversion = 0.2f;  // Increased from 0.1
    float sigma_mr_force = -sigma_mean_reversion * (sigma_tilde - sigma_prior_mean);
    
    // =========================================================================
    // PRIOR GRADIENT (weak regularization)
    // =========================================================================
    float grad_mu_prior = -(mu_tilde - mu_prior_mean) / mu_prior_var;
    float grad_rho_prior = -(rho_tilde - rho_prior_mean) / rho_prior_var;
    float grad_sigma_prior = -(sigma_tilde - sigma_prior_mean) / sigma_prior_var;
    
    // =========================================================================
    // COMBINE
    // =========================================================================
    // H gradient: likelihood + transition (using var_motion for loose spring)
    // The virtual variance lets h follow likelihood freely during panic
    float grad_h = grad_h_lik + grad_h_trans;
    
    // Mu, rho: transition (using var_motion) + prior
    float grad_mu = grad_mu_trans + prior_weight * grad_mu_prior;
    float grad_rho = grad_rho_trans + prior_weight * grad_rho_prior;
    
    // SIGMA gradient: transition + observation force + mean reversion + prior
    // Mean reversion prevents runaway σ growth
    float grad_sigma = grad_sigma_trans + obs_force_sigma + sigma_mr_force 
                     + prior_weight * grad_sigma_prior;
    
    // Clip gradients - panic-dependent σ clip for responsiveness during crash
    grad_h = fmaxf(fminf(grad_h, 10.0f), -10.0f);
    grad_mu = fmaxf(fminf(grad_mu, 5.0f), -5.0f);
    grad_rho = fmaxf(fminf(grad_rho, 5.0f), -5.0f);
    float sigma_clip_hi = is_panic ? 10.0f : 5.0f;
    grad_sigma = fmaxf(fminf(grad_sigma, sigma_clip_hi), -5.0f);
    
    // Store
    d_grad_h[j] = grad_h;
    d_grad_mu[j] = grad_mu;
    d_grad_rho[j] = grad_rho;
    d_grad_sigma[j] = grad_sigma;
}

// =============================================================================
// STEIN TRANSPORT KERNEL (diagonal kernel with adaptive learning)
// =============================================================================

__global__ void svpf_joint_stein_kernel(
    float* __restrict__ d_h,
    float* __restrict__ d_mu_tilde,
    float* __restrict__ d_rho_tilde,
    float* __restrict__ d_sigma_tilde,
    const float* __restrict__ d_grad_h,
    const float* __restrict__ d_grad_mu,
    const float* __restrict__ d_grad_rho,
    const float* __restrict__ d_grad_sigma,
    const float* __restrict__ d_h_prev,  // For surprise detection
    float y_t,  // For surprise detection
    float bw_h, float bw_mu, float bw_rho, float bw_sigma,
    float step_h, float step_mu, float step_rho, float step_sigma,
    int n
) {
    extern __shared__ float smem[];
    
    // Shared memory layout: 8 arrays of size n
    float* sh_h = smem;
    float* sh_mu = smem + n;
    float* sh_rho = smem + 2*n;
    float* sh_sigma = smem + 3*n;
    float* sh_grad_h = smem + 4*n;
    float* sh_grad_mu = smem + 5*n;
    float* sh_grad_rho = smem + 6*n;
    float* sh_grad_sigma = smem + 7*n;
    
    // Cooperative load
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = d_h[k];
        sh_mu[k] = d_mu_tilde[k];
        sh_rho[k] = d_rho_tilde[k];
        sh_sigma[k] = d_sigma_tilde[k];
        sh_grad_h[k] = d_grad_h[k];
        sh_grad_mu[k] = d_grad_mu[k];
        sh_grad_rho[k] = d_grad_rho[k];
        sh_grad_sigma[k] = d_grad_sigma[k];
    }
    __syncthreads();
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    // Current particle
    float h_j = sh_h[j];
    float mu_j = sh_mu[j];
    float rho_j = sh_rho[j];
    float sigma_j = sh_sigma[j];
    
    // =========================================================================
    // SURPRISE-BOOSTED STEP SIZE with PANIC BYPASS
    // =========================================================================
    // When observation is surprising, bypass natural gradient scaling
    // Switch from "careful Riemannian descent" to "raw Euclidean panic mode"
    
    float h_prev_j = d_h_prev[j];
    float y_sq = y_t * y_t;
    float expected_y_sq_prev = expf(h_prev_j);
    float obs_z_sq = y_sq / (expected_y_sq_prev + 1e-8f);
    
    bool is_surprise = (obs_z_sq > 9.0f);  // 3-sigma event (matches gradient kernel)
    
    // =========================================================================
    // STEP SIZE BOOST (during panic/surprise)
    // =========================================================================
    // h gets large boost (5x) to track the observation quickly
    // σ gets moderate-high boost (4x) for faster inflation during crash
    
    float step_boost_h = is_surprise ? 5.0f : 1.0f;
    float step_boost_sigma = is_surprise ? 4.0f : 1.0f;
    
    // Effective step sizes
    float eff_step_h = step_h * step_boost_h;
    float eff_step_mu = step_mu;
    float eff_step_rho = step_rho;
    float eff_step_sigma = step_sigma * step_boost_sigma;
    
    // Precompute inverse squared bandwidths
    float inv_bw_h_sq = 1.0f / (bw_h * bw_h + 1e-8f);
    float inv_bw_mu_sq = 1.0f / (bw_mu * bw_mu + 1e-8f);
    float inv_bw_rho_sq = 1.0f / (bw_rho * bw_rho + 1e-8f);
    float inv_bw_sigma_sq = 1.0f / (bw_sigma * bw_sigma + 1e-8f);
    
    // Accumulate Stein update
    float phi_h = 0.0f;
    float phi_mu = 0.0f;
    float phi_rho = 0.0f;
    float phi_sigma = 0.0f;
    
    for (int i = 0; i < n; i++) {
        // Differences
        float dh = sh_h[i] - h_j;
        float dmu = sh_mu[i] - mu_j;
        float drho = sh_rho[i] - rho_j;
        float dsigma = sh_sigma[i] - sigma_j;
        
        // IMQ (Inverse Multi-Quadratic) kernel: K = 1/(1 + dist²)
        // Heavier tails than RBF, better for SVGD (from svpf_opt)
        float dist_sq = dh * dh * inv_bw_h_sq
                      + dmu * dmu * inv_bw_mu_sq
                      + drho * drho * inv_bw_rho_sq
                      + dsigma * dsigma * inv_bw_sigma_sq;
        float base = 1.0f + dist_sq;
        float k_ij = 1.0f / base;
        float k_sq = k_ij * k_ij;
        
        // Kernel gradient (repulsive term) for IMQ
        // dK/dx = -2 * diff / bw² * K²
        float dk_dh = -2.0f * dh * inv_bw_h_sq * k_sq;
        float dk_dmu = -2.0f * dmu * inv_bw_mu_sq * k_sq;
        float dk_drho = -2.0f * drho * inv_bw_rho_sq * k_sq;
        float dk_dsigma = -2.0f * dsigma * inv_bw_sigma_sq * k_sq;
        
        // Stein: k * grad + grad_k
        phi_h += k_ij * sh_grad_h[i] + dk_dh;
        phi_mu += k_ij * sh_grad_mu[i] + dk_dmu;
        phi_rho += k_ij * sh_grad_rho[i] + dk_drho;
        phi_sigma += k_ij * sh_grad_sigma[i] + dk_dsigma;
    }
    
    // Standard SVGD normalization: 1/n
    float inv_n = 1.0f / (float)n;
    phi_h *= inv_n;
    phi_mu *= inv_n;
    phi_rho *= inv_n;
    phi_sigma *= inv_n;
    
    // Apply update with effective learning rates
    d_h[j] = clamp_h(h_j + eff_step_h * phi_h);
    
    float new_mu = mu_j + eff_step_mu * phi_mu;
    float new_rho = rho_j + eff_step_rho * phi_rho;
    float new_sigma = sigma_j + eff_step_sigma * phi_sigma;
    
    // Clamp parameters
    d_mu_tilde[j] = fmaxf(fminf(new_mu, 5.0f), -10.0f);
    d_rho_tilde[j] = fmaxf(fminf(new_rho, 5.0f), -5.0f);
    d_sigma_tilde[j] = fmaxf(fminf(new_sigma, 2.0f), -5.0f);
}

// =============================================================================
// EXTRACT / DIAGNOSTIC KERNEL
// =============================================================================

__global__ void svpf_joint_extract_kernel(
    const float* __restrict__ d_h,
    const float* __restrict__ d_mu_tilde,
    const float* __restrict__ d_rho_tilde,
    const float* __restrict__ d_sigma_tilde,
    float* __restrict__ d_param_mean,
    float* __restrict__ d_param_std,
    float* __restrict__ d_std_unconstrained,
    int* __restrict__ d_collapse_flags,
    float collapse_thresh_mu,
    float collapse_thresh_rho,
    float collapse_thresh_sigma,
    int n
) {
    // Shared accumulators
    __shared__ float s_h_sum, s_h_sq;
    __shared__ float s_mu_sum, s_rho_sum, s_sigma_sum;
    __shared__ float s_mu_sq, s_rho_sq, s_sigma_sq;
    __shared__ float s_mu_t_sum, s_rho_t_sum, s_sigma_t_sum;
    __shared__ float s_mu_t_sq, s_rho_t_sq, s_sigma_t_sq;
    
    if (threadIdx.x == 0) {
        s_h_sum = s_h_sq = 0.0f;
        s_mu_sum = s_rho_sum = s_sigma_sum = 0.0f;
        s_mu_sq = s_rho_sq = s_sigma_sq = 0.0f;
        s_mu_t_sum = s_rho_t_sum = s_sigma_t_sum = 0.0f;
        s_mu_t_sq = s_rho_t_sq = s_sigma_t_sq = 0.0f;
    }
    __syncthreads();
    
    // Thread-local accumulators
    float h_sum = 0.0f, h_sq = 0.0f;
    float mu_sum = 0.0f, rho_sum = 0.0f, sigma_sum = 0.0f;
    float mu_sq = 0.0f, rho_sq = 0.0f, sigma_sq = 0.0f;
    float mu_t_sum = 0.0f, rho_t_sum = 0.0f, sigma_t_sum = 0.0f;
    float mu_t_sq = 0.0f, rho_t_sq = 0.0f, sigma_t_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float h = d_h[i];
        float mu_t = d_mu_tilde[i];
        float rho_t = d_rho_tilde[i];
        float sigma_t = d_sigma_tilde[i];
        
        // Constrained space
        float mu = mu_t;
        float rho = sigmoid(rho_t);
        float sigma = expf(sigma_t);
        
        h_sum += h;
        h_sq += h * h;
        
        mu_sum += mu;
        rho_sum += rho;
        sigma_sum += sigma;
        
        mu_sq += mu * mu;
        rho_sq += rho * rho;
        sigma_sq += sigma * sigma;
        
        // Unconstrained space (for diversity)
        mu_t_sum += mu_t;
        rho_t_sum += rho_t;
        sigma_t_sum += sigma_t;
        
        mu_t_sq += mu_t * mu_t;
        rho_t_sq += rho_t * rho_t;
        sigma_t_sq += sigma_t * sigma_t;
    }
    
    // Atomic adds
    atomicAdd(&s_h_sum, h_sum);
    atomicAdd(&s_h_sq, h_sq);
    atomicAdd(&s_mu_sum, mu_sum);
    atomicAdd(&s_rho_sum, rho_sum);
    atomicAdd(&s_sigma_sum, sigma_sum);
    atomicAdd(&s_mu_sq, mu_sq);
    atomicAdd(&s_rho_sq, rho_sq);
    atomicAdd(&s_sigma_sq, sigma_sq);
    atomicAdd(&s_mu_t_sum, mu_t_sum);
    atomicAdd(&s_rho_t_sum, rho_t_sum);
    atomicAdd(&s_sigma_t_sum, sigma_t_sum);
    atomicAdd(&s_mu_t_sq, mu_t_sq);
    atomicAdd(&s_rho_t_sq, rho_t_sq);
    atomicAdd(&s_sigma_t_sq, sigma_t_sq);
    __syncthreads();
    
    // Thread 0 computes final statistics
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        
        // Constrained param means
        d_param_mean[0] = s_mu_sum * inv_n;
        d_param_mean[1] = s_rho_sum * inv_n;
        d_param_mean[2] = s_sigma_sum * inv_n;
        d_param_mean[3] = s_h_sum * inv_n;  // h_mean in slot [3]
        
        // Constrained param stds
        d_param_std[0] = sqrtf(fmaxf(s_mu_sq * inv_n - d_param_mean[0] * d_param_mean[0], 0.0f));
        d_param_std[1] = sqrtf(fmaxf(s_rho_sq * inv_n - d_param_mean[1] * d_param_mean[1], 0.0f));
        d_param_std[2] = sqrtf(fmaxf(s_sigma_sq * inv_n - d_param_mean[2] * d_param_mean[2], 0.0f));
        
        // Unconstrained stds (for collapse detection)
        float mu_t_mean = s_mu_t_sum * inv_n;
        float rho_t_mean = s_rho_t_sum * inv_n;
        float sigma_t_mean = s_sigma_t_sum * inv_n;
        
        float std_mu_t = sqrtf(fmaxf(s_mu_t_sq * inv_n - mu_t_mean * mu_t_mean, 0.0f));
        float std_rho_t = sqrtf(fmaxf(s_rho_t_sq * inv_n - rho_t_mean * rho_t_mean, 0.0f));
        float std_sigma_t = sqrtf(fmaxf(s_sigma_t_sq * inv_n - sigma_t_mean * sigma_t_mean, 0.0f));
        
        d_std_unconstrained[0] = std_mu_t;
        d_std_unconstrained[1] = std_rho_t;
        d_std_unconstrained[2] = std_sigma_t;
        
        // Collapse flags
        d_collapse_flags[0] = (std_mu_t < collapse_thresh_mu) ? 1 : 0;
        d_collapse_flags[1] = (std_rho_t < collapse_thresh_rho) ? 1 : 0;
        d_collapse_flags[2] = (std_sigma_t < collapse_thresh_sigma) ? 1 : 0;
        
        // Store h_mean in param_mean[3] (extra slot)
        // Actually we'll handle this in host code
    }
}

// =============================================================================
// BANDWIDTH COMPUTATION (median heuristic approximation)
// =============================================================================

// Simplified: use variance-based approximation
// median ≈ sqrt(variance) for roughly Gaussian distributions
__global__ void svpf_joint_compute_bandwidth_kernel(
    const float* __restrict__ d_values,
    float* __restrict__ d_output,  // [mean, var, bandwidth]
    int n
) {
    __shared__ float s_sum, s_sq;
    
    if (threadIdx.x == 0) {
        s_sum = s_sq = 0.0f;
    }
    __syncthreads();
    
    float sum = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = d_values[i];
        sum += v;
        sq += v * v;
    }
    
    atomicAdd(&s_sum, sum);
    atomicAdd(&s_sq, sq);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        float mean = s_sum * inv_n;
        float var = s_sq * inv_n - mean * mean;
        float std_dev = sqrtf(fmaxf(var, 1e-8f));
        
        // Bandwidth using Silverman's rule (like svpf_opt)
        // bw² = 2 * variance / log(n + 1)
        float bw_sq = 2.0f * var / logf((float)n + 1.0f);
        bw_sq = fmaxf(bw_sq, 1e-6f);
        float bw = sqrtf(bw_sq);
        
        // Clamp
        bw = fmaxf(fminf(bw, 2.0f), 0.01f);
        
        d_output[0] = mean;
        d_output[1] = var;
        d_output[2] = bw;
    }
}
