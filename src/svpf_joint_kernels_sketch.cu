// =============================================================================
// JOINT STATE-PARAMETER SVPF - KERNEL SKETCHES
// =============================================================================
// 
// Each particle carries: [h, μ̃, ρ̃, σ̃]
// - h:  log-volatility (unconstrained)
// - μ̃:  mean level (unconstrained, μ̃ = μ)
// - ρ̃:  persistence (logit-transformed, ρ = sigmoid(ρ̃))
// - σ̃:  vol-of-vol (log-transformed, σ = exp(σ̃))
//
// Memory Layout: Structure of Arrays (SoA) for coalescing
//   d_h[N], d_mu_tilde[N], d_rho_tilde[N], d_sigma_tilde[N]
//
// KEY DESIGN DECISIONS (based on supervisor feedback):
//
// 1. PARAMETER DIFFUSION: Small random walk on θ̃ makes filtering well-posed.
//    Without it, we're doing "variational learning with regularization," not
//    true Bayesian filtering. Diffusion prevents corner-locking and collapse.
//
// 2. DIVERSITY DIAGNOSTICS: Track std(θ̃) in unconstrained space. If collapse
//    detected, it's a structural problem (bandwidth/diffusion), not heuristics.
//
// 3. DIAGONAL KERNEL: Per-dimension bandwidths are necessary but may not be
//    sufficient. Fisher-information scaling is the principled upgrade if needed.
//
// 4. TIMESCALE SEPARATION: Parameters learn slower than state. Enforced via
//    separate learning rates (step_h >> step_θ) and small diffusion rates.
//
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// TRANSFORMS
// -----------------------------------------------------------------------------

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float logit(float p) {
    return logf(p / (1.0f - p + 1e-8f));
}

// Derivative of sigmoid: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
__device__ __forceinline__ float sigmoid_deriv(float rho) {
    return rho * (1.0f - rho);  // rho is already sigmoid(rho_tilde)
}

// -----------------------------------------------------------------------------
// KERNEL 1: JOINT PREDICT
// -----------------------------------------------------------------------------
// Propagates h using particle-local parameters
// Also applies small diffusion to parameters (random walk for well-posed filtering)

__global__ void svpf_joint_predict_kernel(
    float* __restrict__ d_h,
    float* __restrict__ d_h_prev,
    float* __restrict__ d_mu_tilde,
    float* __restrict__ d_rho_tilde,
    float* __restrict__ d_sigma_tilde,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    // Parameter diffusion rates (small random walk)
    float diffusion_mu,
    float diffusion_rho,
    float diffusion_sigma,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load current state
    float h_i = d_h[i];
    
    // Store previous for gradient computation
    d_h_prev[i] = h_i;
    
    // =========================================================================
    // PARAMETER DIFFUSION (small random walk for well-posed filtering)
    // =========================================================================
    // This makes p(θ_t | θ_{t-1}) explicit, preventing:
    // - Parameters getting stuck in corners
    // - Diversity collapse
    // - Brittle behavior from extreme observations
    
    float noise_mu = curand_normal(&rng[i]);
    float noise_rho = curand_normal(&rng[i]);
    float noise_sigma = curand_normal(&rng[i]);
    
    d_mu_tilde[i] += diffusion_mu * noise_mu;
    d_rho_tilde[i] += diffusion_rho * noise_rho;
    d_sigma_tilde[i] += diffusion_sigma * noise_sigma;
    
    // =========================================================================
    // STATE PROPAGATION with particle-local parameters
    // =========================================================================
    
    // Transform parameters from unconstrained space
    float mu = d_mu_tilde[i];                    // μ̃ = μ (identity)
    float rho = sigmoid(d_rho_tilde[i]);         // ρ = sigmoid(ρ̃)
    float sigma = expf(d_sigma_tilde[i]);        // σ = exp(σ̃)
    
    // Clamp sigma for numerical stability
    sigma = fmaxf(sigma, 0.01f);
    sigma = fminf(sigma, 2.0f);
    
    // AR(1) prediction with particle-local params
    float noise = curand_normal(&rng[i]);
    float h_pred = mu + rho * (h_i - mu);
    
    d_h[i] = h_pred + sigma * noise;
}

// -----------------------------------------------------------------------------
// KERNEL 2: JOINT GRADIENT
// -----------------------------------------------------------------------------
// Computes gradients for all 4 dimensions: [∇h, ∇μ̃, ∇ρ̃, ∇σ̃]
//
// Log-posterior = log p(y|h) + log p(h|h_prev, θ) + log p(θ)
//
// Likelihood term (only affects h):
//   ∇_h log p(y|h) = (log(y²) - h + offset) / R   [or exact Student-t]
//
// Transition term (affects all):
//   p(h|h_prev, θ) = N(h; μ + ρ(h_prev - μ), σ²)
//   
//   ∇_h    = -diff / σ²
//   ∇_μ    = (diff / σ²) * (1 - ρ)
//   ∇_ρ̃   = (diff / σ²) * (h_prev - μ) * sigmoid'(ρ̃)
//   ∇_σ̃   = z² - 1    (where z = diff/σ)
//
// Prior term (regularization on θ):
//   Optional: gentle pull toward prior means

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
    float nu,                    // Student-t degrees of freedom
    float student_t_const,
    // Prior hyperparameters (optional regularization)
    float mu_prior_mean,
    float mu_prior_var,
    float rho_prior_mean,        // In unconstrained space
    float rho_prior_var,
    float sigma_prior_mean,      // In log space
    float sigma_prior_var,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    // Load state
    float h_j = d_h[j];
    float h_prev_j = d_h_prev[j];
    
    // Transform parameters
    float mu = d_mu_tilde[j];
    float rho_tilde = d_rho_tilde[j];
    float sigma_tilde = d_sigma_tilde[j];
    
    float rho = sigmoid(rho_tilde);
    float sigma = expf(sigma_tilde);
    sigma = fmaxf(sigma, 0.01f);
    
    float var = sigma * sigma;
    
    // =========================================================================
    // LIKELIHOOD GRADIENT (affects h only)
    // =========================================================================
    float y_sq = y_t * y_t;
    float vol = expf(h_j);
    float scaled_y_sq = y_sq / (vol + 1e-8f);
    float A = scaled_y_sq / nu;
    float one_plus_A = 1.0f + A;
    
    // Log-weight (for ESS / diagnostics)
    d_log_w[j] = student_t_const - 0.5f * h_j 
               - (nu + 1.0f) * 0.5f * log1pf(fmaxf(A, -0.999f));
    
    // Exact Student-t gradient with bias correction
    float lik_offset = 0.30f;  // Tuned bias correction
    float grad_h_lik = -0.5f + 0.5f * (nu + 1.0f) * A / one_plus_A - lik_offset;
    
    // =========================================================================
    // TRANSITION GRADIENT (affects all)
    // =========================================================================
    // p(h | h_prev, θ) = N(h; μ + ρ(h_prev - μ), σ²)
    
    float h_pred = mu + rho * (h_prev_j - mu);
    float diff = h_j - h_pred;
    float z_sq = (diff * diff) / var;
    
    // ∇_h log p(h|h_prev,θ) = -diff / σ²
    float grad_h_trans = -diff / var;
    
    // ∇_μ log p(h|h_prev,θ) = (diff / σ²) * (1 - ρ)
    // Intuition: if diff > 0, h is above prediction, increase μ to raise prediction
    float grad_mu_trans = (diff / var) * (1.0f - rho);
    
    // ∇_ρ̃ log p(h|h_prev,θ) = (diff / σ²) * (h_prev - μ) * sigmoid'(ρ̃)
    // Chain rule: dρ/dρ̃ = ρ(1-ρ)
    // Intuition: if diff > 0 and h_prev > μ, increase ρ to pull prediction up
    float grad_rho_trans = (diff / var) * (h_prev_j - mu) * sigmoid_deriv(rho);
    
    // ∇_σ̃ log p(h|h_prev,θ) = z² - 1
    // Chain rule: dσ/dσ̃ = σ, but this cancels in the derivation
    // Intuition: z² > 1 means σ is too small (can't explain this jump)
    float grad_sigma_trans = z_sq - 1.0f;
    
    // =========================================================================
    // PRIOR GRADIENT (optional regularization)
    // =========================================================================
    // Gentle pull toward prior means, prevents runaway parameters
    
    float grad_mu_prior = -(d_mu_tilde[j] - mu_prior_mean) / mu_prior_var;
    float grad_rho_prior = -(rho_tilde - rho_prior_mean) / rho_prior_var;
    float grad_sigma_prior = -(sigma_tilde - sigma_prior_mean) / sigma_prior_var;
    
    // =========================================================================
    // COMBINE GRADIENTS
    // =========================================================================
    
    // h: likelihood + transition (both are important)
    float grad_h = grad_h_lik + grad_h_trans;
    
    // Parameters: transition + weak prior
    float prior_weight = 0.01f;  // Weak regularization
    float grad_mu = grad_mu_trans + prior_weight * grad_mu_prior;
    float grad_rho = grad_rho_trans + prior_weight * grad_rho_prior;
    float grad_sigma = grad_sigma_trans + prior_weight * grad_sigma_prior;
    
    // Gradient clipping for stability
    grad_h = fmaxf(fminf(grad_h, 10.0f), -10.0f);
    grad_mu = fmaxf(fminf(grad_mu, 5.0f), -5.0f);
    grad_rho = fmaxf(fminf(grad_rho, 5.0f), -5.0f);
    grad_sigma = fmaxf(fminf(grad_sigma, 5.0f), -5.0f);
    
    // Store
    d_grad_h[j] = grad_h;
    d_grad_mu[j] = grad_mu;
    d_grad_rho[j] = grad_rho;
    d_grad_sigma[j] = grad_sigma;
}

// -----------------------------------------------------------------------------
// KERNEL 3: PER-DIMENSION BANDWIDTH (Median Heuristic)
// -----------------------------------------------------------------------------
// Computes bandwidth for each dimension separately
// This is called 4 times (once per dimension) or fused into one kernel

__global__ void svpf_compute_bandwidth_kernel(
    const float* __restrict__ d_values,  // One dimension: d_h, d_mu_tilde, etc.
    float* __restrict__ d_diffs,         // Scratch: N*(N-1)/2 pairwise differences
    int n
) {
    // Compute all pairwise |x_i - x_j| for i < j
    // Then find median on CPU or via parallel selection
    // 
    // For simplicity, can use median of |x_i - x_{i+1}| for sorted particles
    // Or subsample for large N
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n * (n - 1) / 2;
    if (idx >= total_pairs) return;
    
    // Map linear index to (i, j) pair
    // Using triangular indexing
    int i = (int)(sqrtf(2.0f * idx + 0.25f) - 0.5f);
    int j = idx - i * (i + 1) / 2;
    if (j >= i) { i++; j = idx - i * (i + 1) / 2; }
    j = i + 1 + (idx - i * (i + 1) / 2);  // Simplified: just iterate
    
    // Actually, simpler approach for sketch:
    // Each thread handles one (i, j) pair
    i = idx / n;
    j = idx % n;
    if (i >= j) return;
    
    d_diffs[idx] = fabsf(d_values[i] - d_values[j]);
}

// -----------------------------------------------------------------------------
// KERNEL 4: JOINT STEIN TRANSPORT
// -----------------------------------------------------------------------------
// Applies Stein variational update with DIAGONAL kernel
// 
// φ(x_j) = (1/N) Σ_i [ k(x_i, x_j) * ∇_{x_i} log p + ∇_{x_i} k(x_i, x_j) ]
//
// Where x = [h, μ̃, ρ̃, σ̃] and kernel is:
//   k(x, x') = exp( -(h-h')²/bw_h² - (μ̃-μ̃')²/bw_μ² - (ρ̃-ρ̃')²/bw_ρ² - (σ̃-σ̃')²/bw_σ² )

__global__ void svpf_joint_stein_transport_kernel(
    float* __restrict__ d_h,
    float* __restrict__ d_mu_tilde,
    float* __restrict__ d_rho_tilde,
    float* __restrict__ d_sigma_tilde,
    const float* __restrict__ d_grad_h,
    const float* __restrict__ d_grad_mu,
    const float* __restrict__ d_grad_rho,
    const float* __restrict__ d_grad_sigma,
    float bw_h,
    float bw_mu,
    float bw_rho,
    float bw_sigma,
    float step_h,           // Learning rate for h (fast)
    float step_mu,          // Learning rate for μ (slow)
    float step_rho,         // Learning rate for ρ (slow)
    float step_sigma,       // Learning rate for σ (slow)
    int n
) {
    extern __shared__ float smem[];
    
    // Load all particles into shared memory
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
    
    // Precompute inverse squared bandwidths
    float inv_bw_h_sq = 1.0f / (bw_h * bw_h + 1e-8f);
    float inv_bw_mu_sq = 1.0f / (bw_mu * bw_mu + 1e-8f);
    float inv_bw_rho_sq = 1.0f / (bw_rho * bw_rho + 1e-8f);
    float inv_bw_sigma_sq = 1.0f / (bw_sigma * bw_sigma + 1e-8f);
    
    // Accumulate Stein update: φ = Σ_i [k(x_i, x_j) * ∇_i + ∇_i k]
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
        
        // Diagonal kernel: k(x_i, x_j)
        float dist_sq = dh * dh * inv_bw_h_sq
                      + dmu * dmu * inv_bw_mu_sq
                      + drho * drho * inv_bw_rho_sq
                      + dsigma * dsigma * inv_bw_sigma_sq;
        float k_ij = expf(-dist_sq);
        
        // Kernel gradient: ∇_{x_i} k(x_i, x_j)
        // ∂k/∂h_i = k * 2 * (h_i - h_j) / bw_h²
        float dk_dh = k_ij * 2.0f * dh * inv_bw_h_sq;
        float dk_dmu = k_ij * 2.0f * dmu * inv_bw_mu_sq;
        float dk_drho = k_ij * 2.0f * drho * inv_bw_rho_sq;
        float dk_dsigma = k_ij * 2.0f * dsigma * inv_bw_sigma_sq;
        
        // Stein update: k * grad + grad_k (repulsive term)
        phi_h += k_ij * sh_grad_h[i] + dk_dh;
        phi_mu += k_ij * sh_grad_mu[i] + dk_dmu;
        phi_rho += k_ij * sh_grad_rho[i] + dk_drho;
        phi_sigma += k_ij * sh_grad_sigma[i] + dk_dsigma;
    }
    
    // Normalize by N
    float inv_n = 1.0f / (float)n;
    phi_h *= inv_n;
    phi_mu *= inv_n;
    phi_rho *= inv_n;
    phi_sigma *= inv_n;
    
    // Apply update with per-parameter learning rates
    d_h[j] = h_j + step_h * phi_h;
    d_mu_tilde[j] = mu_j + step_mu * phi_mu;
    d_rho_tilde[j] = rho_j + step_rho * phi_rho;
    d_sigma_tilde[j] = sigma_j + step_sigma * phi_sigma;
    
    // Clamp parameters to reasonable ranges (in unconstrained space)
    // μ̃ ∈ [-10, 5] (corresponds to vol 2% to 500%)
    d_mu_tilde[j] = fmaxf(fminf(d_mu_tilde[j], 5.0f), -10.0f);
    
    // ρ̃ ∈ [-3, 4] (corresponds to ρ ∈ [0.05, 0.98])
    d_rho_tilde[j] = fmaxf(fminf(d_rho_tilde[j], 4.0f), -3.0f);
    
    // σ̃ ∈ [-4, 1] (corresponds to σ ∈ [0.018, 2.7])
    d_sigma_tilde[j] = fmaxf(fminf(d_sigma_tilde[j], 1.0f), -4.0f);
}

// -----------------------------------------------------------------------------
// KERNEL 5: EXTRACT PARAMETER ESTIMATES
// -----------------------------------------------------------------------------
// Computes mean and std of parameters across particles

__global__ void svpf_joint_extract_params_kernel(
    const float* __restrict__ d_mu_tilde,
    const float* __restrict__ d_rho_tilde,
    const float* __restrict__ d_sigma_tilde,
    float* __restrict__ d_param_mean,   // [μ_mean, ρ_mean, σ_mean]
    float* __restrict__ d_param_std,    // [μ_std, ρ_std, σ_std]
    int n
) {
    // Simple reduction - in practice use CUB
    __shared__ float s_mu_sum, s_rho_sum, s_sigma_sum;
    __shared__ float s_mu_sq_sum, s_rho_sq_sum, s_sigma_sq_sum;
    
    if (threadIdx.x == 0) {
        s_mu_sum = s_rho_sum = s_sigma_sum = 0.0f;
        s_mu_sq_sum = s_rho_sq_sum = s_sigma_sq_sum = 0.0f;
    }
    __syncthreads();
    
    // Each thread accumulates a chunk
    float mu_sum = 0.0f, rho_sum = 0.0f, sigma_sum = 0.0f;
    float mu_sq = 0.0f, rho_sq = 0.0f, sigma_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // Transform to constrained space
        float mu = d_mu_tilde[i];
        float rho = sigmoid(d_rho_tilde[i]);
        float sigma = expf(d_sigma_tilde[i]);
        
        mu_sum += mu;
        rho_sum += rho;
        sigma_sum += sigma;
        
        mu_sq += mu * mu;
        rho_sq += rho * rho;
        sigma_sq += sigma * sigma;
    }
    
    // Atomic add to shared
    atomicAdd(&s_mu_sum, mu_sum);
    atomicAdd(&s_rho_sum, rho_sum);
    atomicAdd(&s_sigma_sum, sigma_sum);
    atomicAdd(&s_mu_sq_sum, mu_sq);
    atomicAdd(&s_rho_sq_sum, rho_sq);
    atomicAdd(&s_sigma_sq_sum, sigma_sq);
    __syncthreads();
    
    // Thread 0 computes final statistics
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        
        d_param_mean[0] = s_mu_sum * inv_n;
        d_param_mean[1] = s_rho_sum * inv_n;
        d_param_mean[2] = s_sigma_sum * inv_n;
        
        d_param_std[0] = sqrtf(s_mu_sq_sum * inv_n - d_param_mean[0] * d_param_mean[0]);
        d_param_std[1] = sqrtf(s_rho_sq_sum * inv_n - d_param_mean[1] * d_param_mean[1]);
        d_param_std[2] = sqrtf(s_sigma_sq_sum * inv_n - d_param_mean[2] * d_param_mean[2]);
    }
}

// -----------------------------------------------------------------------------
// KERNEL 6: DIVERSITY COLLAPSE DIAGNOSTIC
// -----------------------------------------------------------------------------
// Computes std in UNCONSTRAINED space (more sensitive to collapse)
// Returns flags indicating which parameters have collapsed

__global__ void svpf_joint_diversity_diagnostic_kernel(
    const float* __restrict__ d_mu_tilde,
    const float* __restrict__ d_rho_tilde,
    const float* __restrict__ d_sigma_tilde,
    float* __restrict__ d_std_unconstrained,  // [std_mu_tilde, std_rho_tilde, std_sigma_tilde]
    int* __restrict__ d_collapse_flags,       // [mu_collapsed, rho_collapsed, sigma_collapsed]
    float collapse_thresh_mu,     // e.g., 0.05
    float collapse_thresh_rho,    // e.g., 0.02
    float collapse_thresh_sigma,  // e.g., 0.02
    int n
) {
    __shared__ float s_mu_sum, s_rho_sum, s_sigma_sum;
    __shared__ float s_mu_sq, s_rho_sq, s_sigma_sq;
    
    if (threadIdx.x == 0) {
        s_mu_sum = s_rho_sum = s_sigma_sum = 0.0f;
        s_mu_sq = s_rho_sq = s_sigma_sq = 0.0f;
    }
    __syncthreads();
    
    float mu_sum = 0.0f, rho_sum = 0.0f, sigma_sum = 0.0f;
    float mu_sq = 0.0f, rho_sq = 0.0f, sigma_sq = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float mu = d_mu_tilde[i];
        float rho = d_rho_tilde[i];
        float sigma = d_sigma_tilde[i];
        
        mu_sum += mu;
        rho_sum += rho;
        sigma_sum += sigma;
        
        mu_sq += mu * mu;
        rho_sq += rho * rho;
        sigma_sq += sigma * sigma;
    }
    
    atomicAdd(&s_mu_sum, mu_sum);
    atomicAdd(&s_rho_sum, rho_sum);
    atomicAdd(&s_sigma_sum, sigma_sum);
    atomicAdd(&s_mu_sq, mu_sq);
    atomicAdd(&s_rho_sq, rho_sq);
    atomicAdd(&s_sigma_sq, sigma_sq);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        
        float mean_mu = s_mu_sum * inv_n;
        float mean_rho = s_rho_sum * inv_n;
        float mean_sigma = s_sigma_sum * inv_n;
        
        float std_mu = sqrtf(fmaxf(s_mu_sq * inv_n - mean_mu * mean_mu, 0.0f));
        float std_rho = sqrtf(fmaxf(s_rho_sq * inv_n - mean_rho * mean_rho, 0.0f));
        float std_sigma = sqrtf(fmaxf(s_sigma_sq * inv_n - mean_sigma * mean_sigma, 0.0f));
        
        d_std_unconstrained[0] = std_mu;
        d_std_unconstrained[1] = std_rho;
        d_std_unconstrained[2] = std_sigma;
        
        // Check for collapse
        d_collapse_flags[0] = (std_mu < collapse_thresh_mu) ? 1 : 0;
        d_collapse_flags[1] = (std_rho < collapse_thresh_rho) ? 1 : 0;
        d_collapse_flags[2] = (std_sigma < collapse_thresh_sigma) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// HOST-SIDE: ONE TIMESTEP
// -----------------------------------------------------------------------------

/*
struct SVPFJointConfig {
    // Particle count
    int n_particles;
    int n_stein_steps;
    
    // Learning rates (per-parameter)
    float step_h;       // Fast (0.10)
    float step_mu;      // Slow (0.01)
    float step_rho;     // Very slow (0.005)
    float step_sigma;   // Slow (0.01)
    
    // Parameter diffusion rates (small random walk)
    float diffusion_mu;     // 0.01
    float diffusion_rho;    // 0.001
    float diffusion_sigma;  // 0.005
    
    // Diversity collapse thresholds (unconstrained space)
    float collapse_thresh_mu;     // 0.05
    float collapse_thresh_rho;    // 0.02
    float collapse_thresh_sigma;  // 0.02
    
    // Prior hyperparameters (weak regularization)
    float mu_prior_mean, mu_prior_var;
    float rho_prior_mean, rho_prior_var;
    float sigma_prior_mean, sigma_prior_var;
    
    // Student-t
    float nu;
    float student_t_const;
};

struct SVPFJointDiagnostics {
    // Parameter estimates (constrained space)
    float mu_mean, rho_mean, sigma_mean;
    float mu_std, rho_std, sigma_std;
    
    // Diversity (unconstrained space)
    float std_mu_tilde, std_rho_tilde, std_sigma_tilde;
    
    // Collapse flags
    int mu_collapsed, rho_collapsed, sigma_collapsed;
    
    // State estimate
    float h_mean, vol_mean;
};

void svpf_joint_step(SVPFJointState* state, float y_t, SVPFJointDiagnostics* diag) {
    int n = state->n_particles;
    int block = 256;
    int grid = (n + block - 1) / block;
    SVPFJointConfig* cfg = &state->config;
    
    // 1. Predict (propagate h with particle-local params + parameter diffusion)
    svpf_joint_predict_kernel<<<grid, block>>>(
        state->d_h, state->d_h_prev,
        state->d_mu_tilde, state->d_rho_tilde, state->d_sigma_tilde,
        state->rng_states,
        cfg->diffusion_mu, cfg->diffusion_rho, cfg->diffusion_sigma,
        n
    );
    
    // 2. Compute bandwidths (one per dimension)
    // Simplified: use median heuristic on each array
    state->bw_h = compute_median_bandwidth(state->d_h, n);
    state->bw_mu = compute_median_bandwidth(state->d_mu_tilde, n);
    state->bw_rho = compute_median_bandwidth(state->d_rho_tilde, n);
    state->bw_sigma = compute_median_bandwidth(state->d_sigma_tilde, n);
    
    // 3. Stein iterations
    for (int s = 0; s < cfg->n_stein_steps; s++) {
        // Compute gradients
        svpf_joint_gradient_kernel<<<grid, block>>>(
            state->d_h, state->d_h_prev,
            state->d_mu_tilde, state->d_rho_tilde, state->d_sigma_tilde,
            state->d_grad_h, state->d_grad_mu, state->d_grad_rho, state->d_grad_sigma,
            state->d_log_w,
            y_t, cfg->nu, cfg->student_t_const,
            cfg->mu_prior_mean, cfg->mu_prior_var,
            cfg->rho_prior_mean, cfg->rho_prior_var,
            cfg->sigma_prior_mean, cfg->sigma_prior_var,
            n
        );
        
        // Stein transport
        size_t transport_smem = 8 * n * sizeof(float);
        svpf_joint_stein_transport_kernel<<<grid, block, transport_smem>>>(
            state->d_h,
            state->d_mu_tilde, state->d_rho_tilde, state->d_sigma_tilde,
            state->d_grad_h, state->d_grad_mu, state->d_grad_rho, state->d_grad_sigma,
            state->bw_h, state->bw_mu, state->bw_rho, state->bw_sigma,
            cfg->step_h, cfg->step_mu, cfg->step_rho, cfg->step_sigma,
            n
        );
    }
    
    // 4. Extract parameter estimates
    svpf_joint_extract_params_kernel<<<1, 256>>>(
        state->d_mu_tilde, state->d_rho_tilde, state->d_sigma_tilde,
        state->d_param_mean, state->d_param_std, n
    );
    
    // 5. Diversity diagnostic (optional, can run every N steps)
    svpf_joint_diversity_diagnostic_kernel<<<1, 256>>>(
        state->d_mu_tilde, state->d_rho_tilde, state->d_sigma_tilde,
        state->d_std_unconstrained, state->d_collapse_flags,
        cfg->collapse_thresh_mu, cfg->collapse_thresh_rho, cfg->collapse_thresh_sigma,
        n
    );
    
    // 6. Copy diagnostics to host (if requested)
    if (diag != nullptr) {
        cudaMemcpy(&diag->mu_mean, state->d_param_mean, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&diag->mu_std, state->d_param_std, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&diag->std_mu_tilde, state->d_std_unconstrained, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&diag->mu_collapsed, state->d_collapse_flags, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Compute h_mean and vol_mean
        diag->h_mean = compute_mean(state->d_h, n);
        diag->vol_mean = expf(diag->h_mean * 0.5f);
        
        // Warning if collapsed
        if (diag->mu_collapsed || diag->rho_collapsed || diag->sigma_collapsed) {
            printf("WARNING: Parameter diversity collapsed! mu:%d rho:%d sigma:%d\n",
                   diag->mu_collapsed, diag->rho_collapsed, diag->sigma_collapsed);
        }
    }
}

// Default configuration
SVPFJointConfig svpf_joint_default_config() {
    SVPFJointConfig cfg;
    
    cfg.n_particles = 512;
    cfg.n_stein_steps = 5;
    
    // Learning rates
    cfg.step_h = 0.10f;
    cfg.step_mu = 0.01f;
    cfg.step_rho = 0.005f;
    cfg.step_sigma = 0.01f;
    
    // Parameter diffusion
    cfg.diffusion_mu = 0.01f;
    cfg.diffusion_rho = 0.001f;
    cfg.diffusion_sigma = 0.005f;
    
    // Collapse thresholds
    cfg.collapse_thresh_mu = 0.05f;
    cfg.collapse_thresh_rho = 0.02f;
    cfg.collapse_thresh_sigma = 0.02f;
    
    // Weak priors (regularization)
    cfg.mu_prior_mean = -3.5f;
    cfg.mu_prior_var = 10.0f;
    cfg.rho_prior_mean = 2.0f;   // sigmoid(2.0) ≈ 0.88
    cfg.rho_prior_var = 5.0f;
    cfg.sigma_prior_mean = -2.0f; // exp(-2.0) ≈ 0.14
    cfg.sigma_prior_var = 5.0f;
    
    // Student-t
    cfg.nu = 30.0f;
    cfg.student_t_const = lgammaf(15.5f) - lgammaf(15.0f) - 0.5f * logf(M_PI * 30.0f);
    
    return cfg;
}
*/
