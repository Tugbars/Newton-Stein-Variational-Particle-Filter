/**
 * @file svpf_2d_kernels.cu
 * @brief 2D SVPF kernels for online sigma_z learning
 * 
 * State: (h, λ) where λ = log(σ_z)
 * 
 * Key insight: σ_z varies with market regime. Learning it online
 * allows the filter to adapt to changing volatility-of-volatility.
 */

#include "svpf_kernels.cuh"
#include <stdio.h>

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float clamp_lambda(float lam) {
    // lambda = log(sigma_z), clamp sigma_z to [0.01, 1.0]
    // log(0.01) ≈ -4.6, log(1.0) = 0
    return fminf(fmaxf(lam, -4.6f), 0.0f);
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

__device__ float block_reduce_sum_2d(float val) {
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

__device__ float block_reduce_max_2d(float val) {
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

__device__ float block_reduce_min_2d(float val) {
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

// =============================================================================
// 2D PREDICT: State transition with lambda jittering
// =============================================================================

__global__ void svpf_predict_2d_kernel(
    float* __restrict__ h,
    float* __restrict__ h_prev,
    float* __restrict__ lambda,
    float* __restrict__ lambda_prev,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_y,
    const float* __restrict__ d_h_mean,
    int t,
    float rho, float mu, float gamma,
    float lambda_jitter,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Save previous state
    float h_i = h[i];
    float lam_i = lambda[i];
    h_prev[i] = h_i;
    lambda_prev[i] = lam_i;
    
    // Get two independent normals
    float2 noise = curand_normal2(&rng[i]);
    
    // Jitter lambda (random walk on log-space)
    float lam_new = lam_i + lambda_jitter * noise.y;
    lam_new = clamp_lambda(lam_new);
    lambda[i] = lam_new;
    
    // Transition h using particle-specific sigma
    float sigma_i = __expf(lam_new);
    float y_prev = (t > 0) ? d_y[t - 1] : 0.0f;
    float vol_prev = __expf(h_i * 0.5f);
    float leverage = gamma * y_prev / (vol_prev + 1e-8f);
    
    float h_new = mu + rho * (h_i - mu) + sigma_i * noise.x + leverage;
    h[i] = clamp_logvol(h_new);
}

// =============================================================================
// 2D FUSED GRADIENT: Computes gradients for h and lambda
// =============================================================================

__global__ void svpf_fused_gradient_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    const float* __restrict__ lambda,
    float* __restrict__ grad_h,
    float* __restrict__ grad_lambda,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    int y_idx,
    float rho,
    float mu,
    float beta,
    float nu,
    float student_t_const,
    float lambda_prior_mean,
    float lambda_prior_std,
    int n
) {
    extern __shared__ float smem[];
    // Layout: h_prev[n], lambda[n], mu_trans[n], sigma_sq[n]
    float* sh_h_prev = smem;
    float* sh_lambda = smem + n;
    float* sh_mu_trans = smem + 2 * n;
    float* sh_sigma_sq = smem + 3 * n;
    
    // Cooperative load
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float hp = h_prev[k];
        float lam = lambda[k];
        sh_h_prev[k] = hp;
        sh_lambda[k] = lam;
        sh_mu_trans[k] = mu + rho * (hp - mu);
        float sigma = __expf(lam);
        sh_sigma_sq[k] = sigma * sigma;
    }
    __syncthreads();
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float lam_j = lambda[j];
    float sigma_j = __expf(lam_j);
    float sigma_sq_j = sigma_j * sigma_j;
    float y_t = d_y[y_idx];
    
    // =========================================================================
    // MIXTURE PRIOR GRADIENT (O(N²))
    // p(h_j | Z_{t-1}) = (1/N) Σ_i N(h_j; mu_i, sigma_i²)
    // =========================================================================
    
    // Pass 1: Find max log-weight for numerical stability
    float log_r_max = -1e10f;
    for (int i = 0; i < n; i++) {
        float diff = h_j - sh_mu_trans[i];
        float var_i = sh_sigma_sq[i];
        float log_r_i = -0.5f * diff * diff / var_i - 0.5f * __logf(var_i);
        log_r_max = fmaxf(log_r_max, log_r_i);
    }
    
    // Pass 2: Compute weighted gradients
    float sum_r = 0.0f;
    float gh_prior = 0.0f;
    float glam_prior = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float diff = h_j - sh_mu_trans[i];
        float var_i = sh_sigma_sq[i];
        float log_r_i = -0.5f * diff * diff / var_i - 0.5f * __logf(var_i);
        float r_i = __expf(log_r_i - log_r_max);
        
        sum_r += r_i;
        
        // d/dh: -(h - mu_i) / sigma_i²
        gh_prior += r_i * (-diff / var_i);
        
        // d/dlambda for component i (chain rule through sigma = exp(lambda))
        // d log N / d sigma = (diff² / sigma³) - (1 / sigma)
        // d sigma / d lambda = sigma
        // Combined: diff² / sigma² - 1
        // But this only applies when particle j uses component i's sigma...
        // Actually, for the MIXTURE, each particle j has its OWN lambda_j.
        // The mixture components use sigma from particles i.
        // Gradient w.r.t. lambda_j comes from the transition density p(h_j | h_{j,prev}, lambda_j)
    }
    
    float inv_sum_r = 1.0f / (sum_r + 1e-8f);
    gh_prior *= inv_sum_r;
    
    // =========================================================================
    // GRADIENT w.r.t. lambda_j from OWN transition density
    // p(h_j | h_j_prev, lambda_j) = N(h_j; mu_j, sigma_j²)
    // d log p / d lambda = (h_j - mu_j)² / sigma_j² - 1
    // =========================================================================
    
    float mu_j = mu + rho * (h_prev[j] - mu);
    float diff_j = h_j - mu_j;
    float innovation_sq = diff_j * diff_j;
    
    // Gradient from transition density
    glam_prior = (innovation_sq / sigma_sq_j) - 1.0f;
    
    // Add prior on lambda itself: p(lambda) ~ N(lambda_prior_mean, lambda_prior_std²)
    float lambda_prior_var = lambda_prior_std * lambda_prior_std;
    float glam_prior_reg = -(lam_j - lambda_prior_mean) / lambda_prior_var;
    glam_prior += glam_prior_reg;
    
    // =========================================================================
    // LIKELIHOOD GRADIENT (only affects h, not lambda)
    // =========================================================================
    
    float vol = __expf(h_j);
    float y_sq = y_t * y_t;
    float scaled_y_sq = y_sq / (vol + 1e-8f);
    
    // Log-weight for ESS
    log_w[j] = student_t_const - 0.5f * h_j
             - (nu + 1.0f) * 0.5f * log1pf(fmaxf(scaled_y_sq / nu, -0.999f));
    
    // Log-squared likelihood gradient (robust)
    float log_y2 = __logf(y_sq + 1e-10f);
    float R_noise = 1.4f;
    float gh_lik = (log_y2 - h_j + 1.0f / nu) / R_noise;
    
    // =========================================================================
    // COMBINE AND CLAMP
    // =========================================================================
    
    float g_h = gh_prior + beta * gh_lik;
    g_h = fminf(fmaxf(g_h, -10.0f), 10.0f);
    
    float g_lam = glam_prior;  // No likelihood contribution to lambda
    g_lam = fminf(fmaxf(g_lam, -5.0f), 5.0f);  // Tighter clamp for stability
    
    grad_h[j] = g_h;
    grad_lambda[j] = g_lam;
}

// =============================================================================
// 2D FUSED STEIN + TRANSPORT
// Product kernel: K(x_i, x_j) = K_h(h_i, h_j) * K_λ(λ_i, λ_j)
// =============================================================================

__global__ void svpf_fused_stein_transport_2d_kernel(
    float* __restrict__ h,
    float* __restrict__ lambda,
    const float* __restrict__ grad_h,
    const float* __restrict__ grad_lambda,
    float* __restrict__ v_h,
    float* __restrict__ v_lambda,
    curandStatePhilox4_32_10_t* __restrict__ rng,
    const float* __restrict__ d_bw_h,
    const float* __restrict__ d_bw_lambda,
    float step_size,
    float beta_factor,
    float temperature,
    float rho_rmsprop,
    float epsilon,
    int n
) {
    extern __shared__ float smem[];
    // Layout: h[n], lambda[n], grad_h[n], grad_lambda[n]
    float* sh_h = smem;
    float* sh_lam = smem + n;
    float* sh_gh = smem + 2 * n;
    float* sh_glam = smem + 3 * n;
    
    // Cooperative load
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        sh_h[k] = h[k];
        sh_lam[k] = lambda[k];
        sh_gh[k] = grad_h[k];
        sh_glam[k] = grad_lambda[k];
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float h_i = sh_h[i];
    float lam_i = sh_lam[i];
    
    float bw_h = *d_bw_h;
    float bw_lam = *d_bw_lambda;
    
    float inv_bw_h_sq = 1.0f / (bw_h * bw_h);
    float inv_bw_lam_sq = 1.0f / (bw_lam * bw_lam);
    float inv_2bw_h_sq = 0.5f * inv_bw_h_sq;
    float inv_2bw_lam_sq = 0.5f * inv_bw_lam_sq;
    float inv_n = 1.0f / (float)n;
    
    // =========================================================================
    // STEIN OPERATOR with product kernel
    // φ(x_i) = (1/N) Σ_j [K(x_i, x_j) * ∇log p(x_j) + ∇_{x_j} K(x_i, x_j)]
    // =========================================================================
    
    float phi_h = 0.0f;
    float phi_lam = 0.0f;
    
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        float diff_h = h_i - sh_h[j];
        float diff_lam = lam_i - sh_lam[j];
        
        // Product kernel - compute combined exponent for precision
        float dist_h = diff_h * diff_h * inv_2bw_h_sq;
        float dist_lam = diff_lam * diff_lam * inv_2bw_lam_sq;
        float K = expf(-(dist_h + dist_lam));  // Standard exp for precision
        
        // Stein for h: K * grad_h + grad_K (repulsive: pushes i away from j)
        // grad_K w.r.t. x_j acting on x_i is +diff/bw² (positive = repulsion)
        phi_h += K * sh_gh[j] + K * diff_h * inv_bw_h_sq;
        
        // Stein for lambda: K * grad_lambda + grad_K
        phi_lam += K * sh_glam[j] + K * diff_lam * inv_bw_lam_sq;
    }
    
    phi_h *= inv_n;
    phi_lam *= inv_n;
    
    // =========================================================================
    // RMSPROP + TRANSPORT
    // =========================================================================
    
    float v_h_prev = v_h[i];
    float v_lam_prev = v_lambda[i];
    
    float v_h_new = rho_rmsprop * v_h_prev + (1.0f - rho_rmsprop) * phi_h * phi_h;
    float v_lam_new = rho_rmsprop * v_lam_prev + (1.0f - rho_rmsprop) * phi_lam * phi_lam;
    
    v_h[i] = v_h_new;
    v_lambda[i] = v_lam_new;
    
    float effective_step = step_size * beta_factor;
    
    float precond_h = rsqrtf(v_h_new + epsilon);
    float precond_lam = rsqrtf(v_lam_new + epsilon);
    
    float drift_h = effective_step * phi_h * precond_h;
    float drift_lam = effective_step * phi_lam * precond_lam * 0.5f;  // Slower learning rate for params
    
    // SVLD diffusion
    float diff_h_noise = 0.0f;
    float diff_lam_noise = 0.0f;
    if (temperature > 1e-6f) {
        float2 noise = curand_normal2(&rng[i]);
        float noise_scale = __fsqrt_rn(2.0f * effective_step * temperature);
        diff_h_noise = noise_scale * noise.x;
        diff_lam_noise = noise_scale * noise.y * 0.5f;  // Less noise for params
    }
    
    h[i] = clamp_logvol(h_i + drift_h + diff_h_noise);
    lambda[i] = clamp_lambda(lam_i + drift_lam + diff_lam_noise);
}

// =============================================================================
// 2D BANDWIDTH: Separate bandwidths for h and lambda
// =============================================================================

__global__ void svpf_fused_bandwidth_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ lambda,
    const float* __restrict__ d_y,
    float* __restrict__ d_bw_h,
    float* __restrict__ d_bw_h_sq,
    float* __restrict__ d_bw_lambda,
    float* __restrict__ d_bw_lambda_sq,
    float* __restrict__ d_return_ema,
    float* __restrict__ d_return_var,
    int y_idx,
    float alpha_bw_h,
    float alpha_bw_lambda,
    float alpha_ret,
    int n
) {
    // Compute statistics for h
    float h_sum = 0.0f, h_sum_sq = 0.0f;
    float lam_sum = 0.0f, lam_sum_sq = 0.0f;
    float h_min = 1e10f, h_max = -1e10f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float hi = h[i];
        float li = lambda[i];
        h_sum += hi;
        h_sum_sq += hi * hi;
        lam_sum += li;
        lam_sum_sq += li * li;
        h_min = fminf(h_min, hi);
        h_max = fmaxf(h_max, hi);
    }
    
    h_sum = block_reduce_sum_2d(h_sum);
    __syncthreads();
    h_sum_sq = block_reduce_sum_2d(h_sum_sq);
    __syncthreads();
    lam_sum = block_reduce_sum_2d(lam_sum);
    __syncthreads();
    lam_sum_sq = block_reduce_sum_2d(lam_sum_sq);
    __syncthreads();
    h_min = block_reduce_min_2d(h_min);
    __syncthreads();
    h_max = block_reduce_max_2d(h_max);
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        float log_n = __logf((float)n + 1.0f);
        
        // h bandwidth (Silverman + EMA + adaptive)
        float h_mean = h_sum * inv_n;
        float h_var = h_sum_sq * inv_n - h_mean * h_mean;
        float h_spread = h_max - h_min;
        
        float bw_h_sq_new = 2.0f * h_var / log_n;
        bw_h_sq_new = fmaxf(bw_h_sq_new, 1e-6f);
        
        float bw_h_sq_prev = *d_bw_h_sq;
        float bw_h_sq = (bw_h_sq_prev > 0.0f)
                      ? alpha_bw_h * bw_h_sq_new + (1.0f - alpha_bw_h) * bw_h_sq_prev
                      : bw_h_sq_new;
        
        // Adaptive scaling for h (same as 1D)
        float new_return = d_y[y_idx];
        float abs_ret = fabsf(new_return);
        float ret_ema = *d_return_ema;
        float ret_var = *d_return_var;
        
        ret_ema = (ret_ema > 0.0f) ? alpha_ret * abs_ret + (1.0f - alpha_ret) * ret_ema : abs_ret;
        ret_var = (ret_var > 0.0f) ? alpha_ret * abs_ret * abs_ret + (1.0f - alpha_ret) * ret_var : abs_ret * abs_ret;
        
        *d_return_ema = ret_ema;
        *d_return_var = ret_var;
        
        float vol_ratio = abs_ret / fmaxf(ret_ema, 1e-8f);
        float spread_factor = fminf(h_spread * 0.5f, 2.0f);
        float combined = fmaxf(vol_ratio, spread_factor);
        
        float scale = 1.0f - 0.25f * fminf(combined - 1.0f, 2.0f);
        scale = fmaxf(fminf(scale, 1.0f), 0.5f);
        
        bw_h_sq *= scale;
        float bw_h = __fsqrt_rn(bw_h_sq);
        bw_h = fmaxf(fminf(bw_h, 2.0f), 0.01f);
        
        *d_bw_h_sq = bw_h_sq;
        *d_bw_h = bw_h;
        
        // Lambda bandwidth (slower EMA, no adaptive scaling)
        float lam_mean = lam_sum * inv_n;
        float lam_var = lam_sum_sq * inv_n - lam_mean * lam_mean;
        
        float bw_lam_sq_new = 2.0f * lam_var / log_n;
        bw_lam_sq_new = fmaxf(bw_lam_sq_new, 1e-6f);
        
        float bw_lam_sq_prev = *d_bw_lambda_sq;
        float bw_lam_sq = (bw_lam_sq_prev > 0.0f)
                        ? alpha_bw_lambda * bw_lam_sq_new + (1.0f - alpha_bw_lambda) * bw_lam_sq_prev
                        : bw_lam_sq_new;
        
        float bw_lam = __fsqrt_rn(bw_lam_sq);
        bw_lam = fmaxf(fminf(bw_lam, 1.0f), 0.01f);  // Tighter bounds for lambda
        
        *d_bw_lambda_sq = bw_lam_sq;
        *d_bw_lambda = bw_lam;
    }
}

// =============================================================================
// 2D OUTPUTS: logsumexp + vol + h_mean + sigma_mean
// =============================================================================

__global__ void svpf_fused_outputs_2d_kernel(
    const float* __restrict__ h,
    const float* __restrict__ lambda,
    const float* __restrict__ log_w,
    float* __restrict__ d_loglik,
    float* __restrict__ d_vol,
    float* __restrict__ d_h_mean,
    float* __restrict__ d_sigma_mean,
    int t_out,
    int n
) {
    __shared__ float s_max;
    
    // Pass 1: Find max log-weight
    float local_max = -1e10f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, log_w[i]);
    }
    local_max = block_reduce_max_2d(local_max);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_log_w = s_max;
    
    // Pass 2: Accumulate
    float local_sum_exp = 0.0f;
    float local_sum_vol = 0.0f;
    float local_sum_h = 0.0f;
    float local_sum_sigma = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum_exp += __expf(log_w[i] - max_log_w);
        float h_i = h[i];
        float lam_i = lambda[i];
        local_sum_vol += __expf(h_i * 0.5f);
        local_sum_h += h_i;
        local_sum_sigma += __expf(lam_i);  // sigma = exp(lambda)
    }
    
    local_sum_exp = block_reduce_sum_2d(local_sum_exp);
    __syncthreads();
    local_sum_vol = block_reduce_sum_2d(local_sum_vol);
    __syncthreads();
    local_sum_h = block_reduce_sum_2d(local_sum_h);
    __syncthreads();
    local_sum_sigma = block_reduce_sum_2d(local_sum_sigma);
    
    if (threadIdx.x == 0) {
        float inv_n = 1.0f / (float)n;
        d_loglik[t_out] = max_log_w + __logf(local_sum_exp * inv_n + 1e-10f);
        d_vol[t_out] = local_sum_vol * inv_n;
        *d_h_mean = local_sum_h * inv_n;
        *d_sigma_mean = local_sum_sigma * inv_n;  // Estimated sigma_z
    }
}
