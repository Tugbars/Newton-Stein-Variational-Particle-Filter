/**
 * @file svpf_fully_fused.cu
 * @brief Fully Fused Single-Kernel SVPF Step
 * 
 * Fuses ALL operations into ONE kernel launch:
 * - Predict (AR(1) propagation + MIM jumps)
 * - Guide (EKF update + nudge)
 * - Bandwidth (median-based)
 * - Stein loop (gradient + transport × N iterations)
 * - Outputs (weighted mean, vol, loglik)
 * 
 * Expected: 40-60μs per step (vs 250μs with separate kernels)
 * 
 * Limitation: N ≤ 1024 (single block execution)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float clamp_logvol(float h) {
    return fminf(fmaxf(h, -15.0f), 5.0f);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum_fused(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__device__ float block_reduce_max_fused(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : -1e10f;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// =============================================================================
// FULLY FUSED SVPF STEP KERNEL
// =============================================================================
// 
// Single block, N threads. Everything happens in one kernel.
// 
// Launch: <<<1, N>>> where N = n_particles (≤ 1024)

__global__ void svpf_fully_fused_step_kernel(
    // Particle state [N]
    float* __restrict__ h,
    float* __restrict__ h_prev,
    float* __restrict__ grad_log_p,
    float* __restrict__ log_weights,
    float* __restrict__ d_grad_v,           // RMSprop velocity
    curandStatePhilox4_32_10_t* __restrict__ rng,
    // Inputs
    float y_t,
    float y_prev,
    float h_mean_prev,                      // Previous posterior mean (for guide)
    // Outputs
    float* __restrict__ d_h_mean,
    float* __restrict__ d_vol,
    float* __restrict__ d_loglik,
    float* __restrict__ d_bandwidth_out,
    float* __restrict__ d_ksd,
    // Model parameters
    float rho,
    float sigma_z,
    float mu,
    float nu,
    float lik_offset,
    float student_t_const,
    // MIM parameters
    float mim_jump_prob,
    float mim_jump_scale,
    // Guide parameters
    float guide_strength,
    float guide_mean,                       // EKF estimate (computed externally for now)
    int use_guide,
    // Stein parameters
    float step_size,
    float temperature,
    float rmsprop_rho,
    float rmsprop_eps,
    int n_stein_steps,
    int n_anneal_steps,
    int stein_sign_mode,
    // Control
    int n
) {
    int i = threadIdx.x;
    if (i >= n) return;
    
    // Shared memory layout
    extern __shared__ float smem[];
    float* sh_h = smem;                     // [N]
    float* sh_grad = smem + n;              // [N]
    float* sh_reduce = smem + 2 * n;        // [32] for reductions
    
    __shared__ float sh_bandwidth;
    __shared__ float sh_mean_h;
    
    float inv_n = 1.0f / (float)n;
    float sigma_z_sq = sigma_z * sigma_z;
    float inv_sigma_z_sq = 1.0f / sigma_z_sq;
    
    // ==========================================================================
    // PHASE 1: PREDICT (AR(1) + MIM)
    // ==========================================================================
    
    float h_i = h[i];
    float h_prev_i = h_prev[i];
    
    // AR(1) mean
    float prior_mean = mu + rho * (h_prev_i - mu);
    
    // Sample from prior
    float4 rand4 = curand_normal4(&rng[i]);
    float z1 = rand4.x;
    float z2 = rand4.y;
    
    // MIM: mixture of narrow and wide components
    float is_jump = (curand_uniform(&rng[i]) < mim_jump_prob) ? 1.0f : 0.0f;
    float effective_sigma = sigma_z * (1.0f + is_jump * (mim_jump_scale - 1.0f));
    
    h_i = prior_mean + effective_sigma * z1;
    h_i = clamp_logvol(h_i);
    
    // Store new h_prev
    h_prev[i] = h[i];  // Save current as previous for next step
    h[i] = h_i;
    
    __syncthreads();
    
    // ==========================================================================
    // PHASE 2: GUIDE (nudge toward EKF estimate)
    // ==========================================================================
    
    if (use_guide && guide_strength > 0.0f) {
        // Variance-preserving guide: shift distribution toward guide_mean
        float deviation = h_i - h_mean_prev;
        float guided_h = guide_mean + deviation * (1.0f - guide_strength);
        h_i = guided_h;
        h[i] = clamp_logvol(h_i);
    }
    
    __syncthreads();
    
    // ==========================================================================
    // PHASE 3: BANDWIDTH (Silverman's rule)
    // ==========================================================================
    
    // Load to shared for reductions
    sh_h[i] = h[i];
    __syncthreads();
    
    // Compute mean
    float sum_h = block_reduce_sum_fused(sh_h[i], sh_reduce);
    __syncthreads();
    
    if (i == 0) sh_mean_h = sum_h * inv_n;
    __syncthreads();
    
    float mean_h = sh_mean_h;
    float diff_h = sh_h[i] - mean_h;
    
    // Compute variance
    float sum_sq = block_reduce_sum_fused(diff_h * diff_h, sh_reduce);
    __syncthreads();
    
    if (i == 0) {
        float var_h = sum_sq * inv_n;
        float std_h = sqrtf(fmaxf(var_h, 1e-6f));
        float bw = 1.06f * std_h * powf((float)n, -0.2f);
        bw = fminf(fmaxf(bw, 0.05f), 2.0f);
        sh_bandwidth = bw;
        *d_bandwidth_out = bw;
    }
    __syncthreads();
    
    float bandwidth = sh_bandwidth;
    float bw_sq = bandwidth * bandwidth;
    float inv_bw_sq = 1.0f / bw_sq;
    
    // ==========================================================================
    // PHASE 4: STEIN LOOP
    // ==========================================================================
    
    float beta_schedule[3] = {0.3f, 0.65f, 1.0f};
    int steps_per_anneal = n_stein_steps / n_anneal_steps;
    int remainder = n_stein_steps - steps_per_anneal * n_anneal_steps;
    
    float sign_mult = (stein_sign_mode == 1) ? 1.0f : -1.0f;
    
    int global_step = 0;
    float ksd_final = 0.0f;
    
    for (int ai = 0; ai < n_anneal_steps; ai++) {
        float beta = beta_schedule[ai % 3];
        
        int si = steps_per_anneal;
        if (ai == n_anneal_steps - 1) si += remainder;
        
        for (int s = 0; s < si; s++) {
            global_step++;
            bool is_last = (global_step == n_stein_steps);
            
            // ----- Gradient -----
            h_i = h[i];
            
            // Prior gradient
            float prior_mean_grad = mu + rho * (h_prev[i] - mu);
            float grad_prior = -(h_i - prior_mean_grad) * inv_sigma_z_sq;
            
            // Likelihood (Student-t)
            float vol = expf(h_i * 0.5f);
            float y_scaled = (y_t - lik_offset) / vol;
            float y_sq_norm = y_scaled * y_scaled / nu;
            float one_plus_A = 1.0f + y_sq_norm;
            
            // Log weight
            log_weights[i] = student_t_const - 0.5f * h_i 
                           - 0.5f * (nu + 1.0f) * logf(one_plus_A);
            
            // Likelihood gradient
            float grad_lik = -0.5f + 0.5f * (nu + 1.0f) * y_sq_norm / one_plus_A;
            
            float g = grad_prior + beta * grad_lik;
            g = fminf(fmaxf(g, -10.0f), 10.0f);
            grad_log_p[i] = g;
            
            __syncthreads();
            
            // Load to shared
            sh_h[i] = h[i];
            sh_grad[i] = grad_log_p[i];
            __syncthreads();
            
            // ----- Stein Transport -----
            float s_i = sh_grad[i];
            float k_sum = 0.0f;
            float gk_sum = 0.0f;
            float ksd_sum = 0.0f;
            
            #pragma unroll 8
            for (int j = 0; j < n; j++) {
                float h_j = sh_h[j];
                float s_j = sh_grad[j];
                float diff = h_i - h_j;
                float diff_sq = diff * diff;
                float dist_sq = diff_sq * inv_bw_sq;
                
                // IMQ kernel
                float base = 1.0f + dist_sq;
                float K = 1.0f / base;
                float K_sq = K * K;
                
                k_sum += K * s_j;
                gk_sum += sign_mult * 2.0f * diff * inv_bw_sq * K_sq;
                
                // KSD (last iteration)
                if (is_last) {
                    float grad_x_k = -2.0f * diff * inv_bw_sq * K_sq;
                    float grad_y_k = -grad_x_k;
                    float hess_xy_k = 2.0f * inv_bw_sq * K_sq * (4.0f * dist_sq * K - 1.0f);
                    ksd_sum += K * s_i * s_j + s_i * grad_y_k + s_j * grad_x_k + hess_xy_k;
                }
            }
            
            float phi_i = (k_sum + gk_sum) * inv_n;
            
            // RMSprop
            float v_prev = d_grad_v[i];
            float v_new = rmsprop_rho * v_prev + (1.0f - rmsprop_rho) * phi_i * phi_i;
            d_grad_v[i] = v_new;
            
            // Transport
            float effective_step = step_size * sqrtf(beta);
            float precond = rsqrtf(v_new + rmsprop_eps);
            float drift = effective_step * phi_i * precond;
            
            float diffusion = 0.0f;
            if (temperature > 1e-6f) {
                diffusion = sqrtf(2.0f * effective_step * temperature) * z2;
                z2 = curand_normal(&rng[i]);  // Get new noise for next iteration
            }
            
            h[i] = clamp_logvol(h_i + drift + diffusion);
            
            if (is_last) {
                ksd_final = ksd_sum;
            }
            
            __syncthreads();
        }
    }
    
    // ==========================================================================
    // PHASE 5: OUTPUTS
    // ==========================================================================
    
    // KSD reduction
    float sum_ksd = block_reduce_sum_fused(ksd_final, sh_reduce);
    __syncthreads();
    
    if (i == 0) {
        *d_ksd = sqrtf(fmaxf(sum_ksd * inv_n * inv_n, 0.0f));
    }
    
    // Weighted outputs (log-sum-exp)
    float log_w_i = log_weights[i];
    float max_log_w = block_reduce_max_fused(log_w_i, sh_reduce);
    __syncthreads();
    
    if (i == 0) sh_reduce[0] = max_log_w;
    __syncthreads();
    max_log_w = sh_reduce[0];
    
    float w_i = expf(log_w_i - max_log_w);
    float h_final = h[i];
    float vol_i = expf(h_final * 0.5f);
    
    float sum_w = block_reduce_sum_fused(w_i, sh_reduce);
    __syncthreads();
    
    float sum_wh = block_reduce_sum_fused(w_i * h_final, sh_reduce);
    __syncthreads();
    
    float sum_wvol = block_reduce_sum_fused(w_i * vol_i, sh_reduce);
    __syncthreads();
    
    if (i == 0) {
        float safe_sum = fmaxf(sum_w, 1e-10f);
        *d_h_mean = sum_wh / safe_sum;
        *d_vol = sum_wvol / safe_sum;
        *d_loglik = max_log_w + logf(safe_sum * inv_n);
    }
}

// =============================================================================
// Wrapper
// =============================================================================

extern "C" {

cudaError_t svpf_fully_fused_step(
    float* h, float* h_prev, float* grad_log_p, float* log_weights,
    float* d_grad_v, curandStatePhilox4_32_10_t* rng,
    float y_t, float y_prev, float h_mean_prev,
    float* d_h_mean, float* d_vol, float* d_loglik, float* d_bandwidth, float* d_ksd,
    float rho, float sigma_z, float mu, float nu, float lik_offset, float student_t_const,
    float mim_jump_prob, float mim_jump_scale,
    float guide_strength, float guide_mean, int use_guide,
    float step_size, float temperature, float rmsprop_rho, float rmsprop_eps,
    int n_stein_steps, int n_anneal_steps, int stein_sign_mode,
    int n, cudaStream_t stream
) {
    if (n > 1024) {
        return cudaErrorInvalidConfiguration;
    }
    
    int smem_size = (2 * n + 32) * sizeof(float);
    
    svpf_fully_fused_step_kernel<<<1, n, smem_size, stream>>>(
        h, h_prev, grad_log_p, log_weights, d_grad_v, rng,
        y_t, y_prev, h_mean_prev,
        d_h_mean, d_vol, d_loglik, d_bandwidth, d_ksd,
        rho, sigma_z, mu, nu, lik_offset, student_t_const,
        mim_jump_prob, mim_jump_scale,
        guide_strength, guide_mean, use_guide,
        step_size, temperature, rmsprop_rho, rmsprop_eps,
        n_stein_steps, n_anneal_steps, stein_sign_mode,
        n
    );
    
    return cudaGetLastError();
}

bool svpf_fully_fused_supported(int n_particles) {
    return n_particles <= 1024;
}

} // extern "C"
