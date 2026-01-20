/**
 * @file svpf_finite_diff.cu
 * @brief Parallel finite differences for SVPF parameter learning
 * 
 * Runs 6 SVPFs in parallel with perturbed parameters:
 * [θ+δ_ρ, θ-δ_ρ, θ+δ_σ, θ-δ_σ, θ+δ_μ, θ-δ_μ]
 * 
 * Uses Common Random Numbers (CRN) for variance reduction.
 */

#include "svpf.h"
#include <stdio.h>
#include <math.h>

// =============================================================================
// FINITE DIFFERENCE STRUCTURE
// =============================================================================

struct SVPFFiniteDiff {
    SVPFState* filters[6];
    float delta;
    
    // Unconstrained parameters
    float log_rho_unc;  // rho = 0.999 * sigmoid(log_rho_unc)
    float log_sigma;    // sigma_z = exp(log_sigma)
    float mu;
    
    // Base seed for CRN
    unsigned long long base_seed;
    
    // Configuration
    int n_particles;
    int n_stein_steps;
    float nu;
    
    // CUDA streams for parallel execution
    cudaStream_t streams[6];
};

// =============================================================================
// HELPER: Parameter Transforms
// =============================================================================

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float logit(float p) {
    return logf(p / (1.0f - p + 1e-8f));
}

static void unconstrained_to_constrained(float log_rho_unc, float log_sigma, float mu,
                                          float* rho, float* sigma_z, float* mu_out) {
    *rho = 0.999f * sigmoid(log_rho_unc);
    *sigma_z = expf(log_sigma);
    *mu_out = mu;
}

static void constrained_to_unconstrained(float rho, float sigma_z, float mu,
                                          float* log_rho_unc, float* log_sigma, float* mu_out) {
    *log_rho_unc = logit(rho / 0.999f);
    *log_sigma = logf(sigma_z);
    *mu_out = mu;
}

// =============================================================================
// API IMPLEMENTATION
// =============================================================================

SVPFFiniteDiff* svpf_finite_diff_create(int n_particles, int n_stein_steps, 
                                         float nu, float delta) {
    SVPFFiniteDiff* fd = (SVPFFiniteDiff*)malloc(sizeof(SVPFFiniteDiff));
    if (!fd) return NULL;
    
    fd->delta = delta;
    fd->n_particles = n_particles;
    fd->n_stein_steps = n_stein_steps;
    fd->nu = nu;
    fd->base_seed = 42;
    
    // Initialize parameters
    fd->log_rho_unc = 0.0f;
    fd->log_sigma = -1.5f;
    fd->mu = -5.0f;
    
    // Create streams for parallel execution
    for (int i = 0; i < 6; i++) {
        cudaStreamCreate(&fd->streams[i]);
        fd->filters[i] = svpf_create(n_particles, n_stein_steps, nu, fd->streams[i]);
        if (!fd->filters[i]) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                svpf_destroy(fd->filters[j]);
                cudaStreamDestroy(fd->streams[j]);
            }
            free(fd);
            return NULL;
        }
    }
    
    return fd;
}

void svpf_finite_diff_destroy(SVPFFiniteDiff* fd) {
    if (!fd) return;
    
    for (int i = 0; i < 6; i++) {
        if (fd->filters[i]) {
            svpf_destroy(fd->filters[i]);
        }
        cudaStreamDestroy(fd->streams[i]);
    }
    
    free(fd);
}

void svpf_finite_diff_set_params(SVPFFiniteDiff* fd, float rho, float sigma_z, float mu) {
    constrained_to_unconstrained(rho, sigma_z, mu,
                                  &fd->log_rho_unc, &fd->log_sigma, &fd->mu);
}

void svpf_finite_diff_get_params(SVPFFiniteDiff* fd, float* rho, float* sigma_z, float* mu) {
    unconstrained_to_constrained(fd->log_rho_unc, fd->log_sigma, fd->mu,
                                  rho, sigma_z, mu);
}

float svpf_finite_diff_gradient(SVPFFiniteDiff* fd, const float* observations, int T,
                                 float* grad_rho, float* grad_sigma, float* grad_mu) {
    float delta = fd->delta;
    
    // Create perturbed parameter sets
    // [0]: +δ_ρ, [1]: -δ_ρ, [2]: +δ_σ, [3]: -δ_σ, [4]: +δ_μ, [5]: -δ_μ
    float perturbed[6][3];  // [filter][log_rho_unc, log_sigma, mu]
    
    for (int i = 0; i < 6; i++) {
        perturbed[i][0] = fd->log_rho_unc;
        perturbed[i][1] = fd->log_sigma;
        perturbed[i][2] = fd->mu;
    }
    
    perturbed[0][0] += delta;  // +δ_ρ
    perturbed[1][0] -= delta;  // -δ_ρ
    perturbed[2][1] += delta;  // +δ_σ
    perturbed[3][1] -= delta;  // -δ_σ
    perturbed[4][2] += delta;  // +δ_μ
    perturbed[5][2] -= delta;  // -δ_μ
    
    // Convert to constrained and initialize filters
    SVPFParams params[6];
    for (int i = 0; i < 6; i++) {
        unconstrained_to_constrained(perturbed[i][0], perturbed[i][1], perturbed[i][2],
                                      &params[i].rho, &params[i].sigma_z, &params[i].mu);
        svpf_initialize(fd->filters[i], &params[i], fd->base_seed);
    }
    
    // Run all 6 filters in parallel through the observations
    float log_liks[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    SVPFResult results[6];
    
    for (int t = 0; t < T; t++) {
        float y_t = observations[t];
        unsigned long long step_seed = fd->base_seed + t * 1000000ULL;  // CRN: same seed per step
        
        // Launch all 6 steps in parallel
        for (int i = 0; i < 6; i++) {
            svpf_step_seeded(fd->filters[i], y_t, &params[i], step_seed, &results[i]);
        }
        
        // Synchronize all streams
        for (int i = 0; i < 6; i++) {
            cudaStreamSynchronize(fd->streams[i]);
            log_liks[i] += results[i].log_lik_increment;
        }
    }
    
    // Compute central difference gradients
    // ∇ = (L(θ+δ) - L(θ-δ)) / (2δ)
    *grad_rho = (log_liks[0] - log_liks[1]) / (2.0f * delta);
    *grad_sigma = (log_liks[2] - log_liks[3]) / (2.0f * delta);
    *grad_mu = (log_liks[4] - log_liks[5]) / (2.0f * delta);
    
    // Return mean log-likelihood
    float mean_ll = 0.0f;
    for (int i = 0; i < 6; i++) {
        mean_ll += log_liks[i];
    }
    return mean_ll / 6.0f;
}

// =============================================================================
// ADAM OPTIMIZER STEP
// =============================================================================

typedef struct {
    float m_rho, m_sigma, m_mu;      // First moment
    float v_rho, v_sigma, v_mu;      // Second moment
    int t;                           // Timestep
    float beta1, beta2, epsilon;
} AdamState;

void adam_init(AdamState* adam) {
    adam->m_rho = 0.0f;
    adam->m_sigma = 0.0f;
    adam->m_mu = 0.0f;
    adam->v_rho = 0.0f;
    adam->v_sigma = 0.0f;
    adam->v_mu = 0.0f;
    adam->t = 0;
    adam->beta1 = 0.9f;
    adam->beta2 = 0.999f;
    adam->epsilon = 1e-8f;
}

void adam_step(AdamState* adam, SVPFFiniteDiff* fd, 
               float grad_rho, float grad_sigma, float grad_mu, float lr) {
    adam->t++;
    float t = (float)adam->t;
    
    // Update biased first moment
    adam->m_rho = adam->beta1 * adam->m_rho + (1.0f - adam->beta1) * grad_rho;
    adam->m_sigma = adam->beta1 * adam->m_sigma + (1.0f - adam->beta1) * grad_sigma;
    adam->m_mu = adam->beta1 * adam->m_mu + (1.0f - adam->beta1) * grad_mu;
    
    // Update biased second moment
    adam->v_rho = adam->beta2 * adam->v_rho + (1.0f - adam->beta2) * grad_rho * grad_rho;
    adam->v_sigma = adam->beta2 * adam->v_sigma + (1.0f - adam->beta2) * grad_sigma * grad_sigma;
    adam->v_mu = adam->beta2 * adam->v_mu + (1.0f - adam->beta2) * grad_mu * grad_mu;
    
    // Bias correction
    float m_rho_hat = adam->m_rho / (1.0f - powf(adam->beta1, t));
    float m_sigma_hat = adam->m_sigma / (1.0f - powf(adam->beta1, t));
    float m_mu_hat = adam->m_mu / (1.0f - powf(adam->beta1, t));
    
    float v_rho_hat = adam->v_rho / (1.0f - powf(adam->beta2, t));
    float v_sigma_hat = adam->v_sigma / (1.0f - powf(adam->beta2, t));
    float v_mu_hat = adam->v_mu / (1.0f - powf(adam->beta2, t));
    
    // Update parameters (gradient ASCENT for maximizing likelihood)
    float update_rho = lr * m_rho_hat / (sqrtf(v_rho_hat) + adam->epsilon);
    float update_sigma = lr * m_sigma_hat / (sqrtf(v_sigma_hat) + adam->epsilon);
    float update_mu = lr * m_mu_hat / (sqrtf(v_mu_hat) + adam->epsilon);
    
    // Clamp updates for stability
    update_rho = fminf(fmaxf(update_rho, -0.5f), 0.5f);
    update_sigma = fminf(fmaxf(update_sigma, -0.5f), 0.5f);
    update_mu = fminf(fmaxf(update_mu, -0.5f), 0.5f);
    
    fd->log_rho_unc += update_rho;
    fd->log_sigma += update_sigma;
    fd->mu += update_mu;
}

// =============================================================================
// TRAINING FUNCTION
// =============================================================================

void svpf_train(SVPFFiniteDiff* fd, const float* observations, int T,
                int n_epochs, float lr, int verbose) {
    AdamState adam;
    adam_init(&adam);
    
    if (verbose) {
        float rho, sigma_z, mu;
        svpf_finite_diff_get_params(fd, &rho, &sigma_z, &mu);
        printf("Initial params: ρ=%.4f, σ_z=%.4f, μ=%.2f\n", rho, sigma_z, mu);
    }
    
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Use different base seed each epoch
        fd->base_seed = 42 + epoch;
        
        float grad_rho, grad_sigma, grad_mu;
        float ll = svpf_finite_diff_gradient(fd, observations, T,
                                              &grad_rho, &grad_sigma, &grad_mu);
        
        adam_step(&adam, fd, grad_rho, grad_sigma, grad_mu, lr);
        
        if (verbose && epoch % 20 == 0) {
            float rho, sigma_z, mu;
            svpf_finite_diff_get_params(fd, &rho, &sigma_z, &mu);
            printf("Epoch %4d: LL=%.2f, ρ=%.4f, σ_z=%.4f, μ=%.2f | ∇ρ=%.2f, ∇σ=%.2f\n",
                   epoch, ll, rho, sigma_z, mu, grad_rho, grad_sigma);
        }
    }
    
    if (verbose) {
        float rho, sigma_z, mu;
        svpf_finite_diff_get_params(fd, &rho, &sigma_z, &mu);
        printf("Final params: ρ=%.4f, σ_z=%.4f, μ=%.2f\n", rho, sigma_z, mu);
    }
}

// =============================================================================
// TEST
// =============================================================================

#ifdef TEST_FINITE_DIFF

#include <stdlib.h>

static void generate_test_data(float* y, int T, float rho, float sigma_z, float mu) {
    srand(42);
    
    auto randn = []() -> float {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
    };
    
    float h = mu;
    for (int t = 0; t < T; t++) {
        h = mu + rho * (h - mu) + sigma_z * randn();
        float vol = expf(h / 2.0f);
        y[t] = vol * randn();
    }
}

int main() {
    printf("SVPF Finite Difference Gradient Test\n");
    printf("=====================================\n\n");
    
    // True parameters
    float true_rho = 0.95f;
    float true_sigma = 0.20f;
    float true_mu = -5.0f;
    int T = 1000;
    
    printf("True params: ρ=%.2f, σ_z=%.2f, μ=%.1f\n", true_rho, true_sigma, true_mu);
    
    // Generate data
    float* y = (float*)malloc(T * sizeof(float));
    generate_test_data(y, T, true_rho, true_sigma, true_mu);
    
    // Create finite diff optimizer
    SVPFFiniteDiff* fd = svpf_finite_diff_create(512, 5, 5.0f, 0.02f);
    if (!fd) {
        printf("Failed to create finite diff optimizer\n");
        return 1;
    }
    
    // Set initial (wrong) parameters
    svpf_finite_diff_set_params(fd, 0.5f, 0.5f, -3.0f);
    
    // Train
    printf("\nTraining with T=%d, epochs=100, lr=0.05\n\n", T);
    svpf_train(fd, y, T, 100, 0.05f, 1);
    
    // Check results
    float rho, sigma_z, mu;
    svpf_finite_diff_get_params(fd, &rho, &sigma_z, &mu);
    
    printf("\nErrors:\n");
    printf("  |ρ_est - ρ_true| = %.4f\n", fabsf(rho - true_rho));
    printf("  |σ_est - σ_true| = %.4f\n", fabsf(sigma_z - true_sigma));
    printf("  |μ_est - μ_true| = %.4f\n", fabsf(mu - true_mu));
    
    free(y);
    svpf_finite_diff_destroy(fd);
    
    return 0;
}

#endif // TEST_FINITE_DIFF
