# SVPF: Paper vs Implementation Analysis

## Reference Paper
**"Stein Particle Filtering"** - Fan, Taghvaei, Chen (2021)  
arXiv:2106.10568

---

## 1. The Core Algorithm (Paper's Equation 6)

### Paper's Target Distribution

The paper merges predict + update into a single target:

$$p(x_{t+1}|Z_{t+1}) \propto \underbrace{\left[\frac{1}{n}\sum_{i=1}^{n} p(x_{t+1}|x_t^i)\right]}_{\text{Gaussian Mixture Prior}} \cdot \underbrace{p(z_{t+1}|x_{t+1})}_{\text{Likelihood}}$$

**Key insight:** The prior is a **Gaussian mixture** with n components, one centered at each particle's predicted mean.

For our SV model:
- Prior component i: $p(h_{t+1}|h_t^i) = \mathcal{N}(\mu + \rho(h_t^i - \mu), \sigma_z^2)$
- Likelihood: $p(y_{t+1}|h_{t+1}) = \text{Student-t}(\nu)$ scaled by $\exp(h/2)$

### Paper's Gradient (What SVGD Needs)

SVGD requires $\nabla_h \log p(h|Z)$. For the mixture prior:

$$\nabla_h \log p(h|Z) = \nabla_h \log \left[\sum_{i=1}^n p(h|h_{prev}^i)\right] + \nabla_h \log p(y|h)$$

The prior gradient becomes:

$$\nabla_h \log \left[\sum_{i=1}^n p(h|h_{prev}^i)\right] = \frac{\sum_{i=1}^n p(h|h_{prev}^i) \cdot \nabla_h \log p(h|h_{prev}^i)}{\sum_{i=1}^n p(h|h_{prev}^i)}$$

Which simplifies to a **responsibility-weighted average**:

$$\nabla_h \log p_{prior}(h) = -\sum_{i=1}^n r_i(h) \cdot \frac{h - \mu_i}{\sigma_z^2}$$

where:
- $\mu_i = \mu + \rho(h_{prev}^i - \mu)$ is the prior mean from particle i
- $r_i(h) = \frac{p(h|h_{prev}^i)}{\sum_k p(h|h_{prev}^k)}$ is the **responsibility** (soft assignment)

---

## 2. What We Implemented (WRONG)

### Our Gradient Computation

```cuda
// In svpf_likelihood_grad_kernel:
float mu_prior = mu + rho * (h_prev_i - mu);  // Only THIS particle's prior
float grad_prior = -(h_i - mu_prior) / sigma_z_sq;
```

### The Bug

Each particle j only feels attraction to **its own** prior mean $\mu_j$:

$$\nabla_h \log p_{prior}(h_j) = -\frac{h_j - \mu_j}{\sigma_z^2}$$

This is **WRONG**. It treats each particle as running an independent filter.

### Consequence

- Particles don't share information through the prior
- The "mixture" structure is lost
- Effectively running N independent EKFs with Stein refinement
- Loses the key benefit of particle filtering: representing multimodal posteriors

---

## 3. Correct Implementation

### Option A: Explicit Mixture Prior Gradient (O(N²) per particle)

```cuda
__global__ void svpf_likelihood_grad_kernel_CORRECT(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad,
    float* __restrict__ log_w,
    const float* __restrict__ d_y,
    int t,
    float rho, float sigma_z, float mu, float nu, float student_t_const,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    
    float h_j = h[j];
    float y_t = d_y[t];
    float sigma_z_sq = sigma_z * sigma_z;
    
    // === LIKELIHOOD GRADIENT (unchanged) ===
    float vol = safe_exp(h_j * 0.5f);
    float z = y_t / vol;
    float z_sq = z * z;
    
    log_w[j] = student_t_const - 0.5f * h_j 
             - 0.5f * (nu + 1.0f) * log1pf(z_sq / nu);
    
    float grad_lik = -0.5f + 0.5f * (nu + 1.0f) * z_sq / (nu + z_sq);
    
    // === MIXTURE PRIOR GRADIENT (CORRECTED) ===
    // Sum over ALL particles to compute responsibility-weighted gradient
    float weighted_grad_sum = 0.0f;
    float responsibility_sum = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float mu_i = mu + rho * (h_prev[i] - mu);
        float diff = h_j - mu_i;
        
        // Unnormalized responsibility: p(h_j | h_prev_i) ∝ exp(-diff²/2σ²)
        float log_r_i = -diff * diff / (2.0f * sigma_z_sq);
        float r_i = expf(log_r_i);
        
        // Gradient of log p(h_j | h_prev_i) = -diff / σ²
        weighted_grad_sum += r_i * (-diff / sigma_z_sq);
        responsibility_sum += r_i;
    }
    
    // Normalize to get E_r[grad]
    float grad_prior = weighted_grad_sum / (responsibility_sum + 1e-8f);
    
    // === COMBINE ===
    grad[j] = fminf(fmaxf(grad_prior + grad_lik, -10.0f), 10.0f);
}
```

### Option B: Fuse with Stein Kernel (More Efficient)

Since Stein kernel is already O(N²), we can compute the mixture prior gradient there:

```cuda
// Inside svpf_stein_2d_kernel, for particle i summing over j:

// Standard Stein terms
float K = expf(-diff * diff / (2.0f * bw_sq));
k_sum += K * grad[j];           // K(h_j, h_i) * ∇log p(h_j)
gk_sum += -K * diff / bw_sq;    // ∇_j K(h_j, h_i)

// NEW: Mixture prior contribution
// The grad[j] already includes likelihood gradient
// We need to add the mixture prior gradient BEFORE Stein
```

Actually, the cleanest approach is to compute the mixture prior gradient in a separate kernel BEFORE the Stein kernel, then Stein uses the corrected gradient.

---

## 4. Summary of Differences

| Aspect | Paper (Correct) | Our Implementation (Wrong) |
|--------|-----------------|---------------------------|
| Prior | Gaussian Mixture (N components) | Single Gaussian per particle |
| Prior mean | Weighted avg of all $\mu_i$ | Only own $\mu_j$ |
| Gradient | $-\sum_i r_i(h) \frac{h-\mu_i}{\sigma^2}$ | $-\frac{h_j - \mu_j}{\sigma^2}$ |
| Particle interaction | Through prior responsibilities | Only through Stein kernel |
| Complexity | O(N²) for prior gradient | O(N) for prior gradient |
| Multimodality | Preserved by mixture | Lost (unimodal prior per particle) |

---

## 5. What The Paper's Algorithm Actually Does

### Algorithm 2 from Paper (Sequential Stein PF):

```
Input: particles {x_t^i} approximating p(x_t|Z_t), observation z_{t+1}

1. Initialize: Sample x^i ~ p(x_{t+1}|x_t^i) for each i
   (Sample from transition, one sample per particle)

2. Define target: 
   p(x_{t+1}|Z_{t+1}) ∝ [1/n Σᵢ p(x_{t+1}|x_t^i)] · p(z_{t+1}|x_{t+1})
   
3. Run SVGD with:
   - Initial particles: {x^i} from step 1
   - Target: p(x_{t+1}|Z_{t+1}) from step 2
   - Gradient: ∇log[mixture prior] + ∇log[likelihood]

4. Output: Updated particles {x_{t+1}^i}
```

### What SVGD Needs

The SVGD update is:
$$\phi^*(x) = \frac{1}{n}\sum_{j=1}^n \left[ k(x^j, x) \nabla_{x^j} \log p(x^j|Z) + \nabla_{x^j} k(x^j, x) \right]$$

Where $\nabla_{x^j} \log p(x^j|Z)$ must be the **full gradient including mixture prior**.

---

## 6. Why This Matters

### The Mixture Prior Enables:

1. **Information sharing**: High-probability particles influence all particles through the prior
2. **Multimodal tracking**: The mixture can represent multiple hypotheses
3. **Robustness**: Particles near multiple prior means get pulled to the weighted center
4. **Correct posterior**: Without mixture, we're not approximating the right distribution

### Example:

Consider 2 particles with $h_{prev}^1 = -5$ and $h_{prev}^2 = -3$.

**Our implementation (wrong):**
- Particle 1 feels prior pull toward $\mu_1 = \mu + \rho(-5 - \mu)$
- Particle 2 feels prior pull toward $\mu_2 = \mu + \rho(-3 - \mu)$
- No interaction through prior

**Paper's algorithm (correct):**
- Both particles feel pull from BOTH $\mu_1$ and $\mu_2$
- A particle at $h = -4$ would feel equal pull from both
- A particle at $h = -4.8$ would feel stronger pull from $\mu_1$
- This is proper Bayesian inference with the mixture prior

---

## 7. Implementation Plan

### Step 1: Restore Working Baseline
Get back to the 0.98 RMSE version (need the file)

### Step 2: Add Mixture Prior Gradient Kernel
```cuda
__global__ void svpf_mixture_prior_grad_kernel(
    const float* __restrict__ h,
    const float* __restrict__ h_prev,
    float* __restrict__ grad_prior_out,  // Output: mixture prior gradient
    float rho, float sigma_z, float mu,
    int n
);
```

### Step 3: Modify Likelihood+Grad Kernel
- Compute only likelihood gradient
- Add mixture prior gradient (from step 2)

### Step 4: Benchmark
- Compare against wrong implementation
- Compare against HCRBPF

### Expected Outcome
The mixture prior should:
- Improve tracking during regime changes (multimodal posterior)
- Reduce bias (correct Bayesian inference)
- Possibly close the gap to HCRBPF

---

## 8. Complexity Analysis

| Operation | Our Implementation | Correct Implementation |
|-----------|-------------------|------------------------|
| Prior gradient | O(N) | O(N²) |
| Stein kernel | O(N²) | O(N²) |
| Total | O(N²) | O(N²) |

The mixture prior gradient doesn't change asymptotic complexity since Stein is already O(N²). It adds a constant factor (~2x) to the N² work.

---

## 9. Alternative: Log-Sum-Exp Stable Version

For numerical stability with the mixture:

```cuda
// Compute log responsibilities first
float log_r_max = -INFINITY;
for (int i = 0; i < n; i++) {
    float mu_i = mu + rho * (h_prev[i] - mu);
    float diff = h_j - mu_i;
    float log_r_i = -diff * diff / (2.0f * sigma_z_sq);
    log_r_max = fmaxf(log_r_max, log_r_i);
}

// Log-sum-exp for normalization
float sum_exp = 0.0f;
float weighted_grad = 0.0f;
for (int i = 0; i < n; i++) {
    float mu_i = mu + rho * (h_prev[i] - mu);
    float diff = h_j - mu_i;
    float log_r_i = -diff * diff / (2.0f * sigma_z_sq);
    float r_i = expf(log_r_i - log_r_max);  // Stable
    
    sum_exp += r_i;
    weighted_grad += r_i * (-diff / sigma_z_sq);
}

float grad_prior = weighted_grad / sum_exp;
```

---

## 10. Conclusion

**The fundamental bug**: We computed the prior gradient assuming each particle has its own independent prior, but the paper's algorithm uses a **shared mixture prior** across all particles.

This is not a minor optimization issue - it's computing the gradient of the wrong distribution.

The fix requires O(N²) work for the prior gradient, but since Stein is already O(N²), the total complexity doesn't change.
