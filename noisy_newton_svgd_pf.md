# Noisy Newton-SVGD Particle Filter for Stochastic Volatility Estimation

## Abstract

We present a particle filter for stochastic volatility (SV) models that combines Newton-preconditioned Stein Variational Gradient Descent with Langevin noise injection, operating without the repulsive force term standard in SVGD. The method transports equally-weighted particles through a tempered likelihood path at each observation, using target Hessian preconditioning to set gradient direction and RMSProp to bound step magnitude. Langevin noise at each transport step prevents the variance collapse identified in deterministic SVGD (Ba et al., 2022), serving as an unbiased diversity mechanism that replaces the biased kernel-gradient repulsion of standard SVGD. Under model misspecification—the regime that matters for real financial data—the filter achieves lower RMSE and bias than the Bootstrap Particle Filter with comparable particle counts.

## 1. Problem Setting

We consider the canonical stochastic volatility model:

$$h_t = \mu + \rho(h_{t-1} - \mu) + \sigma_\eta \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)$$
$$y_t = \exp(h_t / 2) \cdot u_t, \quad u_t \sim t_\nu$$

where $h_t$ is the log-volatility, $y_t$ is the observed return, and the observation noise follows a Student-t distribution with degrees of freedom $\nu$. The filtering problem is to estimate $p(h_t | y_{1:t})$ sequentially.

Standard particle filters suffer from weight degeneracy: after resampling, a few particles dominate and diversity collapses. SVGD-based filters replace importance weighting with deterministic transport—all particles are equally weighted and moved toward the posterior via gradient-based updates. However, deterministic SVGD introduces its own pathology: variance collapse from deterministic bias in finite particle regimes.

## 2. Background: Why Standard SVGD Fails

### 2.1 The SVGD Update

The standard SVGD update for particle $x_i$ is:

$$x_i \leftarrow x_i + \frac{\epsilon}{N} \sum_{j=1}^{N} \left[ \underbrace{k(x_j, x_i) \nabla_{x_j} \log p(x_j)}_{\text{S1: driving force}} + \underbrace{\nabla_{x_j} k(x_j, x_i)}_{\text{S2: repulsive force}} \right]$$

where $k$ is a kernel function (typically Cauchy or RBF) and $p$ is the target density.

### 2.2 Variance Collapse and Its Root Cause

Ba et al. (ICLR 2022) identified two interacting causes of SVGD's variance collapse:

1. **High variance of S1:** The driving force term $S1(x_j, x) = k(x_j, x) \nabla \log p(x_j)$ has mean squared error scaling as $\Theta(d)$, while the repulsive term $S2$ scales as $\Theta(1/d)$. In finite particles, S1 estimation error dominates.

2. **Deterministic bias:** The same particles that define the empirical distribution $q$ are used to evaluate S1 under $q$. Without resampling, estimation errors in S1 compound across iterations, systematically biasing the converged particle distribution.

Crucially, Ba et al. showed that **MMD-descent**—which shares the same S2 repulsive term but evaluates S1 under the *target* distribution rather than the particle distribution—does not exhibit variance collapse. The problem is not the kernel or the repulsion, but the circular dependency in S1 evaluation.

### 2.3 Noisy SVGD: The Theoretically Correct Fix

Priser et al. (ICLR 2025) proved that adding Langevin noise to each SVGD iteration:

$$x_i \leftarrow x_i + \epsilon \, \phi(x_i) + \sqrt{2\epsilon\lambda} \, \xi_i, \quad \xi_i \sim \mathcal{N}(0, 1)$$

provably avoids variance collapse for any $\lambda > 0$. The limit set of noisy SVGD is well-defined and approaches the target distribution as $N$ increases. Standard (deterministic) SVGD can converge to Dirac measures; noisy SVGD cannot.

The noise breaks the deterministic bias by ensuring particles explore stochastically around the gradient-informed trajectory, preventing the self-reinforcing collapse that occurs when particles only move according to forces computed from their own positions.

## 3. Our Method: Noisy Newton-SVGD Particle Filter

### 3.1 Key Design Decisions

We make four departures from standard SVGD-based particle filters:

**1. Repulsion removed (S2 = 0).** The standard repulsive force $\nabla_x k(x', x)$ pushes particles apart based on inter-particle distances. While this prevents mode collapse in deterministic SVGD, it introduces systematic bias: repulsion depends on the current particle cloud, creating a feedback loop that compounds across sequential filtering timesteps. With Langevin noise providing diversity, repulsion is redundant and harmful. Removing it reduced bias by 27–38% across test scenarios.

**2. Langevin temperature injection.** At every Stein transport step, we add noise scaled by $\sqrt{2 \cdot \text{step\_size} \cdot T}$, with temperature $T = 0.45$. This is the theoretically justified fix for deterministic bias (Priser et al., 2025). Temperature provides *unbiased* diversity: noise is independent of particle positions, has zero expected bias, and its variance averages out over $N$ particles. In contrast, repulsion provides *biased* diversity: the direction depends on other particles' positions, introducing systematic error.

**3. Newton preconditioning (target Hessian only).** We precondition the transport gradient with the inverse Hessian of the target log-density, kernel-smoothed across particles:

$$H_{\text{weighted}}(x_i) = \frac{\sum_j \nabla^2 \log p(x_j) \cdot K(x_i, x_j)}{\sum_j K(x_i, x_j)}$$

Standard full-Newton SVGD includes an additional geometry term $N_k$ from the kernel's second derivatives with respect to inter-particle distances. We removed this term because it creates a circular dependency analogous to repulsion: $N_k$ depends on particle positions, inflating the Hessian near clusters and suppressing movement exactly when misspecification demands aggressive transport. Removing $N_k$ improved accuracy under misspecification by 10%.

**4. RMSProp step normalization.** Newton preconditioning sets the direction and relative scaling of the transport, but does not bound absolute magnitude. When the likelihood gradient spikes after a volatility event, $\phi_i / H$ can overshoot even with correct curvature scaling. RMSProp's running average of squared transport magnitudes, $v_i \leftarrow \rho \cdot v_i + (1 - \rho) \cdot \phi_i^2$, followed by normalization $\phi_i / \sqrt{v_i + \epsilon}$, adaptively bounds step size within each timestep. Newton and RMSProp are complementary preconditioners: Newton handles *direction*, RMSProp handles *magnitude*.

### 3.2 Algorithm: Single Observation Update

Given particles $\{h_i\}_{i=1}^N$ approximating $p(h_{t-1} | y_{1:t-1})$ and a new observation $y_t$:

**Predict step** (with antithetic sampling):
For each pair $(i, i + N/2)$:
$$\hat{h}_i = \mu + \rho(h_i - \mu) + \sigma_\eta \epsilon_i$$
$$\hat{h}_{i+N/2} = \mu + \rho(h_{i+N/2} - \mu) - \sigma_\eta \epsilon_i$$

Antithetic pairing reduces variance of the ensemble mean at zero computational cost.

**Transport step** (tempered likelihood path):
We anneal the likelihood contribution from $\beta = 0$ to $\beta = 1$ over multiple stages, performing several Stein iterations at each stage. At each iteration with current annealing parameter $\beta$:

1. **Gradient computation.** For each particle $h_j$:
$$\nabla \log p_\beta(h_j) = \underbrace{\frac{-(h_j - \mu - \rho(h_{j,\text{prev}} - \mu))}{\sigma_\eta^2}}_{\text{prior score}} + \beta \cdot \underbrace{\left(-\frac{1}{2} + \frac{\nu+1}{2} \cdot \frac{A_j}{1 + A_j}\right)}_{\text{likelihood score}}$$
where $A_j = y_t^2 \exp(-h_j) / \nu$.

2. **Kernel-smoothed Newton preconditioning.**
$$\phi_i = \frac{\sum_j K(x_i, x_j) \cdot \nabla \log p_\beta(x_j)}{\sum_j K(x_i, x_j)}, \quad \phi_i^{\text{precond}} = 0.95 \cdot \phi_i \cdot H_{\text{weighted}}^{-1}(x_i)$$

3. **RMSProp normalization.**
$$v_i \leftarrow \rho_{\text{rms}} \cdot v_i + (1 - \rho_{\text{rms}}) \cdot (\phi_i^{\text{precond}})^2$$
$$\text{drift}_i = \text{step\_size} \cdot \phi_i^{\text{precond}} / \sqrt{v_i + \epsilon}$$

4. **Langevin noise injection.**
$$h_i \leftarrow h_i + \text{drift}_i + \sqrt{2 \cdot \text{step\_size} \cdot T} \cdot \xi_i, \quad \xi_i \sim \mathcal{N}(0, 1)$$

**Output:** Equally-weighted particle mean and variance as the volatility estimate.

### 3.3 Kernel Choice

We use the Cauchy kernel $k(x, x') = 1 / (1 + \|x - x'\|^2 / \sigma^2)$ rather than the Gaussian RBF. The Cauchy kernel's polynomial tails allow particles in the distribution tails to still receive meaningful kernel weight from particles near the mode, preventing tail particles from receiving near-zero gradient information. Bandwidth is set via a modified Silverman rule: $\sigma^2 = 2 \cdot \text{Var}(h) / \log(N + 1)$, with EMA smoothing and volatility-ratio scaling for stability.

### 3.4 Student-t Observation Likelihood

The observation model uses a Student-t distribution with finite degrees of freedom $\nu \approx 5$. This is critical for robustness: under model misspecification (wrong $\rho$, $\sigma_\eta$, or structural model error), extreme returns generate likelihood gradients that can destabilize Gaussian observation models. The Student-t likelihood bounds the score function's growth rate, preventing gradient explosion during market crashes or flash events. This interacts favorably with the Newton preconditioning, as the Hessian of the Student-t log-likelihood is also bounded.

## 4. Why Each Component Matters: Ablation Results

| Configuration | Effect |
|---|---|
| Full method (baseline) | — |
| + Repulsion (S2 enabled) | +27–38% bias |
| − Temperature (T = 0) | 2× RMSE |
| + Nk geometry term in Hessian | −10% misspec accuracy |
| − RMSProp | Accuracy destroyed |
| − Newton preconditioning | Significant accuracy loss |
| EKF guide bandwidth (replacing Silverman) | No improvement (EKF variance too narrow) |
| Adaptive lik_offset (observation-scaled) | No improvement over fixed |

The pattern is clear: **every removal of inter-particle coupling improved accuracy; every removal of per-particle mechanics degraded it.** Repulsion (inter-particle), Nk (inter-particle Hessian geometry)—both encode circular dependencies where particle positions influence the forces acting on those same particles. Temperature (per-particle noise), Newton (per-particle curvature), RMSProp (per-particle step bounding)—all operate on individual particles using only the target density, breaking the feedback loop.

## 5. Comparison with Bootstrap Particle Filter

Under matched-DGP (data generated from the same model used for filtering), the Bootstrap Particle Filter (BPF) achieves marginally lower RMSE—it is asymptotically exact when the model is correct. Under misspecification (parameters or structure differing from the true DGP), the Noisy Newton-SVGD PF dominates:

| Scenario | SVPF RMSE | BPF RMSE |
|---|---|---|
| Spike | 0.69 | 1.18 |
| Regime change | 1.77 | 2.26 |
| Matched DGP | Slightly higher | Baseline |

The advantage under misspecification is the key practical contribution. Real financial volatility never exactly follows the SV model. A filter that degrades gracefully under model error is more valuable than one that is optimal only when the model is perfectly specified.

## 6. Theoretical Justification

The method sits at the intersection of three lines of theory:

**Noisy SVGD theory (Priser et al., 2025):** Any $\lambda > 0$ in the Langevin noise term avoids variance collapse. Our temperature $T = 0.45$ satisfies this. The proof relies on showing that noisy SVGD trajectories approximate a McKean-Vlasov process whose stationary measure is the target.

**SVGD variance collapse analysis (Ba et al., 2022):** The driving force S1 has $\Theta(d)$ variance while repulsion S2 has $\Theta(1/d)$. With repulsion removed, our update is pure kernel-smoothed S1 + noise. The noise counteracts the deterministic bias in S1 without introducing the systematic bias of S2.

**Stein transport (Nüsken, 2024):** Transporting particles along a tempered path $\pi_\beta \propto \text{prior} \cdot \text{likelihood}^\beta$ from $\beta = 0$ to $\beta = 1$ reaches the posterior at finite time rather than requiring SVGD's infinite-time convergence. Our $\beta$-annealing implements this tempered path, with the Newton preconditioner approximating the optimal transport map at each stage.

## 7. Computational Considerations

The dominant cost is the O(N²) kernel smoothing in the Stein transport step, computed once per iteration. With N = 64–256 particles and 8–32 Stein iterations per observation (across annealing stages), this is roughly 500–8000 kernel evaluations per timestep. On GPU, the kernel matrix fits in shared memory for N ≤ 256, and the entire transport step runs as a single fused kernel launch.

The predict step is O(N) (antithetic Gaussian sampling). The bandwidth computation is O(N) via parallel reduction. The per-timestep cost is dominated by the transport step and scales as O(S · N²) where S is the total number of Stein iterations.

For comparison, the BPF with systematic resampling is O(N log N) per timestep but typically requires 10–100× more particles to achieve comparable accuracy under misspecification, making the effective cost comparable.

## 8. Current Configuration

```
Particles:              64–256
Kernel:                 Cauchy
Bandwidth:              Silverman (2·Var/log(N+1)), EMA-smoothed
Repulsion:              Disabled (stein_repulsive_sign = 0)
Temperature:            0.45
Newton damping:         0.95 (target Hessian only, Nk removed)
RMSProp:                ρ = 0.9, ε = 1e-4
Likelihood offset:      0.08
Observation model:      Student-t (ν ≈ 5)
Annealing:              Multi-stage β: 0 → 1
Predict:                Antithetic sampling
Fan mode:               Enabled (equal weights throughout)
```

## 9. Open Directions

1. **Correct Stein transport score interpolation.** Currently, the annealed gradient scales the likelihood score by $\beta$ but does not weight the prior score by $(1 - \beta)$. The proper tempered score is $\nabla \log \pi_\beta = (1 - \beta) \nabla \log \pi_0 + \beta \nabla \log \pi_1$. This small change may improve the annealing path.

2. **Mini-batch SVGD (VP-SVGD).** Randomly subsampling particles for the kernel matrix while updating the remaining particles breaks the deterministic bias identified by Ba et al. This addresses the root cause rather than just mitigating it with noise.

3. **Temperature tuning.** The optimal noise level $\lambda$ is an open question. With repulsion removed, the theoretically justified range may differ from the value tuned with repulsion present. Systematic sweep from 0.3 to 0.7 is warranted.

4. **O(N) per-particle Newton.** Without repulsion and without kernel-smoothed gradient (just per-particle score + noise), the O(N²) kernel matrix is only needed for gradient smoothing. If Newton preconditioning alone provides sufficient regularization, the kernel smoothing may be removable, reducing cost to O(N) per step.

5. **Differentiable parameter learning.** With all operations being smooth (no resampling, no discrete weight updates), the entire filter is differentiable through time. This opens the possibility of learning SV model parameters ($\rho$, $\sigma_\eta$, $\mu$, $\nu$) by backpropagating through the filtering loss—connecting to the differentiable particle filter literature (Corenflos et al., 2021; Brady et al., 2024).

## References

- Ba, J., Erdogdu, M.A., Ghassemi, M., Sun, S., Suzuki, T., Wu, D., Zhang, T. (2022). Understanding the Variance Collapse of SVGD in High Dimensions. ICLR 2022.
- Priser, V. et al. (2025). Long-time asymptotics of noisy SVGD outside the population limit. ICLR 2025.
- Nüsken, N. (2024). Stein transport for Bayesian inference. arXiv:2409.01464.
- He, Y. et al. (2024). Regularized Stein Variational Gradient Flow. Foundations of Computational Mathematics.
- Gan, W. et al. (2025). Kernel Variational Inference Flow for Nonlinear Filtering Problem. arXiv:2509.18589.
- Maken, F.A., Ramos, F., Ott, L. (2022). Stein Particle Filter for Nonlinear, Non-Gaussian State Estimation. arXiv:2106.10568.
- Liu, Q., Wang, D. (2016). Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm. NeurIPS 2016.
- Detommaso, G. et al. (2018). Stein Variational Gradient Descent with Matrix-Valued Kernels. NeurIPS 2018.
- Corenflos, A. et al. (2021). Differentiable Particle Filtering via Entropy-Regularized Optimal Transport. ICML 2021.
- Brady, J.J. et al. (2024). Differentiable Interacting Multiple Model Particle Filtering. arXiv:2410.00620.
