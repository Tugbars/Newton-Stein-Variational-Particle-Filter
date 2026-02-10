# SVPF: Noisy Newton-SVGD Particle Filter for Stochastic Volatility

A high-performance CUDA particle filter for real-time volatility tracking, combining Newton-preconditioned Stein Variational Gradient Descent with Langevin noise injection for provably correct posterior approximation under model misspecification.

[![CUDA](https://img.shields.io/badge/CUDA-12+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

<img width="4170" height="2370" alt="529603395-23dc2d9c-f92a-4b3c-8fe0-73b165dde93d" src="https://github.com/user-attachments/assets/3490999d-1356-4cea-badb-bf3bb4c7a67c" />

## Performance

| Metric | Value |
|--------|-------|
| **Latency** | ~172 μs/step |
| **Particles** | 512 |

```
Scenario            RMSE (log-vol)
────────────────────────────────────
calm                0.334
crisis              0.568
spike               0.467
regime_shift        0.334
asymmetric          0.382
jumps               0.384
fat_tails           0.432
────────────────────────────────────
```

Under model misspecification (wrong parameters, regime changes), SVPF consistently outperforms the bootstrap particle filter:

```
Scenario       SVPF     BPF
──────────────────────────────
Spike          0.69     1.18
Regime         1.77     2.26
──────────────────────────────
```

---

## The Problem

Volatility is latent — we never observe it directly:

```
Hidden:     h_t = μ + ρ(h_{t-1} − μ) + σ_z·ε_t       (log-volatility, AR(1))
Observed:   y_t = exp(h_t/2)·η_t,  η_t ~ Student-t(ν)  (returns)
```

The challenge: infer `h_t` from `y_t` in real-time, handling regime changes, fat-tailed returns, and model misspecification.

---

## Approach

Standard SVGD (Liu & Wang, 2016) transports particles toward the posterior via a kernel-smoothed gradient field. It has two terms: an **attractive** term that pulls particles toward high-probability regions, and a **repulsive** term that maintains diversity.

This works poorly for sequential filtering. Ba et al. (2022) showed that the attractive term (S1) carries estimation error that scales with dimension, and its use of the *same* particles to both evaluate the score and define the transport creates deterministic bias. The repulsive term (S2) introduces further circular dependencies — particle positions influence inter-particle forces that act on those same particles.

Our implementation addresses these issues by combining insights from several lines of work into a method we call the **Noisy Newton-SVGD Particle Filter**:

1. **Remove repulsion entirely.** Repulsive forces encode circular inter-particle dependencies that compound across timesteps in sequential filtering. Removing them reduces bias by 27–38% in our experiments, consistent with Ba et al.'s analysis that S2 is not the term maintaining diversity.

2. **Add Langevin noise (temperature).** Priser et al. (2025) proved that adding `√(2ε·T)·ξ` noise to each SVGD step prevents variance collapse — the limit set is well-defined and approaches the target as N increases. This replaces repulsion as the diversity mechanism. Critically, noise pushes particles in random directions independent of the cloud (zero expected bias), while repulsion direction depends on the current cloud (systematic bias).

3. **Newton preconditioning on target Hessian only.** We precondition the gradient with the kernel-weighted average of per-particle Hessians (Detommaso et al., 2018), but exclude the kernel geometry term Nk. The Nk term inflates curvature near particle clusters via inter-particle distances — another circular dependency that makes the system conservative exactly when misspecification demands aggressive movement. Removing it yields +10% accuracy under misspecification.

4. **Leave-one-out kernel smoothing.** Each particle's self-kernel K(h_i, h_i) = 1 is the maximum-weight contributor to its own transport field — the most direct S1 circular dependency. We zero this contribution via a branchless mask, breaking the self-interaction without splitting the cloud.

5. **Tempered likelihood path (β-annealing).** We ramp the likelihood contribution from β=0.3 to β=1.0 across multiple stages, letting particles explore before committing. This is related to Stein transport (Nüsken, 2024) where particles follow a predefined tempered path from prior to posterior.

The guiding principle: **every removal of inter-particle coupling improved accuracy; every removal of per-particle mechanics degraded it.** Temperature, Newton preconditioning, RMSProp step bounding — all operate on individual particles using only the target density, breaking the feedback loop that causes bias accumulation in sequential filtering.

---

## Algorithm

```
┌──────────────────────────────────────────────────────────────────┐
│                     SVPF STEP (per tick)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PREDICT        h ~ p(h_t | h_{t-1})                         │
│     └── Antithetic sampling: paired (+ε, −ε) for variance       │
│                          ↓                                       │
│  2. GUIDE          Nudge toward EKF posterior estimate           │
│                          ↓                                       │
│  3. STEIN LOOP     For β in [0.3 → 1.0]:                        │
│     │   a. Gradient  ∇log p(h|y) with Student-t likelihood      │
│     │   b. Newton    H⁻¹ · ∇  (target Hessian, no Nk)          │
│     │   c. Kernel    LOO kernel-smoothed score (Cauchy/IMQ)      │
│     │   d. RMSProp   Bound step magnitude adaptively             │
│     │   e. Step      h += drift + √(2·ε·T)·noise                │
│     └── KSD measured on final iteration                          │
│                          ↓                                       │
│  4. OUTPUT         vol = mean(exp(h/2))                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Why No Repulsion?

Standard SVGD's update is `φ(x) = E[K·∇log p] + E[∇K]`, where the second term is repulsive. In static Bayesian inference with many particles, this converges. In sequential filtering with finite particles, repulsive forces introduce systematic bias that compounds across timesteps:

- Repulsion direction depends on the current particle cloud → circular dependency
- Repulsion pushes particles away from each other, not toward the target
- With 512 particles in 1D, the kernel-smoothed score is already accurate without diversity forcing

Removing repulsion and relying on Langevin noise for diversity gives 27–38% bias reduction across all test scenarios.

### Why Temperature Is Critical

Without temperature noise, RMSE doubles. The predict step adds noise once per timestep, then 8+ Stein iterations all pull toward the mode. Without noise counteracting at every step, the cloud collapses to a point mass. Temperature is unbiased diversity — it pushes in random directions (zero expected bias, variance averages over N particles), unlike repulsion which pushes in cloud-dependent directions (systematic bias).

Priser et al. (2025) formally proved this: any λ > 0 in `√(2ε·λ)·ξ` prevents variance collapse. We use T = 0.45.

### Why Newton + RMSProp (Two Preconditioners)

Newton (H⁻¹) sets direction and relative scaling based on target curvature. RMSProp bounds absolute step magnitude by tracking EMA of squared transport. They are complementary, not redundant: when the transport spikes after a volatility event, Newton scales by 1/H but H may not be large enough to prevent overshoot. RMSProp catches this via accumulated history. Removing RMSProp destroys accuracy.

### Why Cauchy (IMQ) Kernel

The inverse multiquadric kernel `K = 1/(1 + r²/bw²)` has polynomial tail decay vs Gaussian's exponential. During volatility spikes, outlier particles remain connected to the main cloud — their kernel weights decay as 1/r² rather than exp(−r²). This is critical for regime tracking where the posterior can shift several bandwidths in a single step.

### Why Leave-One-Out

Each particle's self-kernel K(h_i, h_i) = 1 is the maximum possible weight. Including it means particle i has outsized influence on its own update — the most direct form of Ba et al.'s S1 circular dependency. Zeroing it via `mask = (float)(j != i)` is branchless and removes the strongest self-interaction. At N=512 the effect is small (1/512 of the sum) but measurable.

---

## Supporting Components

### EKF Guide Density
A parallel Extended Kalman Filter finds the approximate posterior mode and nudges particles to the right neighborhood. The EKF is wrong (assumes Gaussian), but gets particles close; Stein transport then refines the shape. The guide uses a variance-preserving shift — it moves the cloud mean without contracting the spread.

### Adaptive μ Learning
A 1D Kalman filter tracks the long-run mean level μ online. This is a lightweight form of parameter learning that helps during regime shifts when the true μ changes.

### Antithetic Sampling
The predict step pairs particles with (+ε, −ε) noise, halving the variance of the particle mean estimate at zero computational cost.

### Student-t Likelihood
Using Student-t(ν) observations with ν ≈ 5 bounds likelihood gradients during market crashes. Gaussian likelihood produces unbounded gradients on extreme returns, causing numerical instability. The Student-t likelihood is smooth, differentiable, and robust.

### KSD Diagnostics
Kernel Stein Discrepancy is computed on the final Stein iteration to measure convergence quality. Used for adaptive step count decisions and rejuvenation triggering.

---

## Theoretical Foundations

This implementation sits at the intersection of several lines of work:

| Component | Theoretical basis | Key insight |
|-----------|------------------|-------------|
| Remove repulsion | Ba et al. 2022 | S1 carries the problematic bias; S2 isn't needed when noise handles diversity |
| Langevin temperature | Priser et al. 2025 | Noisy SVGD avoids variance collapse; any T > 0 suffices |
| β-annealing | Nüsken 2024 | Stein transport: tempered path from prior to posterior |
| Newton preconditioning | Detommaso et al. 2018 | Hessian-scaled Stein steps; we use target Hessian only |
| Leave-one-out | Ba et al. 2022 | Breaking direct self-interaction in S1 term |
| IMQ kernel | Gorham & Mackey 2017 | Polynomial tails for heavy-tailed targets |

### Ablation Results

Each component's contribution, measured on misspecification scenarios:

| Change | Effect |
|--------|--------|
| Remove repulsion | −27–38% bias |
| Remove Nk (kernel Hessian geometry) | +10% misspec accuracy |
| Remove temperature | +100% RMSE (doubles) |
| Remove RMSProp | Accuracy destroyed |
| Remove Newton | Convergence slows 3–4× |
| lik_offset 0.325 → 0.08 | Cleaner (was compensating for repulsion artifacts) |
| Add LOO masking | Slight additional improvement |

---

## References

- Ba, J., Erdogdu, M. A., Ghassemi, M., Sun, S., Suzuki, T., Wu, D., & Zhang, T. (2022). Understanding the Variance Collapse of SVGD in the Kernel Regime. *ICLR 2022*.
- Priser, R., Ye, N., & Bhatt, U. (2025). Noisy SVGD Avoids Variance Collapse. *ICLR 2025*.
- Nüsken, N. (2024). Stein Transport for Bayesian Inference. *arXiv:2409.01464*.
- He, Y., Balasubramanian, K., & Bhatt, U. (2024). Regularized Stein Variational Gradient Flow. *arXiv:2211.07861*.
- Liu, Q. & Wang, D. (2016). Stein Variational Gradient Descent. *NeurIPS 2016*.
- Detommaso, G., Cui, T., Spantini, A., Marzouk, Y., & Scheichl, R. (2018). A Stein Variational Newton Method. *NeurIPS 2018*.
- Fan, Y., Buchanan, S. B., Goyal, A., & Azizian, W. (2021). Stein Particle Filter. *arXiv:2106.10568*.
- Maken, F. A., Ramos, F., & Ott, L. (2022). Stein Particle Filter for Nonlinear, Non-Gaussian State Estimation. *IEEE RA-L*.
- Gorham, J. & Mackey, L. (2017). Measuring Sample Quality with Kernels. *ICML 2017*.
- Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021). Differentiable Particle Filtering. *ICML 2021*.

---

## Usage

```c
SVPFState* filter = svpf_create(512, 8, 5.0f, stream);
svpf_initialize(filter, &params, seed);

for (int t = 0; t < n_ticks; t++) {
    svpf_step_graph(filter, returns[t], returns[t-1], &params,
                    &loglik, &vol, &h_mean);
}

svpf_destroy(filter);
```

### Configuration

```c
// Core (these matter most)
state->stein_repulsive_sign = 0;    // No repulsion
state->use_fan_mode = 1;            // Uniform weights
state->temperature = 0.45f;         // Langevin noise (critical)
state->use_newton = 1;              // Newton preconditioning
state->use_split_batch = 1;         // Leave-one-out
state->lik_offset = 0.08f;          // Bias correction

// Supporting
state->use_guide = 1;               // EKF guide density
state->use_adaptive_mu = 1;         // Online μ learning
state->use_antithetic = 1;          // Paired noise
state->use_student_t_state = 1;     // Student-t state dynamics
state->nu = 5.0f;                   // Observation DoF
```

---

## License

MIT
