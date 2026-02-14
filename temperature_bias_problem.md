# The Temperature Bias Problem in Noisy Newton-SVGD Particle Filters

## Executive Summary

Temperature injection (SVLD) is theoretically necessary to prevent variance collapse in SVGD-based particle filters, but in finite-particle finite-step regimes it introduces systematic bias. This bias cannot be corrected by a simple observation-dependent offset because it depends on the trajectory state (RMSProp history, annealing progress, Newton preconditioning) rather than just the current observation. The critical unresolved question is whether this bias **accumulates across sequential timesteps**, which would make any local correction strategy fundamentally insufficient.

## The Mechanism

### Why Temperature is Necessary

Standard (deterministic) SVGD suffers from variance collapse: the same particles that define the empirical distribution are used to evaluate the driving force $S1 = k(x', x) \nabla \log p(x')$ under that distribution. This circular dependency creates deterministic bias that compounds across iterations, causing convergence to Dirac measures in high dimensions (Ba et al., ICLR 2022).

Adding Langevin noise breaks this deterministic bias:
```cuda
drift = effective_step * phi * precond;
diffusion = sqrt(2 * effective_step * temperature) * noise;
h[i] = h[i] + drift + diffusion;
```

Priser et al. (ICLR 2025) proved that any $T > 0$ avoids variance collapse—the noise provides stochastic exploration that prevents the self-reinforcing collapse of deterministic SVGD.

### Why Temperature Introduces Bias

At finite particle count $N$ and finite annealing steps, noisy SVGD doesn't sample from $p(x)$—it samples from an effective distribution approximately:

$$p_{\text{eff}}(x) \propto p(x)^{1/T_{\text{eff}}}$$

The particles don't fully equilibrate. They're left with **residual noise-induced displacement** that shifts the effective mode.

The bias magnitude depends on the drift-to-diffusion ratio:

$$\frac{\text{drift}}{\text{diffusion}} = \frac{\phi \cdot H^{-1} \cdot v^{-1/2}}{\sqrt{2T}}$$

where:
- $\phi$ = gradient (observation-dependent)
- $H^{-1}$ = Newton preconditioner (curvature-dependent)
- $v^{-1/2}$ = RMSProp normalizer (trajectory history-dependent)
- $T$ = temperature (constant)

## Observation-Dependent Bias Variation

The likelihood gradient magnitude scales with the observation:

```cuda
float A = (y_t * y_t) / (exp(h) * nu);
float raw_grad = -0.5 + 0.5 * (nu + 1) * A / (1 + A);
```

**Small $|y_t|$**: Likelihood is flat → gradient ≈ -0.5 → diffusion dominates → particles wander  
**Large $|y_t|$**: Likelihood is steep → gradient large → drift dominates → minimal bias

This creates observation-dependent bias:
- Calm markets (small returns) → under-informed likelihood → temperature noise pushes particles around aimlessly
- Volatile markets (large returns) → strong likelihood signal → temperature noise is small relative to gradient

## The lik_offset Correction

```cuda
grad_lik = raw_grad - lik_offset;  // lik_offset = 0.09
```

This **subtracts** from the likelihood score, pulling the gradient downward → pushing particles toward lower log-volatility.

The offset of 0.09 was tuned empirically to correct the trajectory-averaged bias across typical market conditions. It works "on average" but cannot adapt to:
- Individual observation magnitude
- Current trajectory state (RMSProp accumulator values)
- Position in annealing schedule
- Particle cloud geometry

## Why Adaptive lik_offset Failed

An observation-dependent correction `offset(y_t)` cannot work because the bias depends on **trajectory state**, not just the current observation:

### 1. Multi-Step Annealing Compounds Diffusion Non-Linearly

With 4 stages × 3 steps = 12 Stein iterations per observation:
- Cumulative diffusion: $\sqrt{12} \times$ single-step noise
- Cumulative drift: $12 \times$ average gradient (but gradient changes each step due to annealing)

The ratio evolves as $\beta$ increases from 0.25 → 1.0. Early stages (weak likelihood) accumulate more noise-induced bias than late stages.

### 2. RMSProp Carries Historical Information

```cuda
v_new = rho_rmsprop * v_prev + (1 - rho_rmsprop) * phi * phi;
precond = 1 / sqrt(v_new + epsilon);
```

The normalizer $v_i$ at timestep $t$ depends on gradients from $t-1, t-2, \ldots$

A volatility spike at $t-1$ inflates $v$, suppressing drift at $t$ even if $y_t$ is small. The effective step size is **trajectory-history-dependent**, making the bias non-Markovian.

### 3. Newton Preconditioning Depends on Particle Geometry

```cuda
H_weighted = sum(hess[j] * K(h[i], h[j])) / sum(K(h[i], h[j]));
```

The kernel-smoothed Hessian depends on inter-particle distances. If temperature noise has scattered particles, the Hessian estimate changes, altering the transport direction—creating a feedback loop between noise-induced displacement and curvature correction.

### 4. Closed-Loop Correction

The offset itself changes the gradient:
```
grad → Newton(grad) → RMSProp(Newton(grad)) → drift
```

Changing the gradient changes the Hessian, which changes Newton preconditioning, which changes RMSProp's update—the system is a closed loop where the correction propagates through all preconditioners.

## The Critical Diagnostic Question

**Does temperature bias accumulate across sequential timesteps?**

### Hypothesis A: Local Bias (No Accumulation)

Each observation introduces bias, but starting from the correct distribution at $t$ produces the same bias at $t+1$ regardless of history.

**Implication**: A better static offset (or observation-dependent offset with correct functional form) could fix the problem.

### Hypothesis B: Compounding Bias (Accumulation)

Bias at $t$ shifts particle positions → particles at $t+1$ start from biased positions → bias at $t+1$ includes both local bias plus propagated bias from $t$ → exponential compounding.

**Implication**: No local correction can fix this. The filter requires either:
1. Periodic rejuvenation (reset to unbiased distribution)
2. State-dependent offset tracking cumulative drift
3. Reduced temperature (accepting some variance collapse)
4. Different diversity mechanism (mini-batch SVGD, tempering)

## Proposed Diagnostic Test

Run two sequences on identical data:

**Sequence A (Standard):**
- Initialize particles from stationary distribution at $t=0$
- Run filter forward for $T$ steps
- Measure bias at $t = 1, 10, 20, 50, 100$

**Sequence B (Reset):**
- Initialize from stationary distribution at $t=0$
- Run until $t=50$
- **Reset particles to ground truth $h_{50}$** (from true DGP or oracle filter)
- Run from $t=50$ to $t=51$
- Measure bias at $t=51$

**Diagnostic:**
```
If bias(t=51 | reset at 50) ≈ bias(t=1):
    → Local bias (Hypothesis A)
    → Offset correction is feasible
    
If bias(t=51 | reset at 50) << bias(t=51 | no reset):
    → Compounding bias (Hypothesis B)
    → Offset correction is insufficient
    → Need trajectory-level solution
```

### Implementation Sketch

```cuda
// Ground truth oracle (for diagnostic only)
float h_true[n];
generate_from_true_dgp(h_true, t=50);

// Sequence A: bias at t=51 with history
float bias_A = measure_bias(filter_state_with_history, y[51], h_true_51);

// Sequence B: bias at t=51 after reset
cudaMemcpy(filter->h, h_true, n * sizeof(float), H2D);  // Reset to oracle
float bias_B = measure_bias(filter_state_after_reset, y[51], h_true_51);

// Diagnostic ratio
float accumulation_factor = bias_A / bias_B;
```

If `accumulation_factor > 1.5`, bias is compounding. If `accumulation_factor ≈ 1.0`, bias is local.

## Implications for Each Outcome

### If Bias is Local

**Solutions in order of engineering effort:**
1. **Observation-scaled offset**: `offset = base_offset * f(|y_t|, variance(h))`
2. **Annealing-stage-dependent offset**: Different correction at $\beta = 0.25$ vs $\beta = 1.0$
3. **RMSProp-state-dependent offset**: `offset = g(v_mean, volatility_regime)`

All require empirical tuning but are implementable within current architecture.

### If Bias Compounds

**Solutions require architectural changes:**

1. **Mini-batch SVGD (VP-SVGD)**  
   Randomly subsample 50% of particles for kernel matrix, update the other 50%. Breaks the deterministic bias at the root cause rather than compensating with noise. May allow lower temperature → less bias.

2. **Adaptive temperature schedule**  
   Start high ($T = 0.6$) for diversity, anneal down ($T \to 0.2$) as particles converge. Reduces accumulated noise at late stages.

3. **Periodic rejuvenation via resampling**  
   Every 50 steps, resample particles according to importance weights (breaks equal-weight assumption but resets bias accumulation).

4. **Differentiable parameter learning**  
   If bias is structural (wrong $\rho$, $\sigma_\eta$, $\mu$), backprop through filter to learn corrected parameters. May eliminate bias at the source rather than compensating.

5. **Kernel-free transport**  
   Remove kernel smoothing entirely (O(N²) → O(N)), rely on Newton + RMSProp alone. Without kernel smoothing, no inter-particle coupling remains—bias from temperature only, no spatial correlation.

## Current Status

- **Temperature**: 0.45 (tuned empirically)
- **lik_offset**: 0.09 (trajectory-averaged correction)
- **Observation**: Bias varies with $|y_t|$, adaptive offset failed
- **Unknown**: Does bias compound across timesteps?

**Next action**: Run diagnostic test to determine local vs. compounding bias, then select appropriate correction strategy.

---

## References

- Ba, J. et al. (2022). Understanding the Variance Collapse of SVGD in High Dimensions. ICLR.
- Priser, V. et al. (2025). Long-time asymptotics of noisy SVGD outside the population limit. ICLR.
- Nüsken, N. (2024). Stein transport for Bayesian inference. arXiv:2409.01464.
