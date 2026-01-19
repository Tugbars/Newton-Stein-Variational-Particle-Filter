"""
SVPF with Parallel Finite Differences for Parameter Learning

Option B: Run multiple SVPFs in parallel with perturbed parameters,
compute numerical gradients. Unbiased, GPU-friendly, robust.

Key insight: GPU doesn't care if you run 1 filter with 3000 particles
or 6 filters with 500 particles. Same SIMD cost.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Detect GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


@dataclass
class SVParams:
    """True parameters for synthetic data generation."""
    rho: float = 0.95
    sigma_z: float = 0.20
    mu: float = -5.0
    nu: float = 5.0


def generate_synthetic_data(T: int, params: SVParams, seed: int = 42):
    """Generate synthetic stochastic volatility data."""
    torch.manual_seed(seed)
    
    h = torch.zeros(T)
    y = torch.zeros(T)
    
    stationary_var = params.sigma_z**2 / (1 - params.rho**2)
    h[0] = params.mu + np.sqrt(stationary_var) * torch.randn(1).item()
    
    for t in range(1, T):
        h[t] = params.mu + params.rho * (h[t-1] - params.mu) + params.sigma_z * torch.randn(1).item()
        
    vol = torch.exp(h / 2)
    y = vol * torch.randn(T)
    
    return y, h


class BatchedSVPF:
    """
    SVPF that runs multiple parameter configurations in parallel.
    
    No autograd needed - we compute gradients via finite differences.
    Shape convention: [batch, particles] where batch = different θ perturbations
    """
    
    def __init__(self, n_particles: int = 500, n_stein_steps: int = 5, nu: float = 5.0):
        self.n_particles = n_particles
        self.n_stein_steps = n_stein_steps
        self.nu = nu
    
    def rbf_kernel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched RBF kernel. x shape: [batch, particles]
        Returns K, grad_K each of shape [batch, particles, particles]
        """
        # x: [B, N] -> diff: [B, N, N]
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        
        # Median heuristic per batch
        pairwise_dist = torch.abs(diff)
        # Get median of non-zero distances for each batch
        mask = pairwise_dist > 0
        
        # Compute bandwidth per batch
        B, N, _ = diff.shape
        bandwidths = torch.zeros(B, device=x.device)
        for b in range(B):
            dists = pairwise_dist[b][mask[b]]
            if len(dists) > 0:
                bandwidths[b] = torch.median(dists)
            else:
                bandwidths[b] = 1.0
        
        bandwidths = torch.clamp(bandwidths, min=0.01, max=10.0)
        bandwidths = bandwidths / np.log(N + 1)
        bandwidths = torch.clamp(bandwidths, min=0.01, max=10.0)
        
        # Reshape for broadcasting: [B, 1, 1]
        h = bandwidths.view(B, 1, 1)
        
        K = torch.exp(-diff**2 / (2 * h**2))
        grad_K = -K * diff / (h**2)
        
        return K, grad_K
    
    def grad_log_posterior(self, h: torch.Tensor, h_prev: torch.Tensor, y: float,
                           rho: torch.Tensor, sigma_z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Batched gradient of log posterior.
        h, h_prev: [batch, particles]
        rho, sigma_z, mu: [batch] or scalar
        Returns: [batch, particles]
        """
        # Ensure params are broadcastable: [B, 1]
        if rho.dim() == 1:
            rho = rho.unsqueeze(1)
            sigma_z = sigma_z.unsqueeze(1)
            mu = mu.unsqueeze(1)
        
        # Clamp for numerical stability
        h = torch.clamp(h, -15.0, 5.0)
        h_prev = torch.clamp(h_prev, -15.0, 5.0)
        
        # Prior term: AR(1)
        mu_prior = mu + rho * (h_prev - mu)
        grad_prior = -(h - mu_prior) / (sigma_z**2 + 1e-8)
        
        # Likelihood term: Student-t
        vol = torch.exp(h)
        scaled_y_sq = y**2 / (vol + 1e-8)
        scaled_y_sq = torch.clamp(scaled_y_sq, 0.0, 1e6)
        
        grad_lik = 0.5 * ((self.nu + 1) * scaled_y_sq / (self.nu + scaled_y_sq + 1e-8) - 1)
        
        return grad_prior + grad_lik
    
    def stein_update(self, particles: torch.Tensor, grad_log_p: torch.Tensor,
                     step_size: float = 0.1) -> torch.Tensor:
        """
        Batched Stein update.
        particles, grad_log_p: [batch, N]
        """
        grad_log_p = torch.clamp(grad_log_p, -10.0, 10.0)
        
        K, grad_K = self.rbf_kernel(particles)
        
        # K: [B, N, N], grad_log_p: [B, N]
        # attraction[b, i] = sum_j K[b, j, i] * grad_log_p[b, j] / N
        attraction = torch.bmm(K.transpose(1, 2), grad_log_p.unsqueeze(2)).squeeze(2) / self.n_particles
        
        # repulsion[b, i] = sum_j grad_K[b, j, i] / N
        repulsion = grad_K.sum(dim=1) / self.n_particles
        
        update = step_size * (attraction + repulsion)
        update = torch.clamp(update, -1.0, 1.0)
        
        new_particles = particles + update
        new_particles = torch.clamp(new_particles, -15.0, 5.0)
        
        return new_particles
    
    def forward(self, observations: torch.Tensor, 
                rho: torch.Tensor, sigma_z: torch.Tensor, mu: torch.Tensor,
                seed: int = None) -> torch.Tensor:
        """
        Run batched SVPF forward pass.
        
        Args:
            observations: [T] returns
            rho, sigma_z, mu: [batch] parameter values for each parallel run
            seed: Optional seed for reproducibility
            
        Returns:
            log_likelihood: [batch] log-likelihood for each parameter configuration
        """
        T = len(observations)
        device = observations.device
        dtype = observations.dtype
        B = len(rho)  # batch size = number of parameter perturbations
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize particles: [batch, particles]
        # Use stationary distribution for each parameter set
        # CRITICAL: Use shared noise across batch (Common Random Numbers)
        stationary_var = sigma_z**2 / (1 - rho**2 + 1e-6)
        shared_init_noise = torch.randn(1, self.n_particles, device=device, dtype=dtype)
        particles = mu.unsqueeze(1) + torch.sqrt(stationary_var).unsqueeze(1) * shared_init_noise.expand(B, -1)
        particles = torch.clamp(particles, -15.0, 5.0)
        
        # Pre-compute Student-t constant
        nu = torch.tensor(self.nu, device=device, dtype=dtype)
        const = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(torch.pi * nu)
        )
        
        log_likelihood = torch.zeros(B, device=device, dtype=dtype)
        
        # Reshape params for broadcasting: [B, 1]
        rho_b = rho.unsqueeze(1)
        sigma_z_b = sigma_z.unsqueeze(1)
        mu_b = mu.unsqueeze(1)
        
        for t in range(T):
            y_t = observations[t]
            h_prev = particles
            
            # Prediction step: AR(1) - batched
            # CRITICAL: Use shared noise across batch (Common Random Numbers)
            # This isolates the effect of Δθ from stochastic variance
            shared_noise = torch.randn(1, self.n_particles, device=device, dtype=dtype)
            noise = shared_noise.expand(B, -1)
            particles = mu_b + rho_b * (h_prev - mu_b) + sigma_z_b * noise
            particles = torch.clamp(particles, -15.0, 5.0)
            
            # Observation likelihood: [B, N]
            vol = torch.exp(particles)
            scaled_y_sq = y_t**2 / (vol + 1e-8)
            
            log_w = (
                const
                - 0.5 * particles
                - (nu + 1) / 2 * torch.log1p(scaled_y_sq / nu)
            )
            
            # Log-mean-exp per batch: [B]
            max_log_w = log_w.max(dim=1, keepdim=True)[0]
            step_log_lik = max_log_w.squeeze() + torch.log(
                torch.exp(log_w - max_log_w).mean(dim=1) + 1e-10
            )
            log_likelihood = log_likelihood + step_log_lik
            
            # Stein update
            for _ in range(self.n_stein_steps):
                grad_log_p = self.grad_log_posterior(particles, h_prev, y_t,
                                                      rho, sigma_z, mu)
                particles = self.stein_update(particles, grad_log_p)
        
        return log_likelihood


class FiniteDiffOptimizer:
    """
    Parameter optimization via parallel finite differences.
    
    Computes gradients numerically by running SVPF with perturbed parameters.
    """
    
    def __init__(self, svpf: BatchedSVPF, delta: float = 0.01):
        """
        Args:
            svpf: BatchedSVPF instance
            delta: Perturbation size for finite differences
        """
        self.svpf = svpf
        self.delta = delta
        
        # Current parameters (unconstrained)
        self.log_rho_unc = 0.0      # Will be transformed to rho
        self.log_sigma = -1.5       # exp -> sigma_z
        self.mu = -5.0
        
        # Adam state
        self.m = {'rho': 0.0, 'sigma': 0.0, 'mu': 0.0}
        self.v = {'rho': 0.0, 'sigma': 0.0, 'mu': 0.0}
        self.t = 0
    
    def _to_constrained(self, log_rho_unc, log_sigma, mu):
        """Convert unconstrained params to constrained."""
        rho = 0.999 * torch.sigmoid(torch.tensor(log_rho_unc))
        sigma_z = torch.exp(torch.tensor(log_sigma))
        mu = torch.tensor(mu)
        return rho.item(), sigma_z.item(), mu.item()
    
    @property
    def params(self):
        """Current constrained parameters."""
        return self._to_constrained(self.log_rho_unc, self.log_sigma, self.mu)
    
    def compute_gradient(self, observations: torch.Tensor, seed: int = None) -> dict:
        """
        Compute gradient via central finite differences.
        
        Runs 6 parallel SVPFs: ±δ for each of 3 parameters.
        
        Returns:
            Dictionary with gradients for each parameter
        """
        device = observations.device
        dtype = observations.dtype
        
        # Current unconstrained params
        p0 = [self.log_rho_unc, self.log_sigma, self.mu]
        
        # Create perturbations: [+δ_rho, -δ_rho, +δ_sigma, -δ_sigma, +δ_mu, -δ_mu]
        perturbations = []
        for i in range(3):
            p_plus = p0.copy()
            p_plus[i] += self.delta
            perturbations.append(p_plus)
            
            p_minus = p0.copy()
            p_minus[i] -= self.delta
            perturbations.append(p_minus)
        
        # Convert to constrained and batch
        rhos = []
        sigmas = []
        mus = []
        
        for p in perturbations:
            rho, sigma, mu = self._to_constrained(*p)
            rhos.append(rho)
            sigmas.append(sigma)
            mus.append(mu)
        
        rho_batch = torch.tensor(rhos, device=device, dtype=dtype)
        sigma_batch = torch.tensor(sigmas, device=device, dtype=dtype)
        mu_batch = torch.tensor(mus, device=device, dtype=dtype)
        
        # Run batched forward pass
        log_liks = self.svpf.forward(observations, rho_batch, sigma_batch, mu_batch, seed=seed)
        
        # Compute central difference gradients
        # grad = (L(θ+δ) - L(θ-δ)) / (2δ)
        # We want to MAXIMIZE log_lik, so gradient points toward higher lik
        grad_rho = (log_liks[0] - log_liks[1]) / (2 * self.delta)
        grad_sigma = (log_liks[2] - log_liks[3]) / (2 * self.delta)
        grad_mu = (log_liks[4] - log_liks[5]) / (2 * self.delta)
        
        return {
            'rho': grad_rho.item(),
            'sigma': grad_sigma.item(),
            'mu': grad_mu.item(),
            'log_lik': log_liks.mean().item()
        }
    
    def step(self, observations: torch.Tensor, lr: float = 0.01, 
             beta1: float = 0.9, beta2: float = 0.999, seed: int = None):
        """
        One optimization step using Adam with finite-diff gradients.
        """
        grads = self.compute_gradient(observations, seed=seed)
        
        self.t += 1
        
        # Adam update for each parameter
        for name, param_name in [('rho', 'log_rho_unc'), ('sigma', 'log_sigma'), ('mu', 'mu')]:
            g = grads[name]
            
            # Update biased first moment estimate
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            # Update biased second moment estimate
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * g**2
            
            # Bias correction
            m_hat = self.m[name] / (1 - beta1**self.t)
            v_hat = self.v[name] / (1 - beta2**self.t)
            
            # Update parameter (MAXIMIZE log_lik, so add gradient)
            update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Clip update for stability
            update = np.clip(update, -0.5, 0.5)
            
            if param_name == 'log_rho_unc':
                self.log_rho_unc += update
            elif param_name == 'log_sigma':
                self.log_sigma += update
            else:
                self.mu += update
        
        return grads


def train_finite_diff(observations: torch.Tensor, n_epochs: int = 200, 
                      lr: float = 0.05, n_particles: int = 300, n_stein_steps: int = 5,
                      init_rho: float = 0.5, init_sigma: float = 0.5, init_mu: float = -3.0,
                      verbose: bool = True) -> tuple:
    """
    Train SVPF using parallel finite differences.
    
    Returns:
        optimizer: Trained optimizer with final params
        history: Training history
    """
    svpf = BatchedSVPF(n_particles=n_particles, n_stein_steps=n_stein_steps)
    optimizer = FiniteDiffOptimizer(svpf, delta=0.02)
    
    # Set initial params (unconstrained)
    # rho = 0.999 * sigmoid(log_rho_unc) -> log_rho_unc = logit(rho / 0.999)
    optimizer.log_rho_unc = torch.logit(torch.tensor(init_rho / 0.999)).item()
    optimizer.log_sigma = np.log(init_sigma)
    optimizer.mu = init_mu
    
    history = {'rho': [], 'sigma_z': [], 'mu': [], 'log_lik': [], 
               'grad_rho': [], 'grad_sigma': [], 'grad_mu': []}
    
    for epoch in range(n_epochs):
        # Use different seed each epoch for variance
        grads = optimizer.step(observations, lr=lr, seed=epoch)
        
        rho, sigma_z, mu = optimizer.params
        
        history['rho'].append(rho)
        history['sigma_z'].append(sigma_z)
        history['mu'].append(mu)
        history['log_lik'].append(grads['log_lik'])
        history['grad_rho'].append(grads['rho'])
        history['grad_sigma'].append(grads['sigma'])
        history['grad_mu'].append(grads['mu'])
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:4d}: LL={grads['log_lik']:8.2f}, "
                  f"ρ={rho:.4f}, σ_z={sigma_z:.4f}, μ={mu:.2f} | "
                  f"∇ρ={grads['rho']:.2f}, ∇σ={grads['sigma']:.2f}")
    
    return optimizer, history


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_gradient_sanity():
    """Test 1: Verify finite-diff gradients are sensible."""
    print("\n" + "="*60)
    print("TEST 1: Finite Difference Gradient Sanity Check")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    y, h = generate_synthetic_data(T=200, params=true_params)
    y = y.to(DEVICE)
    
    svpf = BatchedSVPF(n_particles=200, n_stein_steps=5)
    optimizer = FiniteDiffOptimizer(svpf, delta=0.02)
    
    # Set to wrong values
    optimizer.log_rho_unc = torch.logit(torch.tensor(0.5 / 0.999)).item()
    optimizer.log_sigma = np.log(0.5)
    optimizer.mu = -3.0
    
    print(f"True params:    ρ={true_params.rho}, σ_z={true_params.sigma_z}, μ={true_params.mu}")
    rho, sigma, mu = optimizer.params
    print(f"Initial params: ρ={rho:.4f}, σ_z={sigma:.4f}, μ={mu:.2f}")
    
    grads = optimizer.compute_gradient(y, seed=42)
    
    print(f"\nFinite Difference Gradients:")
    print(f"  ∇ρ   = {grads['rho']:.4f}")
    print(f"  ∇σ   = {grads['sigma']:.4f}")
    print(f"  ∇μ   = {grads['mu']:.4f}")
    print(f"  LL   = {grads['log_lik']:.2f}")
    
    # Check finite
    assert np.isfinite(grads['rho']), "Gradient for rho is not finite!"
    assert np.isfinite(grads['sigma']), "Gradient for sigma is not finite!"
    assert np.isfinite(grads['mu']), "Gradient for mu is not finite!"
    
    print("\n✓ Gradients are finite")
    
    # Check non-zero (at least one should be non-zero)
    assert abs(grads['rho']) > 1e-6 or abs(grads['sigma']) > 1e-6, "All gradients are zero!"
    print("✓ Gradients are non-zero")
    
    return True


def test_parameter_recovery():
    """Test 2: Verify parameters converge from wrong initialization."""
    print("\n" + "="*60)
    print("TEST 2: Parameter Recovery (Finite Differences)")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    y, h = generate_synthetic_data(T=300, params=true_params)
    y = y.to(DEVICE)
    
    print(f"True params:    ρ={true_params.rho:.4f}, σ_z={true_params.sigma_z:.4f}, μ={true_params.mu:.2f}")
    print(f"Initial params: ρ=0.5000, σ_z=0.5000, μ=-3.00")
    print(f"\nTraining...")
    
    optimizer, history = train_finite_diff(
        y, n_epochs=150, lr=0.05, 
        n_particles=300, n_stein_steps=5,
        init_rho=0.5, init_sigma=0.5, init_mu=-3.0,
        verbose=True
    )
    
    final_rho, final_sigma, final_mu = optimizer.params
    
    print(f"\nFinal params:   ρ={final_rho:.4f}, σ_z={final_sigma:.4f}, μ={final_mu:.2f}")
    
    rho_error = abs(final_rho - true_params.rho)
    sigma_error = abs(final_sigma - true_params.sigma_z)
    
    print(f"\nErrors:")
    print(f"  |ρ_est - ρ_true|     = {rho_error:.4f}")
    print(f"  |σ_est - σ_true|     = {sigma_error:.4f}")
    
    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history['rho'], label='Estimated')
    axes[0, 0].axhline(true_params.rho, color='r', linestyle='--', label='True')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('ρ')
    axes[0, 0].legend()
    axes[0, 0].set_title('Persistence (ρ) Convergence')
    
    axes[0, 1].plot(history['sigma_z'], label='Estimated')
    axes[0, 1].axhline(true_params.sigma_z, color='r', linestyle='--', label='True')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('σ_z')
    axes[0, 1].legend()
    axes[0, 1].set_title('Vol-of-Vol (σ_z) Convergence')
    
    axes[1, 0].plot(history['mu'], label='Estimated')
    axes[1, 0].axhline(true_params.mu, color='r', linestyle='--', label='True')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('μ')
    axes[1, 0].legend()
    axes[1, 0].set_title('Long-run Mean (μ) Convergence')
    
    axes[1, 1].plot(history['log_lik'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Log-Likelihood')
    axes[1, 1].set_title('Log-Likelihood')
    
    plt.tight_layout()
    plt.savefig('svpf_finite_diff_recovery.png', dpi=150)
    plt.close()
    print("\n✓ Saved convergence plot to svpf_finite_diff_recovery.png")
    
    # Success criterion - more lenient since finite diff has noise
    success = rho_error < 0.15 and sigma_error < 0.15
    if success:
        print("\n✓ Parameter recovery successful (within 0.15 of true values)")
    else:
        print("\n✗ Parameter recovery failed")
    
    return success, history


def test_multiple_starting_points():
    """Test 3: Verify convergence from multiple initializations."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Starting Points")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    y, h = generate_synthetic_data(T=300, params=true_params)
    y = y.to(DEVICE)
    
    starting_rhos = [0.3, 0.6, 0.85, 0.99]
    final_rhos = []
    
    for start_rho in starting_rhos:
        optimizer, history = train_finite_diff(
            y, n_epochs=100, lr=0.05,
            n_particles=200, n_stein_steps=5,
            init_rho=start_rho, init_sigma=0.3, init_mu=-4.0,
            verbose=False
        )
        final_rho, _, _ = optimizer.params
        final_rhos.append(final_rho)
        
        print(f"  Start ρ={start_rho:.2f} → Final ρ={final_rho:.4f}")
    
    rho_spread = max(final_rhos) - min(final_rhos)
    print(f"\nSpread of final ρ values: {rho_spread:.4f}")
    
    success = rho_spread < 0.20
    if success:
        print("✓ All starting points converged to similar region")
    else:
        print("✗ Starting points diverged")
    
    return success


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("SVPF FINITE DIFFERENCES VALIDATION SUITE")
    print("="*60)
    
    results = {}
    
    try:
        results['gradient_sanity'] = test_gradient_sanity()
    except Exception as e:
        print(f"Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['gradient_sanity'] = False
    
    try:
        success, _ = test_parameter_recovery()
        results['parameter_recovery'] = success
    except Exception as e:
        print(f"Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['parameter_recovery'] = False
    
    try:
        results['multiple_starts'] = test_multiple_starting_points()
    except Exception as e:
        print(f"Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['multiple_starts'] = False
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗"))
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
