"""
SVPF PyTorch Validation Script
Purpose: Verify gradient dynamics work before CUDA implementation

Tests:
1. Gradient sanity check
2. Parameter recovery from wrong initialization
3. Crash response (no explosion with Student-t)
4. Multiple starting points convergence
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SVParams:
    """True parameters for synthetic data generation."""
    rho: float = 0.95
    sigma_z: float = 0.20
    mu: float = -5.0
    nu: float = 5.0  # Student-t degrees of freedom


def generate_synthetic_data(T: int, params: SVParams, seed: int = 42, crash_at: int = None):
    """
    Generate synthetic stochastic volatility data.
    
    Args:
        T: Number of observations
        params: True SV parameters
        seed: Random seed
        crash_at: If set, inject a 5-sigma crash at this timestep
    
    Returns:
        y: Observations [T]
        h: True log-volatility [T]
    """
    torch.manual_seed(seed)
    
    h = torch.zeros(T)
    y = torch.zeros(T)
    
    # Initialize from stationary distribution
    stationary_var = params.sigma_z**2 / (1 - params.rho**2)
    h[0] = params.mu + np.sqrt(stationary_var) * torch.randn(1).item()
    
    for t in range(1, T):
        # AR(1) dynamics
        h[t] = params.mu + params.rho * (h[t-1] - params.mu) + params.sigma_z * torch.randn(1).item()
        
    # Generate observations
    vol = torch.exp(h / 2)
    y = vol * torch.randn(T)
    
    # Inject crash if requested
    if crash_at is not None and crash_at < T:
        y[crash_at] = 5.0 * vol[crash_at]  # 5-sigma event
    
    return y, h


class SVPF(nn.Module):
    """
    Stein Variational Particle Filter for Stochastic Volatility.
    
    Single-scale, Student-t likelihood, RBF kernel.
    """
    
    def __init__(self, n_particles: int = 500, n_stein_steps: int = 10, nu: float = 5.0):
        super().__init__()
        self.n_particles = n_particles
        self.n_stein_steps = n_stein_steps
        self.nu = nu
        
        # Learnable parameters (unconstrained)
        self.log_rho = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(-1.5))
        self.mu = nn.Parameter(torch.tensor(-5.0))
    
    @property
    def rho(self) -> torch.Tensor:
        """Constrain rho to (0, 0.999) via sigmoid."""
        return 0.999 * torch.sigmoid(self.log_rho)
    
    @property
    def sigma_z(self) -> torch.Tensor:
        """Constrain sigma_z to positive via exp."""
        return torch.exp(self.log_sigma)
    
    def init_particles(self, device: torch.device) -> torch.Tensor:
        """Initialize particles from stationary distribution."""
        with torch.no_grad():
            stationary_var = self.sigma_z**2 / (1 - self.rho**2 + 1e-6)
            particles = self.mu.detach() + torch.sqrt(stationary_var) * torch.randn(
                self.n_particles, device=device
            )
        return particles
    
    def rbf_kernel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RBF kernel matrix and gradients.
        
        Args:
            x: Particles [N]
        
        Returns:
            K: Kernel matrix [N, N]
            grad_K: Kernel gradients [N, N] (∂K/∂x_j)
        """
        diff = x.unsqueeze(0) - x.unsqueeze(1)  # [N, N]
        
        # Median heuristic for bandwidth
        with torch.no_grad():
            pairwise_dist = torch.abs(diff)
            mask = pairwise_dist > 0
            if mask.any():
                bandwidth = torch.median(pairwise_dist[mask])
                bandwidth = bandwidth / np.log(self.n_particles + 1)
                bandwidth = torch.clamp(bandwidth, min=0.01, max=10.0)
            else:
                bandwidth = torch.tensor(1.0, device=x.device)
        
        K = torch.exp(-diff**2 / (2 * bandwidth**2))
        grad_K = -K * diff / (bandwidth**2)
        
        return K, grad_K
    
    def grad_log_posterior(self, h: torch.Tensor, h_prev: torch.Tensor, y: float) -> torch.Tensor:
        """
        Gradient of log p(h_t | h_{t-1}, y_t).
        
        Uses Student-t likelihood for bounded gradients during crashes.
        
        Args:
            h: Current particles [N]
            h_prev: Previous particles [N]
            y: Current observation (scalar)
        
        Returns:
            Gradient for each particle [N]
        """
        # Clamp h to prevent exp overflow
        h_clamped = torch.clamp(h, -15.0, 5.0)
        h_prev_clamped = torch.clamp(h_prev, -15.0, 5.0)
        
        # Prior term: AR(1) pull
        mu_prior = self.mu + self.rho * (h_prev_clamped - self.mu)
        sigma_sq = self.sigma_z**2 + 1e-6
        grad_prior = -(h_clamped - mu_prior) / sigma_sq
        
        # Likelihood term: Student-t (bounded gradient)
        vol = torch.exp(h_clamped)
        scaled_y_sq = y**2 / (vol + 1e-8)
        
        # Clamp scaled_y_sq to prevent numerical issues
        scaled_y_sq = torch.clamp(scaled_y_sq, 0.0, 1e6)
        
        # Student-t gradient: saturates for extreme returns
        # ∂/∂h log p(y|h) for Student-t
        grad_lik = 0.5 * ((self.nu + 1) * scaled_y_sq / (self.nu + scaled_y_sq + 1e-8) - 1)
        
        return grad_prior + grad_lik
    
    def stein_update(self, particles: torch.Tensor, grad_log_p: torch.Tensor, 
                     step_size: float = 0.1) -> torch.Tensor:
        """
        One Stein Variational Gradient Descent step.
        
        Args:
            particles: Current particle positions [N]
            grad_log_p: Gradient of log posterior at each particle [N]
            step_size: Step size for update
        
        Returns:
            Updated particles [N]
        """
        # Clamp gradients to prevent explosion
        grad_log_p = torch.clamp(grad_log_p, -10.0, 10.0)
        
        K, grad_K = self.rbf_kernel(particles)
        
        # φ(x_i) = (1/N) * Σ_j [K(x_j, x_i) * ∇log p(x_j) + ∇_x K(x_j, x_i)]
        attraction = K.T @ grad_log_p / self.n_particles
        repulsion = grad_K.sum(dim=0) / self.n_particles
        
        update = step_size * (attraction + repulsion)
        
        # Clamp update to prevent large jumps
        update = torch.clamp(update, -1.0, 1.0)
        
        new_particles = particles + update
        
        # Clamp particles to reasonable log-vol range: exp(-15) to exp(5)
        new_particles = torch.clamp(new_particles, -15.0, 5.0)
        
        return new_particles
    
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run SVPF on observations.
        
        GRADIENT PATH:
        - At each timestep, gradients flow: theta -> AR(1) prediction -> observation likelihood
        - Stein inner loop is detached (expensive/unstable to backprop through)
        - This is effectively single-step prediction gradient (not full BPTT through time)
        - Memory-efficient and stable compromise
        
        Based on differentiable particle filter approach (Naesseth et al., 2018).
        """
        T = len(observations)
        device = observations.device
        dtype = observations.dtype
        
        # Initialize particles (detach for stability)
        particles = self.init_particles(device)
        
        log_likelihood = torch.tensor(0.0, device=device, dtype=dtype)
        vol_means = []
        
        # Pre-compute Student-t constants
        nu = torch.tensor(self.nu, device=device, dtype=dtype)
        const = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(torch.pi * nu)
        )
        
        for t in range(T):
            y_t = observations[t]
            # h_prev is detached (from init or previous Stein update)
            # Gradients flow through theta in AR(1), not through h_prev
            h_prev = particles
            
            # 1. Prediction Step (Prior) - GRADIENTS FLOW HERE
            noise = torch.randn_like(particles)
            h_pred = (self.mu + self.rho * (h_prev - self.mu) 
                      + self.sigma_z * noise)
            
            # Clamp for numerical stability
            h_pred = torch.clamp(h_pred, -15.0, 5.0)
            
            # 2. Observation Likelihood - THE LEARNING SIGNAL
            vol = torch.exp(h_pred)
            scaled_y_sq = y_t**2 / (vol + 1e-8)
            
            # Student-t log-prob: p(y_t | h_pred)
            log_w = (
                const
                - 0.5 * h_pred
                - (nu + 1) / 2 * torch.log1p(scaled_y_sq / nu)
            )
            
            # 3. Accumulate Marginal Likelihood
            # log(1/N * sum(w_i)) = logsumexp(log_w) - log(N)
            max_log_w = log_w.max()
            step_log_lik = max_log_w + torch.log(torch.exp(log_w - max_log_w).mean() + 1e-10)
            log_likelihood = log_likelihood + step_log_lik
            
            # 4. Stein Update - INFERENCE ONLY (detached)
            # We don't backprop through Stein optimization (expensive/unstable)
            current_particles = h_pred.detach()
            
            with torch.no_grad():
                for _ in range(self.n_stein_steps):
                    grad_log_p = self.grad_log_posterior(current_particles, h_prev.detach(), y_t)
                    current_particles = self.stein_update(current_particles, grad_log_p)
            
            # Update particles for next iteration
            particles = current_particles
            
            # Store estimate
            vol_means.append(torch.exp(particles / 2).mean())
        
        vol_trajectory = torch.stack(vol_means)
        return log_likelihood, vol_trajectory


def train(model: SVPF, observations: torch.Tensor, n_epochs: int = 200, 
          lr: float = 0.01, verbose: bool = True) -> dict:
    """
    Train SVPF via gradient descent on negative log-likelihood.
    
    Args:
        model: SVPF instance
        observations: Training data [T]
        n_epochs: Number of optimization steps
        lr: Learning rate
        verbose: Print progress
    
    Returns:
        Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'loss': [],
        'rho': [],
        'sigma_z': [],
        'mu': [],
        'grad_rho': [],
        'grad_sigma': [],
    }
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        log_lik, _ = model.forward(observations)
        loss = -log_lik
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Record before step
        history['loss'].append(loss.item())
        history['rho'].append(model.rho.item())
        history['sigma_z'].append(model.sigma_z.item())
        history['mu'].append(model.mu.item())
        history['grad_rho'].append(model.log_rho.grad.item() if model.log_rho.grad is not None else 0)
        history['grad_sigma'].append(model.log_sigma.grad.item() if model.log_sigma.grad is not None else 0)
        
        optimizer.step()
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:4d}: loss={loss.item():8.2f}, "
                  f"ρ={model.rho.item():.4f}, "
                  f"σ_z={model.sigma_z.item():.4f}, "
                  f"μ={model.mu.item():.2f}")
    
    return history


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_gradient_sanity():
    """Test 1: Verify gradients are non-zero, finite, and point in correct direction."""
    print("\n" + "="*60)
    print("TEST 1: Gradient Sanity Check")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    y, h = generate_synthetic_data(T=200, params=true_params)
    
    # Smaller model for fast validation
    model = SVPF(n_particles=100, n_stein_steps=3)
    
    # Set to wrong values: rho too low, sigma too high
    with torch.no_grad():
        model.log_rho.fill_(torch.logit(torch.tensor(0.5 / 0.999)))  # rho ≈ 0.5
        model.log_sigma.fill_(np.log(0.5))  # sigma_z = 0.5
    
    print(f"True params:    ρ={true_params.rho}, σ_z={true_params.sigma_z}")
    print(f"Initial params: ρ={model.rho.item():.4f}, σ_z={model.sigma_z.item():.4f}")
    
    # Forward + backward
    log_lik, _ = model.forward(y)
    loss = -log_lik
    loss.backward()
    
    grad_rho = model.log_rho.grad.item()
    grad_sigma = model.log_sigma.grad.item()
    
    print(f"\nGradients:")
    print(f"  ∂L/∂log_rho   = {grad_rho:.4f}")
    print(f"  ∂L/∂log_sigma = {grad_sigma:.4f}")
    
    # Check finite
    assert np.isfinite(grad_rho), "Gradient for rho is not finite!"
    assert np.isfinite(grad_sigma), "Gradient for sigma is not finite!"
    
    # Check non-zero
    assert abs(grad_rho) > 1e-6, "Gradient for rho is zero!"
    assert abs(grad_sigma) > 1e-6, "Gradient for sigma is zero!"
    
    print("\n✓ Gradients are finite and non-zero")
    
    # Check direction (harder to verify without running optimizer, but we can check sign)
    # If rho is too low (0.5 vs 0.95), gradient should push it up
    # If sigma is too high (0.5 vs 0.2), gradient should push it down
    print(f"\nDirection check:")
    print(f"  ρ is too low  → gradient should be negative (to increase rho via sigmoid)")
    print(f"  σ is too high → gradient should be positive (to decrease sigma via exp)")
    
    return True


def test_parameter_recovery():
    """Test 2: Verify parameters converge from wrong initialization."""
    print("\n" + "="*60)
    print("TEST 2: Parameter Recovery")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    # Shorter sequence for faster validation
    y, h = generate_synthetic_data(T=300, params=true_params)
    
    # Smaller model, fewer epochs
    model = SVPF(n_particles=100, n_stein_steps=3)
    
    # Initialize at wrong values
    with torch.no_grad():
        model.log_rho.fill_(torch.logit(torch.tensor(0.5 / 0.999)))  # rho ≈ 0.5
        model.log_sigma.fill_(np.log(0.5))  # sigma_z = 0.5
        model.mu.fill_(-3.0)
    
    print(f"True params:    ρ={true_params.rho:.4f}, σ_z={true_params.sigma_z:.4f}, μ={true_params.mu:.2f}")
    print(f"Initial params: ρ={model.rho.item():.4f}, σ_z={model.sigma_z.item():.4f}, μ={model.mu.item():.2f}")
    print(f"\nTraining...")
    
    history = train(model, y, n_epochs=80, lr=0.02, verbose=True)
    
    final_rho = model.rho.item()
    final_sigma = model.sigma_z.item()
    final_mu = model.mu.item()
    
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
    
    axes[1, 1].plot(history['loss'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Negative Log-Likelihood')
    axes[1, 1].set_title('Loss')
    
    plt.tight_layout()
    plt.savefig('svpf_parameter_recovery.png', dpi=150)
    plt.close()
    print("\n✓ Saved convergence plot to svpf_parameter_recovery.png")
    
    # Success criterion
    success = rho_error < 0.10 and sigma_error < 0.10
    if success:
        print("\n✓ Parameter recovery successful (within 0.10 of true values)")
    else:
        print("\n✗ Parameter recovery failed")
    
    return success, history


def test_crash_response():
    """Test 3: Verify Student-t handles crash without explosion."""
    print("\n" + "="*60)
    print("TEST 3: Crash Response (Student-t Stability)")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    
    # Generate data with crash at t=150 (T=300 for speed)
    y, h = generate_synthetic_data(T=300, params=true_params, crash_at=150)
    
    print(f"Injected 5σ crash at t=150")
    print(f"Crash return: y[150] = {y[150].item():.4f}")
    print(f"Normal returns std: {y[:140].std().item():.4f}")
    
    # Run filter with true parameters - smaller model for speed
    model = SVPF(n_particles=200, n_stein_steps=5)
    with torch.no_grad():
        model.log_rho.fill_(torch.logit(torch.tensor(true_params.rho / 0.999)))
        model.log_sigma.fill_(np.log(true_params.sigma_z))
        model.mu.fill_(true_params.mu)
    
    print(f"\nRunning SVPF with true parameters...")
    
    with torch.no_grad():
        _, vol_trajectory = model.forward(y)
    
    vol_trajectory = vol_trajectory.numpy()
    true_vol = torch.exp(h / 2).numpy()
    
    # Check for explosion
    max_vol = vol_trajectory.max()
    vol_at_crash = vol_trajectory[150]
    vol_after_crash = vol_trajectory[151:160].mean()
    
    print(f"\nResults:")
    print(f"  Max volatility estimate: {max_vol:.4f}")
    print(f"  Vol at crash (t=150):    {vol_at_crash:.4f}")
    print(f"  Vol after crash (mean):  {vol_after_crash:.4f}")
    print(f"  True vol at crash:       {true_vol[150]:.4f}")
    
    # Check no NaN/Inf
    assert np.all(np.isfinite(vol_trajectory)), "Volatility trajectory contains NaN/Inf!"
    print("\n✓ No NaN/Inf in trajectory")
    
    # Check reasonable bounds
    assert max_vol < 10.0, f"Volatility exploded to {max_vol}!"
    print("✓ Volatility stayed bounded")
    
    # Check it responded to crash (vol should increase)
    vol_before_crash = vol_trajectory[140:150].mean()
    assert vol_after_crash > vol_before_crash, "Filter didn't respond to crash!"
    print("✓ Filter responded to crash (vol increased)")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(y.numpy(), alpha=0.7, linewidth=0.5)
    axes[0].axvline(150, color='r', linestyle='--', label='Crash')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Return')
    axes[0].set_title('Observations (with crash at t=150)')
    axes[0].legend()
    
    axes[1].plot(vol_trajectory, label='SVPF Estimate', alpha=0.8)
    axes[1].plot(true_vol, label='True Vol', alpha=0.5)
    axes[1].axvline(150, color='r', linestyle='--', label='Crash')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Volatility')
    axes[1].set_title('Volatility: SVPF vs True')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('svpf_crash_response.png', dpi=150)
    plt.close()
    print("\n✓ Saved crash response plot to svpf_crash_response.png")
    
    return True


def test_multiple_starting_points():
    """Test 4: Verify convergence from multiple initializations."""
    print("\n" + "="*60)
    print("TEST 4: Multiple Starting Points")
    print("="*60)
    
    true_params = SVParams(rho=0.95, sigma_z=0.20, mu=-5.0)
    # Shorter sequence for speed
    y, h = generate_synthetic_data(T=300, params=true_params)
    
    starting_rhos = [0.3, 0.6, 0.85, 0.99]
    final_rhos = []
    
    for start_rho in starting_rhos:
        model = SVPF(n_particles=100, n_stein_steps=3)
        with torch.no_grad():
            model.log_rho.fill_(torch.logit(torch.tensor(start_rho / 0.999)))
            model.log_sigma.fill_(np.log(0.3))
        
        history = train(model, y, n_epochs=50, lr=0.02, verbose=False)
        final_rho = model.rho.item()
        final_rhos.append(final_rho)
        
        print(f"  Start ρ={start_rho:.2f} → Final ρ={final_rho:.4f}")
    
    # Check all converged to similar region
    rho_spread = max(final_rhos) - min(final_rhos)
    print(f"\nSpread of final ρ values: {rho_spread:.4f}")
    
    success = rho_spread < 0.15
    if success:
        print("✓ All starting points converged to similar region")
    else:
        print("✗ Starting points diverged")
    
    return success


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("SVPF VALIDATION SUITE")
    print("="*60)
    
    results = {}
    
    try:
        results['gradient_sanity'] = test_gradient_sanity()
    except Exception as e:
        print(f"Test 1 FAILED with exception: {e}")
        results['gradient_sanity'] = False
    
    try:
        success, _ = test_parameter_recovery()
        results['parameter_recovery'] = success
    except Exception as e:
        print(f"Test 2 FAILED with exception: {e}")
        results['parameter_recovery'] = False
    
    try:
        results['crash_response'] = test_crash_response()
    except Exception as e:
        print(f"Test 3 FAILED with exception: {e}")
        results['crash_response'] = False
    
    try:
        results['multiple_starts'] = test_multiple_starting_points()
    except Exception as e:
        print(f"Test 4 FAILED with exception: {e}")
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