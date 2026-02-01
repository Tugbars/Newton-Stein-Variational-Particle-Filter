"""
SVPF with Parallel Finite Differences for Parameter Learning

2-PARAMETER VERSION: Learns (ρ, μ), fixes σ_z

Key insight: Even with finite differences + truncated windows,
Stein transport eventually absorbs σ_z. So we fix it and let
the adaptive code in production handle vol-of-vol.

Truncated Windows fix particle degeneracy:
- Short window (50) → Particles don't degenerate
- No degeneracy → Likelihood surface is smooth  
- Smooth surface → Finite Difference gradients are accurate
- Result: ρ and μ converge reliably
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from time import time

# Detect GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


@dataclass
class SVParams:
    """True parameters for synthetic data generation."""
    rho: float = 0.94
    sigma_z: float = 0.18
    mu: float = -3.2
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
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        B, N, _ = diff.shape
        
        # Vectorized median heuristic
        pairwise_dist = torch.abs(diff)
        flat_dists = pairwise_dist.reshape(B, -1)
        sorted_dists, _ = flat_dists.sort(dim=1)
        median_idx = int(0.75 * N * N)
        bandwidths = sorted_dists[:, median_idx]
        
        bandwidths = torch.clamp(bandwidths, min=0.01, max=10.0)
        bandwidths = bandwidths / np.log(N + 1)
        bandwidths = torch.clamp(bandwidths, min=0.01, max=10.0)
        
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
        if rho.dim() == 1:
            rho = rho.unsqueeze(1)
            sigma_z = sigma_z.unsqueeze(1)
            mu = mu.unsqueeze(1)
        
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
        """Batched Stein update."""
        grad_log_p = torch.clamp(grad_log_p, -10.0, 10.0)
        
        K, grad_K = self.rbf_kernel(particles)
        
        attraction = torch.bmm(K.transpose(1, 2), grad_log_p.unsqueeze(2)).squeeze(2) / self.n_particles
        repulsion = grad_K.sum(dim=1) / self.n_particles
        
        update = step_size * (attraction + repulsion)
        update = torch.clamp(update, -1.0, 1.0)
        
        new_particles = particles + update
        new_particles = torch.clamp(new_particles, -15.0, 5.0)
        
        return new_particles
    
    def forward(self, observations: torch.Tensor, 
                rho: torch.Tensor, sigma_z: torch.Tensor, mu: torch.Tensor,
                seed: int = None,
                init_particles: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run batched SVPF forward pass.
        
        Args:
            observations: [T] returns
            rho, sigma_z, mu: [batch] parameter values for each parallel run
            seed: Optional seed for reproducibility
            init_particles: Optional [batch, particles] initial state
            
        Returns:
            log_likelihood: [batch] log-likelihood for each parameter configuration
            final_particles: [batch, particles] final particle states
        """
        T = len(observations)
        device = observations.device
        dtype = observations.dtype
        B = len(rho)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize particles
        if init_particles is not None:
            particles = init_particles.clone()
        else:
            # Use stationary distribution with COMMON RANDOM NUMBERS
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
        
        # Reshape params for broadcasting
        rho_b = rho.unsqueeze(1)
        sigma_z_b = sigma_z.unsqueeze(1)
        mu_b = mu.unsqueeze(1)
        
        for t in range(T):
            y_t = observations[t]
            h_prev = particles
            
            # Prediction step with COMMON RANDOM NUMBERS
            shared_noise = torch.randn(1, self.n_particles, device=device, dtype=dtype)
            noise = shared_noise.expand(B, -1)
            particles = mu_b + rho_b * (h_prev - mu_b) + sigma_z_b * noise
            particles = torch.clamp(particles, -15.0, 5.0)
            
            # Observation likelihood BEFORE transport (critical for gradient)
            vol = torch.exp(particles)
            scaled_y_sq = y_t**2 / (vol + 1e-8)
            
            log_w = (
                const
                - 0.5 * particles
                - (nu + 1) / 2 * torch.log1p(scaled_y_sq / nu)
            )
            
            # Log-mean-exp per batch
            max_log_w = log_w.max(dim=1, keepdim=True)[0]
            step_log_lik = max_log_w.squeeze() + torch.log(
                torch.exp(log_w - max_log_w).mean(dim=1) + 1e-10
            )
            log_likelihood = log_likelihood + step_log_lik
            
            # Stein update AFTER likelihood calculation
            for _ in range(self.n_stein_steps):
                grad_log_p = self.grad_log_posterior(particles, h_prev, y_t,
                                                      rho, sigma_z, mu)
                particles = self.stein_update(particles, grad_log_p)
        
        return log_likelihood, particles


class FiniteDiffOptimizer:
    """
    Parameter optimization via parallel finite differences.
    
    Learns: ρ, μ
    Fixed: σ_z (Stein absorbs it even with finite diff)
    
    Computes gradients numerically by running SVPF with perturbed parameters.
    """
    
    def __init__(self, svpf: BatchedSVPF, delta: float = 0.01, fixed_sigma_z: float = 0.15):
        self.svpf = svpf
        self.delta = delta
        self.fixed_sigma_z = fixed_sigma_z  # FIXED, not learned
        
        # Current parameters (unconstrained) - only rho and mu
        self.log_rho_unc = 0.0
        self.mu = -5.0
        
        # Adam state - only for rho and mu
        self.m = {'rho': 0.0, 'mu': 0.0}
        self.v = {'rho': 0.0, 'mu': 0.0}
        self.t = 0
    
    def _to_constrained(self, log_rho_unc, mu):
        """Convert unconstrained params to constrained."""
        rho = 0.999 * torch.sigmoid(torch.tensor(log_rho_unc))
        mu = torch.tensor(mu)
        return rho.item(), mu.item()
    
    @property
    def params(self):
        """Current constrained parameters."""
        rho, mu = self._to_constrained(self.log_rho_unc, self.mu)
        return rho, self.fixed_sigma_z, mu  # sigma_z is fixed
    
    def compute_gradient(self, observations: torch.Tensor, seed: int = None,
                        init_particles: torch.Tensor = None) -> dict:
        """
        Compute gradient via central finite differences.
        
        Runs 4 parallel SVPFs: ±δ for rho and mu (sigma_z is fixed).
        """
        device = observations.device
        dtype = observations.dtype
        
        p0 = [self.log_rho_unc, self.mu]
        
        # Create perturbations for rho and mu only
        perturbations = []
        for i in range(2):  # Only 2 parameters now
            p_plus = p0.copy()
            p_plus[i] += self.delta
            perturbations.append(p_plus)
            
            p_minus = p0.copy()
            p_minus[i] -= self.delta
            perturbations.append(p_minus)
        
        # Convert to constrained and batch
        rhos, sigmas, mus = [], [], []
        for p in perturbations:
            rho, mu = self._to_constrained(*p)
            rhos.append(rho)
            sigmas.append(self.fixed_sigma_z)  # Always fixed
            mus.append(mu)
        
        rho_batch = torch.tensor(rhos, device=device, dtype=dtype)
        sigma_batch = torch.tensor(sigmas, device=device, dtype=dtype)
        mu_batch = torch.tensor(mus, device=device, dtype=dtype)
        
        # Expand init_particles if provided
        if init_particles is not None:
            init_particles = init_particles.unsqueeze(0).expand(4, -1).clone()  # 4 perturbations now
        
        # Run batched forward pass
        log_liks, _ = self.svpf.forward(observations, rho_batch, sigma_batch, mu_batch, 
                                        seed=seed, init_particles=init_particles)
        
        # Central difference gradients
        grad_rho = (log_liks[0] - log_liks[1]) / (2 * self.delta)
        grad_mu = (log_liks[2] - log_liks[3]) / (2 * self.delta)
        
        return {
            'rho': grad_rho.item(),
            'mu': grad_mu.item(),
            'log_lik': log_liks.mean().item()
        }
    
    def step(self, observations: torch.Tensor, lr: float = 0.01, 
             beta1: float = 0.9, beta2: float = 0.999, seed: int = None,
             init_particles: torch.Tensor = None):
        """One optimization step using Adam with finite-diff gradients."""
        grads = self.compute_gradient(observations, seed=seed, init_particles=init_particles)
        
        self.t += 1
        
        for name, param_name in [('rho', 'log_rho_unc'), ('mu', 'mu')]:
            g = grads[name]
            
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * g**2
            
            m_hat = self.m[name] / (1 - beta1**self.t)
            v_hat = self.v[name] / (1 - beta2**self.t)
            
            update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            update = np.clip(update, -0.5, 0.5)
            
            if param_name == 'log_rho_unc':
                self.log_rho_unc += update
            else:
                self.mu += update
        
        return grads


def train_finite_diff_truncated(
    observations: torch.Tensor, 
    true_params: SVParams,
    n_epochs: int = 200, 
    lr: float = 0.05, 
    n_particles: int = 300, 
    n_stein_steps: int = 5,
    init_rho: float = 0.5, 
    fixed_sigma_z: float = 0.15,   # FIXED, not learned
    init_mu: float = -4.0,
    trunc_len: int = 50,           # SHORT windows to prevent degeneracy
    windows_per_epoch: int = 10,   # SGD-style updates
    verbose: bool = True
) -> tuple:
    """
    Train SVPF using parallel finite differences with TRUNCATED WINDOWS.
    
    Learns: ρ, μ
    Fixed: σ_z (Stein transport absorbs it)
    
    Returns:
        optimizer: Trained optimizer with final params
        history: Training history
    """
    svpf = BatchedSVPF(n_particles=n_particles, n_stein_steps=n_stein_steps, nu=true_params.nu)
    optimizer = FiniteDiffOptimizer(svpf, delta=0.02, fixed_sigma_z=fixed_sigma_z)
    
    # Set initial params (unconstrained) - only rho and mu
    optimizer.log_rho_unc = torch.logit(torch.tensor(init_rho / 0.999)).item()
    optimizer.mu = init_mu
    
    history = {'rho': [], 'sigma_z': [], 'mu': [], 'log_lik': [], 
               'grad_rho': [], 'grad_mu': []}
    
    T = len(observations)
    device = observations.device
    
    if verbose:
        print(f"\n{'='*70}")
        print("SVPF FINITE DIFFERENCES - 2-PARAMETER LEARNING (ρ, μ)")
        print(f"{'='*70}")
        print(f"Data:       T={T}, window_len={trunc_len}, windows/epoch={windows_per_epoch}")
        print(f"Particles:  {n_particles}, Stein steps: {n_stein_steps}")
        print(f"True:       ρ={true_params.rho:.3f}, μ={true_params.mu:.2f}")
        print(f"Init:       ρ={init_rho:.3f}, μ={init_mu:.2f}")
        print(f"Fixed:      σ_z={fixed_sigma_z:.3f} (true: {true_params.sigma_z:.3f})")
        print(f"{'─'*70}")
    
    t0 = time()
    
    for epoch in range(n_epochs):
        epoch_ll = 0.0
        epoch_grads = {'rho': 0.0, 'mu': 0.0}
        
        # SGD-style: sample random windows
        for w in range(windows_per_epoch):
            # Random window start
            t_start = np.random.randint(0, max(1, T - trunc_len))
            y_window = observations[t_start : t_start + trunc_len]
            
            # Compute gradient on SHORT window (particles stay healthy)
            seed = epoch * 1000 + w
            grads = optimizer.step(y_window, lr=lr, seed=seed)
            
            epoch_ll += grads['log_lik']
            epoch_grads['rho'] += grads['rho']
            epoch_grads['mu'] += grads['mu']
        
        # Average over windows
        epoch_ll /= windows_per_epoch
        for k in epoch_grads:
            epoch_grads[k] /= windows_per_epoch
        
        rho, sigma_z, mu = optimizer.params
        
        history['rho'].append(rho)
        history['sigma_z'].append(sigma_z)  # Will be constant
        history['mu'].append(mu)
        history['log_lik'].append(epoch_ll)
        history['grad_rho'].append(epoch_grads['rho'])
        history['grad_mu'].append(epoch_grads['mu'])
        
        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            elapsed = time() - t0
            print(f"Epoch {epoch:4d} ({elapsed:5.1f}s): LL={epoch_ll:8.2f}, "
                  f"ρ={rho:.4f} ({true_params.rho:.3f}), "
                  f"μ={mu:.2f} ({true_params.mu:.1f})")
    
    if verbose:
        print(f"{'─'*70}")
        rho, sigma_z, mu = optimizer.params
        print(f"Final:      ρ={rho:.4f}, μ={mu:.2f}")
        print(f"Fixed:      σ_z={sigma_z:.4f}")
        
        # Compute errors
        print(f"\nErrors:")
        print(f"  |ρ - ρ*|   = {abs(rho - true_params.rho):.4f}")
        print(f"  |μ - μ*|   = {abs(mu - true_params.mu):.2f}")
    
    return optimizer, history


def plot_convergence(history: dict, true_params: SVParams, save_path: str = 'svpf_finite_diff.png'):
    """Plot training convergence for 2-parameter learning."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    epochs = range(len(history['rho']))
    
    # ρ
    ax = axes[0, 0]
    ax.plot(epochs, history['rho'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params.rho, color='r', ls='--', linewidth=2, label='True')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ρ')
    ax.set_title('Persistence (ρ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # μ
    ax = axes[0, 1]
    ax.plot(epochs, history['mu'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params.mu, color='r', ls='--', linewidth=2, label='True')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('μ')
    ax.set_title('Mean Level (μ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-likelihood
    ax = axes[1, 0]
    ax.plot(epochs, history['log_lik'], 'g-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Training Log-Likelihood')
    ax.grid(True, alpha=0.3)
    
    # Errors over time
    ax = axes[1, 1]
    rho_err = [abs(r - true_params.rho) for r in history['rho']]
    mu_err = [abs(m - true_params.mu) for m in history['mu']]
    ax.semilogy(epochs, rho_err, 'b-', linewidth=1.5, label='|ρ - ρ*|')
    ax.semilogy(epochs, mu_err, 'r-', linewidth=1.5, label='|μ - μ*|')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Parameter Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # True parameters (crypto-calibrated)
    TRUE_PARAMS = SVParams(
        rho=0.94,
        sigma_z=0.18,
        mu=-3.2,
        nu=5.0,  # Heavy tails for crypto
    )
    
    T = 3000
    
    print("Generating synthetic SV data...")
    y, h = generate_synthetic_data(T=T, params=TRUE_PARAMS, seed=42)
    y = y.to(DEVICE)
    
    print(f"Data statistics: y.std={y.std():.4f}, h.mean={h.mean():.3f}")
    
    # Train with truncated windows - 2 parameters only
    optimizer, history = train_finite_diff_truncated(
        y,
        TRUE_PARAMS,
        n_epochs=200,
        lr=0.03,
        n_particles=300,
        n_stein_steps=5,
        init_rho=0.7,
        fixed_sigma_z=0.15,    # FIXED - adaptive in production
        init_mu=-4.0,
        trunc_len=50,          # Short windows!
        windows_per_epoch=10,
        verbose=True,
    )
    
    # Plot results
    plot_convergence(history, TRUE_PARAMS, save_path='svpf_finite_diff_truncated.png')
    
    # Export
    print("\nParameters ready for production:")
    rho, sigma_z, mu = optimizer.params
    print(f"  rho     = {rho:.6f}f")
    print(f"  sigma_z = {sigma_z:.6f}f")
    print(f"  mu      = {mu:.6f}f")
    print(f"  nu      = {TRUE_PARAMS.nu:.1f}f")
