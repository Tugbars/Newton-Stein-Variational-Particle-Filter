"""
Differentiable SVPF in PyTorch - Parameter Learning via Backprop

Validates that SVPF can learn SV model parameters {ρ, σ_z, μ, ν} through
gradient descent on a loss function.

Key insight: SVPF has NO resampling, so the entire forward pass is differentiable:
- Predict step: reparameterization trick (h = μ + ρ(h_prev - μ) + σ_z * ε)
- Stein transport: gradient-based (differentiable by construction)
- Output: mean of particles (differentiable)

Usage:
    python diff_svpf_torch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# Data Generation (NumPy, not part of computation graph)
# =============================================================================

def generate_sv_data(
    T: int,
    rho: float,
    sigma_z: float,
    mu: float,
    nu: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stochastic volatility data."""
    np.random.seed(seed)
    
    h = np.zeros(T)
    y = np.zeros(T)
    
    # Stationary initialization
    h_std = sigma_z / np.sqrt(1 - rho**2)
    h[0] = mu + h_std * np.random.randn()
    
    for t in range(T):
        if t > 0:
            h[t] = mu + rho * (h[t-1] - mu) + sigma_z * np.random.randn()
        
        vol = np.exp(h[t] / 2)
        # Student-t noise
        y[t] = vol * np.random.standard_t(nu)
    
    return y, h


# =============================================================================
# Differentiable SVPF
# =============================================================================

class DifferentiableSVPF(nn.Module):
    """
    Fully differentiable Stein Variational Particle Filter.
    
    Parameters are nn.Parameters, so gradients flow through them.
    """
    
    def __init__(
        self,
        n_particles: int = 256,
        n_stein_steps: int = 5,
        stein_lr: float = 0.1,
        init_rho: float = 0.9,
        init_sigma_z: float = 0.1,
        init_mu: float = -4.0,
        init_nu: float = 15.0,
    ):
        super().__init__()
        
        self.n_particles = n_particles
        self.n_stein_steps = n_stein_steps
        self.stein_lr = stein_lr
        
        # Learnable parameters (unconstrained)
        # We learn in transformed space to ensure constraints
        self._rho_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(init_rho, 0.5, 0.999)))
        self._log_sigma_z = nn.Parameter(torch.tensor(np.log(init_sigma_z)))
        self._mu = nn.Parameter(torch.tensor(init_mu))
        self._log_nu = nn.Parameter(torch.tensor(np.log(init_nu - 2)))  # nu > 2 for finite variance
        
    @staticmethod
    def _inv_sigmoid(y, lo, hi):
        """Inverse of scaled sigmoid."""
        y_normalized = (y - lo) / (hi - lo)
        return np.log(y_normalized / (1 - y_normalized + 1e-8) + 1e-8)
    
    @property
    def rho(self):
        """ρ ∈ (0.5, 0.999)"""
        return 0.5 + 0.499 * torch.sigmoid(self._rho_logit)
    
    @property
    def sigma_z(self):
        """σ_z > 0"""
        return torch.exp(self._log_sigma_z).clamp(min=0.01, max=1.0)
    
    @property
    def mu(self):
        """μ ∈ (-10, 0)"""
        return self._mu.clamp(min=-10.0, max=0.0)
    
    @property
    def nu(self):
        """ν > 2"""
        return 2.0 + torch.exp(self._log_nu).clamp(min=0.1, max=100.0)
    
    def get_params(self):
        """Return current parameter values."""
        return {
            'rho': self.rho.item(),
            'sigma_z': self.sigma_z.item(),
            'mu': self.mu.item(),
            'nu': self.nu.item(),
        }
    
    def _student_t_log_prob(self, y: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Log probability of Student-t distribution."""
        # log p(y | 0, scale, nu)
        z = y / scale
        return (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * np.pi)
            - torch.log(scale)
            - (nu + 1) / 2 * torch.log(1 + z**2 / nu)
        )
    
    def _imq_kernel(self, h: torch.Tensor, bandwidth: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse Multiquadric kernel and its gradient.
        
        K(x, y) = 1 / sqrt(1 + ||x - y||² / h²)
        """
        # h: [n_particles]
        diff = h.unsqueeze(0) - h.unsqueeze(1)  # [n, n]
        dist_sq = diff**2
        
        base = 1.0 + dist_sq / (bandwidth**2)
        K = 1.0 / torch.sqrt(base)  # [n, n]
        
        # ∇_x K(x, y) = -(x - y) / (h² * base^(3/2))
        grad_K = -diff / (bandwidth**2 * base**(1.5))  # [n, n]
        
        return K, grad_K
    
    def _stein_step(
        self,
        h: torch.Tensor,
        grad_log_p: torch.Tensor,
        bandwidth: float
    ) -> torch.Tensor:
        """
        One step of Stein Variational Gradient Descent.
        
        φ(x) = (1/n) Σ_j [K(x_j, x) ∇log p(x_j) + ∇_x K(x_j, x)]
        """
        n = h.shape[0]
        K, grad_K = self._imq_kernel(h, bandwidth)
        
        # Attraction: K @ grad_log_p
        attraction = K @ grad_log_p / n  # [n]
        
        # Repulsion: sum of kernel gradients
        repulsion = grad_K.sum(dim=1) / n  # [n]
        
        phi = attraction + repulsion
        
        return h + self.stein_lr * phi
    
    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run SVPF on observation sequence.
        
        Args:
            y: Observations [T]
            
        Returns:
            vol_estimates: Volatility estimates [T]
            log_likelihoods: Per-step log-likelihoods [T]
        """
        T = y.shape[0]
        
        # Get current params
        rho = self.rho
        sigma_z = self.sigma_z
        mu = self.mu
        nu = self.nu
        
        # Initialize particles from stationary distribution
        h_std = sigma_z / torch.sqrt(1 - rho**2 + 1e-6)
        h = mu + h_std * torch.randn(self.n_particles, device=y.device)
        
        vol_estimates = []
        log_likelihoods = []
        
        for t in range(T):
            # === PREDICT (reparameterization trick) ===
            eps = torch.randn_like(h)
            h_pred = mu + rho * (h - mu) + sigma_z * eps
            
            # === COMPUTE GRADIENTS ===
            # Prior gradient: ∇_h log p(h | h_prev) -- but h_prev is now h_pred's "source"
            # For predicted particles, gradient toward prior mean
            # Actually for Stein, we need ∇_h log p(h, y) = ∇_h log p(y|h) + ∇_h log p(h)
            
            # Likelihood gradient
            vol = torch.exp(h_pred / 2)
            y_t = y[t]
            
            # ∇_h log p(y | h) for Student-t
            # p(y|h) ∝ (1 + y²/(ν·exp(h)))^(-(ν+1)/2) · exp(-h/2)
            A = y_t**2 / (vol**2 * nu + 1e-8)
            grad_lik = -0.5 + 0.5 * (nu + 1) * A / (1 + A)
            
            # Prior gradient (toward mu)
            grad_prior = -(h_pred - mu) / (sigma_z**2 + 1e-8)
            
            # Combined
            grad_log_p = grad_prior + grad_lik
            grad_log_p = grad_log_p.clamp(-10, 10)  # Stability
            
            # === STEIN TRANSPORT ===
            bandwidth = h_pred.std().item() + 0.1
            h = h_pred
            for _ in range(self.n_stein_steps):
                h = self._stein_step(h, grad_log_p, bandwidth)
                # Recompute gradient at new position
                vol = torch.exp(h / 2)
                A = y_t**2 / (vol**2 * nu + 1e-8)
                grad_lik = -0.5 + 0.5 * (nu + 1) * A / (1 + A)
                grad_prior = -(h - mu) / (sigma_z**2 + 1e-8)
                grad_log_p = (grad_prior + grad_lik).clamp(-10, 10)
            
            # === OUTPUT ===
            vol_mean = torch.exp(h / 2).mean()
            vol_estimates.append(vol_mean)
            
            # Log-likelihood for this step
            log_lik = self._student_t_log_prob(y_t, vol_mean, nu)
            log_likelihoods.append(log_lik)
        
        return torch.stack(vol_estimates), torch.stack(log_likelihoods)


# =============================================================================
# Training Loop
# =============================================================================

def train_svpf(
    y_train: np.ndarray,
    h_true: np.ndarray,
    true_params: dict,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 500,
):
    """Train differentiable SVPF to learn parameters."""
    
    # Convert to torch
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    h_tensor = torch.tensor(h_true, dtype=torch.float32, device=device)
    true_vol = torch.exp(h_tensor / 2)
    
    # Create model with wrong initial params
    model = DifferentiableSVPF(
        n_particles=128,
        n_stein_steps=3,
        stein_lr=0.15,
        init_rho=0.85,      # True: 0.95
        init_sigma_z=0.10,   # True: 0.20
        init_mu=-4.5,        # True: -3.0
        init_nu=20.0,        # True: 8.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track history
    history = {
        'loss': [],
        'rho': [],
        'sigma_z': [],
        'mu': [],
        'nu': [],
    }
    
    T = len(y_train)
    n_batches = (T - 1) // batch_size
    
    print(f"\nTraining on {T} observations, {n_batches} batches of {batch_size}")
    print(f"True params: ρ={true_params['rho']:.3f}, σ_z={true_params['sigma_z']:.3f}, "
          f"μ={true_params['mu']:.3f}, ν={true_params['nu']:.1f}")
    print(f"Initial:     ρ={model.rho.item():.3f}, σ_z={model.sigma_z.item():.3f}, "
          f"μ={model.mu.item():.3f}, ν={model.nu.item():.1f}")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Shuffle batch starts
        batch_starts = np.random.permutation(n_batches) * batch_size
        
        for batch_idx, start in enumerate(batch_starts[:min(5, n_batches)]):  # Limit batches per epoch
            end = min(start + batch_size, T)
            y_batch = y_tensor[start:end]
            true_vol_batch = true_vol[start:end]
            
            optimizer.zero_grad()
            
            # Forward pass
            vol_est, log_liks = model(y_batch)
            
            # Loss: MSE on log-volatility (more stable than raw vol)
            log_vol_est = torch.log(vol_est + 1e-8)
            log_vol_true = torch.log(true_vol_batch + 1e-8)
            loss = torch.mean((log_vol_est - log_vol_true)**2)
            
            # Alternative: negative log-likelihood
            # loss = -log_liks.mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Record
        params = model.get_params()
        history['loss'].append(epoch_loss / min(5, n_batches))
        history['rho'].append(params['rho'])
        history['sigma_z'].append(params['sigma_z'])
        history['mu'].append(params['mu'])
        history['nu'].append(params['nu'])
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={history['loss'][-1]:.4f} | "
                  f"ρ={params['rho']:.4f} ({true_params['rho']:.3f}) | "
                  f"σ={params['sigma_z']:.4f} ({true_params['sigma_z']:.3f}) | "
                  f"μ={params['mu']:.3f} ({true_params['mu']:.3f}) | "
                  f"ν={params['nu']:.2f} ({true_params['nu']:.1f})")
    
    return model, history


def plot_results(history: dict, true_params: dict):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Loss
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    
    # ρ
    axes[0, 1].plot(history['rho'], label='Learned')
    axes[0, 1].axhline(true_params['rho'], color='r', linestyle='--', label='True')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('ρ')
    axes[0, 1].set_title('Persistence (ρ)')
    axes[0, 1].legend()
    
    # σ_z
    axes[0, 2].plot(history['sigma_z'], label='Learned')
    axes[0, 2].axhline(true_params['sigma_z'], color='r', linestyle='--', label='True')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('σ_z')
    axes[0, 2].set_title('Vol-of-Vol (σ_z)')
    axes[0, 2].legend()
    
    # μ
    axes[1, 0].plot(history['mu'], label='Learned')
    axes[1, 0].axhline(true_params['mu'], color='r', linestyle='--', label='True')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('μ')
    axes[1, 0].set_title('Mean Level (μ)')
    axes[1, 0].legend()
    
    # ν
    axes[1, 1].plot(history['nu'], label='Learned')
    axes[1, 1].axhline(true_params['nu'], color='r', linestyle='--', label='True')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ν')
    axes[1, 1].set_title('Tail Index (ν)')
    axes[1, 1].legend()
    
    # Summary
    axes[1, 2].axis('off')
    final = {k: history[k][-1] for k in ['rho', 'sigma_z', 'mu', 'nu']}
    summary = (
        f"Final Results:\n\n"
        f"ρ:   {final['rho']:.4f} (true: {true_params['rho']:.3f})\n"
        f"σ_z: {final['sigma_z']:.4f} (true: {true_params['sigma_z']:.3f})\n"
        f"μ:   {final['mu']:.3f} (true: {true_params['mu']:.3f})\n"
        f"ν:   {final['nu']:.2f} (true: {true_params['nu']:.1f})"
    )
    axes[1, 2].text(0.1, 0.5, summary, fontsize=14, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('svpf_param_learning.png', dpi=150)
    plt.show()
    print("\nSaved plot to svpf_param_learning.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # True parameters
    TRUE_PARAMS = {
        'rho': 0.95,
        'sigma_z': 0.20,
        'mu': -3.0,
        'nu': 8.0,
    }
    
    # Generate data
    print("Generating synthetic SV data...")
    T = 5000
    y, h_true = generate_sv_data(
        T=T,
        rho=TRUE_PARAMS['rho'],
        sigma_z=TRUE_PARAMS['sigma_z'],
        mu=TRUE_PARAMS['mu'],
        nu=TRUE_PARAMS['nu'],
        seed=42
    )
    
    print(f"Data stats: y.std={y.std():.4f}, h.mean={h_true.mean():.3f}, h.std={h_true.std():.3f}")
    
    # Train
    model, history = train_svpf(
        y_train=y,
        h_true=h_true,
        true_params=TRUE_PARAMS,
        n_epochs=100,
        lr=0.02,
        batch_size=500,
    )
    
    # Plot
    plot_results(history, TRUE_PARAMS)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL PARAMETER ESTIMATES")
    print("=" * 70)
    final_params = model.get_params()
    for name, true_val in TRUE_PARAMS.items():
        learned = final_params[name]
        error = abs(learned - true_val)
        print(f"{name:8s}: Learned={learned:8.4f}, True={true_val:8.4f}, Error={error:.4f}")
