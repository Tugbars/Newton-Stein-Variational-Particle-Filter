"""
Differentiable Bootstrap Particle Filter with Optimal Transport Resampling

Purpose: Learn {ρ, σ_z, μ} with filter-parameter consistency.
         No Stein transport = no σ_z collapse problem.

Key Innovation:
- Replace non-differentiable multinomial resampling with Sinkhorn OT
- Transport matrix is differentiable → gradients flow through resampling
- Matches physics of production Bootstrap PF

The Problem with Standard BPF:
    ancestors = torch.multinomial(weights, N)  # ← Non-differentiable!
    
The Solution (OT Resampling):
    P = sinkhorn(uniform, weights)  # ← Differentiable transport matrix
    h_new = P @ h                   # ← Soft resampling

References:
- Corenflos et al. (2021) "Differentiable Particle Filtering via Entropy-Regularized OT"
- Mena et al. (2018) "Learning Latent Permutations with Gumbel-Sinkhorn"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_sv_data(T, rho, sigma_z, mu, nu, seed=42):
    """Generate synthetic SV data."""
    np.random.seed(seed)
    h = np.zeros(T)
    y = np.zeros(T)
    
    h_std = sigma_z / np.sqrt(1 - rho**2)
    h[0] = mu + h_std * np.random.randn()
    
    for t in range(T):
        if t > 0:
            h[t] = mu + rho * (h[t-1] - mu) + sigma_z * np.random.randn()
        vol = np.exp(h[t] / 2)
        y[t] = vol * np.random.standard_t(nu)
    
    return y.astype(np.float32), h.astype(np.float32)


# =============================================================================
# SINKHORN OPTIMAL TRANSPORT
# =============================================================================

def sinkhorn_transport(
    source_weights: torch.Tensor,  # [N] uniform
    target_weights: torch.Tensor,  # [N] normalized particle weights
    cost_matrix: torch.Tensor,     # [N, N] pairwise distances
    epsilon: float = 0.1,          # Entropy regularization
    n_iters: int = 20,             # Sinkhorn iterations
) -> torch.Tensor:
    """
    Compute entropy-regularized optimal transport matrix via Sinkhorn.
    
    Returns P such that:
    - P @ source ≈ target (columns sum to target weights)
    - P.T @ target ≈ source (rows sum to source weights)
    - P minimizes transport cost + entropy regularization
    
    The transport matrix P is differentiable w.r.t. target_weights.
    """
    N = source_weights.shape[0]
    
    # Log-domain Sinkhorn for numerical stability
    # K_ij = exp(-C_ij / epsilon)
    log_K = -cost_matrix / epsilon
    
    # Initialize dual variables
    log_u = torch.zeros(N, device=source_weights.device)
    log_v = torch.zeros(N, device=target_weights.device)
    
    log_source = torch.log(source_weights + 1e-10)
    log_target = torch.log(target_weights + 1e-10)
    
    for _ in range(n_iters):
        # Update v: normalize columns to match target
        log_v = log_target - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        # Update u: normalize rows to match source
        log_u = log_source - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
    
    # Compute transport matrix: P_ij = u_i * K_ij * v_j
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    
    return P


def sinkhorn_resample(
    h: torch.Tensor,               # [N] particle states
    weights: torch.Tensor,         # [N] normalized weights
    epsilon: float = 0.1,
    n_iters: int = 20,
) -> torch.Tensor:
    """
    Differentiable resampling via Sinkhorn OT.
    
    Instead of discrete ancestor selection, compute soft weighted average
    via optimal transport from uniform to current weights.
    """
    N = h.shape[0]
    
    # Source: uniform weights (what we want after resampling)
    source = torch.ones(N, device=h.device) / N
    
    # Cost matrix: squared distance between particle states
    # C_ij = (h_i - h_j)^2
    h_col = h.unsqueeze(1)  # [N, 1]
    h_row = h.unsqueeze(0)  # [1, N]
    cost_matrix = (h_col - h_row) ** 2  # [N, N]
    
    # Compute transport matrix
    P = sinkhorn_transport(source, weights, cost_matrix, epsilon, n_iters)
    
    # Apply transport: h_new[i] = sum_j P[i,j] * h[j]
    # P[i,j] = "how much of particle j goes to position i"
    h_new = P @ h
    
    return h_new


# =============================================================================
# DIFFERENTIABLE BOOTSTRAP PF
# =============================================================================

class DifferentiableBPF(nn.Module):
    """
    Differentiable Bootstrap Particle Filter for SV parameter learning.
    
    Architecture:
    1. PREDICT: h_t = μ + ρ(h_{t-1} - μ) + σ_z·ε  (reparameterized)
    2. WEIGHT:  w_t ∝ p(y_t | h_t)                 (Student-t likelihood)
    3. RESAMPLE: h_new = Sinkhorn(uniform, w) @ h  (Differentiable OT)
    
    Unlike SVPF:
    - No Stein transport (no σ_z absorption)
    - Resampling is soft but respects BPF physics
    - σ_z must provide the diffusion (can't cheat!)
    """
    
    def __init__(
        self,
        n_particles: int = 128,
        nu: float = 8.0,
        init_rho: float = 0.85,
        init_sigma_z: float = 0.15,
        init_mu: float = -4.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iters: int = 20,
        ess_threshold: float = 0.5,  # Resample when ESS < threshold * N
    ):
        super().__init__()
        
        self.n_particles = n_particles
        self.nu = nu
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.ess_threshold = ess_threshold
        
        # Precompute Student-t constant
        self.register_buffer(
            'student_t_const',
            torch.tensor(
                math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            )
        )
        
        # Learnable parameters
        self._rho_logit = nn.Parameter(torch.tensor(self._logit(init_rho)))
        self._log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma_z)))
        self._mu = nn.Parameter(torch.tensor(init_mu))
    
    @staticmethod
    def _logit(x):
        return math.log(x / (1 - x + 1e-8) + 1e-8)
    
    @property
    def rho(self):
        """ρ ∈ (0.5, 0.999)"""
        return 0.5 + 0.499 * torch.sigmoid(self._rho_logit)
    
    @property
    def sigma_z(self):
        """σ_z > 0"""
        return F.softplus(self._log_sigma) + 0.01
    
    @property
    def mu(self):
        return self._mu
    
    def get_params(self):
        return {
            'rho': self.rho.item(),
            'sigma_z': self.sigma_z.item(),
            'mu': self.mu.item(),
            'nu': self.nu,
        }
    
    def _student_t_log_prob(self, y, scale):
        """Log p(y | scale, ν)"""
        z = y / (scale + 1e-8)
        return (
            self.student_t_const
            - torch.log(scale + 1e-8)
            - (self.nu + 1) / 2 * torch.log1p(z**2 / self.nu)
        )
    
    def forward(self, y, truncate_every=50, burn_in=20, resample_mode='adaptive'):
        """
        Run differentiable BPF and compute predictive NLL.
        
        Args:
            y: Returns [T]
            truncate_every: TBPTT window
            burn_in: Steps to exclude from loss
            resample_mode: 'always', 'adaptive', or 'never'
            
        Returns:
            neg_log_lik: Mean predictive NLL
            vol_estimates: [T] volatility estimates
        """
        T = y.shape[0]
        N = self.n_particles
        
        rho = self.rho
        sigma_z = self.sigma_z
        mu = self.mu
        
        # Initialize from stationary distribution
        h_std = sigma_z / torch.sqrt(1 - rho**2 + 1e-6)
        eps_init = torch.randn(N, device=y.device)
        h = mu + h_std * eps_init
        
        # Uniform initial weights
        log_w = torch.zeros(N, device=y.device) - math.log(N)
        
        log_liks = []
        vol_estimates = []
        
        for t in range(T):
            # Truncated BPTT
            if truncate_every > 0 and t > 0 and (t % truncate_every) == 0:
                h = h.detach()
                log_w = log_w.detach()
            
            y_t = y[t]
            
            # ═══════════════════════════════════════════════════════════════
            # RESAMPLE (Differentiable via Sinkhorn OT)
            # ═══════════════════════════════════════════════════════════════
            
            # Compute ESS
            w = F.softmax(log_w, dim=0)
            ess = 1.0 / (w ** 2).sum()
            
            do_resample = (
                resample_mode == 'always' or
                (resample_mode == 'adaptive' and ess < self.ess_threshold * N)
            )
            
            if do_resample and t > 0:
                # Soft resampling via Sinkhorn
                h = sinkhorn_resample(
                    h, w, 
                    epsilon=self.sinkhorn_epsilon,
                    n_iters=self.sinkhorn_iters
                )
                # Reset to uniform weights after resampling
                log_w = torch.zeros(N, device=y.device) - math.log(N)
            
            # ═══════════════════════════════════════════════════════════════
            # PREDICT (Reparameterized for gradients)
            # h_t = μ + ρ(h_{t-1} - μ) + σ_z·ε
            # ═══════════════════════════════════════════════════════════════
            
            # Antithetic sampling for variance reduction
            half_N = N // 2
            eps_half = torch.randn(half_N, device=y.device)
            eps = torch.cat([eps_half, -eps_half])
            
            h_pred = mu + rho * (h - mu) + sigma_z * eps
            
            # ═══════════════════════════════════════════════════════════════
            # WEIGHT (Student-t likelihood)
            # ═══════════════════════════════════════════════════════════════
            
            vol_pred = torch.exp(h_pred / 2)
            log_lik_particles = self._student_t_log_prob(y_t, vol_pred)
            
            # Update log weights
            log_w = log_w + log_lik_particles
            
            # Predictive log-likelihood (before normalization)
            log_lik_t = torch.logsumexp(log_w, dim=0)  # This is log p(y_t | y_{1:t-1})
            log_liks.append(log_lik_t)
            
            # Normalize log_w for next iteration
            log_w = log_w - torch.logsumexp(log_w, dim=0)
            
            # Store vol estimate (weighted mean)
            w_normalized = F.softmax(log_w, dim=0)
            vol_estimates.append((w_normalized * vol_pred).sum())
            
            h = h_pred
        
        # NLL (excluding burn-in)
        # Note: log_liks[t] = log p(y_{1:t}) cumulative, we want increments
        log_liks_tensor = torch.stack(log_liks)
        
        # Incremental log-likelihoods: log p(y_t | y_{1:t-1})
        log_lik_increments = torch.zeros_like(log_liks_tensor)
        log_lik_increments[0] = log_liks_tensor[0]
        log_lik_increments[1:] = log_liks_tensor[1:] - log_liks_tensor[:-1]
        
        neg_log_lik = -log_lik_increments[burn_in:].mean()
        
        return neg_log_lik, torch.stack(vol_estimates)


# =============================================================================
# TRAINING
# =============================================================================

def train_dbpf(
    y_data,
    true_params,
    n_epochs: int = 100,
    lr: float = 0.02,
    batch_size: int = 500,
    burn_in: int = 20,
    sinkhorn_epsilon: float = 0.1,
    verbose: bool = True,
):
    """Train differentiable BPF."""
    
    y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)
    T = len(y_data)
    n_batches = max(1, (T - burn_in - 10) // batch_size)
    
    model = DifferentiableBPF(
        n_particles=128,
        nu=true_params.get('nu', 8.0),
        init_rho=0.85,
        init_sigma_z=0.10,
        init_mu=-4.0,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=20,
        ess_threshold=0.5,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    history = {'nll': [], 'rho': [], 'sigma_z': [], 'mu': []}
    
    if verbose:
        print(f"\n{'='*70}")
        print("DIFFERENTIABLE BOOTSTRAP PF - OT RESAMPLING")
        print(f"{'='*70}")
        print(f"Data:       T={T}, batch_size={batch_size}, burn_in={burn_in}")
        print(f"Sinkhorn:   ε={sinkhorn_epsilon}, iters=20")
        print(f"True:       ρ={true_params['rho']:.3f}, σ_z={true_params['sigma_z']:.3f}, μ={true_params['mu']:.3f}")
        print(f"Init:       ρ={model.rho.item():.3f}, σ_z={model.sigma_z.item():.3f}, μ={model.mu.item():.3f}")
        print(f"Fixed:      ν={model.nu:.1f}")
        print(f"{'─'*70}")
    
    t0 = time()
    
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_batches) * batch_size
        epoch_nll = 0.0
        n_batches_used = 0
        
        for start in perm[:min(10, n_batches)]:
            end = min(start + batch_size, T)
            if end - start < burn_in + 20:
                continue
            y_batch = y_tensor[start:end]
            
            optimizer.zero_grad()
            nll, _ = model(y_batch, truncate_every=50, burn_in=burn_in)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_nll += nll.item()
            n_batches_used += 1
        
        scheduler.step()
        
        p = model.get_params()
        avg_nll = epoch_nll / max(n_batches_used, 1)
        history['nll'].append(avg_nll)
        history['rho'].append(p['rho'])
        history['sigma_z'].append(p['sigma_z'])
        history['mu'].append(p['mu'])
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            elapsed = time() - t0
            print(
                f"Epoch {epoch+1:3d} ({elapsed:5.1f}s) | "
                f"NLL={avg_nll:.4f} | "
                f"ρ={p['rho']:.4f} ({true_params['rho']:.3f}) | "
                f"σ={p['sigma_z']:.4f} ({true_params['sigma_z']:.3f}) | "
                f"μ={p['mu']:.3f} ({true_params['mu']:.3f})"
            )
    
    if verbose:
        print(f"{'─'*70}")
        elapsed = time() - t0
        print(f"Training complete in {elapsed:.1f}s")
    
    return model, history


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(history, true_params, save_path='dbpf_param_learning.png'):
    """Plot training convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(history['nll']) + 1)
    
    # NLL
    ax = axes[0, 0]
    ax.plot(epochs, history['nll'], 'b-', linewidth=1.5)
    ax.set_title('Predictive NLL', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    
    # ρ
    ax = axes[0, 1]
    ax.plot(epochs, history['rho'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params['rho'], color='r', ls='--', linewidth=2, label='True')
    ax.set_title('ρ (Persistence)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # σ_z
    ax = axes[1, 0]
    ax.plot(epochs, history['sigma_z'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params['sigma_z'], color='r', ls='--', linewidth=2, label='True')
    ax.set_title('σ_z (Vol-of-Vol)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # μ
    ax = axes[1, 1]
    ax.plot(epochs, history['mu'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params['mu'], color='r', ls='--', linewidth=2, label='True')
    ax.set_title('μ (Mean Level)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_final_results(model, true_params):
    """Print final estimates."""
    p = model.get_params()
    
    print(f"\n{'='*60}")
    print("FINAL LEARNED PARAMETERS")
    print(f"{'='*60}")
    print(f"{'Parameter':<12} {'Learned':>10} {'True':>10} {'Error':>10} {'Rel.Err':>10}")
    print(f"{'-'*60}")
    
    for key in ['rho', 'sigma_z', 'mu']:
        learned = p[key]
        true = true_params[key]
        err = abs(learned - true)
        rel_err = err / abs(true) * 100 if true != 0 else float('inf')
        print(f"{key:<12} {learned:>10.4f} {true:>10.4f} {err:>10.4f} {rel_err:>9.1f}%")
    
    print(f"{'-'*60}")
    print(f"{'nu (fixed)':<12} {p['nu']:>10.1f}")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    TRUE_PARAMS = {
        'rho': 0.94,
        'sigma_z': 0.18,
        'mu': -3.2,
        'nu': 8.0,
    }
    
    T = 3000
    
    print("Generating synthetic SV data...")
    y, h_true = generate_sv_data(T, **TRUE_PARAMS, seed=42)
    print(f"Data statistics: y.std={y.std():.4f}, h.mean={h_true.mean():.3f}")
    
    # Train
    model, history = train_dbpf(
        y,
        TRUE_PARAMS,
        n_epochs=100,
        lr=0.02,
        batch_size=500,
        burn_in=20,
        sinkhorn_epsilon=0.1,
        verbose=True,
    )
    
    # Results
    print_final_results(model, TRUE_PARAMS)
    plot_results(history, TRUE_PARAMS, save_path='dbpf_param_learning.png')
    
    # Export
    print("Parameters ready for production BPF:")
    p = model.get_params()
    print(f"  rho     = {p['rho']:.6f}f")
    print(f"  sigma_z = {p['sigma_z']:.6f}f")
    print(f"  mu      = {p['mu']:.6f}f")
    print(f"  nu      = {p['nu']:.1f}f  // fixed")
