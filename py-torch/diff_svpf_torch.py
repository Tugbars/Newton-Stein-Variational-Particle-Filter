"""
Differentiable SVPF - Offline Parameter Estimator (Production Version)

Purpose: Learn {ρ, σ_z, μ} from historical returns via backprop.
         Then freeze and feed to real-time CUDA SVPF.

Key decisions:
- ν (Student-t df) is FIXED (poorly identified, noisy to learn)
- Loss: One-step-ahead predictive NLL (computed BEFORE Stein transport)
- Truncated BPTT: Window of 50 steps for temporal credit assignment
- Burn-in: First N steps excluded from loss (cold start bias)
- Antithetic sampling: Variance reduction for gradients
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


class SVPFParamEstimator(nn.Module):
    """
    Offline parameter estimator for SVPF.
    
    Learns: ρ, σ_z, μ
    Fixed:  ν (Student-t df)
    
    Loss: Predictive NLL = -mean(logsumexp(log p(y_t | h_pred)))
    """
    
    def __init__(
        self,
        n_particles: int = 64,
        n_stein_steps: int = 2,
        stein_lr: float = 0.1,
        nu: float = 8.0,  # FIXED
        init_rho: float = 0.9,
        init_sigma_z: float = 0.15,
        init_mu: float = -3.5,
    ):
        super().__init__()
        
        self.n_particles = n_particles
        self.n_stein_steps = n_stein_steps
        self.stein_lr = stein_lr
        self.nu = nu  # Fixed, not learned
        
        # Precompute Student-t constant (since ν is fixed)
        # C(ν) = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*log(ν*π)
        self.register_buffer(
            'student_t_const',
            torch.tensor(
                math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            )
        )
        
        # Learnable parameters (unconstrained space, SOFT transforms)
        self._rho_logit = nn.Parameter(torch.tensor(self._logit(init_rho)))
        self._log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma_z)))
        self._mu = nn.Parameter(torch.tensor(init_mu))  # Unconstrained
    
    @staticmethod
    def _logit(x):
        return math.log(x / (1 - x + 1e-8) + 1e-8)
    
    @property
    def rho(self):
        """ρ ∈ (0.5, 0.999) via scaled sigmoid (smooth, no dead gradients)"""
        return 0.5 + 0.499 * torch.sigmoid(self._rho_logit)
    
    @property
    def sigma_z(self):
        """σ_z > 0 via softplus (smooth, no dead gradients)"""
        return F.softplus(self._log_sigma) + 0.001
    
    @property
    def mu(self):
        """μ unconstrained (daily log-var can be around -9 to -2)"""
        return self._mu
    
    def get_params(self):
        return {
            'rho': self.rho.item(),
            'sigma_z': self.sigma_z.item(),
            'mu': self.mu.item(),
            'nu': self.nu,
        }
    
    def _student_t_log_prob(self, y, scale):
        """Log p(y | scale, ν) using precomputed constant."""
        z = y / (scale + 1e-8)
        return (
            self.student_t_const
            - torch.log(scale + 1e-8)
            - (self.nu + 1) / 2 * torch.log1p(z**2 / self.nu)
        )
    
    def _imq_kernel(self, h, bandwidth):
        """IMQ kernel and gradient."""
        diff = h.unsqueeze(1) - h.unsqueeze(0)
        dist_sq = diff ** 2
        bw_sq = bandwidth ** 2 + 1e-8
        base = 1.0 + dist_sq / bw_sq
        
        K = torch.rsqrt(base)
        base_sqrt = torch.sqrt(base)
        grad_K = -diff / (bw_sq * base * base_sqrt)
        
        return K, grad_K
    
    def _stein_step(self, h, grad_log_p, bandwidth):
        """SVGD update."""
        n = h.shape[0]
        K, grad_K = self._imq_kernel(h, bandwidth)
        phi = K @ grad_log_p / n + grad_K.sum(dim=1) / n
        return h + self.stein_lr * phi
    
    def forward(self, y, truncate_every=50, burn_in=20):
        """
        Forward pass: run SVPF, return predictive NLL.
        
        Args:
            y: Returns [T]
            truncate_every: TBPTT window size
            burn_in: Steps to exclude from NLL (cold start bias)
            
        Returns:
            neg_log_lik: Scalar, mean predictive NLL (excluding burn-in)
            vol_estimates: [T] volatility estimates
        """
        T = y.shape[0]
        N = self.n_particles
        nu = self.nu
        
        rho = self.rho
        sigma_z = self.sigma_z
        mu = self.mu
        
        # Initialize from stationary distribution
        h_std = sigma_z / torch.sqrt(1 - rho**2 + 1e-6)
        h = mu + h_std * torch.randn(N, device=y.device)
        
        log_liks = []
        vol_estimates = []
        
        for t in range(T):
            # Truncated BPTT
            if truncate_every > 0 and t > 0 and (t % truncate_every) == 0:
                h = h.detach()
            
            h_prev = h
            
            # === PREDICT with ANTITHETIC SAMPLING ===
            # Use (ε, -ε) pairs for variance reduction
            half_N = N // 2
            eps_half = torch.randn(half_N, device=y.device)
            eps = torch.cat([eps_half, -eps_half])  # Antithetic pairs
            
            h_pred = mu + rho * (h_prev - mu) + sigma_z * eps
            transition_mean = mu + rho * (h_prev - mu)
            
            y_t = y[t]
            
            # === PREDICTIVE LOG-LIKELIHOOD (before Stein!) ===
            vol_pred = torch.exp(h_pred / 2)
            log_p = self._student_t_log_prob(y_t, vol_pred)
            log_lik_t = torch.logsumexp(log_p, dim=0) - math.log(N)
            log_liks.append(log_lik_t)
            
            # === STEIN TRANSPORT ===
            bw = h_pred.detach().std(unbiased=False) + 0.1
            h = h_pred
            
            for _ in range(self.n_stein_steps):
                vol = torch.exp(h / 2)
                A = y_t**2 / (vol**2 * nu + 1e-8)
                grad_lik = -0.5 + 0.5 * (nu + 1) * A / (1 + A)
                grad_prior = -(h - transition_mean) / (sigma_z**2 + 1e-8)
                grad_log_p = (grad_prior + grad_lik).clamp(-10, 10)
                h = self._stein_step(h, grad_log_p, bw)
            
            vol_estimates.append(torch.exp(h / 2).mean())
        
        # NLL excluding burn-in (cold start bias fix)
        log_liks_tensor = torch.stack(log_liks)
        neg_log_lik = -log_liks_tensor[burn_in:].mean()
        
        return neg_log_lik, torch.stack(vol_estimates)


def train(y_data, true_params, n_epochs=50, lr=0.02, batch_size=250, burn_in=20):
    """Train parameter estimator via NLL."""
    
    y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)
    T = len(y_data)
    n_batches = max(1, (T - 1) // batch_size)
    
    model = SVPFParamEstimator(
        n_particles=64,
        n_stein_steps=2,
        stein_lr=0.15,
        nu=true_params['nu'],
        init_rho=0.85,
        init_sigma_z=0.10,
        init_mu=-4.5,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'nll': [], 'rho': [], 'sigma_z': [], 'mu': []}
    
    print(f"\nTraining: {T} obs, {n_batches} batches of {batch_size}, burn_in={burn_in}")
    print(f"True:    ρ={true_params['rho']:.3f}, σ_z={true_params['sigma_z']:.3f}, μ={true_params['mu']:.3f}")
    print(f"Initial: ρ={model.rho.item():.3f}, σ_z={model.sigma_z.item():.3f}, μ={model.mu.item():.3f}")
    print(f"Fixed:   ν={model.nu:.1f}")
    print("-" * 60)
    
    t0 = time()
    
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_batches) * batch_size
        epoch_nll = 0.0
        n_batches_used = 0
        
        for start in perm[:min(10, n_batches)]:
            end = min(start + batch_size, T)
            if end - start < burn_in + 10:  # Skip too-short batches
                continue
            y_batch = y_tensor[start:end]
            
            optimizer.zero_grad()
            nll, _ = model(y_batch, truncate_every=50, burn_in=burn_in)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_nll += nll.item()
            n_batches_used += 1
        
        p = model.get_params()
        history['nll'].append(epoch_nll / max(n_batches_used, 1))
        history['rho'].append(p['rho'])
        history['sigma_z'].append(p['sigma_z'])
        history['mu'].append(p['mu'])
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time() - t0
            print(f"Epoch {epoch+1:3d} ({elapsed:5.1f}s): NLL={history['nll'][-1]:.3f} | "
                  f"ρ={p['rho']:.4f} ({true_params['rho']:.3f}) | "
                  f"σ={p['sigma_z']:.4f} ({true_params['sigma_z']:.3f}) | "
                  f"μ={p['mu']:.3f} ({true_params['mu']:.3f})")
    
    return model, history


def plot_results(history, true_params):
    """Plot training convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].plot(history['nll'])
    axes[0, 0].set_title('Predictive NLL')
    axes[0, 0].set_xlabel('Epoch')
    
    for ax, key, name in [
        (axes[0, 1], 'rho', 'ρ (persistence)'),
        (axes[1, 0], 'sigma_z', 'σ_z (vol-of-vol)'),
        (axes[1, 1], 'mu', 'μ (mean level)'),
    ]:
        ax.plot(history[key], 'b-', label='Learned')
        ax.axhline(true_params[key], color='r', ls='--', label='True')
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('svpf_param_learning_final.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    TRUE = {'rho': 0.95, 'sigma_z': 0.20, 'mu': -3.0, 'nu': 8.0}
    
    print("Generating synthetic data...")
    y, h = generate_sv_data(5000, **TRUE)
    print(f"Data: y.std={y.std():.4f}, h.mean={h.mean():.3f}")
    
    model, history = train(y, TRUE, n_epochs=50, lr=0.02, batch_size=250, burn_in=20)
    
    plot_results(history, TRUE)
    
    print("\n" + "="*50)
    print("FINAL LEARNED PARAMETERS")
    print("="*50)
    p = model.get_params()
    for k in ['rho', 'sigma_z', 'mu']:
        err = abs(p[k] - TRUE[k])
        print(f"  {k:8s}: {p[k]:.4f} (true: {TRUE[k]:.3f}, err: {err:.4f})")
    print(f"  {'nu':8s}: {p['nu']:.1f} (fixed)")
    