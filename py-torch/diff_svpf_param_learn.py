"""
Differentiable SVPF - Physics-Dominant Parameter Estimator

Purpose: Learn {ρ, σ_z, μ} from historical returns via backprop.
         Then freeze and feed to real-time CUDA SVPF.

Physics-Dominant Configuration:
1. Likelihood-only tempering (Option C): 
   - grad_log_p = grad_prior + (grad_lik / T)
   - Weakens data fitting, preserves physics (transition dynamics)
   - Implicit repulsion boost: when T=5, repulsion is 5× stronger relative to lik
   
2. Unit repulsion (repulsion_scale = 1.0):
   - No explicit boost needed (implicit boost from tempering is enough)
   - Prevents "double boost" failure where optimizer provides variance
   - Forces σ_z to provide the particle spread, not artificial repulsion

3. Low Stein LR (0.02):
   - Prevents particle "teleportation" to optimal positions
   - Forces particles to follow the physics (transition dynamics)

4. Faster temperature schedule:
   - linear_fast: T reaches 1.0 at epoch 50 (not 100)
   - Model must face real likelihood sooner, needs σ_z to survive

The Key Insight:
- Particle spread should come from σ_z (MODEL PHYSICS)
- NOT from artificial repulsion (OPTIMIZER TRICKS)
- If repulsion provides variance, σ_z is free to collapse
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
    """
    Generate synthetic SV data.
    
    Model:
        h_t = μ + ρ*(h_{t-1} - μ) + σ_z*ε_t    (log-volatility)
        y_t = exp(h_t/2) * η_t                  (returns, η ~ Student-t(ν))
    """
    np.random.seed(seed)
    h = np.zeros(T)
    y = np.zeros(T)
    
    # Initialize from stationary distribution
    h_std = sigma_z / np.sqrt(1 - rho**2)
    h[0] = mu + h_std * np.random.randn()
    
    for t in range(T):
        if t > 0:
            h[t] = mu + rho * (h[t-1] - mu) + sigma_z * np.random.randn()
        vol = np.exp(h[t] / 2)
        y[t] = vol * np.random.standard_t(nu)
    
    return y.astype(np.float32), h.astype(np.float32)


# =============================================================================
# ROBUST SVPF PARAMETER ESTIMATOR
# =============================================================================

class RobustSVPFEstimator(nn.Module):
    """
    Differentiable SVPF for offline parameter learning.
    
    Learns: ρ (persistence), σ_z (vol-of-vol), μ (mean level)
    Fixed:  ν (Student-t df) - poorly identified, keep fixed
    
    Key Mechanisms (Physics-Dominant Configuration):
    1. Likelihood-only tempering: Weaken data fitting, preserve physics
    2. Unit repulsion: Let implicit boost from tempering handle diversity
    3. Low Stein LR: Prevent particle "teleportation"
    4. Bandwidth floor: Prevent kernel collapse
    
    The goal: Particle spread should come from σ_z (model physics),
    not from artificial repulsion (optimizer tricks).
    """
    
    def __init__(
        self,
        n_particles: int = 128,
        n_stein_steps: int = 1,
        stein_lr: float = 0.02,      # LOW: Prevent teleportation
        nu: float = 8.0,
        init_rho: float = 0.85,
        init_sigma_z: float = 0.10,
        init_mu: float = -4.0,
        bandwidth_floor: float = 0.5,
    ):
        super().__init__()
        
        self.n_particles = n_particles
        self.n_stein_steps = n_stein_steps
        self.stein_lr = stein_lr
        self.nu = nu
        self.bandwidth_floor = bandwidth_floor
        
        # Precompute Student-t constant (ν is fixed)
        # C(ν) = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*log(ν*π)
        self.register_buffer(
            'student_t_const',
            torch.tensor(
                math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            )
        )
        
        # Learnable parameters (unconstrained space with soft transforms)
        self._rho_logit = nn.Parameter(torch.tensor(self._logit(init_rho)))
        self._log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma_z)))
        self._mu = nn.Parameter(torch.tensor(init_mu))
    
    @staticmethod
    def _logit(x):
        return math.log(x / (1 - x + 1e-8) + 1e-8)
    
    @property
    def rho(self):
        """ρ ∈ (0.5, 0.999) via scaled sigmoid"""
        return 0.5 + 0.499 * torch.sigmoid(self._rho_logit)
    
    @property
    def sigma_z(self):
        """σ_z > 0 via softplus"""
        return F.softplus(self._log_sigma) + 0.001
    
    @property
    def mu(self):
        """μ unconstrained"""
        return self._mu
    
    def get_params(self):
        """Return current parameter estimates as dict."""
        return {
            'rho': self.rho.item(),
            'sigma_z': self.sigma_z.item(),
            'mu': self.mu.item(),
            'nu': self.nu,
        }
    
    def _student_t_log_prob(self, y, scale):
        """Log p(y | scale, ν) for Student-t observation model."""
        z = y / (scale + 1e-8)
        return (
            self.student_t_const
            - torch.log(scale + 1e-8)
            - (self.nu + 1) / 2 * torch.log1p(z**2 / self.nu)
        )
    
    def _imq_kernel(self, h, bandwidth):
        """
        Inverse Multi-Quadric kernel and its gradient.
        
        K(x,y) = 1 / sqrt(1 + ||x-y||²/h²)
        ∇_x K = -(x-y) / (h² * (1 + ||x-y||²/h²)^{3/2})
        """
        diff = h.unsqueeze(1) - h.unsqueeze(0)  # [N, N]
        dist_sq = diff ** 2
        bw_sq = bandwidth ** 2 + 1e-8
        base = 1.0 + dist_sq / bw_sq
        
        K = torch.rsqrt(base)  # 1/sqrt(base)
        base_sqrt = torch.sqrt(base)
        grad_K = -diff / (bw_sq * base * base_sqrt)
        
        return K, grad_K
    
    def _stein_step(self, h, grad_log_p, bandwidth, repulsion_scale=1.0):
        """
        SVGD update with repulsion boosting.
        
        φ(x) = 1/n Σ_j [K(x_j, x)·∇log p(x_j) + ∇_{x_j} K(x_j, x)]
                       └─── attraction ───┘   └─── repulsion ───┘
        
        repulsion_scale > 1.0 forces particles apart (prevents collapse)
        """
        n = h.shape[0]
        K, grad_K = self._imq_kernel(h, bandwidth)
        
        # Attraction: Drive particles toward high probability
        attraction = K @ grad_log_p / n
        
        # Repulsion: Push particles apart (BOOSTED to prevent collapse)
        repulsion = (grad_K.sum(dim=1) / n) * repulsion_scale
        
        phi = attraction + repulsion
        return h + self.stein_lr * phi
    
    def forward(
        self, 
        y, 
        truncate_every: int = 50,
        burn_in: int = 20,
        temperature: float = 1.0,
        repulsion_scale: float = 1.0,
    ):
        """
        Forward pass: Run SVPF and compute predictive NLL.
        
        Args:
            y: Returns tensor [T]
            truncate_every: TBPTT window size (0 = full BPTT)
            burn_in: Steps to exclude from loss (cold start)
            temperature: >1.0 flattens likelihood via tempering (preserves mode!)
            repulsion_scale: >1.0 forces particle diversity
            
        Returns:
            neg_log_lik: Scalar, mean predictive NLL (after burn-in)
            vol_estimates: [T] volatility estimates
            
        Note on Tempering vs Noise Scaling:
            - Noise scaling (vol * scale): SHIFTS the mode (biases μ)
            - Tempering (log_p / T): FLATTENS the peak (preserves mode)
            
            We use tempering because it smooths the landscape for exploration
            without biasing where the optimum is located.
        """
        T = y.shape[0]
        N = self.n_particles
        
        rho = self.rho
        sigma_z = self.sigma_z
        mu = self.mu
        
        # Initialize from stationary distribution
        h_std = sigma_z / torch.sqrt(1 - rho**2 + 1e-6)
        h = mu + h_std * torch.randn(N, device=y.device)
        
        log_liks = []
        vol_estimates = []
        
        for t in range(T):
            # Truncated BPTT: Detach to limit gradient flow
            if truncate_every > 0 and t > 0 and (t % truncate_every) == 0:
                h = h.detach()
            
            h_prev = h
            
            # ═══════════════════════════════════════════════════════════════
            # PREDICT with ANTITHETIC SAMPLING
            # Use (ε, -ε) pairs for variance reduction
            # ═══════════════════════════════════════════════════════════════
            half_N = N // 2
            eps_half = torch.randn(half_N, device=y.device)
            eps = torch.cat([eps_half, -eps_half])
            
            h_pred = mu + rho * (h_prev - mu) + sigma_z * eps
            transition_mean = mu + rho * (h_prev - mu)
            
            y_t = y[t]
            
            # ═══════════════════════════════════════════════════════════════
            # WEIGHTING with LIKELIHOOD TEMPERING
            # Temperature > 1 flattens the likelihood WITHOUT shifting mode
            # This is the correct way to smooth for exploration
            # ═══════════════════════════════════════════════════════════════
            vol_pred = torch.exp(h_pred / 2)  # NO SCALING - preserves mode
            
            # Standard Student-t log-likelihood
            z = y_t / (vol_pred + 1e-8)
            log_p = (
                self.student_t_const 
                - torch.log(vol_pred + 1e-8)
                - (self.nu + 1) / 2 * torch.log1p(z**2 / self.nu)
            )
            
            # TEMPERING: Divide by temperature to flatten (T>1) or sharpen (T<1)
            # Mode location is preserved, only the sharpness changes
            log_p_tempered = log_p / temperature
            
            # Predictive log-likelihood (before Stein transport!)
            log_lik_t = torch.logsumexp(log_p_tempered, dim=0) - math.log(N)
            log_liks.append(log_lik_t)
            
            # Store volatility estimate
            vol_estimates.append(vol_pred.mean())
            
            # ═══════════════════════════════════════════════════════════════
            # STEIN TRANSPORT with BANDWIDTH FLOOR and REPULSION BOOST
            # ═══════════════════════════════════════════════════════════════
            
            # Bandwidth floor prevents gradient death when particles collapse
            bw = torch.max(
                h_pred.detach().std(),
                torch.tensor(self.bandwidth_floor, device=y.device)
            )
            
            h = h_pred
            
            for _ in range(self.n_stein_steps):
                # Gradient of STANDARD likelihood (no scaling on vol!)
                vol = torch.exp(h / 2)
                A = y_t**2 / (vol**2 * self.nu + 1e-8)
                grad_lik_standard = -0.5 + 0.5 * (self.nu + 1) * A / (1 + A)
                
                # Apply temperature to gradient (flattens without shifting)
                grad_lik = grad_lik_standard / temperature
                
                grad_prior = -(h - transition_mean) / (sigma_z**2 + 1e-8)
                grad_log_p = (grad_prior + grad_lik).clamp(-10, 10)
                
                # Stein step with repulsion boosting
                h = self._stein_step(h, grad_log_p, bw, repulsion_scale=repulsion_scale)
        
        # Loss: Negative log-likelihood (excluding burn-in)
        log_liks_tensor = torch.stack(log_liks)
        neg_log_lik = -log_liks_tensor[burn_in:].mean()
        
        return neg_log_lik, torch.stack(vol_estimates)


# =============================================================================
# ANNEALING SCHEDULES
# =============================================================================

def linear_anneal(progress, start, end):
    """Linear interpolation from start to end."""
    return start + (end - start) * progress


def cosine_anneal(progress, start, end):
    """Cosine annealing (smoother transition)."""
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


def exponential_anneal(progress, start, end, rate=3.0):
    """Exponential decay from start toward end."""
    return end + (start - end) * math.exp(-rate * progress)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_annealed(
    y_data,
    true_params,
    n_epochs: int = 100,
    lr: float = 0.015,
    batch_size: int = 500,
    burn_in: int = 20,
    anneal_schedule: str = 'linear_fast',  # Fast linear is better for physics-dominant
    temperature_start: float = 5.0,
    temperature_end: float = 1.0,
    temperature_half_life: int = 30,       # Epochs to reach T≈2.5
    repulsion_scale: float = 1.0,          # UNIT: Let implicit boost handle it
    verbose: bool = True,
):
    """
    Train parameter estimator with likelihood tempering.
    
    Physics-Dominant Configuration:
    - Likelihood-only tempering (Option C): Weaken data, preserve physics
    - Unit repulsion: Implicit boost from T>1 is enough
    - Faster schedule: Force model to face real likelihood sooner
    - Low Stein LR (0.02): Prevent particle teleportation
    
    Args:
        y_data: Numpy array of returns
        true_params: Dict with true parameters (for logging)
        n_epochs: Number of training epochs
        lr: Initial learning rate
        batch_size: Batch size for stochastic updates
        burn_in: Steps to exclude from loss
        anneal_schedule: 'linear_fast', 'exponential', or 'cosine'
        temperature_start/end: Tempering range (T>1 flattens likelihood)
        temperature_half_life: Epochs for T to decay halfway (for exponential)
        repulsion_scale: Keep at 1.0 for physics-dominant mode
        verbose: Print progress
        
    Returns:
        model: Trained model
        history: Dict of training history
    """
    y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)
    T = len(y_data)
    n_batches = max(1, (T - burn_in - 10) // batch_size)
    
    model = RobustSVPFEstimator(
        n_particles=128,
        n_stein_steps=1,
        stein_lr=0.02,               # LOW: Physics-dominant
        nu=true_params.get('nu', 8.0),
        init_rho=0.85,
        init_sigma_z=0.10,
        init_mu=-4.0,
        bandwidth_floor=0.5,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    
    history = {
        'nll': [], 'rho': [], 'sigma_z': [], 'mu': [],
        'temperature': [], 'repulsion_scale': []
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("DIFFERENTIABLE SVPF - PHYSICS-DOMINANT TEMPERING")
        print(f"{'='*70}")
        print(f"Data:       T={T}, batch_size={batch_size}, burn_in={burn_in}")
        print(f"Annealing:  {anneal_schedule} schedule")
        print(f"            temperature: {temperature_start:.1f} → {temperature_end:.1f}")
        print(f"            repulsion:   {repulsion_scale:.1f} (unit, implicit boost only)")
        print(f"            stein_lr:    0.02 (low, prevent teleportation)")
        print(f"True:       ρ={true_params['rho']:.3f}, σ_z={true_params['sigma_z']:.3f}, μ={true_params['mu']:.3f}")
        print(f"Init:       ρ={model.rho.item():.3f}, σ_z={model.sigma_z.item():.3f}, μ={model.mu.item():.3f}")
        print(f"Fixed:      ν={model.nu:.1f}")
        print(f"{'─'*70}")
    
    t0 = time()
    
    for epoch in range(n_epochs):
        # Compute temperature based on schedule
        progress = epoch / max(n_epochs - 1, 1)
        
        if anneal_schedule == 'linear_fast':
            # Linear decay, reaches T=1 at epoch 50 (halfway through)
            temperature = max(temperature_end, 
                            temperature_start - (temperature_start - temperature_end) * (progress * 2))
        elif anneal_schedule == 'exponential':
            # Exponential decay with configurable half-life
            decay_rate = math.log(2) / temperature_half_life
            temperature = temperature_end + (temperature_start - temperature_end) * math.exp(-decay_rate * epoch)
        elif anneal_schedule == 'cosine':
            temperature = cosine_anneal(progress, temperature_start, temperature_end)
        else:
            temperature = linear_anneal(progress, temperature_start, temperature_end)
        
        # Random batch selection
        perm = np.random.permutation(n_batches) * batch_size
        epoch_nll = 0.0
        n_batches_used = 0
        
        for start in perm[:min(10, n_batches)]:  # Limit batches per epoch
            end = min(start + batch_size, T)
            if end - start < burn_in + 20:
                continue
            y_batch = y_tensor[start:end]
            
            optimizer.zero_grad()
            nll, _ = model(
                y_batch,
                truncate_every=50,
                burn_in=burn_in,
                temperature=temperature,
                repulsion_scale=repulsion_scale,  # Unit (1.0)
            )
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_nll += nll.item()
            n_batches_used += 1
        
        # Step LR scheduler
        scheduler.step()
        
        # Record history
        p = model.get_params()
        avg_nll = epoch_nll / max(n_batches_used, 1)
        history['nll'].append(avg_nll)
        history['rho'].append(p['rho'])
        history['sigma_z'].append(p['sigma_z'])
        history['mu'].append(p['mu'])
        history['temperature'].append(temperature)
        history['repulsion_scale'].append(repulsion_scale)
        
        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            elapsed = time() - t0
            print(
                f"Epoch {epoch+1:3d} ({elapsed:5.1f}s) | "
                f"NLL={avg_nll:.3f} | "
                f"ρ={p['rho']:.4f} ({true_params['rho']:.3f}) | "
                f"σ={p['sigma_z']:.4f} ({true_params['sigma_z']:.3f}) | "
                f"μ={p['mu']:.3f} ({true_params['mu']:.3f}) | "
                f"T={temperature:.2f}"
            )
    
    if verbose:
        print(f"{'─'*70}")
        elapsed = time() - t0
        print(f"Training complete in {elapsed:.1f}s")
    
    return model, history


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(history, true_params, save_path='svpf_param_learning.png'):
    """Plot training convergence and annealing schedule."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    epochs = range(1, len(history['nll']) + 1)
    
    # NLL
    ax = axes[0, 0]
    ax.plot(epochs, history['nll'], 'b-', linewidth=1.5)
    ax.set_title('Predictive NLL', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL')
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
    ax = axes[0, 2]
    ax.plot(epochs, history['sigma_z'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params['sigma_z'], color='r', ls='--', linewidth=2, label='True')
    ax.set_title('σ_z (Vol-of-Vol)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # μ
    ax = axes[1, 0]
    ax.plot(epochs, history['mu'], 'b-', linewidth=1.5, label='Learned')
    ax.axhline(true_params['mu'], color='r', ls='--', linewidth=2, label='True')
    ax.set_title('μ (Mean Level)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annealing schedule
    ax = axes[1, 1]
    ax.plot(epochs, history['temperature'], 'g-', linewidth=1.5, label='Temperature')
    ax.plot(epochs, history['repulsion_scale'], 'm-', linewidth=1.5, label='Repulsion Scale')
    ax.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax.set_title('Tempering Schedule', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Scale Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error over time
    ax = axes[1, 2]
    rho_err = [abs(r - true_params['rho']) for r in history['rho']]
    sigma_err = [abs(s - true_params['sigma_z']) for s in history['sigma_z']]
    mu_err = [abs(m - true_params['mu']) for m in history['mu']]
    ax.semilogy(epochs, rho_err, 'b-', linewidth=1.5, label='|ρ - ρ*|')
    ax.semilogy(epochs, sigma_err, 'g-', linewidth=1.5, label='|σ_z - σ_z*|')
    ax.semilogy(epochs, mu_err, 'r-', linewidth=1.5, label='|μ - μ*|')
    ax.set_title('Parameter Errors', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_final_results(model, true_params):
    """Print final parameter estimates with errors."""
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
    # True parameters (crypto-calibrated)
    TRUE_PARAMS = {
        'rho': 0.94,
        'sigma_z': 0.18,
        'mu': -3.2,
        'nu': 8.0,
    }
    
    T = 3000  # Data length
    
    print("Generating synthetic SV data...")
    y, h_true = generate_sv_data(T, **TRUE_PARAMS, seed=42)
    print(f"Data statistics: y.std={y.std():.4f}, h.mean={h_true.mean():.3f}, h.std={h_true.std():.3f}")
    
    # Train with physics-dominant configuration
    # - Likelihood-only tempering (Option C)
    # - Unit repulsion (implicit boost from T>1 is enough)
    # - Faster schedule (linear_fast reaches T=1 at epoch 50)
    # - Low Stein LR (0.02, prevent teleportation)
    model, history = train_annealed(
        y,
        TRUE_PARAMS,
        n_epochs=100,
        lr=0.015,
        batch_size=500,
        burn_in=20,
        anneal_schedule='linear_fast',  # Reaches T=1 at epoch 50
        temperature_start=5.0,
        temperature_end=1.0,
        repulsion_scale=1.0,            # Unit: let physics provide variance
        verbose=True,
    )
    
    # Results
    print_final_results(model, TRUE_PARAMS)
    
    # Plot
    plot_results(history, TRUE_PARAMS, save_path='svpf_param_learning_annealed.png')
    
    # Export for CUDA SVPF
    print("Parameters ready for CUDA SVPF:")
    p = model.get_params()
    print(f"  rho     = {p['rho']:.6f}f")
    print(f"  sigma_z = {p['sigma_z']:.6f}f")
    print(f"  mu      = {p['mu']:.6f}f")
    print(f"  nu      = {p['nu']:.1f}f  // fixed")
