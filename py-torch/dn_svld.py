"""
dN-SVLD: Differentiable Newton-SVLD Particle Filter

PyTorch implementation of the Newton-SVLD particle filter for stochastic
volatility, designed for gradient-based parameter learning via backprop.

Matches the CUDA production filter:
  - No repulsion (pure kernel-smoothed score)
  - Langevin noise (temperature)
  - Newton preconditioning (target Hessian only, no Nk)
  - LOO kernel smoothing (IMQ kernel)
  - RMSProp step bounding
  - Step-size annealing (beta_factor = sqrt(beta))

Differentiability achieved by:
  - Reparameterized noise in predict + Langevin steps
  - Soft clamps (gradient flows at boundaries)
  - Fixed step/stage counts (no adaptive branching)
  - No resampling (fan mode uniform weights)
  - Truncated BPTT for memory efficiency

Usage:
    python dn_svld.py                         # defaults: learn mu, rho
    python dn_svld.py --learn mu rho sigma_z  # also learn sigma_z
    python dn_svld.py --n_particles 256       # more particles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import time
from typing import Optional, Tuple, Dict, Set


# =============================================================================
# Utilities
# =============================================================================

def soft_clamp_sym(x: torch.Tensor, bound: float) -> torch.Tensor:
    """Symmetric soft clamp to [-bound, bound] via scaled tanh."""
    return bound * torch.tanh(x / bound)


def soft_clamp_lower(x: torch.Tensor, lo: float) -> torch.Tensor:
    """Soft lower bound via softplus. x -> lo + softplus(x - lo)."""
    return lo + F.softplus(x - lo)


def inverse_sigmoid(x: float) -> float:
    """Inverse of sigmoid for initialization."""
    x = max(min(x, 0.999), 0.001)
    return math.log(x / (1.0 - x))


def generate_sv_data(
    T: int, mu: float, rho: float, sigma_z: float,
    nu_obs: float = 5.0, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic stochastic volatility data.

    Model:
        h_t = mu + rho * (h_{t-1} - mu) + sigma_z * eps_t
        y_t = exp(h_t / 2) * eta_t,   eta_t ~ Student-t(nu_obs)

    Returns (y, h_true) both shape [T].
    """
    torch.manual_seed(seed)
    h = torch.zeros(T)
    y = torch.zeros(T)

    h[0] = mu
    for t in range(1, T):
        h[t] = mu + rho * (h[t - 1] - mu) + sigma_z * torch.randn(1).item()

    chi2 = torch.distributions.Chi2(nu_obs)
    for t in range(T):
        vol = math.exp(h[t].item() / 2.0)
        eta = torch.randn(1).item() / math.sqrt(chi2.sample().item() / nu_obs)
        y[t] = vol * eta

    return y, h


# =============================================================================
# Model
# =============================================================================

class DNSVLD(nn.Module):
    """Differentiable Newton-SVLD Particle Filter.

    Parameters are stored in unconstrained space and transformed:
        mu:       unconstrained (identity)
        rho:      sigmoid(raw)         -> (0, 1)
        sigma_z:  exp(raw)             -> (0, inf)
        nu:       2 + softplus(raw)    -> (2, inf)
        lik_offset: unconstrained      -> R
    """

    def __init__(
        self,
        n_particles: int = 128,
        n_stein_steps: int = 4,
        n_anneal_stages: int = 3,
        temperature: float = 0.45,
        step_size: float = 0.5,
        rmsprop_rho: float = 0.9,
        newton_damping: float = 0.95,
        learn: Optional[Set[str]] = None,
        init_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.N = n_particles
        self.n_stein_steps = n_stein_steps
        self.n_anneal_stages = n_anneal_stages
        self.temperature = temperature
        self.base_step = step_size
        self.rmsprop_rho_val = rmsprop_rho
        self.rmsprop_eps = 1e-6
        self.newton_damping = newton_damping

        if learn is None:
            learn = {"mu", "rho"}
        if init_params is None:
            init_params = {}

        # Defaults matching production CUDA config
        p = {
            "mu": init_params.get("mu", -1.0),
            "rho": init_params.get("rho", 0.95),
            "sigma_z": init_params.get("sigma_z", 0.2),
            "nu": init_params.get("nu", 5.0),
            "lik_offset": init_params.get("lik_offset", 0.08),
        }

        # Store as raw (unconstrained) parameters
        self.mu_raw = nn.Parameter(torch.tensor(p["mu"], dtype=torch.float32))
        self.rho_raw = nn.Parameter(torch.tensor(inverse_sigmoid(p["rho"]), dtype=torch.float32))
        self.log_sigma_z = nn.Parameter(torch.tensor(math.log(p["sigma_z"]), dtype=torch.float32))
        self.nu_raw = nn.Parameter(torch.tensor(
            math.log(math.exp(p["nu"] - 2.0) - 1.0) if p["nu"] > 2.01 else 1.0,
            dtype=torch.float32
        ))
        self.lik_offset_raw = nn.Parameter(torch.tensor(p["lik_offset"], dtype=torch.float32))

        # Freeze non-learned params
        param_map = {
            "mu": self.mu_raw,
            "rho": self.rho_raw,
            "sigma_z": self.log_sigma_z,
            "nu": self.nu_raw,
            "lik_offset": self.lik_offset_raw,
        }
        self._learn = learn
        for name, param in param_map.items():
            if name not in learn:
                param.requires_grad_(False)

    # ----- Constrained parameter access -----

    @property
    def mu(self) -> torch.Tensor:
        return self.mu_raw

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_raw)

    @property
    def sigma_z(self) -> torch.Tensor:
        return self.log_sigma_z.exp()

    @property
    def nu(self) -> torch.Tensor:
        return 2.0 + F.softplus(self.nu_raw)

    @property
    def lik_offset(self) -> torch.Tensor:
        return self.lik_offset_raw

    def get_params_dict(self) -> Dict[str, float]:
        """Return current constrained parameter values."""
        return {
            "mu": self.mu.item(),
            "rho": self.rho.item(),
            "sigma_z": self.sigma_z.item(),
            "nu": self.nu.item(),
            "lik_offset": self.lik_offset.item(),
        }

    # ----- Predict step -----

    def predict(self, h: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Reparameterized predict: h_new = mu + rho*(h - mu) + sigma_z*eps."""
        mu_pred = self.mu + self.rho * (h - self.mu)
        return mu_pred + self.sigma_z * eps

    # ----- Analytical gradients -----

    def grad_log_posterior(
        self, h: torch.Tensor, h_prev: torch.Tensor, y_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute score and curvature of log p(h | y, h_prev).

        Prior: mixture of Gaussians  p(h|h_prev) = (1/N) sum_i N(h; mu_i, sigma_z^2)
        Likelihood: Student-t(nu)

        Returns:
            grad: [N]  score of tempered posterior
            curvature: [N]  negative Hessian (positive, for Newton)
        """
        N = h.shape[0]
        sigma_z = self.sigma_z
        sigma_z_sq = sigma_z * sigma_z
        nu = self.nu
        lik_offset = self.lik_offset

        # ===== PRIOR =====
        # Mixture means: mu_i = mu + rho * (h_prev_i - mu)
        mu_i = self.mu + self.rho * (h_prev - self.mu)  # [N]

        # diff[j, i] = h[j] - mu_i[i]
        diff = h.unsqueeze(1) - mu_i.unsqueeze(0)  # [N, N]

        # Log responsibilities (log-sum-exp stable)
        log_r = -0.5 * diff * diff / sigma_z_sq  # [N, N]
        log_r_max = log_r.max(dim=1, keepdim=True).values  # [N, 1]
        r = (log_r - log_r_max).exp()  # [N, N]
        sum_r = r.sum(dim=1) + 1e-8  # [N]

        # Weighted gradient: sum_i r_i * (-(h_j - mu_i) / sigma_z^2) / sum_i r_i
        grad_prior = -(r * diff / sigma_z_sq).sum(dim=1) / sum_r  # [N]
        hess_prior = -1.0 / sigma_z_sq  # scalar (Gaussian approx)

        # ===== LIKELIHOOD (Student-t) =====
        # vol = exp(h) is the observation VARIANCE (y_t = exp(h/2)*eta)
        vol = h.exp()  # exp(h), NOT exp(h/2)
        y_sq = y_t * y_t
        A = y_sq / (nu * vol + 1e-8)
        one_plus_A = 1.0 + A

        grad_lik = -0.5 + 0.5 * (nu + 1.0) * A / one_plus_A - lik_offset
        hess_lik = -0.5 * (nu + 1.0) * A / (one_plus_A * one_plus_A)

        # ===== COMBINE =====
        grad = grad_prior + grad_lik
        grad = soft_clamp_sym(grad, 10.0)

        curvature = -(hess_lik + hess_prior)
        curvature = soft_clamp_lower(curvature, 0.1)
        curvature = torch.clamp(curvature, max=100.0)  # hard upper (rarely hit)

        return grad, curvature

    # ----- Stein transport step -----

    def stein_step(
        self,
        h: torch.Tensor,
        grad: torch.Tensor,
        curvature: torch.Tensor,
        v_rms: torch.Tensor,
        eps_noise: torch.Tensor,
        beta_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One N-SVLD transport step.

        Kernel-smoothed score (LOO) -> Newton precondition -> RMSProp -> step + noise.

        Key: kernel matrix uses detached h to prevent N² Jacobian explosion through
        backprop. Gradients flow through the score (theta-dependent) not the kernel
        (transport machinery). This preserves d(score)/d(theta) at each step.
        """
        N = h.shape[0]
        device = h.device

        # Detach h for kernel computation — breaks N² backward chain
        h_det = h.detach()

        # Bandwidth: 2 * var / log(N+1)  (matches CUDA)
        h_var = h_det.var() + 1e-8
        bw_sq = 2.0 * h_var / math.log(N + 1.0)
        bw_sq = torch.clamp(bw_sq, min=1e-6)

        # Pairwise: diff[i,j] = h[i] - h[j]  (detached for kernel)
        diff = h_det.unsqueeze(0) - h_det.unsqueeze(1)  # [N, N]
        dist_sq = diff * diff / bw_sq

        # IMQ kernel: K = 1 / (1 + r^2/bw^2)
        K = 1.0 / (1.0 + dist_sq)  # [N, N]

        # LOO mask
        mask = 1.0 - torch.eye(N, device=device)
        K_masked = K * mask

        # Kernel-smoothed Hessian (Newton preconditioner) — detached
        K_sum = K_masked.sum(dim=1).clamp(min=1e-6)  # [N]
        H_w = (K_masked * curvature.detach().unsqueeze(0)).sum(dim=1) / K_sum  # [N]
        H_w = H_w.clamp(min=0.1, max=100.0)
        inv_H = (1.0 / H_w).detach()

        # Kernel-smoothed score (grad carries theta dependence)
        k_grad_sum = (K_masked * grad.unsqueeze(0)).sum(dim=1)  # [N]

        # Stein update: phi = (1/(N-1)) * K_grad_sum * inv_H * damping
        n_ref = float(N - 1)
        phi = k_grad_sum / n_ref * inv_H * self.newton_damping

        # RMSProp (detach v_rms from graph)
        v_rms_new = (self.rmsprop_rho_val * v_rms + (1.0 - self.rmsprop_rho_val) * phi.detach() * phi.detach()).detach()
        precond = 1.0 / (v_rms_new + self.rmsprop_eps).sqrt()

        # Step
        effective_step = self.base_step * beta_factor
        drift = effective_step * phi * precond

        # Langevin noise (reparameterized)
        diffusion = math.sqrt(2.0 * effective_step * self.temperature) * eps_noise

        h_new = h + drift + diffusion

        # Soft clamp log-vol
        h_new = soft_clamp_sym(h_new, 20.0)

        return h_new, v_rms_new

    # ----- Full forward pass -----

    def forward(
        self,
        y_seq: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
        chunk_size: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Run filter on observation sequence with truncated BPTT.

        Args:
            y_seq: [T] returns
            h_init: [N] initial particles (optional)
            chunk_size: truncated BPTT window

        Returns dict with:
            vol: [T]  volatility estimates
            h_mean: [T]  mean log-vol
            log_lik: [T]  per-step log-likelihood
        """
        T = y_seq.shape[0]
        N = self.N
        device = y_seq.device
        n_stages = self.n_anneal_stages
        n_steps = self.n_stein_steps

        # Initialize particles
        if h_init is None:
            with torch.no_grad():
                h = self.mu.detach().expand(N) + self.sigma_z.detach() * torch.randn(N, device=device)
        else:
            h = h_init.clone()

        v_rms = torch.ones(N, device=device) * 0.01  # warm-start
        vol_out = torch.zeros(T, device=device)
        h_mean_out = torch.zeros(T, device=device)
        ll_out = torch.zeros(T, device=device)

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)

            # Truncated BPTT: detach state at chunk boundary
            h = h.detach()
            v_rms = torch.ones(N, device=device) * 0.01  # warm-start

            for t in range(chunk_start, chunk_end):
                h_prev = h
                y_t = y_seq[t]

                # 1. PREDICT (reparameterized)
                eps_predict = torch.randn(N, device=device)
                h = self.predict(h, eps_predict)

                # 2. PREDICTIVE LOG-LIKELIHOOD (before transport!)
                # This measures how well the model predicted y_t from the prior.
                # After Stein transport, particles are at the mode → ∂logp/∂h ≈ 0 → no gradient.
                # Computing here preserves the gradient signal about θ.
                nu = self.nu
                vol_pred = h.exp()  # exp(h), observation variance
                A_pred = (y_t * y_t) / (nu * vol_pred + 1e-8)
                log_p_pred = (
                    torch.lgamma((nu + 1.0) / 2.0)
                    - torch.lgamma(nu / 2.0)
                    - 0.5 * torch.log(torch.tensor(math.pi, device=device) * nu)
                    - 0.5 * h
                    - (nu + 1.0) / 2.0 * torch.log1p(A_pred)
                )
                ll_out[t] = torch.logsumexp(log_p_pred, dim=0) - math.log(N)

                # 3. STEIN TRANSPORT LOOP (refines particles for next predict)
                for stage in range(n_stages):
                    beta = float(stage + 1) / float(n_stages)
                    beta_factor = math.sqrt(beta)

                    for s in range(n_steps):
                        grad, curvature = self.grad_log_posterior(h, h_prev, y_t)
                        eps_langevin = torch.randn(N, device=device)
                        h, v_rms = self.stein_step(
                            h, grad, curvature, v_rms, eps_langevin, beta_factor
                        )

                # NaN guard: if particles diverge, reset to predict value
                if not torch.isfinite(h).all():
                    h = h_prev.detach()

                # Detach after transport: gradient only flows through predict step
                # This ensures clean ∂loss[t]/∂θ without chain-through-time complications
                h = h.detach()

                # 4. OUTPUT
                h_mean_out[t] = h.mean()
                vol_out[t] = (h / 2.0).exp().mean()

        return {"vol": vol_out, "h_mean": h_mean_out, "log_lik": ll_out}


# =============================================================================
# Training
# =============================================================================

def train(
    model: DNSVLD,
    y_seq: torch.Tensor,
    h_true: Optional[torch.Tensor] = None,
    n_epochs: int = 200,
    lr: float = 3e-3,
    chunk_size: int = 100,
    log_every: int = 10,
) -> Dict[str, list]:
    """Train dN-SVLD via backprop on predictive log-likelihood.

    Loss: -mean(log p(y_t | particles))
    Metric: RMSE(h_mean, h_true)  if h_true provided
    """
    device = y_seq.device
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"loss": [], "rmse": [], "params": []}
    t_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(y_seq, chunk_size=chunk_size)

        # Loss: negative predictive log-likelihood
        loss = -out["log_lik"].mean()

        # NaN guard: skip update if loss is NaN/Inf
        if not torch.isfinite(loss):
            print(f"  [epoch {epoch}] NaN/Inf loss detected, skipping update")
            optimizer.zero_grad()
            continue

        loss.backward()

        # NaN guard: zero any NaN gradients
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Gradient diagnostics
        if epoch % log_every == 0:
            for name, param in [("mu", model.mu_raw), ("rho", model.rho_raw)]:
                if param.grad is not None:
                    print(f"    grad_{name}: {param.grad.item():.6e}")

        optimizer.step()
        scheduler.step()

        # Track
        history["loss"].append(loss.item())
        params = model.get_params_dict()
        history["params"].append(params)

        rmse = float("nan")
        if h_true is not None:
            with torch.no_grad():
                rmse = (out["h_mean"] - h_true.to(device)).pow(2).mean().sqrt().item()
        history["rmse"].append(rmse)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            msg = (
                f"[{epoch:4d}/{n_epochs}] loss={loss.item():8.3f}  "
                f"RMSE={rmse:.4f}  "
                f"mu={params['mu']:+.4f}  rho={params['rho']:.4f}  "
                f"sig_z={params['sigma_z']:.4f}  nu={params['nu']:.2f}  "
                f"lr={lr_now:.1e}  ({elapsed:.1f}s)"
            )
            print(msg)

    return history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="dN-SVLD: Differentiable Newton-SVLD Filter")
    parser.add_argument("--T", type=int, default=500, help="Sequence length")
    parser.add_argument("--n_particles", type=int, default=128, help="Number of particles")
    parser.add_argument("--n_stein_steps", type=int, default=4, help="Stein steps per stage")
    parser.add_argument("--n_anneal_stages", type=int, default=3, help="Annealing stages")
    parser.add_argument("--learn", nargs="+", default=["mu", "rho"],
                        help="Parameters to learn (mu, rho, sigma_z, nu, lik_offset)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--chunk_size", type=int, default=20, help="Truncated BPTT window")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Ground truth parameters ----
    TRUE = {"mu": -1.0, "rho": 0.98, "sigma_z": 0.15, "nu": 5.0}

    # ---- Generate data ----
    y, h_true = generate_sv_data(
        args.T, TRUE["mu"], TRUE["rho"], TRUE["sigma_z"],
        nu_obs=TRUE["nu"], seed=args.seed
    )
    y = y.to(device)
    h_true = h_true.to(device)
    print(f"Data: T={args.T}, y range [{y.min():.3f}, {y.max():.3f}]")
    print(f"True: mu={TRUE['mu']}, rho={TRUE['rho']}, sigma_z={TRUE['sigma_z']}, nu={TRUE['nu']}")

    # ---- Build model with WRONG initial params (to test learning) ----
    wrong_init = {
        "mu": 0.0,       # true: -1.0
        "rho": 0.85,     # true: 0.98
        "sigma_z": 0.15, # true: 0.15 (frozen by default)
        "nu": 5.0,       # true: 5.0  (frozen by default)
        "lik_offset": 0.08,
    }

    model = DNSVLD(
        n_particles=args.n_particles,
        n_stein_steps=args.n_stein_steps,
        n_anneal_stages=args.n_anneal_stages,
        learn=set(args.learn),
        init_params=wrong_init,
    ).to(device)

    learn_str = ", ".join(sorted(args.learn))
    print(f"Learning: {learn_str}")
    print(f"Init:  {model.get_params_dict()}")

    # ---- Baseline (before training) ----
    with torch.no_grad():
        out0 = model(y, chunk_size=args.chunk_size)
        rmse0 = (out0["h_mean"] - h_true).pow(2).mean().sqrt().item()
        ll0 = out0["log_lik"].mean().item()
    print(f"Baseline: RMSE={rmse0:.4f}, mean_ll={ll0:.3f}")

    # ---- Train ----
    print("\n--- Training ---")
    history = train(
        model, y, h_true,
        n_epochs=args.epochs,
        lr=args.lr,
        chunk_size=args.chunk_size,
    )

    # ---- Final eval ----
    with torch.no_grad():
        out_final = model(y, chunk_size=args.chunk_size)
        rmse_final = (out_final["h_mean"] - h_true).pow(2).mean().sqrt().item()
        ll_final = out_final["log_lik"].mean().item()

    print(f"\n--- Results ---")
    print(f"Baseline RMSE: {rmse0:.4f}  ->  Final RMSE: {rmse_final:.4f}")
    print(f"Baseline LL:   {ll0:.3f}  ->  Final LL:   {ll_final:.3f}")
    print(f"Learned: {model.get_params_dict()}")
    print(f"True:    {TRUE}")


if __name__ == "__main__":
    main()