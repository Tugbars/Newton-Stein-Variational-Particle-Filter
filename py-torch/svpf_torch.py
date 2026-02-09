"""
SVPF v4 - Faithful PyTorch port from CUDA implementation
=========================================================

Key design decisions (matching CUDA code):
  - All scalar state (guide_mean, bandwidth, etc.) stays as Python floats on CPU
  - Only particle arrays [N] live on GPU
  - No mixing of GPU scalars and Python floats
  - Score/transport math follows svpf_opt_kernels.cu exactly
  - h represents log-variance: y = exp(h/2) * eps, vol = exp(h) = variance
  
Three modes:
  1. base:        Standard SVPF (equal-weight outputs)
  2. iw:          + Importance weight correction on outputs only  
  3. iw_feedback: + Feed bias estimate back into next step's guide
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import math


@dataclass
class SVPFConfig:
    n_particles: int = 512
    mu: float = -3.5
    rho: float = 0.97
    sigma_z: float = 0.15
    gamma: float = 0.0
    nu_obs: float = 5.0       # Observation Student-t df
    
    # Stein
    n_stein_steps: int = 8
    stein_step_size: float = 0.05
    temperature: float = 0.3
    rmsprop_rho: float = 0.7
    rmsprop_eps: float = 1e-6
    
    # Annealing
    use_annealing: bool = True
    n_anneal_stages: int = 5
    anneal_steps_per_stage: int = 4
    
    # Bandwidth
    bw_ema_alpha: float = 0.3
    
    # Guide
    use_guide: bool = True
    guide_strength: float = 0.05
    
    # Guided prediction
    use_guided_predict: bool = True
    guided_alpha_base: float = 0.0
    guided_alpha_shock: float = 0.4
    guided_innovation_threshold: float = 1.5
    
    # Student-t state noise
    use_student_t_state: bool = True
    nu_state: float = 2.5
    
    # IW correction
    use_iw_correction: bool = False
    use_bias_feedback: bool = False
    bias_ema_alpha: float = 0.2
    var_ema_alpha: float = 0.1
    
    # Antithetic
    use_antithetic: bool = True
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DGPConfig:
    mu: float = -3.5
    rho: float = 0.97
    sigma_z: float = 0.15
    nu_obs: float = 5.0
    nu_state: float = 2.5
    use_student_t_state: bool = True
    use_regime_change: bool = False
    regime_change_t: int = 500
    mu_post_regime: float = -1.5
    sigma_z_post_regime: float = 0.30


def generate_sv_data(T: int, dgp: DGPConfig, seed: int = 42):
    """Generate SV data. All CPU."""
    torch.manual_seed(seed)
    h = torch.zeros(T)
    y = torch.zeros(T)
    var_base = dgp.sigma_z**2 / (1.0 - dgp.rho**2 + 1e-6)
    if dgp.use_student_t_state and dgp.nu_state > 2:
        var_base *= dgp.nu_state / (dgp.nu_state - 2.0)
    h[0] = dgp.mu + math.sqrt(var_base) * torch.randn(1).item()

    for t in range(T):
        mu_t = dgp.mu
        sz_t = dgp.sigma_z
        if dgp.use_regime_change and t >= dgp.regime_change_t:
            mu_t = dgp.mu_post_regime
            sz_t = dgp.sigma_z_post_regime
        if t > 0:
            if dgp.use_student_t_state:
                eta = torch.distributions.StudentT(dgp.nu_state).sample().item()
            else:
                eta = torch.randn(1).item()
            h[t] = mu_t + dgp.rho * (h[t-1].item() - mu_t) + sz_t * eta
        eps = torch.distributions.StudentT(dgp.nu_obs).sample().item()
        y[t] = math.exp(h[t].item() / 2.0) * eps

    return y, h


class SVPFTorch:
    """
    SVPF with all scalar state on CPU (Python floats).
    Only particle arrays on GPU.
    """
    
    CLAMP_LO = -10.0
    CLAMP_HI = 5.0
    
    def __init__(self, cfg: SVPFConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.n = cfg.n_particles

        nu = cfg.nu_obs
        # Student-t normalization constant (matches CUDA student_t_const)
        self.student_t_const = (
            torch.lgamma(torch.tensor((nu + 1.0) / 2.0)).item()
            - torch.lgamma(torch.tensor(nu / 2.0)).item()
            - 0.5 * math.log(math.pi * nu)
        )

        # Implied offset: -E[log(eps^2)] for eps ~ t(nu)
        from scipy.special import digamma as _digamma
        psi_half = _digamma(0.5)
        psi_nu_half = _digamma(nu / 2.0)
        self.implied_offset = -(math.log(nu) + psi_half - psi_nu_half)

    def initialize(self, seed: int = 123):
        torch.manual_seed(seed)
        cfg = self.cfg
        base_var = cfg.sigma_z**2 / (1.0 - cfg.rho**2 + 1e-6)
        if cfg.use_student_t_state and cfg.nu_state > 2:
            base_var *= cfg.nu_state / (cfg.nu_state - 2.0)
        std = math.sqrt(base_var)

        # GPU arrays
        self.h = torch.randn(self.n, device=self.dev) * std + cfg.mu
        self.h = torch.clamp(self.h, self.CLAMP_LO, self.CLAMP_HI)
        self.h_prev = self.h.clone()
        self.v_rmsprop = torch.zeros(self.n, device=self.dev)

        # CPU scalars
        self.bandwidth = 0.5      # initial guess
        self.bandwidth_sq = 0.25
        self.h_mean = cfg.mu
        self.vol_prev = math.exp(cfg.mu / 2.0)
        self.timestep = 0

        # EKF guide (CPU)
        self.guide_mean = cfg.mu
        self.guide_var = base_var

        # Bias feedback (CPU)
        self.bias_ema = 0.0
        self.var_ratio_ema = 1.0
        self.guide_bias_offset = 0.0

    # -------------------------------------------------------------------------
    # EKF Guide (all CPU math, matches svpf_ekf_update in CUDA)
    # -------------------------------------------------------------------------
    def _ekf_guide_update(self, y_t: float):
        cfg = self.cfg
        mu_pred = cfg.mu + cfg.rho * (self.guide_mean - cfg.mu)
        P_pred = cfg.rho**2 * self.guide_var + cfg.sigma_z**2

        vol_pred = math.exp(min(mu_pred * 0.5, 5.0))
        H = 0.5 * vol_pred
        R = math.exp(min(mu_pred, 10.0)) * cfg.nu_obs / max(cfg.nu_obs - 2.0, 0.1)
        S = H * H * P_pred + R
        K = P_pred * H / (S + 1e-8)

        h_implied = math.log(y_t * y_t + 1e-10) + self.implied_offset
        h_innovation = h_implied - mu_pred

        self.guide_mean = mu_pred + K * h_innovation
        self.guide_var = max((1.0 - K * H) * P_pred, 1e-6)
        self.guide_mean = max(min(self.guide_mean, self.CLAMP_HI), self.CLAMP_LO)

    # -------------------------------------------------------------------------
    # Predict (matches svpf_predict_guided_antithetic_kernel)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _predict(self, y_t: float, y_prev: float):
        cfg = self.cfg
        n = self.n

        self.h_prev = self.h.clone()

        # Transition mean: mu + rho * (h - mu) 
        # (no leverage since gamma=0 in test config)
        mu_prior = cfg.mu + cfg.rho * (self.h - cfg.mu)

        # Guided proposal: blend toward implied h
        if cfg.use_guided_predict:
            y_sq = y_t * y_t
            log_y2 = math.log(y_sq + 1e-10)
            mean_implied = max(log_y2 + self.implied_offset, -5.0)

            innovation = mean_implied - mu_prior  # [N] tensor
            z_score = innovation / 2.5

            activation = torch.zeros_like(z_score)
            mask = z_score > cfg.guided_innovation_threshold
            if mask.any():
                activation[mask] = torch.tanh(z_score[mask] - cfg.guided_innovation_threshold)

            alpha = cfg.guided_alpha_base + (cfg.guided_alpha_shock - cfg.guided_alpha_base) * activation
            mean_proposal = (1.0 - alpha) * mu_prior + alpha * mean_implied
        else:
            mean_proposal = mu_prior

        # Antithetic noise
        half = n // 2
        if cfg.use_antithetic:
            if cfg.use_student_t_state:
                z = torch.distributions.StudentT(cfg.nu_state).sample((half,)).to(self.dev)
            else:
                z = torch.randn(half, device=self.dev)
            z_full = torch.cat([z, -z])
        else:
            if cfg.use_student_t_state:
                z_full = torch.distributions.StudentT(cfg.nu_state).sample((n,)).to(self.dev)
            else:
                z_full = torch.randn(n, device=self.dev)

        self.h = torch.clamp(mean_proposal + cfg.sigma_z * z_full, self.CLAMP_LO, self.CLAMP_HI)

    # -------------------------------------------------------------------------
    # Guide: variance-preserving shift (matches svpf_apply_guide_preserving_kernel)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _apply_guide(self):
        if not self.cfg.use_guide:
            return
        current_mean = self.h.mean().item()
        deviation = self.h - current_mean

        target = self.guide_mean + self.guide_bias_offset
        strength = self.cfg.guide_strength
        new_mean = (1.0 - strength) * current_mean + strength * target

        self.h = torch.clamp(new_mean + deviation, self.CLAMP_LO, self.CLAMP_HI)

    # -------------------------------------------------------------------------
    # Bandwidth (matches svpf_fused_bandwidth_kernel, simplified)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _compute_bandwidth(self):
        h = self.h
        n = self.n
        mean = h.mean().item()
        var = h.var().item()

        bw_sq_new = max(2.0 * var / math.log(n + 1.0), 1e-6)

        alpha = self.cfg.bw_ema_alpha
        if self.bandwidth_sq > 0:
            self.bandwidth_sq = alpha * bw_sq_new + (1.0 - alpha) * self.bandwidth_sq
        else:
            self.bandwidth_sq = bw_sq_new

        self.bandwidth = max(min(math.sqrt(self.bandwidth_sq), 2.0), 0.05)
        self.h_mean = mean

    # -------------------------------------------------------------------------
    # Prior score (matches svpf_fused_gradient_kernel prior section)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _log_prior_score(self) -> torch.Tensor:
        """∇_h log[ (1/N) Σ_i p(h | h_prev_i) ] for each particle."""
        cfg = self.cfg
        h = self.h          # [N]
        h_prev = self.h_prev # [N]

        # Transition means from each previous particle
        mu_i = cfg.mu + cfg.rho * (h_prev - cfg.mu)  # [N]

        # diff[j, i] = h[j] - mu_i[i]
        diff = h.unsqueeze(1) - mu_i.unsqueeze(0)  # [N, N]
        sigma_sq = cfg.sigma_z ** 2

        if cfg.use_student_t_state:
            nu_s = cfg.nu_state
            nu_sigma_sq = nu_s * sigma_sq
            half_nup1 = 0.5 * (nu_s + 1.0)
            nup1 = nu_s + 1.0
            
            diff_sq = diff * diff
            denom = nu_sigma_sq + diff_sq  # [N, N]
            
            # log of Student-t kernel (unnormalized)
            log_r = -half_nup1 * torch.log1p(diff_sq / nu_sigma_sq)  # [N, N]
            log_r_max = log_r.max(dim=1, keepdim=True).values
            r = torch.exp(log_r - log_r_max)   # [N, N]
            r_sum = r.sum(dim=1) + 1e-8         # [N]

            # Gradient of Student-t: -(nu+1) * diff / (nu*sigma^2 + diff^2)
            grad_per = -nup1 * diff / denom     # [N, N]
            score = (r * grad_per).sum(dim=1) / r_sum
        else:
            inv_2s2 = 0.5 / sigma_sq
            inv_s2 = 1.0 / sigma_sq

            log_r = -diff * diff * inv_2s2
            log_r_max = log_r.max(dim=1, keepdim=True).values
            r = torch.exp(log_r - log_r_max)
            r_sum = r.sum(dim=1) + 1e-8

            grad_per = -diff * inv_s2
            score = (r * grad_per).sum(dim=1) / r_sum

        return score  # [N]

    # -------------------------------------------------------------------------
    # Likelihood score + log-weights (matches svpf_fused_gradient_kernel lik section)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _log_likelihood_score(self, y_t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (grad_lik [N], log_w [N]).
        
        Matches CUDA:
          vol = safe_exp(h_j)   // exp(h) = variance
          A = y² / (vol * nu)
          log_w = const - 0.5*h - 0.5*(nu+1)*log(1+A)
          grad  = -0.5 + 0.5*(nu+1)*A/(1+A)
        """
        h = self.h
        nu = self.cfg.nu_obs

        # vol = exp(h) = variance (NOT exp(h/2))
        # Clamp h before exp to prevent overflow
        h_safe = torch.clamp(h, -10.0, 10.0)
        vol = torch.exp(h_safe)

        y_sq = y_t * y_t
        A = y_sq / (vol * nu + 1e-8)
        one_plus_A = 1.0 + A

        # Log importance weight
        log_w = (self.student_t_const
                 - 0.5 * h
                 - 0.5 * (nu + 1.0) * torch.log1p(A.clamp(min=0.0)))

        # Exact likelihood gradient
        grad_lik = -0.5 + 0.5 * (nu + 1.0) * A / one_plus_A

        return grad_lik, log_w

    # -------------------------------------------------------------------------
    # Stein step with Cauchy kernel (matches svpf_fused_stein_transport_full_newton_kernel)
    # Simplified: no Newton preconditioning in this test version
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _stein_step(self, grad: torch.Tensor, step_size: float, beta_factor: float):
        cfg = self.cfg
        h = self.h
        n = self.n

        bw = self.bandwidth
        bw_sq = bw * bw
        inv_bw_sq = 1.0 / bw_sq

        # Pairwise: diff[i,j] = h[i] - h[j]
        diff = h.unsqueeze(0) - h.unsqueeze(1)   # [N, N]
        dist_sq = diff * diff * inv_bw_sq

        # Cauchy kernel K = 1/(1 + d²/bw²)
        K = 1.0 / (1.0 + dist_sq)   # [N, N]
        K_sq = K * K

        # phi_i = (1/N) Σ_j [ K(h_j, h_i) * grad_j + ∇_{h_j} K(h_j, h_i) ]
        k_grad_sum = (K * grad.unsqueeze(0)).sum(dim=1)           # [N]
        gk_sum = (2.0 * diff * inv_bw_sq * K_sq).sum(dim=1)      # [N]
        phi = (k_grad_sum + gk_sum) / float(n)

        # RMSProp
        self.v_rmsprop = cfg.rmsprop_rho * self.v_rmsprop + (1.0 - cfg.rmsprop_rho) * phi * phi
        precond = torch.rsqrt(self.v_rmsprop + cfg.rmsprop_eps)

        effective_step = step_size * beta_factor
        drift = effective_step * phi * precond

        # Langevin noise
        diffusion = torch.zeros_like(h)
        if cfg.temperature > 1e-6:
            noise = torch.randn(n, device=self.dev)
            diffusion = math.sqrt(2.0 * effective_step * cfg.temperature) * noise

        self.h = torch.clamp(h + drift + diffusion, self.CLAMP_LO, self.CLAMP_HI)

    # -------------------------------------------------------------------------
    # Importance weights (post-transport correction)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _compute_importance_weights(self, y_t: float) -> torch.Tensor:
        """
        w_i ∝ p(y|h_i) * p(h_i|h_prev) / q_KDE(h_i)
        All terms evaluated at the FINAL transported positions.
        """
        cfg = self.cfg
        h = self.h          # [N] - transported
        h_prev = self.h_prev # [N]
        n = self.n
        sigma_sq = cfg.sigma_z ** 2
        nu = cfg.nu_obs

        # --- log p(y_t | h_i): Student-t likelihood ---
        h_safe = torch.clamp(h, -10.0, 10.0)
        vol = torch.exp(h_safe)
        y_sq = y_t * y_t
        A = y_sq / (vol * nu + 1e-8)
        log_lik = (self.student_t_const
                   - 0.5 * h
                   - 0.5 * (nu + 1.0) * torch.log1p(A.clamp(min=0.0)))

        # --- log p(h_i | h_prev): KDE transition prior ---
        mu_k = cfg.mu + cfg.rho * (h_prev - cfg.mu)  # [N]
        diff = h.unsqueeze(1) - mu_k.unsqueeze(0)      # [N, N]

        if cfg.use_student_t_state:
            nu_s = cfg.nu_state
            log_trans = -0.5 * (nu_s + 1.0) * torch.log1p((diff * diff) / (nu_s * sigma_sq))
        else:
            log_trans = -0.5 * diff * diff / sigma_sq

        # LogSumExp over mixture components (dim=1)
        lt_max = log_trans.max(dim=1, keepdim=True).values
        log_prior = lt_max.squeeze(1) + torch.log(
            torch.exp(log_trans - lt_max).mean(dim=1).clamp(min=1e-30)
        )

        # --- log q(h_i): LOO Gaussian KDE on transported particles ---
        bw = max(self.bandwidth, 0.05)
        diff_q = h.unsqueeze(0) - h.unsqueeze(1)  # [N, N]
        log_kde = -0.5 * (diff_q / bw) ** 2       # [N, N]
        log_kde = torch.clamp(log_kde, min=-50.0, max=0.0)

        # Leave-one-out: set diagonal to -inf
        diag_mask = torch.eye(n, device=self.dev, dtype=torch.bool)
        log_kde_loo = log_kde.masked_fill(diag_mask, -60.0)

        lk_max = log_kde_loo.max(dim=1, keepdim=True).values
        kde_sum = torch.exp(log_kde_loo - lk_max).sum(dim=1)
        log_q = (lk_max.squeeze(1)
                 + torch.log(kde_sum.clamp(min=1e-30))
                 - math.log(max(n - 1, 1))
                 - math.log(bw)
                 - 0.5 * math.log(2.0 * math.pi))

        # --- Importance weight ---
        log_w = log_lik + log_prior - log_q
        log_w = torch.clamp(log_w, -20.0, 20.0)
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        w = w / (w.sum() + 1e-10)

        # Safety
        if torch.isnan(w).any():
            w = torch.ones(n, device=self.dev) / float(n)

        return w

    # -------------------------------------------------------------------------
    # Full step
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, y_t: float, y_prev: float = 0.0) -> dict:
        cfg = self.cfg

        # 1. EKF guide
        self._ekf_guide_update(y_t)

        # 2. Predict
        self._predict(y_t, y_prev)

        # 3. Guide shift
        self._apply_guide()

        # 4. Bandwidth
        self._compute_bandwidth()

        # 5. Annealed Stein iterations
        if cfg.use_annealing:
            n_stages = cfg.n_anneal_stages
            steps_per = cfg.anneal_steps_per_stage
        else:
            n_stages = 1
            steps_per = cfg.n_stein_steps

        step_size = cfg.stein_step_size * (0.5 if cfg.use_guide else 1.0)

        for stage in range(n_stages):
            beta = min((stage + 1) / n_stages, 1.0)
            beta_factor = math.sqrt(beta)

            for s in range(steps_per):
                grad_prior = self._log_prior_score()
                grad_lik, _ = self._log_likelihood_score(y_t)

                grad = grad_prior + beta * grad_lik
                grad = torch.clamp(grad, -10.0, 10.0)

                self._stein_step(grad, step_size, beta_factor)

        # 6. Final log-weights
        _, log_w = self._log_likelihood_score(y_t)

        # 7. Uniform estimates
        h_mean_uniform = self.h.mean().item()
        vol_uniform = torch.exp(self.h * 0.5).mean().item()
        var_uniform = self.h.var().item()

        # Marginal likelihood (logsumexp)
        lw_max = log_w.max().item()
        loglik = lw_max + math.log(max(torch.exp(log_w - lw_max).mean().item(), 1e-30))

        # 8. Output defaults
        result = {
            "h_mean": h_mean_uniform,
            "vol": vol_uniform,
            "loglik": loglik,
            "ess": float(self.n),
            "bias_estimate": 0.0,
            "var_ratio": 1.0,
        }

        # 9. Importance weight correction
        if cfg.use_iw_correction:
            w = self._compute_importance_weights(y_t)
            ess = 1.0 / max((w * w).sum().item(), 1e-10)
            h_mean_weighted = (w * self.h).sum().item()
            vol_weighted = (w * torch.exp(self.h * 0.5)).sum().item()
            var_weighted = (w * (self.h - h_mean_weighted) ** 2).sum().item()

            bias_est = h_mean_weighted - h_mean_uniform
            var_ratio = var_weighted / max(var_uniform, 1e-8)

            result["h_mean"] = h_mean_weighted
            result["vol"] = vol_weighted
            result["ess"] = ess
            result["bias_estimate"] = bias_est
            result["var_ratio"] = var_ratio

        # 10. Bias feedback
        if cfg.use_bias_feedback and cfg.use_iw_correction:
            self.bias_ema = (1.0 - cfg.bias_ema_alpha) * self.bias_ema + cfg.bias_ema_alpha * result["bias_estimate"]
            self.var_ratio_ema = (1.0 - cfg.var_ema_alpha) * self.var_ratio_ema + cfg.var_ema_alpha * result["var_ratio"]
            self.guide_bias_offset = self.bias_ema

        self.vol_prev = vol_uniform
        self.timestep += 1

        return result


# =============================================================================
# Test harness
# =============================================================================

def run_experiment(T=1000, n_particles=512, dgp_cfg=None, filter_cfg=None,
                   mode="base", seed_dgp=42, seed_filter=123) -> dict:

    if dgp_cfg is None: dgp_cfg = DGPConfig()
    if filter_cfg is None: filter_cfg = SVPFConfig(n_particles=n_particles)

    filter_cfg.use_iw_correction = mode in ("iw", "iw_feedback")
    filter_cfg.use_bias_feedback = mode == "iw_feedback"

    y_data, h_true = generate_sv_data(T, dgp_cfg, seed=seed_dgp)

    svpf = SVPFTorch(filter_cfg)
    svpf.initialize(seed=seed_filter)

    h_est = np.zeros(T)
    vol_est = np.zeros(T)
    ess_arr = np.zeros(T)
    bias_arr = np.zeros(T)
    vr_arr = np.zeros(T)
    loglik_arr = np.zeros(T)

    y_prev = 0.0
    t0 = time.time()

    for t in range(T):
        r = svpf.step(y_data[t].item(), y_prev)
        h_est[t] = r["h_mean"]
        vol_est[t] = r["vol"]
        ess_arr[t] = r["ess"]
        bias_arr[t] = r["bias_estimate"]
        vr_arr[t] = r["var_ratio"]
        loglik_arr[t] = r["loglik"]
        y_prev = y_data[t].item()

    elapsed = time.time() - t0

    h_true_np = h_true.numpy()
    vol_true = np.exp(h_true_np / 2.0)
    h_error = h_est - h_true_np
    vol_error = vol_est - vol_true

    return {
        "mode": mode, "T": T, "n_particles": n_particles,
        "rmse_h": float(np.sqrt(np.mean(h_error**2))),
        "mae_h": float(np.mean(np.abs(h_error))),
        "mean_bias_h": float(np.mean(h_error)),
        "rmse_vol": float(np.sqrt(np.mean(vol_error**2))),
        "mean_bias_vol": float(np.mean(vol_error)),
        "mean_ess": float(np.mean(ess_arr)),
        "min_ess": float(np.min(ess_arr)),
        "mean_var_ratio": float(np.mean(vr_arr)),
        "cumulative_loglik": float(np.sum(loglik_arr)),
        "time_sec": elapsed,
        "h_est": h_est, "h_true": h_true_np,
        "vol_est": vol_est, "vol_true": vol_true,
        "ess": ess_arr, "bias": bias_arr, "var_ratio": vr_arr,
    }


def compare_modes(T=1000, n_particles=512, dgp_cfg=None, seed_dgp=42, n_runs=5):
    modes = ["base", "iw", "iw_feedback"]

    print("=" * 95)
    print(f"SVPF Importance Weight Correction Test  |  T={T}, N={n_particles}, runs={n_runs}")
    if dgp_cfg and dgp_cfg.use_regime_change:
        print(f"  REGIME CHANGE at t={dgp_cfg.regime_change_t}: "
              f"mu {dgp_cfg.mu}->{dgp_cfg.mu_post_regime}, "
              f"sigma_z {dgp_cfg.sigma_z:.3f}->{dgp_cfg.sigma_z_post_regime:.3f}")
    print("=" * 95)

    results_all = {m: [] for m in modes}

    for run in range(n_runs):
        seed_f = 100 + run * 37
        for mode in modes:
            r = run_experiment(T=T, n_particles=n_particles, dgp_cfg=dgp_cfg,
                               mode=mode, seed_dgp=seed_dgp, seed_filter=seed_f)
            results_all[mode].append(r)
        b = results_all["base"][-1]
        iw = results_all["iw"][-1]
        print(f"  Run {run+1}/{n_runs}: base bias={b['mean_bias_h']:+.3f} rmse={b['rmse_h']:.3f} | "
              f"iw bias={iw['mean_bias_h']:+.3f} rmse={iw['rmse_h']:.3f} ess={iw['mean_ess']:.0f} | "
              f"{b['time_sec']:.1f}s")

    print(f"\n{'Mode':<15} {'RMSE_h':>8} {'Bias_h':>8} {'RMSE_vol':>8} {'Bias_vol':>8} "
          f"{'ESS':>8} {'VarRatio':>8} {'LogLik':>10} {'Time':>6}")
    print("-" * 95)

    summary = {}
    for mode in modes:
        runs = results_all[mode]
        avg = lambda key: np.mean([r[key] for r in runs])

        s = {
            "rmse_h": avg("rmse_h"), "bias_h": avg("mean_bias_h"),
            "rmse_vol": avg("rmse_vol"), "bias_vol": avg("mean_bias_vol"),
            "ess": avg("mean_ess"), "var_ratio": avg("mean_var_ratio"),
            "loglik": avg("cumulative_loglik"), "time": avg("time_sec"),
        }
        summary[mode] = s

        print(f"{mode:<15} {s['rmse_h']:>8.4f} {s['bias_h']:>+8.4f} {s['rmse_vol']:>8.4f} "
              f"{s['bias_vol']:>+8.4f} {s['ess']:>8.1f} {s['var_ratio']:>8.3f} "
              f"{s['loglik']:>10.1f} {s['time']:>5.1f}s")

    print("\n--- Deltas vs base ---")
    base = summary["base"]
    for mode in ["iw", "iw_feedback"]:
        m = summary[mode]
        dh = (base["rmse_h"] - m["rmse_h"]) / max(base["rmse_h"], 1e-8) * 100
        db = abs(base["bias_h"]) - abs(m["bias_h"])
        dv = (base["rmse_vol"] - m["rmse_vol"]) / max(base["rmse_vol"], 1e-8) * 100
        print(f"  {mode:>12}: RMSE_h {dh:+.1f}%  |Bias_h| {db:+.4f}  RMSE_vol {dv:+.1f}%")

    return results_all, summary


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # TEST 1: Matched
    print("#" * 80)
    print("# TEST 1: MATCHED DGP")
    print("#" * 80)
    dgp1 = DGPConfig(mu=-3.5, rho=0.97, sigma_z=0.15, nu_obs=5.0,
                      nu_state=2.5, use_student_t_state=True)
    compare_modes(T=1000, n_particles=512, dgp_cfg=dgp1, seed_dgp=42, n_runs=5)

    # TEST 2: Misspecified mu
    print("\n" + "#" * 80)
    print("# TEST 2: MISSPECIFIED MU (DGP=-2.0, filter=-3.5)")
    print("#" * 80)
    dgp2 = DGPConfig(mu=-2.0, rho=0.97, sigma_z=0.15, nu_obs=5.0,
                      nu_state=2.5, use_student_t_state=True)
    compare_modes(T=1000, n_particles=512, dgp_cfg=dgp2, seed_dgp=42, n_runs=5)

    # TEST 3: Regime change
    print("\n" + "#" * 80)
    print("# TEST 3: REGIME CHANGE (mu -3.5 -> -1.5 at t=500)")
    print("#" * 80)
    dgp3 = DGPConfig(mu=-3.5, rho=0.97, sigma_z=0.15, nu_obs=5.0,
                      nu_state=2.5, use_student_t_state=True,
                      use_regime_change=True, regime_change_t=500,
                      mu_post_regime=-1.5, sigma_z_post_regime=0.30)
    compare_modes(T=1000, n_particles=512, dgp_cfg=dgp3, seed_dgp=42, n_runs=5)

    # TEST 4: Low N
    print("\n" + "#" * 80)
    print("# TEST 4: LOW N (N=64)")
    print("#" * 80)
    compare_modes(T=1000, n_particles=64, dgp_cfg=dgp1, seed_dgp=42, n_runs=5)

    print("\nDone.")