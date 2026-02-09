"""
SVPF v7 - Noise Strategy Comparison (No Repulsion)
====================================================

All modes use β=0 (no repulsion) since v6 proved repulsion is harmful.
We compare what fills the diversity role instead:

  1. gaussian:       Standard Langevin noise (isotropic Gaussian, fixed T)
  2. student_t:      Student-t noise (ν_langevin) — heavy tails keep outliers alive
  3. obs_scaled:     Gaussian but temperature scales with observation surprise
  4. student_obs:    Student-t + observation-scaled — heavy tails AND adaptive magnitude
  5. base_repulsion: Standard SVGD with repulsion (β=1, Gaussian) for reference
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time
import math


@dataclass
class SVPFConfig:
    n_particles: int = 512
    mu: float = -3.5
    rho: float = 0.97
    sigma_z: float = 0.15
    gamma: float = 0.0
    nu_obs: float = 5.0

    n_stein_steps: int = 8
    stein_step_size: float = 0.05
    temperature: float = 0.3
    rmsprop_rho: float = 0.7
    rmsprop_eps: float = 1e-6

    use_annealing: bool = True
    n_anneal_stages: int = 5
    anneal_steps_per_stage: int = 4

    bw_ema_alpha: float = 0.3

    use_guide: bool = True
    guide_strength: float = 0.05

    use_guided_predict: bool = True
    guided_alpha_base: float = 0.0
    guided_alpha_shock: float = 0.4
    guided_innovation_threshold: float = 1.5

    use_student_t_state: bool = True
    nu_state: float = 2.5

    use_antithetic: bool = True

    # --- Noise / repulsion config (set by harness) ---
    noise_mode: str = "gaussian"  # "gaussian", "student_t", "obs_scaled", "student_obs"
    use_repulsion: bool = False   # True only for base_repulsion reference

    # Noise params
    nu_langevin: float = 3.0          # df for Student-t Langevin noise
    obs_scale_base: float = 0.3       # base temperature for obs-scaled
    obs_scale_sensitivity: float = 0.5 # how much surprise amplifies temperature
    obs_scale_ema_alpha: float = 0.1   # EMA for surprise tracking
    obs_scale_max: float = 2.0         # cap temperature multiplier

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

    CLAMP_LO = -10.0
    CLAMP_HI = 5.0

    def __init__(self, cfg: SVPFConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.n = cfg.n_particles

        nu = cfg.nu_obs
        self.student_t_const = (
            torch.lgamma(torch.tensor((nu + 1.0) / 2.0)).item()
            - torch.lgamma(torch.tensor(nu / 2.0)).item()
            - 0.5 * math.log(math.pi * nu)
        )

        from scipy.special import digamma as _digamma
        psi_half = _digamma(0.5)
        psi_nu_half = _digamma(nu / 2.0)
        self.implied_offset = -(math.log(nu) + psi_half - psi_nu_half)

        # Pre-create Student-t distribution for Langevin if needed
        if cfg.noise_mode in ("student_t", "student_obs"):
            self._student_t_langevin = torch.distributions.StudentT(cfg.nu_langevin)

    def initialize(self, seed: int = 123):
        torch.manual_seed(seed)
        cfg = self.cfg
        base_var = cfg.sigma_z**2 / (1.0 - cfg.rho**2 + 1e-6)
        if cfg.use_student_t_state and cfg.nu_state > 2:
            base_var *= cfg.nu_state / (cfg.nu_state - 2.0)
        std = math.sqrt(base_var)

        self.h = torch.randn(self.n, device=self.dev) * std + cfg.mu
        self.h = torch.clamp(self.h, self.CLAMP_LO, self.CLAMP_HI)
        self.h_prev = self.h.clone()
        self.v_rmsprop = torch.zeros(self.n, device=self.dev)

        self.bandwidth = 0.5
        self.bandwidth_sq = 0.25
        self.h_mean = cfg.mu
        self.vol_prev = math.exp(cfg.mu / 2.0)
        self.timestep = 0

        self.guide_mean = cfg.mu
        self.guide_var = base_var
        self.guide_bias_offset = 0.0

        # Observation surprise tracking
        self.surprise_ema = 1.0  # |y| / vol_prev, EMA
        self.current_temperature = cfg.temperature

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
        self.guide_mean = mu_pred + K * (h_implied - mu_pred)
        self.guide_var = max((1.0 - K * H) * P_pred, 1e-6)
        self.guide_mean = max(min(self.guide_mean, self.CLAMP_HI), self.CLAMP_LO)

    @torch.no_grad()
    def _predict(self, y_t: float):
        cfg = self.cfg
        n = self.n
        self.h_prev = self.h.clone()
        mu_prior = cfg.mu + cfg.rho * (self.h - cfg.mu)

        if cfg.use_guided_predict:
            y_sq = y_t * y_t
            log_y2 = math.log(y_sq + 1e-10)
            mean_implied = max(log_y2 + self.implied_offset, -5.0)
            innovation = mean_implied - mu_prior
            z_score = innovation / 2.5
            activation = torch.where(
                z_score > cfg.guided_innovation_threshold,
                torch.tanh(z_score - cfg.guided_innovation_threshold),
                torch.zeros_like(z_score),
            )
            alpha = cfg.guided_alpha_base + (cfg.guided_alpha_shock - cfg.guided_alpha_base) * activation
            mean_proposal = (1.0 - alpha) * mu_prior + alpha * mean_implied
        else:
            mean_proposal = mu_prior

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

    @torch.no_grad()
    def _compute_bandwidth(self):
        var = self.h.var().item()
        bw_sq_new = max(2.0 * var / math.log(self.n + 1.0), 1e-6)
        alpha = self.cfg.bw_ema_alpha
        if self.bandwidth_sq > 0:
            self.bandwidth_sq = alpha * bw_sq_new + (1.0 - alpha) * self.bandwidth_sq
        else:
            self.bandwidth_sq = bw_sq_new
        self.bandwidth = max(min(math.sqrt(self.bandwidth_sq), 2.0), 0.05)
        self.h_mean = self.h.mean().item()

    @torch.no_grad()
    def _update_observation_surprise(self, y_t: float):
        """Track how surprising the current observation is relative to predicted vol."""
        vol_expected = max(self.vol_prev, 1e-6)
        raw_surprise = abs(y_t) / vol_expected

        # Normalize: for Student-t with nu=5, E[|eps|] ≈ 0.927
        # So raw_surprise ≈ 1.0 when observation matches expectation
        alpha = self.cfg.obs_scale_ema_alpha
        self.surprise_ema = (1.0 - alpha) * self.surprise_ema + alpha * raw_surprise

    @torch.no_grad()
    def _get_effective_temperature(self) -> float:
        """Compute temperature based on noise mode and observation surprise."""
        cfg = self.cfg
        mode = cfg.noise_mode

        if mode in ("obs_scaled", "student_obs"):
            # Scale temperature by surprise
            # surprise_ema ≈ 1.0 when calm, > 2.0 when shocked
            # multiplier: 1.0 at calm, ramps up with surprise
            excess = max(self.surprise_ema - 1.0, 0.0)
            multiplier = 1.0 + cfg.obs_scale_sensitivity * excess
            multiplier = min(multiplier, cfg.obs_scale_max)
            return cfg.obs_scale_base * multiplier
        else:
            return cfg.temperature

    @torch.no_grad()
    def _sample_noise(self, n: int) -> torch.Tensor:
        """Sample noise according to configured mode."""
        cfg = self.cfg
        mode = cfg.noise_mode

        if mode in ("student_t", "student_obs"):
            # Student-t: heavier tails than Gaussian
            # Scale so variance matches Gaussian Langevin: Var(t_nu) = nu/(nu-2)
            raw = self._student_t_langevin.sample((n,)).to(self.dev)
            # Normalize variance to 1 so temperature controls magnitude
            if cfg.nu_langevin > 2:
                scale = math.sqrt((cfg.nu_langevin - 2.0) / cfg.nu_langevin)
                raw = raw * scale
            return raw
        else:
            return torch.randn(n, device=self.dev)

    @torch.no_grad()
    def _log_prior_score(self) -> torch.Tensor:
        cfg = self.cfg
        mu_i = cfg.mu + cfg.rho * (self.h_prev - cfg.mu)
        diff = self.h.unsqueeze(1) - mu_i.unsqueeze(0)
        sigma_sq = cfg.sigma_z ** 2

        if cfg.use_student_t_state:
            nu_s = cfg.nu_state
            nu_sigma_sq = nu_s * sigma_sq
            half_nup1 = 0.5 * (nu_s + 1.0)
            nup1 = nu_s + 1.0
            diff_sq = diff * diff
            denom = nu_sigma_sq + diff_sq
            log_r = -half_nup1 * torch.log1p(diff_sq / nu_sigma_sq)
            log_r_max = log_r.max(dim=1, keepdim=True).values
            r = torch.exp(log_r - log_r_max)
            r_sum = r.sum(dim=1) + 1e-8
            score = (r * (-nup1 * diff / denom)).sum(dim=1) / r_sum
        else:
            inv_2s2 = 0.5 / sigma_sq
            inv_s2 = 1.0 / sigma_sq
            log_r = -diff * diff * inv_2s2
            log_r_max = log_r.max(dim=1, keepdim=True).values
            r = torch.exp(log_r - log_r_max)
            r_sum = r.sum(dim=1) + 1e-8
            score = (r * (-diff * inv_s2)).sum(dim=1) / r_sum

        return score

    @torch.no_grad()
    def _log_likelihood_score(self, y_t: float):
        h = self.h
        nu = self.cfg.nu_obs
        h_safe = torch.clamp(h, -10.0, 10.0)
        vol = torch.exp(h_safe)
        y_sq = y_t * y_t
        A = y_sq / (vol * nu + 1e-8)
        one_plus_A = 1.0 + A
        log_w = (self.student_t_const - 0.5 * h
                 - 0.5 * (nu + 1.0) * torch.log1p(A.clamp(min=0.0)))
        grad_lik = -0.5 + 0.5 * (nu + 1.0) * A / one_plus_A
        return grad_lik, log_w

    @torch.no_grad()
    def _stein_step(self, grad: torch.Tensor, step_size: float, beta_factor: float):
        cfg = self.cfg
        h = self.h
        n = self.n
        inv_bw_sq = 1.0 / (self.bandwidth * self.bandwidth)

        diff = h.unsqueeze(0) - h.unsqueeze(1)
        dist_sq = diff * diff * inv_bw_sq
        K = 1.0 / (1.0 + dist_sq)
        K_sq = K * K

        # Attractive term (always on)
        k_grad_sum = (K * grad.unsqueeze(0)).sum(dim=1)

        # Repulsive term (only if enabled)
        if cfg.use_repulsion:
            gk_sum = (2.0 * diff * inv_bw_sq * K_sq).sum(dim=1)
            phi = (k_grad_sum + gk_sum) / float(n)
        else:
            phi = k_grad_sum / float(n)

        # RMSProp
        self.v_rmsprop = cfg.rmsprop_rho * self.v_rmsprop + (1.0 - cfg.rmsprop_rho) * phi * phi
        precond = torch.rsqrt(self.v_rmsprop + cfg.rmsprop_eps)

        effective_step = step_size * beta_factor
        drift = effective_step * phi * precond

        # Noise injection (mode-dependent)
        temp = self._get_effective_temperature()
        diffusion = 0.0
        if temp > 1e-6:
            noise = self._sample_noise(n)
            diffusion = math.sqrt(2.0 * effective_step * temp) * noise

        self.current_temperature = temp
        self.h = torch.clamp(h + drift + diffusion, self.CLAMP_LO, self.CLAMP_HI)

    @torch.no_grad()
    def step(self, y_t: float) -> dict:
        cfg = self.cfg

        self._ekf_guide_update(y_t)
        self._update_observation_surprise(y_t)
        self._predict(y_t)
        self._apply_guide()
        self._compute_bandwidth()

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
                gp = self._log_prior_score()
                gl, _ = self._log_likelihood_score(y_t)
                grad = torch.clamp(gp + beta * gl, -10.0, 10.0)
                self._stein_step(grad, step_size, beta_factor)

        _, log_w = self._log_likelihood_score(y_t)

        h_mean = self.h.mean().item()
        vol = torch.exp(self.h * 0.5).mean().item()
        h_var = self.h.var().item()

        lw_max = log_w.max().item()
        loglik = lw_max + math.log(max(torch.exp(log_w - lw_max).mean().item(), 1e-30))

        self.vol_prev = vol
        self.timestep += 1

        return {
            "h_mean": h_mean,
            "vol": vol,
            "loglik": loglik,
            "h_var": h_var,
            "temperature": self.current_temperature,
            "surprise": self.surprise_ema,
        }


# =============================================================================
# Test harness
# =============================================================================

def run_experiment(T=1000, n_particles=512, dgp_cfg=None, filter_cfg=None,
                   mode="gaussian", seed_dgp=42, seed_filter=123) -> dict:

    if dgp_cfg is None: dgp_cfg = DGPConfig()
    if filter_cfg is None: filter_cfg = SVPFConfig(n_particles=n_particles)

    if mode == "base_repulsion":
        filter_cfg.use_repulsion = True
        filter_cfg.noise_mode = "gaussian"
    else:
        filter_cfg.use_repulsion = False
        filter_cfg.noise_mode = mode

    y_data, h_true = generate_sv_data(T, dgp_cfg, seed=seed_dgp)
    svpf = SVPFTorch(filter_cfg)
    svpf.initialize(seed=seed_filter)

    h_est = np.zeros(T)
    vol_est = np.zeros(T)
    loglik_arr = np.zeros(T)
    var_arr = np.zeros(T)
    temp_arr = np.zeros(T)
    surp_arr = np.zeros(T)

    t0 = time.time()
    for t in range(T):
        r = svpf.step(y_data[t].item())
        h_est[t] = r["h_mean"]
        vol_est[t] = r["vol"]
        loglik_arr[t] = r["loglik"]
        var_arr[t] = r["h_var"]
        temp_arr[t] = r["temperature"]
        surp_arr[t] = r["surprise"]
    elapsed = time.time() - t0

    h_true_np = h_true.numpy()
    vol_true = np.exp(h_true_np / 2.0)
    h_error = h_est - h_true_np
    vol_error = vol_est - vol_true

    return {
        "mode": mode, "T": T, "n_particles": n_particles,
        "rmse_h": float(np.sqrt(np.mean(h_error**2))),
        "mean_bias_h": float(np.mean(h_error)),
        "rmse_vol": float(np.sqrt(np.mean(vol_error**2))),
        "mean_bias_vol": float(np.mean(vol_error)),
        "mean_h_var": float(np.mean(var_arr)),
        "mean_temp": float(np.mean(temp_arr)),
        "mean_surprise": float(np.mean(surp_arr)),
        "cumulative_loglik": float(np.sum(loglik_arr)),
        "time_sec": elapsed,
    }


def compare_modes(T=1000, n_particles=512, dgp_cfg=None, seed_dgp=42, n_runs=5):
    modes = ["base_repulsion", "gaussian", "student_t", "obs_scaled", "student_obs"]
    labels = {
        "base_repulsion": "SVGD+rep",
        "gaussian": "gauss",
        "student_t": "stud-t",
        "obs_scaled": "obs-scl",
        "student_obs": "stud+obs",
    }

    print("=" * 115)
    print(f"SVPF Noise Strategy Comparison  |  T={T}, N={n_particles}, runs={n_runs}")
    if dgp_cfg and dgp_cfg.use_regime_change:
        print(f"  REGIME CHANGE at t={dgp_cfg.regime_change_t}: "
              f"mu {dgp_cfg.mu}->{dgp_cfg.mu_post_regime}")
    print("  All no-repulsion modes use β=0. base_repulsion uses β=1 for reference.")
    print("=" * 115)

    results_all = {m: [] for m in modes}

    for run in range(n_runs):
        seed_f = 100 + run * 37
        for mode in modes:
            r = run_experiment(T=T, n_particles=n_particles, dgp_cfg=dgp_cfg,
                               mode=mode, seed_dgp=seed_dgp, seed_filter=seed_f)
            results_all[mode].append(r)
        row = " | ".join(
            f"{labels[m]} bias={results_all[m][-1]['mean_bias_h']:+.3f}"
            for m in modes
        )
        print(f"  Run {run+1}/{n_runs}: {row}")

    print(f"\n{'Mode':<16} {'RMSE_h':>8} {'Bias_h':>8} {'RMSE_vol':>8} {'h_var':>8} "
          f"{'avg_T':>6} {'surp':>6} {'LogLik':>10} {'Time':>6}")
    print("-" * 110)

    summary = {}
    for mode in modes:
        runs = results_all[mode]
        avg = lambda key: np.mean([r[key] for r in runs])
        s = {
            "rmse_h": avg("rmse_h"), "bias_h": avg("mean_bias_h"),
            "rmse_vol": avg("rmse_vol"), "bias_vol": avg("mean_bias_vol"),
            "h_var": avg("mean_h_var"), "temp": avg("mean_temp"),
            "surprise": avg("mean_surprise"),
            "loglik": avg("cumulative_loglik"), "time": avg("time_sec"),
        }
        summary[mode] = s
        print(f"{mode:<16} {s['rmse_h']:>8.4f} {s['bias_h']:>+8.4f} {s['rmse_vol']:>8.4f} "
              f"{s['h_var']:>8.4f} {s['temp']:>6.3f} {s['surprise']:>6.2f} "
              f"{s['loglik']:>10.1f} {s['time']:>5.1f}s")

    print("\n--- Deltas vs gaussian (no repulsion baseline) ---")
    base = summary["gaussian"]
    for mode in modes:
        if mode == "gaussian":
            continue
        m = summary[mode]
        dh = (base["rmse_h"] - m["rmse_h"]) / max(base["rmse_h"], 1e-8) * 100
        db = abs(base["bias_h"]) - abs(m["bias_h"])
        dv = (base["rmse_vol"] - m["rmse_vol"]) / max(base["rmse_vol"], 1e-8) * 100
        tag_b = "better" if db > 0 else "worse"
        print(f"  {mode:>15}: RMSE_h {dh:+.1f}%  |Bias_h| {db:+.4f} ({tag_b})  RMSE_vol {dv:+.1f}%")

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
