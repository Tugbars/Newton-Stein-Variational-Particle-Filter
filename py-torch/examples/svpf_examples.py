"""
SVPF Python Usage Examples

Demonstrates:
1. Basic filtering
2. Batch processing
3. Configuration options
4. Real-time usage pattern
"""

import numpy as np
import pysvpf as svpf

# =============================================================================
# Example 1: Basic Usage
# =============================================================================

def basic_example():
    """Minimal working example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create filter with defaults
    filter = svpf.SVPF(n_particles=400, n_stein=8, nu=30.0)
    
    # Initialize with model parameters
    filter.initialize(
        rho=0.97,       # Persistence
        sigma_z=0.15,   # Vol-of-vol
        mu=-3.5,        # Mean log-vol (~3% daily vol)
        gamma=-0.5,     # Leverage effect
        seed=42
    )
    
    # Simulate some returns (in practice, these come from market data)
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02  # ~2% daily vol
    
    # Filter step by step
    for t, y in enumerate(returns[:10]):
        result = filter.step(y)
        print(f"t={t:3d}: vol={result.vol_mean:.4f}, h={result.h_mean:.4f}, ll={result.log_lik:.4f}")
    
    print(f"\nFilter: {filter}")
    print(f"ESS: {filter.get_ess():.1f} / {filter.n_particles}")


# =============================================================================
# Example 2: Batch Processing
# =============================================================================

def batch_example():
    """Process entire sequence at once"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    filter = svpf.SVPF(n_particles=400, n_stein=8)
    filter.initialize(rho=0.97, sigma_z=0.15, mu=-3.5)
    
    # Generate synthetic data with volatility regime change
    np.random.seed(123)
    T = 500
    returns = np.zeros(T)
    
    # Calm regime (t=0-200)
    returns[:200] = np.random.randn(200) * 0.01
    
    # Crisis regime (t=200-350)
    returns[200:350] = np.random.randn(150) * 0.05
    
    # Recovery (t=350-500)
    returns[350:] = np.random.randn(150) * 0.015
    
    # Run filter on entire sequence
    vol, h_mean, loglik = filter.run(returns.astype(np.float32))
    
    print(f"Processed {T} observations")
    print(f"Vol range: [{vol.min():.4f}, {vol.max():.4f}]")
    print(f"Total log-likelihood: {loglik.sum():.2f}")
    
    # Detect regime changes from vol
    calm_vol = vol[:200].mean()
    crisis_vol = vol[200:350].mean()
    recovery_vol = vol[350:].mean()
    
    print(f"\nVolatility by regime:")
    print(f"  Calm (0-200):     {calm_vol:.4f}")
    print(f"  Crisis (200-350): {crisis_vol:.4f}")
    print(f"  Recovery (350+):  {recovery_vol:.4f}")


# =============================================================================
# Example 3: Configuration
# =============================================================================

def config_example():
    """Demonstrate configuration options"""
    print("\n" + "=" * 60)
    print("Example 3: Configuration")
    print("=" * 60)
    
    filter = svpf.SVPF(n_particles=512, n_stein=10, nu=30.0)
    
    # Configure adaptive methods
    filter.set_adaptive_mu(
        enabled=True,
        process_var=0.001,      # Q: how fast mu can drift
        obs_var_scale=11.5,     # R = scale * bandwidth²
        mu_min=-4.0,
        mu_max=-1.0
    )
    
    filter.set_adaptive_sigma(
        enabled=True,
        threshold=1.0,          # Z-score to start boosting
        max_boost=3.0           # Max 3x sigma_z during stress
    )
    
    filter.set_adaptive_guide(
        enabled=True,
        base=0.05,              # Normal guide strength
        max=0.30,               # During surprises
        threshold=1.0           # Innovation threshold
    )
    
    # Configure core components
    filter.set_asymmetric_rho(
        enabled=True,
        rho_up=0.99,            # Vol rises fast
        rho_down=0.92           # Vol decays slow
    )
    
    filter.set_mim(
        enabled=True,
        jump_prob=0.05,         # 5% scouts
        jump_scale=5.0          # 5x wider proposals
    )
    
    filter.set_svld(enabled=True, temperature=0.3)
    filter.set_annealing(enabled=True, n_steps=3)
    filter.set_newton(enabled=True)
    
    # Properties
    filter.nu = 30.0
    filter.lik_offset = 0.70
    
    filter.initialize(rho=0.97, sigma_z=0.15, mu=-3.5)
    
    print("Configuration set:")
    print(f"  - Adaptive mu: ON")
    print(f"  - Adaptive sigma: ON (max 3x boost)")
    print(f"  - Adaptive guide: ON (0.05 → 0.30)")
    print(f"  - Asymmetric rho: ON (up=0.99, down=0.92)")
    print(f"  - MIM: ON (5% scouts)")
    print(f"  - SVLD: ON (T=0.3)")
    print(f"  - Annealing: ON (3 steps)")
    print(f"  - Newton-Stein: ON")
    print(f"  - nu: {filter.nu}")
    print(f"  - lik_offset: {filter.lik_offset}")


# =============================================================================
# Example 4: Real-Time Pattern
# =============================================================================

def realtime_example():
    """Pattern for real-time/HFT usage"""
    print("\n" + "=" * 60)
    print("Example 4: Real-Time Pattern")
    print("=" * 60)
    
    # Pre-create filter (do this at startup)
    filter = svpf.SVPF(n_particles=400, n_stein=8, nu=30.0)
    filter.initialize(rho=0.97, sigma_z=0.15, mu=-3.5)
    
    # First call captures CUDA graph
    dummy = filter.step(0.0)
    print(f"Graph captured: {filter.is_graph_captured()}")
    
    # Simulate real-time loop
    np.random.seed(999)
    
    print("\nSimulating real-time updates:")
    for tick in range(5):
        # In production: y = get_latest_return()
        y = np.random.randn() * 0.02
        
        result = filter.step(y)
        
        # Use vol estimate for position sizing, risk management, etc.
        vol_annualized = result.vol_mean * np.sqrt(252)
        
        print(f"  Tick {tick}: return={y:+.4f}, vol={result.vol_mean:.4f}, "
              f"vol_ann={vol_annualized:.1%}")
    
    # If parameters change (e.g., from Oracle), invalidate graph
    filter.set_params(rho=0.98, sigma_z=0.12, mu=-3.2, gamma=-0.4)
    print(f"\nAfter param update, graph captured: {filter.is_graph_captured()}")
    
    # Next step will recapture
    result = filter.step(0.01)
    print(f"After step, graph captured: {filter.is_graph_captured()}")


# =============================================================================
# Example 5: Diagnostics
# =============================================================================

def diagnostics_example():
    """Access internal state for debugging"""
    print("\n" + "=" * 60)
    print("Example 5: Diagnostics")
    print("=" * 60)
    
    filter = svpf.SVPF(n_particles=400)
    filter.initialize(rho=0.97, sigma_z=0.15, mu=-3.5)
    
    # Run a few steps
    for _ in range(10):
        filter.step(np.random.randn() * 0.02)
    
    # Get particle positions
    particles = filter.get_particles()
    print(f"Particle statistics:")
    print(f"  Mean h: {particles.mean():.4f}")
    print(f"  Std h:  {particles.std():.4f}")
    print(f"  Min h:  {particles.min():.4f}")
    print(f"  Max h:  {particles.max():.4f}")
    
    # Effective sample size
    ess = filter.get_ess()
    print(f"\nESS: {ess:.1f} / {filter.n_particles} ({100*ess/filter.n_particles:.1f}%)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    basic_example()
    batch_example()
    config_example()
    realtime_example()
    diagnostics_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)