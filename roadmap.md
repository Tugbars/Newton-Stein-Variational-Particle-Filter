# SVPF System Roadmap

## Overview

This document outlines the planned extensions to the SVPF (Stein Variational Particle Filter) system for crypto volatility tracking and parameter estimation.

---

## Current State

### What We Have
- **SVPF core**: High-performance GPU particle filter for stochastic volatility
- **Performance**: 148 μs/step @ N=512, 83% compute efficiency
- **Features**: Breathing mechanism, adaptive μ, Student-t likelihood, Stein transport

### What It Does
```
Input:  y_t (log-returns)
Output: h_t (log-volatility estimate), vol_t = exp(h_t/2)
```

Single instrument, single signal, real-time inference.

---

## Phase 1: Multi-Signal Architecture

### Goal
Incorporate additional market microstructure signals to improve vol tracking and anticipate regime changes.

### Signals

| Signal | Source | Update Rate | Purpose |
|--------|--------|-------------|---------|
| **Returns** | Trade stream | Per-trade | Primary SVPF input |
| **Spread** | Order book | 100ms | Liquidity / early warning |
| **Volume** | Trade stream | Per-trade | Signal confidence weighting |
| **Funding rate** | Exchange API | 8h | Regime / crash risk indicator |

### Architecture

```
                        ┌──────────────────────┐
  return_t ────────────→│                      │
                        │        SVPF          │────→ vol_estimate_t
  spread_t ────────────→│   (with enhanced     │────→ vol_uncertainty_t
                        │    breathing)        │
                        └──────────────────────┘
                                  ↑
                        ┌─────────┴──────────┐
                        │    μ Tracker       │
                        │    (NSVPF)         │←──── funding_rate
                        │                    │←──── 24h realized vol
                        └────────────────────┘
```

### Implementation: Enhanced Breathing

Current breathing only reacts to return spikes:
```c
if (return_z > threshold) {
    sigma_boost = 1.0 + severity * (max_boost - 1.0);
}
```

Enhanced version also reacts to spread widening:
```c
// Spread z-score (vs recent average)
float spread_z = (spread_t - spread_ema) / spread_std;

// Trigger on either signal
bool liquidity_warning = spread_z > spread_threshold;
bool price_shock = return_z > return_threshold;

if (liquidity_warning || price_shock) {
    float severity = fmaxf(spread_z, return_z) / 3.0f;
    sigma_boost = 1.0 + severity * (max_boost - 1.0);
}
```

Spread widening often **precedes** vol spikes by seconds—this gives the filter a head start.

---

## Phase 2: NSVPF for μ Tracking

### Goal
Track the slow-moving mean level of log-volatility (μ) as a separate state.

### Why Separate?
- μ moves on **hours-days** timescale
- h_t moves on **seconds-minutes** timescale
- Mixing them in one filter causes either:
  - Too slow tracking of fast dynamics, or
  - Too noisy estimate of slow dynamics

### Model
```
μ_t = μ_{t-1} + σ_μ * ε_μ,t       (slow random walk)
h_t = μ_t + ρ(h_{t-1} - μ_{t-1}) + σ_z * ε_t
y_t ~ Student-t(0, exp(h_t), ν)
```

### Implementation Options

**Option A: Separate Kalman filter** (current `use_adaptive_mu`)
- Simple, already implemented
- Assumes Gaussian, may miss regime shifts

**Option B: Dedicated NSVPF**
- Full particle filter for μ
- Can handle non-Gaussian regime transitions
- More compute, but we have headroom

**Option C: Hierarchical joint filter**
- Single filter tracking (μ, h) jointly
- Most principled, most complex

Recommend: Start with Option A (already working), upgrade to B if needed.

### Regime Indicators for μ

| Indicator | Interpretation |
|-----------|----------------|
| Funding rate > 0.1% | Leverage high, expect vol spike, raise μ |
| Funding rate < -0.1% | Short squeeze risk, raise μ |
| 24h vol / 7d vol > 1.5 | Regime shift up |
| BTC dominance falling | Alt season, correlation breakdown |

These can inform the μ tracker's process noise or provide "soft" observations.

---

## Phase 3: Offline Calibration with CPMMH

### Goal
Calibrate SVPF parameters (ρ, σ, ν, etc.) from historical data using Bayesian inference.

### Why CPMMH?

**PMMH** (Particle Marginal Metropolis-Hastings):
- Uses particle filter inside MCMC
- Particle filter estimates likelihood: p(y_{1:T} | θ)
- MCMC explores parameter posterior: p(θ | y_{1:T})

**CPMMH** (Correlated PMMH):
- Correlates random numbers between MCMC steps
- Dramatically reduces variance of likelihood ratio
- Faster convergence, better mixing

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CPMMH                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  for each MCMC iteration:                             │  │
│  │    1. Propose θ' ~ q(θ' | θ)                          │  │
│  │    2. Run SVPF on full dataset with θ'                │  │
│  │       → get log p(y_{1:T} | θ')                       │  │
│  │    3. Accept/reject via MH ratio                      │  │
│  │    4. Store θ samples                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                              ↓                               │
│                    Posterior samples                         │
│                    p(ρ, σ, ν, μ | data)                      │
└─────────────────────────────────────────────────────────────┘
```

### SVPF as Likelihood Estimator

SVPF already computes log-likelihood increments:
```c
// In svpf_step_graph:
*h_loglik_out = log p(y_t | y_{1:t-1}, θ)
```

Summing these gives the full marginal likelihood:
```c
float total_loglik = 0.0f;
for (int t = 0; t < T; t++) {
    float loglik_t;
    svpf_step_graph(state, y[t], y_prev, &params, &loglik_t, NULL, NULL);
    total_loglik += loglik_t;
    y_prev = y[t];
}
// total_loglik ≈ log p(y_{1:T} | θ)
```

This is an **unbiased estimate** of the marginal likelihood—exactly what PMMH needs.

### Correlation Strategy for CPMMH

Standard PMMH: independent random numbers each iteration → high variance

CPMMH: correlate random numbers via common random numbers (CRN):
```c
// Store random seed/state for proposal
uint64_t seed_current = get_seed();

// Proposal: use correlated seed
uint64_t seed_proposal = correlate(seed_current, correlation_rho);
```

For SVPF, this means correlating:
- RNG states for particle prediction noise
- Resampling randomness

### Parameters to Calibrate

| Parameter | Prior | Notes |
|-----------|-------|-------|
| ρ | Beta(95, 5) → mean 0.95 | Persistence, usually 0.9-0.99 |
| σ | HalfNormal(0.2) | Vol-of-vol, usually 0.05-0.3 |
| ν | Gamma(10, 1) | Tail thickness, usually 5-20 |
| μ | Normal(-4, 2) | Mean log-vol level |
| γ | Normal(0, 0.5) | Leverage effect |

### Implementation Steps

1. **Batch SVPF runner**: Process full historical dataset, return total log-likelihood
2. **MCMC sampler**: Propose parameters, evaluate likelihood via SVPF
3. **Correlation mechanism**: CRN for variance reduction
4. **Diagnostic tools**: Trace plots, R-hat, ESS

---

## Phase 4: Multi-Instrument

### Goal
Run SVPF on multiple instruments simultaneously, exploiting GPU parallelism.

### Options

**Option A: Batched single kernel**
```
[BTC, ETH, SOL, ...] → Single SVPF kernel with batch dimension
```
- Most efficient
- Requires code restructuring

**Option B: Multiple streams**
```
Stream 1: BTC SVPF
Stream 2: ETH SVPF
Stream 3: SOL SVPF
```
- Simple, current code works
- Some overhead from multiple launches

**Option C: Hybrid**
- Batch correlated instruments (BTC, ETH)
- Separate streams for uncorrelated

At 148 μs/step and 100ms market updates, we can easily run 100+ instruments sequentially. Batching is optimization, not necessity.

---

## Implementation Priority

| Phase | Effort | Value | Priority |
|-------|--------|-------|----------|
| 1. Multi-signal (spread) | Low | Medium | **Now** |
| 2. NSVPF for μ | Medium | High | **Next** |
| 3. CPMMH calibration | High | High | Q2 |
| 4. Multi-instrument | Low | Medium | When needed |

---

## Data Pipeline

### Immediate Need

```python
# Collect and log data for development
import websocket
import json
from datetime import datetime

def on_message(ws, msg):
    data = json.loads(msg)
    timestamp = datetime.utcnow().isoformat()
    price = data['p']
    qty = data['q']
    
    # Log to file for offline analysis
    log.write(f"{timestamp},{price},{qty}\n")

# Binance trade stream
ws = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@trade",
    on_message=on_message
)
```

### Order Book for Spread

```python
# Binance order book stream
ws_book = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@bookTicker",
    on_message=on_book_message
)

def on_book_message(ws, msg):
    data = json.loads(msg)
    best_bid = float(data['b'])
    best_ask = float(data['a'])
    spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
    # Use spread in breathing mechanism
```

---

## Summary

```
Current:    SVPF (returns → vol)
                    ↓
Phase 1:    SVPF + spread signal (anticipate spikes)
                    ↓
Phase 2:    SVPF + NSVPF (fast dynamics + slow μ)
                    ↓
Phase 3:    CPMMH[SVPF] (offline calibration)
                    ↓
Phase 4:    Multi-instrument batching
```

The core SVPF engine is complete and optimized. The roadmap is about:
1. Better inputs (spread, funding)
2. Better μ tracking (NSVPF)
3. Better parameters (CPMMH calibration)
4. Scale (multi-instrument)

Each phase builds on thte last. Start with data collection and spread integration.
