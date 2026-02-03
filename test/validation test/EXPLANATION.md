# IV Analysis Pipeline: Complete Explanation

## What This Module Does

This pipeline helps you **make money from volatility mispricing** in crypto options markets.

The core idea is simple:

```
You have: A sophisticated vol filter (SVPF) that estimates "true" volatility
Market has: Implied volatility (IV) priced into options

If your estimate is better than the market's → you can profit from the difference
```

---

## The Big Picture

### The Volatility Trading Game

Options are priced based on **expected future volatility**. The market's estimate is called **Implied Volatility (IV)** — it's the volatility number you'd need to plug into Black-Scholes to get the current market price.

But IV is just a guess. Sometimes the market overestimates vol (fear), sometimes it underestimates (complacency).

If you can estimate volatility more accurately than the market, you can:

| Situation | Your Action | Outcome |
|-----------|-------------|---------|
| IV = 55%, Your RV = 40% | Sell options (short vol) | Collect premium, keep it if vol stays low |
| IV = 30%, Your RV = 45% | Buy options (long vol) | Pay premium, profit when vol spikes |

The difference between IV and realized vol is called the **Variance Risk Premium (VRP)**.

---

## Module Components

### 1. Black-Scholes Engine (`BlackScholes` class)

**What it does:** Converts between option prices and implied volatility.

**The problem:** Options trade at dollar prices ($2,500 for a BTC call), but you need to compare volatilities. Black-Scholes lets you "invert" the price to get IV.

```
Option Price ($2,500) → Black-Scholes Inversion → IV (52%)
```

**Why PyTorch:** The inversion uses Newton-Raphson iteration. PyTorch's autodiff computes the derivative (vega) automatically, and batching lets us process thousands of options at once.

```python
# Batched IV computation for 1000 options
iv = BlackScholes.implied_volatility(prices, spots, strikes, expiries, rates, is_calls)
# Returns tensor of 1000 IVs in one call
```

---

### 2. SVI Surface Model (`SVIModel`, `SVICalibrator`, `IVSurface`)

**What it does:** Fits a smooth, arbitrage-free curve through messy market IV data.

**The problem:** Raw IV data is noisy:

```
Strike    Market IV
48000     51.2%
49000     49.8%
50000     48.1%  ← ATM
51000     52.3%  ← Suspicious jump
52000     50.1%
```

You can't use this directly — some points are stale quotes, bid-ask bounces, or errors.

**The solution:** SVI (Stochastic Volatility Inspired) parametrization fits a smooth curve:

```
Total Variance w(k) = a + b × (ρ(k-m) + √((k-m)² + σ²))

where k = log(Strike / Spot) = "moneyness"
```

Five parameters capture the entire smile shape:

| Parameter | Controls | Typical Value |
|-----------|----------|---------------|
| a | Base variance level | 0.04 |
| b | Overall slope | 0.1 |
| ρ (rho) | Skew direction | -0.3 (negative = downside fear) |
| m | Center of smile | 0.0 |
| σ (sigma) | Curvature | 0.1 |

**Why fit a surface?**

1. **Noise reduction** — Smooth curve filters out quote noise
2. **Arbitrage-free** — Raw data can have calendar/butterfly arbitrage
3. **Interpolation** — Get IV for any strike/expiry, not just traded ones
4. **Compact representation** — 5 numbers instead of hundreds of quotes

```python
# Fit SVI to one expiry slice
surface = IVSurface()
model = surface.fit_slice(strikes, spot, T=30/365, market_ivs)

# Now get IV for any strike
iv_at_55000 = surface.get_iv(strike=55000, spot=50000, T=30/365)
```

---

### 3. Data Fetcher (`DeribitFetcher`)

**What it does:** Pulls option data from Deribit and stores it in SQLite.

**Why Deribit:** Largest crypto options exchange, decent API, good liquidity on BTC/ETH.

**What we fetch:**

```
For each option:
  - Instrument name (BTC-28MAR25-50000-C)
  - Strike, expiry, call/put
  - Mark price, bid, ask
  - Mark IV, bid IV, ask IV
  - Open interest, volume
  - Current spot price
```

**Storage:** SQLite database with timestamps. This builds your historical IV dataset for backtesting.

```python
# Fetch current snapshot
fetcher = DeribitFetcher("iv_data.db")
snapshots = await fetcher.run_snapshot("BTC")

# Later: load historical data
data = fetcher.load_snapshots(currency="BTC", start_time="2024-01-01")
```

---

### 4. VRP Analyzer (`VRPAnalyzer`)

**What it does:** Compares your volatility estimates to market IV and generates trading signals.

**Core computation:**

```python
spread = market_iv - your_rv_estimate

# Normalize to z-score
zscore = (spread - historical_mean) / historical_std

# Generate signal
if zscore > 1.5:
    signal = "SELL_VOL"  # Market overpricing, sell options
elif zscore < -1.5:
    signal = "BUY_VOL"   # Market underpricing, buy options
else:
    signal = "NO_TRADE"  # Within normal range
```

**Edge evaluation:** The key question is whether your SVPF actually beats the market:

```python
# Ground truth: what vol actually was (hindsight)
realized_vol = compute_realized_vol(future_returns)

# Who was closer?
your_error = abs(your_rv - realized_vol)
market_error = abs(market_iv - realized_vol)

# If your_error < market_error consistently, you have edge
```

---

## The Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA COLLECTION                                        │
│                                                                 │
│  Deribit API → Option quotes → SQLite database                  │
│                                                                 │
│  Run continuously: python collector.py --interval 300           │
│  Build 1-4 weeks of data before analysis                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: IV SURFACE CONSTRUCTION                                │
│                                                                 │
│  Raw IV points → BS inversion → SVI fitting → Clean surface     │
│                                                                 │
│  Extract: ATM IV, skew, term structure                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: YOUR SVPF ESTIMATES                                    │
│                                                                 │
│  Price data → Your SVPF filter → RV estimates                   │
│                                                                 │
│  This is YOUR edge — the sophisticated vol tracking you built   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: VRP ANALYSIS                                           │
│                                                                 │
│  Compare: Market IV vs Your RV vs Realized Vol (hindsight)      │
│                                                                 │
│  Questions:                                                     │
│    - Is your RV closer to realized than IV? (edge exists)       │
│    - Does spread predict VRP? (tradeable signal)                │
│    - What's the win rate / P&L? (strategy viability)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: TRADING                                                │
│                                                                 │
│  If edge confirmed:                                             │
│    spread > threshold → Sell straddle/strangle                  │
│    spread < -threshold → Buy straddle/strangle                  │
│                                                                 │
│  Delta hedge to isolate pure vol exposure                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Explained

### Implied Volatility (IV)

The market's **forward-looking** estimate of volatility, embedded in option prices.

```
Option price = f(spot, strike, time, rate, volatility)

Given price, solve for volatility → IV
```

IV is what traders are *willing to pay* for vol exposure. It includes:
- Actual expected vol
- Risk premium (fear/greed)
- Supply/demand imbalances

### Realized Volatility (RV)

**Backward-looking** measure of what volatility actually was.

```
RV = std(log_returns) × √(annualization_factor)

For daily data: RV = std(daily_returns) × √365
```

Your SVPF estimates something closer to "current true vol" — a filtered, real-time estimate that's forward-looking but based on the generative process, not market sentiment.

### Variance Risk Premium (VRP)

The gap between IV and RV:

```
VRP = IV - RV
```

**Why does VRP exist?**

1. **Insurance premium** — Investors pay extra for downside protection
2. **Risk aversion** — Vol spikes hurt more than calm helps
3. **Leverage constraints** — Selling vol requires margin

Historically, VRP is **positive on average** — selling vol is profitable long-term. But it's volatile, and the losses when wrong are severe.

### Why Your SVPF Matters

Standard RV is backward-looking. Market IV contains noise and sentiment.

Your SVPF:
- Estimates the **latent** volatility state
- Uses proper **state-space** inference
- Adapts quickly to **regime changes**
- Provides **uncertainty quantification**

If your filter is genuinely better at tracking true vol, you see mispricings others don't.

---

## Integration with Your SVPF

The pipeline expects your SVPF output as a numpy array aligned with timestamps:

```python
# Option 1: Save SVPF output to file
{
    "timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00", ...],
    "vol_estimates": [0.42, 0.43, 0.45, ...]
}

# Option 2: Direct integration
from your_svpf_module import SVPF

svpf = SVPF(params)
for price in price_series:
    svpf.update(price)
    vol = svpf.get_vol_estimate()
```

The `research.py` script handles alignment with IV timestamps.

---

## What Success Looks Like

After running the analysis, you want to see:

```
Edge Metrics:
  Your wins vs market: 58%      ← > 50% means you're better
  Your MAE: 3.2%                ← Lower is better
  Market MAE: 4.8%              ← You beat this
  Spread predicts VRP: 0.45    ← Positive correlation = tradeable

Strategy P&L:
  Win rate: 62%                 ← > 50% with proper sizing = profit
  Mean per trade: 1.8%          ← Average profit per signal
```

If these metrics are weak, either:
1. Your SVPF needs improvement
2. The market is efficient (no edge exists)
3. Need more data

---

## Risk Warnings

**Selling vol is dangerous:**
- Profits are capped (you keep premium)
- Losses are unlimited (vol can spike 5x)
- One bad event wipes out months of gains

**Mitigations:**
- Position sizing based on confidence
- Stop losses / vol triggers
- Diversification across expiries
- Never sell naked — use spreads

**The pipeline tells you IF you have edge, not HOW MUCH to bet.** Position sizing is a separate problem.

---

## File Summary

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `iv_analysis.py` | Core library | `BlackScholes`, `SVIModel`, `IVSurface`, `DeribitFetcher`, `VRPAnalyzer` |
| `collector.py` | Data collection | Continuous fetching, SQLite storage |
| `research.py` | Backtesting | `VRPBacktester`, edge evaluation |
| `README.md` | Quick reference | Setup, usage examples |
| `EXPLANATION.md` | This document | Conceptual explanation |

---

## Next Steps for You

1. **Collect data** — Run collector for 1-2 weeks minimum
2. **Export SVPF** — Save your vol estimates with timestamps
3. **Run analysis** — Check edge metrics
4. **Validate** — Is spread predictive of VRP?
5. **Paper trade** — Generate signals, track hypothetical P&L
6. **Go live (carefully)** — Small size, strict risk limits

The pipeline is the infrastructure. Your SVPF is the edge. This connects them.
