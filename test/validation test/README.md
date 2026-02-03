# IV Analysis Pipeline

PyTorch-based toolkit for:
1. Fetching Deribit option data
2. Black-Scholes IV inversion
3. SVI surface calibration
4. VRP (Variance Risk Premium) analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy aiohttp
```

### 2. Collect Data

```bash
# Single snapshot
python collector.py --once

# Continuous collection (every 5 minutes)
python collector.py --interval 300

# Check stats
python collector.py --stats
```

### 3. Run Example

```bash
python iv_analysis.py
```

### 4. Run VRP Analysis

```bash
python research.py
```

## Architecture

```
iv_analysis.py    - Core library
├── BlackScholes  - Pricing & IV inversion (PyTorch, batched)
├── SVIModel      - SVI parametrization (trainable)
├── SVICalibrator - Fit SVI to market data (Adam optimizer)
├── IVSurface     - Surface management & interpolation
├── DeribitFetcher - Async data fetching + SQLite storage
└── VRPAnalyzer   - VRP computation & signal generation

collector.py      - Data collection scheduler
research.py       - VRP backtesting framework
```

## Integration with SVPF

The pipeline is designed to work with your SVPF volatility estimates.

### Option 1: File-based Integration

Save your SVPF estimates to JSON:

```json
{
  "timestamps": ["2024-01-01T00:00:00", ...],
  "vol_estimates": [0.45, 0.47, ...]
}
```

Then in `research.py`:

```python
svpf_data = json.load(open("svpf_output.json"))
svpf_rv = SVPFIntegration.align_with_iv_data(
    svpf_data['timestamps'],
    np.array(svpf_data['vol_estimates']),
    iv_timestamps
)
results = backtester.run_analysis(svpf_rv=svpf_rv)
```

### Option 2: Direct Integration

Import your SVPF module directly:

```python
from your_svpf import SVPFilter

# Initialize your filter
svpf = SVPFilter(...)

# Process each timestamp
for spot, timestamp in data:
    svpf.update(spot)
    vol_estimate = svpf.get_volatility()
```

## Key Concepts

### SVI Parameters

```
w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))

a     - Base variance level
b     - Overall slope
ρ     - Skew (-1 to 1, typically negative)
m     - Center of smile
σ     - Curvature
```

### VRP Analysis

```
spread = IV_market - RV_svpf

If spread >> normal:
  → Market overpricing vol
  → Sell vol (short straddle/strangle)
  
If spread << normal:
  → Market underpricing vol  
  → Buy vol (long straddle/strangle)
```

### Trading Signal

```python
zscore = (spread - spread_mean) / spread_std

if zscore > 1.5:
    signal = "SELL_VOL"
elif zscore < -1.5:
    signal = "BUY_VOL"
else:
    signal = "NO_TRADE"
```

## Data Schema

SQLite table `option_snapshots`:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | TEXT | ISO format |
| currency | TEXT | BTC, ETH |
| instrument_name | TEXT | e.g., BTC-28MAR25-50000-C |
| expiry | INTEGER | Unix timestamp (ms) |
| strike | REAL | Strike price |
| is_call | INTEGER | 1=call, 0=put |
| spot | REAL | Index price |
| mark_price | REAL | Mark price (BTC) |
| mark_iv | REAL | Implied vol (%) |
| bid_iv | REAL | Bid implied vol |
| ask_iv | REAL | Ask implied vol |
| open_interest | REAL | Open interest |
| volume | REAL | 24h volume |

## Next Steps

1. **Collect data**: Run collector for 1+ weeks
2. **Validate SVPF**: Compare your estimates vs market IV vs realized
3. **Measure edge**: Does your spread predict VRP?
4. **Paper trade**: Generate signals, track hypothetical P&L
5. **Go live**: Start small, scale with confidence

## Notes

- All volatilities are annualized
- IV from Deribit is in percentage (divide by 100)
- SVI is fit in total variance space (IV² × T)
- Use GPU for batch operations: `DEVICE = torch.device('cuda')`
