#!/usr/bin/env python3
"""
fetch_market_data.py - Download historical market data for SVPF testing

Fetches data covering major market crashes:
  - 2008 Financial Crisis (Sep-Nov 2008)
  - 2010 Flash Crash (May 6, 2010)
  - 2011 US Debt Downgrade (Aug 2011)
  - 2015 China Fears (Aug 2015)
  - 2018 Volmageddon (Feb 2018) + Q4 selloff
  - 2020 COVID Crash (Feb-Mar 2020)
  - 2022 Rate Hike Selloff

Output: CSV files with columns [date, price, return, log_return]
        Ready for loading into CUDA tests

Usage:
    pip install yfinance pandas
    python fetch_market_data.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Output directory
OUTPUT_DIR = "market_data"

def fetch_and_process(ticker: str, start: str, end: str, name: str) -> pd.DataFrame:
    """Fetch data and compute returns."""
    print(f"  Fetching {ticker} ({start} to {end})...")
    
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    if data.empty:
        print(f"    WARNING: No data returned for {ticker}")
        return None
    
    # Handle multi-level columns from yfinance
    # yfinance returns columns like ('Close', 'SPY') as MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten to just the price type (Close, Open, etc.)
        data.columns = data.columns.get_level_values(0)
    
    # Use Close price (Adj Close is no longer available in newer yfinance)
    price_col = 'Close' if 'Close' in data.columns else 'Adj Close'
    
    df = pd.DataFrame()
    df['date'] = data.index
    df['price'] = data[price_col].values
    
    # Simple return: r_t = (P_t - P_{t-1}) / P_{t-1}
    df['return'] = df['price'].pct_change()
    
    # Log return: r_t = log(P_t / P_{t-1})
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    
    # Drop first row (NaN from differencing)
    df = df.dropna().reset_index(drop=True)
    
    # Statistics
    n = len(df)
    mean_ret = df['log_return'].mean() * 252  # Annualized
    std_ret = df['log_return'].std() * np.sqrt(252)  # Annualized
    min_ret = df['log_return'].min()
    max_ret = df['log_return'].max()
    
    # Count tail events
    z = df['log_return'] / df['log_return'].std()
    n_3sigma = (np.abs(z) > 3).sum()
    n_4sigma = (np.abs(z) > 4).sum()
    n_5sigma = (np.abs(z) > 5).sum()
    
    print(f"    {n} observations")
    print(f"    Annualized: μ={mean_ret:.1%}, σ={std_ret:.1%}")
    print(f"    Range: [{min_ret:.2%}, {max_ret:.2%}]")
    print(f"    Tail events: {n_3sigma} (>3σ), {n_4sigma} (>4σ), {n_5sigma} (>5σ)")
    
    return df

def save_csv(df: pd.DataFrame, filename: str):
    """Save to CSV in format suitable for C/CUDA loading."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save with high precision for returns
    df.to_csv(filepath, index=False, float_format='%.10f', date_format='%Y-%m-%d')
    print(f"    Saved to {filepath}")
    
    # Also save returns-only version (simpler to load in C)
    returns_filepath = filepath.replace('.csv', '_returns.csv')
    df[['log_return']].to_csv(returns_filepath, index=False, header=False, float_format='%.10f')
    print(f"    Returns-only: {returns_filepath}")

def save_binary(df: pd.DataFrame, filename: str):
    """Save returns as raw binary float32 (fastest to load in C)."""
    filepath = os.path.join(OUTPUT_DIR, filename.replace('.csv', '.bin'))
    returns = df['log_return'].values.astype(np.float32)
    returns.tofile(filepath)
    print(f"    Binary (float32): {filepath} ({len(returns)} values, {returns.nbytes} bytes)")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print(" Fetching Market Data for SVPF Gradient Testing")
    print("=" * 70)
    
    datasets = []
    
    # =========================================================================
    # 1. SPY (S&P 500 ETF) - Full history with all crashes
    # =========================================================================
    print("\n[1] SPY - Full History (2007-2024)")
    df = fetch_and_process("SPY", "2007-01-01", "2024-12-31", "spy_full")
    if df is not None:
        save_csv(df, "spy_full.csv")
        save_binary(df, "spy_full.csv")
        datasets.append(("spy_full", df))
    
    # =========================================================================
    # 2. Specific crash periods (for focused testing)
    # =========================================================================
    
    # 2008 Financial Crisis
    print("\n[2] SPY - 2008 Financial Crisis")
    df = fetch_and_process("SPY", "2008-01-01", "2009-06-30", "spy_2008_crisis")
    if df is not None:
        save_csv(df, "spy_2008_crisis.csv")
        save_binary(df, "spy_2008_crisis.csv")
        datasets.append(("spy_2008", df))
    
    # 2020 COVID Crash
    print("\n[3] SPY - 2020 COVID Crash")
    df = fetch_and_process("SPY", "2020-01-01", "2020-06-30", "spy_2020_covid")
    if df is not None:
        save_csv(df, "spy_2020_covid.csv")
        save_binary(df, "spy_2020_covid.csv")
        datasets.append(("spy_2020", df))
    
    # 2022 Rate Hike Selloff
    print("\n[4] SPY - 2022 Rate Hikes")
    df = fetch_and_process("SPY", "2022-01-01", "2022-12-31", "spy_2022_rates")
    if df is not None:
        save_csv(df, "spy_2022_rates.csv")
        save_binary(df, "spy_2022_rates.csv")
        datasets.append(("spy_2022", df))
    
    # =========================================================================
    # 3. High-volatility assets (more tail events)
    # =========================================================================
    
    # VIX (Volatility Index) - for reference
    print("\n[5] VIX - Volatility Index")
    df = fetch_and_process("^VIX", "2007-01-01", "2024-12-31", "vix_full")
    if df is not None:
        save_csv(df, "vix_full.csv")
        save_binary(df, "vix_full.csv")
    
    # QQQ (Nasdaq 100) - tech-heavy, more volatile
    print("\n[6] QQQ - Nasdaq 100 ETF")
    df = fetch_and_process("QQQ", "2007-01-01", "2024-12-31", "qqq_full")
    if df is not None:
        save_csv(df, "qqq_full.csv")
        save_binary(df, "qqq_full.csv")
        datasets.append(("qqq_full", df))
    
    # IWM (Russell 2000) - small caps, more volatile
    print("\n[7] IWM - Russell 2000 ETF")
    df = fetch_and_process("IWM", "2007-01-01", "2024-12-31", "iwm_full")
    if df is not None:
        save_csv(df, "iwm_full.csv")
        save_binary(df, "iwm_full.csv")
    
    # =========================================================================
    # 4. Single volatile stocks (extreme tail events)
    # =========================================================================
    
    # TSLA - known for extreme moves
    print("\n[8] TSLA - Tesla (high vol)")
    df = fetch_and_process("TSLA", "2015-01-01", "2024-12-31", "tsla")
    if df is not None:
        save_csv(df, "tsla.csv")
        save_binary(df, "tsla.csv")
    
    # =========================================================================
    # 5. Create combined crash dataset (all crashes concatenated)
    # =========================================================================
    print("\n[9] Creating Combined Crash Dataset...")
    
    crash_periods = [
        ("SPY", "2008-09-01", "2009-03-31"),  # 2008 crisis peak
        ("SPY", "2010-05-01", "2010-06-30"),  # Flash crash
        ("SPY", "2011-07-01", "2011-10-31"),  # Debt downgrade
        ("SPY", "2015-08-01", "2015-09-30"),  # China fears
        ("SPY", "2018-01-15", "2018-04-30"),  # Volmageddon
        ("SPY", "2018-10-01", "2018-12-31"),  # Q4 selloff
        ("SPY", "2020-02-01", "2020-04-30"),  # COVID crash
        ("SPY", "2022-01-01", "2022-10-31"),  # Rate hikes
    ]
    
    crash_dfs = []
    for ticker, start, end in crash_periods:
        df = fetch_and_process(ticker, start, end, f"crash_{start[:4]}")
        if df is not None:
            crash_dfs.append(df)
    
    if crash_dfs:
        combined = pd.concat(crash_dfs, ignore_index=True)
        print(f"\n  Combined crash dataset: {len(combined)} observations")
        
        # Recompute stats for combined
        z = combined['log_return'] / combined['log_return'].std()
        n_3sigma = (np.abs(z) > 3).sum()
        n_4sigma = (np.abs(z) > 4).sum()
        n_5sigma = (np.abs(z) > 5).sum()
        print(f"  Tail events: {n_3sigma} (>3σ), {n_4sigma} (>4σ), {n_5sigma} (>5σ)")
        
        save_csv(combined, "crashes_combined.csv")
        save_binary(combined, "crashes_combined.csv")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"\n Output directory: {os.path.abspath(OUTPUT_DIR)}/")
    print("\n Files created:")
    print("   *.csv         - Full data (date, price, return, log_return)")
    print("   *_returns.csv - Returns only (one value per line)")
    print("   *.bin         - Binary float32 (fastest for C/CUDA)")
    print("\n Recommended for SVPF testing:")
    print("   - crashes_combined.bin  : All crashes, maximum tail events")
    print("   - spy_2008_crisis.bin   : 2008 crisis only")
    print("   - spy_2020_covid.bin    : COVID crash only")
    print("   - spy_full.bin          : Long history for stability testing")
    
    # =========================================================================
    # Generate C header with data info
    # =========================================================================
    header_path = os.path.join(OUTPUT_DIR, "market_data_info.h")
    with open(header_path, 'w') as f:
        f.write("// Auto-generated market data info\n")
        f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
        f.write("#ifndef MARKET_DATA_INFO_H\n")
        f.write("#define MARKET_DATA_INFO_H\n\n")
        
        for name, df in datasets:
            n = len(df)
            f.write(f"#define {name.upper()}_N {n}\n")
        
        f.write("\n// Usage:\n")
        f.write("//   float* returns = (float*)malloc(SPY_FULL_N * sizeof(float));\n")
        f.write("//   FILE* f = fopen(\"market_data/spy_full.bin\", \"rb\");\n")
        f.write("//   fread(returns, sizeof(float), SPY_FULL_N, f);\n")
        f.write("//   fclose(f);\n\n")
        f.write("#endif // MARKET_DATA_INFO_H\n")
    
    print(f"\n C header: {header_path}")
    print("\n Done!")

if __name__ == "__main__":
    main()
