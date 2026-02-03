"""
VRP Research Script
===================
Use this to analyze your collected IV data against your SVPF estimates.

This script shows:
1. How to load and process collected data
2. How to build IV surfaces
3. How to compare your RV estimates vs market IV
4. How to backtest the VRP strategy
"""

import numpy as np
import torch
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

from iv_analysis import (
    DeribitFetcher, 
    IVSurface, 
    VRPAnalyzer,
    BlackScholes,
    DEVICE
)


@dataclass
class AnalysisConfig:
    """Configuration for VRP analysis."""
    db_path: str = "iv_data.db"
    currency: str = "BTC"
    rv_window_days: int = 7          # Window for realized vol computation
    min_expiry_days: int = 5         # Minimum expiry for analysis
    max_expiry_days: int = 45        # Maximum expiry for analysis
    min_open_interest: float = 10    # Minimum OI filter
    zscore_threshold: float = 1.5    # Signal threshold


class DataLoader:
    """Load and preprocess IV data from database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def load_snapshots_by_date(self, 
                                currency: str,
                                date: str) -> List[Dict]:
        """Load all snapshots for a specific date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM option_snapshots 
            WHERE currency = ? 
            AND timestamp LIKE ?
            ORDER BY timestamp
        """, (currency, f"{date}%"))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def load_latest_snapshot(self, currency: str) -> Tuple[str, List[Dict]]:
        """Load the most recent snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest timestamp
        cursor.execute("""
            SELECT MAX(timestamp) FROM option_snapshots 
            WHERE currency = ?
        """, (currency,))
        latest_ts = cursor.fetchone()[0]
        
        if latest_ts is None:
            conn.close()
            return None, []
        
        # Get all options at that timestamp
        cursor.execute("""
            SELECT * FROM option_snapshots 
            WHERE currency = ? AND timestamp = ?
        """, (currency, latest_ts))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        return latest_ts, [dict(zip(columns, row)) for row in rows]
    
    def load_atm_iv_series(self, 
                           currency: str,
                           expiry_days: int = 7,
                           tolerance_days: int = 3) -> List[Dict]:
        """
        Load ATM IV time series for a target expiry.
        
        Finds closest ATM option to specified expiry at each timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all unique timestamps
        cursor.execute("""
            SELECT DISTINCT timestamp FROM option_snapshots 
            WHERE currency = ?
            ORDER BY timestamp
        """, (currency,))
        timestamps = [row[0] for row in cursor.fetchall()]
        
        series = []
        target_dte = expiry_days
        
        for ts in timestamps:
            # Get options at this timestamp
            cursor.execute("""
                SELECT * FROM option_snapshots 
                WHERE currency = ? AND timestamp = ?
            """, (currency, ts))
            
            columns = [desc[0] for desc in cursor.description]
            options = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            if not options:
                continue
            
            spot = options[0]['spot']
            
            # Filter by expiry and find ATM
            best_option = None
            best_score = float('inf')
            
            for opt in options:
                # Parse expiry
                expiry_ms = opt['expiry']
                ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
                expiry_dt = datetime.utcfromtimestamp(expiry_ms / 1000)
                dte = (expiry_dt - ts_dt).days
                
                # Check expiry range
                if abs(dte - target_dte) > tolerance_days:
                    continue
                
                # Score: closeness to ATM + closeness to target expiry
                moneyness = abs(np.log(opt['strike'] / spot))
                expiry_diff = abs(dte - target_dte)
                score = moneyness + 0.1 * expiry_diff
                
                if score < best_score and opt['mark_iv'] is not None:
                    best_score = score
                    best_option = opt
                    best_option['dte'] = dte
            
            if best_option:
                series.append({
                    'timestamp': ts,
                    'spot': spot,
                    'strike': best_option['strike'],
                    'dte': best_option['dte'],
                    'iv': best_option['mark_iv'] / 100,  # Convert to decimal
                    'bid_iv': (best_option['bid_iv'] or 0) / 100,
                    'ask_iv': (best_option['ask_iv'] or 0) / 100,
                })
        
        conn.close()
        return series
    
    def load_spot_series(self, currency: str) -> List[Dict]:
        """Load spot price time series."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, spot 
            FROM option_snapshots 
            WHERE currency = ?
            GROUP BY timestamp
            ORDER BY timestamp
        """, (currency,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{'timestamp': row[0], 'spot': row[1]} for row in rows]


class RealizedVolCalculator:
    """
    Calculate realized volatility from spot prices.
    
    Multiple estimators available.
    """
    
    @staticmethod
    def close_to_close(prices: np.ndarray, window: int, annualize: float = 365) -> np.ndarray:
        """
        Standard close-to-close realized volatility.
        
        Args:
            prices: Array of prices
            window: Rolling window size
            annualize: Annualization factor
        
        Returns:
            Array of realized volatilities
        """
        log_returns = np.diff(np.log(prices))
        
        rv = np.full(len(prices), np.nan)
        for i in range(window, len(log_returns) + 1):
            rv[i] = np.std(log_returns[i-window:i]) * np.sqrt(annualize)
        
        return rv
    
    @staticmethod
    def parkinson(highs: np.ndarray, lows: np.ndarray, 
                  window: int, annualize: float = 365) -> np.ndarray:
        """
        Parkinson volatility estimator (uses high-low range).
        More efficient than close-to-close.
        """
        hl_ratio = np.log(highs / lows)
        factor = 1 / (4 * np.log(2))
        
        rv = np.full(len(highs), np.nan)
        for i in range(window, len(highs)):
            variance = factor * np.mean(hl_ratio[i-window:i]**2)
            rv[i] = np.sqrt(variance * annualize)
        
        return rv
    
    @staticmethod
    def forward_realized_vol(prices: np.ndarray, 
                             forward_window: int,
                             annualize: float = 365) -> np.ndarray:
        """
        Forward-looking realized vol (for backtesting evaluation).
        This is the "ground truth" for what vol actually was.
        """
        log_returns = np.diff(np.log(prices))
        
        rv = np.full(len(prices), np.nan)
        for i in range(len(log_returns) - forward_window):
            rv[i] = np.std(log_returns[i:i+forward_window]) * np.sqrt(annualize)
        
        return rv


class VRPBacktester:
    """
    Backtest VRP strategy.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.loader = DataLoader(config.db_path)
        self.rv_calc = RealizedVolCalculator()
        self.analyzer = VRPAnalyzer()
    
    def prepare_data(self) -> Dict:
        """
        Prepare aligned data for analysis.
        
        Returns dict with:
            - timestamps
            - spot prices
            - ATM IV (market)
            - Realized vol (backward-looking)
            - Forward realized vol (ground truth)
        """
        # Load ATM IV series
        iv_series = self.loader.load_atm_iv_series(
            self.config.currency,
            expiry_days=self.config.rv_window_days
        )
        
        if len(iv_series) < 30:
            raise ValueError(f"Not enough data: {len(iv_series)} snapshots")
        
        # Extract arrays
        timestamps = [s['timestamp'] for s in iv_series]
        spots = np.array([s['spot'] for s in iv_series])
        ivs = np.array([s['iv'] for s in iv_series])
        
        # Compute realized vols
        window = self.config.rv_window_days
        rv_backward = self.rv_calc.close_to_close(spots, window)
        rv_forward = self.rv_calc.forward_realized_vol(spots, window)
        
        return {
            'timestamps': timestamps,
            'spot': spots,
            'market_iv': ivs,
            'rv_backward': rv_backward,
            'rv_forward': rv_forward,
            'n_samples': len(timestamps)
        }
    
    def run_analysis(self, svpf_rv: Optional[np.ndarray] = None) -> Dict:
        """
        Run full VRP analysis.
        
        Args:
            svpf_rv: Your SVPF vol estimates (if None, uses backward RV as proxy)
        
        Returns:
            Analysis results
        """
        data = self.prepare_data()
        
        # If no SVPF provided, use backward RV as baseline
        if svpf_rv is None:
            print("Warning: No SVPF estimates provided, using backward RV as proxy")
            svpf_rv = data['rv_backward']
        
        # Find valid indices (where all data exists)
        valid = ~(np.isnan(data['market_iv']) | 
                  np.isnan(svpf_rv) | 
                  np.isnan(data['rv_forward']))
        
        # Convert to tensors
        market_iv = torch.tensor(data['market_iv'][valid], dtype=torch.float32)
        your_rv = torch.tensor(svpf_rv[valid], dtype=torch.float32)
        realized = torch.tensor(data['rv_forward'][valid], dtype=torch.float32)
        
        # Compute metrics
        edge_metrics = self.analyzer.evaluate_edge(your_rv, market_iv, realized)
        vrp_metrics = self.analyzer.compute_vrp(your_rv, market_iv)
        
        # Compute simple strategy P&L
        spread = (market_iv - your_rv).numpy()
        spread_mean = vrp_metrics['spread_mean'].item()
        spread_std = vrp_metrics['spread_std'].item()
        zscore = (spread - spread_mean) / (spread_std + 1e-8)
        
        # Signal: sell vol when zscore > threshold
        signals = np.where(zscore > self.config.zscore_threshold, 1,
                          np.where(zscore < -self.config.zscore_threshold, -1, 0))
        
        # P&L: profit when IV > RV realized and we sold vol (signal = 1)
        actual_vrp = (market_iv - realized).numpy()
        pnl_per_trade = signals * actual_vrp
        
        results = {
            'n_valid_samples': int(valid.sum()),
            'edge_metrics': edge_metrics,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'signal_count': {
                'sell_vol': int((signals == 1).sum()),
                'buy_vol': int((signals == -1).sum()),
                'no_trade': int((signals == 0).sum())
            },
            'strategy_pnl': {
                'total': float(pnl_per_trade.sum()),
                'mean_per_trade': float(pnl_per_trade[signals != 0].mean()) if (signals != 0).sum() > 0 else 0,
                'win_rate': float((pnl_per_trade[signals != 0] > 0).mean()) if (signals != 0).sum() > 0 else 0,
            }
        }
        
        return results
    
    def print_report(self, results: Dict):
        """Print analysis report."""
        print("\n" + "=" * 60)
        print("VRP ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\nData: {results['n_valid_samples']} valid samples")
        print(f"Spread: {results['spread_mean']*100:.2f}% ± {results['spread_std']*100:.2f}%")
        
        print("\n--- Edge Metrics ---")
        em = results['edge_metrics']
        print(f"Your wins vs market: {em['your_wins_pct']*100:.1f}%")
        print(f"Your MAE: {em['your_mae']*100:.2f}%")
        print(f"Market MAE: {em['market_mae']*100:.2f}%")
        print(f"MAE improvement: {em['mae_improvement']*100:.2f}%")
        print(f"Your corr with realized: {em['your_corr_with_realized']:.3f}")
        print(f"Market corr with realized: {em['market_corr_with_realized']:.3f}")
        print(f"Spread predicts VRP: {em['spread_predicts_vrp_corr']:.3f}")
        
        print("\n--- Trading Signals ---")
        sc = results['signal_count']
        print(f"Sell vol signals: {sc['sell_vol']}")
        print(f"Buy vol signals: {sc['buy_vol']}")
        print(f"No trade: {sc['no_trade']}")
        
        print("\n--- Strategy P&L ---")
        pnl = results['strategy_pnl']
        print(f"Total P&L (vol points): {pnl['total']*100:.2f}%")
        print(f"Mean per trade: {pnl['mean_per_trade']*100:.2f}%")
        print(f"Win rate: {pnl['win_rate']*100:.1f}%")
        
        print("\n" + "=" * 60)


class SVPFIntegration:
    """
    Placeholder for integrating your actual SVPF estimates.
    
    Replace this with your real SVPF output.
    """
    
    @staticmethod
    def load_svpf_estimates(filepath: str) -> np.ndarray:
        """
        Load SVPF vol estimates from file.
        
        Expected format: JSON or CSV with timestamps and vol estimates.
        
        You'll need to implement this based on your SVPF output format.
        """
        # Placeholder - implement based on your format
        raise NotImplementedError("Implement based on your SVPF output format")
    
    @staticmethod
    def align_with_iv_data(svpf_timestamps: List[str], 
                           svpf_vols: np.ndarray,
                           iv_timestamps: List[str]) -> np.ndarray:
        """
        Align SVPF estimates with IV timestamps.
        
        Uses nearest-neighbor matching.
        """
        from datetime import datetime
        
        # Convert to datetime
        svpf_dts = [datetime.fromisoformat(ts.replace('Z', '')) for ts in svpf_timestamps]
        iv_dts = [datetime.fromisoformat(ts.replace('Z', '')) for ts in iv_timestamps]
        
        aligned = np.full(len(iv_dts), np.nan)
        
        for i, iv_dt in enumerate(iv_dts):
            # Find closest SVPF timestamp
            diffs = [abs((svpf_dt - iv_dt).total_seconds()) for svpf_dt in svpf_dts]
            closest_idx = np.argmin(diffs)
            
            # Only use if within 1 hour
            if diffs[closest_idx] < 3600:
                aligned[i] = svpf_vols[closest_idx]
        
        return aligned


def main():
    """Run VRP analysis."""
    
    config = AnalysisConfig(
        db_path="iv_data.db",
        currency="BTC",
        rv_window_days=7,
        zscore_threshold=1.5
    )
    
    # Check if we have data
    loader = DataLoader(config.db_path)
    latest_ts, latest_data = loader.load_latest_snapshot(config.currency)
    
    if latest_ts is None:
        print("No data in database!")
        print("Run: python collector.py --once")
        print("Then run this script again.")
        return
    
    print(f"Latest snapshot: {latest_ts}")
    print(f"Options in snapshot: {len(latest_data)}")
    
    # Run backtest
    backtester = VRPBacktester(config)
    
    try:
        # For now, use backward RV as proxy for SVPF
        # Replace with your actual SVPF estimates:
        # svpf_rv = SVPFIntegration.load_svpf_estimates("your_svpf_output.json")
        
        results = backtester.run_analysis(svpf_rv=None)
        backtester.print_report(results)
        
    except ValueError as e:
        print(f"Analysis error: {e}")
        print("Collect more data before running analysis.")


if __name__ == "__main__":
    main()
