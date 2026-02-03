"""
IV Analysis Pipeline in PyTorch
================================
Complete toolkit for:
1. Deribit data fetching
2. Black-Scholes IV inversion (autodiff Newton-Raphson)
3. SVI surface calibration
4. VRP analysis

Author: For TUGBARS
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from pathlib import Path

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# PART 1: BLACK-SCHOLES IN PYTORCH
# =============================================================================

class BlackScholes:
    """
    Black-Scholes pricing and IV inversion using PyTorch.
    GPU-accelerated, batched operations, autodiff for Newton-Raphson.
    """
    
    @staticmethod
    def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
        """Approximation of standard normal CDF."""
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    
    @staticmethod
    def standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
        """Standard normal PDF."""
        return torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    @classmethod
    def d1_d2(cls, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, 
              r: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute d1 and d2."""
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2
    
    @classmethod
    def price(cls, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
              r: torch.Tensor, sigma: torch.Tensor, 
              is_call: torch.Tensor) -> torch.Tensor:
        """
        Black-Scholes option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            is_call: Boolean tensor (True for call, False for put)
        
        Returns:
            Option price
        """
        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        
        N_d1 = cls.standard_normal_cdf(d1)
        N_d2 = cls.standard_normal_cdf(d2)
        
        discount = torch.exp(-r * T)
        
        call_price = S * N_d1 - K * discount * N_d2
        put_price = K * discount * (1 - N_d2) - S * (1 - N_d1)
        
        return torch.where(is_call, call_price, put_price)
    
    @classmethod
    def vega(cls, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
             r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Vega: sensitivity to volatility."""
        d1, _ = cls.d1_d2(S, K, T, r, sigma)
        return S * cls.standard_normal_pdf(d1) * torch.sqrt(T)
    
    @classmethod
    def implied_volatility(cls, market_price: torch.Tensor, S: torch.Tensor, 
                           K: torch.Tensor, T: torch.Tensor, r: torch.Tensor,
                           is_call: torch.Tensor, 
                           max_iter: int = 100,
                           tol: float = 1e-6) -> torch.Tensor:
        """
        Implied volatility via Newton-Raphson with autodiff.
        
        Batched: computes IV for all options simultaneously.
        
        Returns:
            Implied volatility tensor (NaN where failed to converge)
        """
        # Initial guess: Brenner-Subrahmanyam approximation
        sigma = torch.sqrt(2 * np.pi / T) * market_price / S
        sigma = torch.clamp(sigma, 0.01, 5.0)  # Reasonable bounds
        sigma.requires_grad_(True)
        
        for i in range(max_iter):
            # Compute price and vega
            price = cls.price(S, K, T, r, sigma, is_call)
            vega = cls.vega(S, K, T, r, sigma)
            
            # Newton-Raphson update
            diff = price - market_price
            
            # Avoid division by zero
            vega_safe = torch.where(vega.abs() < 1e-10, 
                                    torch.ones_like(vega) * 1e-10, 
                                    vega)
            
            update = diff / vega_safe
            sigma_new = sigma - update
            
            # Clamp to valid range
            sigma_new = torch.clamp(sigma_new, 0.001, 10.0)
            
            # Check convergence
            converged = diff.abs() < tol
            if converged.all():
                break
            
            sigma = sigma_new.detach().requires_grad_(True)
        
        # Mark non-converged as NaN
        result = sigma.detach()
        result = torch.where(diff.abs() < tol * 100, result, torch.tensor(float('nan')))
        
        return result


# =============================================================================
# PART 2: SVI SURFACE MODEL
# =============================================================================

class SVIModel(nn.Module):
    """
    Stochastic Volatility Inspired (SVI) parametrization.
    
    Raw SVI formula:
        w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
    
    where:
        k = log(K/F) = log-moneyness (forward)
        w = total implied variance = σ_BS² * T
        a, b, ρ, m, σ = parameters
    
    Constraints for no-arbitrage:
        b >= 0
        |ρ| < 1
        σ > 0
        a + b * σ * sqrt(1 - ρ²) >= 0
    """
    
    def __init__(self, initial_params: Optional[Dict[str, float]] = None):
        super().__init__()
        
        if initial_params is None:
            initial_params = {
                'a': 0.04,    # Base variance level
                'b': 0.1,     # Slope
                'rho': -0.3,  # Skew (negative = typical)
                'm': 0.0,     # Center
                'sigma': 0.1  # Curvature
            }
        
        # Use unconstrained parameters, apply constraints in forward
        self.a_raw = nn.Parameter(torch.tensor(initial_params['a']))
        self.b_raw = nn.Parameter(torch.tensor(np.log(initial_params['b'])))  # exp to ensure > 0
        self.rho_raw = nn.Parameter(torch.tensor(np.arctanh(initial_params['rho'])))  # tanh to ensure |ρ| < 1
        self.m = nn.Parameter(torch.tensor(initial_params['m']))
        self.sigma_raw = nn.Parameter(torch.tensor(np.log(initial_params['sigma'])))  # exp to ensure > 0
    
    def get_constrained_params(self) -> Dict[str, torch.Tensor]:
        """Apply constraints to raw parameters."""
        a = self.a_raw
        b = torch.exp(self.b_raw)  # b > 0
        rho = torch.tanh(self.rho_raw)  # |ρ| < 1
        m = self.m
        sigma = torch.exp(self.sigma_raw)  # σ > 0
        
        return {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma}
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compute total implied variance w(k).
        
        Args:
            k: Log-moneyness = log(K/F)
        
        Returns:
            w: Total implied variance
        """
        p = self.get_constrained_params()
        
        k_shifted = k - p['m']
        sqrt_term = torch.sqrt(k_shifted**2 + p['sigma']**2)
        
        w = p['a'] + p['b'] * (p['rho'] * k_shifted + sqrt_term)
        
        # Ensure w > 0 (variance must be positive)
        w = torch.clamp(w, min=1e-8)
        
        return w
    
    def iv_from_w(self, w: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Convert total variance to implied vol."""
        return torch.sqrt(w / T)
    
    def implied_volatility(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Get implied volatility for given moneyness and expiry."""
        w = self.forward(k)
        return self.iv_from_w(w, T)


class SVICalibrator:
    """
    Calibrate SVI model to market IV data.
    Uses PyTorch optimizer (Adam or LBFGS).
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
    
    def calibrate(self, 
                  k: torch.Tensor,           # Log-moneyness
                  T: torch.Tensor,           # Time to expiry (scalar for one slice)
                  market_iv: torch.Tensor,   # Market implied vols
                  weights: Optional[torch.Tensor] = None,
                  max_iter: int = 1000,
                  lr: float = 0.01,
                  verbose: bool = False) -> SVIModel:
        """
        Fit SVI to one expiry slice.
        
        Args:
            k: Log-moneyness for each strike
            T: Time to expiry (years)
            market_iv: Observed implied volatilities
            weights: Optional weights (e.g., inverse bid-ask spread)
            max_iter: Maximum iterations
            lr: Learning rate
            verbose: Print progress
        
        Returns:
            Calibrated SVIModel
        """
        k = k.to(self.device)
        T = T.to(self.device) if isinstance(T, torch.Tensor) else torch.tensor(T, device=self.device)
        market_iv = market_iv.to(self.device)
        
        if weights is None:
            weights = torch.ones_like(market_iv)
        weights = weights.to(self.device)
        
        # Target: total variance
        market_w = (market_iv ** 2) * T
        
        # Initialize model
        model = SVIModel().to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
        
        best_loss = float('inf')
        best_state = None
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # Forward
            pred_w = model(k)
            
            # Weighted MSE loss
            loss = (weights * (pred_w - market_w)**2).mean()
            
            # Regularization: penalize extreme parameters
            params = model.get_constrained_params()
            reg = 0.001 * (params['b']**2 + params['sigma']**2)
            total_loss = loss + reg
            
            # Backward
            total_loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            if verbose and i % 100 == 0:
                print(f"Iter {i}: loss = {loss.item():.6f}")
        
        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return model


# =============================================================================
# PART 3: IV SURFACE MANAGER
# =============================================================================

@dataclass
class IVPoint:
    """Single IV observation."""
    timestamp: datetime
    expiry: datetime
    strike: float
    spot: float
    iv: float
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    is_call: bool = True


class IVSurface:
    """
    Manages IV surface: stores data, fits SVI, interpolates.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.calibrator = SVICalibrator(device)
        self.svi_models: Dict[float, SVIModel] = {}  # T -> SVIModel
        self.expiries: List[float] = []
    
    def fit_slice(self, 
                  strikes: np.ndarray,
                  spot: float,
                  T: float,
                  ivs: np.ndarray,
                  weights: Optional[np.ndarray] = None,
                  verbose: bool = False) -> SVIModel:
        """
        Fit SVI to one expiry slice.
        
        Args:
            strikes: Strike prices
            spot: Current spot price
            T: Time to expiry in years
            ivs: Implied volatilities
            weights: Optional weights
        """
        # Compute log-moneyness
        forward = spot  # Assuming r ≈ 0 for crypto
        k = np.log(strikes / forward)
        
        k_tensor = torch.tensor(k, dtype=torch.float32)
        iv_tensor = torch.tensor(ivs, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32)
        
        if weights is not None:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            weights_tensor = None
        
        model = self.calibrator.calibrate(
            k_tensor, T_tensor, iv_tensor,
            weights=weights_tensor,
            verbose=verbose
        )
        
        self.svi_models[T] = model
        if T not in self.expiries:
            self.expiries.append(T)
            self.expiries.sort()
        
        return model
    
    def get_iv(self, strike: float, spot: float, T: float) -> float:
        """
        Get IV for arbitrary strike and expiry via interpolation.
        
        Uses linear interpolation in total variance space across expiries.
        """
        if len(self.expiries) == 0:
            raise ValueError("No SVI models fitted yet")
        
        forward = spot
        k = np.log(strike / forward)
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        T_tensor = torch.tensor(T, dtype=torch.float32, device=self.device)
        
        # Find bracketing expiries
        if T <= self.expiries[0]:
            # Extrapolate from shortest expiry
            model = self.svi_models[self.expiries[0]]
            w = model(k_tensor)
            iv = model.iv_from_w(w, T_tensor)
            return iv.item()
        
        if T >= self.expiries[-1]:
            # Extrapolate from longest expiry
            model = self.svi_models[self.expiries[-1]]
            w = model(k_tensor)
            iv = model.iv_from_w(w, T_tensor)
            return iv.item()
        
        # Find bracketing expiries
        T_low = None
        T_high = None
        for i, t in enumerate(self.expiries):
            if t >= T:
                T_high = t
                T_low = self.expiries[i - 1]
                break
        
        # Get total variance at each expiry
        model_low = self.svi_models[T_low]
        model_high = self.svi_models[T_high]
        
        T_low_tensor = torch.tensor(T_low, dtype=torch.float32, device=self.device)
        T_high_tensor = torch.tensor(T_high, dtype=torch.float32, device=self.device)
        
        w_low = model_low(k_tensor) * T_low_tensor / T_low  # w = σ² * T
        w_high = model_high(k_tensor) * T_high_tensor / T_high
        
        # Linear interpolation in variance space
        alpha = (T - T_low) / (T_high - T_low)
        w_interp = (1 - alpha) * w_low + alpha * w_high
        
        # Convert back to IV
        iv = torch.sqrt(w_interp / T_tensor)
        
        return iv.item()
    
    def get_atm_iv(self, spot: float, T: float) -> float:
        """Get ATM implied volatility."""
        return self.get_iv(spot, spot, T)
    
    def get_skew(self, spot: float, T: float, delta: float = 0.25) -> float:
        """
        Get skew: 25-delta put IV minus 25-delta call IV.
        
        Approximation: use fixed moneyness instead of delta.
        """
        # Rough approximation: 25-delta ≈ ±10% OTM for typical vol
        put_strike = spot * 0.90
        call_strike = spot * 1.10
        
        put_iv = self.get_iv(put_strike, spot, T)
        call_iv = self.get_iv(call_strike, spot, T)
        
        return put_iv - call_iv


# =============================================================================
# PART 4: DERIBIT DATA FETCHER
# =============================================================================

class DeribitFetcher:
    """
    Async fetcher for Deribit option data.
    Stores data in SQLite for analysis.
    """
    
    BASE_URL = "https://www.deribit.com/api/v2"
    
    def __init__(self, db_path: str = "iv_data.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS option_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                currency TEXT NOT NULL,
                instrument_name TEXT NOT NULL,
                expiry TEXT NOT NULL,
                strike REAL NOT NULL,
                is_call INTEGER NOT NULL,
                spot REAL NOT NULL,
                mark_price REAL,
                bid_price REAL,
                ask_price REAL,
                mark_iv REAL,
                bid_iv REAL,
                ask_iv REAL,
                open_interest REAL,
                volume REAL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON option_snapshots(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_currency_expiry 
            ON option_snapshots(currency, expiry)
        """)
        
        conn.commit()
        conn.close()
    
    async def _request(self, session: aiohttp.ClientSession, 
                       endpoint: str, params: Dict = None) -> Dict:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        async with session.get(url, params=params) as response:
            data = await response.json()
            if 'result' in data:
                return data['result']
            raise ValueError(f"API error: {data}")
    
    async def get_instruments(self, session: aiohttp.ClientSession,
                              currency: str = "BTC") -> List[Dict]:
        """Get all option instruments."""
        return await self._request(session, "public/get_instruments", {
            "currency": currency,
            "kind": "option",
            "expired": "false"
        })
    
    async def get_ticker(self, session: aiohttp.ClientSession,
                         instrument_name: str) -> Dict:
        """Get ticker for single instrument."""
        return await self._request(session, "public/ticker", {
            "instrument_name": instrument_name
        })
    
    async def get_index_price(self, session: aiohttp.ClientSession,
                              currency: str = "BTC") -> float:
        """Get current index (spot) price."""
        result = await self._request(session, "public/get_index_price", {
            "index_name": f"{currency.lower()}_usd"
        })
        return result['index_price']
    
    async def fetch_all_options(self, currency: str = "BTC") -> List[Dict]:
        """
        Fetch all option data for a currency.
        Returns list of option snapshots.
        """
        async with aiohttp.ClientSession() as session:
            # Get spot price
            spot = await self.get_index_price(session, currency)
            
            # Get all instruments
            instruments = await self.get_instruments(session, currency)
            
            # Fetch tickers (batch for efficiency)
            snapshots = []
            timestamp = datetime.utcnow().isoformat()
            
            # Process in batches to avoid rate limits
            batch_size = 20
            for i in range(0, len(instruments), batch_size):
                batch = instruments[i:i + batch_size]
                tasks = [self.get_ticker(session, inst['instrument_name']) 
                         for inst in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for inst, ticker in zip(batch, results):
                    if isinstance(ticker, Exception):
                        continue
                    
                    # Parse instrument name: BTC-28MAR25-100000-C
                    parts = inst['instrument_name'].split('-')
                    is_call = parts[-1] == 'C'
                    strike = float(parts[-2])
                    
                    snapshot = {
                        'timestamp': timestamp,
                        'currency': currency,
                        'instrument_name': inst['instrument_name'],
                        'expiry': inst['expiration_timestamp'],
                        'strike': strike,
                        'is_call': is_call,
                        'spot': spot,
                        'mark_price': ticker.get('mark_price'),
                        'bid_price': ticker.get('best_bid_price'),
                        'ask_price': ticker.get('best_ask_price'),
                        'mark_iv': ticker.get('mark_iv'),
                        'bid_iv': ticker.get('bid_iv'),
                        'ask_iv': ticker.get('ask_iv'),
                        'open_interest': ticker.get('open_interest'),
                        'volume': ticker.get('stats', {}).get('volume')
                    }
                    snapshots.append(snapshot)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            return snapshots
    
    def save_snapshots(self, snapshots: List[Dict]):
        """Save snapshots to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for snap in snapshots:
            cursor.execute("""
                INSERT INTO option_snapshots 
                (timestamp, currency, instrument_name, expiry, strike, is_call,
                 spot, mark_price, bid_price, ask_price, mark_iv, bid_iv, ask_iv,
                 open_interest, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snap['timestamp'], snap['currency'], snap['instrument_name'],
                snap['expiry'], snap['strike'], int(snap['is_call']),
                snap['spot'], snap['mark_price'], snap['bid_price'],
                snap['ask_price'], snap['mark_iv'], snap['bid_iv'],
                snap['ask_iv'], snap['open_interest'], snap['volume']
            ))
        
        conn.commit()
        conn.close()
        print(f"Saved {len(snapshots)} snapshots")
    
    async def run_snapshot(self, currency: str = "BTC"):
        """Fetch and save current snapshot."""
        snapshots = await self.fetch_all_options(currency)
        self.save_snapshots(snapshots)
        return snapshots
    
    def load_snapshots(self, 
                       currency: str = "BTC",
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> List[Dict]:
        """Load snapshots from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM option_snapshots WHERE currency = ?"
        params = [currency]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]


# =============================================================================
# PART 5: VRP ANALYSIS
# =============================================================================

class VRPAnalyzer:
    """
    Variance Risk Premium analysis.
    Compares your RV estimate vs market IV.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
    
    def compute_vrp(self,
                    svpf_rv: torch.Tensor,    # Your RV estimates
                    market_iv: torch.Tensor,   # Market IV (ATM)
                    ) -> Dict[str, torch.Tensor]:
        """
        Compute VRP-related metrics.
        
        Args:
            svpf_rv: Your volatility estimates (annualized)
            market_iv: Market implied volatility (annualized)
        
        Returns:
            Dictionary with VRP metrics
        """
        # Basic spread
        spread = market_iv - svpf_rv  # Positive = IV > RV (sell vol)
        
        # Normalized spread (z-score)
        spread_mean = spread.mean()
        spread_std = spread.std()
        spread_zscore = (spread - spread_mean) / (spread_std + 1e-8)
        
        # Log ratio (symmetric)
        log_ratio = torch.log(market_iv / svpf_rv)
        
        return {
            'spread': spread,
            'spread_zscore': spread_zscore,
            'log_ratio': log_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std
        }
    
    def compute_realized_vrp(self,
                             market_iv: torch.Tensor,
                             realized_vol: torch.Tensor  # Actual future RV
                             ) -> torch.Tensor:
        """
        Compute realized VRP (hindsight).
        This is the ground truth for evaluating predictions.
        """
        return market_iv - realized_vol
    
    def evaluate_edge(self,
                      svpf_rv: torch.Tensor,
                      market_iv: torch.Tensor,
                      realized_vol: torch.Tensor  # Future realized vol
                      ) -> Dict[str, float]:
        """
        Evaluate whether your SVPF has edge over market IV.
        
        Args:
            svpf_rv: Your RV estimates at time t
            market_iv: Market IV at time t
            realized_vol: Actual RV from t to t+T (hindsight)
        
        Returns:
            Dictionary with edge metrics
        """
        # Prediction errors
        your_error = torch.abs(svpf_rv - realized_vol)
        market_error = torch.abs(market_iv - realized_vol)
        
        # Are you better?
        your_wins = (your_error < market_error).float().mean()
        
        # Mean absolute error
        your_mae = your_error.mean()
        market_mae = market_error.mean()
        
        # RMSE
        your_rmse = torch.sqrt((your_error**2).mean())
        market_rmse = torch.sqrt((market_error**2).mean())
        
        # Correlation with realized
        your_corr = torch.corrcoef(torch.stack([svpf_rv, realized_vol]))[0, 1]
        market_corr = torch.corrcoef(torch.stack([market_iv, realized_vol]))[0, 1]
        
        # Spread prediction: does high spread predict profitable short vol?
        spread = market_iv - svpf_rv
        vrp_realized = market_iv - realized_vol
        spread_corr = torch.corrcoef(torch.stack([spread, vrp_realized]))[0, 1]
        
        return {
            'your_wins_pct': your_wins.item(),
            'your_mae': your_mae.item(),
            'market_mae': market_mae.item(),
            'mae_improvement': (market_mae - your_mae).item(),
            'your_rmse': your_rmse.item(),
            'market_rmse': market_rmse.item(),
            'rmse_improvement': (market_rmse - your_rmse).item(),
            'your_corr_with_realized': your_corr.item(),
            'market_corr_with_realized': market_corr.item(),
            'spread_predicts_vrp_corr': spread_corr.item()
        }
    
    def generate_signal(self,
                        svpf_rv: float,
                        market_iv: float,
                        spread_mean: float,
                        spread_std: float,
                        threshold_z: float = 1.5
                        ) -> Dict[str, any]:
        """
        Generate trading signal based on current spread.
        
        Args:
            svpf_rv: Current RV estimate
            market_iv: Current market IV
            spread_mean: Historical mean spread
            spread_std: Historical spread std
            threshold_z: Z-score threshold for trading
        
        Returns:
            Trading signal with confidence
        """
        spread = market_iv - svpf_rv
        zscore = (spread - spread_mean) / (spread_std + 1e-8)
        
        if zscore > threshold_z:
            signal = 'SELL_VOL'
            confidence = min(abs(zscore) / 3.0, 1.0)  # Scale confidence
        elif zscore < -threshold_z:
            signal = 'BUY_VOL'
            confidence = min(abs(zscore) / 3.0, 1.0)
        else:
            signal = 'NO_TRADE'
            confidence = 0.0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'spread': spread,
            'zscore': zscore,
            'svpf_rv': svpf_rv,
            'market_iv': market_iv
        }


# =============================================================================
# PART 6: USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Example of full pipeline."""
    
    print("=" * 60)
    print("IV Analysis Pipeline - Example Usage")
    print("=" * 60)
    
    # 1. Black-Scholes IV inversion
    print("\n1. Black-Scholes IV Inversion")
    print("-" * 40)
    
    # Example: BTC option
    S = torch.tensor([50000.0])  # Spot
    K = torch.tensor([52000.0])  # Strike
    T = torch.tensor([30/365])   # 30 days
    r = torch.tensor([0.0])      # Risk-free rate (crypto)
    market_price = torch.tensor([2500.0])  # Option price
    is_call = torch.tensor([True])
    
    iv = BlackScholes.implied_volatility(market_price, S, K, T, r, is_call)
    print(f"Market price: ${market_price.item():.2f}")
    print(f"Implied Vol: {iv.item() * 100:.2f}%")
    
    # Verify: price back
    price_check = BlackScholes.price(S, K, T, r, iv, is_call)
    print(f"Price check: ${price_check.item():.2f}")
    
    # 2. SVI Calibration
    print("\n2. SVI Surface Calibration")
    print("-" * 40)
    
    # Simulated market data for one expiry slice
    spot = 50000.0
    strikes = np.array([40000, 45000, 48000, 50000, 52000, 55000, 60000])
    market_ivs = np.array([0.65, 0.55, 0.50, 0.48, 0.50, 0.54, 0.62])  # Smile shape
    T_expiry = 30 / 365  # 30 days
    
    surface = IVSurface()
    model = surface.fit_slice(strikes, spot, T_expiry, market_ivs, verbose=False)
    
    params = model.get_constrained_params()
    print("SVI Parameters:")
    print(f"  a (level):  {params['a'].item():.4f}")
    print(f"  b (slope):  {params['b'].item():.4f}")
    print(f"  ρ (skew):   {params['rho'].item():.4f}")
    print(f"  m (center): {params['m'].item():.4f}")
    print(f"  σ (curve):  {params['sigma'].item():.4f}")
    
    # Check fit
    print("\nFit quality:")
    for strike, market_iv in zip(strikes, market_ivs):
        fitted_iv = surface.get_iv(strike, spot, T_expiry)
        print(f"  K={strike}: market={market_iv*100:.1f}%, fitted={fitted_iv*100:.1f}%")
    
    # 3. VRP Analysis
    print("\n3. VRP Analysis")
    print("-" * 40)
    
    # Simulated data
    n_samples = 100
    np.random.seed(42)
    
    # Simulate: your SVPF is slightly better than market
    true_vol = 0.45 + 0.1 * np.random.randn(n_samples).cumsum() * 0.01
    true_vol = np.clip(true_vol, 0.2, 0.8)
    
    market_iv_series = true_vol + 0.05 + 0.03 * np.random.randn(n_samples)  # IV with premium + noise
    svpf_rv_series = true_vol + 0.02 * np.random.randn(n_samples)  # Your estimate (less noise)
    realized_vol = true_vol + 0.01 * np.random.randn(n_samples)  # Actual future RV
    
    # Convert to tensors
    svpf_rv = torch.tensor(svpf_rv_series, dtype=torch.float32)
    market_iv = torch.tensor(market_iv_series, dtype=torch.float32)
    realized = torch.tensor(realized_vol, dtype=torch.float32)
    
    analyzer = VRPAnalyzer()
    
    # Evaluate edge
    edge_metrics = analyzer.evaluate_edge(svpf_rv, market_iv, realized)
    
    print("Edge Evaluation:")
    print(f"  Your wins: {edge_metrics['your_wins_pct']*100:.1f}%")
    print(f"  Your MAE: {edge_metrics['your_mae']*100:.2f}%")
    print(f"  Market MAE: {edge_metrics['market_mae']*100:.2f}%")
    print(f"  MAE improvement: {edge_metrics['mae_improvement']*100:.2f}%")
    print(f"  Your corr with realized: {edge_metrics['your_corr_with_realized']:.3f}")
    print(f"  Market corr with realized: {edge_metrics['market_corr_with_realized']:.3f}")
    print(f"  Spread predicts VRP (corr): {edge_metrics['spread_predicts_vrp_corr']:.3f}")
    
    # 4. Generate trading signal
    print("\n4. Trading Signal Example")
    print("-" * 40)
    
    # Current state
    current_svpf_rv = 0.42
    current_market_iv = 0.55
    historical_spread_mean = 0.05
    historical_spread_std = 0.03
    
    signal = analyzer.generate_signal(
        current_svpf_rv, current_market_iv,
        historical_spread_mean, historical_spread_std,
        threshold_z=1.5
    )
    
    print(f"Current SVPF RV: {signal['svpf_rv']*100:.1f}%")
    print(f"Current Market IV: {signal['market_iv']*100:.1f}%")
    print(f"Spread: {signal['spread']*100:.1f}%")
    print(f"Z-score: {signal['zscore']:.2f}")
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("Pipeline ready for use!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
