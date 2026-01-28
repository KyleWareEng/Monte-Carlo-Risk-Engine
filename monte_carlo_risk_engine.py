"""
Monte Carlo Risk Engine
-----------------------
My attempt at building a financial risk analysis tool using concepts from 
my engineering background. Started this after watching some QuantPy videos
and reading Hull's "Options, Futures and Derivatives" (chapter 14 was helpful).

The idea: use random simulations to predict where stock prices might go,
then figure out how much money you could lose in a bad scenario (VaR).

Author: Kyle W
Course: MEng Automotive Engineering (Year 4)
Date: January 2026

TODO:
- [ ] Add portfolio correlation (currently treats each stock independently)
- [ ] Try implementing GARCH for better volatility estimation  
- [ ] Maybe add a GUI with tkinter or streamlit?
- [ ] Compare results with actual VaR from Bloomberg terminal in uni library
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Quick dataclass to store results - learned about these in my Python course
# Much cleaner than using dictionaries everywhere
@dataclass
class RiskMetrics:
    """Stores all the risk numbers for one stock."""
    ticker: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    current_price: float
    
    def __repr__(self):
        # Formatted output - spent ages getting the alignment right lol
        return (
            f"\n{'='*55}\n"
            f"  {self.ticker} Risk Analysis\n"
            f"{'='*55}\n"
            f"  Current Price:         £{self.current_price:,.2f}\n"
            f"  Annual Return:         {self.annualized_return*100:,.2f}%\n"
            f"  Annual Volatility:     {self.annualized_volatility*100:,.2f}%\n"
            f"  Sharpe Ratio:          {self.sharpe_ratio:,.3f}\n"
            f"  Max Drawdown:          {self.max_drawdown*100:,.2f}%\n"
            f"  {'─'*53}\n"
            f"  95% VaR (1-Year):      £{self.var_95:,.2f}\n"
            f"  99% VaR (1-Year):      £{self.var_99:,.2f}\n"
            f"{'='*55}\n"
        )


class DataLoader:
    """
    Handles getting stock data from Yahoo Finance.
    
    Kept this separate from the simulation stuff because my software engineering
    module taught us about 'separation of concerns'. Makes testing easier too.
    
    Note: Yahoo Finance can be unreliable sometimes, so I added a fallback
    that generates fake data for testing when the network is down.
    """
    
    # Rough estimates based on what I've seen in the market
    # Used for generating test data when Yahoo is down
    FALLBACK_PARAMS = {
        'TSLA': {'price': 250.0, 'annual_return': 0.15, 'annual_vol': 0.55},
        'F': {'price': 11.0, 'annual_return': 0.05, 'annual_vol': 0.35},
        'SPY': {'price': 520.0, 'annual_return': 0.10, 'annual_vol': 0.18},
        'AAPL': {'price': 190.0, 'annual_return': 0.12, 'annual_vol': 0.25},
        'NVDA': {'price': 480.0, 'annual_return': 0.25, 'annual_vol': 0.50},
    }
    
    def __init__(self, tickers: List[str], period: str = '2y', use_fake_data: bool = False):
        self.tickers = tickers
        self.period = period
        self.use_fake_data = use_fake_data
        self._prices = None
        self._returns = None
    
    def _generate_fake_data(self, n_days: int = 504):
        """
        Generate synthetic price data when Yahoo Finance isn't working.
        Uses the same GBM model we use for simulation - bit circular but works for testing.
        
        504 days ≈ 2 years of trading days
        """
        print("Generating synthetic data for testing...")
        print("(Real data unavailable - using realistic estimates)\n")
        
        np.random.seed(42)  # Makes results reproducible for debugging
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
        
        data = {}
        for ticker in self.tickers:
            params = self.FALLBACK_PARAMS.get(ticker, {
                'price': 100.0, 'annual_return': 0.08, 'annual_vol': 0.25
            })
            
            # Generate a price path using GBM
            S0 = params['price']
            mu = params['annual_return']
            sigma = params['annual_vol']
            dt = 1/252  # 1 trading day
            
            prices = np.zeros(n_days)
            prices[0] = S0
            
            for t in range(1, n_days):
                # GBM step - same formula as in the simulator
                z = np.random.standard_normal()
                prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            
            data[ticker] = prices
        
        self._prices = pd.DataFrame(data, index=dates)
        return self._prices
    
    def get_prices(self):
        """Download price data or generate fake data if that fails."""
        if self.use_fake_data:
            return self._generate_fake_data()
        
        print(f"Downloading data for {self.tickers}...")
        try:
            self._prices = yf.download(self.tickers, period=self.period, progress=False)['Close']
            
            # yf returns a Series if only one ticker, need to convert
            if isinstance(self._prices, pd.Series):
                self._prices = self._prices.to_frame(name=self.tickers[0])
            
            self._prices = self._prices.dropna()
            
            if len(self._prices) == 0:
                print("No data received, falling back to synthetic data...")
                return self._generate_fake_data()
            
            print(f"Got {len(self._prices)} days of data\n")
            return self._prices
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("Using synthetic data instead...")
            return self._generate_fake_data()
    
    def get_returns(self):
        """
        Calculate log returns from prices.
        
        Why log returns instead of simple returns?
        1. They're additive over time (easier maths)
        2. They're roughly normally distributed
        3. Prices can't go negative with log returns
        
        Learned this from a quant finance YouTube video - wish my lecturers
        explained it this clearly!
        """
        if self._prices is None:
            self.get_prices()
        
        # Log return = ln(P_t / P_{t-1})
        self._returns = np.log(self._prices / self._prices.shift(1)).dropna()
        return self._returns
    
    @property
    def prices(self):
        if self._prices is None:
            self.get_prices()
        return self._prices
    
    @property
    def returns(self):
        if self._returns is None:
            self.get_returns()
        return self._returns


class MonteCarloSimulator:
    """
    The main simulation engine using Geometric Brownian Motion.
    
    GBM is the standard model for stock prices. The maths comes from
    stochastic calculus (way beyond my module, but I get the intuition).
    
    Basic idea:
    - Stock has a drift (mu) - the average direction it's heading
    - Stock has volatility (sigma) - how much it bounces around
    - Add random noise each day to simulate uncertainty
    
    The formula is:
        S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
    
    where Z is a random number from a normal distribution.
    
    That (mu - sigma²/2) term is called "Ito's correction" - it's a 
    mathematical adjustment needed because of how continuous random 
    processes work. Took me a while to understand why it's there.
    """
    
    def __init__(self, n_paths: int = 1000, n_days: int = 252, seed: int = 42):
        """
        Args:
            n_paths: Number of random scenarios to generate (more = better estimate)
            n_days: How far to simulate (252 trading days = 1 year)
            seed: Random seed for reproducibility
        """
        self.n_paths = n_paths
        self.n_days = n_days
        
        if seed is not None:
            np.random.seed(seed)
    
    def simulate(self, start_price: float, mu: float, sigma: float) -> np.ndarray:
        """
        Run the Monte Carlo simulation.
        
        Returns a 2D array where:
        - Rows = days (0 to n_days)
        - Columns = different simulation paths
        
        I tried to vectorize this properly with numpy but the loop version
        was easier to understand and debug. Could optimize later if speed
        becomes an issue.
        """
        dt = 1 / 252  # Time step = 1 trading day
        
        # Array to hold all price paths
        # Shape: (days+1, paths) - +1 because we include starting price
        paths = np.zeros((self.n_days + 1, self.n_paths))
        paths[0] = start_price
        
        # Generate all the random numbers upfront (faster than doing it in loop)
        random_shocks = np.random.standard_normal((self.n_days, self.n_paths))
        
        # Step through each day
        for t in range(1, self.n_days + 1):
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * random_shocks[t-1]
            paths[t] = paths[t-1] * np.exp(drift + diffusion)
        
        return paths
    
    def calculate_var(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        VaR answers: "What's the most I could lose, with X% confidence?"
        
        For 95% VaR:
        - Look at all 1000 ending prices
        - Calculate profit/loss for each
        - Find the 5th percentile (worst 5%)
        - That's your VaR
        
        Example: If VaR = -£50, you're 95% confident you won't lose more than £50.
        """
        start_price = paths[0, 0]
        end_prices = paths[-1, :]  # All the final prices
        
        profits = end_prices - start_price
        
        # 5th percentile for 95% confidence
        var = np.percentile(profits, (1 - confidence) * 100)
        
        return var


class RiskEngine:
    """
    Main class that ties everything together.
    
    Workflow:
    1. Load historical price data
    2. Calculate statistics (mean return, volatility)
    3. Run Monte Carlo simulation
    4. Calculate risk metrics
    5. Generate visualizations
    
    I structured it this way so each part can be tested independently.
    """
    
    # Using 4% as risk-free rate (roughly UK gilt rate)
    # Should probably make this configurable...
    RISK_FREE_RATE = 0.04
    
    def __init__(
        self, 
        tickers: List[str],
        n_simulations: int = 1000,
        horizon_days: int = 252,
        use_fake_data: bool = False
    ):
        self.tickers = tickers
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        
        # Initialize the components
        self.data_loader = DataLoader(tickers, '2y', use_fake_data)
        self.simulator = MonteCarloSimulator(n_simulations, horizon_days)
        
        # Store results
        self.simulations = {}  # ticker -> price paths array
        self.metrics = {}      # ticker -> RiskMetrics object
    
    def _get_parameters(self, ticker: str) -> Dict:
        """
        Estimate GBM parameters from historical data.
        
        mu = average daily return * 252 (annualized)
        sigma = std of daily returns * sqrt(252) (annualized)
        
        The sqrt(252) for volatility comes from the fact that variance adds
        over time, so std dev scales with sqrt(time). This tripped me up
        initially - I was just multiplying by 252 for both!
        """
        returns = self.data_loader.returns[ticker]
        prices = self.data_loader.prices[ticker]
        
        daily_mu = returns.mean()
        daily_sigma = returns.std()
        
        return {
            'mu': daily_mu * 252,
            'sigma': daily_sigma * np.sqrt(252),
            'current_price': prices.iloc[-1]
        }
    
    def _calc_sharpe(self, ret: float, vol: float) -> float:
        """
        Sharpe Ratio = (Return - Risk-free rate) / Volatility
        
        Measures return per unit of risk. Higher is better.
        - Below 1: meh
        - 1-2: decent  
        - 2-3: good
        - 3+: either amazing or your data is wrong
        """
        if vol == 0:
            return 0.0
        return (ret - self.RISK_FREE_RATE) / vol
    
    def _calc_max_drawdown(self, prices: pd.Series) -> float:
        """
        Maximum Drawdown = biggest peak-to-trough drop.
        
        If a stock goes £100 -> £150 -> £90 -> £120
        The max drawdown is (90-150)/150 = -40%
        
        This is important because a -50% loss needs a +100% gain to recover!
        Learned this the hard way with some meme stocks...
        """
        running_max = prices.expanding().max()
        drawdowns = (prices - running_max) / running_max
        return drawdowns.min()
    
    def run(self) -> Dict[str, RiskMetrics]:
        """Main analysis function - runs everything."""
        
        print("=" * 55)
        print("  MONTE CARLO RISK ENGINE")
        print("=" * 55)
        print(f"  Simulations: {self.n_simulations:,}")
        print(f"  Horizon: {self.horizon_days} days (~1 year)")
        print(f"  Risk-free rate: {self.RISK_FREE_RATE*100}%")
        print("=" * 55 + "\n")
        
        # Load data
        self.data_loader.get_prices()
        self.data_loader.get_returns()
        
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            
            # Get parameters from historical data
            params = self._get_parameters(ticker)
            
            # Run simulation
            paths = self.simulator.simulate(
                start_price=params['current_price'],
                mu=params['mu'],
                sigma=params['sigma']
            )
            self.simulations[ticker] = paths
            
            # Calculate metrics
            var_95 = self.simulator.calculate_var(paths, 0.95)
            var_99 = self.simulator.calculate_var(paths, 0.99)
            sharpe = self._calc_sharpe(params['mu'], params['sigma'])
            max_dd = self._calc_max_drawdown(self.data_loader.prices[ticker])
            
            self.metrics[ticker] = RiskMetrics(
                ticker=ticker,
                annualized_return=params['mu'],
                annualized_volatility=params['sigma'],
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                var_95=var_95,
                var_99=var_99,
                current_price=params['current_price']
            )
        
        print("\nDone!\n")
        return self.metrics
    
    def print_report(self):
        """Print a summary of all results."""
        print("\n" + "█" * 55)
        print("  RISK ANALYSIS RESULTS")
        print("  Generated:", datetime.now().strftime("%d/%m/%Y %H:%M"))
        print("█" * 55)
        
        for ticker, m in self.metrics.items():
            print(m)
        
        # Comparison table
        print("\n" + "=" * 55)
        print("  COMPARISON")
        print("=" * 55)
        print(f"  {'Ticker':<8} {'Return':>10} {'Vol':>10} {'Sharpe':>8} {'VaR 95%':>12}")
        print("  " + "-" * 51)
        for ticker, m in self.metrics.items():
            print(f"  {ticker:<8} {m.annualized_return*100:>9.1f}% {m.annualized_volatility*100:>9.1f}% {m.sharpe_ratio:>8.2f} £{m.var_95:>10,.0f}")
    
    def plot_simulation(self, ticker: str, save_path: str = None):
        """
        Create a fan chart showing all simulation paths.
        
        The chart shows:
        - All 1000 paths (faded lines)
        - Median path (bold line)
        - 50% and 90% confidence intervals (shaded areas)
        - Current price reference line
        - VaR marker
        
        Spent a lot of time on matplotlib formatting to make this look decent.
        """
        if ticker not in self.simulations:
            raise ValueError(f"No data for {ticker} - run analysis first")
        
        paths = self.simulations[ticker]
        m = self.metrics[ticker]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        days = np.arange(paths.shape[0])
        
        # Plot all paths (very transparent)
        ax.plot(days, paths, color='steelblue', alpha=0.03, linewidth=0.5)
        
        # Calculate percentiles for confidence bands
        p5 = np.percentile(paths, 5, axis=1)
        p25 = np.percentile(paths, 25, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p75 = np.percentile(paths, 75, axis=1)
        p95 = np.percentile(paths, 95, axis=1)
        
        # Shaded confidence regions
        ax.fill_between(days, p5, p95, color='steelblue', alpha=0.2, label='90% CI')
        ax.fill_between(days, p25, p75, color='steelblue', alpha=0.3, label='50% CI')
        
        # Median line
        ax.plot(days, p50, color='darkblue', linewidth=2, label='Median')
        
        # Current price reference
        ax.axhline(y=paths[0, 0], color='gray', linestyle='--', 
                   linewidth=1.5, label=f'Start: £{paths[0, 0]:,.2f}')
        
        # VaR marker
        var_price = paths[0, 0] + m.var_95
        ax.scatter([self.horizon_days], [var_price], color='red', s=80, zorder=5, marker='v')
        ax.annotate(f'95% VaR: £{var_price:,.0f}', 
                    xy=(self.horizon_days, var_price),
                    xytext=(self.horizon_days - 40, var_price - 15),
                    fontsize=9, color='red')
        
        # Labels and formatting
        ax.set_xlabel('Trading Days', fontsize=11)
        ax.set_ylabel('Price (£)', fontsize=11)
        ax.set_title(f'{ticker} - Monte Carlo Simulation\n{self.n_simulations:,} paths, 1-year horizon', 
                     fontsize=12, fontweight='bold')
        
        # Stats box
        stats_text = (
            f'μ (drift): {m.annualized_return*100:.1f}%\n'
            f'σ (vol): {m.annualized_volatility*100:.1f}%\n'
            f'Sharpe: {m.sharpe_ratio:.2f}\n'
            f'Max DD: {m.max_drawdown*100:.1f}%'
        )
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.horizon_days)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_comparison(self, save_path: str = None):
        """Side-by-side comparison of all assets."""
        n = len(self.tickers)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        
        if n == 1:
            axes = [axes]
        
        for ax, ticker in zip(axes, self.tickers):
            paths = self.simulations[ticker]
            m = self.metrics[ticker]
            
            days = np.arange(paths.shape[0])
            
            ax.plot(days, paths, color='steelblue', alpha=0.02, linewidth=0.5)
            
            p50 = np.percentile(paths, 50, axis=1)
            p5 = np.percentile(paths, 5, axis=1)
            p95 = np.percentile(paths, 95, axis=1)
            
            ax.fill_between(days, p5, p95, color='steelblue', alpha=0.2)
            ax.plot(days, p50, color='darkblue', linewidth=2)
            ax.axhline(y=paths[0, 0], color='gray', linestyle='--', linewidth=1)
            
            ax.set_title(f'{ticker}\nVaR(95%): £{m.var_95:,.0f}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price (£)')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        
        plt.suptitle('Monte Carlo Simulations - Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_var_distribution(self, ticker: str, save_path: str = None):
        """
        Histogram of profit/loss outcomes with VaR lines.
        
        This shows the full distribution of where you might end up,
        with the left tail (losses) highlighted.
        """
        paths = self.simulations[ticker]
        m = self.metrics[ticker]
        
        end_prices = paths[-1, :]
        start_price = paths[0, 0]
        pnl = end_prices - start_price
        
        fig, ax = plt.subplots(figsize=(11, 5))
        
        # Histogram
        n, bins, patches = ax.hist(pnl, bins=50, density=True, 
                                    color='steelblue', alpha=0.7, edgecolor='white')
        
        # Color the tail red
        for i, (b, patch) in enumerate(zip(bins, patches)):
            if b < m.var_95:
                patch.set_facecolor('red')
                patch.set_alpha(0.7)
        
        # VaR lines
        ax.axvline(x=m.var_95, color='red', linestyle='--', linewidth=2, 
                   label=f'95% VaR: £{m.var_95:,.0f}')
        ax.axvline(x=m.var_99, color='darkred', linestyle=':', linewidth=2,
                   label=f'99% VaR: £{m.var_99:,.0f}')
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, label='Break-even')
        
        ax.set_xlabel('Profit/Loss (£)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'{ticker} - 1-Year P&L Distribution\n({self.n_simulations:,} simulations)', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    # Stocks I'm interested in analysing
    # TSLA - high volatility, interesting to model
    # F (Ford) - traditional automaker, lower volatility
    # SPY - S&P 500 index, good baseline comparison
    STOCKS = ['TSLA', 'F', 'SPY']
    
    # Simulation settings
    N_SIMS = 1000       # Number of random paths (1000 is a good balance)
    DAYS = 252          # 1 year of trading days
    
    # Set to True if Yahoo Finance isn't working
    USE_TEST_DATA = False
    
    # Create and run the engine
    engine = RiskEngine(
        tickers=STOCKS,
        n_simulations=N_SIMS,
        horizon_days=DAYS,
        use_fake_data=USE_TEST_DATA
    )
    
    # Run analysis
    results = engine.run()
    
    # Print report
    engine.print_report()
    
    # Generate charts
    print("\nGenerating charts...")
    
    for ticker in STOCKS:
        engine.plot_simulation(ticker, f'{ticker}_monte_carlo.png')
        plt.close()
    
    engine.plot_comparison('comparison.png')
    plt.close()
    
    engine.plot_var_distribution('TSLA', 'TSLA_pnl_distribution.png')
    plt.close()
    
    print("\nAll done!")