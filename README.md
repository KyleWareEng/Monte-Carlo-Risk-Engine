# Monte Carlo Risk Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Monte Carlo simulation tool for financial risk analysis using Geometric Brownian Motion to calculate Value at Risk (VaR) and other portfolio risk metrics.

![TSLA Monte Carlo](TSLA_monte_carlo.png)

---

## Why I Built This

I'm studying Automotive Engineering but got interested in quantitative finance after learning how hedge funds use mathematical models. It turns out a lot of the maths overlaps with engineering, stochastic processes, differential equations, probability distributions. So I decided to build something practical.

The goal was to answer: **"If I buy this stock today, how much could I lose over the next year?"**

---

## What It Does

1. Downloads historical stock data from Yahoo Finance
2. Calculates statistics (average return, volatility)
3. Simulates 1,000 possible future price paths using Monte Carlo
4. Calculates risk metrics like VaR (Value at Risk) and Sharpe Ratio
5. Creates visualisations with confidence intervals

---

## Results

### Sample Analysis Output

| Metric | TSLA | F | SPY |
|--------|------|---|-----|
| Annual Return | 41.0% | 8.2% | 12.3% |
| Annual Volatility | 62.3% | 45.1% | 18.2% |
| Sharpe Ratio | 0.59 | 0.11 | 0.67 |
| Max Drawdown | -53.8% | -62.4% | -24.5% |
| 95% VaR (1-Year) | -$238 | -$8.50 | -$45 |
| 99% VaR (1-Year) | -$319 | -$10.20 | -$62 |

**Interpretation:** With 95% confidence, a single TSLA share won't lose more than $238 over the next year. However, there's a 5% chance losses could exceed this threshold.

### Simulation Visualisation

The script generates fan charts showing all 1,000 simulated price paths. The blue shaded regions show confidence intervals — 50% of paths fall within the dark band, 90% within the light band.

---

## The Maths

I use **Geometric Brownian Motion** to simulate stock prices. The basic idea:

```
Tomorrow's price = Today's price × exp(drift + randomness)
```

Or more formally:

$$S(t+dt) = S(t) \times \exp\left((\mu - \frac{\sigma^2}{2})dt + \sigma\sqrt{dt} \times Z\right)$$

Where:
- `μ` = average return (the trend)
- `σ` = volatility (how much it fluctuates)
- `Z` = random number from normal distribution

The `(μ - σ²/2)` term is called "Ito's correction" — a mathematical adjustment that accounts for the difference between arithmetic and geometric means when dealing with continuous random processes.

### Why 1,000 Simulations?

More simulations = better estimate of the probability distribution. I tested 100 (too noisy), 10,000 (computationally expensive), and settled on 1,000 as an optimal balance between accuracy and performance.

---

## Risk Metrics Explained

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Value at Risk (VaR)** | Maximum expected loss at a given confidence level | 95% VaR = -$50 means you're 95% confident you won't lose more than $50 |
| **Sharpe Ratio** | Return per unit of risk: `(Return - Risk-free rate) / Volatility` | Below 1 is weak, 1-2 is good, above 2 is excellent |
| **Maximum Drawdown** | Largest peak-to-trough drop historically | Critical because a -50% loss requires a +100% gain to recover |

---

## How to Run

### Installation

```bash
# Clone the repository
git clone https://github.com/KyleWareEng/Monte-Carlo-Risk-Engine.git
cd Monte-Carlo-Risk-Engine

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python monte_carlo_risk_engine.py
```

### Configuration

You can change the stocks being analysed at the bottom of the script:

```python
STOCKS = ['TSLA', 'F', 'SPY']  # Add/remove tickers here
```

---

## Project Structure

```
Monte-Carlo-Risk-Engine/
├── monte_carlo_risk_engine.py   # Main simulation script
├── requirements.txt             # Dependencies
├── README.md
│
└── outputs/
    ├── TSLA_monte_carlo.png     # Generated visualisation
    ├── F_monte_carlo.png
    └── SPY_monte_carlo.png
```

---

## Technical Details

### Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.28
matplotlib>=3.7.0
```

### Key Implementation Details

- **Log returns** used instead of simple returns (standard practice in quantitative finance)
- **Vectorised NumPy operations** for efficient simulation of 1,000 paths
- **Yahoo Finance API** for historical price data (note: occasionally unreliable)

---

## Limitations & Future Work

### Known Limitations

1. **Constant volatility assumption** - Real markets exhibit volatility clustering (large moves follow large moves). A GARCH model would address this.

2. **No correlation modelling** - Currently treats each stock independently. In reality, assets are correlated, especially during market crashes.

3. **Historical parameters** - Uses past data to predict the future. As they say, past performance doesn't guarantee future results.

4. **No transaction costs** - Real trading involves fees, spreads, and slippage.

### Planned Improvements

- [ ] Portfolio optimisation (Markowitz mean-variance)
- [ ] Correlation matrix between assets
- [ ] Interactive dashboard with Streamlit
- [ ] Backtest VaR predictions against actual historical losses
- [ ] GARCH model for dynamic volatility

---

## What I Learned

- How to structure a Python project properly (classes, separation of concerns)
- The mathematics behind option pricing and risk management
- Why quants use log returns instead of simple returns
- Practical matplotlib skills for financial visualisations
- Working with financial data APIs

---

## Resources

- Hull, J.C. - *Options, Futures and Other Derivatives* (Chapter 14)
- QuantPy YouTube channel
- Wikipedia: Geometric Brownian Motion
- Wilmott, P. - *Paul Wilmott Introduces Quantitative Finance*

---

## Contact

**Kyle Ware**

- Email: kyle.ware@outlook.com
- LinkedIn: [linkedin.com/in/kyleaware](https://linkedin.com/in/kyleaware)
- MEng Automotive Engineering

Feel free to reach out if you have questions about this project!

---

*Built with Python | January 2026*
Built by Kyle W | MEng Automotive Engineering | January 2026
