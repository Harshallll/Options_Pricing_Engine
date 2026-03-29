# Options Pricing Engine

A Python-based options pricing engine implementing Black-Scholes, Binomial Tree, and Monte Carlo models with Greeks, VaR/CVaR portfolio risk analysis, and implied volatility calibration via animated Newton-Raphson convergence using live market data.

---

## Features

- **Black-Scholes Model** — analytical closed-form pricing for European calls and puts
- **Live risk-free rate** — fetched automatically from FRED API (10-year US Treasury yield)
- **Greeks** — Delta, Gamma, Vega, Theta, Rho computed analytically
- **Binomial Tree** — CRR tree pricing for European and American options with backward induction
- **Multi-step Tree** — convergence analysis showing binomial → Black-Scholes as N increases
- **Monte Carlo Simulation** — GBM-based options pricing and correlated portfolio simulation via Cholesky decomposition
- **VaR / CVaR** — Value at Risk and Conditional Value at Risk at 95% confidence on a $10,000 portfolio
- **Implied Volatility Solver** — Newton-Raphson iterative solver with bisection fallback, converges to 6 decimal places
- **Animated Convergence** — GIF animation of Newton-Raphson stepping toward the implied volatility solution

---

## Tech Stack

| Library | Usage |
|---------|-------|
| `numpy` | Numerical computation |
| `scipy` | Normal distribution, statistics |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting and animation |
| `yfinance` | Live market data (ASX + US equities) |
| `fredapi` | Live risk-free rate from FRED (GS10) |

---

## Black-Scholes Model

### Formula

The Black-Scholes model prices a European option under the assumption of constant volatility and log-normally distributed stock returns.

```
d1 = [ ln(S/K) + (r + σ²/2) · T ] / (σ · √T)
d2 = d1 - σ · √T

Call:  C = S · N(d1)  - K · e^(-rT) · N(d2)
Put:   P = K · e^(-rT) · N(-d2) - S · N(-d1)
```

### Parameters

| Parameter | Symbol | Description | Value |
|-----------|--------|-------------|-------|
| Underlying price | S | Current stock price | 30 |
| Strike price | K | Option exercise price | 40 |
| Time to expiry | T | In years (240/365) | 0.6575y |
| Volatility | σ | Annualised standard deviation | 0.30 (30%) |
| Risk-free rate | r | 10-yr US Treasury via FRED | fetched live |

### Risk-Free Rate — FRED API

The risk-free rate is fetched live from the Federal Reserve Economic Data (FRED) database using the `GS10` series — the 10-year US Treasury constant maturity rate. This ensures the model always uses the current market rate rather than a hardcoded assumption.

```python
from fredapi import Fred
fred = Fred(api_key='YOUR_API_KEY')
r = fred.get_series_latest_release('GS10').iloc[-1] / 100
```

### Implementation

```python
def blackScholes(r, S, K, T, sigma, type="C"):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if type == "C":
        price = S*norm.cdf(d1,0,1) - K*np.exp(-r*T)*norm.cdf(d2,0,1)
    elif type == "P":
        price = K*np.exp(-r*T)*norm.cdf(-d2,0,1) - S*norm.cdf(-d1,0,1)
    return price
```

---

## Greeks

The five standard risk sensitivities of an option price:

| Greek | Symbol | Formula (Call) | Interpretation |
|-------|--------|----------------|----------------|
| Delta | Δ | N(d1) | Change in price per $1 move in S |
| Gamma | Γ | N'(d1) / (S·σ·√T) | Rate of change of Delta |
| Vega | ν | S·N'(d1)·√T / 100 | Change per 1% move in volatility |
| Theta | Θ | -(S·N'(d1)·σ / 2√T) - rKe^(-rT)·N(d2) | Time decay per day |
| Rho | ρ | K·T·e^(-rT)·N(d2) / 100 | Change per 1% move in r |

---

## Binomial Tree (CRR)

Discrete-time model building a recombining price tree. At each node the stock moves up by factor `u` or down by `d`:

```
u = e^(σ√Δt)
d = 1/u
p = (e^(rΔt) - d) / (u - d)      ← risk-neutral probability
```

Option value at each node computed by backward induction:

```
V = e^(-rΔt) · [ p·V_up + (1-p)·V_down ]
```

For American options, early exercise is checked at every node:

```
V = max(hold value, intrinsic value)
```

---

## Monte Carlo Simulation

### Options Pricing
Simulates stock price paths under Geometric Brownian Motion:

```
S_T = S_0 · exp[ (r - σ²/2)·T + σ·√T·Z ]     Z ~ N(0,1)

Call payoff: max(S_T - K, 0)
Price = e^(-rT) · mean(payoffs)
```

### Portfolio Simulation
Simulates 400 correlated portfolio paths over 100 days using Cholesky decomposition:

```
Daily returns = μ + L · Z
```

Where `L` is the lower triangular Cholesky factor of the covariance matrix and `Z` is a matrix of uncorrelated standard normals. This preserves the real-world correlation structure between stocks.

---

## VaR and CVaR

Computed from 400 Monte Carlo simulations on a **$10,000 portfolio** over **100 days** at **95% confidence**:

```
VaR  (95%) = Portfolio value - 5th percentile of terminal values
CVaR (95%) = Portfolio value - mean of all values below VaR
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| VaR (95%) | ~$339 | Max loss 95% of the time |
| CVaR (95%) | ~$635 | Average loss in the worst 5% of scenarios |

CVaR is always larger than VaR — the gap between them represents tail risk beyond the VaR threshold.

---

## Implied Volatility — Newton-Raphson

Back-solves for the volatility σ implied by a market-observed option price:

```
σ_new = σ_old - (BS(σ_old) - market_price) / vega(σ_old)
```

| Setting | Value |
|---------|-------|
| Initial guess | Brenner-Subrahmanyam: σ₀ = √(2π/T) · (price/S) |
| Tolerance | 1e-6 |
| Max iterations | 200 |
| Fallback | Bisection method if vega ≈ 0 |
| Accuracy | Recovers true volatility to 6 decimal places |

The convergence is animated as a GIF showing each Newton-Raphson step along the Black-Scholes price curve — visually demonstrating how the algorithm homes in on the solution.

---

## Portfolio

Live price data fetched via `yfinance` for 11 stocks across two markets:

| Market | Tickers |
|--------|---------|
| ASX | CBA.AX, BHP.AX, TLS.AX, NAB.AX, WBC.AX, STO.AX |
| US | AAPL, MSFT, GOOGL, JPM, AMZN |

---

## Installation

```bash
git clone https://github.com/Harshall11/Options_Pricing_Engine.git
cd Options_Pricing_Engine
pip install numpy scipy pandas matplotlib yfinance fredapi
```

Get a free FRED API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) and replace `YOUR_API_KEY` in the notebook.

## Usage

Open `OP.ipynb` in Jupyter or VS Code and run all cells top to bottom.

---

## Key Insight

Black-Scholes assumes constant volatility across all strikes and maturities. The implied volatility solver demonstrates that the market prices options at *different* volatilities depending on strike — the foundation of the **volatility smile** phenomenon that practitioners deal with daily. This is a known limitation of Black-Scholes that more advanced models such as Heston stochastic volatility attempt to address.

---

## Author

Harshal — B.E. ECE, MS Quantitative Finance applicant  
GitHub: [@Harshall11](https://github.com/Harshall11)
