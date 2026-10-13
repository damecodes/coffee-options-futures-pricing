# Coffee Option Pricing

Prices a 6-month European option on coffee three ways: the cost of carry model, Black-Scholes and Monte Carlo simulation. It then stress tests the price against supply, demand, weather and geopolitical shocks.

## Inputs

| Input | Value |
| --- | --- |
| Spot price (S) | $1.20 per lb |
| Strike (K) | $1.25 per lb |
| Risk-free rate (r) | 2% per year |
| Storage cost (u) | 1% per year |
| Time to maturity (T) | 0.5 years |
| Volatility (σ) | 25% |

To change them, edit the `Market` dataclass at the top of `coffee_option_pricing.py`.

## How to run

```bash
pip install numpy matplotlib
python coffee_option_pricing.py
```

matplotlib is only used for the charts. The script still runs without it.

## What it does

1. **Cost of carry.** Computes the fair forward price, F = S × e^((r + u − y)T). The convenience yield y is 0 in the base case.
2. **Black-Scholes.** Prices the call and put on the forward (the Black-76 form), with Greeks.
3. **Monte Carlo.** Simulates 1,000,000 prices at maturity with antithetic variates and reports the standard error and a 95% confidence interval.
4. **Weather jumps.** A Merton jump-diffusion model adds sudden frost or drought shocks. A closed-form Merton price checks the simulation.
5. **Scenarios.** Reprices the option under 10 market scenarios: harvest size, demand, frost, El Niño, shipping disruption, tariffs and higher rates.

The script also runs three checks: put-call parity, the Monte Carlo price against the closed form, and the mean simulated price against the forward.

## Results (base case)

| | Call | Put |
| --- | --- | --- |
| Forward price | $1.2181 | |
| Black-Scholes | $0.0712 | $0.1027 |
| Monte Carlo | $0.0711 (SE 0.0001) | $0.1027 |
| Per 37,500 lb contract | $2,670 | $3,853 |

## Outputs

| File | Contents |
| --- | --- |
| `results.json` | All prices, Greeks, checks and scenario results |
| `scenarios.csv` | Call and put values under each scenario |
| `sensitivity_spot.csv`, `sensitivity_vol.csv`, `sensitivity_carry.csv` | Option value as spot, volatility and carry change |
| `chart_paths.png` | 50 simulated price paths |
| `chart_distribution.png` | Distribution of the price at maturity |
| `chart_sensitivity.png` | Option value vs spot and volatility |
| `chart_scenarios.png` | Call value under each scenario |

## Limitations

- Volatility is constant. Real coffee volatility is seasonal and rises with weather risk.
- The option is European. Exchange-traded coffee options are American-style options on futures.
- Scenario shock sizes and jump parameters are illustrative assumptions, not calibrated to market data.
