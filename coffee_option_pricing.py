#!/usr/bin/env python3
"""
coffee_option_pricing.py

Prices a 6-month European option on coffee three ways:
  1. Cost of carry model: fair forward price F = S * exp((r + u - y) * T)
  2. Black-Scholes in its commodity form (Black-76 on the cost of carry forward)
  3. Monte Carlo simulation of the risk-neutral price at maturity

Then it stress tests the price for supply, demand, weather and geopolitical
shocks, and adds a jump-diffusion Monte Carlo for sudden weather events.

Notation
  S  spot price ($/lb)          r  risk-free rate (continuous, per year)
  u  storage cost (per year)    y  convenience yield (per year, 0 in base case)
  b  net cost of carry = r + u - y
  K  strike                     T  time to maturity (years)
  sigma  volatility

Usage:   python coffee_option_pricing.py
Needs:   numpy, matplotlib (matplotlib only for the charts)
Outputs: printed report, results.json, scenarios.csv, sensitivity_*.csv, *.png
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace

import numpy as np

CONTRACT_SIZE_LB = 37_500  # ICE Coffee "C" contract size in pounds
SEED = 20261012


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Market:
    spot: float = 1.20       # S, $/lb
    rate: float = 0.02       # r, continuously compounded
    storage: float = 0.01    # u, storage cost as a % of value per year
    conv_yield: float = 0.0  # y, benefit of holding physical coffee
    maturity: float = 0.5    # T, years
    strike: float = 1.25     # K, $/lb
    vol: float = 0.25        # sigma

    @property
    def carry(self) -> float:
        """Net cost of carry b = r + u - y."""
        return self.rate + self.storage - self.conv_yield


BASE = Market()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------
# 1. Cost of carry
# --------------------------------------------------------------------------
def forward_price(m: Market) -> float:
    """F = S * exp(b * T), with b = r + u - y."""
    return m.spot * math.exp(m.carry * m.maturity)


# --------------------------------------------------------------------------
# 2. Black-Scholes (commodity form = Black-76 on the forward)
# --------------------------------------------------------------------------
def black_scholes(m: Market, kind: str = "call") -> dict:
    """
    Price = exp(-rT) * [F N(d1) - K N(d2)]            (call)
          = exp(-rT) * [K N(-d2) - F N(-d1)]          (put)
    d1 = [ln(F/K) + 0.5 sigma^2 T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)

    This is the generalized Black-Scholes-Merton model with carry b. Setting
    storage = 0 and conv_yield = 0 gives the textbook Black-Scholes formula.
    """
    F = forward_price(m)
    K, T, s, r = m.strike, m.maturity, m.vol, m.rate
    sd = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    disc = math.exp(-r * T)
    carry_disc = math.exp((m.carry - r) * T)  # exp((b - r) T)

    if kind == "call":
        price = disc * (F * norm_cdf(d1) - K * norm_cdf(d2))
        delta = carry_disc * norm_cdf(d1)
    elif kind == "put":
        price = disc * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
        delta = carry_disc * (norm_cdf(d1) - 1.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    gamma = carry_disc * norm_pdf(d1) / (m.spot * sd)
    vega = m.spot * carry_disc * norm_pdf(d1) * math.sqrt(T) / 100.0  # per 1 vol pt

    # Theta: one calendar day of time decay, by finite difference
    one_day = 1.0 / 365.0
    shorter = replace(m, maturity=T - one_day)
    price_1d = _bs_price_only(shorter, kind)
    theta = price_1d - price

    return {
        "price": price, "d1": d1, "d2": d2, "N_d1": norm_cdf(d1), "N_d2": norm_cdf(d2),
        "delta": delta, "gamma": gamma, "vega_per_vol_pt": vega, "theta_per_day": theta,
    }


def _bs_price_only(m: Market, kind: str) -> float:
    F = forward_price(m)
    sd = m.vol * math.sqrt(m.maturity)
    d1 = (math.log(F / m.strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    disc = math.exp(-m.rate * m.maturity)
    if kind == "call":
        return disc * (F * norm_cdf(d1) - m.strike * norm_cdf(d2))
    return disc * (m.strike * norm_cdf(-d2) - F * norm_cdf(-d1))


# --------------------------------------------------------------------------
# 3. Monte Carlo (geometric Brownian motion, risk-neutral drift = b)
# --------------------------------------------------------------------------
def monte_carlo(m: Market, n_paths: int = 1_000_000, seed: int = SEED) -> dict:
    """
    S_T = S * exp((b - 0.5 sigma^2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
    The option is European, so simulating S_T directly is exact (no time
    stepping error). Antithetic variates (Z and -Z) cut the variance.
    Standard error is computed on the antithetic pair averages.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z])
    T = m.maturity
    st = m.spot * np.exp((m.carry - 0.5 * m.vol ** 2) * T + m.vol * math.sqrt(T) * z)
    disc = math.exp(-m.rate * T)

    out = {"n_paths": 2 * half, "mean_ST": float(st.mean())}
    for kind, payoff in (("call", np.maximum(st - m.strike, 0.0)),
                         ("put", np.maximum(m.strike - st, 0.0))):
        pv = disc * payoff
        pairs = 0.5 * (pv[:half] + pv[half:])
        price = float(pairs.mean())
        se = float(pairs.std(ddof=1) / math.sqrt(half))
        out[kind] = {"price": price, "std_error": se,
                     "ci95": [price - 1.96 * se, price + 1.96 * se]}
    out["prob_ITM_call"] = float((st > m.strike).mean())
    out["_terminal"] = st  # kept for charting, dropped before saving
    return out


def monte_carlo_paths(m: Market, n_paths: int = 50, n_steps: int = 126,
                      seed: int = SEED) -> np.ndarray:
    """Daily-step price paths, only used to draw the fan chart."""
    rng = np.random.default_rng(seed)
    dt = m.maturity / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = (m.carry - 0.5 * m.vol ** 2) * dt + m.vol * math.sqrt(dt) * z
    logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return m.spot * np.exp(logp)


# --------------------------------------------------------------------------
# 4. Weather jumps: Merton jump-diffusion
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Jumps:
    """
    Sudden weather events (frost, drought) as Poisson jumps in log price.
    Illustrative assumptions, not calibrated:
      lam    = expected number of major weather shocks per year
      mu_j   = mean log jump size (positive: shocks cut supply)
      sig_j  = std dev of log jump size
    """
    lam: float = 0.40
    mu_j: float = 0.15
    sig_j: float = 0.10

    @property
    def k(self) -> float:
        """Expected proportional jump E[e^J] - 1."""
        return math.exp(self.mu_j + 0.5 * self.sig_j ** 2) - 1.0


def monte_carlo_jump(m: Market, j: Jumps, n_paths: int = 1_000_000,
                     seed: int = SEED) -> dict:
    """
    S_T = S exp((b - lam k - 0.5 sigma^2) T + sigma sqrt(T) Z + sum of jumps)
    The -lam*k term keeps E[S_T] = F, so the forward is unchanged and only
    the shape of the distribution changes (fatter right tail).
    """
    rng = np.random.default_rng(seed)
    T = m.maturity
    z = rng.standard_normal(n_paths)
    n_jumps = rng.poisson(j.lam * T, n_paths)
    jump_sum = n_jumps * j.mu_j + np.sqrt(n_jumps) * j.sig_j * rng.standard_normal(n_paths)
    drift = (m.carry - j.lam * j.k - 0.5 * m.vol ** 2) * T
    st = m.spot * np.exp(drift + m.vol * math.sqrt(T) * z + jump_sum)
    disc = math.exp(-m.rate * T)
    pv = disc * np.maximum(st - m.strike, 0.0)
    price = float(pv.mean())
    se = float(pv.std(ddof=1) / math.sqrt(n_paths))
    return {"price": price, "std_error": se, "mean_ST": float(st.mean()),
            "prob_ITM": float((st > m.strike).mean())}


def merton_closed_form(m: Market, j: Jumps, n_terms: int = 60) -> float:
    """
    Merton (1976) series, used to validate the jump Monte Carlo.
    Given n jumps, ln S_T is normal with variance sigma^2 T + n sig_j^2 and
    E[S_T | n] = S exp(b_n T), b_n = b - lam k + n ln(1 + k) / T.
    So price = sum over n of P(n jumps) * exp(-rT) * Black(F_n, K, sigma_n),
    with P(n jumps) Poisson(lam T). (Hull's version weights by lam(1+k) and
    discounts each term at its own rate; the two are algebraically equal.)
    """
    T = m.maturity
    total = 0.0
    for n in range(n_terms):
        weight = math.exp(-j.lam * T) * (j.lam * T) ** n / math.factorial(n)
        vol_n = math.sqrt(m.vol ** 2 + n * j.sig_j ** 2 / T)
        carry_n = m.carry - j.lam * j.k + n * math.log(1.0 + j.k) / T
        # express carry_n through the storage field so forward_price picks it up
        m_n = replace(m, vol=vol_n, storage=carry_n - m.rate + m.conv_yield)
        total += weight * _bs_price_only(m_n, "call")
    return total


# --------------------------------------------------------------------------
# 5. Scenarios: supply, demand, weather, geopolitics
# --------------------------------------------------------------------------
# Each scenario moves the model inputs through the channel the event works on.
#   spot level -> where prices are today
#   vol        -> how uncertain the next 6 months are
#   storage    -> physical carry costs (warehousing, shipping, insurance)
#   conv_yield -> value of holding physical beans when stocks are tight
#   rate       -> financing cost
# Sizes are illustrative, chosen to be in a plausible range for coffee.
SCENARIOS = [
    ("Base case", "Given inputs", {}),
    ("Supply: bumper Brazil harvest", "Spot -15%, vol 20%",
     dict(spot=BASE.spot * 0.85, vol=0.20)),
    ("Supply: poor harvest, tight stocks", "Spot +15%, vol 30%, conv. yield 4%",
     dict(spot=BASE.spot * 1.15, vol=0.30, conv_yield=0.04)),
    ("Demand: strong consumption growth", "Spot +8%",
     dict(spot=BASE.spot * 1.08)),
    ("Demand: recession cuts consumption", "Spot -10%, vol 30%",
     dict(spot=BASE.spot * 0.90, vol=0.30)),
    ("Weather: frost hits Brazil", "Spot +25%, vol 45%, conv. yield 5%",
     dict(spot=BASE.spot * 1.25, vol=0.45, conv_yield=0.05)),
    ("Weather: El Nino risk priced, no move yet", "Vol 35% only",
     dict(vol=0.35)),
    ("Geopolitics: shipping disruption", "Spot +5%, vol 32%, storage 3%",
     dict(spot=BASE.spot * 1.05, vol=0.32, storage=0.03)),
    ("Geopolitics: tariffs / export curbs", "Spot +10%, vol 35%",
     dict(spot=BASE.spot * 1.10, vol=0.35)),
    ("Macro: rates rise to 4%", "r = 4%",
     dict(rate=0.04)),
]


def run_scenarios() -> list[dict]:
    base_call = black_scholes(BASE, "call")["price"]
    rows = []
    for name, shock, changes in SCENARIOS:
        m = replace(BASE, **changes)
        c = black_scholes(m, "call")
        p = black_scholes(m, "put")
        mc = monte_carlo(m, n_paths=400_000)
        rows.append({
            "scenario": name, "shock": shock,
            "spot": m.spot, "vol": m.vol, "storage": m.storage,
            "conv_yield": m.conv_yield, "rate": m.rate,
            "forward": forward_price(m),
            "call_bs": c["price"], "call_mc": mc["call"]["price"],
            "call_mc_se": mc["call"]["std_error"],
            "put_bs": p["price"], "call_delta": c["delta"],
            "call_change_vs_base": c["price"] - base_call,
            "call_change_pct": (c["price"] / base_call - 1.0) * 100.0,
            "call_per_contract": c["price"] * CONTRACT_SIZE_LB,
        })
    return rows


# --------------------------------------------------------------------------
# 6. Sensitivity grids
# --------------------------------------------------------------------------
def sensitivity_spot() -> list[dict]:
    rows = []
    for s in np.round(np.arange(0.90, 1.6001, 0.05), 2):
        m = replace(BASE, spot=float(s))
        rows.append({"spot": float(s), "call": _bs_price_only(m, "call"),
                     "put": _bs_price_only(m, "put")})
    return rows


def sensitivity_vol() -> list[dict]:
    rows = []
    for v in np.round(np.arange(0.10, 0.6001, 0.05), 2):
        m = replace(BASE, vol=float(v))
        rows.append({"vol": float(v), "call": _bs_price_only(m, "call"),
                     "put": _bs_price_only(m, "put")})
    return rows


def sensitivity_carry() -> list[dict]:
    """Net carry b from strong backwardation to strong contango."""
    rows = []
    for y in np.round(np.arange(-0.04, 0.0801, 0.02), 2):
        m = replace(BASE, conv_yield=float(y))
        rows.append({"conv_yield": float(y), "net_carry": m.carry,
                     "forward": forward_price(m),
                     "call": _bs_price_only(m, "call")})
    return rows


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def make_charts(mc_terminal: np.ndarray, spot_rows, vol_rows, scen_rows) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping charts")
        return []

    ink, muted, accent, accent2 = "#1f2933", "#7b8794", "#2f6f9f", "#c2703d"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    files = []

    # Simulated paths
    paths = monte_carlo_paths(BASE)
    t = np.linspace(0, BASE.maturity * 12, paths.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, paths.T, color=accent, alpha=0.25, lw=0.8)
    ax.axhline(BASE.strike, color=accent2, lw=1.5, ls="--", label=f"Strike ${BASE.strike:.2f}")
    ax.axhline(forward_price(BASE), color=ink, lw=1.2, label=f"Forward ${forward_price(BASE):.4f}")
    ax.set_xlabel("Months"); ax.set_ylabel("Coffee price ($/lb)")
    ax.set_title("50 simulated risk-neutral price paths", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("chart_paths.png", dpi=160); plt.close(fig)
    files.append("chart_paths.png")

    # Terminal distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mc_terminal, bins=200, color=accent, alpha=0.8, density=True)
    ax.axvline(BASE.strike, color=accent2, lw=1.5, ls="--", label="Strike")
    ax.set_xlim(0.6, 2.2)
    ax.set_xlabel("Price at maturity ($/lb)"); ax.set_ylabel("Density")
    ax.set_title("Monte Carlo distribution of the price in 6 months", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("chart_distribution.png", dpi=160); plt.close(fig)
    files.append("chart_distribution.png")

    # Sensitivity to spot and vol
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot([r["spot"] for r in spot_rows], [r["call"] for r in spot_rows], color=accent, lw=2, label="Call")
    axes[0].plot([r["spot"] for r in spot_rows], [r["put"] for r in spot_rows], color=accent2, lw=2, label="Put")
    axes[0].axvline(BASE.spot, color=muted, lw=1, ls=":")
    axes[0].set_xlabel("Spot ($/lb)"); axes[0].set_ylabel("Option value ($/lb)")
    axes[0].set_title("Value vs spot", loc="left"); axes[0].legend(frameon=False)
    axes[1].plot([r["vol"] * 100 for r in vol_rows], [r["call"] for r in vol_rows], color=accent, lw=2, label="Call")
    axes[1].plot([r["vol"] * 100 for r in vol_rows], [r["put"] for r in vol_rows], color=accent2, lw=2, label="Put")
    axes[1].axvline(BASE.vol * 100, color=muted, lw=1, ls=":")
    axes[1].set_xlabel("Volatility (%)")
    axes[1].set_title("Value vs volatility", loc="left"); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig("chart_sensitivity.png", dpi=160); plt.close(fig)
    files.append("chart_sensitivity.png")

    # Scenario bars
    names = [r["scenario"] for r in scen_rows][::-1]
    vals = [r["call_bs"] for r in scen_rows][::-1]
    base_val = scen_rows[0]["call_bs"]
    colors = [accent if v >= base_val else accent2 for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(names, vals, color=colors)
    ax.axvline(base_val, color=ink, lw=1, ls="--")
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"${v:.4f}", va="center", fontsize=8, color=ink)
    ax.set_xlabel("Call value ($/lb)")
    ax.set_title("Call value under each scenario", loc="left")
    fig.tight_layout(); fig.savefig("chart_scenarios.png", dpi=160); plt.close(fig)
    files.append("chart_scenarios.png")
    return files


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    m = BASE
    F = forward_price(m)
    call = black_scholes(m, "call")
    put = black_scholes(m, "put")
    plain = replace(m, storage=0.0)  # textbook BS, ignores storage
    call_plain = black_scholes(plain, "call")
    put_plain = black_scholes(plain, "put")
    mc = monte_carlo(m)
    terminal = mc.pop("_terminal")

    # Validation checks
    parity_lhs = call["price"] - put["price"]
    parity_rhs = math.exp(-m.rate * m.maturity) * (F - m.strike)
    conv = []
    for n in (1_000, 10_000, 100_000, 1_000_000):
        r = monte_carlo(m, n_paths=n)
        conv.append({"n_paths": n, "call": r["call"]["price"],
                     "std_error": r["call"]["std_error"],
                     "error_vs_bs": r["call"]["price"] - call["price"]})

    jumps = Jumps()
    jmc = monte_carlo_jump(m, jumps)
    jcf = merton_closed_form(m, jumps)

    scen = run_scenarios()
    s_spot, s_vol, s_carry = sensitivity_spot(), sensitivity_vol(), sensitivity_carry()

    # ---- Print report ----
    line = "-" * 68
    print(line)
    print("COFFEE OPTION PRICING  (all prices in $ per lb unless noted)")
    print(line)
    print(f"Inputs: S={m.spot}  K={m.strike}  r={m.rate:.0%}  u={m.storage:.0%}  "
          f"T={m.maturity}  sigma={m.vol:.0%}")
    print(f"Net carry b = r + u = {m.carry:.2%}")
    print()
    print("1) COST OF CARRY")
    print(f"   F = S * exp((r+u)T) = {m.spot} * exp({m.carry} * {m.maturity}) = {F:.6f}")
    print(f"   Forward premium over spot: {F - m.spot:.6f} ({(F / m.spot - 1):.4%})")
    print(f"   Forward vs strike: {F - m.strike:+.6f}  -> call is out of the money")
    print()
    print("2) BLACK-SCHOLES (commodity form, on the forward)")
    print(f"   d1 = {call['d1']:.6f}   d2 = {call['d2']:.6f}")
    print(f"   N(d1) = {call['N_d1']:.6f}   N(d2) = {call['N_d2']:.6f}")
    print(f"   Call = {call['price']:.6f}   (${call['price'] * CONTRACT_SIZE_LB:,.2f} per 37,500 lb contract)")
    print(f"   Put  = {put['price']:.6f}   (${put['price'] * CONTRACT_SIZE_LB:,.2f} per contract)")
    print(f"   Textbook BS ignoring storage: call {call_plain['price']:.6f}, put {put_plain['price']:.6f}")
    print(f"   Storage adds {call['price'] - call_plain['price']:+.6f} to the call")
    print(f"   Greeks (call): delta {call['delta']:.4f}  gamma {call['gamma']:.4f}  "
          f"vega {call['vega_per_vol_pt']:.6f}/vol pt  theta {call['theta_per_day']:.6f}/day")
    print(f"   Put-call parity: C - P = {parity_lhs:.8f}, e^-rT (F-K) = {parity_rhs:.8f}")
    print()
    print("3) MONTE CARLO (1,000,000 paths, antithetic)")
    print(f"   Call = {mc['call']['price']:.6f}  SE {mc['call']['std_error']:.6f}  "
          f"95% CI [{mc['call']['ci95'][0]:.6f}, {mc['call']['ci95'][1]:.6f}]")
    print(f"   Put  = {mc['put']['price']:.6f}  SE {mc['put']['std_error']:.6f}")
    print(f"   Mean simulated S_T = {mc['mean_ST']:.6f} (should equal F = {F:.6f})")
    print(f"   Risk-neutral P(S_T > K) = {mc['prob_ITM_call']:.4f}  (N(d2) = {call['N_d2']:.4f})")
    print("   Convergence:")
    for c in conv:
        print(f"     n={c['n_paths']:>9,}  call={c['call']:.6f}  SE={c['std_error']:.6f}  "
              f"error={c['error_vs_bs']:+.6f}")
    print()
    print("4) WEATHER JUMPS (Merton jump-diffusion)")
    print(f"   lam={jumps.lam}/yr  mean jump={jumps.mu_j:.0%}  jump vol={jumps.sig_j:.0%}")
    print(f"   Call MC = {jmc['price']:.6f}  SE {jmc['std_error']:.6f}   closed form = {jcf:.6f}")
    print(f"   Mean S_T = {jmc['mean_ST']:.6f} (forward unchanged)   "
          f"Jump premium vs plain BS = {jcf - call['price']:+.6f}")
    print()
    print("5) SCENARIOS (call)")
    print(f"   {'Scenario':<42}{'Fwd':>8}{'Call':>9}{'Chg %':>9}{'Delta':>8}")
    for r in scen:
        print(f"   {r['scenario']:<42}{r['forward']:>8.4f}{r['call_bs']:>9.4f}"
              f"{r['call_change_pct']:>+9.1f}{r['call_delta']:>8.3f}")
    print(line)

    # ---- Save ----
    results = {
        "inputs": m.__dict__ | {"net_carry": m.carry},
        "cost_of_carry": {"forward": F},
        "black_scholes": {"call": call, "put": put,
                          "call_ignoring_storage": call_plain["price"],
                          "put_ignoring_storage": put_plain["price"],
                          "per_contract_call": call["price"] * CONTRACT_SIZE_LB,
                          "per_contract_put": put["price"] * CONTRACT_SIZE_LB},
        "put_call_parity": {"c_minus_p": parity_lhs, "disc_f_minus_k": parity_rhs},
        "monte_carlo": mc, "mc_convergence": conv,
        "jump_diffusion": {"params": jumps.__dict__ | {"k": jumps.k},
                           "mc": jmc, "closed_form": jcf},
        "scenarios": scen,
        "sensitivity_carry": s_carry,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    write_csv("scenarios.csv", scen)
    write_csv("sensitivity_spot.csv", s_spot)
    write_csv("sensitivity_vol.csv", s_vol)
    write_csv("sensitivity_carry.csv", s_carry)
    files = make_charts(terminal, s_spot, s_vol, scen)
    print("Saved: results.json, scenarios.csv, sensitivity_*.csv" + (", " + ", ".join(files) if files else ""))


if __name__ == "__main__":
    main()