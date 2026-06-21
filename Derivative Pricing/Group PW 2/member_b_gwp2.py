"""Member B implementation for Group PW2
Provides:
- Black-Scholes pricing (call/put)
- Implied volatility via bisection
- Monte-Carlo pricing for European and up-and-out barrier options (GBM)
- Finite-difference Greeks (delta, vega, theta) using MC for exotic, analytic for BS European
- A demo runner when executed as a script

Usage:
    python member_b_gwp2.py --demo

Outputs: prints results and writes plots to ./outputs
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Black-Scholes ---
from math import erf, log, sqrt, exp


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    # S: spot, K: strike, T: time to maturity (years), r: risk-free rate, sigma: vol, q: dividend yield
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # CDF of standard normal
    def Phi(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    if option_type == "call":
        return S * np.exp(-q * T) * Phi(d1) - K * np.exp(-r * T) * Phi(d2)
    else:
        return K * np.exp(-r * T) * Phi(-d2) - S * np.exp(-q * T) * Phi(-d1)


def implied_vol_bisect(price, S, K, T, r, option_type="call", q=0.0, tol=1e-6, maxiter=100):
    # Bisection between vol_low and vol_high
    if price <= 0:
        return 0.0
    vol_low = 1e-6
    vol_high = 5.0
    f_low = bs_price(S, K, T, r, vol_low, option_type, q) - price
    f_high = bs_price(S, K, T, r, vol_high, option_type, q) - price
    if f_low * f_high > 0:
        # No sign change -- return nan
        return np.nan
    for i in range(maxiter):
        mid = 0.5 * (vol_low + vol_high)
        f_mid = bs_price(S, K, T, r, mid, option_type, q) - price
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid <= 0:
            vol_high = mid
            f_high = f_mid
        else:
            vol_low = mid
            f_low = f_mid
    return 0.5 * (vol_low + vol_high)


# --- Monte Carlo ---

def simulate_gbm(S0, r, sigma, T, n_paths, n_steps, q=0.0, seed=None, antithetic=True):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])
        if n_paths % 2 == 1:
            z = np.vstack([z, rng.standard_normal((1, n_steps))])
    else:
        z = rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum(drift + vol * z, axis=1)
    S = np.exp(np.hstack([np.full((n_paths, 1), np.log(S0)), logS]))
    # S shape: (n_paths, n_steps+1)
    return S


def mc_price_european(S0, K, T, r, sigma, option_type="call", n_paths=20000, n_steps=100, seed=42):
    S = simulate_gbm(S0, r, sigma, T, n_paths, n_steps, seed=seed)
    ST = S[:, -1]
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price, payoffs


def mc_price_up_and_out_barrier(S0, K, H, T, r, sigma, option_type="call", n_paths=20000, n_steps=100, seed=42):
    S = simulate_gbm(S0, r, sigma, T, n_paths, n_steps, seed=seed)
    maxS = S.max(axis=1)
    ST = S[:, -1]
    knocked = maxS >= H
    if option_type == "call":
        payoffs = np.where(knocked, 0.0, np.maximum(ST - K, 0.0))
    else:
        payoffs = np.where(knocked, 0.0, np.maximum(K - ST, 0.0))
    price = np.exp(-r * T) * np.mean(payoffs)
    return price, payoffs, knocked


# --- Greeks (finite differences) ---

def finite_diff_greeks_mc(price_func, bump, *args, **kwargs):
    # price_func: callable that accepts changed S0 or sigma via kwargs
    base = price_func(*args, **kwargs)
    # Delta: bump S0
    S0 = kwargs.get("S0")
    if S0 is None:
        raise ValueError("S0 must be provided in kwargs for delta calculation")
    kwargs_up = dict(kwargs)
    kwargs_up["S0"] = S0 + bump
    kwargs_down = dict(kwargs)
    kwargs_down["S0"] = max(1e-12, S0 - bump)

    price_up = price_func(*args, **kwargs_up)
    price_down = price_func(*args, **kwargs_down)
    delta = (price_up - price_down) / (2 * bump)

    # Vega: bump sigma
    sigma = kwargs.get("sigma")
    if sigma is None:
        raise ValueError("sigma must be provided in kwargs for vega calculation")
    kwargs_v_up = dict(kwargs)
    kwargs_v_up["sigma"] = sigma + bump
    kwargs_v_down = dict(kwargs)
    kwargs_v_down["sigma"] = max(1e-12, sigma - bump)
    v_up = price_func(*args, **kwargs_v_up)
    v_down = price_func(*args, **kwargs_v_down)
    vega = (v_up - v_down) / (2 * bump)

    # Theta: bump T (use forward difference)
    T = kwargs.get("T")
    if T is None:
        raise ValueError("T must be provided in kwargs for theta calculation")
    kwargs_t_up = dict(kwargs)
    kwargs_t_up["T"] = max(1e-12, T + bump)
    t_up = price_func(*args, **kwargs_t_up)
    theta = (t_up - base) / bump

    return {"delta": delta, "vega": vega, "theta": theta}


# small adapters for price functions to return scalar price

def price_barrier_adapter(*_args, **_kwargs):
    price, *_ = mc_price_up_and_out_barrier(*_args, **_kwargs)
    return price


def price_european_adapter(*_args, **_kwargs):
    price, *_ = mc_price_european(*_args, **_kwargs)
    return price


# --- Demo / CLI ---

def run_demo(outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    # parameters
    S0 = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.2
    T = 1.0
    H = 140.0

    print("Running Member B demo: BS pricing, implied vol, MC pricing, plots")
    bs_c = bs_price(S0, K, T, r, sigma, option_type="call")
    print(f"BS call price (sigma={sigma:.3f}): {bs_c:.6f}")

    # implied vol from the BS price
    iv = implied_vol_bisect(bs_c, S0, K, T, r, option_type="call")
    print(f"Implied vol recovered: {iv:.6f}")

    # MC European
    mc_price, payoffs = mc_price_european(S0, K, T, r, sigma, option_type="call", n_paths=20000, n_steps=100, seed=2025)
    print(f"MC European call price: {mc_price:.6f}")

    # MC barrier
    barrier_price, bp, knocked = mc_price_up_and_out_barrier(S0, K, H, T, r, sigma, option_type="call", n_paths=20000, n_steps=100, seed=2025)
    print(f"MC up-and-out barrier call price (H={H}): {barrier_price:.6f}")

    # Greeks via finite diff for barrier
    gd = finite_diff_greeks_mc(price_barrier_adapter, 0.5, S0, K, H, T, r, sigma, "call", n_paths=5000, n_steps=50, seed=2025, S0=S0, sigma=sigma, T=T)
    print("Approx Greeks (barrier) via finite differences:")
    print(gd)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(payoffs, bins=50, density=True)
    ax[0].set_title("MC European payoffs (density)")
    ax[0].set_xlabel("Payoff")

    # sample paths
    S_paths = simulate_gbm(S0, r, sigma, T, n_paths=50, n_steps=100, seed=42)
    for i in range(S_paths.shape[0]):
        ax[1].plot(np.linspace(0, T, S_paths.shape[1]), S_paths[i, :], lw=0.8)
    ax[1].axhline(H, color="red", linestyle="--", label=f"Barrier H={H}")
    ax[1].set_title("Sample GBM paths")
    ax[1].set_xlabel("Time")
    ax[1].legend()
    plt.tight_layout()
    plot_path = os.path.join(outdir, "member_b_demo.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved demo plot to {plot_path}")

    # summary CSV
    df = pd.DataFrame({
        "instrument": ["BS_call", "MC_call", "MC_barrier"],
        "price": [bs_c, mc_price, barrier_price],
        "iv_from_bs_price": [iv, np.nan, np.nan],
    })
    csv_path = os.path.join(outdir, "member_b_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run the demo and produce outputs")
    args = parser.parse_args(argv)
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
