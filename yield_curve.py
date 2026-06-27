"""
Nelson-Siegel Yield Curve Fitting
==================================
Fits the Nelson-Siegel model to US Treasury yield data,
extracts level / slope / curvature factors and plots their evolution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── 1. Nelson-Siegel basis functions ──────────────────────────────────────────

def ns_basis(maturities, lam):
    tau = np.array(maturities, dtype=float)
    l1 = np.ones_like(tau)
    l2 = (1 - np.exp(-tau / lam)) / (tau / lam)
    l3 = l2 - np.exp(-tau / lam)
    return np.column_stack([l1, l2, l3])

def ns_yields(maturities, beta, lam):
    return ns_basis(maturities, lam) @ beta

def fit_ns(maturities, observed, lam0=1.5):
    def objective(params):
        beta, lam = params[:3], params[3]
        if lam <= 0.01:
            return 1e10
        return np.sum((ns_yields(maturities, beta, lam) - observed) ** 2)
    x0 = np.array([observed.mean(), -0.5, 0.5, lam0])
    bounds = [(None,None),(None,None),(None,None),(0.05,10.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    beta, lam = res.x[:3], res.x[3]
    fitted = ns_yields(maturities, beta, lam)
    rmse = np.sqrt(np.mean((fitted - observed)**2))
    return beta, lam, fitted, rmse

# ── 2. Hardcoded US Treasury snapshot data ────────────────────────────────────

def load_sample_treasury_data():
    """
    US Treasury constant-maturity yields (% p.a.).
    Source: FRED / US Treasury. Maturities in years.
    """
    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    data = {
        "2020-01-02": [1.57,1.55,1.58,1.59,1.57,1.60,1.69,1.83,1.88,2.14,2.33],
        "2020-07-01": [0.10,0.11,0.14,0.16,0.17,0.19,0.30,0.48,0.67,1.09,1.41],
        "2021-01-04": [0.07,0.08,0.09,0.10,0.13,0.19,0.37,0.65,0.93,1.45,1.65],
        "2022-01-03": [0.04,0.05,0.19,0.39,0.73,1.03,1.37,1.55,1.63,2.05,2.01],
        "2022-07-01": [1.73,2.52,3.04,3.00,2.94,3.00,3.01,3.00,2.98,3.32,3.13],
        "2023-01-03": [4.10,4.42,4.72,4.68,4.40,4.18,3.94,3.87,3.87,4.05,3.96],
        "2024-01-02": [5.40,5.40,5.26,4.79,4.26,4.05,3.93,3.96,3.97,4.29,4.14],
        "2024-07-01": [5.39,5.31,5.16,4.86,4.57,4.43,4.30,4.33,4.37,4.67,4.53],
        "2025-01-02": [4.60,4.32,4.31,4.17,4.24,4.33,4.42,4.55,4.57,4.86,4.78],
    }
    df = pd.DataFrame(data, index=maturities).T
    df.index = pd.to_datetime(df.index)
    df.columns = [f"m{int(m*12)}mo" if m < 1 else f"y{int(m)}yr" for m in maturities]
    return df, maturities

# ── 3. Fit all dates ──────────────────────────────────────────────────────────

def fit_all_dates(df, maturities):
    results = []
    for date, row in df.iterrows():
        observed = row.values.astype(float)
        beta, lam, _, rmse = fit_ns(np.array(maturities), observed)
        results.append({"date": date, "level": beta[0],
                        "slope": -beta[1], "curvature": beta[2],
                        "lambda": lam, "rmse_bps": rmse * 100})
    return pd.DataFrame(results).set_index("date")

# ── 4. Plotting ───────────────────────────────────────────────────────────────

def plot_all(df, maturities, factors):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Nelson-Siegel Yield Curve Analysis\nUS Treasury Constant Maturity Yields",
                 fontsize=13, fontweight="bold")
    mat_fine = np.linspace(1/12, 30, 200)
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(df)))

    ax = axes[0, 0]
    for i, (date, row) in enumerate(df.iterrows()):
        obs = row.values.astype(float)
        beta, lam, _, _ = fit_ns(np.array(maturities), obs)
        ax.plot(mat_fine, ns_yields(mat_fine, beta, lam), color=colors[i], lw=1.5,
                label=date.strftime("%Y-%m"))
        ax.scatter(maturities, obs, color=colors[i], s=18, zorder=5)
    ax.set_title("Fitted Yield Curves"); ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    for ax, col, color, title in [
        (axes[0,1], "level",     "#1A4F8A", "Level Factor (β₁) — Long-Run Yield"),
        (axes[1,0], "slope",     "#D62728", "Slope Factor (−β₂) — Curve Steepness"),
        (axes[1,1], "curvature", "#2CA02C", "Curvature Factor (β₃) — Mid-Maturity Hump"),
    ]:
        ax.plot(factors.index, factors[col], color=color, lw=2, marker="o", ms=5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        if col == "slope":
            ax.fill_between(factors.index, factors[col], 0,
                            where=factors[col] < 0, alpha=0.15, color=color,
                            label="Inverted"); ax.legend(fontsize=8)
        ax.set_title(title); ax.set_ylabel(col.capitalize() + " (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("yield_curve_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved: yield_curve_analysis.png")
    plt.close()

# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    df, maturities = load_sample_treasury_data()
    print("Fitting Nelson-Siegel model...")
    factors = fit_all_dates(df, maturities)
    print(factors.round(4).to_string())
    plot_all(df, maturities, factors)
    idx = factors["slope"].idxmin()
    print(f"\nMost inverted: {idx.date()} (slope={factors.loc[idx,'slope']:.2f}%)")
    print(f"Median RMSE: {factors['rmse_bps'].median():.1f} bps")

if __name__ == "__main__":
    main()
