"""
Debt-at-Risk extraction and WEO baseline re-centering.

Extracts from the pooled distribution:
  DaR      = Q95 (95th percentile — debt-at-risk upper tail)
  Upside   = Q95 − Q50
  Downside = Q50 − Q05

Re-centers the median to the IMF WEO April 2025 baseline projection
for each G4 country (h=3, i.e., 2027 horizon).

Focus country set: FR, DE, IT, ES (EU G4)
Focus horizon    : h=3

Reference: Furceri et al. (2025), IMF WP/25/86, Section IV.B.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from pathlib import Path

RISK_DIR  = Path(__file__).parent
MODEL_DIR = RISK_DIR.parent / "model"

# EU G4 countries
G4_COUNTRIES = ["FRA", "DEU", "ITA", "ESP"]

# IMF WEO April 2025 baseline projections for gross govt debt/GDP in 2027
# Source: IMF World Economic Outlook April 2025
WEO_BASELINE_2027 = {
    "FRA": 117.0,  # France
    "DEU":  68.5,  # Germany
    "ITA": 138.0,  # Italy
    "ESP": 103.5,  # Spain
}


def _fst_quantile(tau: float, xi: float, omega: float, alpha: float, nu: float) -> float:
    """Fernandez-Steel skewed-t quantile function (same as in quantile_fit.py)."""
    from scipy.stats import t as t_dist
    gamma = np.exp(alpha)
    p_mid = 1.0 / (1.0 + gamma)
    if tau < p_mid:
        q_t = t_dist.ppf(tau * (1.0 + gamma) / 2.0, df=nu)
        return xi + omega * q_t / gamma
    else:
        q_t = t_dist.ppf((tau * (1.0 + gamma) - (gamma - 1.0)) / 2.0, df=nu)
        return xi + omega * q_t * gamma


def _pooled_quantile(
    tau: float,
    component_params: list[dict],
    weights: list[float],
    n_grid: int = 500,
) -> float:
    """
    Compute quantile of weighted mixture density via grid inversion.

    The pooled density is: f*(d) = Σ_k w_k · f_k(d)
    Its CDF is directly: F*(d) = Σ_k w_k · F_k(d)
    We invert F* numerically.
    """
    from scipy.stats import t as t_dist

    def fst_cdf(x, xi, omega, alpha, nu):
        """CDF of Fernandez-Steel skewed-t."""
        gamma = np.exp(alpha)
        z = (x - xi) / omega
        if z < 0:
            return t_dist.cdf(-z / gamma, df=nu) / (1.0 + gamma)
        else:
            return (1.0 / (1.0 + gamma) +
                    t_dist.cdf(z * gamma, df=nu) * gamma / (1.0 + gamma))

    def pooled_cdf(x):
        val = 0.0
        for params, w in zip(component_params, weights):
            xi, omega, alpha, nu = params["xi"], params["omega"], params["alpha"], params["nu"]
            if any(pd.isna([xi, omega, alpha, nu])):
                continue
            try:
                val += w * fst_cdf(x, xi, omega, alpha, nu)
            except Exception:
                pass
        return val

    # Find bounds: use most extreme component quantiles
    all_q05 = [_fst_quantile(0.01, p["xi"], p["omega"], p["alpha"], p["nu"])
               for p in component_params
               if not any(pd.isna([p["xi"], p["omega"], p["alpha"], p["nu"]]))]
    all_q95 = [_fst_quantile(0.99, p["xi"], p["omega"], p["alpha"], p["nu"])
               for p in component_params
               if not any(pd.isna([p["xi"], p["omega"], p["alpha"], p["nu"]]))]

    if not all_q05 or not all_q95:
        return np.nan

    lo = min(all_q05) - 20
    hi = max(all_q95) + 20

    try:
        return brentq(lambda x: pooled_cdf(x) - tau, lo, hi, xtol=0.01, maxiter=100)
    except Exception:
        # Fallback: weighted average of component quantiles
        qs = []
        ws = []
        for params, w in zip(component_params, weights):
            xi, omega, alpha, nu = params["xi"], params["omega"], params["alpha"], params["nu"]
            if not any(pd.isna([xi, omega, alpha, nu])):
                qs.append(_fst_quantile(tau, xi, omega, alpha, nu))
                ws.append(w)
        if not qs:
            return np.nan
        ws = np.array(ws)
        ws /= ws.sum()
        return float(np.dot(ws, qs))


def compute_dar(
    skt_params: pd.DataFrame,
    weights: pd.DataFrame,
    horizon: int = 3,
    base_year: int = 2024,
    recenter: bool = True,
) -> pd.DataFrame:
    """
    Compute pooled Debt-at-Risk for G4 countries.

    Parameters
    ----------
    skt_params : DataFrame with skewed-t parameters
    weights    : DataFrame with country-specific pooling weights
    horizon    : forecast horizon (default 3 → 2027)
    base_year  : last data year (observations from this year used for projection)
    recenter   : re-center pooled median to WEO baseline

    Returns
    -------
    DataFrame: iso3, year, horizon, Q05, Q50, Q95, DaR, Upside, Downside,
               weo_baseline, {cond_var}_weight × 7
    """
    taus_to_extract = [0.05, 0.50, 0.95]
    cond_vars = skt_params["cond_var"].unique().tolist()

    results = []

    for iso3 in G4_COUNTRIES:
        # Get params for this country, horizon, latest available year
        country_skt = skt_params[
            (skt_params["iso3"] == iso3) &
            (skt_params["horizon"] == horizon)
        ]
        if country_skt.empty:
            print(f"  Warning: No skt params for {iso3} h={horizon}")
            continue

        # Use most recent year's predictions
        latest_year = country_skt["year"].max()
        params_latest = country_skt[country_skt["year"] == latest_year]

        # Retrieve weights for this country
        country_weights_df = weights[weights["iso3"] == iso3]
        w_dict = dict(zip(country_weights_df["cond_var"], country_weights_df["weight"]))

        # Build component list (one per cond_var)
        component_params = []
        component_weights = []
        for cv in cond_vars:
            row = params_latest[params_latest["cond_var"] == cv]
            if row.empty:
                continue
            r = row.iloc[0]
            if any(pd.isna([r["xi"], r["omega"], r["alpha"], r["nu"]])):
                continue
            component_params.append({
                "xi": float(r["xi"]), "omega": float(r["omega"]),
                "alpha": float(r["alpha"]), "nu": float(r["nu"]),
            })
            component_weights.append(w_dict.get(cv, 1.0 / len(cond_vars)))

        if not component_params:
            print(f"  Warning: No valid components for {iso3}")
            continue

        # Normalise weights
        wt = np.array(component_weights)
        wt /= wt.sum()

        # Extract pooled quantiles
        q_values = {}
        for tau in taus_to_extract:
            q_values[tau] = _pooled_quantile(tau, component_params, wt.tolist())

        q05, q50, q95 = q_values[0.05], q_values[0.50], q_values[0.95]

        # Re-center to WEO baseline
        weo_med = WEO_BASELINE_2027.get(iso3, q50)
        shift   = (weo_med - q50) if recenter and np.isfinite(q50) else 0.0
        q05 += shift
        q50 += shift
        q95 += shift

        dar     = q95
        upside  = q95 - q50
        downside = q50 - q05

        row_out = {
            "iso3":         iso3,
            "year":         latest_year,
            "horizon":      horizon,
            "proj_year":    latest_year + horizon,
            "Q05":          round(q05, 2),
            "Q50":          round(q50, 2),
            "Q95":          round(q95, 2),
            "DaR":          round(dar, 2),
            "Upside":       round(upside, 2),
            "Downside":     round(downside, 2),
            "weo_baseline": weo_med,
        }
        # Append per-driver weights (for waterfall decomposition)
        for k, cv in enumerate(cond_vars):
            row_out[f"w_{cv}"] = float(wt[k]) if k < len(wt) else 0.0

        results.append(row_out)

    out = pd.DataFrame(results)

    # ── Driver contribution to upside risk (waterfall) ────────────────────────
    # For each driver, compute how much its individual Q95-Q50 contributes
    for cv in cond_vars:
        out[f"upside_{cv}"] = np.nan

    for idx, row in out.iterrows():
        iso3 = row["iso3"]
        total_upside = row["Upside"]
        country_skt = skt_params[
            (skt_params["iso3"] == iso3) &
            (skt_params["horizon"] == horizon) &
            (skt_params["year"] == row["year"])
        ]
        driver_upsides = {}
        for cv in cond_vars:
            cv_row = country_skt[country_skt["cond_var"] == cv]
            if cv_row.empty:
                continue
            r = cv_row.iloc[0]
            if any(pd.isna([r["xi"], r["omega"], r["alpha"], r["nu"]])):
                continue
            try:
                q95_cv = _fst_quantile(0.95, r["xi"], r["omega"], r["alpha"], r["nu"])
                q50_cv = _fst_quantile(0.50, r["xi"], r["omega"], r["alpha"], r["nu"])
                driver_upsides[cv] = max(q95_cv - q50_cv, 0.0)
            except Exception:
                pass

        total_driver = sum(driver_upsides.values())
        for cv, du in driver_upsides.items():
            share = (du / total_driver * total_upside) if total_driver > 0 else 0.0
            out.loc[idx, f"upside_{cv}"] = round(share, 2)

    out_path = RISK_DIR / "dar_results.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved DaR results → {out_path}")
    print(out[["iso3", "Q05", "Q50", "Q95", "DaR", "Upside", "Downside", "weo_baseline"]])
    return out


def load_dar(horizon: int = 3) -> pd.DataFrame:
    """Load cached DaR results."""
    cache = RISK_DIR / "dar_results.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from model.quantile_fit import load_skt_params
    from risk.pooling import load_pooling_weights
    skt     = load_skt_params()
    weights = load_pooling_weights(horizon=horizon)
    return compute_dar(skt, weights, horizon=horizon)


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from risk.pooling import load_pooling_weights

    skt     = load_skt_params()
    weights = load_pooling_weights(horizon=3)
    dar     = compute_dar(skt, weights, horizon=3)
    print(dar.T)
