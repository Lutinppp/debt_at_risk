"""
Machado-Santos Silva (2019) Location-Scale Quantile Regression estimator.

Model:
    d_{i,t+h} = α_i + X'β + (δ_i + X'γ) · ε_{i,t+h}

Three-step procedure:
  Step 1: Fixed-effects OLS on d_{i,t+h} ~ [X_{i,t}, d_{i,t}] → residuals ê
  Step 2: Fixed-effects OLS on |ê|    ~ [X_{i,t}, d_{i,t}] → fitted scale ŝ
  Step 3: z = ê / ŝ → empirical quantiles q(τ)
  Predicted: Q(τ) = (α̂_i + δ̂_i · q(τ)) + X'β̂ + X'γ̂ · q(τ)

Reference: Machado & Santos Silva (2019), JOE 213(1), 145-173.
           Furceri et al. (2025), IMF WP/25/86.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# linearmodels for panel fixed-effects OLS
from linearmodels import PanelOLS

MODEL_DIR = Path(__file__).parent
DATA_DIR  = MODEL_DIR.parent / "data"

QUANTILES  = [0.05, 0.25, 0.50, 0.75, 0.95]
HORIZONS   = [1, 3, 5]

# Conditioning variables available in the panel
COND_VARS = {
    "primary_balance": "primary_balance_gdp",
    "rgdp_growth":     "rgdp_growth",
    "cpi_inflation":   "cpi_inflation",
    "initial_debt":    "debt_gdp_lag",
    "fsi":             "fsi",
    "spread_10y":      "spread_10y",
    "wui":             "wui",
}


def _prepare_panel(df: pd.DataFrame, cond_col: str, horizon: int) -> pd.DataFrame:
    """
    Prepare a balanced sub-panel for one conditioning variable × horizon.

    Returns DataFrame with MultiIndex (entity, time) as required by linearmodels.
    """
    dep_var = f"debt_gdp_fwd{horizon}"
    required = ["debt_gdp_lag", cond_col, dep_var]

    sub = df[["iso3", "year"] + required].dropna(subset=required).copy()

    # linearmodels needs a MultiIndex(entity, time)
    sub = sub.set_index(["iso3", "year"])
    return sub


def _ols_fe(df: pd.DataFrame, dep: str, indep: list[str]) -> tuple:
    """
    Run fixed-effects OLS via linearmodels.PanelOLS.

    Returns (fitted_values, residuals, params_df, entity_effects)
    """
    formula_vars = " + ".join(indep) + " + EntityEffects"
    # Build formula string
    formula = f"{dep} ~ 1 + {' + '.join(indep)} + EntityEffects"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model  = PanelOLS.from_formula(formula, data=df, drop_absorbed=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)

    fitted    = result.fitted_values.squeeze()
    residuals = result.resids.squeeze()
    params    = result.params

    # Extract entity effects
    fe_df = result.estimated_effects.copy() if hasattr(result, "estimated_effects") else pd.DataFrame()

    return fitted, residuals, params, fe_df


def run_location_scale(
    panel: pd.DataFrame,
    cond_var_name: str,
    horizon: int,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run the MSS three-step estimator for one conditioning variable and horizon.

    Parameters
    ----------
    panel         : clean panel DataFrame (iso3, year indexed)
    cond_var_name : name key in COND_VARS dict
    horizon       : forecast horizon h ∈ {1, 3, 5}
    quantiles     : list of τ values

    Returns
    -------
    DataFrame with columns: iso3, year, horizon, cond_var,
                            Q05, Q25, Q50, Q75, Q95
    """
    if quantiles is None:
        quantiles = QUANTILES

    cond_col = COND_VARS[cond_var_name]
    dep_var  = f"debt_gdp_fwd{horizon}"

    sub = _prepare_panel(panel, cond_col, horizon)
    if len(sub) < 50:
        print(f"    Skipping {cond_var_name} h={horizon}: insufficient data ({len(sub)} obs)")
        return pd.DataFrame()

    indep_cols = ["debt_gdp_lag", cond_col]

    # ── Step 1: OLS on level ─────────────────────────────────────────────────
    try:
        fitted1, resid1, params1, fe1 = _ols_fe(sub, dep_var, indep_cols)
    except Exception as exc:
        print(f"    Step 1 failed ({cond_var_name}, h={horizon}): {exc}")
        return pd.DataFrame()

    # ── Step 2: OLS on absolute residuals ────────────────────────────────────
    sub2 = sub.copy()
    sub2["__abs_resid__"] = np.abs(resid1.reindex(sub2.index).values)
    sub2 = sub2.dropna(subset=["__abs_resid__"])

    try:
        fitted2, _, params2, fe2 = _ols_fe(sub2, "__abs_resid__", indep_cols)
    except Exception as exc:
        print(f"    Step 2 failed ({cond_var_name}, h={horizon}): {exc}")
        return pd.DataFrame()

    # ── Step 3: standardise residuals, get empirical quantiles ───────────────
    scale_hat = fitted2.reindex(sub2.index).clip(lower=1e-6)
    abs_resid_aligned = sub2["__abs_resid__"]
    common_idx = scale_hat.index.intersection(abs_resid_aligned.index)

    z = resid1.reindex(common_idx) / scale_hat.reindex(common_idx)
    z = z.dropna()

    q_empirical = {tau: float(np.quantile(z, tau)) for tau in quantiles}

    # ── Predicted quantile: Q(τ | X_{i,t}) ──────────────────────────────────
    # Q(τ) = (μ̂_{i,t}) + q(τ) · ŝ_{i,t}
    #   where μ̂ = fitted from Step 1
    #         ŝ = fitted from Step 2

    mu_hat = fitted1.reindex(common_idx)
    s_hat  = scale_hat.reindex(common_idx)

    records = []
    for idx in common_idx:
        iso3, year = idx
        mu  = float(mu_hat[idx])
        s   = float(s_hat[idx])
        row = {"iso3": iso3, "year": int(year), "horizon": horizon, "cond_var": cond_var_name}
        for tau, qz in q_empirical.items():
            col = f"Q{int(round(tau * 100)):02d}"
            row[col] = mu + qz * s
        records.append(row)

    return pd.DataFrame(records)


def run_all(panel: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """
    Run MSS estimator for all conditioning variables × all horizons.

    Returns merged DataFrame of quantile predictions.
    """
    if horizons is None:
        horizons = HORIZONS

    results = []
    for h in horizons:
        print(f"\n  Horizon h={h}:")
        for var_name in COND_VARS:
            print(f"    {var_name} …", end=" ", flush=True)
            df_q = run_location_scale(panel, var_name, h)
            if not df_q.empty:
                results.append(df_q)
                print(f"{len(df_q)} rows")
            else:
                print("skipped")

    if not results:
        raise RuntimeError("No quantile predictions produced. Check panel data.")

    out = pd.concat(results, ignore_index=True)
    out_path = MODEL_DIR / "quantile_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved quantile predictions → {out_path}")
    return out


def load_quantile_predictions() -> pd.DataFrame:
    """Load cached predictions; rerun if missing."""
    cache = MODEL_DIR / "quantile_predictions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from data.panel_builder import load_panel
    return run_all(load_panel())


if __name__ == "__main__":
    from data.panel_builder import build_panel
    panel = build_panel()
    preds = run_all(panel)
    print(preds.head(20))
    print(f"Shape: {preds.shape}")
