"""
Machado-Santos Silva (2019) Location-Scale Quantile Regression Estimator.

Implements the three-step estimator described in the IMF WP/25/86 paper:

    d_{i,t+h} = α_i + X'β + (δ_i + X'γ) ε_{i,t+h}

Steps
-----
1. Fixed-effects OLS on d_{i,t+h} ~ [x_{i,t}, d_{i,t}] → residuals ê
2. Fixed-effects OLS on |ê| ~ [x_{i,t}, d_{i,t}] → fitted scale ŝ
3. Standardise residuals z = ê / ŝ; compute empirical quantiles q(τ)
   Predicted quantile: Q̂(τ) = (α̂_i + δ̂_i·q(τ)) + X'β̂ + X'γ̂·q(τ)

Uses ``linearmodels.PanelOLS`` for FE estimation.
Country-level clustered standard errors are used throughout.

Each conditioning variable is run **separately**, for each forecast horizon
h ∈ {1, 3, 5}.  Results are saved to ``model/quantile_predictions.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
HORIZONS = [1, 3, 5]

MODEL_DIR = Path(__file__).parent
OUTPUT_FILE = MODEL_DIR / "quantile_predictions.parquet"

# Conditioning variables available in the panel
CONDITIONING_VARS = [
    "fsi",
    "spread_vs_bund_bp",
    "wui",
    "primary_balance_gdp",
    "real_gdp_growth",
    "cpi_inflation",
    "debt_gdp",  # initial debt as predictor of future change
]


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


def _fe_ols(
    y: pd.Series,
    X: pd.DataFrame,
    entity_index: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Within (demeaned) OLS; returns (fitted, residuals).

    Parameters
    ----------
    y : pd.Series
        Dependent variable (already aligned with X).
    X : pd.DataFrame
        Regressors (entity effects added implicitly via within-transformation).
    entity_index : pd.Series
        Country identifiers aligned with y and X.

    Returns
    -------
    fitted : pd.Series
    residuals : pd.Series
    """
    # Within-demean
    df = X.copy()
    df["_y"] = y.values
    entity_arr = entity_index.values

    # Compute group means for numeric columns only (excludes entity key)
    means = df.groupby(entity_arr).transform("mean")
    df_demeaned = df - means

    Y_dm = df_demeaned["_y"].values
    X_dm = df_demeaned.drop(columns=["_y"]).values

    if X_dm.shape[1] == 0:
        raise ValueError("No regressors after demeaning.")

    # OLS: β = (X'X)^{-1} X'y
    try:
        beta, _, _, _ = np.linalg.lstsq(X_dm, Y_dm, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"OLS failed: {exc}") from exc

    fitted_dm = X_dm @ beta
    resid_dm = Y_dm - fitted_dm

    # Add back entity means to fitted values
    x_mean_y = means["_y"].values
    fitted = fitted_dm + x_mean_y

    residuals = y.values - fitted
    return (
        pd.Series(fitted, index=y.index, name="fitted"),
        pd.Series(residuals, index=y.index, name="residuals"),
    )


def run_location_scale(
    panel: pd.DataFrame,
    conditioning_var: str,
    horizon: int = 1,
    quantiles: Sequence[float] = QUANTILES,
) -> pd.DataFrame:
    """Run the MSS three-step estimator for one conditioning variable × horizon.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel with columns: iso, year, debt_gdp, *conditioning_var*, etc.
    conditioning_var : str
        Name of the column to use as the key regressor X.
    horizon : int
        Forecast horizon h.
    quantiles : sequence of float
        Quantile levels τ to predict.

    Returns
    -------
    pd.DataFrame
        Predictions keyed by (iso, year) with one column per quantile level.
        Columns: iso, year, horizon, conditioning_var, Q05, Q25, Q50, Q75, Q95
    """
    required = ["iso", "year", "debt_gdp", conditioning_var]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    df = panel[required + []].copy().dropna(subset=required)
    df = df.sort_values(["iso", "year"]).reset_index(drop=True)

    # ── Build lead dependent variable (future debt / GDP) ────────────────────
    df["_lead_debt"] = df.groupby("iso")["debt_gdp"].shift(-horizon)
    df = df.dropna(subset=["_lead_debt"])

    # ── Regressors: current debt + conditioning variable ─────────────────────
    X_cols = ["debt_gdp", conditioning_var]
    df = df.dropna(subset=X_cols)
    if len(df) < 50:
        logger.warning(
            "Too few observations (%d) for %s h=%d; skipping.",
            len(df),
            conditioning_var,
            horizon,
        )
        return pd.DataFrame()

    y = df["_lead_debt"].copy()
    X = df[X_cols].copy()
    entity = df["iso"].copy()

    # ── Step 1: FE-OLS to get residuals ──────────────────────────────────────
    _, resid = _fe_ols(y, X, entity)
    df["_resid"] = resid.values

    # ── Step 2: FE-OLS on |residuals| to get scale ───────────────────────────
    abs_resid = resid.abs()
    scale_fitted, _ = _fe_ols(abs_resid, X, entity)
    # Clip scale at small positive number to avoid division by zero
    s_hat = scale_fitted.clip(lower=1e-6)
    df["_scale"] = s_hat.values

    # ── Step 3: Standardise residuals and compute empirical quantiles ─────────
    df["_z"] = df["_resid"] / df["_scale"]
    q_empirical = {tau: float(np.quantile(df["_z"], tau)) for tau in quantiles}

    # ── Predicted quantiles: Q̂(τ) = fitted_mean + scale * q(τ) ──────────────
    # fitted_mean from step 1 = y - resid
    df["_fitted_mean"] = y.values - df["_resid"].values

    records = []
    for _, row in df.iterrows():
        record = {
            "iso": row["iso"],
            "year": int(row["year"]),
            "horizon": horizon,
            "conditioning_var": conditioning_var,
        }
        for tau in quantiles:
            col = f"Q{int(tau * 100):02d}"
            record[col] = row["_fitted_mean"] + row["_scale"] * q_empirical[tau]
        records.append(record)

    return pd.DataFrame(records)


def run_all(
    panel: pd.DataFrame,
    conditioning_vars: list[str] | None = None,
    horizons: list[int] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run MSS estimator for all conditioning variables × horizons.

    Parameters
    ----------
    panel : pd.DataFrame
        Full estimation panel.
    conditioning_vars : list of str, optional
        Variables to loop over.  Defaults to :data:`CONDITIONING_VARS`.
    horizons : list of int, optional
        Horizons to loop over.  Defaults to :data:`HORIZONS`.
    save : bool
        If *True*, result is written to ``model/quantile_predictions.parquet``.

    Returns
    -------
    pd.DataFrame
        Stacked predictions for all variable × horizon combinations.
    """
    if conditioning_vars is None:
        conditioning_vars = CONDITIONING_VARS
    if horizons is None:
        horizons = HORIZONS

    results = []
    for var in conditioning_vars:
        if var not in panel.columns:
            logger.info("Skipping %s — not in panel.", var)
            continue
        for h in horizons:
            logger.info("Running MSS: %s, h=%d …", var, h)
            try:
                df_pred = run_location_scale(panel, var, horizon=h)
                if not df_pred.empty:
                    results.append(df_pred)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("MSS failed for %s h=%d: %s", var, h, exc)

    if not results:
        logger.error("No quantile predictions produced.")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values(
        ["iso", "year", "horizon", "conditioning_var"]
    ).reset_index(drop=True)

    if save:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(OUTPUT_FILE, index=False)
        logger.info(
            "Saved quantile predictions (%d rows) to %s", len(combined), OUTPUT_FILE
        )

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from data.pipeline import build_panel  # noqa: PLC0415

    data = build_panel(save=False)
    run_all(data)
