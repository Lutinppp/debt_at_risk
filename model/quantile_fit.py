"""
Skewed-t density fitting for Debt-at-Risk.

For each (iso, year, horizon, conditioning_var) tuple we have five quantile
predictions {Q05, Q25, Q50, Q75, Q95}.  This module fits an
Azzalini-Capitanio skewed-t distribution to those five points by minimising
the sum of squared distances between predicted quantiles and the theoretical
quantile function of the skewed-t.

The fitted distribution is characterised by four parameters:
  ξ (location), ω (scale > 0), α (shape / skewness), ν (degrees of freedom > 2)

The PPF of the skewed-t is evaluated via a fast grid-interpolation approach
to avoid nested numerical inversions.

Saves individual density parameters per observation to
``model/density_params.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
DENSITY_PARAMS_FILE = MODEL_DIR / "density_params.parquet"

QUANTILE_COLS = ["Q05", "Q25", "Q50", "Q75", "Q95"]
QUANTILE_PROBS = [0.05, 0.25, 0.50, 0.75, 0.95]

# Grid for fast CDF→PPF inversion (standardised skew-t)
_GRID_SIZE = 500
_GRID_Z = np.linspace(-10, 10, _GRID_SIZE)


def _skewt_cdf_grid(alpha: float, nu: float) -> np.ndarray:
    """CDF of standardised Azzalini skew-t evaluated on _GRID_Z."""
    z = _GRID_Z
    t_cdf_z = stats.t.cdf(z, df=nu)
    inner = alpha * z * np.sqrt((nu + 1) / np.maximum(nu + z**2, 1e-9))
    t_cdf2 = stats.t.cdf(inner, df=nu + 1)
    return 2.0 * t_cdf_z * t_cdf2


def _skewt_ppf_fast(
    probs: np.ndarray,
    xi: float,
    omega: float,
    alpha: float,
    nu: float,
) -> np.ndarray:
    """Fast PPF via grid interpolation.

    Builds a CDF grid for the standardised Azzalini skew-t distribution
    (ξ=0, ω=1) and interpolates to obtain quantiles, then transforms.
    """
    cdf_vals = _skewt_cdf_grid(alpha, nu)
    # Ensure monotonicity (clip and deduplicate)
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    # Interpolate
    q_std = np.interp(probs, cdf_vals, _GRID_Z)
    return xi + omega * q_std


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit_one(quantile_values: np.ndarray) -> dict[str, float] | None:
    """Fit skewed-t parameters to five quantile values.

    Parameters
    ----------
    quantile_values : np.ndarray, shape (5,)
        Empirical quantile predictions at [0.05, 0.25, 0.50, 0.75, 0.95].

    Returns
    -------
    dict with keys: xi, omega, alpha, nu  — or None on failure.
    """
    probs = np.array(QUANTILE_PROBS)
    q = quantile_values.astype(float)

    if np.any(np.isnan(q)):
        return None

    # Initial parameter guesses from the quantile summary
    q50 = q[2]
    iqr = q[3] - q[1]  # Q75 - Q25
    omega0 = max(iqr / 1.35, 1e-3)
    xi0 = q50
    # Skewness proxy from Bowley's coefficient
    skew_bowley = (q[4] - 2 * q50 + q[0]) / max(q[4] - q[0], 1e-6)
    alpha0 = float(np.clip(skew_bowley * 3, -8, 8))
    nu0 = 10.0

    x0 = np.array([xi0, np.log(omega0), alpha0, np.log(nu0 - 2)])

    def _objective(params: np.ndarray) -> float:
        xi_, log_omega, alpha_, log_nu_shift = params
        omega_ = np.exp(log_omega)
        nu_ = float(np.exp(log_nu_shift)) + 2.0
        if nu_ < 2.01 or omega_ < 1e-9:
            return 1e10
        try:
            q_theory = _skewt_ppf_fast(probs, xi_, omega_, alpha_, nu_)
        except Exception:  # pylint: disable=broad-except
            return 1e10
        return float(np.sum((q - q_theory) ** 2))

    try:
        result = optimize.minimize(
            _objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
        )
        xi_ = result.x[0]
        omega_ = float(np.exp(result.x[1]))
        alpha_ = result.x[2]
        nu_ = float(np.exp(result.x[3])) + 2.0
        return {"xi": xi_, "omega": omega_, "alpha": alpha_, "nu": nu_}
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Skewed-t fit failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def fit_densities(
    predictions: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Fit skewed-t distributions to all quantile prediction rows.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of :func:`model.location_scale.run_all`, containing columns
        ``[iso, year, horizon, conditioning_var, Q05, Q25, Q50, Q75, Q95]``.
    save : bool
        Write result to ``model/density_params.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Columns: iso, year, horizon, conditioning_var, xi, omega, alpha, nu
        (plus Q05 … Q95 for downstream convenience).
    """
    required = ["iso", "year", "horizon", "conditioning_var"] + QUANTILE_COLS
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions DataFrame missing columns: {missing}")

    records = []
    for _, row in predictions.iterrows():
        q_vals = row[QUANTILE_COLS].values.astype(float)
        params = _fit_one(q_vals)
        if params is None:
            continue
        rec = {
            "iso": row["iso"],
            "year": row["year"],
            "horizon": row["horizon"],
            "conditioning_var": row["conditioning_var"],
            **params,
        }
        # Keep quantile columns for convenience
        for col, val in zip(QUANTILE_COLS, q_vals):
            rec[col] = val
        records.append(rec)

    if not records:
        logger.warning("No successful density fits.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(
        ["iso", "year", "horizon", "conditioning_var"]
    ).reset_index(drop=True)

    logger.info("Fitted %d density parameter sets.", len(df))

    if save:
        DENSITY_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DENSITY_PARAMS_FILE, index=False)
        logger.info("Saved density params to %s", DENSITY_PARAMS_FILE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preds = pd.read_parquet(MODEL_DIR / "quantile_predictions.parquet")
    fit_densities(preds)
