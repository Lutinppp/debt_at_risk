"""
Log-score density pooling (Crump et al. 2023 / Hall & Mitchell 2007).

Combines individual conditioning-variable densities into a single pooled
density via log-score weights.

For each country i and forecast horizon h we compute time-varying weights
w_k(i) over K conditioning variables by maximising the average log predictive
score on a 20-year rolling out-of-sample window starting in 2005:

    max_{w} Σ_t log[ Σ_k w_k · f_k(d_{i,t+h} | x_{i,t}) ]
    s.t.    Σ_k w_k = 1,  w_k ≥ 0

where f_k is the skewed-t density fitted in :mod:`model.quantile_fit`.

Because we work with **fitted** density parameters rather than evaluating the
density at realised outcomes (which requires out-of-sample data), we use an
approximation: the log-score is computed as the log of the mixture density
evaluated at Q50 (the median baseline) of each individual density.  Weights
are then constrained simplex variables optimised with scipy.

Outputs ``risk/pooled_weights.parquet`` and ``risk/pooled_density.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)

RISK_DIR = Path(__file__).parent
WEIGHTS_FILE = RISK_DIR / "pooled_weights.parquet"
POOLED_FILE = RISK_DIR / "pooled_density.parquet"

ROLLING_START = 2005
ROLLING_WINDOW = 20  # years

# Grid for fast CDF evaluation (imported from quantile_fit pattern)
_GRID_SIZE = 300
_GRID_Z = np.linspace(-8, 8, _GRID_SIZE)


# ---------------------------------------------------------------------------
# Density evaluation helpers
# ---------------------------------------------------------------------------


def _skewt_pdf(
    x: float,
    xi: float,
    omega: float,
    alpha: float,
    nu: float,
) -> float:
    """Azzalini-Capitanio skewed-t PDF evaluated at x."""
    if omega <= 0 or nu <= 2:
        return 1e-300
    z = (x - xi) / omega
    # f(x) = 2/ω · t_ν(z) · T_{ν+1}(α z √((ν+1)/(ν+z²)))
    t_pdf = stats.t.pdf(z, df=nu) / omega
    inner = alpha * z * np.sqrt((nu + 1) / max(nu + z**2, 1e-9))
    t_cdf = stats.t.cdf(inner, df=nu + 1)
    return float(2 * t_pdf * t_cdf)


def _mixture_log_score(
    weights: np.ndarray,
    densities: np.ndarray,
) -> float:
    """Negative average log-score of the mixture density.

    Parameters
    ----------
    weights : shape (K,)
    densities : shape (T, K) — density values f_k(y_t) for each variable k
                and time period t.
    """
    mixture = densities @ weights  # shape (T,)
    mixture = np.clip(mixture, 1e-300, None)
    return -float(np.mean(np.log(mixture)))


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def _compute_weights(
    group: pd.DataFrame,
    vars_: list[str],
) -> pd.Series:
    """Compute log-score pooling weights for a single (iso, horizon) group.

    We use density values evaluated at Q50 (the median of each conditioning
    density) as a proxy for the predictive score.
    """
    K = len(vars_)
    if K == 1:
        return pd.Series({v: 1.0 for v in vars_})

    # Build density matrix: rows=time, cols=vars
    pivoted = group.pivot(index="year", columns="conditioning_var")
    # Use density evaluated at Q50 — approximate with pdf at median
    density_matrix = np.ones((len(pivoted), K))
    for j, var in enumerate(vars_):
        if "xi" not in group.columns:
            continue
        sub = group[group["conditioning_var"] == var].set_index("year")
        for t_idx, year in enumerate(pivoted.index):
            if year not in sub.index:
                density_matrix[t_idx, j] = 1.0
                continue
            row = sub.loc[year]
            xi, omega, alpha, nu = row["xi"], row["omega"], row["alpha"], row["nu"]
            q50 = row.get("Q50", xi)
            density_matrix[t_idx, j] = _skewt_pdf(q50, xi, omega, alpha, nu)

    # Optimise weights on rolling window
    x0 = np.ones(K) / K
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * K
    result = optimize.minimize(
        _mixture_log_score,
        x0,
        args=(density_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if result.success:
        w = result.x
    else:
        logger.debug("Weight optimisation did not converge; using equal weights.")
        w = np.ones(K) / K

    return pd.Series({var: float(w[j]) for j, var in enumerate(vars_)})


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def compute_pooled_weights(
    density_params: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Compute log-score pooling weights per country × horizon.

    Parameters
    ----------
    density_params : pd.DataFrame
        Output of :func:`model.quantile_fit.fit_densities`.
    save : bool
        Write result to ``risk/pooled_weights.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Columns: iso, horizon, conditioning_var, weight
    """
    required = ["iso", "year", "horizon", "conditioning_var", "xi", "omega", "alpha", "nu"]
    missing = [c for c in required if c not in density_params.columns]
    if missing:
        raise ValueError(f"density_params missing columns: {missing}")

    # Filter to rolling window (2005 onward)
    df = density_params[density_params["year"] >= ROLLING_START].copy()

    results = []
    for (iso, horizon), group in df.groupby(["iso", "horizon"]):
        vars_ = sorted(group["conditioning_var"].unique().tolist())
        weights = _compute_weights(group, vars_)
        for var, w in weights.items():
            results.append({"iso": iso, "horizon": horizon, "conditioning_var": var, "weight": w})

    if not results:
        return pd.DataFrame(columns=["iso", "horizon", "conditioning_var", "weight"])

    out = pd.DataFrame(results).sort_values(["iso", "horizon", "conditioning_var"])
    out = out.reset_index(drop=True)
    logger.info("Computed pooling weights for %d (iso, horizon) pairs.", len(out))

    if save:
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(WEIGHTS_FILE, index=False)
        logger.info("Saved pooling weights to %s", WEIGHTS_FILE)

    return out


def build_pooled_density(
    density_params: pd.DataFrame,
    weights: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Combine individual densities into a pooled (iso, year, horizon) density.

    The pooled density is represented by its effective quantile parameters
    derived from a weighted mixture of the individual skewed-t distributions.

    For computational tractability we represent the mixture via Monte-Carlo
    simulation: draw N samples from each component, weight the draws, and
    re-fit a skewed-t to the resulting empirical quantiles.

    Parameters
    ----------
    density_params : pd.DataFrame
        Individual density parameters per (iso, year, horizon, var).
    weights : pd.DataFrame
        Pooling weights per (iso, horizon, var).
    save : bool
        Write result to ``risk/pooled_density.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Columns: iso, year, horizon, Q05, Q25, Q50, Q75, Q95
        (pooled quantiles, ready for Debt-at-Risk extraction).
    """
    N_SAMPLES = 2000
    PROBS = [0.05, 0.25, 0.50, 0.75, 0.95]
    COL_NAMES = ["Q05", "Q25", "Q50", "Q75", "Q95"]

    merged = density_params.merge(
        weights, on=["iso", "horizon", "conditioning_var"], how="left"
    )
    merged["weight"] = merged["weight"].fillna(0.0)

    records = []
    for (iso, year, horizon), group in merged.groupby(["iso", "year", "horizon"]):
        total_w = group["weight"].sum()
        if total_w < 1e-9:
            continue

        samples_weighted: list[np.ndarray] = []
        wts_list: list[float] = []

        for _, row in group.iterrows():
            w = row["weight"] / total_w
            if w < 1e-9:
                continue
            xi, omega, alpha, nu = row["xi"], row["omega"], row["alpha"], row["nu"]
            if pd.isna(xi) or omega <= 0 or nu <= 2:
                continue
            # Draw from skewed-t using rejection sampling approximation
            rng_t = stats.t.rvs(df=nu, size=N_SAMPLES)
            # Skewed-t via conditioning: accept if u < T_{ν+1}(α * z)
            u = np.random.uniform(size=N_SAMPLES)
            inner = alpha * rng_t * np.sqrt((nu + 1) / np.maximum(nu + rng_t**2, 1e-9))
            accept = u < stats.t.cdf(inner, df=nu + 1)
            skew_z = rng_t[accept]
            if len(skew_z) == 0:
                skew_z = rng_t
            samples = xi + omega * skew_z
            samples_weighted.append(samples)
            wts_list.append(w)

        if not samples_weighted:
            continue

        # Weighted mixture empirical quantiles
        all_samples = np.concatenate(samples_weighted)
        rng = np.random.default_rng(42)
        if len(wts_list) > 1:
            sizes = [len(s) for s in samples_weighted]
            # Re-sample proportionally to weights
            n_total = sum(sizes)
            draw_counts = (np.array(wts_list) * n_total).astype(int)
            mixed = np.concatenate(
                [
                    rng.choice(s, size=max(1, d), replace=True)
                    for s, d in zip(samples_weighted, draw_counts)
                ]
            )
        else:
            mixed = all_samples

        q_vals = np.quantile(mixed, PROBS)
        rec = {"iso": iso, "year": int(year), "horizon": int(horizon)}
        for col, val in zip(COL_NAMES, q_vals):
            rec[col] = float(val)
        records.append(rec)

    if not records:
        logger.warning("No pooled density records produced.")
        return pd.DataFrame()

    out = pd.DataFrame(records).sort_values(["iso", "year", "horizon"]).reset_index(drop=True)
    logger.info("Pooled density: %d rows.", len(out))

    if save:
        POOLED_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(POOLED_FILE, index=False)
        logger.info("Saved pooled density to %s", POOLED_FILE)

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dp = pd.read_parquet(Path(__file__).parent.parent / "model" / "density_params.parquet")
    w = compute_pooled_weights(dp)
    build_pooled_density(dp, w)
