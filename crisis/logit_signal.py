"""
Fiscal crisis early-warning logit model.

Uses the IMF/Laeven-Valencia fiscal crisis database as the binary crisis
variable. Runs panel logit:

  crisis_{i,t+1,t+2} ~ (Q95 − Q50)_{i,t}

separately for each conditioning variable, mirroring Figure 10 of WP/25/86.

Reference: Laeven & Valencia (2020), "Systemic Banking Crises Database II"
           Furceri et al. (2025), IMF WP/25/86, Section IV.D.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

CRISIS_DIR = Path(__file__).parent
DATA_DIR   = CRISIS_DIR.parent / "data"
MODEL_DIR  = CRISIS_DIR.parent / "model"
RISK_DIR   = CRISIS_DIR.parent / "risk"

G4_COUNTRIES = ["FRA", "DEU", "ITA", "ESP"]

# ── Laeven-Valencia crisis episodes (fiscal crises subset) ──────────────────
# Source: Laeven & Valencia (2020) Table A1 — country, start_year
# Coding: crisis = 1 for start_year and start_year+1
LV_FISCAL_CRISES = {
    "ARG": [2001, 2014, 2018],
    "BLZ": [2006, 2012],
    "BOL": [1994],
    "BRA": [1994, 2015, 2002],
    "CAN": [1995],
    "CHL": [1985],
    "CMR": [1988],
    "CIV": [1983, 2011],
    "COG": [1986, 1992],
    "DOM": [2003],
    "ECU": [1999, 2008],
    "EGY": [1981, 2016],
    "GRC": [2010],
    "GTM": [1986],
    "HND": [1981],
    "IRQ": [2014],
    "JAM": [2010, 2013],
    "JOR": [1989],
    "KEN": [1992],
    "LBN": [2020],
    "LBR": [1980],
    "MDG": [1988, 2009],
    "MEX": [1982, 1994],
    "MRT": [1992],
    "MOZ": [2016],
    "NGA": [1983, 1986, 2016],
    "PAK": [1998, 2019],
    "PAN": [1983, 1989],
    "PRY": [1986, 2002],
    "PER": [1983],
    "PHL": [1983],
    "PRT": [2011],
    "RUS": [1998, 2014],
    "SLE": [1997],
    "TGO": [1991, 2002],
    "TTO": [1988],
    "TUN": [1991],
    "TUR": [1978, 1982, 2001],
    "UKR": [1998, 2014],
    "URY": [1983, 2002],
    "VEN": [1983, 1995],
    "ZMB": [1983],
    "ZWE": [2006],
    # European episodes
    "ESP": [1977, 2010],
    "ITA": [],
    "DEU": [],
    "FRA": [],
    "IRL": [2010],
    "ISL": [2008],
    "CYP": [2011],
}


def _build_crisis_variable(iso3_list: list[str], year_range: range) -> pd.DataFrame:
    """
    Construct binary crisis panel from Laeven-Valencia database.

    crisis_{i,t} = 1 if a fiscal crisis begins in year t or t+1 for country i.
    """
    records = []
    for iso3 in iso3_list:
        crisis_starts = LV_FISCAL_CRISES.get(iso3, [])
        for year in year_range:
            # 1-2 year ahead crisis indicator
            crisis = int(
                any(year <= cs <= year + 2 for cs in crisis_starts)
            )
            records.append({"iso3": iso3, "year": year, "crisis": crisis})

    return pd.DataFrame(records)


def run_logit(
    skt_params: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int = 3,
    forecast_years: list[int] | None = None,
    qpreds: pd.DataFrame | None = None,
) -> dict:
    """
    Run panel logit models and produce crisis probability scores for G4.

    Parameters
    ----------
    skt_params     : DataFrame with skewed-t parameters (for G4 forecasts)
    panel          : estimation panel
    horizon        : quantile forecast horizon (default 3)
    forecast_years : years for which to produce G4 crisis signals (default 2025–2026)
    qpreds         : raw quantile predictions (used as upside proxy for non-G4 training)
    """
    if forecast_years is None:
        forecast_years = [2025, 2026]

    cond_vars = skt_params["cond_var"].unique().tolist()
    if not cond_vars and qpreds is not None:
        cond_vars = qpreds["cond_var"].unique().tolist() if "cond_var" in qpreds.columns else []

    # ── 1. Build crisis panel ─────────────────────────────────────────────────
    all_iso3 = panel["iso3"].unique().tolist()
    year_range = range(int(panel["year"].min()), int(panel["year"].max()) + 1)
    crisis_df  = _build_crisis_variable(all_iso3, year_range)

    # ── 2. Compute upside risk predictor ──────────────────────────────────────
    # Use SKT-derived upside where available; use raw qpreds Q95-Q50 elsewhere
    if len(skt_params) > 0:
        upside_df = _compute_upside(skt_params, horizon=horizon)
    elif qpreds is not None and "Q95" in qpreds.columns:
        # Build from raw quantile predictions
        sub_q = qpreds[qpreds["horizon"] == horizon].copy()
        sub_q["upside"] = (sub_q["Q95"] - sub_q["Q50"]).clip(lower=0)
        upside_df = sub_q[["iso3", "year", "cond_var", "upside"]].dropna()
    else:
        return {"logit_results": {}, "crisis_scores": pd.DataFrame(),
                "pooled_scores": pd.DataFrame()}

    # Merge with crisis
    model_df = upside_df.merge(crisis_df, on=["iso3", "year"], how="inner")
    model_df = model_df.dropna(subset=["upside", "crisis"])

    logit_results  = {}
    crisis_records = []

    for cv in cond_vars:
        sub = model_df[model_df["cond_var"] == cv].copy()
        if len(sub) < 30 or sub["crisis"].sum() < 3:
            continue

        X = sm.add_constant(sub["upside"].values, has_constant="add")
        y = sub["crisis"].values.astype(float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model  = Logit(y, X)
                result = model.fit(disp=False, maxiter=200)
            logit_results[cv] = result

            # ── Predict G4 crisis probabilities for forecast years ────────────
            for iso3 in G4_COUNTRIES:
                g4_sub = skt_params[
                    (skt_params["iso3"] == iso3) &
                    (skt_params["horizon"] == horizon) &
                    (skt_params["cond_var"] == cv)
                ]
                if g4_sub.empty:
                    continue

                for fy in forecast_years:
                    # Use most recent available year's distribution
                    latest = g4_sub.sort_values("year").iloc[-1]
                    if any(pd.isna([latest["xi"], latest["omega"],
                                   latest["alpha"], latest["nu"]])):
                        continue
                    try:
                        from model.quantile_fit import skt_quantile_from_params
                        q95 = skt_quantile_from_params(0.95, latest)
                        q50 = skt_quantile_from_params(0.50, latest)
                    except Exception:
                        continue
                    upside_val = max(q95 - q50, 0.0)
                    X_pred     = np.array([[1.0, upside_val]])
                    prob       = float(result.predict(X_pred)[0])
                    crisis_records.append({
                        "iso3":       iso3,
                        "year":       fy,
                        "cond_var":   cv,
                        "crisis_prob": prob,
                        "upside":     upside_val,
                    })

        except Exception as exc:
            print(f"    Logit failed for {cv}: {exc}")
            continue

    crisis_scores = pd.DataFrame(crisis_records)

    # ── 3. Equal-weight pooled crisis score across drivers ────────────────────
    pooled = (
        crisis_scores.groupby(["iso3", "year"])["crisis_prob"]
        .mean()
        .reset_index()
        .rename(columns={"crisis_prob": "crisis_prob_pooled"})
    )

    # Save outputs
    crisis_scores.to_parquet(CRISIS_DIR / "crisis_scores.parquet", index=False)
    pooled.to_parquet(CRISIS_DIR / "crisis_scores_pooled.parquet", index=False)
    print(f"Saved crisis scores → {CRISIS_DIR / 'crisis_scores.parquet'}")

    return {
        "logit_results": logit_results,
        "crisis_scores": crisis_scores,
        "pooled_scores": pooled,
    }


def _compute_upside(skt_params: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute upside risk (Q95 - Q50) for all country × year × cond_var.
    Falls back to raw quantile predictions if skt_params is unavailable.
    """
    from model.quantile_fit import _fst_quantile

    records = []
    sub = skt_params[skt_params["horizon"] == horizon].copy()

    for _, row in sub.iterrows():
        # If SKT params available, use fitted distribution
        if not any(pd.isna([row.get("xi"), row.get("omega"),
                             row.get("alpha"), row.get("nu")])):
            try:
                q95 = _fst_quantile(0.95, row["xi"], row["omega"],
                                    row["alpha"], row["nu"])
                q50 = _fst_quantile(0.50, row["xi"], row["omega"],
                                    row["alpha"], row["nu"])
                records.append({
                    "iso3":     row["iso3"],
                    "year":     row["year"],
                    "cond_var": row["cond_var"],
                    "upside":   max(q95 - q50, 0.0),
                })
                continue
            except Exception:
                pass
        # Fallback: use raw Q95 - Q50 from quantile predictions columns
        if "Q95" in row and "Q50" in row and pd.notna(row.get("Q95")) and pd.notna(row.get("Q50")):
            records.append({
                "iso3":     row["iso3"],
                "year":     row["year"],
                "cond_var": row["cond_var"],
                "upside":   max(float(row["Q95"]) - float(row["Q50"]), 0.0),
            })

    return pd.DataFrame(records)


def load_crisis_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached crisis scores (per-driver, pooled)."""
    cs_path = CRISIS_DIR / "crisis_scores.parquet"
    cp_path = CRISIS_DIR / "crisis_scores_pooled.parquet"
    if cs_path.exists() and cp_path.exists():
        return pd.read_parquet(cs_path), pd.read_parquet(cp_path)

    from model.quantile_fit import load_skt_params
    from data.panel_builder import load_panel

    skt   = load_skt_params()
    panel = load_panel()
    out   = run_logit(skt, panel)
    return out["crisis_scores"], out["pooled_scores"]


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from data.panel_builder import load_panel

    skt   = load_skt_params()
    panel = load_panel()
    out   = run_logit(skt, panel)

    print("\nPooled crisis probability scores for G4 (2025-2026):")
    print(out["pooled_scores"])
