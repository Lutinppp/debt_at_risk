"""
Debt-at-Risk (DaR) extraction from the pooled density.

Extracts from the pooled distribution:
  • DaR      = P95   (upside debt risk)
  • Upside   = P95 − P50
  • Downside = P50 − P5

For the G4 board presentation we focus on h = 3 (three-year ahead, 2027
horizon).  The median (P50) is re-centred to match the IMF WEO April 2025
baseline projection for each G4 country.

WEO baselines (% GDP, approximate 2027 projection):
  France  (FRA): 117 %
  Germany (DEU):  65 %
  Italy   (ITA): 135 %
  Spain   (ESP): 109 %

Saves results to ``risk/dar.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RISK_DIR = Path(__file__).parent
DAR_FILE = RISK_DIR / "dar.parquet"

G4_ISO = ["FRA", "DEU", "ITA", "ESP"]
BOARD_HORIZON = 3  # years ahead

# IMF WEO April 2025 baseline projections (% GDP)
WEO_BASELINES: dict[str, float] = {
    "FRA": 117.0,
    "DEU": 65.0,
    "ITA": 135.0,
    "ESP": 109.0,
}


def extract_dar(
    pooled: pd.DataFrame,
    weo_baselines: dict[str, float] | None = None,
    horizon: int = BOARD_HORIZON,
    base_year: int = 2024,
    save: bool = True,
) -> pd.DataFrame:
    """Extract Debt-at-Risk metrics from the pooled density.

    Parameters
    ----------
    pooled : pd.DataFrame
        Output of :func:`risk.pooling.build_pooled_density` with columns
        [iso, year, horizon, Q05, Q25, Q50, Q75, Q95].
    weo_baselines : dict, optional
        WEO median re-centring targets keyed by ISO-3 code.
    horizon : int
        Forecast horizon to extract (default = 3).
    base_year : int
        The "current" year from which horizon is counted (default = 2024).
    save : bool
        Write result to ``risk/dar.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Columns: iso, year, horizon, Q05, Q50, Q95, DaR, Upside, Downside,
                 Q50_weo (re-centred median), DaR_weo, Upside_weo
    """
    if weo_baselines is None:
        weo_baselines = WEO_BASELINES

    df = pooled[pooled["horizon"] == horizon].copy()
    if df.empty:
        logger.warning("No pooled density data for horizon=%d.", horizon)
        return pd.DataFrame()

    # Focus on the base year (current projection)
    df_base = df[df["year"] == base_year].copy()
    if df_base.empty:
        # Fall back to latest available year
        latest = df["year"].max()
        logger.warning(
            "No data for year=%d; using latest available year=%d.", base_year, latest
        )
        df_base = df[df["year"] == latest].copy()

    df_base["DaR"] = df_base["Q95"]
    df_base["Upside"] = df_base["Q95"] - df_base["Q50"]
    df_base["Downside"] = df_base["Q50"] - df_base["Q05"]

    # Re-centre median to WEO baseline
    df_base["Q50_weo"] = df_base["iso"].map(
        lambda iso: weo_baselines.get(iso, float("nan"))
    )

    def _shift(row: pd.Series) -> pd.Series:
        if pd.isna(row.get("Q50_weo")):
            row["DaR_weo"] = row["DaR"]
            row["Upside_weo"] = row["Upside"]
        else:
            shift = row["Q50_weo"] - row["Q50"]
            row["DaR_weo"] = row["Q95"] + shift
            row["Upside_weo"] = row["Q95"] + shift - row["Q50_weo"]
        return row

    df_base = df_base.apply(_shift, axis=1)
    df_base = df_base.reset_index(drop=True)

    logger.info("DaR extracted for %d countries at h=%d.", len(df_base), horizon)

    if save:
        DAR_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_base.to_parquet(DAR_FILE, index=False)
        logger.info("Saved DaR to %s", DAR_FILE)

    return df_base


def get_g4_dar(
    pooled: pd.DataFrame | None = None,
    dar_path: Path | None = None,
) -> pd.DataFrame:
    """Return DaR metrics for the EU G4 countries.

    Loads from cache if available; otherwise computes from ``pooled``.
    """
    dar_path = dar_path or DAR_FILE
    if pooled is None and dar_path.exists():
        df = pd.read_parquet(dar_path)
    elif pooled is not None:
        df = extract_dar(pooled)
    else:
        raise FileNotFoundError(
            f"DaR file not found at {dar_path} and no pooled density provided."
        )
    return df[df["iso"].isin(G4_ISO)].copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pooled_path = Path(__file__).parent / "pooled_density.parquet"
    if pooled_path.exists():
        pooled = pd.read_parquet(pooled_path)
        extract_dar(pooled)
    else:
        logger.error("Pooled density file not found at %s", pooled_path)
