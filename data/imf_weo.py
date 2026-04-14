"""
IMF WEO Data Pull via IMF JSON RESTful API (no key required).

Fetches for all available countries, 1980–2024:
  - GGXWDG_NGDP : gross government debt / GDP
  - GGXCNL_NGDP : primary balance / GDP
  - NGDP_RPCH   : real GDP growth
  - PCPIPCH     : CPI inflation

Saves cleaned wide panel to data/panel.parquet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

INDICATORS = {
    "GGXWDG_NGDP": "debt_gdp",
    "GGXCNL_NGDP": "primary_balance_gdp",
    "NGDP_RPCH": "real_gdp_growth",
    "PCPIPCH": "cpi_inflation",
}

START_YEAR = 1980
END_YEAR = 2024

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "panel.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_indicator(indicator: str, retries: int = 3, backoff: float = 2.0) -> dict:
    """Fetch a single WEO indicator from the IMF DataMapper API.

    Returns the raw JSON ``values`` dict keyed by ISO country code.
    """
    url = f"{BASE_URL}/{indicator}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("values", {}).get(indicator, {})
        except requests.RequestException as exc:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, indicator, exc)
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    logger.error("All retries failed for indicator %s", indicator)
    return {}


def _indicator_to_df(raw: dict, col_name: str) -> pd.DataFrame:
    """Convert the raw country→year→value dict to a tidy DataFrame."""
    records = []
    for iso, yearly in raw.items():
        for year_str, value in yearly.items():
            try:
                year = int(year_str)
                val = float(value)
            except (ValueError, TypeError):
                continue
            if START_YEAR <= year <= END_YEAR:
                records.append({"iso": iso, "year": year, col_name: val})
    if not records:
        return pd.DataFrame(columns=["iso", "year", col_name])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fetch_weo(save: bool = True) -> pd.DataFrame:
    """Pull all WEO indicators and return a merged panel DataFrame.

    Parameters
    ----------
    save:
        If *True*, the result is also written to ``data/panel.parquet``.

    Returns
    -------
    pd.DataFrame
        Long panel with columns [iso, year, debt_gdp, primary_balance_gdp,
        real_gdp_growth, cpi_inflation].
    """
    frames = []
    for imf_code, col_name in INDICATORS.items():
        logger.info("Fetching %s (%s) …", imf_code, col_name)
        raw = _fetch_indicator(imf_code)
        df = _indicator_to_df(raw, col_name)
        logger.info("  → %d observations", len(df))
        frames.append(df)

    if not frames:
        logger.error("No data fetched — check network connectivity.")
        return pd.DataFrame()

    # Merge on iso × year
    panel = frames[0]
    for df in frames[1:]:
        panel = panel.merge(df, on=["iso", "year"], how="outer")

    panel = panel.sort_values(["iso", "year"]).reset_index(drop=True)
    logger.info("WEO panel shape: %s", panel.shape)

    if save:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Saved WEO panel to %s", OUTPUT_FILE)

    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_weo()
