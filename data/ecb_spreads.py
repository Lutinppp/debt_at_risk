"""
ECB Statistical Data Warehouse (SDW) — 10-year sovereign spreads vs. Bund.

Fetches daily yield series for France (FR), Italy (IT), Spain (ES),
and Germany (DE) from the ECB SDW REST API (no key required), then
computes annual average spread vs. the German Bund.

ECB SDW series keys used (IRTS dataset, AAA-rated long-term):
  DE: IRS.M.DE.L.L40.CI.0000.EUR.N.Z (10Y Bund yield, monthly)
  FR: IRS.M.FR.L.L40.CI.0000.EUR.N.Z
  IT: IRS.M.IT.L.L40.CI.0000.EUR.N.Z
  ES: IRS.M.ES.L.L40.CI.0000.EUR.N.Z

Output columns: iso, year, spread_vs_bund_bp  (basis points)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
# IRS = Interest Rate Statistics, monthly, 10-year government bond yields
SERIES_KEYS = {
    "DE": "IRS/M.DE.L.L40.CI.0000.EUR.N.Z",
    "FR": "IRS/M.FR.L.L40.CI.0000.EUR.N.Z",
    "IT": "IRS/M.IT.L.L40.CI.0000.EUR.N.Z",
    "ES": "IRS/M.ES.L.L40.CI.0000.EUR.N.Z",
}

ISO_MAP = {"DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP"}

START_YEAR = 1993
END_YEAR = 2024

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "ecb_spreads.parquet"


def _fetch_series(
    series_key: str, retries: int = 3, backoff: float = 2.0
) -> pd.Series:
    """Fetch a single ECB SDW time series and return a monthly pd.Series."""
    url = f"{ECB_BASE}/{series_key}?format=csvdata"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            from io import StringIO

            df = pd.read_csv(StringIO(resp.text))
            # ECB CSV has columns: KEY, FREQ, ..., TIME_PERIOD, OBS_VALUE
            if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
                logger.warning("Unexpected ECB CSV columns: %s", df.columns.tolist())
                return pd.Series(dtype=float)
            s = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
            s.index = pd.to_datetime(df["TIME_PERIOD"])
            return s.dropna()
        except requests.RequestException as exc:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, series_key, exc)
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    return pd.Series(dtype=float)


def fetch_ecb_spreads(save: bool = True) -> pd.DataFrame:
    """Fetch 10-year sovereign spreads vs. Bund for EU G4 countries.

    Returns
    -------
    pd.DataFrame
        Columns: [iso, year, spread_vs_bund_bp]
        *iso* uses IMF 3-letter codes (DEU, FRA, ITA, ESP).
        Spread for DEU is always 0 by construction.
    """
    yields: dict[str, pd.Series] = {}
    for country, key in SERIES_KEYS.items():
        logger.info("Fetching ECB yield for %s …", country)
        s = _fetch_series(key)
        if s.empty:
            logger.warning("No ECB data for %s; will use NaN.", country)
        else:
            yields[country] = s

    if "DE" not in yields:
        logger.warning("No Bund yield available — spread computation will fail.")
        bund = pd.Series(dtype=float)
    else:
        bund = yields["DE"]

    records = []
    for country, iso3 in ISO_MAP.items():
        if country == "DE":
            # Spread vs itself = 0 for all years where Bund exists
            for year in range(START_YEAR, END_YEAR + 1):
                records.append({"iso": iso3, "year": year, "spread_vs_bund_bp": 0.0})
            continue

        if country not in yields:
            continue

        country_yield = yields[country]
        # Align on monthly dates then compute annual mean spread
        combined = pd.DataFrame({"country": country_yield, "bund": bund})
        combined = combined.loc[
            (combined.index.year >= START_YEAR)
            & (combined.index.year <= END_YEAR)
        ]
        combined["spread_bp"] = (combined["country"] - combined["bund"]) * 100
        annual = combined["spread_bp"].groupby(combined.index.year).mean()
        for year, spread in annual.items():
            if pd.notna(spread):
                records.append({"iso": iso3, "year": int(year), "spread_vs_bund_bp": spread})

    if not records:
        logger.warning("No ECB spread data — returning empty DataFrame.")
        return pd.DataFrame(columns=["iso", "year", "spread_vs_bund_bp"])

    df = pd.DataFrame(records).sort_values(["iso", "year"]).reset_index(drop=True)
    logger.info("ECB spreads panel shape: %s", df.shape)

    if save:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Saved ECB spreads to %s", OUTPUT_FILE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_ecb_spreads()
