"""
IMF WEO API pull — retrieves fiscal and macro indicators for all countries.

Indicators:
  GGXWDG_NGDP  – gross government debt / GDP
  GGXCNL_NGDP  – primary balance / GDP
  NGDP_RPCH    – real GDP growth (%)
  PCPIPCH      – CPI inflation (%)

Uses the IMF JSON RESTful API (no key required):
  https://www.imf.org/external/datamapper/api/v1/
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time
from data.api_config import add_api_key

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"
INDICATORS = {
    "GGXWDG_NGDP": "debt_gdp",
    "GGXCNL_NGDP": "primary_balance_gdp",
    "NGDP_RPCH":   "rgdp_growth",
    "PCPIPCH":     "cpi_inflation",
}
START_YEAR = 1980
END_YEAR   = 2025

DATA_DIR = Path(__file__).parent


def _fetch_indicator(indicator: str) -> pd.DataFrame:
    """Return long-form DataFrame (iso3, year, value) for one WEO indicator."""
    url = f"{BASE_URL}/{indicator}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=add_api_key(), timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch {indicator}: {exc}") from exc
            time.sleep(2 ** attempt)

    payload = resp.json()
    values_block = payload.get("values", {}).get(indicator, {})

    records = []
    for iso3, year_dict in values_block.items():
        for yr_str, val in year_dict.items():
            try:
                year = int(yr_str)
                value = float(val)
            except (ValueError, TypeError):
                continue
            if START_YEAR <= year <= END_YEAR:
                records.append({"iso3": iso3, "year": year, "value": value})

    return pd.DataFrame(records)


def fetch_weo(save: bool = True) -> pd.DataFrame:
    """
    Pull all four WEO indicators and return a wide panel DataFrame.

    Columns: iso3, year, debt_gdp, primary_balance_gdp, rgdp_growth, cpi_inflation
    """
    frames = {}
    for imf_code, col_name in INDICATORS.items():
        print(f"  Fetching WEO {imf_code} …")
        df = _fetch_indicator(imf_code)
        df = df.rename(columns={"value": col_name})
        frames[col_name] = df

    # Merge all indicators on iso3 × year
    merged = None
    for col_name, df in frames.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["iso3", "year"], how="outer")

    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)

    if save:
        out = DATA_DIR / "weo_raw.parquet"
        merged.to_parquet(out, index=False)
        print(f"  Saved → {out}")

    return merged


def load_weo() -> pd.DataFrame:
    """Load cached WEO data; re-fetch if not present."""
    cache = DATA_DIR / "weo_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return fetch_weo(save=True)


if __name__ == "__main__":
    df = fetch_weo()
    print(df.head(10))
    print(f"Shape: {df.shape}")
