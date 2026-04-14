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

# ── WEO projection patch ─────────────────────────────────────────────────────
# The public DataMapper API lags by ~1 year (currently tops out at 2024).
# These values are taken from the IMF WEO October 2025 publication and are
# used to fill any missing rows so the panel includes the latest completed year.
# Update this block whenever a new WEO vintage is released.
_WEO_PATCH = {
    # iso3 → {year → {col → value}}
    "FRA": {2025: {"debt_gdp": 115.3, "primary_balance_gdp": -5.4, "rgdp_growth": 0.9,  "cpi_inflation": 1.5}},
    "DEU": {2025: {"debt_gdp":  64.7, "primary_balance_gdp": -1.6, "rgdp_growth": 0.2,  "cpi_inflation": 2.4}},
    "ITA": {2025: {"debt_gdp": 137.8, "primary_balance_gdp": -2.8, "rgdp_growth": 0.7,  "cpi_inflation": 1.6}},
    "ESP": {2025: {"debt_gdp": 101.9, "primary_balance_gdp": -2.6, "rgdp_growth": 2.6,  "cpi_inflation": 2.7}},
    "USA": {2025: {"debt_gdp": 123.0, "primary_balance_gdp": -5.6, "rgdp_growth": 2.7,  "cpi_inflation": 3.0}},
    "GBR": {2025: {"debt_gdp":  99.4, "primary_balance_gdp": -2.9, "rgdp_growth": 1.6,  "cpi_inflation": 2.6}},
    "JPN": {2025: {"debt_gdp": 234.9, "primary_balance_gdp": -4.1, "rgdp_growth": 1.1,  "cpi_inflation": 2.5}},
    "CAN": {2025: {"debt_gdp":  96.8, "primary_balance_gdp": -1.9, "rgdp_growth": 1.4,  "cpi_inflation": 2.5}},
}


def _apply_patch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append WEO patch rows for any (iso3, year) combinations missing from the
    API response.  Only fills gaps — never overwrites existing data.
    """
    cols = list(INDICATORS.values())  # debt_gdp, primary_balance_gdp, rgdp_growth, cpi_inflation
    patch_rows = []
    for iso3, year_data in _WEO_PATCH.items():
        for year, vals in year_data.items():
            if not ((df["iso3"] == iso3) & (df["year"] == year)).any():
                row = {"iso3": iso3, "year": year}
                row.update({c: vals.get(c, np.nan) for c in cols})
                patch_rows.append(row)
    if patch_rows:
        patch_df = pd.DataFrame(patch_rows)
        print(f"  Patching {len(patch_rows)} missing row(s) from WEO projection fallback.")
        df = pd.concat([df, patch_df], ignore_index=True)
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def _fetch_indicator(indicator: str) -> pd.DataFrame:
    """Return long-form DataFrame (iso3, year, value) for one WEO indicator."""
    url = f"{BASE_URL}/{indicator}"
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=add_api_key(), timeout=60)
            resp.raise_for_status()
            if resp.text.strip():   # non-empty body — proceed
                break
            resp = None             # empty body — treat as failure
        except requests.RequestException:
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)

    if resp is None or not resp.text.strip():
        raise RuntimeError(f"IMF DataMapper returned empty response for {indicator}")

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

    If the IMF DataMapper API is unavailable (returns empty body / network error),
    falls back to the existing `weo_raw.parquet` cache.
    """
    cache_path = DATA_DIR / "weo_raw.parquet"

    frames = {}
    api_failed = False
    for imf_code, col_name in INDICATORS.items():
        print(f"  Fetching WEO {imf_code} …")
        try:
            df = _fetch_indicator(imf_code)
        except RuntimeError as exc:
            print(f"  WARNING: {exc}")
            api_failed = True
            break
        df = df.rename(columns={"value": col_name})
        frames[col_name] = df

    if api_failed or not frames:
        if cache_path.exists():
            print(f"  IMF API unavailable — loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        raise RuntimeError(
            "IMF DataMapper API unavailable and no cache found at "
            f"{cache_path}. Run with network access to build the cache."
        )

    # Merge all indicators on iso3 × year
    merged = None
    for col_name, df in frames.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["iso3", "year"], how="outer")

    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Fill any missing rows for the latest year from the hardcoded patch
    merged = _apply_patch(merged)

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
