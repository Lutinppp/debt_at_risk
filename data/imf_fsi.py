"""
IMF Financial Stress Index (FSI) pull via IMF JSON RESTful API.

Dataset: Ahir, Bloom & Furceri (2023) World Financial Stress Indicator.
Fetched via the IMF DataMapper API under indicator ``FSI``.

Output columns: iso, year, fsi
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"
INDICATOR = "FSI"
START_YEAR = 1980
END_YEAR = 2024

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "fsi.parquet"


def _fetch_fsi_raw(retries: int = 3, backoff: float = 2.0) -> dict:
    url = f"{BASE_URL}/{INDICATOR}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("values", {}).get(INDICATOR, {})
        except requests.RequestException as exc:
            logger.warning("Attempt %d failed for FSI: %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    logger.error("All retries exhausted for FSI; returning empty dict.")
    return {}


def fetch_fsi(save: bool = True) -> pd.DataFrame:
    """Pull IMF Financial Stress Index for all countries.

    Parameters
    ----------
    save:
        Write result to ``data/fsi.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Columns: [iso, year, fsi]
    """
    logger.info("Fetching IMF FSI …")
    raw = _fetch_fsi_raw()

    records = []
    for iso, yearly in raw.items():
        for year_str, value in yearly.items():
            try:
                year = int(year_str)
                val = float(value)
            except (ValueError, TypeError):
                continue
            if START_YEAR <= year <= END_YEAR:
                records.append({"iso": iso, "year": year, "fsi": val})

    if not records:
        logger.warning("No FSI data retrieved — returning empty DataFrame.")
        return pd.DataFrame(columns=["iso", "year", "fsi"])

    df = pd.DataFrame(records).sort_values(["iso", "year"]).reset_index(drop=True)
    logger.info("FSI panel shape: %s", df.shape)

    if save:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Saved FSI panel to %s", OUTPUT_FILE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_fsi()
