"""
World Uncertainty Index (WUI) — CSV ingest.

The WUI data (Ahir, Bloom & Furceri 2018, updated) is available as an Excel
file from https://worlduncertaintyindex.com/data/  (WUI_Data.xlsx).

Usage
-----
Either:
  (a) Pass the path to the downloaded XLSX/CSV explicitly, or
  (b) Let the module attempt to download from the canonical URL.

Output columns: iso, year, wui
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WUI_URL = "https://worlduncertaintyindex.com/data/WUI_Data.xlsx"

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "wui.parquet"
CACHE_XLSX = DATA_DIR / "WUI_Data.xlsx"

START_YEAR = 1980
END_YEAR = 2024


def _download_wui(dest: Path, retries: int = 3) -> bool:
    """Attempt to download the WUI Excel file; return True on success."""
    for attempt in range(retries):
        try:
            resp = requests.get(WUI_URL, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info("Downloaded WUI data to %s", dest)
            return True
        except requests.RequestException as exc:
            logger.warning("WUI download attempt %d failed: %s", attempt + 1, exc)
    return False


def _parse_wui_xlsx(path: Path) -> pd.DataFrame:
    """Parse the WUI Excel workbook and return a tidy [iso, year, wui] DataFrame.

    The WUI Excel file typically has countries as columns and year-quarters
    as rows (e.g., '1990Q1').  We aggregate to annual means.
    """
    xl = pd.ExcelFile(path)
    # Try the first sheet
    sheet = xl.sheet_names[0]
    raw = xl.parse(sheet, header=0, index_col=0)

    # Drop fully-NaN rows/columns
    raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # The index is expected to be quarters like "1990Q1"; extract year
    raw.index = raw.index.astype(str)
    raw["year"] = raw.index.str.extract(r"(\d{4})")[0].astype(float)
    raw = raw.dropna(subset=["year"])
    raw["year"] = raw["year"].astype(int)
    raw = raw[(raw["year"] >= START_YEAR) & (raw["year"] <= END_YEAR)]

    # Annual mean across quarters
    annual = raw.groupby("year").mean(numeric_only=True)

    # Melt to long format; column names should be ISO-3 country codes
    records = []
    for iso in annual.columns:
        col = annual[iso].dropna()
        for year, val in col.items():
            records.append({"iso": str(iso).upper(), "year": int(year), "wui": float(val)})

    return pd.DataFrame(records)


def fetch_wui(
    path: str | Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Load WUI data from a local file or (attempt to) download it.

    Parameters
    ----------
    path:
        Explicit path to the WUI XLSX or CSV file.  If *None*, the function
        first checks for a cached copy at ``data/WUI_Data.xlsx`` and then
        tries to download from the canonical URL.
    save:
        Write result to ``data/wui.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Columns: [iso, year, wui]
    """
    if path is not None:
        source = Path(path)
    elif CACHE_XLSX.exists():
        source = CACHE_XLSX
    else:
        logger.info("WUI file not found locally; attempting download …")
        if not _download_wui(CACHE_XLSX):
            logger.warning("WUI download failed — returning empty DataFrame.")
            return pd.DataFrame(columns=["iso", "year", "wui"])
        source = CACHE_XLSX

    logger.info("Parsing WUI data from %s …", source)
    try:
        df = _parse_wui_xlsx(source)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to parse WUI file: %s", exc)
        return pd.DataFrame(columns=["iso", "year", "wui"])

    df = df.sort_values(["iso", "year"]).reset_index(drop=True)
    logger.info("WUI panel shape: %s", df.shape)

    if save:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Saved WUI panel to %s", OUTPUT_FILE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_wui()
