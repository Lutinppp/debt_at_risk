"""
Data pipeline — build the estimation panel.

Merges WEO core variables with FSI, ECB spreads, and WUI.
Filters to countries with continuous debt / growth / primary-balance data
from 1990 onward and saves the result to data/panel.parquet.

Run as a script::

    python -m data.pipeline

or call :func:`build_panel` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "panel.parquet"

# Minimum consecutive years of core data required for a country to be included
MIN_COVERAGE_START = 1990
MIN_COVERAGE_YEARS = 10

CORE_COLS = ["debt_gdp", "real_gdp_growth", "primary_balance_gdp"]


def _load_parquet_or_empty(path: Path, fallback_columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("File not found: %s — using empty DataFrame.", path)
    return pd.DataFrame(columns=fallback_columns)


def build_panel(
    weo_path: Path | None = None,
    fsi_path: Path | None = None,
    ecb_path: Path | None = None,
    wui_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Assemble the full estimation panel.

    Parameters
    ----------
    *_path:
        Explicit paths to pre-built parquet files.  If *None*, defaults to
        the standard locations under ``data/``.
    save:
        Write the assembled panel to ``data/panel.parquet`` when *True*.

    Returns
    -------
    pd.DataFrame
        Country × year panel with all conditioning variables.
    """
    weo_path = weo_path or DATA_DIR / "panel.parquet"
    fsi_path = fsi_path or DATA_DIR / "fsi.parquet"
    ecb_path = ecb_path or DATA_DIR / "ecb_spreads.parquet"
    wui_path = wui_path or DATA_DIR / "wui.parquet"

    # ── Load each dataset ────────────────────────────────────────────────────
    # WEO panel is already written by imf_weo.fetch_weo(); if missing, try to
    # build it on the fly.
    if not weo_path.exists():
        logger.info("WEO parquet not found — running imf_weo.fetch_weo() …")
        from data.imf_weo import fetch_weo  # noqa: PLC0415

        weo = fetch_weo(save=True)
    else:
        weo = pd.read_parquet(weo_path)

    fsi = _load_parquet_or_empty(
        fsi_path, ["iso", "year", "fsi"]
    )
    ecb = _load_parquet_or_empty(
        ecb_path, ["iso", "year", "spread_vs_bund_bp"]
    )
    wui = _load_parquet_or_empty(
        wui_path, ["iso", "year", "wui"]
    )

    # ── Merge ────────────────────────────────────────────────────────────────
    panel = weo.copy()
    for aux in [fsi, ecb, wui]:
        if not aux.empty:
            merge_cols = list(aux.columns)
            key_cols = ["iso", "year"]
            panel = panel.merge(
                aux[merge_cols],
                on=key_cols,
                how="left",
            )

    # ── Filter: keep countries with sufficient core coverage from 1990 ───────
    panel_90 = panel[panel["year"] >= MIN_COVERAGE_START]
    coverage = (
        panel_90.groupby("iso")[CORE_COLS]
        .apply(lambda g: g.notna().all(axis=1).sum())
    )
    keep = coverage[coverage >= MIN_COVERAGE_YEARS].index
    logger.info(
        "Countries with ≥%d years of core data from %d: %d / %d",
        MIN_COVERAGE_YEARS,
        MIN_COVERAGE_START,
        len(keep),
        panel["iso"].nunique(),
    )
    panel = panel[panel["iso"].isin(keep)].copy()

    panel = panel.sort_values(["iso", "year"]).reset_index(drop=True)
    logger.info("Final panel shape: %s", panel.shape)

    if save:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Saved merged panel to %s", OUTPUT_FILE)

    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_panel()
