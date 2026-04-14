"""
Panel builder — assembles clean estimation panel from all data sources.

Saves: data/panel.parquet
Columns: iso3, year, debt_gdp, primary_balance_gdp, rgdp_growth,
         cpi_inflation, fsi, spread_10y, wui

Filter: countries with continuous debt/growth/primary balance data 1990–2024
        (target ~40–60 countries).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent


def build_panel(min_obs: int = 20, start_year: int = 1990, end_year: int = 2025) -> pd.DataFrame:
    """
    Merge all data sources, filter to estimation panel.

    Parameters
    ----------
    min_obs   : minimum non-missing core observations per country
    start_year: first year to include
    end_year  : last year to include
    """
    from data.imf_weo    import load_weo
    from data.imf_fsi    import load_fsi
    from data.ecb_spreads import load_spreads
    from data.wui        import load_wui

    print("Building panel …")

    # -- Load all components
    weo     = load_weo()
    fsi     = load_fsi()
    spreads = load_spreads()[["iso3", "year", "spread_10y"]]
    wui     = load_wui()

    # -- Merge
    panel = (
        weo
        .merge(fsi,     on=["iso3", "year"], how="left")
        .merge(spreads, on=["iso3", "year"], how="left")
        .merge(wui,     on=["iso3", "year"], how="left")
    )

    # -- Year filter
    panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)].copy()

    # -- Core columns that must be non-missing for estimation
    core_cols = ["debt_gdp", "primary_balance_gdp", "rgdp_growth"]

    # Count non-missing core observations per country
    country_counts = (
        panel.groupby("iso3")[core_cols]
        .apply(lambda x: x.notna().all(axis=1).sum())
        .reset_index()
    )
    country_counts.columns = ["iso3", "n_core_obs"]
    good_countries = country_counts.loc[country_counts["n_core_obs"] >= min_obs, "iso3"]

    panel = panel[panel["iso3"].isin(good_countries)].copy()
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # -- Winsorize extreme outliers (debt/GDP > 300, inflation > 200%)
    panel["debt_gdp"] = panel["debt_gdp"].clip(lower=0, upper=300)
    panel["cpi_inflation"] = panel["cpi_inflation"].clip(lower=-5, upper=200)

    # -- Lagged debt (for use as regressor)
    panel["debt_gdp_lag"] = panel.groupby("iso3")["debt_gdp"].shift(1)

    # -- Forward debt at horizons h = 1, 3, 5
    for h in [1, 3, 5]:
        panel[f"debt_gdp_fwd{h}"] = panel.groupby("iso3")["debt_gdp"].shift(-h)

    out_path = DATA_DIR / "panel.parquet"
    panel.to_parquet(out_path, index=False)
    n_countries = panel["iso3"].nunique()
    print(f"Panel: {panel.shape[0]} rows × {panel.shape[1]} cols, {n_countries} countries")
    print(f"Saved → {out_path}")

    return panel


def load_panel() -> pd.DataFrame:
    """Load cached panel; rebuild if not present."""
    cache = DATA_DIR / "panel.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return build_panel()


if __name__ == "__main__":
    df = build_panel()
    print(df.describe())
