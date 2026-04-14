"""
Eurostat 10-year Maastricht criterion bond yields — sovereign spreads vs. Bund.

Primary source: Eurostat SDMX 2.1 REST API, dataset IRT_LT_MCBY_A
  https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/IRT_LT_MCBY_A
  ?format=SDMX-CSV&startPeriod=<YEAR>&geo=FR+DE+IT+ES

Spread = country yield − Germany (Bund) yield, annual average.
Countries: FR (FRA), DE (DEU), IT (ITA), ES (ESP)
Fallback: compiled historical table if Eurostat API is unavailable.
"""

import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR      = Path(__file__).parent
EUROSTAT_URL  = ("https://ec.europa.eu/eurostat/api/dissemination/"
                 "sdmx/2.1/data/IRT_LT_MCBY_A")
GEO_TO_ISO3   = {"FR": "FRA", "DE": "DEU", "IT": "ITA", "ES": "ESP"}
G4_GEOS       = list(GEO_TO_ISO3.keys())   # ["FR", "DE", "IT", "ES"]
START_YEAR    = 1990
END_YEAR      = 2025


def _fetch_eurostat_yields() -> pd.DataFrame | None:
    """
    Download annual 10Y Maastricht bond yields for G4 from Eurostat IRT_LT_MCBY_A.
    Returns DataFrame with columns [iso3, year, yield_10y, spread_10y], or None on failure.
    """
    params = {
        "format":      "SDMX-CSV",
        "startPeriod": str(START_YEAR),
        "geo":         "+".join(G4_GEOS),
    }
    try:
        resp = requests.get(EUROSTAT_URL, params=params, timeout=60)
        if resp.status_code != 200:
            print(f"    Eurostat returned HTTP {resp.status_code} — falling back.")
            return None
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        print(f"    Eurostat fetch error: {exc} — falling back.")
        return None

    df = df.rename(columns={"OBS_VALUE": "yield_10y", "TIME_PERIOD": "year", "geo": "geo2"})
    df["iso3"]     = df["geo2"].map(GEO_TO_ISO3)
    df["year"]     = pd.to_numeric(df["year"], errors="coerce")
    df["yield_10y"] = pd.to_numeric(df["yield_10y"], errors="coerce")
    df = df.dropna(subset=["iso3", "year", "yield_10y"])
    df["year"] = df["year"].astype(int)

    # Spread = country yield − Bund yield
    bund = df[df["iso3"] == "DEU"].set_index("year")["yield_10y"]
    df["spread_10y"] = df.apply(
        lambda r: r["yield_10y"] - bund.get(r["year"], np.nan), axis=1
    )

    return df[["iso3", "year", "yield_10y", "spread_10y"]].sort_values(["iso3", "year"])


def _compiled_fallback() -> pd.DataFrame:
    """
    Compiled historical 10Y sovereign yield data (from ECB/Eurostat publications).
    Used only when the live Eurostat API is unavailable.
    Values in % p.a. (annual average).
    """
    # Annual average 10Y yields (% p.a.) — compiled from ECB/Eurostat publications
    yields = {
        "DEU": {
            1993: 6.37, 1994: 6.89, 1995: 6.85, 1996: 6.24, 1997: 5.67,
            1998: 4.57, 1999: 4.49, 2000: 5.27, 2001: 4.80, 2002: 4.78,
            2003: 4.07, 2004: 4.04, 2005: 3.35, 2006: 3.77, 2007: 4.22,
            2008: 3.98, 2009: 3.22, 2010: 2.74, 2011: 2.61, 2012: 1.50,
            2013: 1.57, 2014: 1.16, 2015: 0.50, 2016: 0.09, 2017: 0.32,
            2018: 0.39, 2019: -0.25, 2020: -0.57, 2021: -0.37, 2022: 1.19,
            2023: 2.46, 2024: 2.38, 2025: 2.55,
        },
        "FRA": {
            1993: 6.88, 1994: 7.45, 1995: 7.54, 1996: 6.30, 1997: 5.59,
            1998: 4.64, 1999: 4.61, 2000: 5.39, 2001: 4.94, 2002: 4.86,
            2003: 4.13, 2004: 4.10, 2005: 3.41, 2006: 3.80, 2007: 4.30,
            2008: 3.99, 2009: 3.65, 2010: 3.12, 2011: 3.32, 2012: 2.54,
            2013: 2.20, 2014: 1.67, 2015: 0.84, 2016: 0.47, 2017: 0.81,
            2018: 0.78, 2019: 0.13, 2020: -0.34, 2021: 0.06, 2022: 1.87,
            2023: 3.07, 2024: 3.11, 2025: 3.30,
        },
        "ITA": {
            1993: 11.29, 1994: 10.58, 1995: 12.21, 1996: 9.40, 1997: 6.86,
            1998: 4.88, 1999: 4.73, 2000: 5.58, 2001: 5.19, 2002: 5.03,
            2003: 4.25, 2004: 4.26, 2005: 3.56, 2006: 4.05, 2007: 4.49,
            2008: 4.68, 2009: 4.31, 2010: 4.05, 2011: 5.42, 2012: 5.49,
            2013: 4.32, 2014: 2.89, 2015: 1.71, 2016: 1.49, 2017: 2.11,
            2018: 2.62, 2019: 1.94, 2020: 1.22, 2021: 0.92, 2022: 3.34,
            2023: 4.24, 2024: 3.89, 2025: 3.70,
        },
        "ESP": {
            1993: 10.17, 1994: 10.03, 1995: 11.30, 1996: 8.74, 1997: 6.40,
            1998: 4.83, 1999: 4.73, 2000: 5.53, 2001: 5.12, 2002: 4.96,
            2003: 4.12, 2004: 4.02, 2005: 3.39, 2006: 3.79, 2007: 4.31,
            2008: 4.37, 2009: 3.97, 2010: 4.25, 2011: 5.44, 2012: 5.85,
            2013: 4.56, 2014: 2.72, 2015: 1.74, 2016: 1.39, 2017: 1.56,
            2018: 1.43, 2019: 0.66, 2020: 0.45, 2021: 0.41, 2022: 2.57,
            2023: 3.61, 2024: 3.29, 2025: 3.20,
        },
    }

    records = []
    bund = yields["DEU"]
    for iso3, yr_data in yields.items():
        for year, yld in yr_data.items():
            bund_yld = bund.get(year, float("nan"))
            spread   = yld - bund_yld if not (pd.isna(bund_yld) or pd.isna(yld)) else float("nan")
            records.append({"iso3": iso3, "year": year, "yield_10y": yld, "spread_10y": spread})

    return pd.DataFrame(records)


def fetch_spreads(save: bool = True) -> pd.DataFrame:
    """
    Fetch 10Y sovereign yields and spreads vs. Bund for FR, DE, IT, ES.
    Primary: Eurostat IRT_LT_MCBY_A.  Fallback: compiled historical table.

    Returns DataFrame: iso3, year, yield_10y, spread_10y (percentage points).
    """
    print("  Fetching Eurostat 10Y sovereign yields (IRT_LT_MCBY_A) …")
    df = _fetch_eurostat_yields()

    if df is None or df["spread_10y"].notna().sum() == 0:
        print("  Eurostat unavailable — using compiled historical fallback.")
        df = _compiled_fallback()

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    if save:
        out = DATA_DIR / "ecb_spreads_raw.parquet"
        df.to_parquet(out, index=False)
        print(f"  Saved → {out}")

    return df


def load_spreads() -> pd.DataFrame:
    """Load cached spreads; re-fetch if not present."""
    cache = DATA_DIR / "ecb_spreads_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return fetch_spreads(save=True)


if __name__ == "__main__":
    df = fetch_spreads()
    print(df)
