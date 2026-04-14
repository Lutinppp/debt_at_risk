import io
from pathlib import Path
import requests
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

ECB_TO_ISO3 = {
   "AT":"AUT","BE":"BEL","BG":"BGR","CY":"CYP","CZ":"CZE",
   "DE":"DEU","DK":"DNK","EE":"EST","ES":"ESP","FI":"FIN",
   "FR":"FRA","GB":"GBR","GR":"GRC","HR":"HRV","HU":"HUN",
   "IE":"IRL","IT":"ITA","LT":"LTU","LU":"LUX","LV":"LVA",
   "MT":"MLT","NL":"NLD","PL":"POL","PT":"PRT","RO":"ROU",
   "SE":"SWE","SI":"SVN","SK":"SVK"
}
BASE = "https://data-api.ecb.europa.eu/service/data/CLIFS/"
def fetch_fsi(save=True):
   """Fetch ECB CLIFS financial stress index for EU-27 + UK.
   Returns long DataFrame with columns [iso3, year, fsi].
   Non-EU countries in the estimation panel will have NaN fsi —
   the pooling step assigns them zero weight automatically.
   Note: CLIFS is dimensionless [0,1], unlike Ahir et al. FSI (×1000 scaled).
   """
   print("Fetching CLIFS from ECB Data Portal …")
   records = []
   for cc, iso3 in ECB_TO_ISO3.items():
       series_key = f"M.{cc}._Z.4F.EC.CLIFS_CI.IDX"
       url = f"{BASE}{series_key}?format=csvdata&detail=dataonly"
       try:
           r = requests.get(url, timeout=15)
           if r.status_code != 200:
               print(f"  WARNING: {cc} returned HTTP {r.status_code} — skipping")
               continue
           df = pd.read_csv(io.StringIO(r.text))
           # Column names vary slightly; normalise
           df.columns = [c.strip() for c in df.columns]
           time_col = next(c for c in df.columns if "TIME" in c.upper())
           val_col  = next(c for c in df.columns if "OBS_VALUE" in c.upper()
                           or "VALUE" in c.upper())
           df = df[[time_col, val_col]].copy()
           df.columns = ["period", "fsi"]
           df["fsi"] = pd.to_numeric(df["fsi"], errors="coerce")
           df["date"] = pd.to_datetime(df["period"], format="%Y-%m")
           df["year"] = df["date"].dt.year
           # Require ≥6 monthly obs per year for a valid annual value
           annual = (df.groupby("year")["fsi"]
                       .agg(lambda x: x.mean() if x.notna().sum() >= 6 else float("nan"))
                       .reset_index())
           annual["iso3"] = iso3
           records.append(annual[["iso3", "year", "fsi"]])
           print(f"  {cc} ({iso3}): {annual['fsi'].notna().sum()} annual obs")
       except Exception as e:
           print(f"  WARNING: {cc} failed — {e}")
           continue
   if not records:
       raise RuntimeError("CLIFS fetch returned no data for any country.")
   out = pd.concat(records, ignore_index=True).sort_values(["iso3","year"])
   print(f"CLIFS total: {len(out):,} rows, {out['iso3'].nunique()} countries, "
         f"{int(out['year'].min())}–{int(out['year'].max())}")
   if save:
       out.to_parquet(DATA_DIR / "fsi_raw.parquet", index=False)
       print(f"  Saved → {DATA_DIR / 'fsi_raw.parquet'}")
   return out


def load_fsi() -> pd.DataFrame:
    """Load cached CLIFS data; re-fetch if not present."""
    cache = DATA_DIR / "fsi_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return fetch_fsi(save=True)


if __name__ == "__main__":
    df = fetch_fsi()
    print(df.head(10))
    print("\nG4 sample:")
    print(df[df["iso3"].isin(["FRA", "DEU", "ITA", "ESP"])].tail(12))