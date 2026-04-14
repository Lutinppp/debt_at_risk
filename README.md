# EU G4 Debt-at-Risk

A complete Python implementation of the **Debt-at-Risk (DaR)** methodology
from [IMF Working Paper WP/25/86](https://www.imf.org/en/Publications/WP/Issues/2025/05/)
(Furceri, Giannone, Kisat, Lam, Li — May 2025), applied to the EU G4
(France, Germany, Italy, Spain).

The project produces a board-ready PowerPoint presentation showing:
1. Current debt risk levels (P5 / P50 / P95) for each G4 country
2. Fiscal crisis early-warning signals for 2025–2026

---

## Repository structure

```
eu-debt-at-risk/
├── data/
│   ├── imf_weo.py        # WEO API pull (debt, primary balance, growth, CPI)
│   ├── imf_fsi.py        # IMF Financial Stress Index pull
│   ├── ecb_spreads.py    # ECB SDW 10Y sovereign spreads vs. Bund
│   ├── wui.py            # World Uncertainty Index CSV ingest
│   └── pipeline.py       # Merge + filter → data/panel.parquet
├── model/
│   ├── location_scale.py # MSS (2019) three-step quantile estimator
│   └── quantile_fit.py   # Azzalini-Capitanio skewed-t fitting
├── risk/
│   ├── pooling.py        # Log-score density combination (Crump et al. 2023)
│   └── dar.py            # P5/P50/P95 extraction + WEO re-centering
├── crisis/
│   └── logit_signal.py   # Fiscal crisis panel logit signal
├── output/
│   ├── charts.py         # Fan charts, waterfall, DaR comparison, crisis
│   └── deck.py           # python-pptx 5-slide deck builder
├── notebooks/
│   └── eu_g4_dar.ipynb   # End-to-end walkthrough notebook
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Lutinppp/debt_at_risk.git
cd debt_at_risk

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the pipeline

### Option A — Jupyter Notebook (recommended)

```bash
jupyter notebook notebooks/eu_g4_dar.ipynb
```

The notebook runs the complete pipeline end-to-end with inline charts.

### Option B — Command line, phase by phase

```bash
# Phase 1: Data
python -m data.imf_weo          # → data/panel.parquet (WEO)
python -m data.imf_fsi          # → data/fsi.parquet
python -m data.ecb_spreads      # → data/ecb_spreads.parquet
python -m data.wui              # → data/wui.parquet  (needs WUI_Data.xlsx)
python -m data.pipeline         # → data/panel.parquet (merged)

# Phase 2: Model
python -m model.location_scale  # → model/quantile_predictions.parquet
python -m model.quantile_fit    # → model/density_params.parquet

# Phase 3: Risk
python -m risk.pooling          # → risk/pooled_weights.parquet
                                #   risk/pooled_density.parquet
python -m risk.dar              # → risk/dar.parquet

# Phase 4: Crisis signal
python -m crisis.logit_signal   # → crisis/crisis_scores.parquet

# Phase 5: Output
python -m output.charts         # → output/charts/*.png
python -m output.deck           # → output/eu_g4_debt_at_risk.pptx
```

---

## Data sources

| Dataset | Source | API / URL |
|---------|--------|-----------|
| WEO (debt, growth, CPI, primary balance) | IMF DataMapper | `https://www.imf.org/external/datamapper/api/v1/` |
| Financial Stress Index (FSI) | IMF (Ahir et al. 2023) | `https://www.imf.org/external/datamapper/api/v1/FSI` |
| 10-year sovereign bond yields | ECB SDW | `https://data-api.ecb.europa.eu/service/data/IRS/` |
| World Uncertainty Index (WUI) | Ahir, Bloom & Furceri (2018) | `https://worlduncertaintyindex.com/data/` |
| Fiscal crisis episodes | Laeven & Valencia (2020) | IMF WP/20/162 |

> **Note on WUI**: Download `WUI_Data.xlsx` from https://worlduncertaintyindex.com/data/
> and place it in the `data/` folder before running `data/wui.py`.  The module
> will also attempt an automatic download; if blocked, the WUI variable is
> gracefully excluded from the pooled density.

---

## Methodology

### Location-scale quantile regression (Phase 2)

Implements the **Machado-Santos Silva (2019)** three-step estimator:

```
d_{i,t+h} = α_i + X'β + (δ_i + X'γ) ε_{i,t+h}
```

1. **Step 1** — Fixed-effects OLS: `d_{i,t+h}` on `[X_{i,t}, d_{i,t}]` → residuals `ê`
2. **Step 2** — Fixed-effects OLS: `|ê|` on `[X_{i,t}, d_{i,t}]` → fitted scale `ŝ`
3. **Step 3** — Standardised residuals `z = ê/ŝ`; empirical quantiles `q(τ)`
   → `Q̂(τ) = fitted_mean + ŝ · q(τ)`

Each conditioning variable is run separately for horizons h ∈ {1, 3, 5}.

### Density fitting (Phase 2)

The five predicted quantiles `{Q05, Q25, Q50, Q75, Q95}` are fitted to an
**Azzalini-Capitanio skewed-t** distribution via `scipy.optimize.minimize`
(Nelder-Mead, minimising squared quantile distances).

### Log-score pooling (Phase 3)

Individual conditioning-variable densities are combined via
**log-score weighting** (Crump et al. 2023):

- 20-year rolling window from 2005
- Country-specific weights constrained to the unit simplex
- Solved with `scipy.optimize.minimize` (SLSQP)

### Debt-at-Risk (Phase 3)

- **DaR** = P95 of the pooled distribution (3-year horizon)
- **Upside** = P95 − P50
- **Downside** = P50 − P5
- Median re-centred to IMF WEO April 2025 baseline projections

### Fiscal crisis signal (Phase 4)

Panel logit with the IMF/Laeven-Valencia crisis database:

```
P(crisis_{i,t+1}) = Λ(α + β · Upside_{k,i,t})
```

Run separately for each conditioning variable; averaged for the headline score.

---

## Output

| File | Description |
|------|-------------|
| `output/charts/fan_charts.png` | 2×2 fan chart grid for G4 |
| `output/charts/dar_comparison.png` | DaR vs. WEO baseline bar chart |
| `output/charts/waterfall_*.png` | Risk decomposition per country |
| `output/charts/crisis_signals.png` | Crisis probability scores |
| `output/eu_g4_debt_at_risk.pptx` | 5-slide board deck |

---

## References

- Furceri, D., Giannone, D., Kisat, M., Lam, W.R., & Li, B. (2025).
  *Debt-at-Risk: A Framework for Assessing Fiscal Vulnerabilities.*
  IMF Working Paper WP/25/86.
- Machado, J.A.F. & Santos Silva, J.M.C. (2019).
  *Quantiles via Moments.* Journal of Econometrics, 213(1), 145–173.
- Azzalini, A. & Capitanio, A. (2003).
  *Distributions generated by perturbation of symmetry.*
  Journal of the Royal Statistical Society B, 65(2), 367–389.
- Laeven, L. & Valencia, F. (2020).
  *Systemic Banking Crises Database II.* IMF WP/20/162.
- Ahir, H., Bloom, N. & Furceri, D. (2018, updated).
  *The World Uncertainty Index.* NBER Working Paper 29763.
- Crump, R.K., Eusepi, S., Giannoni, M. & Şahin, A. (2023).
  *The Term Structure of Expectations.* FRB NY Staff Report No. 992.
