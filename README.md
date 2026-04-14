# EU G4 Debt-at-Risk

Full Python implementation of the **Debt-at-Risk (DaR)** methodology from:

> Furceri, Giannone, Kisat, Lam & Li (May 2025). *"Debt-at-Risk"*, IMF Working Paper WP/25/86.

Applied to the **EU G4**: France, Germany, Italy, Spain — producing a board-ready PowerPoint presentation with current debt risk levels and fiscal crisis early-warning signals.

---

## Repository structure

```
eu-debt-at-risk/
├── data/
│   ├── imf_weo.py          # IMF WEO API pull (debt, growth, inflation, balance)
│   ├── imf_fsi.py          # Financial stress: ECB CLIFS (EU-27 + UK)
│   ├── ecb_spreads.py      # 10Y sovereign yields & spreads vs. Bund (Eurostat)
│   ├── wui.py              # World Uncertainty Index (worlduncertaintyindex.com)
│   └── panel_builder.py    # Merge & filter estimation panel → panel.parquet
├── model/
│   ├── location_scale.py   # MSS (2019) three-step quantile estimator
│   └── quantile_fit.py     # Skewed-t distribution fitting
├── risk/
│   ├── pooling.py          # Log-score density combination (Crump et al. 2022)
│   └── dar.py              # P5/P50/P95 extraction + WEO re-centering
├── crisis/
│   └── logit_signal.py     # Fiscal crisis panel logit (Laeven-Valencia)
├── output/
│   ├── charts.py           # Fan charts, waterfall, comparison, crisis signal
│   └── deck.py             # python-pptx 5-slide deck builder
├── notebooks/
│   └── eu_g4_dar.ipynb     # End-to-end walkthrough with inline charts
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd eu-debt-at-risk
pip install -r requirements.txt
```

Python 3.10+ required.

---

## Running the pipeline

### Option A — Jupyter notebook (recommended)
```bash
jupyter lab notebooks/eu_g4_dar.ipynb
```
Run cells top-to-bottom. Each phase caches intermediate output to `data/*.parquet` and `model/*.parquet` so individual phases can be re-run independently.

### Option B — Command line, phase by phase
```bash
# Phase 1: pull all data
python -m data.panel_builder

# Phase 2: run location-scale model
python -m model.location_scale

# Phase 3: fit distributions + pool + extract DaR
python -m model.quantile_fit
python -m risk.pooling
python -m risk.dar

# Phase 4: crisis signal
python -m crisis.logit_signal

# Phase 5: generate charts + deck
python -m output.charts
python -m output.deck
```

### Option C — Full pipeline in one go
```python
from data.panel_builder   import build_panel
from model.location_scale import run_all
from model.quantile_fit   import fit_distributions
from risk.pooling         import compute_country_weights
from risk.dar             import compute_dar
from crisis.logit_signal  import run_logit
from output.charts        import generate_all_charts
from output.deck          import build_deck

panel   = build_panel()
qpreds  = run_all(panel)
skt     = fit_distributions(qpreds)
weights = compute_country_weights(skt, panel, horizon=3)
dar     = compute_dar(skt, weights, horizon=3)
crisis  = run_logit(skt, panel)
charts  = generate_all_charts(panel, dar, skt, crisis["pooled_scores"])
build_deck(dar, crisis["pooled_scores"], charts)
```

---

## Output

| File | Description |
|------|-------------|
| `data/panel.parquet` | Clean estimation panel (~220 countries, 1990–2025) |
| `model/quantile_predictions.parquet` | Predicted debt quantiles Q5/Q25/Q50/Q75/Q95 |
| `model/skt_params.parquet` | Fitted skewed-t parameters (ξ, ω, α, ν) |
| `risk/pooling_weights.parquet` | Country-specific log-score pooling weights |
| `risk/dar_results.parquet` | DaR / Upside / Downside for G4 |
| `crisis/crisis_scores.parquet` | Per-driver crisis probabilities (2025–2026) |
| `output/fig0_global_context.png` | G4 in world debt distribution |
| `output/fig1_fan_charts.png` | Fan charts for all G4 countries |
| `output/fig2_comparison_bar.png` | DaR vs. WEO baseline bar chart |
| `output/fig3_waterfall.png` | Upside risk waterfall decomposition |
| `output/fig4_crisis_signal.png` | Fiscal crisis probability scores |
| `output/eu_g4_debt_at_risk.pptx` | **Board-ready 5-slide PowerPoint deck** |

---

## Current results (h=3, 2027 horizon)

| Country | WEO Baseline | Median P50 | DaR P95 | Upside | Downside |
|---------|-------------|-----------|---------|--------|----------|
| France  | 117.0% | 117.0% | 128.1% | +11.1pp | −15.1pp |
| Germany | 68.5%  | 68.5%  | 84.8%  | +16.3pp | −22.1pp |
| Italy   | 138.0% | 138.0% | 147.2% | +9.2pp  | −12.5pp |
| Spain   | 103.5% | 103.5% | 114.5% | +11.0pp | −14.9pp |

All figures % of GDP. DaR = 95th percentile of the 3-year-ahead debt distribution, re-centred to April 2025 WEO baseline.

---

## Methodology

### Location-scale quantile regression (Phase 2)

Following Machado & Santos Silva (2019), the conditional debt distribution is modelled as:

$$d_{i,t+h} = \alpha_i + X'\beta + (\delta_i + X'\gamma)\,\varepsilon_{i,t+h}$$

**Step 1:** FE-OLS on $d_{i,t+h}$ → residuals $\hat{e}$  
**Step 2:** FE-OLS on $|\hat{e}|$ → fitted scale $\hat{s}$  
**Step 3:** $z = \hat{e}/\hat{s}$; empirical quantile $q(\tau)$; predicted: $Q(\tau) = \hat{\mu}_{i,t} + q(\tau)\cdot\hat{s}_{i,t}$

Conditioning variables run separately per variable: primary balance, GDP growth, CPI inflation, initial debt, financial stress (CLIFS), 10Y sovereign spread, World Uncertainty Index (WUI). Horizons: $h \in \{1, 3, 5\}$.

### Density pooling (Phase 3)

Individual conditioning-variable densities are combined via log-score optimal weights:

$$w^* = \arg\max_{w \geq 0,\,\sum w = 1} \sum_{t \in \mathrm{val}} \log\!\left[\sum_k w_k\, f_k(d_{t+h} \mid X_t)\right]$$

Rolling 20-year out-of-sample window from 2005; weights are country-specific. For EU G4, sovereign spreads carry the dominant financial stress signal — consistent with how eurozone fiscal stress manifested in the 2011–2012 European debt crisis.

### Fiscal crisis logit (Phase 4)

$$\Pr(\text{crisis}_{i,t+1,t+2}) = \Lambda(\beta_0 + \beta_1\,(Q_{95} - Q_{50})_{i,t})$$

Binary crisis indicator from Laeven & Valencia (2020). Estimated separately per conditioning variable; probability scores pooled equally across drivers with positive and significant coefficients (CLIFS β = +0.36, spread β = +1.41).

---

## Data sources

| Variable | Module | Source | Series / endpoint |
|----------|--------|--------|-------------------|
| Debt/GDP, primary balance, GDP growth, CPI inflation | `imf_weo.py` | IMF DataMapper API | `GGXWDG_NGDP`, `GGXCNL_NGDP`, `NGDP_RPCH`, `PCPIPCH` |
| Financial stress (CLIFS) | `imf_fsi.py` | ECB Data Portal | `CLIFS.M.{CC}._Z.4F.EC.CLIFS_CI.IDX` — EU-27 + UK, monthly from 1970, annualised |
| 10Y sovereign yields & spreads | `ecb_spreads.py` | Eurostat | `IRT_LT_MCBY_A` — Maastricht criterion 10Y bond yields, annual, FR/DE/IT/ES from 1990 |
| World Uncertainty Index | `wui.py` | worlduncertaintyindex.com | `WUI_Data.xlsx`, sheet T2 — 143 countries, quarterly from 1990, annualised |
| Fiscal crisis episodes | `logit_signal.py` | Laeven & Valencia (2020) | IMF WP/20/175 |

### Note on the financial stress variable

The original Furceri et al. (2025) paper uses the **Ahir, Dell'Ariccia, Furceri, Papageorgiou & Qi (2023) Financial Stress Index** (FSI) — a text-based measure constructed from EIU country reports covering 110 countries. That dataset is distributed as a one-off file alongside IMF WP/2023/217 and is not available via any live API.

This implementation substitutes the **ECB Country-Level Index of Financial Stress (CLIFS)** for EU/EEA countries. CLIFS is a market-based composite across equity, bond, and FX stress, published monthly by the ECB and freely accessible via their Data Portal API. It covers EU-27 + UK from 1970. For non-EU countries in the estimation panel, the financial stress variable is unavailable and the pooling step assigns those observations zero weight on the FSI dimension, relying on the economic conditioning variables instead.

CLIFS and the Ahir et al. FSI measure related but distinct concepts: CLIFS captures multi-market asset price dislocations; the Ahir et al. FSI captures narrative-based credit supply disruption. For eurozone sovereigns the two series correlate strongly during major stress episodes (GFC, European debt crisis, COVID). The substitution is methodologically conservative — CLIFS covers only 28 countries versus 110 for the original FSI, which reduces the statistical power of the financial stress quantile regression for non-EU panel countries.

---

## References

- Furceri, D., Giannone, D., Kisat, F., Lam, W.R. & Li, H. (2025). *Debt-at-Risk*. IMF Working Paper WP/25/86.
- Machado, J.A.F. & Santos Silva, J.M.C. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145–173.
- Crump, R.K., Everaert, M., Giannone, D. & Hundtofte, S. (2023). Changing Risk-Return Profiles. Federal Reserve Bank of New York Staff Report No. 850.
- Laeven, L. & Valencia, F. (2020). Systemic Banking Crises Database II. *IMF Economic Review*, 68, 307–361.
- Ahir, H., Dell'Ariccia, G., Furceri, D., Papageorgiou, C. & Qi, H. (2023). Financial Stress and Economic Activity: Evidence from a New Worldwide Index. IMF Working Paper WP/2023/217.
- Ahir, H., Bloom, N. & Furceri, D. (2022). The World Uncertainty Index. NBER Working Paper 29763.
- Duprey, T. & Klaus, B. (2015). Dating Systemic Financial Stress Episodes in the EU Countries. ECB Working Paper No. 1873.