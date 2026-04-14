# EU G4 Debt-at-Risk

Full Python implementation of the **Debt-at-Risk (DaR)** methodology from:

> Furceri, Giannone, Kisat, Lam & Li (May 2025). *"Debt-at-Risk: A Probabilistic Framework for Public Debt Sustainability Analysis"*, IMF Working Paper WP/25/86.

Applied to the **EU G4**: France, Germany, Italy, Spain — producing a board-ready PowerPoint presentation with current debt risk levels and fiscal crisis early-warning signals.

---

## Repository structure

```
eu-debt-at-risk/
├── data/
│   ├── imf_weo.py          # IMF WEO API pull (debt, growth, inflation, balance)
│   ├── imf_fsi.py          # Financial Stress Index (IMF API / proxy)
│   ├── ecb_spreads.py      # 10Y sovereign spreads vs. Bund (ECB SDW)
│   ├── wui.py              # World Uncertainty Index ingest
│   └── panel_builder.py    # Merge & filter estimation panel → panel.parquet
├── model/
│   ├── location_scale.py   # MSS (2019) three-step quantile estimator
│   └── quantile_fit.py     # Azzalini-Capitanio skewed-t fitting
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
# From project root
import sys; sys.path.insert(0, ".")
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
| `data/panel.parquet` | Clean estimation panel (~40–60 countries, 1990–2024) |
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

## Methodology note

### Location-scale quantile regression (Phase 2)

Following Machado & Santos Silva (2019), the conditional debt distribution is modelled as:

$$d_{i,t+h} = \alpha_i + X'\beta + (\delta_i + X'\gamma)\,\varepsilon_{i,t+h}$$

**Step 1:** FE-OLS on $d_{i,t+h}$ → residuals $\hat{e}$  
**Step 2:** FE-OLS on $|\hat{e}|$ → fitted scale $\hat{s}$  
**Step 3:** $z = \hat{e}/\hat{s}$; empirical quantile $q(\tau)$; predicted: $Q(\tau) = \hat{\mu}_{i,t} + q(\tau)\cdot\hat{s}_{i,t}$

Conditioning variables: primary balance, GDP growth, inflation, initial debt, financial stress (FSI), sovereign spread, World Uncertainty Index (WUI).  
Horizons: $h \in \{1, 3, 5\}$.

### Density pooling (Phase 3)

Individual conditioning-variable densities are combined via log-score optimal weights:

$$w^* = \arg\max_{w \geq 0,\,\sum w = 1} \sum_{t \in \mathrm{val}} \log\!\left[\sum_k w_k\, f_k(d_{t+h} \mid X_t)\right]$$

Rolling 20-year out-of-sample window from 2005; country-specific weights.

### Fiscal crisis logit (Phase 4)

$$\Pr(\text{crisis}_{i,t+1,t+2}) = \Lambda(\beta_0 + \beta_1 (Q_{95} - Q_{50})_{i,t})$$

Binary crisis indicator from Laeven & Valencia (2020). Estimated separately per conditioning variable; probability scores pooled equally across drivers.

---

## Data sources

| Source | Data | URL / Reference |
|--------|------|-----------------|
| IMF DataMapper API | Debt/GDP, primary balance, GDP growth, inflation | `https://www.imf.org/external/datamapper/api/v1/` |
| IMF FSI | Financial Stress Index (Ahir et al. 2023) | IMF DataMapper API |
| ECB SDW | 10Y sovereign bond yields (FR, DE, IT, ES) | `https://data-api.ecb.europa.eu/service/data/` |
| WUI | World Uncertainty Index (Ahir, Bloom, Furceri 2023) | `https://worlduncertaintyindex.com` |
| Laeven & Valencia (2020) | Fiscal crisis episodes | IMF WP/20/175 |

---

## References

- Furceri, D., Giannone, D., Kisat, F., Lam, W.R. & Li, C. (2025). *Debt-at-Risk: A Probabilistic Framework for Public Debt Sustainability Analysis*. IMF Working Paper WP/25/86.
- Machado, J.A.F. & Santos Silva, J.M.C. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145–173.
- Crump, R.K., Eusepi, S., Giannoni, M. & Sahin, A. (2022). A Large Bayesian VAR of the United States Economy. *Federal Reserve Bank of New York Staff Report*, No. 976.
- Laeven, L. & Valencia, F. (2020). Systemic Banking Crises Database II. *IMF Economic Review*, 68, 307–361.
- Ahir, H., Bloom, N. & Furceri, D. (2023). The World Uncertainty Index. *NBER Working Paper* 29763.
Analysis using Furceri's 2025 paper framework and IMF data 
