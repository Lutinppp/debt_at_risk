"""
Script to generate the eu_g4_dar.ipynb notebook programmatically.
Run: python notebooks/generate_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

# ── Metadata ─────────────────────────────────────────────────────────────────
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
}

cells = []

def md(source):
    return nbf.v4.new_markdown_cell(source)

def code(source):
    return nbf.v4.new_code_cell(source)


# ── Cell 0: Title ────────────────────────────────────────────────────────────
cells.append(md("""# EU G4 Debt-at-Risk — End-to-End Walkthrough

**Methodology:** Furceri, Giannone, Kisat, Lam & Li (May 2025), IMF WP/25/86  
**Countries:** France (FRA), Germany (DEU), Italy (ITA), Spain (ESP)  
**Horizon:** 3-year ahead (2027)  

This notebook runs the full pipeline:
1. Data pull (WEO, FSI, ECB spreads, WUI)
2. Location-scale quantile regression (MSS 2019)
3. Skewed-t distribution fitting
4. Log-score density pooling
5. Debt-at-Risk extraction (P5/P50/P95)
6. Fiscal crisis early-warning logit
7. Charts and PowerPoint deck
"""))

# ── Cell 1: Setup ────────────────────────────────────────────────────────────
cells.append(code("""\
import sys, os
# Ensure project root on path
ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.dpi"] = 120

print("Python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
"""))

# ── Cell 2: Phase 1 — Data Pipeline ─────────────────────────────────────────
cells.append(md("""## Phase 1 — Data Pipeline

Pull macro-fiscal data from IMF DataMapper API, ECB SDW, and WUI.
All data cached to `data/*.parquet` for reproducibility.
"""))

cells.append(code("""\
from data.imf_weo    import fetch_weo
from data.imf_fsi    import fetch_fsi
from data.ecb_spreads import fetch_spreads
from data.wui        import fetch_wui

print("=== WEO ===")
weo = fetch_weo(save=True)
print(f"WEO: {weo.shape[0]:,} rows, {weo['iso3'].nunique()} countries")
print(weo.head(5))
"""))

cells.append(code("""\
print("=== FSI ===")
fsi = fetch_fsi(save=True)
print(f"FSI: {fsi.shape[0]:,} rows, {fsi['iso3'].nunique()} countries")
print(fsi.head(5))
"""))

cells.append(code("""\
print("=== ECB Sovereign Spreads ===")
spreads = fetch_spreads(save=True)
print(spreads)
"""))

cells.append(code("""\
print("=== World Uncertainty Index ===")
wui = fetch_wui(save=True)
print(f"WUI: {wui.shape[0]:,} rows, {wui['iso3'].nunique()} countries")
print(wui.head(5))
"""))

cells.append(code("""\
from data.panel_builder import build_panel

print("=== Building Estimation Panel ===")
panel = build_panel(min_obs=20)
print(panel.describe())
"""))

cells.append(code("""\
# Quick QA: coverage for G4
G4 = ["FRA", "DEU", "ITA", "ESP"]
for iso3 in G4:
    sub = panel[panel["iso3"] == iso3]
    nobs = sub[["debt_gdp", "primary_balance_gdp", "rgdp_growth"]].notna().all(axis=1).sum()
    print(f"{iso3}: {nobs} complete core observations  "
          f"(debt range: {sub['debt_gdp'].min():.0f}–{sub['debt_gdp'].max():.0f}% GDP)")
"""))

# ── Cell 3: Visualise Historical Debt ────────────────────────────────────────
cells.append(md("""### Historical Debt/GDP trajectories for G4"""))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"FRA": "#3498DB", "DEU": "#27AE60", "ITA": "#E74C3C", "ESP": "#F39C12"}
labels = {"FRA": "France", "DEU": "Germany", "ITA": "Italy", "ESP": "Spain"}

for iso3 in G4:
    hist = panel[(panel["iso3"] == iso3) & (panel["year"] >= 1995)].sort_values("year")
    ax.plot(hist["year"], hist["debt_gdp"], lw=2.2,
            color=colors[iso3], label=labels[iso3])

ax.axhline(60, color="gold", lw=1.2, ls="--", alpha=0.8, label="Maastricht 60%")
ax.set_title("Gross Government Debt/GDP — EU G4 (1995–2024)", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("% of GDP")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.show()
"""))

# ── Cell 4: Phase 2 — Location-Scale Model ───────────────────────────────────
cells.append(md("""## Phase 2 — Location-Scale Quantile Regression (MSS 2019)

The Machado-Santos Silva three-step estimator:

$$d_{i,t+h} = \\alpha_i + X'\\beta + (\\delta_i + X'\\gamma)\\,\\varepsilon_{i,t+h}$$

Three steps:
1. FE-OLS on level → residuals $\\hat{e}$
2. FE-OLS on $|\\hat{e}|$ → scale $\\hat{s}$
3. $z = \\hat{e}/\\hat{s}$; empirical quantiles; predicted $Q(\\tau) = \\hat{\\mu} + q_z(\\tau)\\cdot\\hat{s}$
"""))

cells.append(code("""\
from model.location_scale import run_all

print("Running MSS quantile regression (all conditioning variables × horizons) …")
print("This may take several minutes …")
qpreds = run_all(panel, horizons=[1, 3, 5])
print(f"\\nQuantile predictions: {qpreds.shape[0]:,} rows")
print(qpreds.head(10))
"""))

cells.append(code("""\
# Check predictions for G4
print("Sample quantile predictions for France (h=3):")
print(qpreds[(qpreds["iso3"] == "FRA") & (qpreds["horizon"] == 3)].tail(10))
"""))

# ── Cell 5: Phase 3a — Skewed-t Fitting ──────────────────────────────────────
cells.append(md("""## Phase 3a — Skewed-t Distribution Fitting

Fit Azzalini-Capitanio skewed-t parameters (ξ, ω, α, ν) to each set of five predicted quantiles
{Q5, Q25, Q50, Q75, Q95} via least-squares minimisation.
"""))

cells.append(code("""\
from model.quantile_fit import fit_distributions

print("Fitting skewed-t distributions …")
skt_params = fit_distributions(qpreds)
print(f"SKT params: {skt_params.shape[0]:,} rows")
print(skt_params[skt_params["iso3"] == "FRA"].head(8))
"""))

cells.append(code("""\
# Visualise fitted distributions for Italy h=3, one conditioning variable
from model.quantile_fit import _fst_quantile
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Italy — Fitted Skewed-t Distributions (h=3)", fontweight="bold")

cond_vars_show = ["primary_balance", "rgdp_growth", "fsi", "spread_10y"]
x_grid = np.linspace(60, 200, 300)

for ax, cv in zip(axes, cond_vars_show):
    sub = skt_params[
        (skt_params["iso3"] == "ITA") &
        (skt_params["horizon"] == 3) &
        (skt_params["cond_var"] == cv)
    ]
    if sub.empty: continue
    r = sub.sort_values("year").iloc[-1]
    if any(pd.isna([r["xi"], r["omega"], r["alpha"], r["nu"]])): continue

    from scipy.stats import t as t_dist
    gamma = np.exp(r["alpha"])
    omega, nu = r["omega"], r["nu"]
    xi    = r["xi"]

    def pdf(x):
        z = (x - xi) / omega
        c = 2.0 / (omega * (gamma + 1.0 / gamma))
        if z < 0:
            return c * t_dist.pdf(-z / gamma, df=nu) / gamma
        else:
            return c * t_dist.pdf(z * gamma, df=nu) * gamma

    y_pdf = np.array([pdf(xv) for xv in x_grid])
    ax.plot(x_grid, y_pdf, color="#1B2A4A", lw=2)
    ax.fill_between(x_grid, y_pdf, alpha=0.2, color="#C8A951")
    ax.axvline(_fst_quantile(0.05, xi, omega, r["alpha"], nu),
               color="blue", lw=1, ls="--", label="P5")
    ax.axvline(_fst_quantile(0.50, xi, omega, r["alpha"], nu),
               color="black", lw=1.5, ls="-", label="P50")
    ax.axvline(_fst_quantile(0.95, xi, omega, r["alpha"], nu),
               color="red", lw=1, ls="--", label="P95")
    ax.set_title(cv.replace("_", " ").title(), fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.show()
"""))

# ── Cell 6: Phase 3b — Log-Score Pooling ─────────────────────────────────────
cells.append(md("""## Phase 3b — Log-Score Density Pooling

Optimal mixture weights computed via rolling out-of-sample log-score maximisation
(Crump et al. 2022), country-specific, from 2005 onward.

$$w^* = \\arg\\max_{w\\geq 0,\\,\\sum w=1} \\sum_{t \\in \\text{val}} \\log\\left[\\sum_k w_k f_k(d_{t+h}|X_t)\\right]$$
"""))

cells.append(code("""\
from risk.pooling import compute_country_weights

print("Computing log-score pooling weights (h=3) …")
weights = compute_country_weights(skt_params, panel, horizon=3)

# Show G4 weights
print("\\nPooling weights for G4:")
print(weights[weights["iso3"].isin(["FRA","DEU","ITA","ESP"])]
      .pivot(index="iso3", columns="cond_var", values="weight")
      .round(3))
"""))

cells.append(code("""\
# Visualise pooling weights for G4 as stacked bars
g4_w = weights[weights["iso3"].isin(G4)].pivot(
    index="iso3", columns="cond_var", values="weight"
).reindex(G4)

driver_colors = {
    "primary_balance": "#E74C3C", "rgdp_growth": "#3498DB",
    "cpi_inflation": "#F39C12",   "initial_debt": "#9B59B6",
    "fsi": "#1ABC9C", "spread_10y": "#E67E22", "wui": "#2ECC71",
}

fig, ax = plt.subplots(figsize=(9, 4))
bottom = np.zeros(4)
x = np.arange(4)
labels_g4 = [{"FRA":"France","DEU":"Germany","ITA":"Italy","ESP":"Spain"}[c] for c in G4]

for cv in g4_w.columns:
    vals = g4_w[cv].fillna(0).values
    ax.bar(x, vals, bottom=bottom, color=driver_colors.get(cv, "grey"),
           label=cv.replace("_"," ").title(), alpha=0.9)
    bottom += vals

ax.set_xticks(x); ax.set_xticklabels(labels_g4)
ax.set_ylabel("Pooling weight")
ax.set_title("Log-Score Pooling Weights by Driver — G4 (h=3)", fontweight="bold")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.show()
"""))

# ── Cell 7: DaR Extraction ────────────────────────────────────────────────────
cells.append(md("""## Phase 3c — Debt-at-Risk (P5 / P50 / P95)

Extract quantiles from the pooled distribution. Re-center median to IMF WEO
April 2025 baseline projections for 2027.
"""))

cells.append(code("""\
from risk.dar import compute_dar, WEO_BASELINE_2027

print("Computing Debt-at-Risk …")
dar = compute_dar(skt_params, weights, horizon=3, recenter=True)

print("\\nDebt-at-Risk results:")
print(dar[["iso3","Q05","Q50","Q95","DaR","Upside","Downside","weo_baseline"]].to_string(index=False))
"""))

cells.append(code("""\
# DaR summary visualisation
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(G4))
labels_g4 = ["France", "Germany", "Italy", "Spain"]
width = 0.22

q05_vals = [dar[dar["iso3"]==c]["Q05"].values[0] if not dar[dar["iso3"]==c].empty else 0 for c in G4]
q50_vals = [dar[dar["iso3"]==c]["Q50"].values[0] if not dar[dar["iso3"]==c].empty else 0 for c in G4]
q95_vals = [dar[dar["iso3"]==c]["Q95"].values[0] if not dar[dar["iso3"]==c].empty else 0 for c in G4]
weo_vals = [WEO_BASELINE_2027.get(c, 0) for c in G4]

ax.bar(x - width*1.5, weo_vals, width, label="WEO Baseline", color="#27AE60", alpha=0.85)
ax.bar(x - width*0.5, q50_vals, width, label="P50 (Model)", color="#7BAFD4", alpha=0.85)
ax.bar(x + width*0.5, q95_vals, width, label="DaR (P95)",   color="#C0392B", alpha=0.85)
ax.bar(x + width*1.5, q05_vals, width, label="P5",          color="#BDC3C7", alpha=0.85)

ax.axhline(60, color="gold", lw=1.2, ls="--", alpha=0.8, label="Maastricht 60%")
ax.set_xticks(x); ax.set_xticklabels(labels_g4)
ax.set_ylabel("Gross Govt Debt (% GDP)")
ax.set_title("EU G4 — Debt-at-Risk (2027 Horizon, h=3)", fontweight="bold")
ax.legend(framealpha=0.8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.show()
"""))

# ── Cell 8: Phase 4 — Crisis Signal ──────────────────────────────────────────
cells.append(md("""## Phase 4 — Fiscal Crisis Early-Warning Logit

$$\\Pr(\\text{crisis}_{i,t+1,t+2}) = \\Lambda\\!\\left(\\beta_0 + \\beta_1 \\cdot (Q_{95} - Q_{50})_{i,t}\\right)$$

Estimated separately for each conditioning variable. Binary crisis indicator from
Laeven & Valencia (2020).
"""))

cells.append(code("""\
from crisis.logit_signal import run_logit

print("Running fiscal crisis logit models …")
crisis_out = run_logit(skt_params, panel, horizon=3, forecast_years=[2025, 2026])

print("\\nLogit results summary:")
for cv, res in crisis_out["logit_results"].items():
    coef_upside = res.params[1] if len(res.params) > 1 else float("nan")
    pval        = res.pvalues[1] if len(res.pvalues) > 1 else float("nan")
    print(f"  {cv:20s}  β_upside = {coef_upside:+.4f}  (p={pval:.3f})")
"""))

cells.append(code("""\
# G4 crisis probability scores
pooled = crisis_out["pooled_scores"]
print("\\nG4 Fiscal Crisis Probability Scores:")
print(pooled[pooled["iso3"].isin(G4)].to_string(index=False))

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, yr in zip(axes, [2025, 2026]):
    sub = pooled[(pooled["iso3"].isin(G4)) & (pooled["year"] == yr)].copy()
    if sub.empty:
        ax.set_title(str(yr))
        continue
    sub = sub.set_index("iso3").reindex(G4)
    probs = sub["crisis_prob_pooled"].fillna(0).values * 100
    labels_g4 = ["France", "Germany", "Italy", "Spain"]
    bar_colors = ["#C0392B" if p > 15 else "#C8A951" if p > 8 else "#7BAFD4"
                  for p in probs]
    ax.barh(labels_g4, probs, color=bar_colors, edgecolor="#1B2A4A", linewidth=0.5)
    ax.axvline(8,  color="#C8A951", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(15, color="#C0392B", lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel("Crisis probability (%)")
    ax.set_title(f"Fiscal Crisis Signal — {yr}", fontweight="bold")
    ax.set_xlim(0, max(probs.max() + 5, 25))
    ax.spines[["top","right"]].set_visible(False)
    ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

# ── Cell 9: Phase 5 — Full Chart Suite ───────────────────────────────────────
cells.append(md("""## Phase 5 — Generate All Charts"""))

cells.append(code("""\
from output.charts import generate_all_charts

chart_paths = generate_all_charts(panel, dar, skt_params, pooled)
for name, path in chart_paths.items():
    print(f"  {name}: {path}")
"""))

cells.append(code("""\
# Display fan charts inline
from IPython.display import Image, display
fan_path = chart_paths.get("fan_charts")
if fan_path and fan_path.exists():
    display(Image(filename=str(fan_path), width=950))
"""))

cells.append(code("""\
# Display waterfall
wf_path = chart_paths.get("waterfall")
if wf_path and wf_path.exists():
    display(Image(filename=str(wf_path), width=950))
"""))

# ── Cell 10: Build PPTX ───────────────────────────────────────────────────────
cells.append(md("""## Build PowerPoint Deck"""))

cells.append(code("""\
from output.deck import build_deck

deck_path = build_deck(dar, pooled, chart_paths)
print(f"\\nDeck saved to: {deck_path}")
"""))

# ── Cell 11: Summary Table ────────────────────────────────────────────────────
cells.append(md("""## Summary — EU G4 Debt-at-Risk Results

| Country | WEO Baseline | Median (P50) | DaR (P95) | Upside | Downside |
|---------|-------------|-------------|-----------|--------|----------|
"""))

cells.append(code("""\
summary = dar[["iso3","weo_baseline","Q50","DaR","Upside","Downside"]].copy()
summary.columns = ["Country","WEO Baseline","Median (P50)","DaR (P95)","Upside (+pp)","Downside (−pp)"]
summary["Country"] = summary["Country"].map(
    {"FRA":"France","DEU":"Germany","ITA":"Italy","ESP":"Spain"}
)
print(summary.to_string(index=False))
"""))

nb.cells = cells

out = Path("notebooks/eu_g4_dar.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
import nbformat
with open(str(out), "w") as f:
    nbformat.write(nb, f)

print(f"Notebook written → {out}")
