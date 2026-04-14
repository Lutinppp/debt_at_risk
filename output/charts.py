"""
Chart generation for EU G4 Debt-at-Risk presentation.

Charts produced:
  1. Fan charts per country — historical debt/GDP + P5/P50/P95 fan to 2027
  2. Country comparison bar chart — DaR vs. WEO baseline for FR, DE, IT, ES
  3. Waterfall chart — upside risk decomposition by driver, per country
  4. Crisis signal chart — fiscal crisis probability scores ranked across G4

All charts saved to output/ as .png (300 dpi).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ── Palette (institutional navy / gold) ──────────────────────────────────────
NAVY   = "#1B2A4A"
GOLD   = "#C8A951"
LIGHT_BLUE = "#7BAFD4"
RED    = "#C0392B"
GREEN  = "#27AE60"
GREY   = "#BDC3C7"
WHITE  = "#FFFFFF"

G4_LABELS = {"FRA": "France", "DEU": "Germany", "ITA": "Italy", "ESP": "Spain"}
G4_ORDER  = ["FRA", "DEU", "ITA", "ESP"]

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor": NAVY,
    "xtick.color":    NAVY,
    "ytick.color":    NAVY,
    "text.color":     NAVY,
    "figure.facecolor": WHITE,
    "axes.facecolor":   WHITE,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. FAN CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fan_charts(
    panel: pd.DataFrame,
    dar: pd.DataFrame,
    skt_params: pd.DataFrame,
    horizon: int = 3,
    hist_start: int = 2000,
) -> Path:
    """
    2×2 grid of debt-fan charts, one per G4 country.
    Historical line + P5/P50/P95 shaded fans to 2027.
    """
    from model.quantile_fit import _fst_quantile
    from risk.pooling import compute_country_weights

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "EU G4 — Government Debt/GDP: Historical & Debt-at-Risk Fan (2024–2027)",
        fontsize=14, fontweight="bold", color=NAVY, y=1.01,
    )

    proj_years = list(range(2024, 2024 + horizon + 1))

    for ax, iso3 in zip(axes.flat, G4_ORDER):
        # Historical series
        hist = panel[(panel["iso3"] == iso3) & (panel["year"] >= hist_start)].copy()
        hist = hist.sort_values("year")

        ax.plot(hist["year"], hist["debt_gdp"],
                color=NAVY, lw=2.0, label="Historical", zorder=5)

        # Get DaR row
        dar_row = dar[dar["iso3"] == iso3]
        if dar_row.empty:
            ax.set_title(G4_LABELS[iso3])
            continue
        dr = dar_row.iloc[0]

        # WEO anchor point (2024 actual)
        weo_2024 = float(hist[hist["year"] == hist["year"].max()]["debt_gdp"].values[0]) \
                   if not hist.empty else dr["Q50"]

        # Build fan from Q05 / Q50 / Q95
        # Linear interpolation from 2024 anchor to 2027 projected quantiles
        y_q05 = np.linspace(weo_2024, dr["Q05"], len(proj_years))
        y_q50 = np.linspace(weo_2024, dr["Q50"], len(proj_years))
        y_q95 = np.linspace(weo_2024, dr["Q95"], len(proj_years))

        # Fan shading
        ax.fill_between(proj_years, y_q05, y_q95,
                        color=GOLD, alpha=0.25, label="P5–P95 fan", zorder=2)
        ax.fill_between(proj_years, y_q05, y_q50,
                        color=LIGHT_BLUE, alpha=0.30, label="P5–P50", zorder=3)

        ax.plot(proj_years, y_q50, color=GOLD, lw=2.0, ls="--",
                label="Median (P50)", zorder=4)
        ax.plot(proj_years, y_q95, color=RED,  lw=1.5, ls=":",
                label="DaR (P95)", zorder=4)

        # WEO baseline marker
        ax.axhline(dr["weo_baseline"], color=GREEN, lw=1.2, ls="-.",
                   alpha=0.8, label="WEO baseline 2027")

        # Vertical divider between history and projection
        ax.axvline(2024, color=GREY, lw=1.0, ls="--", zorder=1)
        ax.text(2024.1, ax.get_ylim()[1] * 0.95, "Projection →",
                fontsize=7, color=GREY)

        ax.set_title(G4_LABELS[iso3], color=NAVY)
        ax.set_xlabel("Year")
        ax.set_ylabel("% of GDP")
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.annotate(
            f"DaR {int(dr['proj_year'])}: {dr['DaR']:.1f}%",
            xy=(proj_years[-1], dr["Q95"]),
            xytext=(-40, 8), textcoords="offset points",
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
        )

    out_path = OUTPUT_DIR / "fig1_fan_charts.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. COUNTRY COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def comparison_bar(dar: pd.DataFrame) -> Path:
    """Grouped bar: DaR (P95) vs. WEO baseline vs. P50 for G4."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title(
        "EU G4 — Debt-at-Risk vs. WEO Baseline (2027 Horizon)",
        fontweight="bold", color=NAVY,
    )

    x      = np.arange(len(G4_ORDER))
    width  = 0.25
    labels = [G4_LABELS[c] for c in G4_ORDER]

    weo_vals = []
    q50_vals = []
    q95_vals = []
    for iso3 in G4_ORDER:
        row = dar[dar["iso3"] == iso3]
        if row.empty:
            weo_vals.append(0); q50_vals.append(0); q95_vals.append(0)
        else:
            r = row.iloc[0]
            weo_vals.append(r["weo_baseline"])
            q50_vals.append(r["Q50"])
            q95_vals.append(r["Q95"])

    bar1 = ax.bar(x - width, weo_vals, width, label="WEO baseline (Apr 2025)",
                  color=GREEN, alpha=0.85, edgecolor=NAVY, linewidth=0.5)
    bar2 = ax.bar(x,          q50_vals, width, label="Median projection (P50)",
                  color=LIGHT_BLUE, alpha=0.85, edgecolor=NAVY, linewidth=0.5)
    bar3 = ax.bar(x + width,  q95_vals, width, label="Debt-at-Risk (P95)",
                  color=RED, alpha=0.85, edgecolor=NAVY, linewidth=0.5)

    # Value labels
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8, color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=NAVY)
    ax.set_ylabel("Gross Govt Debt (% of GDP)")
    ax.legend(framealpha=0.8)

    # Maastricht reference line at 60%
    ax.axhline(60, color=GOLD, lw=1.2, ls="--", alpha=0.8)
    ax.text(3.6, 61.5, "Maastricht 60%", fontsize=7.5, color=GOLD)

    out_path = OUTPUT_DIR / "fig2_comparison_bar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. WATERFALL CHART — upside risk decomposition
# ─────────────────────────────────────────────────────────────────────────────

DRIVER_LABELS = {
    "primary_balance": "Primary\nBalance",
    "rgdp_growth":     "GDP\nGrowth",
    "cpi_inflation":   "Inflation",
    "initial_debt":    "Initial\nDebt",
    "fsi":             "Financial\nStress",
    "spread_10y":      "Sovereign\nSpread",
    "wui":             "Uncertainty\n(WUI)",
}
DRIVER_COLORS = {
    "primary_balance": "#E74C3C",
    "rgdp_growth":     "#3498DB",
    "cpi_inflation":   "#F39C12",
    "initial_debt":    "#9B59B6",
    "fsi":             "#1ABC9C",
    "spread_10y":      "#E67E22",
    "wui":             "#2ECC71",
}


def waterfall_charts(dar: pd.DataFrame) -> Path:
    """
    2×2 waterfall decomposition of upside risk (DaR - P50) by driver.
    """
    driver_cols = [f"upside_{d}" for d in DRIVER_LABELS]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "EU G4 — Decomposition of Upside Debt Risk by Driver (2027 Horizon)",
        fontsize=14, fontweight="bold", color=NAVY, y=1.01,
    )

    for ax, iso3 in zip(axes.flat, G4_ORDER):
        row = dar[dar["iso3"] == iso3]
        if row.empty:
            ax.set_title(G4_LABELS[iso3])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=GREY)
            continue

        r = row.iloc[0]
        values, labels, colors = [], [], []

        for d_key, d_label in DRIVER_LABELS.items():
            col = f"upside_{d_key}"
            if col in r and pd.notna(r[col]) and r[col] > 0.01:
                values.append(float(r[col]))
                labels.append(d_label)
                colors.append(DRIVER_COLORS[d_key])

        if not values:
            ax.set_title(G4_LABELS[iso3])
            ax.text(0.5, 0.5, "No driver data", ha="center", va="center",
                    transform=ax.transAxes, color=GREY)
            continue

        # Sort descending
        order = np.argsort(values)[::-1]
        values = [values[i] for i in order]
        labels = [labels[i] for i in order]
        colors = [colors[i] for i in order]

        # Waterfall: cumulative running total
        running = [0.0]
        for v in values:
            running.append(running[-1] + v)

        x_pos = np.arange(len(values) + 1)  # +1 for total bar

        # Individual driver bars (stacked waterfall style)
        for i, (v, c) in enumerate(zip(values, colors)):
            ax.bar(i, v, bottom=running[i], color=c, edgecolor=NAVY,
                   linewidth=0.5, alpha=0.9)
            ax.text(i, running[i] + v / 2, f"+{v:.1f}",
                    ha="center", va="center", fontsize=7.5, color=WHITE,
                    fontweight="bold")

        # Total bar
        total = sum(values)
        ax.bar(len(values), total, color=RED, edgecolor=NAVY,
               linewidth=0.8, alpha=0.9, label=f"Total = {total:.1f}%")
        ax.text(len(values), total / 2, f"{total:.1f}",
                ha="center", va="center", fontsize=9, color=WHITE,
                fontweight="bold")

        x_labels = labels + ["Total\nUpside"]
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8, color=NAVY)
        ax.set_ylabel("pp of GDP")
        ax.set_title(f"{G4_LABELS[iso3]} — Upside Risk: {total:.1f} pp", color=NAVY)

        # P50 and DaR annotations
        ax.text(0.98, 0.97, f"P50={r['Q50']:.1f}%  DaR={r['DaR']:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=NAVY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=GOLD, alpha=0.3))

    # Legend for drivers
    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in zip(DRIVER_LABELS.values(), DRIVER_COLORS.values())]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))

    out_path = OUTPUT_DIR / "fig3_waterfall.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. FISCAL CRISIS PROBABILITY SCORES
# ─────────────────────────────────────────────────────────────────────────────

def crisis_signal_chart(pooled_scores: pd.DataFrame) -> Path:
    """
    Horizontal bar chart of fiscal crisis probability for G4 (2025–2026).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    fig.suptitle(
        "EU G4 — Fiscal Crisis Early-Warning Probability Scores",
        fontsize=13, fontweight="bold", color=NAVY,
    )

    for ax, yr in zip(axes, [2025, 2026]):
        yr_data = pooled_scores[pooled_scores["year"] == yr].copy()

        if yr_data.empty:
            ax.text(0.5, 0.5, f"No data for {yr}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(str(yr))
            continue

        # Sort by probability descending
        yr_data = yr_data.set_index("iso3").reindex(G4_ORDER).reset_index()
        yr_data["label"] = yr_data["iso3"].map(G4_LABELS)
        yr_data["prob"]  = yr_data["crisis_prob_pooled"].fillna(0.0) * 100

        bar_colors = [RED if p > 15 else GOLD if p > 8 else LIGHT_BLUE
                      for p in yr_data["prob"]]

        bars = ax.barh(yr_data["label"], yr_data["prob"],
                       color=bar_colors, edgecolor=NAVY, linewidth=0.5,
                       height=0.5)

        # Value labels
        for bar, p in zip(bars, yr_data["prob"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=9, color=NAVY)

        # Risk threshold lines
        ax.axvline(8,  color=GOLD, lw=1.2, ls="--", alpha=0.8)
        ax.axvline(15, color=RED,  lw=1.2, ls="--", alpha=0.8)
        ax.text(8.2,  -0.6, "Elevated", fontsize=7, color=GOLD)
        ax.text(15.2, -0.6, "High",     fontsize=7, color=RED)

        ax.set_xlim(0, max(yr_data["prob"].max() + 5, 25))
        ax.set_xlabel("Crisis probability (%)")
        ax.set_title(f"Horizon: {yr}", color=NAVY)
        ax.invert_yaxis()

    out_path = OUTPUT_DIR / "fig4_crisis_signal.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. GLOBAL CONTEXT — G4 in World Debt Distribution (slide 1)
# ─────────────────────────────────────────────────────────────────────────────

def global_context_chart(panel: pd.DataFrame, latest_year: int = 2023) -> Path:
    """
    Kernel-density of world debt/GDP with G4 vertical markers.
    """
    from scipy.stats import gaussian_kde

    yr_data = panel[panel["year"] == latest_year]["debt_gdp"].dropna()
    if yr_data.empty:
        yr_data = panel[panel["year"] == panel["year"].max()]["debt_gdp"].dropna()
        latest_year = int(panel["year"].max())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        f"Global Debt/GDP Distribution ({latest_year}) — EU G4 Highlighted",
        fontweight="bold", color=NAVY,
    )

    # KDE
    kde = gaussian_kde(yr_data, bw_method=0.3)
    x   = np.linspace(0, 280, 400)
    ax.fill_between(x, kde(x), color=NAVY, alpha=0.15)
    ax.plot(x, kde(x), color=NAVY, lw=2)

    # G4 markers
    from risk.dar import WEO_BASELINE_2027
    g4_debt = panel[(panel["iso3"].isin(["FRA", "DEU", "ITA", "ESP"])) &
                    (panel["year"] == latest_year)].set_index("iso3")["debt_gdp"]
    g4_colors = {"FRA": "#3498DB", "DEU": "#27AE60", "ITA": "#E74C3C", "ESP": "#F39C12"}

    for iso3 in G4_ORDER:
        val = g4_debt.get(iso3, np.nan)
        if pd.isna(val):
            val = WEO_BASELINE_2027.get(iso3, 100)
        ax.axvline(val, color=g4_colors[iso3], lw=2.2, ls="--",
                   label=f"{G4_LABELS[iso3]}: {val:.0f}%")

    ax.axvline(60, color=GOLD, lw=1.5, ls=":", alpha=0.8)
    ax.text(61, ax.get_ylim()[1] * 0.95, "Maastricht\n60%", fontsize=7.5, color=GOLD)

    ax.set_xlabel("Gross Government Debt (% of GDP)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim(0, 280)

    out_path = OUTPUT_DIR / "fig0_global_context.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def generate_all_charts(
    panel: pd.DataFrame,
    dar: pd.DataFrame,
    skt_params: pd.DataFrame,
    pooled_scores: pd.DataFrame,
) -> dict[str, Path]:
    """
    Generate all five charts and return dict of chart_name → path.
    """
    print("Generating charts …")
    return {
        "global_context": global_context_chart(panel),
        "fan_charts":     fan_charts(panel, dar, skt_params),
        "comparison_bar": comparison_bar(dar),
        "waterfall":      waterfall_charts(dar),
        "crisis_signal":  crisis_signal_chart(pooled_scores),
    }


if __name__ == "__main__":
    from data.panel_builder import load_panel
    from risk.dar import load_dar
    from model.quantile_fit import load_skt_params
    from crisis.logit_signal import load_crisis_scores

    panel   = load_panel()
    dar     = load_dar()
    skt     = load_skt_params()
    _, pooled = load_crisis_scores()

    paths = generate_all_charts(panel, dar, skt, pooled)
    for name, p in paths.items():
        print(f"  {name}: {p}")
