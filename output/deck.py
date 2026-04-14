"""
PowerPoint deck builder for EU G4 Debt-at-Risk presentation.

5-slide deck:
  Slide 1: Global context — G4 in world debt distribution
  Slide 2: Fan charts for FR, DE, IT, ES
  Slide 3: Drivers waterfall per country
  Slide 4: Fiscal crisis probability scores
  Slide 5: Policy implications (bullet points, editable)

Palette: navy (#1B2A4A) and gold (#C8A951)
Export: output/eu_g4_debt_at_risk.pptx
"""

from pathlib import Path
from datetime import date

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt, Emu

OUTPUT_DIR = Path(__file__).parent

# Colour constants
NAVY_RGB  = RGBColor(0x1B, 0x2A, 0x4A)
GOLD_RGB  = RGBColor(0xC8, 0xA9, 0x51)
WHITE_RGB = RGBColor(0xFF, 0xFF, 0xFF)
RED_RGB   = RGBColor(0xC0, 0x39, 0x2B)
GREEN_RGB = RGBColor(0x27, 0xAE, 0x60)

G4_LABELS = {"FRA": "France", "DEU": "Germany", "ITA": "Italy", "ESP": "Spain"}
G4_ORDER  = ["FRA", "DEU", "ITA", "ESP"]

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── Helper functions ──────────────────────────────────────────────────────────

def _rgb(hex_str: str) -> RGBColor:
    """Convert hex colour string to RGBColor."""
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _set_bg(slide, color: RGBColor) -> None:
    """Fill slide background with a solid colour."""
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_text_box(
    slide, text: str, left: float, top: float,
    width: float, height: float,
    font_size: int = 16, bold: bool = False,
    color: RGBColor = WHITE_RGB, align=PP_ALIGN.LEFT,
    word_wrap: bool = True,
) -> object:
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = word_wrap
    p    = tf.paragraphs[0]
    p.alignment = align
    run  = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    return txBox


def _add_picture(slide, img_path: Path,
                 left: float, top: float,
                 width: float, height: float) -> None:
    """Insert an image onto a slide; skip if missing."""
    if not img_path.exists():
        print(f"  Warning: image not found — {img_path}")
        return
    slide.shapes.add_picture(
        str(img_path),
        Inches(left), Inches(top),
        Inches(width), Inches(height),
    )


def _header_bar(slide, title: str, subtitle: str = "") -> None:
    """Add a navy header bar with gold title text."""
    # Background rectangle
    from pptx.util import Inches
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), SLIDE_W, Inches(1.1),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY_RGB
    shape.line.fill.background()

    _add_text_box(slide, title, 0.2, 0.05, 11.0, 0.65,
                  font_size=22, bold=True, color=GOLD_RGB)
    if subtitle:
        _add_text_box(slide, subtitle, 0.2, 0.68, 11.0, 0.34,
                      font_size=11, bold=False, color=WHITE_RGB)

    # Date stamp (top-right)
    today = date.today().strftime("%B %Y")
    _add_text_box(slide, today, 11.3, 0.05, 2.0, 0.4,
                  font_size=9, color=GOLD_RGB, align=PP_ALIGN.RIGHT)


def _footer(slide, text: str = "IMF WP/25/86 Methodology | Confidential") -> None:
    """Add a slim footer line."""
    shape = slide.shapes.add_shape(
        1,
        Inches(0), Inches(7.15), SLIDE_W, Inches(0.3),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY_RGB
    shape.line.fill.background()
    _add_text_box(slide, text, 0.15, 7.14, 13.0, 0.28,
                  font_size=7.5, color=WHITE_RGB)


# ── Slide builders ────────────────────────────────────────────────────────────

def _slide1_global(prs: Presentation, img_path: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_bg(slide, WHITE_RGB)
    _header_bar(
        slide,
        "EU G4 Debt-at-Risk — Board Presentation",
        subtitle="Based on IMF Working Paper WP/25/86 · Furceri, Giannone, Kisat, Lam & Li (May 2025)",
    )
    _add_picture(slide, img_path, left=0.3, top=1.2, width=12.7, height=5.7)
    _footer(slide)


def _slide2_fan(prs: Presentation, img_path: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, WHITE_RGB)
    _header_bar(
        slide,
        "Debt-at-Risk Fan Charts — EU G4 (2024–2027 Horizon)",
        subtitle="Shaded bands: P5–P95 distribution conditional on current macro-financial environment",
    )
    _add_picture(slide, img_path, left=0.3, top=1.15, width=12.7, height=6.05)
    _footer(slide)


def _slide3_waterfall(prs: Presentation, img_path: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, WHITE_RGB)
    _header_bar(
        slide,
        "Upside Risk Decomposition by Driver",
        subtitle="Contribution of macro-financial drivers to DaR (P95 − P50) — 3-year horizon",
    )
    _add_picture(slide, img_path, left=0.3, top=1.15, width=12.7, height=5.9)
    _footer(slide)


def _slide4_crisis(prs: Presentation, img_path: Path, pooled_scores: pd.DataFrame) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, WHITE_RGB)
    _header_bar(
        slide,
        "Fiscal Crisis Early-Warning Signals — EU G4 (2025–2026)",
        subtitle="Logit model: crisis probability ~ upside debt risk (P95 − P50) | Laeven-Valencia crisis episodes",
    )
    _add_picture(slide, img_path, left=0.3, top=1.15, width=8.5, height=5.7)

    # Score table (right side)
    _add_text_box(slide, "Crisis Probability Summary",
                  9.1, 1.2, 3.8, 0.4,
                  font_size=10, bold=True, color=NAVY_RGB)

    y = 1.65
    for iso3 in G4_ORDER:
        for yr in [2025, 2026]:
            row = pooled_scores[
                (pooled_scores["iso3"] == iso3) &
                (pooled_scores["year"] == yr)
            ]
            prob = f"{row['crisis_prob_pooled'].values[0]*100:.1f}%" if not row.empty else "N/A"
            risk_level = ""
            if not row.empty:
                p = row['crisis_prob_pooled'].values[0] * 100
                risk_level = " ▲ HIGH" if p > 15 else (" ▲" if p > 8 else " ●")
            line = f"{G4_LABELS[iso3]} {yr}: {prob}{risk_level}"
            color = RED_RGB if "HIGH" in risk_level else (GOLD_RGB if "▲" in risk_level else NAVY_RGB)
            _add_text_box(slide, line, 9.1, y, 3.8, 0.32,
                          font_size=9, color=color)
            y += 0.34

    _footer(slide)


def _slide5_policy(prs: Presentation, dar: pd.DataFrame) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, WHITE_RGB)
    _header_bar(
        slide,
        "Policy Implications & Key Takeaways",
        subtitle="Edit this slide with current policy conclusions",
    )

    # Summary table
    _add_text_box(slide, "Debt-at-Risk Summary (2027 Horizon)",
                  0.5, 1.25, 12.0, 0.45,
                  font_size=12, bold=True, color=NAVY_RGB)

    y_start = 1.75
    for iso3 in G4_ORDER:
        row = dar[dar["iso3"] == iso3]
        if row.empty:
            continue
        r = row.iloc[0]
        line = (
            f"{G4_LABELS[iso3]:10s}  "
            f"WEO Baseline: {r['weo_baseline']:.1f}%  │  "
            f"Median: {r['Q50']:.1f}%  │  "
            f"DaR (P95): {r['DaR']:.1f}%  │  "
            f"Upside: +{r['Upside']:.1f} pp"
        )
        _add_text_box(slide, line, 0.5, y_start, 12.5, 0.38,
                      font_size=10, color=NAVY_RGB)
        y_start += 0.42

    # Bullet points (editable)
    bullets = [
        "1.  Italy and France face the highest Debt-at-Risk: model-implied P95 paths exceed 130–145% of GDP "
        "by 2027 under adverse macro-financial scenarios.",
        "2.  Primary balance consolidation remains the dominant driver of upside risk across all G4 economies, "
        "followed by sovereign spread widening for periphery sovereigns.",
        "3.  Fiscal crisis early-warning scores are elevated for Spain and Italy (2025–2026), "
        "warranting contingency planning.",
        "4.  Germany's lower DaR reflects fiscal space; however, uncertainty-driven tail risks "
        "have risen post-2022.",
        "5.  [EDIT] Recommended policy action: Commit to credible medium-term fiscal frameworks "
        "to anchor debt expectations and narrow the DaR-to-baseline gap.",
    ]

    y_b = 3.5
    for b in bullets:
        _add_text_box(slide, b, 0.5, y_b, 12.5, 0.52,
                      font_size=9.5, color=NAVY_RGB)
        y_b += 0.58

    # Source note
    _add_text_box(
        slide,
        "Methodology: Machado-Santos Silva (2019) location-scale quantile regression · "
        "Log-score density pooling (Crump et al. 2022) · IMF WP/25/86",
        0.5, 7.0, 12.5, 0.3,
        font_size=7, color=GOLD_RGB,
    )
    _footer(slide)


# ── Main deck builder ─────────────────────────────────────────────────────────

def build_deck(
    dar: pd.DataFrame,
    pooled_scores: pd.DataFrame,
    chart_paths: dict[str, Path] | None = None,
) -> Path:
    """
    Assemble the 5-slide PPTX deck.

    Parameters
    ----------
    dar           : DaR results DataFrame
    pooled_scores : pooled crisis probability scores
    chart_paths   : dict mapping chart names to file paths (from charts.py)

    Returns
    -------
    Path to saved PPTX file
    """
    if chart_paths is None:
        chart_paths = {}

    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    # Default blank layout
    blank_layout = prs.slide_layouts[6]  # Blank

    print("  Building slides …")

    # Slide 1
    _slide1_global(prs, chart_paths.get("global_context", Path("missing.png")))

    # Slide 2 — Fan charts
    _slide2_fan(prs, chart_paths.get("fan_charts", Path("missing.png")))

    # Slide 3 — Waterfall
    _slide3_waterfall(prs, chart_paths.get("waterfall", Path("missing.png")))

    # Slide 4 — Crisis signal
    _slide4_crisis(prs, chart_paths.get("crisis_signal", Path("missing.png")), pooled_scores)

    # Slide 5 — Policy implications
    _slide5_policy(prs, dar)

    out_path = OUTPUT_DIR / "eu_g4_debt_at_risk.pptx"
    prs.save(str(out_path))
    print(f"  Saved deck → {out_path}")
    return out_path


if __name__ == "__main__":
    from risk.dar import load_dar
    from crisis.logit_signal import load_crisis_scores
    from output.charts import generate_all_charts
    from data.panel_builder import load_panel
    from model.quantile_fit import load_skt_params

    panel   = load_panel()
    dar     = load_dar()
    skt     = load_skt_params()
    _, pooled = load_crisis_scores()

    chart_paths = generate_all_charts(panel, dar, skt, pooled)
    deck_path   = build_deck(dar, pooled, chart_paths)
    print(f"\nDeck saved: {deck_path}")
