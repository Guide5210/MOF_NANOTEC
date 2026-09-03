"""
Professional PDF report using reportlab (Python equivalent of QuestPDF).
Multi-page layout with title bar, numbered sections, colour-coded KPI
boxes, banded tables, and a methodology page.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .analyzer import AnalysisResult
from .constants import PALETTE


# ── Unicode fonts ───────────────────────────────────────────────────────────
# Helvetica/Courier (reportlab built-ins) lack glyphs for ₂, ρ, Δ, Σ, ≤, …
# which render as ■ boxes. Register matplotlib's bundled DejaVu fonts (always
# present since matplotlib is a dependency) so the methodology text and KPI
# units render correctly. Falls back to the built-ins if anything is missing.
def _register_fonts() -> tuple[str, str, str]:
    """Return (regular, bold, mono) font names — DejaVu if available."""
    try:
        from matplotlib import get_data_path
        ttf = Path(get_data_path()) / "fonts" / "ttf"
        faces = {
            "DejaVuSans":       ttf / "DejaVuSans.ttf",
            "DejaVuSans-Bold":  ttf / "DejaVuSans-Bold.ttf",
            "DejaVuSansMono":   ttf / "DejaVuSansMono.ttf",
        }
        for name, path in faces.items():
            if not path.exists():
                raise FileNotFoundError(path)
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold")
        return "DejaVuSans", "DejaVuSans-Bold", "DejaVuSansMono"
    except Exception:
        return F_BODY, F_BOLD, "Courier"


F_BODY, F_BOLD, F_MONO = _register_fonts()


# ── Palette ───────────────────────────────────────────────────────────────
C_TITLE    = colors.HexColor("#1F4E79")
C_ACCENT   = colors.HexColor("#2E75B6")
C_LABEL_BG = colors.HexColor("#DDEBF7")
C_ROW_ALT  = colors.HexColor("#F2F2F2")
C_KPI_BG   = colors.HexColor("#E2EFDA")
C_BORDER   = colors.HexColor("#BFBFBF")
C_MUTED    = colors.HexColor("#595959")
C_PURITY   = colors.HexColor("#1F77B4")
C_RECOVERY = colors.HexColor("#ED7D31")
C_PRODUCT  = colors.HexColor("#70AD47")


def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", parent=base["Title"],
                                fontName=F_BOLD, fontSize=22,
                                textColor=C_TITLE, leading=26,
                                spaceAfter=2)
    s["subtitle"] = ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontName=F_BODY, fontSize=10,
                                   textColor=C_MUTED, leading=13,
                                   spaceAfter=2)
    s["meta"] = ParagraphStyle("meta", parent=base["Normal"],
                               fontName=F_BODY, fontSize=8,
                               textColor=C_MUTED, leading=11)
    s["section"] = ParagraphStyle("section", parent=base["Heading2"],
                                  fontName=F_BOLD, fontSize=14,
                                  textColor=C_TITLE, spaceBefore=10,
                                  spaceAfter=6, leading=18)
    s["body"] = ParagraphStyle("body", parent=base["Normal"],
                               fontName=F_BODY, fontSize=10,
                               textColor=colors.black, leading=15)
    s["mono"] = ParagraphStyle("mono", parent=base["Code"],
                               fontName=F_MONO, fontSize=9,
                               textColor=colors.black, leading=12)
    s["caption"] = ParagraphStyle("caption", parent=base["Normal"],
                                  fontName=F_BODY, fontSize=8,
                                  textColor=C_MUTED, leading=10,
                                  alignment=1)
    return s


def _header_footer(canvas_obj: canvas.Canvas, doc) -> None:
    """Draw a coloured title bar at top + page number footer."""
    w, h = A4
    # top accent bar
    canvas_obj.setFillColor(C_TITLE)
    canvas_obj.rect(0, h - 28, w, 4, stroke=0, fill=1)
    # report title in top-left
    canvas_obj.setFillColor(C_MUTED)
    canvas_obj.setFont(F_BODY, 8)
    canvas_obj.drawString(2 * cm, h - 1.2 * cm, "PSA Analyzer — Report")
    # footer
    canvas_obj.setFillColor(C_MUTED)
    canvas_obj.setFont(F_BODY, 8)
    canvas_obj.drawString(2 * cm, 1.2 * cm,
                          "PSA Analyzer v1.0 — Pressure Swing Adsorption")
    canvas_obj.drawRightString(w - 2 * cm, 1.2 * cm,
                               f"Page {doc.page}")


def _kpi_table(result: AnalysisResult) -> Table:
    p = result.params
    n_cyc = len(result.cycles)
    kpis = [
        ("Final Purity",       f"{result.final_purity:.2f}",        "%",                C_PURITY),
        ("Final Recovery",     f"{result.final_recovery:.2f}",      "%",                C_RECOVERY),
        ("Final Productivity", f"{result.final_productivity:.4f}",  "t CO₂/m³·day",     C_PRODUCT),
        ("Elapsed Time",       f"{result.total_elapsed_hours:.2f}", "hours",            C_MUTED),
        ("Total Cycles",       f"{n_cyc}",                          "",                 C_MUTED),
    ]
    labels = [k[0] for k in kpis]
    values = [k[1] for k in kpis]
    units  = [k[2] for k in kpis]

    tbl = Table([labels, values, units],
                colWidths=[3.4 * cm] * 5, rowHeights=[0.7 * cm, 1.2 * cm, 0.55 * cm])
    style = TableStyle([
        # Labels row
        ("BACKGROUND", (0, 0), (-1, 0), C_KPI_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_MUTED),
        ("FONTNAME",   (0, 0), (-1, 0), F_BOLD),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        # Values row
        ("FONTNAME",   (0, 1), (-1, 1), F_BOLD),
        ("FONTSIZE",   (0, 1), (-1, 1), 16),
        # Units row
        ("FONTSIZE",   (0, 2), (-1, 2), 9),
        ("TEXTCOLOR",  (0, 2), (-1, 2), C_MUTED),
        # Borders
        ("BOX",        (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LINEBELOW",  (0, 0), (-1, 0), 0.5, C_BORDER),
        ("LINEBELOW",  (0, 1), (-1, 1), 0.5, C_BORDER),
        ("LINEAFTER",  (0, 0), (-2, -1), 0.5, C_BORDER),
    ])
    # Per-column value colours
    for i, (_, _, _, color) in enumerate(kpis):
        style.add("TEXTCOLOR", (i, 1), (i, 1), color)
    tbl.setStyle(style)
    return tbl


def _stats_table(result: AnalysisResult) -> Table | None:
    """Descriptive statistics over the final averaging window."""
    stats = getattr(result, "final_stats", None)
    if not stats:
        return None
    order = [("purity", "Purity [%]", C_PURITY),
             ("recovery", "Recovery [%]", C_RECOVERY),
             ("productivity", "Productivity", C_PRODUCT)]
    header = ["Metric", "Mean", "Std", "Min", "Max", "CV [%]"]
    rows = [header]
    fmt = lambda v, d: ("—" if v != v else f"{v:.{d}f}")
    for key, label, _ in order:
        s = stats.get(key, {})
        d = 4 if key == "productivity" else 2
        rows.append([label,
                     fmt(s.get("mean", float("nan")), d),
                     fmt(s.get("std", float("nan")), d),
                     fmt(s.get("min", float("nan")), d),
                     fmt(s.get("max", float("nan")), d),
                     fmt(s.get("cv", float("nan")), 1)])
    tbl = Table(rows, colWidths=[4.4 * cm, 2.5 * cm, 2.5 * cm,
                                 2.5 * cm, 2.5 * cm, 2.6 * cm])
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), F_BOLD),
        ("FONTNAME",   (0, 1), (0, -1), F_BOLD),
        ("FONTNAME",   (1, 1), (-1, -1), F_BODY),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",      (0, 0), (0, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    for r, (key, _, color) in enumerate(order, start=1):
        style.add("TEXTCOLOR", (0, r), (0, r), color)
        if r % 2 == 0:
            style.add("BACKGROUND", (1, r), (-1, r), C_ROW_ALT)
    tbl.setStyle(style)
    return tbl


def _params_table(result: AnalysisResult,
                  source_file: str | None) -> Table:
    p = result.params
    total_bed_mL = p.steps_per_cycle * p.bed_volume_mL
    rows = []
    if source_file:
        rows.append(("Source file", source_file))
    rows += [
        ("Adsorbent",            p.adsorbent_name),
        ("Temperature",          f"{p.temperature_C:.1f} °C"),
        ("Bed volume (per bed)", f"{p.bed_volume_mL:.3f} mL"),
        ("Steps / cycle",        str(p.steps_per_cycle)),
        ("Total bed volume",     f"{total_bed_mL:.3f} mL"),
        ("Smoothing",            "adaptive (window 3 for cycles 1–4, 5 thereafter)"),
        ("Final values from",    f"cycle {p.final_cycle_start} to last"),
    ]
    tbl = Table(rows, colWidths=[5.5 * cm, 11.5 * cm])
    style = TableStyle([
        ("FONTNAME", (0, 0), (0, -1), F_BOLD),
        ("FONTNAME", (1, 0), (1, -1), F_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), C_TITLE),
        ("BACKGROUND", (0, 0), (0, -1), C_LABEL_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, C_BORDER),
    ])
    for r in range(len(rows)):
        if r % 2 == 1:
            style.add("BACKGROUND", (1, r), (1, r), C_ROW_ALT)
    tbl.setStyle(style)
    return tbl


def _cycle_table(result: AnalysisResult) -> Table:
    cyc = result.cycles
    has_cap = "recovery_corrected_pct" in cyc.columns
    headers = ["Cycle", "dt [s]", "Σ CO₂ in", "Σ CO₂ out",
               "Purity [%]", "Recov [%]", "Recov cap", "Product."]
    rows = [headers]
    over_rows: list[int] = []   # 1-based row indices where raw recovery > 100%
    for i, (_, r) in enumerate(cyc.iterrows(), start=1):
        rec_raw = float(r["recovery_pct"])
        if rec_raw > 100.0:
            over_rows.append(i)
        rec_cap = (f"{float(r['recovery_corrected_pct']):.2f}" if has_cap
                   else f"{min(rec_raw, 100.0):.2f}")
        rows.append([
            int(r["cycle_id"]),
            f"{r['dt_s']:.0f}"  if "dt_s" in cyc.columns else "—",
            f"{r['sum_co2_in']:.3f}"  if "sum_co2_in" in cyc.columns else "—",
            f"{r['sum_co2_out']:.3f}" if "sum_co2_out" in cyc.columns else "—",
            f"{r['purity_pct']:.2f}",
            f"{rec_raw:.2f}",
            rec_cap,
            f"{r['productivity_tCO2_m3_day']:.4f}",
        ])
    col_widths = [1.3, 1.5, 2.0, 2.0, 2.0, 1.9, 1.9, 2.2]
    tbl = Table(rows, colWidths=[w * cm for w in col_widths],
                repeatRows=1)
    style = TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), F_BOLD),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    for r in range(1, len(rows)):
        if r % 2 == 0:
            style.add("BACKGROUND", (0, r), (-1, r), C_ROW_ALT)
    # Flag raw-recovery cells that exceeded 100% (capped for Final KPIs).
    for r in over_rows:
        style.add("TEXTCOLOR", (5, r), (5, r), colors.HexColor("#C00000"))
        style.add("FONTNAME",  (5, r), (5, r), F_BOLD)
    tbl.setStyle(style)
    return tbl


def _dashboard_png(result: AnalysisResult) -> io.BytesIO:
    cyc = result.cycles
    x = cyc["cycle_id"]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.5), sharex=True)

    def _col(name, fallback):
        return cyc[name] if name in cyc.columns else cyc[fallback]

    # Each panel: faint raw scatter behind the smoothed/corrected line.
    pur_avg  = _col("purity_avg_pct", "purity_pct")
    rec_corr = (cyc["recovery_corrected_pct"] if "recovery_corrected_pct"
                in cyc.columns else cyc["recovery_pct"].clip(upper=100.0))
    prod_avg = _col("productivity_avg", "productivity_tCO2_m3_day")

    panels = [
        (axes[0], cyc["purity_pct"], pur_avg,  PALETTE.purity,
         "Purity [%]",                 "averaged"),
        (axes[1], cyc["recovery_pct"], rec_corr, PALETTE.recovery,
         "Recovery [%]",               "avg, ≤100%"),
        (axes[2], cyc["productivity_tCO2_m3_day"], prod_avg, PALETTE.productivity,
         "Productivity\n[t CO₂/m³·day]", "averaged"),
    ]
    for ax, raw, smooth, color, ylabel, avg_label in panels:
        ax.scatter(x, raw, s=14, color=color, alpha=0.30,
                   edgecolors="none", zorder=1, label="raw")
        ax.plot(x, smooth, color=color, marker="o", markersize=4,
                linewidth=1.6, zorder=2, label=avg_label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="best", framealpha=0.85,
                  handletextpad=0.4, borderpad=0.3)

    axes[0].set_title(
        f"{result.params.adsorbent_name} @ {result.params.temperature_C:.0f}°C",
        fontsize=11, fontweight="bold", color="#1F4E79")
    axes[2].set_xlabel("Cycle #", fontsize=9)

    # Vertical marker at the cycle where final-value averaging begins.
    start = int(result.params.final_cycle_start)
    for ax in axes:
        ax.axvline(start, color="#9467bd", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=0)
    axes[0].annotate(f"avg from cycle {start}", xy=(start, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(3, -10), textcoords="offset points",
                     fontsize=7, color="#9467bd", va="top")

    for ax in axes:
        ax.grid(True, color="#cccccc", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


_METHODOLOGY_HTML = """
<b>STAGE 1 — Row-level derivatives</b><br/>
• elapsed_s = (timestamp − first_timestamp).total_seconds()<br/>
• Recovery uses instantaneous CO₂%:
CO₂ outlet = MFC-07 (AD-GAS) × CO₂%/100, N₂ outlet = MFC-07 × (100−CO₂%)/100<br/>
• Purity uses a 7-row forward rolling MAX of CO₂% (Excel MAX(AYₙ:AYₙ₊₆)):
CO₂ outlet = MFC-07 × MAX₇(CO₂%)/100<br/>
• Step boundary: BPR-01 drop ≤ 0.001 bar (rising edge → new step)<br/>
• step_id = cumulative rising-edge count + 1
(rows before first edge are step 1 — matches researcher's Excel)<br/>
<br/>
<b>STAGE 2 — Per-step aggregation (group by step_id)</b><br/>
• Σ CO₂ out, Σ N₂ out, Σ CO₂ in over rows of the step<br/>
• Purity<sub>step</sub>   = Σ CO₂ out(MAX₇) / (Σ CO₂ out(MAX₇) + Σ N₂ out(MAX₇)) × 100<br/>
• Recovery<sub>step</sub> = Σ CO₂ out(inst.) / Σ CO₂ in × 100<br/>
<br/>
<b>STAGE 3 — Per-cycle aggregation (4 consecutive steps = 1 cycle)</b><br/>
• cycle_id = (step_id − 1) // steps_per_cycle + 1<br/>
• Σ CO₂ out per cycle = sum of step values<br/>
• Purity<sub>cycle</sub>   = Σ CO₂ out / (Σ CO₂ out + Σ N₂ out) × 100<br/>
• Recovery<sub>cycle</sub> = Σ CO₂ out / Σ CO₂ in × 100<br/>
• Productivity = (Σ CO₂ out × ρ<sub>CO₂</sub> / Δt<sub>min</sub>)
                 ÷ (V<sub>bed,total</sub> × Δt<sub>day</sub>)<br/>
&nbsp;&nbsp;ρ<sub>CO₂</sub> = 1.8 × 10⁻⁶ t/L (researcher's reference)<br/>
&nbsp;&nbsp;V<sub>bed,total</sub> = steps × bed_volume (e.g. 4 × 60 mL = 240 mL)<br/>
<br/>
<b>STAGE 3b — Smoothing (display only)</b><br/>
• Plotted lines use an adaptive centred average: window 3 for cycles 1–4,
window 5 for cycle 5+ (applied to Purity, Recovery, Productivity;
Excel columns BW / BX).<br/>
• Recovery is capped at 100% — Corrected = min(100, averaged) (column BY).<br/>
• The faint dots behind each line are the unsmoothed per-cycle values.<br/>
<br/>
<b>STAGE 4 — Final values &amp; statistics</b><br/>
• Final Purity / Recovery / Productivity = mean of every cycle from
"Final values from" (default cycle 3) to the last cycle.<br/>
• Final Recovery uses the raw (uncapped) per-cycle values; the 100% cap is
display-only (plot "Corrected" mode and the "Recov cap" table column).<br/>
• Std / Min / Max and CV (= Std ÷ Mean × 100) are computed over the
same window as a stability indicator.<br/>
• Total elapsed time = elapsed_s at end of last cycle ÷ 3600.<br/>
"""


def export_pdf_report(result: AnalysisResult, out_path: Path,
                      source_file: str | None = None) -> Path:
    """Generate the styled multi-page PDF report."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(str(out_path), pagesize=A4,
                          leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2.4 * cm, bottomMargin=2 * cm)
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="normal")
    template = PageTemplate(id="main", frames=[frame],
                            onPage=_header_footer)
    doc.addPageTemplates([template])

    st = _styles()
    story: list = []

    # ─── Title block ────────────────────────────────────────────────────────
    story.append(Paragraph("PSA Analyzer — Cycle Report", st["title"]))
    story.append(Paragraph(
        f"Pressure Swing Adsorption — {result.params.adsorbent_name}",
        st["subtitle"]))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           st["meta"]))
    story.append(Spacer(1, 6 * mm))

    # ─── 1. Parameters ──────────────────────────────────────────────────────
    story.append(Paragraph("1. Experimental Parameters", st["section"]))
    story.append(_params_table(result, source_file))
    story.append(Spacer(1, 6 * mm))

    # ─── 2. Final results KPIs ──────────────────────────────────────────────
    story.append(Paragraph(
        f"2. Final Results (mean of cycles "
        f"{int(result.params.final_cycle_start)}–{len(result.cycles)})",
        st["section"]))
    story.append(_kpi_table(result))
    story.append(Spacer(1, 5 * mm))

    # Descriptive statistics over the same averaging window.
    stats_tbl = _stats_table(result)
    if stats_tbl is not None:
        story.append(Paragraph(
            "Statistics over the averaging window "
            "(Std / Min / Max, CV = coefficient of variation):",
            st["caption"]))
        story.append(Spacer(1, 1.5 * mm))
        story.append(stats_tbl)
    story.append(Spacer(1, 6 * mm))

    # ─── 3. Dashboard ───────────────────────────────────────────────────────
    story.append(Paragraph("3. Performance Dashboard", st["section"]))
    img = Image(_dashboard_png(result), width=17 * cm, height=15 * cm)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Paragraph(
        "Figure 1. Per-cycle Purity, Recovery, and Productivity. Solid line = "
        "adaptive-averaged (Recovery capped at 100%); faint dots = raw "
        "per-cycle values. Dashed line marks where final-value averaging begins.",
        st["caption"]))

    story.append(PageBreak())

    # ─── 4. Per-cycle table ─────────────────────────────────────────────────
    story.append(Paragraph("4. Per-Cycle Data", st["section"]))
    story.append(_cycle_table(result))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "“Recov [%]” is the raw per-cycle recovery (values &gt; 100% shown "
        "in red); “Recov cap” is capped at 100% for display only. Final "
        "Recovery and the statistics use the raw (uncapped) values.",
        st["caption"]))

    story.append(Spacer(1, 8 * mm))

    # ─── 5. Methodology ─────────────────────────────────────────────────────
    story.append(Paragraph("5. Calculation Methodology", st["section"]))
    story.append(Paragraph(_METHODOLOGY_HTML, st["body"]))

    doc.build(story)
    return out_path
