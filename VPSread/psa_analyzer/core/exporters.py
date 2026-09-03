"""
Export utilities — CSV with summary header, publication figures, and PDF report.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

# Use the non-interactive Agg backend so exporting works even from a
# worker thread without a display attached.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .analyzer import AnalysisResult
from .constants import PALETTE
# Re-export the styled writers (kept in separate modules to stay readable)
from .report_pdf import export_pdf_report  # noqa: F401
from .report_xlsx import export_xlsx_report  # noqa: F401


# ---------------------------------------------------------------------------
# CSV export — Excel-style with Summary header + per-cycle data
# ---------------------------------------------------------------------------
def export_cycles_csv(result: AnalysisResult, out_path: Path) -> Path:
    """
    Write a sectioned CSV: Summary stats at top, then per-cycle table.

    Layout mirrors the lab's ``Summary - <name>`` Excel format so the
    file opens nicely in Excel/Google Sheets.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cyc = result.cycles
    p = result.params
    n_cyc = len(cyc)
    total_bed_mL = p.steps_per_cycle * p.bed_volume_mL

    lines: list[str] = []
    w = lambda *cells: lines.append(",".join(str(c) for c in cells))

    # ── Header ──
    w(f"Summary - {p.adsorbent_name}")
    w()
    w("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    w()

    # ── Experimental parameters ──
    w("--- EXPERIMENTAL PARAMETERS ---")
    w("Adsorbent",           p.adsorbent_name)
    w("Temperature (°C)",    f"{p.temperature_C:.1f}")
    w("Bed volume per bed (mL)", f"{p.bed_volume_mL:.3f}")
    w("Steps per cycle",     p.steps_per_cycle)
    w("Total bed volume (mL)", f"{total_bed_mL:.3f}")
    w("Smoothing window",    "adaptive (3 cycles for 1-4, 5 thereafter)")
    w("Final cycle start",   p.final_cycle_start)
    w()

    # ── Results summary ──
    w("--- RESULTS (last cycle) ---")
    w("Total cycles",          n_cyc)
    w("Elapsed time (h)",      f"{result.total_elapsed_hours:.4f}")
    w("Final Purity (%)",      f"{result.final_purity:.4f}")
    w("Final Recovery (%)",    f"{result.final_recovery:.4f}")
    w("Final Productivity (t CO2/m³·day)", f"{result.final_productivity:.6f}")
    w()

    # ── Per-cycle table ──
    w("--- PER-CYCLE DATA ---")
    cols = ["cycle_id", "elapsed_s", "dt_s", "dt_min",
            "sum_co2_out", "sum_n2_out", "sum_co2_in",
            "purity_pct", "recovery_pct", "productivity_tCO2_m3_day"]
    cols = [c for c in cols if c in cyc.columns]
    w(*cols)
    for _, row in cyc.iterrows():
        w(*(f"{row[c]:.6g}" if isinstance(row[c], (int, float))
            else str(row[c]) for c in cols))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    # Sibling JSON of params for reproducibility
    params_path = out_path.with_name(out_path.stem + "_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(p.to_dict(), f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Figure export
# ---------------------------------------------------------------------------
def _apply_publication_style(ax) -> None:
    """Apply consistent grid + spine style to a matplotlib axis."""
    ax.grid(True, color=PALETTE.grid, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_dashboard(fig, result: AnalysisResult) -> None:
    """Draw the 3-panel dashboard into an existing matplotlib figure."""
    cyc = result.cycles
    x = cyc["cycle_id"]
    axes = fig.subplots(3, 1, sharex=True)

    pur_avg = cyc["purity_avg_pct"] if "purity_avg_pct" in cyc.columns \
        else cyc["purity_pct"]
    axes[0].scatter(x, cyc["purity_pct"], s=14, color=PALETTE.purity,
                    alpha=0.30, edgecolors="none", zorder=1, label="raw")
    axes[0].plot(x, pur_avg, color=PALETTE.purity, marker="o", markersize=4,
                 linewidth=1.6, zorder=2, label="Purity (avg)")
    axes[0].set_ylabel("Purity [%]")
    axes[0].set_title(
        f"{result.params.adsorbent_name} — PSA Performance "
        f"({result.params.temperature_C:.0f}°C)"
    )
    _apply_publication_style(axes[0])

    rec_corr = cyc["recovery_corrected_pct"] if "recovery_corrected_pct" \
        in cyc.columns else cyc["recovery_pct"].clip(upper=100.0)
    axes[1].scatter(x, cyc["recovery_pct"], s=14, color=PALETTE.recovery,
                    alpha=0.30, edgecolors="none", zorder=1, label="raw")
    axes[1].plot(x, rec_corr, color=PALETTE.recovery, marker="o", markersize=4,
                 linewidth=1.6, zorder=2, label="Recovery (avg, ≤100%)")
    axes[1].set_ylabel("Recovery [%]")
    _apply_publication_style(axes[1])

    prod_avg = cyc["productivity_avg"] if "productivity_avg" in cyc.columns \
        else cyc["productivity_tCO2_m3_day"]
    axes[2].scatter(x, cyc["productivity_tCO2_m3_day"], s=14,
                    color=PALETTE.productivity, alpha=0.30, edgecolors="none",
                    zorder=1, label="raw")
    axes[2].plot(x, prod_avg, color=PALETTE.productivity, marker="o",
                 markersize=4, linewidth=1.6, zorder=2,
                 label="Productivity (avg)")
    axes[2].set_ylabel(r"Productivity [t CO$_2$/m$^3$·day]")
    axes[2].set_xlabel("Cycle #")
    _apply_publication_style(axes[2])

    # Mark where final-value averaging begins, and show a small legend.
    start = int(result.params.final_cycle_start)
    for ax in axes:
        ax.axvline(start, color="#9467bd", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=0)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    axes[0].annotate(f"avg from cycle {start}", xy=(start, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(3, -10), textcoords="offset points",
                     fontsize=8, color="#9467bd", va="top")


def export_dashboard(result: AnalysisResult, out_path: Path,
                     dpi: int = 300) -> Path:
    """Render the 3-panel dashboard to PNG / SVG / PDF (single page)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 9))
    _draw_dashboard(fig, result)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_single_metric(result: AnalysisResult, metric: str,
                         out_path: Path, dpi: int = 300) -> Path:
    """Export a single-panel figure for one metric."""
    cyc = result.cycles
    x = cyc["cycle_id"]

    metric = metric.lower()
    if metric == "purity":
        y = cyc["purity_avg_pct"] if "purity_avg_pct" in cyc.columns \
            else cyc["purity_pct"]
        ylabel, color = "Purity [%] (avg)", PALETTE.purity
    elif metric == "recovery":
        y = cyc["recovery_corrected_pct"] if "recovery_corrected_pct" \
            in cyc.columns else cyc["recovery_pct"].clip(upper=100.0)
        ylabel, color = "Recovery [%] (avg, ≤100%)", PALETTE.recovery
    elif metric == "productivity":
        y, ylabel, color = (cyc["productivity_avg"] if "productivity_avg"
                            in cyc.columns else cyc["productivity_tCO2_m3_day"],
                            r"Productivity [t CO$_2$/m$^3$·day]",
                            PALETTE.productivity)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color=color, marker="o", markersize=4, linewidth=1.6)
    ax.set_xlabel("Cycle #")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{result.params.adsorbent_name} — {metric.title()}")
    _apply_publication_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Legacy matplotlib PDF report — kept only as a no-reportlab fallback.
# The PUBLIC ``export_pdf_report`` is the styled reportlab writer re-exported
# at the top of this module (from .report_pdf). Do not rename this back to
# ``export_pdf_report`` or it will shadow the styled version.
# ---------------------------------------------------------------------------
_METHODOLOGY_TEXT = r"""
CALCULATION METHODOLOGY

The PSA pipeline processes data in four stages:

STAGE 1 — Row-level derivatives
  • elapsed_s = (timestamp - first_timestamp).total_seconds()
  • CO$_2$ outlet flow [mL/min] = MFC-07 (AD-GAS, SMLM) × CO$_2$%/100
  • N$_2$  outlet flow [mL/min] = MFC-07 × (100 - CO$_2$%)/100
  • Step boundaries: BPR-01 drop $\leq$ 0.001 bar (rising edge → new step)
  • step_id = cumulative rising-edge count + 1 (rows before first edge = step 1)

STAGE 2 — Per-step aggregation (group by step_id)
  • $\sum$CO$_2$ out, $\sum$N$_2$ out, $\sum$CO$_2$ in over rows of the step
  • Purity$_{step}$  = $\sum$CO$_2$ out / ($\sum$CO$_2$ out + $\sum$N$_2$ out) × 100
  • Recovery$_{step}$ = $\sum$CO$_2$ out / $\sum$CO$_2$ in × 100

STAGE 3 — Per-cycle aggregation (4 consecutive steps = 1 cycle)
  • cycle_id = (step_id - 1) // steps_per_cycle + 1
  • $\sum$CO$_2$ out per cycle = sum of 4 step values
  • Purity$_{cycle}$ = $\sum$CO$_2$ out / ($\sum$CO$_2$ out + $\sum$N$_2$ out) × 100
  • Recovery$_{cycle}$ = $\sum$CO$_2$ out / $\sum$CO$_2$ in × 100
  • Productivity = ($\sum$CO$_2$ out × $\rho_{CO_2}$ / $\Delta$t$_{min}$)
                  ÷ (V$_{bed,total}$ × $\Delta$t$_{day}$)
    where $\rho_{CO_2}$ = 1.8 × 10$^{-6}$ t/L (researcher's reference),
    V$_{bed,total}$ = steps × bed_volume (e.g. 4 × 60 mL = 240 mL)

STAGE 3b — Smoothing (display only)
  • Adaptive centred average: window 3 for cycles 1-4, 5 for cycle 5+
  • Recovery capped at 100%: corrected = min(100, averaged)

STAGE 4 — Final values
  • Final = mean of every cycle from "Final avg start" (default 3) to last
  • Final Recovery averaged from the 100%-capped per-cycle values
  • Total elapsed (h) = elapsed_s at end of last cycle / 3600
""".strip()


def _export_pdf_report_matplotlib(result: AnalysisResult, out_path: Path,
                                  source_file: str | None = None) -> Path:
    """Legacy multi-page PDF (matplotlib). Unused — see module note above."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result.params
    n_cyc = len(result.cycles)
    total_bed_mL = p.steps_per_cycle * p.bed_volume_mL
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with PdfPages(out_path) as pdf:
        # ─── Page 1: cover + KPIs ───────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.06, 0.94, "PSA Analyzer — Report",
                 fontsize=20, fontweight="bold")
        fig.text(0.06, 0.91, f"Adsorbent: {p.adsorbent_name}   |   "
                              f"Generated: {now}", fontsize=9, color="#6b7280")
        if source_file:
            fig.text(0.06, 0.89, f"Source: {source_file}",
                     fontsize=9, color="#6b7280")

        # Experimental parameters block
        ex_text = (
            f"EXPERIMENTAL PARAMETERS\n"
            f"  • Temperature              {p.temperature_C:.1f} °C\n"
            f"  • Bed volume per bed       {p.bed_volume_mL:.3f} mL\n"
            f"  • Steps per cycle          {p.steps_per_cycle}\n"
            f"  • Total bed volume         {total_bed_mL:.3f} mL\n"
            f"  • Rolling window           {p.rolling_window} cycles\n"
            f"  • Final cycle start        {p.final_cycle_start}"
        )
        fig.text(0.06, 0.78, ex_text, fontsize=10, family="monospace",
                 verticalalignment="top")

        # KPI cards
        kpi_y = 0.55
        cards = [
            ("Final Purity",       f"{result.final_purity:.2f} %",      PALETTE.purity),
            ("Final Recovery",     f"{result.final_recovery:.2f} %",    PALETTE.recovery),
            ("Final Productivity", f"{result.final_productivity:.4f}\nt CO₂/m³·day",
                                                                        PALETTE.productivity),
            ("Elapsed",            f"{result.total_elapsed_hours:.2f} h", "#6b7280"),
            ("Total cycles",       f"{n_cyc}",                          "#6b7280"),
        ]
        n = len(cards)
        for i, (label, val, color) in enumerate(cards):
            x_left = 0.06 + (0.88 / n) * i
            w_card = 0.88 / n - 0.01
            ax = fig.add_axes([x_left, kpi_y, w_card, 0.12])
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="#f9fafb",
                                       edgecolor="#e5e7eb"))
            ax.text(0.5, 0.78, label, ha="center", va="center",
                    fontsize=8, color="#6b7280", fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.38, val, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color,
                    transform=ax.transAxes)

        # Footer
        fig.text(0.06, 0.04,
                 "Generated by PSA Analyzer  •  see following pages for "
                 "dashboard charts, per-cycle data, and calculation methodology.",
                 fontsize=8, color="#6b7280")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 2: dashboard charts ───────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Dashboard — Purity / Recovery / Productivity",
                     fontsize=14, fontweight="bold", y=0.97)
        _draw_dashboard(fig, result)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 3: per-cycle table ────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Per-Cycle Data", fontsize=14, fontweight="bold", y=0.97)
        ax = fig.add_axes([0.05, 0.05, 0.90, 0.88])
        ax.axis("off")
        cyc = result.cycles
        show_cols = ["cycle_id", "dt_s", "sum_co2_in", "sum_co2_out",
                     "purity_pct", "recovery_pct", "productivity_tCO2_m3_day"]
        show_cols = [c for c in show_cols if c in cyc.columns]
        header = ["Cycle", "dt [s]", "CO₂ in", "CO₂ out",
                  "Purity %", "Recovery %", "Productivity"]
        header = header[:len(show_cols)]
        cells = [[f"{cyc.iloc[i][c]:.4g}"
                  if isinstance(cyc.iloc[i][c], (int, float))
                  else str(cyc.iloc[i][c])
                  for c in show_cols]
                 for i in range(len(cyc))]
        tbl = ax.table(cellText=cells, colLabels=header,
                       loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 4: methodology ────────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.95, "Methodology",
                 fontsize=16, fontweight="bold")
        fig.text(0.06, 0.92,
                 "How Purity, Recovery, and Productivity are derived from raw sensor data.",
                 fontsize=9, color="#6b7280")
        fig.text(0.06, 0.88, _METHODOLOGY_TEXT, fontsize=9,
                 family="monospace", verticalalignment="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return out_path
