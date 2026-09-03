"""
Interactive plotting canvas built on pyqtgraph.

* Three stacked plots (Purity, Recovery, Productivity) with a shared X axis.
* Mouse wheel zooms, drag pans — standard pyqtgraph interactions.
* A hover crosshair on each plot shows precise values in a monospaced label.

The canvas is designed to also accept live samples in the future via
:meth:`append_live_sample` (currently a no-op stub).
"""

from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from psa_analyzer.core import PALETTE
from psa_analyzer.ui.theme import MONO_STACK, Theme, pyqtgraph_palette


# pyqtgraph global config — apply once at import time.
pg.setConfigOptions(antialias=True, useOpenGL=False)


def _cycle_axis_label(cyc) -> str:
    """'Cycle #' plus how long a cycle actually lasted, from ``dt_min``."""
    if cyc is None or cyc.empty or "dt_min" not in cyc.columns:
        return "Cycle #"
    d = pd.to_numeric(cyc["dt_min"], errors="coerce").dropna()
    d = d[d > 0]
    if d.empty:
        return "Cycle #"
    if len(d) == 1 or abs(d.max() - d.min()) < 0.05:
        return f"Cycle #   (each ≈ {d.mean():.2f} min)"
    return (f"Cycle #   (mean {d.mean():.2f} min, "
            f"range {d.min():.2f}–{d.max():.2f} min)")


class PlotCanvas(QWidget):
    """
    Three stacked interactive plots sharing the cycle-number X axis.

    Public API
    ----------
    plot_result(result)   -- draw the cycle data from an AnalysisResult
    clear()               -- wipe all curves
    apply_theme(theme)    -- restyle for Light or Dark
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar: Home (reset zoom) + Back (one step in zoom history).
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)
        self._btn_home = QPushButton("⌂ Home")
        self._btn_home.setToolTip("Reset zoom to fit all data")
        self._btn_home.clicked.connect(self._reset_zoom)
        self._btn_back = QPushButton("← Back")
        self._btn_back.setToolTip("Go back to previous zoom")
        self._btn_back.clicked.connect(self._zoom_back)
        toolbar.addWidget(self._btn_home)
        toolbar.addWidget(self._btn_back)
        toolbar.addStretch(1)

        # View toggle: how Recovery / Productivity are displayed.
        toolbar.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Averaged", "Raw", "Corrected"])
        self._view_combo.setToolTip(
            "Recovery / Productivity display mode:\n"
            "  Averaged  — adaptive centred smoothing (default)\n"
            "  Raw       — per-cycle values, unsmoothed\n"
            "  Corrected — averaged then capped at 100%"
        )
        self._view_combo.currentTextChanged.connect(self._on_view_changed)
        toolbar.addWidget(self._view_combo)
        layout.addLayout(toolbar)

        # GraphicsLayoutWidget hosts multiple PlotItems stacked vertically.
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # Zoom-history bookkeeping
        self._zoom_stack: list[list[tuple]] = []
        self._navigating: bool = False

        self._plot_purity       = self._glw.addPlot(row=0, col=0)
        self._plot_recovery     = self._glw.addPlot(row=1, col=0)
        self._plot_productivity = self._glw.addPlot(row=2, col=0)

        for p, label in [
            (self._plot_purity,       "Purity [%]"),
            (self._plot_recovery,     "Recovery [%]"),
            (self._plot_productivity, "Productivity [t CO₂/m³·day]"),
        ]:
            p.setLabel("left", label)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.addLegend(offset=(-10, 10), labelTextSize="8pt")

        self._plot_productivity.setLabel("bottom", "Cycle #")

        # Link X axes so panning/zooming one pans/zooms all.
        self._plot_recovery.setXLink(self._plot_purity)
        self._plot_productivity.setXLink(self._plot_purity)

        # Track view-range changes for the Back-button zoom history
        for p in (self._plot_purity, self._plot_recovery, self._plot_productivity):
            p.vb.sigRangeChangedManually.connect(self._push_zoom_snapshot)

        # Cache curve handles for fast updates
        self._curves: dict[str, pg.PlotDataItem] = {}
        # Last result, kept so the View toggle can re-draw without re-analysing
        self._result = None

        # Hover crosshairs
        self._crosshairs: list[tuple] = []
        for p in (self._plot_purity, self._plot_recovery, self._plot_productivity):
            self._install_crosshair(p)

        self._current_theme: Theme = Theme.LIGHT
        self.apply_theme(Theme.LIGHT)

    # -- Crosshair ----------------------------------------------------------
    def _install_crosshair(self, plot: pg.PlotItem) -> None:
        """Add a hover crosshair + monospaced label to a PlotItem."""
        v_line = pg.InfiniteLine(angle=90, movable=False,
                                 pen=pg.mkPen("#9ca3af", style=Qt.DashLine))
        h_line = pg.InfiniteLine(angle=0,  movable=False,
                                 pen=pg.mkPen("#9ca3af", style=Qt.DashLine))
        v_line.setVisible(False)
        h_line.setVisible(False)
        plot.addItem(v_line, ignoreBounds=True)
        plot.addItem(h_line, ignoreBounds=True)

        label = pg.TextItem(anchor=(0, 1), color="#1f2937",
                            fill=pg.mkBrush(255, 255, 255, 220))
        label.setFont(QFont("Consolas", 9))
        label.setVisible(False)
        plot.addItem(label, ignoreBounds=True)

        def on_move(evt) -> None:
            try:
                pos: QPointF = evt[0] if isinstance(evt, tuple) else evt
                if not plot.sceneBoundingRect().contains(pos):
                    v_line.setVisible(False); h_line.setVisible(False); label.setVisible(False)
                    return
                mp = plot.vb.mapSceneToView(pos)
                v_line.setPos(mp.x()); h_line.setPos(mp.y())
                v_line.setVisible(True); h_line.setVisible(True); label.setVisible(True)
                label.setText(f"x = {mp.x():>7.2f}\ny = {mp.y():>8.3f}")
                label.setPos(mp.x(), mp.y())
            except RuntimeError:
                # Underlying C++ object was deleted — disconnect ourselves
                try:
                    plot.scene().sigMouseMoved.disconnect(on_move)
                except (TypeError, RuntimeError):
                    pass

        plot.scene().sigMouseMoved.connect(on_move)
        self._crosshairs.append((plot, v_line, h_line, label, on_move))

    # -- Theme --------------------------------------------------------------
    def apply_theme(self, theme: Theme) -> None:
        """Recolour background, axes, and crosshair label for the given theme."""
        self._current_theme = theme
        palette = pyqtgraph_palette(theme)
        self._glw.setBackground(palette["background"])

        axis_color  = palette["foreground"]
        label_bg    = (255, 255, 255, 220) if theme is Theme.LIGHT else (40, 40, 40, 220)
        label_text  = "#1f2937" if theme is Theme.LIGHT else "#e5e7eb"

        for p in (self._plot_purity, self._plot_recovery, self._plot_productivity):
            for axis in ("left", "bottom"):
                ax = p.getAxis(axis)
                ax.setPen(pg.mkPen(axis_color))
                ax.setTextPen(pg.mkPen(axis_color))

        for entry in self._crosshairs:
            lbl = entry[3]
            lbl.setColor(label_text)
            lbl.fill = pg.mkBrush(*label_bg)

    # -- Data ---------------------------------------------------------------
    def clear(self) -> None:
        """Remove every plotted curve, but keep crosshair items alive."""
        self._result = None
        # Disconnect old crosshair handlers BEFORE clearing the plot —
        # otherwise they fire against deleted C++ objects and crash.
        for plot, v, h, lbl, handler in self._crosshairs:
            try:
                plot.scene().sigMouseMoved.disconnect(handler)
            except (TypeError, RuntimeError):
                pass
        self._crosshairs.clear()

        for p in (self._plot_purity, self._plot_recovery, self._plot_productivity):
            p.clear()
        self._curves.clear()

        for p in (self._plot_purity, self._plot_recovery, self._plot_productivity):
            self._install_crosshair(p)
        self.apply_theme(self._current_theme)

    def plot_result(self, result) -> None:
        """Draw the three panels from an :class:`AnalysisResult`."""
        self._result = result
        self._redraw()

    def _on_view_changed(self, _text: str = "") -> None:
        """Re-draw Recovery / Productivity when the View combo changes."""
        if self._result is not None:
            self._redraw()

    def _col(self, cyc, name: str, fallback: str):
        """Return cyc[name] as an array, falling back if the column is absent
        (e.g. results produced before the adaptive-smoothing change)."""
        col = name if name in cyc.columns else fallback
        return cyc[col].to_numpy()

    def _redraw(self) -> None:
        """Render the three panels for the current result + view mode."""
        result = self._result
        if result is None:
            return
        self.clear()              # clears the canvas (and nulls _result)...
        self._result = result     # ...so restore it for the View toggle
        cyc = result.cycles
        x = cyc["cycle_id"].to_numpy()
        mode = self._view_combo.currentText()

        # "Cycle #" alone never says how long a cycle took, which is the first
        # thing anyone asks of these axes. The duration is already computed per
        # cycle (dt_min), so put its range on the axis label.
        self._plot_productivity.setLabel("bottom", _cycle_axis_label(cyc))

        # Purity / Recovery / Productivity column choice depends on view mode.
        pur_col, pur_label = {
            "Raw":       ("purity_pct",     "Purity (raw)"),
            "Averaged":  ("purity_avg_pct", "Purity (avg)"),
            "Corrected": ("purity_avg_pct", "Purity (avg)"),
        }[mode]
        rec_col, rec_label = {
            "Raw":       ("recovery_pct",           "Recovery (raw)"),
            "Averaged":  ("recovery_avg_pct",       "Recovery (avg)"),
            "Corrected": ("recovery_corrected_pct", "Recovery (capped 100%)"),
        }[mode]
        prod_col, prod_label = {
            "Raw":       ("productivity_tCO2_m3_day", "Productivity (raw)"),
            "Averaged":  ("productivity_avg",         "Productivity (avg)"),
            "Corrected": ("productivity_corrected",   "Productivity (avg)"),
        }[mode]

        # Purity
        self._curves["purity"] = self._plot_purity.plot(
            x, self._col(cyc, pur_col, "purity_pct"),
            pen=pg.mkPen(PALETTE.purity, width=2),
            symbol="o", symbolSize=5, symbolBrush=PALETTE.purity,
            name=pur_label,
        )

        # Recovery
        self._curves["recovery"] = self._plot_recovery.plot(
            x, self._col(cyc, rec_col, "recovery_pct"),
            pen=pg.mkPen(PALETTE.recovery, width=2),
            symbol="o", symbolSize=5, symbolBrush=PALETTE.recovery,
            name=rec_label,
        )

        # Productivity
        self._curves["prod"] = self._plot_productivity.plot(
            x, self._col(cyc, prod_col, "productivity_tCO2_m3_day"),
            pen=pg.mkPen(PALETTE.productivity, width=2),
            symbol="o", symbolSize=5, symbolBrush=PALETTE.productivity,
            name=prod_label,
        )

        # Auto-range now that data is in, and seed the zoom history with
        # the home view so Back can return to it.
        self._navigating = True
        try:
            for p in self._all_plots():
                p.vb.autoRange()
        finally:
            self._navigating = False
        self._zoom_stack.clear()
        self._push_zoom_snapshot()

    # -- Zoom history -------------------------------------------------------
    def _all_plots(self) -> tuple:
        return (self._plot_purity, self._plot_recovery, self._plot_productivity)

    def _push_zoom_snapshot(self) -> None:
        """Push the current view range of every panel onto the zoom history."""
        if self._navigating:
            return
        snap = [tuple(map(tuple, p.vb.viewRange())) for p in self._all_plots()]
        # Avoid pushing a no-op duplicate
        if self._zoom_stack and self._zoom_stack[-1] == snap:
            return
        self._zoom_stack.append(snap)
        # Cap history length
        if len(self._zoom_stack) > 50:
            self._zoom_stack.pop(0)

    def _reset_zoom(self) -> None:
        """Home — auto-range every panel to fit all data."""
        self._navigating = True
        try:
            for p in self._all_plots():
                p.vb.autoRange()
        finally:
            self._navigating = False
        self._zoom_stack.clear()
        self._push_zoom_snapshot()

    def _zoom_back(self) -> None:
        """Back — restore the previous view range from the history stack."""
        if len(self._zoom_stack) < 2:
            self._reset_zoom()
            return
        self._zoom_stack.pop()  # discard the current state
        prev = self._zoom_stack[-1]
        self._navigating = True
        try:
            for p, (xr, yr) in zip(self._all_plots(), prev):
                p.vb.setRange(xRange=xr, yRange=yr, padding=0)
        finally:
            self._navigating = False

    # -- Live mode stub -----------------------------------------------------
    def append_live_sample(self, sample: dict) -> None:
        """
        Append a single live sample (future serial-port use).

        Currently a no-op — see workers/serial_worker.py.
        """
        # TODO: implement scrolling buffer when live mode goes live
        pass
