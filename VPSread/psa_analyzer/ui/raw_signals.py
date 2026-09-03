"""
Raw-signal explorer — the right-hand half of the analysis window.

The left half answers "what did the experiment achieve" (Purity / Recovery /
Productivity per cycle). This half answers "what did the instruments actually
do", by drawing the untouched columns of the loaded workbook against time.

That distinction matters when a KPI looks wrong: the only way to tell a real
result from a sensor problem is to look at the signal that produced it.

Design notes
------------
* **One panel per signal, stacked and time-linked** -- the same shape as the
  KPI canvas on the left. Overlaying them on shared axes buried anything small
  (0.006 SLPM against 40 vol%) in the bottom pixel row; a panel each keeps
  every trace readable while the linked x-axis still lines events up.
* **Two ways to pick a time window**, because they suit different questions:
  drag on a plot to chase something you can see, or type exact hours to return
  to a window you already know. The two stay in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from psa_analyzer.ui.theme import pyqtgraph_palette

# Colour-blind-safe qualitative palette; cycles if more signals are picked.
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Derived columns the analyzer adds to `result.rows`. They are not instrument
# readings, so they are hidden by default to keep the list to real signals.
_DERIVED_PREFIXES = (
    "co2_outlet", "n2_outlet", "co2_pct_max7", "elapsed_s", "edge", "step_id",
)

# The picker shows every signal at once -- no scrolling -- by spreading them
# across as many columns as the pane is wide enough for. Fewer columns than
# this looks like a long strip; more than this and the labels start colliding.
_PICKER_MIN_COLUMNS = 2
_PICKER_MAX_COLUMNS = 8
_CHECKBOX_CHROME_PX = 34    # indicator + spacing around each label

_PANEL_HEIGHT = 130         # px per stacked plot before the area starts scrolling


# Engineering units that trail a tag name and carry no identity of their own.
_UNIT_WORDS = {"slpm", "smlm", "nlpm", "lpm", "bar", "mbar", "vol%", "%",
               "c", "°c", "degc", "pa", "kpa", "ppm"}


def _short_label(column: str) -> str:
    """Just the tag: 'MFC-01 (CO2 ) SLPM' and 'BPR -01 SLPM' -> 'MFC-01', 'BPR-01'.

    Long labels force the picker into two columns and a very tall list, so the
    decoration is dropped here and kept as the checkbox tooltip. Plots keep the
    full column name.
    """
    head = str(column).split("(")[0]
    words = [w for w in head.split() if w]
    while words and words[-1].lower().strip(",;") in _UNIT_WORDS:
        words.pop()
    short = " ".join(words).replace(" -", "-").replace("- ", "-").strip()
    return short or str(column).strip()


class RawSignalPanel(QWidget):
    """Pick any columns of the loaded file and plot each against time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._hours: np.ndarray | None = None
        self._boxes: dict[str, QCheckBox] = {}
        self._plots: list[pg.PlotItem] = []
        self._stats: dict[str, object] = {}     # name -> LabelItem beside plot
        self._series: dict[str, np.ndarray] = {}
        self._syncing = False
        self._theme = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        title = QLabel("Raw signals")
        title.setStyleSheet("font-weight:600;")
        root.addWidget(title)

        self._hint = QLabel("Load a file to list its signals.")
        self._hint.setStyleSheet("color:#6b7280; font-size:9pt;")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        # -- signal picker: every signal visible at once, no scrolling --------
        self._list_host = QWidget()
        self._grid = QGridLayout(self._list_host)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(1)
        self._columns = _PICKER_MIN_COLUMNS
        root.addWidget(self._list_host)

        # -- time window -----------------------------------------------------
        rng = QHBoxLayout()
        rng.addWidget(QLabel("From"))
        self.spin_from = self._hour_spin()
        rng.addWidget(self.spin_from)
        rng.addWidget(QLabel("to"))
        self.spin_to = self._hour_spin()
        rng.addWidget(self.spin_to)
        rng.addWidget(QLabel("h"))
        self.btn_full = QPushButton("Full range")
        self.btn_full.clicked.connect(self.show_full_range)
        rng.addWidget(self.btn_full)
        self.btn_none = QPushButton("Clear all")
        self.btn_none.setToolTip("Untick every signal")
        self.btn_none.clicked.connect(self.clear_selection)
        rng.addWidget(self.btn_none)
        rng.addStretch(1)
        self.chk_overlay = QCheckBox("Overlay in one plot")
        self.chk_overlay.setToolTip(
            "Off: one panel per signal, each with its own y-axis (best when the "
            "signals have different units).\n"
            "On: all signals share one plot — good for comparing shapes, but a "
            "small signal can be flattened by a large one.")
        self.chk_overlay.stateChanged.connect(self._redraw)
        rng.addWidget(self.chk_overlay)
        root.addLayout(rng)

        # -- stacked plots ---------------------------------------------------
        pg.setConfigOptions(antialias=True)
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#ffffff")
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(self._glw)
        root.addWidget(plot_scroll, stretch=1)
        self._plot_scroll = plot_scroll

        self.spin_from.valueChanged.connect(self._spins_to_view)
        self.spin_to.valueChanged.connect(self._spins_to_view)
        self._set_enabled(False)

    # -- theme --------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        """Match the KPI canvas so both halves recolour together."""
        self._theme = theme
        palette = pyqtgraph_palette(theme)
        self._glw.setBackground(palette["background"])
        for p in self._plots:
            for axis in ("left", "bottom"):
                ax = p.getAxis(axis)
                ax.setPen(pg.mkPen(palette["foreground"]))
                ax.setTextPen(pg.mkPen(palette["foreground"]))

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _hour_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(3)
        s.setRange(0.0, 1e6)
        s.setSingleStep(0.25)
        s.setMaximumWidth(90)
        return s

    def _set_enabled(self, on: bool) -> None:
        for w in (self.spin_from, self.spin_to, self.btn_full, self.btn_none):
            w.setEnabled(on)

    # -- picker layout ------------------------------------------------------
    def _best_columns(self) -> int:
        """How many columns of checkboxes fit across the current width."""
        if not self._boxes:
            return _PICKER_MIN_COLUMNS
        fm = QFontMetrics(self.font())
        widest = max(fm.horizontalAdvance(b.text())
                     for b in self._boxes.values()) + _CHECKBOX_CHROME_PX
        # the pane's own width is the authority; the host lags a resize by one
        # layout pass and would otherwise ask for a column too many
        avail = max(self.width() - 12, 200)
        cols = max(_PICKER_MIN_COLUMNS,
                   min(_PICKER_MAX_COLUMNS, avail // max(widest, 1)))
        # Never use more columns than would leave the last row nearly empty.
        return int(min(cols, len(self._boxes)))

    def _reflow(self, columns: int) -> None:
        """Re-lay the checkboxes into ``columns`` columns, top-left first."""
        if columns < 1:
            return
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        for i, box in enumerate(self._boxes.values()):
            self._grid.addWidget(box, i // columns, i % columns)

    def resizeEvent(self, event) -> None:      # noqa: N802 (Qt naming)
        """Re-flow the picker so it always shows everything without scrolling."""
        super().resizeEvent(event)
        want = self._best_columns()
        if want != self._columns:
            self._reflow(want)

    # -- data ---------------------------------------------------------------
    def set_dataframe(self, df: pd.DataFrame | None) -> None:
        """Load the row-level table and rebuild the signal list."""
        self.clear()
        if df is None or df.empty:
            return
        self._df = df

        if "elapsed_s" in df.columns:
            secs = pd.to_numeric(df["elapsed_s"], errors="coerce").to_numpy(float)
        else:                       # fall back to sample index if unavailable
            secs = np.arange(len(df), dtype=float)
        self._hours = secs / 3600.0

        cols = [c for c in df.columns
                if not str(c).startswith(_DERIVED_PREFIXES)
                and pd.api.types.is_numeric_dtype(df[c])]
        # Short labels ("MFC-01" rather than "MFC-01 (CO2 ) SLPM") so the whole
        # list fits without scrolling; the full column name is the tooltip and
        # still labels the plot itself. Any short name that would be ambiguous
        # keeps its full text.
        short = {c: _short_label(c) for c in cols}
        seen: dict[str, int] = {}
        for s in short.values():
            seen[s] = seen.get(s, 0) + 1
        for name in cols:
            label = short[name] if seen[short[name]] == 1 else str(name)
            box = QCheckBox(label)
            box.setToolTip(str(name))
            box.stateChanged.connect(self._redraw)
            self._boxes[str(name)] = box
        self._reflow(self._best_columns())

        span = float(np.nanmax(self._hours)) if len(self._hours) else 0.0
        self._hint.setText(
            f"{len(cols)} signals · {len(df):,} rows · {span:.2f} h. "
            f"All listed below — tick any number, each gets its own panel "
            f"sharing one time axis. Drag on a panel or type hours to zoom.")
        self._set_enabled(True)

        # Start with the pipeline's own inputs ticked — the ones a KPI problem
        # is most often traced back to. Each pattern needs every keyword, so
        # "CO2 + vol" cannot be satisfied by "MFC-01 (CO2) SLPM".
        for keywords in (("co2", "vol"), ("mfc-01",), ("bpr",)):
            for name, box in self._boxes.items():
                low = name.lower()
                if box.isChecked() or not all(k in low for k in keywords):
                    continue
                box.setChecked(True)
                break

        self.show_full_range()

    def clear(self) -> None:
        self._df = None
        self._hours = None
        while self._grid.count():
            self._grid.takeAt(0)
        for box in self._boxes.values():
            box.setParent(None)
        self._boxes.clear()
        self._glw.clear()
        self._plots.clear()
        self._stats.clear()
        self._series.clear()
        self._hint.setText("Load a file to list its signals.")
        self._set_enabled(False)

    def clear_selection(self) -> None:
        for box in self._boxes.values():
            box.setChecked(False)

    # -- time window --------------------------------------------------------
    def _x_range(self) -> tuple[float, float] | None:
        if not self._plots:
            return None
        return tuple(self._plots[0].getViewBox().viewRange()[0])

    def show_full_range(self) -> None:
        if self._hours is None or not len(self._hours):
            return
        lo = float(np.nanmin(self._hours))
        hi = float(np.nanmax(self._hours))
        if hi <= lo:
            hi = lo + 1e-3
        self._syncing = True
        self.spin_from.setValue(lo)
        self.spin_to.setValue(hi)
        for p in self._plots:
            p.setXRange(lo, hi, padding=0)
        self._syncing = False

    def _spins_to_view(self) -> None:
        if self._syncing or self._hours is None or not self._plots:
            return
        lo, hi = self.spin_from.value(), self.spin_to.value()
        if hi <= lo:
            return
        self._syncing = True
        self._plots[0].setXRange(lo, hi, padding=0)   # the rest are linked
        self._syncing = False

    def _view_to_spins(self) -> None:
        if self._syncing or self._hours is None:
            return
        rng = self._x_range()
        if rng is None:
            return
        lo, hi = rng
        self._syncing = True
        self.spin_from.setValue(max(0.0, lo))
        self.spin_to.setValue(max(lo + 1e-3, hi))
        self._syncing = False

    # -- drawing ------------------------------------------------------------
    def selected(self) -> list[str]:
        return [n for n, b in self._boxes.items() if b.isChecked()]

    def _redraw(self) -> None:
        if self._df is None or self._hours is None:
            return
        keep = self._x_range()          # survive a re-tick without losing zoom
        self._glw.clear()
        self._plots.clear()
        self._stats.clear()
        self._series.clear()

        chosen = self.selected()
        overlay = self.chk_overlay.isChecked()

        for i, name in enumerate(chosen):
            y = pd.to_numeric(self._df[name], errors="coerce").to_numpy(float)
            self._series[name] = y
            pen = pg.mkPen(_COLORS[i % len(_COLORS)], width=2)

            if overlay:
                if not self._plots:
                    p = self._new_plot(0)
                    p.addLegend(offset=(-10, 10), labelTextSize="8pt")
                    self._plots.append(p)
                self._plots[0].plot(self._hours, y, pen=pen, name=str(name))
            else:
                p = self._new_plot(i)
                p.setLabel("left", str(name))
                p.plot(self._hours, y, pen=pen)
                if i:
                    p.setXLink(self._plots[0])
                # running average for this panel, to the right of the trace
                lbl = self._glw.addLabel("", row=i, col=1,
                                         justify="left", size="8pt")
                self._stats[name] = lbl
                self._plots.append(p)

        if self._plots:
            self._plots[-1].setLabel("bottom", "Elapsed [h]")
            if overlay:
                self._plots[0].setLabel("left", "Signal")
            self._plots[0].getViewBox().sigXRangeChanged.connect(
                self._on_x_changed)
            self._glw.setMinimumHeight(
                _PANEL_HEIGHT * (1 if overlay else len(self._plots)))
            if keep and keep[1] > keep[0]:
                self._syncing = True
                self._plots[0].setXRange(keep[0], keep[1], padding=0)
                self._syncing = False
            else:
                self.show_full_range()
        else:
            self._glw.setMinimumHeight(0)

        self._update_stats()
        if self._theme is not None:
            self.apply_theme(self._theme)

    def _new_plot(self, row: int) -> pg.PlotItem:
        p = self._glw.addPlot(row=row, col=0)
        p.showGrid(x=True, y=True, alpha=0.25)
        # Files run to tens of thousands of rows; let pyqtgraph thin the drawn
        # points instead of pushing every one through the GPU.
        p.setDownsampling(auto=True, mode="peak")
        p.setClipToView(True)
        p.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        return p

    def _on_x_changed(self) -> None:
        self._view_to_spins()
        self._update_stats()

    def _update_stats(self) -> None:
        """Average (and range) of each signal *over the visible window*.

        Averaging the whole file would answer a question nobody asked once the
        user has zoomed into one stretch of the run, so the figures follow the
        time window on screen.
        """
        if not self._stats or self._hours is None:
            return
        rng = self._x_range()
        if rng is None:
            return
        lo, hi = rng
        sel = (self._hours >= lo) & (self._hours <= hi)
        if not sel.any():
            sel = np.ones_like(self._hours, dtype=bool)
        for name, lbl in self._stats.items():
            y = self._series.get(name)
            if y is None:
                continue
            w = y[sel]
            w = w[np.isfinite(w)]
            if not w.size:
                lbl.setText("no data")
                continue
            lbl.setText(f"avg {np.mean(w):,.4g}<br>"
                        f"min {np.min(w):,.4g}<br>"
                        f"max {np.max(w):,.4g}<br>"
                        f"n {w.size:,}")
