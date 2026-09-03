"""
DataCropperDialog — pick a time window from a multi-experiment workbook.

Shows an overview plot of the inlet CO2 flow (MFC-01) and BPR-01 against
time. The user drags a pyqtgraph LinearRegionItem to mark the experiment
they want to analyse, then clicks "Apply Crop & Analyze".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from psa_analyzer.core import PALETTE
from psa_analyzer.core.analyzer import _build_timestamp


class DataCropperDialog(QDialog):
    """
    Modal dialog that lets the user crop the raw DataFrame to a time window.

    Use:
        dlg = DataCropperDialog(df, colmap, parent=self)
        if dlg.exec() == QDialog.Accepted:
            cropped_df = dlg.cropped_df()
    """

    def __init__(self, df: pd.DataFrame, colmap: dict[str, str],
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop Timeline — select experiment window")
        self.resize(1100, 600)

        self._df = df
        self._colmap = colmap
        self._cropped: pd.DataFrame | None = None

        # Build a continuous datetime index. Drop rows whose timestamp
        # didn't parse — they'd break the region selector.
        ts = _build_timestamp(df, colmap)
        keep = ts.notna()
        if keep.sum() < 2:
            # Last-resort: synthesize a 1 Hz timestamp from row position so
            # the cropper still opens. Durations won't be wall-clock accurate.
            ts = pd.Series(
                pd.date_range("1970-01-01", periods=len(df), freq="s")
            )
            keep = pd.Series([True] * len(df))
        self._df = df.loc[keep].reset_index(drop=True)
        self._ts = ts.loc[keep].reset_index(drop=True)
        # x-axis in seconds since the first sample, easier for the region item
        self._t0 = self._ts.iloc[0]
        self._x = (self._ts - self._t0).dt.total_seconds().to_numpy()

        self._build_ui()
        self._plot_overview()

    # -- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        header = QLabel(
            "Drag the shaded region's edges to mark the experiment you want "
            "to analyze. Scroll-wheel zooms the plot."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #4b5563;")
        v.addWidget(header)

        self._glw = pg.GraphicsLayoutWidget()
        v.addWidget(self._glw, stretch=1)

        # Read-out row
        info = QHBoxLayout()
        self._lbl_start = QLabel("Start: —")
        self._lbl_end   = QLabel("End: —")
        self._lbl_dur   = QLabel("Duration: —")
        for lbl in (self._lbl_start, self._lbl_end, self._lbl_dur):
            lbl.setStyleSheet("font-family: Consolas, monospace;")
            info.addWidget(lbl)
        info.addStretch(1)
        self._btn_full = QPushButton("Reset to full range")
        self._btn_full.clicked.connect(self._reset_range)
        info.addWidget(self._btn_full)
        v.addLayout(info)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._btn_apply = QPushButton("Apply Crop && Analyze")
        self._btn_apply.setDefault(True)
        self._btn_apply.clicked.connect(self._on_apply)
        buttons.addButton(self._btn_apply, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def _plot_overview(self) -> None:
        # Two stacked panels: CO2 inlet on top, BPR-01 below
        self._plot_co2 = self._glw.addPlot(row=0, col=0)
        self._plot_bpr = self._glw.addPlot(row=1, col=0)
        self._plot_co2.setLabel("left", "MFC-01 CO2 [SLPM]")
        self._plot_bpr.setLabel("left", "BPR-01 [bar]")
        self._plot_bpr.setLabel("bottom", "Elapsed time [h]")
        for p in (self._plot_co2, self._plot_bpr):
            p.showGrid(x=True, y=True, alpha=0.25)
        self._plot_bpr.setXLink(self._plot_co2)

        x_hours = self._x / 3600.0
        co2 = pd.to_numeric(self._df[self._colmap["co2_in"]],
                            errors="coerce").to_numpy()
        bpr = pd.to_numeric(self._df[self._colmap["bpr"]],
                            errors="coerce").to_numpy()

        self._plot_co2.plot(x_hours, co2, pen=pg.mkPen(PALETTE.purity, width=1))
        self._plot_bpr.plot(x_hours, bpr,
                            pen=pg.mkPen(PALETTE.productivity, width=1))

        # LinearRegionItem on the top plot — covers full range by default.
        x_min, x_max = float(x_hours[0]), float(x_hours[-1])
        self._region = pg.LinearRegionItem(
            values=(x_min, x_max),
            brush=pg.mkBrush(31, 119, 180, 40),
            hoverBrush=pg.mkBrush(31, 119, 180, 70),
        )
        self._region.setZValue(10)
        self._plot_co2.addItem(self._region)

        # Mirror region onto the BPR plot for visual feedback
        self._region_mirror = pg.LinearRegionItem(
            values=(x_min, x_max),
            brush=pg.mkBrush(31, 119, 180, 40),
            movable=False,
        )
        self._region_mirror.setZValue(10)
        self._plot_bpr.addItem(self._region_mirror)

        self._region.sigRegionChanged.connect(self._on_region_changed)
        self._on_region_changed()  # initialise labels

    # -- Slots --------------------------------------------------------------
    def _on_region_changed(self) -> None:
        lo_h, hi_h = self._region.getRegion()
        if hi_h < lo_h:
            lo_h, hi_h = hi_h, lo_h
        self._region_mirror.setRegion((lo_h, hi_h))

        t_lo = self._t0 + pd.Timedelta(hours=lo_h)
        t_hi = self._t0 + pd.Timedelta(hours=hi_h)
        dur_h = hi_h - lo_h
        self._lbl_start.setText(f"Start: {t_lo.strftime('%Y-%m-%d %H:%M:%S')}")
        self._lbl_end.setText(  f"End:   {t_hi.strftime('%Y-%m-%d %H:%M:%S')}")
        self._lbl_dur.setText(  f"Duration: {dur_h:.3f} h")

    def _reset_range(self) -> None:
        x_hours = self._x / 3600.0
        self._region.setRegion((float(x_hours[0]), float(x_hours[-1])))

    def _on_apply(self) -> None:
        lo_h, hi_h = self._region.getRegion()
        if hi_h < lo_h:
            lo_h, hi_h = hi_h, lo_h
        lo_s, hi_s = lo_h * 3600.0, hi_h * 3600.0
        mask = (self._x >= lo_s) & (self._x <= hi_s)
        if mask.sum() < 10:
            # Refuse silly-tiny selections
            self._lbl_dur.setText("Duration: too few rows — widen the range.")
            return
        self._cropped = self._df.loc[mask].reset_index(drop=True)
        self.accept()

    # -- Public API ---------------------------------------------------------
    def cropped_df(self) -> pd.DataFrame | None:
        """The cropped DataFrame, available once the dialog is accepted."""
        return self._cropped
