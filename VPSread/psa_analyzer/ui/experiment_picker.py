"""
ExperimentPickerDialog — pick from auto-detected experiment windows.

Shown right after a workbook is loaded. The user either picks one of the
detected experiments (click 'Analyze' on the corresponding row) or falls
back to manual cropping with the 'Manual Crop' button.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from psa_analyzer.core.detector import ExperimentSegment, detect_experiments
from psa_analyzer.core.filename_parser import classify_group


def _fmt(t, fmt: str) -> str:
    """strftime that survives NaT / None."""
    try:
        if t is None or pd.isna(t):
            return "—"
        return t.strftime(fmt)
    except Exception:
        return "—"


def _fmt_num(v, fmt: str) -> str:
    """Number formatter that prints '—' for None / NaN."""
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    try:
        return fmt.format(v)
    except Exception:
        return "—"


class ExperimentPickerDialog(QDialog):
    """
    Modal dialog listing detected experiments.

    Outcomes (via :meth:`outcome`):
        ("analyze", df_cropped)   — user picked a row
        ("manual",  None)         — user clicked Manual Crop
        ("cancel",  None)         — user closed / cancelled
    """

    def __init__(self, df: pd.DataFrame, colmap: dict[str, str],
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detected experiments")
        self.resize(1100, 480)

        self._df = df
        self._colmap = colmap
        self._segments: list[ExperimentSegment] = detect_experiments(df, colmap)
        self._outcome: tuple[str, pd.DataFrame | None] = ("cancel", None)

        self._build_ui()

    # -- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        header = QLabel(
            f"Auto-detected <b>{len(self._segments)}</b> experiment "
            f"window(s) using the MFC-01 (CO2) feed signal. "
            f"Click 'Analyze' on the experiment you want to process, "
            f"or use 'Manual Crop' for full control."
        )
        header.setWordWrap(True)
        v.addWidget(header)

        self._table = QTableWidget(len(self._segments), 11, self)
        self._table.setHorizontalHeaderLabels(
            ["#", "Type", "Date", "Start → End", "Duration",
             "P [bar]", "Flow [mL/min]", "T [°C]", "Ads [s]",
             "Cycles", ""]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        # Style: pattern columns get a subtle bg so they stand out
        for col in (5, 6, 7, 8):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents)
        v.addWidget(self._table, stretch=1)

        self._populate_table()

        # Button row
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        self._btn_manual = QPushButton("Manual Crop…")
        self._btn_manual.clicked.connect(self._on_manual)
        buttons.addButton(self._btn_manual, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    _FLAG_LABEL = {
        "warmup": ("Warmup",       "#9ca3af", "#f3f4f6"),
        "long":   ("⚠ Long",       "#92400e", "#fef3c7"),
        "short":  ("⚠ Short",      "#92400e", "#fef3c7"),
        "":       ("Experiment",   "#065f46", "#d1fae5"),
    }

    def _populate_table(self) -> None:
        if not self._segments:
            self._table.setRowCount(1)
            empty = QTableWidgetItem(
                "No experiments detected. Use 'Manual Crop' to pick a window."
            )
            empty.setTextAlignment(Qt.AlignCenter)
            self._table.setSpan(0, 0, 1, 11)
            self._table.setItem(0, 0, empty)
            return

        for row, seg in enumerate(self._segments):
            label, fg, bg = self._FLAG_LABEL.get(seg.flag,
                                                self._FLAG_LABEL[""])
            self._set_cell(row, 0, str(seg.index))
            self._set_cell(row, 1, label, fg=fg, bg=bg)
            self._set_cell(row, 2, _fmt(seg.start_time, "%d %b %Y"))
            self._set_cell(
                row, 3,
                f"{_fmt(seg.start_time, '%H:%M:%S')} → "
                f"{_fmt(seg.end_time, '%H:%M:%S')}",
                align_left=True)
            self._set_cell(row, 4, f"{seg.duration_h:.2f} h")
            # 4-signal pattern detection columns
            self._set_cell(row, 5, _fmt_num(seg.pressure_bar, "{:g}"))
            self._set_cell(row, 6, _fmt_num(seg.flow_mlmin, "{:.0f}"))
            self._set_cell(row, 7, _fmt_num(seg.temperature_C, "{:.0f}"))
            self._set_cell(row, 8, _fmt_num(seg.ads_time_s, "{:.0f}"))
            self._set_cell(row, 9, str(seg.bpr_pulses))

            btn = QPushButton("→ Analyze")
            tip = f"Signature: {seg.signature()}"
            if seg.pressure_bar and seg.flow_mlmin and seg.temperature_C \
                    and seg.ads_time_s:
                grp = classify_group({
                    "temp_C": int(seg.temperature_C),
                    "flow_mlmin": int(seg.flow_mlmin),
                    "pressure_barg": int(seg.pressure_bar),
                    "ads_time_s": int(seg.ads_time_s),
                })
                tip += f"\nOFAT group: {grp}"
            if seg.flag == "warmup":
                tip += ("\n\nFew BPR pulses — likely warmup / no PSA "
                        "cycling. Analyze anyway?")
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, s=seg: self._on_analyze(s))
            self._table.setCellWidget(row, 10, btn)

    def _set_cell(self, row: int, col: int, text: str,
                  fg: str | None = None, bg: str | None = None,
                  align_left: bool = False) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(
            (Qt.AlignVCenter | Qt.AlignLeft) if align_left else Qt.AlignCenter
        )
        if fg:
            item.setForeground(QBrush(QColor(fg)))
        if bg:
            item.setBackground(QBrush(QColor(bg)))
        self._table.setItem(row, col, item)

    # -- Slots --------------------------------------------------------------
    def _on_analyze(self, seg: ExperimentSegment) -> None:
        cropped = self._df.iloc[seg.start_idx:seg.end_idx + 1].reset_index(drop=True)
        self._outcome = ("analyze", cropped)
        self.accept()

    def _on_manual(self) -> None:
        self._outcome = ("manual", None)
        self.accept()

    # -- Public API ---------------------------------------------------------
    def outcome(self) -> tuple[str, pd.DataFrame | None]:
        return self._outcome
