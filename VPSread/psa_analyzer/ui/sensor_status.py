"""
Machine & sensor status window for the Live Monitor.

Shows every VPSA sensor in one table — tag, group, Modbus address, the latest
value, unit, and a live/idle/not-exposed status light — plus a one-line summary
of what the machine is doing right now. Opened from the Live Monitor's View
menu; updated on every poll batch.

Clicking a row opens that one sensor's trend (:class:`SensorTrendDialog`). Each
sensor's recent samples are kept here as they stream in, so the trend is
available for any sensor without the main window having to plot all fifty.
"""

from __future__ import annotations

import time
from collections import deque

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from psa_analyzer.core import PALETTE
from psa_analyzer.core.plc_map import COL_BPR, COL_MFC01, COL_MFC07

_GREEN = QColor("#16a34a")
_GRAY  = QColor("#9ca3af")
_RED   = QColor("#dc2626")
_AMBER = QColor("#d97706")

_HEADERS = ["Tag", "Group", "Addr", "Value", "Unit", "Status"]

# Per-sensor history kept for the trend view: 30 minutes, matching the main
# scrolling plot, which is enough to see one cycle's shape without the memory
# cost of holding a whole run for every one of the ~50 tags.
_HISTORY_MIN = 30.0
_HISTORY_MAX_POINTS = 5000


class SensorTrendDialog(QDialog):
    """One sensor's value against time, refreshed live."""

    def __init__(self, tag: str, unit: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tag = tag
        self.setWindowTitle(f"Trend — {tag}")
        self.resize(720, 420)
        self.setModal(False)
        # A plain QDialog only offers a close button, so watching several
        # sensors at once meant a pile of windows that could not be put aside.
        # Give it the full title-bar set: minimise, maximise, close.
        self.setWindowFlags(Qt.Window
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowCloseButtonHint)

        root = QVBoxLayout(self)
        self._caption = QLabel("")
        self._caption.setStyleSheet("color:#6b7280;")
        root.addWidget(self._caption)

        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#ffffff")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("left", f"{tag} [{unit}]" if unit else tag)
        self._plot.setLabel("bottom", "Elapsed [min]")
        root.addWidget(self._plot, stretch=1)
        self._curve = self._plot.plot([], [], pen=pg.mkPen(PALETTE.purity, width=2))

    def set_series(self, minutes: list[float], values: list[float]) -> None:
        self._curve.setData(minutes, values)
        if values:
            self._caption.setText(
                f"{len(values):,} samples · last {values[-1]:,.4f} · "
                f"min {min(values):,.4f} · max {max(values):,.4f}")
        else:
            self._caption.setText("No samples yet for this sensor.")


class SensorStatusDialog(QDialog):
    """Live table of all sensors + a machine-state summary line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VPSA — Machine & Sensor Status")
        self.resize(720, 640)
        # non-modal so the live monitor keeps updating behind it
        self.setModal(False)

        self._row_by_key: dict[str, int] = {}
        self._latest: dict[str, float] = {}
        # Keys the probe found unreachable. The four canonical analyzer columns
        # are present (blank) in every row even when their device is dead, so
        # without this they would be relabelled "no signal" and look like a
        # different fault from the other channels of the very same device.
        self._unreachable: set[str] = set()
        # key -> (deque[t_seconds], deque[value]) for the trend view
        self._hist: dict[str, tuple[deque, deque]] = {}
        self._meta_by_key: dict[str, dict] = {}
        self._trends: dict[str, SensorTrendDialog] = {}
        self._t0: float | None = None

        root = QVBoxLayout(self)

        self._state = QLabel("Waiting for data…")
        self._state.setStyleSheet(
            "font-size:13pt; font-weight:600; padding:6px 2px;")
        root.addWidget(self._state)

        self._summary = QLabel("")
        self._summary.setStyleSheet("color:#6b7280;")
        root.addWidget(self._summary)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(_HEADERS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._table.cellClicked.connect(self._open_trend)
        root.addWidget(self._table, stretch=1)

        hint = QLabel("Click any sensor to open its trend. Several can be open "
                      "at once — each window can be minimised.")
        hint.setStyleSheet("color:#6b7280; font-size:9pt;")
        root.addWidget(hint)

    # -- populate -----------------------------------------------------------
    def set_report(self, report: list[dict]) -> None:
        """Build the table from the worker's one-off sensor report."""
        self._row_by_key.clear()
        self._unreachable.clear()
        report = sorted(report, key=lambda r: (r.get("group", ""), r.get("address", 0)))
        self._table.setRowCount(len(report))
        n_ok = n_bad = 0
        for r, s in enumerate(report):
            self._table.setItem(r, 0, QTableWidgetItem(str(s.get("tag", ""))))
            self._table.setItem(r, 1, QTableWidgetItem(str(s.get("group", ""))))
            # two devices are polled (PLC + gas analyzer), so name the source
            addr = str(s.get("address", ""))
            if s.get("host"):
                addr = f"{addr} @ {s['host']}"
            self._table.setItem(r, 2, QTableWidgetItem(addr))
            val_item = QTableWidgetItem("")
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(r, 3, val_item)
            self._table.setItem(r, 4, QTableWidgetItem(str(s.get("unit", ""))))
            self._table.setItem(r, 5, QTableWidgetItem(""))
            key = s.get("key", s.get("tag", ""))
            self._row_by_key[key] = r
            self._meta_by_key[key] = s

            reachable = bool(s.get("reachable"))
            if s.get("bus") == "rs485":
                self._set_status(r, "RS-485 (not polled)", _GRAY)
            elif reachable:
                n_ok += 1
                self._apply_value(r, s.get("value"))
            else:
                n_bad += 1
                self._unreachable.add(key)
                self._set_status(r, "✕ not exposed", _RED)
                val_item.setText("—")
        self._summary.setText(
            f"{n_ok} sensors live · {n_bad} not exposed over Modbus · "
            f"{len(report)} total")

    # -- live updates -------------------------------------------------------
    def update_values(self, row: dict) -> None:
        """Refresh value/status cells from a poll batch row."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        for key, val in row.items():
            r = self._row_by_key.get(key)
            if r is None:
                continue
            try:
                f = float(val)
            except (TypeError, ValueError):
                if key in self._unreachable:
                    continue        # keep the honest "not exposed" verdict
                self._apply_value(r, val)
                continue
            self._latest[key] = f
            self._record(key, now, f)
            self._apply_value(r, val)
        self._refresh_trends()
        self._update_state()

    # -- per-sensor history -------------------------------------------------
    def _record(self, key: str, now: float, value: float) -> None:
        """Append one sample and drop anything older than the history window."""
        ts, vs = self._hist.setdefault(
            key, (deque(maxlen=_HISTORY_MAX_POINTS),
                  deque(maxlen=_HISTORY_MAX_POINTS)))
        ts.append(now)
        vs.append(value)
        cutoff = now - _HISTORY_MIN * 60.0
        while ts and ts[0] < cutoff:
            ts.popleft()
            vs.popleft()

    def _open_trend(self, row: int, _col: int) -> None:
        """Show (or raise) the trend window for the clicked sensor."""
        key = next((k for k, r in self._row_by_key.items() if r == row), None)
        if key is None:
            return
        dlg = self._trends.get(key)
        if dlg is None:
            meta = self._meta_by_key.get(key, {})
            dlg = SensorTrendDialog(str(meta.get("tag", key)),
                                    str(meta.get("unit", "")), parent=self)
            dlg.finished.connect(lambda _r, k=key: self._trends.pop(k, None))
            self._trends[key] = dlg
        self._push_series(key, dlg)
        dlg.show()
        dlg.raise_()

    def _push_series(self, key: str, dlg: SensorTrendDialog) -> None:
        ts, vs = self._hist.get(key, (deque(), deque()))
        if not ts:
            dlg.set_series([], [])
            return
        t0 = ts[0]
        dlg.set_series([(t - t0) / 60.0 for t in ts], list(vs))

    def _refresh_trends(self) -> None:
        for key, dlg in list(self._trends.items()):
            self._push_series(key, dlg)

    # -- helpers ------------------------------------------------------------
    def _apply_value(self, r: int, val) -> None:
        try:
            f = float(val)
        except (TypeError, ValueError):
            self._set_status(r, "✕ no signal", _RED)
            return
        item = self._table.item(r, 3)
        if item is not None:
            item.setText(f"{f:,.4f}" if abs(f) < 1000 else f"{f:,.1f}")
        if abs(f) < 1e-9:
            self._set_status(r, "○ idle / zero", _GRAY)
        else:
            self._set_status(r, "● live", _GREEN)

    def _set_status(self, r: int, text: str, color: QColor) -> None:
        item = self._table.item(r, 5)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(r, 5, item)
        item.setText(text)
        item.setForeground(color)

    def _update_state(self) -> None:
        """One-line machine state from the headline flows / pressure."""
        co2_in = self._latest.get(COL_MFC01, 0.0)
        ad_gas = self._latest.get(COL_MFC07, 0.0)
        bpr    = self._latest.get(COL_BPR, 0.0)
        running = (co2_in or 0) > 0.001 or (ad_gas or 0) > 0.001
        if running:
            self._state.setText("VPSA: ● RUNNING")
            self._state.setStyleSheet(
                "font-size:13pt; font-weight:600; color:#16a34a; padding:6px 2px;")
        else:
            self._state.setText("VPSA: ○ IDLE / standby")
            self._state.setStyleSheet(
                "font-size:13pt; font-weight:600; color:#6b7280; padding:6px 2px;")
        self._summary.setText(
            f"CO₂ inlet {co2_in:.3f} SLPM · AD-GAS {ad_gas:.3f} NLPM · "
            f"BPR-01 {bpr:.3f}")
