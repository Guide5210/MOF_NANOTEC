"""
Live Monitor window — real-time view of eServer's CSV stream.

Opened as a separate top-level window from the sidebar so the existing
file-analysis workflow is untouched. It:

* tails the CSV on a background thread (:class:`LiveCsvWorker`),
* shows the latest raw sensor values as big number cards,
* scrolls a time-series of CO2%, the two MFC flows, and BPR-01, and
* re-runs the *unchanged* PSA pipeline every few seconds to update the
  cycle KPIs (Purity / Recovery / Productivity / cycle count).
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QStatusBar, QVBoxLayout, QWidget,
)

from psa_analyzer.core import PALETTE
from psa_analyzer.core.live_buffer import LiveBuffer, parse_timestamps
from psa_analyzer.core.plc_map import HORIBA_GROUP
from psa_analyzer.ui.data_table import DataTableDialog
from psa_analyzer.ui.kpi_cards import KPICard
from psa_analyzer.ui.sensor_status import SensorStatusDialog
from psa_analyzer.workers import start_live, start_modbus


# Rolling window for the scrolling plot (samples). 1800 = 30 min at 1 Hz.
_PLOT_WINDOW = 1800

# Trailing window for the continuously-available Purity/Recovery cards, and
# how often they are recomputed. These do not need BPR edges, so they keep
# reporting while the cycle-based KPIs are still waiting for a full cycle.
_KPI_WINDOW_MIN = 5.0
_KPI_REFRESH_S = 300.0


class LiveMonitorWindow(QMainWindow):
    """Standalone real-time monitor driven by the eServer CSV."""

    def __init__(self, source, params_provider,
                 interval_ms: int = 1000, parent: QWidget | None = None,
                 modbus_config: dict | None = None) -> None:
        super().__init__(parent)
        title = "PSA Analyzer — Live Monitor"
        title += " (Modbus)" if modbus_config else " (CSV)"
        self.setWindowTitle(title)
        self.resize(1200, 820)

        # Either a CSV path/folder, or a Modbus config dict (mutually exclusive)
        self._modbus_config = modbus_config
        self._explorer_window = None
        self._source = None if modbus_config else Path(source)
        self._params_provider = params_provider     # callable -> AnalysisParams
        self._interval_ms = interval_ms

        self._buffer = LiveBuffer()
        self._thread = None
        self._worker = None

        # rolling plot storage
        self._t: list[float] = []           # elapsed minutes (real timestamp)
        self._t0 = None                      # first sample's timestamp
        self._series: dict[str, list[float]] = {
            "co2": [], "mfc01": [], "mfc07": [], "bpr": []}
        self._n = 0
        self._batches = 0
        self._analyze_every = max(1, int(5000 / max(interval_ms, 1)))  # ~5 s
        self._last_kpi_t = 0.0      # wall clock of the last window-KPI refresh
        # A device that refused the connection is reported once at start-up and
        # would otherwise be buried by the routine "Live - N samples" line, so
        # the warning is kept and re-appended to every later status message.
        self._device_warning = ""
        self._flow_warning = ""     # set while a flow sensor reads negative
        # separate slot: the negative-flow check clears its own warning every
        # refresh and would otherwise wipe this one out
        self._adgas_warning = ""
        # Why the window KPIs are blank, when they are. Not a fault — an
        # explanation, so "—" is never read as "the program is broken".
        self._idle_reason = ""    # set while AD-GAS is a typed-in stand-in

        # recording (continuous CSV log) + dialogs
        self._rec_file = None
        self._rec_writer = None
        self._rec_path: Path | None = None
        self._rec_count = 0
        self._status_dialog: SensorStatusDialog | None = None
        self._data_dialog: DataTableDialog | None = None
        self._sensor_report: list = []

        self._build_ui()
        self._build_menu()
        self._start()

    # -- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Row 1 — current raw sensor values
        live_row = QHBoxLayout()
        live_row.setSpacing(10)
        self.card_co2   = KPICard("CO₂", "vol%")
        self.card_mfc01 = KPICard("MFC-01 (CO₂)", "SLPM")
        # MFC-07 is a 20 NLPM device and the pipeline uses its reading raw
        # (no mL->L conversion), so the value really is in L/min. The old
        # "SMLM" label was wrong; only the label changes, never the maths.
        self.card_mfc07 = KPICard("MFC-07 (AD-GAS)", "NLPM")
        self.card_bpr   = KPICard("BPR-01", "SLPM")
        for c in (self.card_co2, self.card_mfc01, self.card_mfc07, self.card_bpr):
            live_row.addWidget(c)
        root.addLayout(live_row)

        # Row 2 — rolling-window KPIs (no BPR edges needed) + cycle KPIs
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)
        self.card_purity_win = KPICard(f"Purity ({_KPI_WINDOW_MIN:g} min)", "%")
        self.card_recov_win  = KPICard(f"Recovery ({_KPI_WINDOW_MIN:g} min)", "%")
        self.card_purity = KPICard("Final Purity", "%")
        self.card_recov  = KPICard("Final Recovery", "%")
        self.card_prod   = KPICard("Final Productivity", "t CO₂/m³·day")
        self.card_cycles = KPICard("Cycles", "")
        for c in (self.card_purity_win, self.card_recov_win, self.card_purity,
                  self.card_recov, self.card_prod, self.card_cycles):
            kpi_row.addWidget(c)
        root.addLayout(kpi_row)

        # Row 3 — scrolling plots
        pg.setConfigOptions(antialias=True)
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#ffffff")
        root.addWidget(self._glw, stretch=1)

        # Three panels: CO2, then each MFC on its own axis. The two flows
        # differ by orders of magnitude, so sharing one axis flattened the
        # smaller one into a straight line. BPR-01 is still recorded and still
        # drives cycle detection — it just no longer needs a panel here.
        self._p_co2   = self._glw.addPlot(row=0, col=0)
        self._p_mfc01 = self._glw.addPlot(row=1, col=0)
        self._p_mfc07 = self._glw.addPlot(row=2, col=0)
        for p, lbl in [(self._p_co2, "CO₂ [vol%]"),
                       (self._p_mfc01, "MFC-01 [SLPM]"),
                       (self._p_mfc07, "MFC-07 [NLPM]")]:
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", lbl)
            p.addLegend(offset=(-10, 10), labelTextSize="8pt")
        self._p_mfc01.setXLink(self._p_co2)
        self._p_mfc07.setXLink(self._p_co2)
        self._p_mfc07.setLabel("bottom", "Elapsed [min]")

        self._c_co2   = self._p_co2.plot([], [], pen=pg.mkPen(PALETTE.purity, width=2), name="CO₂")
        self._c_mfc01 = self._p_mfc01.plot([], [], pen=pg.mkPen(PALETTE.productivity, width=2), name="MFC-01 (CO₂)")
        self._c_mfc07 = self._p_mfc07.plot([], [], pen=pg.mkPen(PALETTE.recovery, width=2), name="MFC-07 (AD-GAS)")

        self._plots = (self._p_co2, self._p_mfc01, self._p_mfc07)

        # Row 4 — controls
        ctrl = QHBoxLayout()
        self.btn_toggle = QPushButton("⏸ Pause")
        self.btn_toggle.clicked.connect(self._toggle)
        ctrl.addWidget(self.btn_toggle)
        ctrl.addStretch(1)
        self._count_lbl = QLabel("0 samples")
        self._count_lbl.setStyleSheet("color:#6b7280;")
        ctrl.addWidget(self._count_lbl)
        root.addLayout(ctrl)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Starting live monitor...")

    def _build_menu(self) -> None:
        """Menu bar: File (save/record/export), Edit, View, Tools.

        Tools duplicates the main window's diagnostics on purpose. Identifying
        a register means watching it move while the rig cycles, and this is the
        window you are already looking at when that happens — sending the
        operator back to another window to open the Address Explorer is exactly
        the friction that made it unfindable.
        """
        mb = self.menuBar()
        # Qt gives the menu bar C++ ownership, but the Python wrappers are
        # untracked; without a reference they are collected and every menu
        # opens empty.
        self._menus: list = []

        m_file = mb.addMenu("&File")
        act_save = QAction("💾 Save data snapshot to CSV…", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_snapshot)
        m_file.addAction(act_save)

        self._act_record = QAction("⏺ Start recording to CSV…", self)
        self._act_record.setShortcut("Ctrl+R")
        self._act_record.triggered.connect(self._toggle_recording)
        m_file.addAction(self._act_record)

        act_pdf = QAction("📄 Export PDF report…", self)
        act_pdf.triggered.connect(self._export_pdf)
        m_file.addAction(act_pdf)
        m_file.addSeparator()
        act_close = QAction("Close", self)
        act_close.triggered.connect(self.close)
        m_file.addAction(act_close)

        m_edit = mb.addMenu("&Edit")
        act_table = QAction("📋 View / edit recorded data…", self)
        act_table.triggered.connect(self._open_data_table)
        m_edit.addAction(act_table)
        act_clear = QAction("🗑 Clear all data", self)
        act_clear.triggered.connect(self._clear_data)
        m_edit.addAction(act_clear)

        m_view = mb.addMenu("&View")
        act_status = QAction("🖥 Machine && sensor status…", self)
        act_status.triggered.connect(self._open_status)
        m_view.addAction(act_status)
        act_reset = QAction("🔍 Reset plot view", self)
        act_reset.setShortcut("Ctrl+0")
        act_reset.triggered.connect(self.reset_plot_view)
        m_view.addAction(act_reset)

        m_tools = mb.addMenu("&Tools")
        act_expl = QAction("🔎 Address Explorer — which register is which?",
                           self)
        act_expl.setShortcut("Ctrl+E")
        act_expl.setToolTip("Watch addresses live and compare them against the "
                            "eServer screen. Read-only.")
        act_expl.triggered.connect(self._open_address_explorer)
        m_tools.addAction(act_expl)

        self._menus = [m_file, m_edit, m_view, m_tools]

    def _open_address_explorer(self) -> None:
        """Open the Address Explorer pointed at the PLC this monitor polls."""
        from psa_analyzer.ui.address_explorer import AddressExplorerWindow
        if self._explorer_window is None:
            # Point it at the very endpoint this monitor is polling, so the
            # two windows can never disagree about which device is being read.
            cfg = self._modbus_config or {}
            self._explorer_window = AddressExplorerWindow(
                host=str(cfg.get("host") or "192.168.1.5"),
                port=int(cfg.get("port") or 502),
                unit=int(cfg.get("unit") or 1),
                parent=None)
        self._explorer_window.show()
        self._explorer_window.raise_()
        self._explorer_window.activateWindow()
        self._say("Address Explorer opened — read-only, it cannot write to "
                  "the PLC.")

    # -- worker lifecycle ---------------------------------------------------
    def _start(self) -> None:
        cbs = dict(
            on_batch=self._on_batch,
            on_status=self._on_status,
            on_failed=self._on_failed,
            on_finished=lambda: self.statusBar().showMessage("Live monitor stopped."),
        )
        if self._modbus_config is not None:
            self._thread, self._worker = start_modbus(
                self._modbus_config, self._interval_ms,
                on_sensors=self._on_sensors, **cbs)
        else:
            self._thread, self._worker = start_live(
                self._source, self._interval_ms, **cbs)
        self.btn_toggle.setText("⏸ Pause")

    def _toggle(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
            self.btn_toggle.setText("▶ Resume")
            self.statusBar().showMessage("Paused.")
        else:
            self._start()

    def _on_failed(self, msg: str) -> None:
        self.statusBar().showMessage("Live monitor failed.")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Live monitor error", msg)

    # -- status -------------------------------------------------------------
    def _on_status(self, msg: str) -> None:
        """Show a worker status line, remembering unreachable-device warnings.

        A blank CO2 card is indistinguishable from a broken program unless the
        reason stays on screen, so "Cannot connect ..." is latched and shown
        alongside every later message until the monitor is restarted.
        """
        if "Cannot connect" in msg:
            self._device_warning = msg
        elif "stand-in" in msg:
            # a substituted value must never quietly pass for a measurement
            self._adgas_warning = msg
        self._say(msg)

    def _say(self, msg: str) -> None:
        """Status-bar text with any latched warnings appended."""
        for warn in (self._device_warning, self._flow_warning,
                     self._adgas_warning, self._idle_reason):
            if warn and warn not in msg:
                msg = f"{msg}   ⚠ {warn}"
        self.statusBar().showMessage(msg)

    # -- data handling ------------------------------------------------------
    def _f(self, row: dict, col: str | None) -> float:
        """Pull a float from a raw row dict; NaN on miss/blank."""
        if not col or col not in row:
            return float("nan")
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return float("nan")

    def _on_batch(self, rows: list) -> None:
        if not rows:
            return
        self._buffer.add_rows(rows)

        # continuous CSV log + live status table
        if self._rec_file is not None:
            self._write_rows(rows)
        if self._status_dialog is not None:
            for r in rows:
                self._status_dialog.update_values(r)

        cm = self._buffer.colmap() or {}

        # Real elapsed-minutes x-axis from the DATE/TIME column (handles the
        # 5-second log interval exactly); fall back to a sample-count estimate
        # if timestamps are missing/unparseable.
        tcol = cm.get("time")
        ts = None
        if tcol:
            ts = parse_timestamps(pd.Series([r.get(tcol) for r in rows]))
        dt_min = self._interval_ms / 1000.0 / 60.0
        for idx, row in enumerate(rows):
            self._n += 1
            tval = ts.iloc[idx] if ts is not None else pd.NaT
            if pd.notna(tval):
                if self._t0 is None:
                    self._t0 = tval
                self._t.append((tval - self._t0).total_seconds() / 60.0)
            else:
                self._t.append(self._n * dt_min)
            self._series["co2"].append(self._f(row, cm.get("co2_pct")))
            self._series["mfc01"].append(self._f(row, cm.get("co2_in")))
            self._series["mfc07"].append(self._f(row, cm.get("ad_gas")))
            self._series["bpr"].append(self._f(row, cm.get("bpr")))

        # trim rolling window
        if len(self._t) > _PLOT_WINDOW:
            keep = slice(-_PLOT_WINDOW, None)
            self._t = self._t[keep]
            for k in self._series:
                self._series[k] = self._series[k][keep]

        # update big numbers from the most recent row
        latest = rows[-1]
        self.card_co2.set_value(self._f(latest, cm.get("co2_pct")), fmt="{:.2f}")
        self.card_mfc01.set_value(self._f(latest, cm.get("co2_in")), fmt="{:.3f}")
        self.card_mfc07.set_value(self._f(latest, cm.get("ad_gas")), fmt="{:.3f}")
        self.card_bpr.set_value(self._f(latest, cm.get("bpr")), fmt="{:.3f}")
        self._count_lbl.setText(f"{len(self._buffer):,} samples")

        # redraw curves (BPR is still buffered and recorded — just not plotted)
        t = np.asarray(self._t)
        self._c_co2.setData(t, np.asarray(self._series["co2"]))
        self._c_mfc01.setData(t, np.asarray(self._series["mfc01"]))
        self._c_mfc07.setData(t, np.asarray(self._series["mfc07"]))
        self._heal_zoom(t)

        # periodic re-analysis with the unchanged pipeline
        self._batches += 1
        if self._batches % self._analyze_every == 0:
            self._reanalyze()

        # trailing-window Purity/Recovery: once immediately so the cards are
        # not blank for the first five minutes, then on the refresh interval
        now = time.monotonic()
        if self._last_kpi_t == 0.0 or now - self._last_kpi_t >= _KPI_REFRESH_S:
            self._last_kpi_t = now
            self._update_window_kpis()

    # -- plot view ----------------------------------------------------------
    def reset_plot_view(self) -> None:
        """Put every panel back on auto-range."""
        for p in self._plots:
            vb = p.getViewBox()
            vb.enableAutoRange(x=True, y=True)
            vb.autoRange()
        self._say("Plot view reset.")

    def _heal_zoom(self, t: np.ndarray) -> None:
        """Undo an absurd zoom-out so an unattended run cannot lose its plots.

        pyqtgraph zooms on the mouse wheel and turns auto-range off as soon as
        the view is touched. One stray scroll over the plot — easily sent by a
        remote-desktop session — can widen the x-range by many orders of
        magnitude, collapsing hours of data into less than a pixel. The data is
        never affected, but the window looks broken and stays that way.

        A view more than 1000x wider than the data it holds is never something
        a person asked for, so recover from it; ordinary zooming is untouched.
        """
        if t.size < 2:
            return
        span = float(t[-1] - t[0])
        if not np.isfinite(span) or span <= 0:
            return
        for p in self._plots:
            vb = p.getViewBox()
            lo, hi = vb.viewRange()[0]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                vb.enableAutoRange(x=True, y=True)
                continue
            if (hi - lo) > span * 1000.0:
                vb.enableAutoRange(x=True, y=True)
                vb.autoRange()

    def _update_window_kpis(self) -> None:
        """Refresh the Purity/Recovery cards from the trailing time window."""
        try:
            m = self._buffer.window_metrics(_KPI_WINDOW_MIN)
        except Exception as exc:        # noqa: BLE001
            self._say(f"Window KPI skipped: {exc}")
            return
        self.card_purity_win.set_value(m["purity"], fmt="{:.2f}")
        self.card_recov_win.set_value(m["recovery"], fmt="{:.2f}")

        # A blank KPI card is ambiguous: it looks identical whether the rig is
        # simply standing still or the acquisition is broken. Say which, so an
        # idle bench is never mistaken for a fault (or the other way round).
        purity_blank = m["purity"] != m["purity"]        # NaN
        if purity_blank and float(m.get("ad_gas_sum") or 0.0) <= 0.0:
            self._idle_reason = (
                "Purity/Recovery are blank because AD-GAS (MFC-07) is not "
                "flowing — nothing is leaving the bed to measure. This is "
                "normal on standby; the numbers appear once the rig runs.")
        elif purity_blank and m.get("n", 0) < 2:
            self._idle_reason = (
                f"Purity/Recovery need a few samples in the last "
                f"{_KPI_WINDOW_MIN:g} min — only {m.get('n', 0)} so far.")
        else:
            self._idle_reason = ""

        # Negative outlet flow is physically impossible, and it drags Recovery
        # negative too. Keep showing the raw number — hiding data is worse —
        # but make it impossible to mistake for a real result.
        if m.get("flow_negative"):
            self._flow_warning = (
                "MFC-07 (AD-GAS) is reading NEGATIVE — Purity/Recovery are "
                "not physically meaningful. Do not record this as data.")
        else:
            self._flow_warning = ""

    def _reanalyze(self) -> None:
        if not self._buffer.can_analyze():
            return
        try:
            params = self._params_provider()
            result = self._buffer.analyze(params)
        except Exception as exc:        # noqa: BLE001
            self.statusBar().showMessage(f"Live analysis skipped: {exc}")
            return
        self.card_purity.set_value(result.final_purity, fmt="{:.2f}")
        self.card_recov.set_value(result.final_recovery, fmt="{:.2f}")
        self.card_prod.set_value(result.final_productivity, fmt="{:.4f}")
        self.card_cycles.set_value(len(result.cycles), fmt="{:.0f}")
        self._say(f"Live — {len(self._buffer):,} samples, "
                  f"{len(result.cycles)} cycles, "
                  f"{result.total_elapsed_hours:.2f} h elapsed.")

    # -- sensor status report ----------------------------------------------
    def _on_sensors(self, report: list) -> None:
        """Cache the one-off sensor report; feed an open status dialog."""
        self._sensor_report = report
        if self._status_dialog is not None:
            self._status_dialog.set_report(report)

        # The gas analyzer is a second device. If none of its channels answer,
        # CO2 will be blank for the whole session — say so plainly rather than
        # leaving an empty card that looks like a broken program.
        gas = [s for s in report if s.get("group") == HORIBA_GROUP]
        if gas and not any(s.get("reachable") for s in gas):
            host = next((s.get("host") for s in gas if s.get("host")), "the analyzer")
            self._device_warning = (
                f"Cannot connect to {host} — CO₂ unavailable, so Purity "
                f"cannot be computed.")
            self._say("Gas analyzer not responding.")

    # -- File menu ----------------------------------------------------------
    def _save_snapshot(self) -> None:
        """Save everything collected so far to a CSV the user picks."""
        if len(self._buffer) == 0:
            QMessageBox.information(self, "Save data", "No data collected yet.")
            return
        default = f"vpsa_live_{datetime.now():%Y%m%d_%H%M}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save data snapshot", default, "CSV files (*.csv)")
        if not path:
            return
        try:
            self._buffer.to_dataframe().to_csv(path, index=False)
        except Exception as exc:        # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {len(self._buffer):,} rows to {Path(path).name}")

    def _toggle_recording(self) -> None:
        if self._rec_file is not None:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin appending every new sample to a CSV on disk (reliable log)."""
        default = f"vpsa_record_{datetime.now():%Y%m%d_%H%M}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Record live data to CSV", default, "CSV files (*.csv)")
        if not path:
            return
        try:
            self._rec_file = open(path, "w", newline="", encoding="utf-8-sig")
        except Exception as exc:        # noqa: BLE001
            QMessageBox.critical(self, "Recording failed", str(exc))
            self._rec_file = None
            return
        self._rec_path = Path(path)
        self._rec_writer = None
        self._rec_count = 0
        # flush any data already buffered, then keep appending live
        existing = self._buffer.rows_snapshot()
        if existing:
            self._write_rows(existing)
        self._act_record.setText("⏹ Stop recording")
        self.statusBar().showMessage(f"● Recording to {self._rec_path.name}")

    def _write_rows(self, rows: list) -> None:
        if self._rec_file is None or not rows:
            return
        try:
            if self._rec_writer is None:
                fields = self._fieldnames(rows)
                self._rec_writer = csv.DictWriter(
                    self._rec_file, fieldnames=fields,
                    extrasaction="ignore", restval="")
                self._rec_writer.writeheader()
            self._rec_writer.writerows(rows)
            self._rec_file.flush()
            self._rec_count += len(rows)
        except Exception as exc:        # noqa: BLE001
            self.statusBar().showMessage(f"Recording error: {exc}")

    @staticmethod
    def _fieldnames(rows: list) -> list:
        """Ordered union of keys across the given rows ('DATE / TIME' first)."""
        fields: list[str] = []
        seen: set = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
        if "DATE / TIME" in fields:
            fields.remove("DATE / TIME")
            fields.insert(0, "DATE / TIME")
        return fields

    def _stop_recording(self) -> None:
        if self._rec_file is None:
            return
        try:
            self._rec_file.close()
        except Exception:
            pass
        name = self._rec_path.name if self._rec_path else "file"
        count = self._rec_count
        self._rec_file = None
        self._rec_writer = None
        self._act_record.setText("⏺ Start recording to CSV…")
        self.statusBar().showMessage(f"Recording stopped — {count:,} rows in {name}")

    def _export_pdf(self) -> None:
        """Run the unchanged pipeline and export the styled PDF report."""
        if not self._buffer.can_analyze():
            QMessageBox.information(
                self, "Export PDF",
                "Not enough data yet (need the cycle sensors + at least one "
                "completed cycle).")
            return
        default = f"vpsa_report_{datetime.now():%Y%m%d_%H%M}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF report", default, "PDF files (*.pdf)")
        if not path:
            return
        try:
            from psa_analyzer.core import export_pdf_report
            result = self._buffer.analyze(self._params_provider())
            export_pdf_report(result, path, source_file="Live Monitor")
        except Exception as exc:        # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"PDF report saved to {Path(path).name}")

    # -- Edit menu ----------------------------------------------------------
    def _open_data_table(self) -> None:
        if self._data_dialog is None:
            self._data_dialog = DataTableDialog(self._buffer, parent=self)
        else:
            self._data_dialog.refresh()
        self._data_dialog.show()
        self._data_dialog.raise_()

    def _clear_data(self) -> None:
        if QMessageBox.question(
                self, "Clear data",
                "Discard all collected samples and reset the plots?") \
                != QMessageBox.Yes:
            return
        self._buffer.clear()
        self._t.clear()
        self._t0 = None
        self._n = 0
        for k in self._series:
            self._series[k].clear()
        for c in (self._c_co2, self._c_mfc01, self._c_mfc07):
            c.setData([], [])
        for card in (self.card_co2, self.card_mfc01, self.card_mfc07,
                     self.card_bpr, self.card_purity_win, self.card_recov_win,
                     self.card_purity, self.card_recov,
                     self.card_prod, self.card_cycles):
            card.set_value("—")
        self._last_kpi_t = 0.0
        self._count_lbl.setText("0 samples")
        self.statusBar().showMessage("Data cleared.")

    # -- View menu ----------------------------------------------------------
    def _open_status(self) -> None:
        if self._status_dialog is None:
            self._status_dialog = SensorStatusDialog(parent=self)
            if self._sensor_report:
                self._status_dialog.set_report(self._sensor_report)
        self._status_dialog.show()
        self._status_dialog.raise_()

    # -- shutdown -----------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._stop_recording()
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
