"""
Main application window — composes the sidebar, KPI strip, and plot canvas.

Workflow:
    Browse / drag-and-drop xlsx
        -> LoadWorker reads workbook
        -> DataCropperDialog lets user pick a time window
        -> AnalysisWorker runs the PSA pipeline on the cropped DataFrame
        -> results pushed into KPI strip + plot canvas
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from psa_analyzer.core import (
    AnalysisResult,
    export_dashboard,
    export_pdf_report,
    export_single_metric,
    export_xlsx_report,
)
from psa_analyzer.core.filename_parser import classify_group, try_parse_filename
from psa_analyzer.core.steady_state import extract_results, find_steady_state
from psa_analyzer.ui.cropper import DataCropperDialog
from psa_analyzer.ui.experiment_picker import ExperimentPickerDialog
from psa_analyzer.ui.kpi_cards import KPIStrip
from psa_analyzer.ui.live_window import LiveMonitorWindow
from psa_analyzer.ui.plot_canvas import PlotCanvas
from psa_analyzer.ui.raw_signals import RawSignalPanel
from psa_analyzer.ui.sidebar import Sidebar
from psa_analyzer.ui.theme import Theme, get_stylesheet
from psa_analyzer.workers import start_analysis, start_load


class MainWindow(QMainWindow):
    """Top-level QMainWindow. Owns sidebar, KPI strip, plot canvas, workers."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PSA Analyzer — Demo 1")
        self.resize(1400, 880)
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)

        self._result: Optional[AnalysisResult] = None
        self._raw_df: Optional[pd.DataFrame] = None
        self._colmap: Optional[dict[str, str]] = None
        self._cropped_df: Optional[pd.DataFrame] = None

        self._load_thread: Optional[QThread] = None
        self._load_worker = None
        self._analysis_thread: Optional[QThread] = None
        self._analysis_worker = None
        self._live_window: Optional[LiveMonitorWindow] = None
        self._scanner_window = None
        self._explorer_window = None

        self._theme = Theme.LIGHT

        self._build_ui()
        self._build_menubar()
        self._wire_signals()
        self.apply_theme(Theme.LIGHT)

    # -- UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        workspace = QWidget()
        v = QVBoxLayout(workspace)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # Sticky warning banner — shown when results look suspicious.
        self.warning_banner = QLabel("")
        self.warning_banner.setWordWrap(True)
        self.warning_banner.setStyleSheet(
            "background-color: #fee2e2; color: #991b1b; "
            "border: 1px solid #fca5a5; border-radius: 6px; "
            "padding: 8px 12px; font-weight: 600;"
        )
        self.warning_banner.setVisible(False)
        v.addWidget(self.warning_banner)

        self.kpi_strip = KPIStrip()
        v.addWidget(self.kpi_strip)

        # Left: the per-cycle KPI plots. Right: the untouched instrument
        # signals behind them. A splitter so either half can be widened —
        # chasing a sensor problem needs the right, reporting needs the left.
        self.plot_canvas = PlotCanvas()
        self.raw_panel = RawSignalPanel()
        self.plot_split = QSplitter(Qt.Horizontal)
        self.plot_split.addWidget(self.plot_canvas)
        self.plot_split.addWidget(self.raw_panel)
        self.plot_split.setStretchFactor(0, 1)
        self.plot_split.setStretchFactor(1, 1)
        self.plot_split.setChildrenCollapsible(True)
        self.plot_split.setHandleWidth(10)
        # Both panes hold buttons and combos whose size hints add up to a wide
        # minimum, which stopped the handle roughly 40 % across. An Ignored
        # horizontal policy lets the splitter hand each pane whatever width the
        # user drags to; the loser is simply clipped, which is the point.
        for pane in (self.plot_canvas, self.raw_panel):
            pane.setMinimumWidth(0)
            pane.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        # Dragging is the main gesture, but one click for the three useful
        # extremes saves fighting the mouse when switching between "reporting"
        # and "diagnosing".
        split_bar = QHBoxLayout()
        split_bar.setSpacing(6)
        split_bar.addWidget(QLabel("Plot area:"))
        for text, tip, frac in (
            ("◀ KPI full", "Show only the per-cycle KPI plots", 1.0),
            ("Split 50/50", "Give both halves equal width", 0.5),
            ("Raw full ▶", "Show only the raw instrument signals", 0.0),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setMaximumWidth(120)
            b.clicked.connect(lambda _=False, f=frac: self.set_plot_split(f))
            split_bar.addWidget(b)
        split_bar.addStretch(1)
        v.addLayout(split_bar)

        v.addWidget(self.plot_split, stretch=1)

        root.addWidget(workspace, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Ready. Drop an Excel file here or use Browse to begin."
        )

    def _build_menubar(self) -> None:
        """Menu bar over the sidebar's buttons, plus the register-hunting tools.

        Everything here re-uses an existing handler — the menu is a second door
        to the same rooms, not a second implementation. The one entry that is
        only here is Address Explorer, which belongs with the diagnostics
        rather than in the analysis sidebar.
        """
        bar = self.menuBar()
        # Hold references. Qt gives the menu bar C++ ownership of these menus,
        # but the Python wrappers are not tracked, so without a reference they
        # are garbage-collected and every menu opens empty — the same silent
        # class of failure as the missing `datetime` import in the live window.
        self._menus: list = []

        m_file = bar.addMenu("&File")
        act = m_file.addAction("&Open Excel file...")
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self.sidebar.browse)
        act = m_file.addAction("&Close file")
        act.triggered.connect(self.sidebar.clear_file)
        m_file.addSeparator()
        act = m_file.addAction("Start &Analysis")
        act.setShortcut("Ctrl+R")
        act.triggered.connect(self._on_analyze_again)
        m_file.addSeparator()
        m_exp = m_file.addMenu("&Export")
        m_exp.addAction("Excel (.xlsx)...").triggered.connect(self._on_export_csv)
        m_exp.addAction("Graphs (SVG/PNG)...").triggered.connect(
            self._on_export_graphs)
        m_exp.addAction("PDF report...").triggered.connect(self._on_export_pdf)
        m_file.addSeparator()
        act = m_file.addAction("E&xit")
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)

        m_tools = bar.addMenu("&Tools")
        act = m_tools.addAction("● &Live Monitor (PLC)...")
        act.triggered.connect(self._on_open_live)
        m_tools.addSeparator()
        act = m_tools.addAction("&Address Explorer — which register is which?")
        act.setShortcut("Ctrl+E")
        act.setToolTip("Watch chosen addresses live and compare them against "
                       "the eServer screen until one matches")
        act.triggered.connect(self._on_open_address_explorer)
        act = m_tools.addAction("⚙ Modbus &Scanner (sweep a range)...")
        act.triggered.connect(self._on_open_scanner)

        m_view = bar.addMenu("&View")
        for text, frac in (("KPI plots full width", 1.0),
                           ("Split 50 / 50", 0.5),
                           ("Raw signals full width", 0.0)):
            m_view.addAction(text).triggered.connect(
                lambda _=False, f=frac: self.set_plot_split(f))
        m_view.addSeparator()
        act = m_view.addAction("Toggle &dark theme")
        act.setShortcut("Ctrl+D")
        act.triggered.connect(self._on_toggle_theme)

        self._menus = [m_file, m_exp, m_tools, m_view]

    def set_plot_split(self, left_fraction: float) -> None:
        """Give the left pane ``left_fraction`` of the width (0.0-1.0)."""
        total = max(self.plot_split.width(), 1)
        left = int(round(total * max(0.0, min(1.0, left_fraction))))
        self.plot_split.setSizes([left, total - left])

    def _wire_signals(self) -> None:
        self.sidebar.file_selected.connect(self._on_file_selected)
        self.sidebar.file_cleared.connect(self._on_file_cleared)
        self.sidebar.analyze_requested.connect(self._on_analyze_again)
        self.sidebar.live_requested.connect(self._on_open_live)
        self.sidebar.scan_requested.connect(self._on_open_scanner)
        self.sidebar.export_csv_requested.connect(self._on_export_csv)
        self.sidebar.export_graphs_requested.connect(self._on_export_graphs)
        self.sidebar.export_pdf_requested.connect(self._on_export_pdf)
        self.sidebar.theme_toggled.connect(self._on_toggle_theme)

    # -- Drag and drop ------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".xlsx", ".xls")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in (".xlsx", ".xls") and p.exists():
                # Just stage the file — don't auto-load. User clicks
                # 'Start Analysis' (or removes it) when ready.
                self.sidebar.set_file(p)
                self._raw_df = None
                self._colmap = None
                self._cropped_df = None
                self.status.showMessage(
                    f"Loaded reference to {p.name}. Click 'Start Analysis' to begin."
                )
                event.acceptProposedAction()
                return
        event.ignore()

    def _on_file_cleared(self) -> None:
        """User removed the selected file — drop all cached state."""
        self._raw_df = None
        self._colmap = None
        self._cropped_df = None
        self._result = None
        self.kpi_strip.clear()
        self.plot_canvas.clear()
        self.raw_panel.clear()
        self.warning_banner.setVisible(False)
        self.status.showMessage(
            "Ready. Drop an Excel file here or use Browse to begin."
        )

    # -- Stage 1: load workbook --------------------------------------------
    def _on_file_selected(self, path: Path) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return

        self._raw_df = None
        self._colmap = None
        self._cropped_df = None

        self.sidebar.set_analysis_running(True)
        self.kpi_strip.clear()
        self.plot_canvas.clear()
        self.raw_panel.clear()
        self.status.showMessage(f"Loading {path.name}...")

        self._load_thread, self._load_worker = start_load(
            path,
            on_progress=self._on_progress,
            on_finished=self._on_load_finished,
            on_failed=self._on_failed,
        )
        self._load_thread.finished.connect(self._on_load_thread_finished)

    def _on_load_thread_finished(self) -> None:
        self._load_thread = None
        self._load_worker = None

    def _on_load_finished(self, df, colmap, path) -> None:
        self._raw_df = df
        self._colmap = colmap
        self.sidebar.set_analysis_running(False)

        # Try to parse experimental params from the filename and auto-fill.
        fp = try_parse_filename(str(path))
        if fp is not None:
            self.sidebar.apply_filename_params(fp)
            group = classify_group(fp)
            msg = (f"Loaded {len(df):,} rows from {path.name}. "
                   f"Detected: {fp.material} • {fp.temp_C}°C • "
                   f"{fp.flow_mlmin} mL/min • {fp.pressure_barg} bar • "
                   f"{fp.ads_time_min} min  ({group})")
        else:
            msg = (f"Loaded {len(df):,} rows from {path.name} — "
                   f"choose an experiment.")
        self.status.showMessage(msg)

        self._open_picker()

    # -- Stage 2a: auto-detect experiments ---------------------------------
    def _open_picker(self) -> None:
        if self._raw_df is None or self._colmap is None:
            return
        picker = ExperimentPickerDialog(self._raw_df, self._colmap, parent=self)
        if picker.exec() != QDialog.Accepted:
            self.status.showMessage(
                "Cancelled. Click 'Start Analysis' to re-open the picker."
            )
            return
        action, cropped = picker.outcome()
        if action == "analyze" and cropped is not None and not cropped.empty:
            self._cropped_df = cropped
            self._run_analysis()
        elif action == "manual":
            self._open_cropper()

    # -- Stage 2b: manual crop dialog --------------------------------------
    def _open_cropper(self) -> None:
        if self._raw_df is None or self._colmap is None:
            return
        dlg = DataCropperDialog(self._raw_df, self._colmap, parent=self)
        if dlg.exec() == QDialog.Accepted:
            cropped = dlg.cropped_df()
            if cropped is None or cropped.empty:
                QMessageBox.warning(self, "Empty crop",
                                    "No rows selected. Please widen the range.")
                return
            self._cropped_df = cropped
            self._run_analysis()
        else:
            self.status.showMessage(
                "Crop cancelled. Click 'Start Analysis' to re-open."
            )

    def _on_analyze_again(self) -> None:
        """Sidebar 'Start Analysis' button — re-opens the picker if data
        is already loaded; otherwise re-loads from the selected path."""
        if self._raw_df is not None:
            self._open_picker()
        elif self.sidebar.selected_path() is not None:
            self._on_file_selected(self.sidebar.selected_path())

    # -- Live monitor (Modbus PLC direct, or eServer CSV) ------------------
    def _on_open_live(self) -> None:
        """Ask the data source (Modbus or CSV), then open the Live Monitor."""
        box = QMessageBox(self)
        box.setWindowTitle("Live Monitor — choose data source")
        box.setText("Read live data from:")
        btn_modbus = box.addButton("Modbus (PLC direct)", QMessageBox.AcceptRole)
        btn_csv    = box.addButton("eServer CSV file", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_modbus:
            self._open_live_modbus()
        elif clicked is btn_csv:
            self._open_live_csv()

    def _open_live_modbus(self) -> None:
        from psa_analyzer.ui.modbus_live_config import ModbusLiveConfigDialog
        dlg = ModbusLiveConfigDialog(host="192.168.1.5", parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._live_window = LiveMonitorWindow(
            None,
            params_provider=self.sidebar.current_params,
            interval_ms=dlg.interval_ms(),
            parent=None,
            modbus_config=dlg.config(),
        )
        self._live_window.show()
        self.status.showMessage("Live monitor (Modbus) opened.")

    def _open_live_csv(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select eServer live CSV", "",
            "CSV files (*.csv);;All files (*)"
        )
        source = path_str
        if not source:
            source = QFileDialog.getExistingDirectory(
                self, "...or select the folder eServer writes CSVs to"
            )
        if not source:
            return
        self._live_window = LiveMonitorWindow(
            source,
            params_provider=self.sidebar.current_params,
            interval_ms=1000,
            parent=None,
        )
        self._live_window.show()
        self.status.showMessage(f"Live monitor opened on {source}")

    def _on_open_scanner(self) -> None:
        """Open the Modbus Register Scanner (register-map discovery tool)."""
        from psa_analyzer.ui.modbus_scanner import ModbusScannerDialog
        self._scanner_window = ModbusScannerDialog(host="192.168.1.5", parent=None)
        self._scanner_window.show()
        self.status.showMessage("Modbus Scanner opened.")

    def _on_open_address_explorer(self) -> None:
        """Open the Address Explorer (identify a register against eServer)."""
        from psa_analyzer.ui.address_explorer import AddressExplorerWindow
        if self._explorer_window is None:
            self._explorer_window = AddressExplorerWindow(
                host="192.168.1.5", port=502, unit=1, theme=self._theme,
                parent=None)
        self._explorer_window.show()
        self._explorer_window.raise_()
        self._explorer_window.activateWindow()
        self.status.showMessage(
            "Address Explorer opened — read-only, it cannot write to the PLC.")

    # -- Stage 3: run analysis ---------------------------------------------
    def _run_analysis(self) -> None:
        if self._cropped_df is None or self._colmap is None:
            return
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return

        params = self.sidebar.current_params()
        self.sidebar.set_analysis_running(True)
        self.sidebar.set_results_available(False)
        self.kpi_strip.clear()
        self.plot_canvas.clear()
        self.raw_panel.clear()
        self.warning_banner.setVisible(False)
        self.status.showMessage(
            f"Analyzing cropped data ({len(self._cropped_df):,} rows, "
            f"steps/cycle = {params.steps_per_cycle})..."
        )

        self._analysis_thread, self._analysis_worker = start_analysis(
            self._cropped_df, self._colmap, params,
            on_progress=self._on_progress,
            on_finished=self._on_analysis_finished,
            on_failed=self._on_failed,
        )
        self._analysis_thread.finished.connect(self._on_analysis_thread_finished)

    def _on_analysis_thread_finished(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None

    def _on_progress(self, pct: int, msg: str) -> None:
        self.sidebar.set_progress(pct, msg)
        self.status.showMessage(msg)

    def _on_analysis_finished(self, result: AnalysisResult) -> None:
        self._result = result
        self.sidebar.set_analysis_running(False)
        self.sidebar.set_results_available(True)
        try:
            self.kpi_strip.update_from_result(result)
            self.plot_canvas.plot_result(result)
            # row-level table = every raw column of the workbook plus elapsed_s
            self.raw_panel.set_dataframe(result.rows)
        except Exception as exc:
            import traceback
            QMessageBox.critical(
                self, "Render failed",
                f"{exc}\n\n{traceback.format_exc()}",
            )
            return
        # Run steady-state finder on the cropped data — surface the SS
        # KPIs alongside the cycle-based ones in the status bar.
        ss_msg = ""
        try:
            params = self.sidebar.current_params()
            ss_win = find_steady_state(
                self._cropped_df, self._colmap,
                temp_setpoint_C=params.temperature_C,
            )
            if ss_win is not None:
                ss = extract_results(
                    self._cropped_df, self._colmap, ss_win,
                    bed_volume_mL=params.bed_volume_mL,
                    n_beds=params.steps_per_cycle,
                )
                ss_msg = (f"  |  SS window: {ss.window.n_rows} rows, "
                          f"Purity={ss.purity_pct:.2f}%  "
                          f"Recovery={ss.recovery_pct:.2f}%  "
                          f"Prod={ss.productivity_tCO2_m3_day:.4f}")
        except Exception:
            pass

        self.status.showMessage(
            f"Done — {len(result.cycles)} cycles, "
            f"{result.total_elapsed_hours:.2f} h elapsed." + ss_msg
        )

        # Sanity check: raw recovery shouldn't exceed ~100%. Anything above
        # 150% usually means the crop window includes warmup/cooldown noise
        # or the wrong experiment block was selected.
        try:
            max_recov = float(result.cycles["recovery_pct"].max())
        except Exception:
            max_recov = float("nan")
        if max_recov == max_recov and max_recov > 150.0:
            self.warning_banner.setText(
                f"⚠ Recovery max = {max_recov:.1f}% (> 150%) — the crop "
                f"window probably includes warmup/cooldown or the wrong "
                f"experiment block. Re-open the picker and tighten the "
                f"selection."
            )
            self.warning_banner.setVisible(True)
        else:
            self.warning_banner.setVisible(False)

    def _on_failed(self, message: str) -> None:
        self.sidebar.set_analysis_running(False)
        self.sidebar.set_progress(0, "Failed.")
        self.status.showMessage("Operation failed — see error dialog.")
        QMessageBox.critical(self, "Operation failed", message)

    # -- Slots: exports -----------------------------------------------------
    def _on_export_csv(self) -> None:
        """Export styled Excel (.xlsx) — replaces the old plain CSV path."""
        if self._result is None:
            return
        src = self.sidebar.selected_path()
        default = (src.parent / f"{src.stem}_summary_UI.xlsx") if src \
            else Path("summary_UI.xlsx")
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export Excel report",
            str(default), "Excel files (*.xlsx)"
        )
        if not path_str:
            return
        try:
            out = export_xlsx_report(
                self._result, Path(path_str),
                source_file=src.name if src else None,
            )
            self.status.showMessage(f"Excel report written: {out}")
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Excel export failed",
                                 f"{exc}\n\n{traceback.format_exc()}")

    def _on_export_pdf(self) -> None:
        if self._result is None:
            return
        src = self.sidebar.selected_path()
        default = (src.parent / f"{src.stem}_report_UI.pdf") if src \
            else Path("report_UI.pdf")
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export PDF report",
            str(default), "PDF files (*.pdf)"
        )
        if not path_str:
            return
        try:
            out = export_pdf_report(
                self._result, Path(path_str),
                source_file=src.name if src else None,
            )
            self.status.showMessage(f"PDF report written: {out}")
            QMessageBox.information(self, "Export complete",
                                    f"PDF report saved to:\n{out}")
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "PDF export failed",
                                 f"{exc}\n\n{traceback.format_exc()}")

    def _on_export_graphs(self) -> None:
        if self._result is None:
            return
        out_dir_str = QFileDialog.getExistingDirectory(
            self, "Choose output folder for figures"
        )
        if not out_dir_str:
            return
        out_dir = Path(out_dir_str)
        stem = self.sidebar.selected_path().stem if self.sidebar.selected_path() else "psa"

        try:
            export_dashboard(self._result,       out_dir / f"{stem}_dashboard.png")
            export_dashboard(self._result,       out_dir / f"{stem}_dashboard.svg")
            export_single_metric(self._result, "purity",       out_dir / f"{stem}_purity.png")
            export_single_metric(self._result, "recovery",     out_dir / f"{stem}_recovery.png")
            export_single_metric(self._result, "productivity", out_dir / f"{stem}_productivity.png")
            self.status.showMessage(f"Figures written to {out_dir}")
            QMessageBox.information(
                self, "Export complete",
                f"5 figures written to:\n{out_dir}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # -- Slots: theme -------------------------------------------------------
    def _on_toggle_theme(self) -> None:
        new_theme = Theme.DARK if self._theme is Theme.LIGHT else Theme.LIGHT
        self.apply_theme(new_theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.setStyleSheet(get_stylesheet(theme))
        self.plot_canvas.apply_theme(theme)
        self.raw_panel.apply_theme(theme)
        if self._explorer_window is not None:
            self._explorer_window.apply_theme(theme)
