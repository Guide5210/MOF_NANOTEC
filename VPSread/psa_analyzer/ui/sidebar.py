"""
Left sidebar: file selection, dynamic parameter inputs, action buttons.

Emits high-level signals so the main window can wire actions to handlers
without the sidebar needing any knowledge of the analyser or workers.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)

from psa_analyzer.core import AnalysisParams


class Sidebar(QFrame):
    """
    Control panel docked to the left edge of the main window.

    Signals
    -------
    file_selected(Path)
    analyze_requested()
    export_csv_requested()
    export_graphs_requested()
    theme_toggled()
    """

    file_selected         = Signal(Path)
    file_cleared          = Signal()
    analyze_requested     = Signal()
    live_requested        = Signal()
    scan_requested        = Signal()
    export_csv_requested  = Signal()
    export_graphs_requested = Signal()
    export_pdf_requested  = Signal()
    theme_toggled         = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(280)

        self._selected_path: Path | None = None
        self._build_ui()

    # -- UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(8)

        # --- Header ---
        title = QLabel("PSA Analyzer")
        title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        v.addWidget(title)
        subtitle = QLabel("Pressure Swing Adsorption")
        subtitle.setStyleSheet("color: #6b7280; font-size: 9pt;")
        v.addWidget(subtitle)

        # --- File section ---
        v.addWidget(self._section_header("Data Source"))

        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #6b7280; font-size: 9pt; padding: 4px 0;")
        file_row.addWidget(self.file_label, stretch=1)
        self.btn_clear = QPushButton("✕")
        self.btn_clear.setFixedSize(22, 22)
        self.btn_clear.setToolTip("Remove the selected file")
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(self._on_clear)
        file_row.addWidget(self.btn_clear)
        v.addLayout(file_row)

        self.btn_browse = QPushButton("Browse Excel file...")
        self.btn_browse.clicked.connect(self._on_browse)
        v.addWidget(self.btn_browse)

        # --- Parameters section ---
        v.addWidget(self._section_header("Experimental Parameters"))

        self.adsorbent_input = QLineEdit("CALF-20")
        v.addLayout(self._labelled("Adsorbent", self.adsorbent_input))

        # Bed volume = spin box + unit dropdown (defaults to mL to avoid
        # the easy-to-mistype µL).
        bed_row = QHBoxLayout()
        bed_row.setContentsMargins(0, 0, 0, 0)
        bed_row.setSpacing(4)
        self.bed_volume_input = QDoubleSpinBox()
        self.bed_volume_input.setRange(0.001, 1e6)
        self.bed_volume_input.setDecimals(3)
        self.bed_volume_input.setValue(60.0)
        self.bed_volume_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.bed_volume_input.setToolTip(
            "Volume per single bed (60 mL standard). "
            "Productivity uses total = steps/cycle × bed volume."
        )
        self.bed_volume_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bed_row.addWidget(self.bed_volume_input, stretch=1)
        self.bed_unit_input = QComboBox()
        self.bed_unit_input.addItems(["mL", "µL", "L"])
        self.bed_unit_input.setCurrentText("mL")
        self.bed_unit_input.setFixedWidth(62)
        bed_row.addWidget(self.bed_unit_input)
        v.addLayout(self._labelled_row("Bed volume", bed_row))

        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 32)
        self.steps_input.setValue(4)
        v.addLayout(self._labelled("Steps / cycle", self.steps_input))

        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(-50.0, 300.0)
        self.temperature_input.setDecimals(1)
        self.temperature_input.setValue(40.0)
        self.temperature_input.setSuffix(" °C")
        self.temperature_input.setToolTip(
            "Recorded on the PDF/Excel report and in plot titles as the "
            "experiment's condition. It does not enter any calculation."
        )
        v.addLayout(self._labelled("Temperature", self.temperature_input))

        # "Rolling window" used to sit here. It was never read by the pipeline
        # — smoothing uses a fixed adaptive window (3 cycles for the first 4,
        # 5 thereafter) to match the lab's Excel — so the control only looked
        # like it did something. Removed rather than left as a decoy.

        self.final_cycle_input = QSpinBox()
        self.final_cycle_input.setRange(1, 9999)
        self.final_cycle_input.setValue(3)
        self.final_cycle_input.setToolTip(
            "Final KPIs = average of every cycle from this number to the "
            "last cycle (lab's revised Excel starts at cycle 3)."
        )
        v.addLayout(self._labelled("Final from cycle", self.final_cycle_input))

        # --- Actions ---
        v.addWidget(self._section_header("Actions"))

        self.btn_analyze = QPushButton("Start Analysis")
        self.btn_analyze.setObjectName("PrimaryButton")
        self.btn_analyze.setToolTip(
            "Runs the pipeline on the data already loaded — press it again "
            "after changing any parameter to recompute without re-reading "
            "the file."
        )
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze_requested.emit)
        v.addWidget(self.btn_analyze)

        self.btn_live = QPushButton("● Live Monitor (PLC)")
        self.btn_live.setToolTip(
            "Open a real-time monitor that reads eServer's live CSV "
            "and updates numbers + charts continuously."
        )
        self.btn_live.clicked.connect(self.live_requested.emit)
        v.addWidget(self.btn_live)

        self.btn_scan = QPushButton("⚙ Modbus Scanner")
        self.btn_scan.setToolTip(
            "Probe the PLC/HMI Modbus registers to find which one holds "
            "MFC-01 / MFC-07 / BPR-01 / CO2 and confirm the float word order."
        )
        self.btn_scan.clicked.connect(self.scan_requested.emit)
        v.addWidget(self.btn_scan)

        self.btn_export_csv = QPushButton("Export Excel (.xlsx)")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self.export_csv_requested.emit)
        v.addWidget(self.btn_export_csv)

        self.btn_export_graphs = QPushButton("Export Graphs (SVG/PNG)")
        self.btn_export_graphs.setEnabled(False)
        self.btn_export_graphs.clicked.connect(self.export_graphs_requested.emit)
        v.addWidget(self.btn_export_graphs)

        self.btn_export_pdf = QPushButton("Export PDF Report")
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_pdf.clicked.connect(self.export_pdf_requested.emit)
        v.addWidget(self.btn_export_pdf)

        # --- Progress bar ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        v.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #6b7280; font-size: 9pt;")
        self.progress_label.setWordWrap(True)
        v.addWidget(self.progress_label)

        # --- Spacer + theme toggle at bottom ---
        v.addStretch(1)
        self.btn_theme = QPushButton("Toggle Light / Dark")
        self.btn_theme.clicked.connect(self.theme_toggled.emit)
        v.addWidget(self.btn_theme)

    def _section_header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionHeader")
        return lbl

    def _labelled(self, text: str, widget: QWidget) -> QHBoxLayout:
        """A two-column row: label on the left, widget on the right."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(text)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet("color: #6b7280;")
        row.addWidget(lbl)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(widget)
        return row

    def _labelled_row(self, text: str, inner: QHBoxLayout) -> QHBoxLayout:
        """Like _labelled but accepts an existing layout instead of a single widget."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(text)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet("color: #6b7280;")
        row.addWidget(lbl)
        row.addLayout(inner, stretch=1)
        return row

    # -- Slots --------------------------------------------------------------
    def _on_browse(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select raw PSA Excel file", "",
            "Excel files (*.xlsx *.xls);;All files (*)"
        )
        if not path_str:
            return
        self.set_file(Path(path_str))
        self.file_selected.emit(self._selected_path)

    # -- Public API ---------------------------------------------------------
    def browse(self) -> None:
        """Open the file chooser (also reachable from the File menu)."""
        self._on_browse()

    def clear_file(self) -> None:
        """Drop the selected file (also reachable from the File menu)."""
        self._on_clear()

    def selected_path(self) -> Path | None:
        """Return the currently selected file path, or None."""
        return self._selected_path

    def set_file(self, path: Path) -> None:
        """Programmatically set the selected file (used by drag-and-drop)."""
        self._selected_path = Path(path)
        self.file_label.setText(self._selected_path.name)
        self.file_label.setStyleSheet(
            "color: #111827; font-size: 9pt; padding: 4px 0;")
        self.btn_analyze.setEnabled(True)
        self.btn_clear.setVisible(True)

    def _on_clear(self) -> None:
        """Remove the currently selected file."""
        self._selected_path = None
        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet(
            "color: #6b7280; font-size: 9pt; padding: 4px 0;")
        self.btn_clear.setVisible(False)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText("Start Analysis")
        self.set_results_available(False)
        self.set_progress(0, "")
        self.file_cleared.emit()

    def current_params(self) -> AnalysisParams:
        """Snapshot the input fields into an AnalysisParams instance."""
        return AnalysisParams(
            bed_volume_mL     = self._bed_volume_in_mL(),
            steps_per_cycle   = int(self.steps_input.value()),
            final_cycle_start = int(self.final_cycle_input.value()),
            temperature_C     = float(self.temperature_input.value()),
            adsorbent_name    = self.adsorbent_input.text().strip() or "Unnamed",
        )

    def apply_filename_params(self, fp) -> None:
        """Auto-fill matching inputs from a parsed FilenameParams."""
        if fp is None:
            return
        self.adsorbent_input.setText(fp.material)
        self.temperature_input.setValue(float(fp.temp_C))

    def _bed_volume_in_mL(self) -> float:
        """Convert the bed volume input to millilitres based on the unit dropdown."""
        value = float(self.bed_volume_input.value())
        unit = self.bed_unit_input.currentText()
        if unit == "µL":
            return value * 1e-3
        if unit == "L":
            return value * 1e3
        return value  # mL

    def set_progress(self, pct: int, msg: str) -> None:
        """Update the progress bar + label from worker signals."""
        self.progress.setValue(pct)
        self.progress_label.setText(msg)

    def set_results_available(self, available: bool) -> None:
        """Enable/disable export buttons based on whether results exist."""
        self.btn_export_csv.setEnabled(available)
        self.btn_export_graphs.setEnabled(available)
        self.btn_export_pdf.setEnabled(available)

    def set_analysis_running(self, running: bool) -> None:
        """Disable controls while an analysis is in flight."""
        self.btn_analyze.setEnabled(not running and self._selected_path is not None)
        self.btn_browse.setEnabled(not running)
