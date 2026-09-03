"""
Modbus Register Scanner — discover which register holds which sensor.

Sweeps a range of holding registers over Modbus TCP and shows each aligned
register pair as the 32-bit float it forms. Match the float values against the
HMI screen to identify the MFC-01 / MFC-07 / BPR-01 / CO2 registers.

Because we read through the HMI (which only exposes a limited address range)
the eServer "D" numbers are NOT our Modbus addresses — so the right approach
is to sweep a wide range, tick "non-zero only", run the rig, and look for the
value that matches the HMI (e.g. CO2 ~85).

Opened from the sidebar; works on the eServer PC through the bundled .exe.
"""

from __future__ import annotations

from pymodbus.client import ModbusTcpClient
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from psa_analyzer.core.modbus_client import ModbusReader


class ModbusScannerDialog(QWidget):
    """Standalone window for sweeping Modbus holding registers."""

    def __init__(self, host: str = "192.168.1.5", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Modbus Register Scanner")
        self.resize(560, 660)
        self.setWindowFlag(Qt.Window, True)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._read)

        self._build_ui(host)

    def _build_ui(self, host: str) -> None:
        root = QVBoxLayout(self)

        box = QGroupBox("Connection")
        form = QFormLayout(box)
        self.ip_input = QLineEdit(host)
        form.addRow("IP", self.ip_input)
        self.port_input = QSpinBox(); self.port_input.setRange(1, 65535); self.port_input.setValue(502)
        form.addRow("Port", self.port_input)
        self.unit_input = QSpinBox(); self.unit_input.setRange(0, 255); self.unit_input.setValue(1)
        form.addRow("Unit / station ID", self.unit_input)
        self.word_order = QComboBox(); self.word_order.addItems(["little", "big"])
        form.addRow("Float word order", self.word_order)
        self.start_input = QSpinBox(); self.start_input.setRange(0, 65535); self.start_input.setValue(0)
        form.addRow("Start address", self.start_input)
        self.end_input = QSpinBox(); self.end_input.setRange(1, 65535); self.end_input.setValue(2000)
        self.end_input.setToolTip("Sweep up to this address. Unreadable chunks are skipped.")
        form.addRow("End address", self.end_input)
        root.addWidget(box)

        ctrl = QHBoxLayout()
        from PySide6.QtWidgets import QPushButton
        self.btn_read = QPushButton("Scan once")
        self.btn_read.clicked.connect(self._read)
        ctrl.addWidget(self.btn_read)
        self.auto = QCheckBox("Auto-refresh (1 s)")
        self.auto.toggled.connect(self._toggle_auto)
        ctrl.addWidget(self.auto)
        self.nonzero = QCheckBox("Non-zero only")
        self.nonzero.setChecked(True)
        self.nonzero.toggled.connect(self._read)
        ctrl.addWidget(self.nonzero)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        self.status = QLabel("Set End address, press Scan once. Tick 'Non-zero only' "
                             "to hide empty registers.")
        self.status.setStyleSheet("color:#6b7280;")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Addr (pair)", "Words (hex)", "Float32"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 180)
        root.addWidget(self.table, stretch=1)

        hint = QLabel(
            "Each row = one float at [addr, addr+1]. Run the rig, then look "
            "for the value that matches the HMI: CO₂ ≈ 85, the MFC flows, BPR. "
            "If floats look wrong, switch word order.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280; font-size:9pt;")
        root.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        root.addWidget(bb)

    # -- actions ------------------------------------------------------------
    def _toggle_auto(self, on: bool) -> None:
        if on:
            self._read(); self._timer.start()
        else:
            self._timer.stop()

    def _read(self) -> None:
        host = self.ip_input.text().strip()
        wo = self.word_order.currentText()
        start, end = self.start_input.value(), self.end_input.value()
        if end <= start:
            self.status.setText("End address must be greater than Start.")
            return
        try:
            reader = ModbusReader(host, port=self.port_input.value(),
                                  device_id=self.unit_input.value(), word_order=wo)
            if not reader.connect():
                self.status.setText(f"Could not connect to {host}:{self.port_input.value()}")
                return
            words = reader.read_range(start, end)
            reader.close()
        except Exception as exc:        # noqa: BLE001
            self.status.setText(f"Read error: {exc}")
            return

        self.table.setRowCount(0)
        if not words:
            self.status.setText("Connected but no registers were readable in "
                                "this range. Try a different range / unit id.")
            return

        nonzero_only = self.nonzero.isChecked()
        rows = []
        a = start
        while a <= end:
            if a in words and (a + 1) in words:
                w0, w1 = words[a], words[a + 1]
                try:
                    fval = float(ModbusTcpClient.convert_from_registers(
                        [w0, w1], ModbusTcpClient.DATATYPE.FLOAT32, word_order=wo))
                except Exception:
                    fval = float("nan")
                if not nonzero_only or (fval == fval and abs(fval) > 1e-9):
                    rows.append((a, w0, w1, fval))
            a += 2

        self.table.setRowCount(len(rows))
        for i, (addr, w0, w1, fval) in enumerate(rows):
            self._set(i, 0, f"{addr}–{addr+1}")
            self._set(i, 1, f"0x{w0:04X} 0x{w1:04X}")
            self._set(i, 2, f"{fval:.4f}")
        self.status.setText(
            f"Scanned {start}–{end}: {len(words)} readable words, "
            f"showing {len(rows)} float(s)"
            f"{' (non-zero)' if nonzero_only else ''}, {wo} word order.")

    def _set(self, r: int, c: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, c, item)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
