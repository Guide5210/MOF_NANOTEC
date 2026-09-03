"""
Config dialog for the Modbus live monitor.

Collects the connection settings and the register address of each sensor, then
hands back a config dict for :class:`LiveMonitorWindow` (``modbus_config=...``).
Addresses are editable so the researcher can update them after scanning / after
the PLC is changed — no rebuild needed.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from psa_analyzer.core import plc_map
from psa_analyzer.core.plc_map import COL_BPR, COL_CO2, COL_MFC01, COL_MFC07


class ModbusLiveConfigDialog(QDialog):
    """Ask for IP/unit/word order + the four sensor register addresses."""

    def __init__(self, host: str = "192.168.1.5", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Live Monitor — Modbus setup")
        self.setMinimumWidth(420)

        # Full editable sensor map (the four essentials are also surfaced as
        # quick-edit spinboxes below; everything else is edited via "Edit all…").
        self._sensors = plc_map.default_sensors()

        root = QVBoxLayout(self)

        conn = QGroupBox("Connection")
        cf = QFormLayout(conn)
        self.ip = QLineEdit(host); cf.addRow("IP", self.ip)
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(502)
        cf.addRow("Port", self.port)
        self.unit = QSpinBox(); self.unit.setRange(0, 255); self.unit.setValue(1)
        cf.addRow("Unit / station ID", self.unit)
        self.word_order = QComboBox(); self.word_order.addItems(["little", "big"])
        cf.addRow("Float word order", self.word_order)
        self.interval = QDoubleSpinBox(); self.interval.setRange(0.2, 60.0)
        self.interval.setValue(1.0); self.interval.setSuffix(" s")
        cf.addRow("Poll interval", self.interval)
        root.addWidget(conn)

        regs = QGroupBox("PLC register address of each sensor (32-bit float)")
        rf = QFormLayout(regs)
        # Verified against eServer 2026-08-18 by matching values minute-for-
        # minute; the PLC publishes a compacted window, so these are not the
        # D-numbers (see core/plc_map.py).
        self.a_mfc01 = self._addr(100)
        self.a_mfc07 = self._addr(112)    # NOT published by the PLC
        self.a_bpr   = self._addr(108)    # 4.9 matched eServer exactly
        rf.addRow("MFC-01 (CO₂ inlet)", self.a_mfc01)
        rf.addRow("MFC-07 (AD-GAS)", self.a_mfc07)
        rf.addRow("BPR-01", self.a_bpr)
        root.addWidget(regs)

        # -- stand-in for AD-GAS while the PLC does not publish it -----------
        # MFC-07 is a mass-flow *controller*: it holds whatever setpoint the
        # operator dialled in, so that setpoint is a defensible stand-in for a
        # reading we cannot get. Purity is exact either way (ad_gas is a weight
        # that cancels top and bottom); Recovery and Productivity scale with it,
        # so they are only as good as the assumption that the MFC holds setpoint.
        adg = QGroupBox("AD-GAS stand-in (MFC-07 is not published by the PLC)")
        af = QFormLayout(adg)
        self.chk_adgas = QCheckBox("Use a fixed AD-GAS value")
        self.chk_adgas.setToolTip(
            "Only used when the AD-GAS register cannot be read. Enter the "
            "MFC-07 setpoint.\n"
            "Purity stays exact — ad_gas is a weighting term that cancels.\n"
            "Recovery and Productivity are correct only while the MFC actually "
            "holds that setpoint, so check it on eServer before trusting them.")
        self.chk_adgas.setChecked(False)
        af.addRow(self.chk_adgas)
        self.adgas_value = QDoubleSpinBox()
        self.adgas_value.setRange(0.0, 20.0)
        self.adgas_value.setDecimals(4)
        self.adgas_value.setSingleStep(0.1)
        self.adgas_value.setValue(0.0)
        self.adgas_value.setSuffix(" NLPM")
        self.adgas_value.setEnabled(False)
        self.chk_adgas.toggled.connect(self.adgas_value.setEnabled)
        af.addRow("MFC-07 setpoint", self.adgas_value)
        root.addWidget(adg)

        # -- the gas analyzer is a separate device on the same LAN ----------
        gas = QGroupBox("Gas analyzer — HORIBA VA-5000 (separate device)")
        gf = QFormLayout(gas)
        self.gas_ip = QLineEdit(plc_map.HORIBA_HOST)
        gf.addRow("IP", self.gas_ip)
        self.gas_port = QSpinBox(); self.gas_port.setRange(1, 65535)
        self.gas_port.setValue(plc_map.HORIBA_PORT)
        gf.addRow("Port", self.gas_port)
        self.gas_unit = QSpinBox(); self.gas_unit.setRange(0, 255)
        self.gas_unit.setValue(plc_map.HORIBA_UNIT)
        gf.addRow("Slave address", self.gas_unit)
        # Which of the analyzer's four channels carries CO2 — this is the only
        # gas the PSA pipeline needs, and the channel order is per-machine.
        self.co2_comp = QComboBox()
        for n in (1, 2, 3, 4):
            self.co2_comp.addItem(f"Component {n}  (address {plc_map.horiba_address(n)})", n)
        self.co2_comp.setCurrentIndex(1)     # verified on site: CO2 = Component 2
        self.co2_comp.currentIndexChanged.connect(self._sync_co2_addr)
        gf.addRow("CO₂ is", self.co2_comp)
        self.a_co2 = self._addr(plc_map.horiba_address(2))
        gf.addRow("CO₂ address", self.a_co2)
        root.addWidget(gas)

        # Advanced: edit every sensor address (PTs, gases, temperatures, …)
        adv_row = QHBoxLayout()
        self.btn_edit_all = QPushButton("⚙ Edit all sensor addresses…")
        self.btn_edit_all.clicked.connect(self._edit_all)
        adv_row.addWidget(self.btn_edit_all)
        adv_row.addStretch(1)
        root.addLayout(adv_row)

        note = QLabel(
            "The VA-5000 is its own Modbus/TCP server: slave address is fixed "
            "to 255 and its floats are stored high word first, both handled "
            "automatically. Confirm which Component is CO₂ by comparing "
            "tools/read_horiba.py against the analyzer's own screen.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7280; font-size:9pt;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Start Live")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _addr(self, default: int) -> QSpinBox:
        s = QSpinBox(); s.setRange(0, 65535); s.setValue(default)
        return s

    def _sync_co2_addr(self) -> None:
        """Keep the CO₂ address box in step with the chosen component."""
        self.a_co2.setValue(plc_map.horiba_address(self.co2_comp.currentData()))

    def _edit_all(self) -> None:
        """Open the full sensor-address editor and keep any changes."""
        dlg = AllAddressesDialog(self._sensors, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self._sensors = dlg.sensors()

    def config(self) -> dict:
        """Return the Modbus config dict for LiveMonitorWindow."""
        # Point the VA-5000 channels at the analyzer endpoint, then let the
        # quick-edit spinboxes win for the four canonical sensors; everything
        # else comes from the (possibly edited) full map.
        sensors = plc_map.with_analyzer_endpoint(
            self._sensors, self.gas_ip.text(), self.gas_port.value(),
            self.gas_unit.value())
        sensors = plc_map.with_overrides(sensors, {
            COL_MFC01: self.a_mfc01.value(),
            COL_MFC07: self.a_mfc07.value(),
            COL_BPR:   self.a_bpr.value(),
            COL_CO2:   self.a_co2.value(),
        })
        return {
            "host": self.ip.text().strip(),
            "port": self.port.value(),
            "unit": self.unit.value(),
            "word_order": self.word_order.currentText(),
            "sensors": sensors,
            "ad_gas_fixed": (float(self.adgas_value.value())
                             if self.chk_adgas.isChecked() else 0.0),
        }

    def interval_ms(self) -> int:
        return int(self.interval.value() * 1000)


class AllAddressesDialog(QDialog):
    """Edit the Modbus address + poll flag of every sensor in the map."""

    _HEADERS = ["Tag", "Group", "Address", "Type", "Poll"]

    def __init__(self, sensors: list, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit all sensor addresses")
        self.resize(560, 620)
        self._src = sensors

        root = QVBoxLayout(self)
        note = QLabel(
            "Address = Modbus address (for the low range this equals the Delta "
            "D-number). Untick Poll to skip a sensor. RS-485 controllers can't "
            "be read over Modbus TCP and stay off.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7280; font-size:9pt;")
        root.addWidget(note)

        self._table = QTableWidget(len(sensors), len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(self._HEADERS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)

        self._addr_spins: list[QSpinBox] = []
        self._poll_boxes: list[QCheckBox] = []
        for r, s in enumerate(sensors):
            self._table.setItem(r, 0, QTableWidgetItem(s.tag))
            self._table.setItem(r, 1, QTableWidgetItem(s.group))
            spin = QSpinBox(); spin.setRange(0, 65535); spin.setValue(int(s.address))
            self._table.setCellWidget(r, 2, spin)
            self._addr_spins.append(spin)
            self._table.setItem(r, 3, QTableWidgetItem(s.kind))
            box = QCheckBox()
            box.setChecked(bool(s.poll) and s.bus == "tcp")
            box.setEnabled(s.bus == "tcp")
            self._table.setCellWidget(r, 4, box)
            self._poll_boxes.append(box)
        root.addWidget(self._table, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def sensors(self) -> list:
        """Return a new sensor list with the edited address/poll values."""
        out = []
        for r, s in enumerate(self._src):
            out.append(replace(
                s,
                address=self._addr_spins[r].value(),
                poll=self._poll_boxes[r].isChecked(),
            ))
        return out
