"""
Address Explorer — identify a Modbus register by watching it against eServer.

Modbus sends numbers with no labels. The PLC answers 2250 addresses and every
one of them is anonymous, so the only way to learn that (say) address 108 is
BPR-01 is to read a value off the eServer screen and find the address that is
showing the same number *at the same moment*.

This window is that loop, made interactive:

  1. put some addresses on the watch list,
  2. leave it refreshing while you look at eServer,
  3. when a row's number matches the screen, name it and press Confirm.

The crucial difference from the Modbus Scanner is that **every row is decoded
every way at once** — raw word, signed, /10, /100, int32 LE/BE, float LE/BE.
eServer's own address table stores the TI channels as ``Unsigned / Word /
Read Count 1``, i.e. 29.5 °C lives in memory as the integer 295, which a
float-only view cannot see at all. Showing them side by side means the operator
never has to guess the storage format before looking.

The Device selector matters just as much. Delta publishes each AS-series device
file at its own address base, and the X file (inputs) answers **only** to
function code 04 — an FC03 sweep reports it as absent rather than as an error,
so anything the program leaves in X is invisible until the code is right.

The Function selector carries Modbus Poll's own four entries (01/02/03/04) so
the two screens can be compared line for line, and "PLC addresses (Base 1)"
matches its address convention — with that ticked, Modbus Poll calls D3000
"3001" while the wire still carries 3000.

**Read-only and deliberately gentle.** Only *read* function codes are ever
issued — 01, 02, 03, 04 — through a single reused connection, at 1 s by
default. The PLC allows just 8 simultaneous connections in total (shared with
eServer and the HMI), which is why this window opens exactly one and keeps it.
Nothing here can write to the PLC or disturb the running control system.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from psa_analyzer.core.modbus_client import AS_DEVICE_RANGES, ModbusReader
from psa_analyzer.core.plc_map import default_sensors
from psa_analyzer.ui.theme import Theme, get_stylesheet

# Where confirmed identifications are kept between sessions. Home rather than
# next to the .exe, which on the eServer PC may sit in a read-only folder.
MAP_FILE = Path.home() / ".psa_analyzer" / "address_map.json"

# A screen value rounded to 2-3 decimals still has to match a register holding
# slightly more precision, so matching is relative.
FIND_RTOL = 0.02

# Guard rail: the watch list spans one block read. A careless "add range" of
# thousands would turn a 1 s refresh into a burst of requests at the PLC.
MAX_WATCH = 400


# Every way one register (or a register pair) can be read. Shown side by side
# so the operator never has to guess the storage format before looking.
ALL_FORMATS = ["word", "signed", "/10", "/100",
               "int32 LE", "int32 BE", "float LE", "float BE"]

# Bit function codes return 0/1 per address; the word views are meaningless.
BIT_FORMATS = ["bit"]

# Formats that are whole numbers and must never be shown in scientific form.
_INTEGRAL = {"word", "signed", "int32 LE", "int32 BE", "bit"}


def _signed(word: int) -> int:
    return word - 65536 if word > 32767 else word


def _signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value > 0x7FFFFFFF else value


def _decode_float(w0: int, w1: int, order: str) -> float:
    import struct
    if order == "little":
        combined = ((w1 & 0xFFFF) << 16) | (w0 & 0xFFFF)
    else:
        combined = ((w0 & 0xFFFF) << 16) | (w1 & 0xFFFF)
    return struct.unpack(">f", struct.pack(">I", combined))[0]


def decodings(addr: int, words: dict[int, int]) -> dict[str, float | None]:
    """Every plausible reading of ``addr``, given the words we have.

    Returns ``{format: value}`` with ``None`` where the words needed for that
    format were not readable. The float entries consume ``addr`` and ``addr+1``.
    """
    out: dict[str, float | None] = {k: None for k in ALL_FORMATS}
    w = words.get(addr)
    if w is not None:
        s = _signed(w)
        out["word"] = float(w)
        out["signed"] = float(s)
        out["/10"] = s / 10.0
        out["/100"] = s / 100.0
    w1 = words.get(addr + 1)
    if w is not None and w1 is not None:
        # ISPSoft's monitor table shows a 32-bit integer view beside the float
        # one, so a tag stored as a long is matched the same way a float is.
        out["int32 LE"] = float(_signed32((w1 << 16) | w))
        out["int32 BE"] = float(_signed32((w << 16) | w1))
        out["float LE"] = _decode_float(w, w1, "little")
        out["float BE"] = _decode_float(w, w1, "big")
    return out


def bit_decodings(addr: int, bits: dict[int, int]) -> dict[str, float | None]:
    """A coil / discrete input is one bit — there is nothing else to try."""
    b = bits.get(addr)
    return {"bit": None if b is None else float(b)}


def _fmt(v: float | None, integral: bool = False) -> str:
    """Render one decoded value.

    ``integral`` is set for the word / signed / int32 columns: those are whole
    numbers by definition, and printing 309347343 as "3.093e+08" makes the one
    thing this table exists for — matching a number against another screen —
    impossible.
    """
    if v is None:
        return "—"
    if v != v:                       # NaN
        return "nan"
    if integral:
        return f"{v:.0f}"
    if abs(v) >= 1e7:
        return f"{v:.3e}"
    if float(v).is_integer() and abs(v) < 1e7:
        return f"{v:.0f}"
    return f"{v:.4g}"


class AddressExplorerWindow(QWidget):
    """Live watch list over raw Modbus addresses, decoded every which way."""

    _WORD_COLS = ["Address", "What you think it is", "moving?",
                  "word", "signed", "÷10", "÷100",
                  "int32 LE", "int32 BE", "float LE", "float BE"]
    _BIT_COLS = ["Address", "What you think it is", "moving?", "bit"]

    # Modbus Poll's four read functions, in its own wording so the two screens
    # can be compared line for line.
    _FUNCTIONS = [
        ("03 Read Holding Registers (4x)", 3),
        ("04 Read Input Registers (3x)", 4),
        ("01 Read Coils (0x)", 1),
        ("02 Read Discrete Inputs (1x)", 2),
    ]

    @property
    def _COLS(self) -> list[str]:
        return self._BIT_COLS if self._fc in (1, 2) else self._WORD_COLS

    @property
    def _FORMATS(self) -> list[str]:
        return BIT_FORMATS if self._fc in (1, 2) else ALL_FORMATS

    def _decode(self, addr: int) -> dict[str, float | None]:
        if self._fc in (1, 2):
            return bit_decodings(addr, self._words)
        return decodings(addr, self._words)

    def __init__(self, host: str = "192.168.1.5", port: int = 502,
                 unit: int = 1, theme: Theme = Theme.LIGHT,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Address Explorer — which register is which?")
        self.resize(1080, 780)
        self.setWindowFlag(Qt.Window, True)

        self._watch: list[int] = []
        self._confirmed: list[dict] = []
        self._words: dict[int, int] = {}
        # Notes are keyed by address, never by row: hiding zero rows or a
        # search reorders the table, and a note that slid onto a neighbouring
        # address would be worse than no note at all.
        self._notes: dict[int, str] = {}
        self._reader: ModbusReader | None = None
        self._highlight: set[int] = set()
        self._fc = 3                      # 3 = holding regs, 4 = X input regs
        # Per-address movement since the window opened. Comparing two scans by
        # hand is what separated the four live flows from ten setpoints in the
        # 0..4000 sweeps; doing it continuously turns that into one glance.
        # BPR-01 is the most dynamic signal on the rig, so "what is moving"
        # finds it faster than any value search can.
        self._motion: dict[int, dict] = {}
        # What the map currently claims, so each row shows the claim next to
        # the live number that either backs it up or refutes it.
        try:
            self._expected = list(default_sensors())
        except Exception:                        # noqa: BLE001
            self._expected = []
        # Modbus Poll shows "PLC Addresses (Base 1)" ticked, so the number on
        # the technician's screen is one HIGHER than the protocol address we
        # put on the wire: his 3001 is our 3000. Typing his number straight in
        # would silently read the neighbouring register.
        self._base1 = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)

        self._build_ui(host, port, unit)
        self._load_map()
        self.apply_theme(theme)
        self.add_range(100, 130)          # the range every hunt starts from

    # -- construction -------------------------------------------------------
    def _build_ui(self, host: str, port: int, unit: int) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "Watch addresses live while you look at eServer. When a number "
            "here matches the number on that screen, you have found the tag — "
            "name it and press Confirm. Read-only: this never writes to the PLC.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#6b7280;")
        root.addWidget(intro)

        # -- connection + watch controls ------------------------------------
        conn = QGroupBox("Connection")
        c = QHBoxLayout(conn)
        c.addWidget(QLabel("IP"))
        self.ip_input = QLineEdit(host)
        self.ip_input.setMaximumWidth(130)
        c.addWidget(self.ip_input)
        c.addWidget(QLabel("Port"))
        self.port_input = QSpinBox(); self.port_input.setRange(1, 65535)
        self.port_input.setValue(int(port)); self.port_input.setMaximumWidth(80)
        c.addWidget(self.port_input)
        c.addWidget(QLabel("Unit"))
        self.unit_input = QSpinBox(); self.unit_input.setRange(0, 255)
        self.unit_input.setValue(int(unit)); self.unit_input.setMaximumWidth(60)
        c.addWidget(self.unit_input)
        c.addWidget(QLabel("Refresh"))
        self.interval = QComboBox()
        self.interval.addItems(["1 s", "2 s", "5 s"])
        self.interval.setMaximumWidth(70)
        self.interval.currentIndexChanged.connect(self._retime)
        c.addWidget(self.interval)
        self.chk_live = QCheckBox("Live")
        self.chk_live.setChecked(True)
        self.chk_live.toggled.connect(self._toggle_live)
        c.addWidget(self.chk_live)
        self.btn_once = QPushButton("Read once")
        self.btn_once.clicked.connect(self._poll)
        c.addWidget(self.btn_once)
        c.addStretch(1)
        root.addWidget(conn)

        watch = QGroupBox("Watch list")
        w = QHBoxLayout(watch)
        # Delta publishes each device file (D, X, Y, SR, ...) at its own address
        # base and X is reachable only through FC04. Picking the device sets
        # both, so a hunt can reach ranges a hand-typed number never would.
        w.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self.device.setMinimumWidth(170)
        for name in AS_DEVICE_RANGES:
            self.device.addItem(name, name)
        self.device.setToolTip(
            "AS-series device file. D holds the plant tags; X is the analog / "
            "digital input file and needs function code 04.")
        self.device.currentIndexChanged.connect(self._on_device_changed)
        w.addWidget(self.device)
        w.addWidget(QLabel("Function"))
        self.function = QComboBox()
        self.function.setMinimumWidth(210)
        for label, fc in self._FUNCTIONS:
            self.function.addItem(label, fc)
        self.function.setToolTip(
            "The Modbus read function, same list as Modbus Poll. Picking a "
            "device above sets this for you; override it here to try a file "
            "the other way round.")
        self.function.currentIndexChanged.connect(self._on_function_changed)
        w.addWidget(self.function)
        w.addWidget(QLabel("From"))
        self.from_addr = QSpinBox(); self.from_addr.setRange(0, 65535)
        self.from_addr.setValue(100); self.from_addr.setMaximumWidth(90)
        w.addWidget(self.from_addr)
        w.addWidget(QLabel("to"))
        self.to_addr = QSpinBox(); self.to_addr.setRange(0, 65535)
        self.to_addr.setValue(130); self.to_addr.setMaximumWidth(90)
        w.addWidget(self.to_addr)
        b_add = QPushButton("+ Add")
        b_add.setToolTip("Put this address range on the watch list")
        b_add.clicked.connect(
            lambda: self.add_range(self.from_addr.value(), self.to_addr.value()))
        w.addWidget(b_add)
        # A float occupies two words, so listing every address shows each tag
        # twice: once as itself and once as the second half of the pair, whose
        # own word is 0 and whose float decoding is garbage. Those look like
        # extra moving registers and are pure noise while hunting.
        self.chk_pairs = QCheckBox("Float pairs only")
        self.chk_pairs.setChecked(True)
        self.chk_pairs.setToolTip(
            "List every second address, so each row is the START of a float "
            "pair — which is how the plant tags are laid out (D100, D102, "
            "D104 …). Untick to see every single word.")
        self.chk_pairs.toggled.connect(self._on_pairs_changed)
        w.addWidget(self.chk_pairs)
        # "if it doesn't match, keep moving" — the whole window slides so the
        # operator can walk the address space without retyping bounds.
        for text, step in (("◀ −10", -10), ("◀ −1", -1),
                           ("+1 ▶", 1), ("+10 ▶", 10)):
            b = QPushButton(text)
            b.setMaximumWidth(64)
            b.setToolTip("Slide the whole watch window by this many addresses")
            b.clicked.connect(lambda _=False, s=step: self._slide(s))
            w.addWidget(b)
        b_drop = QPushButton("Remove selected")
        b_drop.clicked.connect(self._remove_selected)
        w.addWidget(b_drop)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear_watch)
        w.addWidget(b_clear)
        self.chk_base1 = QCheckBox("PLC addresses (Base 1)")
        self.chk_base1.setToolTip(
            "Tick this to enter and display addresses the way Modbus Poll "
            "does with 'PLC Addresses (Base 1)' on: D3000 shows as 3001. "
            "Off = the protocol address actually sent on the wire.")
        self.chk_base1.toggled.connect(self._on_base_changed)
        w.addWidget(self.chk_base1)
        self.chk_nonzero = QCheckBox("Hide all-zero rows")
        self.chk_nonzero.toggled.connect(self._refresh_table)
        w.addWidget(self.chk_nonzero)
        self.chk_moving = QCheckBox("Only what is moving")
        self.chk_moving.setToolTip(
            "Hide every address whose value has not changed since this window "
            "opened. A running rig moves very few registers, and BPR-01 moves "
            "the most of all — this is the quickest way to find it.")
        self.chk_moving.toggled.connect(self._refresh_table)
        w.addWidget(self.chk_moving)
        b_reset = QPushButton("Reset movement")
        b_reset.clicked.connect(self._reset_motion)
        w.addWidget(b_reset)
        w.addStretch(1)
        root.addWidget(watch)

        # -- search by the value on the eServer screen -----------------------
        find = QGroupBox("Find the value you see on eServer")
        f = QHBoxLayout(find)
        f.addWidget(QLabel("Value"))
        self.find_value = QLineEdit()
        self.find_value.setPlaceholderText("e.g. 29.5")
        self.find_value.setMaximumWidth(110)
        self.find_value.returnPressed.connect(self._search)
        f.addWidget(self.find_value)
        f.addWidget(QLabel("in"))
        self.find_from = QSpinBox(); self.find_from.setRange(0, 65535)
        self.find_from.setValue(0); self.find_from.setMaximumWidth(90)
        f.addWidget(self.find_from)
        f.addWidget(QLabel("..."))
        self.find_to = QSpinBox(); self.find_to.setRange(1, 65535)
        self.find_to.setValue(1200); self.find_to.setMaximumWidth(90)
        f.addWidget(self.find_to)
        b_find = QPushButton("Search")
        b_find.setToolTip("Sweep that range once and list every address "
                          "holding this value, in any format")
        b_find.clicked.connect(self._search)
        f.addWidget(b_find)
        self.find_result = QLabel("")
        self.find_result.setWordWrap(True)
        self.find_result.setStyleSheet("color:#6b7280;")
        f.addWidget(self.find_result, stretch=1)
        root.addWidget(find)

        # -- the two tables --------------------------------------------------
        split = QSplitter(Qt.Vertical)

        live_box = QWidget()
        lv = QVBoxLayout(live_box)
        lv.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(self._COLS))
        self.table.setHorizontalHeaderLabels(self._COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.itemChanged.connect(self._on_note_edited)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Interactive)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(1, 170)
        self.table.setToolTip(
            "The second column is pre-filled from plc_map with a trailing '?' "
            "— that is what the map CLAIMS lives at this address. Check the "
            "live number against eServer, then confirm or correct it.")
        lv.addWidget(self.table)

        # -- confirm panel ---------------------------------------------------
        conf = QGroupBox("Found it? Name it and confirm")
        cf = QHBoxLayout(conf)
        self.sel_label = QLabel("Select a row above")
        self.sel_label.setMinimumWidth(120)
        cf.addWidget(self.sel_label)
        cf.addWidget(QLabel("Tag"))
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("e.g. MFC-07 (AD GAS)")
        self.tag_input.setMaximumWidth(200)
        cf.addWidget(self.tag_input)
        cf.addWidget(QLabel("Format"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.setMinimumWidth(190)
        self.fmt_combo.setToolTip(
            "Pick the reading that matches eServer — that is the tag's format")
        cf.addWidget(self.fmt_combo)
        self.btn_confirm = QPushButton("✓ Matches eServer — confirm")
        self.btn_confirm.setObjectName("PrimaryButton")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._confirm)
        cf.addWidget(self.btn_confirm)
        cf.addStretch(1)
        lv.addWidget(conf)
        split.addWidget(live_box)

        conf_box = QWidget()
        cv = QVBoxLayout(conf_box)
        cv.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("Confirmed addresses"))
        head.addStretch(1)
        b_del = QPushButton("Remove selected")
        b_del.clicked.connect(self._remove_confirmed)
        head.addWidget(b_del)
        b_exp = QPushButton("Export...")
        b_exp.setToolTip("Save as JSON plus a ready-to-paste plc_map.py snippet")
        b_exp.clicked.connect(self._export)
        head.addWidget(b_exp)
        cv.addLayout(head)
        self.conf_table = QTableWidget(0, 6)
        self.conf_table.setHorizontalHeaderLabels(
            ["Address", "Tag", "Function", "Format",
             "Value when confirmed", "When"])
        self.conf_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.conf_table.verticalHeader().setVisible(False)
        self.conf_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        cv.addWidget(self.conf_table)
        split.addWidget(conf_box)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, stretch=1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#6b7280;")
        root.addWidget(self.status)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        root.addWidget(bb)

        self._retime()

    def apply_theme(self, theme: Theme) -> None:
        self.setStyleSheet(get_stylesheet(theme))

    # -- device files -------------------------------------------------------
    def _set_fc(self, fc: int) -> None:
        """Switch function code, rebuilding the table if the shape changed."""
        was_bits = self._fc in (1, 2)
        self._fc = int(fc)
        i = self.function.findData(self._fc)
        if i >= 0 and self.function.currentIndex() != i:
            self.function.blockSignals(True)
            self.function.setCurrentIndex(i)
            self.function.blockSignals(False)
        if was_bits != (self._fc in (1, 2)):
            self.table.setColumnCount(len(self._COLS))
            self.table.setHorizontalHeaderLabels(self._COLS)

    def _on_function_changed(self) -> None:
        fc = self.function.currentData()
        if fc is None:
            return
        self._set_fc(fc)
        self._words = {}
        self._refresh_table()
        self._poll()

    def _on_base_changed(self) -> None:
        """Only the display convention changes — the wire address does not."""
        self._base1 = self.chk_base1.isChecked()
        self._refresh_table()
        self._say("Addresses shown as "
                  + ("PLC / Base 1, matching Modbus Poll (D3000 shows as 3001)."
                     if self._base1 else
                     "protocol addresses, as sent on the wire (D3000 = 3000)."))

    def _shown(self, addr: int) -> int:
        """The address as the operator's other screen numbers it."""
        return addr + 1 if self._base1 else addr

    def _on_device_changed(self) -> None:
        """Point the spin boxes and the function code at the chosen file."""
        name = self.device.currentData()
        lo, hi, fc = AS_DEVICE_RANGES[name]
        self._set_fc(fc)
        span = min(30, hi - lo)
        self.from_addr.setValue(lo)
        self.to_addr.setValue(lo + span)
        self.find_from.setValue(lo)
        self.find_to.setValue(hi)
        self._clear_watch()
        self.add_range(lo, lo + span)
        self._say(f"{name}: addresses {lo}..{hi}, read with FC{fc:02d}."
                  + ("  X is an input file — FC03 cannot see it at all."
                     if fc == 4 else ""))

    def _device_of(self, addr: int) -> str:
        """Name ``addr`` the way the PLC programmer would, e.g. 49152 -> SR0."""
        for name, (lo, hi, _fc) in AS_DEVICE_RANGES.items():
            if lo <= addr <= hi:
                return f"{name.split()[0]}{addr - lo}"
        return ""

    # -- watch list ---------------------------------------------------------
    def _on_pairs_changed(self) -> None:
        """Re-add the current span at the new step."""
        if not self._watch:
            return
        lo, hi = min(self._watch), max(self._watch)
        self._watch.clear()
        self.add_range(lo, hi)
        self._say("Listing " + ("every second address — each row is the start "
                                "of a float pair."
                                if self.chk_pairs.isChecked()
                                else "every single word."))

    def add_range(self, lo: int, hi: int) -> None:
        lo, hi = int(min(lo, hi)), int(max(lo, hi))
        step = 2 if self.chk_pairs.isChecked() else 1
        added = 0
        for a in range(lo, hi + 1, step):
            if a not in self._watch:
                if len(self._watch) >= MAX_WATCH:
                    self._say(f"Watch list is full at {MAX_WATCH} addresses — "
                              f"remove some before adding more.")
                    break
                self._watch.append(a)
                added += 1
        self._watch.sort()
        self._refresh_table()
        if added:
            self._poll()

    def _slide(self, step: int) -> None:
        """Move the whole watch window — 'not this one, try the next'."""
        if not self._watch:
            return
        lo, hi = min(self._watch), max(self._watch)
        span = hi - lo
        lo = max(0, lo + step)
        stride = 2 if self.chk_pairs.isChecked() else 1
        self._watch = list(range(lo, lo + span + 1, stride))
        self.from_addr.setValue(lo)
        self.to_addr.setValue(lo + span)
        self._refresh_table()
        self._poll()

    def _remove_selected(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        keep = [a for i, a in enumerate(self._visible()) if i not in rows]
        gone = set(self._visible()) - set(keep)
        self._watch = [a for a in self._watch if a not in gone]
        self._refresh_table()

    def _clear_watch(self) -> None:
        self._watch.clear()
        self._highlight.clear()
        self._refresh_table()

    def _visible(self) -> list[int]:
        """Addresses currently shown, honouring both row filters."""
        out = []
        for a in self._watch:
            if a in self._highlight:
                out.append(a)
                continue
            if self.chk_nonzero.isChecked():
                w0, w1 = self._words.get(a), self._words.get(a + 1)
                if not (w0 or w1):
                    continue
            if self.chk_moving.isChecked() and not self._motion_of(a)[0]:
                continue
            out.append(a)
        return out

    # -- polling ------------------------------------------------------------
    def _retime(self) -> None:
        secs = int(self.interval.currentText().split()[0])
        self._timer.setInterval(secs * 1000)
        if self.chk_live.isChecked():
            self._timer.start()

    def _toggle_live(self, on: bool) -> None:
        if on:
            self._poll()
            self._timer.start()
        else:
            self._timer.stop()

    def _connect(self) -> ModbusReader | None:
        """One connection, reused. Reopened only if the endpoint changed."""
        host = self.ip_input.text().strip()
        port, unit = self.port_input.value(), self.unit_input.value()
        ep = (host, port, unit)
        if self._reader is not None and getattr(self._reader, "_ep", None) == ep:
            return self._reader
        self._close_reader()
        reader = ModbusReader(host, port=port, device_id=unit, timeout=1.0)
        if not reader.connect():
            reader.close()
            # Connecting blocks for the full timeout, so retrying every second
            # against a dead endpoint would freeze the window. Stop refreshing
            # and let the operator fix the address and press Live again.
            note = ""
            if self.chk_live.isChecked():
                self.chk_live.setChecked(False)
                note = " Live refresh paused — fix the address, then tick Live."
            self._say(f"Cannot connect to {host}:{port}.{note}")
            return None
        reader._ep = ep                          # noqa: SLF001 - our own tag
        self._reader = reader
        return reader

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _poll(self) -> None:
        if not self._watch:
            self._refresh_table()
            return
        reader = self._connect()
        if reader is None:
            return
        lo, hi = min(self._watch), max(self._watch) + 1
        try:
            if hi - lo + 1 <= 500:
                self._words = reader.read_range(lo, hi, fc=self._fc)
            else:
                words: dict[int, int] = {}
                for a in self._watch:
                    words.update(reader.read_range(a, a + 1, fc=self._fc))
                self._words = words
        except Exception as exc:                 # noqa: BLE001
            self._close_reader()
            self._say(f"Read error: {exc}")
            return
        self._track_motion()
        got = len(self._words)
        self._refresh_table()
        span = self._device_of(lo)
        where = f"{lo}..{hi}" + (f"  ({span} onwards)" if span else "")
        self._say(f"{datetime.now():%H:%M:%S} — FC{self._fc:02d}: {got} of "
                  f"{hi - lo + 1} words readable in {where}.")

    def _track_motion(self) -> None:
        """Remember how much each raw word has moved since we started."""
        for a, w in self._words.items():
            m = self._motion.get(a)
            if m is None:
                self._motion[a] = {"lo": w, "hi": w, "last": w, "changes": 0}
                continue
            if w != m["last"]:
                m["changes"] += 1
                m["last"] = w
            m["lo"] = min(m["lo"], w)
            m["hi"] = max(m["hi"], w)

    def _motion_of(self, addr: int) -> tuple[int, str]:
        """(change count, label) for a row — a float row watches both words."""
        span = 1 if self._fc in (1, 2) else 2
        changes = 0
        for a in range(addr, addr + span):
            m = self._motion.get(a)
            if m:
                changes += m["changes"]
        if not changes:
            return 0, "still"
        return changes, f"MOVING  x{changes}"

    def _reset_motion(self) -> None:
        self._motion.clear()
        self._refresh_table()
        self._say("Movement counters cleared — watching again from now.")

    # -- table --------------------------------------------------------------
    def _refresh_table(self) -> None:
        addrs = self._visible()
        # Keep the selection across refreshes — losing it every second would
        # make the confirm panel unusable.
        sel_addr = self._selected_address()

        self.table.blockSignals(True)
        self.table.setRowCount(len(addrs))
        for r, a in enumerate(addrs):
            d = self._decode(a)
            dev = self._device_of(a)
            shown = self._shown(a)
            self._set(r, 0, f"{shown}  {dev}" if dev else str(shown),
                      center=True)
            n_moved, motion = self._motion_of(a)
            self._set(r, 2, motion, center=True)
            note = (self._notes.get(a) or self._confirmed_tag(a)
                    or self._expected_tag(a))
            n = QTableWidgetItem(note)
            n.setFlags(n.flags() | Qt.ItemIsEditable)
            self.table.setItem(r, 1, n)
            for c, key in enumerate(self._FORMATS, start=3):
                self._set(r, c, _fmt(d[key], key in _INTEGRAL), center=True)
            if n_moved:
                cell = self.table.item(r, 2)
                if cell is not None:
                    cell.setForeground(QBrush(QColor("#b45309")))
            if a in self._highlight:
                for c in range(len(self._COLS)):
                    cell = self.table.item(r, c)
                    if cell is not None:
                        cell.setBackground(QBrush(QColor("#fef3c7")))
        self.table.blockSignals(False)

        if sel_addr is not None and sel_addr in addrs:
            self.table.selectRow(addrs.index(sel_addr))
        self._on_row_selected()

    def _set(self, r: int, c: int, text: str, center: bool = False) -> None:
        item = QTableWidgetItem(text)
        if center:
            item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, c, item)

    def _on_note_edited(self, item: QTableWidgetItem) -> None:
        """Remember an edited note against its address, not its row."""
        if item.column() != 1:
            return
        addr_item = self.table.item(item.row(), 0)
        if addr_item is None:
            return
        shown = int(addr_item.text().split()[0])
        addr = shown - 1 if self._base1 else shown
        self._notes[addr] = item.text().strip()

    def _selected_address(self) -> int | None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return None
        item = self.table.item(min(rows), 0)
        if item is None:
            return None
        shown = int(item.text().split()[0])
        return shown - 1 if self._base1 else shown

    def _on_row_selected(self) -> None:
        addr = self._selected_address()
        if addr is None:
            self.sel_label.setText("Select a row above")
            self.fmt_combo.clear()
            self.btn_confirm.setEnabled(False)
            return
        dev = self._device_of(addr)
        self.sel_label.setText(f"Address {self._shown(addr)}"
                               + (f"  ({dev})" if dev else ""))
        d = self._decode(addr)
        keep = self.fmt_combo.currentData()
        self.fmt_combo.blockSignals(True)
        self.fmt_combo.clear()
        for key in self._FORMATS:
            self.fmt_combo.addItem(
                f"{key}  →  {_fmt(d[key], key in _INTEGRAL)}", key)
        if keep is not None:
            i = self.fmt_combo.findData(keep)
            if i >= 0:
                self.fmt_combo.setCurrentIndex(i)
        self.fmt_combo.blockSignals(False)
        self.btn_confirm.setEnabled(True)

    # -- search -------------------------------------------------------------
    def _search(self) -> None:
        text = self.find_value.text().strip()
        try:
            want = float(text)
        except ValueError:
            self.find_result.setText("Type a number, e.g. 29.5")
            return
        reader = self._connect()
        if reader is None:
            return
        lo, hi = self.find_from.value(), self.find_to.value()
        if hi <= lo:
            self.find_result.setText("The end address must be above the start.")
            return
        self.find_result.setText(f"Sweeping {lo}..{hi} ...")
        self.find_result.repaint()
        try:
            words = reader.read_range(lo, hi, fc=self._fc)
        except Exception as exc:                 # noqa: BLE001
            self._close_reader()
            self.find_result.setText(f"Read error: {exc}")
            return

        hits: list[tuple[int, str, float]] = []
        for a in sorted(words):
            decode = bit_decodings if self._fc in (1, 2) else decodings
            for key, v in decode(a, words).items():
                if v is None or v != v or v == 0:
                    continue
                tol = max(abs(want) * FIND_RTOL, 5e-4)
                if abs(v - want) <= tol:
                    hits.append((a, key, v))
                    break

        # A float at [a, a+1] read little-endian and the float at [a+1, a+2]
        # read big-endian are the same four bytes seen twice whenever the
        # outer words are zero. Reporting both turns a clean single answer
        # into a fake ambiguity, so drop the shifted twin.
        le = {a for a, k, _ in hits if k == "float LE"}
        be = {a for a, k, _ in hits if k == "float BE"}
        vals = {(a, k): v for a, k, v in hits}
        alias = {(a, "float BE") for a in be
                 if a - 1 in le and vals[(a - 1, "float LE")] == vals[(a, "float BE")]}
        alias |= {(a, "float LE") for a in le
                  if a - 1 in be and vals[(a - 1, "float BE")] == vals[(a, "float LE")]}
        hits = [h for h in hits if (h[0], h[1]) not in alias]

        if not hits:
            self.find_result.setText(
                f"{want:g} is not in {lo}..{hi} in any format. Either the tag "
                f"is outside this range, or the PLC does not publish it.")
            return

        self._highlight = {a for a, _, _ in hits}
        for a, _, _ in hits:
            if a not in self._watch and len(self._watch) < MAX_WATCH:
                self._watch.append(a)
        self._watch.sort()
        self._poll()
        listing = ", ".join(
            f"{self._shown(a)}"
            f"{' ' + self._device_of(a) if self._device_of(a) else ''} ({k})"
            for a, k, _ in hits[:12])
        more = f" ... +{len(hits) - 12} more" if len(hits) > 12 else ""
        self.find_result.setText(
            f"{len(hits)} match: {listing}{more}. "
            + ("Exactly one — that is your tag."
               if len(hits) == 1
               else "Several: note a second eServer value now and search again; "
                    "the address answering both is the one."))

    # -- confirm / persist --------------------------------------------------
    def _confirmed_tag(self, addr: int) -> str:
        for c in self._confirmed:
            if c["address"] == addr:
                return c["tag"]
        return ""

    def _expected_tag(self, addr: int) -> str:
        """What plc_map claims lives here — the thing being put to the test.

        Pre-filling this saves flipping to the Machine & Sensor Status window
        to remember which tag address 104 is supposed to be. It is a claim, not
        a confirmation: the Confirmed table below is the record of what has
        actually been checked against eServer.
        """
        for sen in self._expected:
            if sen.address == addr and sen.host is None and sen.bus == "tcp":
                return f"{sen.tag}?"
        return ""

    def _confirm(self) -> None:
        addr = self._selected_address()
        if addr is None:
            return
        tag = self.tag_input.text().strip() or self._notes.get(addr, "").strip()
        if not tag:
            # The "?" suffix marks a suggestion from plc_map. Accept it, but
            # strip the mark: what gets confirmed is a claim the operator has
            # just checked, not a guess.
            tag = self._expected_tag(addr).rstrip("?")
        if not tag:
            QMessageBox.information(
                self, "Name it first",
                "Type the tag name as eServer shows it, e.g. MFC-07 (AD GAS).")
            return
        fmt = self.fmt_combo.currentData() or self._FORMATS[0]
        value = self._decode(addr).get(fmt)
        self._confirmed = [c for c in self._confirmed if c["address"] != addr]
        self._confirmed.append({
            "address": addr, "tag": tag, "format": fmt, "fc": self._fc,
            "value": None if value is None else round(float(value), 6),
            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._confirmed.sort(key=lambda c: c["address"])
        self._save_map()
        self._refresh_conf_table()
        self.tag_input.clear()
        self._say(f"Confirmed: {addr} = {tag} ({fmt}). Saved to {MAP_FILE}")

    def _remove_confirmed(self) -> None:
        rows = sorted({i.row() for i in self.conf_table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._confirmed):
                self._confirmed.pop(r)
        self._save_map()
        self._refresh_conf_table()

    def _refresh_conf_table(self) -> None:
        self.conf_table.setRowCount(len(self._confirmed))
        for r, c in enumerate(self._confirmed):
            for col, text in enumerate((str(c["address"]), c["tag"],
                                        f"FC{int(c.get('fc', 3)):02d}",
                                        c["format"], _fmt(c["value"]),
                                        c["when"])):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.conf_table.setItem(r, col, item)
        self._refresh_table()

    def _load_map(self) -> None:
        try:
            if MAP_FILE.exists():
                data = json.loads(MAP_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._confirmed = [d for d in data if "address" in d]
        except Exception:                        # noqa: BLE001
            self._confirmed = []
        self._refresh_conf_table()

    def _save_map(self) -> None:
        try:
            MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
            MAP_FILE.write_text(
                json.dumps(self._confirmed, indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception as exc:                 # noqa: BLE001
            self._say(f"Could not save {MAP_FILE}: {exc}")

    def _export(self) -> None:
        if not self._confirmed:
            QMessageBox.information(self, "Nothing to export",
                                    "Confirm at least one address first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export confirmed addresses", "address_map.json",
            "JSON (*.json)")
        if not path:
            return
        p = Path(path)
        try:
            p.write_text(json.dumps(self._confirmed, indent=2,
                                    ensure_ascii=False), encoding="utf-8")
            snippet = p.with_suffix(".plc_map.txt")
            snippet.write_text(self._snippet(), encoding="utf-8")
        except Exception as exc:                 # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._say(f"Exported {p.name} and {snippet.name}")

    def _snippet(self) -> str:
        """A paste-ready plc_map.py block for the confirmed addresses."""
        kind = {"word": "uint16", "signed": "int16", "/10": "uint16",
                "/100": "uint16", "int32 LE": "int32", "int32 BE": "int32",
                "float LE": "float", "float BE": "float", "bit": "bit"}
        lines = ["# Confirmed against the eServer screen with Address Explorer",
                 f"# {datetime.now():%Y-%m-%d %H:%M}", ""]
        for c in self._confirmed:
            note = ""
            if c["format"] in ("/10", "/100"):
                note = (f"  # value = word {c['format']}"
                        f" — SensorDef has no scale field yet")
            elif c["format"] == "float BE":
                note = "  # big word order — differs from the PLC default"
            if c.get("fc", 3) != 3:
                note += f"  # read with FC{int(c['fc']):02d}, not FC03"
            lines.append(
                f'SensorDef("{c["tag"]}", {c["address"]}, '
                f'"{kind.get(c["format"], "float")}"),{note}')
        return "\n".join(lines) + "\n"

    # -- misc ---------------------------------------------------------------
    def _say(self, text: str) -> None:
        self.status.setText(text)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._close_reader()
        super().closeEvent(event)
