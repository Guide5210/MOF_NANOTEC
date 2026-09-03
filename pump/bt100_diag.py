"""
BT100-1L RS485 Diagnostic Tool — God Mode
Run with:  py -3.11 "C:\\Users\\srnan\\Downloads\\bt100_diag.py"
"""

import sys
import time
import re
from collections import Counter
from threading import Lock

import serial
import serial.tools.list_ports

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QGridLayout, QHBoxLayout, QVBoxLayout,
    QGroupBox, QFrame, QSplitter, QFileDialog
)


# ============================================================
# COM PORT HELPERS
# ============================================================
def normalize_windows_com(port: str) -> str:
    port = port.strip().upper()
    if re.fullmatch(r"COM\d+", port) and int(port[3:]) >= 10:
        return r"\\.\{}".format(port)
    return port


def extract_display_com(port_info) -> str:
    text = f"{port_info.device} {port_info.description} {port_info.hwid}"
    matches = re.findall(r"COM\d+", text.upper())
    return matches[-1] if matches else port_info.device


def list_serial_ports():
    ports = list(serial.tools.list_ports.comports())
    valid = []
    for p in ports:
        desc = (p.description or "").upper()
        hwid = (p.hwid or "").upper()
        if "BLUETOOTH" in desc or "BTHENUM" in hwid:
            continue
        valid.append(p)
    return valid


def parse_hex_string(s: str) -> bytes:
    cleaned = re.sub(r"0x|0X|,|;|:|\s+|-", "", s.strip())
    if not cleaned:
        raise ValueError("Hex string is empty.")
    if len(cleaned) % 2 != 0:
        raise ValueError("Hex string must have an even number of nibbles.")
    if not re.fullmatch(r"[0-9a-fA-F]+", cleaned):
        raise ValueError("Hex string contains non-hex characters.")
    return bytes.fromhex(cleaned)


# ============================================================
# RS485 DIAGNOSTIC EXCEPTION TRANSLATOR
# ============================================================
def diagnose_serial_exception(exc: Exception):
    msg = str(exc); low = msg.lower()
    if "access is denied" in low or "permissionerror" in low or "errno 13" in low:
        return ("Port in use / Access denied",
            "This COM port is already opened by another program.\n"
            "• Close any other terminal (PuTTY, Arduino IDE, RealTerm, prior instance of this app).\n"
            "• Unplug and reinsert the USB-RS485 adapter.\n"
            "• Verify in Device Manager that the COM number is what you selected.")
    if "could not open port" in low or "filenotfound" in low or "errno 2" in low or "the system cannot find" in low:
        return ("Port not found",
            "The selected COM port does not exist anymore.\n"
            "• Click 'Refresh Ports' — the adapter may have changed COM number.\n"
            "• Check that the USB-RS485 dongle is plugged in.\n"
            "• Reinstall the CH340/FTDI/CP210x driver if the port never appears.")
    if "semaphore" in low or "errno 121" in low or "timed out" in low or "timeout" in low:
        return ("Timeout / No response",
            "The pump did not reply within the timeout window.\n"
            "• Swap RS485 A and B wires — polarity is the #1 cause of silence.\n"
            "• Confirm baud rate is 1200 bps, 8E1 (factory default for BT100-1L).\n"
            "• Verify pump address matches the one set in the pump's menu (default 1).\n"
            "• Check 24 V power supply to the pump; the panel must be lit.\n"
            "• Add a 120 ohm termination resistor across A-B if cable is long.")
    if "input/output error" in low or "clearcommerror" in low or "errno 5" in low:
        return ("Cable / driver I/O error",
            "The OS reported a low-level I/O failure on the adapter.\n"
            "• Unplug the USB adapter and plug it back in.\n"
            "• Try a different USB port (avoid unpowered hubs).\n"
            "• Update or reinstall the USB-Serial driver.")
    return ("Serial error",
        f"Unrecognised serial error:\n  {msg}\n"
        "• Check cabling, power, and port selection.\n"
        "• See README of pyserial for OS-specific notes.")


def diagnose_rx(tx: bytes, rx: bytes):
    if not rx:
        return ("NO REPLY",
            "No bytes received from the pump.\n"
            "• A/B polarity reversed — try swapping the two RS485 wires.\n"
            "• Wrong pump address (the pump ignores frames not for it).\n"
            "• Baud / parity mismatch — pump expects 1200 bps, 8E1.\n"
            "• Pump unpowered or in local-only mode.")
    counts = Counter(rx)
    most_common_byte, most_common_count = counts.most_common(1)[0]
    if len(rx) >= 3 and most_common_count == len(rx) and most_common_byte in (0x00, 0xFF):
        return (f"GARBAGE ({most_common_count}x 0x{most_common_byte:02X})",
            "The receive line is reading a constant value — the bus is floating or framing is wrong.\n"
            "• Connect GND between the pump and the USB adapter (don't rely on chassis).\n"
            "• Check baud rate — a mismatched baud often shows as 0xFF runs.\n"
            "• Ensure RS485 transceiver has bias resistors (some adapters lack them).\n"
            "• A 120 ohm terminator at each end of the bus reduces reflections.")
    if rx[0] != 0xE9:
        return ("BAD START FLAG",
            f"First byte is 0x{rx[0]:02X}, expected 0xE9 (LongerPump start flag).\n"
            "• Likely a partial frame — increase wait time or read again.\n"
            "• Could also be an echo from your own TX if the adapter loops back.")
    if len(rx) < 5:
        return ("FRAME TRUNCATED",
            f"Only {len(rx)} bytes received — frame is too short.\n"
            "• Increase the read wait time.\n"
            "• Check for noise on the line truncating the message.")
    return ("OK", "")


# ============================================================
# LONGERPUMP PROTOCOL
# ============================================================
class BT100Link:
    def __init__(self, port: str, addr: int, baud: int = 1200):
        self.port = port
        self.addr = int(addr)
        self.baud = int(baud)
        self.ser = None
        self.lock = Lock()

    def open(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baud,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE, timeout=1.0,
            )

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    @staticmethod
    def checksum(addr: int, pdu: bytes) -> int:
        x = addr ^ len(pdu)
        for b in pdu: x ^= b
        return x

    @staticmethod
    def escape(data: bytes) -> bytes:
        out = bytearray()
        for b in data:
            if b == 0xE8:   out.extend([0xE8, 0x00])
            elif b == 0xE9: out.extend([0xE8, 0x01])
            else:           out.append(b)
        return bytes(out)

    def frame(self, pdu: bytes) -> bytes:
        fcs = self.checksum(self.addr, pdu)
        body = bytes([self.addr, len(pdu)]) + pdu + bytes([fcs])
        return bytes([0xE9]) + self.escape(body)

    def send(self, pdu: bytes, wait_s: float = 0.25):
        return self.send_raw(self.frame(pdu), wait_s=wait_s)

    def send_raw(self, tx: bytes, wait_s: float = 0.25):
        with self.lock:
            self.open()
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.write(tx)
            self.ser.flush()
            time.sleep(wait_s)
            rx = self.ser.read_all()
            return bytes(tx), rx

    def set_speed(self, rpm: float, cw: bool = True):
        rpm = max(0.0, min(float(rpm), 100.0))
        raw = int(round(rpm * 10))
        pdu = b"XL" + raw.to_bytes(2, "big") + bytes([0x01, 0x01 if cw else 0x00])
        return self.send(pdu)

    def stop(self, cw: bool = True):
        pdu = b"XL" + b"\x00\x00" + bytes([0x00, 0x01 if cw else 0x00])
        return self.send(pdu)

    def read_speed(self):
        return self.send(b"DL", wait_s=0.45)


def parse_rx_frame(rx: bytes) -> dict:
    out = {"start": None, "addr": None, "length": None,
           "command": None, "data": None, "checksum": None,
           "checksum_ok": None, "decoded": None, "error": None,
           "checksum_index": None}
    if not rx:
        out["error"] = "empty"; return out
    out["start"] = bytes([rx[0]])
    if rx[0] != 0xE9:
        out["error"] = "bad start flag"; return out
    if len(rx) < 5:
        out["error"] = "frame too short"; return out
    out["addr"] = bytes([rx[1]])
    length = rx[2]; out["length"] = bytes([length])
    pdu_end = 3 + length
    if pdu_end + 1 > len(rx):
        out["error"] = "length byte exceeds buffer"; return out
    pdu = rx[3:pdu_end]
    out["command"] = pdu[:2] if len(pdu) >= 2 else pdu
    out["data"] = pdu[2:]
    out["checksum"] = bytes([rx[pdu_end]])
    out["checksum_index"] = pdu_end
    expected = rx[1] ^ length
    for b in pdu: expected ^= b
    out["checksum_ok"] = (expected == rx[pdu_end])
    if out["command"] == b"DL" and len(out["data"]) == 4:
        raw = int.from_bytes(out["data"][:2], "big")
        s1, s2 = out["data"][2], out["data"][3]
        out["decoded"] = {"rpm": raw / 10.0, "running": bool(s1 & 0x01),
                          "direction": "CW" if s2 & 0x01 else "CCW",
                          "raw_rpm_word": raw, "state1": s1, "state2": s2}
    return out


# ============================================================
# HEX COLORIZER
# ============================================================
BYTE_COLOR_START    = "#dc2626"
BYTE_COLOR_ADDR     = "#16a34a"
BYTE_COLOR_LENGTH   = "#0ea5e9"
BYTE_COLOR_COMMAND  = "#7c3aed"
BYTE_COLOR_DATA     = "#172033"
BYTE_COLOR_CHECKSUM = "#ea580c"
BYTE_COLOR_PLAIN    = "#172033"


def colorize_rx_html(rx: bytes) -> str:
    if not rx:
        return "<i style='color:#9ca3af'>&lt;no bytes&gt;</i>"
    parsed = parse_rx_frame(rx)
    spans = []
    cs_idx = parsed["checksum_index"]
    has_valid_frame = parsed["error"] is None
    for i, b in enumerate(rx):
        hex_b = f"{b:02X}"
        if not has_valid_frame:
            color = BYTE_COLOR_PLAIN
            if i == 0 and b == 0xE9:
                color = BYTE_COLOR_START
        elif i == 0:    color = BYTE_COLOR_START
        elif i == 1:    color = BYTE_COLOR_ADDR
        elif i == 2:    color = BYTE_COLOR_LENGTH
        elif i == cs_idx: color = BYTE_COLOR_CHECKSUM
        elif 3 <= i <= 4: color = BYTE_COLOR_COMMAND
        else:           color = BYTE_COLOR_DATA
        spans.append(f"<span style='color:{color};font-weight:700'>{hex_b}</span>")
    return " ".join(spans)


def colorize_tx_html(tx: bytes) -> str:
    if not tx:
        return "<i style='color:#9ca3af'>-</i>"
    n = len(tx); spans = []
    for i, b in enumerate(tx):
        hex_b = f"{b:02X}"
        if i == 0:                color = BYTE_COLOR_START
        elif i == 1:              color = BYTE_COLOR_ADDR
        elif i == 2:              color = BYTE_COLOR_LENGTH
        elif i == n - 1 and n >= 5: color = BYTE_COLOR_CHECKSUM
        elif 3 <= i <= 4:         color = BYTE_COLOR_COMMAND
        else:                     color = BYTE_COLOR_DATA
        spans.append(f"<span style='color:{color};font-weight:700'>{hex_b}</span>")
    return " ".join(spans)


# ============================================================
# LED INDICATOR
# ============================================================
class LedIndicator(QLabel):
    def __init__(self, label: str, on_color: str, off_color: str = "#475569",
                 size: int = 16, flash_ms: int = 100):
        super().__init__()
        self.label = label
        self.on_color = on_color
        self.off_color = off_color
        self.size = size
        self.flash_ms = flash_ms
        self.setFixedSize(size, size)
        self.setToolTip(f"{label} indicator")
        self._apply(self.off_color, glow=False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._reset)

    def _apply(self, color: str, glow: bool):
        border = color if glow else "#1e293b"
        self.setStyleSheet(
            f"background:{color}; border:2px solid {border};"
            f"border-radius:{self.size // 2}px;"
        )

    def flash(self):
        self._apply(self.on_color, glow=True)
        self._timer.start(self.flash_ms)

    def _reset(self):
        self._apply(self.off_color, glow=False)


# ============================================================
# MONITOR THREAD
# ============================================================
class MonitorWorker(QThread):
    transaction = pyqtSignal(bytes, bytes)
    failure = pyqtSignal(str, str)

    def __init__(self, link: BT100Link, interval_s: float = 0.5):
        super().__init__()
        self.link = link
        self.interval_s = interval_s
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                tx, rx = self.link.read_speed()
                self.transaction.emit(tx, rx)
            except serial.SerialException as e:
                title, hint = diagnose_serial_exception(e)
                self.failure.emit(title, hint)
                break
            except Exception as e:
                self.failure.emit("Worker error", str(e))
            time.sleep(self.interval_s)


# ============================================================
# PACKET DECODER WIDGET
# ============================================================
class PacketDecoder(QGroupBox):
    FIELD_COLORS = {
        "Start":    "#fde68a",
        "Address":  "#bbf7d0",
        "Length":   "#bae6fd",
        "Command":  "#ddd6fe",
        "Data":     "#fecaca",
        "Checksum": "#fed7aa",
    }

    def __init__(self):
        super().__init__("Packet Decoder")
        self.tx_label = QLabel("-")
        self.tx_label.setFont(QFont("Consolas", 11))
        self.tx_label.setTextFormat(Qt.TextFormat.RichText)
        self.tx_label.setStyleSheet("padding:4px; background:#ffffff;"
                                    "border:1px solid #e2e8f0; border-radius:6px;")
        self.tx_label.setWordWrap(True)

        self.rx_label = QLabel("-")
        self.rx_label.setFont(QFont("Consolas", 11))
        self.rx_label.setTextFormat(Qt.TextFormat.RichText)
        self.rx_label.setStyleSheet("padding:4px; background:#ffffff;"
                                    "border:1px solid #e2e8f0; border-radius:6px;")
        self.rx_label.setWordWrap(True)

        self.legend_label = QLabel(self._legend_html())
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)
        self.legend_label.setStyleSheet("font-size:10px; padding:2px;")

        self.frame_row = QHBoxLayout()
        self.frame_row.setSpacing(4)

        self.params_label = QLabel("No frame decoded yet.")
        self.params_label.setStyleSheet(
            "background:#0f172a; color:#dbeafe; padding:8px;"
            "border-radius:6px; font-family:Consolas; font-size:13px;"
        )
        self.params_label.setMinimumHeight(110)
        self.params_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.diag_label = QLabel("Waiting for traffic.")
        self.diag_label.setWordWrap(True)
        self.diag_label.setStyleSheet(
            "padding:6px; border:1px solid #d7deea; border-radius:6px; background:#f8fafc;"
        )

        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.addWidget(QLabel("<b>Raw TX</b>"), 0, 0)
        layout.addWidget(self.tx_label, 0, 1)
        layout.addWidget(QLabel("<b>Raw RX</b>"), 1, 0)
        layout.addWidget(self.rx_label, 1, 1)
        layout.addWidget(self.legend_label, 2, 1)
        layout.addWidget(QLabel("<b>RX Frame</b>"), 3, 0)
        layout.addLayout(self.frame_row, 3, 1)
        layout.addWidget(QLabel("<b>Parameters</b>"), 4, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.params_label, 4, 1)
        layout.addWidget(QLabel("<b>RX Health</b>"), 5, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.diag_label, 5, 1)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)

    @staticmethod
    def _legend_html() -> str:
        items = [
            ("Start", BYTE_COLOR_START), ("Address", BYTE_COLOR_ADDR),
            ("Length", BYTE_COLOR_LENGTH), ("Command", BYTE_COLOR_COMMAND),
            ("Data", BYTE_COLOR_DATA), ("Checksum", BYTE_COLOR_CHECKSUM),
        ]
        return "&nbsp;&nbsp;".join(
            f"<span style='color:{c};font-weight:700'>&#9632;</span> {n}" for n, c in items
        )

    def _clear_frame_row(self):
        while self.frame_row.count():
            item = self.frame_row.takeAt(0)
            w = item.widget()
            if w is not None: w.deleteLater()

    def _add_field(self, name: str, value: bytes, ok: bool = True):
        if value is None: return
        color = self.FIELD_COLORS.get(name, "#e5e7eb")
        border = "#16a34a" if ok else "#dc2626"
        box = QFrame()
        box.setStyleSheet(
            f"background:{color}; border:1px solid {border}; border-radius:6px; padding:4px;"
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(4, 2, 4, 2); v.setSpacing(0)
        tag = QLabel(name)
        tag.setStyleSheet("font-size:10px; color:#1f2937; background:transparent;")
        val = QLabel(value.hex(" ").upper())
        val.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        val.setStyleSheet("background:transparent;")
        v.addWidget(tag); v.addWidget(val)
        self.frame_row.addWidget(box)

    def update(self, tx: bytes, rx: bytes):
        self.tx_label.setText(colorize_tx_html(tx) if tx else "-")
        self.rx_label.setText(colorize_rx_html(rx))

        self._clear_frame_row()
        parsed = parse_rx_frame(rx)
        for name, key in [("Start", "start"), ("Address", "addr"), ("Length", "length"),
                          ("Command", "command"), ("Data", "data")]:
            self._add_field(name, parsed[key])
        self._add_field("Checksum", parsed["checksum"], ok=bool(parsed["checksum_ok"]))
        self.frame_row.addStretch()

        if parsed["decoded"]:
            d = parsed["decoded"]
            running = "RUNNING" if d["running"] else "STOPPED"
            self.params_label.setText(
                f"  Actual RPM     : {d['rpm']:>6.1f}    (raw word = {d['raw_rpm_word']} = 0x{d['raw_rpm_word']:04X})\n"
                f"  Direction      : {d['direction']}        (state2 = 0x{d['state2']:02X})\n"
                f"  Run state      : {running}     (state1 = 0x{d['state1']:02X})\n"
                f"  Checksum       : {'VALID' if parsed['checksum_ok'] else 'INVALID'}"
            )
        elif parsed["error"]:
            self.params_label.setText(f"  Cannot decode: {parsed['error']}")
        else:
            self.params_label.setText("  Reply is not a DL (read-speed) response.")

        status, hint = diagnose_rx(tx, rx)
        if not hint:
            self.diag_label.setStyleSheet("padding:6px; border:1px solid #16a34a; border-radius:6px;"
                                          "background:#dcfce7; color:#14532d;")
            self.diag_label.setText(f"<b>{status}</b> - RX frame looks healthy.")
        else:
            self.diag_label.setStyleSheet("padding:6px; border:1px solid #dc2626; border-radius:6px;"
                                          "background:#fee2e2; color:#7f1d1d;")
            self.diag_label.setText(f"<b>{status}</b>\n{hint}")


# ============================================================
# MAIN WINDOW
# ============================================================
class DiagnosticWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BT100-1L RS485 Diagnostic Tool - God Mode")
        self.resize(1180, 820)
        self.link = None
        self.monitor = None

        self.port_box = QComboBox()
        self.addr_edit = QLineEdit("1")
        self.baud_edit = QLineEdit("1200")
        self.rpm_edit = QLineEdit("30.0")
        self.interval_edit = QLineEdit("0.5")
        self.hex_edit = QLineEdit()
        self.hex_edit.setPlaceholderText("e.g.  E9 01 02 44 4C 0D    (Read-speed frame)")
        self.hex_edit.setFont(QFont("Consolas", 11))
        self.dir_box = QComboBox(); self.dir_box.addItems(["CW", "CCW"])

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color:gray; font-size:18px;")
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("font-weight:700;")

        self.tx_led = LedIndicator("TX", on_color="#3b82f6")
        self.rx_led = LedIndicator("RX", on_color="#22c55e")

        self.decoder = PacketDecoder()

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "background:#0f172a; color:#e2e8f0; border-radius:8px; padding:6px;"
            "font-family:Consolas; font-size:11px;"
        )

        self._build_ui()
        self.refresh_ports()

    def _build_ui(self):
        connection = QGroupBox("Connection")
        g = QGridLayout()
        g.setHorizontalSpacing(6); g.setVerticalSpacing(4)
        btn_refresh = QPushButton("Refresh")
        btn_connect = QPushButton("Connect")
        btn_disconnect = QPushButton("Disconnect")

        led_row = QHBoxLayout()
        led_row.setSpacing(6)
        led_row.addWidget(QLabel("TX"))
        led_row.addWidget(self.tx_led)
        led_row.addSpacing(12)
        led_row.addWidget(QLabel("RX"))
        led_row.addWidget(self.rx_led)
        led_row.addStretch()
        led_container = QWidget(); led_container.setLayout(led_row)

        g.addWidget(self.status_dot, 0, 0)
        g.addWidget(self.status_label, 0, 1, 1, 2)
        g.addWidget(led_container, 0, 3)
        g.addWidget(QLabel("COM Port"), 1, 0)
        g.addWidget(self.port_box,    1, 1, 1, 2)
        g.addWidget(btn_refresh,      1, 3)
        g.addWidget(QLabel("Address"), 2, 0)
        g.addWidget(self.addr_edit,    2, 1)
        g.addWidget(QLabel("Baud"),    2, 2)
        g.addWidget(self.baud_edit,    2, 3)
        g.addWidget(btn_connect,    3, 0, 1, 2)
        g.addWidget(btn_disconnect, 3, 2, 1, 2)
        connection.setLayout(g)

        control = QGroupBox("Pump Control")
        c = QGridLayout()
        c.setHorizontalSpacing(6); c.setVerticalSpacing(4)
        btn_start = QPushButton("Start")
        btn_stop = QPushButton("Stop")
        btn_read = QPushButton("Read")
        btn_monitor = QPushButton("Start Monitor")
        btn_stop_mon = QPushButton("Stop Monitor")
        btn_start.setStyleSheet("background:#16a34a; color:white; font-weight:700; padding:6px;")
        btn_stop.setStyleSheet("background:#ea580c; color:white; font-weight:700; padding:6px;")
        c.addWidget(QLabel("Target RPM"), 0, 0)
        c.addWidget(self.rpm_edit,        0, 1)
        c.addWidget(QLabel("Direction"),  0, 2)
        c.addWidget(self.dir_box,         0, 3)
        c.addWidget(QLabel("Poll (s)"),   1, 0)
        c.addWidget(self.interval_edit,   1, 1)
        c.addWidget(btn_read,             1, 2)
        c.addWidget(btn_start,    2, 0, 1, 2)
        c.addWidget(btn_stop,     2, 2, 1, 2)
        c.addWidget(btn_monitor,  3, 0, 1, 2)
        c.addWidget(btn_stop_mon, 3, 2, 1, 2)
        control.setLayout(c)

        manual = QGroupBox("Manual Hex Sender")
        m = QVBoxLayout()
        m.addWidget(QLabel("Type any hex bytes - sent verbatim, no framing applied."))
        m.addWidget(self.hex_edit)
        hex_row = QHBoxLayout()
        btn_send_hex = QPushButton("Send Raw")
        btn_send_hex.setStyleSheet("background:#7c3aed; color:white; font-weight:700; padding:6px;")
        btn_clear_hex = QPushButton("Clear")
        hex_row.addWidget(btn_send_hex)
        hex_row.addWidget(btn_clear_hex)
        m.addLayout(hex_row)
        manual.setLayout(m)

        left_col = QVBoxLayout()
        left_col.addWidget(connection)
        left_col.addWidget(control)
        left_col.addWidget(manual)
        left_col.addStretch()
        left_panel = QWidget(); left_panel.setLayout(left_col)
        left_panel.setMaximumWidth(380)
        left_panel.setMinimumWidth(320)

        log_group = QGroupBox("Diagnostic Log")
        log_lay = QVBoxLayout()
        log_buttons = QHBoxLayout()
        btn_save_log = QPushButton("Save Log")
        btn_clear_log = QPushButton("Clear Log")
        log_buttons.addStretch()
        log_buttons.addWidget(btn_save_log)
        log_buttons.addWidget(btn_clear_log)
        log_lay.addWidget(self.log)
        log_lay.addLayout(log_buttons)
        log_group.setLayout(log_lay)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.decoder)
        right_split.addWidget(log_group)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

        root = QHBoxLayout()
        root.addWidget(left_panel)
        root.addWidget(right_split, 1)
        self.setLayout(root)

        btn_refresh.clicked.connect(self.refresh_ports)
        btn_connect.clicked.connect(self.connect_pump)
        btn_disconnect.clicked.connect(self.disconnect_pump)
        btn_start.clicked.connect(self.start_pump)
        btn_stop.clicked.connect(self.stop_pump)
        btn_read.clicked.connect(self.read_once)
        btn_monitor.clicked.connect(self.start_monitor)
        btn_stop_mon.clicked.connect(self.stop_monitor)
        btn_send_hex.clicked.connect(self.send_manual_hex)
        btn_clear_hex.clicked.connect(self.hex_edit.clear)
        self.hex_edit.returnPressed.connect(self.send_manual_hex)
        btn_clear_log.clicked.connect(self.log.clear)
        btn_save_log.clicked.connect(self.save_log)

    def set_status(self, text: str, color: str):
        self.status_label.setText(text)
        self.status_dot.setStyleSheet(f"color:{color}; font-size:18px;")

    def log_line(self, text: str, kind: str = "info"):
        colors = {"info": "#e2e8f0", "tx": "#7dd3fc", "rx": "#86efac",
                  "warn": "#fbbf24", "err": "#fca5a5"}
        c = colors.get(kind, "#e2e8f0")
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"<span style='color:#64748b'>[{stamp}]</span> "
                        f"<span style='color:{c}'>{text}</span>")

    def report_failure(self, title: str, hint: str):
        self.set_status(title, "red")
        self.log_line(f"<b>{title}</b>", "err")
        for line in hint.splitlines():
            self.log_line(f"  {line}", "warn")

    def refresh_ports(self):
        current = self.port_box.currentData()
        self.port_box.clear()
        ports = list_serial_ports()
        for p in ports:
            real = extract_display_com(p)
            open_port = normalize_windows_com(real)
            self.port_box.addItem(f"{real} | {p.description}", open_port)
        if current:
            for i in range(self.port_box.count()):
                if self.port_box.itemData(i) == current:
                    self.port_box.setCurrentIndex(i); break
        self.log_line(f"Ports refreshed - found {len(ports)}.")
        if not ports:
            self.log_line("No USB-Serial adapter detected. Plug in the RS485 dongle.", "warn")

    def _validated_inputs(self):
        port = self.port_box.currentData()
        if not port:
            raise ValueError("No COM port selected - click Refresh and pick one.")
        addr = int(self.addr_edit.text().strip())
        if not 1 <= addr <= 31:
            raise ValueError("Address must be between 1 and 31.")
        baud = int(self.baud_edit.text().strip())
        if baud <= 0:
            raise ValueError("Baud must be positive (factory default 1200).")
        return port, addr, baud

    def connect_pump(self):
        self.disconnect_pump(quiet=True)
        try:
            port, addr, baud = self._validated_inputs()
        except Exception as e:
            self.report_failure("Invalid input", str(e)); return
        try:
            self.link = BT100Link(port, addr, baud=baud)
            self.link.open()
            self.set_status(f"Connected {port} @ {baud} 8E1, addr={addr}", "green")
            self.log_line(f"Opened {port} @ {baud} bps 8E1, addr={addr}.", "info")
            self.log_line("Tip: click 'Read' to confirm the pump replies before starting it.", "warn")
        except serial.SerialException as e:
            self.link = None
            title, hint = diagnose_serial_exception(e)
            self.report_failure(title, hint)
        except Exception as e:
            self.link = None
            self.report_failure("Connect failed", str(e))

    def disconnect_pump(self, quiet: bool = False):
        self.stop_monitor()
        try:
            if self.link:
                try: self.link.stop()
                except Exception: pass
                self.link.close()
        finally:
            self.link = None
        self.set_status("Disconnected", "gray")
        if not quiet:
            self.log_line("Disconnected.", "info")

    def _require_link(self):
        if self.link is None:
            raise RuntimeError("Not connected. Click Connect first.")
        self.link.addr = int(self.addr_edit.text().strip())
        return self.link

    def _do_transaction(self, fn, label: str):
        try:
            tx, rx = fn()
        except serial.SerialException as e:
            title, hint = diagnose_serial_exception(e)
            self.report_failure(title, hint)
            self.disconnect_pump(quiet=True)
            return
        except Exception as e:
            self.report_failure(f"{label} error", str(e)); return

        self.tx_led.flash()
        if rx:
            self.rx_led.flash()
        self.decoder.update(tx, rx)
        self.log_line(f"{label} TX: {tx.hex(' ').upper()}", "tx")
        self.log_line(f"{label} RX: {rx.hex(' ').upper() if rx else '<no bytes>'}", "rx")
        status, hint = diagnose_rx(tx, rx)
        if hint:
            self.report_failure(status, hint)
        else:
            self.set_status(f"{label} OK - {status}", "green")

    def start_pump(self):
        try:
            link = self._require_link()
            rpm = float(self.rpm_edit.text().strip())
            cw = self.dir_box.currentText() == "CW"
        except Exception as e:
            self.report_failure("Cannot start", str(e)); return
        self._do_transaction(lambda: link.set_speed(rpm, cw=cw), "START")

    def stop_pump(self):
        try:
            link = self._require_link()
            cw = self.dir_box.currentText() == "CW"
        except Exception as e:
            self.report_failure("Cannot stop", str(e)); return
        self._do_transaction(lambda: link.stop(cw=cw), "STOP")

    def read_once(self):
        try:
            link = self._require_link()
        except Exception as e:
            self.report_failure("Cannot read", str(e)); return
        self._do_transaction(lambda: link.read_speed(), "READ")

    def send_manual_hex(self):
        try:
            data = parse_hex_string(self.hex_edit.text())
        except Exception as e:
            self.report_failure("Bad hex string", str(e)); return
        try:
            link = self._require_link()
        except Exception as e:
            self.report_failure("Cannot send", str(e)); return
        self.log_line(f"MANUAL -> {len(data)} byte(s): {data.hex(' ').upper()}", "info")
        self._do_transaction(lambda: link.send_raw(data, wait_s=0.45), "HEX")

    def start_monitor(self):
        try:
            link = self._require_link()
            interval = max(0.2, float(self.interval_edit.text().strip()))
        except Exception as e:
            self.report_failure("Cannot monitor", str(e)); return
        self.stop_monitor()
        w = MonitorWorker(link, interval_s=interval)
        w.transaction.connect(self._handle_monitor_tx)
        w.failure.connect(self.report_failure)
        self.monitor = w
        w.start()
        self.set_status(f"Monitoring every {interval:.2f}s", "green")
        self.log_line(f"Monitor started ({interval:.2f}s interval).", "info")

    def stop_monitor(self):
        if self.monitor and self.monitor.isRunning():
            self.monitor.stop()
            self.monitor.wait(1500)
            self.log_line("Monitor stopped.", "info")
        self.monitor = None

    def _handle_monitor_tx(self, tx: bytes, rx: bytes):
        self.tx_led.flash()
        if rx:
            self.rx_led.flash()
        self.decoder.update(tx, rx)
        self.log_line(f"MON TX: {tx.hex(' ').upper()}", "tx")
        self.log_line(f"MON RX: {rx.hex(' ').upper() if rx else '<no bytes>'}", "rx")
        status, hint = diagnose_rx(tx, rx)
        if hint:
            self.report_failure(status, hint)

    def save_log(self):
        default_name = f"bt100_diag_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Diagnostic Log", default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# BT100-1L RS485 Diagnostic Log\n")
                f.write(f"# Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# " + "=" * 60 + "\n\n")
                f.write(self.log.toPlainText())
            self.log_line(f"Log saved to {path}", "info")
        except Exception as e:
            self.report_failure("Save failed", str(e))

    def closeEvent(self, event):
        self.disconnect_pump(quiet=True)
        event.accept()


STYLESHEET = """
QWidget { font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px;
          color: #172033; background: #f4f7fb; }
QGroupBox { background: #ffffff; border: 1px solid #d7deea; border-radius: 10px;
            margin-top: 12px; padding: 10px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px;
                   color: #1f3a5f; background: #f4f7fb; }
QLineEdit, QComboBox { background: #ffffff; border: 1px solid #b8c4d6;
                       border-radius: 6px; padding: 5px 7px; min-height: 22px; }
QLineEdit:focus, QComboBox:focus { border: 2px solid #2f80ed; padding: 4px 6px; }
QPushButton { background: #e9eff8; color: #172033; border: 1px solid #b9c7da;
              border-radius: 6px; padding: 6px 10px; font-weight: 650; }
QPushButton:hover { background: #dbe7f7; border-color: #8ea9ce; }
QPushButton:pressed { background: #c5d7ee; }
"""


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = DiagnosticWindow()
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
