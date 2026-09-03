"""
BT100-1L RS485 Dual-Pump Controller
===================================
Controls TWO Longer BT100-1L peristaltic pumps at the same time, either

    * both on ONE RS485 bus  (same COM port, different Pump I.D.)  <- recommended
    * or on TWO separate USB-RS485 adapters (different COM ports)

Features
    * live RPM read-out per pump (polled with the DL command)
    * slider + spin box to change speed, applied on the fly while running
    * run / direction state, optional flow-rate estimate
    * single I/O thread => no bus collisions when both pumps share a port
    * packet decoder + colour log kept from the diagnostic tool

Run with:  py -3.11 "C:\\MOF_NanoTec\\pump\\bt100_dual.py"
"""

import os
import sys
import time
import json
import queue
import re
from collections import Counter

import serial
import serial.tools.list_ports

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QCheckBox,
    QTextEdit, QGridLayout, QHBoxLayout, QVBoxLayout, QSlider, QButtonGroup,
    QGroupBox, QFrame, QSplitter, QFileDialog, QDoubleSpinBox, QSpinBox,
)

# BT100-1L technical specification (operating manual p.11)
RPM_MIN = 0.1
RPM_MAX = 100.0
PUMP_NAMES = ("Pump 1", "Pump 2")
PUMP_TINT = ("#2f80ed", "#7c3aed")


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


def adapter_key(port_info) -> str:
    """A name for the physical USB-RS485 dongle that survives a COM renumber.

    Windows hands out COM numbers per machine and per USB socket, so COM13 on
    one laptop can be COM7 on the next. The FTDI/CH340 serial number does not
    change, so that is what the app remembers instead.
    """
    if port_info.serial_number:
        return f"SER:{port_info.serial_number}"
    vid = f"{port_info.vid:04X}" if port_info.vid is not None else "----"
    pid = f"{port_info.pid:04X}" if port_info.pid is not None else "----"
    # no serial number (common on cheap CH340 clones): fall back to the USB
    # socket the dongle is plugged into, which is at least stable on one machine
    return f"USB:{vid}:{pid}@{port_info.location or '?'}"


def adapter_label(port_info) -> str:
    key = adapter_key(port_info)
    tail = key.split(":", 1)[1]
    return f"{extract_display_com(port_info)} | {port_info.description or 'serial port'} | {tail}"


SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bt100_dual_settings.json")


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}                      # first run, or the file was hand-edited badly


def save_settings(data: dict):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False                   # a read-only folder must not kill the app


def list_serial_ports():
    valid = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").upper()
        hwid = (p.hwid or "").upper()
        if "BLUETOOTH" in desc or "BTHENUM" in hwid:
            continue
        valid.append(p)
    return valid


# ============================================================
# DIAGNOSTIC TRANSLATORS
# ============================================================
def diagnose_serial_exception(exc: Exception):
    msg = str(exc); low = msg.lower()
    if "access is denied" in low or "permissionerror" in low or "errno 13" in low:
        return ("Port in use / Access denied",
            "This COM port is already opened by another program.\n"
            "- Close any other terminal (PuTTY, Arduino IDE, the diagnostic tool).\n"
            "- If both pumps share one bus, point BOTH pumps at the same COM port\n"
            "  here with different addresses - this app opens the port only once.\n"
            "- Unplug and reinsert the USB-RS485 adapter.")
    if "could not open port" in low or "filenotfound" in low or "errno 2" in low or "the system cannot find" in low:
        return ("Port not found",
            "The selected COM port does not exist anymore.\n"
            "- Click 'Refresh' - the adapter may have changed COM number.\n"
            "- Check that the USB-RS485 dongle is plugged in.")
    if "semaphore" in low or "errno 121" in low or "timed out" in low or "timeout" in low:
        return ("Timeout / No response",
            "The pump did not reply within the timeout window.\n"
            "- Swap RS485 A and B wires - polarity is the #1 cause of silence.\n"
            "- Confirm 1200 bps, 8E1 (factory default for BT100-1L).\n"
            "- Verify the Pump I.D. matches (MENU > 4 Pump I.D.).\n"
            "- Set MENU > 6 Remote Control = On, otherwise RS485 is ignored.")
    if "input/output error" in low or "clearcommerror" in low or "errno 5" in low:
        return ("Cable / driver I/O error",
            "The OS reported a low-level I/O failure on the adapter.\n"
            "- Unplug the USB adapter and plug it back in.\n"
            "- Try a different USB port (avoid unpowered hubs).")
    return ("Serial error", f"Unrecognised serial error:\n  {msg}")


def diagnose_rx(tx: bytes, rx: bytes, expect_addr=None):
    if not rx:
        return ("NO REPLY",
            "No bytes received from the pump.\n"
            "- A/B polarity reversed - try swapping the two RS485 wires.\n"
            "- Wrong Pump I.D. (the pump ignores frames not addressed to it).\n"
            "- Remote Control is Off in the pump menu (MENU > 6).\n"
            "- Baud / parity mismatch - pump expects 1200 bps, 8E1.")
    counts = Counter(rx)
    most_common_byte, most_common_count = counts.most_common(1)[0]
    if len(rx) >= 3 and most_common_count == len(rx) and most_common_byte in (0x00, 0xFF):
        return (f"GARBAGE ({most_common_count}x 0x{most_common_byte:02X})",
            "The receive line reads a constant value - the bus floats or framing is wrong.\n"
            "- Connect GND between the pump and the USB adapter.\n"
            "- Check baud rate; a mismatch often shows as 0xFF runs.\n"
            "- A 120 ohm terminator at each end of the bus reduces reflections.")
    if rx[0] != 0xE9:
        return ("BAD START FLAG",
            f"First byte is 0x{rx[0]:02X}, expected 0xE9 (LongerPump start flag).\n"
            "- Likely a partial frame, or an echo of your own TX.\n"
            "- On a shared bus this also happens if two masters talk at once.")
    if len(rx) < 5:
        return ("FRAME TRUNCATED", f"Only {len(rx)} bytes received - frame is too short.")
    if expect_addr is not None and rx[1] != expect_addr:
        return ("WRONG ADDRESS",
            f"Reply came from address {rx[1]}, but {expect_addr} was addressed.\n"
            "- Both pumps probably share the same Pump I.D. Give them different IDs\n"
            "  in MENU > 4 Pump I.D. (e.g. 01# and 02#).")
    return ("OK", "")


# ============================================================
# LONGERPUMP PROTOCOL
# ============================================================
def unescape(body: bytes) -> bytes:
    """Undo the 0xE8 transparency encoding used inside a frame body."""
    out = bytearray(); esc = False
    for b in body:
        if esc:
            out.append(0xE8 if b == 0x00 else 0xE9); esc = False
        elif b == 0xE8:
            esc = True
        else:
            out.append(b)
    return bytes(out)


class SerialBus:
    """One physical COM port, shared by every pump wired to that bus."""

    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.ser = None
        self.refs = 0

    def open(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baud,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05, write_timeout=3.0,
            )

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    def transact(self, tx: bytes, timeout: float = 1.0):
        """Write a frame, then read until a complete reply arrives or time is up.

        Called only from the single I/O thread, so two pumps on the same bus
        can never talk over each other.
        """
        self.open()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.write(tx)
        self.ser.flush()

        raw = bytearray(); body = bytearray()
        started = False; esc = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.ser.read(16)
            if not chunk:
                continue
            for b in chunk:
                raw.append(b)
                if not started:
                    if b == 0xE9:
                        started = True
                    continue
                if esc:
                    body.append(0xE8 if b == 0x00 else 0xE9); esc = False
                elif b == 0xE8:
                    esc = True
                else:
                    body.append(b)
            if len(body) >= 3 and len(body) >= 2 + body[1] + 1:
                break
        time.sleep(0.02)          # RS485 line turnaround guard
        return bytes(tx), bytes(raw)


class BT100Link:
    """Protocol layer for one pump: one address on one bus."""

    def __init__(self, bus: SerialBus, addr: int):
        self.bus = bus
        self.addr = int(addr)

    @staticmethod
    def checksum(addr: int, pdu: bytes) -> int:
        x = addr ^ len(pdu)
        for b in pdu:
            x ^= b
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

    def send(self, pdu: bytes, timeout: float = 1.0):
        return self.bus.transact(self.frame(pdu), timeout=timeout)

    def set_speed(self, rpm: float, cw: bool = True):
        """XL sets speed AND run, so resending it while running retunes rpm live."""
        rpm = max(0.0, min(float(rpm), RPM_MAX))
        raw = int(round(rpm * 10))
        pdu = b"XL" + raw.to_bytes(2, "big") + bytes([0x01, 0x01 if cw else 0x00])
        return self.send(pdu)

    def stop(self, cw: bool = True):
        pdu = b"XL" + b"\x00\x00" + bytes([0x00, 0x01 if cw else 0x00])
        return self.send(pdu)

    def read_speed(self):
        return self.send(b"DL", timeout=1.2)


def parse_rx_frame(rx: bytes) -> dict:
    out = {"start": None, "addr": None, "length": None,
           "command": None, "data": None, "checksum": None,
           "checksum_ok": None, "decoded": None, "error": None,
           "checksum_index": None}
    if not rx:
        out["error"] = "empty"; return out
    if rx[0] != 0xE9:
        out["error"] = "bad start flag"; return out
    out["start"] = b"\xE9"

    body = unescape(rx[1:])
    if len(body) < 4:
        out["error"] = "frame too short"; return out
    length = body[1]
    if 2 + length + 1 > len(body):
        out["error"] = "length byte exceeds buffer"; return out

    pdu = body[2:2 + length]
    out["addr"] = bytes([body[0]])
    out["length"] = bytes([length])
    out["command"] = pdu[:2] if len(pdu) >= 2 else pdu
    out["data"] = pdu[2:]
    out["checksum"] = bytes([body[2 + length]])
    out["checksum_index"] = len(rx) - 1     # exact when nothing was escaped

    expected = body[0] ^ length
    for b in pdu:
        expected ^= b
    out["checksum_ok"] = (expected == body[2 + length])

    if out["command"] == b"DL" and len(out["data"]) == 4:
        raw = int.from_bytes(out["data"][:2], "big")
        s1, s2 = out["data"][2], out["data"][3]
        out["decoded"] = {"rpm": raw / 10.0, "running": bool(s1 & 0x01),
                          "direction": "CW" if s2 & 0x01 else "CCW",
                          "raw_rpm_word": raw, "state1": s1, "state2": s2}
    return out


# ============================================================
# I/O WORKER - the only thread that touches a serial port
# ============================================================
class IOWorker(QThread):
    """Serialises every transaction of every pump through one command queue.

    The GUI never blocks on the bus: it posts a command and gets a signal back.
    Polling happens whenever the queue is idle, round-robin over the pumps, so
    two pumps sharing one RS485 bus can never transmit at the same time.
    """

    txrx     = pyqtSignal(int, str, object, object)   # slot, label, tx, rx
    linkinfo = pyqtSignal(int, bool, str)             # slot, connected, message
    failed   = pyqtSignal(int, str, str)              # slot, title, hint
    scanline = pyqtSignal(int, object)                # address, rx bytes
    scandone = pyqtSignal(object)                     # list of addresses that answered

    def __init__(self):
        super().__init__()
        self.cmds = queue.Queue()
        self.links = {}            # slot -> BT100Link
        self.buses = {}            # port -> SerialBus
        self.polling = {}          # slot -> bool
        self.last_poll = {}        # slot -> timestamp
        self.interval = 1.0
        self._running = True

    # ---------- called from the GUI thread ----------
    def submit(self, **cmd):
        self.cmds.put(cmd)

    def shutdown(self):
        self._running = False

    # ---------- worker thread ----------
    def run(self):
        while self._running:
            try:
                cmd = self.cmds.get(timeout=0.02)
            except queue.Empty:
                self._poll_due()
                continue
            try:
                self._handle(cmd)
            except serial.SerialException as e:
                slot = cmd.get("slot", 0)
                title, hint = diagnose_serial_exception(e)
                self.failed.emit(slot, title, hint)
                self._close(slot, quiet=True)
            except Exception as e:
                self.failed.emit(cmd.get("slot", 0), f"{cmd.get('op')} error", str(e))
        for slot in list(self.links):
            self._close(slot, quiet=True)

    def _poll_due(self):
        now = time.time()
        for slot, link in list(self.links.items()):
            if not self.polling.get(slot):
                continue
            if now - self.last_poll.get(slot, 0.0) < self.interval:
                continue
            self.last_poll[slot] = now
            try:
                tx, rx = link.read_speed()
                self.txrx.emit(slot, "POLL", tx, rx)
            except serial.SerialException as e:
                title, hint = diagnose_serial_exception(e)
                self.failed.emit(slot, title, hint)
                self._close(slot, quiet=True)
            except Exception as e:
                self.failed.emit(slot, "Poll error", str(e))
            return          # one pump per pass -> fair round-robin

    def _handle(self, cmd):
        op = cmd["op"]
        slot = cmd.get("slot", 0)

        if op == "interval":
            self.interval = max(0.3, float(cmd["value"])); return
        if op == "poll":
            self.polling[slot] = bool(cmd["value"]); return
        if op == "open":
            self._open(slot, cmd["port"], int(cmd["baud"]), int(cmd["addr"])); return
        if op == "close":
            self._close(slot); return
        if op == "scan":
            self._scan(cmd["port"], int(cmd["baud"]),
                       int(cmd["lo"]), int(cmd["hi"])); return

        link = self.links.get(slot)
        if link is None:
            self.failed.emit(slot, "Not connected", "Click Connect for this pump first.")
            return

        if op == "set_speed":
            tx, rx = link.set_speed(cmd["rpm"], cw=cmd["cw"])
            label = "SPEED" if cmd.get("live") else "START"
        elif op == "stop":
            tx, rx = link.stop(cw=cmd["cw"]); label = "STOP"
        elif op == "read":
            tx, rx = link.read_speed(); label = "READ"
        else:
            return
        self.txrx.emit(slot, label, tx, rx)
        self.last_poll[slot] = time.time()      # keep the poll from crowding it

    def _scan(self, port, baud, lo, hi):
        """Send a DL read to every Pump I.D. in turn and report who answers.

        This is the only way to tell 'pump is on another I.D.' apart from
        'pump is not on the bus at all' - both look like NO REPLY otherwise.
        """
        bus = self.buses.get(port)
        borrowed = bus is not None and bus.refs > 0
        if bus is None:
            bus = SerialBus(port, baud)
            self.buses[port] = bus
        bus.open()

        found = []
        for addr in range(lo, hi + 1):
            probe = BT100Link(bus, addr)
            # a valid reply is 10 bytes = 92 ms at 1200 baud, so 0.5 s is ample
            _, rx = bus.transact(probe.frame(b"DL"), timeout=0.5)
            self.scanline.emit(addr, rx)
            if rx:
                found.append(addr)
        if not borrowed and bus.refs == 0:
            bus.close()
            self.buses.pop(port, None)
        self.scandone.emit(found)

    def _open(self, slot, port, baud, addr):
        self._close(slot, quiet=True)
        bus = self.buses.get(port)
        if bus is not None and bus.refs == 0 and bus.baud != baud:
            bus.close(); bus = None
        if bus is None:
            bus = SerialBus(port, baud)
            self.buses[port] = bus
        elif bus.refs > 0 and bus.baud != baud:
            # the port is already open for the other pump; one bus = one baud rate
            self.failed.emit(slot, "Baud rate ignored",
                             f"{port} is already open at {bus.baud} bps for the other pump. "
                             f"Both pumps on one RS485 bus must share the same baud rate.")
        bus.open()
        bus.refs += 1
        self.links[slot] = BT100Link(bus, addr)
        self.last_poll[slot] = 0.0
        shared = " (shared bus)" if bus.refs > 1 else ""
        self.linkinfo.emit(slot, True, f"{port} @ {baud} 8E1, addr={addr}{shared}")

    def _close(self, slot, quiet=False):
        link = self.links.pop(slot, None)
        self.polling[slot] = False
        if link is None:
            if not quiet:
                self.linkinfo.emit(slot, False, "Disconnected")
            return
        try:
            link.stop()
        except Exception:
            pass
        bus = link.bus
        bus.refs = max(0, bus.refs - 1)
        if bus.refs == 0:
            bus.close()
        if not quiet:
            self.linkinfo.emit(slot, False, "Disconnected")


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


def _colorize(data: bytes, cs_idx) -> str:
    if not data:
        return "<i style='color:#9ca3af'>-</i>"
    spans = []
    for i, b in enumerate(data):
        if i == 0:            color = BYTE_COLOR_START if b == 0xE9 else BYTE_COLOR_PLAIN
        elif i == 1:          color = BYTE_COLOR_ADDR
        elif i == 2:          color = BYTE_COLOR_LENGTH
        elif i == cs_idx:     color = BYTE_COLOR_CHECKSUM
        elif 3 <= i <= 4:     color = BYTE_COLOR_COMMAND
        else:                 color = BYTE_COLOR_DATA
        spans.append(f"<span style='color:{color};font-weight:700'>{b:02X}</span>")
    return " ".join(spans)


def colorize_rx_html(rx: bytes) -> str:
    if not rx:
        return "<i style='color:#9ca3af'>&lt;no bytes&gt;</i>"
    parsed = parse_rx_frame(rx)
    return _colorize(rx, parsed["checksum_index"] if parsed["error"] is None else -1)


def colorize_tx_html(tx: bytes) -> str:
    return _colorize(tx, len(tx) - 1 if len(tx) >= 5 else -1)


# ============================================================
# SMALL WIDGETS
# ============================================================
class LedIndicator(QLabel):
    def __init__(self, label: str, on_color: str, off_color: str = "#cbd5e1",
                 size: int = 14, flash_ms: int = 90):
        super().__init__()
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
        border = color if glow else "#94a3b8"
        self.setStyleSheet(f"background:{color}; border:2px solid {border};"
                           f"border-radius:{self.size // 2}px;")

    def flash(self):
        self._apply(self.on_color, glow=True)
        self._timer.start(self.flash_ms)

    def _reset(self):
        self._apply(self.off_color, glow=False)


class Chip(QLabel):
    """Small coloured status pill."""

    def __init__(self, text: str = "-"):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state(text, "#64748b", "#e2e8f0")

    def set_state(self, text: str, fg: str, bg: str):
        self.setText(text)
        self.setStyleSheet(f"color:{fg}; background:{bg}; border-radius:9px;"
                           f"padding:3px 10px; font-weight:700; font-size:11px;")


# ============================================================
# PUMP PANEL - one per pump, identical behaviour
# ============================================================
class PumpPanel(QGroupBox):
    def __init__(self, slot: int, submit, log):
        super().__init__(PUMP_NAMES[slot])
        self.slot = slot
        self.submit = submit
        self.log = log
        self.connected = False
        self.running = False
        self.addr = 1
        self.peer = None          # the other PumpPanel, set by the main window
        self.port_keys = {}       # COM name -> stable adapter id
        self.want_key = None      # adapter this pump was last connected to
        self._syncing = False

        self.tint = PUMP_TINT[slot]

        # --- connection widgets ---
        self.port_box = QComboBox()
        self.addr_spin = QSpinBox(); self.addr_spin.setRange(1, 99)
        # Both pumps normally stay on the factory I.D. 01#, each on its own
        # USB-RS485 adapter. Two pumps may only share one address if they are
        # on different COM ports - do_connect() enforces that.
        self.addr_spin.setValue(1)
        self.addr_spin.setToolTip("Pump I.D. from the pump menu (MENU > 4).\n"
                                  "Leave both pumps at 1 and give each one its own COM port.\n"
                                  "Only change this if both pumps share a single RS485 bus.")
        self.baud_box = QComboBox(); self.baud_box.addItems(["1200", "2400", "4800", "9600"])
        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)

        self.status_chip = Chip("OFFLINE")
        self.tx_led = LedIndicator("TX", "#3b82f6")
        self.rx_led = LedIndicator("RX", "#22c55e")

        # --- live read-out ---
        self.rpm_label = QLabel("--.-")
        self.rpm_label.setFont(QFont("Consolas", 46, QFont.Weight.Bold))
        self.rpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rpm_label.setStyleSheet(f"color:{self.tint}; padding:0px;")
        self.unit_label = QLabel("RPM  (actual, read from pump)")
        self.unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unit_label.setStyleSheet("color:#64748b; font-size:11px;")

        self.run_chip = Chip("STOPPED")
        self.dir_chip = Chip("CW")
        self.flow_label = QLabel("flow: -")
        self.flow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flow_label.setStyleSheet("color:#475569; font-size:11px;")

        # --- setpoint widgets ---
        self.rpm_spin = QDoubleSpinBox()
        self.rpm_spin.setRange(0.0, RPM_MAX); self.rpm_spin.setDecimals(1)
        self.rpm_spin.setSingleStep(0.1); self.rpm_spin.setValue(30.0)
        self.rpm_spin.setSuffix(" rpm")

        self.rpm_slider = QSlider(Qt.Orientation.Horizontal)
        self.rpm_slider.setRange(0, int(RPM_MAX * 10)); self.rpm_slider.setValue(300)
        self.rpm_slider.setStyleSheet(
            "QSlider::groove:horizontal { height:6px; background:#dbe4f0; border-radius:3px; }"
            f"QSlider::handle:horizontal {{ background:{self.tint}; width:18px; margin:-7px 0;"
            "  border-radius:9px; }"
            f"QSlider::sub-page:horizontal {{ background:{self.tint}; border-radius:3px; }}"
        )

        self.btn_ccw = QPushButton("◀  CCW")
        self.btn_cw = QPushButton("CW  ▶")
        for b in (self.btn_ccw, self.btn_cw):
            b.setCheckable(True)
            b.setMinimumHeight(30)
        self.btn_cw.setChecked(True)
        self.dir_group = QButtonGroup(self)
        self.dir_group.setExclusive(True)
        self.dir_group.addButton(self.btn_ccw, 0)
        self.dir_group.addButton(self.btn_cw, 1)
        self._style_dir_buttons()
        self.live_chk = QCheckBox("Live apply while running")
        self.live_chk.setChecked(True)
        self.live_chk.setToolTip("Resend the speed command as soon as the slider moves,\n"
                                 "so the pump retunes without stopping.")
        self.poll_chk = QCheckBox("Poll speed")
        self.poll_chk.setChecked(True)

        self.ulrev_spin = QDoubleSpinBox()
        self.ulrev_spin.setRange(0.0, 99999.0); self.ulrev_spin.setDecimals(1)
        self.ulrev_spin.setValue(0.0); self.ulrev_spin.setSuffix(" uL/rev")
        self.ulrev_spin.setToolTip("Volume per revolution of this pump head + tubing.\n"
                                   "Calibrate as in the manual (p.4), then flow = rpm x uL/rev.\n"
                                   "Leave 0 to hide the flow estimate.")

        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_read = QPushButton("Read once")
        self.btn_start.setStyleSheet("background:#16a34a; color:white; font-size:14px;"
                                     "font-weight:800; padding:10px; border:none; border-radius:8px;")
        self.btn_stop.setStyleSheet("background:#ea580c; color:white; font-size:14px;"
                                    "font-weight:800; padding:10px; border:none; border-radius:8px;")

        self._build()
        self._wire()
        self._set_controls_enabled(False)

    # ---------- layout ----------
    def _build(self):
        head = QHBoxLayout()
        head.addWidget(self.status_chip)
        head.addStretch()
        head.addWidget(QLabel("TX")); head.addWidget(self.tx_led)
        head.addSpacing(8)
        head.addWidget(QLabel("RX")); head.addWidget(self.rx_led)

        conn = QGridLayout()
        conn.setHorizontalSpacing(6); conn.setVerticalSpacing(4)
        conn.addWidget(QLabel("COM"), 0, 0)
        conn.addWidget(self.port_box, 0, 1, 1, 3)
        conn.addWidget(QLabel("Pump I.D."), 1, 0)
        conn.addWidget(self.addr_spin, 1, 1)
        conn.addWidget(QLabel("Baud"), 1, 2)
        conn.addWidget(self.baud_box, 1, 3)
        conn.addWidget(self.btn_connect, 2, 0, 1, 2)
        conn.addWidget(self.btn_disconnect, 2, 2, 1, 2)
        conn.setColumnStretch(1, 1); conn.setColumnStretch(3, 1)

        readout = QVBoxLayout(); readout.setSpacing(2)
        readout.addWidget(self.rpm_label)
        readout.addWidget(self.unit_label)
        chips = QHBoxLayout(); chips.addStretch()
        chips.addWidget(self.run_chip); chips.addWidget(self.dir_chip)
        chips.addStretch()
        readout.addLayout(chips)
        readout.addWidget(self.flow_label)
        box = QFrame()
        box.setObjectName("readoutBox")
        box.setStyleSheet("QFrame#readoutBox { background:#f8fafc;"
                          " border:1px solid #e2e8f0; border-radius:10px; }")
        box.setLayout(readout)

        sp = QGridLayout()
        sp.setHorizontalSpacing(6); sp.setVerticalSpacing(4)
        sp.addWidget(QLabel("Target"), 0, 0)
        sp.addWidget(self.rpm_spin, 0, 1)
        sp.addWidget(self.btn_ccw, 0, 2)
        sp.addWidget(self.btn_cw, 0, 3)
        sp.addWidget(self.rpm_slider, 1, 0, 1, 4)
        sp.addWidget(self.live_chk, 2, 0, 1, 2)
        sp.addWidget(self.poll_chk, 2, 2, 1, 2)
        sp.addWidget(QLabel("Calib"), 3, 0)
        sp.addWidget(self.ulrev_spin, 3, 1)
        sp.addWidget(self.btn_read, 3, 2, 1, 2)
        sp.addWidget(self.btn_start, 4, 0, 1, 2)
        sp.addWidget(self.btn_stop, 4, 2, 1, 2)
        sp.setColumnStretch(1, 1); sp.setColumnStretch(3, 1)

        root = QVBoxLayout()
        root.addLayout(head)
        root.addLayout(conn)
        root.addWidget(box, 1)      # the RPM read-out absorbs any extra height,
        root.addLayout(sp)          # e.g. when the log is switched off
        self.setLayout(root)

    def _wire(self):
        self.btn_connect.clicked.connect(self.do_connect)
        self.btn_disconnect.clicked.connect(self.do_disconnect)
        self.btn_start.clicked.connect(self.do_start)
        self.btn_stop.clicked.connect(self.do_stop)
        self.btn_read.clicked.connect(lambda: self.submit(op="read", slot=self.slot))
        self.poll_chk.toggled.connect(
            lambda on: self.submit(op="poll", slot=self.slot, value=on and self.connected))

        self.rpm_slider.valueChanged.connect(self._slider_moved)
        self.rpm_spin.valueChanged.connect(self._spin_moved)
        self.btn_cw.clicked.connect(self._direction_clicked)
        self.btn_ccw.clicked.connect(self._direction_clicked)

        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(250)          # debounce a dragged slider
        self._live_timer.timeout.connect(self._live_apply)

    # ---------- helpers ----------
    def _style_dir_buttons(self):
        for b, active in ((self.btn_ccw, self.btn_ccw.isChecked()),
                          (self.btn_cw, self.btn_cw.isChecked())):
            if active:
                b.setStyleSheet(f"background:{self.tint}; color:white; border:none;"
                                "border-radius:6px; font-size:14px; font-weight:800;")
            else:
                b.setStyleSheet("background:#e9eff8; color:#64748b; border:1px solid #b9c7da;"
                                "border-radius:6px; font-size:14px; font-weight:700;")

    def _set_controls_enabled(self, on: bool):
        for w in (self.btn_start, self.btn_stop, self.btn_read,
                  self.rpm_slider, self.rpm_spin, self.btn_cw, self.btn_ccw):
            w.setEnabled(on)

    def target_rpm(self) -> float:
        return self.rpm_spin.value()

    def is_cw(self) -> bool:
        return self.btn_cw.isChecked()

    def _direction_clicked(self):
        self._style_dir_buttons()
        # a direction change while running must be pushed out, or the read-out
        # and the pump disagree until the next START
        if self.connected and self.running:
            self.submit(op="set_speed", slot=self.slot,
                        rpm=self.target_rpm(), cw=self.is_cw(), live=True)

    def _slider_moved(self, value: int):
        if self._syncing:
            return
        self._syncing = True
        self.rpm_spin.setValue(value / 10.0)
        self._syncing = False
        self._maybe_live()

    def _spin_moved(self, value: float):
        if self._syncing:
            return
        self._syncing = True
        self.rpm_slider.setValue(int(round(value * 10)))
        self._syncing = False
        self._maybe_live()

    def _maybe_live(self):
        if self.connected and self.running and self.live_chk.isChecked():
            self._live_timer.start()

    def _live_apply(self):
        if self.connected and self.running:
            self.submit(op="set_speed", slot=self.slot,
                        rpm=self.target_rpm(), cw=self.is_cw(), live=True)

    # ---------- actions ----------
    def do_connect(self):
        display = self.port_box.currentData()
        if not display:
            self.log(self.slot, "No COM port selected. Click Refresh.", "warn"); return

        # Two pumps on one bus must not answer to the same Pump I.D.; if both are
        # left on the factory I.D. 1 they each need their own USB-RS485 adapter.
        peer = self.peer
        if (peer is not None and peer.connected
                and peer.port_box.currentData() == display
                and peer.addr_spin.value() == self.addr_spin.value()):
            self.log(self.slot,
                     f"{peer.title()} is already using I.D. {peer.addr_spin.value()} on {display}.",
                     "err")
            self.log(self.slot,
                     "   Two pumps cannot share one COM port AND one I.D. - they would "
                     "answer at the same time and both start together.", "warn")
            self.log(self.slot,
                     "   Fix: put this pump on a second USB-RS485 adapter (its own COM port), "
                     "or give it a different I.D. in MENU > 4 Pump I.D.", "warn")
            self.set_error("I.D. CLASH")
            return

        self.addr = self.addr_spin.value()
        self.submit(op="open", slot=self.slot,
                    port=normalize_windows_com(display),
                    baud=int(self.baud_box.currentText()),
                    addr=self.addr)

    def do_disconnect(self):
        self.submit(op="close", slot=self.slot)

    def do_start(self):
        if not self.connected:
            self.log(self.slot, "Not connected.", "warn"); return
        rpm = self.target_rpm()
        if rpm < RPM_MIN:
            self.log(self.slot, f"Target {rpm:.1f} rpm is below the {RPM_MIN} rpm minimum.", "warn")
            return
        self.submit(op="set_speed", slot=self.slot, rpm=rpm, cw=self.is_cw(), live=False)

    def do_stop(self):
        if not self.connected:
            return
        self.submit(op="stop", slot=self.slot, cw=self.is_cw())

    # ---------- state from the worker ----------
    def set_connected(self, ok: bool, message: str):
        self.connected = ok
        self.btn_connect.setEnabled(not ok)
        self.btn_disconnect.setEnabled(ok)
        self.port_box.setEnabled(not ok)
        self.addr_spin.setEnabled(not ok)
        self.baud_box.setEnabled(not ok)
        self._set_controls_enabled(ok)
        if ok:
            self.status_chip.set_state("ONLINE", "#065f46", "#d1fae5")
            self.submit(op="poll", slot=self.slot, value=self.poll_chk.isChecked())
        else:
            self.status_chip.set_state("OFFLINE", "#64748b", "#e2e8f0")
            self.running = False
            self.rpm_label.setText("--.-")
            self.run_chip.set_state("STOPPED", "#64748b", "#e2e8f0")
            self.flow_label.setText("flow: -")

    def set_error(self, title: str):
        self.status_chip.set_state(title.upper()[:18], "#7f1d1d", "#fee2e2")

    def update_readout(self, d: dict):
        rpm = d["rpm"]
        self.running = d["running"]
        self.rpm_label.setText(f"{rpm:.1f}")
        if self.running:
            self.run_chip.set_state("RUNNING", "#065f46", "#bbf7d0")
        else:
            self.run_chip.set_state("STOPPED", "#64748b", "#e2e8f0")
        self.dir_chip.set_state("◀ " + d["direction"] if d["direction"] == "CCW"
                                else d["direction"] + " ▶", "#1e3a8a", "#dbeafe")

        ulrev = self.ulrev_spin.value()
        if ulrev > 0:
            ulmin = rpm * ulrev
            if ulmin >= 1000:
                self.flow_label.setText(f"flow: {ulmin / 1000:.2f} mL/min")
            else:
                self.flow_label.setText(f"flow: {ulmin:.1f} uL/min")
        else:
            self.flow_label.setText("flow: set uL/rev to estimate")

    def fill_ports(self, ports):
        """Rebuild the port list and re-select this pump's own adapter.

        Selection is by adapter_key, not by COM number, so moving the setup to
        another laptop (or another USB socket) still lands on the right dongle.
        """
        was_com = self.port_box.currentData()
        was_key = self.port_keys.get(was_com)
        self.port_box.blockSignals(True)
        self.port_box.clear()
        self.port_keys = {}
        for p in ports:
            com = extract_display_com(p)
            key = adapter_key(p)
            self.port_keys[com] = key
            self.port_box.addItem(adapter_label(p), com)
            self.port_box.setItemData(self.port_box.count() - 1,
                                      f"adapter id: {key}", Qt.ItemDataRole.ToolTipRole)

        wanted = self.want_key or was_key
        idx = -1
        if wanted:
            for i in range(self.port_box.count()):
                if self.port_keys.get(self.port_box.itemData(i)) == wanted:
                    idx = i; break
        if idx < 0 and was_com:
            idx = self.port_box.findData(was_com)
        if idx >= 0:
            self.port_box.setCurrentIndex(idx)
        elif self.port_box.count() > 0:
            # default: pump 1 -> first adapter, pump 2 -> second (or the same one)
            self.port_box.setCurrentIndex(self.slot if self.port_box.count() > self.slot else 0)
        self.port_box.blockSignals(False)

        chosen = self.port_box.currentData()
        if (self.want_key and idx >= 0 and was_com and chosen != was_com):
            self.log(self.slot, f"adapter {self.want_key.split(':', 1)[1]} moved "
                                f"{was_com} -> {chosen}, re-selected automatically.", "info")
        return chosen

    def current_key(self):
        return self.port_keys.get(self.port_box.currentData())

    def dump_settings(self) -> dict:
        return {"adapter": self.current_key(),
                "com": self.port_box.currentData(),
                "addr": self.addr_spin.value(),
                "baud": self.baud_box.currentText(),
                "rpm": self.rpm_spin.value(),
                "cw": self.is_cw(),
                "ulrev": self.ulrev_spin.value(),
                "poll": self.poll_chk.isChecked(),
                "live": self.live_chk.isChecked()}

    def apply_settings(self, d: dict):
        """Restore what the user had last time; anything missing keeps its default."""
        if not isinstance(d, dict):
            return
        self.want_key = d.get("adapter") or None
        self.addr_spin.setValue(int(d.get("addr", 1)))
        baud = str(d.get("baud", "1200"))
        if self.baud_box.findText(baud) >= 0:
            self.baud_box.setCurrentText(baud)
        self.rpm_spin.setValue(float(d.get("rpm", 30.0)))
        self.ulrev_spin.setValue(float(d.get("ulrev", 0.0)))
        self.poll_chk.setChecked(bool(d.get("poll", True)))
        self.live_chk.setChecked(bool(d.get("live", True)))
        if d.get("cw", True):
            self.btn_cw.setChecked(True)
        else:
            self.btn_ccw.setChecked(True)
        self._style_dir_buttons()


# ============================================================
# PACKET DECODER
# ============================================================
class PacketDecoder(QGroupBox):
    FIELD_COLORS = {
        "Start": "#fde68a", "Address": "#bbf7d0", "Length": "#bae6fd",
        "Command": "#ddd6fe", "Data": "#fecaca", "Checksum": "#fed7aa",
    }

    def __init__(self, title: str = "Packet Decoder", tint: str = "#1f3a5f"):
        super().__init__(title)
        self.source = QLabel("-")
        self.source.setStyleSheet(f"font-weight:700; color:{tint};")

        self.tx_label = self._mono()
        self.rx_label = self._mono()

        self.legend_label = QLabel(self._legend_html())
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)
        self.legend_label.setStyleSheet("font-size:10px; padding:2px;")

        self.frame_row = QHBoxLayout(); self.frame_row.setSpacing(4)

        self.params_label = QLabel("No frame decoded yet.")
        self.params_label.setStyleSheet("background:#0f172a; color:#dbeafe; padding:8px;"
                                        "border-radius:6px; font-family:Consolas; font-size:12px;")
        self.params_label.setMinimumHeight(86)
        self.params_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.diag_label = QLabel("Waiting for traffic.")
        self.diag_label.setWordWrap(True)
        self.diag_label.setStyleSheet("padding:6px; border:1px solid #d7deea;"
                                      "border-radius:6px; background:#f8fafc;")

        lay = QGridLayout()
        lay.setHorizontalSpacing(8); lay.setVerticalSpacing(5)
        lay.addWidget(QLabel("<b>Source</b>"), 0, 0); lay.addWidget(self.source, 0, 1)
        lay.addWidget(QLabel("<b>Raw TX</b>"), 1, 0); lay.addWidget(self.tx_label, 1, 1)
        lay.addWidget(QLabel("<b>Raw RX</b>"), 2, 0); lay.addWidget(self.rx_label, 2, 1)
        lay.addWidget(self.legend_label, 3, 1)
        lay.addWidget(QLabel("<b>RX Frame</b>"), 4, 0); lay.addLayout(self.frame_row, 4, 1)
        lay.addWidget(QLabel("<b>Parameters</b>"), 5, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.params_label, 5, 1)
        lay.addWidget(QLabel("<b>RX Health</b>"), 6, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.diag_label, 6, 1)
        lay.setColumnStretch(1, 1)
        self.setLayout(lay)

    @staticmethod
    def _mono():
        lab = QLabel("-")
        lab.setFont(QFont("Consolas", 11))
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setStyleSheet("padding:4px; background:#ffffff; border:1px solid #e2e8f0;"
                          "border-radius:6px;")
        lab.setWordWrap(True)
        return lab

    @staticmethod
    def _legend_html() -> str:
        items = [("Start", BYTE_COLOR_START), ("Address", BYTE_COLOR_ADDR),
                 ("Length", BYTE_COLOR_LENGTH), ("Command", BYTE_COLOR_COMMAND),
                 ("Data", BYTE_COLOR_DATA), ("Checksum", BYTE_COLOR_CHECKSUM)]
        return "&nbsp;&nbsp;".join(
            f"<span style='color:{c};font-weight:700'>&#9632;</span> {n}" for n, c in items)

    def _clear_frame_row(self):
        while self.frame_row.count():
            item = self.frame_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_field(self, name, value: bytes, ok=None):
        if value is None:
            return
        box = QFrame()
        colour = self.FIELD_COLORS.get(name, "#e2e8f0")
        border = "#16a34a" if ok is True else ("#dc2626" if ok is False else "#cbd5e1")
        box.setStyleSheet(f"background:{colour}; border:2px solid {border}; border-radius:6px;")
        tag = QLabel(name); tag.setStyleSheet("font-size:9px; color:#334155; border:none;")
        tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel(value.hex(" ").upper() or "-")
        val.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        val.setStyleSheet("border:none;")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v = QVBoxLayout(); v.setContentsMargins(6, 3, 6, 3); v.setSpacing(0)
        v.addWidget(tag); v.addWidget(val)
        box.setLayout(v)
        self.frame_row.addWidget(box)

    def update_frame(self, source: str, tx: bytes, rx: bytes, expect_addr=None):
        self.source.setText(source)
        self.tx_label.setText(colorize_tx_html(tx))
        self.rx_label.setText(colorize_rx_html(rx))

        self._clear_frame_row()
        parsed = parse_rx_frame(rx)
        for name, key in [("Start", "start"), ("Address", "addr"), ("Length", "length"),
                          ("Command", "command"), ("Data", "data")]:
            self._add_field(name, parsed[key])
        self._add_field("Checksum", parsed["checksum"], ok=bool(parsed["checksum_ok"]))
        self.frame_row.addStretch()

        d = parsed["decoded"]
        if d:
            running = "RUNNING" if d["running"] else "STOPPED"
            self.params_label.setText(
                f"  Actual RPM : {d['rpm']:>6.1f}   (raw word = {d['raw_rpm_word']} = 0x{d['raw_rpm_word']:04X})\n"
                f"  Direction  : {d['direction']:<6}   (state2 = 0x{d['state2']:02X})\n"
                f"  Run state  : {running:<7}  (state1 = 0x{d['state1']:02X})\n"
                f"  Checksum   : {'VALID' if parsed['checksum_ok'] else 'INVALID'}")
        elif parsed["error"]:
            self.params_label.setText(f"  Cannot decode: {parsed['error']}")
        else:
            self.params_label.setText("  Command acknowledged (not a DL speed reply).")

        status, hint = diagnose_rx(tx, rx, expect_addr)
        if not hint:
            self.diag_label.setStyleSheet("padding:6px; border:1px solid #16a34a;"
                                          "border-radius:6px; background:#dcfce7; color:#14532d;")
            self.diag_label.setText(f"<b>{status}</b> - RX frame looks healthy.")
        else:
            self.diag_label.setStyleSheet("padding:6px; border:1px solid #dc2626;"
                                          "border-radius:6px; background:#fee2e2; color:#7f1d1d;")
            self.diag_label.setText(f"<b>{status}</b><br>{hint.replace(chr(10), '<br>')}")
        return status, hint


# ============================================================
# MAIN WINDOW
# ============================================================
class DualPumpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BT100-1L RS485 Dual-Pump Controller")
        self.resize(1380, 900)

        self.worker = IOWorker()
        self.worker.txrx.connect(self.on_txrx)
        self.worker.linkinfo.connect(self.on_linkinfo)
        self.worker.failed.connect(self.on_failed)
        self.worker.scanline.connect(self.on_scanline)
        self.worker.scandone.connect(self.on_scandone)
        self.worker.start()

        self.panels = [PumpPanel(0, self.worker.submit, self.log_line),
                       PumpPanel(1, self.worker.submit, self.log_line)]
        self.panels[0].peer = self.panels[1]
        self.panels[1].peer = self.panels[0]

        # one decoder per pump, so the two traffic streams never overwrite each other
        self.decoders = [PacketDecoder(f"Packet Decoder - {PUMP_NAMES[i]}", PUMP_TINT[i])
                         for i in (0, 1)]
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#0f172a; color:#e2e8f0; border-radius:8px;"
                               "padding:6px; font-family:Consolas; font-size:11px;")

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.3, 10.0); self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(1.0); self.interval_spin.setSuffix(" s")
        self.interval_spin.setToolTip("Poll period per pump. At 1200 baud one read takes\n"
                                      "roughly 0.2 s, so keep this at 0.5 s or more for two pumps.")

        self.log_enable = QCheckBox("Enable log")
        self.log_enable.setChecked(True)
        self.log_enable.setToolTip("Turn the log off to stop it collecting text\n"
                                   "during long unattended runs.")

        self._build()

        # restore the previous session before listing ports, so fill_ports can
        # look for the adapter each pump was actually wired to last time
        cfg = load_settings()
        for slot, panel in enumerate(self.panels):
            panel.apply_settings((cfg.get("pumps") or [{}, {}])[slot]
                                 if len(cfg.get("pumps") or []) > slot else {})
        self.interval_spin.setValue(float(cfg.get("interval", 1.0)))
        self.log_enable.setChecked(bool(cfg.get("log", True)))

        self.refresh_ports()
        self.interval_spin.valueChanged.connect(
            lambda v: self.worker.submit(op="interval", value=v))
        self.worker.submit(op="interval", value=self.interval_spin.value())
        self.log_line(None, "Ready. Both pumps on I.D. 1: give each one its own COM port.", "info")
        self.log_line(None, "Pump menu must have 6 Remote Control = On for RS485 to work.", "warn")
        if cfg:
            self.log_line(None, f"Settings restored from {os.path.basename(SETTINGS_PATH)}.", "info")

    # ---------- layout ----------
    def _build(self):
        bar = QGroupBox("Master Control")
        b = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Ports")
        self.btn_scan = QPushButton("Scan Bus (I.D. 1-16)")
        self.btn_scan.setToolTip("Ask every Pump I.D. from 1 to 16 who is out there.\n"
                                 "Use it when a pump answers NO REPLY - it tells you\n"
                                 "whether the pump sits on a different I.D. or is not\n"
                                 "on the bus at all. Takes about 10 s at 1200 baud.")
        btn_all_start = QPushButton("START BOTH")
        btn_all_stop = QPushButton("STOP BOTH")
        btn_estop = QPushButton("EMERGENCY STOP")
        btn_all_start.setStyleSheet("background:#16a34a; color:white; font-weight:800;"
                                    "padding:8px 18px; border:none; border-radius:8px;")
        btn_all_stop.setStyleSheet("background:#ea580c; color:white; font-weight:800;"
                                   "padding:8px 18px; border:none; border-radius:8px;")
        btn_estop.setStyleSheet("background:#b91c1c; color:white; font-weight:800;"
                                "padding:8px 18px; border:none; border-radius:8px;")
        b.addWidget(btn_refresh)
        b.addWidget(self.btn_scan)
        b.addSpacing(12)
        b.addWidget(QLabel("Poll every"))
        b.addWidget(self.interval_spin)
        b.addSpacing(12)
        b.addWidget(self.log_enable)
        b.addStretch()
        b.addWidget(btn_all_start); b.addWidget(btn_all_stop); b.addWidget(btn_estop)
        bar.setLayout(b)

        btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_scan.clicked.connect(self.scan_bus)
        btn_all_start.clicked.connect(self.start_both)
        btn_all_stop.clicked.connect(self.stop_both)
        btn_estop.clicked.connect(self.emergency_stop)

        pumps = QHBoxLayout()
        pumps.addWidget(self.panels[0])
        pumps.addWidget(self.panels[1])

        # the log now sits where the manual hex sender used to be
        self.log_group = QGroupBox("Diagnostic Log")
        lg = QVBoxLayout()
        row = QHBoxLayout()
        btn_save = QPushButton("Save Log"); btn_clear = QPushButton("Clear Log")
        row.addStretch(); row.addWidget(btn_save); row.addWidget(btn_clear)
        lg.addLayout(row); lg.addWidget(self.log)
        self.log_group.setLayout(lg)
        self.log.setMinimumHeight(150)
        btn_save.clicked.connect(self.save_log)
        btn_clear.clicked.connect(self.log.clear)
        self.log_enable.toggled.connect(self._toggle_log)

        left = QVBoxLayout()
        left.addWidget(bar)
        left.addLayout(pumps, 3)          # the pump panels take the freed space
        left.addWidget(self.log_group, 2) # when the log is hidden
        left_panel = QWidget(); left_panel.setLayout(left)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.decoders[0])
        right.addWidget(self.decoders[1])
        right.setSizes([400, 400])

        root = QHBoxLayout()
        root.addWidget(left_panel, 3)
        root.addWidget(right, 2)
        self.setLayout(root)

    def _toggle_log(self, on: bool):
        """Hide the whole group, not just the text box, so the pumps grow into it.

        The check box itself lives in the master bar, never inside the group,
        or turning the log off would hide the only way to turn it back on.
        """
        self.log_group.setVisible(on)

    # ---------- helpers ----------
    def refresh_ports(self):
        ports = list_serial_ports()
        available = {adapter_key(p) for p in ports}
        for p in self.panels:
            p.fill_ports(ports)

        # do not let both pumps default to the same dongle when two are present
        if (len(ports) > 1 and not self.panels[1].connected
                and self.panels[0].port_box.currentData() == self.panels[1].port_box.currentData()
                and not self.panels[1].want_key):
            box = self.panels[1].port_box
            box.setCurrentIndex((box.currentIndex() + 1) % box.count())

        self.log_line(None, f"Ports refreshed - found {len(ports)}.", "info")
        if not ports:
            self.log_line(None, "No USB-Serial adapter detected. Plug in the RS485 dongle.", "warn")
        for slot, panel in enumerate(self.panels):
            if panel.want_key and panel.want_key not in available:
                self.log_line(slot, f"saved adapter {panel.want_key.split(':', 1)[1]} is not "
                                    f"plugged in - falling back to "
                                    f"{panel.port_box.currentData() or 'nothing'}.", "warn")

    def log_line(self, slot, text: str, kind: str = "info"):
        if not self.log_enable.isChecked():
            return
        colors = {"tx": "#60a5fa", "rx": "#4ade80", "warn": "#fbbf24",
                  "err": "#f87171", "info": "#e2e8f0"}
        tag = f"[{PUMP_NAMES[slot]}] " if slot is not None else ""
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"<span style='color:{colors.get(kind, '#e2e8f0')}'>"
                        f"[{stamp}] {tag}{text}</span>")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def scan_bus(self):
        """Probe every Pump I.D. on the bus that the pumps are wired to."""
        port = baud = None
        for p in self.panels:                       # prefer a bus that is already open
            if p.connected:
                port = normalize_windows_com(p.port_box.currentData())
                baud = int(p.baud_box.currentText())
                break
        if port is None:
            display = self.panels[0].port_box.currentData()
            if not display:
                self.log_line(None, "No COM port selected - click Refresh Ports first.", "warn")
                return
            port = normalize_windows_com(display)
            baud = int(self.panels[0].baud_box.currentText())

        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Scanning...")
        self.log_line(None, f"Scanning {port} @ {baud} 8E1 for Pump I.D. 1-16 ...", "info")
        self.worker.submit(op="scan", port=port, baud=baud, lo=1, hi=16)

    def on_scanline(self, addr: int, rx: bytes):
        if not rx:
            return                                   # silence is the normal case, don't spam
        parsed = parse_rx_frame(rx)
        d = parsed["decoded"]
        detail = f"{d['rpm']:.1f} rpm, {'RUNNING' if d['running'] else 'STOPPED'}" if d else "replied"
        self.log_line(None, f"  I.D. {addr:>2} -> {detail}   [{rx.hex(' ').upper()}]", "rx")

    def on_scandone(self, found):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("Scan Bus (I.D. 1-16)")
        if not found:
            self.log_line(None, "Scan finished: NO pump answered on any I.D. 1-16.", "err")
            for line in ("Nothing is talking on this bus at all, so the problem is not the I.D.:",
                         "- swap the RS485 A and B wires,",
                         "- check the pump is powered and MENU > 6 Remote Control = On,",
                         "- check A/B of the second pump is really daisy-chained to the same pair,",
                         "- connect GND between both pumps and the USB adapter."):
                self.log_line(None, "   " + line, "warn")
            return

        self.log_line(None, f"Scan finished: pump(s) answered on I.D. {found}.", "info")
        wanted = [p.addr_spin.value() for p in self.panels]
        for slot, want in enumerate(wanted):
            if want not in found:
                free = [a for a in found if a not in wanted]
                hint = (f"try setting {PUMP_NAMES[slot]} to I.D. {free[0]} here, "
                        f"or change the pump to {want} in MENU > 4 Pump I.D.") if free else \
                       (f"no spare I.D. was found - the two pumps are probably both set to "
                        f"{found[0]}, which they must not share.")
                self.log_line(slot, f"I.D. {want} did not answer: {hint}", "warn")

    def start_both(self):
        for p in self.panels:
            if p.connected:
                p.do_start()

    def stop_both(self):
        for p in self.panels:
            if p.connected:
                p.do_stop()

    def emergency_stop(self):
        for p in self.panels:
            p.live_chk.setChecked(False)
            if p.connected:
                p.do_stop()
                p.do_stop()          # sent twice: a dropped frame must not leave it running
        self.log_line(None, "EMERGENCY STOP issued to both pumps.", "err")

    # ---------- worker signals ----------
    def on_txrx(self, slot: int, label: str, tx: bytes, rx: bytes):
        panel = self.panels[slot]
        panel.tx_led.flash()
        if rx:
            panel.rx_led.flash()

        parsed = parse_rx_frame(rx)
        if parsed["decoded"]:
            panel.update_readout(parsed["decoded"])

        # a poll every second would drown the log, so only log real commands
        if label != "POLL":
            self.log_line(slot, f"{label} TX: {tx.hex(' ').upper()}", "tx")
            self.log_line(slot, f"{label} RX: {rx.hex(' ').upper() if rx else '<no bytes>'}", "rx")

        status, hint = self.decoders[slot].update_frame(
            f"{label}   (I.D. {panel.addr})", tx, rx, expect_addr=panel.addr)
        if hint and label != "POLL":
            self.log_line(slot, f"{status}: {hint.splitlines()[0]}", "err")
            panel.set_error(status)
        elif hint and label == "POLL":
            panel.set_error(status)
        elif panel.connected:
            panel.status_chip.set_state("ONLINE", "#065f46", "#d1fae5")

    def on_linkinfo(self, slot: int, connected: bool, message: str):
        panel = self.panels[slot]
        panel.set_connected(connected, message)
        self.log_line(slot, message, "info" if connected else "warn")
        if connected:
            # remember which dongle this pump is on, by hardware id not COM number
            panel.want_key = panel.current_key()
            self.store_settings()
            self.worker.submit(op="read", slot=slot)

    def store_settings(self):
        cfg = {"pumps": [p.dump_settings() for p in self.panels],
               "interval": self.interval_spin.value(),
               "log": self.log_enable.isChecked()}
        if not save_settings(cfg):
            self.log_line(None, f"Could not write {SETTINGS_PATH} - settings not saved.", "warn")

    def on_failed(self, slot: int, title: str, hint: str):
        self.panels[slot].set_error(title)
        self.log_line(slot, f"{title}", "err")
        for line in hint.splitlines():
            self.log_line(slot, "   " + line, "warn")

    def save_log(self):
        default_name = f"bt100_dual_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", default_name,
                                              "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# BT100-1L Dual-Pump Log\n")
                f.write(f"# Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.log.toPlainText())
            self.log_line(None, f"Log saved to {path}", "info")
        except Exception as e:
            self.log_line(None, f"Save failed: {e}", "err")

    def closeEvent(self, event):
        self.store_settings()
        for slot in (0, 1):
            self.worker.submit(op="close", slot=slot)
        self.worker.shutdown()
        self.worker.wait(3000)
        event.accept()


STYLESHEET = """
QWidget { font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px;
          color: #172033; background: #f4f7fb; }
QGroupBox { background: #ffffff; border: 1px solid #d7deea; border-radius: 10px;
            margin-top: 12px; padding: 10px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px;
                   color: #1f3a5f; background: #f4f7fb; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff; border: 1px solid #b8c4d6; border-radius: 6px;
    padding: 4px 6px; min-height: 22px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #2f80ed; padding: 3px 5px; }
QPushButton { background: #e9eff8; color: #172033; border: 1px solid #b9c7da;
              border-radius: 6px; padding: 6px 10px; font-weight: 650; }
QPushButton:hover { background: #dbe7f7; border-color: #8ea9ce; }
QPushButton:pressed { background: #c5d7ee; }
QPushButton:disabled { background: #eef2f7; color: #9aa7b8; border-color: #dde4ee; }
QCheckBox { spacing: 6px; }
"""


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = DualPumpWindow()
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
