#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 p4_bridge.py — ช่องคุยกับจอ ESP32-P4 (NDJSON บน USB-Serial-JTAG)
============================================================================
 กฎเหล็กของไฟล์นี้

  1. ห้ามแตะพอร์ตของบอร์ด CONTROL เด็ดขาด
     P1 ไม่ได้เป็นเจ้าของ logger ด้วยซ้ำ — ระบบเดิมยังเก็บ CSV อยู่ตามปกติ
  2. ห้ามเขียนอะไรลงโฟลเดอร์ของ legacy
     เขียนได้เฉพาะ data/events/ ของโปรเจกต์นี้เท่านั้น
  3. bridge ตายต้องไม่ทำให้อย่างอื่นตาย — ห่อทั้ง run() ด้วย try/except
  4. UI ห้ามอ่านสถานะจากเธรดนี้ตรง ๆ — ดึงผ่าน snapshot() ที่ล็อกไว้แล้ว

 จังหวะเวลา (ตัวเลขมาจาก pc_bridge.c ฝั่งจอ ไม่ใช่จากเอกสาร)
    ส่ง state ทุก 3 วินาที      (จอถือว่า PC หายเมื่อเงียบเกิน 10 วินาที)
    จอเงียบเกิน 10 วินาที       -> P4 OFFLINE
============================================================================
"""

import socket
import threading
import time
from collections import deque

from . import protocol as P
from .mask import MaskSync

LINK_DISABLED = "DISABLED"     # ปิดด้วย config — ไม่ใช่ความผิดพลาด
LINK_OFFLINE  = "OFFLINE"      # เปิดอยู่แต่ยังไม่เห็นจอ
LINK_ONLINE   = "ONLINE"
LINK_ERROR    = "ERROR"

LINK_TEXT = {
    LINK_DISABLED: "P4 BRIDGE DISABLED",
    LINK_OFFLINE:  "P4 OFFLINE",
    LINK_ONLINE:   "P4 CONNECTED",
    LINK_ERROR:    "P4 BRIDGE ERROR",
}


# ============================================================================
#  Transport — แยกออกมาเพื่อให้เทสต์ไม่ต้องมีพอร์ตจริง
# ============================================================================
class Transport(object):
    def read_line(self, timeout=0.2):  raise NotImplementedError
    def write(self, data):             raise NotImplementedError
    def close(self):                   pass
    @property
    def connected(self):               return False


class LoopbackTransport(Transport):
    """ป้อนบรรทัดจาก list ตรง ๆ — ใช้ในเทสต์"""

    def __init__(self, lines=None):
        self.inbox = deque(lines or [])
        self.sent = []
        self._open = True

    def feed(self, line):
        self.inbox.append(line if isinstance(line, bytes)
                          else line.encode("utf-8"))

    def read_line(self, timeout=0.2):
        return self.inbox.popleft() if self.inbox else None

    def write(self, data):
        self.sent.append(data)
        return True

    def close(self):
        self._open = False

    @property
    def connected(self):
        return self._open


class SocketTransport(Transport):
    """รับการเชื่อมต่อจาก mock_p4.py บน localhost

    ใช้ TCP แทนพอร์ตอนุกรมเพื่อให้ทดสอบได้ทั้งบน Windows และ Linux
    โดยไม่ต้องมี com0com หรือ pty
    """

    def __init__(self, host="127.0.0.1", port=8781):
        self.addr = (host, port)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(self.addr)
        self._srv.listen(1)
        self._srv.settimeout(0.2)
        self._cli = None
        self._buf = b""

    def _accept(self):
        try:
            self._cli, _ = self._srv.accept()
            self._cli.settimeout(0.2)
            self._buf = b""
        except (socket.timeout, OSError):
            pass

    def read_line(self, timeout=0.2):
        if self._cli is None:
            self._accept()
            return None
        while b"\n" not in self._buf:
            try:
                chunk = self._cli.recv(4096)
            except socket.timeout:
                return None
            except OSError:
                chunk = b""
            if not chunk:
                self._drop()
                return None
            self._buf += chunk
            if len(self._buf) > 64 * 1024:      # กันบัฟเฟอร์บวมจากบรรทัดที่ไม่จบ
                self._buf = b""
                return None
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def write(self, data):
        if self._cli is None:
            return False
        try:
            self._cli.sendall(data)
            return True
        except OSError:
            self._drop()
            return False

    def _drop(self):
        try:
            self._cli.close()
        except Exception:
            pass
        self._cli = None

    def close(self):
        self._drop()
        try:
            self._srv.close()
        except Exception:
            pass

    @property
    def connected(self):
        return self._cli is not None


class SerialTransport(Transport):
    """พอร์ตจริงของจอ — เปิดใหม่เองเมื่อจอรีบูต

    ⚠️ USB-Serial-JTAG หายไปจากระบบตอนจอรีบูตเป็นเรื่องปกติ ไม่ใช่ความผิดพลาด
       ต้องหาพอร์ตใหม่ทุกครั้ง ไม่ใช่จำเลข COM เดิมไว้
    """

    def __init__(self, port=None, baud=115200, retry_s=2.0):
        self.port = port
        self.baud = baud
        self.retry_s = retry_s
        self._ser = None
        self._next_try = 0.0
        self.reconnects = 0
        self.device = port
        self.last_reason = None

    def _ensure(self):
        if self._ser is not None:
            return True
        if time.time() < self._next_try:
            return False
        self._next_try = time.time() + self.retry_s
        try:
            import serial
            from ..core import ports as PORTS
            if self.port:
                dev, reason = self.port, PORTS.PICK_OK
            else:
                dev, reason, cands = PORTS.find_detailed(PORTS.ROLE_P4_BRIDGE)
                if reason == PORTS.PICK_AMBIGUOUS:
                    # ⚠️ เจอ 303A:1001 หลายตัว — ห้ามเดา
                    #    เปิดพอร์ตผิดตัวแปลว่าไปยึดพอร์ตของบอร์ดอื่นไว้ด้วย
                    self.last_reason = (
                        "เจอพอร์ต 303A:1001 %d ตัว (%s) — ระบุด้วย --bridge-port"
                        % (len(cands), ", ".join(c["device"] for c in cands)))
                    return False
            self.last_reason = None if dev else "ไม่พบพอร์ต 303A:1001"
            if not dev:
                return False
            # ⚠️ ต้องกด DTR/RTS ลงก่อน open() ไม่งั้นการเปิดพอร์ตจะรีเซ็ตจอ
            #
            #    USB-Serial-JTAG ของ ESP32-P4 ใช้ DTR/RTS เป็นสัญญาณ reset/boot
            #    เหมือน USB-serial ทั่วไป  pyserial ยืนยันสองเส้นนี้ให้อัตโนมัติ
            #    ตอนเปิด จอจึงรีบูตทุกครั้งที่เราต่อ และรีบูตอีกครั้งตอนเราปิด
            #
            #    เจอจริงตอน soak 2 ชั่วโมง: จบขั้น E แล้วจอรีบูตทันที
            #    rst:0x17 (CHIP_USB_UART_RESET) ไม่ใช่ crash แต่เป็นเราเอง
            #    ในห้องแล็บอาการนี้จะดูเหมือน "จอรีสตาร์ตเองแบบสุ่ม" ซึ่งไล่หา
            #    สาเหตุยากมากเพราะฝั่งจอไม่มีอะไรผิดให้เห็นเลย
            #
            #    ต้องตั้งค่าบนอ็อบเจกต์ที่ยังไม่เปิด แล้วค่อย open()
            #    การตั้งหลัง open() ไม่ช่วย เพราะจอรีเซ็ตไปแล้ว
            ser = serial.Serial()
            ser.port = dev
            ser.baudrate = self.baud
            ser.timeout = 0.2
            ser.dtr = False
            ser.rts = False
            ser.open()
            self._ser = ser
            self.device = dev
            self.reconnects += 1
            return True
        except Exception as exc:
            self.last_reason = "เปิดพอร์ตไม่ได้: %s" % exc
            self._ser = None
            return False

    def read_line(self, timeout=0.2):
        if not self._ensure():
            return None
        try:
            line = self._ser.readline()
        except Exception:
            self._drop()
            return None
        return line.rstrip(b"\r\n") if line else None

    def write(self, data):
        if not self._ensure():
            return False
        try:
            self._ser.write(data)
            self._ser.flush()
            return True
        except Exception:
            self._drop()
            return False

    def _drop(self):
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def close(self):
        self._drop()

    @property
    def connected(self):
        return self._ser is not None


# ============================================================================
#  Bridge
# ============================================================================
class P4Bridge(object):
    def __init__(self, cfg, event_log, state_source, transport=None,
                 on_event=None, raw=None):
        self.cfg = cfg
        self.log = event_log
        self.state_source = state_source
        self.on_event = on_event          # callback ให้ UI (เรียกจากเธรดนี้)
        b = cfg.get("bridge", {})
        self.enabled = bool(b.get("enabled", True))
        self.state_interval = float(b.get("state_interval_s", 3.0))
        self.offline_after = float(b.get("offline_after_s", 10.0))
        self.transport = transport

        self.mask = MaskSync()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        # ---- สถานะที่ UI อ่าน ----
        self.link = LINK_DISABLED if not self.enabled else LINK_OFFLINE
        self.error = None
        self.last_hb = None
        self.last_hb_at = 0.0
        self.boot_id = None
        self.reboots = 0
        self.counters = dict(rx_frames=0, rx_lines=0, dropped_oversize=0,
                             dropped_parse=0, dropped_version=0,
                             dropped_field=0, dup_events=0, events=0,
                             cmds_nacked=0, state_sent=0, state_failed=0,
                             heartbeats=0, hb_gap_max_s=0.0,
                             link_drops=0, port_reopens=0)
        self.raw = raw                      # RawCapture หรือ None
        self.last_frame_at = 0.0            # เฟรมที่ "ใช้ได้" ล่าสุด
        self.proto_ver_seen = None
        self.display_mask_prev = None
        self.started_at = time.time()

    # ------------------------------------------------------------------
    def start(self):
        if not self.enabled:
            return False
        self._thread = threading.Thread(target=self._guard, name="p4-bridge",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout=3.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if self.transport:
            self.transport.close()

    def _guard(self):
        """ห่อ run() ไว้ — bridge ตายต้องไม่ลากอย่างอื่นตายไปด้วย"""
        try:
            self._run()
        except Exception as exc:
            with self._lock:
                self.link = LINK_ERROR
                self.error = "{}: {}".format(type(exc).__name__, exc)

    # ------------------------------------------------------------------
    def _run(self):
        next_state = 0.0
        while not self._stop.is_set():
            line = self.transport.read_line(0.2) if self.transport else None
            if line is not None:
                self.counters["rx_lines"] += 1
                self._handle(line)

            now = time.time()
            if self.last_hb_at and (now - self.last_hb_at) > self.offline_after:
                with self._lock:
                    if self.link == LINK_ONLINE:
                        self.link = LINK_OFFLINE
                        self.counters["link_drops"] += 1
                        self.mask.on_p4_silent()

            if now >= next_state:
                next_state = now + self.state_interval
                self._send_state()

            if line is None:
                time.sleep(0.02)

    def _send_state(self):
        try:
            frame, _ = self.state_source.build_frame()
            data = P.dumps_line(frame)
            if data and self.raw is not None:
                self.raw.add("tx", data)
            if data and self.transport and self.transport.write(data):
                self.counters["state_sent"] += 1
            else:
                self.counters["state_failed"] += 1
        except Exception:
            self.counters["state_failed"] += 1

    # ------------------------------------------------------------------
    def _handle(self, line):
        if self.raw is not None:
            self.raw.add("rx", line)
        res = P.parse_line(line)
        if not res.ok:
            r = res.reason or ""
            if r == P.R_OVERSIZE:
                self.counters["dropped_oversize"] += 1
            elif r == P.R_BAD_VERSION:
                self.counters["dropped_version"] += 1
            elif r.startswith("MISSING_FIELD") or r.startswith("BAD_FIELD"):
                self.counters["dropped_field"] += 1
            elif r in (P.R_EMPTY, P.R_UNKNOWN_TYPE):
                pass                       # ไม่ใช่ความผิดพลาด
            else:
                self.counters["dropped_parse"] += 1
            return

        self.counters["rx_frames"] += 1
        self.last_frame_at = time.time()
        self.proto_ver_seen = P.PROTO_VER
        fr = res.frame
        if isinstance(fr, P.Heartbeat):
            self._on_hb(fr)
        elif isinstance(fr, P.ReadingSaved):
            self._on_reading(fr)
        elif isinstance(fr, P.ContextEvent):
            self._on_context(fr)
        elif isinstance(fr, P.Command):
            self._on_cmd(fr)

    def _on_hb(self, hb):
        with self._lock:
            now = time.time()
            if self.last_hb_at:
                gap = now - self.last_hb_at
                if gap > self.counters["hb_gap_max_s"]:
                    self.counters["hb_gap_max_s"] = round(gap, 2)
            self.counters["heartbeats"] += 1
            if hb.display_mask is not None and hb.display_mask != 0xFF:
                cur = self.mask.p4_mask
                if cur is not None and cur != (hb.display_mask & 0x0F):
                    self.display_mask_prev = cur
            if self.boot_id is not None and hb.boot_id != self.boot_id:
                self.reboots += 1
                self.log.append_local("P4_REBOOT", boot_id=hb.boot_id,
                                      previous_boot_id=self.boot_id)
            self.boot_id = hb.boot_id
            self.last_hb = hb
            self.last_hb_at = time.time()
            self.link = LINK_ONLINE
            self.error = None
            ev = self.mask.on_heartbeat(hb, time.time())

        if ev.kind in ("INITIAL", "CHANGED"):
            self.log.append_local(
                "DISPLAY_MASK_" + ev.kind, boot_id=hb.boot_id,
                **{"from": ev.old, "to": ev.new})
            self._notify("mask", ev)
        elif ev.kind in ("REJECTED", "OUT_OF_RANGE"):
            self.log.append_local("DISPLAY_MASK_" + ev.kind, boot_id=hb.boot_id,
                                  raw=hb.display_mask, detail=ev.detail)
        self._notify("hb", hb)

    def _on_reading(self, fr):
        rec = P.reading_saved_record(fr)
        extra = self._session_link()
        if self.log.append(rec, extra):
            self.counters["events"] += 1
            self._notify("reading_saved", fr)
        else:
            self.counters["dup_events"] += 1

    def _on_context(self, fr):
        rec = P.context_event_record(fr)
        if self.log.append(rec, self._session_link()):
            self.counters["events"] += 1
            self._notify("event", fr)
        else:
            self.counters["dup_events"] += 1

    def _session_link(self):
        """ผูก event เข้ากับ session ที่กำลังเปิดอยู่ ถ้ามี

        ⚠️ ห้ามสร้าง session ให้อัตโนมัติ  ถ้าไม่มี session เปิดอยู่
           ให้ติดธง unassigned_session ไว้แล้วให้คนตัดสินทีหลัง
        """
        try:
            s = self.state_source.snapshot()
        except Exception:
            return {"unassigned_session": True}
        if s.get("session_mask"):
            return {"sample_id": s.get("sample_id") or "",
                    "session_mask": s.get("session_mask")}
        return {"unassigned_session": True}

    def _on_cmd(self, cmd):
        """P1 ไม่ทำคำสั่งใด ๆ — แต่ต้องตอบ ไม่ใช่เงียบ

        ⚠️ ถ้าเงียบ จอจะรอจน CMD_TIMEOUT_MS (5 วินาที) แล้วขึ้นว่า
           "ไม่ทราบผล" ซึ่งชวนให้ผู้ใช้กดซ้ำ
           การตอบ NACK ทันทีบอกชัดว่า "ระบบยังไม่เปิดให้ทำ" ไม่ใช่ "อาจทำไปแล้ว"
        """
        self.counters["cmds_nacked"] += 1
        self.log.append_local("CMD_REJECTED", request_id=cmd.request_id,
                              action=cmd.action, sensor=cmd.sensor,
                              code="COMMANDS_DISABLED")
        nack = P.build_nack(cmd.request_id, cmd.action, "COMMANDS_DISABLED",
                            "PC ยังไม่เปิดรับคำสั่งในเฟสนี้")
        data = P.dumps_line(nack)
        if data and self.transport:
            self.transport.write(data)
        self._notify("cmd_rejected", cmd)

    def _notify(self, kind, obj):
        if self.on_event:
            try:
                self.on_event(kind, obj)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def snapshot(self):
        """สถานะทั้งหมดที่ UI ต้องใช้ — คัดลอกออกมาใต้ล็อก"""
        with self._lock:
            hb = self.last_hb
            age = (time.time() - self.last_hb_at) if self.last_hb_at else None
            tr = self.transport
            last_frame_age = ((time.time() - self.last_frame_at)
                              if self.last_frame_at else None)
            return {
                "link": self.link,
                "link_text": LINK_TEXT[self.link],
                "error": self.error,
                "hb_age_s": age,
                "last_frame_age_s": last_frame_age,
                "proto_ver_seen": self.proto_ver_seen,
                "display_mask_prev": self.display_mask_prev,
                "uptime_s": time.time() - self.started_at,
                "port": getattr(tr, "device", None),
                "port_reason": getattr(tr, "last_reason", None),
                "port_reopens": getattr(tr, "reconnects", 0),
                "raw_capture": (self.raw.snapshot() if self.raw else None),
                "boot_id": self.boot_id,
                "reboots": self.reboots,
                "queued": hb.queued if hb else None,
                "heap": hb.heap if hb else None,
                "heap_big": hb.heap_big if hb else None,
                "rs485_round": hb.rs485_round if hb else None,
                "p4_sees_pc": hb.link if hb else None,
                "display_mask": self.mask.p4_mask,
                "view_mask": self.mask.effective(),
                "mask_state": self.mask.state,
                "mask_text": self.mask.ui_text(),
                "counters": dict(self.counters),
            }
