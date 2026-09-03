#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 app.py — ตัวโปรแกรมฝั่ง PC (P1-B)  ผู้อ่านอย่างเดียว
============================================================================
   python -m ecstation.app                 ต่อจอจริง (USB-Serial-JTAG)
   python -m ecstation.app --mock          รอ tools/mock_p4.py ต่อเข้ามา
   python -m ecstation.app --no-bridge     ดูข้อมูลอย่างเดียว ไม่แตะพอร์ต

 ⚠️ โปรแกรมนี้ไม่เขียนอะไรลงโฟลเดอร์ของระบบเดิมเลย
    เขียนได้แค่ data/events/*.jsonl และ data/ui_state.json ของตัวเอง
============================================================================
"""

import argparse
import os
import queue
import sys
import tkinter as tk

if __name__ == "__main__" and __package__ is None:      # รันไฟล์ตรง ๆ ได้ด้วย
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "ecstation"

from .bridge import p4_bridge as B                      # noqa: E402
from .bridge import protocol as P                       # noqa: E402
from .bridge.event_log import EventLog                  # noqa: E402
from .bridge.pc_state import PcStateSource              # noqa: E402
from .bridge.raw_capture import RawCapture              # noqa: E402
from .diag import snapshot as DIAG                      # noqa: E402
from .core import config as CFG                         # noqa: E402
from .core import ports as PORTS                        # noqa: E402
from .core.csv_source import CsvTail                    # noqa: E402
from .ui import lab_theme as T                          # noqa: E402
from .ui.dashboard import Dashboard                     # noqa: E402
from .ui.view_model import DashboardModel               # noqa: E402

REFRESH_MS = 1000


def banner(cfg, mode):
    print("=" * 70)
    print(" EC MEASUREMENT STATION — PC viewer  [{}]".format(T.UI_VERSION))
    print("=" * 70)
    print("  ไฟล์ที่รันอยู่ : {}".format(os.path.abspath(__file__)))
    print("  ธีม           : {}".format(os.path.abspath(T.__file__)))
    print("  bridge        : {}".format(mode))
    print("  data_dir      : {}".format(cfg["data_dir"]))
    print("  legacy (อ่าน) : {}".format(cfg["legacy"].get("root") or "— ไม่ได้ตั้ง —"))
    print("-" * 70)
    print("  โปรแกรมนี้ไม่เขียนอะไรลงโฟลเดอร์ของระบบเดิม")
    print("=" * 70)


class App(object):
    def __init__(self, cfg, mode="serial", port=None, raw_capture=False):
        self.cfg = cfg
        self.mode = mode
        self.port = port
        lg = cfg.get("legacy", {})
        self.csv = CsvTail(lg.get("data_dir") or "")
        self.event_log = EventLog(cfg["data_dir"])
        self.state_source = PcStateSource(cfg)
        self.model = DashboardModel(T.load_ui_config())
        self._q = queue.Queue()
        self._pc = {}
        self.bridge = None

        b = dict(cfg.get("bridge", {}))
        self.raw = RawCapture(
            cfg["data_dir"],
            enabled=bool(raw_capture or b.get("raw_capture")),
            max_bytes=int(b.get("raw_capture_max_mb", 32)) * 1024 * 1024)
        if mode != "off":
            b["enabled"] = True
            c2 = dict(cfg); c2["bridge"] = b
            tr = (B.SocketTransport("127.0.0.1", 8781) if mode == "mock"
                  else B.SerialTransport(port=port))
            self.bridge = B.P4Bridge(c2, self.event_log, self.state_source,
                                     transport=tr, on_event=self._on_bridge,
                                     raw=self.raw)
            self.bridge.start()

    # ------------------------------------------------------------------
    def _on_bridge(self, kind, obj):
        """เรียกจากเธรด bridge — ห้ามแตะวิดเจ็ตที่นี่ ส่งเข้าคิวอย่างเดียว"""
        self._q.put((kind, obj))

    def drain(self):
        n = 0
        while True:
            try:
                kind, obj = self._q.get_nowait()
            except queue.Empty:
                break
            n += 1
            f = self.model.events
            if kind == "reading_saved":
                f.add_reading_saved(obj)
            elif kind == "event":
                f.add_context(obj)
            elif kind == "cmd_rejected":
                f.add("CMD_REJECTED", sensor=obj.sensor, source="pc",
                      detail=obj.action)
            elif kind == "mask" and obj.kind in ("INITIAL", "CHANGED"):
                f.add("DISPLAY_MASK_" + obj.kind, source="pc",
                      detail="%s → %s" % (obj.old, obj.new))
        return n

    # ------------------------------------------------------------------
    def bridge_snapshot(self):
        if self.bridge is None:
            return {"link": "DISABLED", "link_text": "P4 BRIDGE DISABLED",
                    "counters": {}, "display_mask": None, "view_mask": None}
        return self.bridge.snapshot()

    def pc_snapshot(self):
        return self._pc

    def legacy_reports_dir(self):
        return (self.cfg.get("legacy") or {}).get("reports_dir") or ""

    def save_ui_state(self):
        T.save_ui_config({"window": self.model.window_minutes,
                          "chart_mode": self.model.chart_mode,
                          "ec_decimals": self.model.ec_decimals})

    def poll(self):
        self.csv.poll()
        try:
            self._pc = self.state_source.snapshot()
        except Exception as e:                       # noqa
            self._pc = {"liveness": "OFFLINE",
                        "liveness_text": "PC LOGGER OFFLINE",
                        "error": str(e)}
        self.drain()

    def export_diag(self, note=None):
        """เขียน snapshot วินิจฉัยลง data/diag/ — ปลอดภัยที่จะส่งต่อ"""
        payload = DIAG.build(self.bridge_snapshot(), self.pc_snapshot(),
                             csv=self.csv, feed=self.model.events,
                             raw=self.raw, ports=PORTS.describe(),
                             extra=note)
        return DIAG.export(self.cfg["data_dir"], payload)

    def shutdown(self):
        if self.bridge:
            self.bridge.stop(2.0)
        self.raw.close()
        self.event_log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="EC Measurement Station — viewer")
    ap.add_argument("--mock", action="store_true", help="รอจอจำลองทาง TCP")
    ap.add_argument("--no-bridge", action="store_true",
                    help="ไม่เปิด bridge เลย — ไม่แตะพอร์ตใด ๆ")
    ap.add_argument("--bridge-port", default=None, metavar="COMn",
                    help="ระบุพอร์ต NDJSON ของจอเอง (ข้ามการค้นหาอัตโนมัติ)")
    ap.add_argument("--raw-capture", action="store_true",
                    help="จับ NDJSON ดิบลง data/raw/ (สำหรับวินิจฉัยเท่านั้น)")
    ap.add_argument("--diag-export", action="store_true",
                    help="export snapshot วินิจฉัยตอนปิดโปรแกรม")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    cfg = CFG.load(a.config)
    mode = "off" if a.no_bridge else ("mock" if a.mock else "serial")
    banner(cfg, mode)
    if mode == "serial":
        au = PORTS.audit()
        r = au["roles"][PORTS.ROLE_P4_BRIDGE]
        print("  พอร์ตจอ      : {}".format(
            a.bridge_port or r["device"] or "— ({}) —".format(r["reason"])))
        if not a.bridge_port and r["reason"] == PORTS.PICK_AMBIGUOUS:
            print("  ⚠️ เจอ 303A:1001 หลายตัว: {} — ใช้ --bridge-port ระบุเอง"
                  .format(", ".join(c["device"] for c in r["candidates"])))
        print("-" * 70)

    T.enable_dpi_awareness()             # ต้องก่อน tk.Tk() เสมอ
    T.apply_mpl_rc()
    root = tk.Tk()
    T.init_scaling(root)
    T.apply_ttk_theme(root)
    root.title("EC Measurement Station — %s" % T.UI_VERSION)
    root.geometry("1280x820")
    root.minsize(960, 640)
    root.configure(bg=T.BG)

    app = App(cfg, mode, port=a.bridge_port, raw_capture=a.raw_capture)
    ui = Dashboard(root, app)
    ui.pack(fill="both", expand=True)

    def tick():
        try:
            app.poll()
            ui.refresh()
        except Exception as e:           # noqa — UI ตายห้ามลาก bridge ตามไป
            print("[ui] รอบวาดล้มเหลว:", type(e).__name__, e)
        root.after(REFRESH_MS, tick)

    def on_close():
        if a.diag_export:
            try:
                print("[diag] เขียน snapshot:", app.export_diag("ปิดโปรแกรม"))
            except Exception as e:                       # noqa
                print("[diag] export ไม่ได้:", e)
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(200, tick)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
