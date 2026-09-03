#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bridge.py — เปิด bridge แบบไม่มีหน้าจอ

  โหมด mock  : รอ tools/mock_p4.py ต่อเข้ามาทาง TCP 127.0.0.1:8781
  โหมด serial: หาพอร์ต USB-Serial-JTAG ของจอเอง (303A:1001)

ใช้ตรวจว่า bridge ทำงานถูกก่อนจะมี UI  ไม่ใช่ตัวโปรแกรมหลัก
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecstation.bridge import p4_bridge as B          # noqa: E402
from ecstation.bridge.event_log import EventLog      # noqa: E402
from ecstation.bridge.pc_state import PcStateSource  # noqa: E402
from ecstation.core import config as CFG             # noqa: E402


def make_transport(mode, host, port):
    if mode == "mock":
        return B.SocketTransport(host, port)
    return B.SerialTransport()


def main():
    ap = argparse.ArgumentParser(description="เปิด bridge แบบไม่มีหน้าจอ")
    ap.add_argument("--mode", choices=("mock", "serial"), default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = ไม่จำกัด")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = CFG.load(a.config)
    log = EventLog(cfg["data_dir"])
    state = PcStateSource(cfg)
    br = B.P4Bridge(cfg, log, state, transport=make_transport(a.mode, a.host,
                                                              a.port))
    print("[bridge] โหมด %s · data_dir=%s" % (a.mode, cfg["data_dir"]))
    br.start()
    end = (time.time() + a.seconds) if a.seconds else None
    try:
        while end is None or time.time() < end:
            time.sleep(1.0)
            s = br.snapshot()
            print("[bridge] %-18s mask=%s ev=%d dup=%d bad=%d nack=%d state=%d"
                  % (s["link_text"], s["display_mask"],
                     s["counters"]["events"], s["counters"]["dup_events"],
                     s["counters"]["dropped_parse"]
                     + s["counters"]["dropped_field"]
                     + s["counters"]["dropped_version"]
                     + s["counters"]["dropped_oversize"],
                     s["counters"]["cmds_nacked"], s["counters"]["state_sent"]))
    except KeyboardInterrupt:
        pass
    finally:
        br.stop(2.0)
        log.close()
    print("[bridge] จบ")


if __name__ == "__main__":
    main()
