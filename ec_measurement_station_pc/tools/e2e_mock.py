#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2e_mock.py — รันทั้ง 13 ฉากของ mock_p4 ผ่าน bridge จริง แล้วตรวจผล

ต่างจากเทสต์ยูนิต: ชุดนี้เดินผ่าน socket จริง, เธรดจริง, ไฟล์จริง
สิ่งที่ยูนิตเทสต์จับไม่ได้และชุดนี้จับได้คือ การประกอบร่างผิด — เช่นเธรดค้าง,
บรรทัดถูกตัดครึ่งตอนอ่านจาก socket, หรือไฟล์ event ไม่ถูก flush

ใช้: python tools/e2e_mock.py [--seconds 2.0] [--only normal,flood]
"""
import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from ecstation.bridge import p4_bridge as B          # noqa: E402
from ecstation.bridge.event_log import EventLog      # noqa: E402
import mock_p4 as MP                                 # noqa: E402


class FakeState(object):
    def __init__(self):
        self.seq = 0

    def snapshot(self):
        return {"session_mask": 0b0110, "sample_id": "CALF-20 B3",
                "recording": True, "csv_rows": 42, "active_mask": 0b0111,
                "active_mask_assumed": True, "cal_busy": False}

    def build_frame(self):
        from ecstation.bridge import protocol as P
        self.seq += 1
        return P.build_state(self.seq, True, True, 0b0110, 0b0111, False, 42,
                             "CALF-20 B3"), {}


# ---- สิ่งที่แต่ละฉากต้องทำให้เกิด (ไม่ใช่แค่ "รันแล้วไม่ error") ----
def _has(c, k, n=1):
    return c[k] >= n


EXPECT = {
    "normal":       lambda s, c, ev: c["events"] > 0 and s["link"] == "ONLINE",
    "mask-change":  lambda s, c, ev: any(e.get("event", "").startswith("DISPLAY_MASK_CHANGED") for e in ev),
    "mask-boot":    lambda s, c, ev: any(e.get("event") == "DISPLAY_MASK_INITIAL" for e in ev),
    "mask-invalid": lambda s, c, ev: any(e.get("event", "").startswith("DISPLAY_MASK_") and e.get("event") in ("DISPLAY_MASK_OUT_OF_RANGE", "DISPLAY_MASK_REJECTED") for e in ev),
    "reboot":       lambda s, c, ev: s["reboots"] >= 1 and any(e.get("event") == "P4_REBOOT" for e in ev),
    # ฉากนี้ส่งแต่ hb ไม่มี reading — ตัวชี้วัดคือ "ตกแล้วกลับมาได้เอง"
    #   (ตรวจลำดับจริงใน run_one ไม่ใช่ที่นี่ เพราะต้องดูลำดับก่อนหลัง)
    "disconnect":   lambda s, c, ev: s["link"] == "ONLINE",
    "dup":          lambda s, c, ev: c["dup_events"] > 0,
    # ฉากนี้ตั้งใจไม่มีเฟรมดีเลยสักอัน (สองบรรทัดที่หน้าตาเหมือน event
    # ถูกออกแบบให้กำกวม: อันหนึ่งใช้คีย์ "ec" อีกอันมี stable_ec_us_cm
    # ขัดกับ ec_us_cm)  ตัวชี้วัดจึงเป็น "ทิ้งขยะได้ และ hb ถัดไปยังรับได้"
    "malformed":    lambda s, c, ev: (c["dropped_parse"] + c["dropped_field"] + c["dropped_version"] + c["dropped_oversize"]) >= 4 and c["rx_frames"] >= 2 and s["link"] == "ONLINE",
    "queue-full":   lambda s, c, ev: s["queued"] is not None and c["rx_frames"] > 0,
    "heap-leak":    lambda s, c, ev: s["heap"] is not None and s["heap_big"] is not None,
    "cmd":          lambda s, c, ev: c["cmds_nacked"] > 0 and any(e.get("event") == "CMD_REJECTED" for e in ev),
    "flood":        lambda s, c, ev: c["rx_frames"] > 20 and s["error"] is None,
    "soak":         lambda s, c, ev: c["events"] > 0 and s["error"] is None,
}


def read_events(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def run_one(name, seconds, port, speed):
    data_dir = tempfile.mkdtemp(prefix="ec_e2e_%s_" % name)
    log = EventLog(data_dir)
    # ย่อเวลาฝั่งจอเท่าไร ต้องย่อเกณฑ์ OFFLINE ฝั่ง PC เท่านั้นด้วย
    # ไม่งั้นฉาก disconnect จะผ่านเพราะ "ยังไม่ทันหมดเวลา" ไม่ใช่เพราะถูกต้อง
    cfg = {"bridge": {"enabled": True,
                      "state_interval_s": 0.5 / speed,
                      "offline_after_s": 10.0 / speed}}
    tr = B.SocketTransport("127.0.0.1", port)
    br = B.P4Bridge(cfg, log, FakeState(), transport=tr)
    br.start()

    err = []
    seen_links = set()
    link_seq = []                    # ลำดับการเปลี่ยนสถานะ ไม่ใช่แค่เซต
    watching = threading.Event()

    def watcher():
        while not watching.is_set():
            cur = br.link
            seen_links.add(cur)
            if not link_seq or link_seq[-1] != cur:
                link_seq.append(cur)
            watching.wait(0.05)

    wt = threading.Thread(target=watcher, name="link-watch", daemon=True)
    wt.start()

    def client():
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError as e:
            err.append("ต่อไม่ติด: %s" % e)
            return
        m = MP.Mock(s)
        try:
            MP.SCENARIOS[name](m, seconds)
        except SystemExit:
            pass                     # ฉาก disconnect ตั้งใจให้ปิดกลางคัน
        except Exception as e:       # noqa
            err.append("%s: %s" % (type(e).__name__, e))
        finally:
            try:
                s.close()
            except Exception:
                pass

    t = threading.Thread(target=client, name="mock-client")
    t.start()
    t.join(seconds + 10)
    time.sleep(0.4)                  # ให้ bridge เก็บบรรทัดสุดท้าย
    watching.set()
    wt.join(1.0)
    br.stop(3.0)
    snap = br.snapshot()
    log.close()
    ev = read_events(log.path_for())

    ok = not err and snap["error"] is None
    check = EXPECT.get(name)
    passed = ok and (check is None or check(snap, snap["counters"], ev))
    if name == "disconnect":
        # ต้องเป็นลำดับ ONLINE -> OFFLINE -> ONLINE จริง ๆ
        # "เคยเห็น OFFLINE" ไม่พอ เพราะตอนเพิ่งสตาร์ตก็เป็น OFFLINE อยู่แล้ว
        # ถ้าเช็กแค่นั้น เทสต์จะผ่านแม้ bridge ไม่เคยตรวจจับการหลุดเลย
        try:
            i_on = link_seq.index("ONLINE")
            i_off = link_seq.index("OFFLINE", i_on)
            recovered = "ONLINE" in link_seq[i_off:]
        except ValueError:
            recovered = False
        passed = passed and recovered
    return {
        "scenario": name, "ok": passed, "errors": err,
        "link": snap["link"], "bridge_error": snap["error"],
        "display_mask": snap["display_mask"], "reboots": snap["reboots"],
        "events_file": len(ev), "links_seen": ">".join(link_seq) or "-",
        **snap["counters"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--only", default="")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--speed", type=float, default=8.0,
                    help="ย่อเวลารอของฉากลงกี่เท่า")
    a = ap.parse_args()
    MP.TIME_SCALE = max(1.0, a.speed)

    names = ([x.strip() for x in a.only.split(",") if x.strip()]
             or list(MP.SCENARIOS))
    rows, bad = [], 0
    devnull = open(os.devnull, "w")
    real_out = sys.stdout
    for i, n in enumerate(names):
        sys.stdout = devnull                 # ปิดเสียง mock ที่พิมพ์ทุกบรรทัด
        try:
            r = run_one(n, a.seconds, a.port + i, max(1.0, a.speed))
        finally:
            sys.stdout = real_out
        rows.append(r)
        if not r["ok"]:
            bad += 1
        print("%-14s %s  link=%-30s ev=%-5d dup=%-4d bad=%-6d nack=%-3d state=%-4d %s"
              % (n, "ผ่าน" if r["ok"] else "ตก", r["links_seen"], r["events"],
                 r["dup_events"],
                 r["dropped_parse"] + r["dropped_field"]
                 + r["dropped_version"] + r["dropped_oversize"],
                 r["cmds_nacked"], r["state_sent"],
                 ("  << " + "; ".join(r["errors"])) if r["errors"] else ""))
    devnull.close()
    print("\n%d/%d ฉากผ่าน" % (len(rows) - bad, len(rows)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
