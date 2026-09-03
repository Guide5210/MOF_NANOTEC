# -*- coding: utf-8 -*-
"""
test_logger_unaffected — bridge ต้องไม่ไปรบกวนจังหวะการเก็บข้อมูลของ logger

เกณฑ์ที่ผู้ใช้กำหนด: ที่จังหวะ CONTROL 2.5 วินาที ช่องว่างระหว่างแถว CSV
ต้องไม่เกิน 3.0 วินาที  พร้อมรายงาน จำนวนแถว · seq ซ้ำ/ขาด · ค่ากลาง/p95/
แย่สุดของช่วงเวลา · เฟรมพังที่เจอ · exception · อัตรา event

สามฉากที่วัด
  1. baseline   — ไม่มี bridge เลย  (ถ้าไม่มีเส้นฐาน ตัวเลขอื่นไม่มีความหมาย)
  2. heartbeat  — bridge ทำงานปกติ รับ hb + reading_saved ตามจังหวะจริง
  3. flood      — ป้อนบรรทัดพังรัว ๆ ไม่หยุด  (เคสที่แย่ที่สุดที่คิดออก)

⚠️ ขอบเขตของเทสต์นี้ — ต้องพูดให้ชัด ไม่งั้นมันจะถูกอ้างเกินจริง
   เทสต์นี้วัด "การแย่ง GIL/CPU ในโปรเซสเดียวกัน" ซึ่งเป็นความเสี่ยงจริง
   ของสถาปัตยกรรมนี้  แต่มัน **ไม่ได้** วัดสายจริง: บน UART ที่ไม่มี
   flow control ถ้าลูปอ่านช้า ไบต์จะถูกทิ้ง ไม่ใช่แค่มาช้า
   การยืนยันเรื่องนั้นต้องรันกับฮาร์ดแวร์จริง

⚠️ "seq ซ้ำ/ขาด" ที่รายงานคือ seq ของแถวใน CSV ที่ฝั่ง PC เป็นคนใส่เอง
   เฟรม DATA, ของ CONTROL ปัจจุบัน **ไม่มี seq** จึงตรวจของจริงไม่ได้
   จนกว่าจะมีเฟรม ECV2 — ห้ามรายงานตัวเลขนี้ราวกับว่าครอบคลุมสายจริง
"""
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime

from _helpers import ROOT, tmp_cfg  # noqa
from ecstation.bridge import p4_bridge as B
from ecstation.bridge.event_log import EventLog

POLL_S = 2.5                 # จังหวะ CONTROL จริง
MAX_GAP_S = 3.0              # เกณฑ์ที่ต้องผ่าน
ROWS = int(os.environ.get("EC_LOGGER_ROWS", "6"))
ROWS_BASE = int(os.environ.get("EC_LOGGER_ROWS_BASE", "4"))

REPORTS = []


class FakeControlLogger(object):
    """จำลองลูปเก็บข้อมูลของ logger เดิม — เขียน CSV ทุก POLL_S วินาที

    ใช้กำหนดเวลาแบบ absolute deadline เหมือนของจริง ไม่ใช่ sleep(2.5) ต่อรอบ
    เพราะ sleep แบบหลังจะกลบความล่าช้าที่เราต้องการวัดพอดี
    """

    def __init__(self, path, rows):
        self.path = path
        self.rows = rows
        self.stamps = []
        self.seqs = []
        self.errors = []
        self._t = threading.Thread(target=self._run, name="fake-logger")

    def start(self):
        self._t.start()

    def join(self):
        self._t.join()

    def _run(self):
        t0 = time.monotonic()
        try:
            with open(self.path, "w", encoding="utf-8", buffering=1) as fh:
                fh.write("seq,timestamp,sensor,ec,temp\n")
                for i in range(self.rows):
                    target = t0 + i * POLL_S
                    now = time.monotonic()
                    if target > now:
                        time.sleep(target - now)
                    fh.write("%d,%s,1,1146.0,20.6\n"
                             % (i, datetime.now().isoformat()))
                    fh.flush()
                    os.fsync(fh.fileno())
                    self.stamps.append(time.monotonic())
                    self.seqs.append(i)
        except Exception as exc:            # noqa
            self.errors.append("%s: %s" % (type(exc).__name__, exc))


def gaps(stamps):
    return [b - a for a, b in zip(stamps, stamps[1:])]


def p95(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[k]


def csv_row_check(path):
    """นับแถวจริงและตรวจ seq ที่ PC ใส่เอง (ไม่ใช่ seq จากสาย — ยังไม่มี)"""
    seqs = []
    with open(path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            line = line.strip()
            if line:
                seqs.append(int(line.split(",", 1)[0]))
    dup = len(seqs) - len(set(seqs))
    missing = 0
    if seqs:
        missing = (max(seqs) - min(seqs) + 1) - len(set(seqs))
    return {"rows": len(seqs), "dup_seq": dup, "missing_seq": missing}


class LoggerCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_loggertest_")
        self.csv = os.path.join(self.d, "water_log.csv")
        self.exceptions = []

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    # --------------------------------------------------------------
    def _record(self, name, lg, br, extra=None):
        g = gaps(lg.stamps)
        rep = {
            "scenario": name,
            "poll_s": POLL_S,
            **csv_row_check(self.csv),
            "seq_source": "PC-generated (สาย CONTROL ยังไม่มี seq — ต้องรอ ECV2)",
            "median_gap_s": round(statistics.median(g), 3) if g else None,
            "p95_gap_s": round(p95(g), 3) if g else None,
            "max_gap_s": round(max(g), 3) if g else None,
            "late_rows": sum(1 for x in g if x > MAX_GAP_S),
            "logger_exceptions": lg.errors,
            "process_exceptions": self.exceptions,
        }
        if br is not None:
            c = br.snapshot()["counters"]
            span = (lg.stamps[-1] - lg.stamps[0]) if len(lg.stamps) > 1 else 1.0
            rep.update({
                "bridge_link": br.snapshot()["link"],
                "bridge_error": br.snapshot()["error"],
                "rx_lines": c["rx_lines"],
                "rx_frames": c["rx_frames"],
                "malformed_frames": (c["dropped_parse"] + c["dropped_oversize"]
                                     + c["dropped_version"] + c["dropped_field"]),
                "events": c["events"],
                "event_rate_per_s": round(c["events"] / span, 3),
                "line_rate_per_s": round(c["rx_lines"] / span, 1),
            })
        rep.update(extra or {})
        REPORTS.append(rep)
        return rep

    def _assert_healthy(self, rep):
        self.assertEqual(rep["rows"], rep["expected_rows"])
        self.assertEqual(rep["dup_seq"], 0)
        self.assertEqual(rep["missing_seq"], 0)
        self.assertEqual(rep["logger_exceptions"], [])
        self.assertEqual(rep["process_exceptions"], [])
        self.assertLessEqual(rep["max_gap_s"], MAX_GAP_S,
                             "ช่องว่างสูงสุด %.3f s เกินเกณฑ์ %.1f s"
                             % (rep["max_gap_s"], MAX_GAP_S))

    # --------------------------------------------------------------
    def test_1_baseline_without_bridge(self):
        """เส้นฐาน — ถ้าเส้นฐานเองยังไม่ผ่าน ตัวเลขฉากอื่นก็ไม่มีความหมาย"""
        lg = FakeControlLogger(self.csv, ROWS_BASE)
        lg.start()
        lg.join()
        rep = self._record("baseline (ไม่มี bridge)", lg, None,
                           {"expected_rows": ROWS_BASE})
        self._assert_healthy(rep)

    def test_2_normal_heartbeat_traffic(self):
        """จราจรปกติ: hb ทุก 1 วินาที + reading_saved เป็นระยะ"""
        log = EventLog(self.d)
        tr = B.LoopbackTransport()
        cfg = tmp_cfg(self.d)
        cfg["bridge"]["state_interval_s"] = 1.0
        br = B.P4Bridge(cfg, log, _State(), transport=tr)

        stop = threading.Event()

        def producer():
            i = 0
            try:
                while not stop.is_set():
                    tr.feed(_hb(ts=i * 1000))
                    if i % 3 == 0:
                        tr.feed(_saved("p4-aaaa-%06d" % i, ts=i * 1000))
                    i += 1
                    stop.wait(1.0)
            except Exception as exc:              # noqa
                self.exceptions.append("producer %s: %s"
                                       % (type(exc).__name__, exc))

        pt = threading.Thread(target=producer, name="p4-producer")
        br.start()
        pt.start()
        lg = FakeControlLogger(self.csv, ROWS)
        lg.start()
        lg.join()
        stop.set()
        pt.join()
        br.stop(2.0)
        log.close()

        rep = self._record("heartbeat ปกติ", lg, br, {"expected_rows": ROWS})
        self._assert_healthy(rep)
        self.assertEqual(rep["bridge_link"], B.LINK_ONLINE)
        self.assertIsNone(rep["bridge_error"])
        self.assertGreater(rep["events"], 0)
        self.assertEqual(rep["malformed_frames"], 0)

    def test_3_malformed_flood(self):
        """ป้อนบรรทัดพังรัวไม่หยุด — bridge ต้องกลืนไว้เองไม่ลามไปหา logger"""
        log = EventLog(self.d)
        tr = B.LoopbackTransport()
        cfg = tmp_cfg(self.d)
        cfg["bridge"]["state_interval_s"] = 0.2      # ยิง state ถี่ ๆ ซ้ำเข้าไป
        br = B.P4Bridge(cfg, log, _State(), transport=tr)

        stop = threading.Event()

        def flooder():
            i = 0
            try:
                while not stop.is_set():
                    if len(tr.inbox) < 5000:         # กันหน่วยความจำบวม
                        for k in range(200):
                            n = i + k
                            if n % 7 == 0:
                                tr.feed(b"x" * 900)                  # ยาวเกิน
                            elif n % 7 == 1:
                                tr.feed(u"{พัง".encode("utf-8"))     # ไม่ใช่ json
                            elif n % 7 == 2:
                                tr.feed(b"[1,2,3]")                  # ไม่ใช่ object
                            elif n % 7 == 3:
                                tr.feed(json.dumps(
                                    {"v": 9, "type": "hb"}).encode())
                            elif n % 7 == 4:
                                tr.feed(json.dumps(
                                    {"v": 1, "type": "hb"}).encode())
                            elif n % 7 == 5:
                                tr.feed(_hb(ts=n))                   # ของดีปนมา
                            else:
                                tr.feed(_saved("p4-aaaa-%06d" % n, ts=n))
                        i += 200
                    stop.wait(0.005)
            except Exception as exc:              # noqa
                self.exceptions.append("flooder %s: %s"
                                       % (type(exc).__name__, exc))

        ft = threading.Thread(target=flooder, name="p4-flood")
        br.start()
        ft.start()
        lg = FakeControlLogger(self.csv, ROWS)
        lg.start()
        lg.join()
        stop.set()
        ft.join()
        br.stop(2.0)
        log.close()

        rep = self._record("flood บรรทัดพัง", lg, br, {"expected_rows": ROWS})
        self._assert_healthy(rep)
        self.assertGreater(rep["malformed_frames"], 100,
                           "flood ต้องหนักจริง ไม่งั้นเทสต์ไม่ได้พิสูจน์อะไร")
        self.assertGreater(rep["events"], 0, "ของดีที่ปนมาต้องไม่หายไปกับขยะ")
        self.assertEqual(rep["bridge_link"], B.LINK_ONLINE)
        self.assertIsNone(rep["bridge_error"])


# ---------------------------------------------------------------- ตัวช่วย
class _State(object):
    def snapshot(self):
        return {"session_mask": 0b0110, "sample_id": "CALF-20 B3",
                "recording": True, "csv_rows": 0, "active_mask": 0b0111,
                "active_mask_assumed": True, "cal_busy": False}

    def build_frame(self):
        from ecstation.bridge import protocol as P
        return P.build_state(1, True, True, 0b0110, 0b0111, False, 0,
                             "CALF-20 B3"), {}


def _hb(ts=0, boot="p4-aaaa", mask=0b0111):
    return json.dumps({"v": 1, "type": "hb", "boot_id": boot, "ts_ms": ts,
                       "queued": 0, "link": "online", "heap": 180000,
                       "heap_big": 90000, "display_mask": mask}).encode()


def _saved(eid, boot="p4-aaaa", ts=0):
    return json.dumps({"v": 1, "type": "event", "event_id": eid,
                       "boot_id": boot, "event": "reading_saved", "sensor": 2,
                       "ec_us_cm": 1146.0, "temperature_c": 20.6,
                       "tolerance_us_cm": 11.5, "stable_for_ms": 15000,
                       "after_link_error": False, "ts_ms": ts}).encode()


def tearDownModule():
    if not REPORTS:
        return
    out = ["", "=" * 74,
           " รายงานผล test_logger_unaffected  (เกณฑ์: gap สูงสุด <= %.1f s)"
           % MAX_GAP_S, "=" * 74]
    for r in REPORTS:
        out.append("")
        out.append("[%s]" % r["scenario"])
        for k in ("rows", "dup_seq", "missing_seq", "seq_source",
                  "median_gap_s", "p95_gap_s", "max_gap_s", "late_rows",
                  "rx_lines", "line_rate_per_s", "rx_frames",
                  "malformed_frames", "events", "event_rate_per_s",
                  "bridge_link", "bridge_error", "logger_exceptions",
                  "process_exceptions"):
            if k in r:
                out.append("    %-18s %s" % (k, r[k]))
    out.append("")
    out.append("หมายเหตุ: วัดการแย่ง GIL ในโปรเซสเดียวกัน ไม่ได้วัด UART จริง")
    out.append("=" * 74)
    sys.stderr.write("\n".join(out) + "\n")


if __name__ == "__main__":
    unittest.main()
