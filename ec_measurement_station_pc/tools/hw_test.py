#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 hw_test.py — ชุดทดสอบฮาร์ดแวร์จริง P1-C (matrix A-E)
============================================================================
   python tools\\hw_test.py --steps A,B,C,D          ทดสอบตามลำดับ
   python tools\\hw_test.py --steps E --soak 120     soak 120 นาที
   python tools\\hw_test.py --bridge-port COM7       ระบุพอร์ตจอเอง

 ⚠️ ปิดหน้าต่าง viewer ก่อนรัน
    พอร์ตอนุกรมหนึ่งตัวเปิดได้ทีละโปรเซส  ถ้า viewer เปิดค้างอยู่
    เครื่องมือนี้จะเปิดพอร์ตไม่ได้ แล้วรายงานว่า "ไม่พบจอ" ซึ่งไม่จริง

 ⚠️ เครื่องมือนี้ไม่แตะพอร์ตของบอร์ด CONTROL เลย
    logger เดิมต้องรันต่อไปตามปกติตลอดการทดสอบ — นั่นคือสิ่งที่กำลังพิสูจน์

 ⚠️ ไม่เขียนอะไรลงโฟลเดอร์ของระบบเดิม  ผลทั้งหมดลง data/diag/
============================================================================
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ecstation.bridge import p4_bridge as B          # noqa: E402
from ecstation.bridge.event_log import EventLog      # noqa: E402
from ecstation.bridge.pc_state import PcStateSource  # noqa: E402
from ecstation.bridge.raw_capture import RawCapture  # noqa: E402
from ecstation.core import config as CFG             # noqa: E402
from ecstation.core import ports as PORTS            # noqa: E402
from ecstation.core.csv_source import CsvTail        # noqa: E402
from ecstation.diag import snapshot as DIAG          # noqa: E402

# ============================================================================
#  เกณฑ์ผ่าน/ไม่ผ่าน — ตั้งไว้ก่อนเริ่ม ไม่ใช่ปรับตามผลที่ได้
# ============================================================================
# ⚠️ ทำไมไม่มีเกณฑ์ p95 ของช่องว่าง CSV
#    คอลัมน์ timestamp ของ logger เดิมมีความละเอียด 1 วินาที  ที่จังหวะ 2.5 s
#    ช่องว่างที่บันทึกได้จะสลับ 2 s / 3 s เสมอ ทำให้ p95 = 3.0 โดยธรรมชาติ
#    การตั้งเกณฑ์ p95 <= 2.8 s จึงเป็นการอ้างความละเอียดที่ข้อมูลไม่มี
#    — เทสต์ที่ตกเพราะหน่วยวัด ไม่ใช่เพราะระบบ คือเทสต์ที่คนจะเลิกเชื่อ
#    ตัวชี้วัดที่ไม่ขึ้นกับความละเอียดคือ "แถวที่หายไปกี่ %" ซึ่งใช้แทน
TH = {
    "csv_gap_max_s":        3.0,    # เกณฑ์เดิมจาก P1-A (วัดจาก timestamp ในไฟล์)
    "csv_missing_pct_max":  1.0,    # แถวที่ควรมีแต่ไม่มี เทียบจังหวะ 2.5 s
    "hb_age_max_s":        12.0,    # จอส่งทุก 5 s -> เกิน 12 s = หาย 2 ครั้งติด
    "link_drops_max":         0,    # เด้งเองโดยไม่ได้ถอดสาย = 16 s ยังไม่พอ
    "malformed_max":          0,    # USB สะอาด เฟรมพังแม้เฟรมเดียวคือสัญญาณจริง
    "stored_duplicates_max":  0,    # ซ้ำที่ "ถูกปฏิเสธ" มีได้ ที่ "ถูกเก็บ" ไม่ได้
    "queued_max":             8,    # จาก 32 — เกินนี้แปลว่า PC ตามไม่ทัน
    "heap_big_retain":     0.90,    # ชม.ท้าย >= 90% ของ ชม.แรก
    "exceptions_max":         0,
}
SAMPLE_S = 1.0
# ⚠️ รอบวัดจริงใช้เวลาเท่าไรไม่มีใครรู้ล่วงหน้า
#    จุ่มหัววัด -> ค่าไหลลง -> นิ่งครบ 6 ตัวอย่าง -> คนเดินมากด
#    90 วินาทีที่เคยตั้งไว้สั้นเกินจริงมาก  เทสต์ที่หมดเวลาก่อนคนจะกดเสร็จ
#    จะถูกรายงานว่า "ช่องทาง event เสีย" ทั้งที่ไม่มีอะไรเสียเลย
WAIT_MEASURE_S = 900.0
CSV_POLL_S = 0.25        # อ่านไฟล์ถี่กว่ารอบเก็บตัวอย่าง — เป็นการอ่านต่อยอด ถูกมาก
CONTROL_POLL_S = 2.5     # จังหวะของ CONTROL ใช้คำนวณจำนวนแถวที่ควรมี


def hb(title, char="="):
    print("\n" + char * 78); print(" " + title); print(char * 78)


UNATTENDED = False


def ask(msg):
    print("\n>>> %s" % msg)
    if UNATTENDED:
        print("    [unattended] ข้ามการรอ")
        time.sleep(1.0)
        return
    try:
        input("    ทำเสร็จแล้วกด Enter (Ctrl+C เพื่อหยุด) ... ")
    except EOFError:
        time.sleep(1.0)


def p95(xs):
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))]


# ============================================================================
#  ยามเฝ้าโฟลเดอร์ของระบบเดิม
# ============================================================================
class LegacyGuard(object):
    """ยืนยันว่าโฟลเดอร์ legacy ถูก "ต่อท้าย" เท่านั้น ไม่ถูกแก้

    ⚠️ เทียบ sha256 ทั้งไฟล์ใช้ไม่ได้กับ CSV เพราะ logger เดิมกำลังเขียนอยู่จริง
       สิ่งที่ต้องพิสูจน์คือ "ส่วนที่เคยมีไม่ถูกแก้" ไม่ใช่ "ไฟล์ไม่โต"
       จึงจำขนาดเดิมไว้แล้วเทียบ hash เฉพาะ N ไบต์แรก
       ส่วนซอร์ส/เอกสาร/คอนฟิก ต้องเท่าเดิมทุกไบต์
    """

    RUNTIME = ("rec_status.json", "cal_status.json", "sessions_3ec.json")

    def __init__(self, root):
        self.root = root or ""
        self.base = {}
        self.new_files = []
        if self.root and os.path.isdir(self.root):
            self.base = self._scan()

    def _scan(self):
        out = {}
        for b, dirs, names in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for n in sorted(names):
                p = os.path.join(b, n)
                rel = os.path.relpath(p, self.root).replace("\\", "/")
                try:
                    size = os.path.getsize(p)
                    with open(p, "rb") as fh:
                        head = fh.read(size)
                    out[rel] = (size, hashlib.sha256(head).hexdigest())
                except OSError:
                    pass
        return out

    def check(self):
        """คืน (ok, problems, summary)"""
        problems = []
        now = self._scan()
        appended = frozen = 0
        for rel, (size0, h0) in self.base.items():
            if rel not in now:
                problems.append("หายไป: %s" % rel)
                continue
            size1 = now[rel][0]
            is_data = (rel.endswith(".csv") or os.path.basename(rel) in self.RUNTIME)
            if size1 == size0 and now[rel][1] == h0:
                frozen += 1
                continue
            if not is_data:
                problems.append("ถูกแก้ (ไม่ใช่ไฟล์ข้อมูล): %s" % rel)
                continue
            if os.path.basename(rel) in self.RUNTIME:
                frozen += 1          # logger เดิมเขียนทับไฟล์สถานะของตัวเองได้
                continue
            if size1 < size0:
                problems.append("CSV สั้นลง (ถูกเขียนทับ?): %s" % rel)
                continue
            try:
                with open(os.path.join(self.root, rel), "rb") as fh:
                    head = fh.read(size0)
            except OSError as e:
                problems.append("อ่านไม่ได้: %s (%s)" % (rel, e))
                continue
            if hashlib.sha256(head).hexdigest() != h0:
                problems.append("CSV ส่วนเดิมถูกแก้: %s" % rel)
            else:
                appended += 1
        self.new_files = sorted(set(now) - set(self.base))
        for rel in self.new_files:
            if not (rel.endswith(".csv") or os.path.basename(rel) in self.RUNTIME
                    or rel.startswith("reports/")):
                problems.append("ไฟล์ใหม่ที่ไม่ควรมี: %s" % rel)
        return (not problems), problems, {
            "files_tracked": len(self.base), "unchanged": frozen,
            "appended_only": appended, "new_files": self.new_files}


# ============================================================================
#  ตัวเก็บตัวอย่าง
# ============================================================================
class Runner(object):
    def __init__(self, cfg, port=None, raw=False, mock=False):
        self.cfg = cfg
        lg = cfg.get("legacy", {})
        self.csv = CsvTail(lg.get("data_dir") or "")
        self.csv.poll()
        self.guard = LegacyGuard(lg.get("root") or "")
        self.log = EventLog(cfg["data_dir"])
        self.raw = RawCapture(cfg["data_dir"], enabled=raw)
        self.state = PcStateSource(cfg)
        self.events = []
        self.exceptions = []
        self.samples = []
        self.csv_stamps = []          # เวลาที่ "ลูปนี้เห็น" แถวใหม่ (ข้อมูลประกอบ)
        self.row_times = []           # timestamp ที่อยู่ในไฟล์ (ตัวชี้วัดหลัก)
        self._rows = len(self.csv.rows)
        self._last_row_t = (self.csv.rows[-1]["t"] if self.csv.rows else None)
        self._next_csv = 0.0

        b = dict(cfg.get("bridge", {})); b["enabled"] = True
        c2 = dict(cfg); c2["bridge"] = b
        # ⚠️ โหมด mock มีไว้ทดสอบตัวชุดทดสอบเอง ไม่ใช่ทดแทนการทดสอบฮาร์ดแวร์
        #    ชุดทดสอบที่ไม่เคยถูกทดสอบ คือชุดทดสอบที่จะพังตอนที่ต้องใช้จริง
        tr = (B.SocketTransport("127.0.0.1", 8781) if mock
              else B.SerialTransport(port=port))
        self.mock = mock
        self.bridge = B.P4Bridge(c2, self.log, self.state, transport=tr,
                                 on_event=self._on_event, raw=self.raw)
        self.bridge.start()

    def _on_event(self, kind, obj):
        try:
            self.events.append({"at": time.time(), "kind": kind,
                                "event_id": getattr(obj, "event_id", None),
                                "event": getattr(obj, "event", kind),
                                "sensor": getattr(obj, "sensor", 0),
                                "stable_ec_us_cm":
                                    getattr(obj, "stable_ec_us_cm", None)})
        except Exception as e:                                  # noqa
            self.exceptions.append("on_event %s: %s" % (type(e).__name__, e))

    def poll_csv(self):
        """อ่าน CSV ถี่ ๆ — เก็บ timestamp ที่อยู่ในไฟล์ ไม่ใช่เวลาที่เราเห็น

        ⚠️ วัดช่องว่างจากเวลาที่ลูปเราเห็นแถวใหม่ = วัดจังหวะของลูปเราเอง
           ไม่ใช่จังหวะของ logger  ตอน dry run วิธีนั้นให้ 3.002 s
           ทั้งที่ของจริงคือ 2.5 s — ตัวเลขที่ผิดแบบนี้จะถูกตีความว่า
           "ระบบใหม่ไปกวน logger" ซึ่งเป็นข้อสรุปที่ผิดและแก้ผิดจุด
        """
        now = time.time()
        if now < self._next_csv:
            return 0
        self._next_csv = now + CSV_POLL_S
        added = self.csv.poll()
        if added:
            self._rows += added
            self.csv_stamps.append(now)
            for row in list(self.csv.rows)[-added:]:
                if self._last_row_t is None or row["t"] > self._last_row_t:
                    self.row_times.append(row["t"])
                    self._last_row_t = row["t"]
        return added

    def tick(self):
        try:
            self.poll_csv()
            s = self.bridge.snapshot()
            self.samples.append({
                "t": time.time(), "link": s["link"],
                "hb_age_s": s["hb_age_s"], "boot_id": s["boot_id"],
                "display_mask": s["display_mask"],
                "mask_state": s["mask_state"], "queued": s["queued"],
                "heap": s["heap"], "heap_big": s["heap_big"],
                "rs485_round": s.get("rs485_round"),
                "error": s["error"], "csv_rows": self._rows,
                "counters": dict(s["counters"])})
            return s
        except Exception as e:                                  # noqa
            self.exceptions.append("tick %s: %s" % (type(e).__name__, e))
            return self.bridge.snapshot()

    def wait(self, seconds, until=None, msg=None):
        """เก็บตัวอย่างไปเรื่อย ๆ จนครบเวลา หรือจน until(snap) เป็นจริง"""
        end = time.time() + seconds
        last = None
        while time.time() < end:
            s = self.tick()
            if msg and int(time.time()) % 5 == 0:
                sys.stdout.write("\r    %-58s" % msg(s)); sys.stdout.flush()
            if until is not None and until(s):
                last = s
                break
            t_end = time.time() + SAMPLE_S
            while time.time() < t_end:
                self.poll_csv()
                time.sleep(CSV_POLL_S)
        if msg:
            sys.stdout.write("\r" + " " * 64 + "\r")
        return last

    def stop(self):
        self.bridge.stop(3.0)
        self.raw.close()
        self.log.close()

    # ---- สรุปตัวเลข ----
    def csv_gaps(self):
        """ช่องว่างจาก timestamp ในไฟล์ — ความละเอียด 1 วินาที"""
        return [(b - a).total_seconds()
                for a, b in zip(self.row_times, self.row_times[1:])]

    def observed_gaps(self):
        """ช่องว่างตามที่ลูปนี้เห็น — ข้อมูลประกอบ ไม่ใช้ตัดสิน"""
        return [b - a for a, b in zip(self.csv_stamps, self.csv_stamps[1:])]

    def jsonl_rows(self):
        p = self.log.path_for()
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]


# ============================================================================
#  ขั้นตอนทดสอบ
# ============================================================================
def step_A(r):
    hb("A. STARTUP", "-")
    mock = getattr(r, "mock", False)
    au = PORTS.audit()
    role = au["roles"][PORTS.ROLE_P4_BRIDGE]
    ctl = au["roles"][PORTS.ROLE_CONTROL]
    print("  พอร์ตจอ (NDJSON) : %s  [%s]" % (role["device"] or "—", role["reason"]))
    print("  พอร์ต CONTROL    : %s  [%s]  (เครื่องมือนี้ไม่แตะ)"
          % (ctl["device"] or "—", ctl["reason"]))
    rows0 = len(r.csv.rows)
    print("  รอ heartbeat แรก (สูงสุด 40 วินาที) ...")
    s = r.wait(40, until=lambda s: s["link"] == "ONLINE",
               msg=lambda s: "link=%s  เฟรมที่ใช้ได้=%d"
                             % (s["link"], s["counters"]["rx_frames"]))
    ok_link = bool(s and s["link"] == "ONLINE")
    r.wait(12)
    rows1 = len(r.csv.rows)
    res = {
        "port_detected": bool(role["device"]),
        "port_reason": role["reason"],
        "p4_connected": ok_link,
        "display_mask": (s or {}).get("display_mask"),
        "boot_id": (s or {}).get("boot_id"),
        "csv_rows_advanced": rows1 - rows0,
        "control_untouched": True,
    }
    # โหมด mock ไม่มีพอร์ตจริง — ข้อนี้ตรวจได้เฉพาะกับของจริง
    res["port_check_skipped_mock"] = mock
    res["pass"] = ((res["port_detected"] or mock) and ok_link
                   and res["csv_rows_advanced"] > 0)
    return res


def step_B(r):
    hb("B. DISPLAY MASK", "-")
    seen = []
    for want in (7, 6, 2, 7):
        ask("ตั้ง display_mask บนจอเป็น %d  (%s)"
            % (want, ", ".join("%02d" % (i + 1) for i in range(3)
                               if (want >> i) & 1)))
        s = r.wait(30, until=lambda s: s["display_mask"] == want,
                   msg=lambda s: "รอ mask=%d  ตอนนี้=%s" % (want, s["display_mask"]))
        got = (s or r.bridge.snapshot())["display_mask"]
        seen.append({"want": want, "got": got, "ok": got == want})
        print("    mask -> %s  %s" % (got, "ผ่าน" if got == want else "ไม่ตรง"))
    ok, problems, summary = r.guard.check()
    return {"transitions": seen, "legacy_ok": ok, "legacy_problems": problems,
            "legacy_summary": summary,
            "pass": all(x["ok"] for x in seen) and ok}


def step_C(r):
    hb("C. SAVED READING", "-")
    before = {e["event_id"] for e in r.events if e["kind"] == "reading_saved"}
    n0 = len(r.jsonl_rows())
    print("  จะรอสูงสุด %.0f นาที — ไม่ต้องรีบ" % (WAIT_MEASURE_S / 60.0))
    ask("บนจอ: จุ่มหัววัด · MEASUREMENT RUN -> Start run · รอขึ้น STABLE"
        " · แล้วกด Save reading **หนึ่งครั้ง**")
    t0 = time.time()
    r.wait(WAIT_MEASURE_S, until=lambda s: any(
        e["kind"] == "reading_saved" and e["event_id"] not in before
        for e in r.events),
        msg=lambda s: "รอ reading_saved ... %d:%02d  (event ที่เก็บ %d)"
                      % (int(time.time() - t0) // 60,
                         int(time.time() - t0) % 60,
                         s["counters"]["events"]))
    got = [e for e in r.events if e["kind"] == "reading_saved"
           and e["event_id"] not in before]
    rows = r.jsonl_rows()
    saved = [x for x in rows if x.get("event") == "reading_saved"]
    ids = [x.get("event_id") for x in saved]
    res = {
        "ui_rows": len(got),
        "jsonl_rows_added": len(rows) - n0,
        "jsonl_reading_saved_total": len(saved),
        "unique_event_ids": len(set(ids)),
        "has_stable_ec": all("stable_ec_us_cm" in x for x in saved),
        "sample": saved[-1] if saved else None,
    }
    ok, problems, summary = r.guard.check()
    res["legacy_ok"] = ok
    res["legacy_problems"] = problems
    res["note"] = ("กด Save reading กี่ครั้ง ต้องได้เท่านั้นแถวพอดี — "
                   "ได้ %d แถว" % len(got))
    res["pass"] = (len(got) == 1 and len(ids) == len(set(ids))
                   and res["has_stable_ec"] and ok)
    return res


def step_D(r):
    hb("D. DISCONNECT / RECONNECT / REBOOT", "-")
    off_after = float(r.cfg["bridge"].get("offline_after_s", 16.0))
    rows0 = len(r.csv.rows)
    boot0 = r.bridge.snapshot()["boot_id"]
    n0 = len([x for x in r.jsonl_rows() if x.get("event") == "reading_saved"])

    ask("ถอดสาย USB ของจอ (เส้น USB-Serial-JTAG) ออก")
    t0 = time.time()
    s = r.wait(off_after + 20, until=lambda s: s["link"] != "ONLINE",
               msg=lambda s: "รอ P4 OFFLINE ... link=%s" % s["link"])
    t_off = time.time() - t0
    went_off = bool(s and s["link"] != "ONLINE")
    rows_during = len(r.csv.rows) - rows0
    print("    P4 OFFLINE ใน %.1f s (เกณฑ์ %.0f-%.0f s) · CSV เพิ่ม %d แถวระหว่างนั้น"
          % (t_off, off_after, off_after + 8, rows_during))

    ask("เสียบสาย USB ของจอกลับ")
    s = r.wait(60, until=lambda s: s["link"] == "ONLINE",
               msg=lambda s: "รอ P4 CONNECTED ... link=%s" % s["link"])
    back = bool(s and s["link"] == "ONLINE")
    mask_back = (s or {}).get("display_mask")
    r.wait(15)

    ask("กดปุ่ม reset บนจอ (หรือถอดไฟ) เพื่อให้ boot_id เปลี่ยน")
    s = r.wait(90, until=lambda s: s["boot_id"] and s["boot_id"] != boot0,
               msg=lambda s: "รอ boot_id ใหม่ ... %s" % s["boot_id"])
    boot1 = (s or r.bridge.snapshot())["boot_id"]
    r.wait(20)
    snap = r.bridge.snapshot()
    n1 = len([x for x in r.jsonl_rows() if x.get("event") == "reading_saved"])
    ok, problems, _ = r.guard.check()
    res = {
        "went_offline": went_off, "offline_after_s": round(t_off, 1),
        "csv_rows_during_outage": rows_during,
        "reconnected": back, "mask_after_reconnect": mask_back,
        "boot_id_before": boot0, "boot_id_after": boot1,
        "boot_changed": bool(boot1 and boot1 != boot0),
        "reconnect_count": snap["reboots"],
        "port_reopens": snap.get("port_reopens"),
        "duplicate_events_rejected": snap["counters"]["dup_events"],
        "reading_saved_added_after_reconnect": n1 - n0,
        "legacy_ok": ok, "legacy_problems": problems,
    }
    res["pass"] = (went_off and back and rows_during > 0
                   and res["boot_changed"] and res["reading_saved_added_after_reconnect"] == 0
                   and ok)
    return res


def step_E(r, minutes):
    hb("E. SOAK %d นาที" % minutes, "-")
    print("  ปล่อยให้ระบบทำงานตามปกติ  เปลี่ยน display_mask และกด Save reading")
    print("  ระหว่างทางได้ตามสะดวก  Ctrl+C เพื่อจบก่อนกำหนด\n")
    end = time.time() + minutes * 60
    try:
        while time.time() < end:
            s = r.tick()
            left = int(end - time.time())
            sys.stdout.write(
                "\r    เหลือ %02d:%02d  link=%-8s hb=%4s s  heap_big=%-9s "
                "queued=%-3s เฟรมพัง=%d"
                % (left // 60, left % 60, s["link"],
                   ("%.1f" % s["hb_age_s"]) if s["hb_age_s"] else "—",
                   "{:,}".format(s["heap_big"]) if s["heap_big"] else "—",
                   s["queued"], sum(s["counters"][k] for k in
                                    ("dropped_parse", "dropped_field",
                                     "dropped_version", "dropped_oversize"))))
            sys.stdout.flush()
            time.sleep(SAMPLE_S)
    except KeyboardInterrupt:
        print("\n    หยุดก่อนกำหนดตามคำสั่ง")
    print()
    return evaluate_soak(r)


def gap_detail(r, gaps, limit, keep=20):
    """ช่องว่างที่เกินเกณฑ์ พร้อมเวลาที่เกิด — เอาไปเทียบกับ log ของจอได้

    เวลาที่คืนเป็นเวลาของ "แถวก่อนหน้าช่องว่าง" คือจุดที่ข้อมูลขาดหายไป
    ไม่ใช่จุดที่กลับมา  ตรงกับที่ find_csv_gaps.py รายงาน จะได้เทียบกันตรง ๆ
    """
    if not gaps or len(r.row_times) < 2:
        return []
    hits = [(t, g) for t, g in zip(r.row_times, gaps) if g > limit]
    hits.sort(key=lambda x: x[1], reverse=True)
    return [{"at": t.strftime("%Y-%m-%d %H:%M:%S"), "gap_s": round(g, 3)}
            for t, g in hits[:keep]]


def evaluate_soak(r):
    gaps = r.csv_gaps()
    obs = r.observed_gaps()
    ages = [s["hb_age_s"] for s in r.samples
            if s["link"] == "ONLINE" and s["hb_age_s"] is not None]
    queued = [s["queued"] for s in r.samples if s["queued"] is not None]
    hbig = [(s["t"], s["heap_big"]) for s in r.samples if s["heap_big"]]
    c = r.samples[-1]["counters"] if r.samples else {}
    bad = sum(c.get(k, 0) for k in ("dropped_parse", "dropped_field",
                                    "dropped_version", "dropped_oversize"))
    rows = r.jsonl_rows()
    ids = [x.get("event_id") for x in rows if x.get("event_id")]
    boots = sorted({s["boot_id"] for s in r.samples if s["boot_id"]})

    retain = None
    if len(hbig) > 60:
        n = max(1, len(hbig) // 4)
        first, last = hbig[:n], hbig[-n:]
        a = min(v for _t, v in first); b = min(v for _t, v in last)
        retain = round(b / float(a), 3) if a else None

    ok, problems, summary = r.guard.check()
    m = {
        "duration_s": round(r.samples[-1]["t"] - r.samples[0]["t"], 1) if len(r.samples) > 1 else 0,
        "csv_rows": r.samples[-1]["csv_rows"] if r.samples else 0,
        "csv_gap_source": "timestamp ในไฟล์ (ความละเอียด 1 s)",
        "csv_gap_median_s": round(statistics.median(gaps), 3) if gaps else None,
        "csv_gap_p95_s": round(p95(gaps), 3) if gaps else None,
        "csv_gap_max_s": round(max(gaps), 3) if gaps else None,
        #  ⚠️ ค่าสูงสุดค่าเดียวตอบไม่ได้ว่าต้องไปแก้อะไร
        #     ช่องว่าง 9 วินาทีครั้งเดียวใน 24 ชั่วโมง กับวันละร้อยครั้ง
        #     ให้ csv_gap_max เท่ากันเป๊ะ แต่เป็นคนละเรื่องกันสิ้นเชิง
        #     soak 24 ชม. 3 ก.ย. ตกข้อนี้ข้อเดียวแล้วเราไล่ต่อไม่ได้เพราะ
        #     ไม่รู้ว่าเกิดกี่ครั้งและตอนไหน จึงต้องเก็บรายละเอียดไว้ด้วย
        "csv_gap_over_threshold": gap_detail(r, gaps, TH["csv_gap_max_s"]),
        "csv_rows_seen": len(r.row_times),
        "observed_gap_max_s": round(max(obs), 3) if obs else None,
        "hb_age_max_s": round(max(ages), 2) if ages else None,
        "hb_age_p95_s": round(p95(ages), 2) if ages else None,
        "hb_gap_max_s": c.get("hb_gap_max_s"),
        "heartbeats": c.get("heartbeats", 0),
        "valid_frames": c.get("rx_frames", 0),
        "malformed_total": bad,
        "malformed_by_reason": {k: c.get(k, 0) for k in
                                ("dropped_parse", "dropped_field",
                                 "dropped_version", "dropped_oversize")},
        "duplicates_rejected": c.get("dup_events", 0),
        "stored_duplicates": len(ids) - len(set(ids)),
        "events_stored": c.get("events", 0),
        "link_drops": c.get("link_drops", 0),
        "queued_max": max(queued) if queued else None,
        "heap_big_first": hbig[0][1] if hbig else None,
        "heap_big_last": hbig[-1][1] if hbig else None,
        "heap_big_retain": retain,
        "boot_ids": boots,
        "reboots": r.bridge.snapshot()["reboots"],
        "bridge_error": r.bridge.snapshot()["error"],
        "process_exceptions": r.exceptions,
        "legacy_ok": ok, "legacy_problems": problems,
        "legacy_summary": summary,
        "raw_capture": r.raw.snapshot(),
    }
    # ------------------------------------------------------------------
    #  แถวที่หายไป — เทียบกับตัวนับรอบของจอ ไม่ใช่เดาจาก timestamp
    #
    #  ⚠️ timestamp ใน CSV เชื่อไม่ได้สำหรับงานนี้โดยธรรมชาติ
    #     logger ประทับเวลาตอน "เขียนไฟล์" ไม่ใช่ตอนบอร์ดวัด เวลาจึงกระเพื่อม
    #     ตามภาระของเครื่อง ช่องว่างที่กว้างขึ้นดูเหมือนแถวหายทั้งที่ข้อมูลครบ
    #
    #     ข้อมูลชุดเดียวกันเคยให้คำตอบสามแบบ
    #       สูตรเดิม (หารด้วยค่าคงที่ 2.5)      -> 3.44%   ผิด
    #       สูตรนับช่องว่าง (วัดจังหวะเอง)      -> 10.0%   ผิดกว่าเดิม
    #       เทียบกับตัวนับรอบของจอ              -> 0.72%   ของจริง
    #
    #     สองแบบแรกผิดเพราะพยายามอนุมาน "ควรมีกี่แถว" จากข้อมูลที่ไม่มี
    #     ข้อมูลนั้นอยู่ในตัว  ตัวนับรอบของจอคือความจริงที่มีอยู่แล้ว
    #
    #  ⚠️ ถ้าจอรีบูตกลางทาง ตัวนับจะกลับไปเริ่มใหม่ ค่านี้จึงใช้ไม่ได้
    #     กรณีนั้นรายงาน None แล้วข้ามเกณฑ์ไป ดีกว่ารายงานตัวเลขที่ผิด
    # ------------------------------------------------------------------
    expected = missing_pct = None
    rounds = None
    #  ใช้ csv_rows จาก sample เดียวกับที่อ่าน rs485_round มา
    #  ทั้งสองค่าจึงอยู่ในหน้าต่างเวลาเดียวกันโดยอัตโนมัติ ไม่ต้องจับคู่เวลาเอง
    rs = [(x["csv_rows"], x["rs485_round"]) for x in r.samples
          if x.get("rs485_round") is not None and x.get("csv_rows") is not None]
    one_boot = len(m.get("boot_ids") or []) <= 1
    if len(rs) >= 2 and one_boot and rs[-1][1] >= rs[0][1]:
        rounds = rs[-1][1] - rs[0][1]
        rows_in_window = rs[-1][0] - rs[0][0]
        if rounds > 0:
            expected = rounds
            missing_pct = round(
                max(0.0, (rounds - rows_in_window) / float(rounds) * 100.0), 2)
    m["rs485_rounds_in_window"] = rounds
    m["csv_rows_expected"] = expected
    m["csv_rows_missing_pct"] = missing_pct
    m["csv_missing_source"] = ("ตัวนับรอบของจอ" if missing_pct is not None
                               else "ไม่มีข้อมูลอ้างอิง — ข้ามเกณฑ์นี้")

    checks = []

    def chk(name, value, ok_, note=""):
        checks.append({"check": name, "value": value, "pass": bool(ok_),
                       "note": note})

    n_over = len(m.get("csv_gap_over_threshold") or [])
    chk("ช่องว่างเกินเกณฑ์ (ครั้ง)", n_over, True)   # ข้อมูลประกอบ ไม่ตัดสิน
    chk("csv_gap_max <= %.1f s" % TH["csv_gap_max_s"], m["csv_gap_max_s"],
        m["csv_gap_max_s"] is not None and m["csv_gap_max_s"] <= TH["csv_gap_max_s"])
    chk("csv แถวหาย <= %.1f%%" % TH["csv_missing_pct_max"],
        m["csv_rows_missing_pct"],
        m["csv_rows_missing_pct"] is None
        or m["csv_rows_missing_pct"] <= TH["csv_missing_pct_max"],
        "ตัวชี้วัดที่ไม่ขึ้นกับความละเอียด 1 s ของ timestamp")
    chk("hb_age_max <= %.0f s" % TH["hb_age_max_s"], m["hb_age_max_s"],
        m["hb_age_max_s"] is not None and m["hb_age_max_s"] <= TH["hb_age_max_s"])
    chk("link_drops == 0", m["link_drops"], m["link_drops"] <= TH["link_drops_max"],
        "นับเฉพาะตอนไม่ได้ถอดสาย")
    chk("malformed == 0", m["malformed_total"], m["malformed_total"] <= TH["malformed_max"])
    chk("stored_duplicates == 0", m["stored_duplicates"],
        m["stored_duplicates"] <= TH["stored_duplicates_max"])
    chk("queued_max <= %d" % TH["queued_max"], m["queued_max"],
        m["queued_max"] is None or m["queued_max"] <= TH["queued_max"])
    chk("heap_big retain >= %.2f" % TH["heap_big_retain"], m["heap_big_retain"],
        m["heap_big_retain"] is None or m["heap_big_retain"] >= TH["heap_big_retain"],
        "ข้ามถ้าเวลาสั้นเกินจะสรุป")
    chk("process exceptions == 0", len(m["process_exceptions"]),
        len(m["process_exceptions"]) <= TH["exceptions_max"])
    chk("bridge error == None", m["bridge_error"], m["bridge_error"] is None)
    chk("legacy ต่อท้ายอย่างเดียว", m["legacy_summary"], m["legacy_ok"])
    m["checks"] = checks
    m["pass"] = all(x["pass"] for x in checks)
    return m


# ============================================================================
def render(report):
    out = []
    out.append("# ผลทดสอบฮาร์ดแวร์ P1-C")
    out.append("")
    out.append("- เวลา: %s" % report["started_at"])
    out.append("- พอร์ตจอ: %s" % (report.get("port") or "—"))
    out.append("- เกณฑ์ที่ใช้ตัดสิน (ตั้งก่อนเริ่ม): `%s`" % json.dumps(TH))
    if report.get("mock"):
        out.append("")
        out.append("> **โหมด MOCK — ไม่ใช่ผลการทดสอบฮาร์ดแวร์จริง**")
    out.append("")
    for name, res in report["steps"].items():
        out.append("## ขั้น %s — %s" % (name, "ผ่าน" if res.get("pass") else "**ไม่ผ่าน**"))
        out.append("")
        out.append("```json")
        out.append(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        out.append("```")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="A,B,C,D")
    ap.add_argument("--soak", type=int, default=0, help="นาที (ใช้กับขั้น E)")
    ap.add_argument("--bridge-port", default=None)
    ap.add_argument("--raw-capture", action="store_true")
    ap.add_argument("--wait-measure", type=float, default=None,
                    metavar="SEC", help="เวลารอ Save reading ในขั้น C (ค่าเริ่มต้น 900)")
    ap.add_argument("--mock", action="store_true",
                    help="ใช้จอจำลองแทนของจริง — ทดสอบตัวชุดทดสอบเอง")
    ap.add_argument("--unattended", action="store_true",
                    help="ไม่รอกด Enter (ใช้กับ --mock เท่านั้น)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    global UNATTENDED, WAIT_MEASURE_S
    UNATTENDED = bool(a.unattended)
    if a.wait_measure:
        WAIT_MEASURE_S = float(a.wait_measure)

    cfg = CFG.load(a.config)
    hb("P1-C — ทดสอบฮาร์ดแวร์จริง")
    print("  data_dir      : %s" % cfg["data_dir"])
    print("  legacy (อ่าน) : %s" % (cfg["legacy"].get("root") or "— ไม่ได้ตั้ง —"))
    print("  offline_after : %.0f s" % float(cfg["bridge"].get("offline_after_s", 16)))
    print("\n  เกณฑ์ผ่าน/ไม่ผ่าน (ตั้งไว้ก่อนเริ่ม ไม่ปรับตามผล):")
    for k, v in TH.items():
        print("    %-24s %s" % (k, v))
    print("\n  ⚠️ ปิดหน้าต่าง viewer ก่อน — พอร์ตเดียวเปิดได้ทีละโปรเซส")
    print("  ⚠️ logger เดิมต้องรันอยู่ตลอด นั่นคือสิ่งที่กำลังพิสูจน์")
    ask("พร้อมแล้วกด Enter เพื่อเริ่ม")

    if a.mock:
        print("\n  *** โหมด MOCK — ผลนี้ไม่ใช่ผลการทดสอบฮาร์ดแวร์ ***")
    r = Runner(cfg, port=a.bridge_port, raw=a.raw_capture, mock=a.mock)
    report = {"started_at": datetime.now().isoformat(timespec="seconds"),
              "thresholds": TH, "port": a.bridge_port, "steps": {},
              "mock": bool(a.mock)}
    steps = [x.strip().upper() for x in a.steps.split(",") if x.strip()]
    try:
        if "A" in steps:
            report["steps"]["A"] = step_A(r)
        if "B" in steps:
            report["steps"]["B"] = step_B(r)
        if "C" in steps:
            report["steps"]["C"] = step_C(r)
        if "D" in steps:
            report["steps"]["D"] = step_D(r)
        if "E" in steps:
            report["steps"]["E"] = step_E(r, a.soak or 120)
        if "E" not in steps:
            report["steps"]["summary"] = evaluate_soak(r)
    except KeyboardInterrupt:
        print("\n  ถูกหยุดกลางคัน — บันทึกเท่าที่ได้")
        report["aborted"] = True
    finally:
        snap = r.bridge.snapshot()
        report["port"] = report["port"] or snap.get("port")
        report["final_snapshot"] = snap
        r.stop()

    d = os.path.join(cfg["data_dir"], "diag")
    os.makedirs(d, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    jp = os.path.join(d, "hwtest_%s.json" % stamp)
    mp = os.path.join(d, "hwtest_%s.md" % stamp)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render(report))
    DIAG.export(cfg["data_dir"],
                DIAG.build(report["final_snapshot"], r.state.snapshot(),
                           csv=r.csv, raw=r.raw, ports=PORTS.describe(),
                           extra="หลังจบ hw_test"),
                "hwtest_%s_diag.json" % stamp)

    hb("สรุป")
    allpass = True
    for name, res in report["steps"].items():
        good = res.get("pass")
        allpass = allpass and bool(good)
        print("  ขั้น %-8s %s" % (name, "ผ่าน" if good else "ไม่ผ่าน"))
        for chkr in res.get("checks", []):
            print("      %-34s %-14s %s" % (chkr["check"], chkr["value"],
                                            "ok" if chkr["pass"] else "FAIL"))
    print("\n  รายงาน: %s" % mp)
    print("          %s" % jp)
    print("\n  ผลรวม: %s" % ("ผ่านทั้งหมด" if allpass else "มีข้อที่ไม่ผ่าน"))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
