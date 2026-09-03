# -*- coding: utf-8 -*-
"""
P1-C — เครื่องมือก่อนแตะฮาร์ดแวร์

⚠️ ชุดทดสอบฮาร์ดแวร์เองก็ต้องถูกทดสอบ
   ถ้า hw_test.py มีบั๊ก มันจะรายงานว่า "ระบบพัง" ทั้งที่ตัวมันเองพัง
   แล้วเราจะไปแก้ผิดจุดตอนที่มีฮาร์ดแวร์ต่ออยู่ตรงหน้า ซึ่งแพงที่สุด
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

from _helpers import ROOT, FakePort  # noqa

sys.path.insert(0, os.path.join(ROOT, "tools"))
import hw_test as HW                              # noqa: E402
from ecstation.bridge.raw_capture import RawCapture  # noqa: E402
from ecstation.core import ports as P             # noqa: E402
from ecstation.diag import snapshot as DIAG       # noqa: E402


# ==========================================================  ports
class TestPortSelection(unittest.TestCase):
    def p(self, dev, desc, hw, vid=None, pid=None, sn=None):
        f = FakePort(dev, desc, hw)
        f.vid, f.pid, f.serial_number = vid, pid, sn
        return f

    def test_single_p4_is_picked(self):
        ps = [self.p("COM3", "CH340", "USB VID:PID=1A86:7523", 0x1a86, 0x7523),
              self.p("COM7", "USB Serial Device",
                     "USB VID:PID=303A:1001 SER=ABC123", 0x303a, 0x1001, "ABC123")]
        dev, reason, _ = P.find_detailed(P.ROLE_P4_BRIDGE, lambda: ps)
        self.assertEqual((dev, reason), ("COM7", P.PICK_OK))

    def test_two_esp32_boards_refuse_to_guess(self):
        """303A:1001 เป็นของ ESP32 ทุกตัว ไม่ใช่ของ P4 ตัวเดียว

        เดาแล้วเปิดพอร์ตผิดตัว = ไปยึดพอร์ตของบอร์ดอื่นไว้ด้วย
        แล้วนั่งรอข้อมูลที่ไม่มีวันมา ซึ่งไล่หาสาเหตุยากมาก
        """
        ps = [self.p("COM7", "USB Serial Device", "USB VID:PID=303A:1001",
                     0x303a, 0x1001),
              self.p("COM9", "USB JTAG/serial debug unit",
                     "USB VID:PID=303A:1001", 0x303a, 0x1001)]
        dev, reason, cands = P.find_detailed(P.ROLE_P4_BRIDGE, lambda: ps)
        self.assertIsNone(dev)
        self.assertEqual(reason, P.PICK_AMBIGUOUS)
        self.assertEqual(len(cands), 2)

    def test_control_never_picks_the_p4_ports(self):
        ps = [self.p("COM7", "USB Serial Device", "USB VID:PID=303A:1001",
                     0x303a, 0x1001),
              self.p("COM8", "USB-Enhanced-SERIAL CH343",
                     "USB VID:PID=1A86:55D3", 0x1a86, 0x55d3)]
        dev, reason, _ = P.find_detailed(P.ROLE_CONTROL, lambda: ps)
        self.assertIsNone(dev)
        self.assertEqual(reason, P.PICK_NONE)

    def test_serial_number_is_optional_not_required(self):
        """USB-Serial-JTAG มี descriptor ใน ROM — บางเครื่องไม่เห็นเลขซีเรียล"""
        no_sn = self.p("COM7", "USB Serial Device", "USB VID:PID=303A:1001",
                       0x303a, 0x1001)
        self.assertIsNone(P.serial_number(no_sn))
        dev, reason, _ = P.find_detailed(P.ROLE_P4_BRIDGE, lambda: [no_sn])
        self.assertEqual((dev, reason), ("COM7", P.PICK_OK))

    def test_serial_number_read_from_hwid_when_attribute_missing(self):
        f = FakePort("COM7", "x", "USB VID:PID=303A:1001 SER=F0F5BD4A LOCATION=1-2")
        self.assertEqual(P.serial_number(f), "F0F5BD4A")

    def test_audit_shape(self):
        ps = [self.p("COM3", "CH340", "USB VID:PID=1A86:7523", 0x1a86, 0x7523)]
        au = P.audit(lambda: ps)
        self.assertEqual(au["roles"][P.ROLE_CONTROL]["device"], "COM3")
        self.assertEqual(au["roles"][P.ROLE_P4_BRIDGE]["reason"], P.PICK_NONE)
        self.assertEqual(au["ports"][0]["vid_pid"], "1A86:7523")


# ==========================================================  raw capture
class TestRawCapture(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_raw_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_disabled_by_default_writes_nothing(self):
        r = RawCapture(self.d)
        self.assertFalse(r.add("rx", b'{"v":1}'))
        self.assertFalse(os.path.isdir(os.path.join(self.d, "raw")))

    def test_writes_into_our_own_raw_dir_only(self):
        r = RawCapture(self.d, enabled=True)
        r.add("rx", b'{"v":1,"type":"hb"}')
        r.close()
        self.assertTrue(r.path.startswith(os.path.join(self.d, "raw")))
        with open(r.path, encoding="utf-8") as fh:
            self.assertIn('"type":"hb"', fh.read())

    def test_stops_at_the_size_limit_instead_of_filling_the_disk(self):
        """flood ที่วัดได้ 2,396 บรรทัด/วินาที จะทำให้ไฟล์โตหลาย GB ในไม่กี่นาที
        ดิสก์เต็มจะพา logger เดิมล่มไปด้วย ซึ่งเป็นสิ่งเดียวที่ห้ามเกิด
        """
        r = RawCapture(self.d, enabled=True, max_bytes=2000)
        for i in range(500):
            r.add("rx", b"x" * 100)
        r.close()
        self.assertTrue(r.stopped)
        self.assertLessEqual(os.path.getsize(r.path), 2400)
        self.assertGreater(r.dropped_after_limit, 0)

    def test_write_failure_never_kills_the_bridge(self):
        r = RawCapture(self.d, enabled=True)
        r.path = os.path.join(self.d, "ไม่มีโฟลเดอร์นี้", "x.log")
        self.assertFalse(r.add("rx", b"hello"))
        self.assertFalse(r.enabled)


# ==========================================================  diag snapshot
class TestDiagSnapshot(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_diag_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_no_laboratory_data_leaks_into_the_export(self):
        """ไฟล์นี้มักถูกแนบส่งไปขอความช่วยเหลือ — ต้องปลอดภัยโดยโครงสร้าง"""
        payload = DIAG.build(
            {"link": "ONLINE", "counters": {"events": 3}},
            {"liveness": "ONLINE", "sample_id": "CALF-20 B3",
             "session_mask": 6, "csv_rows": 34560})
        blob = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("CALF-20", blob)
        self.assertNotIn("sample_id", blob)

    def test_paths_are_redacted(self):
        class C(object):
            rows, skipped_lines, read_errors = [], 0, 0
            dir = r"C:/MOF_NanoTec/test_realtime/water_data"
        payload = DIAG.build({}, {}, csv=C())
        blob = json.dumps(payload)
        self.assertNotIn("MOF_NanoTec", blob)
        self.assertEqual(payload["csv"]["dir"], ".../water_data")

    def test_export_lands_in_our_data_dir(self):
        p = DIAG.export(self.d, DIAG.build({}, {}))
        self.assertTrue(p.startswith(os.path.join(self.d, "diag")))
        with open(p, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["schema"], "ecstation.diag/1")

    def test_export_cannot_escape_with_a_path_in_the_name(self):
        p = DIAG.export(self.d, DIAG.build({}, {}), "../../escape.json")
        self.assertTrue(p.startswith(os.path.join(self.d, "diag")))
        self.assertTrue(p.endswith("escape.json"))

    def test_malformed_total_is_derived_not_trusted(self):
        payload = DIAG.build(
            {"counters": {"dropped_parse": 2, "dropped_field": 1,
                          "dropped_version": 0, "dropped_oversize": 3}}, {})
        self.assertEqual(payload["counters_derived"]["malformed_total"], 6)


# ==========================================================  legacy guard
class TestLegacyGuard(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_guard_")
        os.makedirs(os.path.join(self.d, "water_data"))
        self.csv = os.path.join(self.d, "water_data", "water_log_x.csv")
        with open(self.csv, "w") as fh:
            fh.write("timestamp,EC1\n2026-08-28 12:00:00,1.0\n")
        with open(os.path.join(self.d, "logger_3ec.py"), "w") as fh:
            fh.write("# legacy\n")
        with open(os.path.join(self.d, "rec_status.json"), "w") as fh:
            json.dump({"mask": 6}, fh)
        self.g = HW.LegacyGuard(self.d)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_append_to_csv_is_allowed(self):
        """logger เดิมกำลังเขียนอยู่จริง — โตขึ้นคือปกติ ไม่ใช่ความผิดพลาด"""
        with open(self.csv, "a") as fh:
            fh.write("2026-08-28 12:00:02,2.0\n")
        ok, problems, summary = self.g.check()
        self.assertTrue(ok, problems)
        self.assertEqual(summary["appended_only"], 1)

    def test_rewriting_the_start_of_a_csv_is_caught(self):
        """ต่อท้ายได้ แต่ของเดิมห้ามเปลี่ยน — นี่คือสิ่งที่ต้องจับให้ได้"""
        with open(self.csv, "w") as fh:
            fh.write("timestamp,EC1\n2026-08-28 12:00:00,999.0\n")
        ok, problems, _ = self.g.check()
        self.assertFalse(ok)
        self.assertTrue(any("ส่วนเดิมถูกแก้" in x for x in problems))

    def test_truncating_a_csv_is_caught(self):
        with open(self.csv, "w") as fh:
            fh.write("t\n")
        ok, problems, _ = self.g.check()
        self.assertFalse(ok)

    def test_modifying_legacy_source_is_caught(self):
        with open(os.path.join(self.d, "logger_3ec.py"), "a") as fh:
            fh.write("print('x')\n")
        ok, problems, _ = self.g.check()
        self.assertFalse(ok)
        self.assertTrue(any("ไม่ใช่ไฟล์ข้อมูล" in x for x in problems))

    def test_runtime_status_file_may_change(self):
        """rec_status.json เป็นของ logger เดิม มันเขียนทับตัวเองได้"""
        with open(os.path.join(self.d, "rec_status.json"), "w") as fh:
            json.dump({"mask": 7}, fh)
        ok, problems, _ = self.g.check()
        self.assertTrue(ok, problems)

    def test_unexpected_new_file_is_caught(self):
        with open(os.path.join(self.d, "ec_events.jsonl"), "w") as fh:
            fh.write("{}\n")
        ok, problems, _ = self.g.check()
        self.assertFalse(ok)
        self.assertTrue(any("ไฟล์ใหม่" in x for x in problems))

    def test_deleted_file_is_caught(self):
        os.remove(os.path.join(self.d, "logger_3ec.py"))
        ok, problems, _ = self.g.check()
        self.assertFalse(ok)
        self.assertTrue(any("หายไป" in x for x in problems))


# ==========================================================  evaluate_soak
class FakeRunner(object):
    def __init__(self, samples, row_times, jsonl=None, exceptions=None,
                 guard_ok=True):
        self.samples = samples
        self.row_times = row_times
        self.csv_stamps = []
        self._jsonl = jsonl or []
        self.exceptions = exceptions or []
        self.raw = RawCapture(tempfile.mkdtemp(prefix="ec_fr_"))

        class G(object):
            def check(_s):
                return (guard_ok, [] if guard_ok else ["พัง"], {"files_tracked": 1})
        self.guard = G()

        class Br(object):
            def snapshot(_s):
                return {"reboots": 0, "error": None}
        self.bridge = Br()

    def csv_gaps(self):
        return [(b - a).total_seconds()
                for a, b in zip(self.row_times, self.row_times[1:])]

    def observed_gaps(self):
        return []

    def jsonl_rows(self):
        return self._jsonl


def mk(n=120, hb_age=2.0, heap_big=250000, queued=0, bad=0, dup_stored=0,
       lost=0):
    t0 = 1000.0
    c = {"dropped_parse": bad, "dropped_field": 0, "dropped_version": 0,
         "dropped_oversize": 0, "dup_events": 0, "events": 5,
         "link_drops": 0, "heartbeats": 20, "rx_frames": 100,
         "hb_gap_max_s": 5.1}
    #  rs485_round เดินทีละหนึ่งพร้อม csv_rows = ไม่มีแถวหาย
    #  lost = จำนวนรอบที่บอร์ดเดินแต่ logger เขียนไม่ทัน (แถวหายจริง)
    samples = [{"t": t0 + i, "link": "ONLINE", "hb_age_s": hb_age,
                "boot_id": "p4-a", "display_mask": 7, "mask_state": "FOLLOWING",
                "queued": queued, "heap": 300000, "heap_big": heap_big,
                "error": None,
                "csv_rows": 100 + i - (lost if i == n - 1 else 0),
                "rs485_round": 500 + i,
                "counters": dict(c)}
               for i in range(n)]
    base = datetime(2026, 8, 28, 12, 0, 0)
    # จังหวะ 2.5 s บน timestamp ที่ละเอียด 1 s -> สลับ 2/3 วินาที
    rows, acc = [], 0.0
    for i in range(n):
        acc += 2.5
        rows.append(base + timedelta(seconds=round(acc)))
    j = [{"event": "reading_saved", "event_id": "e%d" % i} for i in range(3)]
    j += [{"event": "reading_saved", "event_id": "e0"}] * dup_stored
    return FakeRunner(samples, rows, j)


class TestEvaluateSoak(unittest.TestCase):
    def test_healthy_run_passes(self):
        m = HW.evaluate_soak(mk())
        self.assertTrue(m["pass"], [c for c in m["checks"] if not c["pass"]])
        self.assertEqual(m["csv_rows_missing_pct"], 0.0)

    def test_one_second_timestamp_resolution_does_not_fail_the_run(self):
        """ที่จังหวะ 2.5 s บน timestamp ละเอียด 1 s ช่องว่างจะสลับ 2/3 วินาที
        เกณฑ์ต้องยอมรับ 3.0 s ได้ ไม่งั้นเทสต์ตกเพราะหน่วยวัด ไม่ใช่เพราะระบบ
        """
        m = HW.evaluate_soak(mk())
        self.assertEqual(m["csv_gap_max_s"], 3.0)
        self.assertNotIn("csv_gap_p95_s", HW.TH)
        self.assertTrue(next(c for c in m["checks"]
                             if c["check"].startswith("csv_gap_max"))["pass"])

    def test_missing_rows_are_caught(self):
        """บอร์ดเดินครบทุกรอบ แต่ logger เขียนได้น้อยกว่า = แถวหายจริง

        วัดจากตัวนับรอบของจอ ไม่ใช่จาก timestamp — timestamp กระเพื่อมตาม
        ภาระของเครื่องจนแยกไม่ออกว่าแถวหายหรือแค่เขียนช้า
        """
        m = HW.evaluate_soak(mk(lost=20))
        self.assertGreater(m["csv_rows_missing_pct"], 1.0)
        self.assertFalse(m["pass"])

    def test_missing_pct_is_skipped_when_display_rebooted(self):
        """จอรีบูตแล้วตัวนับรอบกลับไปเริ่มใหม่ — ห้ามรายงานตัวเลขที่ผิด"""
        r = mk()
        r.samples[-1]["boot_id"] = "p4-b"        # บูตใหม่กลางทาง
        r.samples[-1]["rs485_round"] = 3         # ตัวนับรีเซ็ต
        m = HW.evaluate_soak(r)
        self.assertIsNone(m["csv_rows_missing_pct"])
        self.assertIn("ไม่มีข้อมูลอ้างอิง", m["csv_missing_source"])

    def test_malformed_frame_fails_the_run(self):
        m = HW.evaluate_soak(mk(bad=1))
        self.assertEqual(m["malformed_total"], 1)
        self.assertFalse(m["pass"])

    def test_stored_duplicate_fails_but_rejected_duplicate_does_not(self):
        self.assertFalse(HW.evaluate_soak(mk(dup_stored=1))["pass"])
        r = mk()
        for s in r.samples:
            s["counters"]["dup_events"] = 9      # ถูกปฏิเสธ ไม่ได้ถูกเก็บ
        m = HW.evaluate_soak(r)
        self.assertEqual(m["duplicates_rejected"], 9)
        self.assertEqual(m["stored_duplicates"], 0)
        self.assertTrue(m["pass"])

    def test_heartbeat_age_over_threshold_fails(self):
        self.assertFalse(HW.evaluate_soak(mk(hb_age=13.0))["pass"])

    def test_heap_big_decline_is_caught(self):
        r = mk(n=400)
        for i, s in enumerate(r.samples):
            s["heap_big"] = int(250000 - i * 300)     # ลดเรื่อย ๆ
        m = HW.evaluate_soak(r)
        self.assertLess(m["heap_big_retain"], 0.90)
        self.assertFalse(m["pass"])

    def test_queue_backlog_is_caught(self):
        self.assertFalse(HW.evaluate_soak(mk(queued=20))["pass"])

    def test_legacy_problem_fails_the_run(self):
        r = mk()
        r.guard = type("G", (), {"check": lambda _s: (False, ["ถูกแก้"], {})})()
        self.assertFalse(HW.evaluate_soak(r)["pass"])

    def test_thresholds_are_declared_up_front(self):
        """เกณฑ์ต้องเป็นค่าคงที่ที่พิมพ์ออกมาก่อนเริ่ม ไม่ใช่คำนวณจากผล"""
        for k in ("csv_gap_max_s", "csv_missing_pct_max", "hb_age_max_s",
                  "link_drops_max", "malformed_max", "stored_duplicates_max",
                  "queued_max", "heap_big_retain", "exceptions_max"):
            self.assertIn(k, HW.TH)
            self.assertIsInstance(HW.TH[k], (int, float))


if __name__ == "__main__":
    unittest.main()


class TestGapDetail(unittest.TestCase):
    """ช่องว่างที่เกินเกณฑ์ต้องรายงานเวลาที่เกิด ไม่ใช่แค่ค่าสูงสุด

    soak 24 ชม. ตกที่ csv_gap_max ข้อเดียวแล้วไล่ต่อไม่ได้ เพราะรายงานบอกแค่
    9.0 วินาที ไม่บอกว่าเกิดกี่ครั้งและตอนไหน ซึ่งเป็นคนละคำถามกันสิ้นเชิง
    """

    def test_reports_when_and_how_many(self):
        r = mk(n=120)
        base = r.row_times[0]
        # ยัดช่องว่างกว้างสองที่ ที่รู้เวลาแน่นอน
        for i in range(30, len(r.row_times)):
            r.row_times[i] += timedelta(seconds=9)
        for i in range(80, len(r.row_times)):
            r.row_times[i] += timedelta(seconds=20)
        m = HW.evaluate_soak(r)
        over = m["csv_gap_over_threshold"]
        self.assertEqual(len(over), 2)
        # เรียงจากกว้างสุดก่อน
        self.assertGreater(over[0]["gap_s"], over[1]["gap_s"])
        self.assertIn("at", over[0])
        # เวลาที่รายงานคือแถวก่อนช่องว่าง ไม่ใช่แถวที่กลับมา
        self.assertEqual(over[1]["at"], r.row_times[29].strftime("%Y-%m-%d %H:%M:%S"))

    def test_empty_when_no_gaps(self):
        self.assertEqual(HW.evaluate_soak(mk())["csv_gap_over_threshold"], [])
