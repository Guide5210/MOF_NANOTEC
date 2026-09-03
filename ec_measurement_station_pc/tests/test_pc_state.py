# -*- coding: utf-8 -*-
"""
pc_state / legacy_read — สถานะฝั่ง PC ที่ส่งให้จอ

หัวใจของชุดนี้คือ **ห้ามโกหกจอ**
  - ความมีชีวิตของ logger ต้องมีสามระดับ ไม่ใช่ ออนไลน์/ไม่ออนไลน์
  - active_mask ที่ส่งไปเป็นค่าสมมติ ต้องมีธงกำกับเสมอ
  - ไม่มี rec_status.json = OFFLINE ห้ามเดาว่ายังเก็บอยู่
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta

from _helpers import ROOT, tmp_cfg  # noqa
from ecstation.core import legacy_read as LR
from ecstation.bridge.pc_state import PcStateSource, CONTROL_ASSUMED_ACTIVE_MASK


# ------------------------------------------------------------------ liveness
class TestLiveness(unittest.TestCase):
    """เกณฑ์ที่ผู้ใช้กำหนด: <=10 s ONLINE · <=30 s STALE · เกินนั้น OFFLINE"""

    def test_three_states_and_their_boundaries(self):
        cases = [
            (0.0,   LR.PC_ONLINE),
            (9.9,   LR.PC_ONLINE),
            (10.0,  LR.PC_ONLINE),      # ขอบบนยังนับว่าออนไลน์
            (10.1,  LR.PC_STALE),
            (29.9,  LR.PC_STALE),
            (30.0,  LR.PC_STALE),
            (30.1,  LR.PC_OFFLINE),
            (3600., LR.PC_OFFLINE),
        ]
        for age, want in cases:
            self.assertEqual(LR.pc_liveness(age), want, "age=%s" % age)

    def test_missing_file_is_offline_not_online(self):
        self.assertEqual(LR.pc_liveness(None), LR.PC_OFFLINE)

    def test_texts_match_the_agreed_wording(self):
        self.assertEqual(LR.PC_TEXT[LR.PC_ONLINE], "PC LOGGER ONLINE")
        self.assertEqual(LR.PC_TEXT[LR.PC_STALE], "PC STATUS DELAYED")
        self.assertEqual(LR.PC_TEXT[LR.PC_OFFLINE], "PC LOGGER OFFLINE")


# --------------------------------------------------------------- rec_status
class TestRecStatus(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_legacy_")
        self.p = os.path.join(self.d, "rec_status.json")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def write(self, obj):
        with open(self.p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    def test_new_format_with_mask_and_sample(self):
        self.write({"active": [False, True, True], "mask": 6,
                    "sample": [None, "CALF-20 B3", "POND-A"],
                    "updated": datetime.now().isoformat()})
        rec, age = LR.read_rec_status(self.p)
        sv = LR.session_view(rec)
        self.assertEqual(sv["mask"], 6)
        self.assertTrue(sv["recording"])
        self.assertEqual(sv["sample_id"], "CALF-20 B3")
        self.assertLess(age, 2.0)

    def test_old_format_without_mask_still_reads(self):
        """logger รุ่นก่อนไม่มี field mask — ต้องคำนวณจาก active ให้ได้"""
        self.write({"active": [True, False, True],
                    "updated": datetime.now().isoformat()})
        rec, _ = LR.read_rec_status(self.p)
        self.assertEqual(LR.session_view(rec)["mask"], 0b101)

    def test_no_session_open(self):
        self.write({"active": [False, False, False], "mask": 0,
                    "updated": datetime.now().isoformat()})
        rec, _ = LR.read_rec_status(self.p)
        sv = LR.session_view(rec)
        self.assertFalse(sv["recording"])
        self.assertEqual(sv["sample_id"], "")

    def test_corrupt_json_does_not_raise(self):
        with open(self.p, "w", encoding="utf-8") as fh:
            fh.write("{ครึ่งบรรทัด")
        rec, age = LR.read_rec_status(self.p)
        self.assertIsNone(rec)
        self.assertIsNone(age)
        self.assertEqual(LR.session_view(rec)["mask"], 0)

    def test_bad_timestamp_falls_back_to_file_mtime(self):
        """ถ้า field updated เพี้ยน ยังต้องรู้อายุจาก mtime ไม่ใช่ยอมแพ้"""
        self.write({"active": [True], "mask": 1, "updated": "ไม่ใช่เวลา"})
        rec, age = LR.read_rec_status(self.p)
        self.assertIsNotNone(rec)
        self.assertIsNotNone(age)
        self.assertLess(age, 5.0)

    def test_missing_file(self):
        rec, age = LR.read_rec_status(os.path.join(self.d, "ไม่มีจริง.json"))
        self.assertIsNone(rec)
        self.assertIsNone(age)


# ------------------------------------------------------------- CSV row count
class TestCsvRowCounter(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_csv_")
        self.p = os.path.join(
            self.d, "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def append(self, n, start=0):
        new = not os.path.exists(self.p)
        with open(self.p, "a", encoding="utf-8") as fh:
            if new:
                fh.write("timestamp,sensor,ec,temp\n")
            for i in range(n):
                fh.write("2026-08-28 12:00:%02d,1,1146.0,20.6\n"
                         % ((start + i) % 60))

    def test_header_is_not_counted_as_a_row(self):
        self.append(5)
        self.assertEqual(LR.CsvRowCounter(self.d).today(), 5)

    def test_incremental_counting_matches_a_full_recount(self):
        c = LR.CsvRowCounter(self.d)
        self.append(10)
        self.assertEqual(c.today(), 10)
        self.append(7)
        self.assertEqual(c.today(), 17)          # ต่อยอด ไม่ใช่เริ่มนับใหม่
        self.assertEqual(LR.CsvRowCounter(self.d).today(), 17)

    def test_file_rewritten_smaller_resets_cleanly(self):
        c = LR.CsvRowCounter(self.d)
        self.append(20)
        self.assertEqual(c.today(), 20)
        with open(self.p, "w", encoding="utf-8") as fh:
            fh.write("timestamp,sensor,ec,temp\n")
        self.append(3)
        self.assertEqual(c.today(), 3)

    def test_empty_dir_is_zero_not_an_error(self):
        self.assertEqual(LR.CsvRowCounter(self.d).today(), 0)
        self.assertEqual(LR.CsvRowCounter("").today(), 0)
        self.assertEqual(LR.CsvRowCounter("/ไม่มี/โฟลเดอร์นี้").today(), 0)


# ------------------------------------------------------------- state source
class TestPcStateSource(unittest.TestCase):
    def setUp(self):
        self.legacy = tempfile.mkdtemp(prefix="ec_legacyroot_")
        self.data = os.path.join(self.legacy, "water_data")
        os.makedirs(self.data)
        self.rec = os.path.join(self.legacy, "rec_status.json")
        self.mine = tempfile.mkdtemp(prefix="ec_new_")
        self.cfg = tmp_cfg(self.mine, self.legacy)

    def tearDown(self):
        shutil.rmtree(self.legacy, ignore_errors=True)
        shutil.rmtree(self.mine, ignore_errors=True)

    def write_rec(self, ago_s=0.0, mask=6):
        when = datetime.now() - timedelta(seconds=ago_s)
        with open(self.rec, "w", encoding="utf-8") as fh:
            json.dump({"active": [False, True, True], "mask": mask,
                       "sample": [None, "CALF-20 B3", None],
                       "updated": when.isoformat()}, fh)

    def write_rows(self, last_ago_s=0.0, n=5):
        """เขียน CSV ให้แถวสุดท้ายเก่า last_ago_s วินาที

        ⚠️ ความมีชีวิตของ logger ตัดสินจากแถวนี้ ไม่ใช่จาก rec_status.json
        """
        p = os.path.join(self.data,
                         "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("timestamp,EC1,T1,EC2,T2,EC3,T3,ok1,ok2,ok3\n")
            for i in range(n):
                t = (datetime.now()
                     - timedelta(seconds=last_ago_s + 2.5 * (n - 1 - i)))
                fh.write("%s,1362.4,20.6,1146.0,20.5,84.6,20.4,1,1,1\n"
                         % t.strftime("%Y-%m-%d %H:%M:%S"))
        return p

    def test_online_snapshot(self):
        self.write_rec(0.0)
        self.write_rows(0.0)
        s = PcStateSource(self.cfg).snapshot()
        self.assertEqual(s["liveness"], LR.PC_ONLINE)
        self.assertEqual(s["liveness_text"], "PC LOGGER ONLINE")
        self.assertTrue(s["recording"])
        self.assertEqual(s["session_mask"], 6)
        self.assertEqual(s["sample_id"], "CALF-20 B3")

    def test_stale_then_offline(self):
        self.write_rec(0.0)
        self.write_rows(20.0)
        self.assertEqual(PcStateSource(self.cfg).snapshot()["liveness"],
                         LR.PC_STALE)
        self.write_rows(120.0)
        self.assertEqual(PcStateSource(self.cfg).snapshot()["liveness"],
                         LR.PC_OFFLINE)

    def test_old_rec_status_does_not_mean_offline(self):
        """บั๊กที่เจอจากฮาร์ดแวร์จริง

        logger เดิมเขียน rec_status.json เฉพาะตอนเปิดโปรแกรม / เริ่ม / จบ session
        (logger_3ec.py:337,403,411)  ระหว่างเก็บข้อมูลปกติไฟล์นั้นจะเก่าขึ้น
        เรื่อย ๆ ตลอดกาล  ถ้าเอามาตัดสิน liveness จะขึ้น OFFLINE ถาวร
        ทั้งที่ CSV ยาวขึ้นทุก 2.5 วินาที — และจะส่ง pc_online=false ไปให้จอด้วย
        """
        self.write_rec(3600.0)        # ไฟล์สถานะเก่าหนึ่งชั่วโมง
        self.write_rows(1.0)          # แต่ข้อมูลเพิ่งเข้ามาเมื่อวินาทีที่แล้ว
        s = PcStateSource(self.cfg).snapshot()
        self.assertEqual(s["liveness"], LR.PC_ONLINE)
        self.assertGreater(s["rec_age_s"], 3000)
        self.assertLess(s["data_age_s"], 5)
        frame, _ = PcStateSource(self.cfg).build_frame()
        self.assertTrue(frame["pc_online"])

    def test_fresh_rec_status_does_not_hide_a_dead_logger(self):
        """ด้านกลับ: เพิ่งกด start session แล้ว logger ตาย — ต้องจับได้"""
        self.write_rec(0.0)
        self.write_rows(300.0)
        s = PcStateSource(self.cfg).snapshot()
        self.assertEqual(s["liveness"], LR.PC_OFFLINE)

    def test_stale_still_counts_as_online_to_the_screen(self):
        """STALE = ยังไม่รู้ว่าตาย  จอยังควรเห็นว่า PC เชื่อมอยู่
        เพราะการรีบขึ้น PC OFFLINE ทำให้คนวิ่งไปไล่สายทั้งที่ไม่มีอะไรพัง
        """
        self.write_rec(0.0)
        self.write_rows(20.0)
        frame, s = PcStateSource(self.cfg).build_frame()
        self.assertEqual(s["liveness"], LR.PC_STALE)
        self.assertTrue(frame["pc_online"])

    def test_offline_tells_the_screen_the_truth(self):
        self.write_rec(0.0)
        self.write_rows(120.0)
        frame, _ = PcStateSource(self.cfg).build_frame()
        self.assertFalse(frame["pc_online"])

    def test_no_rec_status_is_offline_and_not_recording(self):
        frame, s = PcStateSource(self.cfg).build_frame()
        self.assertEqual(s["liveness"], LR.PC_OFFLINE)
        self.assertFalse(frame["pc_online"])
        self.assertFalse(frame["recording"])
        self.assertEqual(frame["session_mask"], 0)

    def test_session_info_still_comes_from_rec_status(self):
        """rec_status ยังเป็นแหล่งของ 'session ไหนเปิดอยู่' — คนละคำถามกับ liveness"""
        self.write_rec(3600.0, mask=6)
        self.write_rows(0.0)
        s = PcStateSource(self.cfg).snapshot()
        self.assertEqual(s["session_mask"], 6)
        self.assertEqual(s["sample_id"], "CALF-20 B3")
        self.assertTrue(s["recording"])

    def test_active_mask_is_assumed_and_says_so(self):
        """CONTROL ยังเป็น #define N_SENSORS 3 ไม่มีทางบอก active_mask จริง
        จึงต้องส่งธงกำกับ ไม่ใช่แต่งค่าแล้วแสดงเหมือนเป็นข้อเท็จจริง
        """
        self.write_rec(0.0)
        self.write_rows(0.0)
        frame, s = PcStateSource(self.cfg).build_frame()
        self.assertEqual(frame["active_mask"], CONTROL_ASSUMED_ACTIVE_MASK)
        self.assertEqual(frame["active_mask"], 0b0111)
        self.assertTrue(frame["active_mask_assumed"])
        self.assertTrue(s["active_mask_assumed"])

    def test_seq_increases_monotonically(self):
        src = PcStateSource(self.cfg)
        seqs = [src.build_frame()[0]["seq"] for _ in range(5)]
        self.assertEqual(seqs, [1, 2, 3, 4, 5])

    def test_frame_always_fits_the_line_limit(self):
        from ecstation.bridge import protocol as P
        self.write_rec(0.0)
        self.write_rows(0.0)
        frame, _ = PcStateSource(self.cfg).build_frame()
        self.assertIsNotNone(P.dumps_line(frame))

    def test_csv_rows_reported(self):
        self.write_rec(0.0)
        p = os.path.join(self.data,
                         "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("timestamp,sensor,ec,temp\n")
            for i in range(9):
                fh.write("2026-08-28 12:00:00,1,1146.0,20.6\n")
        self.assertEqual(PcStateSource(self.cfg).snapshot()["csv_rows"], 9)

    def test_cal_busy_is_false_in_p1_and_honest_about_it(self):
        """P1 ยังไม่มีตัวสังเกตการคาลิเบรต — ต้องเป็น false เสมอ ไม่ใช่เดา"""
        self.write_rec(0.0)
        self.write_rows(0.0)
        self.assertFalse(PcStateSource(self.cfg).snapshot()["cal_busy"])


if __name__ == "__main__":
    unittest.main()
