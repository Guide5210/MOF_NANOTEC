# -*- coding: utf-8 -*-
"""
เงื่อนไขการยอมรับของ P1-B — สี่ข้อ

ทั้งหมดยิงที่ view_model / series_painter ซึ่งไม่พึ่ง tkinter
จึงรันได้บนเครื่องที่ไม่มีหน้าจอ และเป็นเหตุผลที่ตรรกะพวกนี้ถูกแยกออกมา
"""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure   # noqa: E402

from _helpers import ROOT, tmp_cfg     # noqa
from ecstation.bridge import protocol as P            # noqa: E402
from ecstation.bridge.event_log import EventLog       # noqa: E402
from ecstation.core.csv_source import CsvTail         # noqa: E402
from ecstation.ui import lab_theme as T               # noqa: E402
from ecstation.ui import view_model as VM             # noqa: E402
from ecstation.ui.series_painter import SeriesPainter  # noqa: E402


def write_csv(path, n=12, base=(1362.4, 1146.0, 84.6), ok=(1, 1, 1)):
    now = datetime.now()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("timestamp,EC1,T1,EC2,T2,EC3,T3,ok1,ok2,ok3\n")
        for i in range(n):
            t = now - timedelta(seconds=2.5 * (n - 1 - i))
            fh.write("%s,%.1f,20.6,%.1f,20.5,%.1f,20.4,%d,%d,%d\n" % (
                t.strftime("%Y-%m-%d %H:%M:%S"),
                base[0], base[1], base[2], ok[0], ok[1], ok[2]))


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_vm_")
        self.data = os.path.join(self.d, "water_data")
        os.makedirs(self.data)
        self.csv_path = os.path.join(
            self.data, "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        write_csv(self.csv_path)
        self.csv = CsvTail(self.data)
        self.csv.poll()
        self.model = VM.DashboardModel()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def snap(self, link="ONLINE", mask=0b111):
        return {"link": link, "link_text": "P4 CONNECTED", "display_mask": mask,
                "view_mask": mask, "counters": {}}


# ==========================================================================
#  A1 — mask 7 -> 6 -> 2 -> 7  ต้องไม่มีเส้นค้าง
# ==========================================================================
class TestMaskTransition(Base):
    SEQUENCE = [(0b111, [1, 2, 3]), (0b110, [2, 3]), (0b010, [2]),
                (0b111, [1, 2, 3])]

    def test_cards_follow_the_mask(self):
        for mask, want in self.SEQUENCE:
            sensors = self.model.sensors(self.csv, mask)
            vis = self.model.visible_sensors(sensors)
            self.assertEqual([s.number for s in vis], want,
                             "mask=%s" % bin(mask))
            self.assertEqual(len(sensors), 3,
                             "ต้องคำนวณครบทุกตัวเสมอ แค่ไม่แสดงบางตัว")

    def test_chart_has_no_ghost_series(self):
        rows = self.csv.since(None)
        for mode in ("split", "overlay"):
            painter = SeriesPainter(Figure())
            for mask, want in self.SEQUENCE:
                sensors = self.model.sensors(self.csv, mask)
                painter.draw(rows, self.model.chart_series(sensors), mode)
                idx = [n - 1 for n in want]
                self.assertEqual(painter.sensors_drawn(), idx,
                                 "%s mask=%s เส้นไม่ตรง" % (mode, bin(mask)))
                # ตัวเลขนี้คือที่ ghost ซ่อนตัว: Line2D ที่ยังอยู่บนแกน
                self.assertEqual(painter.artist_count(), len(idx),
                                 "%s mask=%s มีเส้นค้างบนแกน" % (mode, bin(mask)))
                self.assertEqual(len(painter.legend_labels()), len(idx),
                                 "%s mask=%s legend ค้าง" % (mode, bin(mask)))

    def test_series_colour_never_shifts_after_hide_and_show(self):
        """ตัวที่ 3 ต้องเป็นสีเดิมเสมอ แม้เคยถูกซ่อนแล้วเปิดกลับ

        นี่คืออาการจริงของ ghost series: prop_cycle เลื่อน แล้วสีสลับ
        ผู้ใช้ที่จับคู่เส้นด้วยสีจากจอ P4 จะอ่านผิดโดยไม่มีอะไรเตือน
        """
        rows = self.csv.since(None)
        painter = SeriesPainter(Figure())
        seen = {}
        for mask, _ in self.SEQUENCE:
            sensors = self.model.sensors(self.csv, mask)
            painter.draw(rows, self.model.chart_series(sensors), "overlay")
            for (_ax, s), ln in painter._lines.items():
                col = ln.get_color().upper()
                self.assertEqual(col, T.SENSOR_SERIES[s].upper())
                seen.setdefault(s, col)
                self.assertEqual(seen[s], col)

    def test_empty_mask_draws_nothing_not_a_broken_panel(self):
        painter = SeriesPainter(Figure())
        sensors = self.model.sensors(self.csv, 0)
        painter.draw(self.csv.since(None),
                     self.model.chart_series(sensors), "split")
        self.assertEqual(painter.sensors_drawn(), [])
        self.assertEqual(painter.artist_count(), 0)
        self.assertEqual(self.model.visible_sensors(sensors), [])


# ==========================================================================
#  A2 — ซ่อนบนจอ ต้องไม่กระทบข้อมูลดิบ
# ==========================================================================
class TestHiddenRawIntegrity(Base):
    def test_hidden_sensor_stays_in_the_legacy_csv(self):
        before = sha(self.csv_path)
        for mask in (0b111, 0b011, 0b001, 0b111):
            sensors = self.model.sensors(self.csv, mask)
            self.model.visible_sensors(sensors)
            self.model.chart_series(sensors)
            self.csv.poll()
        self.assertEqual(sha(self.csv_path), before,
                         "การกรองที่ UI ไปแตะไฟล์ CSV ของระบบเดิม")

    def test_hidden_sensor_still_has_data_and_state(self):
        """ซ่อน = ไม่แสดง ไม่ใช่ไม่มีข้อมูล"""
        sensors = self.model.sensors(self.csv, 0b011)
        s3 = sensors[2]
        self.assertTrue(s3.hidden)
        self.assertIsNotNone(s3.ec)
        self.assertEqual(s3.ec, 84.6)
        self.assertNotEqual(s3.state, T.DISABLED)   # ไม่ใช่ error ไม่ใช่ปิดโพล

    def test_hidden_sensor_is_never_rendered_as_zero_or_error(self):
        sensors = self.model.sensors(self.csv, 0b011)
        vis = self.model.visible_sensors(sensors)
        self.assertNotIn(3, [s.number for s in vis])
        # ที่สำคัญกว่า: ไม่มีการ์ดไหนที่โชว์ 0.0 แทนตัวที่ซ่อน
        for s in vis:
            self.assertNotEqual(s.ec_text(), "0.0")

    def test_engineering_view_reveals_with_a_label(self):
        self.model.engineering = True
        sensors = self.model.sensors(self.csv, 0b011)
        vis = self.model.visible_sensors(sensors)
        self.assertEqual([s.number for s in vis], [1, 2, 3])
        labels = [lbl for _i, _c, lbl, hidden in
                  self.model.chart_series(sensors) if hidden]
        self.assertEqual(len(labels), 1)
        self.assertIn("hidden on HMI", labels[0])

    def test_view_model_module_has_no_write_path(self):
        for name in ("view_model.py", "series_painter.py"):
            with open(os.path.join(ROOT, "ecstation", "ui", name),
                      encoding="utf-8") as fh:
                src = fh.read()
            for bad in ('open(', 'os.remove', 'os.rename', 'shutil'):
                self.assertNotIn(bad, src, "%s มีเส้นทางเขียนไฟล์" % name)


# ==========================================================================
#  A3 — reading_saved ซ้ำ 100 ครั้ง ต้องขึ้นครั้งเดียว
# ==========================================================================
class TestEventDedup(Base):
    def _frame(self, eid="p4-aaaa-000017"):
        line = json.dumps({
            "v": 1, "type": "event", "event_id": eid, "boot_id": "p4-aaaa",
            "event": "reading_saved", "sensor": 2, "ec_us_cm": 1146.0,
            "temperature_c": 20.6, "tolerance_us_cm": 11.5,
            "stable_for_ms": 15000, "after_link_error": False,
            "ts_ms": 182345}).encode()
        res = P.parse_line(line)
        self.assertTrue(res.ok, res.reason)
        return res.frame

    def test_hundred_duplicates_appear_once_in_the_ui(self):
        feed = self.model.events
        fr = self._frame()
        for _ in range(100):
            feed.add_reading_saved(fr)
        rows = [r for r in feed.visible() if r["kind"] == "reading_saved"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(feed.duplicates, 99)

    def test_hundred_duplicates_stored_once_on_disk(self):
        log = EventLog(self.d)
        fr = self._frame()
        rec = P.reading_saved_record(fr)
        for _ in range(100):
            log.append(rec)
        log.close()
        with open(log.path_for(), encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(log.duplicates, 99)

    def test_reading_saved_row_shows_stable_ec(self):
        """ต้องแสดง stable_ec_us_cm ไม่ใช่ค่าสดตอนนั้น"""
        feed = self.model.events
        feed.add_reading_saved(self._frame())
        row = feed.visible()[0]
        self.assertEqual(row["value"], 1146.0)
        self.assertEqual(row["text"], "READING SAVED")
        self.assertIn("±11.5", row["detail"])
        self.assertIn("15.0 s", row["detail"])

    def test_different_ids_are_kept_and_ordered_newest_first(self):
        feed = self.model.events
        for i in range(5):
            feed.add_reading_saved(self._frame("p4-aaaa-%06d" % i))
        rows = feed.visible()
        self.assertEqual(len(rows), 5)
        self.assertGreaterEqual(rows[0]["when"], rows[-1]["when"])

    def test_event_ids_are_not_shown_on_the_dashboard_rows(self):
        """id เต็มเป็นเรื่องของ Diagnostics — dashboard แสดงความหมาย ไม่ใช่รหัส"""
        feed = self.model.events
        feed.add_reading_saved(self._frame())
        row = feed.visible()[0]
        self.assertNotIn("p4-aaaa", row["text"])
        self.assertNotIn("p4-aaaa", row["detail"])
        self.assertIsNotNone(row["event_id"])       # แต่ยังเก็บไว้ให้ Diagnostics


# ==========================================================================
#  A4 — จอหลุด ต้องคงค่าที่เห็นล่าสุดพร้อมป้ายบอก
# ==========================================================================
class TestP4StateLoss(Base):
    def test_keeps_last_mask_with_a_clear_label(self):
        online = self.snap("ONLINE", 0b110)
        self.assertEqual(self.model.resolve_mask(online), (0b110, None))

        offline = {"link": "OFFLINE", "link_text": "P4 OFFLINE",
                   "display_mask": 0b110, "view_mask": 0b110, "counters": {}}
        mask, note = self.model.resolve_mask(offline)
        self.assertEqual(mask, 0b110, "จอหลุดแล้วรีเซ็ต mask")
        self.assertEqual(note, "P4 OFFLINE — showing last HMI selection")

    def test_does_not_silently_reset_to_all_sensors(self):
        offline = {"link": "OFFLINE", "display_mask": 0b010,
                   "view_mask": 0b010, "counters": {}}
        mask, note = self.model.resolve_mask(offline)
        self.assertNotEqual(mask, 0b111, "รีเซ็ตเป็นครบทุกตัวโดยไม่บอกใคร")
        self.assertEqual(mask, 0b010)
        self.assertIsNotNone(note)

    def test_does_not_hide_the_cards(self):
        offline = {"link": "OFFLINE", "display_mask": 0b110,
                   "view_mask": 0b110, "counters": {}}
        mask, _ = self.model.resolve_mask(offline)
        vis = self.model.visible_sensors(self.model.sensors(self.csv, mask))
        self.assertEqual([s.number for s in vis], [2, 3])

    def test_no_label_while_the_screen_is_still_answering(self):
        _mask, note = self.model.resolve_mask(self.snap("ONLINE", 0b110))
        self.assertIsNone(note, "ขึ้นป้ายทั้งที่จอยังตอบอยู่")

    def test_never_knew_the_mask_shows_everything_without_a_false_label(self):
        """ไม่เคยได้ mask เลย ต่างจาก 'เคยได้แล้วจอหาย' — ห้ามใช้ป้ายเดียวกัน"""
        never = {"link": "OFFLINE", "display_mask": None, "view_mask": None,
                 "counters": {}}
        mask, note = self.model.resolve_mask(never)
        self.assertEqual(mask, VM.FULL_MASK)
        self.assertIsNone(note)

    def test_event_history_survives_the_disconnect(self):
        """สายหลุดต้องไม่ลบหลักฐานงานที่เพิ่งทำไป"""
        feed = self.model.events
        feed.add("reading_saved", sensor=2, value=1146.0,
                 event_id="p4-aaaa-000001")
        before = len(feed.visible())
        self.model.resolve_mask({"link": "OFFLINE", "display_mask": 0b110,
                                 "view_mask": 0b110, "counters": {}})
        self.assertEqual(len(feed.visible()), before)
        self.assertEqual(feed.visible()[0]["value"], 1146.0)

    def test_reconnect_returns_to_following_the_screen(self):
        for snap in ({"link": "OFFLINE", "display_mask": 0b110,
                      "view_mask": 0b110, "counters": {}},
                     self.snap("ONLINE", 0b001)):
            mask, note = self.model.resolve_mask(snap)
        self.assertEqual(mask, 0b001)
        self.assertIsNone(note)


class TestSummaryAndBadges(Base):
    def test_summary_counts_only_visible_sensors(self):
        sensors = self.model.sensors(self.csv, 0b011)
        txt = self.model.summary(sensors, 0b011)
        self.assertIn("2 SENSORS", txt)

    def test_link_badge_colours(self):
        self.assertEqual(VM.link_badge(self.snap("ONLINE"))["colour"], T.OK)
        self.assertEqual(
            VM.link_badge({"link": "ERROR", "link_text": "x"})["colour"],
            T.ERROR)
        self.assertEqual(
            VM.link_badge({"link": "DISABLED",
                           "link_text": "P4 BRIDGE DISABLED"})["colour"],
            T.IDLE)

    def test_history_note_only_when_paused(self):
        self.assertIsNone(self.model.chart_note())
        self.model.history_paused = True
        self.assertEqual(self.model.chart_note(),
                         "HISTORY VIEW — LIVE FOLLOW PAUSED")


if __name__ == "__main__":
    unittest.main()
