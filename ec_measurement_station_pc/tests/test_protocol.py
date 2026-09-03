# -*- coding: utf-8 -*-
"""ตรวจตัวแยกเฟรม NDJSON — ครอบคลุมทุกทางที่ปฏิเสธได้"""
import json
import unittest

from _helpers import ROOT  # noqa
from ecstation.bridge import protocol as P

# เฟรม reading_saved ที่เฟิร์มแวร์ "ส่งจริง" วันนี้ (pc_bridge.c:461-471)
WIRE_TODAY = json.dumps({
    "v": 1, "type": "event", "event_id": "p4-c96e-a820-000017",
    "boot_id": "p4-c96e-a820", "event": "reading_saved", "sensor": 2,
    "ec_us_cm": 1146.0, "temperature_c": 20.6, "tolerance_us_cm": 11.5,
    "stable_for_ms": 15000, "after_link_error": False,
    "ts_ms": 182345, "wall": "27 Aug 2026  14:42:19"})

# เฟรมเดียวกันถ้าวันหนึ่งเฟิร์มแวร์เปลี่ยนไปใช้ชื่อที่ชัดเจนกว่า
WIRE_FUTURE = json.dumps({
    "v": 1, "type": "event", "event_id": "p4-c96e-a820-000018",
    "boot_id": "p4-c96e-a820", "event": "reading_saved", "sensor": 2,
    "stable_ec_us_cm": 1146.0, "temperature_c": 20.6, "tolerance_us_cm": 11.5,
    "stable_for_ms": 15000, "device_mono_ms": 182345})

HB = json.dumps({"v": 1, "type": "hb", "boot_id": "p4-c96e-a820",
                 "ts_ms": 182345, "queued": 0, "link": "online",
                 "heap": 328399, "heap_big": 253952, "display_mask": 6})


class TestReadingSavedFieldNames(unittest.TestCase):
    """test_reading_saved_field_names — สัญญาของ 'ระเบียน' ที่ PC เก็บ

    ชื่อบนสายเปลี่ยนได้ตามเฟิร์มแวร์ แต่ชื่อในระเบียนต้องนิ่งและไม่กำกวม
    """

    REQUIRED = ("v", "type", "event_id", "boot_id", "event", "sensor",
                "stable_ec_us_cm", "temperature_c", "tolerance_us_cm",
                "stable_for_ms", "device_mono_ms")

    def test_record_has_exactly_the_required_names(self):
        for wire in (WIRE_TODAY, WIRE_FUTURE):
            r = P.parse_line(wire)
            self.assertTrue(r.ok, r.reason)
            rec = P.reading_saved_record(r.frame)
            for k in self.REQUIRED:
                self.assertIn(k, rec, "ระเบียนขาด field %s" % k)
            self.assertEqual(rec["stable_ec_us_cm"], 1146.0)
            self.assertEqual(rec["device_mono_ms"], 182345)
            self.assertEqual(P.READING_SAVED_RECORD_FIELDS, self.REQUIRED)

    def test_accepts_firmware_wire_names(self):
        """เฟิร์มแวร์วันนี้ส่ง ec_us_cm / ts_ms — ต้องรับได้ ไม่งั้นค่าจะเป็น null ทุกครั้ง"""
        r = P.parse_line(WIRE_TODAY)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.frame.stable_ec_us_cm, 1146.0)
        self.assertEqual(r.frame.device_mono_ms, 182345)

    def test_rejects_ambiguous_ec_names(self):
        for bad in ("ec", "value", "ec_value", "reading"):
            d = json.loads(WIRE_TODAY)
            d.pop("ec_us_cm")
            d[bad] = 1146.0
            r = P.parse_line(json.dumps(d))
            self.assertFalse(r.ok, "ควรปฏิเสธชื่อกำกวม %r" % bad)

    def test_rejects_conflicting_ec_values(self):
        d = json.loads(WIRE_TODAY)
        d["stable_ec_us_cm"] = 10.0        # ต่างจาก ec_us_cm = 1146.0
        r = P.parse_line(json.dumps(d))
        self.assertFalse(r.ok)
        self.assertIn("stable_ec_us_cm!=ec_us_cm", r.reason)

    def test_missing_ec_is_rejected(self):
        d = json.loads(WIRE_TODAY); d.pop("ec_us_cm")
        self.assertFalse(P.parse_line(json.dumps(d)).ok)


class TestHeartbeat(unittest.TestCase):
    def test_ok(self):
        r = P.parse_line(HB)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.frame.display_mask, 6)
        self.assertEqual(r.frame.heap_big, 253952)
        self.assertEqual(r.frame.device_mono_ms, 182345)

    def test_display_mask_255_means_unknown(self):
        """0xFF คือค่าตั้งต้นในเฟิร์มแวร์ ไม่ใช่ 'เปิดครบทุกตัว'

        255 & 0b111 = 7 ซึ่งดูสมเหตุสมผลแต่ผิด
        """
        d = json.loads(HB); d["display_mask"] = 255
        r = P.parse_line(json.dumps(d))
        self.assertTrue(r.ok)
        self.assertIsNone(r.frame.display_mask)

    def test_optional_fields_may_be_absent(self):
        d = json.loads(HB)
        for k in ("heap", "heap_big", "display_mask"):
            d.pop(k)
        r = P.parse_line(json.dumps(d))
        self.assertTrue(r.ok, r.reason)

    def test_bad_link_value(self):
        d = json.loads(HB); d["link"] = "maybe"
        self.assertFalse(P.parse_line(json.dumps(d)).ok)

    def test_missing_required(self):
        for k in ("boot_id", "queued", "link"):
            d = json.loads(HB); d.pop(k)
            r = P.parse_line(json.dumps(d))
            self.assertFalse(r.ok, "ควรปฏิเสธเมื่อขาด %s" % k)
            self.assertIn(k, r.reason)


class TestEnvelope(unittest.TestCase):
    def test_oversize(self):
        r = P.parse_line(b'{"v":1,"type":"hb","pad":"' + b"A" * 600 + b'"}')
        self.assertEqual(r.reason, P.R_OVERSIZE)

    def test_not_utf8(self):
        self.assertEqual(P.parse_line(b"\xff\xfe abc").reason, P.R_NOT_UTF8)

    def test_not_json(self):
        self.assertEqual(P.parse_line("{ไม่ใช่ json").reason, P.R_NOT_JSON)

    def test_not_object(self):
        self.assertEqual(P.parse_line("[1,2,3]").reason, P.R_NOT_OBJECT)

    def test_bad_version(self):
        d = json.loads(HB); d["v"] = 99
        self.assertEqual(P.parse_line(json.dumps(d)).reason, P.R_BAD_VERSION)

    def test_unknown_type_is_skipped_quietly(self):
        r = P.parse_line('{"v":1,"type":"ดาวอังคาร"}')
        self.assertEqual(r.reason, P.R_UNKNOWN_TYPE)

    def test_unknown_fields_are_ignored(self):
        d = json.loads(HB); d["ของใหม่ในอนาคต"] = {"a": 1}
        self.assertTrue(P.parse_line(json.dumps(d)).ok)

    def test_never_raises(self):
        for junk in (b"", "  ", "{", "null", "123", '{"v":1}',
                     '{"v":1,"type":123}', b"\x00\x01\x02",
                     '{"v":1,"type":"event"}'):
            P.parse_line(junk)      # ต้องไม่ raise


class TestContextEventAndCommand(unittest.TestCase):
    def test_context_event(self):
        s = json.dumps({"v": 1, "type": "event", "event_id": "e1",
                        "boot_id": "b", "event": "STABILITY_REACHED",
                        "sensor": 2, "ec_us_cm": 84.6, "ts_ms": 5})
        r = P.parse_line(s)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.frame.event, "STABILITY_REACHED")
        self.assertEqual(r.frame.ec_us_cm, 84.6)

    def test_context_event_name_is_upper(self):
        s = json.dumps({"v": 1, "type": "event", "event_id": "e2",
                        "boot_id": "b", "event": "run_started",
                        "sensor": 1, "ts_ms": 5})
        self.assertEqual(P.parse_line(s).frame.event, "RUN_STARTED")

    def test_sensor_zero_means_whole_system(self):
        s = json.dumps({"v": 1, "type": "cmd", "request_id": 7, "boot_id": "b",
                        "action": "recording_start", "sensor": 0, "ts_ms": 5})
        r = P.parse_line(s)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.frame.sensor, 0)

    def test_sensor_out_of_range(self):
        s = json.dumps({"v": 1, "type": "event", "event_id": "e", "boot_id": "b",
                        "event": "RUN_STARTED", "sensor": 9, "ts_ms": 1})
        self.assertFalse(P.parse_line(s).ok)


class TestStateFrame(unittest.TestCase):
    def test_fits_in_frame_limit(self):
        f = P.build_state(999999, True, True, 6, 7, False, 1234567, "CALF-20 B3")
        data = P.dumps_line(f)
        self.assertIsNotNone(data)
        self.assertLessEqual(len(data), P.MAX_LINE)

    def test_sample_id_truncated_to_23(self):
        f = P.build_state(1, True, False, 0, 7, False, 0, "x" * 60)
        self.assertLessEqual(len(f["sample_id"]), 23)

    def test_assumed_flag_present(self):
        f = P.build_state(1, True, False, 0, 7, False, 0, "")
        self.assertTrue(f["active_mask_assumed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
