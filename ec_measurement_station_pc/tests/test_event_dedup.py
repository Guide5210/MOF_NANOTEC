# -*- coding: utf-8 -*-
"""event_log — ต่อท้ายอย่างเดียว และห้ามมี reading_saved ซ้ำ"""
import json
import os
import shutil
import tempfile
import unittest

from _helpers import ROOT  # noqa
from ecstation.bridge.event_log import EventLog


def rec(eid, ec=1146.0):
    return {"v": 1, "type": "event", "event_id": eid, "boot_id": "b1",
            "event": "reading_saved", "sensor": 2, "stable_ec_us_cm": ec,
            "temperature_c": 20.6, "tolerance_us_cm": 11.5,
            "stable_for_ms": 15000, "device_mono_ms": 1}


class TestDedup(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_evlog_")
        self.log = EventLog(self.d)

    def tearDown(self):
        self.log.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def _lines(self):
        p = self.log.path_for()
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def test_duplicate_written_once(self):
        for _ in range(3):
            self.log.append(rec("p4-b1-000017"))
        self.assertEqual(len(self._lines()), 1)
        self.assertEqual(self.log.duplicates, 2)

    def test_distinct_ids_all_written(self):
        for i in range(50):
            self.log.append(rec("p4-b1-%06d" % i))
        self.assertEqual(len(self._lines()), 50)

    def test_dedup_survives_restart(self):
        """เปิดโปรแกรมใหม่แล้วจอ replay คิว — ต้องไม่ได้บรรทัดซ้ำ"""
        self.log.append(rec("p4-b1-000017"))
        self.log.close()
        again = EventLog(self.d)
        self.assertTrue(again.seen("p4-b1-000017"))
        self.assertFalse(again.append(rec("p4-b1-000017")))
        again.close()
        self.assertEqual(len(self._lines()), 1)

    def test_reboot_does_not_clear_dedup(self):
        """event_id มี boot_id อยู่ในตัว จึงไม่ต้องล้างตอนจอรีบูต"""
        self.log.append(rec("p4-b1-000001"))
        self.log.append(rec("p4-b2-000001"))     # boot ใหม่ id ไม่ชนกัน
        self.assertFalse(self.log.append(rec("p4-b1-000001")))
        self.assertEqual(len(self._lines()), 2)

    def test_dedup_set_is_bounded(self):
        log = EventLog(self.d, dedup_max=64)
        for i in range(500):
            log.append(rec("p4-b1-%06d" % i))
        self.assertLessEqual(len(log._seen), 64 + 1)
        log.close()

    def test_record_is_append_only(self):
        self.log.append(rec("a"))
        first = self._lines()[0]
        self.log.append(rec("b"))
        self.assertEqual(self._lines()[0], first, "บรรทัดเก่าต้องไม่ถูกแก้")

    def test_recv_wall_and_source_added(self):
        self.log.append(rec("a"))
        line = self._lines()[0]
        self.assertIn("recv_wall", line)
        self.assertEqual(line["source"], "p4")

    def test_local_event_has_pc_source(self):
        self.log.append_local("DISPLAY_MASK_CHANGED", **{"from": 7, "to": 6})
        line = self._lines()[-1]
        self.assertEqual(line["source"], "pc")
        self.assertEqual(line["to"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
