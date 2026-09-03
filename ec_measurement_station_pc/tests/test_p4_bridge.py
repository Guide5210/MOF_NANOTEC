# -*- coding: utf-8 -*-
"""
p4_bridge — พฤติกรรมของช่องคุยกับจอ

สิ่งที่เทสต์ชุดนี้ยืนยัน (เรียงตามความสำคัญ)
  1. เฟรมพังไม่ทำให้ bridge ตาย และไม่ทำให้ event ที่ดีถูกทิ้ง
  2. คำสั่งจากจอต้องถูก "ตอบ NACK" ไม่ใช่เงียบ  (เงียบ = จอรอ 5 วิ แล้วขึ้น
     "ไม่ทราบผล" ซึ่งชวนให้ผู้ใช้กดซ้ำ)
  3. จอเงียบ -> OFFLINE, จอรีบูต -> นับและบันทึก
  4. event ที่ยังไม่มี session ต้องติดธง unassigned_session ห้ามสร้าง session เอง
"""
import json
import os
import shutil
import tempfile
import time
import unittest

from _helpers import ROOT, tmp_cfg  # noqa
from ecstation.bridge import protocol as P
from ecstation.bridge import p4_bridge as B
from ecstation.bridge.event_log import EventLog


def hb(boot="p4-aaaa", ts=1000, mask=0b0111, queued=0, link="online",
       heap=180000, heap_big=90000):
    d = {"v": 1, "type": "hb", "boot_id": boot, "ts_ms": ts, "queued": queued,
         "link": link, "heap": heap, "heap_big": heap_big,
         "display_mask": mask}
    return json.dumps(d).encode()


def saved(eid, boot="p4-aaaa", sensor=2, ec=1146.0, ts=2000):
    d = {"v": 1, "type": "event", "event_id": eid, "boot_id": boot,
         "event": "reading_saved", "sensor": sensor, "ec_us_cm": ec,
         "temperature_c": 20.6, "tolerance_us_cm": 11.5,
         "stable_for_ms": 15000, "after_link_error": False, "ts_ms": ts,
         "wall": "2026-08-28 12:00:00"}
    return json.dumps(d).encode()


def cmd(rid=7, action="start_session", sensor=2, boot="p4-aaaa", ts=3000):
    return json.dumps({"v": 1, "type": "cmd", "request_id": rid,
                       "boot_id": boot, "action": action, "sensor": sensor,
                       "ts_ms": ts}).encode()


class FakeState(object):
    """แทน PcStateSource — คุมค่าได้เต็มที่ ไม่ต้องมีไฟล์ของ legacy"""

    def __init__(self, session_mask=0, sample_id="", raise_on_snapshot=False):
        self.session_mask = session_mask
        self.sample_id = sample_id
        self.raise_on_snapshot = raise_on_snapshot
        self.seq = 0

    def snapshot(self):
        if self.raise_on_snapshot:
            raise RuntimeError("อ่าน rec_status ไม่ได้")
        return {"session_mask": self.session_mask, "sample_id": self.sample_id,
                "recording": bool(self.session_mask), "csv_rows": 12,
                "active_mask": 0b0111, "active_mask_assumed": True,
                "cal_busy": False}

    def build_frame(self):
        self.seq += 1
        s = self.snapshot()
        return P.build_state(self.seq, True, s["recording"], s["session_mask"],
                             s["active_mask"], False, s["csv_rows"],
                             s["sample_id"]), s


class BridgeCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ec_bridge_")
        self.log = EventLog(self.d)
        self.tr = B.LoopbackTransport()
        self.state = FakeState()
        self.events = []
        self.br = B.P4Bridge(tmp_cfg(self.d), self.log, self.state,
                             transport=self.tr,
                             on_event=lambda k, o: self.events.append((k, o)))

    def tearDown(self):
        self.br.stop(0.5)
        self.log.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def lines(self):
        p = self.log.path_for()
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def sent(self):
        return [json.loads(x.decode()) for x in self.tr.sent]


# ----------------------------------------------------------- heartbeat / link
class TestLink(BridgeCase):
    def test_hb_brings_link_online(self):
        self.assertEqual(self.br.link, B.LINK_OFFLINE)
        self.br._handle(hb())
        s = self.br.snapshot()
        self.assertEqual(s["link"], B.LINK_ONLINE)
        self.assertEqual(s["link_text"], "P4 CONNECTED")
        self.assertEqual(s["boot_id"], "p4-aaaa")
        self.assertEqual(s["heap"], 180000)
        self.assertEqual(s["heap_big"], 90000)
        self.assertEqual(s["display_mask"], 0b0111)

    def test_disabled_bridge_is_not_an_error(self):
        cfg = tmp_cfg(self.d)
        cfg["bridge"]["enabled"] = False
        br = B.P4Bridge(cfg, self.log, self.state, transport=self.tr)
        self.assertEqual(br.link, B.LINK_DISABLED)
        self.assertEqual(br.snapshot()["link_text"], "P4 BRIDGE DISABLED")
        self.assertFalse(br.start())          # ไม่สตาร์ตเธรดเลย

    def test_silence_flips_to_offline(self):
        """จอเงียบเกินกำหนดต้องกลายเป็น OFFLINE เอง ไม่ใช่ค้างที่ CONNECTED

        การค้างที่ CONNECTED อันตรายกว่าไม่แสดงอะไร เพราะผู้ใช้จะเชื่อว่า
        ยังคุยกันอยู่ทั้งที่สายหลุดไปแล้ว
        """
        self.br.offline_after = 0.05
        self.br.state_interval = 10.0         # ไม่ให้ไปยุ่งกับจังหวะทดสอบ
        self.br._handle(hb())
        self.assertEqual(self.br.link, B.LINK_ONLINE)
        self.br.start()
        time.sleep(0.35)
        self.br.stop(1.0)
        self.assertEqual(self.br.link, B.LINK_OFFLINE)
        self.assertEqual(self.br.mask.state, "STALE")

    def test_reboot_counted_and_logged(self):
        self.br._handle(hb(boot="p4-aaaa"))
        self.br._handle(hb(boot="p4-bbbb"))
        self.assertEqual(self.br.reboots, 1)
        ev = [l for l in self.lines() if l.get("event") == "P4_REBOOT"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["previous_boot_id"], "p4-aaaa")
        self.assertEqual(ev[0]["boot_id"], "p4-bbbb")

    def test_mask_unknown_ff_is_not_treated_as_seven(self):
        self.br._handle(hb(mask=0xFF))
        self.assertIsNone(self.br.snapshot()["display_mask"])
        self.assertEqual(self.br.snapshot()["view_mask"], 0b0111)  # ปลอดภัยไว้ก่อน

    def test_mask_change_logged_once(self):
        self.br._handle(hb(mask=0b0111))
        self.br._handle(hb(mask=0b0111))
        self.br._handle(hb(mask=0b0101))
        kinds = [l.get("event") for l in self.lines()]
        self.assertEqual(kinds.count("DISPLAY_MASK_INITIAL"), 1)
        self.assertEqual(kinds.count("DISPLAY_MASK_CHANGED"), 1)


# ------------------------------------------------------------------- events
class TestEvents(BridgeCase):
    def test_reading_saved_written_with_contract_fields(self):
        self.br._handle(saved("p4-aaaa-000017"))
        rows = [l for l in self.lines() if l.get("event") == "reading_saved"]
        self.assertEqual(len(rows), 1)
        for f in P.READING_SAVED_RECORD_FIELDS:
            self.assertIn(f, rows[0])
        self.assertEqual(rows[0]["stable_ec_us_cm"], 1146.0)
        self.assertEqual(rows[0]["device_mono_ms"], 2000)
        self.assertEqual(self.br.counters["events"], 1)

    def test_duplicate_event_id_counted_not_written_twice(self):
        for _ in range(3):
            self.br._handle(saved("p4-aaaa-000017"))
        rows = [l for l in self.lines() if l.get("event") == "reading_saved"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.br.counters["dup_events"], 2)
        self.assertEqual(self.br.counters["events"], 1)

    def test_event_without_session_is_flagged_never_auto_assigned(self):
        self.br._handle(saved("p4-aaaa-000018"))
        row = [l for l in self.lines() if l.get("event") == "reading_saved"][0]
        self.assertTrue(row.get("unassigned_session"))
        self.assertNotIn("sample_id", row)

    def test_event_with_open_session_carries_sample(self):
        self.state.session_mask = 0b0110
        self.state.sample_id = "CALF-20 B3"
        self.br._handle(saved("p4-aaaa-000019"))
        row = [l for l in self.lines() if l.get("event") == "reading_saved"][0]
        self.assertEqual(row["sample_id"], "CALF-20 B3")
        self.assertEqual(row["session_mask"], 0b0110)
        self.assertNotIn("unassigned_session", row)

    def test_state_source_failure_does_not_lose_the_event(self):
        """อ่านสถานะ PC ไม่ได้ ต้องไม่ทำให้ผลการวัดหาย — แค่ผูก session ไม่ได้"""
        self.state.raise_on_snapshot = True
        self.br._handle(saved("p4-aaaa-000020"))
        rows = [l for l in self.lines() if l.get("event") == "reading_saved"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].get("unassigned_session"))


# ----------------------------------------------------------------- commands
class TestCommands(BridgeCase):
    def test_command_is_nacked_immediately(self):
        self.br._handle(cmd(rid=7, action="start_session"))
        acks = [m for m in self.sent() if m.get("type") == "ack"]
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]["request_id"], 7)
        self.assertFalse(acks[0]["ok"])
        self.assertEqual(acks[0]["code"], "COMMANDS_DISABLED")
        self.assertEqual(acks[0]["action"], "start_session")
        self.assertEqual(self.br.counters["cmds_nacked"], 1)

    def test_every_command_is_audited(self):
        for i, a in enumerate(("start_session", "stop_session", "calibrate")):
            self.br._handle(cmd(rid=i, action=a))
        rej = [l for l in self.lines() if l.get("event") == "CMD_REJECTED"]
        self.assertEqual(len(rej), 3)
        self.assertEqual({r["action"] for r in rej},
                         {"start_session", "stop_session", "calibrate"})

    def test_no_command_path_executes_anything(self):
        """P1 ต้องไม่มีเส้นทางไหนที่ทำคำสั่งจริง — ตรวจที่ซอร์สโดยตรง"""
        with open(os.path.join(ROOT, "ecstation", "bridge", "p4_bridge.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("COMMANDS_DISABLED", src)
        self.assertNotIn("build_ack(", src)      # ยังไม่มีทางตอบ ok=true


# ------------------------------------------------------------- ทนต่อขยะ
class TestRobustness(BridgeCase):
    def test_malformed_lines_are_counted_by_reason(self):
        self.br._handle(u"{ไม่ใช่ json".encode("utf-8"))
        self.br._handle(b"[1,2,3]")
        self.br._handle(json.dumps({"v": 9, "type": "hb"}).encode())
        self.br._handle(json.dumps({"v": 1, "type": "hb"}).encode())   # ขาด field
        self.br._handle(b"x" * (P.MAX_LINE + 1))
        c = self.br.counters
        self.assertEqual(c["dropped_version"], 1)
        self.assertEqual(c["dropped_field"], 1)
        self.assertEqual(c["dropped_oversize"], 1)
        self.assertEqual(c["dropped_parse"], 2)
        self.assertEqual(c["rx_frames"], 0)

    def test_garbage_burst_does_not_block_a_good_frame_after_it(self):
        for i in range(200):
            self.br._handle(b"garbage %d" % i)
        self.br._handle(hb())
        self.assertEqual(self.br.link, B.LINK_ONLINE)
        self.assertEqual(self.br.counters["rx_frames"], 1)

    def test_unknown_type_is_ignored_not_an_error(self):
        self.br._handle(json.dumps({"v": 1, "type": "future_thing"}).encode())
        self.assertEqual(self.br.counters["dropped_parse"], 0)
        self.assertEqual(self.br.counters["rx_frames"], 0)

    def test_guard_catches_a_crash_instead_of_killing_the_process(self):
        class Boom(object):
            def read_line(self, timeout=0.2):
                raise RuntimeError("พอร์ตหลุดกลางคัน")
            def write(self, d):
                return False
            def close(self):
                pass

        self.br.transport = Boom()
        self.br._guard()                      # ต้องไม่โยนออกมา
        s = self.br.snapshot()
        self.assertEqual(s["link"], B.LINK_ERROR)
        self.assertIn("พอร์ตหลุดกลางคัน", s["error"])

    def test_notify_exception_does_not_break_ingestion(self):
        def bad(kind, obj):
            raise ValueError("UI พัง")
        self.br.on_event = bad
        self.br._handle(saved("p4-aaaa-000021"))
        self.assertEqual(self.br.counters["events"], 1)


# ---------------------------------------------------------------- state out
class TestStateOut(BridgeCase):
    def test_state_frame_shape(self):
        self.br._send_state()
        st = [m for m in self.sent() if m.get("type") == "state"]
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0]["v"], 1)
        self.assertTrue(st[0]["active_mask_assumed"])
        self.assertEqual(st[0]["active_mask"], 0b0111)
        self.assertEqual(self.br.counters["state_sent"], 1)

    def test_state_send_failure_is_counted_not_raised(self):
        class Dead(B.LoopbackTransport):
            def write(self, d):
                return False
        self.br.transport = Dead()
        self.br._send_state()
        self.assertEqual(self.br.counters["state_failed"], 1)

    def test_state_frame_fits_the_512_byte_limit(self):
        self.state.sample_id = "S" * 60          # ยาวเกินจงใจ
        frame, _ = self.state.build_frame()
        data = P.dumps_line(frame)
        self.assertIsNotNone(data)
        self.assertLessEqual(len(data), P.MAX_LINE)


# ---------------------------------------------------------------- end-to-end
class TestThreadedRun(BridgeCase):
    def test_full_loop_over_transport(self):
        self.br.state_interval = 0.05
        for ln in (hb(), saved("p4-aaaa-000030"), cmd(rid=1),
                   b"junk", hb(mask=0b0011)):
            self.tr.feed(ln)
        self.br.start()
        time.sleep(0.5)
        self.br.stop(1.0)

        s = self.br.snapshot()
        self.assertEqual(s["link"], B.LINK_ONLINE)
        self.assertEqual(s["display_mask"], 0b0011)
        self.assertEqual(s["counters"]["events"], 1)
        self.assertEqual(s["counters"]["cmds_nacked"], 1)
        self.assertEqual(s["counters"]["dropped_parse"], 1)
        self.assertGreaterEqual(s["counters"]["state_sent"], 1)
        self.assertIsNone(s["error"])


if __name__ == "__main__":
    unittest.main()
