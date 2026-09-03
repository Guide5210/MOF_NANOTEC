# -*- coding: utf-8 -*-
"""display_mask — การตีความและการซิงก์ระหว่างจอกับ PC"""
import unittest
from _helpers import ROOT  # noqa
from ecstation.bridge.mask import (MaskSync, mask_to_list, mask_text,
                                   FOLLOWING, OVERRIDE, STALE, CONFLICT, UNKNOWN)
from ecstation.bridge.protocol import Heartbeat


def hb(mask, boot="b1"):
    return Heartbeat(boot_id=boot, device_mono_ms=1, queued=0, link="online",
                     heap=1, heap_big=1, display_mask=mask)


class TestInterpretation(unittest.TestCase):
    def test_all_valid_masks(self):
        for m in range(1, 8):
            s = MaskSync()
            s.on_heartbeat(hb(m))
            self.assertEqual(s.p4_mask, m)
            self.assertEqual(s.effective(), m)
            self.assertEqual(len(mask_to_list(m)), bin(m).count("1"))

    def test_mask_zero_rejected_and_previous_kept(self):
        s = MaskSync(); s.on_heartbeat(hb(6))
        ev = s.on_heartbeat(hb(0))
        self.assertEqual(ev.kind, "REJECTED")
        self.assertEqual(s.p4_mask, 6)          # ใช้ค่าเดิมต่อ

    def test_out_of_range_bits_are_trimmed(self):
        s = MaskSync()
        ev = s.on_heartbeat(hb(0b1111))
        self.assertEqual(s.p4_mask, 0b0111)
        self.assertIn(ev.kind, ("OUT_OF_RANGE", "INITIAL"))

    def test_255_never_becomes_7(self):
        """0xFF ถูกแปลงเป็น None ตั้งแต่ชั้น protocol แล้ว — ที่นี่ต้องไม่เดาต่อ"""
        s = MaskSync()
        ev = s.on_heartbeat(hb(None))
        self.assertEqual(ev.kind, "NONE")
        self.assertIsNone(s.p4_mask)
        self.assertEqual(s.state, UNKNOWN)

    def test_first_mask_is_initial_not_changed(self):
        s = MaskSync()
        self.assertEqual(s.on_heartbeat(hb(6)).kind, "INITIAL")

    def test_change_is_reported_once(self):
        s = MaskSync(); s.on_heartbeat(hb(7))
        ev = s.on_heartbeat(hb(6))
        self.assertEqual(ev.kind, "CHANGED")
        self.assertEqual((ev.old, ev.new), (7, 6))
        self.assertEqual(s.on_heartbeat(hb(6)).kind, "NONE")   # ซ้ำ ไม่รายงาน

    def test_reboot_is_initial_not_user_change(self):
        """บูตใหม่แล้ว mask ต่าง = จอโหลดค่าจาก NVS ไม่ใช่ผู้ใช้กดเปลี่ยน"""
        s = MaskSync(); s.on_heartbeat(hb(7, "b1"))
        ev = s.on_heartbeat(hb(3, "b2"))
        self.assertEqual(ev.kind, "INITIAL")


class TestSyncStateMachine(unittest.TestCase):
    def test_follows_p4_by_default(self):
        s = MaskSync(); s.on_heartbeat(hb(6))
        self.assertEqual(s.state, FOLLOWING)
        self.assertEqual(s.effective(), 6)

    def test_user_override_is_not_overwritten(self):
        s = MaskSync(); s.on_heartbeat(hb(6))
        s.set_pc_mask(7)
        self.assertEqual(s.state, OVERRIDE)
        s.on_heartbeat(hb(6))
        self.assertEqual(s.effective(), 7, "ค่าที่ผู้ใช้ตั้งต้องไม่ถูกทับเงียบ ๆ")

    def test_follow_button_returns_to_p4(self):
        s = MaskSync(); s.on_heartbeat(hb(6)); s.set_pc_mask(7)
        s.follow_p4()
        self.assertEqual(s.state, FOLLOWING)
        self.assertEqual(s.effective(), 6)

    def test_goes_stale_when_p4_silent(self):
        s = MaskSync(); s.on_heartbeat(hb(6))
        s.on_p4_silent()
        self.assertEqual(s.state, STALE)
        self.assertEqual(s.effective(), 6, "ยังใช้ค่าล่าสุดที่รู้")

    def test_conflict_after_reconnect_needs_user(self):
        s = MaskSync(cached_mask=7)
        s.on_heartbeat(hb(6))
        self.assertEqual(s.state, CONFLICT)
        self.assertEqual(s.effective(), 7)      # ยังไม่ทับ รอผู้ใช้ตัดสิน
        s.follow_p4()
        self.assertEqual(s.effective(), 6)

    def test_effective_never_none(self):
        self.assertEqual(MaskSync().effective(), 0b0111)

    def test_ui_text_mentions_both_sides_on_conflict(self):
        s = MaskSync(cached_mask=7); s.on_heartbeat(hb(6))
        t = s.ui_text()
        self.assertIn("02, 03", t)
        self.assertIn("01, 02, 03", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
