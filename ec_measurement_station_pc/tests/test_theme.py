# -*- coding: utf-8 -*-
"""
test_theme — ธีมของ PC ต้องตามหลัง ui_tokens.h ของ P4 เสมอ

⚠️ เทสต์ชุดนี้จงใจ "ตอกค่าไว้" ไม่ใช่ไปอ่านจาก lab_theme มาเทียบกับตัวเอง
   ถ้าอ่านจากไฟล์เดียวกันทั้งสองฝั่ง เทสต์จะผ่านเสมอไม่ว่าใครแก้อะไร
   ค่าที่ตอกไว้ข้างล่างนี้คือค่าใน hello_world/main/ui/ui_tokens.h
   ถ้าฝั่งจอเปลี่ยน ต้องมาแก้ที่นี่ด้วยมือ — ซึ่งเป็นเรื่องที่ควรต้องตั้งใจทำ
"""
import unittest

from _helpers import ROOT  # noqa
from ecstation.ui import lab_theme as T

# ---- ค่าจริงใน ui_tokens.h (คัดมาด้วยมือ) ----
UI_TOKENS = {
    "UI_BG":          "#F4F7F8",
    "UI_SURFACE":     "#FFFFFF",
    "UI_SURFACE_ALT": "#EDF2F4",
    "UI_BORDER":      "#D9E2E6",
    "UI_TEXT":        "#1D2A31",
    "UI_TEXT_DIM":    "#66757D",
    "UI_ACCENT":      "#007C83",
    "UI_OK":          "#008B7A",
    "UI_WARN":        "#B77700",
    "UI_ERROR":       "#B3263E",
    "UI_IDLE":        "#9AA7AE",
}
UI_SERIES = ["#007C83", "#8A5A00", "#4A5FA5"]


class TestTokensMatchP4(unittest.TestCase):
    def test_semantic_tokens(self):
        pairs = [("UI_BG", T.BG), ("UI_SURFACE", T.SURFACE),
                 ("UI_SURFACE_ALT", T.SURFACE_ALT), ("UI_BORDER", T.BORDER),
                 ("UI_TEXT", T.TEXT), ("UI_TEXT_DIM", T.TEXT_DIM),
                 ("UI_ACCENT", T.ACCENT), ("UI_OK", T.OK),
                 ("UI_WARN", T.WARN), ("UI_ERROR", T.ERROR),
                 ("UI_IDLE", T.IDLE)]
        for name, got in pairs:
            self.assertEqual(got.upper(), UI_TOKENS[name],
                             "%s ไม่ตรงกับ ui_tokens.h" % name)

    def test_ok_and_idle_are_the_corrected_values(self):
        """สองค่านี้เคยต่างจากจอ — ล็อกไว้กันถอยกลับ"""
        self.assertEqual(T.OK.upper(), "#008B7A")
        self.assertEqual(T.IDLE.upper(), "#9AA7AE")
        self.assertNotEqual(T.OK.upper(), "#178F7A")
        self.assertNotEqual(T.IDLE.upper(), "#95A2A8")


class TestSeriesColours(unittest.TestCase):
    def test_exactly_three_series_in_p1b(self):
        self.assertEqual(len(T.SENSOR_SERIES), 3)
        self.assertEqual([c.upper() for c in T.SENSOR_SERIES], UI_SERIES)

    def test_series_04_is_reserved_not_used(self):
        """P4 ยังไม่มี UI_SERIES_4 — ห้ามเดาค่าแล้วใช้ไปก่อน"""
        self.assertIsNone(T.SERIES_04_RESERVED)
        self.assertEqual(T.MAX_SENSORS, 3)
        self.assertNotIn("#7A5A87", [c.upper() for c in T.SENSOR_SERIES])

    def test_order_must_match_p4(self):
        """สลับลำดับ = ผู้ใช้ที่ดูจอแล้วหันมาดู PC จับคู่เส้นผิดโดยไม่รู้ตัว"""
        self.assertEqual(T.SENSOR_SERIES[0].upper(), "#007C83")
        self.assertEqual(T.SENSOR_SERIES[1].upper(), "#8A5A00")
        self.assertEqual(T.SENSOR_SERIES[2].upper(), "#4A5FA5")

    def test_amber_and_crimson_are_never_series_colours(self):
        """สีเตือนกับสีผิดพลาดห้ามเป็นสีเส้นปกติ

        ถ้าเส้นเซนเซอร์เป็นสี amber ผู้ใช้จะอ่านว่า 'ตัวนี้มีปัญหา'
        ทั้งที่มันแค่เป็นตัวที่สอง
        """
        banned = {T.WARN.upper(), T.ERROR.upper(), T.REC.upper(),
                  T.WARN_SOFT.upper(), T.ERROR_SOFT.upper()}
        for c in T.SENSOR_SERIES:
            self.assertNotIn(c.upper(), banned)

    def test_identity_and_status_palettes_stay_separate(self):
        """สีตัวตนกับสีสถานะทับกันได้ตัวเดียวคือ ACCENT (teal) ซึ่งตั้งใจ"""
        status = {T.OK.upper(), T.WARN.upper(), T.ERROR.upper(),
                  T.IDLE.upper()}
        overlap = status & {c.upper() for c in T.SENSOR_SERIES}
        self.assertEqual(overlap, set(),
                         "สีสถานะไปโผล่เป็นสีเส้นเซนเซอร์: %s" % overlap)

    def test_series_alias_stays_in_sync(self):
        self.assertIs(T.SERIES, T.SENSOR_SERIES)


class TestStatusMapping(unittest.TestCase):
    def test_every_p4_state_word_exists(self):
        words = ["LIVE", "CHANGING", "STEADY", "OBSERVING", "STABLE", "SAVED",
                 "STALE", "NO RESPONSE", "SENSOR FAULT", "DISABLED", "OFFLINE"]
        for w in words:
            self.assertEqual(T.status_style(w)["label"], w)

    def test_live_uses_accent_and_ok_is_reserved_for_steady(self):
        """คัดจาก sensor_card.c::sensor_state_colour() ของ P4 ตรง ๆ"""
        self.assertEqual(T.status_style("LIVE")["colour"], T.ACCENT)
        self.assertEqual(T.status_style("CHANGING")["colour"], T.ACCENT)
        self.assertEqual(T.status_style("STEADY")["colour"], T.OK)

    def test_steady_and_stable_are_not_the_same_thing(self):
        """STEADY = ค่าสดนิ่งตอนนี้ · STABLE = ผ่านเกณฑ์รอบวัดแล้ว"""
        self.assertNotEqual(T.STEADY, T.STABLE)
        self.assertIn(T.STEADY, T.MONITOR_STATES)
        self.assertIn(T.STABLE, T.RUN_STATES)
        self.assertNotIn(T.STABLE, T.MONITOR_STATES)

    def test_disabled_offline_and_no_response_are_distinct(self):
        for a, b in (("DISABLED", "OFFLINE"), ("OFFLINE", "NO RESPONSE"),
                     ("NO RESPONSE", "SENSOR FAULT")):
            self.assertNotEqual(a, b)
        self.assertTrue(T.status_style("DISABLED")["quiet"])
        self.assertFalse(T.status_style("NO RESPONSE")["quiet"])


class TestNoDarkTheme(unittest.TestCase):
    def _lum(self, hexv):
        h = hexv.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def test_background_is_light(self):
        self.assertGreater(self._lum(T.BG), 0.85)
        self.assertGreater(self._lum(T.SURFACE), 0.95)

    def test_text_has_contrast_on_surface(self):
        self.assertLess(self._lum(T.TEXT), 0.25)


class TestMplRc(unittest.TestCase):
    def test_rc_uses_the_exact_series_order(self):
        try:
            rc = T.apply_mpl_rc()
        except ImportError:
            self.skipTest("ไม่มี matplotlib ในเครื่องนี้")
        colours = [d["color"] for d in rc["axes.prop_cycle"]]
        self.assertEqual([c.upper() for c in colours], UI_SERIES)

    def test_rc_has_no_dark_surface(self):
        self.assertEqual(T.LAB_RC["figure.facecolor"], T.SURFACE)
        self.assertEqual(T.LAB_RC["axes.facecolor"], T.SURFACE)
        self.assertFalse(T.LAB_RC["axes.spines.top"])


if __name__ == "__main__":
    unittest.main()
