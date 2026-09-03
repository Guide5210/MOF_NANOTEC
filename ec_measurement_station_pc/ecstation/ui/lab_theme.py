#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 lab_theme.py — design tokens ของ PC dashboard
============================================================================
 ธีม: light scientific laboratory — ชุดเดียวกับจอสัมผัส ESP32-P4

 ⚠️ ไฟล์นี้คือคู่แฝดของ  hello_world/main/ui/ui_tokens.h
    ทุกสี ทุกระยะ ทุกขนาดตัวอักษรของ PC ต้องมาจากที่นี่ที่เดียว
    ห้ามใส่เลขสีดิบ ๆ ลงใน desktop_ui.py เพราะพอกระจายแล้วจะแก้ธีมไม่ได้
    และหน้าจอสองเครื่องจะค่อย ๆ เพี้ยนจากกันจนดูไม่เหมือนเครื่องมือชิ้นเดียวกัน

 กฎการใช้สี (ยกมาจาก ui_tokens.h ทั้งชุด)
   - สีใช้เพื่อ "สื่อสถานะ" เท่านั้น ไม่ใช้ตกแต่ง
   - ห้ามมี gradient / neon glow / เงาหนัก
   - ทุกสถานะต้องมี "จุด + ข้อความ" เสมอ ห้ามสื่อด้วยสีอย่างเดียว
   - teal  = action หลัก / ข้อมูลกำลังไหล
   - amber = ต้องรอ / ข้อมูลเก่า
   - crimson = อ่านไม่ได้จริง ๆ
   - เทา  = ปิดไว้ / ยังไม่เคยมีข้อมูล  (ไม่ใช่ error)
============================================================================
"""

import math
import os

# เลขเวอร์ชันของชุดธีม — ใช้ยืนยันว่ากำลังรันไฟล์ชุดไหนอยู่จริง
UI_VERSION = "P1-B viewer"

# ============================================================================
#  สี — ตรงกับ ui_tokens.h
# ============================================================================
BG          = "#F4F7F8"   # UI_BG          พื้นหลังหน้าต่าง
SURFACE     = "#FFFFFF"   # UI_SURFACE     พื้น card / chart
SURFACE_ALT = "#EDF2F4"   # UI_SURFACE_ALT card ที่ปิดไว้ / แถบรอง
BORDER      = "#D9E2E6"   # UI_BORDER      เส้นแบ่ง 1 px
TEXT        = "#1D2A31"   # UI_TEXT        ข้อความหลัก / ค่า EC
TEXT_DIM    = "#66757D"   # UI_TEXT_DIM    หน่วย / เวลา / แกนกราฟ

ACCENT      = "#007C83"   # UI_ACCENT      ปุ่มหลัก, LIVE, CHANGING
ACCENT_DEEP = "#00666C"   # สถานะกดปุ่ม teal (ไม่มีใน P4 — PC ต้องมีเพราะใช้เมาส์)
OK          = "#008B7A"   # UI_OK          STEADY / STABLE / SAVED
WARN        = "#B77700"   # UI_WARN        STALE / OBSERVING / ดูย้อนหลัง
ERROR       = "#B3263E"   # UI_ERROR       NO RESPONSE / SENSOR FAULT
IDLE        = "#9AA7AE"   # UI_IDLE        DISABLED / OFFLINE / ค่าที่หมดอายุ

# พื้นอ่อนของแถบแจ้งเตือน — PC มีก่อน P4 (จอยังไม่มี banner)
ACCENT_SOFT = "#DCEFED"
WARN_SOFT   = "#F7ECD3"
ERROR_SOFT  = "#F8E1E6"
ACCENT_LINE = "#A9D6D3"
WARN_LINE   = "#E3C98F"
ERROR_LINE  = "#E8AFBC"

ON_ACCENT = "#FFFFFF"     # ข้อความบนพื้น teal ทึบ (ปุ่มหลัก / ตัวเลือกที่เลือกอยู่)
GRID = "#E3EAED"          # เส้นตารางกราฟ (P4 ไม่มีเส้นตาราง)
REC  = "#B3263E"          # จุด REC — แยก token จาก ERROR เพราะคนละความหมาย

# ----------------------------------------------------------------------------
#  สีเส้นกราฟรายเซนเซอร์
#
#  ⚠️ ลำดับนี้ต้องตรงกับ UI_SERIES_1..3 ใน ui_tokens.h เป๊ะ
#     ถ้าสลับ ผู้ใช้ที่ดูกราฟบนจอแล้วหันมาดู PC จะจับคู่เส้นผิดทันที
#     ตัวที่ 4 ยังไม่มีบน P4 — เมื่อ P4 เพิ่ม ต้องใช้ค่าเดียวกันนี้
# ----------------------------------------------------------------------------
SENSOR_SERIES = [
    "#007C83",   # UI_SERIES_1  teal
    "#8A5A00",   # UI_SERIES_2  น้ำตาลอมส้ม
    "#4A5FA5",   # UI_SERIES_3  น้ำเงินอมม่วง
]

# ตัวที่ 4: P4 ยังไม่มี `UI_SERIES_4` — ห้ามเดาค่าแล้วใช้ไปก่อน
#
# ⚠️ ถ้าใส่สีที่ P4 ยังไม่รับรอง ผู้ใช้ที่ดูกราฟบนจอแล้วหันมาดู PC จะเห็น
#    เส้นที่ 4 คนละสี แล้วจับคู่ผิดโดยไม่รู้ตัว — ซึ่งแย่กว่าไม่มีเส้นที่ 4 เลย
#    ค่าที่เสนอไว้คือ 0x7A5A87 (ดู docs/P4_TOKEN_SYNC.md ฝั่งจอ)
#    เมื่อ P4 นิยามแล้ว ให้ต่อท้าย SENSOR_SERIES ที่นี่ที่เดียว
SERIES_04_RESERVED = None

# ชื่อเดิมที่โค้ดบางส่วนอ้างถึง — คงไว้ให้เท่ากับ SENSOR_SERIES เสมอ
SERIES = SENSOR_SERIES

# ============================================================================
#  ระยะห่าง — 8 px grid เหมือน P4 ห้ามใช้เลขนอกชุดนี้
# ============================================================================
SP_1, SP_2, SP_3, SP_4 = 8, 16, 24, 32
BORDER_W = 1

# ============================================================================
#  ตัวอักษร
#
#  ⚠️ ของเดิมใช้ font=("Sans", ...) ซึ่งเป็น alias ของ X11 — บน Windows ไม่มี
#     Tk จึง fallback เงียบ ๆ หน้าตาที่เห็นจริงเลยไม่ใช่สิ่งที่โค้ดระบุ
#     ตรงนี้จึงเลือกจากรายชื่อฟอนต์ที่ติดตั้งอยู่จริง
# ============================================================================
_BASE_FONT = {"value_xl": 56, "value_l": 40, "value": 34, "value_s": 30,
              "title": 19, "section": 16, "state": 15, "body": 14, "label": 12}

FONT_VALUE_XL = 56   # ค่า EC ในโหมด 1 เซนเซอร์
FONT_VALUE_L  = 40   # ค่า EC ในโหมด 2 เซนเซอร์
FONT_VALUE    = 34   # ค่า EC ในโหมด 3 เซนเซอร์
FONT_VALUE_S  = 30   # ค่า EC ในโหมด 4 เซนเซอร์ (การ์ดเรียงสองแถว)
FONT_TITLE    = 19   # ชื่อโปรแกรม
FONT_SECTION  = 16   # หัวข้อส่วน
FONT_STATE    = 15   # คำสถานะบนการ์ด
FONT_BODY     = 14   # ข้อความทั่วไป / ปุ่ม
FONT_LABEL    = 12   # ป้ายกำกับ / เวลา — เล็กที่สุดที่ยอมให้มี

_FAMILY = None
_MONO = None

# ============================================================================
#  DPI — ต้นเหตุที่ทำให้ layout ถูกบีบบน Windows
# ============================================================================
_SCALE = 1.0          # DPI จริงของจอ หารด้วย 96
_FONT_SCALE = 1.0     # ตัวคูณขนาดตัวอักษร (จำกัดเพดานไว้ ดูเหตุผลข้างล่าง)


def enable_dpi_awareness():
    """บอก Windows ว่าโปรแกรมนี้จัดการ DPI เอง

    ⚠️ ต้องเรียก "ก่อน" สร้าง tk.Tk() เท่านั้น เรียกทีหลังไม่มีผลเลย

    ถ้าไม่เรียก Windows จะให้โปรแกรมวาดที่ 96 DPI แล้ว "ยืดภาพทั้งหน้าต่าง"
    ขึ้นตามสเกลของจอ (125% / 150%) ผลที่ตามมาสองข้อ:

      1. ตัวอักษรและเส้นเบลอ เพราะเป็นการขยายบิตแมป ไม่ใช่วาดใหม่
      2. **Tk เห็นความสูงจอแค่ 2/3 ของจริง** — ที่สเกล 150% จอสูง 1140 px
         จะถูกรายงานเป็น 760 px  พื้นที่ที่ layout มีให้ใช้หายไปหนึ่งในสาม
         ทุกอย่างที่ความสูงตายตัว (การ์ด / แถบปุ่ม / event log) ยังกินที่เท่าเดิม
         สิ่งเดียวที่ยืดหยุ่นได้คือแผงกราฟ มันจึงถูกบีบจนแบนติดกับ toolbar

    เรียกแล้วหน้าต่างจะได้พิกเซลจริงเต็มจำนวน และตัวอักษรคมขึ้นด้วย
    """
    import sys
    if sys.platform != "win32":
        return False
    import ctypes
    for fn in (lambda: ctypes.windll.shcore.SetProcessDpiAwareness(1),
               lambda: ctypes.windll.user32.SetProcessDPIAware()):
        try:
            fn()
            return True
        except Exception:
            continue
    return False


def init_scaling(root):
    """คำนวณตัวคูณจาก DPI จริง แล้วปรับขนาดตัวอักษรทั้งชุด

    ⚠️ ตัวคูณของ "ตัวอักษร" จำกัดเพดานไว้ที่ 1.25 โดยตั้งใจ

    ถ้าคูณเต็ม 1.5 ตามสเกลจอ ความสูงที่ layout ต้องใช้ก็คูณ 1.5 ตามไปด้วย
    แล้วเราจะกลับไปเจอปัญหาเดิมทันทีบนจอที่สูงไม่พอ  ที่ 1.25 ตัวอักษรยัง
    ใหญ่พอสำหรับจอ 150% และเหลือที่ให้กราฟจริง ๆ

    ระยะห่าง (SP_*) ไม่คูณเลย เพราะมันเป็นงบพื้นที่ ไม่ใช่เรื่องการอ่านออก
    """
    global _SCALE, _FONT_SCALE
    global FONT_VALUE_XL, FONT_VALUE_L, FONT_VALUE, FONT_VALUE_S
    global FONT_TITLE, FONT_SECTION, FONT_STATE, FONT_BODY, FONT_LABEL
    try:
        _SCALE = root.winfo_fpixels("1i") / 96.0
    except Exception:
        _SCALE = 1.0
    _FONT_SCALE = min(max(_SCALE, 1.0), 1.25)

    k = _FONT_SCALE
    FONT_VALUE_XL = int(round(_BASE_FONT["value_xl"] * k))
    FONT_VALUE_L  = int(round(_BASE_FONT["value_l"]  * k))
    FONT_VALUE    = int(round(_BASE_FONT["value"]    * k))
    FONT_VALUE_S  = int(round(_BASE_FONT["value_s"]  * k))
    FONT_TITLE    = int(round(_BASE_FONT["title"]    * k))
    FONT_SECTION  = int(round(_BASE_FONT["section"]  * k))
    FONT_STATE    = int(round(_BASE_FONT["state"]    * k))
    FONT_BODY     = int(round(_BASE_FONT["body"]     * k))
    FONT_LABEL    = int(round(_BASE_FONT["label"]    * k))
    return _SCALE


def scale():
    return _SCALE


def font_scale():
    return _FONT_SCALE


def _pick(cands, default):
    try:
        from tkinter import font as tkfont
        avail = {f.lower() for f in tkfont.families()}
    except Exception:
        return default
    for c in cands:
        if c.lower() in avail:
            return c
    return default


def font_family():
    """ฟอนต์หลัก — ต้องมีอักษรไทยด้วย เพราะข้อความช่วยเหลือยังเป็นไทย"""
    global _FAMILY
    if _FAMILY is None:
        _FAMILY = _pick(["Segoe UI", "Noto Sans Thai", "Noto Sans",
                         "Ubuntu", "DejaVu Sans", "Arial"], "TkDefaultFont")
    return _FAMILY


def mono_family():
    """ตัวเลขต้องกว้างเท่ากันทุกตัว ไม่งั้นค่าที่เปลี่ยนจะทำให้ layout ขยับ"""
    global _MONO
    if _MONO is None:
        _MONO = _pick(["Consolas", "DejaVu Sans Mono", "Noto Sans Mono",
                       "Courier New"], "TkFixedFont")
    return _MONO


# ⚠️ ขนาดข้างบนเป็น "พิกเซล" ไม่ใช่พอยต์
#    Tk ตีความเลขบวกเป็นพอยต์ ซึ่งที่ 96 DPI จะใหญ่กว่าที่ระบุราว 33%
#    (34 -> ~45 px) ทำให้การ์ดสูงเกินงบจนกราฟไม่เหลือที่ และไม่ตรงกับสเปกที่
#    กำหนดเป็นพิกเซลไว้  เลขติดลบคือหน่วยพิกเซลตามสเปกของ Tk
def f(size, weight="normal"):
    """ชุดฟอนต์ปกติ — size เป็นพิกเซล"""
    return (font_family(), -abs(int(size)), weight)


def fm(size, weight="normal"):
    """ชุดฟอนต์ตัวเลข (tabular) — size เป็นพิกเซล"""
    return (mono_family(), -abs(int(size)), weight)


# ============================================================================
#  นโยบายการตัดสินสถานะ
# ----------------------------------------------------------------------------
#  ⚠️ ตัวเลขทุกตัวในบล็อกนี้คัดลอกมาจาก measure_cfg_default() ใน
#     hello_world/main/ui/measure_policy.c  ห้ามเดาเอง
#     ถ้าฝั่งจอเปลี่ยน ต้องแก้ที่นี่ด้วย ไม่งั้นสองเครื่องจะบอกสถานะไม่ตรงกัน
#     ทั้งที่ดูข้อมูลชุดเดียวกัน ซึ่งแย่กว่าการไม่บอกสถานะเลย
# ============================================================================
POLL_INTERVAL_S       = 2.5     # cfg->poll_interval_ms    = 2500
MISSED_SAMPLE_LIMIT   = 3       # cfg->missed_sample_limit = 3
JITTER_MARGIN_S       = 1.25    # max(0.5, poll/2)
STALE_AFTER_S         = POLL_INTERVAL_S * MISSED_SAMPLE_LIMIT + JITTER_MARGIN_S  # 8.75
STABLE_WINDOW_SAMPLES = 6       # cfg->stable_window_samples = 6
STABLE_ABS_TOL        = 2.0     # cfg->stable_abs_tol_us_cm  = 2.0
STABLE_REL_TOL        = 0.01    # cfg->stable_rel_tol        = 0.01

# DEAD_AFTER_FAILS ในเฟิร์มแวร์ CONTROL (water_monitor_3ec.ino:327)
FAULT_AFTER_FAILS = 3

MAX_SENSORS = len(SENSOR_SERIES)   # 3 — ผูกกับจำนวนสีที่ P4 รับรองจริง


def tolerance_for(avg):
    """win_tolerance() — นิ่งเมื่อ max-min <= max(abs_tol, rel_tol x ค่าเฉลี่ย)

    เกณฑ์เดี่ยวใช้ไม่ได้ทั้งช่วง: EC 10 ยอมแกว่ง 15 คือกว้างกว่าค่าจริง
    ส่วน EC 5,000 ยอมแกว่ง 15 คือเข้มจนแทบไม่มีวันนิ่ง
    """
    return max(STABLE_ABS_TOL, STABLE_REL_TOL * abs(avg))


# ============================================================================
#  สถานะ
#
#  ⚠️ สองชุดนี้ต้องแยกกันเสมอ ห้ามยุบเป็น dict เดียว
#     STEADY = ค่าสดนิ่งอยู่ตอนนี้ (เฝ้าดูเฉย ๆ)
#     STABLE = ผ่านเกณฑ์ของรอบวัดจริงแล้ว (มีผลการวัด)
#     ถ้าใช้สลับกัน ผู้ใช้จะเข้าใจว่าค่าที่เห็นเป็นผลการวัดทั้งที่ไม่ใช่
# ============================================================================
LIVE, CHANGING, STEADY   = "LIVE", "CHANGING", "STEADY"
STALE, NO_RESPONSE       = "STALE", "NO RESPONSE"
SENSOR_FAULT             = "SENSOR FAULT"
DISABLED, OFFLINE        = "DISABLED", "OFFLINE"

OBSERVING, STABLE, SAVED = "OBSERVING", "STABLE", "SAVED"

#   colour     = สีของจุดและคำสถานะ
#   dim_value  = ค่า EC ต้องหรี่ไหม (ค่าที่เห็นไม่ใช่ค่าปัจจุบันแล้ว)
#   show_value = มีค่าให้แสดงไหม
#   quiet      = พื้นการ์ดเป็นสีเทาไหม (ปิดไว้โดยตั้งใจ ไม่ใช่ error)
MONITOR_STATES = {
    LIVE:         {"colour": ACCENT, "dim_value": False, "show_value": True,  "quiet": False},
    CHANGING:     {"colour": ACCENT, "dim_value": False, "show_value": True,  "quiet": False},
    STEADY:       {"colour": OK,     "dim_value": False, "show_value": True,  "quiet": False},
    STALE:        {"colour": WARN,   "dim_value": True,  "show_value": True,  "quiet": False},
    NO_RESPONSE:  {"colour": ERROR,  "dim_value": True,  "show_value": False, "quiet": False},
    SENSOR_FAULT: {"colour": ERROR,  "dim_value": True,  "show_value": False, "quiet": False},
    DISABLED:     {"colour": IDLE,   "dim_value": True,  "show_value": False, "quiet": True},
    OFFLINE:      {"colour": IDLE,   "dim_value": True,  "show_value": False, "quiet": True},
}

RUN_STATES = {
    OBSERVING: {"colour": WARN},
    STABLE:    {"colour": OK},
    SAVED:     {"colour": OK},
}

# เหตุผลที่อ่านไม่ได้ ต้องบอกให้ผู้ใช้รู้ว่าไปแก้ตรงไหน
# คำว่า ERROR เฉย ๆ ไม่ช่วยอะไรกับคนที่ยืนอยู่หน้าเครื่อง
STATE_HINT = {
    DISABLED:     "Excluded from polling by configuration",
    OFFLINE:      "No data yet",
    NO_RESPONSE:  "No reply this cycle",
    SENSOR_FAULT: "No reply for {n} cycles - check the probe wiring",
}


def status_style(state):
    """คืนชุดการแสดงผลของสถานะหนึ่ง — จุดเดียวที่แปลงสถานะเป็นสี"""
    st = MONITOR_STATES.get(state) or RUN_STATES.get(state)
    if st is None:
        st = MONITOR_STATES[OFFLINE]
    out = dict(st)
    out.setdefault("dim_value", True)
    out.setdefault("show_value", False)
    out.setdefault("quiet", False)
    out["label"] = state
    return out


def monitor_state(window_vals, window_ok, age_s, enabled=True):
    """ตัดสินสถานะของ sensor หนึ่งตัว — มิเรอร์ monitor_state() ของ P4

    window_vals / window_ok = ค่าล่าสุดเรียงเก่า -> ใหม่ (ไม่เกินขนาดหน้าต่าง)
    age_s = อายุของแถวล่าสุด (วินาที) หรือ None ถ้ายังไม่เคยมีข้อมูลเลย

    ลำดับการตัดสินต้องเหมือนฝั่งจอเป๊ะ ไม่งั้นสองเครื่องจะบอกคนละอย่าง
    """
    if not enabled:
        return DISABLED
    if age_s is None or not window_ok:
        return OFFLINE
    if age_s > STALE_AFTER_S:
        return STALE

    if not window_ok[-1]:
        # นับว่าพลาดติดกันมากี่รอบ — พลาดรอบเดียวยังไม่ใช่หัววัดเสีย
        fails = 0
        for ok in reversed(window_ok):
            if ok:
                break
            fails += 1
        return SENSOR_FAULT if fails >= FAULT_AFTER_FAILS else NO_RESPONSE

    vals = [v for v in window_vals if v is not None]
    if len(vals) < STABLE_WINDOW_SAMPLES:
        return LIVE          # ยังเก็บไม่ครบหน้าต่าง บอกได้แค่ว่ามีค่าสด

    win = vals[-STABLE_WINDOW_SAMPLES:]
    avg = sum(win) / len(win)
    return STEADY if (max(win) - min(win)) <= tolerance_for(avg) else CHANGING


def consecutive_fails(window_ok):
    """จำนวนรอบที่พลาดติดกันล่าสุด — ใช้เติมข้อความ SENSOR_FAULT"""
    n = 0
    for ok in reversed(window_ok or []):
        if ok:
            break
        n += 1
    return n


def state_hint(state, fails=0):
    tpl = STATE_HINT.get(state)
    return tpl.format(n=max(fails, FAULT_AFTER_FAILS)) if tpl else ""


# ============================================================================
#  การจัดรูปแบบตัวเลขและเวลา
# ============================================================================
NO_VALUE = "— — —"    # "— — —"


def format_ec(v, decimals=1):
    """ค่า EC พร้อมตัวคั่นหลักพัน

    "1,362.4" อ่านเร็วกว่า "1362.4" มากเมื่อมองผ่าน ๆ (เหตุผลเดียวกับ
    format_ec() ใน sensor_card.c) แต่ PC เก็บทศนิยม 1 ตำแหน่งไว้ ต่างจากจอ
    ที่ปัดเป็นจำนวนเต็ม เพราะ CSV / Excel / PDF ของ PC ใช้ 1 ตำแหน่งทั้งหมด
    ถ้าจอ PC ปัดเป็นจำนวนเต็ม ตัวเลขบนจอจะไม่ตรงกับรายงานของตัวเอง
    ซึ่งอันตรายกว่าการต่างจากจอสัมผัส
    """
    if v is None:
        return NO_VALUE
    try:
        v = float(v)
    except (TypeError, ValueError):
        return NO_VALUE
    if math.isnan(v):
        return NO_VALUE
    return "{:,.{d}f}".format(v, d=decimals)


def format_temp(v):
    if v is None:
        return ""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v):
        return ""
    return "{:.1f} °C".format(v)


def format_freshness(age_s):
    """ถ้อยคำเดียวกับ format_age() ใน sensor_card.c ทุกตัวอักษร"""
    if age_s is None:
        return "No data yet"
    if age_s < 1.5:
        return "Updated now"
    if age_s < 60:
        return "Updated {}s ago".format(int(age_s))
    m = int(age_s // 60)
    return "Updated {}m ago".format(min(m, 999))


def sensor_card_style(state):
    """สีพื้น / ขอบ / ตัวเลข ของการ์ดหนึ่งใบ"""
    st = status_style(state)
    quiet = st["quiet"]
    return {
        "bg":        SURFACE_ALT if quiet else SURFACE,
        "border":    BORDER,
        "state_fg":  st["colour"],
        "value_fg":  IDLE if st["dim_value"] else TEXT,
        "show_value": st["show_value"],
    }


def summary_text(counts):
    """สรุปบน header — สูตรเดียวกับ screen_overview_update() ของ P4

    counts = dict ของจำนวนแต่ละกลุ่ม
    """
    parts = ["{} SENSORS".format(counts.get("total", 0))]
    for key, word in (("steady", "STEADY"), ("live", "LIVE"), ("stale", "STALE"),
                      ("fault", "FAULT"), ("disabled", "DISABLED"),
                      ("offline", "OFFLINE")):
        n = counts.get(key, 0)
        if n:
            parts.append("{} {}".format(n, word))
    return "  •  ".join(parts)


def grid_for(n):
    """(rows, cols) ของการ์ดตามจำนวน sensor ที่เปิดใช้"""
    if n <= 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n == 3:
        return (1, 3)
    return (2, 2)


def value_font_size(n, density="roomy"):
    """ค่า EC ใหญ่ขึ้นเมื่อมี sensor น้อย — ใช้พื้นที่ที่ว่างให้เป็นประโยชน์

    ที่ 4 ตัวต้องเล็กลงอีกขั้น เพราะการ์ดเรียงสองแถวแล้วกินความสูงเป็นสองเท่า
    ถ้าไม่ลด กราฟจะไม่เหลือที่เลย
    """
    if n <= 1:
        size = FONT_VALUE_XL
    elif n == 2:
        size = FONT_VALUE_L
    elif n == 3:
        size = FONT_VALUE
    else:
        size = FONT_VALUE_S
    # จอเตี้ย: ย่อตัวเลขลงอีกขั้น เพื่อคืนความสูงให้กราฟ
    if density in ("tight", "min"):
        size = int(round(size * 0.82))
    return size


def mask_to_list(mask, maximum=MAX_SENSORS):
    return [i for i in range(maximum) if (mask >> i) & 1]


# ============================================================================
#  ttk
#
#  ⚠️ ต้องใช้ธีม "clam" เท่านั้น
#     ธีม vista / xpnative บน Windows วาดด้วย native renderer ซึ่งไม่สนใจ
#     ค่าสีที่เราตั้ง Combobox จึงยังเป็นกล่องขาวของระบบโดดออกมาจากธีม
# ============================================================================
def apply_ttk_theme(root):
    from tkinter import ttk
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    body = f(FONT_BODY)

    style.configure("Lab.TCombobox",
                    font=body,
                    fieldbackground=SURFACE, background=SURFACE,
                    foreground=TEXT, arrowcolor=TEXT_DIM,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    selectbackground=SURFACE, selectforeground=TEXT,
                    padding=4)
    style.map("Lab.TCombobox",
              fieldbackground=[("readonly", SURFACE), ("disabled", SURFACE_ALT)],
              foreground=[("disabled", IDLE)],
              bordercolor=[("focus", ACCENT)],
              arrowcolor=[("disabled", IDLE)])

    # รายการที่หล่นลงมาเป็น Tk listbox ธรรมดา ตั้งสีผ่าน option database เท่านั้น
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)
    root.option_add("*TCombobox*Listbox.font", body)

    style.configure("Lab.Vertical.TScrollbar",
                    background=SURFACE_ALT, troughcolor=BG,
                    bordercolor=BORDER, arrowcolor=TEXT_DIM,
                    lightcolor=SURFACE_ALT, darkcolor=SURFACE_ALT)
    return style


# ============================================================================
#  matplotlib
# ----------------------------------------------------------------------------
#  ⚠️ ห้ามแตะ rcParams ระดับ global เด็ดขาด
#     report_3ec.py สร้าง PDF ด้วย matplotlib ตัวเดียวกันในโปรเซสเดียวกัน
#     (desktop_ui เรียก generate_pdf_3ec ตอนกด Export) ถ้าเราตั้ง rcParams
#     ทิ้งไว้ เช่น ปิด spine บน/ขวา หรือเปลี่ยนสี grid หน้าตาของรายงานที่ส่ง
#     อาจารย์จะเปลี่ยนไปด้วยโดยไม่มีใครตั้งใจ — ซึ่งอยู่นอกขอบเขตงานรอบนี้
#     จึงตั้ง style ทีละ figure/axes ของเราเองเท่านั้น
# ============================================================================
# ----------------------------------------------------------------------------
#  rcParams กลาง — ตั้งครั้งเดียวตอนแอปเริ่ม
#
#  ⚠️ ในโปรเจกต์ "เดิม" ห้ามแตะ rcParams เด็ดขาด เพราะ report_3ec.py แชร์
#     โปรเซสเดียวกัน การเปลี่ยน rcParams จะทำให้ PDF/Excel ที่ออกมาหน้าตา
#     เปลี่ยนไปโดยไม่มีใครสั่ง — ซึ่งเป็นการแก้ผลลัพธ์งานวิจัยโดยอุบัติเหตุ
#
#     ในโปรเจกต์ "นี้" ไม่มีตัวสร้างรายงานอยู่ในโปรเซสเลย (บังคับด้วย
#     tests/test_no_legacy_mutation.py ที่ห้าม import report_3ec / openpyxl /
#     reportlab) จึงตั้ง rcParams กลางได้อย่างปลอดภัย และเป็นวิธีที่ถูกต้อง
#     กว่าการไล่ style ทีละแกน เพราะกราฟที่เพิ่มทีหลังจะได้ธีมเดียวกันอัตโนมัติ
# ----------------------------------------------------------------------------
LAB_RC = {
    "figure.facecolor":  SURFACE,
    "figure.edgecolor":  SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.linewidth":    0.8,
    "axes.labelcolor":   TEXT_DIM,
    "axes.titlecolor":   TEXT,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.prop_cycle":   None,          # เติมตอน apply (ต้อง import cycler)
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
    "grid.alpha":        1.0,
    "text.color":        TEXT,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "xtick.labelcolor":  TEXT_DIM,
    "ytick.labelcolor":  TEXT_DIM,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "legend.framealpha": 1.0,
    "legend.fontsize":   9,
    "lines.linewidth":   1.6,
    "lines.solid_capstyle": "round",
}


def apply_mpl_rc():
    """ตั้งธีม matplotlib ทั้งโปรเซส — เรียกครั้งเดียวใน app.py"""
    import matplotlib
    from cycler import cycler
    rc = dict(LAB_RC)
    rc["axes.prop_cycle"] = cycler(color=list(SENSOR_SERIES))
    matplotlib.rcParams.update(rc)
    return rc


def mpl_style_figure(fig):
    fig.patch.set_facecolor(SURFACE)
    return fig


def mpl_style_axes(ax, labelbottom=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_DIM, labelsize=9, length=3, width=0.8,
                   labelbottom=labelbottom)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(TEXT_DIM)
    return ax


def mpl_style_legend(leg):
    if leg is None:
        return None
    fr = leg.get_frame()
    fr.set_facecolor(SURFACE)
    fr.set_edgecolor(BORDER)
    fr.set_linewidth(0.8)
    for txt in leg.get_texts():
        txt.set_color(TEXT_DIM)
        txt.set_fontsize(9)
    return leg


# ============================================================================
#  marker ของเหตุการณ์บนกราฟ — จองไว้ให้ Phase 3b เสียบเข้ามา
#  (รอบนี้ยังไม่วาด แต่นิยามไว้ที่เดียวเพื่อไม่ต้องมาแก้ styling อีก)
# ============================================================================
MARKER_STYLES = {
    "cal":    {"kind": "span",  "facecolor": WARN_SOFT,  "edgecolor": WARN_LINE},
    "fault":  {"kind": "span",  "facecolor": ERROR_SOFT, "edgecolor": ERROR_LINE},
    "saved":  {"kind": "point", "color": OK,     "marker": "o", "size": 28},
    "stable": {"kind": "vline", "color": OK,     "linestyle": ":"},
    "session": {"kind": "vline", "color": ACCENT, "linestyle": "--"},
}


# ============================================================================
#  ค่าตั้งของ UI  (display-only — ไม่ใช่ config ของฮาร์ดแวร์)
# ----------------------------------------------------------------------------
#  ⚠️ active_mask ในไฟล์นี้ "ไม่" ไปสั่งอะไรกับบอร์ดเลย
#     CONTROL ยังคง poll ครบทุกตัวเหมือนเดิม (จำเป็นด้วย เพราะจอ P4 ใช้การเห็น
#     คำถามถึง address 1 เป็นตัวตัดรอบ ถ้าข้ามจะค้าง STALE ทั้งเครื่อง)
#     ที่นี่เป็นแค่การบอกว่า "หน้าจอ PC ไม่ต้องแสดงตัวนี้"
#     ⇒ ปลอดภัยกับข้อมูล 100% และย้อนกลับได้ด้วยการแก้ตัวเลขเดียว
# ============================================================================
# ⚠️ ค่าตั้งหน้าจอต้องอยู่ใน data/ ของโปรเจกต์นี้เท่านั้น
#    เขียนข้าง ๆ ซอร์สจะทำให้ไฟล์นี้หลุดเข้า git และชนกับสำเนาอื่น
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(_PKG_ROOT, "data")
UI_CONFIG_FILE = os.path.join(BASE_DIR, "ui_state.json")

UI_CONFIG_DEFAULT = {
    "active_mask":  0b0111,
    "sensor_names": ["SENSOR 01", "SENSOR 02", "SENSOR 03", "SENSOR 04"],
    # null = คำนวณจากขนาดจอจริงตอนเปิด (แนะนำให้ใช้ null)
    "window":       None,
    "chart_mode":   "split",
    "ec_decimals":  1,
}


# ============================================================================
#  ความสูงคงที่ของแต่ละส่วน (พิกเซลจริง)
#
#  ⚠️ ส่วนพวกนี้ต้อง "ตายตัว" ไม่ใช่ปล่อยให้โตตามเนื้อหา
#     ไม่งั้นเมื่อพื้นที่ไม่พอ Tk จะไปบีบสิ่งที่ยืดหยุ่นได้ตัวเดียว คือแผงกราฟ
#     ซึ่งเป็นสิ่งที่ผู้ใช้ต้องดูมากที่สุด — ตรงกันข้ามกับที่ควรเป็น
# ============================================================================
CARD_H_WIDE    = 120   # 1 เซนเซอร์ — การ์ดแนวนอน
CARD_H_NORMAL  = 172   # 2-3 เซนเซอร์ — แถวเดียว
CARD_H_COMPACT = 136   # 4 เซนเซอร์ — สองแถว
EVENT_ROW_H    = 30
EVENT_HEAD_H   = 34
CHART_MIN_H    = 170   # ความสูงขั้นต่ำของ canvas กราฟ


# ----------------------------------------------------------------------------
#  ความหนาแน่นของหน้าจอ ตามความสูงหน้าต่างจริง
#
#  ⚠️ ลำดับการยอมเสียสละต้องเป็นแบบนี้เท่านั้น:
#        event log  ->  ความสูงการ์ด  ->  (ห้ามแตะกราฟ)
#     ของเดิมกลับหัว: ทุกอย่างความสูงตายตัว เหลือกราฟตัวเดียวที่ยืดหยุ่นได้
#     พอพื้นที่ไม่พอ กราฟจึงโดนบีบจนแบน ทั้งที่เป็นสิ่งที่ผู้ใช้ต้องดูที่สุด
# ----------------------------------------------------------------------------
def density_for(window_h):
    if window_h >= 900:
        return "roomy"
    if window_h >= 800:
        return "normal"
    if window_h >= 720:
        return "tight"
    return "min"


EVENT_ROWS_BY_DENSITY = {"roomy": 3, "normal": 2, "tight": 1, "min": 0}


def event_rows_for(window_h):
    return EVENT_ROWS_BY_DENSITY[density_for(window_h)]


def card_height_for(n_active, rows, density="roomy"):
    if n_active <= 1:
        return CARD_H_WIDE
    if rows > 1:
        return CARD_H_COMPACT
    # แถวเดียวแต่จอเตี้ย: ใช้ความสูงแบบกระชับ คืนพื้นที่ให้กราฟ
    return CARD_H_COMPACT if density in ("tight", "min") else CARD_H_NORMAL


def load_ui_config():
    import json
    cfg = dict(UI_CONFIG_DEFAULT)
    try:
        with open(UI_CONFIG_FILE, encoding="utf-8") as fh:
            user = json.load(fh)
        if isinstance(user, dict):
            cfg.update({k: v for k, v in user.items() if k in UI_CONFIG_DEFAULT})
    except FileNotFoundError:
        pass
    except Exception as e:
        print("[ui] ui_state.json อ่านไม่ได้ ({}) - ใช้ค่าเริ่มต้น".format(e))

    # กันค่าที่ทำให้หน้าจอว่างเปล่าจนกดอะไรไม่ได้
    mask = int(cfg.get("active_mask", 0b0111)) & ((1 << MAX_SENSORS) - 1)
    cfg["active_mask"] = mask or 0b0111
    names = cfg.get("sensor_names") or []
    cfg["sensor_names"] = [
        (names[i] if i < len(names) and names[i] else "SENSOR {:02d}".format(i + 1))
        for i in range(MAX_SENSORS)
    ]
    if cfg.get("chart_mode") not in ("split", "overlay"):
        cfg["chart_mode"] = "split"
    return cfg


def save_ui_config(cfg):
    import json
    try:
        keep = {k: cfg[k] for k in UI_CONFIG_DEFAULT if k in cfg}
        os.makedirs(BASE_DIR, exist_ok=True)
        tmp = UI_CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(keep, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, UI_CONFIG_FILE)
        return True
    except Exception as e:
        print("[ui] เขียน ui_state.json ไม่ได้: {}".format(e))
        return False
