#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 view_model.py — ตรรกะทั้งหมดของหน้าจอ โดยไม่มี tkinter สักบรรทัด
============================================================================
 ทำไมต้องแยกออกมา
 -----------------
 เงื่อนไขการยอมรับของ P1-B เป็นเรื่อง "พฤติกรรม" ไม่ใช่ "พิกเซล":
   mask 7->6->2->7 ต้องไม่มีเส้นค้าง · event ซ้ำ 100 ครั้งต้องเก็บครั้งเดียว ·
   P4 หลุดต้องคงค่าที่เห็นล่าสุด  ถ้าตรรกะพวกนี้ฝังอยู่ในโค้ด Tkinter
   จะทดสอบได้แค่ด้วยตาคน ซึ่งพิสูจน์อะไรไม่ได้และถอยกลับไม่ได้ด้วย

 ⚠️ ไฟล์นี้ไม่เขียนไฟล์ใด ๆ ทั้งสิ้น  มันคำนวณอย่างเดียว
============================================================================
"""

from datetime import datetime

from . import lab_theme as T

MAX_SENSORS = T.MAX_SENSORS
FULL_MASK = (1 << MAX_SENSORS) - 1

# ป้ายที่ต้องขึ้นเมื่อจอหลุด — ข้อความนี้เป็นสัญญากับผู้ใช้ อย่าเปลี่ยนเงียบ ๆ
P4_OFFLINE_MASK_NOTE = "P4 OFFLINE — showing last HMI selection"
HISTORY_NOTE = "HISTORY VIEW — LIVE FOLLOW PAUSED"

SRC_P4 = "p4"
SRC_PC = "pc"

# เหตุการณ์ที่ผู้ใช้ควรเห็นบน dashboard  ส่วนที่เหลือไปอยู่ Diagnostics
EVENT_TEXT = {
    "reading_saved":      "READING SAVED",
    "STABILITY_REACHED":  "STABILITY REACHED",
    "STABILITY_LOST":     "STABILITY LOST",
    "LINK_ERROR":         "LINK ERROR",
    "P4_REBOOT":          "P4 RESTARTED",
    "DISPLAY_MASK_CHANGED":  "HMI SELECTION CHANGED",
    "DISPLAY_MASK_INITIAL":  "HMI SELECTION RECEIVED",
    "CMD_REJECTED":       "COMMAND REJECTED BY PC",
}

# เหตุการณ์ที่เป็นเรื่องภายใน — ไม่ขึ้น dashboard แต่ยังอยู่ใน Diagnostics
DIAGNOSTIC_ONLY = ("DISPLAY_MASK_REJECTED", "DISPLAY_MASK_OUT_OF_RANGE")


class EventFeed(object):
    """รายการเหตุการณ์ตามเวลา — ต่อท้ายอย่างเดียว และกันซ้ำด้วย event_id

    ⚠️ ต้องเก็บของตัวเองไว้ ไม่ใช่ดึงจาก bridge ทุกครั้งที่วาด
       เพราะเมื่อจอหลุด ผู้ใช้ยังต้องเห็นสิ่งที่เกิดขึ้นก่อนหน้า
       การล้างรายการตอนสายหลุดคือการทำลายหลักฐานของงานที่เพิ่งทำไป
    """

    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.rows = []
        self._seen = set()
        self.duplicates = 0

    def add(self, kind, when=None, sensor=0, detail="", value=None,
            source=SRC_P4, event_id=None, extra=None):
        if event_id is not None:
            if event_id in self._seen:
                self.duplicates += 1
                return False
            self._seen.add(event_id)
        self.rows.append({
            "when": when or datetime.now(),
            "kind": kind,
            "text": EVENT_TEXT.get(kind, kind.replace("_", " ")),
            "sensor": int(sensor or 0),
            "value": value,
            "detail": detail,
            "source": source,
            "event_id": event_id,
            "extra": extra or {},
        })
        if len(self.rows) > self.maxlen:
            del self.rows[:len(self.rows) - self.maxlen]
        return True

    # -- ตัวแปลงจากเฟรมของ bridge -------------------------------------
    def add_reading_saved(self, fr):
        """reading_saved ต้องแสดง stable_ec_us_cm เสมอ ไม่ใช่ค่าสดตอนนั้น"""
        return self.add(
            "reading_saved", sensor=fr.sensor, event_id=fr.event_id,
            value=fr.stable_ec_us_cm,
            detail="±{} µS/cm · stable {:.1f} s".format(
                T.format_ec(fr.tolerance_us_cm), (fr.stable_for_ms or 0) / 1000.0),
            extra={"temperature_c": fr.temperature_c,
                   "tolerance_us_cm": fr.tolerance_us_cm,
                   "stable_for_ms": fr.stable_for_ms,
                   "boot_id": fr.boot_id})

    def add_context(self, fr):
        return self.add(fr.event, sensor=fr.sensor, event_id=fr.event_id,
                        value=fr.ec_us_cm, extra={"boot_id": fr.boot_id})

    def visible(self, limit=None):
        """แถวที่ควรขึ้น dashboard — ใหม่สุดอยู่บน"""
        rows = [r for r in self.rows if r["kind"] not in DIAGNOSTIC_ONLY]
        rows = list(reversed(rows))
        return rows[:limit] if limit else rows


class SensorView(object):
    """สิ่งที่การ์ดหนึ่งใบต้องรู้ — ไม่มีอะไรเกี่ยวกับวิดเจ็ต"""

    __slots__ = ("index", "name", "colour", "state", "style", "ec", "temp",
                 "age_s", "hidden", "hint")

    def __init__(self, index, name, colour, state, style, ec, temp, age_s,
                 hidden, hint):
        self.index, self.name, self.colour = index, name, colour
        self.state, self.style = state, style
        self.ec, self.temp, self.age_s = ec, temp, age_s
        self.hidden, self.hint = hidden, hint

    @property
    def number(self):
        return self.index + 1

    def ec_text(self, decimals=1):
        return T.format_ec(self.ec if self.style["show_value"] else None, decimals)

    def freshness_text(self):
        return T.format_freshness(self.age_s)


class DashboardModel(object):
    """สถานะทั้งหน้าจอ ณ วินาทีหนึ่ง — UI แค่วาดสิ่งที่นี่บอก"""

    def __init__(self, ui_cfg=None):
        cfg = ui_cfg or {}
        self.events = EventFeed()
        self.engineering = False        # เผยเซนเซอร์ที่ถูกซ่อนไว้
        self.chart_mode = cfg.get("chart_mode", "split")
        self.window_minutes = cfg.get("window")
        self.ec_decimals = int(cfg.get("ec_decimals", 1))
        self.sensor_names = list(cfg.get("sensor_names") or
                                 ["SENSOR {:02d}".format(i + 1)
                                  for i in range(MAX_SENSORS)])
        self.history_paused = False
        self._pc_mask_override = None   # ผู้ใช้เลือกเองที่ PC (ยังไม่มี UI ในเฟสนี้)

    # ------------------------------------------------------------------
    #  mask ที่ใช้แสดงจริง
    # ------------------------------------------------------------------
    def resolve_mask(self, bridge_snap):
        """คืน (mask, note) — note = ข้อความที่ต้องขึ้นให้ผู้ใช้เห็น หรือ None

        ⚠️ กฎที่ห้ามละเมิด: จอหลุดแล้ว **ห้ามรีเซ็ตเป็นครบทุกตัว และห้ามซ่อนการ์ด**
           สองอย่างนั้นทำให้ผู้ใช้เชื่อว่าการเลือกของตัวเองหายไป แล้วไปกดตั้งใหม่
           ทั้งที่จอยังจำค่าเดิมอยู่ — พอจอกลับมาจะได้ค่าที่ขัดกันสองชุด
           ค่าที่ถูกต้องคือ "ค่าล่าสุดที่รู้ พร้อมป้ายบอกว่ามันเก่าแล้ว"
        """
        if self._pc_mask_override:
            return self._pc_mask_override, None

        link = (bridge_snap or {}).get("link")
        p4 = (bridge_snap or {}).get("display_mask")
        view = (bridge_snap or {}).get("view_mask")

        if link == "ONLINE" and p4:
            return p4 & FULL_MASK, None
        if p4:                                   # เคยรู้ค่า แต่ตอนนี้จอไม่ตอบ
            return p4 & FULL_MASK, P4_OFFLINE_MASK_NOTE
        if view:                                 # ยังไม่เคยได้ mask จากจอเลย
            return view & FULL_MASK, None
        return FULL_MASK, None

    def set_pc_mask(self, mask):
        mask = (mask or 0) & FULL_MASK
        self._pc_mask_override = mask or None
        return self._pc_mask_override

    # ------------------------------------------------------------------
    #  การ์ด
    # ------------------------------------------------------------------
    def sensors(self, csv, mask, now=None):
        """สร้าง SensorView ครบทุกตัว พร้อมธง hidden

        คืนครบทุกตัวเสมอ (ไม่กรองทิ้งที่นี่) เพื่อให้โหมด Engineering
        เผยตัวที่ซ่อนได้โดยไม่ต้องคำนวณใหม่ และเพื่อให้เทสต์ตรวจได้ว่า
        การซ่อน "ไม่ได้แปลว่าไม่มีข้อมูล"
        """
        out = []
        age = csv.age_s(now) if csv else None
        latest = csv.latest() if csv else None
        for i in range(MAX_SENSORS):
            shown = bool((mask >> i) & 1)
            vals, oks = (csv.window(i, T.STABLE_WINDOW_SAMPLES)
                         if csv else ([], []))
            state = T.monitor_state(vals, oks, age, enabled=True)
            style = T.status_style(state)
            fails = T.consecutive_fails(oks)
            hint = T.state_hint(state, fails)
            ec = temp = None
            if latest and i < len(latest["ec"]):
                ec, temp = latest["ec"][i], latest["tw"][i]
            out.append(SensorView(
                index=i, name=self.sensor_names[i],
                colour=T.SENSOR_SERIES[i], state=state, style=style,
                ec=ec, temp=temp, age_s=age, hidden=not shown, hint=hint))
        return out

    def visible_sensors(self, sensors):
        """ตัวที่ต้องวาดการ์ด

        ⚠️ เซนเซอร์ที่ถูกซ่อนต้อง "ไม่ถูกวาดเลย" ไม่ใช่วาดเป็น 0.0 หรือ error
           การวาด 0.0 ให้ตัวที่ไม่ได้แสดง คือการรายงานค่าที่ไม่มีอยู่จริง
        """
        if self.engineering:
            return list(sensors)
        return [s for s in sensors if not s.hidden]

    def summary(self, sensors, mask):
        counts = {"total": bin(mask & FULL_MASK).count("1")}
        key = {T.STEADY: "steady", T.LIVE: "live", T.CHANGING: "live",
               T.STALE: "stale", T.SENSOR_FAULT: "fault",
               T.NO_RESPONSE: "fault", T.DISABLED: "disabled",
               T.OFFLINE: "offline"}
        for s in sensors:
            if s.hidden:
                continue
            k = key.get(s.state)
            if k:
                counts[k] = counts.get(k, 0) + 1
        text = T.summary_text(counts)
        # โหมด Engineering แสดงมากกว่าที่จอเลือกไว้ — ต้องบอกให้ชัด
        # ไม่งั้นหัวข้อจะเขียนว่า 2 SENSORS ทั้งที่เห็นการ์ด 3 ใบ
        extra = sum(1 for s in sensors if s.hidden) if self.engineering else 0
        if extra:
            text += "  •  +{} HIDDEN SHOWN".format(extra)
        return text

    # ------------------------------------------------------------------
    #  กราฟ
    # ------------------------------------------------------------------
    def chart_series(self, sensors):
        """คู่ (index, สี, ชื่อ, ป้ายเสริม) ของเส้นที่ต้องมีอยู่ตอนนี้

        ⚠️ ผู้เรียกต้องลบเส้นที่ไม่อยู่ในรายการนี้ทิ้งจริง ๆ ด้วย `.remove()`
           การแค่ `set_data([], [])` ทำให้เส้นหายจากตา แต่ยังอยู่ใน legend
           และยังกิน prop_cycle — ซึ่งคือ ghost series ที่เทสต์ A1 ไล่จับ
        """
        out = []
        for s in self.visible_sensors(sensors):
            label = s.name
            if s.hidden:                 # เห็นได้เพราะโหมด Engineering เท่านั้น
                label += "  (hidden on HMI)"
            out.append((s.index, s.colour, label, s.hidden))
        return out

    def chart_note(self):
        return HISTORY_NOTE if self.history_paused else None


def link_badge(bridge_snap):
    """ข้อความสั้นบน header สำหรับสถานะจอ"""
    snap = bridge_snap or {}
    link = snap.get("link", "OFFLINE")
    text = snap.get("link_text", "P4 OFFLINE")
    mask = snap.get("display_mask")
    if mask:
        text += " · HMI " + mask_text(mask)
    colour = {"ONLINE": T.OK, "OFFLINE": T.IDLE,
              "DISABLED": T.IDLE, "ERROR": T.ERROR}.get(link, T.IDLE)
    return {"text": text, "colour": colour, "link": link}


def mask_text(mask):
    if not mask:
        return "—"
    got = [i + 1 for i in range(MAX_SENSORS) if (mask >> i) & 1]
    return ", ".join("%02d" % n for n in got) if got else "—"
