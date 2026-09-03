#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 protocol.py — ตรวจและแปลงเฟรม NDJSON จากจอ ESP32-P4
============================================================================
 ⚠️ ไฟล์นี้ "ไม่มี I/O เลย"  ไม่เปิดพอร์ต ไม่อ่านไฟล์ ไม่พิมพ์อะไร
    จึงทดสอบได้ครบทุกเส้นทางโดยไม่ต้องมีฮาร์ดแวร์

 ⚠️ ทุกฟังก์ชันในไฟล์นี้ "ห้าม raise" — คืน ParseResult ที่มีเหตุผลเสมอ
    บรรทัดเสียหนึ่งบรรทัดต้องไม่ทำให้ระบบที่กำลังเก็บข้อมูลอยู่ล้ม

 ----------------------------------------------------------------------
  เรื่องสำคัญที่สุดในไฟล์นี้: ชื่อ field บนสาย ≠ ชื่อ field ในระเบียนของ PC
 ----------------------------------------------------------------------
  เฟิร์มแวร์วันนี้ (pc_bridge.c:461-471) ส่งชื่อพวกนี้ออกมา:

        "ec_us_cm"   "ts_ms"

  แต่ค่าที่อยู่ข้างในคือค่าที่ถูกล็อกไว้แล้วว่านิ่ง — ui_model.c:195 ส่ง
  v.stable_ec_us_cm เข้ามาตรง ๆ  และ ts_ms คือ now_ms() ซึ่งเป็นเวลานับจาก
  บูตของจอ ไม่ใช่เวลาผนัง

  ชื่อบนสายจึง "กำกวม" จริงตามที่กังวล:
     - ec_us_cm ในเฟรม reading_saved  = ค่าที่นิ่งแล้วและถูกบันทึก
     - ec_us_cm ในเฟรม context event  = ค่าสดตอนนั้น
     ชื่อเดียวกัน คนละความหมาย

  วิธีแก้ที่ไฟล์นี้ใช้:
     รับได้ทั้งชื่อเก่าและชื่อใหม่ แล้ว **แปลงเป็นชื่อที่ไม่กำกวมทันที**

        ec_us_cm | stable_ec_us_cm   ->  stable_ec_us_cm
        ts_ms    | device_mono_ms    ->  device_mono_ms

     ระเบียนที่เก็บลงไฟล์และที่ UI เห็น จึงใช้ชื่อที่ชัดเจนเสมอ
     และวันที่เฟิร์มแวร์เปลี่ยนไปใช้ชื่อใหม่ ฝั่ง PC ไม่ต้องแก้อะไรเลย

     ถ้าเฟรมมีทั้งสองชื่อและค่าไม่ตรงกัน = กำกวมจริง -> ปฏิเสธเฟรมนั้น
     ชื่อหลวม ๆ อย่าง "ec" / "value" -> ปฏิเสธ
============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json

PROTO_VER = 1
MAX_LINE = 512                 # PC_BRIDGE_FRAME_MAX ฝั่งจอ
DISPLAY_MASK_UNKNOWN = 0xFF    # ค่าตั้งต้นใน pc_bridge.c ก่อนถูกเซ็ตจริง
SENSOR_MAX = 4                 # โครงรองรับ 4 · P1 ใช้จริง 3

# ---------------------------------------------------------------- เหตุผลที่ปฏิเสธ
R_OK             = None
R_EMPTY          = "EMPTY"
R_OVERSIZE       = "OVERSIZE"
R_NOT_UTF8       = "NOT_UTF8"
R_NOT_JSON       = "NOT_JSON"
R_NOT_OBJECT     = "NOT_OBJECT"
R_BAD_VERSION    = "BAD_VERSION"
R_MISSING_TYPE   = "MISSING_TYPE"
R_UNKNOWN_TYPE   = "UNKNOWN_TYPE"


def _missing(name):
    return "MISSING_FIELD:" + name


def _bad(name):
    return "BAD_FIELD:" + name


# ---------------------------------------------------------------- เฟรม
@dataclass(frozen=True)
class Heartbeat:
    boot_id: str
    device_mono_ms: int
    queued: int
    link: str                        # "online" | "offline" — จอมองว่า PC อยู่ไหม
    heap: Optional[int] = None
    heap_big: Optional[int] = None
    display_mask: Optional[int] = None   # None = จอยังไม่ได้ตั้งค่า (0xFF)
    rs485_round: Optional[int] = None    # รอบ polling ที่จอนับได้ตั้งแต่บูต


@dataclass(frozen=True)
class ReadingSaved:
    event_id: str
    boot_id: str
    sensor: int
    stable_ec_us_cm: float
    temperature_c: float
    tolerance_us_cm: float
    stable_for_ms: int
    device_mono_ms: int
    after_link_error: bool = False
    device_wall: Optional[str] = None

    event: str = "reading_saved"


@dataclass(frozen=True)
class ContextEvent:
    event_id: str
    boot_id: str
    event: str                       # RUN_STARTED / STABILITY_REACHED / ...
    sensor: int
    device_mono_ms: int
    ec_us_cm: Optional[float] = None   # ค่าสด ไม่ใช่ค่าที่บันทึก — ชื่อจึงต่างกันโดยตั้งใจ


@dataclass(frozen=True)
class Command:
    request_id: int
    boot_id: str
    action: str
    device_mono_ms: int
    sensor: int = 0                  # 0 = ไม่ระบุตัว/ทั้งระบบ
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseResult:
    frame: Any = None
    reason: Optional[str] = None
    raw: str = ""

    @property
    def ok(self):
        return self.frame is not None


# ---------------------------------------------------------------- ตัวช่วยดึงค่า
def _req_str(d, key, maxlen=64):
    v = d.get(key)
    if v is None:
        return None, _missing(key)
    if not isinstance(v, str) or not v or len(v) > maxlen:
        return None, _bad(key)
    return v, None


def _req_int(d, key, lo=None, hi=None):
    v = d.get(key)
    if v is None:
        return None, _missing(key)
    if isinstance(v, bool) or not isinstance(v, int):
        return None, _bad(key)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None, _bad(key)
    return v, None


def _req_num(d, key):
    v = d.get(key)
    if v is None:
        return None, _missing(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, _bad(key)
    if v != v:                       # NaN
        return None, _bad(key)
    return float(v), None


def _opt_int(d, key, lo=None, hi=None):
    if key not in d or d[key] is None:
        return None, None
    return _req_int(d, key, lo, hi)


def _sensor(d, required=True):
    """sensor 0 = ไม่ระบุตัว · 1-4 = ตัวที่ระบุ

    ⚠️ 0 ไม่ใช่ 'sensor ตัวที่ 0' — มาจาก pc_bridge.c ที่เขียน sensor+1
       ของค่า -1  ผู้เรียกต้องตีความเป็น 'ทั้งระบบ'
    """
    v = d.get("sensor")
    if v is None:
        return (None, _missing("sensor")) if required else (0, None)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > SENSOR_MAX:
        return None, _bad("sensor")
    return v, None


# ---------------------------------------------------------------- ชื่อกำกวม
#  ชื่อที่ "ไม่ยอมรับเด็ดขาด" สำหรับค่า EC ในเฟรม reading_saved
#  เพราะอ่านแล้วไม่รู้ว่าเป็นค่าสด ค่าเฉลี่ย หรือค่าที่ล็อกไว้
_AMBIGUOUS_EC = ("ec", "value", "ec_value", "reading", "ec_val")


def _stable_ec(d):
    """ดึงค่า EC ที่ถูกบันทึก — รับได้ทั้งชื่อบนสายวันนี้และชื่อที่ชัดเจนกว่า"""
    for bad in _AMBIGUOUS_EC:
        if bad in d:
            return None, _bad(bad)

    a = d.get("stable_ec_us_cm")
    b = d.get("ec_us_cm")
    if a is None and b is None:
        return None, _missing("stable_ec_us_cm|ec_us_cm")
    if a is not None and b is not None:
        # มีทั้งสองชื่อ — ยอมรับได้ถ้าค่าตรงกัน ถ้าต่างกันคือกำกวมจริง
        try:
            if abs(float(a) - float(b)) > 1e-6:
                return None, _bad("stable_ec_us_cm!=ec_us_cm")
        except (TypeError, ValueError):
            return None, _bad("stable_ec_us_cm")
    src = "stable_ec_us_cm" if a is not None else "ec_us_cm"
    return _req_num(d, src)


def _mono_ms(d):
    """เวลานับจากบูตของจอ — รับได้ทั้ง ts_ms (วันนี้) และ device_mono_ms (อนาคต)"""
    a = d.get("device_mono_ms")
    b = d.get("ts_ms")
    if a is None and b is None:
        return None, _missing("device_mono_ms|ts_ms")
    if a is not None and b is not None and a != b:
        return None, _bad("device_mono_ms!=ts_ms")
    return _req_int(d, "device_mono_ms" if a is not None else "ts_ms", lo=0)


# ---------------------------------------------------------------- ตัวแยกเฟรม
def _parse_hb(d):
    boot, e = _req_str(d, "boot_id", 32)
    if e: return ParseResult(reason=e)
    mono, e = _mono_ms(d)
    if e: return ParseResult(reason=e)
    queued, e = _req_int(d, "queued", lo=0, hi=65535)
    if e: return ParseResult(reason=e)
    link, e = _req_str(d, "link", 16)
    if e: return ParseResult(reason=e)
    if link not in ("online", "offline"):
        return ParseResult(reason=_bad("link"))

    heap, e = _opt_int(d, "heap", lo=0)
    if e: return ParseResult(reason=e)
    heap_big, e = _opt_int(d, "heap_big", lo=0)
    if e: return ParseResult(reason=e)

    rnd, e = _opt_int(d, "rs485_round", lo=0)
    mask, e = _opt_int(d, "display_mask", lo=0, hi=255)
    if e: return ParseResult(reason=e)
    if mask == DISPLAY_MASK_UNKNOWN:
        # ⚠️ 0xFF คือค่าตั้งต้นใน pc_bridge.c ก่อน hello_world_main.c เซ็ตค่าจริง
        #    ห้ามตีความเป็น 'เปิดครบทุกตัว' เพราะ 0xFF & 0b111 = 7 ซึ่งดูสมเหตุสมผล
        #    แต่ผิด — มันแปลว่า 'จอยังไม่ได้บอก'
        mask = None

    return ParseResult(frame=Heartbeat(boot_id=boot, device_mono_ms=mono,
                                       queued=queued, link=link, heap=heap,
                                       heap_big=heap_big, display_mask=mask,
                                       rs485_round=rnd))


def _parse_event(d):
    ev, e = _req_str(d, "event", 40)
    if e: return ParseResult(reason=e)
    eid, e = _req_str(d, "event_id", 64)
    if e: return ParseResult(reason=e)
    boot, e = _req_str(d, "boot_id", 32)
    if e: return ParseResult(reason=e)
    sensor, e = _sensor(d)
    if e: return ParseResult(reason=e)
    mono, e = _mono_ms(d)
    if e: return ParseResult(reason=e)

    if ev.lower() == "reading_saved":
        ec, e = _stable_ec(d)
        if e: return ParseResult(reason=e)
        temp, e = _req_num(d, "temperature_c")
        if e: return ParseResult(reason=e)
        tol, e = _req_num(d, "tolerance_us_cm")
        if e: return ParseResult(reason=e)
        hold, e = _req_int(d, "stable_for_ms", lo=0)
        if e: return ParseResult(reason=e)
        ale = d.get("after_link_error", False)
        if not isinstance(ale, bool):
            return ParseResult(reason=_bad("after_link_error"))
        wall = d.get("wall") or d.get("device_wall")
        if wall is not None and not isinstance(wall, str):
            return ParseResult(reason=_bad("wall"))
        return ParseResult(frame=ReadingSaved(
            event_id=eid, boot_id=boot, sensor=sensor,
            stable_ec_us_cm=ec, temperature_c=temp, tolerance_us_cm=tol,
            stable_for_ms=hold, device_mono_ms=mono,
            after_link_error=ale, device_wall=wall))

    # event บริบท — ec_us_cm ที่นี่คือค่าสด ไม่ใช่ค่าที่บันทึก จึงไม่แปลงชื่อ
    ec = None
    if d.get("ec_us_cm") is not None:
        ec, e = _req_num(d, "ec_us_cm")
        if e: return ParseResult(reason=e)
    return ParseResult(frame=ContextEvent(
        event_id=eid, boot_id=boot, event=ev.upper(), sensor=sensor,
        device_mono_ms=mono, ec_us_cm=ec))


def _parse_cmd(d):
    rid, e = _req_int(d, "request_id", lo=0)
    if e: return ParseResult(reason=e)
    boot, e = _req_str(d, "boot_id", 32)
    if e: return ParseResult(reason=e)
    act, e = _req_str(d, "action", 32)
    if e: return ParseResult(reason=e)
    mono, e = _mono_ms(d)
    if e: return ParseResult(reason=e)
    sensor, e = _sensor(d, required=False)
    if e: return ParseResult(reason=e)
    payload = d.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return ParseResult(reason=_bad("payload"))
    return ParseResult(frame=Command(request_id=rid, boot_id=boot, action=act,
                                     device_mono_ms=mono, sensor=sensor,
                                     payload=payload))


_DISPATCH = {"hb": _parse_hb, "event": _parse_event, "cmd": _parse_cmd}


# ---------------------------------------------------------------- ทางเข้าหลัก
def parse_line(raw):
    """แปลงหนึ่งบรรทัดเป็นเฟรม — ห้าม raise ไม่ว่ากรณีใด

    raw รับได้ทั้ง bytes และ str (ไม่รวม '\\n')
    """
    try:
        if isinstance(raw, (bytes, bytearray)):
            if len(raw) > MAX_LINE:
                return ParseResult(reason=R_OVERSIZE, raw="")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return ParseResult(reason=R_NOT_UTF8, raw="")
        else:
            text = raw
            if len(text.encode("utf-8", "replace")) > MAX_LINE:
                return ParseResult(reason=R_OVERSIZE, raw="")

        text = text.strip()
        if not text:
            return ParseResult(reason=R_EMPTY, raw="")

        try:
            d = json.loads(text)
        except Exception:
            return ParseResult(reason=R_NOT_JSON, raw=text)
        if not isinstance(d, dict):
            return ParseResult(reason=R_NOT_OBJECT, raw=text)

        if d.get("v") != PROTO_VER:
            # ⚠️ เวอร์ชันไม่ตรง = ไม่ประมวลผลเด็ดขาด
            #    การเดาความหมายของเฟรมที่ไม่รู้จักคือทางที่ทำให้ระบบทำสิ่งที่
            #    ไม่มีใครสั่ง ซึ่งอันตรายกว่าการไม่ทำอะไรเลย  (กติกาเดียวกับฝั่งจอ)
            return ParseResult(reason=R_BAD_VERSION, raw=text)

        t = d.get("type")
        if not isinstance(t, str) or not t:
            return ParseResult(reason=R_MISSING_TYPE, raw=text)
        fn = _DISPATCH.get(t)
        if fn is None:
            # type ที่ยังไม่รู้จัก — ข้ามเงียบตามกฎ forward compatibility
            return ParseResult(reason=R_UNKNOWN_TYPE, raw=text)

        res = fn(d)
        return ParseResult(frame=res.frame, reason=res.reason, raw=text)
    except Exception as exc:                     # กันไว้ชั้นสุดท้าย
        return ParseResult(reason="INTERNAL:" + type(exc).__name__, raw="")


# ---------------------------------------------------------------- ระเบียนของ PC
#  ชื่อ field ในระเบียนนี้คือ "สัญญา" ที่เหลือของระบบยึด
#  ไม่ผูกกับชื่อบนสายซึ่งเปลี่ยนได้ตามเฟิร์มแวร์
READING_SAVED_RECORD_FIELDS = (
    "v", "type", "event_id", "boot_id", "event", "sensor",
    "stable_ec_us_cm", "temperature_c", "tolerance_us_cm",
    "stable_for_ms", "device_mono_ms",
)


def reading_saved_record(fr):
    """แปลง ReadingSaved เป็น dict ที่จะถูกเขียนลง JSONL"""
    rec = {
        "v": PROTO_VER,
        "type": "event",
        "event_id": fr.event_id,
        "boot_id": fr.boot_id,
        "event": "reading_saved",
        "sensor": fr.sensor,
        "stable_ec_us_cm": fr.stable_ec_us_cm,
        "temperature_c": fr.temperature_c,
        "tolerance_us_cm": fr.tolerance_us_cm,
        "stable_for_ms": fr.stable_for_ms,
        "device_mono_ms": fr.device_mono_ms,
        "after_link_error": fr.after_link_error,
    }
    if fr.device_wall:
        rec["device_wall"] = fr.device_wall
    return rec


def context_event_record(fr):
    rec = {
        "v": PROTO_VER,
        "type": "event",
        "event_id": fr.event_id,
        "boot_id": fr.boot_id,
        "event": fr.event,
        "sensor": fr.sensor,
        "device_mono_ms": fr.device_mono_ms,
    }
    if fr.ec_us_cm is not None:
        rec["ec_us_cm"] = fr.ec_us_cm
    return rec


# ---------------------------------------------------------------- PC -> P4
def build_state(seq, pc_online, recording, session_mask, active_mask,
                cal_busy, csv_rows, sample_id, active_mask_assumed=True):
    """เฟรม state ที่ส่งกลับไปให้จอ

    ⚠️ ส่งเฉพาะ field ที่ handle_state() ฝั่งจออ่านจริง + ธงบอกความไม่แน่นอน
       จอข้าม field ที่ไม่รู้จักอยู่แล้ว การส่งธงเพิ่มจึงปลอดภัย
       และทำให้ฝั่งจอ (หรือคนอ่าน log) รู้ว่า active_mask เป็นค่าสมมติ
    """
    return {
        "v": PROTO_VER, "type": "state", "seq": int(seq),
        "pc_online": bool(pc_online),
        "recording": bool(recording),
        "session_mask": int(session_mask) & 0x0F,
        "active_mask": int(active_mask) & 0x0F,
        "cal_busy": bool(cal_busy),
        "csv_rows": int(csv_rows),
        "sample_id": (sample_id or "")[:23],
        "active_mask_assumed": bool(active_mask_assumed),
    }


def build_nack(request_id, action, code, message):
    return {"v": PROTO_VER, "type": "ack", "request_id": int(request_id),
            "ok": False, "action": action, "code": code, "message": message[:120]}


def dumps_line(obj):
    """serialize หนึ่งเฟรม — คืน bytes พร้อม '\\n' หรือ None ถ้ายาวเกิน"""
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    b = s.encode("utf-8") + b"\n"
    return b if len(b) <= MAX_LINE else None
