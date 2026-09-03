#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 ports.py — เลือกพอร์ตด้วย VID:PID และ "บทบาท" ไม่ใช่ COM แรกที่เจอ
============================================================================
 บนเครื่องนี้มีพอร์ตที่เกี่ยวข้องสามตัวและหน้าที่คนละเรื่องกันสิ้นเชิง

   1A86:7523   CH340              บอร์ด CONTROL — ส่ง "DATA," ให้ logger
   1A86:55D3   CH343              จอ ESP32-P4 — log ของจอ + ใช้แฟลช
   303A:1001   USB-Serial-JTAG    จอ ESP32-P4 — ช่อง NDJSON ของ bridge

 ⚠️ ข้อที่เคยพลาดมาแล้วและต้องไม่พลาดอีก
    พอร์ต 303A:1001 ขึ้นชื่อว่า "USB Serial Device" หรือ
    "USB JTAG/serial debug unit" ซึ่ง *มีคำว่า usb* อยู่ในนั้น
    ถ้าเลือกพอร์ตด้วยคำอย่างหลวม ๆ แล้วถอดบอร์ด CONTROL ออก
    ตัวหา CONTROL จะไปคว้าพอร์ต NDJSON ของจอมาแทน แล้วนั่งรอบรรทัด DATA,
    ที่ไม่มีวันมา พร้อมกับยึดพอร์ตไม่ให้ bridge ใช้
    อาการคือ "เงียบทั้งสองฝั่ง" ซึ่งไล่หาสาเหตุยากมาก

 ⚠️ VID 303A เป็นของ Espressif — บอร์ด CONTROL (ESP32 + CH340) เป็นไปไม่ได้
    ที่จะขึ้นเป็นเลขนี้ในโหมดใช้งาน  ถ้าเอกสารไหนบอกว่า CONTROL คือ
    303A:1001 แปลว่าเอกสารนั้นผิด
============================================================================
"""

CONTROL_IDS   = ("1a86:7523",)
P4_BRIDGE_IDS = ("303a:1001",)
P4_LOG_IDS    = ("1a86:55d3",)

ROLE_CONTROL   = "control"
ROLE_P4_BRIDGE = "p4_bridge"


def _blob(p):
    return "{} {} {}".format(getattr(p, "device", ""),
                             getattr(p, "description", ""),
                             getattr(p, "hwid", "")).lower()


def _has(blob, ids):
    return any(i in blob for i in ids)


def score(p, role):
    """คะแนนของพอร์ตหนึ่งตัวสำหรับบทบาทหนึ่ง — ติดลบ = ห้ามใช้เด็ดขาด"""
    b = _blob(p)
    if role == ROLE_CONTROL:
        if _has(b, P4_BRIDGE_IDS) or _has(b, P4_LOG_IDS) or "jtag" in b or "ch343" in b:
            return -1
        if _has(b, CONTROL_IDS):
            return 100
        for i, key in enumerate(("ch340", "cp210", "ch910")):
            if key in b:
                return 90 - i
        if any(k in b for k in ("usb", "uart", "ttyusb", "ttyacm")):
            return 10
        return 0
    if role == ROLE_P4_BRIDGE:
        if _has(b, P4_BRIDGE_IDS):
            return 100
        # ห้ามเดา — ถ้าไม่ใช่ VID:PID นี้ก็ไม่ใช่ช่อง NDJSON
        return -1
    raise ValueError("role ไม่รู้จัก: %r" % (role,))


# เหตุผลที่เลือก / ไม่เลือก — ส่งกลับไปให้คนอ่าน ไม่ใช่แค่ None เปล่า ๆ
PICK_OK        = "OK"
PICK_NONE      = "NOT_FOUND"
PICK_AMBIGUOUS = "AMBIGUOUS"


def _list(comports):
    if comports is None:
        try:
            import serial.tools.list_ports as lp
            comports = lp.comports
        except Exception:
            return []
    try:
        return list(comports())
    except Exception:
        return []


def serial_number(p):
    """เลขซีเรียลของอุปกรณ์ ถ้า OS บอกมา — ไม่รับประกันว่ามี

    ⚠️ USB-Serial-JTAG ของ ESP32 มี descriptor อยู่ใน ROM  โปรเจกต์เฟิร์มแวร์
       ตั้งค่านี้ไม่ได้  บางระบบจึงไม่เห็นเลขซีเรียลเลย
       โค้ดที่ *ต้องมี* เลขซีเรียลจึงถือว่าผิดตั้งแต่ต้น — ใช้เป็นข้อมูลเสริม
       สำหรับให้คนยืนยันด้วยตาเท่านั้น
    """
    sn = getattr(p, "serial_number", None)
    if sn:
        return str(sn)
    hw = (getattr(p, "hwid", "") or "")
    for tok in hw.replace(",", " ").split():
        if tok.upper().startswith("SER="):
            return tok[4:]
    return None


def find_detailed(role, comports=None):
    """คืน (device, reason, candidates)

    ⚠️ เจอผู้สมัครมากกว่าหนึ่ง = ไม่เลือก
       `303A:1001` ไม่ได้เป็นของ P4 ตัวเดียว — ESP32-C3/S3/C6 ทุกตัวที่เสียบ
       อยู่ก็ขึ้น VID:PID เดียวกัน  การหยิบตัวแรกมาใช้เงียบ ๆ แปลว่าวันหนึ่ง
       ที่มีบอร์ดอื่นเสียบอยู่ ระบบจะไปเปิดพอร์ตผิดตัวแล้วรอข้อมูลที่ไม่มีวันมา
       พร้อมกับยึดพอร์ตของบอร์ดนั้นไว้ด้วย  ให้คนตัดสินด้วย --bridge-port ดีกว่า
    """
    ports = [p for p in _list(comports) if score(p, role) > 0]
    if not ports:
        return None, PICK_NONE, []
    best = max(score(p, role) for p in ports)
    top = [p for p in ports if score(p, role) == best]
    if len(top) > 1:
        return None, PICK_AMBIGUOUS, [describe_port(p) for p in top]
    return top[0].device, PICK_OK, [describe_port(top[0])]


def find(role, comports=None):
    """คืน device ของพอร์ตที่เหมาะที่สุด หรือ None (รูปแบบเดิม)"""
    dev, _reason, _c = find_detailed(role, comports)
    return dev


def role_of(p):
    b = _blob(p)
    if _has(b, CONTROL_IDS):
        return "CONTROL (CH340)"
    if _has(b, P4_BRIDGE_IDS):
        return "P4 bridge (USB-Serial-JTAG)"
    if _has(b, P4_LOG_IDS):
        return "P4 log/flash (CH343)"
    return "—"


def vid_pid(p):
    vid, pid = getattr(p, "vid", None), getattr(p, "pid", None)
    if vid is not None and pid is not None:
        return "%04X:%04X" % (vid, pid)
    hw = (getattr(p, "hwid", "") or "").upper()
    for tok in hw.replace("\\", " ").replace(",", " ").split():
        if tok.startswith("VID:PID="):
            return tok[8:]
    return "—"


def describe_port(p):
    return {"device": getattr(p, "device", "?"),
            "description": getattr(p, "description", "") or "",
            "hwid": getattr(p, "hwid", "") or "",
            "vid_pid": vid_pid(p),
            "serial_number": serial_number(p),
            "manufacturer": getattr(p, "manufacturer", None),
            "product": getattr(p, "product", None),
            "location": getattr(p, "location", None),
            "role": role_of(p)}


def describe(comports=None):
    """รายการพอร์ตพร้อมบทบาทที่เดาได้ — ใช้ในหน้า Diagnostics"""
    return [describe_port(p) for p in _list(comports)]


def audit(comports=None):
    """ภาพรวมสำหรับ tools/port_audit.py — พอร์ตทั้งหมด + ผลการเลือกแต่ละบทบาท"""
    ports = _list(comports)
    out = {"ports": [describe_port(p) for p in ports], "roles": {}}
    for role in (ROLE_CONTROL, ROLE_P4_BRIDGE):
        dev, reason, cands = find_detailed(role, lambda: ports)
        out["roles"][role] = {"device": dev, "reason": reason,
                              "candidates": cands}
    return out
