#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 snapshot.py — export สถานะเพื่อการวินิจฉัย ลง data/diag/ ของโปรเจกต์นี้
============================================================================
 ⚠️ ไฟล์ที่ export ต้อง "ปลอดภัยที่จะส่งต่อ"
    จึงมีแต่ตัวเลขสถานะและชื่อพอร์ต — ไม่มีค่าที่วัดได้ ไม่มี sample_id
    ไม่มี path ของข้อมูลห้องแล็บ  เพราะไฟล์แบบนี้มักถูกแนบไปในอีเมลหรือ
    แชตเพื่อขอความช่วยเหลือ และไม่มีใครมานั่งตรวจว่ามีอะไรอยู่ในนั้นบ้าง

 ⚠️ เขียนได้เฉพาะ data/diag/ ห้ามแตะโฟลเดอร์ของระบบเดิม
============================================================================
"""

import json
import os
import platform
import sys
from datetime import datetime

# key ที่ยอมให้หลุดออกไปจาก pc_state — ทุกอย่างนอกรายการนี้ถูกตัดทิ้ง
PC_SAFE_KEYS = ("liveness", "liveness_text", "rec_age_s", "session_mask",
                "recording", "csv_rows", "active_mask", "active_mask_assumed",
                "cal_busy")


def _redact_path(p):
    """เก็บแค่ว่า 'มี/ไม่มี' และชื่อโฟลเดอร์สุดท้าย ไม่ใช่ path เต็ม"""
    if not p:
        return None
    return ".../" + os.path.basename(os.path.normpath(p))


def build(bridge_snap, pc_snap, csv=None, feed=None, raw=None, ports=None,
          extra=None):
    now = datetime.now()
    c = dict((bridge_snap or {}).get("counters", {}))
    bad = (c.get("dropped_parse", 0) + c.get("dropped_field", 0)
           + c.get("dropped_version", 0) + c.get("dropped_oversize", 0))
    out = {
        "schema": "ecstation.diag/1",
        "captured_at": now.isoformat(timespec="seconds"),
        "host": {"python": sys.version.split()[0],
                 "platform": platform.platform(),
                 "machine": platform.machine()},
        "p4": {
            "link": (bridge_snap or {}).get("link"),
            "link_text": (bridge_snap or {}).get("link_text"),
            "error": (bridge_snap or {}).get("error"),
            "boot_id": (bridge_snap or {}).get("boot_id"),
            "reconnects": (bridge_snap or {}).get("reboots"),
            "heartbeat_age_s": (bridge_snap or {}).get("hb_age_s"),
            "last_frame_age_s": (bridge_snap or {}).get("last_frame_age_s"),
            "protocol_version": (bridge_snap or {}).get("proto_ver_seen"),
            "queued": (bridge_snap or {}).get("queued"),
            "heap": (bridge_snap or {}).get("heap"),
            "heap_big": (bridge_snap or {}).get("heap_big"),
            "p4_sees_pc": (bridge_snap or {}).get("p4_sees_pc"),
            "display_mask": (bridge_snap or {}).get("display_mask"),
            "display_mask_prev": (bridge_snap or {}).get("display_mask_prev"),
            "mask_state": (bridge_snap or {}).get("mask_state"),
        },
        "counters": c,
        "counters_derived": {"malformed_total": bad},
        "pc": {k: (pc_snap or {}).get(k) for k in PC_SAFE_KEYS},
    }
    if csv is not None:
        out["csv"] = {"rows_in_memory": len(getattr(csv, "rows", [])),
                      "skipped_lines": getattr(csv, "skipped_lines", 0),
                      "read_errors": getattr(csv, "read_errors", 0),
                      "dir": _redact_path(getattr(csv, "dir", ""))}
    if feed is not None:
        out["event_feed"] = {"rows": len(getattr(feed, "rows", [])),
                             "duplicates_rejected": getattr(feed, "duplicates", 0)}
    if raw is not None:
        r = dict(raw.snapshot() if hasattr(raw, "snapshot") else raw)
        r["path"] = _redact_path(r.get("path"))
        out["raw_capture"] = r
    if ports is not None:
        out["ports"] = [{"device": p.get("device"), "vid_pid": p.get("vid_pid"),
                         "role": p.get("role"),
                         "has_serial_number": bool(p.get("serial_number"))}
                        for p in ports]
    if extra:
        out["notes"] = extra
    return out


def export(data_dir, payload, name=None):
    """เขียนลง data/diag/ แล้วคืน path — ไม่มีทางเขียนที่อื่น"""
    d = os.path.join(data_dir, "diag")
    os.makedirs(d, exist_ok=True)
    name = name or "diag_{:%Y-%m-%d_%H%M%S}.json".format(datetime.now())
    path = os.path.join(d, os.path.basename(name))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return path
