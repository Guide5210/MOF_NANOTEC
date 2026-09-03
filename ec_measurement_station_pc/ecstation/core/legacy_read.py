#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 legacy_read.py — อ่านสถานะและข้อมูลของระบบเดิม แบบอ่านอย่างเดียว
============================================================================
 ⚠️ ไฟล์นี้ "ห้ามเขียนอะไรลงโฟลเดอร์ของ legacy เด็ดขาด"
    ไม่มีฟังก์ชันไหนในนี้เปิดไฟล์ด้วยโหมดเขียน และต้องเป็นแบบนั้นตลอดไป
    raw CSV / session / รายงาน เป็นของระบบเดิมทั้งหมด

 ⚠️ นับแถว CSV ด้วย index ต่อไฟล์ ไม่อ่านซ้ำทั้งไฟล์ทุกครั้ง
    ของเดิมใน desktop_ui อ่านทุกไฟล์ทุกบรรทัดทุก 2 วินาที
    ที่ข้อมูล 24 ชั่วโมงคือ 0.8 วินาทีต่อครั้ง และโตเป็นเส้นตรง
============================================================================
"""

import glob
import json
import os
from datetime import datetime

# สถานะความมีชีวิตของ logger เดิม — สามระดับ ไม่ใช่ boolean
PC_ONLINE  = "ONLINE"
PC_STALE   = "STALE"
PC_OFFLINE = "OFFLINE"

PC_TEXT = {
    PC_ONLINE:  "PC LOGGER ONLINE",
    PC_STALE:   "PC STATUS DELAYED",
    PC_OFFLINE: "PC LOGGER OFFLINE",
}


def read_rec_status(path):
    """คืน dict ของ rec_status.json + อายุเป็นวินาที (None = ไม่มีไฟล์)"""
    if not path or not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return None, None
    age = None
    try:
        age = (datetime.now() - datetime.fromisoformat(d["updated"])).total_seconds()
    except Exception:
        try:
            age = max(0.0, datetime.now().timestamp() - os.path.getmtime(path))
        except Exception:
            age = None
    return d, age


def pc_liveness(age_s, online_within=10.0, stale_within=30.0):
    """แปลงอายุของ "แถวข้อมูลล่าสุด" เป็นสามสถานะ

    ⚠️ อย่าใช้อายุของ rec_status.json มาตัดสินข้อนี้
       logger เดิมเขียนไฟล์นั้นเฉพาะตอนเปิดโปรแกรม / เริ่ม session / จบ session
       (logger_3ec.py:337, 403, 411) ไม่ได้เขียนเป็นจังหวะ
       ระหว่างเก็บข้อมูลปกติไฟล์นั้นจะเก่าขึ้นเรื่อย ๆ ตลอดกาล
       ถ้าเอามาตัดสิน จะขึ้น PC LOGGER OFFLINE ทั้งที่ logger เขียน CSV
       อยู่ทุก 2.5 วินาที — และจะส่ง pc_online=false ไปให้จอด้วย
       ซึ่งทำให้ผู้ใช้ที่หน้าจอเชื่อว่าไม่มีใครเก็บข้อมูล ทั้งที่เก็บอยู่

       สิ่งที่พิสูจน์ว่า logger ยังมีชีวิตคือ "CSV ยาวขึ้น" ไม่ใช่ไฟล์สถานะ

    ⚠️ ต้องมีสามระดับ ไม่ใช่ 'ออนไลน์/ไม่ออนไลน์'
       ช่วงกลางคือ 'ยังไม่รู้ว่าตายหรือแค่ช้า' ซึ่งเป็นคนละเรื่องกับตายจริง
       การรีบบอกว่าตายทำให้คนวิ่งไปไล่สายทั้งที่ไม่มีอะไรพัง
    """
    if age_s is None:
        return PC_OFFLINE
    if age_s <= online_within:
        return PC_ONLINE
    if age_s <= stale_within:
        return PC_STALE
    return PC_OFFLINE


def session_view(rec):
    """สรุป session จาก rec_status.json — รองรับทั้งรูปแบบเก่าและใหม่"""
    if not rec:
        return {"mask": 0, "recording": False, "sample_id": "", "active": []}
    active = list(rec.get("active") or [])
    mask = rec.get("mask")
    if mask is None:
        mask = 0
        for i, a in enumerate(active):
            if a:
                mask |= (1 << i)
    samples = list(rec.get("sample") or [])
    sample_id = next((s for s in samples if s), "")
    return {"mask": int(mask), "recording": bool(mask),
            "sample_id": sample_id, "active": active}


def _row_time(line):
    """เวลาในคอลัมน์แรกของแถว CSV — คืน None ถ้าไม่ใช่แถวข้อมูล"""
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    head = line.split(",", 1)[0].strip()
    try:
        return datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


class CsvRowCounter(object):
    """นับแถวของไฟล์ CSV โดยอ่านเฉพาะส่วนที่ต่อท้ายเข้ามาใหม่

    เก็บเวลาของแถวล่าสุดไว้ด้วย — ใช้ตัดสินว่า logger เดิมยังมีชีวิตอยู่ไหม
    """

    def __init__(self, data_dir):
        self.dir = data_dir
        self._pos = {}       # path -> (offset, rows)
        self._last_ts = None

    def _count_file(self, path):
        off, rows = self._pos.get(path, (0, 0))
        try:
            size = os.path.getsize(path)
        except OSError:
            return 0
        if size < off:                    # ไฟล์ถูกเขียนใหม่ตั้งแต่ต้น
            off, rows = 0, 0
        if size == off:
            return rows
        try:
            with open(path, "rb") as fh:
                fh.seek(off)
                chunk = fh.read(size - off)
        except OSError:
            return rows
        rows += chunk.count(b"\n")
        if off == 0 and rows > 0:
            rows -= 1                     # หัวตาราง
        self._pos[path] = (size, rows)

        # เวลาของ "แถวสุดท้ายที่เขียนจบแล้ว" — แถวที่ยังเขียนไม่จบต้องข้าม
        if b"\n" in chunk:
            parts = chunk.rsplit(b"\n", 2)
            if len(parts) >= 2:
                ts = _row_time(parts[-2])
                if ts is not None:
                    self._last_ts = ts
        return rows

    def newest_path(self):
        """ไฟล์ของวันนี้ก่อน ถ้าไม่มีก็ไฟล์ล่าสุดที่มี

        ⚠️ ต้องมีทางถอยตอนข้ามเที่ยงคืน  ไฟล์ของวันใหม่ยังไม่ถูกสร้างจนกว่า
           logger จะเขียนแถวแรก ถ้าดูแต่ไฟล์ของวันนี้ จะขึ้น OFFLINE ชั่วขณะ
           ทุกคืนโดยไม่มีอะไรผิด
        """
        if not self.dir or not os.path.isdir(self.dir):
            return None
        p = os.path.join(self.dir,
                         "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        if os.path.exists(p):
            return p
        files = sorted(glob.glob(os.path.join(self.dir, "water_log_*.csv")))
        return files[-1] if files else None

    def today(self):
        if not self.dir or not os.path.isdir(self.dir):
            return 0
        p = os.path.join(self.dir, "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        return self._count_file(p) if os.path.exists(p) else 0

    def last_row_time(self, refresh=True):
        """เวลาของแถวข้อมูลล่าสุดที่เขียนจบแล้ว (None = ยังไม่มีข้อมูลเลย)"""
        if refresh:
            p = self.newest_path()
            if p:
                self._count_file(p)
        return self._last_ts

    def row_age_s(self, now=None, refresh=True):
        """อายุของแถวล่าสุดเป็นวินาที — ตัวชี้วัดว่า logger เดิมยังมีชีวิต"""
        ts = self.last_row_time(refresh=refresh)
        if ts is None:
            return None
        return max(0.0, ((now or datetime.now()) - ts).total_seconds())

    def total(self):
        if not self.dir or not os.path.isdir(self.dir):
            return 0
        return sum(self._count_file(p)
                   for p in glob.glob(os.path.join(self.dir, "water_log_*.csv")))
