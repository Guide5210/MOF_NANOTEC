#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 csv_source.py — อ่าน raw CSV ของระบบเดิม แบบอ่านอย่างเดียว
============================================================================
 ⚠️ ห้ามมีการเปิดไฟล์โหมดเขียนในไฟล์นี้ตลอดไป
    raw CSV เป็นของ logger เดิม  โปรเจกต์นี้เป็นผู้อ่าน ไม่ใช่ผู้ร่วมเขียน

 สคีมา 3EC (logger_3ec.py:223)
    timestamp, EC1, T1, EC2, T2, EC3, T3, ok1, ok2, ok3 [, tag]
    tag == "CAL" -> แถวที่เก็บระหว่างคาลิเบรต ต้องข้าม ไม่ใช่ค่าวัดจริง

 ⚠️ ในโฟลเดอร์เดียวกันยังมีไฟล์สคีมาเก่า 8 คอลัมน์
    (timestamp,EC_uScm,Tw_C,Salinity_ppm,TDS_ppm,pH,pH_mV,rs485_ok)
    ต้องข้ามทั้งไฟล์ ไม่ใช่พยายามตีความ — ตีความผิดจะได้กราฟที่ดูเหมือนจริง
    แต่จับคู่ค่ากับเซนเซอร์ผิดตัว ซึ่งอันตรายกว่าไม่แสดงอะไรเลย

 ⚠️ อ่านแบบต่อยอด ไม่ใช่อ่านทั้งไฟล์ใหม่ทุก 2 วินาที
    ของเดิมใน desktop_ui อ่านทุกไฟล์ทุกบรรทัดทุกรอบ ที่ข้อมูล 24 ชั่วโมง
    ใช้เวลา 0.8 วินาทีต่อรอบ และโตเป็นเส้นตรง
============================================================================
"""

import glob
import os
from collections import deque
from datetime import datetime, timedelta

N_COLS_3EC = 10
TS_FMT = "%Y-%m-%d %H:%M:%S"


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def parse_row(line, n_sensors=3):
    """แปลงหนึ่งบรรทัดเป็น dict — คืน None ถ้าใช้ไม่ได้ (ไม่เคยโยน exception)"""
    p = line.rstrip("\r\n").split(",")
    if len(p) < N_COLS_3EC:
        return None                      # สคีมาเก่า หรือบรรทัดขาด
    if len(p) > N_COLS_3EC and p[N_COLS_3EC].strip().upper() == "CAL":
        return None
    try:
        t = datetime.strptime(p[0].strip(), TS_FMT)
    except ValueError:
        return None
    ec, tw, ok = [], [], []
    for i in range(n_sensors):
        ec.append(_num(p[1 + i * 2]))
        tw.append(_num(p[2 + i * 2]))
        raw = p[7 + i].strip()
        ok.append(raw not in ("", "0", "False", "false"))
    return {"t": t, "ec": ec, "tw": tw, "ok": ok}


class CsvTail(object):
    """เก็บแถวล่าสุดไว้ในหน่วยความจำ แล้วต่อยอดทีละส่วนที่เขียนเพิ่ม"""

    def __init__(self, data_dir, n_sensors=3, maxlen=20000):
        self.dir = data_dir or ""
        self.n = n_sensors
        self.rows = deque(maxlen=maxlen)
        self._path = None
        self._off = 0
        self._pending = ""
        self.skipped_lines = 0           # บรรทัดที่ใช้ไม่ได้ (ไว้โชว์ใน Diagnostics)
        self.read_errors = 0

    # ------------------------------------------------------------------
    def _today_path(self):
        if not self.dir or not os.path.isdir(self.dir):
            return None
        p = os.path.join(self.dir,
                         "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        return p if os.path.exists(p) else None

    def _latest_path(self):
        """ไฟล์ล่าสุดที่มีอยู่จริง — ใช้ตอนเปิดโปรแกรมนอกเวลาที่ logger ทำงาน"""
        if not self.dir or not os.path.isdir(self.dir):
            return None
        files = sorted(glob.glob(os.path.join(self.dir, "water_log_*.csv")))
        return files[-1] if files else None

    def poll(self):
        """อ่านส่วนที่เพิ่มเข้ามา คืนจำนวนแถวใหม่ (0 = ไม่มีอะไรเปลี่ยน)"""
        path = self._today_path() or self._latest_path()
        if path is None:
            return 0
        if path != self._path:            # ข้ามวัน หรือเพิ่งเริ่ม
            self._path, self._off, self._pending = path, 0, ""
        try:
            size = os.path.getsize(path)
        except OSError:
            self.read_errors += 1
            return 0
        if size < self._off:              # ไฟล์ถูกเขียนใหม่ตั้งแต่ต้น
            self._off, self._pending = 0, ""
            self.rows.clear()
        if size == self._off:
            return 0
        try:
            with open(path, "rb") as fh:  # rb — ไม่มีทางเขียนกลับ
                fh.seek(self._off)
                chunk = fh.read(size - self._off)
        except OSError:
            self.read_errors += 1
            return 0
        self._off = size

        text = self._pending + chunk.decode("utf-8", "replace")
        lines = text.split("\n")
        self._pending = lines.pop()       # บรรทัดสุดท้ายอาจยังเขียนไม่จบ
        added = 0
        for line in lines:
            if not line.strip():
                continue
            if line.startswith("timestamp"):
                continue
            row = parse_row(line, self.n)
            if row is None:
                self.skipped_lines += 1
                continue
            self.rows.append(row)
            added += 1
        return added

    # ------------------------------------------------------------------
    def latest(self):
        return self.rows[-1] if self.rows else None

    def age_s(self, now=None):
        """อายุของแถวล่าสุดเป็นวินาที (None = ยังไม่เคยมีข้อมูล)"""
        r = self.latest()
        if r is None:
            return None
        return max(0.0, ((now or datetime.now()) - r["t"]).total_seconds())

    def window(self, sensor, n):
        """ค่า n ตัวล่าสุดของเซนเซอร์หนึ่ง เรียงเก่า -> ใหม่"""
        vals, oks = [], []
        for r in list(self.rows)[-n:]:
            if sensor < len(r["ec"]):
                vals.append(r["ec"][sensor])
                oks.append(bool(r["ok"][sensor]) and r["ec"][sensor] is not None)
        return vals, oks

    def since(self, minutes=None, maxpts=3000):
        """แถวในช่วงเวลาที่ขอ ลดจำนวนจุดลงให้พอวาด"""
        rows = list(self.rows)
        if minutes is not None:
            cut = datetime.now() - timedelta(minutes=minutes)
            rows = [r for r in rows if r["t"] >= cut]
        step = max(1, len(rows) // maxpts)
        return rows[::step]
