#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 raw_capture.py — จับบรรทัด NDJSON ดิบไว้ตรวจสอบ (ปิดโดยปริยาย)
============================================================================
 ⚠️ นี่คือเครื่องมือวินิจฉัย ไม่ใช่ระบบบันทึกข้อมูล
    บรรทัดที่จับได้ **ไม่ใช่หลักฐานการวัด** — หลักฐานคือ data/events/*.jsonl
    ที่ผ่านการตรวจสอบและกันซ้ำแล้ว  ไฟล์ในนี้มีทั้งเฟรมพัง เฟรมซ้ำ และ
    เศษบรรทัด ห้ามเอาไปใช้แทนกัน

 ⚠️ ต้องมีเพดานขนาด
    ตอน soak 2 ชั่วโมง heartbeat ทุก 5 วินาที = ~1,440 บรรทัด ซึ่งเล็กมาก
    แต่ถ้าวันไหนเจอ flood (เคยวัดได้ 2,396 บรรทัด/วินาที) ไฟล์จะโตหลาย GB
    ภายในไม่กี่นาทีจนดิสก์เต็ม แล้วพา logger เดิมล่มไปด้วย
    — ซึ่งเป็นสิ่งเดียวที่โปรเจกต์นี้ต้องไม่ทำเด็ดขาด  จึงหยุดเองเมื่อถึงเพดาน

 เขียนได้เฉพาะใน data/raw/ ของโปรเจกต์นี้เท่านั้น
============================================================================
"""

import os
from datetime import datetime

DEFAULT_MAX_BYTES = 32 * 1024 * 1024      # 32 MB แล้วหยุดเอง
STOPPED_NOTE = "ถึงเพดานขนาดแล้ว — หยุดจับต่อ (ตัวนับใน Diagnostics ยังเดินปกติ)"


class RawCapture(object):
    def __init__(self, data_dir, enabled=False, max_bytes=DEFAULT_MAX_BYTES):
        self.dir = os.path.join(data_dir, "raw")
        self.enabled = bool(enabled)
        self.max_bytes = int(max_bytes)
        self.written = 0
        self.lines = 0
        self.dropped_after_limit = 0
        self.stopped = False
        self.path = None
        self._fh = None
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)
            self.path = os.path.join(
                self.dir, "ndjson_{:%Y-%m-%d_%H%M%S}.log".format(datetime.now()))

    # ------------------------------------------------------------------
    def _open(self):
        if self._fh is None and self.enabled and not self.stopped:
            self._fh = open(self.path, "a", encoding="utf-8", errors="replace")
        return self._fh

    def add(self, direction, raw):
        """direction: 'rx' หรือ 'tx'  ·  raw: bytes หรือ str"""
        if not self.enabled or self.stopped:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.rstrip("\r\n")
        line = "%s %s %s\n" % (datetime.now().isoformat(timespec="milliseconds"),
                               direction, raw)
        n = len(line.encode("utf-8"))
        if self.written + n > self.max_bytes:
            self.stopped = True
            self.dropped_after_limit += 1
            try:
                fh = self._open()
                if fh:
                    fh.write("# %s\n" % STOPPED_NOTE)
                    fh.flush()
            except Exception:
                pass
            self.close()
            return False
        try:
            fh = self._open()
            if fh is None:
                return False
            fh.write(line)
            fh.flush()          # ไม่ fsync — นี่เป็นไฟล์วินิจฉัย ไม่ใช่หลักฐาน
        except Exception:
            self.enabled = False        # เขียนไม่ได้ก็เลิก ไม่ทำให้ bridge ตาย
            return False
        self.written += n
        self.lines += 1
        return True

    def snapshot(self):
        return {"enabled": self.enabled, "path": self.path,
                "lines": self.lines, "bytes": self.written,
                "stopped_at_limit": self.stopped,
                "max_bytes": self.max_bytes}

    def close(self):
        if self._fh:
            try:
                self._fh.close()
            finally:
                self._fh = None
