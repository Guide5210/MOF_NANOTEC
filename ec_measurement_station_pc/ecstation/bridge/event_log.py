#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 event_log.py — บันทึก event จากจอแบบต่อท้ายอย่างเดียว
============================================================================
 ⚠️ ไฟล์นี้เขียนได้เฉพาะใน data_dir ของโปรเจกต์ใหม่เท่านั้น
    ห้ามเขียนอะไรลงโฟลเดอร์ของ legacy เด็ดขาด — raw CSV, session, รายงาน
    เป็นของระบบเดิมทั้งหมด และ P1 เป็นผู้อ่านอย่างเดียว

 ⚠️ ต่อท้ายอย่างเดียว ไม่มีการแก้หรือลบบรรทัดเก่า
    ผลการวัดที่ผู้ใช้กดยืนยันแล้วคือหลักฐาน ไม่ใช่ cache

 การกันซ้ำ
 ---------
 event_id ของจอมี boot_id อยู่ในตัว (รูปแบบ "<boot_id>-<seq6>") จึงไม่มีทาง
 ชนกันข้ามการบูต  ชุดกันซ้ำจึง **ไม่ต้องล้างเมื่อจอรีบูต**
 และเมื่อเปิดโปรแกรมใหม่ ต้องอ่าน event_id ของไฟล์วันนี้กลับเข้ามาก่อน
 ไม่งั้นการ replay คิวของจอหลังต่อใหม่จะได้บรรทัดซ้ำ
============================================================================
"""

import json
import os
from collections import deque
from datetime import datetime


class EventLog(object):
    def __init__(self, data_dir, dedup_max=4096):
        self.dir = os.path.join(data_dir, "events")
        os.makedirs(self.dir, exist_ok=True)
        self._seen = set()
        self._order = deque(maxlen=dedup_max)
        self._path = None
        self._fh = None
        self.written = 0
        self.duplicates = 0
        self._load_today()

    # ------------------------------------------------------------------
    def path_for(self, when=None):
        when = when or datetime.now()
        return os.path.join(self.dir, "p4_events_{:%Y-%m-%d}.jsonl".format(when))

    def _load_today(self):
        """อ่าน event_id ของวันนี้กลับมา กันบรรทัดซ้ำหลังเปิดโปรแกรมใหม่"""
        p = self.path_for()
        if not os.path.exists(p):
            return
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        eid = json.loads(line).get("event_id")
                    except Exception:
                        continue          # บรรทัดเสียก็ข้าม ไม่ทำให้เปิดไม่ได้
                    if eid:
                        self._remember(eid)
        except Exception:
            pass

    def _remember(self, eid):
        if len(self._order) == self._order.maxlen and self._order:
            self._seen.discard(self._order[0])
        self._order.append(eid)
        self._seen.add(eid)

    def seen(self, eid):
        return eid in self._seen

    # ------------------------------------------------------------------
    def _open(self):
        p = self.path_for()
        if p != self._path:
            if self._fh:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(p, "a", encoding="utf-8")
            self._path = p
        return self._fh

    def append(self, record, extra=None):
        """เขียนหนึ่งระเบียน คืน True ถ้าเขียนจริง / False ถ้าเป็นตัวซ้ำ"""
        eid = record.get("event_id")
        if eid and self.seen(eid):
            self.duplicates += 1
            return False

        now = datetime.now()
        out = {"recv_wall": now.isoformat(timespec="milliseconds"), "source": "p4"}
        out.update(record)
        if extra:
            out.update(extra)

        fh = self._open()
        fh.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())     # ผลการวัดต้องอยู่บนดิสก์จริงก่อนคืนค่า
        except Exception:
            pass

        if eid:
            self._remember(eid)
        self.written += 1
        return True

    def append_local(self, event, **fields):
        """เหตุการณ์ที่ PC เป็นคนสังเกตเอง เช่น DISPLAY_MASK_CHANGED

        ไม่มี event_id จากจอ จึงไม่เข้าระบบกันซ้ำ
        """
        rec = {"source": "pc", "event": event}
        rec.update(fields)
        now = datetime.now()
        out = {"recv_wall": now.isoformat(timespec="milliseconds")}
        out.update(rec)
        fh = self._open()
        fh.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
        self.written += 1
        return True

    def close(self):
        if self._fh:
            try:
                self._fh.close()
            finally:
                self._fh = None
