#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 pc_state.py — ประกอบ state snapshot ที่ส่งให้จอ
============================================================================
 ⚠️ P1 ยังไม่ได้เป็นเจ้าของ logger  สถานะทั้งหมดจึงอ่านจากไฟล์ที่ระบบเดิม
    เขียนอยู่แล้ว (rec_status.json + ไฟล์ CSV) ไม่ได้ไปแตะอะไรของมัน

 ⚠️ ห้ามบอกจอว่า pc_online = true ถ้าไม่มีใครเก็บข้อมูลอยู่จริง
    การโกหกจอเรื่องนี้แย่กว่าการไม่ส่งอะไรเลย เพราะผู้ใช้จะกด Save reading
    ต่อไปโดยเชื่อว่ามีคนรับ

 ⚠️ ...และห้ามบอกว่า false ทั้งที่เก็บอยู่ด้วย
    เดิมตัดสินจากอายุของ rec_status.json ซึ่ง **ผิด** — logger เดิมเขียนไฟล์นั้น
    เฉพาะตอนเปิดโปรแกรม / เริ่ม / จบ session เท่านั้น (logger_3ec.py:337,403,411)
    ระหว่างเก็บข้อมูลปกติมันจะเก่าขึ้นเรื่อย ๆ จนขึ้น OFFLINE ถาวร
    ทั้งที่ CSV ยาวขึ้นทุก 2.5 วินาที  ตอนนี้ตัดสินจาก "อายุของแถวล่าสุดใน CSV"
    ส่วน rec_status ใช้ตอบว่า *session ไหนเปิดอยู่* ซึ่งเป็นคนละคำถาม

 ⚠️ active_mask ที่ส่งไปเป็น "ค่าสมมติ"
    CONTROL ยังเป็น #define N_SENSORS 3 และไม่มี NVS จึงไม่มีช่องทางใดที่มัน
    บอก active_mask จริงให้ PC ได้  เราจึงส่ง 0b111 พร้อมธง
    active_mask_assumed=true ไม่ใช่แต่งค่าขึ้นมาแล้วแสดงเหมือนเป็นข้อเท็จจริง
============================================================================
"""

from ..core import legacy_read as LR
from . import protocol as P

CONTROL_ASSUMED_ACTIVE_MASK = 0b0111


class PcStateSource(object):
    def __init__(self, cfg):
        self.cfg = cfg
        lg = cfg.get("legacy", {})
        self.rec_path = lg.get("rec_status") or ""
        self.counter = LR.CsvRowCounter(lg.get("data_dir") or "")
        live = cfg.get("pc_liveness", {})
        self.online_within = float(live.get("online_within_s", 10.0))
        self.stale_within = float(live.get("stale_within_s", 30.0))
        self.seq = 0

    def snapshot(self):
        rec, rec_age = LR.read_rec_status(self.rec_path)
        data_age = self.counter.row_age_s()
        liveness = LR.pc_liveness(data_age, self.online_within,
                                  self.stale_within)
        sv = LR.session_view(rec)
        return {
            "liveness": liveness,
            "liveness_text": LR.PC_TEXT[liveness],
            "data_age_s": data_age,        # อายุแถวล่าสุด — ตัวตัดสิน liveness
            "rec_age_s": rec_age,          # อายุไฟล์สถานะ — ข้อมูลประกอบเท่านั้น
            "session_mask": sv["mask"],
            "recording": sv["recording"],
            "sample_id": sv["sample_id"],
            "csv_rows": self.counter.today(),
            "active_mask": CONTROL_ASSUMED_ACTIVE_MASK,
            "active_mask_assumed": True,
            "cal_busy": False,          # P1 ยังไม่มีตัวสังเกตการคาลิเบรต
        }

    def build_frame(self):
        s = self.snapshot()
        self.seq += 1
        return P.build_state(
            seq=self.seq,
            pc_online=(s["liveness"] != LR.PC_OFFLINE),
            recording=s["recording"],
            session_mask=s["session_mask"],
            active_mask=s["active_mask"],
            cal_busy=s["cal_busy"],
            csv_rows=s["csv_rows"],
            sample_id=s["sample_id"],
            active_mask_assumed=s["active_mask_assumed"],
        ), s
