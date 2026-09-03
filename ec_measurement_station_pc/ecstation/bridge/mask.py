#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 mask.py — display_mask: ใครเป็นเจ้าของ และซิงก์กันอย่างไร
============================================================================
 มีสามอย่างที่ชื่อคล้ายกันมาก ต้องแยกให้ขาด

   control_active_mask  CONTROL ถามหัววัดตัวไหนบ้าง        (บอร์ดเป็นเจ้าของ)
   p4_display_mask      จอแสดง/บันทึกตัวไหน                (ผู้ใช้เป็นเจ้าของ)
   pc_view_mask         หน้าจอ PC แสดงตัวไหน               (ไฟล์นี้ดูแล)

 ⚠️ display_mask ไม่แตะการเก็บ raw CSV เลย
    ปิดการแสดงผลไม่เท่ากับหยุดเก็บข้อมูล  legacy logger เก็บครบทุกช่องเสมอ

 ⚠️ ไม่มี event เฉพาะสำหรับการเปลี่ยน mask — มันมากับ heartbeat เท่านั้น
    PC จึงต้องตรวจการเปลี่ยนด้วยการเทียบ heartbeat ติดกันเอง
============================================================================
"""

from dataclasses import dataclass
from typing import Optional

VALID_MASK = 0b0111          # P1 รองรับ 3 ตัว

UNKNOWN    = "UNKNOWN"       # ยังไม่เคยได้ mask ที่ใช้ได้
FOLLOWING  = "FOLLOWING"     # PC ตามจอ
OVERRIDE   = "OVERRIDE"      # ผู้ใช้ตั้งค่าที่ PC เอง
STALE      = "STALE"         # จอเงียบ ใช้ค่าที่รู้ล่าสุด
CONFLICT   = "CONFLICT"      # จอกลับมาแล้วค่าต่างจาก PC — รอผู้ใช้ตัดสิน


@dataclass
class MaskEvent:
    kind: str                # NONE | INITIAL | CHANGED | REJECTED | OUT_OF_RANGE
    old: Optional[int] = None
    new: Optional[int] = None
    detail: str = ""


def mask_to_list(mask):
    return [i + 1 for i in range(3) if mask is not None and (mask >> i) & 1]


def mask_text(mask):
    if mask is None:
        return "—"
    got = mask_to_list(mask)
    return ", ".join("%02d" % n for n in got) if got else "ไม่มี"


class MaskSync(object):
    """เก็บสถานะการซิงก์ mask ระหว่างจอกับ PC"""

    def __init__(self, cached_mask=None):
        self.p4_mask = None            # ค่าล่าสุดที่จอบอก
        self.pc_mask = cached_mask     # ค่าที่หน้าจอ PC ใช้อยู่จริง
        self.state = UNKNOWN if cached_mask is None else STALE
        self.boot_id = None
        self.last_change = None

    # ------------------------------------------------------------------
    def on_heartbeat(self, hb, now=None):
        """ป้อน heartbeat หนึ่งอัน คืน MaskEvent ที่ควรบันทึกลง log"""
        raw = hb.display_mask
        boot_changed = (self.boot_id is not None and hb.boot_id != self.boot_id)
        first_boot = self.boot_id is None
        self.boot_id = hb.boot_id

        if raw is None:
            # จอยังไม่ได้ตั้งค่า (0xFF) — ห้ามเดาว่าเปิดครบ
            return MaskEvent("NONE", detail="จอยังไม่ได้บอก display_mask")

        ev = MaskEvent("NONE")
        if raw & ~VALID_MASK:
            ev = MaskEvent("OUT_OF_RANGE", new=raw,
                           detail="บิตนอกช่วงถูกตัดทิ้ง")
            raw &= VALID_MASK
        if raw == 0:
            return MaskEvent("REJECTED", new=0,
                             detail="mask ว่าง — ใช้ค่าเดิมต่อ")

        old = self.p4_mask
        self.p4_mask = raw

        if old is None or first_boot or boot_changed:
            kind = "INITIAL"
        elif raw != old:
            kind = "CHANGED"
        else:
            kind = ev.kind          # NONE หรือ OUT_OF_RANGE

        # ---- ตัดสินสถานะการซิงก์ ----
        if self.state in (UNKNOWN, STALE):
            if self.pc_mask is None or self.pc_mask == raw:
                self._follow(raw)
            else:
                self.state = CONFLICT
        elif self.state == FOLLOWING:
            self._follow(raw)
        elif self.state == OVERRIDE:
            pass                    # ผู้ใช้สั่งคงไว้ ไม่ทับ
        elif self.state == CONFLICT:
            if self.pc_mask == raw:
                self._follow(raw)

        if kind in ("INITIAL", "CHANGED"):
            self.last_change = now
            return MaskEvent(kind, old=old, new=raw, detail=ev.detail)
        return ev

    # ------------------------------------------------------------------
    def on_p4_silent(self):
        """เรียกเมื่อไม่ได้ยิน heartbeat นานเกินกำหนด"""
        if self.state in (FOLLOWING, CONFLICT):
            self.state = STALE

    def follow_p4(self):
        """ผู้ใช้กด 'ตามจอ'"""
        if self.p4_mask:
            self._follow(self.p4_mask)
            return True
        return False

    def keep_pc(self):
        """ผู้ใช้กด 'คงค่า PC'"""
        self.state = OVERRIDE
        return True

    def set_pc_mask(self, mask):
        """ผู้ใช้แก้ mask ที่หน้าจอ PC เอง"""
        mask &= VALID_MASK
        if mask == 0:
            return False
        self.pc_mask = mask
        self.state = OVERRIDE if (self.p4_mask and mask != self.p4_mask) else FOLLOWING
        return True

    def _follow(self, mask):
        self.pc_mask = mask
        self.state = FOLLOWING

    # ------------------------------------------------------------------
    def effective(self):
        """mask ที่หน้าจอ PC ควรใช้ตอนนี้ — ไม่เคยคืน None"""
        if self.pc_mask:
            return self.pc_mask
        if self.p4_mask:
            return self.p4_mask
        return VALID_MASK       # ยังไม่รู้อะไรเลย -> แสดงทุกตัว ปลอดภัยที่สุด

    def ui_text(self):
        if self.state == UNKNOWN:
            return "HMI DISPLAY: รอสถานะจากจอ"
        if self.state == STALE:
            return "HMI DISPLAY: {} (ค่าล่าสุดที่รู้)".format(mask_text(self.p4_mask))
        if self.state == CONFLICT:
            return "จอแสดง {} · PC แสดง {}".format(
                mask_text(self.p4_mask), mask_text(self.pc_mask))
        if self.state == OVERRIDE:
            return "OVERRIDE — จอกำลังแสดง {}".format(mask_text(self.p4_mask))
        return "HMI DISPLAY: Sensors {}".format(mask_text(self.p4_mask))
