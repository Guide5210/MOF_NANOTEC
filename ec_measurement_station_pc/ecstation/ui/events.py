#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
แผงเหตุการณ์การวัด — เวลา · แหล่ง/เซนเซอร์ · เหตุการณ์ · ค่า/รายละเอียด

⚠️ ไม่มี event_id, boot_id, หรือ raw JSON ที่นี่ — พวกนั้นอยู่ Diagnostics
   แผงนี้ตอบคำถามว่า "เกิดอะไรขึ้นกับการวัดของฉัน" ไม่ใช่ "ลิงก์เป็นยังไง"

⚠️ ประวัติต้องอยู่ต่อเมื่อจอหลุด  รายการนี้เป็นของ EventFeed ในฝั่ง PC
   ไม่ได้ดึงจากจอใหม่ทุกครั้ง การล้างตอนสายหลุดคือการลบหลักฐานงานที่เพิ่งทำ
"""

import tkinter as tk

from . import lab_theme as T
from .widgets import Dot

KIND_COLOUR = {
    "reading_saved":     T.OK,
    "STABILITY_REACHED": T.OK,
    "STABILITY_LOST":    T.WARN,
    "LINK_ERROR":        T.ERROR,
    "P4_REBOOT":         T.WARN,
    "CMD_REJECTED":      T.WARN,
}


class EventPanel(tk.Frame):
    def __init__(self, parent, rows=3):
        tk.Frame.__init__(self, parent, bg=T.SURFACE,
                          highlightbackground=T.BORDER,
                          highlightthickness=T.BORDER_W, bd=0)
        self.max_rows = rows
        head = tk.Frame(self, bg=T.SURFACE)
        head.pack(fill="x", padx=T.SP_2, pady=(T.SP_1, 2))
        tk.Label(head, text="MEASUREMENT EVENTS", bg=T.SURFACE, fg=T.TEXT_DIM,
                 font=T.f(T.FONT_LABEL, "bold")).pack(side="left")
        self.count = tk.Label(head, text="", bg=T.SURFACE, fg=T.TEXT_DIM,
                              font=T.f(T.FONT_LABEL))
        self.count.pack(side="right")

        self.body = tk.Frame(self, bg=T.SURFACE)
        self.body.pack(fill="both", expand=True, padx=T.SP_2,
                       pady=(0, T.SP_1))
        self._rows = []

    def _row(self, i):
        while len(self._rows) <= i:
            fr = tk.Frame(self.body, bg=T.SURFACE)
            fr.pack(fill="x", pady=1)
            dot = Dot(fr, 8)
            dot.pack(side="left", padx=(0, T.SP_1))
            when = tk.Label(fr, bg=T.SURFACE, fg=T.TEXT_DIM,
                            font=T.fm(T.FONT_LABEL), width=9, anchor="w")
            who = tk.Label(fr, bg=T.SURFACE, fg=T.TEXT_DIM,
                           font=T.f(T.FONT_LABEL), width=11, anchor="w")
            what = tk.Label(fr, bg=T.SURFACE, fg=T.TEXT,
                            font=T.f(T.FONT_LABEL, "bold"), width=24,
                            anchor="w")
            val = tk.Label(fr, bg=T.SURFACE, fg=T.TEXT,
                           font=T.fm(T.FONT_LABEL), anchor="w")
            for w in (when, who, what, val):
                w.pack(side="left", padx=(0, T.SP_1))
            self._rows.append((fr, dot, when, who, what, val))
        return self._rows[i]

    def render(self, feed, decimals=1):
        rows = feed.visible(self.max_rows)
        for i in range(self.max_rows):
            fr, dot, when, who, what, val = self._row(i)
            if i >= len(rows):
                fr.pack_forget()
                continue
            fr.pack(fill="x", pady=1)
            r = rows[i]
            dot.set(KIND_COLOUR.get(r["kind"], T.IDLE))
            when.configure(text=r["when"].strftime("%H:%M:%S"))
            who.configure(text="SENSOR %02d" % r["sensor"] if r["sensor"]
                          else "SYSTEM")
            what.configure(text=r["text"])
            bits = []
            if r["value"] is not None:
                bits.append("%s µS/cm" % T.format_ec(r["value"], decimals))
            if r["detail"]:
                bits.append(r["detail"])
            val.configure(text="   ".join(bits))
        n = len(feed.visible())
        # ตารางนี้ใช้คำศัพท์ชุดเดียวกับจอ P4 ซึ่งเป็นอังกฤษทั้งหมด
        # การสลับภาษากลางตารางทำให้สายตาสะดุดและจัดคอลัมน์ยาก
        self.count.configure(text="%d events" % n if n else "no events yet")
