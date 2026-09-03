#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
การ์ดเซนเซอร์ — 1 ตัว = ใบกว้าง · 2 ตัว = สองคอลัมน์ · 3 ตัว = สามใบ

⚠️ เซนเซอร์ที่ถูกซ่อนจะไม่มีการ์ดเลย ไม่ใช่การ์ดที่แสดง 0.0 หรือ error
   การแสดง 0.0 ให้ตัวที่ไม่ได้เลือก คือการรายงานค่าที่ไม่มีอยู่จริง
   ส่วน error สื่อว่ามีอะไรพัง ทั้งที่ผู้ใช้เป็นคนสั่งซ่อนเอง
"""

import tkinter as tk

from . import lab_theme as T
from .widgets import Dot, card


class SensorCard(tk.Frame):
    def __init__(self, parent, view, value_size, compact=False):
        tk.Frame.__init__(self, parent, bg=T.BG)
        self.index = view.index
        self.compact = compact
        self.box = card(self)
        self.box.pack(fill="both", expand=True)

        pad = T.SP_2
        head = tk.Frame(self.box, bg=T.SURFACE)
        head.pack(fill="x", padx=pad, pady=(pad, 0))

        # แถบสีตัวตนของเซนเซอร์ — คนละเรื่องกับสีสถานะ จึงอยู่คนละที่
        self.tag = tk.Frame(self.box, bg=view.colour, height=3)
        self.tag.place(x=0, y=0, relwidth=1.0)

        self.name = tk.Label(head, text=view.name, bg=T.SURFACE, fg=T.TEXT_DIM,
                             font=T.f(T.FONT_LABEL, "bold"), anchor="w")
        self.name.pack(side="left")

        state_box = tk.Frame(head, bg=T.SURFACE)
        state_box.pack(side="right")
        self.dot = Dot(state_box, 10)
        self.dot.pack(side="left", padx=(0, 6))
        self.state = tk.Label(state_box, text="", bg=T.SURFACE,
                              font=T.f(T.FONT_STATE, "bold"))
        self.state.pack(side="left")

        body = tk.Frame(self.box, bg=T.SURFACE)
        body.pack(fill="both", expand=True, padx=pad, pady=(4, 0))
        self.value = tk.Label(body, text=T.NO_VALUE, bg=T.SURFACE, fg=T.TEXT,
                              font=T.f(value_size, "bold"), anchor="w")
        self.value.pack(side="left" if compact else "top",
                        anchor="w", fill="x" if not compact else None)

        unit_box = tk.Frame(body, bg=T.SURFACE)
        unit_box.pack(side="left" if compact else "top", anchor="w",
                      fill="x", padx=(T.SP_2 if compact else 0, 0))
        self.unit = tk.Label(unit_box, text="µS/cm", bg=T.SURFACE,
                             fg=T.TEXT_DIM, font=T.f(T.FONT_LABEL))
        self.unit.pack(side="left")
        self.temp = tk.Label(unit_box, text="", bg=T.SURFACE, fg=T.TEXT_DIM,
                             font=T.f(T.FONT_LABEL))
        self.temp.pack(side="left", padx=(T.SP_2, 0))

        # ⚠️ freshness กับ hint ต้องอยู่คนละบรรทัด
        #    บรรทัดเดียวกันแล้วจัด left/right จะทับกันทันทีที่ hint ยาว
        #    (เคยเห็นเป็น "Updated now o reply for 6 cycles" ซึ่งอ่านไม่รู้เรื่อง
        #     และที่แย่กว่าคือทำให้คำว่า "No" หายไปจนความหมายกลับด้าน)
        foot = tk.Frame(self.box, bg=T.SURFACE)
        foot.pack(fill="x", padx=pad, pady=(2, pad))
        self.fresh = tk.Label(foot, text="", bg=T.SURFACE, fg=T.TEXT_DIM,
                              font=T.f(T.FONT_LABEL), anchor="w")
        self.fresh.pack(fill="x", anchor="w")
        self.hint = tk.Label(foot, text="", bg=T.SURFACE, fg=T.TEXT_DIM,
                             font=T.f(T.FONT_LABEL), anchor="w",
                             justify="left")
        self.hint.pack(fill="x", anchor="w")

        self.update_view(view)

    # ------------------------------------------------------------------
    def update_view(self, view, decimals=1):
        st = view.style
        bg = T.SURFACE_ALT if st["quiet"] else T.SURFACE
        for w in (self.box, self.name, self.state, self.value, self.unit,
                  self.temp, self.fresh, self.hint):
            w.configure(bg=bg)
        for w in self.box.winfo_children():
            try:
                w.configure(bg=bg)
                for c in w.winfo_children():
                    c.configure(bg=bg)
            except tk.TclError:
                pass
        self.dot.configure(bg=bg)
        self.dot.set(st["colour"])
        self.state.configure(text=st["label"], fg=st["colour"])
        self.value.configure(text=view.ec_text(decimals),
                             fg=T.TEXT_DIM if st["dim_value"] else T.TEXT)
        self.unit.configure(text="µS/cm" if st["show_value"] else "")
        self.temp.configure(text=T.format_temp(view.temp)
                            if st["show_value"] else "")
        self.fresh.configure(text=view.freshness_text())
        # ป้าย hidden ต้องเป็นข้อความ ไม่ใช่แค่สีจาง — สีจางอ่านว่า "เสีย" ได้
        self.hint.configure(
            text="HIDDEN ON HMI" if view.hidden else (view.hint or ""),
            fg=T.WARN if view.hidden else T.TEXT_DIM)
        self.tag.configure(bg=T.IDLE if st["quiet"] else view.colour)


class CardArea(tk.Frame):
    """จัดวางการ์ดใหม่ทุกครั้งที่ชุดที่มองเห็นเปลี่ยน

    ⚠️ สร้างใหม่เมื่อ "ชุด" เปลี่ยนเท่านั้น ไม่ใช่ทุกรอบวาด
       การสร้างวิดเจ็ตใหม่ทุก 2 วินาทีทำให้หน้าจอกระพริบและกิน handle เพิ่มเรื่อย ๆ
    """

    def __init__(self, parent):
        tk.Frame.__init__(self, parent, bg=T.BG)
        self.cards = {}
        self._key = None

    def render(self, views, density="roomy", decimals=1):
        key = tuple((v.index, v.hidden) for v in views)
        if key != self._key:
            self._rebuild(views, density)
            self._key = key
        for v in views:
            c = self.cards.get(v.index)
            if c is not None:
                c.update_view(v, decimals)

    def _rebuild(self, views, density):
        for w in self.winfo_children():
            w.destroy()
        self.cards.clear()
        n = len(views)
        if n == 0:
            tk.Label(self, text="ไม่มีเซนเซอร์ที่เลือกให้แสดง",
                     bg=T.BG, fg=T.TEXT_DIM, font=T.f(T.FONT_BODY)).pack(
                         pady=T.SP_3)
            return
        rows, cols = T.grid_for(n)
        size = T.value_font_size(n, density)
        compact = (n <= 1)
        for c in range(cols):
            self.grid_columnconfigure(c, weight=1, uniform="card")
        for r in range(rows):
            self.grid_rowconfigure(r, weight=1)
        for i, v in enumerate(views):
            w = SensorCard(self, v, size, compact=compact)
            w.grid(row=i // cols, column=i % cols, sticky="nsew",
                   padx=(0 if i % cols == 0 else T.SP_1 // 2, 0),
                   pady=(0, T.SP_1))
            self.cards[v.index] = w
