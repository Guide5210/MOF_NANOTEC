#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ชิ้นส่วน UI ที่ใช้ซ้ำ — ไม่มีสีดิบสักตัว ทุกสีมาจาก lab_theme"""

import tkinter as tk
from tkinter import ttk

from . import lab_theme as T


def card(parent, **kw):
    """กรอบขาว ขอบบาง 1 px — พื้นฐานของทุกกล่องในหน้าจอ"""
    f = tk.Frame(parent, bg=kw.pop("bg", T.SURFACE),
                 highlightbackground=T.BORDER, highlightcolor=T.BORDER,
                 highlightthickness=T.BORDER_W, bd=0, **kw)
    return f


def label(parent, text="", size=None, colour=None, bg=T.SURFACE,
          weight="normal", **kw):
    return tk.Label(parent, text=text, bg=bg,
                    fg=colour or T.TEXT,
                    font=T.f(size or T.FONT_BODY, weight), **kw)


class Dot(tk.Canvas):
    """จุดสถานะ — ต้องมีข้อความกำกับเสมอ ห้ามใช้สีสื่อความหมายเดี่ยว ๆ"""

    def __init__(self, parent, size=10, bg=T.SURFACE):
        px = max(8, int(round(size * T.scale())))
        tk.Canvas.__init__(self, parent, width=px, height=px, bg=bg,
                           highlightthickness=0, bd=0)
        self._id = self.create_oval(1, 1, px - 2, px - 2,
                                    fill=T.IDLE, outline="")

    def set(self, colour):
        self.itemconfigure(self._id, fill=colour)


class Banner(tk.Frame):
    """แถบแจ้งเตือนเต็มความกว้าง — ซ่อนตัวเองเมื่อไม่มีอะไรจะบอก

    ⚠️ ต้องสร้างใน "ช่อง" ที่ pack ไว้ตั้งแต่ตอนสร้างหน้าจอ
       ถ้าเรียก pack() ตอนจะแสดง มันจะไปต่อท้าย pack order ซึ่งอาจกลายเป็น
       อยู่ล่างสุดของหน้าจอ — แถบเตือนที่โผล่ใต้กราฟไม่มีใครเห็น
       และแถบที่ย้ายที่ไปมาตามลำดับการเกิดเหตุยิ่งแย่กว่า
    """

    KINDS = {
        "info":  (T.ACCENT_SOFT, T.ACCENT_LINE, T.ACCENT_DEEP),
        "warn":  (T.WARN_SOFT,   T.WARN_LINE,   T.WARN),
        "error": (T.ERROR_SOFT,  T.ERROR_LINE,  T.ERROR),
    }

    def __init__(self, parent):
        tk.Frame.__init__(self, parent, bg=T.WARN_SOFT,
                          highlightbackground=T.WARN_LINE,
                          highlightthickness=T.BORDER_W, bd=0)
        self.lbl = tk.Label(self, text="", bg=T.WARN_SOFT, fg=T.WARN,
                            font=T.f(T.FONT_BODY, "bold"), anchor="w",
                            padx=T.SP_2, pady=6)
        self.lbl.pack(fill="x")
        self._shown = False

    def show(self, text, kind="warn"):
        bg, line, fg = self.KINDS.get(kind, self.KINDS["warn"])
        self.configure(bg=bg, highlightbackground=line)
        self.lbl.configure(text=text, bg=bg, fg=fg)
        if not self._shown:
            self.pack(fill="x", padx=T.SP_2, pady=(0, T.SP_1))
            self._shown = True

    def hide(self):
        if self._shown:
            self.pack_forget()
            self._shown = False


class Segmented(tk.Frame):
    """ปุ่มเลือกแบบกลุ่ม — ตัวที่เลือกอยู่เป็นพื้น teal ทึบ"""

    def __init__(self, parent, options, value=None, command=None,
                 bg=T.SURFACE):
        tk.Frame.__init__(self, parent, bg=bg)
        self.command = command
        self.value = value if value is not None else options[0][1]
        self._btns = {}
        for text, val in options:
            b = tk.Label(self, text=text, font=T.f(T.FONT_LABEL),
                         padx=T.SP_2 - 2, pady=4, cursor="hand2",
                         highlightbackground=T.BORDER, highlightthickness=1)
            b.pack(side="left", padx=(0, 2))
            b.bind("<Button-1>", lambda _e, v=val: self.set(v, notify=True))
            self._btns[val] = b
        self._paint()

    def _paint(self):
        for val, b in self._btns.items():
            on = (val == self.value)
            b.configure(bg=T.ACCENT if on else T.SURFACE,
                        fg=T.ON_ACCENT if on else T.TEXT_DIM,
                        highlightbackground=T.ACCENT if on else T.BORDER)

    def set(self, value, notify=False):
        if value not in self._btns:
            return
        self.value = value
        self._paint()
        if notify and self.command:
            self.command(value)


class Button(tk.Label):
    """ปุ่มธรรมดา — ใช้ Label เพราะ tk.Button บน Windows ไม่ยอมรับสีพื้น"""

    def __init__(self, parent, text, command=None, primary=False,
                 bg=T.SURFACE):
        self.primary = primary
        self.command = command
        tk.Label.__init__(
            self, parent, text=text, font=T.f(T.FONT_BODY),
            padx=T.SP_2, pady=5, cursor="hand2",
            bg=T.ACCENT if primary else T.SURFACE,
            fg=T.ON_ACCENT if primary else T.TEXT,
            highlightbackground=T.ACCENT if primary else T.BORDER,
            highlightthickness=1)
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda _e: self.configure(
            bg=T.ACCENT_DEEP if primary else T.SURFACE_ALT))
        self.bind("<Leave>", lambda _e: self.configure(
            bg=T.ACCENT if primary else T.SURFACE))

    def _click(self, _e=None):
        if self.command:
            self.command()


class KeyValue(tk.Frame):
    """ตารางสองคอลัมน์สำหรับ Diagnostics — คีย์หรี่ ค่าเข้ม"""

    def __init__(self, parent, bg=T.SURFACE, mono=True):
        tk.Frame.__init__(self, parent, bg=bg)
        self.bg = bg
        self.mono = mono
        self._vals = {}
        self._row = 0
        self.grid_columnconfigure(1, weight=1)

    def add(self, key, value="—"):
        k = tk.Label(self, text=key, bg=self.bg, fg=T.TEXT_DIM,
                     font=T.f(T.FONT_LABEL), anchor="w")
        v = tk.Label(self, text=str(value), bg=self.bg, fg=T.TEXT,
                     font=(T.fm(T.FONT_LABEL) if self.mono
                           else T.f(T.FONT_LABEL)), anchor="w")
        k.grid(row=self._row, column=0, sticky="w", padx=(0, T.SP_2), pady=2)
        v.grid(row=self._row, column=1, sticky="w", pady=2)
        self._vals[key] = v
        self._row += 1
        return v

    def set(self, key, value, colour=None):
        w = self._vals.get(key)
        if w is None:
            w = self.add(key, value)
        w.configure(text=str(value), fg=colour or T.TEXT)
