#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
แผงกราฟ — สีเส้นตรง UI_SERIES_1..3 ของ P4 เป๊ะ

⚠️ ghost series คืออะไร และทำไมต้องระวัง
   เมื่อผู้ใช้ซ่อนเซนเซอร์ วิธีที่ง่ายที่สุดคือ set_data([], []) แล้วปล่อยไว้
   เส้นจะหายจากตา แต่ Line2D ยังอยู่ใน ax.lines, ยังอยู่ใน legend,
   และยังกิน prop_cycle ไปเรื่อย ๆ  พอเปิดกลับมาสีจะเลื่อน ผู้ใช้ที่จับคู่
   เส้นกับเซนเซอร์ด้วยสี (แบบเดียวกับที่ทำบนจอ P4) จะอ่านผิดทันที
   ที่นี่จึง .remove() ของจริงเสมอ และมีเทสต์ A1 ไล่จับ
"""

import tkinter as tk

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from . import lab_theme as T
from .series_painter import MAX_SPLIT_PANELS, SeriesPainter   # noqa: F401


class LabToolbar(NavigationToolbar2Tk):
    """เหลือเฉพาะปุ่มที่ใช้จริง — Back/Forward เรนเดอร์เป็นตารางหมากรุกบนพื้นขาว"""
    toolitems = [t for t in NavigationToolbar2Tk.toolitems
                 if t[0] in ("Home", "Pan", "Zoom", "Save")]

    def __init__(self, canvas, parent, on_user_zoom=None):
        NavigationToolbar2Tk.__init__(self, canvas, parent,
                                      pack_toolbar=False)
        self.on_user_zoom = on_user_zoom
        self.configure(bg=T.SURFACE)
        for child in self.winfo_children():
            try:
                child.configure(bg=T.SURFACE)
            except tk.TclError:
                pass

    def set_message(self, s):
        pass                     # ไม่แสดงพิกัดเมาส์ — เป็น debug noise

    def release_zoom(self, event):
        NavigationToolbar2Tk.release_zoom(self, event)
        if self.on_user_zoom:
            self.on_user_zoom(True)

    def release_pan(self, event):
        NavigationToolbar2Tk.release_pan(self, event)
        if self.on_user_zoom:
            self.on_user_zoom(True)

    def home(self, *a):
        NavigationToolbar2Tk.home(self, *a)
        if self.on_user_zoom:
            self.on_user_zoom(False)


class ChartPanel(tk.Frame):
    def __init__(self, parent, on_history_change=None):
        tk.Frame.__init__(self, parent, bg=T.SURFACE,
                          highlightbackground=T.BORDER,
                          highlightthickness=T.BORDER_W, bd=0)
        self.on_history_change = on_history_change
        self.fig = Figure(figsize=(8, 2.6), dpi=100)
        self.painter = SeriesPainter(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = LabToolbar(self.canvas, self,
                                  on_user_zoom=self._user_zoom)
        # toolbar ต้อง pack ก่อน canvas ไม่งั้นถูก canvas ที่ expand บีบหายไป
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    @property
    def paused(self):
        return self.painter.paused

    def _user_zoom(self, paused):
        if paused != self.painter.paused:
            self.painter.paused = paused
            if self.on_history_change:
                self.on_history_change(paused)

    def draw(self, rows, series, mode="split"):
        self.painter.draw(rows, series, mode)
        self.canvas.draw_idle()

    def reset_view(self):
        self.painter.reset_view()
        self.canvas.draw_idle()

    def line_keys(self):
        return self.painter.line_keys()
