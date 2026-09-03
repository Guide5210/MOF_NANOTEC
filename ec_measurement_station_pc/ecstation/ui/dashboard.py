#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
หน้าจอหลัก — ผู้อ่านอย่างเดียว

⚠️ ไม่มีปุ่มไหนในหน้านี้ที่เขียนอะไรลงโฟลเดอร์ของระบบเดิม
   ปุ่ม Reports เปิดโฟลเดอร์อย่างเดียว (os.startfile) ไม่สร้างรายงาน
   การสร้างรายงานยังเป็นของ report_3ec.py ในระบบเดิมทั้งหมด
"""

import os
import tkinter as tk

from . import lab_theme as T
from . import view_model as VM
from .cards import CardArea
from .chart import ChartPanel
from .diagnostics import DiagnosticsWindow
from .events import EventPanel
from .widgets import Banner, Button, Dot, Segmented, card

WINDOWS = [("15m", 15), ("1h", 60), ("6h", 360), ("All", None)]


class Dashboard(tk.Frame):
    def __init__(self, root, app):
        tk.Frame.__init__(self, root, bg=T.BG)
        self.root = root
        self.app = app
        self.model = app.model
        self._diag = None
        self._build()

    # ==================================================================
    def _build(self):
        pad = T.SP_2
        # ---------------- header ----------------
        head = card(self)
        head.pack(fill="x", padx=pad, pady=(pad, T.SP_1))
        inner = tk.Frame(head, bg=T.SURFACE)
        inner.pack(fill="x", padx=pad, pady=T.SP_1)

        # ปุ่มขวาต้อง pack ก่อนฝั่งซ้ายที่ expand ไม่งั้นข้อความจะถูกตัด
        btns = tk.Frame(inner, bg=T.SURFACE)
        btns.pack(side="right")
        Button(btns, "Diagnostics", command=self.open_diagnostics).pack(
            side="right", padx=(T.SP_1, 0))
        Button(btns, "Settings", command=self.open_settings).pack(
            side="right", padx=(T.SP_1, 0))
        Button(btns, "Reports", command=self.open_reports).pack(side="right")

        status = tk.Frame(inner, bg=T.SURFACE)
        status.pack(side="right", padx=(0, T.SP_3))
        r1 = tk.Frame(status, bg=T.SURFACE); r1.pack(anchor="e")
        self.pc_dot = Dot(r1, 10); self.pc_dot.pack(side="left", padx=(0, 6))
        self.pc_txt = tk.Label(r1, text="PC LOGGER OFFLINE", bg=T.SURFACE,
                               fg=T.IDLE, font=T.f(T.FONT_BODY, "bold"))
        self.pc_txt.pack(side="left")
        r2 = tk.Frame(status, bg=T.SURFACE); r2.pack(anchor="e")
        self.p4_dot = Dot(r2, 10); self.p4_dot.pack(side="left", padx=(0, 6))
        self.p4_txt = tk.Label(r2, text="P4 OFFLINE", bg=T.SURFACE,
                               fg=T.IDLE, font=T.f(T.FONT_LABEL))
        self.p4_txt.pack(side="left")

        left = tk.Frame(inner, bg=T.SURFACE)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="EC MEASUREMENT STATION", bg=T.SURFACE, fg=T.TEXT,
                 font=T.f(T.FONT_TITLE, "bold"), anchor="w").pack(anchor="w")
        self.sub = tk.Label(left, text="", bg=T.SURFACE, fg=T.TEXT_DIM,
                            font=T.f(T.FONT_LABEL), anchor="w")
        self.sub.pack(anchor="w")

        # ---------------- แถบแจ้งเตือน ----------------
        # ช่องว่างที่จองไว้ใต้ header — สูง 0 เมื่อไม่มีอะไรจะบอก
        self.banner_slot = tk.Frame(self, bg=T.BG)
        self.banner_slot.pack(fill="x")
        self.banner = Banner(self.banner_slot)

        # ---------------- การ์ด ----------------
        self.cards = CardArea(self)
        self.cards.pack(fill="x", padx=pad)

        # ---------------- เหตุการณ์ ----------------
        # ⚠️ ต้อง pack ก่อนกราฟ และยึด side="bottom"
        #    กราฟมี expand=True ถ้าปล่อยให้แผงนี้ pack ทีหลัง มันจะถูกบีบจน
        #    แถวล่างโดนตัดครึ่ง — ซึ่งคือแถวที่มีเหตุการณ์เก่ากว่า
        self.events = EventPanel(self, rows=3)
        self.events.pack(side="bottom", fill="x", padx=pad, pady=(T.SP_1, pad))

        # ---------------- กราฟ ----------------
        chart_box = tk.Frame(self, bg=T.BG)
        chart_box.pack(fill="both", expand=True, padx=pad, pady=(T.SP_1, 0))
        bar = tk.Frame(chart_box, bg=T.BG)
        bar.pack(fill="x", pady=(0, 4))
        tk.Label(bar, text="TREND", bg=T.BG, fg=T.TEXT_DIM,
                 font=T.f(T.FONT_LABEL, "bold")).pack(side="left")
        self.win_sel = Segmented(bar, WINDOWS, self.model.window_minutes,
                                 command=self._set_window, bg=T.BG)
        self.win_sel.pack(side="right")
        self.mode_sel = Segmented(bar, [("Split", "split"),
                                        ("Overlay", "overlay")],
                                  self.model.chart_mode,
                                  command=self._set_mode, bg=T.BG)
        self.mode_sel.pack(side="right", padx=(0, T.SP_2))
        self.eng = tk.IntVar(value=0)
        eng = tk.Checkbutton(
            bar, text="Engineering view", variable=self.eng,
            command=self._toggle_eng, bg=T.BG, fg=T.TEXT_DIM,
            activebackground=T.BG, activeforeground=T.TEXT,
            selectcolor=T.SURFACE, font=T.f(T.FONT_LABEL),
            highlightthickness=0, bd=0)
        eng.pack(side="right", padx=(0, T.SP_2))

        self.chart = ChartPanel(chart_box,
                                on_history_change=self._history_changed)
        self.chart.pack(fill="both", expand=True)

    # ==================================================================
    def _set_window(self, minutes):
        self.model.window_minutes = minutes
        self.app.save_ui_state()
        self.refresh()

    def _set_mode(self, mode):
        self.model.chart_mode = mode
        self.app.save_ui_state()
        self.refresh()

    def _toggle_eng(self):
        self.model.engineering = bool(self.eng.get())
        self.refresh()

    def _history_changed(self, paused):
        self.model.history_paused = paused
        self.refresh()

    # ------------------------------------------------------------------
    def open_reports(self):
        """เปิดโฟลเดอร์รายงานของระบบเดิม — อ่านอย่างเดียว ไม่สร้างอะไร"""
        d = self.app.legacy_reports_dir()
        if not d or not os.path.isdir(d):
            self.banner.show("ยังไม่ได้ตั้งค่าโฟลเดอร์รายงานของระบบเดิม "
                             "(config/app_config.json → legacy.reports_dir)",
                             "info")
            return
        try:
            os.startfile(d)
        except AttributeError:
            print("[reports] โฟลเดอร์:", d)
        except OSError as e:
            self.banner.show("เปิดโฟลเดอร์รายงานไม่ได้: %s" % e, "error")

    def open_settings(self):
        self.banner.show(
            "P1-B เป็นตัวแสดงผลอย่างเดียว — ค่าตั้งอยู่ที่ "
            "config/app_config.json และ data/ui_state.json", "info")

    def open_diagnostics(self):
        if self._diag is None or not self._diag.winfo_exists():
            self._diag = DiagnosticsWindow(self.root, self.app)
        self._diag.deiconify()
        self._diag.lift()

    # ==================================================================
    def refresh(self):
        app = self.app
        snap = app.bridge_snapshot()
        pc = app.pc_snapshot()
        mask, mask_note = self.model.resolve_mask(snap)

        sensors = self.model.sensors(app.csv, mask)
        visible = self.model.visible_sensors(sensors)

        # ---- header ----
        badge = VM.link_badge(snap)
        self.p4_dot.set(badge["colour"])
        self.p4_txt.configure(text=badge["text"], fg=badge["colour"])
        live = {"ONLINE": T.OK, "STALE": T.WARN,
                "OFFLINE": T.IDLE}.get(pc.get("liveness"), T.IDLE)
        self.pc_dot.set(live)
        self.pc_txt.configure(text=pc.get("liveness_text", "—"), fg=live)
        sample = pc.get("sample_id") or "ยังไม่ได้ตั้งชื่อตัวอย่าง"
        self.sub.configure(text="%s  ·  %s  ·  %s" % (
            sample, T.format_freshness(app.csv.age_s()),
            self.model.summary(sensors, mask)))

        # ---- แถบแจ้งเตือน: เรื่องที่ร้ายแรงกว่าอยู่ก่อน ----
        if snap.get("error"):
            self.banner.show("P4 BRIDGE ERROR — %s" % snap["error"], "error")
        elif pc.get("liveness") == "OFFLINE":
            self.banner.show("PC LOGGER OFFLINE — ไม่มีใครกำลังเก็บข้อมูล",
                             "error")
        elif mask_note:
            self.banner.show(mask_note, "warn")
        elif self.model.history_paused:
            self.banner.show(VM.HISTORY_NOTE, "warn")
        else:
            self.banner.hide()

        # ---- การ์ด ----
        density = T.density_for(self.root.winfo_height())
        self.cards.render(visible, density, self.model.ec_decimals)

        # ---- กราฟ ----
        rows = app.csv.since(self.model.window_minutes)
        self.chart.draw(rows, self.model.chart_series(sensors),
                        self.model.chart_mode)

        # ---- เหตุการณ์ ----
        self.events.render(self.model.events, self.model.ec_decimals)

        if self._diag is not None and self._diag.winfo_exists():
            try:
                self._diag.refresh(snap, pc, app.csv, self.model.events)
            except tk.TclError:
                pass
