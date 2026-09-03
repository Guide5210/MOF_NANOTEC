#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
หน้าต่าง Diagnostics — แยกจาก dashboard โดยตั้งใจ

⚠️ ตัวเลขในนี้ (heap, queued, malformed, boot_id) ตอบคำถามของคนที่กำลัง
   ไล่ปัญหาของ "ระบบ"  ส่วน dashboard ตอบคำถามของคนที่กำลังทำ "การวัด"
   ถ้าเอามาปนกัน คนวัดจะเห็นตัวเลขที่ตัวเองตีความไม่ได้เต็มหน้าจอ
   แล้วเริ่มไม่ไว้ใจตัวเลขที่สำคัญจริง ๆ ที่อยู่ข้าง ๆ กัน
"""

import os
import tkinter as tk

from . import lab_theme as T
from .view_model import mask_text
from .widgets import Button, KeyValue, card

LINK_COLOUR = {"ONLINE": T.OK, "OFFLINE": T.IDLE,
               "DISABLED": T.IDLE, "ERROR": T.ERROR}


class DiagnosticsWindow(tk.Toplevel):
    def __init__(self, parent, app):
        tk.Toplevel.__init__(self, parent, bg=T.BG)
        self.app = app
        self.title("Diagnostics — EC Measurement Station")
        self.geometry("660x660")
        self.configure(bg=T.BG)
        self.protocol("WM_DELETE_WINDOW", self.hide)

        pad = T.SP_2
        # ---- ลิงก์ ----
        box = card(self)
        box.pack(fill="x", padx=pad, pady=(pad, T.SP_1))
        tk.Label(box, text="P4 LINK", bg=T.SURFACE, fg=T.TEXT_DIM,
                 font=T.f(T.FONT_LABEL, "bold")).pack(anchor="w", padx=pad,
                                                      pady=(T.SP_1, 0))
        self.link = KeyValue(box)
        self.link.pack(fill="x", padx=pad, pady=(4, T.SP_1))
        for k in ("link", "heartbeat age", "boot_id", "reboots / reconnects",
                  "display_mask", "mask sync state", "queued",
                  "heap", "heap_big", "P4 sees PC"):
            self.link.add(k)

        # ---- เฟรม ----
        box2 = card(self)
        box2.pack(fill="x", padx=pad, pady=(0, T.SP_1))
        tk.Label(box2, text="FRAMES", bg=T.SURFACE, fg=T.TEXT_DIM,
                 font=T.f(T.FONT_LABEL, "bold")).pack(anchor="w", padx=pad,
                                                      pady=(T.SP_1, 0))
        self.frames = KeyValue(box2)
        self.frames.pack(fill="x", padx=pad, pady=(4, T.SP_1))
        for k in ("lines received", "valid frames", "malformed (by reason)",
                  "events stored", "duplicate events", "commands rejected",
                  "state sent / failed", "last error"):
            self.frames.add(k)

        # ---- ฝั่ง PC ----
        box3 = card(self)
        box3.pack(fill="x", padx=pad, pady=(0, T.SP_1))
        tk.Label(box3, text="PC LOGGER / DATA", bg=T.SURFACE, fg=T.TEXT_DIM,
                 font=T.f(T.FONT_LABEL, "bold")).pack(anchor="w", padx=pad,
                                                      pady=(T.SP_1, 0))
        self.pc = KeyValue(box3)
        self.pc.pack(fill="x", padx=pad, pady=(4, T.SP_1))
        for k in ("logger state", "rec_status age", "session mask",
                  "sample", "CSV rows today", "last row age", "lines skipped",
                  "event log folder"):
            self.pc.add(k)

        # ---- event ล่าสุดพร้อม id เต็ม ----
        box4 = card(self)
        box4.pack(fill="both", expand=True, padx=pad, pady=(0, T.SP_1))
        tk.Label(box4, text="RAW EVENT IDS", bg=T.SURFACE, fg=T.TEXT_DIM,
                 font=T.f(T.FONT_LABEL, "bold")).pack(anchor="w", padx=pad,
                                                      pady=(T.SP_1, 0))
        self.raw = tk.Text(box4, bg=T.SURFACE, fg=T.TEXT, bd=0,
                           highlightthickness=0, font=T.fm(T.FONT_LABEL),
                           height=6, wrap="none")
        self.raw.pack(fill="both", expand=True, padx=pad, pady=(4, T.SP_1))
        self.raw.configure(state="disabled")

        bar = tk.Frame(self, bg=T.BG)
        bar.pack(fill="x", padx=pad, pady=(0, pad))
        Button(bar, "Open event log folder", command=self._open_events,
               bg=T.BG).pack(side="left")
        Button(bar, "Close", command=self.hide, bg=T.BG).pack(side="right")

    # ------------------------------------------------------------------
    def hide(self):
        self.withdraw()

    def _open_events(self):
        d = self.app.event_log.dir
        try:
            os.startfile(d)                     # Windows เท่านั้น
        except AttributeError:
            print("[diag] โฟลเดอร์ event log:", d)
        except OSError as e:
            print("[diag] เปิดโฟลเดอร์ไม่ได้:", e)

    # ------------------------------------------------------------------
    def refresh(self, snap, pc, csv, feed):
        c = snap.get("counters", {})
        age = snap.get("hb_age_s")
        self.link.set("link", snap.get("link_text", "—"),
                      LINK_COLOUR.get(snap.get("link"), T.TEXT))
        self.link.set("heartbeat age",
                      "—" if age is None else "%.1f s ago" % age,
                      T.WARN if (age or 0) > 10 else T.TEXT)
        self.link.set("boot_id", snap.get("boot_id") or "—")
        self.link.set("reboots / reconnects", snap.get("reboots", 0))
        dm = snap.get("display_mask")
        self.link.set("display_mask",
                      "—  (not reported yet)" if dm is None
                      else "%d  (%s)" % (dm, mask_text(dm)))
        self.link.set("mask sync state", snap.get("mask_state", "—"))
        q = snap.get("queued")
        self.link.set("queued", "—" if q is None else "%d / 32" % q,
                      T.WARN if (q or 0) >= 24 else T.TEXT)
        for k, f in (("heap", "heap"), ("heap_big", "heap_big")):
            v = snap.get(f)
            self.link.set(k, "—" if v is None else "{:,}".format(v))
        self.link.set("P4 sees PC", snap.get("p4_sees_pc") or "—")

        bad = (c.get("dropped_parse", 0) + c.get("dropped_field", 0)
               + c.get("dropped_version", 0) + c.get("dropped_oversize", 0))
        self.frames.set("lines received", "{:,}".format(c.get("rx_lines", 0)))
        self.frames.set("valid frames", "{:,}".format(c.get("rx_frames", 0)))
        self.frames.set(
            "malformed (by reason)",
            "{:,}   json {} · field {} · version {} · oversize {}".format(
                bad, c.get("dropped_parse", 0), c.get("dropped_field", 0),
                c.get("dropped_version", 0), c.get("dropped_oversize", 0)),
            T.WARN if bad else T.TEXT)
        self.frames.set("events stored", "{:,}".format(c.get("events", 0)))
        self.frames.set("duplicate events", "{:,}  (UI {})".format(
            c.get("dup_events", 0), feed.duplicates))
        self.frames.set("commands rejected", c.get("cmds_nacked", 0))
        self.frames.set("state sent / failed", "%d / %d" % (
            c.get("state_sent", 0), c.get("state_failed", 0)))
        err = snap.get("error")
        self.frames.set("last error", err or "—",
                        T.ERROR if err else T.TEXT)

        self.pc.set("logger state", pc.get("liveness_text", "—"),
                    {"ONLINE": T.OK, "STALE": T.WARN,
                     "OFFLINE": T.IDLE}.get(pc.get("liveness"), T.TEXT))
        d = pc.get("data_age_s")
        self.pc.set("last row age", "—" if d is None else "%.1f s" % d,
                    T.OK if (d is not None and d <= 10) else T.WARN)
        a = pc.get("rec_age_s")
        # ไฟล์นี้อัปเดตเฉพาะตอนเริ่ม/จบ session — เก่าไม่ได้แปลว่าผิด
        self.pc.set("rec_status age",
                    "—" if a is None else "%.0f s  (อัปเดตเฉพาะตอนเริ่ม/จบ session)" % a)
        self.pc.set("session mask", "%d  (%s)" % (pc.get("session_mask", 0),
                                                  mask_text(pc.get("session_mask", 0))))
        self.pc.set("sample", pc.get("sample_id") or "—")
        self.pc.set("CSV rows today", "{:,}".format(pc.get("csv_rows", 0)))
        self.pc.set("lines skipped", "%d  (read errors %d)" % (
            csv.skipped_lines, csv.read_errors))
        self.pc.set("event log folder", self.app.event_log.dir)

        self.raw.configure(state="normal")
        self.raw.delete("1.0", "end")
        for r in feed.visible(30):
            self.raw.insert("end", "%s  %-22s %-20s %s\n" % (
                r["when"].strftime("%H:%M:%S"), r["kind"],
                r["event_id"] or "-", r["extra"] or ""))
        self.raw.configure(state="disabled")
