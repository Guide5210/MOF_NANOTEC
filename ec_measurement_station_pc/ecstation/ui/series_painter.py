#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 series_painter.py — ตรรกะเส้นกราฟ โดยไม่แตะ tkinter เลย
============================================================================
 ⚠️ ghost series คืออะไร และทำไมต้องแยกไฟล์นี้ออกมา

    เมื่อผู้ใช้ซ่อนเซนเซอร์ วิธีที่ง่ายที่สุดคือ set_data([], []) แล้วปล่อยไว้
    เส้นจะหายจากตา แต่ Line2D ยังอยู่ใน ax.lines, ยังอยู่ใน legend, และยัง
    กิน prop_cycle ไปเรื่อย ๆ  พอเปิดกลับมาสีจะเลื่อน ผู้ใช้ที่จับคู่เส้นกับ
    เซนเซอร์ด้วยสี (แบบเดียวกับที่ทำบนจอ P4) จะอ่านผิดทันที

    ไฟล์นี้ .remove() ของจริงเสมอ — และเพราะไม่พึ่ง tkinter จึงทดสอบได้
    บนเครื่องที่ไม่มีหน้าจอ ซึ่งเป็นเงื่อนไขที่ทำให้เทสต์ A1 มีอยู่จริงได้
============================================================================
"""

import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from . import lab_theme as T

MAX_SPLIT_PANELS = 3


class SeriesPainter(object):
    """จัดการเส้นบน Figure หนึ่งอัน — ไม่ผูกกับ Tk เลย

    ⚠️ แยกออกมาเพื่อให้เทสต์ ghost series รันได้โดยไม่ต้องมีหน้าจอ
       ถ้าตรรกะนี้ฝังอยู่ในวิดเจ็ต จะทดสอบได้แค่ด้วยตาคน
    """

    def __init__(self, fig):
        self.fig = fig
        T.mpl_style_figure(fig)
        self._axes = []
        self._lines = {}          # (ax_i, sensor) -> Line2D
        self._layout_key = None
        self.paused = False

    # ------------------------------------------------------------------
    def _ensure_layout(self, series, mode):
        n = len(series)
        split = (mode == "split" and 0 < n <= MAX_SPLIT_PANELS)
        key = (tuple(s[0] for s in series), split)
        if key == self._layout_key:
            return split
        self._layout_key = key

        self.fig.clear()
        self._axes, self._lines = [], {}
        if n == 0:
            ax = self.fig.add_subplot(111)
            T.mpl_style_axes(ax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "ไม่มีเซนเซอร์ที่เลือกให้แสดง",
                    ha="center", va="center", color=T.TEXT_DIM,
                    fontsize=10, transform=ax.transAxes)
            self._axes = [ax]
            return split

        panels = n if split else 1
        for i in range(panels):
            ax = self.fig.add_subplot(panels, 1, i + 1)
            T.mpl_style_axes(ax, labelbottom=(i == panels - 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self._axes.append(ax)
        # top เว้นที่ให้แถบ legend ของโหมด overlay · hspace กันป้ายแกน y ชนกัน
        self.fig.subplots_adjust(left=0.085, right=0.99,
                                 top=0.86 if panels == 1 else 0.97,
                                 bottom=0.22, hspace=0.36)
        return split

    # ------------------------------------------------------------------
    def draw(self, rows, series, mode="split"):
        """series = [(index, colour, label, hidden), ...] ตามลำดับที่จะวาด"""
        split = self._ensure_layout(series, mode)
        if not series:
            return
        xs = [r["t"] for r in rows]
        wanted = set()
        for k, (idx, colour, lbl, hidden) in enumerate(series):
            ax_i = k if split else 0
            ax = self._axes[ax_i]
            key = (ax_i, idx)
            wanted.add(key)
            ys = [(r["ec"][idx] if idx < len(r["ec"]) else None) for r in rows]
            ln = self._lines.get(key)
            if ln is None:
                ln, = ax.plot(xs, ys, color=colour, label=lbl, linewidth=1.6,
                              linestyle="--" if hidden else "-")
                self._lines[key] = ln
            else:
                ln.set_data(xs, ys)
                ln.set_color(colour)
                ln.set_label(lbl)
                ln.set_linestyle("--" if hidden else "-")

        # ---- ลบเส้นที่ไม่ควรมีอยู่แล้วออกจริง ๆ (กัน ghost series) ----
        for key in [k for k in self._lines if k not in wanted]:
            try:
                self._lines[key].remove()
            except Exception:
                pass
            del self._lines[key]

        for ax_i, ax in enumerate(self._axes):
            has = [ln for (a, _s), ln in self._lines.items() if a == ax_i]
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            if has and not split:
                # legend อยู่ "เหนือ" แกน ไม่ใช่ทับบนเส้น
                # ทับบนเส้นแล้วบังข้อมูลช่วงต้น ซึ่งเป็นช่วงที่ค่าเปลี่ยนเร็วที่สุด
                T.mpl_style_legend(ax.legend(
                    loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.14),
                    mode="expand", borderaxespad=0, ncol=max(1, len(has))))
            elif has and split:
                # ในโหมด split ป้ายแกน y แคบมาก — ใช้ชื่อสั้นอย่างเดียว
                # ส่วน "(hidden on HMI)" สื่อด้วยเส้นประ + ป้ายบนการ์ดแทน
                lbl = has[0].get_label().split("  (")[0]
                ax.set_ylabel(lbl, color=T.TEXT_DIM, fontsize=9, labelpad=2)
            if not self.paused:
                ax.relim()
                ax.autoscale_view()

    # ---- ไว้ให้เทสต์ตรวจว่าไม่มีเส้นค้าง ----
    def line_keys(self):
        return sorted(self._lines)

    def sensors_drawn(self):
        return sorted({s for _ax, s in self._lines})

    def artist_count(self):
        """จำนวน Line2D ที่ยังอยู่บนแกนจริง ๆ — ตัวเลขนี้คือที่ ghost ซ่อนตัว"""
        return sum(len(ax.lines) for ax in self._axes)

    def legend_labels(self):
        out = []
        for ax in self._axes:
            leg = ax.get_legend()
            if leg is not None:
                out += [t.get_text() for t in leg.get_texts()]
            elif ax.get_ylabel():
                out.append(ax.get_ylabel())
        return out

    def reset_view(self):
        for ax in self._axes:
            ax.relim(); ax.autoscale_view()
        self.paused = False
