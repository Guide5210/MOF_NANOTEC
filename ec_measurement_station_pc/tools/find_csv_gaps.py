# -*- coding: utf-8 -*-
"""หาว่าช่องว่างใน CSV เกิดตอนไหน แล้วบอกว่าอะไรเกิดขึ้นตอนนั้น

ทำไมต้องมี
    hw_test รายงานแค่ csv_gap_max ซึ่งเป็นค่าสูงสุดค่าเดียว
    ช่องว่าง 9 วินาทีครั้งเดียวใน 24 ชั่วโมง กับ 9 วินาทีวันละร้อยครั้ง
    ให้ตัวเลขเดียวกันเป๊ะ แต่เป็นคนละเรื่องกันสิ้นเชิง

    ตัวนี้บอกทั้งจำนวน เวลาที่เกิด และจับคู่กับ log ของจอให้อัตโนมัติ
    เพื่อแยกว่าเป็นเรื่องของ host (USB หลุด / เครื่องติดขัด) หรือของระบบจริง

ใช้
    python tools\\find_csv_gaps.py
    python tools\\find_csv_gaps.py --csv <ไฟล์> --p4log <ไฟล์> --min-gap 6
"""
import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime, timedelta


def out(text):
    sys.stdout.buffer.write((text + "\n").encode("utf-8", "replace"))


def load_rows(path):
    times = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if not row or not row[0][:4].isdigit():
                continue
            try:
                times.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    return times


def load_p4log(path):
    """คืน [(เวลานาฬิกา, ข้อความ)] — ต้องมี wall clock ในไฟล์จึงจะจับคู่ได้"""
    if not path or not os.path.exists(path):
        return []
    hits = []
    pat = re.compile(r"^\[\s*\d+\]\s+(\d\d:\d\d:\d\d)\s+(.*)$")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.match(line.rstrip())
            if m:
                hits.append((m.group(1), m.group(2)))
    return hits


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV ของ logger เดิม")
    ap.add_argument("--p4log", default=None, help="log ของจอจาก watch_p4_log.py")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="ช่องว่างกี่วินาทีขึ้นไปถึงจะรายงาน (0 = คิดจากจังหวะจริง)")
    a = ap.parse_args()

    path = a.csv or newest("C:/MOF_NanoTec/test_realtime/water_data/water_log_*.csv")
    if not path:
        out("ไม่พบไฟล์ CSV")
        return 2
    p4 = a.p4log or newest(os.path.join(root, "data", "diag", "p4log_*.log"))

    times = load_rows(path)
    if len(times) < 10:
        out("แถวน้อยเกินไป (%d) — ไฟล์ผิดหรือเปล่า" % len(times))
        return 2

    gaps = [(a_, (b - a_).total_seconds()) for a_, b in zip(times, times[1:])]
    span = (times[-1] - times[0]).total_seconds()
    cadence = span / (len(times) - 1)

    # เกณฑ์ปกติ: กว้างกว่าสองรอบ = มีแถวหายจริงอย่างน้อยหนึ่งแถว
    # บวก 1 วินาทีเผื่อความละเอียดของ timestamp ที่เป็นวินาทีเต็ม
    limit = a.min_gap if a.min_gap > 0 else cadence * 2 + 1.0

    big = [(t, g) for t, g in gaps if g >= limit]

    out("=" * 74)
    out(" ไฟล์: %s" % path)
    out("=" * 74)
    out("  แถวทั้งหมด      %s" % format(len(times), ","))
    out("  ช่วงเวลา        %s ถึง %s (%.2f ชม.)"
        % (times[0], times[-1], span / 3600.0))
    out("  จังหวะเฉลี่ย     %.4f วิ/แถว" % cadence)
    out("  เกณฑ์ช่องว่าง    >= %.1f วิ" % limit)
    out("")
    out("  ช่องว่างที่เกินเกณฑ์: %d ครั้ง จาก %s ช่อง (%.4f%%)"
        % (len(big), format(len(gaps), ","), 100.0 * len(big) / len(gaps)))
    if big:
        lost = sum(int(round(g / cadence)) - 1 for _, g in big)
        out("  แถวที่หายไปรวม     ~%d แถว (%.4f%% ของที่ควรมี)"
            % (lost, 100.0 * lost / (len(times) + lost)))
    out("")

    if not big:
        out("  ไม่มีช่องว่างผิดปกติเลย")
        return 0

    p4lines = load_p4log(p4)
    if p4:
        out("  จับคู่กับ log ของจอ: %s" % os.path.basename(p4))
    out("")
    out("  %-21s %8s   %s" % ("เกิดตอน", "กว้าง", "จอพูดอะไรในช่วงนั้น"))
    out("  " + "-" * 70)
    for t, g in big[:40]:
        note = ""
        if p4lines:
            # ดูบรรทัดของจอในหน้าต่าง [เริ่มช่องว่าง, จบช่องว่าง + 5 วิ]
            lo = t.strftime("%H:%M:%S")
            hi = (t + timedelta(seconds=g + 5)).strftime("%H:%M:%S")
            near = [m for w, m in p4lines if lo <= w <= hi]
            hot = [m for m in near
                   if any(k in m for k in ("พอร์ตหลุด", "Guru", "CORRUPT",
                                           "ESP-ROM", "watchdog", " E "))]
            if hot:
                note = hot[0][:44]
            elif near:
                note = "ปกติ (%d บรรทัด)" % len(near)
            else:
                note = "จอเงียบ"
        # ข้ามวันใหม่ = logger ปิดไฟล์เก่าเปิดไฟล์ใหม่ ซึ่งกินเวลา
        if t.hour == 23 and t.minute == 59:
            note = (note + "  <- ข้ามวัน (เปลี่ยนไฟล์)").strip()
        out("  %-21s %6.0f วิ   %s" % (t, g, note))
    if len(big) > 40:
        out("  ... อีก %d ครั้ง" % (len(big) - 40))
    out("")
    out("  วิธีอ่าน")
    out("    'พอร์ตหลุด' หรือ 'ESP-ROM'  -> เรื่องของ host / จอรีบูต ไม่ใช่ข้อมูลหาย")
    out("    'ปกติ' หรือ 'จอเงียบ'        -> จอทำงานปกติ ช่องว่างอยู่ฝั่ง PC")
    out("    'ข้ามวัน'                    -> logger เปลี่ยนไฟล์ เป็นช่องว่างที่คาดได้")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
