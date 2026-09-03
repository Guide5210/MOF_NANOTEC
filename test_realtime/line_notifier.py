#!/usr/bin/env python3
"""
============================================================================
 line_notifier.py — ESP32 Water Monitor : แจ้งเตือนผ่าน LINE
============================================================================
 อ่าน CSV ที่ logger.py เขียน (ไม่แตะ serial) แล้วส่งข้อความเข้า LINE ผ่าน
 LINE Messaging API (broadcast — ส่งถึงทุกคนที่เป็นเพื่อนกับ OA ของคุณ
 ไม่ต้องหา userId ไม่ต้องทำ webhook)

 ทำ 3 อย่าง:
   1) รายงานสรุปอัตโนมัติทุก REPORT_EVERY ชั่วโมง
      (ค่าล่าสุด + สถิติช่วงที่ผ่านมา + ลิงก์เปิด dashboard / โหลด PDF)
   2) ตรวจจับ "EC ไต่ขึ้นแล้วเริ่มคงที่" -> แจ้งเตือนทันที (ครั้งเดียวต่อรอบ)
   3) แจ้งถ้าข้อมูลหยุดไหล (logger/เซนเซอร์ตาย) เกิน STALL_MIN นาที

 ประหยัดโควตา LINE ฟรี (~200-500 ข้อความ/เดือน):
   - ไม่ส่งรายงานถ้าข้อมูลหยุดไหล (ส่งแจ้ง stall ครั้งเดียวแทน)
   - แจ้งเตือนแต่ละชนิดมี cooldown กันสแปม

 ตั้งค่า: แก้ค่าในบล็อก CONFIG ด้านล่าง แล้ว
   ทดสอบ:      python3 line_notifier.py --test
   รันจริง:     python3 line_notifier.py
   เป็น service: ใช้ water-line.service
============================================================================
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta

# ============================ CONFIG (แก้ตรงนี้) ============================
# 1) Channel access token (long-lived) จาก LINE Developers console
#    (Messaging API channel -> แท็บ Messaging API -> Channel access token -> Issue)
CHANNEL_ACCESS_TOKEN = P60L0ADUGiAL4BqDlVzIYws6vMSLUejb5sjhQnNihYdh9VsSL9H8C/3FzEtxHDJXCejVBeDnY9GSmvmz0c2epsm5UJYymJRwDcpY5s9rC9LiEL2OWe8a8v5+9eAmsyVYA5Mdn5nTIETX0JnWkmwLjQdB04t89/1O/w1cDnyilFU=

# 2) ลิงก์ dashboard (Tailscale IP ของ Ubuntu) — ใช้แปะท้ายข้อความ
DASHBOARD_URL = "http://100.84.225.79:8080"

# 3) รอบรายงานอัตโนมัติ (ชั่วโมง) — 3 ชม. = ~240 ข้อความ/เดือน (พอดีโควตาฟรี)
#    ถ้าโควตาไม่พอ เปลี่ยนเป็น 6
REPORT_EVERY_H = 3

# 4) เกณฑ์ตรวจ "EC ไต่ขึ้นแล้วคงที่" (หน่วยจากข้อมูลเฉลี่ยราย 1 นาที)
RISE_WINDOW_MIN   = 30      # ช่วงมองย้อนหา "ขาขึ้น" (นาที)
RISE_MIN_DELTA    = 20.0    # ขาขึ้นต้องเพิ่ม >= เท่านี้ (µS/cm) ใน RISE_WINDOW
STABLE_WINDOW_MIN = 15      # ช่วงตรวจ "คงที่" (นาที)
STABLE_MAX_RANGE  = 5.0     # ในช่วงคงที่ max-min ต้อง <= เท่านี้ (µS/cm)
REARM_DROP_PCT    = 20.0    # EC ตกจากจุดคงที่เกิน % นี้ = เริ่มตัวอย่างใหม่ พร้อมแจ้งรอบหน้า

# 5) แจ้งข้อมูลหยุดไหล ถ้าไม่มีแถวใหม่เกิน (นาที)
STALL_MIN = 5

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_data")
CHECK_EVERY_S = 60          # รอบตรวจหลัก (วินาที)
# ============================================================================

API_URL = "https://api.line.me/v2/bot/message/broadcast"


# ---------------------------------------------------------------- LINE send
def line_send(text):
    """broadcast ข้อความถึงเพื่อนทุกคนของ OA (คืน True/False)"""
    body = json.dumps({"messages": [{"type": "text", "text": text[:4900]}]}).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        print(f"[line] HTTP {e.code}: {e.read().decode(errors='ignore')[:200]}")
        return False
    except Exception as e:
        print(f"[line] ส่งไม่ได้: {e}")
        return False
    print(f"[line] ส่งแล้ว ({len(text)} ตัวอักษร)")
    return ok


# ---------------------------------------------------------------- อ่านข้อมูล
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_recent(minutes):
    """อ่านแถวช่วง N นาทีล่าสุด (จากไฟล์วันนี้+เมื่อวานถ้าคร่อมเที่ยงคืน)"""
    cutoff = datetime.now() - timedelta(minutes=minutes)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "water_log_*.csv")))[-2:]
    rows = []
    for f in files:
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        t = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
                    except (KeyError, ValueError):
                        continue
                    if t < cutoff:
                        continue
                    rows.append((t, _f(r.get("EC_uScm")), _f(r.get("pH")),
                                 _f(r.get("Tw_C"))))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0])
    return rows


def one_min_avg(rows):
    """ย่อเป็นราย 1 นาที -> list ของ (นาที, ec_เฉลี่ย)"""
    buckets = {}
    for t, ec, _, _ in rows:
        if ec is None:
            continue
        k = t.replace(second=0)
        buckets.setdefault(k, []).append(ec)
    return sorted((k, sum(v) / len(v)) for k, v in buckets.items())


# ---------------------------------------------------------------- ข้อความ
def fmt_summary(rows, hours):
    last_t, ec, ph, tw = rows[-1]
    ecs = [r[1] for r in rows if r[1] is not None]
    phs = [r[2] for r in rows if r[2] is not None]
    lines = [
        "🌊 Water Monitor — รายงานสรุป",
        f"⏱ {last_t:%Y-%m-%d %H:%M}",
        "",
        f"EC ตอนนี้: {ec:.1f} µS/cm" if ec is not None else "EC: --",
        f"pH ตอนนี้: {ph:.2f}" if ph is not None else "pH: --",
        f"อุณหภูมิน้ำ: {tw:.1f} °C" if tw is not None else "Tw: --",
        "",
        f"ช่วง {hours} ชม.ที่ผ่านมา:",
    ]
    if ecs:
        lines.append(f"  EC  min {min(ecs):.1f} | max {max(ecs):.1f} | เฉลี่ย {sum(ecs)/len(ecs):.1f}")
        delta = ecs[-1] - ecs[0]
        trend = "↑ เพิ่มขึ้น" if delta > 2 else ("↓ ลดลง" if delta < -2 else "→ ทรงตัว")
        lines.append(f"  แนวโน้ม EC: {trend} ({delta:+.1f} µS/cm)")
    if phs:
        lines.append(f"  pH  min {min(phs):.2f} | max {max(phs):.2f}")
    lines += [
        "",
        f"📊 Dashboard: {DASHBOARD_URL}",
        f"📄 PDF ช่วงนี้: {DASHBOARD_URL}/api/export.pdf?minutes={hours*60}",
        f"📁 CSV ช่วงนี้: {DASHBOARD_URL}/api/export.csv?minutes={hours*60}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- EC stability
class StabilityDetector:
    """
    state machine: WAIT_RISE -> RISING -> (คงที่ครบ STABLE_WINDOW) -> แจ้ง -> ARMED_OFF
    รีเซ็ตกลับ WAIT_RISE เมื่อ EC ตกจากระดับคงที่เกิน REARM_DROP_PCT %
    """
    def __init__(self):
        self.notified_level = None    # EC ระดับที่แจ้งไปแล้ว (None = ยังไม่แจ้งรอบนี้)
        self.notified_at = None       # เวลาแจ้งครั้งล่าสุด
        self.min_since = None         # ค่าต่ำสุด "หลังเวลาแจ้ง" (ใช้จับ "ล้างใหม่")

    def check(self):
        span = RISE_WINDOW_MIN + STABLE_WINDOW_MIN + 5
        avg = one_min_avg(read_recent(span))
        if len(avg) < RISE_WINDOW_MIN + STABLE_WINDOW_MIN:
            return None                       # ข้อมูลยังไม่พอ

        ecs = [v for _, v in avg]

        # re-arm: ดูค่าต่ำสุดเฉพาะข้อมูล "หลังเวลาที่แจ้ง" เท่านั้น
        # (กันจุดต่ำเก่าที่ค้างในหน้าต่างทำให้ re-arm+แจ้งซ้ำวน)
        if self.notified_level is not None:
            after = [v for t, v in avg if self.notified_at and t > self.notified_at]
            if after:
                wmin = min(after)
                self.min_since = wmin if self.min_since is None else min(self.min_since, wmin)
            if (self.min_since is not None and
                    self.min_since < self.notified_level * (1 - REARM_DROP_PCT / 100)):
                print(f"[stab] EC เคยตกถึง {self.min_since:.0f} "
                      f"(จาก {self.notified_level:.0f}) หลังแจ้ง : re-arm")
                self.notified_level = None
                self.notified_at = None
                self.min_since = None
                # ไม่ return — ตรวจรอบนี้ต่อเลย เผื่อรูปแบบใหม่ครบแล้ว
            else:
                return None                   # แจ้งรอบนี้ไปแล้ว ไม่แจ้งซ้ำ

        stable_seg = ecs[-STABLE_WINDOW_MIN:]
        rise_seg   = ecs[-(RISE_WINDOW_MIN + STABLE_WINDOW_MIN):-STABLE_WINDOW_MIN]

        stable_now = (max(stable_seg) - min(stable_seg)) <= STABLE_MAX_RANGE
        rose_before = (rise_seg[-1] - rise_seg[0]) >= RISE_MIN_DELTA
        level = sum(stable_seg) / len(stable_seg)

        if rose_before and stable_now:
            self.notified_level = level
            self.notified_at = avg[-1][0]     # เวลาแท่งข้อมูลล่าสุด
            return level
        return None


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description="LINE notifier for water monitor")
    ap.add_argument("--test", action="store_true", help="ส่งข้อความทดสอบ 1 ครั้งแล้วจบ")
    args = ap.parse_args()

    if "PASTE_YOUR" in CHANNEL_ACCESS_TOKEN:
        print("!! ยังไม่ได้ใส่ CHANNEL_ACCESS_TOKEN — แก้ในไฟล์ก่อน (บล็อก CONFIG)")
        sys.exit(1)

    if args.test:
        ok = line_send("✅ ทดสอบระบบแจ้งเตือน Water Monitor สำเร็จ!\n"
                       f"Dashboard: {DASHBOARD_URL}")
        sys.exit(0 if ok else 1)

    print(f"[notifier] เริ่ม | รายงานทุก {REPORT_EVERY_H} ชม. | ตรวจทุก {CHECK_EVERY_S}s")
    det = StabilityDetector()
    last_report = datetime.now()          # เริ่มนับจากตอนสตาร์ท (ไม่ยิงทันที)
    stall_notified = False

    while True:
        try:
            now = datetime.now()
            recent = read_recent(STALL_MIN + 1)
            flowing = bool(recent) and (now - recent[-1][0]).total_seconds() < STALL_MIN * 60

            # --- แจ้งข้อมูลหยุดไหล (ครั้งเดียวจนกว่าจะกลับมา) ---
            if not flowing and not stall_notified:
                line_send("⚠️ Water Monitor: ข้อมูลหยุดไหลเกิน "
                          f"{STALL_MIN} นาที\nเช็ก logger/สาย USB/เซนเซอร์\n{DASHBOARD_URL}")
                stall_notified = True
            elif flowing and stall_notified:
                line_send("✅ Water Monitor: ข้อมูลกลับมาไหลปกติแล้ว")
                stall_notified = False

            # --- EC คงที่ ---
            if flowing:
                level = det.check()
                if level is not None:
                    line_send("🔔 EC เริ่มคงที่แล้ว!\n"
                              f"ระดับ ~{level:.1f} µS/cm "
                              f"(นิ่งต่อเนื่อง {STABLE_WINDOW_MIN} นาที หลังช่วงไต่ขึ้น)\n"
                              f"📊 {DASHBOARD_URL}")

            # --- รายงานตามรอบ (ส่งเฉพาะถ้าข้อมูลไหลอยู่ = ไม่เปลืองโควตา) ---
            if (now - last_report) >= timedelta(hours=REPORT_EVERY_H):
                last_report = now
                if flowing:
                    rows = read_recent(REPORT_EVERY_H * 60)
                    if rows:
                        line_send(fmt_summary(rows, REPORT_EVERY_H))

        except Exception as e:
            print(f"[notifier] error (ทำงานต่อ): {e}")

        time.sleep(CHECK_EVERY_S)


if __name__ == "__main__":
    main()
