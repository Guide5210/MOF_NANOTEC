#!/usr/bin/env python3
"""
============================================================================
 logger.py  —  ESP32 Water Monitor : ตัวเก็บข้อมูล realtime ลง CSV
============================================================================
 หน้าที่: อ่านบรรทัด "DATA,..." จาก ESP32 ผ่าน serial แล้วเขียนลง CSV ทันที
          ออกแบบให้รันต่อเนื่องหลายวัน ปลอดภัยจากไฟดับ/USB หลุด

 คุณสมบัติ:
   - เขียนลง CSV ทุกบรรทัดทันที (flush) -> ไฟดับก็ไม่เสียข้อมูลที่ผ่านมา
   - แยกไฟล์ตามวัน  water_log_YYYY-MM-DD.csv  (rotating)
   - Python ใส่ timestamp เวลาคอมเอง (แม่นกว่า RTC บนบอร์ด)
   - Auto-reconnect: ถ้า USB หลุด จะพยายามต่อใหม่เองทุก 5 วินาที ไม่ตาย
   - รองรับ Windows (COMx) และ Linux/Ubuntu (/dev/ttyUSBx)

 การใช้งาน:
   Windows :  python logger.py --port COM7
   Linux   :  python3 logger.py --port /dev/ttyUSB0
   auto    :  python3 logger.py            (หา port อัตโนมัติ)

 หยุด: กด Ctrl+C  (ข้อมูลถูกบันทึกครบแล้วทุกบรรทัด)
============================================================================
"""

import serial
import serial.tools.list_ports
import csv
import os
import sys
import time
import signal
import argparse
from datetime import datetime

# import ตัวสร้างรายงาน (ต้องมี report.py อยู่โฟลเดอร์เดียวกัน)
try:
    import report as report_module
    HAVE_REPORT = True
except Exception as e:
    HAVE_REPORT = False
    _report_err = str(e)

# ให้ systemctl stop (SIGTERM) ปิดโปรแกรมแบบเดียวกับ Ctrl+C (ปิด CSV สะอาด)
def _sigterm_handler(signum, frame):
    raise KeyboardInterrupt()
signal.signal(signal.SIGTERM, _sigterm_handler)

BAUD = 115200
DATA_DIR = "water_data"          # โฟลเดอร์เก็บ CSV
RECONNECT_DELAY = 5              # วินาที รอก่อน reconnect
CSV_HEADER = ["timestamp", "EC_uScm", "Tw_C", "Salinity_ppm",
              "TDS_ppm", "pH", "pH_mV", "rs485_ok"]


def find_port():
    """หา serial port อัตโนมัติ (มองหา CP210x / CH340 / USB serial ทั่วไป)"""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        # ชิป USB-serial ที่พบบ่อยบนบอร์ด ESP32
        for key in ("cp210", "ch340", "ch910", "usb", "uart", "ttyusb", "ttyacm"):
            if key in desc or key in hwid or key in p.device.lower():
                return p.device
    # ถ้าไม่เจอ คืน port แรกที่มี (ถ้ามี)
    return ports[0].device if ports else None


def daily_csv_path():
    """คืน path ของไฟล์ CSV ประจำวันนี้ (สร้างโฟลเดอร์ถ้ายังไม่มี)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = f"water_log_{datetime.now():%Y-%m-%d}.csv"
    return os.path.join(DATA_DIR, fname)


def open_csv(path):
    """เปิดไฟล์ CSV แบบ append; เขียน header ถ้าเป็นไฟล์ใหม่"""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if is_new:
        w.writerow(CSV_HEADER)
        f.flush()
    return f, w


def parse_data_line(line):
    """
    แปลงบรรทัด 'DATA,EC,Tw,Sal,TDS,pH,mV,rs485' เป็น list ของค่า
    คืน None ถ้าไม่ใช่บรรทัด DATA หรือรูปแบบผิด
    """
    line = line.strip()
    if not line.startswith("DATA,"):
        return None
    parts = line.split(",")
    if len(parts) != 8:          # DATA + 7 ค่า
        return None
    vals = parts[1:]             # ตัด "DATA" ออก
    out = []
    for v in vals:
        if v == "NaN":
            out.append("")       # ช่องว่างใน CSV = ไม่มีข้อมูล
        else:
            out.append(v)
    return out


def run(port, auto_report=True, auto_open=True, meta=None):
    if meta is None:
        meta = {}
    print(f"[logger] เริ่มทำงาน | port={port} | baud={BAUD}")
    print(f"[logger] เก็บข้อมูลที่โฟลเดอร์: {os.path.abspath(DATA_DIR)}/")
    print("[logger] กด Ctrl+C เพื่อหยุด (ข้อมูลถูกบันทึกครบทุกบรรทัด)\n")

    session_start = datetime.now()     # จำเวลาเริ่ม session สำหรับ auto-report
    session_files = set()              # ไฟล์ CSV ที่ session นี้เขียน
    current_day = None
    csv_file = None
    csv_writer = None
    ser = None
    row_count = 0

    try:
        while True:
            # --- เปิด serial (พร้อม auto-reconnect) ---
            if ser is None or not ser.is_open:
                try:
                    ser = serial.Serial(port, BAUD, timeout=2)
                    time.sleep(2)             # รอ ESP32 reset หลังเปิด port
                    ser.reset_input_buffer()
                    print(f"[logger] เชื่อมต่อ {port} สำเร็จ")
                except (serial.SerialException, OSError) as e:
                    print(f"[logger] ต่อ port ไม่ได้: {e} | ลองใหม่ใน {RECONNECT_DELAY}s")
                    time.sleep(RECONNECT_DELAY)
                    continue

            # --- อ่านบรรทัด ---
            try:
                raw = ser.readline().decode("utf-8", errors="ignore")
            except (serial.SerialException, OSError) as e:
                print(f"[logger] USB หลุด: {e} | จะ reconnect")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(RECONNECT_DELAY)
                continue

            if not raw:
                continue

            values = parse_data_line(raw)
            if values is None:
                continue          # ข้ามบรรทัด log มนุษย์/debug

            # --- สลับไฟล์ถ้าข้ามวัน ---
            today = datetime.now().strftime("%Y-%m-%d")
            if today != current_day:
                if csv_file:
                    csv_file.close()
                path = daily_csv_path()
                csv_file, csv_writer = open_csv(path)
                current_day = today
                session_files.add(path)
                print(f"[logger] เขียนลงไฟล์: {path}")

            # --- เขียนข้อมูล + timestamp เวลาคอม ---
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([ts] + values)
            csv_file.flush()               # << สำคัญ: เขียนลงดิสก์ทันที
            os.fsync(csv_file.fileno())    # << บังคับ OS flush (กันไฟดับ)

            row_count += 1
            # แสดงความคืบหน้าบนจอทุก 10 แถว
            if row_count % 10 == 0:
                ec = values[0] or "--"
                ph = values[4] or "--"
                print(f"[{ts}] rows={row_count}  EC={ec}  pH={ph}")

    except KeyboardInterrupt:
        print(f"\n[logger] หยุดโดยผู้ใช้ | บันทึกทั้งหมด {row_count} แถว")
    finally:
        if csv_file:
            csv_file.close()
        if ser and ser.is_open:
            ser.close()
        print("[logger] ปิดไฟล์และ port เรียบร้อย")

        # ---------- auto-report เฉพาะข้อมูล session นี้ ----------
        if auto_report and row_count > 0:
            if not HAVE_REPORT:
                print(f"[logger] ข้าม auto-report: โหลด report.py ไม่ได้ ({_report_err})")
                print("         สร้างเองได้ด้วย: python report.py")
                return
            print("\n[logger] กำลังสร้างรายงาน PDF ของ session นี้...")
            try:
                stamp = session_start.strftime("%Y%m%d_%H%M%S")
                base = os.path.join(DATA_DIR, f"session_{stamp}")
                report_module.generate_report(
                    inputs=sorted(session_files),
                    output=base,
                    since=session_start,          # กรองเฉพาะช่วง session
                    auto_open=auto_open,          # เปิด PDF ขึ้นมาเลย
                    want_excel=True,
                    meta=meta,
                )
            except Exception as e:
                print(f"[logger] สร้างรายงานไม่สำเร็จ: {e}")
                print("         ข้อมูล CSV ปลอดภัย — สร้างเองได้ด้วย: python report.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ESP32 Water Monitor logger")
    ap.add_argument("--port", help="serial port (เช่น COM7 หรือ /dev/ttyUSB0)")
    ap.add_argument("--no-report", action="store_true",
                    help="ไม่ต้องสร้าง PDF อัตโนมัติตอนกดปิด")
    ap.add_argument("--no-open", action="store_true",
                    help="สร้าง PDF แต่ไม่ต้องเปิดขึ้นมาดูอัตโนมัติ")
    ap.add_argument("--sample", default="-", help="ชื่อ/ID ตัวอย่าง (ใส่ในรายงาน)")
    ap.add_argument("--ec-factor", default="1.0367", help="EC correction factor ที่ใช้")
    ap.add_argument("--service", action="store_true",
                    help="โหมด service (headless): ไม่สร้าง/เปิด PDF, ปิดเงียบเมื่อ stop")
    args = ap.parse_args()

    # โหมด service = headless -> ปิด auto-report/auto-open ทั้งหมด
    if args.service:
        args.no_report = True
        args.no_open = True

    port = args.port or find_port()
    if not port:
        print("!! หา serial port ไม่เจอ — ระบุด้วย --port")
        print("   ตัวอย่าง: python logger.py --port COM7")
        print("            python3 logger.py --port /dev/ttyUSB0")
        sys.exit(1)

    meta = {"sample": args.sample, "ec_factor": args.ec_factor}
    run(port, auto_report=not args.no_report, auto_open=not args.no_open, meta=meta)
