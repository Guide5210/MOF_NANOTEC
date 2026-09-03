# -*- coding: utf-8 -*-
"""เฝ้า ESP_LOG ของจอ (CH343) ระหว่างที่ hw_test ถือพอร์ต NDJSON อยู่

ทำไมต้องมีตัวนี้แยกต่างหาก
    hw_test เห็นจอผ่าน USB-Serial-JTAG (303A:1001) ซึ่งส่งแต่ NDJSON
    ถ้าจอพังจริง สิ่งที่ฝั่งนั้นเห็นคือ "heartbeat หายไป" เท่านั้น
    ไม่มีทางรู้ว่าพังเพราะอะไร

    backtrace, panic และ core dump ออกทาง CH343 (1A86:55D3) ซึ่งเป็นสายคนละเส้น
    เปิดพร้อมกันได้ไม่ชนกัน  ตัวนี้จึงเป็นตัวเดียวที่จะบอกได้ว่าเกิดอะไรขึ้น

⚠️ ตั้ง dtr/rts เป็น False ก่อน open() เสมอ
   ESP32-P4 ใช้สองเส้นนี้เป็นสัญญาณ reset/boot  ถ้าไม่กดลงก่อน แค่เปิดพอร์ต
   ก็รีเซ็ตจอแล้ว และการทดสอบ 12 ชั่วโมงจะพังตั้งแต่วินาทีแรกโดยที่เราเป็นคนทำเอง
"""
import argparse
import os
import sys
import time

import serial
import serial.tools.list_ports

CH343 = (0x1A86, 0x55D3)

# บรรทัดที่ต้องเด้งขึ้นมาให้เห็นทันที ไม่ปนกับ log ปกติ
HOT = (
    "Guru Meditation", "panic", "Backtrace", "abort()", "assert failed",
    "CORRUPT HEAP", "Bad tail", "Bad head",
    "Stack canary", "stack overflow", "watchdog", "WDT",
    "rst:0x",
)


def find_port():
    hits = [p.device for p in serial.tools.list_ports.comports()
            if (p.vid, p.pid) == CH343]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    # เจอหลายตัวห้ามเดา — เปิดผิดตัวแปลว่าไปยึดพอร์ตของบอร์ดอื่นไว้ด้วย
    print("เจอ CH343 หลายตัว: %s — ระบุด้วย --port" % ", ".join(hits))
    raise SystemExit(2)


def out(text):
    # log ของจอเป็นภาษาไทย ต้องเขียนเป็น utf-8 ตรง ๆ
    # ไม่งั้น console ที่เป็น cp1252/cp874 จะโยน UnicodeEncodeError แล้วตัวเฝ้าตาย
    sys.stdout.buffer.write((text + "\n").encode("utf-8", "replace"))
    sys.stdout.flush()


def wall():
    """เวลานาฬิกาแบบสั้น — ใช้เทียบกับ CSV และรายงานของ hw_test"""
    return time.strftime("%H:%M:%S")


def open_port(port):
    """เปิดพอร์ตโดยไม่รีเซ็ตจอ — แยกเป็นฟังก์ชันเพราะต้องเรียกซ้ำตอนต่อใหม่"""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.timeout = 1.0
    ser.dtr = False            # ⚠️ ต้องอยู่ก่อน open() ดูหัวไฟล์
    ser.rts = False
    ser.open()
    return ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="ระบุเองถ้าหาอัตโนมัติไม่ได้ เช่น COM3")
    ap.add_argument("--hours", type=float, default=25.0,
                    help="เฝ้านานเท่าไหร่ (ค่าเริ่มต้นเผื่อไว้เกิน soak 24 ชม.)")
    ap.add_argument("--log", default=None, help="ไฟล์เก็บ log ทั้งหมด")
    a = ap.parse_args()

    port = a.port or find_port()
    if not port:
        out("ไม่พบพอร์ต CH343 (1A86:55D3) — เสียบสายเส้นที่สองของจอหรือยัง")
        return 2

    path = a.log or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "diag",
        time.strftime("p4log_%Y-%m-%d_%H%M%S.log"))
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        ser = open_port(port)
    except serial.SerialException as exc:
        # เคสที่เจอบ่อยที่สุดคือมีตัวเฝ้าเก่าค้างอยู่ หรือเปิด idf.py monitor ทิ้งไว้
        # traceback ดิบ ๆ ไม่ได้บอกเรื่องนี้เลย ทำให้เสียเวลาไปหาที่ผิดที่อื่น
        out("เปิด %s ไม่ได้: %s" % (port, exc))
        out("พอร์ตนี้เปิดได้ทีละโปรเซส — ปิด idf.py monitor / ตัวเฝ้าตัวเก่า")
        out("ดูว่ามีอะไรค้างอยู่:  powershell Get-Process python")
        return 2

    out("เฝ้า %s นาน %.1f ชม.  เก็บลง %s" % (port, a.hours, path))
    out("ถ้าบรรทัดถัดไปเป็น boot log แปลว่าการเปิดพอร์ตรีเซ็ตจอ — หยุดแล้วแจ้ง")

    end = time.time() + a.hours * 3600
    n = boots = 0
    last_note = 0.0
    t0 = time.time()

    #  ⚠️ ห้ามตายเมื่อพอร์ตหลุด — ต้องต่อใหม่เอง
    #
    #     รอบทดสอบ 2 ก.ย. ตัวเฝ้าตายที่นาทีที่ 18 จาก USB ของ host สะดุด
    #     (ClearCommError PermissionError) แล้วเราตาบอดไปตลอดช่วงที่เหลือ
    #     สำหรับรันข้ามคืน 24 ชั่วโมง นั่นแปลว่าถ้าจอพังตอนตีสามจะไม่มีหลักฐานเลย
    #
    #     การที่ USB หลุดเองเป็นข้อมูลสำคัญในตัวมันเอง จึงนับและบันทึกเวลาไว้
    #     เอาไปเทียบกับช่องว่างใน CSV ได้ว่าเป็นเหตุการณ์เดียวกันหรือคนละเรื่อง
    usb_drops = []

    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        while time.time() < end:
            if ser is None:
                try:
                    ser = open_port(port)
                    el = int(time.time() - t0)
                    msg = "ต่อ %s กลับได้แล้ว" % port
                    out("  ** [%5ds] %s" % (el, msg))
                    fh.write("[%6d] %s  %s\n" % (el, wall(), msg))
                    fh.flush()
                except (serial.SerialException, OSError):
                    time.sleep(2.0)       # ยังไม่กลับมา — รอแล้วลองใหม่
                    continue

            try:
                raw = ser.readline()
            except (serial.SerialException, OSError) as exc:
                el = int(time.time() - t0)
                usb_drops.append(wall())
                out("  !! [%5ds] %s พอร์ตหลุด ครั้งที่ %d: %s"
                    % (el, wall(), len(usb_drops), exc))
                fh.write("[%6d] %s  พอร์ตหลุด: %s\n" % (el, wall(), exc))
                fh.flush()
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                continue

            el = int(time.time() - t0)
            if raw:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    n += 1
                    #  เวลานาฬิกาจำเป็น ไม่ใช่ของประดับ — ต้องเอาบรรทัดนี้ไปเทียบ
                    #  กับ timestamp ใน CSV และเวลาใน hwtest report ได้
                    #  วินาทีที่นับจากตอนเริ่มอย่างเดียวเทียบข้ามไฟล์ไม่ได้
                    fh.write("[%6d] %s  %s\n" % (el, wall(), line))
                    fh.flush()
                    if "ESP-ROM:" in line:
                        boots += 1
                        out("  !! [%5ds] %s จอบูตใหม่ ครั้งที่ %d"
                            % (el, wall(), boots))
                    elif any(h in line for h in HOT):
                        out("  >> [%5ds] %s %s" % (el, wall(), line))
                    elif " E " in line:
                        out("     [%5ds] %s %s" % (el, wall(), line))
            if el - last_note >= 1800:
                last_note = el
                out("     [%5ds] %s เงียบดี  %d บรรทัด  บูตใหม่ %d  USB หลุด %d"
                    % (el, wall(), n, boots, len(usb_drops)))

    if ser is not None:
        ser.close()
    out("")
    out("จบ %.1f ชม. | บูตใหม่ %d ครั้ง | USB หลุด %d ครั้ง | %d บรรทัด"
        % ((time.time() - t0) / 3600.0, boots, len(usb_drops), n))
    if usb_drops:
        out("USB หลุดตอน: %s" % ", ".join(usb_drops))
        out("(USB หลุดเป็นเรื่องของ host ไม่ใช่จอพัง — เอาเวลาไปเทียบกับ CSV)")
    out("log: %s" % path)
    # บูตใหม่แม้ครั้งเดียวก็ถือว่าไม่ผ่าน — แต่ต้องไปดูสาเหตุใน log ก่อนสรุปว่าพัง
    # rst:0x17 (CHIP_USB_UART_RESET) คือ host รีเซ็ต ไม่ใช่จอพัง
    return 1 if boots else 0


if __name__ == "__main__":
    raise SystemExit(main())
