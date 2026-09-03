#!/usr/bin/env python3
"""
swu10u_probe.py
================
สคริปต์สำหรับตรวจสอบ/ดึงข้อมูลจาก ULVAC SWU10-U ผ่านพอร์ต USB (ชิป FTDI)

หลักการทางวิศวกรรม: เนื่องจาก ULVAC ไม่เปิดเผย protocol ของ SWU10-U ต่อสาธารณะ
เราจึงต้องพิสูจน์เชิงประจักษ์ (empirical) เป็น 2 ขั้นก่อนที่จะไปถึงขั้น reverse-engineer
เต็มรูปแบบด้วย packet sniffer ซึ่งใช้เวลามากกว่ามาก:

  ขั้นที่ 1 (PASSIVE)  : เปิดพอร์ตแล้วฟังเฉยๆ 10 วินาที
                         -> เครื่องวัดสุญญากาศราคาประหยัดจำนวนมาก "สตรีม" ค่าออกมาเองอัตโนมัติ
                            ทุก 0.1-1 วินาทีโดยไม่ต้องส่งคำสั่งใดๆ เลย ถ้าใช่ ปัญหาจบตรงนี้
  ขั้นที่ 2 (ACTIVE)   : ถ้าขั้น 1 เงียบ ให้ลองส่งคำสั่งสั้นๆ ที่เป็นมาตรฐานทั่วไปของ
                         เครื่องมือวัดผ่าน RS232/USB-serial (ENQ, '?', 'P', 'READ' ฯลฯ)
                         ทีละตัว แล้วดูว่ามีการตอบกลับหรือไม่ วิธีนี้ไม่ทำลายอุปกรณ์
                         (เป็นเพียงการส่ง byte แล้วอ่านกลับ)

ถ้าทั้งสองขั้นไม่ได้ผล แปลว่าโปรโตคอลน่าจะซับซ้อนกว่านั้น (มี checksum/addressing
เหมือนเครื่องรุ่น GP-2001G, GI-M001 ของ ULVAC เอง) ขั้นต่อไปที่แนะนำคือ sniff การสื่อสาร
จริงระหว่างแอป UL-MOBI กับตัวเครื่องด้วย Wireshark + USBPcap (Windows) แล้วนำ raw bytes
ที่ capture ได้มาป้อนให้ผมช่วย decode รูปแบบ (pattern) ต่อ

การใช้งาน:
    pip install pyserial --break-system-packages
    python swu10u_probe.py COM5          (Windows)
    python swu10u_probe.py /dev/ttyUSB0  (Linux/macOS)

ถ้าไม่แน่ใจชื่อพอร์ต ให้รันโดยไม่ใส่ argument สคริปต์จะแสดงพอร์ต serial ที่เจอในเครื่องให้เลือก
"""

import sys
import time
import serial
import serial.tools.list_ports

BAUD = 38400  # ตายตัวตามคู่มือ SWU10-U (เปลี่ยนค่านี้แล้วแอป UL-MOBI จะต่อไม่ติด)
LISTEN_SECONDS = 10
PROBE_COMMANDS = [
    b"\x05",          # ENQ - มาตรฐานเก่าแก่สำหรับขอข้อมูลจากเครื่องมือวัด
    b"?\r\n",
    b"?\r",
    b"P\r\n",
    b"P?\r\n",
    b"READ\r\n",
    b"READ?\r\n",
    b"MEAS?\r\n",
    b"*IDN?\r\n",      # SCPI-style identify, บาง instrument รองรับ
    b"D\r\n",
    b"\r\n",
]


def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("ไม่พบพอร์ต serial ใดๆ เลย ตรวจสอบว่าเสียบสาย USB และติดตั้งไดรเวอร์ FTDI แล้ว")
        sys.exit(1)
    print("พอร์ต serial ที่พบในเครื่อง:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} - {p.description}")
    idx = input("เลือกหมายเลขพอร์ตที่เชื่อมกับ SWU10-U: ").strip()
    return ports[int(idx)].device


def hexdump(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def main():
    port_name = sys.argv[1] if len(sys.argv) > 1 else pick_port()

    print(f"\n--- เปิดพอร์ต {port_name} @ {BAUD} 8N1 ---")
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
        )
    except serial.SerialException as e:
        print(f"เปิดพอร์ตไม่ได้: {e}")
        sys.exit(1)

    time.sleep(0.3)
    ser.reset_input_buffer()

    # ---------- ขั้นที่ 1: ฟังเฉยๆ ----------
    print(f"\n[ขั้น 1] กำลังฟังพอร์ตแบบ passive เป็นเวลา {LISTEN_SECONDS} วินาที (ไม่ส่งอะไรเลย)...")
    end_time = time.time() + LISTEN_SECONDS
    got_passive_data = False
    while time.time() < end_time:
        n = ser.in_waiting
        if n:
            raw = ser.read(n)
            got_passive_data = True
            print(f"  RAW: {hexdump(raw)}")
            try:
                print(f"  ASCII: {raw.decode('ascii', errors='replace')!r}")
            except Exception:
                pass
        else:
            time.sleep(0.05)

    if got_passive_data:
        print("\n>>> เครื่องสตรีมข้อมูลออกมาเองโดยไม่ต้องขอ! ดูรูปแบบ RAW/ASCII ด้านบน "
              "แล้วส่งตัวอย่าง 5-10 บรรทัดมาให้ผมช่วย parse เป็นค่าความดันได้เลย")
        ser.close()
        return

    print("  ไม่มีข้อมูลไหลออกมาเองในขั้นที่ 1")

    # ---------- ขั้นที่ 2: ลองส่งคำสั่งทั่วไป ----------
    print("\n[ขั้น 2] ลองส่งคำสั่งทดสอบทีละตัว...")
    for cmd in PROBE_COMMANDS:
        ser.reset_input_buffer()
        ser.write(cmd)
        time.sleep(0.3)
        n = ser.in_waiting
        resp = ser.read(n) if n else b""
        status = "มีการตอบกลับ!!" if resp else "(เงียบ)"
        print(f"  ส่ง {hexdump(cmd):<20}  -> {status}  {hexdump(resp)}  {resp!r}")

    ser.close()
    print(
        "\n--- สรุป ---\n"
        "ถ้าทุกคำสั่งข้างบน '(เงียบ)' หมดเลย แปลว่า SWU10-U ไม่ตอบสนองต่อคำสั่งข้อความมาตรฐาน\n"
        "ทั่วไป (มีความเป็นไปได้ว่าโปรโตคอลจริงเป็น binary frame เฉพาะของ ULVAC เช่นเดียวกับ\n"
        "รุ่น GP-2001G/GI-M001) ขั้นต่อไปที่แนะนำคือ sniff การสื่อสารจริงระหว่างแอป UL-MOBI\n"
        "กับตัวเครื่องด้วย Wireshark + USBPcap แล้วส่ง capture ไฟล์ (.pcapng) มาให้ช่วย decode\n"
        "หรือคู่ขนานกันคือติดต่อ ULVAC / ตัวแทนจำหน่ายในไทยขอเอกสาร 'Communication Protocol'\n"
        "ของรุ่น SWU10-U โดยตรง (ผู้ผลิตหลายรายมีเอกสารนี้แยกจากคู่มือผู้ใช้ทั่วไป และมักจะให้\n"
        "แก่ผู้ใช้ที่ต้องการทำ system integration)"
    )


if __name__ == "__main__":
    main()