#!/usr/bin/env python3
"""
============================================================================
 console_utf8.py — ทำให้พิมพ์ภาษาไทยลงคอนโซลได้ทุก OS
============================================================================
 บน Linux คอนโซลเป็น UTF-8 อยู่แล้ว แต่บน Windows ค่าปริยายเป็น cp1252
 ซึ่งไม่มีตัวอักษรไทย  พอ print() ข้อความไทยจะโยน UnicodeEncodeError
 แล้วโปรแกรมตายทันที (ไม่ใช่แค่แสดงผลเพี้ยน)

 แก้สองชั้น:
   1. บอก Windows ให้ใช้ code page 65001 (UTF-8) กับ output ของคอนโซล
   2. ตั้ง stdout/stderr ของ Python เป็น UTF-8 พร้อม errors="replace"
      เพื่อว่าถ้าคอนโซลรุ่นเก่ายังแสดงไม่ได้ ก็แค่เห็นเป็นกล่อง ไม่ crash

 เรียก enable() ครั้งเดียวตอนโปรแกรมเริ่ม
============================================================================
"""

import sys


def enable():
    if sys.platform == "win32":
        try:
            import ctypes
            # 65001 = UTF-8 — ต้องตั้งก่อน reconfigure ไม่งั้นคอนโซลยังตีความเป็น cp1252
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass        # stdout ถูก redirect ไปที่อื่น หรือ Python เก่าเกินไป
