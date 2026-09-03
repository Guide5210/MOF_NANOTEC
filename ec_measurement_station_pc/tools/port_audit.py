#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
port_audit.py — ดูว่าเครื่องนี้เห็นพอร์ตอะไรบ้าง และตัวไหนทำหน้าที่อะไร

    python tools\\port_audit.py

⚠️ ไม่เปิดพอร์ตใด ๆ ทั้งสิ้น — แค่ถาม OS ว่ามีอะไรอยู่
   ปลอดภัยที่จะรันขณะ logger เดิมกำลังทำงาน
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecstation.core import ports as P          # noqa: E402

ROLE_NAME = {P.ROLE_CONTROL: "บอร์ด CONTROL (logger เดิมใช้)",
             P.ROLE_P4_BRIDGE: "ช่อง NDJSON ของจอ P4"}


def main():
    au = P.audit()
    ports = au["ports"]
    print("=" * 78)
    print(" พอร์ตอนุกรมที่เครื่องนี้มองเห็น  (%d ตัว)" % len(ports))
    print("=" * 78)
    if not ports:
        print("  ไม่พบพอร์ตเลย — ตรวจสายและไดรเวอร์")
    for p in ports:
        print("  %-8s %-11s %-28s %s" % (p["device"], p["vid_pid"],
                                         p["role"], p["description"][:28]))
        sn = p["serial_number"]
        print("           serial=%-22s location=%s"
              % (sn or "— (อุปกรณ์ไม่ได้บอก)", p["location"] or "—"))
    print("-" * 78)
    for role, info in au["roles"].items():
        head = "  %-30s" % ROLE_NAME.get(role, role)
        if info["reason"] == P.PICK_OK:
            print("%s -> %s" % (head, info["device"]))
        elif info["reason"] == P.PICK_AMBIGUOUS:
            print("%s -> ไม่เลือก (กำกวม %d ตัว: %s)"
                  % (head, len(info["candidates"]),
                     ", ".join(c["device"] for c in info["candidates"])))
            if role == P.ROLE_P4_BRIDGE:
                print("     ใช้  python -m ecstation.app --bridge-port COMn  ระบุเอง")
        else:
            print("%s -> ไม่พบ" % head)
    print("=" * 78)
    print("หมายเหตุ: VID:PID 303A:1001 เป็นของ ESP32 ทุกตัวที่ใช้ USB-Serial-JTAG")
    print("          ไม่ได้เจาะจงว่าเป็นจอ P4  ถ้ามีบอร์ด ESP32 อื่นเสียบอยู่ด้วย")
    print("          ต้องระบุพอร์ตเอง — เครื่องมือนี้จะไม่เดาให้")
    if "--json" in sys.argv:
        print(json.dumps(au, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
