"""
azd_kd_bringup.py  —  สคริปต์ commissioning / ตรวจการสื่อสาร Modbus
====================================================================
ใช้ครั้งแรกหลังเดินสาย RS-485 เสร็จ เพื่อยืนยันว่า Python คุยกับ AZD-KD ได้
ก่อนจะไปสั่งหมุนจริง แยกปัญหา "สื่อสาร" ออกจาก "สั่งงาน" ให้ชัด

ต้องมีไฟล์ azd_kd_controller.py (v2) อยู่โฟลเดอร์เดียวกัน

รัน:  python azd_kd_bringup.py
"""

import sys
import time
from serial.tools import list_ports
from azd_kd_controller import AZDKD, REG_MON_STATUS, REG_MON_POSITION


def step1_list_ports():
    print("=" * 60)
    print("ด่าน 3: เลือก COM port ของ USB-RS485 converter")
    print("=" * 60)
    ports = list(list_ports.comports())
    if not ports:
        print("  ไม่พบ COM port เลย — เสียบ converter เข้า USB หรือยัง?")
        sys.exit(1)

    for i, p in enumerate(ports):
        # ไดรเวอร์ mini-USB จะโชว์ชื่อ Oriental Motor — เตือนว่าอย่าเลือกตัวนี้
        warn = ""
        desc = f"{p.description} {p.manufacturer or ''}"
        if "oriental" in desc.lower() or "common virtual" in desc.lower():
            warn = "  <-- นี่คือ mini-USB ของไดรเวอร์ (MEXE02) อย่าเลือก!"
        print(f"  [{i}] {p.device:8s} : {p.description}{warn}")

    idx = input("\nเลือกหมายเลขพอร์ตของ converter: ").strip()
    try:
        return ports[int(idx)].device
    except (ValueError, IndexError):
        print("เลือกไม่ถูกต้อง")
        sys.exit(1)


def step2_comm_check(port, slave_id=1):
    print("\n" + "=" * 60)
    print("ด่าน 4: ตรวจการสื่อสาร Modbus")
    print("=" * 60)
    try:
        with AZDKD(port=port, slave_id=slave_id) as az:
            # อ่าน register ที่ verify แล้ว 2 ตัว = พิสูจน์ว่าสื่อสารครบวงจร
            status = az.read_register32(REG_MON_STATUS, signed=False)
            pos = az.read_register32(REG_MON_POSITION, signed=True)
            print("  ✅ สื่อสารสำเร็จ! (ไฟ C-DAT ควรเป็นเขียว)")
            print(f"     driver status word : 0x{status:08X}")
            print(f"     actual position    : {pos} step")
            print("\n  >> เทียบ 'actual position' กับหน้า Teaching ของ MEXE02")
            print("     ถ้าตรงกัน = การสื่อสารและ mapping ถูกต้องแน่นอน")
            return True
    except Exception as e:
        print(f"  ❌ สื่อสารไม่สำเร็จ: {e}\n")
        print("  ไล่เช็กตามลำดับ:")
        print("   1) เลือก COM port ถูกตัวไหม (ต้องเป็น converter ไม่ใช่ COM9)")
        print("   2) TR+/TR- สลับกันไหม (A+→pin3, B-→pin6)")
        print("   3) BAUD switch = 4 (115200) ตรงกับโค้ดไหม")
        print("   4) SW1-No.2 = ON (โหมด Modbus) และ cycle power แล้วหรือยัง")
        print("   5) ID = 1 ตรงกับ slave_id ไหม")
        print("   6) ปิด MEXE02 แล้วหรือยัง (กัน master ชนกัน)")
        return False


def step3_probe_registers(port, slave_id=1):
    """สแกน register เพื่อจับคู่ address ของ speed/torque/temp กับค่าใน MEXE02
       หมุนมอเตอร์ค้างไว้ที่ความเร็วคงที่ผ่าน MEXE02 หรือ run_continuous ก่อน
       แล้วดูว่า address ไหนคืนค่าที่ตรงกับ Actual Speed ที่ MEXE02 โชว์"""
    print("\n" + "=" * 60)
    print("(ทางเลือก) สแกนหา register monitor ที่เหลือ")
    print("=" * 60)
    print("  ให้มอเตอร์หมุนค้างที่ความเร็วคงที่ก่อน แล้วกด Enter")
    input("  พร้อมแล้วกด Enter (หรือ Ctrl+C ข้าม)...")
    with AZDKD(port=port, slave_id=slave_id) as az:
        # สแกนย่านที่ monitor commands มักอยู่ (0x00C0–0x0100)
        for addr in range(0x00C0, 0x0100, 2):
            try:
                val = az.read_register32(addr, signed=True)
                if val != 0:
                    print(f"    0x{addr:04X} = {val}")
            except Exception:
                pass
        print("\n  จับคู่ค่าที่ตรงกับ Actual Speed / temperature ใน MEXE02")
        print("  แล้วเอา address ไปใส่ REG_MON_SPEED_RPM ฯลฯ ใน azd_kd_controller.py")


if __name__ == "__main__":
    port = step1_list_ports()
    if step2_comm_check(port):
        print("\n🎉 พร้อมสั่งหมุนแล้ว — ใช้ az.run_continuous('cw', 5000) ได้เลย")
        ans = input("\nอยากสแกนหา register monitor ที่เหลือด้วยไหม? (y/n): ")
        if ans.strip().lower() == "y":
            step3_probe_registers(port)
