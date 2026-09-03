"""
scan_modbus.py — วินิจฉัยการเชื่อมต่อ AZD-KD ผ่าน RS-485
ลองทุก baud rate และทุก slave address เพื่อหาว่าไดรเวอร์ตอบที่ค่าใด

วิธีใช้:  python scan_modbus.py [COM8]
ถ้าเจอ -> จะพิมพ์ baud + address ที่ใช้ได้ ให้เอาไปตั้งใน azd_kd_controller.py
ถ้าไม่เจอเลย -> ปัญหาอยู่ที่ฮาร์ดแวร์ (สวิตช์ / สาย A-B / ไฟ)
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pymodbus.client import ModbusSerialClient

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM9"

# AZ Modbus รองรับ: 9600,19200,38400,57600,115200,230400  (BAUD switch 0-5)
BAUDS = [115200, 9600, 19200, 38400, 57600, 230400]
# AZ Modbus = even parity, 1 stop bit เป็นค่ามาตรฐาน (ลอง none เผื่อไว้)
PARITIES = ["E", "N"]
ADDRESSES = list(range(1, 11))   # 1..10 (0 = broadcast ข้ามไป)

# register ที่อ่านปลอดภัย: 0x0120 = present position (32-bit, 2 reg)
TEST_ADDR = 0x0120


def kw(addr):
    # pymodbus 3.7+ ใช้ device_id
    return {"device_id": addr}


print(f"สแกนพอร์ต {PORT} ...\n")
found = []

for parity in PARITIES:
    for baud in BAUDS:
        client = ModbusSerialClient(
            port=PORT, baudrate=baud, parity=parity,
            stopbits=1, bytesize=8, timeout=0.2, retries=0,
        )
        if not client.connect():
            print(f"!! เปิดพอร์ต {PORT} ไม่ได้ — พอร์ตถูกใช้งานโดยโปรแกรมอื่น (MEXE02?) หรือชื่อพอร์ตผิด")
            sys.exit(1)

        hit_this_combo = False
        for addr in ADDRESSES:
            try:
                rr = client.read_holding_registers(address=TEST_ADDR, count=2, **kw(addr))
                if not rr.isError():
                    print(f"  ✅ ตอบกลับ! baud={baud} parity={parity} address={addr}  regs={rr.registers}")
                    found.append((baud, parity, addr))
                    hit_this_combo = True
            except Exception:
                pass
        if not hit_this_combo:
            print(f"  - baud={baud:>6} parity={parity}: ไม่มีใครตอบ (address 1-31)")
        client.close()

print()
if found:
    b, p, a = found[0]
    print("=" * 60)
    print(f"เจอไดรเวอร์! ตั้งค่าใน azd_kd_controller.py เป็น:")
    print(f"  AZDKD(port=\"{PORT}\", slave_id={a}, baudrate={b}, parity=\"{p}\")")
    print("=" * 60)
else:
    print("=" * 60)
    print("ไม่พบไดรเวอร์ที่ค่าใดเลย — ตรวจฮาร์ดแวร์:")
    print("  1. SW1-No.2 = ON (เปิดโหมด Modbus RTU)  << สาเหตุพบบ่อยสุด")
    print("  2. ปิด-เปิดไฟไดรเวอร์ใหม่ หลังเปลี่ยน SW1")
    print("  3. ลองสลับสาย A+ <-> B-  (TR+ pin3 <-> TR- pin6)")
    print("  4. SW1-No.3+No.4 = ON ทั้งคู่ (termination 120 ohm)")
    print("  5. ไฟเลี้ยงไดรเวอร์ 24/48VDC เข้าแล้ว, LED POWER ติดเขียว")
    print("=" * 60)
