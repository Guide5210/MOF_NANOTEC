"""
azd_kd_verify_speed.py — verify register ความเร็วครั้งเดียวจบ
สั่งหมุนที่ความเร็วที่รู้ค่า แล้วอ่าน 0x00CE/0x00D0 มาเทียบกับค่าคาดหวัง
คาดหวัง (resolution โรงงาน 1000 P/R): 3000 Hz -> 0x00D0 ~ 3000, 0x00CE ~ 180 r/min
"""
import time
from azd_kd_controller import AZDKD

PORT = "COM8"
SPEED_HZ = 3000   # ที่เพลามอเตอร์ (หลังเกียร์ 30:1 = เอาต์พุตหมุนช้าลง 30 เท่า)

with AZDKD(port=PORT, slave_id=1) as az:
    az.reset_alarm()
    print(f"สั่งหมุน CW {SPEED_HZ} Hz ...")
    az.run_continuous("cw", SPEED_HZ)
    time.sleep(2)  # รอเข้าความเร็วคงที่

    for i in range(5):
        rpm = az.read_register32(0x00CE)
        hz  = az.read_register32(0x00D0)
        pos = az.read_register32(0x00CC)
        print(f"  0x00CE={rpm:6d} r/min | 0x00D0={hz:6d} Hz | pos={pos}")
        time.sleep(0.5)

    az.stop()
    print("\nถ้า 0x00D0 ~ 3000 และ 0x00CE ~ 180 = ยืนยันครบ ปิดจ๊อบ monitor ได้เลย")
    print("(ค่า r/min ขึ้นกับ resolution; ถ้าไม่ใช่ 1000 P/R ตัวเลขจะสเกลต่างแต่ทิศต้องบวก)")
