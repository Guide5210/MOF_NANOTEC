"""
azd_kd_controller.py  (v2 — register addresses verified)
====================================================================
ควบคุม Oriental Motor AZD-KD (AZ Series, built-in controller) + AZM46AK-TS30
ผ่าน Modbus RTU (RS-485) ด้วย pymodbus 3.x

เลข register ในไฟล์นี้ตรวจสอบแล้วจาก:
  - คู่มือ AZ Function Edition (HM-60262E): operation-type codes, ทิศ = เครื่องหมาย speed
  - ไลบรารีที่ใช้งานจริง hideakitai/ofxModbusOriental: address ของ command/direct-data/monitor

หลักการ AZ ที่ฝังไว้:
  * ทุกค่าเป็น 32-bit = 2 register ติดกัน, high word อยู่ address ต่ำ (big-endian word order)
  * เขียนบล็อก Direct Data Operation ทั้ง 16 register ในทรานแซกชันเดียว (FC16)
    การเขียนทีละตัวเสี่ยง error "outside settings range"
  * ทิศหมุน CW/CCW มาจาก "เครื่องหมายของ velocity" (บวก=หน้า, ลบ=กลับ) ไม่ใช่ op type แยก

>>> ก่อนรันจริง <<<
  - ตั้งสวิตช์: SW1-No.2 = Modbus, ID = slave, BAUD ตรงกับ master, TERM=ON ที่ไดรเวอร์ปลายสุด
  - ต่อ RS-485 ที่ CN6/CN7 ผ่าน USB<->RS-485 (mini-USB ของไดรเวอร์ใช้กับ MEXE02 เท่านั้น)
  - ตั้ง home position ก่อน (จากโรงงานยังไม่ตั้งมา) มิฉะนั้น absolute positioning จะ error
  - เลข monitor เพิ่มเติม (speed/torque/temp): เปิดตาราง "Monitor commands" p.348 แล้วเติมใน MON_*
"""

from __future__ import annotations
import struct
import time
from pymodbus.client import ModbusSerialClient


# ====================================================================
# REGISTER MAP (address = high word ; low word = +1)
# ====================================================================
# --- Command register (I/O command) : FC16, 2 registers ---
REG_COMMAND   = 0x007C   # คำสั่งอยู่ที่ word ล่าง (0x007D)
REG_NET_SELECT = 0x007A  # เลือก stored-data number ผ่านเน็ต

# Command bits (เขียนลง word ล่างของ 0x007C)
CMD_CLEAR   = 0x0000     # เคลียร์บิตทั้งหมด (รีเซ็ต trigger หลังสั่งงาน)
CMD_START   = 0x0008
CMD_HOME    = 0x0010
CMD_STOP    = 0x0020
CMD_FREE    = 0x0040
CMD_RESET   = 0x0080     # reset alarm
CMD_JOG_FWD = 0x1000     # continuous jog CW  (ความเร็วจาก REG_JOG_SPEED)
CMD_JOG_BWD = 0x2000     # continuous jog CCW

# --- Direct Data Operation block : FC16, เขียน 16 register รวดที่ 0x0058 ---
REG_DIRECT_BASE = 0x0058
DD_DRIVE_NO  = 0xFF      # 0xFF = direct (ไม่อ้าง stored data)
DD_MODE_CONTINUOUS_SPEED = 16   # continuous operation (speed control)
DD_MODE_ABS_POSITION     = 1    # absolute positioning
DD_MODE_INC_POSITION     = 2    # incremental positioning (command pos)

# --- JOG parameters (FC16, 2 registers) ---
REG_JOG_STEPS = 0x02A0   # ระยะต่อ inching
REG_JOG_SPEED = 0x02A2   # ความเร็ว jog (Hz) สำหรับ CMD_JOG_FWD/BWD
REG_PRESET    = 0x038C   # ตั้งตำแหน่ง origin/preset

# --- Monitor (FC03) ---
REG_MON_STATUS   = 0x007E   # *verified* สถานะ output (READY/MOVE/ALM/HOME-END ...)
REG_MON_POSITION = 0x0120   # *verified* ตำแหน่งปัจจุบัน (step, signed)
# เติมจากตาราง Monitor commands (p.348) — ค่าเริ่มต้น None = ยังไม่ยืนยัน
REG_MON_CMD_POS   = 0x00C6  # *verified จากสแกนจริง* ตำแหน่งคำสั่ง (step)
REG_MON_FB_POS    = 0x00CC  # *verified จากสแกนจริง* ตำแหน่งจริงจาก encoder (step)
REG_MON_SPEED_RPM = 0x00CE  # ความเร็วจริง (r/min, signed) — verify ตอนหมุน
REG_MON_SPEED_HZ  = 0x00D0  # ความเร็วจริง (Hz, signed) — verify ตอนหมุน
REG_MON_TEMP_DRV  = 0x00F8  # *verified จากสแกนจริง* อุณหภูมิไดรเวอร์ (0.1 degC)
REG_MON_TEMP_MOT  = 0x00FA  # *verified จากสแกนจริง* อุณหภูมิมอเตอร์ (0.1 degC)
REG_MON_ODOMETER  = 0x00FC  # รอบสะสมตลอดอายุ (kRev)
REG_MON_TRIPMETER = 0x00FE  # รอบสะสมตั้งแต่เปิดไฟ (kRev)
REG_MON_TORQUE    = None    # แรงบิด/โหลด (0.1 %) — ยังไม่ยืนยัน
REG_MON_ALARM     = None    # โค้ดอะลาร์มปัจจุบัน — ยังไม่ยืนยัน

GEAR_RATIO = 30.0           # TS30


class AZDKD:
    def __init__(self, port="/dev/ttyUSB0", slave_id=1, baudrate=115200,
                 parity="E", stopbits=1, bytesize=8, timeout=1.0):
        self.slave_id = slave_id
        self.client = ModbusSerialClient(
            port=port, baudrate=baudrate, parity=parity,
            stopbits=stopbits, bytesize=bytesize, timeout=timeout,
        )

    # ---------- connection ----------
    def connect(self):
        if not self.client.connect():
            raise ConnectionError(f"เปิดพอร์ตไม่ได้: {self.client}")
        return True

    def close(self):
        self.client.close()

    def __enter__(self):
        self.connect(); return self

    def __exit__(self, *exc):
        self.close()

    # ---------- pymodbus version shim (slave / device_id) ----------
    def _kw(self):
        try:
            from pymodbus import __version__ as v
            mj, mn = (int(x) for x in v.split(".")[:2])
            return {"device_id": self.slave_id} if (mj, mn) >= (3, 7) else {"slave": self.slave_id}
        except Exception:
            return {"slave": self.slave_id}

    # ---------- 32-bit helpers (hi word ก่อน) ----------
    @staticmethod
    def _u32_words(value):
        return list(struct.unpack(">HH", struct.pack(">i", value)))  # รองรับติดลบ

    def _write_regs(self, addr, words):
        wr = self.client.write_registers(address=addr, values=words, **self._kw())
        if wr.isError():
            raise IOError(f"เขียน 0x{addr:04X} ผิดพลาด: {wr}")

    def _read32_signed(self, addr):
        rr = self.client.read_holding_registers(address=addr, count=2, **self._kw())
        if rr.isError():
            raise IOError(f"อ่าน 0x{addr:04X} ผิดพลาด: {rr}")
        return struct.unpack(">i", struct.pack(">HH", *rr.registers))[0]

    def _read32_unsigned(self, addr):
        return self._read32_signed(addr) & 0xFFFFFFFF

    # ---------- command register ----------
    def _command(self, cmd_bits):
        self._write_regs(REG_COMMAND, [0x0000, cmd_bits & 0xFFFF])

    def stop(self):
        self._command(CMD_STOP)
        self._command(CMD_CLEAR)

    def reset_alarm(self):
        self._command(CMD_RESET); time.sleep(0.05); self._command(CMD_CLEAR)

    def go_home(self):
        self._command(CMD_HOME); self._command(CMD_CLEAR)

    # ================================================================
    # วิธีที่ 1 (แนะนำ): Direct Data Operation — ปรับ speed/accel ได้อิสระ
    # ================================================================
    def run_continuous(self, direction, speed_hz, accel=1_000_000,
                       decel=1_000_000, current_pct=100.0):
        """หมุนต่อเนื่องด้วย continuous speed control จนกว่าจะ stop()
           direction: 'cw' หรือ 'ccw'  (CW = velocity บวก)
           speed_hz : Hz ที่เพลามอเตอร์ (ก่อนเกียร์ 30:1)
           accel/decel: 1 = 0.001 kHz/s
        """
        vel = abs(int(speed_hz)) if direction.lower() == "cw" else -abs(int(speed_hz))
        crnt = int(current_pct * 10)        # 0.1%
        # บล็อก 16 register: [DriveNo, Mode, Position, Velocity, Accel, Decel, Current, Trigger]
        words = []
        words += [0x0000, DD_DRIVE_NO]                 # 0x0058
        words += [0x0000, DD_MODE_CONTINUOUS_SPEED]    # 0x005A
        words += self._u32_words(0)                    # 0x005C position (ไม่ใช้)
        words += self._u32_words(vel)                  # 0x005E velocity (signed → ทิศ)
        words += self._u32_words(int(accel))           # 0x0060
        words += self._u32_words(int(decel))           # 0x0062
        words += self._u32_words(crnt)                 # 0x0064
        words += [0x0000, 0x0001]                      # 0x0066 trigger = 1
        self._write_regs(REG_DIRECT_BASE, words)       # เขียนรวดเดียว (atomic)

    def set_speed(self, direction, speed_hz, accel=1_000_000, decel=1_000_000, current_pct=100.0):
        """เปลี่ยนความเร็ว/ทิศ ระหว่างหมุน (ส่งบล็อกใหม่ทับ)"""
        self.run_continuous(direction, speed_hz, accel, decel, current_pct)

    def move_absolute(self, position_step, speed_hz, accel=1_000_000,
                      decel=1_000_000, current_pct=100.0):
        """ตัวอย่าง positioning: เลื่อนไปตำแหน่งสัมบูรณ์ (step)"""
        crnt = int(current_pct * 10)
        words = []
        words += [0x0000, DD_DRIVE_NO]
        words += [0x0000, DD_MODE_ABS_POSITION]
        words += self._u32_words(int(position_step))
        words += self._u32_words(abs(int(speed_hz)))
        words += self._u32_words(int(accel))
        words += self._u32_words(int(decel))
        words += self._u32_words(crnt)
        words += [0x0000, 0x0001]
        self._write_regs(REG_DIRECT_BASE, words)

    # ================================================================
    # วิธีที่ 2 (ง่ายสุด): continuous jog ผ่าน command register
    # ================================================================
    def jog(self, direction, speed_hz=None):
        """หมุนค้างด้วยบิต jog ของ command register (ความเร็วจากพารามิเตอร์ jog)"""
        if speed_hz is not None:
            self._write_regs(REG_JOG_SPEED, self._u32_words(abs(int(speed_hz))))
        self._command(CMD_JOG_FWD if direction.lower() == "cw" else CMD_JOG_BWD)

    # ================================================================
    # อ่านค่า monitor
    # ================================================================
    def read_register32(self, addr, signed=True):
        """อ่าน register 32-bit ใดก็ได้ — ใช้ทดลองเลขจากตาราง Monitor commands"""
        return self._read32_signed(addr) if signed else self._read32_unsigned(addr)

    def read_monitor(self):
        out = {
            "status_word": self._read32_unsigned(REG_MON_STATUS),
            "command_position_step": self._read32_signed(REG_MON_CMD_POS),
            "position_step": self._read32_signed(REG_MON_FB_POS),
        }
        # เติมอัตโนมัติเฉพาะ address ที่ยืนยันแล้ว (ไม่เป็น None)
        optional = {
            "speed_rpm_motor": (REG_MON_SPEED_RPM, True, 1),
            "speed_hz_motor":  (REG_MON_SPEED_HZ, True, 1),
            "torque_load_pct": (REG_MON_TORQUE, True, 10),
            "driver_temp_c":   (REG_MON_TEMP_DRV, True, 10),
            "motor_temp_c":    (REG_MON_TEMP_MOT, True, 10),
            "odometer_krev":   (REG_MON_ODOMETER, True, 1),
            "alarm_code":      (REG_MON_ALARM, False, 1),
        }
        for key, (addr, signed, div) in optional.items():
            if addr is not None:
                val = self._read32_signed(addr) if signed else self._read32_unsigned(addr)
                out[key] = val / div if div != 1 else val
        if "speed_rpm_motor" in out:
            out["output_rpm_after_gear"] = round(out["speed_rpm_motor"] / GEAR_RATIO, 3)
        return out


    def read_monitor_fast(self):
        """อ่าน monitor ทั้งชุดแบบ block (3 ทรานแซกชัน) เหมาะกับ GUI polling
           คืน dict เดียวกับ read_monitor() + speed_hz + tripmeter"""
        import struct as _st

        def _blk(addr, count):
            rr = self.client.read_holding_registers(address=addr, count=count, **self._kw())
            if rr.isError():
                raise IOError(f"อ่าน block 0x{addr:04X} ผิดพลาด: {rr}")
            return rr.registers

        def _s32(regs, i):
            return _st.unpack(">i", _st.pack(">HH", regs[i], regs[i+1]))[0]

        st = _blk(REG_MON_STATUS, 2)                 # 0x007E..0x007F
        mid = _blk(REG_MON_CMD_POS, 12)              # 0x00C6..0x00D1
        tail = _blk(REG_MON_TEMP_DRV, 8)             # 0x00F8..0x00FF

        speed_rpm = _s32(mid, 8)
        return {
            "status_word": (st[0] << 16 | st[1]) & 0xFFFFFFFF,
            "command_position_step": _s32(mid, 0),
            "position_step": _s32(mid, 6),
            "speed_rpm_motor": speed_rpm,
            "speed_hz_motor": _s32(mid, 10),
            "output_rpm_after_gear": round(speed_rpm / GEAR_RATIO, 3),
            "driver_temp_c": _s32(tail, 0) / 10.0,
            "motor_temp_c": _s32(tail, 2) / 10.0,
            "odometer_krev": _s32(tail, 4),
            "tripmeter_krev": _s32(tail, 6),
        }


# ====================================================================
# ตัวอย่างใช้งาน
# ====================================================================
if __name__ == "__main__":
    with AZDKD(port="/dev/ttyUSB0", slave_id=1) as az:
        az.reset_alarm()

        print("หมุน CW 5000 Hz (continuous speed control)")
        az.run_continuous("cw", 5000)
        time.sleep(3)

        print("เปลี่ยนเป็น 10000 Hz")
        az.set_speed("cw", 10000)
        time.sleep(2)
        print("monitor:", az.read_monitor())

        print("กลับทิศ CCW 5000 Hz")
        az.set_speed("ccw", 5000)
        time.sleep(3)

        az.stop()
        print("หยุดแล้ว")
