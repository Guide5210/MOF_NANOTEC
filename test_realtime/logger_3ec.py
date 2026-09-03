#!/usr/bin/env python3
"""
============================================================================
 logger_3ec.py — เก็บ EC 3 ตัว + คุม session แยกแต่ละตัวผ่านคีย์บอร์ด
============================================================================
 - เก็บ CSV รวม 3 ตัว "ตลอดเวลา" (fsync) — ข้อมูลดิบไม่มีวันขาด
 - แต่ละ sensor มี session อิสระ (เริ่ม/หยุดคนละเวลา สำหรับ parallel experiment)
 - หยุด session ตัวไหน -> ออก PDF + Excel ของตัวนั้นทันที ลง folder ของมัน
     sensor_1/  sensor_2/  sensor_3/

 คีย์ควบคุม (ในหน้าต่างนี้ ระหว่าง logger รันอยู่):
     1 / 2 / 3   = เริ่ม/หยุด session ของ EC#1 / #2 / #3
     p           = ดูสถานะ session ทั้ง 3
     q           = ออก (ปิด session ที่ค้าง + ออกไฟล์ให้)

 ใช้:
   python3 logger_3ec.py --port /dev/ttyUSB0 --sample "CALF-20 batch 3"
   python3 logger_3ec.py --service         # headless (ไม่มีคีย์บอร์ด, เก็บ CSV อย่างเดียว)
============================================================================
"""

import argparse
import csv
import json
import os
import sys
import time
import signal
from datetime import datetime

import serial
import serial.tools.list_ports


# ---------------------------------------------------------------- preflight
# โปรเจกต์นี้เป็นไฟล์ .py หลายตัวที่ต้องอยู่ในโฟลเดอร์เดียวกัน  เวลาคัดลอกไป
# อีกเครื่องแล้วตกหล่นบางไฟล์ Python จะโยน traceback ที่ไม่ได้บอกว่าต้องทำอะไร
# ต่อ  ตรวจตั้งแต่ต้นแล้วบอกตรง ๆ ว่าขาดไฟล์ไหนและต้องเอาไปวางที่ไหน
def _preflight():
    here = os.path.dirname(os.path.abspath(__file__))
    required = {
        "calibration.py": "calibration system, the [c] key",
        "report_3ec.py":  "PDF/Excel report on session close",
        "report_jobs.py": "background report worker (keeps the CSV gap-free)",
    }
    optional = {
        "console_utf8.py": "Thai text in the Windows console",
        "desktop_ui.py":   "desktop viewer, not used by the logger",
    }
    missing = [f for f in required if not os.path.exists(os.path.join(here, f))]
    if missing:
        print("!! Required files are missing from this folder:")
        for f in missing:
            print(f"     {f}    <- {required[f]}")
        print(f"\n   Running from: {here}")
        print("   Copy the missing files next to logger_3ec.py and run again")
        sys.exit(1)
    for f in optional:
        if not os.path.exists(os.path.join(here, f)):
            print(f"[logger3] note: {f} not found ({optional[f]}) - continuing")


_preflight()

import calibration
# ⚠️ ต้อง import หลัง _preflight() — ถ้าไฟล์หายไป ผู้ใช้ควรได้ข้อความที่บอกว่า
#    ขาดไฟล์ไหนและต้องเอาไปวางที่ไหน ไม่ใช่ ImportError ดิบ ๆ
import report_jobs

# ----------------------------------------------------------------------------
#  lab_theme เป็นที่เก็บ "คำสถานะ" กับ "รูปแบบตัวเลข" ชุดเดียวกับ desktop_ui.py
#  และจอสัมผัส ESP32-P4  ใช้ที่นี่เพื่อให้สิ่งที่พิมพ์ในคอนโซลกับสิ่งที่ขึ้นบนจอ
#  เป็นคำเดียวกัน ไม่ใช่คนละภาษาบนเครื่องเดียวกัน
#
#  ⚠️ เป็นเรื่องการแสดงผลล้วน ๆ ถ้าไฟล์นี้ไม่อยู่ logger ต้องยังเก็บข้อมูลได้
#     ตามปกติ ไม่ใช่ตายทั้งตัว (กฎเดียวกับ console_utf8 ข้างล่าง)
# ----------------------------------------------------------------------------
try:
    import lab_theme as _T
except Exception:
    _T = None
# console_utf8 ทำหน้าที่แค่ให้คอนโซล Windows แสดงภาษาไทยได้ — เป็นเรื่องการแสดงผล
# ล้วน ๆ ไม่เกี่ยวกับการเก็บข้อมูล  ถ้าไฟล์นี้ไม่อยู่ในโฟลเดอร์ โปรแกรมต้องยัง
# ทำงานได้ตามปกติ ไม่ใช่ตายทั้งตัว  (เคยเกิดจริงตอนคัดลอกโปรเจกต์ไปอีกเครื่อง
# แล้วลืมไฟล์นี้ — logger ล้มทั้งระบบเพราะ helper ที่ไม่สำคัญเลย)
try:
    import console_utf8
    console_utf8.enable()
except ImportError:
    def _enable_utf8_console():
        """สำเนาย่อของ console_utf8.enable() — ใช้เมื่อหาไฟล์นั้นไม่เจอ"""
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                ctypes.windll.kernel32.SetConsoleCP(65001)
            except Exception:
                pass
        for _st in (sys.stdout, sys.stderr):
            try:
                _st.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    _enable_utf8_console()


# ============================================================================
#  อ่านคีย์ทีละตัวแบบไม่บล็อก — รองรับทั้ง Windows และ Linux
# ----------------------------------------------------------------------------
#  เดิมใช้ termios/tty/select ซึ่งมีเฉพาะบน Unix พอรันบน Windows แล้ว import
#  ล้มเหลว ทำให้ interactive = False คีย์ 1/2/3 จึงใช้ไม่ได้ทั้งหมด
#  ตรงนี้แยกเป็นสองรุ่นตาม OS โดยให้หน้าตาการใช้งานเหมือนกัน
# ============================================================================
if os.name == "nt":
    import msvcrt

    class KeyInput:
        """Windows: ใช้ msvcrt ไม่ต้องสลับโหมด terminal"""
        available = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self):
            """คืนคีย์ที่กด หรือ None ถ้าไม่มี — ไม่บล็อก"""
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            # ปุ่มพิเศษ (ลูกศร/ฟังก์ชัน) ส่งมาสองตัว ต้องกินตัวที่สองทิ้ง
            if ch in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                return None
            return ch

        def line(self, prompt):
            """อ่านทั้งบรรทัด (สำหรับถาม Note)"""
            try:
                return input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return ""

else:
    try:
        import termios
        import tty
        import select
        _HAVE_TTY = True
    except ImportError:
        _HAVE_TTY = False

    class KeyInput:
        """Linux/macOS: ตั้ง terminal เป็น cbreak เพื่ออ่านทีละคีย์"""
        available = _HAVE_TTY

        def __init__(self):
            self.fd = None
            self.old = None

        def __enter__(self):
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, *exc):
            if self.old is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            return False

        def get(self):
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

        def line(self, prompt):
            # สลับกลับโหมดปกติชั่วคราว ไม่งั้นพิมพ์ทั้งบรรทัดไม่ได้
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            try:
                return input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return ""
            finally:
                tty.setcbreak(self.fd)


def _sigterm(signum, frame):
    raise KeyboardInterrupt()
signal.signal(signal.SIGTERM, _sigterm)

BAUD = 115200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "water_data")
SESSION_LOG = os.path.join(BASE_DIR, "sessions_3ec.json")
REC_STATUS_FILE = os.path.join(BASE_DIR, "rec_status.json")   # สถานะบันทึกสด (UI อ่าน)
RECONNECT_DELAY = 5

# ----------------------------------------------------------------------------
#  ตัวที่ผู้ใช้ปิดไว้บนหน้าจอ (ec_ui_config.json)
#
#  ⚠️ ใช้ "เฉพาะตอนพิมพ์" เท่านั้น
#     บอร์ดยังอ่านครบทุกตัว และ CSV ยังเก็บครบทุกคอลัมน์เหมือนเดิมทุกประการ
#     ค่าที่ปิดไว้จึงไม่หายไปจากข้อมูลดิบ แค่ไม่รกคอนโซล
# ----------------------------------------------------------------------------
def _display_mask():
    if _T is None:
        return 0b0111
    try:
        return _T.load_ui_config()["active_mask"]
    except Exception:
        return 0b0111


def _shown(i):
    return bool(_display_mask() >> i & 1)
# คอลัมน์ที่ 11 "flag" — ต่อท้าย ไฟล์ CSV เก่าจึงยังอ่านได้เหมือนเดิม
#   ""    = ข้อมูลการทดลองปกติ
#   "CAL" = เก็บตอนคาลิเบรต (หัววัดอยู่ในน้ำยามาตรฐาน ไม่ใช่ตัวอย่าง)
# ข้อมูลดิบไม่ขาดช่วง แต่รายงาน/กราฟจะข้ามแถว CAL ให้เอง ไม่ปนกัน
HEADER = ["timestamp", "EC1", "T1", "EC2", "T2", "EC3", "T3",
          "ok1", "ok2", "ok3", "flag"]


def find_port():
    """
    เดา port ของบอร์ดอ่านเซนเซอร์

    บนเครื่องนี้จอ ESP32-P4 ขึ้นเป็น COM ถึง "สองพอร์ต" พร้อมกัน:

        CH343            1A86:55D3   ใช้แฟลชและดู log ของจอ
        USB-Serial-JTAG  303A:1001   ช่อง NDJSON ที่ pc_bridge ใช้คุยกับ PC

    ทั้งคู่ไม่ใช่บอร์ดเซนเซอร์ ต้องตัดออกทั้งคู่

    ⚠️ ข้อที่เกือบพลาด: USB-Serial-JTAG ขึ้นชื่อว่า "USB Serial Device" หรือ
       "USB JTAG/serial debug unit" ซึ่ง *มีคำว่า usb* จึงได้คะแนน 10 จากกฎ
       ท้ายสุด  ถ้าถอดบอร์ดเซนเซอร์ออกแล้วเปิด logger มันจะไปเปิดพอร์ตของจอ
       แล้วนั่งรอบรรทัด DATA, ที่ไม่มีวันมา พร้อมกับยึดพอร์ตไม่ให้ bridge ใช้
       — อาการคือ "ทั้งสองฝั่งเงียบ" ซึ่งไล่หาสาเหตุยากมาก
    """
    def score(p):
        blob = f"{p.device} {p.description} {p.hwid}".lower()
        # ตัดด้วย VID:PID เป็นหลัก เพราะข้อความบรรยายต่างกันไปตาม OS/ไดรเวอร์
        for bad in ("303a:1001", "1a86:55d3", "ch343", "jtag"):
            if bad in blob:
                return -1                   # เป็นพอร์ตของจอ ไม่ใช่บอร์ดเซนเซอร์
        for i, key in enumerate(("ch340", "cp210", "ch910")):
            if key in blob:
                return 100 - i
        if any(k in blob for k in ("usb", "uart", "ttyusb", "ttyacm")):
            return 10
        return 0

    ports = [p for p in serial.tools.list_ports.comports() if score(p) >= 0]
    if not ports:
        return None
    ports.sort(key=score, reverse=True)

    if len(ports) > 1:
        print("[logger3] multiple ports found:")
        for p in ports:
            print(f"           {p.device}  {p.description}")
        print(f"[logger3] using {ports[0].device} (override with --port)")
    return ports[0].device


def daily_path():
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"water_log_{datetime.now():%Y-%m-%d}.csv")


def open_csv(path):
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if is_new:
        w.writerow(HEADER); f.flush()
    return f, w


EOL = b"\n"


class LineReader:
    """อ่านทีละบรรทัดโดยไม่ให้บรรทัดขาดครึ่งหลุดออกไป

    ⚠️ ser.readline() ที่ timeout=0.5 จะคืน "ครึ่งบรรทัด" ถ้าหมดเวลาตอนข้อมูล
       ยังมาไม่ครบ  parse() จะตีว่าใช้ไม่ได้แล้วทิ้งเงียบ ๆ ส่วนที่เหลือมาถึง
       รอบถัดไปก็ขึ้นต้นไม่ตรงอีก ⇒ เสียสองรอบจากบรรทัดเดียว

       รอบทดสอบ 2 ชั่วโมงวัดได้ว่าแถวหาย 3.44% และช่องว่างสูงสุด 5.0 วินาที
       ซึ่งเท่ากับสองรอบ polling พอดี (รอบละ 2.57 วิ) ตรงกับอาการนี้

       ตัวนี้เก็บเศษไว้ในบัฟเฟอร์แล้วต่อกับข้อมูลรอบหน้า จึงคืนเฉพาะบรรทัดที่
       จบด้วยขึ้นบรรทัดใหม่จริง ๆ ไม่ว่าข้อมูลจะมาเป็นก้อนหรือมาทีละไบต์
    """

    def __init__(self):
        self._buf = b""

    def reset(self):
        self._buf = b""

    def poll(self, ser):
        """คืนบรรทัดที่สมบูรณ์หนึ่งบรรทัด หรือ None ถ้ายังไม่ครบ"""
        if EOL not in self._buf:
            waiting = ser.in_waiting
            chunk = ser.read(waiting if waiting else 1)
            if not chunk:
                return None
            self._buf += chunk
            if EOL not in self._buf:
                return None
        line, _, self._buf = self._buf.partition(EOL)
        return line.decode("utf-8", "ignore")


def parse(line):
    line = line.strip()
    if not line.startswith("DATA,"):
        return None
    p = line.split(",")
    if len(p) != 8:
        return None
    vals = [("" if v == "NaN" else v) for v in p[1:7]]
    okbits = p[7].strip()
    if len(okbits) != 3 or any(c not in "01" for c in okbits):
        return None
    return vals + list(okbits)


def append_session_log(entry):
    try:
        data = []
        if os.path.exists(SESSION_LOG):
            with open(SESSION_LOG, encoding="utf-8") as f:
                data = json.load(f)
        data.append(entry)
        with open(SESSION_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[logger3] warning: cannot write session log: {e}")


# ============================================================================
#  session control
# ============================================================================
class SessionMgr:
    """จัดการ session อิสระ 3 ตัว (เก็บเวลาเริ่ม/หยุด + ออกไฟล์รายตัว)

    ขอบเขตของแต่ละสิ่ง (ตกลงกันไว้ตอนออกแบบ bridge กับจอสัมผัส)
    ------------------------------------------------------------
        session    = รายเซนเซอร์   เริ่ม/หยุดอิสระ เป็นหน่วยที่ออกรายงาน
        sample_id  = รายเซนเซอร์   จับตอน "เริ่ม" session  <- ย้ายมาจากระดับ run
        note       = รายเซนเซอร์   ถามตอน "หยุด" session
        recording  = any(session)  ให้ปุ่มเดียวบนจอสอดคล้องกับหลาย session

    ⚠️ ของเดิม sample เป็นค่าเดียวของทั้งรอบการรัน (argument --sample)
       ซึ่งใช้ไม่ได้กับจอที่เปลี่ยน sample ระหว่างทางได้
       ตอนนี้ --sample กลายเป็น "ค่าตั้งต้นของ session ใหม่" แทน
    """

    def __init__(self, sample, auto_open, note_reader=None, jobs=None):
        self.start = [None, None, None]     # datetime หรือ None
        self.sample_of = [None, None, None]  # sample ที่จับไว้ตอนเริ่มแต่ละตัว
        self.default_sample = sample        # ค่าตั้งต้นของ session ใหม่
        self.pending_sample = None          # ตั้งจากจอด้วย sample_set (ภายหลัง)
        self.auto_open = auto_open
        self.note_reader = note_reader      # ฟังก์ชันอ่าน Note (None = ข้าม)
        self.jobs = jobs                    # None = ออกรายงานแบบบล็อกเหมือนเดิม
        self._write_status()

    # ---- สถานะที่ผู้อื่นอ่าน ----

    def session_mask(self):
        """bit0 = sensor 1 — รูปแบบเดียวกับที่ pc_bridge ฝั่งจอคาดหวัง"""
        m = 0
        for i, st in enumerate(self.start):
            if st is not None:
                m |= (1 << i)
        return m

    def any_active(self):
        return self.session_mask() != 0

    def sample_now(self):
        """sample ที่จะใช้กับ session ถัดไป (หรือของตัวที่กำลังรันอยู่)"""
        for s in self.sample_of:
            if s:
                return s
        return self.pending_sample or self.default_sample

    def set_sample(self, sample):
        """ตั้ง sample ของ session ที่จะเริ่มถัดไป — ยังไม่แตะตัวที่รันอยู่

        ไม่เปลี่ยนย้อนหลังโดยตั้งใจ: รายงานที่ออกไปแล้วกับ session ที่กำลัง
        เก็บอยู่ต้องอ้างชื่อตัวอย่างเดียวกันตลอดช่วงเวลาของมัน
        """
        self.pending_sample = (sample or "").strip() or None
        return self.pending_sample

    def _write_status(self):
        """เขียนสถานะบันทึกสด -> rec_status.json (ให้ desktop UI อ่าน)"""
        try:
            with open(REC_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump({"active": [s is not None for s in self.start],
                           "mask": self.session_mask(),
                           "sample": list(self.sample_of),
                           "updated": datetime.now().isoformat()}, f,
                          ensure_ascii=False)
        except Exception:
            pass

    # ---- เริ่ม / หยุด ----

    def toggle(self, i):
        if self.start[i] is None:
            self.start_session(i)
        else:
            self._close(i, datetime.now())

    def start_session(self, i, sample=None):
        """เริ่ม session คืน False ถ้ามีอยู่แล้ว

        ⚠️ ต้อง idempotent — จอไม่ retry เอง แต่ผู้ใช้กดซ้ำได้เมื่อเห็น
           "Request status unknown"  ถ้าการกดซ้ำไปสร้าง session ใหม่ทับ
           จะได้ session ซ้อนทันที  ผู้เรียกต้องถือว่า False = "สำเร็จอยู่แล้ว"
           ไม่ใช่ "ล้มเหลว"
        """
        if self.start[i] is not None:
            return False
        self.start[i] = datetime.now()
        self.sample_of[i] = (sample or self.pending_sample
                             or self.default_sample or "-")
        print(f"\n  > SENSOR {i+1:02d} START session @ {self.start[i]:%H:%M:%S}"
              f"  sample={self.sample_of[i]}")
        self._write_status()
        return True

    def _close(self, i, end):
        start = self.start[i]
        sample = self.sample_of[i] or self.default_sample or "-"
        self.start[i] = None
        self.sample_of[i] = None
        self._write_status()
        dur = end - start
        print(f"\n  # SENSOR {i+1:02d} STOP session (duration {dur})")
        # ถาม Note (ภาษาอังกฤษสั้น ๆ) ก่อนออกไฟล์
        note = ""
        if self.note_reader:
            try:
                note = self.note_reader(i)
            except Exception:
                note = ""
        append_session_log({"sensor": i + 1, "start": start.isoformat(),
                            "end": end.isoformat(), "sample": sample,
                            "note": note})

        # ------------------------------------------------------------------
        #  ออกรายงานใน worker แยก
        #
        #  ⚠️ ห้ามเรียกตรงนี้เด็ดขาด — ระหว่างสร้าง PDF จะไม่มีใครเรียก
        #     ser.readline() เลย ข้อมูลจะค้างใน buffer ของ OS แล้วถูกอ่าน
        #     รวดเดียว ทำให้ timestamp ของ CSV กระจุกผิดตำแหน่ง
        #     (วัดจริง: session 8 ชม. = 2.0 วิ, หยุด 3 ตัว = 5.9 วิ)
        #     รายละเอียดใน report_jobs.py
        # ------------------------------------------------------------------
        def _make():
            import report_3ec
            return report_3ec.export_sensor_session(
                i, since=start, until=end, data_dir=DATA_DIR,
                sample=sample, auto_open=self.auto_open, note=note)

        if self.jobs is not None:
            self.jobs.submit(f"SENSOR {i+1:02d} session report", _make)
            print(f"  รายงานเข้าคิวแล้ว — การเก็บ CSV ไม่หยุด")
        else:
            try:
                _make()
            except Exception as e:
                print(f"  !! SENSOR {i+1:02d} export failed: {e}")
                print(f"     CSV data is safe - generate later with report_3ec.py")

    def status(self):
        print("\n  --- session status ---")
        for i in range(3):
            if not _shown(i):
                print(f"    SENSOR {i+1:02d}: DISABLED (hidden on the dashboard)")
            elif self.start[i]:
                dur = datetime.now() - self.start[i]
                print(f"    SENSOR {i+1:02d}: REC   since {self.start[i]:%H:%M:%S}  ({dur})")
            else:
                print(f"    SENSOR {i+1:02d}: idle")
        print("  ---------------------")

    def close_all(self):
        end = datetime.now()
        any_open = False
        for i in range(3):
            if self.start[i]:
                any_open = True
                self._close(i, end)
        return any_open


def _help():
    print("  keys: [1][2][3]=start/stop that sensor | [c]=calibrate | "
          "[p]=status | [q]=quit")


# ============================================================================
#  main loop
# ============================================================================
def run(port, service=False, sample="-", auto_open=True):
    interactive = (not service) and KeyInput.available and sys.stdin.isatty()

    print(f"[logger3] start | port={port} | baud={BAUD}")
    print(f"[logger3] combined CSV: {DATA_DIR}/  | per-sensor output: sensor_1/ 2/ 3/")
    if interactive:
        _help()
    elif not service:
        print("[logger3] (non-interactive terminal - logging CSV only, Ctrl+C to stop)")

    jobs = report_jobs.ReportJobs(tag="report")
    mgr = SessionMgr(sample, auto_open, jobs=jobs)
    run_start = datetime.now()          # เวลาเริ่ม run (สำหรับรายงานรวมตอนปิด)
    cur_day, f, w, ser, n = None, None, None, None, 0
    latest = [None, None, None]         # ค่า EC ล่าสุดของแต่ละตัว (ให้ตัวคาลิเบรตใช้)

    keys = KeyInput() if interactive else None
    if keys:
        keys.__enter__()
        mgr.note_reader = lambda i: keys.line(
            f"  Note EC#{i+1} (English, Enter to skip): ")

    # เตือนถ้ายังไม่ได้คาลิเบรตวันนี้ — ผู้ใช้ต้อง cal ทุกวัน
    if not service:
        for i in range(3):
            if not _shown(i):
                continue          # ปิดไว้อยู่แล้ว ไม่ต้องเตือนให้ไปคาลิเบรต
            d = calibration.days_since_calibration(i)
            if d is None:
                print(f"[logger3] SENSOR {i+1:02d}: never calibrated - press [c] to calibrate")
            elif d >= 1.0:
                print(f"[logger3] SENSOR {i+1:02d}: last calibrated {d:.1f} days ago - recalibrate")

    def do_calibrate():
        """คาลิเบรตระหว่าง logger รันอยู่ — ใช้สตรีมเดิม ไม่ต้องแย่ง port"""
        raw = keys.line("  calibrate which sensor? [1/2/3, Enter=cancel]: ")
        if raw not in ("1", "2", "3"):
            print("  cancelled")
            return
        idx = int(raw) - 1

        std_txt = keys.line(
            f"  standard solution uS/cm [Enter={calibration.DEFAULT_STANDARD}]: ")
        try:
            std = float(std_txt) if std_txt else calibration.DEFAULT_STANDARD
        except ValueError:
            print("  invalid value - cancelled")
            return

        def pump():
            """ดูดข้อมูลจาก serial ต่อระหว่างคาลิเบรต ไม่ให้ CSV ขาดช่วง"""
            nonlocal n
            try:
                line = ser.readline().decode("utf-8", "ignore")
            except Exception:
                return ""
            vals = parse(line)
            if vals:
                for k in range(3):
                    latest[k] = float(vals[k * 2]) if vals[k * 2] else None
                if w:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    w.writerow([ts] + vals + ["CAL"])
                    f.flush()
                    n += 1
            return line

        cal = calibration.Calibrator(ser, lambda k: latest[k], pump=pump)
        cal.run(idx, std, ask=lambda p: keys.line(p) != "x")
        _help()

    try:
        reader = LineReader()
        n_bad = 0            # แถว DATA ที่อ่านไม่ได้ — ตัวนี้คือสัญญาณจริง
        n_other = 0          # บรรทัดสถานะ/debug ของบอร์ด — ปกติ ไม่ใช่ปัญหา

        while True:
            # ---- เช็กคีย์บอร์ด (ไม่บล็อก) ----
            if keys:
                ch = keys.get()
                if ch in ("1", "2", "3"):
                    mgr.toggle(int(ch) - 1)
                elif ch == "c":
                    if ser is not None and ser.is_open:
                        do_calibrate()
                    else:
                        print("\n  board not connected - cannot calibrate")
                elif ch == "p":
                    mgr.status()
                elif ch == "q":
                    print("\n[logger3] quit (q)")
                    break

            # ---- serial ----
            if ser is None or not ser.is_open:
                try:
                    ser = serial.Serial(port, BAUD, timeout=0.5)
                    time.sleep(2); ser.reset_input_buffer()
                    print(f"[logger3] connected to {port}")
                    if interactive:
                        _help()
                except (serial.SerialException, OSError) as e:
                    print(f"[logger3] connect failed: {e} | retry {RECONNECT_DELAY}s")
                    time.sleep(RECONNECT_DELAY); continue

            try:
                raw = reader.poll(ser)
            except (serial.SerialException, OSError) as e:
                print(f"[logger3] USB dropped: {e} | reconnecting")
                try: ser.close()
                except Exception: pass
                ser = None; reader.reset()
                time.sleep(RECONNECT_DELAY); continue

            if raw is None:
                continue                  # ยังไม่ครบบรรทัด — ไม่ใช่ความผิดพลาด

            vals = parse(raw)
            if vals is None:
                # ⚠️ นับเฉพาะบรรทัดที่ "ควรจะอ่านได้แต่อ่านไม่ได้" เท่านั้น
                #
                #    บอร์ด CONTROL พ่นบรรทัดอื่นออกมาตลอดเวลาโดยตั้งใจ เช่น
                #      #1 ERR  #2 EC:84.8 T:21.3  #3 EC:0.0 T:21.7
                #      [espnow] seq=163 sent=162 fail=0 txerr=0
                #    พวกนี้ปกติดี ไม่ใช่ความผิดพลาด
                #
                #    รอบแรกผมนับทุกบรรทัดที่ไม่ใช่ DATA ซึ่งได้เลขหลักหมื่นใน
                #    12 ชั่วโมง แล้วบรรทัดที่เสียจริงสิบครั้งจะจมหายไปในนั้น
                #    ตัวนับที่ส่งเสียงตลอดเวลาเท่ากับไม่มีตัวนับ
                #
                #    บรรทัดที่ขาดครึ่งจะขึ้นต้นด้วย DATA, เสมอแต่ฟิลด์ไม่ครบ
                #    จึงกรองด้วยเงื่อนไขนั้น
                if raw.strip().startswith("DATA,"):
                    n_bad += 1
                    if n_bad <= 5 or n_bad % 20 == 0:
                        print("[logger3] แถว DATA เสียครั้งที่ %d: %r"
                              % (n_bad, raw[:70]))
                else:
                    n_other += 1          # บรรทัดปกติของบอร์ด — เงียบไว้
                continue

            for k in range(3):
                latest[k] = float(vals[k * 2]) if vals[k * 2] else None

            today = datetime.now().strftime("%Y-%m-%d")
            if today != cur_day:
                if f: f.close()
                f, w = open_csv(daily_path()); cur_day = today
                print(f"[logger3] writing file: {daily_path()}")

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([ts] + vals + [""])
            f.flush()
            n += 1
            #  fsync บังคับให้ดิสก์เขียนจริง ซึ่งบล็อกได้นานบน Windows
            #  ของเดิมทำทุกแถว = ~2,800 ครั้งใน 2 ชั่วโมง และระหว่างที่บล็อกอยู่
            #  ไม่มีใครอ่าน serial เลย
            #  flush() อย่างเดียวข้อมูลก็ถึง OS แล้ว จึงไม่หายแม้โปรแกรมถูกฆ่า
            #  จะหายก็ต่อเมื่อไฟดับทั้งเครื่องภายใน 20 แถว (~50 วินาที)
            #  ซึ่งแลกกับการอ่าน serial ไม่ทันแล้วคุ้มกว่ามาก
            if n % 20 == 0:
                os.fsync(f.fileno())

            # ----------------------------------------------------------------
            #  บรรทัดสรุป — ใช้คำชุดเดียวกับ desktop_ui.py และจอสัมผัส
            #
            #  ของเดิมพิมพ์ "EC1=-- " เมื่ออ่านไม่ได้ ซึ่งหน้าตาเหมือนกันหมด
            #  ไม่ว่าหัววัดจะถูกถอดออกเอง หรือเพิ่งเงียบไปเฉย ๆ
            #  ตรงนี้จึงแยก DISABLED ออกจาก NO RESPONSE ให้ชัด
            # ----------------------------------------------------------------
            if n % 5 == 0 and not service:
                parts = []
                for i in range(3):
                    tag = f"#{i+1}"
                    if not _shown(i):
                        parts.append(f"{tag} DISABLED")
                        continue
                    ok = vals[6 + i] == "1"
                    raw = vals[i * 2]
                    if ok and raw:
                        val = _T.format_ec(float(raw)) if _T else raw
                    else:
                        val = "NO RESPONSE"
                    parts.append(f"{tag} {val}" + ("  REC" if mgr.start[i] else ""))
                print(f"[{ts[11:]}] rows={n:,}   " + "   ".join(parts))

    except KeyboardInterrupt:
        print(f"\n[logger3] stopped (Ctrl+C) | logged {n} rows"
              + (f" | แถว DATA เสีย {n_bad}" if n_bad else " | แถว DATA เสีย 0")
              + f" | บรรทัดอื่นของบอร์ด {n_other}")
    finally:
        if keys:
            keys.__exit__(None, None, None)     # คืนโหมด terminal (เฉพาะ Linux)
        # ปิด session ที่ยังค้าง + เข้าคิวไฟล์แยกรายตัว
        if mgr.close_all():
            print("[logger3] closed open sessions - reports queued")
        if f: f.close()
        if ser and ser.is_open: ser.close()

        # รายงานรวม 3 ตัว ของทั้ง run (ตอนปิด terminal) -> reports/
        if (not service) and n > 0:
            run_note = ""
            if interactive:
                try:
                    run_note = input(
                        "\n[logger3] Note for combined run report (Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    run_note = ""
            run_end = datetime.now()

            def _combined():
                import report_3ec
                return report_3ec.export_combined_report(
                    since=run_start, until=run_end, data_dir=DATA_DIR,
                    sample=sample, note=run_note, auto_open=auto_open)

            jobs.submit("combined run report", _combined)

        # ต้องรอให้เขียนไฟล์จนจบก่อนปิด process
        # ไม่รอ = ได้ PDF ที่เขียนไม่จบ ซึ่งแย่กว่าไม่ได้ไฟล์เลย
        jobs.shutdown(wait=True)

        print("[logger3] closed cleanly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--sample", default="-", help="sample name / ID (written into the reports)")
    ap.add_argument("--service", action="store_true",
                    help="headless: no keyboard, no reports, CSV only")
    ap.add_argument("--no-open", action="store_true", help="generate files but do not open the PDF")
    args = ap.parse_args()

    port = args.port or find_port()
    if not port:
        print("!! port not found - specify one, e.g. --port COM5"); sys.exit(1)
    run(port, service=args.service, sample=args.sample, auto_open=not args.no_open)
