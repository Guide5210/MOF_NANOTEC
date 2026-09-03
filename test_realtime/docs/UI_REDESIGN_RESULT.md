# PC Dashboard — Lab Theme Redesign : ผลการทำงาน

- วันที่: 25 สิงหาคม 2026
- ขอบเขต: visual design / layout / component style / chart style เท่านั้น
- ตรรกะข้อมูล · CSV parser · Modbus · serial protocol · session logic ·
  report calculation · command bridge — **ไม่ถูกแตะ**

---

## 1. ไฟล์ที่เปลี่ยน

| ไฟล์ | สถานะ | บรรทัด |
|---|---|---|
| `lab_theme.py` | **ใหม่** | ~400 |
| `ec_ui_config.json` | **ใหม่** | 7 |
| `desktop_ui.py` | เขียนใหม่เฉพาะชั้นแสดงผล | 548 → 1,318 |
| `logger_3ec.py` | แก้เฉพาะข้อความในคอนโซล | 522 → 571 |
| `desktop_ui.py.bak` · `logger_3ec.py.bak` | **สำรองของเดิม** | — |

**ไม่แตะ:** `report_3ec.py` · `calibration.py` · `console_utf8.py` ·
`water_monitor_3ec.ino` · ทุกไฟล์ใน `hello_world/`

---

## 2. คำสั่งรัน

```bash
cd C:\MOF_NanoTec\test_realtime

# ต้องมี matplotlib + openpyxl (ของเดิมอยู่แล้ว)
python -m pip install matplotlib openpyxl

# ตรวจว่าคอมไพล์ผ่าน
python -m py_compile desktop_ui.py lab_theme.py logger_3ec.py

# รันหน้าจอ
python desktop_ui.py

# รัน logger เหมือนเดิมทุกประการ
python logger_3ec.py --sample "CALF-20 batch 3"
```

**เปลี่ยนว่าจะแสดงหัววัดตัวไหน:** กดปุ่ม `Display setup` มุมขวาบน
หรือแก้ `ec_ui_config.json` โดยตรง

```json
{ "active_mask": 6 }     ← 0b0110 = แสดง #2 #3, ซ่อน #1 (ค่าที่ตั้งไว้ตอนนี้)
```

| mask | ความหมาย |
|---|---|
| `2` | เฉพาะ #2 |
| `6` | #2 #3 — **ค่าปัจจุบัน** (#1 ที่เสียถูกซ่อน) |
| `7` | #1 #2 #3 — พฤติกรรมเดิมทั้งหมด |
| `15` | ครบ 4 ตัว |
| `13` | #1 #3 #4 (ข้าม #2) |

---

## 3. ผลการตรวจสอบ

| # | ตรวจอะไร | ผล |
|---|---|---|
| **V1** | ฟังก์ชันข้อมูล 12 ตัวใน `desktop_ui.py` ไม่เปลี่ยน | ✅ **เหมือนเดิมทุกไบต์** — เทียบด้วย AST ระหว่าง `.bak` กับไฟล์ใหม่ `list_files · _num · read_range · read_recent · downsample · make_mock_csv · load_sessions · save_sessions · export_csv · _stats · export_excel · export_pdf` |
| **V2** | อ่าน CSV เดิมได้ | ✅ 960 แถวจากไฟล์ทดสอบ ค่าตรงกัน |
| **V3** | Export ยังทำงานครบ | ✅ `t.csv` 46 KB · `t.xlsx` 37 KB · `report_3ec` PDF 108 KB (9 หน้า) · `report_3ec` xlsx 37 KB · fallback PDF 48 KB |
| **V3b** | **`report_3ec.py` ไม่ถูกธีมของ PC กวน** | ✅ ตรวจ `rcParams` ก่อน/หลังสร้าง UI — เปลี่ยนแค่ `backend_fallback` ที่ matplotlib ตั้งเอง ธีมทั้งหมดตั้งทีละ figure/axes ไม่แตะ global |
| **V4** | Session dropdown | ✅ อ่านจาก `sessions_3ec.json` เดิม เลือกแล้วกราฟเปลี่ยนช่วง |
| **V5** | mask `0b0110` | ✅ #1 = `DISABLED` เทา ไม่มีแผงกราฟของ #1 |
| **V6** | mask `0b0111` | ✅ กลับเป็น 3 คอลัมน์ |
| **V7** | mask 1 / 2 / 4 ตัว | ✅ ทุกผัง (ดูภาพ) |
| **V8** | ไม่กะพริบ / zoom ไม่หาย | ✅ ใช้ `set_data()` + `draw_idle()` autoscale เฉพาะตอนตามค่าสด |
| **V9** | Windows | ⏳ **ต้องให้คุณรันยืนยัน** — ทดสอบบน Linux + Xvfb แล้ว ฟอนต์เลือกอัตโนมัติ (`Segoe UI` บน Windows) |
| **V10** | Linux | ✅ รันจริงบน Ubuntu / Tk 8.6 / matplotlib 3.11 |
| **V11** | ไม่มีสีดิบใน `desktop_ui.py` | ✅ **0 จุด** (เดิม 11 จุด) |
| **V12** | ไม่มีข้อความเล็กกว่า 12 px | ✅ เล็กสุด 12 px (เดิมมี 9 px สองจุด) |
| **V13** | screenshot | ✅ แนบแล้ว |

---

## 4. สิ่งที่เจอระหว่างทำ และแก้ไปด้วย

| # | เรื่อง | ทำไมสำคัญ |
|---|---|---|
| 1 | **`font=("Sans", ...)` ของเดิมใช้ไม่ได้บน Windows** — `Sans` เป็น alias ของ X11 Tk จึง fallback เงียบ ๆ | หน้าตาที่คุณเห็นบน Windows ไม่เคยเป็นสิ่งที่โค้ดระบุ ตอนนี้เลือกจากฟอนต์ที่ติดตั้งจริง |
| 2 | **Tk ตีความขนาดฟอนต์เป็นพอยต์ ไม่ใช่พิกเซล** — 34 → ~45 px | สเปกที่คุณกำหนดเป็นพิกเซล ถ้าไม่แก้ การ์ดจะสูงเกินงบจนกราฟไม่เหลือที่ ตอนนี้ใช้เลขติดลบ = พิกเซลตามสเปกเป๊ะ |
| 3 | **matplotlib toolbar: ปุ่ม Back/Forward ที่ถูก disable ขึ้นเป็นลายตาราง** บนพื้นขาว | ตัดปุ่มที่ไม่ได้ใช้ออก เหลือ Home / Pan / Zoom / Save |
| 4 | **`ax.clear()` ทุก 10 วินาที ล้าง zoom ที่ผู้ใช้เพิ่งตั้ง** | เปลี่ยนเป็น `set_data()` — zoom อยู่ครบ ไม่กะพริบ |
| 5 | **หน้าต่างสถานะ 5 นาทีทำให้บอร์ดที่เงียบเกิน 5 นาทีเด้งไป `OFFLINE`** ทั้งที่ควรเป็น `STALE` | ขยายเป็น 30 นาที — ไม่มีต้นทุนเพิ่ม เพราะ `read_range()` อ่านทุกไฟล์อยู่แล้ว |
| 6 | **4 แผงกราฟในโหมด Split อ่านไม่ออก** (แผงละ ~50 px) | เกิน 3 แผงบังคับใช้ Overlay และปิดปุ่ม Split พร้อมบอกเหตุผลบนหัวแผง |
| 7 | **`toggle_mock()` เดิมเรียก `self._logger_active()` ที่ไม่มีอยู่จริง** | เป็นบั๊กค้างที่ถูกกลืนโดย `except` — เอาออกแล้ว |
| 8 | **`_stats` / `export_*` จะโดนธีมของเราเปลี่ยนหน้าตา ถ้าตั้ง `rcParams` แบบ global** | จึงตั้ง style ทีละ figure/axes เท่านั้น รายงานที่ส่งอาจารย์หน้าตาไม่เปลี่ยน |

---

## 5. Rollback

| ระดับ | วิธี | เวลา |
|---|---|---|
| **แค่ปิด/เปิดหัววัด** | แก้ `active_mask` ใน `ec_ui_config.json` หรือกด `Display setup` | ทันที |
| **กลับ UI เดิมทั้งหมด** | `copy desktop_ui.py.bak desktop_ui.py` | 5 วินาที |
| **กลับ logger เดิม** | `copy logger_3ec.py.bak logger_3ec.py` | 5 วินาที |
| **ลบของใหม่ทิ้ง** | ลบ `lab_theme.py` และ `ec_ui_config.json` | — |

`lab_theme.py` และ `ec_ui_config.json` เป็น **ไฟล์เพิ่ม** ล้วน ๆ
ลบทิ้งแล้วไฟล์ `.bak` ทำงานได้ทันทีโดยไม่ต้องแก้อะไรอีก

> `logger_3ec.py` ใหม่ import `lab_theme` แบบ optional — ถ้าไฟล์หาย
> logger ยังเก็บข้อมูลได้ตามปกติ แค่กลับไปพิมพ์แบบไม่มีตัวคั่นหลักพัน

---

## 6. สิ่งที่ยังไม่ได้ทำ (ตามที่ตกลงไว้ว่าอยู่นอกรอบนี้)

- **marker เหตุการณ์บนกราฟ** — นิยาม `MARKER_STYLES` ไว้ใน `lab_theme.py` แล้ว
  รอ Phase 3b มาเสียบข้อมูล ไม่ต้องแก้ styling อีก
- **แถบ `CALIBRATING`** — โค้ดพร้อมแล้ว อ่านจาก `cal_status.json`
  ไฟล์นั้นจะเกิดขึ้นเมื่อทำ Phase 3b (cal observer)
- **EVENT LOG** — ตอนนี้แสดงเฉพาะสิ่งที่ PC รู้เอง
  (session เริ่ม/หยุด · export · mock · board offline/recovered)
  เหตุการณ์จากจอสัมผัส (`SAVED` / `STABLE` / `CALIBRATION`) จะเข้ามาใน Phase 3b
- **`active_mask` จริงจากบอร์ด** — ตอนนี้เป็นค่าการแสดงผลของ PC เท่านั้น
  บอร์ดยังอ่านครบ 3 ตัว และ CSV ยังเก็บครบทุกคอลัมน์ (จำเป็น เพราะจอ P4 ใช้
  การเห็นคำถามถึง address 1 เป็นตัวตัดรอบ — ดู R1 ในเอกสาร Phase 1)

---

## 7. สิ่งที่ขอให้แก้ฝั่ง P4 (4 บรรทัดใน `ui_tokens.h`)

ดู `docs/P4_TOKEN_SYNC.md`
