# PSA Analyzer — Demo 1

โปรแกรมวิเคราะห์ข้อมูล Pressure Swing Adsorption (PSA) พร้อมโหมด **Live Monitor**
อ่านข้อมูลเรียลไทม์จาก PLC และเครื่องวิเคราะห์แก๊สโดยตรงผ่าน Modbus TCP

> 📋 **สถานะล่าสุด ข้อค้นพบ และรายการที่ยังค้าง อยู่ใน [PROGRESS.md](PROGRESS.md)**
> — อ่านไฟล์นั้นก่อนเริ่มงานต่อ จะได้ไม่ไล่ทางตันซ้ำ

---

## 1. การติดตั้ง / การรัน (เครื่องที่มี eServer)

ไฟล์ `dist/PSA_Analyzer_Demo1.exe` เป็น **standalone** — รวม Python + ไลบรารีทุกอย่างไว้ในไฟล์เดียว
**ไม่ต้องติดตั้ง Python** บนเครื่องปลายทาง

1. ก๊อปไฟล์ `PSA_Analyzer_Demo1.exe` ไปวางบนเครื่อง eServer (เช่นบน Desktop)
2. ดับเบิลคลิกเปิด
3. ครั้งแรก Windows SmartScreen อาจเตือน "unknown publisher" → กด **More info → Run anyway**
   (ปกติของไฟล์ที่ยังไม่ได้ code-sign)
4. ครั้งแรกเปิดอาจช้า ~5–10 วิ (ต้องแตกไฟล์ลง temp ก่อน)

---

## 2. วิธีใช้งาน

### โหมดวิเคราะห์ไฟล์ (เดิม)
1. ลากไฟล์ Excel เข้า หรือกด **Browse Excel file**
2. กด **Start Analysis** → เลือกช่วง/การทดลอง → ดูผล + กราฟ
3. Export ได้: **Excel (.xlsx)**, **PDF Report**, **กราฟ (PNG/SVG)**

### โหมด Live Monitor (ใหม่ — อ่านจาก eServer)
1. ตั้ง eServer ให้เขียน CSV (timestamp + MFC-01, MFC-07, BPR-01, CO2 vol%)
2. ในแอปกดปุ่ม **"● Live Monitor (PLC)"**
3. เลือกไฟล์ CSV ที่ eServer เขียน (หรือเลือกโฟลเดอร์ ถ้า eServer สร้างไฟล์ใหม่ตามวัน)
4. ดูเลขสด + กราฟ scrolling + KPI cycle อัปเดตอัตโนมัติ

> สูตรคำนวณในโหมด Live **เหมือนโหมดไฟล์ทุกประการ** (MAX7 purity, cycle จาก BPR,
> Final เฉลี่ยจาก cycle 3, productivity) — ใช้โค้ดเดียวกัน

---

## 3. การแก้ไข / พัฒนาต่อในอนาคต (สำหรับนักวิจัย/ผู้ดูแลโค้ด)

`.exe` เป็นไฟล์ที่ build แล้ว **แก้ในไฟล์ .exe โดยตรงไม่ได้** — ต้องแก้ที่ซอร์สโค้ด
แล้ว build ใหม่ บนเครื่องที่มี Python

### เตรียมเครื่อง dev (ครั้งเดียว)
```powershell
# ติดตั้ง Python 3.11 ก่อน แล้วในโฟลเดอร์โปรเจกต์:
pip install -r requirements.txt
pip install pyinstaller
```

### แก้โค้ด → ทดสอบ
```powershell
python main.py            # รันจากซอร์สเพื่อทดสอบทันที (ไม่ต้อง build)
```

### build .exe ใหม่
```powershell
pyinstaller PSA_Analyzer.spec --noconfirm --clean
# ได้ไฟล์ใหม่ที่ dist\PSA_Analyzer_Demo1.exe
```

### โครงสร้างโค้ด (ที่ที่ควรแก้)
| ไฟล์ | หน้าที่ |
|---|---|
| `psa_analyzer/core/analyzer.py` | **สูตรคำนวณหลัก** (purity/recovery/productivity/cycle) |
| `psa_analyzer/core/constants.py` | ชื่อคอลัมน์, ค่า default, ค่าคงที่ฟิสิกส์ |
| `psa_analyzer/core/plc_map.py` | **แผนที่ register ของเซ็นเซอร์ทุกตัว** (PLC + เครื่องวิเคราะห์แก๊ส) |
| `psa_analyzer/core/modbus_client.py` | ตัวอ่าน Modbus TCP (float/uint16/int16, เลือก word order ได้) |
| `psa_analyzer/core/live_csv.py` | อ่านไฟล์ CSV แบบ tail (เฉพาะแถวใหม่) |
| `psa_analyzer/core/live_buffer.py` | สะสมข้อมูลสด + parse timestamp |
| `psa_analyzer/workers/modbus_worker.py` | thread poll Modbus (คุยได้ทั้ง PLC และ VA-5000 พร้อมกัน) |
| `psa_analyzer/workers/live_worker.py` | thread อ่านไฟล์ทุก ~1 วิ |
| `psa_analyzer/ui/live_window.py` | หน้าจอ Live Monitor |
| `psa_analyzer/ui/modbus_live_config.py` | หน้าต่างตั้งค่า IP / register ก่อนเริ่ม Live |
| `psa_analyzer/ui/main_window.py` | หน้าต่างหลัก, KPI, กราฟ |
| `psa_analyzer/ui/sidebar.py` | แถบควบคุมซ้าย (พารามิเตอร์, ปุ่ม) |

### เครื่องมือทดสอบ
```powershell
# จำลอง eServer เขียน CSV โตขึ้นเรื่อยๆ (ทดสอบ Live โดยไม่ต้องมี PLC)
python tools\fake_eserver.py

# อ่านค่าแก๊สจากเครื่อง HORIBA VA-5000 โดยตรง (ยืนยัน address / ดูค่าสด)
python tools\read_horiba.py 192.168.1.100 --watch
```

---

## 5. แหล่งข้อมูลของค่าแก๊ส — HORIBA VA-5000 (ไม่ใช่ PLC)

CO / CO₂ / CH₄ / O₂ **ไม่ได้มาจากเซ็นเซอร์ของ PLC** แต่มาจากเครื่องวิเคราะห์แก๊ส
HORIBA VA-5000 ซึ่งเป็น Modbus/TCP server ของตัวเองบนวงเดียวกัน โปรแกรมจึงเปิด
การเชื่อมต่อที่สองไปหาเครื่องนี้โดยตรง (ไม่ต้องพึ่ง PLC)

ค่าทั้งหมดอ้างจากคู่มือ VA-5000 บทที่ 8 "External Input/Output":

| หัวข้อ | ค่า |
|---|---|
| IP (จากหน้าจอเครื่อง COMMUNICATION 1/2) | `192.168.1.100` |
| Port / Protocol | 502 / Modbus-TCP |
| **Slave address** | **255 (fixed)** — ไม่ใช่ 1 และเครื่องบังคับใช้จริง |
| Function code | 03 (Read Holding Registers) |
| **Word order ของ float** | **big** (high word ก่อน, Fig. 94) — ตรงข้ามกับ PLC ที่เป็น little |
| Component 1–4 concentration | `10506` / `10508` / `10510` / `10512` (float, 2 word) |
| หน่วยของแต่ละ component | `10874`–`10877` (0 = vol%) |
| สถานะกำลังวัด | `10840` bit 9 |
| แก๊สที่ไหลอยู่ | `11520` (0 = sample gas, 1 = calibration gas) |

> ⚠️ คู่มือระบุว่า **ห้ามอ่าน/เขียน address ที่ไม่อยู่ในตาราง** เพราะ
> "may affect the performance of the analyzer" — ดังนั้นห้ามใช้ `tools\scan_horiba.py`
> (ตัวกวาดหา address แบบสุ่ม) ยิงใส่เครื่องนี้ ให้ใช้ `tools\read_horiba.py` แทน

**Component ไหนคือแก๊สอะไร — ยืนยันหน้างานแล้ว 13 ส.ค. 2026** โดยเทียบค่า Modbus
กับหน้าจอ MEASUREMENT ของเครื่อง (CH₄ −0.0 / CO₂ 14.5 / O₂ 9.76):

| Component | Address | แก๊ส | Range |
|---|---|---|---|
| 1 | 10506 | CH₄ | 100 vol% |
| 2 | **10508** | **CO₂** ← ตัวที่ purity ใช้ | 100 vol% |
| 3 | 10510 | O₂ | 25 vol% |
| 4 | 10512 | *ไม่มี* (range 0, digits 0) | — |

> เครื่องนี้ **ไม่มีช่อง CO** เลย — label เดิมใน eServer (CO/CO₂/CH₄/O₂) เลื่อนไปหนึ่งช่อง
> ตรงกับที่เคยบันทึกไว้ว่า "CO จริงๆ คือ CH₄"

ถ้าย้ายไปใช้กับเครื่องอื่นที่เรียง component ไม่เหมือนกัน เปลี่ยนได้ในหน้าตั้งค่า
Live Monitor ช่อง "CO₂ is Component …" โดยไม่ต้อง build ใหม่

---

## 4. หมายเหตุข้อมูล eServer (Demo 1)

- Timestamp: `M/D/YYYY  h:mm:ss AM/PM` (12 ชม.) — parser รองรับแล้ว
- คอลัมน์ที่ใช้คำนวณ: `DATE / TIME`, `MFC-01 (CO2) SLPM`, `MFC-07 (AD-GAS) SMLM`,
  `BPR-01 SLPM`, `CO2 (vol%)`
- คอลัมน์ก๊าซที่ label สลับ (`CO`→จริงคือ CH4, `CH4`→จริงคือ O2) และ `O2` ตัวสุดท้าย
  **ไม่ถูกใช้ในการคำนวณ** จึงไม่กระทบผลลัพธ์
- ถ้า eServer เปลี่ยนรูปแบบไฟล์/ชื่อคอลัมน์ → แก้ที่ `constants.py` (COLUMN_KEYWORDS)
  และ `live_buffer.py` (parse_timestamps)

---

*Build: Demo 1 — PyInstaller onefile, Python 3.11*
