# ติดตั้งบนเครื่องที่ต่อบอร์ดจริง

## คำตอบสั้น: ต้องคัดลอก **สองโฟลเดอร์** ไม่ใช่โฟลเดอร์เดียว

```
C:\MOF_NanoTec\
├── test_realtime\                 ← ระบบเดิม เป็นคนเก็บข้อมูลจริง  **ขาดไม่ได้**
└── ec_measurement_station_pc\     ← โปรเจกต์นี้ เป็นผู้อ่านและคุยกับจอ
```

โปรเจกต์นี้ **ไม่ได้เก็บข้อมูลเอง** มันอ่าน CSV และ `rec_status.json` ที่
`test_realtime` เขียน  ถ้าเครื่องปลายทางมีแต่โฟลเดอร์นี้:

- การ์ดเซนเซอร์จะขึ้น `OFFLINE` ทุกใบ · กราฟว่าง · `PC LOGGER OFFLINE`
- `hw_test.bat` ขั้น A จะตกทันที
- soak จะไม่ได้พิสูจน์อะไรเลย เพราะข้ออ้างหลักคือ *"bridge ไม่กวน logger"*
  ซึ่งพิสูจน์ไม่ได้ถ้าไม่มี logger รันอยู่

**สองโฟลเดอร์นี้ต้องเป็นพี่น้องกัน ห้ามซ้อนกัน** — `data/` ของโปรเจกต์นี้
ต้องไม่อยู่ใน `test_realtime` (มีด่านกันไว้ใน `config.py` และมีเทสต์คุม)

---

## ขั้นตอน

### 1. คัดลอกทั้งสองโฟลเดอร์ไปที่เครื่องปลายทาง

คัดลอกทั้งก้อนได้เลย (รวม `.git` ก็ได้ ไม่เสียหาย)
สิ่งที่ **ไม่ต้อง** คัดลอกไป: `__pycache__\` · `data\` (สร้างใหม่เอง)

> ถ้าใช้ `git clone` แทน: `config\app_config.json` และ `data\` จะไม่ตามไป
> เพราะอยู่ใน `.gitignore` — ต้องสร้างเองตามข้อ 3

### 2. Python + ไลบรารี

Python 3.8 ขึ้นไป ตอนติดตั้ง **ต้องติ๊ก `tcl/tk and IDLE`** (คือ tkinter
ซึ่งลงผ่าน pip ไม่ได้ ถ้าลืมติ๊กต้องลง Python ใหม่)

```bat
pip install -r requirements.txt
```

### 3. สร้างไฟล์ตั้งค่า

```bat
cd C:\MOF_NanoTec\ec_measurement_station_pc
copy config\app_config.example.json config\app_config.json
```

แล้วแก้ให้ path ตรงกับเครื่องนั้นจริง ๆ:

```json
"legacy": {
  "enabled": true,
  "root":       "C:/MOF_NanoTec/test_realtime",
  "data_dir":   "C:/MOF_NanoTec/test_realtime/water_data",
  "rec_status": "C:/MOF_NanoTec/test_realtime/rec_status.json",
  "sessions":   "C:/MOF_NanoTec/test_realtime/sessions_3ec.json",
  "reports_dir":"C:/MOF_NanoTec/test_realtime/reports",
  "read_only": true
}
```

### 4. ตรวจก่อนเริ่ม

```bat
run_check.bat
```

บอกครบว่าอะไรขาดและแก้ยังไง — ไม่เปิดพอร์ต ไม่เขียนอะไรลง legacy
รันขณะ logger เดิมทำงานอยู่ได้

ต้องได้ **"ส่วนที่จำเป็นครบแล้ว"** ก่อนไปต่อ

### 5. ดูว่าพอร์ตไหนเป็นอะไร

```bat
run_port_audit.bat
```

ต้องเห็นทั้ง `พอร์ต CONTROL` และ `พอร์ตจอ P4 (NDJSON)`
ถ้าพอร์ตจอขึ้นว่า **กำกวม** ให้จด COM ไว้ใส่ `--bridge-port` ทุกครั้ง

### 6. ลำดับการเปิด

```bat
:: หน้าต่างที่ 1 — logger เดิม  เปิดค้างไว้ตลอด
cd C:\MOF_NanoTec\test_realtime
run_logger.bat

:: หน้าต่างที่ 2 — เลือกอย่างใดอย่างหนึ่ง (พอร์ตจอเปิดได้ทีละโปรเซส)
cd C:\MOF_NanoTec\ec_measurement_station_pc
run_viewer.bat                              :: ดูหน้าจอ
run_hw_test.bat --steps A,B,C,E --soak 120  :: ชุดทดสอบ
```

⚠️ **viewer กับ hw_test รันพร้อมกันไม่ได้** ทั้งคู่ต้องการพอร์ต NDJSON ของจอ
   ถ้าเปิดค้างไว้ อีกตัวจะรายงานว่า "ไม่พบจอ" ซึ่งไม่จริง

---

## สิ่งที่ไม่ตามไปกับโฟลเดอร์ (ตั้งใจ)

| | ทำไม |
|---|---|
| `config\app_config.json` | มี path เฉพาะเครื่อง ถ้าตามไปจะชี้ผิดที่เงียบ ๆ |
| `data\` | หลักฐานการวัดของเครื่องนั้น ๆ ไม่ใช่ของโปรเจกต์ |
| ข้อมูลห้องแล็บทุกชนิด | `.gitignore` กันไว้ — CSV/PDF/Excel/JSONL ไม่เข้า git เด็ดขาด |

---

## เครื่องปลายทางไม่มีอินเทอร์เน็ต

`pip install` ต้องใช้เน็ต ถ้าเครื่องนั้นออกเน็ตไม่ได้ ให้ดาวน์โหลดล่วงหน้า
จากเครื่องที่ออกได้:

```bat
pip download -r requirements.txt -d wheels
:: คัดลอกโฟลเดอร์ wheels ไปด้วย แล้วที่เครื่องปลายทาง:
pip install --no-index --find-links wheels -r requirements.txt
```

`tkinter` มากับตัวติดตั้ง Python เอง ไม่ต้องใช้เน็ต

---

## ย้อนกลับ

โปรเจกต์นี้ไม่เขียนอะไรลง `test_realtime` เลย  ลบโฟลเดอร์นี้ทิ้งแล้ว
ระบบเดิมทำงานต่อได้ทันทีโดยไม่ต้องแก้อะไร (ดู `docs/ROLLBACK.md`)
