# PC Desktop Dashboard — Lab Theme Redesign Proposal

ปรับ `desktop_ui.py` ให้ใช้ design system เดียวกับ ESP32-P4 touchscreen HMI

- วันที่: 25 สิงหาคม 2026
- ขอบเขต: **visual design / layout / component style / chart style เท่านั้น**
- **ไม่แตะ:** raw data schema · CSV parser · Modbus · serial protocol · session logic ·
  report calculation · command bridge
- **ยังไม่มีการแก้โค้ด** — รออนุมัติ
- ไฟล์คู่กัน: `ui_mockup_lab_theme.html` (เปิดในเบราว์เซอร์เพื่อดูภาพเป้าหมาย)

---

# 1. Audit — visual style ที่ฝังอยู่ใน `desktop_ui.py`

## 1.1 สรุปตัวเลข

| | จำนวน |
|---|---|
| จุดที่ตั้งค่าหน้าตา (`bg=` `fg=` `font=` `padx=` `pady=` `width=` `height=`) | **53 จุด** |
| ค่าสีดิบที่ฝังในโค้ด | **11 ค่า** |
| ชุดฟอนต์ที่ฝัง | **11 จุด / 5 ขนาด** |
| ไฟล์ token กลาง | **ไม่มีเลย** |

## 1.2 สีที่ฝังอยู่ และปลายทางใหม่

| บรรทัด | ค่าเดิม | ใช้ทำอะไร | ปลายทาง |
|---|---|---|---|
| 46 | `COLORS = ["#1565C0","#2E7D32","#EF6C00"]` | เส้นกราฟ + แถบสีหัวการ์ด | `SERIES[0..3]` |
| 47 | `BG = "#0f1720"` | พื้นหลังหน้าต่าง | `BG = "#F4F7F8"` |
| 47 | `CARD = "#1a2430"` | พื้น card + ปุ่ม + facecolor กราฟ | **แยกเป็น 3 token** — `SURFACE`, `SURFACE_ALT`, `CHART_FACE` |
| 47 | `TXT = "#e6edf3"` | ข้อความหลัก | `TEXT = "#1D2A31"` |
| 47 | `MUT = "#8b98a5"` | ข้อความรอง · spine · tick | `TEXT_DIM = "#66757D"` |
| 48 | `GREEN = "#2ecc71"` | dot live + สถานะ LIVE | `OK = "#178F7A"` |
| 48 | `RED = "#e74c3c"` | dot offline + ERROR + REC | **แยก 2 token** — `ERROR` (สถานะ) กับ `REC` (จุดบันทึก) |
| 303 | `"#F9A825"` | ป้ายโหมดทดสอบ | `WARN` |
| 488 | `"#4a3800"` | พื้นปุ่ม mock ตอนเปิด | `WARN_SOFT` |

**ข้อสังเกตสำคัญ:** `CARD` ตัวเดียวถูกใช้ทั้ง *พื้น card*, *พื้นปุ่ม* และ *facecolor ของ figure*
พอเปลี่ยนเป็นธีมสว่างจะแยกไม่ออกทันที เพราะ card ต้องขาว แต่ปุ่มรองต้องเทาอ่อน
⇒ ต้องแตก token ตั้งแต่ต้น ไม่ใช่แค่เปลี่ยนค่า

## 1.3 ฟอนต์ที่ฝังอยู่

| บรรทัด | เดิม | ใหม่ |
|---|---|---|
| 299 | `("Sans", 15, "bold")` — ชื่อโปรแกรม | `FONT_TITLE` 19 px semibold |
| 314 | `("Sans", 26, "bold")` — ค่า EC | `FONT_VALUE` 34 px (56 px ในโหมด 1 sensor) |
| 300, 313, 317, 319, 344, 347 | `("Sans", 10)` / `("Sans", 10, "bold")` | `FONT_BODY` 14 px / `FONT_LABEL` 12 px |
| 316, 377 | `("Sans", 9)` | `FONT_SMALL` 12 px — **เดิม 9 px เล็กกว่าเกณฑ์ที่คุณตั้งไว้ (ห้ามต่ำกว่า 12)** |

`"Sans"` เป็นชื่อ alias ของ X11 ซึ่ง **บน Windows ไม่มี** Tk จึง fallback เงียบ ๆ ไปที่
`TkDefaultFont` ⇒ หน้าตาที่เห็นบน Windows วันนี้ไม่ใช่สิ่งที่โค้ดระบุ
⇒ ต้องเลือกฟอนต์ตาม OS จริง: `Segoe UI` (Win) → `Noto Sans Thai` / `DejaVu Sans` (Linux) → `TkDefaultFont`

## 1.4 สถานะที่แสดงได้ตอนนี้ vs ที่ต้องการ

| ตอนนี้ (`desktop_ui.py:411`) | |
|---|---|
| `● LIVE` (เขียว) | เมื่อ `ok == "1"` |
| `● ERROR` (แดง) | เมื่อ `ok != "1"` |

**มีแค่ 2 สถานะ** เทียบกับ 11 สถานะฝั่ง P4 ⇒ ปัญหาที่เห็นในภาพหน้าจอที่คุณส่งมา:
sensor #1 ที่ **ถอดออกโดยตั้งใจ** ขึ้นเป็น `ERROR` สีแดงตลอดเวลา
ทั้งที่ไม่มีอะไรผิดพลาดเลย — นี่คือกรณีตรงที่ทำให้ผู้ใช้เลิกเชื่อสีแดง

## 1.5 ปัญหา rendering ที่ต้องแก้ไปพร้อมกัน

| # | ปัญหา | จุด | ผล |
|---|---|---|---|
| U1 | `refresh_charts()` เรียก `ax.clear()` ทั้ง 3 แกน + `canvas.draw()` ทุก 10 วินาที | 431-448 | กราฟกะพริบทั้งแผ่น toolbar zoom ที่ผู้ใช้ตั้งไว้หายทุกครั้ง |
| U2 | `_hl()` เขียน `b.config(bg=...)` ทุกปุ่มทุกรอบ | 384-387 | ปุ่มกะพริบเบา ๆ |
| U3 | `refresh_now()` เขียน `.config(text=...)` ทุก label ทุก 2 วินาที แม้ค่าไม่เปลี่ยน | 399-415 | สิ้นเปลือง — P4 แก้ปัญหานี้ไปแล้วด้วย `ui_set_text()` |
| U4 | ไม่มี `ttk.Style` ⇒ `ttk.Combobox` ใช้ธีมของ OS พื้นขาว บนพื้นหลังเข้ม | 353 | ในภาพหน้าจอเห็นเป็นกล่องขาวโดดออกมา |
| U5 | `NavigationToolbar2Tk` ใช้สีเริ่มต้นของ Tk | 373 | แถบเทาสว่างบนพื้นเข้ม (ในภาพเห็นชัด) |

**U1 และ U3 คือสิ่งเดียวกับที่ฝั่ง P4 เจอมาแล้วและแก้ด้วย `ui_set_text()`**
(`ui_theme.h` เขียนไว้ว่าเคยทำให้ task watchdog ทำงาน) — PC จะยืมวิธีเดียวกัน

---

# 2. เทียบ P4 tokens กับที่คุณกำหนดให้ PC

อ่านจาก `hello_world/main/ui/ui_tokens.h`, `ui_theme.h`, `sensor_card.c`

## 2.1 ตรงกันแล้ว ✓

`UI_BG` #F4F7F8 · `UI_SURFACE` #FFFFFF · `UI_SURFACE_ALT` #EDF2F4 · `UI_BORDER` #D9E2E6 ·
`UI_TEXT` #1D2A31 · `UI_TEXT_DIM` #66757D · `UI_ACCENT` #007C83 · `UI_WARN` #B77700 ·
`UI_ERROR` #B3263E

**9 จาก 11 สีหลักตรงกันเป๊ะอยู่แล้ว** — spec ที่คุณเขียนมาสอดคล้องกับ P4 เกือบสมบูรณ์

## 2.2 ไม่ตรงกัน — ต้องตัดสินใจ

| token | P4 | ที่คุณกำหนด | ต่างกันแค่ไหน | ข้อเสนอ |
|---|---|---|---|---|
| OK / live | `UI_OK` **#008B7A** | **#178F7A** | ΔE ≈ 3 — ตาแทบแยกไม่ออก | **ใช้ #178F7A ทั้งสองฝั่ง** แล้วแก้ `UI_OK` ฝั่ง P4 ให้ตรง (P-13) |
| idle | `UI_IDLE` **#9AA7AE** | **#95A2A8** | ΔE ≈ 2 | **ใช้ #95A2A8 ทั้งสองฝั่ง** (P-13) |
| **series 2** | `UI_SERIES_2` **#8A5A00** น้ำตาล | **#3E6E9C** น้ำเงิน | **คนละสีคนละโทน** | ต้องเลือก — ดู 2.3 |
| **series 3** | `UI_SERIES_3` **#4A5FA5** น้ำเงิน | **#8A6F3E** น้ำตาล | **คนละสีคนละโทน** | ต้องเลือก — ดู 2.3 |
| series 4 | ไม่มี | #7A5A87 ม่วง | — | เพิ่มฝั่ง P4 (P-13) |
| soft tints | ไม่มี | 3 ค่า | — | PC มีก่อน แล้วค่อยเพิ่มฝั่ง P4 เมื่อทำ banner บนจอ |
| grid | ไม่มี | #E3EAED | — | PC-only (P4 ไม่มีเส้นตารางกราฟ) |

## 2.3 ⚠️ สีเส้น sensor 2 / 3 สลับกันระหว่างสองเครื่อง

```
sensor 2 :  PC = น้ำเงิน #3E6E9C     P4 = น้ำตาล #8A5A00
sensor 3 :  PC = น้ำตาล #8A6F3E     P4 = น้ำเงิน #4A5FA5
```

นี่คือ **จุดที่ทำลายเป้าหมาย "ธีมเดียวกัน" มากที่สุด** — ผู้ใช้ที่ดูกราฟบนจอแล้วหันมาดู PC
จะจับคู่เส้นผิดทันที

**ทางเลือก:**

| | ก. PC ตาม P4 | ข. P4 ตาม PC | ค. ใช้ชุดใหม่ทั้งคู่ |
|---|---|---|---|
| ต้องแก้ | `lab_theme.py` เท่านั้น | `ui_tokens.h` + build + flash | ทั้งสองฝั่ง |
| แก้ฝั่งที่คุณห้ามแตะ | ไม่ต้อง | ต้อง (เป็น `.md` request) | ต้อง |
| ลำดับสีตามหลักแยกด้วยตาบอดสี | teal → น้ำตาล → น้ำเงิน | teal → น้ำเงิน → น้ำตาล | — |
| กราฟที่มี 2 เส้น (กรณีใช้จริงตอนนี้: #2 #3) | teal-ish + **น้ำตาล/น้ำเงิน** | **น้ำเงิน + น้ำตาล** | — |

**ข้อเสนอ: (ก) — PC ใช้ลำดับของ P4** `S1=#007C83 · S2=#8A5A00 · S3=#4A5FA5 · S4=#7A5A87`

เหตุผล: P4 เป็นของที่ทำงานอยู่แล้ว ทดสอบแล้ว และเป็นฝั่งที่คุณห้ามแก้ในรอบนี้
การให้ฝั่งที่แก้ได้เดินตามฝั่งที่แก้ไม่ได้ คือทางที่ทำให้ทั้งสองตรงกันได้ **ในรอบนี้เลย**
โดยไม่ต้อง flash เฟิร์มแวร์ ส่วนสี #3E6E9C / #8A6F3E ที่คุณเลือกมานั้นสวยกว่าเล็กน้อย
(อ่อนกว่า เข้ากับพื้นสว่างดีกว่า) — ถ้าอยากได้จริง ๆ ให้เข้าคิวเป็น (ข) พร้อมรอบที่แก้ P4 อยู่แล้ว

## 2.4 การจับคู่สถานะ → สี (ต้นฉบับจาก `sensor_card.c:6-17`)

```c
MONITOR_STEADY       → UI_OK
MONITOR_LIVE         → UI_ACCENT
MONITOR_CHANGING     → UI_ACCENT
MONITOR_STALE        → UI_WARN
MONITOR_NO_RESPONSE  → UI_ERROR
MONITOR_SENSOR_FAULT → UI_ERROR
default (OFFLINE)    → UI_IDLE
```

⚠️ สังเกตว่า **P4 ใช้ teal (`UI_ACCENT`) กับ LIVE ไม่ใช่ `UI_OK`** — และ `UI_OK` สงวนไว้ให้
`STEADY` เท่านั้น ต่างจากที่ spec ของคุณเขียนว่า "Live/online: #178F7A"
⇒ **ข้อเสนอ: ยึดตาม P4** เพราะมันแยกความหมายได้ละเอียดกว่า
(teal = "ข้อมูลกำลังไหลอยู่" · เขียว-teal = "และค่านิ่งแล้ว")

---

# 3. ตารางจับคู่สถานะ · สี · ข้อความ (ฉบับสมบูรณ์)

## 3.1 Monitor states (หน้า Overview / การ์ดบน PC)

| สถานะ | token | สี | ค่า EC | ข้อความบรรทัดล่าง | ที่มาฝั่ง PC |
|---|---|---|---|---|---|
| `LIVE` | ACCENT | #007C83 | เข้ม | `Updated now` | มีข้อมูลใหม่ แต่ยังไม่มีหน้าต่างพอตัดสิน |
| `CHANGING` | ACCENT | #007C83 | เข้ม | `Updated Ns ago` | spread ในหน้าต่าง > tolerance |
| `STEADY` | OK | #178F7A | เข้ม | `Updated Ns ago` | spread ≤ tolerance |
| `STALE` | WARN | #B77700 | **หรี่** | `Updated Nm ago` | อายุแถวล่าสุด > 8.75 s |
| `NO RESPONSE` | ERROR | #B3263E | `— — —` | `No reply this cycle` | `ok = 0` |
| `SENSOR FAULT` | ERROR | #B3263E | `— — —` | `No reply for N cycles — check the probe wiring` | `ok = 0` ติดกัน ≥ 3 |
| `DISABLED` | IDLE | #95A2A8 | `— — —` | `Excluded from polling by configuration` | ไม่อยู่ใน mask |
| `OFFLINE` | IDLE | #95A2A8 | `— — —` | `No data yet` | ยังไม่เคยได้ข้อมูลเลย |

## 3.2 Measurement-run states (แผง event)

| สถานะ | token | ความหมายที่ต้องไม่สับสน |
|---|---|---|
| `OBSERVING` | WARN | รอบวัดเริ่มแล้ว กำลังรอค่านิ่ง |
| `STABLE` | OK | **ผ่านเกณฑ์ของ measure policy แล้ว** — ต่างจาก `STEADY` ซึ่งเป็นแค่ค่าสดที่นิ่ง |
| `SAVED` | OK | ผู้ใช้กดยืนยันแล้ว บันทึกลงไฟล์แล้ว |

**กฎการเขียนโค้ด:** `STEADY` และ `STABLE` ต้องไม่มีทางถูกใช้สลับกัน
⇒ แยกเป็นสอง dict คนละตัวใน `lab_theme.py` (`MONITOR_STATES` / `RUN_STATES`)
ไม่ใช่ dict เดียวรวมกัน

## 3.3 สีที่ **ห้าม** ใช้เป็นเส้นกราฟปกติ

`WARN` #B77700 และ `ERROR` #B3263E สงวนไว้สำหรับ **เหตุการณ์บนกราฟ** เท่านั้น
(marker ช่วง CAL, ช่วง fault) — เส้นข้อมูลปกติใช้เฉพาะชุด `SERIES` 4 สี

## 3.4 ความสอดคล้องของถ้อยคำที่ต้องคัดลอกมาตรง ๆ

| เรื่อง | ต้นฉบับ P4 | PC ต้องใช้แบบเดียวกัน |
|---|---|---|
| freshness | `format_age()` ใน `sensor_card.c:41` | `No data yet` · `Updated now` (<1.5 s) · `Updated Ns ago` (<60 s) · `Updated Nm ago` |
| ชื่อการ์ด | `snprintf("SENSOR %02d")` | `SENSOR 01` … `SENSOR 04` (**ไม่ใช่** `EC ภาชนะ #1`) |
| สรุปบน header | `"%u SENSORS • %u STEADY • %u LIVE • %u FAULT • %u OFFLINE"` | ใช้สูตรเดียวกันเป๊ะ + เพิ่ม `• %u DISABLED` |
| ค่าที่อ่านไม่ได้ | `"- - -"` | `— — —` |

---

# 4. แผนไฟล์

## 4.1 ไฟล์ใหม่

### `lab_theme.py` — token กลาง (ประมาณ 220 บรรทัด ไม่มี logic)

```python
"""
lab_theme.py — design tokens ของ PC dashboard
ทุกสี ทุกระยะ ทุกฟอนต์ต้องมาจากไฟล์นี้ที่เดียว
คู่แฝดของ hello_world/main/ui/ui_tokens.h — ถ้าแก้ที่นี่ ต้องเช็คอีกฝั่งด้วย
"""

# ---- สี (ตรงกับ ui_tokens.h) ----
BG, SURFACE, SURFACE_ALT, BORDER   = "#F4F7F8", "#FFFFFF", "#EDF2F4", "#D9E2E6"
TEXT, TEXT_DIM                     = "#1D2A31", "#66757D"
ACCENT, OK, WARN, ERROR, IDLE      = "#007C83", "#178F7A", "#B77700", "#B3263E", "#95A2A8"
ACCENT_SOFT, WARN_SOFT, ERROR_SOFT = "#DCEFED", "#F7ECD3", "#F8E1E6"
GRID, REC                          = "#E3EAED", "#B3263E"
SERIES = ["#007C83", "#8A5A00", "#4A5FA5", "#7A5A87"]   # ← ตาม ui_tokens.h

# ---- ระยะ 8 px grid (เหมือน P4) ----
SP_1, SP_2, SP_3, SP_4 = 8, 16, 24, 32
RADIUS, BORDER_W       = 10, 1

# ---- ฟอนต์: เลือกตาม OS จริง ----
def font_family() -> str: ...          # Segoe UI → Noto Sans Thai → DejaVu Sans → TkDefaultFont
def mono_family() -> str: ...          # Consolas → DejaVu Sans Mono
FONT_VALUE, FONT_VALUE_XL = 34, 56
FONT_TITLE, FONT_SECTION  = 19, 16
FONT_BODY, FONT_LABEL     = 14, 12

# ---- สถานะ ----
MONITOR_STATES = {...}   # 8 รายการตามตาราง 3.1
RUN_STATES     = {...}   # 3 รายการตามตาราง 3.2

# ---- helper ที่ทุกไฟล์ต้องเรียกผ่าน ----
def status_style(state: str) -> dict          # {"colour","label","dim_value","dot"}
def format_ec(v, decimals=1) -> str           # "1,362.4" / "84.6" / "— — —"
def format_freshness(age_s) -> str            # ถ้อยคำเดียวกับ format_age() ของ P4
def sensor_card_style(state, enabled) -> dict # {"bg","border","value_fg","state_fg"}
def apply_ttk_theme(root) -> None             # ttk.Style: clam + Combobox/Scrollbar/Notebook
def apply_mpl_theme() -> None                 # rcParams ของ matplotlib
def grid_for(n_active) -> tuple               # (rows, cols) จากจำนวน sensor
```

### `ec_ui_config.json` — ค่าตั้งของ UI (ไม่ใช่ config ของฮาร์ดแวร์)

```json
{
  "active_mask": 6,
  "sensor_names": ["SENSOR 01", "SENSOR 02", "SENSOR 03", "SENSOR 04"],
  "window": [1100, 820],
  "chart_mode": "split"
}
```

> **นี่คือคำตอบทันทีของ "หัววัด #1 ปิดไปเฉย ๆ ก่อน"**
> `active_mask = 6` (`0b0110`) ทำให้การ์ด #1 ขึ้น `DISABLED` สีเทา แทน `ERROR` สีแดง
> และหายจากกราฟ **โดยไม่แตะ logger, CSV, Modbus หรือ firmware แม้แต่บรรทัดเดียว**
> — CONTROL ยังคง poll ครบ 3 ตัวเหมือนเดิม (จำเป็นด้วย เพราะการตัดรอบของจอ P4 ยังผูกกับ
> การเห็น address 1 — ดู R1 ในเอกสาร Phase 1) นี่เป็นการซ่อนเชิงการแสดงผลล้วน ๆ
> ⇒ ปลอดภัย 100% และย้อนได้ด้วยการแก้ตัวเลขเดียว

## 4.2 ไฟล์ที่แก้

| ไฟล์ | แก้อะไร | **ไม่แตะ** |
|---|---|---|
| `desktop_ui.py` | ทุกส่วนที่เป็นการแสดงผล: `_build_*`, `refresh_*`, `_hl`, สี/ฟอนต์ทั้ง 53 จุด | `read_range()` · `read_recent()` · `downsample()` · `load_sessions()` · `export_csv/excel/pdf()` · `make_mock_csv()` — **ฟังก์ชันข้อมูลทั้งหมดคงเดิมทุกบรรทัด** |
| `logger_3ec.py` | **แก้จุดเดียว:** บรรทัดสรุปในคอนโซล (บรรทัด 472-475) ให้ใช้คำ `STEADY/LIVE/…` ชุดเดียวกัน และเคารพ `active_mask` ตอนพิมพ์ | serial · CSV · session · calibration · report — ไม่แตะทั้งหมด |

> การแก้ `logger_3ec.py` เป็น **ทางเลือก** ถ้าอยากให้รอบนี้แตะไฟล์เดียวจริง ๆ ตัดออกได้
> โดยไม่กระทบอะไรเลย

## 4.3 ไฟล์ที่ **ไม่แตะเด็ดขาด** ในรอบนี้

`report_3ec.py` · `calibration.py` · `console_utf8.py` · `water_monitor_3ec.ino` ·
ทุกไฟล์ใน `hello_world/`

---

# 5. Before / After wireframe

## 5.1 ก่อน (ตามภาพหน้าจอที่ส่งมา)

```
┌───────────────────────────────────────────────────────────────┐ พื้น #0f1720
│ ● ESP32 Water Monitor — EC 3 ภาชนะ   live • 00:20:05 (4s ago) │
├───────────────┬───────────────┬───────────────────────────────┤
│ EC ภาชนะ #1   │ EC ภาชนะ #2   │ EC ภาชนะ #3                   │  การ์ด #1a2430
│  --           │  84.6         │  85.4                         │  มีแถบสีบนหัวการ์ด
│  µS/cm        │  µS/cm        │  µS/cm                        │
│  T -- °C      │  T 20.4 °C    │  T 20.5 °C                    │
│  ● ERROR ←──── แดง ทั้งที่ตั้งใจถอดออก                          │
├───────────────┴───────────────┴───────────────────────────────┤
│ ช่วง: [10 min][1 hr][6 hr][24 hr][All]      ⬇PDF ⬇Excel ⬇CSV  │  ปุ่มเดี่ยวพื้นเข้ม
│ บันทึก (คุมจาก terminal): #1 ○  #2 ●REC  #3 ●REC | ดู session:▼│  Combobox ขาวโดด
├───────────────────────────────────────────────────────────────┤
│  กราฟ 3 แผง พื้น #1a2430 เส้น Material blue/green/orange       │
│  แผง #1 ว่างเปล่า แกน -0.050 … 0.050  ← ไม่มีข้อมูลแต่ยังกินที่  │
│  [NavigationToolbar สีเทาสว่างของ Tk]                          │
└───────────────────────────────────────────────────────────────┘
```

**ปัญหาที่เห็นในภาพ:** แผงกราฟ #1 ว่างเปล่ากินพื้นที่ 1 ใน 3 · การ์ด #1 แดงตลอดเวลา ·
Combobox ขาวโดด · toolbar เทาสว่าง · แถบสีบนหัวการ์ดเป็นการตกแต่งที่ไม่สื่อสถานะ

## 5.2 หลัง

```
┌───────────────────────────────────────────────────────────────────┐ พื้น #F4F7F8
│ ● EC MEASUREMENT STATION  3 SENSORS • 2 STEADY • 1 DISABLED       │ แถบขาว
│                        Sample CALF-20 B3 · Updated now · [Settings]│ ขอบล่าง 1px
├───────────────────────────────────────────────────────────────────┤
│ ┌─ SENSOR 01 ─────┐ ┌─ SENSOR 02 ─────┐ ┌─ SENSOR 03 ─────┐       │ การ์ดขาว
│ │ ● DISABLED  เทา │ │ ● STEADY        │ │ ● STEADY        │       │ ขอบ 1px #D9E2E6
│ │  — — —          │ │  84.6           │ │  85.4           │       │ ไม่มีเงา
│ │  uS/cm          │ │  uS/cm  20.4 °C │ │  uS/cm  20.5 °C │       │ ไม่มีแถบสีตกแต่ง
│ │ ─────────────── │ │ ─────────────── │ │ ─────────────── │       │
│ │ Excluded from   │ │ Updated now     │ │ Updated 2s ago  │       │ พื้นการ์ด #1
│ │ polling by cfg  │ │                 │ │                 │       │ = #EDF2F4
│ └─────────────────┘ └─────────────────┘ └─────────────────┘       │
├───────────────────────────────────────────────────────────────────┤
│ Range [10 min│1 hr│6 hr│24 hr│All]  [Split│Overlay]                │ segmented
│                              Export CSV  Export Excel  [Export PDF]│ teal = หลัก
│ Recording  ⊘#1 DISABLED  ●#2 REC  ●#3 REC   Session [All data ▼]  │
├───────────────────────────────────────────────────────────────────┤
│ EC vs TIME — SPLIT                                                │ พื้นขาว
│  แผงของ sensor ที่ enabled เท่านั้น (ไม่มีแผงว่าง)                  │ grid #E3EAED
│  แกน/tick #66757D · spine #D9E2E6 · เส้นตามชุด SERIES             │ spine บาง
│  [toolbar สีเข้ากับธีม]                                            │
├───────────────────────────────────────────────────────────────────┤
│ EVENT LOG                                                         │
│ 00:20:04 ● SAVED · Sensor 02 · 84.6 uS/cm · stable 15s            │ ไม่ใช่ terminal
│ 00:11:03 ● CALIBRATION · Sensor 03 @ 84 uS/cm · from touchscreen  │ เรียงเวลาล่าสุดบน
└───────────────────────────────────────────────────────────────────┘
```

## 5.3 แถบสถานะที่โผล่เฉพาะเมื่อเข้าเงื่อนไข

| แถบ | เงื่อนไข | สี |
|---|---|---|
| `HISTORY VIEW — LIVE FOLLOW PAUSED` + ปุ่ม `Return to live` | ผู้ใช้เลือก session หรือ pan/zoom ออกจากขวาสุด | WARN_SOFT |
| `CALIBRATING SENSOR 0N @ X uS/cm` + เวลาที่ผ่านไป | อ่านจาก `cal_status.json` (Phase 3b) | ACCENT_SOFT |
| `SENSOR BOARD OFFLINE` | ไม่มีแถวใหม่ > 20 วินาที | ERROR_SOFT |
| `MOCK DATA` | โหมดทดสอบเปิดอยู่ | WARN_SOFT |

**กฎ:** แถบแสดงทีละอันตามลำดับความสำคัญ `OFFLINE > CALIBRATING > MOCK > HISTORY`
ไม่ซ้อนกันหลายแถบจนดันเนื้อหาลง

---

# 6. ผัง 1 / 2 / 3 / 4 sensors

`grid_for(n)` คืนค่าจากจำนวน sensor ที่ **enabled** (ไม่ใช่จำนวนทั้งหมด)

| n | การ์ด | ขนาดค่า EC | กราฟ (split) | หมายเหตุ |
|---|---|---|---|---|
| **1** | 1 การ์ดเต็มความกว้าง — ค่าชิดขวา ชื่อ/สถานะชิดซ้าย | **56 px** | 1 แผงเต็มพื้นที่ | ใช้พื้นที่ที่เหลือไปกับกราฟทั้งหมด |
| **2** | 2 คอลัมน์ `grid(row=0, column=0..1)` | 40 px | 2 แผงแชร์แกน x | |
| **3** | 3 คอลัมน์ — **ผังเดิมทุกประการ** | 34 px | 3 แผง — เดิม | ต้องผ่าน screenshot diff |
| **4** | **2×2** | 34 px | 4 แผง + โหมด `Overlay` | ดู 6.2 |

## 6.1 กลไก

```python
active = [i for i in range(MAX_SENSORS) if mask & (1 << i)]   # เช่น [1, 2]
rows, cols = grid_for(len(active))                            # (1, 2)
```

- ใช้ `grid()` ไม่ใช่ `pack(side="left")` — `pack` แบบเดิมเปลี่ยนจำนวนคอลัมน์ไม่ได้
- **สร้างการ์ดใหม่เฉพาะตอน mask เปลี่ยน** ไม่ใช่ทุกรอบ refresh
- `mask` ที่เปลี่ยนกลางทางจะ `destroy()` เฟรมเดิมแล้วสร้างใหม่ครั้งเดียว

## 6.2 กรณี 4 sensors — ทำไมต้องมีโหมด Overlay

พื้นที่กราฟที่เหลือหลังหักการ์ด 2×2 คือประมาณ 300 px
หาร 4 แผงได้แผงละ **75 px** ซึ่งอ่านไม่ออก

⇒ เมื่อ `n == 4` ให้ default เป็น **Overlay** (4 เส้นบนแกนเดียว สูงเต็ม 300 px)
และมีปุ่ม `Split` ให้สลับ — ตอนนั้นใช้ `Notebook` 2 แท็บแทนการ scroll

**ทำไมไม่ scroll:** `NavigationToolbar2Tk` (zoom/pan ที่ใช้อยู่) จะเพี้ยนเมื่อ canvas
ถูกห่อใน scrollable frame — แท็บให้ผลเดียวกันโดยไม่แตะ toolbar

## 6.3 sensor ที่ปิดไว้

- **ไม่สร้าง subplot** — ไม่มีแผงว่างเหมือนในภาพหน้าจอปัจจุบัน
- **ไม่ plot เป็น 0 และไม่ plot เป็น error**
- การ์ดยังแสดงอยู่ (พื้น `SURFACE_ALT` ข้อความ `IDLE`) เพื่อให้ผู้ใช้รู้ว่ามีอยู่แต่ปิดไว้
  — ถ้าซ่อนหายไปเลย ผู้ใช้จะไม่รู้ว่าทำไมมีแค่ 2 การ์ด

---

# 7. Chart styling

## 7.1 `rcParams` ที่ตั้ง (ใน `apply_mpl_theme()`)

| key | ค่า |
|---|---|
| `figure.facecolor` / `savefig.facecolor` | `SURFACE` |
| `axes.facecolor` | `SURFACE` |
| `axes.edgecolor` | `BORDER` |
| `axes.linewidth` | `0.8` |
| `axes.labelcolor` / `xtick.color` / `ytick.color` | `TEXT_DIM` |
| `axes.grid` / `grid.color` / `grid.linewidth` / `grid.alpha` | `True` / `GRID` / `0.8` / `1.0` |
| `axes.spines.top` / `.right` | `False` — เหลือ 2 ด้าน อ่านง่ายกว่า |
| `legend.frameon` / `legend.facecolor` / `legend.edgecolor` | `True` / `SURFACE` / `BORDER` |
| `font.family` / `font.size` | จาก `font_family()` / `11` |
| `lines.linewidth` | `1.5` (เส้นที่เลือก `2.2`) |
| `date.autoformatter.minute` | `%H:%M` |

## 7.2 แก้ U1 — กราฟกะพริบ

**ตอนนี้:**
```python
for i, ax in enumerate(self.axes):
    ax.clear()            # ← ทำลาย artist ทั้งหมด + ล้าง zoom ที่ผู้ใช้ตั้งไว้
    ax.plot(ts, ys, ...)
self.canvas.draw()        # ← วาดใหม่ทั้งแผ่นแบบ blocking
```

**ที่เสนอ:**
```python
# สร้าง Line2D ครั้งเดียวตอนสร้างแกน
self.lines[i].set_data(ts, ys)          # เปลี่ยนแค่ข้อมูล
ax.relim(); ax.autoscale_view(scaley=True, scalex=not self.user_zoomed)
self.canvas.draw_idle()                 # เข้าคิว ไม่บล็อก
```

ผลที่ได้:
- ไม่กะพริบ
- **zoom/pan ที่ผู้ใช้ตั้งไว้ไม่หาย** (ตรวจด้วย `ax.get_navigate_mode()` / callback `xlim_changed`)
- เมื่อผู้ใช้ pan ออกจากขอบขวา → ตั้ง `user_zoomed = True` → ขึ้นแถบ
  `HISTORY VIEW — LIVE FOLLOW PAUSED` + ปุ่ม `Return to live` (ซึ่งตั้งกลับเป็น `False`)
- สร้าง/ทำลาย subplot เฉพาะตอน `mask` หรือ `chart_mode` เปลี่ยน

## 7.3 แก้ U3 — เขียน label เฉพาะตอนค่าเปลี่ยน

ยืมวิธีของ `ui_set_text()` ฝั่ง P4 ตรง ๆ:
```python
def set_text(widget, txt):
    if widget._last_text != txt:
        widget.config(text=txt)
        widget._last_text = txt
```

## 7.4 downsampling

ใช้ `downsample()` เดิมทั้งหมด ไม่แตะ — แต่ย้ายป้าย `(ย่อ)` / `(raw)`
ไปอยู่ที่มุมขวาล่างของกราฟในรูปแบบ `1,842 points (downsampled)`

## 7.5 event marker

**ไม่ทำในรอบนี้** — แต่จองที่ไว้: `apply_mpl_theme()` นิยาม `MARKER_STYLES` ให้พร้อม
(`cal` = แถบ `WARN_SOFT` แนวตั้ง, `saved` = จุด `OK`, `fault` = แถบ `ERROR_SOFT`)
เพื่อให้ Phase 3b เสียบเข้ามาได้โดยไม่ต้องแก้ styling อีก

---

# 8. ข้อจำกัดของ Tkinter ที่ต้องยอมรับ (ตรงไปตรงมา)

| spec ที่คุณตั้งไว้ | ทำได้ไหมใน Tk | ทางออก |
|---|---|---|
| ขอบ 1 px บาง | ✅ | `highlightthickness=1, highlightbackground=BORDER, bd=0` |
| ไม่มีเงา | ✅ | `relief="flat"` ทุกที่ |
| **มุมโค้ง 8-12 px** | ❌ **Tk ไม่รองรับ radius** | ดู 8.1 |
| segmented control | ✅ | `Frame` + `Button` ชิดกัน `relief=flat` เลือก = พื้น teal |
| ปุ่ม teal ทึบ | ✅ | `bg=ACCENT, fg="white", activebackground=` เฉดเข้มขึ้น |
| tabular figures | ⚠️ บางส่วน | Segoe UI ไม่มี tnum ผ่าน Tk → ใช้ `Consolas` เฉพาะค่าตัวเลข |
| Combobox ตามธีม | ✅ | `ttk.Style(theme="clam")` + `map()` — **ต้องใช้ `clam` เท่านั้น** ธีม `vista`/`xpnative` บน Windows ไม่ยอมให้เปลี่ยนสี |
| toolbar ตามธีม | ✅ | ไล่ `toolbar.winfo_children()` แล้วตั้ง `bg` (เป็น Tk widget ธรรมดา) |
| ฟอนต์ไทยในกราฟ | ❌ ตามที่โค้ดเดิมเขียนไว้ | **กราฟใช้ภาษาอังกฤษล้วน** — คงกติกาเดิมของโปรเจกต์ |

## 8.1 เรื่องมุมโค้ง — ข้อเสนอ

| | ก. ขอบเหลี่ยม 1 px | ข. วาดการ์ดบน Canvas |
|---|---|---|
| ตรงกับ spec 8-12 px | ❌ | ✅ |
| ตรงกับหน้าตา P4 | ⚠️ P4 มี `UI_RADIUS 12` | ✅ |
| ความซับซ้อน | 0 | สูง — ต้องวาง label เป็น canvas text item เอง จัดตำแหน่งเอง ฟอนต์ไทยเพี้ยนง่าย |
| ความเสี่ยงตอน resize | 0 | ต้องวาดใหม่ทุกครั้ง |

**ข้อเสนอ: (ก) ขอบเหลี่ยม 1 px**

เหตุผล: หน้าตาที่ได้ยังอ่านเป็น "เครื่องมือห้องแล็บ" ครบทุกอย่าง — สิ่งที่สร้างความรู้สึกนั้นคือ
**พื้นสว่าง + ขอบบาง + ไม่มีเงา + สีมีความหมาย** ไม่ใช่มุมโค้ง
ส่วน (ข) แลกความเสี่ยงเรื่องการวางข้อความและฟอนต์ไทยกับสิ่งที่มองแทบไม่เห็นที่ 10 px
ถ้าคุณอยากได้มุมโค้งจริง ๆ ทำเฉพาะการ์ด sensor ได้ (เป็น label ล้วน ไม่มี widget ซ้อน)
แต่ขอแยกเป็นรอบต่างหากหลังจากธีมหลักนิ่งแล้ว

---

# 9. แผนตรวจสอบหลังทำเสร็จ

| # | ตรวจอะไร | วิธี | เกณฑ์ผ่าน |
|---|---|---|---|
| V1 | ฟังก์ชันข้อมูลไม่เปลี่ยน | `git diff` เฉพาะ `read_range` `read_recent` `downsample` `export_*` `load_sessions` `make_mock_csv` | **diff ว่างเปล่า** |
| V2 | อ่าน CSV เก่าได้เท่าเดิม | รัน `read_range()` บนไฟล์ทั้ง 6 ก่อน/หลัง | จำนวนแถวและค่าตรงกันทุกตัว |
| V3 | Export ยังทำงาน | กด CSV / Excel / PDF | ได้ไฟล์ · PDF ยังเป็น 9 หน้าจาก `report_3ec` |
| V4 | Session dropdown | เลือก session เดิม | ช่วงเวลาตรงกับ `sessions_3ec.json` |
| V5 | mask 0b0110 | เปิดโปรแกรม | #1 = `DISABLED` เทา · ไม่มีแผงกราฟของ #1 · #2 #3 ปกติ |
| V6 | mask 0b0111 | ตั้งกลับ | ผังเหมือนเดิม 3 คอลัมน์ |
| V7 | mask 1 / 2 / 4 ตัว | แก้ `ec_ui_config.json` | ผังตามตาราง 6 |
| V8 | ไม่กะพริบ | ปล่อยไว้ 10 นาที + zoom ค้างไว้ | zoom ไม่หาย · ไม่มีการกะพริบ |
| V9 | Windows | รันบนเครื่องจริง | ฟอนต์ Segoe UI · Combobox ตามธีม · toolbar ตามธีม |
| V10 | Linux | รันบน Ubuntu | ฟอนต์ fallback ทำงาน · ไม่มี exception |
| V11 | ไม่มีสีฝังในโค้ด | `grep -n '"#[0-9A-Fa-f]\{6\}"' desktop_ui.py` | **ไม่เจอเลย** |
| V12 | ไม่มีข้อความเล็กกว่า 12 px | ตรวจ `lab_theme.py` | ผ่าน |
| V13 | screenshot | ก่อน/หลัง 3 sensors | แนบในรายงานส่งมอบ |

**Rollback:** `git checkout desktop_ui.py` (หลัง `git init` ตาม Q9) หรือสำเนา
`desktop_ui.py.bak` — ไฟล์ใหม่ทั้งหมดเป็นไฟล์เพิ่ม ลบทิ้งได้โดยไม่กระทบอะไร

---

# 10. คำถามที่ต้องตัดสินใจก่อนลงมือ

| # | คำถาม | **ที่แนะนำ** |
|---|---|---|
| **U-Q1** | สีเส้นกราฟ sensor 2/3 ที่สลับกัน (หัวข้อ 2.3) | **PC เดินตาม P4** — `#8A5A00` สำหรับ #2, `#4A5FA5` สำหรับ #3 ⇒ ตรงกันได้ในรอบนี้เลยโดยไม่ต้อง flash |
| **U-Q2** | ภาษาบน UI | **คำสถานะและชื่อ sensor เป็นอังกฤษทั้งหมด** (ต้องตรงกับจอ) · ข้อความช่วยเหลือ/ปุ่ม export เป็นไทยได้ · **กราฟอังกฤษล้วน** ตามกติกาเดิม |
| **U-Q3** | ค่า EC บนการ์ด | **1 ตำแหน่ง `84.6`** — เพราะ CSV/Excel/PDF ทั้งหมดใช้ 1 ตำแหน่ง ถ้าปัดเป็นจำนวนเต็มแบบ P4 (`85`) ตัวเลขบนจอ PC จะไม่ตรงกับตัวเลขในรายงานของตัวเอง ซึ่งแย่กว่าการต่างจาก P4 · แต่ยืมรูปแบบ `1,362.4` และ `— — —` มาใช้ |
| **U-Q4** | มุมโค้ง (หัวข้อ 8.1) | **ขอบเหลี่ยม 1 px** |
| **U-Q5** | `active_mask` ตั้งที่ไหน | **`ec_ui_config.json` (display-only)** ⇒ ปิด #1 ได้ทันทีโดยไม่แตะ data path · Phase 4 ค่อยย้ายไปเป็น mask จริงจาก CONTROL |
| **U-Q6** | แผง Event log | **ใส่เลย** แต่ Phase นี้แสดงเฉพาะสิ่งที่ PC รู้อยู่แล้ว (session start/stop, export, mock, board offline) · เหตุการณ์จากจอ (SAVED/STABLE/CAL) จะเสียบเข้ามาตอน Phase 3b |
| **U-Q7** | แก้ `logger_3ec.py` ด้วยไหม | **แก้แค่บรรทัดสรุปในคอนโซล** ให้ใช้คำชุดเดียวกัน · ถ้าอยากให้รอบนี้แตะไฟล์เดียว ตัดออกได้ |
| **U-Q8** | ขนาดหน้าต่าง 1100×820 | **ได้** (คุณตอบไว้แล้วว่าไม่ใช่จอสัมผัส ⇒ ปุ่มใช้ 32 px สูงพอ ไม่ต้อง 44 px) |
| **U-Q9** | ให้ผมส่ง `.md` ขอแก้ฝั่ง P4 (P-13: `UI_OK`, `UI_IDLE`, `UI_SERIES_4`, soft tints) ด้วยเลยไหม | **ส่ง** — เป็นการแก้ 4 บรรทัดใน `ui_tokens.h` ทำตอนไหนก็ได้ ไม่เร่ง |

---

**No code has been changed. Waiting for approval.**
