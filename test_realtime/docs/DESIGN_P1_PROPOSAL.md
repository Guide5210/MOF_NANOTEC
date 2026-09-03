# Phase 0 Audit + Phase 1 Design Proposal
## ESP32 Water Monitor — P4 Command Console + Configurable 1–4 Sensors

เอกสารนี้เป็นผลของ **Phase 0 (audit เท่านั้น)** และ **Phase 1 (design proposal เท่านั้น)**
ยังไม่มีการแก้โค้ดใด ๆ ทั้งสิ้น

- วันที่: 25 สิงหาคม 2026
- ขอบเขตที่ผู้เขียนรับผิดชอบลงมือ: **ฝั่ง PC เท่านั้น** (`C:\MOF_NanoTec\test_realtime`)
- ฝั่งจอ P4 (`C:\Users\srnan\Documents\Arduino\ESP-IDF`): **อ่านอย่างเดียว** — ข้อเสนอทั้งหมด
  ที่กระทบ P4 อยู่ในหัวข้อ 11.4 และจะส่งมอบเป็นไฟล์ `.md` แยก
- ฝั่ง CONTROL V1.2 (`.ino`): มีสำเนาอยู่ใน `test_realtime/` แต่ **ยังไม่แตะ** จนกว่าจะอนุมัติ

---

# 1. ความเข้าใจระบบปัจจุบัน

## 1.1 โครงสร้างที่เห็นจริงจากโค้ด

```
                     RS485 4800 8N1 Modbus RTU
   ┌──────────────┐  ◄─────────────────────────►  ┌────────────────────┐
   │ ESP32-P4 จอ  │   0x01-0x03 = SEN0706          │ CONTROL V1.2       │
   │ ESP-IDF 6.0.2│   0x20 บอร์ด→จอ  (เวลา DS3231)  │ Arduino            │
   │ LVGL 9       │   0x21 จอ→บอร์ด  (คำสั่ง)       │ master ตัวเดียว     │
   │ ฟังบัสเงียบ ๆ │   0x22 บอร์ด→จอ  (ACK)         │ POLL 2500 ms       │
   └──────┬───────┘                                └─────────┬──────────┘
          │ USB CH343 (1A86:55D3)                            │ USB CH340
          │ = console + ESP_LOG + คำสั่ง S/B/V/C/D/K          │ 115200
          │ ปัจจุบัน "ไม่มี" ใครฝั่ง PC เปิดพอร์ตนี้            │
          ▼                                                  ▼
        (ว่าง)                                     logger_3ec.py  ← เจ้าของพอร์ตเดียว
                                                              │
                                                   water_data/*.csv
                                                   sessions_3ec.json
                                                   rec_status.json
                                                   calibration_log.json
                                                              │
                                          desktop_ui.py / report_3ec.py (อ่านไฟล์อย่างเดียว)
```

## 1.2 สิ่งที่ **มีอยู่แล้ว** และเปลี่ยนสมมติฐานของโจทย์อย่างมาก

นี่คือหัวใจของ audit รอบนี้ — โจทย์ตั้งอยู่บนสมมติฐานว่า "P4 ยังสั่งอะไรไม่ได้เลย"
แต่โค้ดจริงบอกตรงกันข้าม:

| สิ่งที่มีอยู่แล้ว | หลักฐาน |
|---|---|
| **P4 สั่ง calibrate ผ่าน RS485 ได้แล้ว และทดสอบผ่านแล้ว** | `ec_rs485.h:56 ec_rs485_calibrate()`, `.ino:520 handleBusCommand() case CMD_CAL`, `docs/SYSTEM.md §7 "คาลิเบรตผ่านบัส หัววัด #3 ที่ 1413 → Done"` |
| **มีช่องคำสั่ง P4→CONTROL ครบวงจร พร้อม state machine + retry + ACK** | `ec_rs485.h:44-54` `EC_CMD_IDLE/SENDING/WAIT_ACK/WORKING/DONE/FAILED/TIMEOUT` |
| **CONTROL พิมพ์ผลการ cal ออก COM-A อยู่แล้วทุกครั้ง** | `.ino:534 "[bus] จอสั่งคาลิเบรต EC#%u ที่ %u uS/cm"`, `.ino:452 "[cal] ok n=%d std=%.0f"`, `.ino:454 "[cal] fail n=%d rc=0x%02X"` |
| **`logger_3ec.py` เห็นบรรทัดพวกนี้แล้ว แต่ทิ้งทันที** | `logger_3ec.py:237-248 parse()` — `if not line.startswith("DATA,"): return None` |
| **measure event ที่โจทย์ระบุ มีครบแล้วบน P4** | `measure_policy.h:109-119` `MEASURE_EV_RUN_STARTED … READING_SAVED` ครบ 10 ตัวตรงกับที่ขอ |
| **P4 มี `EC_SENSOR_COUNT` เป็นจุดเดียว** | `ec_packet.h:15`, `ui_model.h:25 #define UI_SENSOR_COUNT EC_SENSOR_COUNT` |
| **`logger_3ec.py` กัน CH343 ไว้แล้ว** | `logger_3ec.py:201-202` — `if "ch343" in blob: return -1` (จอ P4 ไม่ถูกเลือกเป็น CONTROL) |
| **มี rollback switch ฝั่ง P4 อยู่แล้ว** | `CONFIG_EC_UI_V2` — UI เก่า `ec_ui.c` ยังคอมไพล์อยู่ สลับได้จาก menuconfig |

**ผลที่ตามมาต่อการออกแบบ:** เป้าหมาย "กด cal บนจอ แล้ว PC ขึ้น cal ด้วย"
**ไม่ต้องใช้ transport ใหม่เลย** — ดูหัวข้อ 4.1

---

# 2. ความเสี่ยงและข้อจำกัดที่ต่อรองไม่ได้

## 2.1 ความเสี่ยงระดับ "ทำแล้วพังแน่ถ้าไม่รู้"

| # | ความเสี่ยง | หลักฐาน | ผลถ้าพลาด |
|---|---|---|---|
| **R1** | **ปิด sensor #1 จะทำให้จอ P4 ค้าง STALE ทั้งเครื่อง** | `docs/BUS_PROTOCOL.md §1` — จอใช้ "เห็น `01 03 0000` อีกครั้ง = ขึ้นรอบใหม่" เป็นตัวตัดรอบ (`on_request()` ใน `ec_rs485.c`) | mask `0b0110` (ซึ่งคือกรณีใช้งานจริงข้อแรก เพราะหัววัด #1 เสีย) ทำให้จอไม่ปิดรอบ ข้อมูลค้าง ขึ้น STALE ทั้งที่ #2 #3 อ่านได้ปกติ |
| **R2** | **schema CSV ชนกันเงียบ ๆ อยู่แล้ววันนี้** | `water_log_2026-08-05.csv` และ `2026-08-07.csv` มี header ของ `logger.py` (7 คอลัมน์ `EC_uScm,Tw_C,Salinity_ppm,TDS_ppm,pH,pH_mV,rs485_ok`) แต่ข้อมูลข้างในเป็นของ 3EC (`0.0,19.8,0.0,19.8,0.0,20.5,111`) | `logger.py` (เก่า, 1 เซนเซอร์) กับ `logger_3ec.py` เขียน **ชื่อไฟล์รูปแบบเดียวกัน** `water_log_YYYY-MM-DD.csv` ในโฟลเดอร์เดียวกัน → ข้อมูลปนแล้วแยกไม่ออก |
| **R3** | **ทุก reader ฝั่ง PC อ่านคอลัมน์ด้วยตำแหน่ง ไม่ใช่ชื่อ** | `logger_3ec.py:241`, `desktop_ui.py:73-91`, `report_3ec.py:644-660` ใช้ `p[1] p[3] p[5]` + `if len(p) < 10: continue` | CSV v2 ที่มี 15 คอลัมน์จะ **ผ่านเงื่อนไข `len(p) >= 10`** แล้วถูกอ่านผิดตำแหน่งอย่างเงียบสนิท → กราฟและรายงานผิดโดยไม่มี error |
| **R4** | **heap ของ P4 ยังมีบั๊ก PANIC ระหว่าง soak ที่ยังหาสาเหตุไม่เจอ** | `docs/SYSTEM.md §8` — `Guru Meditation Error: Load access fault` ที่นาทีที่ ~38, heap ลด 44 KB | เพิ่ม JSON parser ที่จองฮีปทุกเฟรมตอนนี้ = เพิ่มตัวแปรให้บั๊กที่ยังหาไม่เจอ |
| **R5** | **ESP-IDF v6.0.2 ไม่มี component `json`/cJSON ใน core แล้ว** | `ls v6.0.2/esp-idf/components` → ไม่มี `json`; `hello_world/idf_component.yml` ไม่มี cjson | สมมติว่า "P4 น่าจะมี JSON lib อยู่แล้ว" ผิด — ต้องเพิ่ม managed component หรือเขียนเอง |
| **R6** | **`calibration.Calibrator.run()` เป็น blocking workflow ที่ผูกกับคีย์บอร์ด** | `calibration.py:565` `run(sensor, standard, ask=...)` เรียก `wait_stable()` ที่วนรอถึง 120 วิ; `logger_3ec.py:377-414 do_calibrate()` ใช้ `keys.line()` | cal ที่สั่งจาก P4 จะไปเรียกฟังก์ชันนี้ไม่ได้ตรง ๆ ถ้าไม่แยก presenter ออกมาก่อน |
| **R7** | **`ec_packet_t` มี `_Static_assert(sizeof == 41)` ผูกกับ 3 เซนเซอร์ และต้องตรงกับ `.ino` ทุกไบต์** | `ec_packet.h:28` | เปลี่ยน `EC_SENSOR_COUNT` เป็น 4 = build fail ทันที (ดีแล้ว — แต่ต้องแก้สองฝั่งพร้อมกัน) |
| **R8** | **หัววัด #1 เสียจริง และวงจรขับ RS485 ที่เสียลากบัสตายทั้งเส้น** | `docs/SYSTEM.md §1` | "disabled" กับ "fault" ต้องแยกกันจริง ๆ ไม่ใช่แค่เรื่องความสวยของ UI |

## 2.2 ข้อจำกัดที่ต่อรองไม่ได้ (คงไว้ทุกข้อ)

1. CONTROL V1.2 เป็น Modbus master ตัวเดียว — P4 ห้ามเป็น master ใน normal operation
2. `logger_3ec.py` เป็น process เดียวที่เปิด COM-A
3. `desktop_ui.py` / `report_3ec.py` ห้ามเปิด serial ใด ๆ
4. Modbus task ห้ามเรียก LVGL API / UI อ่านผ่าน `ui_model` เท่านั้น
5. raw CSV ฝั่ง PC ต้องไม่ขาดช่วงเพราะเหตุใดก็ตาม (P4 หลุด, report กำลังสร้าง, cal กำลังทำ)
6. **เพิ่มใหม่ (จาก R1):** ต้องมีตัวตัดรอบ polling ที่ชัดเจน ไม่ใช่การอนุมานจากการเห็น address 1
7. **เพิ่มใหม่ (จาก R3):** reader ทุกตัวต้องเลิกอ่านคอลัมน์ด้วยตำแหน่ง

---

# 3. ผล Audit แบบละเอียด

## 3.1 สมมติฐาน "3 เซนเซอร์" ที่ฝังอยู่ (fixed-3 assumptions)

### ฝั่ง PC — `C:\MOF_NanoTec\test_realtime`

| ไฟล์ | บรรทัด | สิ่งที่ฝัง | ความยาก |
|---|---|---|---|
| `logger_3ec.py` | 187-188 | `HEADER = [timestamp,EC1,T1,EC2,T2,EC3,T3,ok1,ok2,ok3,flag]` | สูง — เป็น schema |
| `logger_3ec.py` | 241-248 | `parse()` : `if len(p) != 8`, `len(okbits) != 3` | สูง |
| `logger_3ec.py` | 268 | `SessionMgr.start = [None, None, None]` | กลาง |
| `logger_3ec.py` | 320, 330, 371, 402, 457, 473, 738 | `for i in range(3)` × 7 จุด | ต่ำ |
| `logger_3ec.py` | 379-383 | `do_calibrate()` : `if raw not in ("1","2","3")` | ต่ำ |
| `logger_3ec.py` | 420 | `if ch in ("1","2","3")` | ต่ำ |
| `logger_3ec.py` | 351 | ข้อความ `sensor_1/ 2/ 3/` | ต่ำ |
| `logger_3ec.py` | 499 | `"generating combined 3-sensor report"` | ต่ำ |
| `calibration.py` | 148 | `TRACE_HEADER = [...,"EC1","EC2","EC3"]` | สูง — schema |
| `calibration.py` | 215 | `"ec": [num("EC1"), num("EC2"), num("EC3")]` | สูง |
| `calibration.py` | 331, 356, 437, 607, 676 | `for i/k in range(3)` × 5 | ต่ำ |
| `calibration.py` | 437 | `read_all = lambda: [read_ec(i) for i in range(3)]` | ต่ำ |
| `calibration.py` | 604-607 | `stats_post_all` = list ยาว 3 (ลง JSON) | กลาง — schema |
| `calibration.py` | 674-690 | `cross_spread_us` คำนวณจาก `len(good) == 3` เท่านั้น | กลาง |
| `calibration.py` | 728, 738, 749 | `latest=[None]*3`, `targets=[0,1,2]` | ต่ำ |
| `calibration.py` | 613-614 | ข้อความ "put all three probes in the SAME beaker" | ต่ำ |
| `desktop_ui.py` | 46 | `COLORS = [...]` 3 สี | ต่ำ |
| `desktop_ui.py` | 73-91 | `read_range()` positional `p[1] p[3] p[5]`, `len(p) < 10` | **สูง — R3** |
| `desktop_ui.py` | 109-142 | `make_mock_csv()` เขียน header 10 คอลัมน์แบบ hard-code | กลาง |
| `desktop_ui.py` | 165, 220 | `export_csv/export_excel` header hard-code | กลาง |
| `desktop_ui.py` | 208, 252 | `for i in range(3)` ในสถิติ | ต่ำ |
| `desktop_ui.py` | 233-243 | `export_pdf` : `fig.add_subplot(3,1,i+1)` | กลาง |
| `desktop_ui.py` | 309-321 | `_build_cards()` : `for i in range(3)` pack side=left | **สูง — layout** |
| `desktop_ui.py` | 346-350 | `_build_session_bar()` rec label × 3 | กลาง |
| `desktop_ui.py` | 366-370 | `_build_charts()` : `add_subplot(3,1,i+1)` | **สูง — layout** |
| `desktop_ui.py` | 419, 425 | `active = [False,False,False]` จาก `rec_status.json` | กลาง |
| `desktop_ui.py` | 275, 298 | title `"EC x3"`, `"EC 3 ภาชนะ"` | ต่ำ |
| `report_3ec.py` | 42 | `COLORS` 3 สี | ต่ำ |
| `report_3ec.py` | 313 | `recs = [_cal_record(i) for i in range(3)]` | ต่ำ |
| `report_3ec.py` | 608, 617, 620 | `range(3)` ในการสร้างหน้า | กลาง |
| `report_3ec.py` | 614 | `total = 1 + 3 + 1 + 3 + 1` (9 หน้าตายตัว) | **สูง — layout** |
| `report_3ec.py` | 644-660 | `read_csv_rows()` positional `p[1] p[3] p[5]`, `len(p) < 10` | **สูง — R3** |
| `report_3ec.py` | 463, 547, 867 | `for i in range(3)` | ต่ำ |
| `report_3ec.py` | 880 | Excel raw header hard-code 10 คอลัมน์ | กลาง |
| `report_3ec.py` | 194, 241, 246, 843 | ข้อความ `"3 samples"` | ต่ำ |
| `report_3ec.py` | 936-937 | ชื่อไฟล์ `run_3ec_{stamp}.pdf` | ต่ำ |
| `rec_status.json` | — | `{"active": [false,false,false]}` | กลาง — schema |
| `sessions_3ec.json` | — | ไม่ผูกกับ 3 (มี field `sensor` อยู่แล้ว) ✅ | — |
| `calibration_log.json` | — | `stats_post_all` ยาว 3, `cross_medians` ยาว 3 | กลาง |

> ⚠️ **พบเพิ่มเติม:** `calibration_log.json` และโฟลเดอร์ `calibration_data/`
> **ยังไม่มีอยู่จริงบนเครื่อง** ⇒ ประวัติการคาลิเบรตฝั่ง PC ว่างเปล่า และหน้า
> calibration ในรายงาน PDF (`_page_calibration_all`) แสดง "ไม่มีข้อมูล" ทุกครั้ง
> ทั้งที่ `SYSTEM.md §7` ยืนยันว่าคาลิเบรตผ่านบัสจากจอสำเร็จแล้ว
> ⇒ **นี่คือหลักฐานตรงว่าการ cal จากจอไม่เคยถูกบันทึกฝั่ง PC เลย** และเป็นเหตุผล
> ที่ทำให้ Phase 3b มีคุณค่าทันที

**รวมจุดที่ต้องแก้ฝั่ง PC: ~48 จุด ใน 4 ไฟล์ + 2 schema ไฟล์**

### ฝั่ง CONTROL — `water_monitor_3ec.ino`

| บรรทัด | สิ่งที่ฝัง |
|---|---|
| 142 | `#define N_SENSORS 3` ✅ เป็นจุดเดียวอยู่แล้ว |
| 143 | `const uint8_t ADDRS[N_SENSORS] = {1,2,3};` — hard-code address |
| 159-164 | array ทั้งหมดใช้ `N_SENSORS` ✅ |
| 290 | `sparkBuf[N_SENSORS][SPARK_N]` ✅ |
| 533 | `if (a >= 1 && a <= N_SENSORS ...)` ✅ |
| 839 | `sscanf` cal: `n >= 1 && n <= N_SENSORS` ✅ |
| 1105-1112 | DATA frame — วน `N_SENSORS` ✅ แต่ **format ผูกกับ 3 ในฝั่ง parser** |
| — | **ไม่มี** NVS/Preferences สำหรับ config เลย (ตรวจแล้ว ไม่มี `#include <Preferences.h>`) |
| — | `ENABLE_ESPNOW 1` ส่ง `ec_packet_t` 41 ไบต์ ไปที่ `PEER_MAC` |

**ข่าวดี:** ฝั่ง CONTROL แทบพร้อมอยู่แล้ว — `N_SENSORS` เป็นจุดเดียวจริง
**ข่าวร้าย:** ยังไม่มี NVS เลย ต้องเพิ่ม `Preferences` ใหม่ทั้งหมด

### ฝั่ง P4 — `hello_world/` (อ่านอย่างเดียว)

| ไฟล์ | สิ่งที่ฝัง |
|---|---|
| `main/ec_packet.h:15` | `#define EC_SENSOR_COUNT 3` — **จุดเดียว** ✅ |
| `main/ec_packet.h:28` | `_Static_assert(sizeof(ec_packet_t) == 41)` — ผูกกับ 3 |
| `main/ui/ui_model.h:25` | `#define UI_SENSOR_COUNT EC_SENSOR_COUNT` ✅ |
| `main/ec_rs485.h:105` | `ec_rs485_scan_found()` คืน bitmask `bit0..bit2` — 8 bit รองรับ 4 ได้อยู่แล้ว ✅ |
| `main/ui/screen_settings.c:260` | ปุ่มเลือก probe วน 3 ตัว |
| `main/ui/screen_overview.c` | layout 3 คอลัมน์ |
| `main/ui/screen_measure.c` | 3 bays เรียงซ้าย→ขวา |
| `main/ui/screen_trend.c` | 3 เส้นบนแกนเดียว |
| `main/ec_rs485.c` `on_request()` | **ใช้ `01 03 0000` เป็นตัวตัดรอบ — R1** |

**สรุป:** ฝั่ง P4 สะอาดกว่าฝั่ง PC มาก โครงสร้าง `EC_SENSOR_COUNT` + `ui_model` เป็นชั้นกลาง
ทำให้เปลี่ยนจำนวนได้ค่อนข้างตรงไปตรงมา งานหนักอยู่ที่ **layout** กับ **R1** เท่านั้น

## 3.2 Serial ownership

| พอร์ต | อุปกรณ์ | VID:PID / ชิป | ใครเปิดตอนนี้ | ใครจะเปิดหลัง refactor |
|---|---|---|---|---|
| COM-A | CONTROL V1.2 | CH340 (`logger_3ec.py:203` ให้คะแนน `ch340/cp210/ch910`) | `logger_3ec.py` เท่านั้น | เหมือนเดิม |
| COM-B | ESP32-P4 | **CH343 `1A86:55D3`** (`tools/p4port.py:11`) | **ไม่มีใครเปิด** — logger กันไว้แล้ว (`score() → -1`) | `p4_bridge` (thread ใน logger process เดียวกัน) |

**ยืนยัน:** ไม่มี process ไหนแย่งพอร์ตกันอยู่ตอนนี้ และ `find_port()` กัน CH343 ไว้ให้แล้ว
โดยตั้งใจ — โครงสร้างพร้อมรับ COM-B อยู่แล้ว

**สิ่งที่ต้องระวัง:** COM-B (CH343) ตอนนี้คือ **console + ESP_LOG + คำสั่ง `S/B/V/C/D/K`**
(`ec_clock.c:229 cmd_task()` ใช้ `getchar()` จาก stdin ซึ่งผูกกับ `CONFIG_ESP_CONSOLE_UART_NUM=0`)
ถ้า PC จะใช้พอร์ตนี้เป็น command channel ต้องแก้ปัญหา log ปนกับ protocol — ดูหัวข้อ 6

## 3.3 Protocol assumptions

| สมมติฐาน | ที่ฝัง | ยังจริงมั้ย |
|---|---|---|
| `DATA,` มี 8 fields เป๊ะ | `logger_3ec.py:242` | จริง |
| `okbits` ยาว 3 ตัว เป็น "0"/"1" | `logger_3ec.py:246` | จริง |
| ค่าที่อ่านไม่ได้ = `NaN` (string) แปลงเป็น `""` | `logger_3ec.py:244`, `.ino:1109` | จริง |
| ไม่มี seq / ไม่มี timestamp จากบอร์ด | ตรวจแล้ว — DATA ไม่มี seq | **จริง — ตรวจ gap ไม่ได้เลยตอนนี้** |
| ผลลัพธ์ cal มาทาง `[cal] ok/fail` | `calibration.py:456-458` `"ok" in line.split("[cal]")[1][:12]` | จริง — แต่เปราะ |
| จอไม่ส่งอะไรมาที่ PC เลย | จริง | จริง |
| จอตัดรอบ polling จากการเห็น address 1 | `BUS_PROTOCOL.md §1` | **จริง — R1** |

## 3.4 CSV / report assumptions

- ไฟล์ raw: `water_data/water_log_YYYY-MM-DD.csv` — glob pattern เดียวกันทุก reader
- header จริงบนดิสก์มี **3 เวอร์ชัน** ปนกันแล้ว:
  1. `timestamp,EC1,T1,EC2,T2,EC3,T3,ok1,ok2,ok3` (10 คอลัมน์ — ไฟล์ 07-25, 07-26, 08-03, 08-04)
  2. `timestamp,EC_uScm,Tw_C,Salinity_ppm,TDS_ppm,pH,pH_mV,rs485_ok` (8 คอลัมน์ header / 7 คอลัมน์ data — ไฟล์ 08-05, 08-07) ← **เขียนโดย `logger.py` ตัวเก่า**
  3. `...,ok3,flag` (11 คอลัมน์ — สิ่งที่ `logger_3ec.py:187` เขียนวันนี้ แต่ยังไม่มีไฟล์ไหนบนดิสก์)
- ทุก reader `skip` header บรรทัดแรกด้วย `fh.readline()` แล้วอ่านตำแหน่ง — **ไม่เคยตรวจ header เลย**
- `report_3ec.py` สร้างหน้า PDF ตายตัว 9 หน้า (`total = 1+3+1+3+1`)
- `export_sensor_session()` เขียนลง `sensor_{idx+1}/`, `export_combined_report()` เขียนลง `reports/`

## 3.5 UI assumptions (PC)

- `desktop_ui.py` เป็น **view-only** จริง — ไม่มีการเปิด serial, ไม่มีการเขียนไฟล์ควบคุมใด ๆ
  (มี `save_sessions()` แต่ไม่มีใครเรียก)
- สถานะ "กำลังบันทึก" อ่านจาก `rec_status.json` ทุก 2 วิ (`_schedule()`)
- ความ "live" อนุมานจากอายุแถวล่าสุด `< 15 วินาที` (`refresh_now():403`)
  — **ไม่มี heartbeat ของ logger จริง ๆ** ถ้า logger ตายแต่ CSV ยังใหม่ UI จะยังบอกว่า live
- **ไม่มีอะไรบน UI ที่บอกว่ากำลัง calibrate อยู่เลย** — แม้ `logger_3ec.py` จะรู้ (มันเขียน flag `CAL`
  ลง CSV ตอนนั้น) แต่ UI แค่ *ข้าม* แถว CAL ไปเงียบ ๆ (`desktop_ui.py:76`)
  ⇒ ผู้ใช้เห็นกราฟ "หยุดเดิน" โดยไม่รู้ว่าทำไม

---

# 4. สถาปัตยกรรมที่แนะนำ

## 4.1 หลักการตัดสินใจข้อที่สำคัญที่สุด — แยก "cal" ออกจาก "command console"

> **cal ไม่ควรวิ่งผ่าน PC และไม่ควรใช้ NDJSON**

เหตุผลเชิงวิศวกรรม 4 ข้อ:

1. **เส้นทางมีอยู่แล้วและทดสอบผ่านแล้ว** (`P4 → 0x21 → CONTROL → SEN0706 → 0x22 → P4`)
   การสร้างเส้นทางที่สองที่ทำงานเดียวกัน = มี "ความจริง" สองชุด
2. **ด่านความปลอดภัยอยู่ที่ CONTROL ไม่ใช่ที่ PC** — `calibrateSensor()` ตรวจ
   `|ec - std| / std > 0.5 → CAL_RC_NOT_IN_SOL (0xF0)` ก่อนยิง Modbus ทุกครั้ง
   ไม่ว่าคำสั่งจะมาจากปุ่มบนจอ จากบัส หรือจาก serial ของ PC
   (`.ino:434-441` เขียนไว้ชัดว่า "ด่านนี้อยู่ที่จุดลงมือ จึงคุ้มครองทุกทาง")
   ถ้าให้ PC เป็นคน validate เราย้ายด่านออกจากจุดที่ปลอดภัยที่สุด
3. **cal ต้องทำได้แม้ PC ปิดอยู่** — เป็นข้อกำหนดของโจทย์เองข้อ 11
4. **CONTROL รายงานผลออก COM-A อยู่แล้วทุกครั้ง** — PC แค่ต้อง "ฟัง" ไม่ต้อง "สั่ง"

### ผลลัพธ์: PC เป็น **observer** ของ cal ไม่ใช่ **orchestrator**

```
ผู้ใช้กด Calibrate บนจอ P4
        │
        ├──► P4: ec_rs485_calibrate()  ──0x21──►  CONTROL
        │                                            │
        │                                            ├─► Serial.printf("[bus] จอสั่งคาลิเบรต EC#2 ที่ 84 uS/cm")
        │                                            ├─► calibrateSensor()  ─Modbus 0x0110─► SEN0706
        │                                            ├─► Serial.printf("[cal] ok n=2 std=84 (reg=840)")
        │                                            └─► sendAckFrame() ──0x22──► P4 (แสดง "Done")
        │                                                     │
        │                                                COM-A (มีอยู่แล้ว)
        │                                                     ▼
        │                                        logger_3ec.py : control_events.classify()
        │                                                     │
        │                          ┌──────────────────────────┼──────────────────────┐
        │                          ▼                          ▼                      ▼
        │                  cal_status.json          calibration_log.json      CSV flag="CAL"
        │                  (สถานะสด, UI อ่าน)        (entry origin="p4")       (มีอยู่แล้ว)
        │                          │
        │                          ▼
        │                  desktop_ui.py : แถบ "⚗ CALIBRATING EC#2 @ 84 µS/cm"
        │                                   + แถบสีบนกราฟช่วง cal
        └──► (ไม่ต้องใช้ COM-B เลย)
```

**ต้นทุน: แก้ Python 2 ไฟล์ ไม่แตะ firmware ทั้งสองตัว ไม่ต้องต่อสายเพิ่ม**

### `COM-B` / NDJSON ยังจำเป็นสำหรับอะไร

สำหรับสิ่งที่ **CONTROL ไม่รู้เรื่อง** เท่านั้น — คือ session / recording / sample_id / note /
report / measurement event ของหน้า Measure  เพราะข้อมูลพวกนี้ไม่มีทางเดินผ่านบัส RS485
โดยไม่ทำให้ CONTROL ต้องรู้จักเรื่องที่ไม่ใช่หน้าที่มัน

## 4.2 ภาพรวมสถาปัตยกรรมเป้าหมาย

```
                                 ┌─────────────────────────────────────┐
                                 │   Windows PC (Python)               │
                                 │                                     │
  CONTROL ──COM-A──────────────► │  logger_ec.py  (process เดียว)      │
   ▲   │      DATA / ECV2        │   ├─ serial_reader   (thread หลัก)  │
   │   │      [cal] [bus] [scan] │   ├─ control_events  (classifier)   │
   │   │      [cfg]              │   ├─ csv_writer      (fsync ทุกแถว)  │
   │   │                         │   ├─ session_mgr     (1-4 sessions) │
   │   └──── X / C / Z / T ──────┤   ├─ cal_observer                   │
   │         (คำสั่ง PC→CONTROL)  │   ├─ report_jobs     (thread pool)  │
   │                             │   └─ p4_bridge       (thread) ◄──┐  │
   │ RS485                       │                                  │  │
   │ 0x21/0x22 (cal, oled, cfg)  └──────────────────────────────────┼──┘
   │ 0x20 (time)                        │ เขียนไฟล์สถานะ            │
   │ 0x23 (cycle end — ใหม่)             ▼                          │
   │                              pc_state.json                     │ COM-B
   │                              cal_status.json                   │ NDJSON
   │                              sessions_ec.json                  │ @J1 prefix
   ▼                              measurement_events_*.jsonl        │
 ESP32-P4 ◄──────────────────────────────────────────────────────── ┘
   │
   └─ desktop_ui.py (view-only, อ่าน pc_state.json + CSV)
      report_ec.py  (อ่าน CSV + event log)
```

**กฎที่เพิ่มเข้ามา:**

- **P4 bridge เป็น thread ใน `logger_ec.py` ไม่ใช่ process แยก** — เพราะ command เช่น
  `session_start` ต้องแก้ state ที่อยู่ใน memory ของ logger ถ้าแยก process ต้องมี IPC
  อีกชั้นซึ่งเป็นความซับซ้อนที่ไม่ได้อะไรกลับมา
- **thread นี้ห้ามแตะ `ser` ของ COM-A เด็ดขาด** — สื่อสารกับ main loop ผ่าน
  `queue.Queue` สองทางเท่านั้น (`cmd_q` เข้า, `state_q` ออก)
- **report job รันบน `ThreadPoolExecutor(max_workers=1)` แยก** — เพราะ `generate_pdf_3ec()`
  ใช้เวลาหลายวินาที ถ้ารันใน main loop CSV จะขาดช่วง (ปัญหานี้มีอยู่แล้ววันนี้
  ตอนกดหยุด session แล้ว export — `logger_3ec.py:312` เรียกตรง ๆ ใน `_close()`)

---

# 5. ตารางความเป็นเจ้าของข้อมูล

## 5.1 ยืนยัน + แย้งข้อเสนอเดิม

| ข้อมูล | เจ้าของ (ข้อเสนอเดิม) | **ข้อสรุปหลัง audit** | เหตุผล |
|---|---|---|---|
| `active_sensor_mask` | CONTROL | ✅ **CONTROL** | มันเป็นตัวเดียวที่ยิง Modbus จริง ถ้าไม่ใช่มันเป็นเจ้าของ จะมี config สองชุดที่ไม่ตรงกัน |
| Modbus addresses | CONTROL | ✅ **CONTROL** | `ADDRS[]` อยู่ที่นั่นและ `Z` scan ก็ทำจากที่นั่น (จอ scan ไม่ได้ผลจริง — `BUS_PROTOCOL.md`) |
| poll interval | CONTROL | ⚠️ **CONTROL เป็นเจ้าของ แต่ต้องประกาศออกมา** | `ui_model_init()` ตั้ง `poll_interval_ms=2500` ซ้ำไว้ฝั่งจอ และ `stale_after_ms` คำนวณจากค่านี้ — ถ้า CONTROL เปลี่ยนแล้วจอไม่รู้ ทุกอย่างขึ้น STALE ⇒ ต้องส่งค่าจริงมากับ frame |
| hardware scan result | CONTROL | ✅ **CONTROL** | พิสูจน์แล้วว่า scan จากฝั่งจอไม่ได้ผล |
| **calibration ของหัววัด** | (ไม่ได้ระบุ) | ✅ **SEN0706 เอง / สั่งโดย CONTROL** | ค่าที่ cal ฝังอยู่ในหัววัด (register 0x0110) ไม่ใช่ในซอฟต์แวร์ — `calibration.py:11-13` เขียนไว้ชัด |
| **calibration history / report** | (ไม่ได้ระบุ) | ✅ **PC** | เป็นหลักฐานงานแล็บ ต้องอยู่ที่ที่มีดิสก์และสำรองได้ |
| session | PC | ✅ **PC** | |
| raw CSV | PC | ✅ **PC** | |
| reports | PC | ✅ **PC** | |
| event archive | PC | ✅ **PC** | |
| sample metadata | PC | ✅ **PC** | |
| theme / brightness / volume / click | P4 | ✅ **P4** (NVS `ecmon`) | มีอยู่แล้วใน `ec_prefs_t` |
| **immersion mode (MANUAL/AUTO)** | (ไม่ได้ระบุ) | ✅ **P4** | เป็นพฤติกรรม UI ล้วน อยู่ใน `ec_prefs_t.immersion_mode` แล้ว |
| **threshold เตือนรายเซนเซอร์** | (ไม่ได้ระบุ) | ⚠️ **P4 ตอนนี้ → ควรย้ายไป PC ภายหลัง** | `ec_config_t` เก็บใน NVS ของจอ แต่รายงานฝั่ง PC ควรอ้างเกณฑ์เดียวกันได้ — ยกไว้ Phase 5 |
| measurement run / stable decision | P4 | ✅ **P4** | `measure_policy.c` เป็นตรรกะล้วนที่ทดสอบแล้ว 82 ข้อ |

## 5.2 จุดที่ **ขอแย้ง** ข้อเสนอเดิม

### แย้งข้อ 1: cal ไม่ควรผ่าน PC (ดูหัวข้อ 4.1)

โจทย์ข้อ 2 จัด `calibration_request` ไว้ใน Phase 2 commands ที่ต้องผ่าน PC validation
**ไม่เห็นด้วย** — ควรเป็น: P4 → CONTROL โดยตรง (ทางเดิม), PC เป็นผู้สังเกตและบันทึก
สิ่งที่ PC ควรทำได้คือ **"ขอจอง"** ไม่ใช่ "อนุญาต":

- PC เขียน `pc_state.json` บอกว่า `"session_active": [2,3]`
- P4 อ่านผ่าน state snapshot แล้ว **แสดงคำเตือน** บนหน้า Settings ว่า
  `PC IS RECORDING SENSOR 2 — CALIBRATING WILL INSERT A GAP`
- แต่ **ไม่บล็อก** — เพราะถ้า PC ตาย ผู้ใช้ต้อง cal ได้อยู่ดี
- PC ฝั่งตัวเองจัดการเอง: แถวช่วง cal ได้ flag `CAL` และรายงานข้ามให้ (ทำอยู่แล้ว)

### แย้งข้อ 2: `sensor_config_apply` **ควร**ผ่าน PC (ตรงข้ามกับ cal)

โจทย์เสนอ P4 → PC → CONTROL สำหรับ config — **เห็นด้วยเต็มที่** และเหตุผลต่างจาก cal:
config เปลี่ยน **schema ของ CSV** ที่ PC กำลังเขียนอยู่ ถ้า CONTROL เปลี่ยน mask
โดย PC ไม่รู้ ไฟล์ CSV จะมี "ครึ่งไฟล์เป็น mask เก่า ครึ่งไฟล์เป็น mask ใหม่"
โดยไม่มีเครื่องหมายอะไรเลย

⇒ **ถ้า PC offline ต้องเปลี่ยน config ไม่ได้** และนั่นถูกต้องแล้ว
(แสดงบนจอว่า `PC NOT CONNECTED — CONFIG LOCKED`)

**สรุปกฎเดียวที่จำง่าย:**
> คำสั่งที่เปลี่ยน *ฮาร์ดแวร์* → ไปตรงผ่านบัส (PC ฟัง)
> คำสั่งที่เปลี่ยน *ความหมายของไฟล์ที่ PC เขียนอยู่* → ต้องผ่าน PC

---

# 6. Transport ระหว่าง P4 กับ PC — ข้อเสนอ

## 6.1 CH343 vs USB CDC-ACM/TinyUSB

| | CH343 (UART0, พอร์ตที่ใช้อยู่) | USB CDC-ACM (TinyUSB บน USB-OTG) |
|---|---|---|
| ไดรเวอร์ Windows | มีอยู่แล้ว (ใช้แฟลชอยู่ทุกวัน) | CDC class driver ในตัว Windows 10+ |
| VID:PID คงที่ | ✅ `1A86:55D3` — `tools/p4port.py` ใช้อยู่แล้ว | ต้องกำหนดเอง เสี่ยงชนกับ device อื่น |
| หายตอนบอร์ดรีบูต | ไม่หาย (CH343 เป็นชิปแยก พอร์ตอยู่ตลอด) | **หาย** — enumerate ใหม่ทุกครั้งที่ P4 reset ⇒ COM เปลี่ยนเลข, ต้อง reconnect ทุกครั้ง |
| แชร์กับ ESP_LOG | **ต้องแชร์** — ต้องทำ framing | แยกได้ (log อยู่ UART0, protocol อยู่ CDC) |
| กระทบบั๊ก PANIC ที่ยังหาไม่เจอ (R4) | ไม่กระทบ | **กระทบ** — TinyUSB จองหน่วยความจำ + task เพิ่ม |
| แฟลช/monitor ระหว่างพัฒนา | ต้องปิด bridge ก่อน | ทำพร้อมกันได้ |
| งานที่ต้องทำฝั่ง P4 | เพิ่ม prefix + mutex (~150 บรรทัด) | เพิ่ม TinyUSB + descriptor + task (~400 บรรทัด + config) |

### ข้อเสนอ: **ใช้ CH343 (UART0) พร้อม line-prefix framing**

เหตุผลชี้ขาด: **บั๊ก PANIC ระหว่าง soak ยังหาสาเหตุไม่ได้ (R4)**
การเพิ่ม USB stack ตอนนี้คือการเพิ่มตัวแปรให้บั๊กที่กำลังไล่อยู่ ซึ่งจะทำให้แยกไม่ออกว่า
พังเพราะอะไร  CDC-ACM เป็นทางเลือกที่ดีกว่า *ในระยะยาว* แต่ต้องรอให้ soak ผ่าน 2 ชั่วโมงก่อน

**ทางที่สาม (สำรอง ถ้า framing มีปัญหาจริงตอนทดสอบ):** ใช้ UART2 บนขาว่าง
(`CONFIG_SOC_UART_NUM=6` — UART0=console, UART1=RS485, เหลือ 4 ตัว) ต่อ USB-TTL dongle
เป็น COM-C แยกกายภาพ 100% ต้นทุนคือสาย+dongle 1 ชุด

## 6.2 การแยก machine protocol ออกจาก human log บนสายเดียว

**กฎ 4 ข้อ:**

1. **ทุกเฟรมโปรโตคอลขึ้นต้นด้วย `@J1 ` (4 ไบต์) แล้วตามด้วย JSON object และ `\n`**
   ```
   @J1 {"v":1,"type":"event","event":"reading_saved",...}
   ```
   PC parser รับเฉพาะบรรทัดที่ match `^@J1 \{.*\}$` เท่านั้น อย่างอื่นถือเป็น log ทิ้งได้
   (หรือเก็บลง `p4_console.log` เพื่อ debug)

2. **เขียนเฟรมทั้งบรรทัดในการเรียกเดียวภายใต้ mutex** — `xSemaphoreTake(tx_mux)` →
   `fwrite(buf, 1, len, stdout)` → `fflush(stdout)` → `xSemaphoreGive()`
   กัน ESP_LOG จาก task อื่นแทรกกลางบรรทัด

3. **ปิดสี ANSI ในทุก build:** `CONFIG_LOG_COLORS=n`
   เพราะ escape sequence ทำให้ regex ฝั่ง PC พลาดได้

4. **ห้าม `ESP_LOG*` ใช้ tag ที่ขึ้นต้นด้วย `@J1`** — บังคับด้วย code review

**ทำไมไม่ใช้ `\x1e` (ASCII RS) หรือ COBS:** อ่านไม่ออกด้วยตาเปล่าใน `idf.py monitor`
ซึ่งเป็นเครื่องมือ debug หลักของโปรเจกต์นี้  prefix ที่เป็น ASCII ล้วนแลกความ "สวย"
กับความ "ไล่ปัญหาได้" ซึ่งคุ้มกว่ามากในสถานะปัจจุบันของโปรเจกต์

## 6.3 JSON บน P4 — **ไม่ใช้ cJSON**

**ข้อเท็จจริง:** ESP-IDF v6.0.2 ไม่มี component `json` แล้ว (ตรวจ `components/` ครบแล้ว)
ต้องเพิ่ม `espressif/cjson` จาก registry

**ข้อเสนอ: ไม่เพิ่ม** — เขียน scanner แบบ flat-key ~120 บรรทัด แทน

เหตุผล:
- P4 ต้องอ่านจาก PC แค่ **6 field** เท่านั้น: `type`, `request_id`, `ok`, `code`, `seq`, `pc_online`
  (+ `sessions` ซึ่งย่อเป็น bitmask ได้ — ดู 7.4)
- cJSON จองฮีปต่อ node ทุกครั้ง ⇒ ข้อความ 512 ไบต์ = จอง/คืน ~30 ก้อนย่อย ทุก 2-5 วินาที
  **นี่คือรูปแบบการใช้ฮีปที่ทำให้ heap แตกเป็นเสี่ยง** ซึ่งเป็นบั๊กที่โปรเจกต์นี้เจอมาแล้ว
  (`SYSTEM.md §6` — "heap เหลือเท่าเดิมแต่แตกเป็นเสี่ยง จะจองก้อนใหญ่ไม่ได้")
- scanner แบบ flat-key ใช้ **static buffer อย่างเดียว จองฮีป 0 ไบต์**
- **P4 เป็นฝ่าย "สร้าง" JSON ซึ่งง่าย** (`snprintf`) — ความยากอยู่ที่การ "อ่าน" เท่านั้น
  และเราจำกัดสิ่งที่ต้องอ่านให้เล็กที่สุดได้

**งบหน่วยความจำที่เสนอ (P4):**

| รายการ | ขนาด | ที่เก็บ |
|---|---|---|
| RX line buffer | 512 B | static |
| TX frame buffer | 512 B | static |
| offline event queue | 32 × 96 B = 3 KB | static (ไม่ใช่ฮีป) |
| bridge task stack | 4 KB | FreeRTOS |
| **รวม** | **~8 KB คงที่ ไม่แตะฮีปเลย** | |

---

# 7. ข้อเสนอโปรโตคอล NDJSON

## 7.1 กฎกรอบ

| หัวข้อ | ค่า | เหตุผล |
|---|---|---|
| encoding | UTF-8 | note ภาษาไทยได้ |
| max frame | **512 ไบต์** (รวม `@J1 ` และ `\n`) | ให้อยู่ใน static buffer และไม่กิน UART FIFO |
| เกิน max | ทั้งสองฝั่ง **ทิ้งทั้งบรรทัด** + นับ `frames_dropped_oversize` + log 1 ครั้ง/10 วิ | ห้าม truncate แล้ว parse ต่อ — จะได้ JSON ที่ "ถูกต้อง" แต่ผิดความหมาย |
| protocol version | `"v": 1` บังคับทุกเฟรม | |
| `v` ไม่ตรง | ตอบ NACK `code:"UNSUPPORTED_VERSION"` แล้ว **ไม่ประมวลผล** | |
| unknown field | **ต้อง ignore** ทั้งสองฝั่ง | forward compatibility |
| unknown `action` | NACK `code:"UNKNOWN_ACTION"` | |
| malformed JSON | ทิ้งเงียบ + นับ `frames_dropped_parse` | ตอบ NACK ไม่ได้เพราะไม่รู้ `request_id` |
| rate limit (P4→PC) | **10 เฟรม/วินาที** เกินแล้ว PC ทิ้ง + NACK `RATE_LIMITED` | กัน P4 ที่บั๊กท่วม PC |
| state snapshot (PC→P4) | ทุก **3 วินาที** + ทุกครั้งที่ state เปลี่ยน | อยู่ในช่วง 2-5 วิ ที่ขอ และหารลงตัวกับ poll 2500 ms พอดี |
| P4 ถือว่า PC offline | ไม่เห็น state snapshot **10 วินาที** (3× interval + margin) | สอดคล้องกับสูตร `stale_after_ms` ที่ใช้อยู่ |

## 7.2 P4 → PC : command request

```json
@J1 {"v":1,"type":"cmd","request_id":1042,"boot_id":"p4-20260825-a1b2","action":"session_start","sensor":2,"ts_ms":182345600,"payload":{"sample_id":"POND-A","note":""}}
```

| field | ชนิด | บังคับ | หมายเหตุ |
|---|---|---|---|
| `v` | int | ✅ | = 1 |
| `type` | str | ✅ | `"cmd"` |
| `request_id` | uint32 | ✅ | เพิ่มทีละ 1 ต่อ boot ไม่รีเซ็ต |
| `boot_id` | str ≤ 20 | ✅ | `p4-YYYYMMDD-xxxx` (xxxx = 16 bit สุ่มตอนบูต) |
| `action` | str ≤ 24 | ✅ | |
| `sensor` | int 1-4 | ตามคำสั่ง | 0 = ไม่ระบุ/ทั้งระบบ |
| `ts_ms` | uint32 | ✅ | monotonic จาก `esp_timer_get_time()/1000` |
| `payload` | object | ไม่ | field ภายในแล้วแต่ action |

**การ retry:** P4 ส่งซ้ำได้เฉพาะ **idempotent action** และต้องใช้ `request_id` เดิม
- retry ได้: `session_start`, `session_stop`, `recording_start`, `recording_stop`,
  `sample_set`, `note_set` — เพราะผลลัพธ์เป็น "สถานะปลายทาง" ไม่ใช่ "การกระทำ"
- **retry ไม่ได้: `report_request`** — เป็นการสร้างไฟล์ ต้องรอ ACK อย่างเดียว
  ถ้า timeout ให้แสดง `REQUEST STATUS UNKNOWN — CHECK PC` ไม่ใช่ส่งซ้ำ
- retry policy: 3 ครั้ง ห่าง 1s / 2s / 4s แล้วเลิก

**PC dedupe:** เก็บ `set[(boot_id, request_id)]` ล่าสุด 256 รายการ (deque)
ถ้าซ้ำ → **ตอบ ACK เดิมซ้ำ** (ไม่ execute ใหม่) ⇒ retry ปลอดภัยเสมอ

## 7.3 PC → P4 : ACK / NACK

```json
@J1 {"v":1,"type":"ack","request_id":1042,"ok":true,"action":"session_start","payload":{"session_id":"S2-20260825-001","active_sensor_mask":7}}
@J1 {"v":1,"type":"ack","request_id":1042,"ok":false,"action":"session_start","code":"SESSION_ALREADY_ACTIVE","message":"Sensor 2 already has an active session."}
```

**ชุดรหัส `code` ที่นิยามไว้ (ห้ามส่งรหัสนอกลิสต์):**

| code | ความหมาย | ข้อความบนจอ P4 |
|---|---|---|
| `SESSION_ALREADY_ACTIVE` | เริ่มซ้ำ | Session already running |
| `SESSION_NOT_ACTIVE` | หยุดทั้งที่ไม่ได้เริ่ม | No session to stop |
| `SENSOR_DISABLED` | sensor นั้นไม่อยู่ใน mask | Sensor is disabled |
| `SENSOR_OUT_OF_RANGE` | sensor > MAX_SENSORS | Invalid sensor |
| `LOGGER_NOT_CONNECTED` | COM-A หลุด | Sensor board offline |
| `CONFIG_LOCKED_SESSION` | มี session ค้าง | Stop all sessions first |
| `CONFIG_LOCKED_RUN` | มี run ค้างบน P4 | Finish the measurement first |
| `CONFIG_LOCKED_CAL` | กำลัง cal | Calibration in progress |
| `REPORT_BUSY` | มี job ค้าง | A report is already running |
| `NO_DATA_IN_WINDOW` | ไม่มีข้อมูล | No data in that window |
| `RATE_LIMITED` | ส่งถี่เกิน | Slow down |
| `UNKNOWN_ACTION` / `UNSUPPORTED_VERSION` | | Unsupported command |
| `INTERNAL_ERROR` | exception ฝั่ง PC | PC error — check the log |

## 7.4 PC → P4 : state snapshot

```json
@J1 {"v":1,"type":"state","seq":851,"pc_online":true,"recording":true,"sample_id":"POND-A","active_mask":7,"session_mask":2,"cal":{"busy":false},"report":{"state":"idle"},"csv_rows":18422}
```

**เปลี่ยนจากข้อเสนอเดิม:** ใช้ `session_mask` (bitmask) แทน object `sessions{}`
ที่มี `session_id` ครบทุกตัว

เหตุผล: object แบบเดิมกิน ~180 ไบต์ และบังคับให้ P4 ต้อง parse nested object
ซึ่งเป็นสิ่งเดียวที่ทำให้ต้องใช้ cJSON  bitmask ให้ข้อมูลที่ P4 **ใช้จริง**
(ไฟติดหรือไม่ติด) ครบถ้วน ส่วน `session_id` แบบเต็มส่งมากับ ACK ของ `session_start`
อยู่แล้ว ซึ่งเป็นจังหวะที่ P4 ต้องใช้จริงเพียงจังหวะเดียว

ถ้าภายหลังต้องการ `session_id` ทุกตัวบนจอ ค่อยเพิ่ม `type:"sessions"` เป็นเฟรมแยก
ที่ส่งเฉพาะตอนเปลี่ยน — ไม่ต้องยัดเข้า snapshot ที่ส่งทุก 3 วินาที

## 7.5 P4 → PC : measurement event

```json
@J1 {"v":1,"type":"event","event_id":"p4-20260825-a1b2-000017","boot_id":"p4-20260825-a1b2","event":"reading_saved","sensor":2,"ec_us_cm":84.6,"temperature_c":20.4,"tolerance_us_cm":2.0,"stable_for_ms":15000,"ts_ms":182345600,"note":""}
```

**PC บันทึกลง `water_data/measurement_events_YYYY-MM-DD.jsonl`** — append-only, fsync ทุกบรรทัด
- ห้ามแตะ CSV เด็ดขาด (ข้อกำหนดโจทย์ ข้อ 4)
- dedupe ด้วย `event_id` (deque 512 รายการ) — `event_id` ผูกกับ `boot_id` แล้ว จึงไม่ซ้ำข้ามการรีบูต
- PC **ไม่ ACK event** เพื่อประหยัด traffic — แต่ echo `last_event_id` ที่รับได้ล่าสุด
  ใน state snapshot ⇒ P4 รู้ว่าอันไหนถึงแล้ว ⇒ ล้างคิว offline ได้ถูกต้อง

**metadata ที่เพียงพอสำหรับปักหมุดบนกราฟและรายงาน:** ต้องมี `ts_ms` (monotonic) **และ**
`wall_ms` (เวลาผนัง จาก `ec_clock`) เพราะ CSV ฝั่ง PC ใช้เวลาผนังของ PC
⇒ **เพิ่ม field `wall`** (ISO-8601 string) เข้าไปในทุก event
PC จะจับคู่ด้วย `wall` เป็นหลัก และใช้ `ts_ms` ตรวจความสอดคล้องภายในของ run เดียวกัน

## 7.6 คิว offline

| ข้อมูล | เก็บที่ไหน | จำนวน | ทำไม |
|---|---|---|---|
| `reading_saved` | **RAM (static array)** + SD | 32 ใน RAM / ไม่จำกัดบน SD | มีอยู่ใน `readings.csv` บน SD แล้ว — RAM queue แค่ช่วยให้ replay ได้เร็ว |
| `calibration_result` | RAM | 8 | เกิดไม่บ่อย |
| event อื่น (`run_started` ฯลฯ) | **RAM เท่านั้น ทิ้งได้** | 32 (ring, ทับตัวเก่า) | เป็นข้อมูลบริบท ไม่ใช่ผลการวัด หายได้ |
| `sensor_config_requested` | **ไม่เข้าคิวเลย** | — | config ที่ apply ตอน PC กลับมา = อันตราย ผู้ใช้ต้องกดใหม่ |

**กฎ:** ถ้า RAM ring เต็ม ให้ทับตัวเก่าสุด **ยกเว้น `reading_saved`** ซึ่งถ้าเต็มให้
หยุดรับแล้วขึ้น `EVENT QUEUE FULL — PC OFFLINE` บนจอ (ผลการวัดหายไม่ได้)

---

# 8. การออกแบบ config 1–4 เซนเซอร์

## 8.1 โครงสร้างกลาง

```c
/* ใช้ร่วมกันทั้ง CONTROL และ P4 — คัดลอกไฟล์ให้ตรงกัน */
#define MAX_SENSORS 4

typedef struct {
    uint8_t  ver;                        /* = 1 */
    uint8_t  active_mask;                /* bit0=#1 … bit3=#4 */
    uint8_t  modbus_addr[MAX_SENSORS];   /* 1-247 */
    uint32_t poll_interval_ms;           /* 1000-10000 */
    uint16_t crc;                        /* กัน NVS blob เพี้ยน */
} sensor_config_t;
```

**ค่าเริ่มต้น (legacy) — บังคับ:**
```c
{ .ver=1, .active_mask=0b0111, .modbus_addr={1,2,3,4}, .poll_interval_ms=2500 }
```

## 8.2 mask เป็น source of truth — และกฎ 3 ชั้นที่ต้องแยกกัน

ทุกชั้นต้องแยก 3 สถานะนี้ออกจากกันเสมอ **ห้ามยุบรวม**:

| สถานะ | นิยาม | ที่มา | แสดงผล |
|---|---|---|---|
| `DISABLED` | `!(active_mask & (1<<i))` | config | สีเทา / ซ่อน — **ไม่ใช่ error** |
| `NO DATA` | enabled แต่ `ok=0` รอบล่าสุด | runtime | สีเหลือง `NO RESPONSE` |
| `FAULT` | enabled + `ok=0` ติดกัน ≥ `DEAD_AFTER_FAILS` (3) | runtime | สีแดง `SENSOR FAULT` |

**เหตุผลที่ต้องแยกจริง ๆ ไม่ใช่แค่เรื่อง UI:** หัววัด #1 ที่เสียตอนนี้ลากบัสตายทั้งเส้น
(`SYSTEM.md §1`) การแยก `DISABLED` ออกมาคือวิธีเดียวที่จะเอามันออกจากรอบ polling
โดยที่ระบบยังบอกได้ว่า "ตั้งใจเอาออก" ไม่ใช่ "พังเงียบ ๆ"

## 8.3 ⚠️ ปัญหาที่ต้องแก้ก่อนอย่างอื่น — R1: การตัดรอบ polling

**สถานะปัจจุบัน:** จอ P4 รู้ว่ารอบใหม่เริ่มจากการเห็นเฟรม `01 03 0000` (ถาม EC ตัวที่ 1)
ถ้า mask = `0b0110` (ปิด #1 ซึ่งเป็นกรณีใช้งานจริงข้อแรก เพราะหัววัด #1 เสีย)
**จอจะไม่ปิดรอบเลย → STALE ทั้งเครื่อง**

### ทางเลือก

| | A. บังคับ poll addr 1 เสมอ | B. เฟรม end-of-cycle ใหม่ (0x23) | C. ให้จอใช้ timeout |
|---|---|---|---|
| ความถูกต้อง | ได้ แต่เป็นข้อยกเว้นที่ซ่อนอยู่ | ✅ ชัดเจน | เดาเวลา — เปราะ |
| กระทบ CONTROL | น้อย | ต้องเพิ่ม ~30 บรรทัด | 0 |
| กระทบ P4 | 0 | ต้องแก้ `on_request()` | ต้องแก้ `on_request()` |
| ลบ coupling ที่ซ่อนอยู่ | ❌ ยิ่งฝังลึก | ✅ ลบถาวร | ❌ |
| ได้ metadata เพิ่ม | 0 | ✅ ได้ `seq` + `active_mask` + `poll_ms` ฟรี | 0 |
| ถ้า addr 1 เสียลากบัสตาย | ❌ **แก้ไม่ได้เลย** | ✅ แก้ได้ | ✅ |

### **ข้อเสนอ: B — เฟรม end-of-cycle บน address 0x23**

```
CONTROL → จอ (หลังอ่านครบทุกตัวในรอบ), 15 ไบต์ โครงเดียวกับ 0x20/0x22
23 10 00 00 00 03 06 <seq_lo> <seq_hi> <active_mask> <poll_ms/100> <ver> <rsv> crc_lo crc_hi
```

- ใช้ address `0x23` — ว่างอยู่ (0x20/0x21/0x22 ใช้แล้ว, `BUS_PROTOCOL.md §4.4` เตือนไว้ตรงนี้พอดี)
- ส่งทุกรอบ (2.5 วิ) เพิ่ม traffic 15 ไบต์/รอบ = **0.05 % ของ 4800 baud** — ไม่มีนัยสำคัญ
- **ให้ประโยชน์เกินกว่าที่ตั้งใจ:** จอได้ `seq` (ตรวจ gap ได้ครั้งแรก), ได้ `active_mask`
  (ไม่ต้องเดาว่า config เปลี่ยนแล้ว), ได้ `poll_interval_ms` จริง (แก้ปัญหา `stale_after_ms`
  ที่ต้อง sync สองที่ตามที่ `BUS_PROTOCOL.md §1` เตือนไว้)
- **migration ปลอดภัย:** ฝั่งจอเก็บกติกาเดิม (`01 03 0000`) ไว้เป็น fallback
  ถ้าไม่เห็นเฟรม 0x23 ภายใน 3 รอบ → กลับไปใช้กติกาเดิม
  ⇒ จอรุ่นใหม่ทำงานกับ CONTROL รุ่นเก่าได้ และกลับกัน

**ข้อนี้เป็น blocker ของ Phase 4 — ต้องทำเสร็จและทดสอบก่อนเปิดให้ปิด sensor ตัวใดตัวหนึ่งได้**

## 8.4 ลำดับการ apply config

```
[P4] ผู้ใช้แก้ใน SENSOR SETUP → กด Apply
  │  ตรวจในเครื่องก่อน: ไม่มี run ค้าง / mask ≥ 1 ตัว / addr ไม่ซ้ำ
  ▼
[P4→PC] @J1 {"type":"cmd","action":"sensor_config_apply","payload":{"mask":13,"addr":[1,2,3,4],"poll_ms":2500}}
  │
  ▼
[PC] ตรวจ: ไม่มี session ค้าง / ไม่มี report job running / ไม่มี cal ค้าง / COM-A ต่ออยู่
  │  ถ้าไม่ผ่าน → NACK code=CONFIG_LOCKED_* → จบ (P4 คง config เดิม)
  ▼
[PC→CONTROL] "X1,13,1,2,3,4,2500\n"   (คำสั่ง serial ใหม่ — ver,mask,addr×4,poll)
  │
  ▼
[CONTROL] ตรวจ: mask≠0, addr 1-247 ไม่ซ้ำ, poll 1000-10000
  │  เขียน Preferences (NVS) → อ่านกลับมายืนยัน → apply เข้า runtime
  ▼
[CONTROL→PC] "[cfg] applied ver=1 mask=13 addr=1,2,3,4 poll=2500"    (หรือ "[cfg] reject reason=...")
  │
  ▼
[PC] - ปิดไฟล์ CSV ปัจจุบัน แล้วเปิดไฟล์ใหม่ (mask เปลี่ยน = schema เปลี่ยน)
  │  - เขียน config event ลง measurement_events_*.jsonl
  │  - อัปเดต pc_state.json
  ▼
[PC→P4] ACK ok=true payload={"mask":13,"addr":[...],"poll_ms":2500}
  │
  ▼
[P4] rebuild UI จาก config ที่ **ตอบกลับมา** ไม่ใช่จากที่ตัวเองส่งไป
```

**กฎเหล็ก:** P4 rebuild UI จาก **ACK payload** เท่านั้น
ถ้า timeout / NACK → คง config เดิม 100% + ขึ้นเหตุผล

**ห้ามเปลี่ยน config เมื่อ:** มี P4 run active / มี PC session active / กำลัง cal /
มี report job running / กำลัง scan bus / กำลัง apply config อยู่แล้ว

---

# 9. ข้อเสนอ CSV migration

## 9.1 ปัญหาที่ต้องแก้ให้ได้ 3 ข้อ

1. ไฟล์เก่าต้องอ่านได้ด้วยเครื่องมือเก่าโดยไม่แตะเลย
2. เครื่องมือ **เก่า** ต้องไม่เผลออ่านไฟล์ **ใหม่** ผิด (**R3** — `len(p) >= 10` ผ่านหมด)
3. โฟลเดอร์นี้มี schema ปนกันอยู่แล้ววันนี้ (**R2**)

## 9.2 เปรียบเทียบทางเลือก

| | A. metadata header ในไฟล์เดิม | B. **ชื่อไฟล์ใหม่** | C. เพิ่มคอลัมน์ต่อท้ายไปเรื่อย ๆ |
|---|---|---|---|
| reader เก่าไม่อ่านผิด | ❌ `fh.readline()` ข้าม header ทิ้ง แล้วอ่านตำแหน่งต่อ | ✅ glob `water_log_*.csv` ไม่แมตช์ | ❌ **R3** |
| แยก schema ในโฟลเดอร์เดียว | ❌ | ✅ | ❌ |
| แก้ปัญหา R2 ที่มีอยู่แล้ว | ❌ | ✅ | ❌ |
| reader ใหม่ทำงานยาก | ต้องตรวจ header | ตรวจจากชื่อ + header | ต้องเดาจำนวนคอลัมน์ |
| ไฟล์เดิมยังใช้ต่อได้ | ✅ | ✅ (อ่านได้ทั้งสองแบบ) | ✅ |

## 9.3 **ข้อเสนอ: B + metadata header (ทำทั้งสองอย่าง)**

**ชื่อไฟล์ใหม่:** `water_data/ec_log_v2_YYYY-MM-DD.csv`

> ใช้ prefix `ec_log_v2_` ไม่ใช่ `water_log_v2_` — เพราะ `list_files()` และ
> `read_csv_rows()` ใช้ glob `water_log_*.csv` ซึ่งจะยัง **แมตช์** `water_log_v2_*.csv`
> การเปลี่ยน prefix ทั้งคำเป็นวิธีเดียวที่ทำให้เครื่องมือเก่า "มองไม่เห็น" ไฟล์ใหม่
> **โดยไม่ต้องแก้เครื่องมือเก่าเลย** — ซึ่งคือนิยามของ rollback ที่ปลอดภัย

**หัวไฟล์:**
```csv
# schema=ec_log/2
# created=2026-08-25T13:47:10+07:00
# max_sensors=4
# active_mask=7
# modbus_addr=1,2,3,4
# poll_interval_ms=2500
# control_fw=8.3-3EC
timestamp,seq,mono_ms,active_mask,EC1,T1,ok1,EC2,T2,ok2,EC3,T3,ok3,EC4,T4,ok4,flag
2026-08-25 13:47:10,12547,183248100,7,1385.5,20.6,1,302.1,20.5,1,78.3,20.7,1,nan,nan,0,
```

**กฎ:**
- `active_mask` อยู่ **ทุกแถว** (ไม่ใช่แค่หัวไฟล์) — ราคาคือ 1-2 ไบต์/แถว
  แต่ทำให้ไฟล์อ่านได้ด้วยตัวเองแม้ถูกตัดหัวออก และรองรับกรณีที่ mask เปลี่ยนกลางไฟล์
  (ซึ่งเราห้ามไว้แล้ว — แต่ defensive)
- **นอกจากนี้ยังเขียน config event ลง `measurement_events_*.jsonl` ด้วย** ทุกครั้งที่ mask
  เปลี่ยน — เพื่อให้รายงานตีความช่วงเวลาได้โดยไม่ต้องสแกน CSV ทั้งไฟล์
- ค่าที่ไม่มี = `nan` (**ไม่ใช่ `0.0` และไม่ใช่ค่าว่าง**)
  ⚠️ **สังเกต:** ปัจจุบัน `logger_3ec.py:244` แปลง `NaN` → `""` (ค่าว่าง)
  v2 ต้องเขียน `nan` ตรง ๆ — pandas/`float()` อ่านได้เป็น `nan` ถูกต้อง
- `ok` = 0/1 แยกจาก `nan` ชัดเจน: `nan,nan,0` = ไม่มีข้อมูล, `84.6,20.4,0` = **เป็นไปไม่ได้** (reject)
- **หนึ่งไฟล์ = หนึ่ง schema เสมอ** — mask เปลี่ยน → ปิดไฟล์ เปิดไฟล์ใหม่
  ชื่อไฟล์ชนกันได้ในวันเดียวกัน ⇒ ต่อท้าย `_a`, `_b`, … (`ec_log_v2_2026-08-25_b.csv`)

**reader ใหม่ (ทั้ง logger, ui, report) ต้อง:**
- ใช้ `csv.DictReader` — **เลิกอ่านตำแหน่งทั้งหมด**
- ตรวจ `# schema=` ถ้ามี ไม่มีก็ดูจากชื่อไฟล์แล้วเดา v1
- อ่านได้ทั้ง `water_log_*.csv` (v1, 10/11 คอลัมน์) และ `ec_log_v2_*.csv` (v2)
- v1 → normalize เป็นโครงเดียวกันโดยตั้ง `active_mask = 0b0111`, `seq = None`

**การกันปัญหา R2 ซ้ำ:** เสนอย้าย `logger.py` (ตัวเก่า 1 เซนเซอร์) และ
`report.py` / `web_dashboard.py` / `line_notifier.py` ไปไว้ใน `legacy/`
เพราะ `logger.py` ยังเขียน `water_log_YYYY-MM-DD.csv` ทับได้ทุกเมื่อ

---

# 10. ข้อเสนอ layout 1/2/3/4 เซนเซอร์

## 10.1 PC — `desktop_ui.py` (ส่วนที่ผู้เขียนรับผิดชอบ)

หน้าต่าง 1040×780 (ค่าปัจจุบัน) → เสนอ **1100×820**

```
┌────────────────────────────────────────────────────────────────────┐
│ ● ESP32 Water Monitor    live • 13:47:10 (2s)    ACTIVE 3/4 · 2 LIVE · 1 FAULT │
├────────────────────────────────────────────────────────────────────┤
│ ⚗ CALIBRATING EC#2 @ 84 µS/cm — waiting for a stable reading  ⟳   │ ← แถบ cal (ซ่อนได้)
├────────────────────────────────────────────────────────────────────┤
│  การ์ดตาม mask (ดูตารางด้านล่าง)                                     │
├────────────────────────────────────────────────────────────────────┤
│ ช่วง: [10m][1h][6h][24h][All]    P4: ● connected   ⬇PDF ⬇XLSX ⬇CSV │
├────────────────────────────────────────────────────────────────────┤
│ บันทึก: #1 ○  #2 ● REC  #3 ○  #4 ⊘DISABLED  | ดู session: [▼]      │
├────────────────────────────────────────────────────────────────────┤
│  กราฟตาม mask                                                       │
└────────────────────────────────────────────────────────────────────┘
```

| mask | การ์ด | กราฟ |
|---|---|---|
| 1 ตัว | 1 การ์ดเต็มความกว้าง ตัวเลข 48pt | 1 subplot เต็มพื้นที่ |
| 2 ตัว | 2 คอลัมน์ ตัวเลข 34pt | 2 subplot แชร์แกน x |
| 3 ตัว | 3 คอลัมน์ (เหมือนเดิมเป๊ะ) 26pt | 3 subplot (เหมือนเดิมเป๊ะ) |
| 4 ตัว | **2×2 grid** 26pt | 4 subplot + **`Notebook` 2 แท็บ**: `แยก` / `รวม (overlay)` |
| custom `0b1101` | เรียงเฉพาะที่ enabled ตามลำดับหมายเลข | เฉพาะที่ enabled — **ไม่สร้าง subplot ว่าง** |

**เหตุผลที่ 4 ตัวใช้แท็บแทน scroll:** matplotlib ใน Tk ที่ต้อง scroll ต้องห่อ `Canvas`
ที่มี scrollbar ซึ่งทำให้ `NavigationToolbar2Tk` (zoom/pan ที่ใช้อยู่) ทำงานเพี้ยน
แท็บได้ผลเดียวกันโดยไม่แตะ toolbar

**สถานะที่ต้องแยกให้เห็นบนการ์ด (ตรงกับ 8.2):**

| สถานะ | การ์ด | ป้าย |
|---|---|---|
| LIVE | ปกติ | `● LIVE` เขียว |
| NO DATA | ตัวเลขจาง | `● NO RESPONSE` เหลือง |
| FAULT | ตัวเลข `--` | `● SENSOR FAULT` แดง |
| DISABLED | การ์ดสีเทาทึบ ตัวเลข `⊘` | `DISABLED` เทา |
| CALIBRATING | ขอบการ์ดกะพริบ | `⚗ CALIBRATING` ฟ้า |

## 10.2 P4 — ข้อเสนอเชิงหลักการ (จะส่งเป็น `.md` แยก)

| หน้า | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Overview | การ์ดเดียว 1024×~380 | 2 คอลัมน์ 500 px | 3 คอลัมน์ 330 px (เดิม) | **2×2 grid 500×190** |
| Measure | 1 bay เต็มจอ | 2 bays | 3 bays (เดิม) | **2 หน้า × 2 bays + แถบเลื่อนแนวนอน** |
| Trend | 1 เส้น | 2 เส้น | 3 เส้น (เดิม) | 4 เส้น แกนเดียว + legend |

**เหตุผลที่ Measure 4 ตัวต้องเป็น "2 หน้า × 2 bays" ไม่ใช่ 2×2 หรือ 4 ช่องเรียง:**

- `sensor_bay.c` ปัจจุบันวาดหัววัด + บีกเกอร์ + อนิเมชันจุ่ม-ยก + ripple
  ความสูงที่ใช้จริงประมาณ 420-450 px — จอสูง 600 px หัก header/footer เหลือ ~480 px
  ⇒ **2×2 ทำให้แต่ละแถวเหลือ ~240 px ซึ่งไม่พอวาดอนิเมชัน** ต้องเขียน `sensor_bay` ใหม่ทั้งไฟล์
- 4 ช่องเรียงแนวนอน = ช่องละ 256 px กว้าง — ปุ่ม `Confirm immersed` (44 px สูง)
  จะเหลือความกว้างไม่พอใส่ข้อความ
- **2 หน้า × 2 bays: bay กว้าง 500 px สูงเต็ม — ใช้โค้ด `sensor_bay.c` เดิมได้ 100%**
  แลกกับผู้ใช้ต้องปัดเพื่อดูอีกสองตัว ซึ่งยอมรับได้เพราะการวัดทำทีละตัวอยู่แล้ว
- ต้องมีตัวบอกสถานะย่อของ bay ที่อยู่คนละหน้า (จุดเล็ก ๆ บน page indicator)
  ไม่งั้นผู้ใช้จะไม่รู้ว่าอีกหน้ามีค่าที่ STABLE รออยู่

**หน้า SENSOR SETUP (ใหม่ ใน Settings):**
```
SENSOR SETUP                                    Maximum supported: 4
──────────────────────────────────────────────────────────────────
 [✓] Sensor 01    Address [ 1 ]    ● SENSOR FAULT
 [✓] Sensor 02    Address [ 2 ]    ● LIVE     84.6 uS/cm
 [✓] Sensor 03    Address [ 3 ]    ● LIVE     85.4 uS/cm
 [ ] Sensor 04    Address [ 4 ]      DISABLED
──────────────────────────────────────────────────────────────────
 Poll interval  [ 2500 ] ms                          Active: 3 / 4
 
 [ Scan bus ]                    [ Apply changes ]   ← disabled ถ้า lock
 
 PC IS RECORDING SENSOR 2 — STOP THE SESSION BEFORE CHANGING CONFIG
```

---

# 11. แผนไฟล์แยกตามเฟส

## Phase 2 — core 4-capacity (legacy ต้องเหมือนเดิมทุกอย่าง)

| ไฟล์ | สถานะ | การเปลี่ยนแปลง |
|---|---|---|
| `ec_schema.py` | **ใหม่** | `MAX_SENSORS=4`, `SensorConfig`, `Row`, `read_any_csv()` (v1+v2), `write_v2_header()` — **ไฟล์เดียวที่รู้จัก schema** |
| `ec_features.py` | **ใหม่** | feature flags ทั้งหมด อ่านจาก env/ไฟล์ — `ECV2_FRAME`, `P4_BRIDGE`, `CSV_V2`, `CONFIG_APPLY` ทุกตัว default `False` |
| `logger_3ec.py` | แก้ | `parse()` → `parse_data()` + `parse_ecv2()`; `HEADER` → จาก `ec_schema`; `SessionMgr` → `MAX_SENSORS` |
| `calibration.py` | แก้ | `TRACE_HEADER` → dynamic; `range(3)` → `range(MAX_SENSORS)`; `read_all` → mask-aware |
| `desktop_ui.py` | แก้ | `read_range()` → `ec_schema.read_any_csv()`; layout 1-4 |
| `report_3ec.py` | แก้ | `read_csv_rows()` → `ec_schema.read_any_csv()`; หน้า PDF dynamic |
| `water_monitor_3ec.ino` | แก้ | `N_SENSORS` → `MAX_SENSORS 4` + `activeMask`; ECV2 frame หลัง `#if ENABLE_ECV2` |
| `tests/` | **ใหม่** | `test_schema.py`, `test_parse.py`, `test_session.py`, `mock_control.py` |
| **ไม่แตะ** | | `report.py`, `logger.py`, `web_dashboard.py`, `line_notifier.py` (→ ย้ายไป `legacy/`) |

**ไม่เปลี่ยนชื่อไฟล์ใน Phase 2** — ทำตอน Phase 5 พร้อม shim:
```python
# logger_3ec.py (หลัง rename)
from logger_ec import *          # noqa
import warnings
warnings.warn("logger_3ec.py ถูกเปลี่ยนชื่อเป็น logger_ec.py", DeprecationWarning)
```

## Phase 3 — PC↔P4 bridge

| ไฟล์ | สถานะ |
|---|---|
| `p4_bridge.py` | **ใหม่** — thread + NDJSON codec + dedupe + rate limit + reconnect |
| `p4_protocol.py` | **ใหม่** — นิยาม action/code/schema validation (ไม่มี I/O เลย → unit test ง่าย) |
| `pc_state.py` | **ใหม่** — เขียน `pc_state.json` atomically (`.tmp` + `os.replace`) |
| `event_log.py` | **ใหม่** — append `measurement_events_*.jsonl` + fsync + dedupe |
| `report_jobs.py` | **ใหม่** — `ThreadPoolExecutor(1)` + สถานะ accepted/running/complete/failed |
| `logger_3ec.py` | แก้ — สร้าง thread, ต่อ queue, ไม่แตะ COM-A จาก thread อื่น |
| `desktop_ui.py` | แก้ — อ่าน `pc_state.json`, แสดงสถานะ P4 |

## Phase 3b — **cal observer (ทำได้ก่อน bridge ไม่ต้องรอ)**

| ไฟล์ | สถานะ | หมายเหตุ |
|---|---|---|
| `control_events.py` | **ใหม่** | classifier บรรทัด `[cal]` / `[bus]` / `[scan]` / `[cfg]` / `[rtc]` → dataclass |
| `cal_state.py` | **ใหม่** | สถานะ cal สด → `cal_status.json` |
| `calibration.py` | แก้ | แยก `Calibrator` เป็น 2 ชั้น: `CalCore` (state machine ไม่มี I/O) + `ConsolePresenter` — **แก้ R6** |
| `logger_3ec.py` | แก้ | ป้อนทุกบรรทัดที่ไม่ใช่ DATA เข้า `control_events.classify()` |
| `desktop_ui.py` | แก้ | แถบ CAL + แถบสีบนกราฟช่วง cal |
| `report_3ec.py` | แก้ | หน้า calibration รองรับ entry ที่ `origin="p4"` (ซึ่งไม่มี trace CSV) |

**นี่คือกลุ่มที่ตอบโจทย์ "กด cal บนจอ แล้ว PC ขึ้น cal ด้วย" ครบถ้วน โดยไม่ต้องมี COM-B เลย**

## Phase 4 — configurable sensors

`sensor_config.py` (ใหม่) · `.ino`: `Preferences` + คำสั่ง `X` + เฟรม `0x23` ·
`p4_protocol.py`: `sensor_config_apply` · `desktop_ui.py`: layout ตาม mask

## Phase 5 — advanced

`report_request` async · `bus_scan_request` · offline replay · rename ไฟล์ + shim ·
ย้าย threshold มาฝั่ง PC · (ทางเลือก) Wi-Fi/WebSocket

## 11.4 สิ่งที่ต้องขอให้ฝั่ง P4 ทำ (จะส่งเป็น `P4_CHANGES_REQUESTED.md`)

| # | เรื่อง | ไฟล์ | เฟส | ความสำคัญ |
|---|---|---|---|---|
| P-1 | รับเฟรม `0x23` end-of-cycle เป็นตัวตัดรอบ (fallback = กติกาเดิม) | `ec_rs485.c` `on_request()` | 4 | **blocker ของ Phase 4** |
| P-2 | ใช้ `poll_interval_ms` จากเฟรม `0x23` แทนค่า hard-code 2500 | `ui_model.c` `ui_model_init()` | 4 | สูง |
| P-3 | `pc_bridge.c/.h` — NDJSON บน UART0 + prefix `@J1 ` + mutex กับ ESP_LOG | ใหม่ | 3 | สูง |
| P-4 | flat-key JSON scanner (ไม่ใช้ cJSON ไม่จองฮีป) | ใหม่ `nd_scan.c` | 3 | สูง |
| P-5 | `CONFIG_EC_PC_BRIDGE` (default `n`) — rollback switch | `Kconfig.projbuild` | 3 | สูง |
| P-6 | ส่ง measure event ออก bridge (hook ที่ `measure_policy` emit อยู่แล้ว) | `ui_model.c` | 3 | กลาง |
| P-7 | `EC_SENSOR_COUNT` 3 → `EC_MAX_SENSORS 4` + `active_mask` + แก้ `_Static_assert` | `ec_packet.h` | 4 | สูง |
| P-8 | Overview/Trend/Measure รองรับ 1-4 + แยก DISABLED / FAULT | `screen_*.c` | 4 | สูง |
| P-9 | หน้า `SENSOR SETUP` | `screen_settings.c` | 4 | สูง |
| P-10 | `CONFIG_LOG_COLORS=n` | `sdkconfig.defaults` | 3 | กลาง |
| P-11 | แสดง `PC NOT CONNECTED` / `REQUEST NOT APPLIED` / `LOCAL MEASUREMENT STILL AVAILABLE` | `ui_theme.c` + ทุกหน้า | 3 | สูง |
| P-12 | **บั๊กเล็ก:** `config EC_PROBE_ART_DETAILED` อยู่นอกบล็อก `menu`/`endmenu` ใน `Kconfig.projbuild` | `Kconfig.projbuild:32` | 0 | ต่ำ |

---

# 12. Test matrix

## A. Legacy regression (ต้องผ่านก่อนทุกเฟส)

| # | ทดสอบ | วิธี | เกณฑ์ผ่าน |
|---|---|---|---|
| A1 | `parse("DATA,1385.5,20.6,302.1,20.5,78.3,20.7,111")` | unit | ได้ 9 ค่าเท่าเดิมทุกตัว |
| A2 | อ่าน CSV เก่าทั้ง 6 ไฟล์ใน `water_data/` | unit | จำนวนแถวและค่าตรงกับก่อน refactor bit-for-bit |
| A3 | สร้างรายงานจาก `sessions_3ec.json` เดิม | integration | PDF 9 หน้า ตัวเลขสถิติตรงกับ `reports/run_3ec_20260807_104548.pdf` |
| A4 | `desktop_ui` mask=0b0111 | manual + screenshot diff | หน้าตาเหมือนเดิม 100% |
| A5 | mock CONTROL ส่ง DATA 1000 บรรทัด | integration | CSV 1000 แถว ไม่ขาด ไม่ซ้ำ |
| A6 | ถอด USB กลางทาง | manual | reconnect ภายใน ~5 วิ CSV ต่อได้ |
| A7 | `flag=CAL` ยังถูกข้ามในรายงาน | unit | เหมือนเดิม |

## B. Config scenarios

| # | ทดสอบ | เกณฑ์ผ่าน |
|---|---|---|
| B1-B4 | mask = 1 / 3 / 7 / 15 ตัว | UI/CSV/report สร้างครบตามจำนวน ไม่มี panel ว่าง |
| B5 | mask `0b1101` (ปิด #2) | #2 เป็น `DISABLED` ไม่ใช่ error; CSV คอลัมน์ #2 = `nan,nan,0` |
| B6 | **mask `0b1110` (ปิด #1)** | **จอไม่ขึ้น STALE** ← ทดสอบ R1 โดยตรง |
| B7 | sensor enabled แต่ไม่ตอบ 1 ครั้ง | `NO RESPONSE` เหลือง |
| B8 | sensor enabled ไม่ตอบ 3 ครั้งติด | `SENSOR FAULT` แดง |
| B9 | apply config ขณะมี PC session | NACK `CONFIG_LOCKED_SESSION`, config เดิมไม่ขยับ |
| B10 | apply config ขณะมี P4 run | NACK `CONFIG_LOCKED_RUN` |
| B11 | apply config ขณะ cal | NACK `CONFIG_LOCKED_CAL` |
| B12 | CONTROL reboot หลัง apply | อ่าน NVS แล้วได้ mask เดิม |
| B13 | addr ซ้ำกัน (1,2,2,4) | CONTROL reject, PC ส่ง NACK |
| B14 | mask = 0 | reject ทุกชั้น |
| B15 | mask เปลี่ยน → CSV ปิดไฟล์เปิดไฟล์ใหม่ | ไฟล์ใหม่มี header ใหม่ ไฟล์เก่าไม่ถูกแตะ |

## C. Command protocol

| # | ทดสอบ |
|---|---|
| C1 | `session_start` valid → ACK มี `session_id` |
| C2 | `session_start` ซ้ำ → NACK `SESSION_ALREADY_ACTIVE` |
| C3 | ถอด COM-B ระหว่างรอ ACK → P4 ขึ้น `PC NOT CONNECTED`, **CSV ฝั่ง PC ไม่ขาดแม้แถวเดียว** |
| C4 | เสียบ COM-B กลับ → reconnect + replay คิว event, ไม่มี event ซ้ำ |
| C5 | ส่ง `request_id` ซ้ำ → PC ตอบ ACK เดิม ไม่ execute ซ้ำ (ตรวจจำนวน session ในไฟล์) |
| C6 | ส่ง `event_id` ซ้ำ 3 ครั้ง → `.jsonl` มีบรรทัดเดียว |
| C7 | ACK timeout → P4 คง state เดิม |
| C8 | JSON เสีย 100 บรรทัดติด → PC ไม่ crash, `frames_dropped_parse` = 100 |
| C9 | บรรทัดยาว 4096 ไบต์ | ทิ้ง + นับ, บรรทัดถัดไปยัง parse ได้ |
| C10 | `action:"nuke_everything"` → NACK `UNKNOWN_ACTION` |
| C11 | `"v":99` → NACK `UNSUPPORTED_VERSION` |
| C12 | PC restart ระหว่าง P4 รอ ACK → P4 timeout แล้ว retry ด้วย id เดิม → ได้ ACK |
| C13 | P4 restart → `boot_id` เปลี่ยน → PC ยอมรับ `request_id` ที่เริ่มใหม่จาก 1 |
| C14 | `report_request` → ACK `accepted` ทันที (< 200 ms) แล้ว state ไล่ running → complete |
| C15 | report ล้มเหลว (ไม่มีข้อมูล) → state `failed` + `code:NO_DATA_IN_WINDOW` |
| C16 | ส่ง 50 เฟรมใน 1 วิ → NACK `RATE_LIMITED` ตั้งแต่เฟรมที่ 11 |
| C17 | ESP_LOG พิมพ์ระหว่างส่งเฟรม | เฟรมไม่ถูกตัดกลาง (ทดสอบด้วยการเปิด log level VERBOSE) |

## D. Performance

| # | ทดสอบ | เกณฑ์ |
|---|---|---|
| D1 | ยิง command 10/วิ นาน 10 นาที | `seq` ของ RS485 poll ไม่หายแม้รอบเดียว (`ok/err` ตรงกับ baseline) |
| D2 | เดียวกัน | LVGL lock สูงสุด **< 30 ms** (baseline ปัจจุบัน 11 ms) |
| D3 | สร้าง report 8 หน้าระหว่าง logging | **ไม่มีแถว CSV หาย** — เทียบจำนวนแถวกับ `seq` ของ ECV2 |
| D4 | เดียวกัน | UI ยังตอบสนอง, `p4_bridge` ยังส่ง state ทุก 3 วิ |
| D5 | soak 2 ชั่วโมง | heap ก้อนใหญ่สุดฝั่ง P4 ไม่ลด, **ไม่ PANIC** ← ต้องผ่าน **ก่อน** merge Phase 3 |
| D6 | P4 ส่ง cmd ตอน PC ไม่ตอบ | P4 ไม่บล็อก — LVGL ยังลื่น (วัดด้วย frame time) |
| D7 | CSV fsync | ดึงปลั๊ก PC → เปิดไฟล์แล้วแถวสุดท้ายสมบูรณ์ |

## E. Rollback

| # | ทดสอบ |
|---|---|
| E1 | ทุก feature flag = `False` → พฤติกรรมเหมือน git tag `baseline-3ec` ทุกประการ (A1-A7 ผ่านหมด) |
| E2 | `ECV2_FRAME=False` → CONTROL ส่งแค่ `DATA,` เหมือนเดิม |
| E3 | `P4_BRIDGE=False` → ไม่มีใครเปิด COM-B, `idf.py monitor` ใช้ได้ปกติ |
| E4 | `CSV_V2=False` → เขียน `water_log_*.csv` แบบเดิม |
| E5 | `CONFIG_EC_PC_BRIDGE=n` บน P4 → เฟิร์มแวร์ไม่ส่ง `@J1` เลย |
| E6 | flash เฟิร์มแวร์ CONTROL เก่ากลับ + logger ใหม่ | ยังทำงานได้ (parser รับ DATA ได้) |
| E7 | flash CONTROL ใหม่ + logger เก่า | ยังทำงานได้ (DATA frame ยังถูกส่งเสมอ) |
| E8 | ไฟล์ CSV v1 เดิมทั้ง 6 ไฟล์ | checksum ไม่เปลี่ยนหลังรัน Phase 2-5 ทั้งหมด |

---

# 13. แผน rollback

## 13.1 ก่อนเริ่ม — สิ่งที่ต้องมี (blocker)

1. **`git init` ใน `C:\MOF_NanoTec\test_realtime`** — ตอนนี้ยังไม่มี git
   (มีแต่ `_backup_before_calreport/` ซึ่งเป็นสำเนามือ ไม่มีประวัติ ย้อนได้จุดเดียว)
   ⇒ commit สถานะปัจจุบัน แล้ว tag `baseline-3ec-20260825`
2. **สำรอง `water_data/`, `sensor_1..3/`, `reports/`, `sessions_3ec.json`,
   `calibration_log.json`, `calibration_data/` ออกนอกโฟลเดอร์โปรเจกต์**
   (git ไม่ควรถือข้อมูลดิบ — ใส่ `.gitignore`)
3. ฝั่ง P4: มี `_baseline_20260819_2332/` อยู่แล้ว + git repo ที่ `ESP-IDF/.git`
   ⇒ tag เพิ่มก่อนเริ่ม
4. ฝั่ง CONTROL: บันทึก `.ino` ปัจจุบัน + **เก็บไฟล์ `.bin` ที่ compile แล้ว**
   เพื่อ flash กลับได้โดยไม่ต้องหวังว่า toolchain จะให้ผลเหมือนเดิม

## 13.2 กลไก rollback 3 ชั้น

| ชั้น | กลไก | เวลาที่ใช้ย้อน |
|---|---|---|
| **1. Runtime** | `ec_features.py` — แก้ค่า `False` แล้วรีสตาร์ท logger | **< 10 วินาที** |
| **2. Build** | `CONFIG_EC_PC_BRIDGE=n`, `CONFIG_EC_UI_V2=n`, `#define ENABLE_ECV2 0` | ~2 นาที (build+flash) |
| **3. Source** | `git checkout baseline-3ec-20260825` | ~1 นาที |

**ชั้นที่ 1 คือชั้นที่ต้องใช้จริงตอนอยู่หน้างานแล้วอะไรพัง** — ออกแบบให้ flag ทุกตัว
อ่านตอน **start** ของ logger ไม่ใช่ตอน import ⇒ แก้ไฟล์แล้วรีสตาร์ทพอ

```python
# ec_features.py — ทุกตัว default False จนกว่าจะผ่านการทดสอบของเฟสนั้น
FEATURES = {
    "ECV2_FRAME":     False,   # Phase 2 — parse/emit ECV2
    "CSV_V2":         False,   # Phase 2 — เขียน ec_log_v2_*.csv
    "CAL_OBSERVER":   False,   # Phase 3b — ฟัง [cal]/[bus] จาก COM-A
    "P4_BRIDGE":      False,   # Phase 3 — เปิด COM-B
    "CONFIG_APPLY":   False,   # Phase 4 — ยอมรับ sensor_config_apply
    "REPORT_ASYNC":   False,   # Phase 5
}
```

## 13.3 กฎระหว่างพัฒนา

- **ห้ามลบโค้ดเก่าจนกว่า regression A1-A7 จะผ่านครบ 3 รอบติด**
- **ห้ามแก้ไฟล์ที่ไม่เกี่ยวกับเฟสนั้น** — แต่ละ PR แตะเฉพาะไฟล์ที่ระบุในหัวข้อ 11
- ทุกเฟสต้องส่ง: ไฟล์ที่เปลี่ยน / คำสั่ง build+test / output จริง / วิธี rollback
- ข้อมูลดิบเก่า (`water_data/*.csv`) **ห้ามเขียนทับหรือแปลงในที่**
  — เครื่องมือใหม่อ่าน v1 ได้ แต่ห้ามเขียนกลับ

---

# 14. คำถามที่ต้องให้คุณตัดสินใจ

| # | คำถาม | ตัวเลือก | **ที่ผู้เขียนแนะนำ** |
|---|---|---|---|
| **Q1** | เห็นด้วยกับการ **แยก cal ออกจาก NDJSON** (cal ไปตรงผ่านบัสเหมือนเดิม, PC เป็นผู้ฟัง) ไหม | (a) เห็นด้วย (b) ให้ cal ผ่าน PC ตามโจทย์เดิม | **(a)** — ได้ผลตามโจทย์ครบ โดยไม่ต้องแตะ firmware เลย และไม่ย้ายด่านความปลอดภัยออกจาก CONTROL |
| **Q2** | ให้ทำ **Phase 3b (cal observer) ก่อน Phase 3 (bridge)** ไหม | (a) ทำ 3b ก่อน (b) ตามลำดับเดิม | **(a)** — 3b แก้ Python 2 ไฟล์ ได้ผลลัพธ์ที่คุณขอทันที ไม่ต้องรออะไรเลย |
| **Q3** | Transport ฝั่ง P4 | (a) CH343 + prefix `@J1 ` (b) TinyUSB CDC (c) UART2 + dongle แยก | **(a)** — บั๊ก PANIC ที่ยังหาไม่เจอทำให้ตอนนี้ไม่ควรเพิ่ม USB stack; (c) เป็นแผนสำรองถ้า framing มีปัญหา |
| **Q4** | JSON parser บน P4 | (a) flat-key scanner เขียนเอง 0 heap (b) `espressif/cjson` | **(a)** — โปรเจกต์นี้มีประวัติ heap แตกเป็นเสี่ยงจนจอดับ |
| **Q5** | ชื่อไฟล์ CSV v2 | (a) `ec_log_v2_*.csv` (b) `water_log_v2_*.csv` (c) header อย่างเดียว | **(a)** — (b) ยังแมตช์ glob `water_log_*` ของเครื่องมือเก่า; (c) ไม่กัน R3 |
| **Q6** | ตัวตัดรอบ polling (R1) | (a) เฟรม `0x23` ใหม่ (b) บังคับ poll addr 1 เสมอ | **(a)** — ได้ seq + mask + poll_ms ฟรี และแก้กรณีหัววัด #1 ที่ลากบัสตาย |
| **Q7** | `logger.py` / `report.py` / `web_dashboard.py` / `line_notifier.py` (ชุดเก่า 1 เซนเซอร์) | (a) ย้ายไป `legacy/` (b) ลบ (c) เก็บไว้ที่เดิม | **(a)** — `logger.py` เขียนชื่อไฟล์ชนกับ `logger_3ec.py` และทำข้อมูลปนไปแล้ว 2 วัน (R2) |
| **Q8** | Measure 4 เซนเซอร์บน P4 | (a) 2 หน้า × 2 bays (b) 2×2 grid (c) แท็บ | **(a)** — ใช้ `sensor_bay.c` เดิมได้ 100%, (b) ต้องเขียน bay ใหม่ทั้งไฟล์ |
| **Q9** | `git init` ใน `test_realtime` ก่อนเริ่ม | (a) ให้ผู้เขียนทำ (b) คุณทำเอง (c) ไม่ใช้ git | **(a)** — เป็น blocker ของ rollback ชั้นที่ 3 |
| **Q10** | หน้าต่าง `desktop_ui.py` ขยายจาก 1040×780 → 1100×820 ได้ไหม | (a) ได้ (b) ต้องคงขนาดเดิม | **(a)** — โหมด 4 เซนเซอร์ต้องการพื้นที่เพิ่ม |
| **Q11** | จอ PC ที่ใช้จริงเป็นจอสัมผัสด้วยไหม (มีผลกับขนาดปุ่ม/ระยะกด) | (a) เมาส์อย่างเดียว (b) สัมผัสด้วย | ต้องการคำตอบ — ถ้า (b) จะปรับปุ่มเป็นอย่างน้อย 44×44 px และเพิ่มระยะห่าง |
| **Q12** | หัววัด #4 ที่จะเพิ่ม มีของจริงแล้วหรือยัง / จะใช้ Modbus address เท่าไร | — | ถ้ายังไม่มี จะทดสอบ Phase 4 ด้วย mock CONTROL ทั้งหมด |
| **Q13** | หัววัด #1 ที่เสีย จะซ่อม/เปลี่ยน หรือจะปิดถาวรด้วย mask | (a) ซ่อม (b) ปิดด้วย mask ไปก่อน | ถ้า (b) → **B6 (mask `0b1110`) กลายเป็น test case ที่สำคัญที่สุดของโปรเจกต์** |
| **Q14** | เริ่ม Phase ไหนก่อน หลังอนุมัติ | (a) 3b เท่านั้น (b) 2 + 3b (c) ตามลำดับ 2→3→4→5 | **(b)** — Phase 2 ไม่เปลี่ยนพฤติกรรมอะไรเลย (flags ปิดหมด) แต่ปูทางไว้ครบ และ 3b ให้ผลที่คุณเห็นได้ทันที |

---

# 15.

**No code has been changed. Waiting for approval.**

ยังไม่มีไฟล์ใดในทั้งสอง path ถูกแก้ไข สร้าง หรือลบ
เอกสารฉบับนี้เป็นผลของการอ่านอย่างเดียว (`cat` / `grep` / `sed -n`)
