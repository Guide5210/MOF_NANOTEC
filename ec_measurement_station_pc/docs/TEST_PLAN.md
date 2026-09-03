# แผนการทดสอบ — วัดอะไร และ **ไม่ได้** วัดอะไร

```bat
run_tests.bat                  :: unit ทั้งหมด (~36 วินาที)
run_e2e.bat                    :: 13 ฉากผ่าน bridge จริง (~40 วินาที)
```

หรือ: `python -m unittest discover -s tests -t tests -q` (stdlib ล้วน ไม่ต้องมี pytest)

## เทสต์ที่มี

| ไฟล์ | จำนวน | ยืนยันอะไร |
|---|---|---|
| `test_protocol.py` | 25 | สคีมา NDJSON · การ normalize ชื่อ field · เฟรมพังไม่โยน exception |
| `test_display_mask.py` | | เครื่องสถานะ mask ครบทุกทางเดิน · `0xFF` ไม่กลายเป็น 7 |
| `test_event_dedup.py` | | ต่อท้ายอย่างเดียว · กันซ้ำ · กันซ้ำรอดหลังเปิดโปรแกรมใหม่ |
| `test_ports.py` | | แยก CONTROL / P4-log / P4-bridge ด้วย VID:PID |
| `test_p4_bridge.py` | 22 | ลิงก์ · รีบูต · NACK · ทนขยะ · `_guard` · state |
| `test_pc_state.py` | 23 | liveness 3 ระดับ · `rec_status` ทุกรูปแบบ · นับแถว CSV · `active_mask_assumed` |
| `test_no_legacy_mutation.py` | 14 | **ห้ามแตะ legacy** — ตรวจ 3 ชั้น |
| `test_logger_unaffected.py` | 3 | จังหวะ CSV ที่ 2.5 s · เกณฑ์ gap ≤ 3.0 s |
| `test_theme.py` | 16 | token ตรง `ui_tokens.h` · Series 04 ยังสงวน · amber/crimson ไม่เป็นสีเส้น |
| `test_view_model.py` | 24 | เงื่อนไขการยอมรับ P1-B ทั้งสี่ข้อ |

## เทสต์ที่ผู้ใช้สั่งเป็นพิเศษ

### 1. `test_no_legacy_mutation` — สามชั้น

| ชั้น | วิธี | จับอะไรที่ชั้นอื่นจับไม่ได้ |
|---|---|---|
| **static** | ไล่ AST ทุกไฟล์ใน `ecstation/` | เส้นทางเขียนไฟล์ที่เทสต์ไม่เคยเดินผ่าน |
| **dynamic** | สร้าง legacy จำลอง → `sha256+size+mtime` ทุกไฟล์ → รัน bridge เต็มรูปแบบ → เทียบใหม่ | พฤติกรรมจริงตอนรัน ที่การอ่านโค้ดมองไม่เห็น |
| **guard** | `config` ที่ชี้ `data_dir` เข้าไปใน legacy ต้องถูกปัดตก | config พิมพ์ผิดบรรทัดเดียว |

static ห้าม: `open(mode=w/a/x/+)` · `remove` `unlink` `rename` `rmtree` `move` …
· `import subprocess/shutil` · สตริง `test_realtime`/`MOF_NanoTec` ฝังในโค้ด
· อ้างถึง `report_3ec` `report_jobs` `calibration` `matplotlib` `openpyxl` `pandas`
โมดูลเดียวที่เขียนไฟล์ได้คือ `event_log.py` (และ `config.py` สร้าง `data_dir` ของตัวเอง)

dynamic สร้าง legacy จำลองครบทุกชนิดที่สั่งห้ามแตะ: source `.py` 6 ไฟล์ ·
`docs/SYSTEM.md` · `ec_ui_config.json` · `sessions_3ec.json` · `rec_status.json` ·
CSV 50 แถว · `reports/*.pdf` · `reports/*.xlsx`  แล้วยืนยันว่าไม่มีไฟล์
หรือโฟลเดอร์ใหม่โผล่ และทุกอย่างที่เขียนตกอยู่ใน `data/events/` เท่านั้น

### 2. `test_reading_saved_field_names`

อยู่ใน `test_protocol.py` + `test_p4_bridge.py`:
ระเบียนต้องมีครบ 11 field ตามสัญญา · คีย์กำกวม (`ec` `value` `ec_value`
`reading` `ec_val`) ต้องถูกปฏิเสธ · เฟรมจริงจากเฟิร์มแวร์ (`ec_us_cm` + `ts_ms`)
ต้องรับได้และถูก normalize เป็น `stable_ec_us_cm` + `device_mono_ms`

### 3. `test_logger_unaffected`

| ฉาก | ผล (รันจริง 28 ส.ค. 2026) |
|---|---|
| baseline ไม่มี bridge | median 2.497 · p95 2.504 · **max 2.504 s** · ช้า 0 แถว |
| heartbeat ปกติ | median 2.499 · p95 2.563 · **max 2.563 s** · ช้า 0 · event 0.4/s |
| flood บรรทัดพัง | median 2.500 · p95 2.501 · **max 2.501 s** · ช้า 0 · **2,396 บรรทัด/s · พัง 21,390 เฟรม · event 342/s** |

ทุกฉาก: seq ซ้ำ 0 · ขาด 0 · exception 0 · `bridge_error` = None

ปรับจำนวนแถวได้ด้วย `EC_LOGGER_ROWS` / `EC_LOGGER_ROWS_BASE`

## เงื่อนไขการยอมรับของ P1-B

| # | เทสต์ | ยืนยัน |
|---|---|---|
| A1 | `TestMaskTransition` | `7→6→2→7` ทั้ง split และ overlay — การ์ดตรง · เส้นตรง · `artist_count()` เท่ากับจำนวนที่ควรมี (ตัวเลขนี้คือที่ ghost ซ่อนตัว) · legend ไม่ค้าง · **สีเส้นไม่เลื่อนหลังซ่อนแล้วเปิดกลับ** · mask ว่างวาดเปล่าไม่ใช่แผงพัง |
| A2 | `TestHiddenRawIntegrity` | sha256 ของ CSV ไม่เปลี่ยนหลังสลับ mask · เซนเซอร์ที่ซ่อนยังมีค่าและสถานะจริง (ไม่ใช่ `DISABLED`) · ไม่มีการ์ดไหนแสดง `0.0` แทน · Engineering view เผยพร้อมป้าย · `view_model.py` / `series_painter.py` ไม่มีเส้นทางเขียนไฟล์เลย |
| A3 | `TestEventDedup` | `reading_saved` id เดิม 100 ครั้ง → แถวใน UI 1 · บรรทัดใน JSONL 1 · แสดง `stable_ec_us_cm` ไม่ใช่ค่าสด · `event_id` ไม่โผล่บน dashboard แต่ยังเก็บไว้ให้ Diagnostics |
| A4 | `TestP4StateLoss` | จอหลุด → mask เดิม + ป้าย `P4 OFFLINE — showing last HMI selection` · **ไม่รีเซ็ตเป็นครบทุกตัว ไม่ซ่อนการ์ด** · "ไม่เคยได้ mask เลย" ต้องไม่ใช้ป้ายเดียวกัน · ประวัติเหตุการณ์รอดข้ามการหลุด · ต่อกลับแล้วตามจอต่อ |

## ขอบเขต — สิ่งที่เทสต์เหล่านี้ **ไม่ได้** พิสูจน์

1. **ไม่ได้วัด UART จริง** — วัดการแย่ง GIL/CPU ในโปรเซสเดียวกัน ซึ่งเป็นความเสี่ยง
   จริงของสถาปัตยกรรมนี้ แต่บนสายจริงที่ไม่มี flow control ถ้าลูปอ่านช้า
   **ไบต์จะถูกทิ้ง ไม่ใช่แค่มาช้า** — ต้องยืนยันกับฮาร์ดแวร์จริง
2. **"seq ซ้ำ/ขาด" เป็น seq ที่ PC ใส่เอง** — เฟรม `DATA,` ของ CONTROL ปัจจุบัน
   ไม่มี seq จึงตรวจการหายของแพ็กเก็ตบนสายไม่ได้จนกว่าจะมีเฟรม ECV2
3. **ไม่ได้ทดสอบกับจอจริง** — `tools/e2e_mock.py` เดินผ่าน TCP + เธรด + ไฟล์จริง
   แต่ตัวจอเป็นของจำลอง
4. **ไม่ได้ทดสอบพิกเซล** — เทสต์ทั้งหมดยิงที่ `view_model` / `series_painter`
   ซึ่งเป็นชั้นที่ตัดสินใจ  การจัดวางจริงบน Tk ตรวจด้วยภาพหน้าจอ
   (`docs/UI.md`) ไม่ใช่ด้วยเทสต์อัตโนมัติ

## e2e 13 ฉาก (`tools/e2e_mock.py`)

| ฉาก | ตัวชี้วัด |
|---|---|
| `normal` | มี event · ลิงก์ ONLINE |
| `mask-change` | เขียน `DISPLAY_MASK_CHANGED` |
| `mask-boot` | `0xFF` ไม่ถูกนับเป็นค่า แล้วค่อยได้ `INITIAL` |
| `mask-invalid` | mask 0 ถูกปฏิเสธ · บิตนอกช่วงถูกตัด |
| `reboot` | นับรีบูต + เขียน `P4_REBOOT` |
| `disconnect` | **ลำดับ ONLINE → OFFLINE → ONLINE** (แค่ "เคยเห็น OFFLINE" ไม่นับ เพราะตอนสตาร์ตก็ OFFLINE) |
| `dup` | `event_id` ซ้ำ เก็บครั้งเดียว |
| `malformed` | ทิ้งขยะ ≥4 เฟรม แล้ว hb ถัดไปยังรับได้ (ฉากนี้ตั้งใจไม่มีเฟรมดีเลย) |
| `queue-full` | `queued` ถึง 32 แล้วยังทำงานต่อ |
| `heap-leak` | อ่าน `heap`/`heap_big` ได้ |
| `cmd` | NACK ครบ + เขียน `CMD_REJECTED` |
| `flood` | รับ >20 เฟรม · ไม่มี error |
| `soak` | มี event · ไม่มี error |

`--speed N` ย่อเวลารอของฉากลง N เท่า และ **ย่อ `offline_after_s` ตามอัตโนมัติ**
ไม่งั้นฉาก `disconnect` จะผ่านเพราะ "ยังไม่ทันหมดเวลา" ไม่ใช่เพราะถูกต้อง
