# NDJSON v1 — สคีมาทุกเฟรม

หนึ่งบรรทัด = หนึ่ง JSON object · UTF-8 · ยาวไม่เกิน **512 ไบต์รวม `\n`**
บรรทัดที่ยาวเกิน / ไม่ใช่ JSON / ไม่ใช่ object ถูกทิ้งพร้อมนับ ไม่ทำให้ลิงก์ตก

`parse_line()` **ไม่เคยโยน exception** — คืน `ParseResult(ok=False, reason=...)` เสมอ

---

## P4 → PC

### `hb` — heartbeat

```json
{"v":1,"type":"hb","boot_id":"p4-c96e-a820","ts_ms":182345,"queued":0,
 "link":"online","heap":328399,"heap_big":253952,"display_mask":7}
```

| field | ชนิด | หมายเหตุ |
|---|---|---|
| `boot_id` | str ≤32 | เปลี่ยน = จอรีบูต → นับ + เขียน `P4_REBOOT` |
| `ts_ms` | int ≥0 | เวลานับจากบูตของจอ (รับ `device_mono_ms` ได้ด้วย) |
| `queued` | int | event ค้างในคิวของจอ (เต็มที่ 32) |
| `link` | `online`/`offline` | **จอมองเห็น PC หรือไม่** ไม่ใช่สถานะของจอเอง |
| `heap`, `heap_big` | int | ไว้ดู memory leak |
| `display_mask` | int | **`255` = จอยังไม่ได้ตั้งค่า → `None` ห้ามตีความเป็น 7** |

### `event` — `reading_saved`

รูปแบบบนสาย (จาก `pc_bridge.c:461-471`):

```json
{"v":1,"type":"event","event_id":"p4-c96e-a820-000017","boot_id":"p4-c96e-a820",
 "event":"reading_saved","sensor":2,"ec_us_cm":1146.0,"temperature_c":20.6,
 "tolerance_us_cm":11.5,"stable_for_ms":15000,"after_link_error":false,
 "ts_ms":182345,"wall":"28 Aug 2026  12:00:00"}
```

**ชื่อบนสาย ≠ ชื่อในระเบียนที่เก็บ** — ตัวแปลงรับได้ทั้งสองแบบแล้ว normalize:

| บนสายวันนี้ | ในระเบียนที่เก็บ (สัญญา) |
|---|---|
| `ec_us_cm` หรือ `stable_ec_us_cm` | `stable_ec_us_cm` |
| `ts_ms` หรือ `device_mono_ms` | `device_mono_ms` |

ระเบียนที่เขียนลง JSONL ต้องมีครบ 11 field:
`v · type · event_id · boot_id · event · sensor · stable_ec_us_cm ·
temperature_c · tolerance_us_cm · stable_for_ms · device_mono_ms`

**คีย์ EC ที่ถูกปฏิเสธเสมอ:** `ec` · `value` · `ec_value` · `reading` · `ec_val`
กำกวมเกินกว่าจะเดาว่าเป็นค่าที่นิ่งแล้วหรือค่าดิบ — ผลการวัดที่ผู้ใช้กดยืนยัน
ต้องไม่ถูกเดา  ถ้ามีทั้ง `stable_ec_us_cm` และ `ec_us_cm` แล้วค่าต่างกัน → ปฏิเสธ

`sensor` เป็น 1-based (`0` = ไม่ระบุ / ทั้งระบบ มาจาก `sensor + 1` ของ `-1`)

### `event` — เหตุการณ์บริบทอื่น

`STABILITY_REACHED` · `STABILITY_LOST` · `LINK_ERROR` … ต้องมี
`event_id · boot_id · event · ts_ms` เป็นอย่างน้อย · `ec_us_cm` ใส่มาก็ได้

### `cmd` — คำขอจากจอ

```json
{"v":1,"type":"cmd","request_id":7,"boot_id":"p4-c96e-a820",
 "action":"recording_start","sensor":0,"ts_ms":182400,"payload":{}}
```

**P1 ตอบ NACK ทุกคำสั่ง** พร้อมเขียน `CMD_REJECTED` ลง log

---

## PC → P4

### `state` — ทุก 3 วินาที

```json
{"v":1,"type":"state","seq":42,"pc_online":true,"recording":true,
 "session_mask":6,"active_mask":7,"cal_busy":false,"csv_rows":34560,
 "sample_id":"CALF-20 B3","active_mask_assumed":true}
```

`sample_id` ตัดที่ 23 ตัวอักษร · `active_mask_assumed` บอกว่า `active_mask`
เป็นค่าสมมติ เพราะ CONTROL ยังเป็น `#define N_SENSORS 3` และไม่มี NVS
จึงไม่มีช่องทางบอกค่าจริง — ส่งธงดีกว่าแต่งค่าแล้วแสดงเหมือนเป็นข้อเท็จจริง

### `ack` — ตอบคำสั่ง

```json
{"v":1,"type":"ack","request_id":7,"ok":false,"action":"recording_start",
 "code":"COMMANDS_DISABLED","message":"PC ยังไม่เปิดรับคำสั่งในเฟสนี้"}
```

---

## การกันซ้ำ

`event_id` = `"<boot_id>-<seq6>"` มี boot_id อยู่ในตัว จึงไม่ชนกันข้ามการบูต
ชุดกันซ้ำ **ไม่ต้องล้างเมื่อจอรีบูต** และตอนเปิดโปรแกรมต้องอ่าน `event_id`
ของไฟล์วันนี้กลับเข้ามาก่อน ไม่งั้นการ replay คิวของจอหลังต่อใหม่จะได้บรรทัดซ้ำ

## เหตุผลที่ถูกทิ้ง (ตัวนับใน `snapshot()["counters"]`)

| reason | ตัวนับ | ถือเป็นความผิดพลาดไหม |
|---|---|---|
| `OVERSIZE` | `dropped_oversize` | ใช่ |
| `BAD_VERSION` | `dropped_version` | ใช่ |
| `MISSING_FIELD*` / `BAD_FIELD*` | `dropped_field` | ใช่ |
| `NOT_JSON` / `NOT_OBJECT` / `NOT_UTF8` | `dropped_parse` | ใช่ |
| `EMPTY` / `UNKNOWN_TYPE` | — | **ไม่** — เผื่อเฟิร์มแวร์รุ่นใหม่ส่ง type ที่เรายังไม่รู้จัก |
