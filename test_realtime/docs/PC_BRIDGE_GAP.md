# ฝั่ง PC — สิ่งที่ยังขาด เพื่อรับ bridge จากจอ (หลัง commit a8b4c6c)

- วันที่: 27 สิงหาคม 2026
- อ่านจาก: `pc_bridge.h/.c` · `nd_json.h` · `ui/ui_model.h` · `tools/mock_pc.py`
- ตัวเลขทุกตัวในเอกสารนี้ **วัดจากการรันจริง** ไม่ใช่ประมาณ

---

# 1. สัญญาที่ฝั่ง PC ต้องทำตาม (ถอดจากโค้ดจริง ไม่ใช่จากข้อเสนอเดิม)

| หัวข้อ | ค่าจริงในโค้ด | ที่มา |
|---|---|---|
| พอร์ต | USB-Serial-JTAG **VID:PID `303A:1001`** (คนละหัวกับ CH343) | `pc_bridge.h` |
| เฟรม | NDJSON บรรทัดละ object · สูงสุด **512 ไบต์** | `PC_BRIDGE_FRAME_MAX` |
| เวอร์ชัน | `"v":1` — ไม่ตรง = จอไม่ประมวลผลเลย | `handle_line()` |
| **PC ต้องส่ง `state` ทุกไม่เกิน 10 วินาที** | `PC_STALE_MS = 10000` | `pc_bridge.c:49` |
| **PC ต้อง ACK ภายใน 5 วินาที** | `CMD_TIMEOUT_MS = 5000` | `pc_bridge.c:50` |
| คำสั่งค้างได้พร้อมกัน | **4** | `CMD_SLOTS` |
| คิว event ฝั่งจอ | **32** | `EVENT_Q_LEN` |
| **จอไม่ retry เอง** | ไม่มีโค้ดส่งซ้ำ — PENDING → TIMEOUT แล้วจบ | `pc_bridge.c:313-317` |

## 1.1 เฟรมที่จอส่งมา

```jsonc
// heartbeat
{"v":1,"type":"hb","boot_id":"p4-c96e-a820","ts_ms":182345,"queued":0,"link":"online"}

// ผลการวัดที่ผู้ใช้กดยืนยัน — ห้ามหาย
{"v":1,"type":"event","event_id":"p4-c96e-a820-000017","boot_id":"p4-c96e-a820",
 "event":"reading_saved","sensor":2,"ec_us_cm":84.6,"temperature_c":20.4,
 "tolerance_us_cm":2.0,"stable_for_ms":15000,"after_link_error":false,
 "ts_ms":182345,"wall":"..."}

// event บริบท — ชื่อมาจาก measure_event_name()
{"v":1,"type":"event","event_id":"...","boot_id":"...","event":"MEASURE_EV_STABILITY_REACHED",
 "sensor":2,"ec_us_cm":84.6,"ts_ms":182345}

// คำสั่ง
{"v":1,"type":"cmd","request_id":7,"boot_id":"p4-c96e-a820",
 "action":"recording_start","sensor":0,"ts_ms":182345,"payload":{}}
```

> ⚠️ **`"sensor":0` ไม่ใช่ sensor #0** — `ui_model.c:498` เรียก `pc_bridge_cmd(action, -1, NULL)`
> แล้ว `pc_bridge.c:499` เขียน `sensor + 1` ลงเฟรม ⇒ **0 = "ไม่ระบุตัว / ทั้งระบบ"**
> ฝั่ง PC ต้องตีความแบบนี้ ไม่งั้นจะไปหา `sessions[−1]`

## 1.2 เฟรมที่ PC ต้องส่งกลับ — field ที่จอ "อ่านจริง"

จาก `handle_state()` และ `handle_ack()` — field อื่นถูกข้ามเงียบ ๆ ตามกฎ forward compatibility

| type | field ที่จออ่าน |
|---|---|
| `state` | `recording` (bool) · `session_mask` (int) · `active_mask` (int) · `cal_busy` (bool) · `csv_rows` (int) · `sample_id` (str ≤23) |
| `ack` | `request_id` · `ok` · `code` · `message` (ถ้าไม่มี `message` จอใช้ `code` แทน) |

## 1.3 action ที่จอส่งได้ "ตอนนี้"

`recording_start` · `recording_stop` — เท่านั้น ทั้งคู่ `sensor: 0`

---

# 2. สิ่งที่ฝั่ง PC ยังไม่มีเลย

| # | ต้องมี | ตอนนี้ |
|---|---|---|
| G1 | เปิดพอร์ต `303A:1001` + อ่าน/เขียน NDJSON | **ไม่มี** |
| G2 | ส่ง `state` ทุก 3 วินาที + ทุกครั้งที่สถานะเปลี่ยน | **ไม่มี** |
| G3 | ตอบ `ack` ภายใน 5 วินาที | **ไม่มี** |
| G4 | เก็บ `event` ลง `measurement_events_YYYY-MM-DD.jsonl` (append + fsync) | **ไม่มี** |
| G5 | dedupe `event_id` และ `(boot_id, request_id)` | **ไม่มี** |
| G6 | ตีความ `recording_start` / `recording_stop` | **ไม่มี** |
| G7 | คำนวณ `session_mask` / `csv_rows` / `sample_id` ส่งกลับ | **ไม่มี** (มี `rec_status.json` แต่คนละรูปแบบ) |
| G8 | หน้าจอ PC แสดงสถานะลิงก์ P4 + คิว event ค้าง | **ไม่มี** |

---

# 3. ⚠️ ปัญหาที่ bridge จะเปิดโปงทันที (มีอยู่แล้ววันนี้ แต่ยังไม่มีใครเห็น)

## D1 — สร้างรายงานบล็อกการอ่าน serial

`logger_3ec.py:312` เรียก `report_3ec.export_sensor_session()` **ตรง ๆ ใน main loop**
ระหว่างนั้นไม่มี `ser.readline()` เลย

**วัดจริงบนชุดข้อมูล 24 ชั่วโมง (34,560 แถว):**

| งาน | เวลา |
|---|---|
| `read_csv_rows()` อ่านทั้งโฟลเดอร์ | 0.8 s |
| `export_sensor_session()` (session 8 ชม.) | **2.0 s** |
| `export_combined_report()` (PDF 9 หน้า) | **5.7 s** |
| **หยุด session 3 ตัวติดกัน** | **5.9 s** |

ผลที่ตามมาสามข้อ พร้อมกัน:

1. **จอ timeout** — `CMD_TIMEOUT_MS = 5000` ⇒ 5.9 s > 5.0 s
   จอจะขึ้น `Request status unknown - check the PC` ทั้งที่ PC ทำสำเร็จแล้ว
2. **จอเด้งไป PC OFFLINE** ถ้างานยาวเกิน 10 s (`export_combined_report` ตอนปิด logger ก็เข้าข่าย)
3. **timestamp ของ CSV เพี้ยน** — logger ใส่เวลาด้วย `datetime.now()` ตอน *อ่าน* ไม่ใช่ตอน *วัด*
   ข้อมูลที่ CONTROL พ่นออกมาระหว่างนั้นค้างใน buffer ของ OS แล้วถูกอ่านรวดเดียว
   ⇒ ได้หลายแถวที่ timestamp กระจุกอยู่ท้ายสุด **ข้อมูลไม่หาย แต่แกนเวลาผิด**
   ซึ่งแย่กว่าข้อมูลหาย เพราะมองไม่ออกว่าผิด

> นี่ไม่ใช่บั๊กใหม่จาก bridge — กดคีย์ `3` บนคีย์บอร์ดวันนี้ก็เกิดเหมือนกัน
> แต่ bridge จะทำให้มันเกิดบ่อยขึ้นและเห็นชัดขึ้น
> **ต้องแก้ก่อนต่อ bridge ไม่ใช่หลัง**

## D2 — `find_port()` อาจไปคว้าพอร์ตของจอมาเป็น CONTROL

`logger_3ec.py:199-208` กัน `ch343` ไว้ตัวเดียว แต่ USB-Serial-JTAG ขึ้นเป็น
`USB Serial Device` / `USB JTAG/serial debug unit` ซึ่ง **มีคำว่า `usb`** ⇒ ได้คะแนน 10

⇒ ถ้าถอด CONTROL ออกแล้วเปิด logger มันจะเปิดพอร์ต NDJSON ของจอแทน
แล้วนั่งรอ `DATA,` ที่ไม่มีวันมา พร้อมกับ**ยึดพอร์ตไม่ให้ bridge ใช้**

แก้ง่ายมาก: ตัด `303a:1001` ออกเหมือนที่ทำกับ `ch343` — แต่ต้องรู้ก่อนถึงจะแก้

## D3 — หน้าจอ PC อ่านไฟล์ทั้งโฟลเดอร์ทุก 2 วินาที

`read_range()` วนอ่าน **ทุกไฟล์ทุกบรรทัด** ทุกครั้ง แล้วค่อยกรองด้วยเวลา
ที่ข้อมูล 24 ชั่วโมง = 0.8 s ต่อครั้ง × ~2 ครั้งต่อ 4 วินาที ≈ **40% ของหนึ่งคอร์**
และโตเป็นเส้นตรงตามปริมาณข้อมูล — หนึ่งสัปดาห์ก็ใช้ไม่ได้แล้ว

ไม่เกี่ยวกับ bridge โดยตรง แต่ถ้า bridge ไปอยู่ใน process เดียวกัน
มันจะแย่ง CPU กับการตอบ ACK ภายใน 5 วินาที ⇒ กลายเป็นเรื่องเดียวกัน

---

# 4. ตอบคำถามที่ค้างไว้ — "session ผูกกับอะไร"

## 4.1 ของจริงที่เป็นอยู่

| สิ่ง | ขอบเขตปัจจุบัน | ที่มา |
|---|---|---|
| session | **รายเซนเซอร์** เริ่ม/หยุดอิสระ | `SessionMgr.start[i]`, คีย์ `1/2/3` |
| ผลลัพธ์ของ session | PDF + Excel ของเซนเซอร์ตัวนั้น ลง `sensor_N/` | `export_sensor_session()` |
| `sample` | **รายรอบการรัน** — argument `--sample` ตัวเดียวทั้ง run | `logger_3ec.py:513` |
| `note` | รายเซนเซอร์ ถามตอนกดหยุด | `note_reader` |

⇒ **`sample` เป็นระดับ run แต่จอคิดว่ามันเปลี่ยนได้ระหว่างทาง** — ตรงนี้ไม่ตรงกัน

## 4.2 ข้อเสนอ (ผมแนะนำแบบนี้ และจะทำแบบนี้ถ้าไม่แย้ง)

```
session   = รายเซนเซอร์          (คงเดิม — เป็นหน่วยที่ออกรายงานอยู่แล้ว
                                  และตรงกับเหตุผลที่มี 3 หัววัดคือทดลองขนานกัน)
sample_id = ผูกกับ session        (ย้ายจากระดับ run) — จับตอนเริ่ม session
note      = ผูกกับ session        (คงเดิม)
recording = any(session_mask)     (ให้ปุ่มเดียวบนจอสอดคล้องกับหลาย session)
```

**`--sample` กลายเป็นค่าตั้งต้นของ session ใหม่** ไม่ใช่ค่าตายตัวของทั้ง run
เป็นการแก้เล็ก ๆ แต่ทำให้ `sample_set` จากจอมีความหมายจริง

## 4.3 การตีความคำสั่งที่จอส่งมาแล้ววันนี้

| action | `sensor` | PC ทำอะไร |
|---|---|---|
| `recording_start` | 0 | **เริ่ม session ให้ทุกเซนเซอร์ที่เปิดใช้และยังไม่มี session** |
| `recording_stop` | 0 | **หยุดทุก session ที่ค้างอยู่** แล้วออกรายงาน (แบบไม่บล็อก) |

### ⚠️ ต้อง idempotent เท่านั้น

จอ **ไม่ retry** แต่ผู้ใช้ retry ได้ — เห็น `Request status unknown` แล้วกดซ้ำ
ถ้า `recording_start` ครั้งที่สองไปสร้าง session ใหม่ทับ จะได้ session ซ้อนทันที
ซึ่งเป็นสิ่งที่คุณกังวลไว้พอดี

**กฎ:** `recording_start` ที่ยิงซ้ำตอนมี session อยู่แล้ว ต้องตอบ `ok:true`
พร้อม `session_id` เดิม **ไม่ใช่ NACK** — เพราะสถานะปลายทางถูกต้องแล้ว
NACK เก็บไว้ให้กรณีที่ทำไม่ได้จริง ๆ (บอร์ดหลุด / กำลัง cal)

### ลำดับที่ต้องเป็น: ACK ก่อน ลงมือทีหลัง

เหมือนที่ CONTROL ทำกับ cal อยู่แล้ว (`sendAckFrame(status=2)` ก่อน `calibrateSensor()`)

```
รับ cmd -> ตรวจความถูกต้อง -> เปลี่ยน state ในหน่วยความจำ -> ส่ง ACK  (< 100 ms)
                                                          -> โยนงานออกรายงานเข้า worker
```

ถ้ารอออกรายงานเสร็จก่อนค่อย ACK จะชน 5 วินาทีตาม D1 แน่นอน

## 4.4 คำสั่งรุ่นถัดไป (หลังจากนี้)

`session_start` / `session_stop` พร้อม `sensor: 1..4` · `sample_set` · `note_set` · `report_request`
— ตอนนั้นปุ่มบนจอควรอยู่ที่ **หน้า Measure รายช่อง** ไม่ใช่ Settings
เพราะ session ผูกกับเซนเซอร์ ไม่ใช่กับเครื่อง

---

# 5. แผนไฟล์

## P0 — ต้องทำก่อนต่อ bridge (แก้ D1 D2)

| ไฟล์ | ทำอะไร |
|---|---|
| `report_jobs.py` | **ใหม่** — `ThreadPoolExecutor(1)` + สถานะ accepted/running/done/failed |
| `logger_3ec.py` | `SessionMgr._close()` โยนงานออกรายงานเข้า worker แทนเรียกตรง ๆ · `find_port()` ตัด `303a:1001` · `sample` ย้ายไปผูกกับ session |

**ผลทันทีแม้ยังไม่มี bridge:** กดหยุด session แล้ว logger ไม่ค้าง timestamp ไม่เพี้ยน

## P1 — bridge (แก้ G1-G7)

| ไฟล์ | ทำอะไร |
|---|---|
| `p4_protocol.py` | **ใหม่** — นิยาม action/code + validate เฟรม ไม่มี I/O เลย ⇒ unit test ได้ตรง ๆ |
| `p4_bridge.py` | **ใหม่** — thread: หาพอร์ต `303A:1001` · reconnect · rx/tx NDJSON · dedupe · rate limit |
| `event_log.py` | **ใหม่** — append `.jsonl` + fsync · ห้ามแตะ CSV |
| `pc_state.py` | **ใหม่** — แหล่งเดียวของ state ที่ส่งให้จอ **และ** เขียน `pc_state.json` ให้ desktop_ui |
| `logger_3ec.py` | สร้าง thread + คิวสองทาง (`cmd_q` / `state_q`) — **thread ห้ามแตะ `ser` ของ COM-A** |

## P2 — หน้าจอ PC (แก้ G8)

| ไฟล์ | ทำอะไร |
|---|---|
| `lab_theme.py` | เพิ่มสถานะ `P4 BRIDGE DISABLED` / `P4 OFFLINE` / `P4 CONNECTED` — **ใช้คำและกฎเดียวกับที่จอใช้กับ PC** (สมมาตร) |
| `desktop_ui.py` | การ์ด/แถบสถานะ P4 + จำนวน event ค้าง (แดงเมื่อใกล้เต็ม 32) + event จากจอไหลเข้า EVENT LOG |

## P3 — แยกต่างหาก ไม่ผูกกับ bridge

- cal observer (Phase 3b ที่อนุมัติไว้แล้ว) → เติม `cal_busy` ใน snapshot ให้เป็นค่าจริง
- แก้ D3 (index CSV ตามวัน / cache) — จะจำเป็นเมื่อข้อมูลเกินสองสามวัน

---

# 6. เรื่องที่ยังเห็นต่าง / ต้องตัดสิน

## 6.1 `active_mask` ในเฟรม state — ส่งอันไหน

ฝั่ง PC มีสองอย่างที่ชื่อคล้ายกัน:

| | คืออะไร | ค่าตอนนี้ |
|---|---|---|
| mask ที่ CONTROL poll จริง | บอร์ดถามหัววัดตัวไหนบ้าง | `0b0111` เสมอ (ยังไม่ configurable) |
| mask การแสดงผลของ PC | `ec_ui_config.json` — ซ่อน #1 ที่เสีย | `0b0110` |

**ผมจะส่งอันแรก** (`0b0111`) เพราะจอใช้ค่านี้ตัดสินว่าจะโชว์ช่องวัดกี่ช่อง
และช่องวัดต้องตรงกับหัววัดที่ถูกถามจริง ไม่ใช่ตรงกับความชอบของหน้าจอ PC
ถ้าวันหลังต้องการให้จอรู้ค่าที่สอง ค่อยเพิ่ม field `display_mask` แยก

## 6.2 เห็นด้วยกับคุณเรื่องลำดับ

> "soak 1–2 ชั่วโมง + PANIC (R4) — แนะนำให้ปิดเรื่องนี้ก่อนเปิด bridge ใช้จริง"

**เห็นด้วยเต็มที่** และเสริมอีกข้อ: ฝั่ง PC ก็ควรปิด D1 ก่อนเหมือนกัน
ไม่งั้นเวลา timeout เกิดขึ้น จะแยกไม่ออกว่าเป็นเพราะจอค้าง หรือเพราะ PC ไปสร้าง PDF อยู่
— บั๊กสองตัวที่อาการเหมือนกันเป๊ะ

ลำดับที่ปลอดภัยที่สุด:

```
1. PC: P0 (report worker + find_port)     <- ทำได้เลย ไม่ต้องมีฮาร์ดแวร์
2. P4: ปิด soak/PANIC
3. PC: P1 + P2 ทดสอบกับ mock_pc.py กลับด้าน (ผมทำ mock_p4.py)
4. ต่อของจริงสองหัว USB
```

ข้อ 1 กับ 3 ผมทำได้โดยไม่ต้องรอฮาร์ดแวร์เลย
