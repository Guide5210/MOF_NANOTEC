# P0 เสร็จแล้ว — ฝั่ง PC พร้อมรับ bridge แล้วในเชิงโครงสร้าง

- วันที่: 27 สิงหาคม 2026
- ขอบเขต: แก้ D1 (report บล็อก serial) · D2 (`find_port` คว้าพอร์ตจอ) · session/sample model
- **ยังไม่ต่อ bridge** — นี่คือขั้นที่ต้องทำก่อน ไม่ใช่ตัว bridge เอง

---

## 1. ไฟล์ที่เปลี่ยน

| ไฟล์ | สถานะ |
|---|---|
| `report_jobs.py` | **ใหม่** — คิวงานหนัก + worker หนึ่งตัว |
| `logger_3ec.py` | แก้ 3 จุด (`find_port` · `SessionMgr` · `run()`) |
| `logger_3ec.py.bak` | สำรองของเดิม (ยังอยู่) |

**ไม่แตะ:** `report_3ec.py` · `calibration.py` · `desktop_ui.py` · `lab_theme.py` ·
`parse()` · `open_csv()` · `daily_path()` · `append_session_log()` — บล็อกอ่าน/เขียน CSV
ยืนยันด้วย diff แล้วว่าเหมือนเดิมทุกไบต์

---

## 2. ผลการทดสอบ

### D1 — เวลาที่ `_close()` บล็อกลูปอ่าน serial

ชุดข้อมูลจริง 24 ชั่วโมง (34,560 แถว) · ปิด session ที่กินเวลา 8 ชั่วโมง

| | บล็อกลูปนาน |
|---|---|
| เดิม (เรียก `export_sensor_session()` ตรง ๆ) | **13.64 s** |
| ใหม่ (เข้าคิว worker) | **0.00 s** |

| | หยุด 3 session ติดกัน | vs `CMD_TIMEOUT_MS = 5.0 s` |
|---|---|---|
| เดิม | 40.9 s | **เกิน 8 เท่า** |
| ใหม่ | 0.0 s | ทัน |

รายงานยังถูกสร้างจริงครบทุกใบ — แค่ไปทำในเบื้องหลัง

### D1 — end-to-end ด้วย pty จริง + pyserial จริง

จำลองบอร์ด CONTROL ด้วย pty ที่พ่น `DATA,` ตามจังหวะ แล้ววัดช่องว่างระหว่างแถวที่เขียนลง CSV

**ที่จังหวะเร่ง 10 เท่า (0.25 s) — ให้เห็นอาการชัด**

| | `_close()` | gap แย่สุด | แถวที่ช้า | แถวที่กระจุก |
|---|---|---|---|---|
| เดิม | 1.72 s | 1.73 s | 1 | **6** |
| ใหม่ | 0.00 s | 0.52 s | 1 | 1 |

**ที่จังหวะจริง 2.5 s**

| | `_close()` | gap กลาง | gap แย่สุด | แถวที่ช้า | แถวที่กระจุก |
|---|---|---|---|---|---|
| ใหม่ | 0.00 s | 2.50 s | 2.50 s | **0** | **0** |

> **ข้อสังเกตที่ต้องรู้:** บน pty ตัวเขียนจะถูกบล็อกเมื่อบัฟเฟอร์เต็ม
> "gap แย่สุด" ที่วัดได้จึงสั้นกว่าเวลาที่ `_close()` บล็อกจริง
> **บน UART จริงไม่มี flow control — ไบต์จะถูกทิ้งไปเลย**
> อาการจริงบนฮาร์ดแวร์จึงแรงกว่าที่เห็นในตาราง: ข้อมูลหาย ไม่ใช่แค่เวลาเพี้ยน
>
> ส่วน gap 0.52 s ที่เหลือของฝั่งใหม่ที่จังหวะเร่ง มาจาก worker แย่ง GIL ตอน
> matplotlib ทำงาน ที่จังหวะจริง 2.5 s มีเวลาเหลือพอจนไม่มีแถวไหนช้าเลย
> ถ้าวันไหนกลายเป็นปัญหา ทางแก้คือย้าย worker ไป `multiprocessing`
> (args ทุกตัวเป็น pickle ได้อยู่แล้ว) — ยังไม่ทำเพราะยังไม่จำเป็น

### D2 — `find_port()`

| พอร์ตที่มีอยู่ | เลือกได้ |
|---|---|
| CONTROL + CH343 + USB-Serial-JTAG | `COM3` (CONTROL) |
| **ถอด CONTROL ออก** ← เคสที่เคยพัง | `None` |
| เหลือแต่พอร์ต NDJSON ของจอ | `None` |
| CONTROL + JTAG (ชื่ออีกแบบ) | `COM3` |
| อุปกรณ์ USB อื่นที่ไม่ใช่จอ | `COM9` |

ตัดด้วย **VID:PID** เป็นหลัก (`303A:1001`, `1A86:55D3`) ไม่พึ่งข้อความบรรยาย
ซึ่งต่างกันไปตาม OS และไดรเวอร์

### session / sample model

```
sample ผูกกับ session ไม่ใช่กับ run
  เริ่ม #2 ด้วยค่าตั้งต้น       -> sample = CALF-20 B3
  ตั้ง sample ใหม่ แล้วเริ่ม #3  -> sample = POND-A
  #2 ที่รันอยู่ไม่ถูกเปลี่ยนย้อนหลัง: CALF-20 B3

idempotency
  start_session(#2) ซ้ำ -> False   (= "มีอยู่แล้ว" ไม่ใช่ "ล้มเหลว")
  session_mask          -> 0b0110

sessions_3ec.json บันทึก sample ของ session นั้น ๆ ถูกต้อง
  sensor 2  sample=CALF-20 B3   note=wash step 2
  sensor 3  sample=POND-A       note=wash step 2
```

---

## 3. API ใหม่ที่ bridge จะใช้ต่อ (มีแล้ว ยังไม่มีใครเรียก)

```python
mgr.session_mask()          # -> 0b0110  รูปแบบเดียวกับที่ pc_bridge คาดหวัง
mgr.any_active()            # -> recording
mgr.start_session(i, sample=None)   # idempotent: False = มีอยู่แล้ว
mgr.set_sample("POND-A")    # ตั้งของ session ถัดไป ไม่แตะตัวที่รันอยู่
mgr.sample_now()            # -> sample_id ที่จะส่งใน state snapshot

jobs.snapshot()             # {busy, running, queued, done, failed, last_error}
                            # -> ใช้ตอบ report_request แบบ async ภายหลัง
```

`rec_status.json` เพิ่ม field (ของเดิมยังอยู่ครบ desktop_ui อ่านได้เหมือนเดิม):

```json
{"active":[false,true,true], "mask":6,
 "sample":[null,"CALF-20 B3","POND-A"], "updated":"..."}
```

---

## 4. ผลข้างเคียงที่ได้มาฟรี (แม้ยังไม่มี bridge)

- กดคีย์ `1/2/3` หยุด session แล้ว **logger ไม่ค้างอีกแล้ว** — เดิมค้าง 13 วินาที
- ตอนปิดโปรแกรม รายงานรวมเข้าคิวแล้วรอจนเขียนไฟล์จบก่อน process จบ
  (ถ้าไม่รอจะได้ PDF ที่เขียนไม่จบ ซึ่งแย่กว่าไม่ได้ไฟล์)
- รายงานที่ล้มเหลวไม่ทำให้ session อื่นหยุดตาม และไม่ทำให้ logger ตาย
- `plt.close("all")` หลังทุกงาน — กันรูปค้างจากเส้นทางที่ raise กลางคัน

---

## 5. ถัดไป

| | งาน | ต้องรออะไร |
|---|---|---|
| P1 | `p4_protocol.py` · `p4_bridge.py` · `event_log.py` · `pc_state.py` | ไม่ต้องรอ — ทดสอบกับ `mock_p4.py` ได้ |
| P2 | สถานะ P4 บน desktop_ui (สมมาตรกับที่จอแสดง PC) | หลัง P1 |
| — | เปิด bridge ใช้จริง | **หลังปิด soak/PANIC ฝั่งจอ (R4)** ตามที่ตกลงกัน |

**Rollback:** `copy logger_3ec.py.bak logger_3ec.py` แล้วลบ `report_jobs.py`
`SessionMgr(..., jobs=None)` ยังทำงานแบบเดิม (บล็อก) ได้อยู่ ถ้าอยากเทียบ
