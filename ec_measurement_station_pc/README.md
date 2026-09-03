# EC Measurement Station (PC)

โปรแกรมฝั่ง PC ตัวใหม่ของระบบวัด EC สำหรับล้าง CALF-20
ทำงานคู่กับจอสัมผัส **ESP32-P4-WIFI6-Touch-LCD-7B** ผ่าน NDJSON บน USB-Serial-JTAG

> **สถานะปัจจุบัน: P1-C — เครื่องมือทดสอบฮาร์ดแวร์พร้อม ยังไม่ได้รันกับของจริง**
>
> ยังไม่เปิด: คำสั่งจากจอมา PC (ตอบ NACK ทุกอัน) · การสร้างรายงาน ·
> การย้ายความเป็นเจ้าของ logger

---

## กฎข้อเดียวที่สำคัญที่สุด

**ระบบเดิมที่ `C:\MOF_NanoTec\test_realtime` เป็นของจริงที่ใช้งานอยู่
โปรเจกต์นี้อ่านได้อย่างเดียว ห้ามเขียนอะไรลงไปเด็ดขาด**

- raw CSV · session JSON · รายงาน PDF/Excel · การคาลิเบรต → เป็นของระบบเดิมทั้งหมด
- โปรเจกต์นี้เขียนได้ที่เดียวคือ `data/events/` ของตัวเอง
- มีด่านกันสามชั้น: `config._guard_data_dir()` ตอนรัน,
  `tests/test_no_legacy_mutation.py` ตอนเทสต์, และ `.gitignore` ตอน commit

จุดอ้างอิงกลับ: tag `legacy-baseline-2026-08-28` ใน repo ของ `test_realtime`

---

## โครงไฟล์

```
ecstation/
  app.py            entry point — python -m ecstation.app [--mock|--no-bridge]
  ui/
    lab_theme.py    token (ตาม ui_tokens.h) + นโยบายสถานะ + rcParams
    view_model.py   ★ ตรรกะหน้าจอทั้งหมด — ไม่ import tkinter
    series_painter.py ★ ตรรกะเส้นกราฟ — ไม่ import tkinter
    dashboard.py · cards.py · chart.py · events.py · diagnostics.py · widgets.py
  bridge/
    protocol.py     แปลง/สร้างเฟรม NDJSON — ไม่มี I/O ไม่เคยโยน exception
    p4_bridge.py    เธรดคุยกับจอ + สถานะลิงก์ + ตัวนับ
    event_log.py    เขียน data/events/*.jsonl ต่อท้ายอย่างเดียว + กันซ้ำ
    mask.py         ซิงก์ display_mask ระหว่างจอกับ PC
    pc_state.py     ประกอบ state ที่ส่งให้จอ (อ่านจากไฟล์ของระบบเดิม)
  core/
    ports.py        แยกพอร์ต CONTROL / P4-log / P4-bridge ด้วย VID:PID
    legacy_read.py  อ่านของระบบเดิม — ไม่มีฟังก์ชันไหนเปิดไฟล์โหมดเขียน
    csv_source.py   อ่าน raw CSV ของ legacy แบบต่อยอด (read-only)
    config.py       ค่าตั้ง + ด่านกัน data_dir ทับ legacy
tools/
  mock_p4.py        จอจำลอง 13 ฉาก (ต่อเข้ามาทาง TCP)
  check_install.py  ตรวจความพร้อมของเครื่องก่อนเริ่ม
  port_audit.py     ดูพอร์ตทั้งหมด + บทบาท (ไม่เปิดพอร์ตใด ๆ)
  hw_test.py        ชุดทดสอบฮาร์ดแวร์ A-E + ตัดสิน pass/fail
  run_bridge.py     เปิด bridge แบบไม่มีหน้าจอ
  e2e_mock.py       รันครบ 13 ฉากผ่าน bridge จริงแล้วตรวจผลอัตโนมัติ
tests/              เทสต์ stdlib unittest (ไม่ต้องมี pytest)
data/events/        event log ของโปรเจกต์นี้ (ไม่เข้า git)
data/ui_state.json  ค่าตั้งหน้าจอ (ไม่เข้า git)
```

## เริ่มใช้

> **ย้ายไปเครื่องอื่น?** ต้องคัดลอก **สองโฟลเดอร์** — `test_realtime`
> (ระบบเดิม เป็นคนเก็บข้อมูลจริง) และโฟลเดอร์นี้  ดู `docs/INSTALL.md`

```bat
copy config\app_config.example.json config\app_config.json
run_check.bat          :: ตรวจว่าเครื่องนี้พร้อมหรือยัง
run_port_audit.bat     :: ดูว่าพอร์ตไหนเป็นอะไร
run_hw_test.bat --steps A,B,C,D   :: ทดสอบฮาร์ดแวร์
run_viewer.bat         :: เปิดหน้าจอ (ต่อจอจริง)
run_viewer_mock.bat normal   :: เปิดหน้าจอ + จอจำลอง
run_tests.bat          :: เทสต์ทั้งหมด
run_e2e.bat            :: 13 ฉากผ่าน bridge จริง
run_mock.bat           :: เปิด bridge + จอจำลอง (สองหน้าต่าง)
```

ต่อกับจอจริง: `python tools\run_bridge.py --mode serial`

## เอกสาร

| ไฟล์ | เรื่อง |
|---|---|
| `docs/ARCHITECTURE.md` | ใครเป็นเจ้าของอะไร · เธรด · ทิศทางข้อมูล |
| `docs/UI.md` | ชั้นของหน้าจอ · กฎการแสดงผลที่ห้ามละเมิด · ภาษาบนหน้าจอ |
| `docs/INSTALL.md` | **ติดตั้งบนเครื่องที่ต่อบอร์ดจริง** |
| `docs/P1C_HARDWARE.md` | **ขั้นตอนทดสอบฮาร์ดแวร์ + เกณฑ์ผ่าน/ไม่ผ่าน** |
| `docs/P4_BRIDGE_PROTOCOL.md` | สคีมา NDJSON ทุกเฟรม + กฎการตรวจ |
| `docs/DISPLAY_MASK.md` | สถานะการซิงก์ mask และเหตุผล |
| `docs/TEST_PLAN.md` | เทสต์ทั้งหมดวัดอะไร และ **ไม่ได้** วัดอะไร |
| `docs/ROLLBACK.md` | ถอยกลับอย่างไรในแต่ละระดับ |
