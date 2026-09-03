# การถอยกลับ

## ระดับ 0 — ไม่ต้องทำอะไร

โปรเจกต์นี้อยู่คนละโฟลเดอร์กับระบบเดิมและ **ไม่เขียนอะไรลงโฟลเดอร์ของระบบเดิม**
ปิดโปรแกรมนี้ทิ้งไป ระบบเดิมทำงานต่อได้ทันทีโดยไม่ต้องแก้อะไร

นี่คือเหตุผลทั้งหมดของการแยกโปรเจกต์ — การถอยกลับต้องเป็น "หยุดใช้" ไม่ใช่ "กู้คืน"

## ระดับ 1 — ปิดเฉพาะ bridge

`config/app_config.json`:

```json
{"bridge": {"enabled": false}}
```

ลิงก์จะเป็น `P4 BRIDGE DISABLED` (ไม่ใช่ ERROR) · ไม่เปิดพอร์ต · ไม่เขียน event
ส่วนที่เหลือยังใช้ได้

## ระดับ 2 — ลบโปรเจกต์นี้ทิ้ง

```bat
rmdir /S /Q C:\MOF_NanoTec\ec_measurement_station_pc
```

ไม่มีผลใด ๆ ต่อ `test_realtime` — ยืนยันด้วย `tests/test_no_legacy_mutation.py`
ข้อมูล event ที่เก็บไว้จะหายไปด้วย ถ้าอยากเก็บให้คัดลอก `data/events/` ออกก่อน

## ระดับ 3 — ถอยระบบเดิมกลับไปที่ baseline

ใน `C:\MOF_NanoTec\test_realtime` (repo แยกของมันเอง):

```bat
git -C C:\MOF_NanoTec\test_realtime status --short
git -C C:\MOF_NanoTec\test_realtime checkout legacy-baseline-2026-08-28 -- .
```

tag `legacy-baseline-2026-08-28` (commit `54a2c6e`, 35 ไฟล์) ตรวจแล้วว่ากู้กลับ
ได้ตรงทุกไบต์  **ข้อมูลห้องแล็บไม่ได้อยู่ใน git** — `water_data/` · รายงาน ·
`sessions_3ec.json` · `rec_status.json` ไม่ถูกแตะโดยคำสั่งนี้

## ระดับ 4 — ฝั่งจอ

เป็น repo คนละตัว (`hello_world`) ถอยด้วย git ของมันเอง
ปิด bridge ฝั่งจอได้ที่ SETTINGS → SYSTEM → PC LINK
จอทำงานได้ครบทุกอย่างโดยไม่มี PC และ PC ทำงานได้โดยไม่มีจอ

## สิ่งที่ยังไม่ได้เปิด (ไม่ต้องถอย เพราะยังไม่ได้เริ่ม)

- คำสั่งจาก P4 มา PC — ตอบ NACK ทุกคำสั่ง
- การย้ายความเป็นเจ้าของ logger — ยังเป็นของระบบเดิม
- bridge การคาลิเบรต · การสร้างรายงาน · การแก้ schema CSV
