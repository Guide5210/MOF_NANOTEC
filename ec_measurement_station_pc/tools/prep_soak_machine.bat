@echo off
REM ============================================================================
REM  เตรียมเครื่องก่อนรัน soak ยาว (12-24 ชม.)
REM ============================================================================
REM  ทำไมต้องมี:
REM
REM  โปรแกรม Python ที่รันอยู่ "ไม่" กันเครื่องหลับ  Windows ตัดสินจากการที่
REM  ไม่มีคนแตะเครื่อง ไม่ได้ดูว่ามีงานทำอยู่หรือไม่  ค่าเริ่มต้นของ Windows
REM  คือหลับใน 60 นาที ⇒ soak 24 ชม. จะตายตั้งแต่ชั่วโมงแรกโดยไม่มีใครรู้
REM
REM  และ USB selective suspend ทำให้พอร์ตอนุกรมหลุดเป็นระยะ ซึ่งเคยเกิดจริง
REM  รอบ 2 ก.ย. 2026 ทั้งสามพอร์ตสะดุดพร้อมกัน ~210 วินาที แล้วเทสต์ตก
REM  โดยที่จอไม่ได้พังเลย (reboots 0, ไม่มี Guru Meditation)
REM
REM  ⚠️ ตั้งเฉพาะตอนเสียบปลั๊ก (AC) ไม่แตะค่าตอนใช้แบตเตอรี่
REM     เครื่องที่รัน soak ต้องเสียบปลั๊กอยู่แล้ว และการไปแก้ค่าแบตจะทำให้
REM     โน้ตบุ๊กของคนอื่นแบตหมดเร็วโดยไม่ได้ตั้งใจ
REM ============================================================================

setlocal
echo.
echo ==============================================================================
echo  เตรียมเครื่องสำหรับ soak ยาว
echo ==============================================================================
echo.
echo ค่าเดิมก่อนแก้ (จดไว้เผื่ออยากคืนค่า):
for /f "tokens=2 delims=:" %%a in ('powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE ^| findstr /c:"Current AC"') do echo    sleep timeout  ^(AC^) = %%a
for /f "tokens=2 delims=:" %%a in ('powercfg /q SCHEME_CURRENT SUB_DISK DISKIDLE ^| findstr /c:"Current AC"') do echo    disk timeout   ^(AC^) = %%a
echo.

echo กำลังตั้งค่า ...
REM 0 = ไม่ทำงานเลย
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_DISK DISKIDLE 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_VIDEO VIDEOIDLE 0

REM USB selective suspend: subgroup 2a737441-... / setting 48e6b7a6-...
REM 0 = ปิด (ห้าม suspend พอร์ต USB)
powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0

powercfg /setactive SCHEME_CURRENT

echo.
echo ค่าใหม่:
for /f "tokens=2 delims=:" %%a in ('powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE ^| findstr /c:"Current AC"') do echo    sleep timeout  ^(AC^) = %%a  ^(0 = ไม่หลับ^)
for /f "tokens=2 delims=:" %%a in ('powercfg /q SCHEME_CURRENT SUB_DISK DISKIDLE ^| findstr /c:"Current AC"') do echo    disk timeout   ^(AC^) = %%a  ^(0 = ไม่หยุดหมุน^)
echo.
echo ==============================================================================
echo  เสร็จแล้ว
echo.
echo  ยังต้องทำเองอีกสองอย่างที่สคริปต์ทำแทนไม่ได้:
echo    1. เสียบสาย USB ตรงเข้าเครื่อง อย่าผ่าน hub ที่ไม่มีไฟเลี้ยง
echo    2. เสียบปลั๊กไฟไว้ตลอด อย่าใช้แบตเตอรี่
echo.
echo  คืนค่าเดิมหลังเทสต์เสร็จ (ค่า Windows มาตรฐาน):
echo    powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 1800
echo    powercfg /setacvalueindex SCHEME_CURRENT SUB_DISK DISKIDLE 1200
echo    powercfg /setactive SCHEME_CURRENT
echo ==============================================================================
echo.
pause
