@echo off
REM ======================================================================
REM  run_dashboard.bat - เปิดหน้าจอ EC Measurement Station
REM
REM  ใช้ %~dp0 = โฟลเดอร์ที่ไฟล์ .bat นี้อยู่ จึงชี้ไปที่ desktop_ui.py
REM  ตัวที่อยู่ข้าง ๆ กันเสมอ ไม่ว่าจะกดจาก shortcut ที่ไหนก็ตาม
REM ======================================================================
cd /d "%~dp0"
echo.
echo  folder: %CD%
echo.
python "%~dp0desktop_ui.py"
if errorlevel 1 (
  echo.
  echo  ** เปิดไม่สำเร็จ - อ่านข้อความข้างบน **
  pause
)
