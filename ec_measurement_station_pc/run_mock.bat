@echo off
cd /d "%~dp0"
start "EC bridge (mock)" cmd /k python tools\run_bridge.py --mode mock
timeout /t 2 >nul
start "P4 mock" cmd /k python tools\mock_p4.py --scenario %1 --seconds 60
