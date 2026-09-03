@echo off
cd /d "%~dp0"
python tools\run_bridge.py --mode serial
pause
