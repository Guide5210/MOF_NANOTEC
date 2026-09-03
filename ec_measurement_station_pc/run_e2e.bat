@echo off
cd /d "%~dp0"
python tools\e2e_mock.py %*
pause
