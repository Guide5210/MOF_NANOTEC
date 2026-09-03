@echo off
cd /d "%~dp0"
python -m ecstation.app %*
if errorlevel 1 pause
