@echo off
cd /d "%~dp0"
python -m unittest discover -s tests -t tests -q
if errorlevel 1 pause
