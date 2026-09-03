@echo off
cd /d "%~dp0"
start "EC viewer (mock)" cmd /k python -m ecstation.app --mock
timeout /t 3 >nul
python tools\mock_p4.py --scenario %1 --seconds 300
