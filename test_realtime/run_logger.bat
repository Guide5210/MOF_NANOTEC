@echo off
cd /d "%~dp0"
echo.
echo  folder: %CD%
echo.
python "%~dp0logger_3ec.py" %*
pause
