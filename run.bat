@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if not errorlevel 1 goto use_python

where py >nul 2>nul
if not errorlevel 1 goto use_py

echo Python was not found. Install Python, then double-click run.bat again.
pause
exit /b 1

:use_py
py -3 run_server.py
goto finished

:use_python
python run_server.py

:finished
pause
