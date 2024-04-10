@echo off
set PYTHON_PATH=..\python_env\py38_gdt\python.exe

if not exist "%PYTHON_PATH%" (
    echo Python is not installed or the path is incorrect.
)
set SCRIPT_PATH=stop.py
"%PYTHON_PATH%" "%SCRIPT_PATH%"

set SCRIPT_PATH=start.py
"%PYTHON_PATH%" "%SCRIPT_PATH%" "%PYTHON_PATH%"

pause