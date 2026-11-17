@echo off
chcp 65001 >nul <nul
title AstroPipe
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Программа ещё не установлена. Сначала запустите install.bat двойным щелчком.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m astropipe wizard %1
pause
