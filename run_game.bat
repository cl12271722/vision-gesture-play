@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo 未找到 .venv，请先按照 README.md 创建虚拟环境。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py %*
if errorlevel 1 (
    pause
)
