@echo off
cd /d "%~dp0"
if not exist "%~dp0.for_yashi_key" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup_For_Yashi_Model.ps1"
  if not exist "%~dp0.for_yashi_key" exit /b 1
)
start "" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0For_Yashi_Server.ps1"
exit /b 0
