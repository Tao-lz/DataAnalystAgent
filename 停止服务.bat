@echo off
setlocal enabledelayedexpansion
chcp 936 >nul
title 停止服务
cd /d "%~dp0"

echo.
echo  正在停止占用 8000 / 5173 端口的服务...
echo.

call :KILL_PORT 8000
call :KILL_PORT 5173

echo.
echo  处理完成。可以关闭本窗口。
ping -n 4 127.0.0.1 >nul
exit /b 0

:KILL_PORT
set "HIT=0"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /c:":%~1 " ^| findstr /c:"LISTENING"') do (
  echo    - 结束进程 PID %%p，端口 %~1
  taskkill /f /t /pid %%p >nul 2>&1
  set "HIT=1"
)
if "!HIT!"=="0" echo    - 端口 %~1 上没有正在运行的服务
exit /b 0
