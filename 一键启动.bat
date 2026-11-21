@echo off
setlocal enabledelayedexpansion
chcp 936 >nul
title 数据分析 Agent 一键启动
cd /d "%~dp0"

echo.
echo  ============================================================
echo    DataAnalysisAgent  -  数据分析 Agent
echo  ============================================================
echo.

if not exist "run.py" goto :BAD_DIR
if not exist "backend\app\main.py" goto :BAD_DIR

rem ---------- 找一个能用的 Python ----------
set "PY=python"
%PY% -c "import sys" >nul 2>&1
if not errorlevel 1 goto :PY_OK
set "PY=py -3"
%PY% -c "import sys" >nul 2>&1
if not errorlevel 1 goto :PY_OK
goto :NO_PYTHON
:PY_OK

rem ---------- 检查 .env ----------
if not exist ".env" goto :NO_ENV

rem ---------- 检查依赖 ----------
%PY% -c "import fastapi,uvicorn,langgraph,pandas,matplotlib" >nul 2>&1
if errorlevel 1 goto :INSTALL

goto :START

:INSTALL
echo  [*] 第一次运行，正在安装依赖，大约 2-5 分钟，请勿关闭窗口...
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto :PIP_FAIL
echo.
goto :START

:START
if not exist "storage" mkdir "storage"
if not exist "data" mkdir "data"

set "ARG="
if not "%~1"=="" set "ARG=--port %~1"

echo  [*] 启动服务...
echo      控制台     : http://127.0.0.1:8000
echo      接口文档   : http://127.0.0.1:8000/docs
echo      浏览器会在 2 秒后自动打开
echo      关闭本窗口即停止服务
echo.
%PY% run.py %ARG%
set "RC=%errorlevel%"
if "%RC%"=="0" goto :END
goto :RUNTIME_FAIL

rem ================= 出错提示 =================

:BAD_DIR
echo  [x] 当前目录不是项目根目录。
echo      请把本脚本放在能看到 run.py 的那一层，再双击运行。
goto :FAIL

:NO_PYTHON
echo  [x] 没有找到可用的 Python。
echo      请安装 Python 3.11 或更高版本。
echo      安装时务必勾选 "Add Python to PATH"。
echo      下载地址：https://www.python.org/downloads/
goto :FAIL

:NO_ENV
echo  [x] 缺少 .env 配置文件。
if not exist ".env.example" goto :FAIL
copy ".env.example" ".env" >nul
echo      已从 .env.example 生成 .env，正在打开记事本。
echo      请把 DEEPSEEK_API_KEY 填成你自己的 Key，保存后重新双击运行。
start "" notepad ".env"
goto :FAIL

:PIP_FAIL
echo.
echo  [x] 依赖安装失败。常见原因与处理：
echo      1. 网络慢或超时，改用国内镜像重试：
echo         python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo      2. 权限不足，右键本脚本选择"以管理员身份运行"
echo      3. Python 版本过低，建议 3.11 及以上
goto :FAIL

:RUNTIME_FAIL
echo.
echo  [x] 服务启动失败，退出码 %RC%
echo      1. 端口 8000 被占用：换一个端口重试，例如  一键启动.bat 8080
echo      2. storage 目录不可写：确认项目不在只读位置
echo      3. 看上面的红色报错，按提示处理
goto :FAIL

:FAIL
echo.
pause
exit /b 1

:END
echo.
echo  [*] 服务已退出。
pause
exit /b 0
