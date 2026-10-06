@echo off
rem ============================================
rem  llm_go quick start: launch Go gateway (web UI)
rem  Usage: double-click start.bat  or  run in cmd
rem  Optional: set PYTHON_CMD=/path/to/python before running
rem ============================================
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "GATEWAY_PORT=8080"

rem --- locate python: PYTHON_CMD > project venv > PATH (实测能否 import yaml) ---
rem 只挑真正装了依赖的解释器: PATH 里的 python 常常是精简版, 会让每个节点 ModuleNotFoundError
if not defined PYTHON_CMD (
    for %%d in ("%PROJECT_DIR%.venv" "%PROJECT_DIR%venv" "%PROJECT_DIR%env") do (
        if not defined PYTHON_CMD if exist "%%~d\Scripts\python.exe" (
            "%%~d\Scripts\python.exe" -c "import yaml" >nul 2>&1
            if not errorlevel 1 set "PYTHON_CMD=%%~d\Scripts\python.exe"
        )
    )
)
if not defined PYTHON_CMD (
    for /f "delims=" %%i in ('where python 2^>nul ^| findstr /v "WindowsApps"') do (
        if not defined PYTHON_CMD (
            "%%i" -c "import yaml" >nul 2>&1
            if not errorlevel 1 set "PYTHON_CMD=%%i"
        )
    )
)
if not defined PYTHON_CMD (
    echo [start] WARNING: no python with pyyaml found. Set PYTHON_CMD first:
    echo [start]   set PYTHON_CMD=C:\path\to\venv\Scripts\python.exe
    echo [start] then install deps:  pip install -r requirements.txt
    echo [start] (training/export nodes will fail with ModuleNotFoundError; gateway/web UI still works)
    set "PYTHON_CMD=python"
)

echo [start] project dir: %PROJECT_DIR%
echo [start] python cmd : %PYTHON_CMD%

rem --- check gateway binary ---
if not exist "%PROJECT_DIR%server\llm-gateway.exe" (
    echo [start] llm-gateway.exe not found, building...
    pushd "%PROJECT_DIR%server"
    go build -o llm-gateway.exe .
    if errorlevel 1 (
        popd
        echo [start] BUILD FAILED - check that Go 1.22+ is installed
        pause
        exit /b 1
    )
    popd
)

rem --- free the port if a stale gateway is running ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%GATEWAY_PORT% " ^| findstr "LISTENING"') do (
    echo [start] killing stale process on port %GATEWAY_PORT% ^(pid %%p^)
    taskkill /F /PID %%p >nul 2>&1
)

echo [start] gateway running at http://localhost:%GATEWAY_PORT%/web/
echo [start] press Ctrl+C to stop
start "" http://localhost:%GATEWAY_PORT%/web/

cd /d "%PROJECT_DIR%server"
llm-gateway.exe
pause
