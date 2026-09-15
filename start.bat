@echo off
rem ============================================
rem  llm_go quick start: launch Go gateway (web UI)
rem  Usage: double-click start.bat  or  run in cmd
rem ============================================
setlocal

set PROJECT_DIR=%~dp0
set PYTHON_CMD=C:\Users\kevin\.workbuddy\binaries\python\envs\default\Scripts\python.exe
set GATEWAY_PORT=8080

echo [start] project dir: %PROJECT_DIR%
echo [start] python cmd : %PYTHON_CMD%

rem --- check gateway binary ---
if not exist "%PROJECT_DIR%server\llm-gateway.exe" (
    echo [start] llm-gateway.exe not found, building...
    pushd "%PROJECT_DIR%server"
    go build -o llm-gateway.exe . || (popd & echo [start] BUILD FAILED & pause & exit /b 1)
    popd
)

rem --- free the port if a stale gateway is running ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%GATEWAY_PORT% " ^| findstr "LISTENING"') do (
    echo [start] killing stale process on port %GATEWAY_PORT% (pid %%p)
    taskkill /F /PID %%p >nul 2>&1
)

echo [start] gateway running at http://localhost:%GATEWAY_PORT%/web/
echo [start] press Ctrl+C to stop
start "" http://localhost:%GATEWAY_PORT%/web/

cd /d "%PROJECT_DIR%server"
set PYTHON_CMD=%PYTHON_CMD%
llm-gateway.exe
pause
