@echo off
rem ============================================
rem  llm_go stop: kill gateway + llama-server
rem ============================================
echo [stop] killing llm-gateway / llama-server ...
taskkill /F /IM llm-gateway.exe >nul 2>&1
taskkill /F /IM llama-server.exe >nul 2>&1
echo [stop] done.
timeout /t 2 >nul
