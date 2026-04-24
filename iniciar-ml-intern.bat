@echo off
color 0A
echo ==========================================
echo   ?? ML-Intern - Hugging Face AI Agent
echo ==========================================
echo.
echo Activando entorno virtual...
call .venv\Scripts\activate.bat
echo.
echo Para salir escribe: exit o presiona Ctrl+C
echo.
.venv\Scripts\ml-intern.exe %*
pause
