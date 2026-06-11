@echo off
setlocal
set PY=%~1
if "%PY%"=="" set PY=C:\Users\HP\.conda\envs\agent_env\python.exe

echo Using: %PY%
echo.
echo [1/2] Upgrading charset-normalizer (fix _FREQUENCIES_SET import)...
"%PY%" -m pip install --upgrade --force-reinstall "charset-normalizer>=3.4.0"
if errorlevel 1 (
    echo.
    echo Install failed. Stop app.py / Docker web first, then run this script again.
    exit /b 1
)

echo.
echo [2/2] Installing document parse dependencies...
"%PY%" -m pip install "pdfplumber>=0.10.0" "markitdown[docx,pptx,xlsx]>=0.1.2"
if errorlevel 1 exit /b 1

echo.
echo Verifying imports...
"%PY%" -c "from charset_normalizer.constant import _FREQUENCIES_SET; from markitdown import MarkItDown; import pdfplumber; print('All OK')"
if errorlevel 1 exit /b 1

echo.
echo Done. Restart: python app.py
pause
