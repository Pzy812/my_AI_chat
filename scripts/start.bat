@echo off
setlocal
cd /d "%~dp0.."

if not exist ".env" (
    if exist ".env.example" (
        echo [提示] 未找到 .env，已从 .env.example 复制一份，请编辑 .env 填入 ZHIPUAI_API_KEY
        copy /Y ".env.example" ".env" >nul
    ) else (
        echo [错误] 未找到 .env，请创建并配置 ZHIPUAI_API_KEY
        exit /b 1
    )
)

echo Starting AI Chat Web on http://localhost:5001
echo MCP will auto-start if not running on port 8090
echo.
python app.py
exit /b %ERRORLEVEL%
