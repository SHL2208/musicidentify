@echo off
chcp 65001 > nul
echo =======================================================
echo    🎵 AI MUSIC DETECTOR AGENT - FAANG ARCHITECTURE 🎵
echo =======================================================

:: Tự động bổ sung PATH nếu uv cài trong Python Scripts hoặc .local\bin
if exist "%LOCALAPPDATA%\Programs\Python\Python312\Scripts\uv.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
)
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Đang khởi động giao diện Web bằng uv (app.py)...
    uv run streamlit run app.py
) else (
    echo [ERROR] Không tìm thấy uv trên hệ thống. Vui lòng cài đặt uv hoặc thêm uv vào biến môi trường PATH.
    pause
)
