@echo off
chcp 65001 > nul
echo =======================================================
echo    🎵 AI MUSIC DETECTOR AGENT - FORENSIC 4-TIER 🎵
echo =======================================================

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Đang khởi động giao diện Web bằng uv...
    uv run streamlit run src\ui\app.py
) else (
    if exist .venv\Scripts\python.exe (
        echo Đang khởi động giao diện Web bằng .venv...
        .venv\Scripts\python.exe -m streamlit run src\ui\app.py
    ) else (
        echo Không tìm thấy Python hoặc uv. Vui lòng cài đặt uv hoặc tạo môi trường .venv.
        pause
    )
)
pause
