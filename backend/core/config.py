import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Tự động nạp biến môi trường từ .env (khi chạy Local)
load_dotenv()

# Tự động đồng bộ Streamlit Secrets vào os.environ (khi chạy trên Streamlit Cloud)
try:
    import streamlit as st

    if hasattr(st, "secrets"):
        for _sk, _sv in st.secrets.items():
            if isinstance(_sv, (str, int, float, bool)) and _sk not in os.environ:
                os.environ[_sk] = str(_sv)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = BASE_DIR / "data" / "temp"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Google AI Studio Configuration
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1", "").strip()
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "").strip()
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3", "").strip()

# Thu thập các key có giá trị theo đúng thứ tự ưu tiên 1 > 2 > 3...
_keys_list: list[str] = []
for i in range(1, 10):
    k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
    if k and k not in _keys_list:
        _keys_list.append(k)

# Tương thích ngược: nếu có biến GEMINI_API_KEY đơn lẻ
_legacy_key = os.getenv("GEMINI_API_KEY", "").strip()
if _legacy_key and _legacy_key not in _keys_list:
    _keys_list.insert(0, _legacy_key)

GEMINI_API_KEYS: list[str] = _keys_list
GEMINI_API_KEY: str = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# Model Selection: Gemini 3.8 Flash (chính) -> Gemini 3.7 Flash -> Gemini 3.6 Flash (dự phòng)
DEFAULT_GEMINI_MODEL = (
    os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.8-flash") or "gemini-3.8-flash"
).strip()

# Thứ tự mô hình dự phòng tự động khi cạn Quota hoặc lỗi (Multi-tier Model Failover)
DEFAULT_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

_env_fallback = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
if _env_fallback:
    GEMINI_FALLBACK_MODELS = [m.strip() for m in _env_fallback.split(",") if m.strip()]
else:
    GEMINI_FALLBACK_MODELS = DEFAULT_FALLBACK_MODELS

AVAILABLE_MODELS = list(
    dict.fromkeys(
        GEMINI_FALLBACK_MODELS
        + [
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]
    )
)

# Audio Processing Configuration
AUDIO_SAMPLE_DURATION = int(os.getenv("AUDIO_SAMPLE_DURATION", "30"))
AUDIO_SAMPLE_OFFSET = int(os.getenv("AUDIO_SAMPLE_OFFSET", "45"))
BATCH_SLEEP_INTERVAL = float(os.getenv("BATCH_SLEEP_INTERVAL", "3.0"))

# Media Cloud Storage Configuration
CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD = os.getenv(
    "CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD", "true"
).strip().lower() in ("true", "1", "yes")

# Server / API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


def get_ffmpeg_binary() -> str:
    """Trả về đường dẫn tới ffmpeg executable (từ PATH hoặc từ imageio-ffmpeg)."""
    ffmpeg_sys = shutil.which("ffmpeg")
    if ffmpeg_sys:
        return ffmpeg_sys
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_cookie_file_path() -> Path | None:
    """
    Trả về đường dẫn file cookies.txt hợp lệ nếu có.
    Hỗ trợ:
    1. Biến môi trường YOUTUBE_COOKIES_CONTENT (dành cho Cloud Secrets như Hugging Face / Streamlit Cloud).
    2. Các file cục bộ: cookies.txt hoặc cookies_master.txt tại BASE_DIR hoặc TEMP_DIR.
    """
    env_cookies = os.getenv("YOUTUBE_COOKIES_CONTENT", "").strip()
    if not env_cookies:
        try:
            import streamlit as st

            if hasattr(st, "secrets") and "YOUTUBE_COOKIES_CONTENT" in st.secrets:
                env_cookies = str(st.secrets["YOUTUBE_COOKIES_CONTENT"]).strip()
        except Exception:
            pass

    if env_cookies:
        runtime_cookie = TEMP_DIR / "cookie_runtime.txt"
        try:
            runtime_cookie.write_text(env_cookies, encoding="utf-8")
            return runtime_cookie
        except Exception:
            pass

    for candidate in [
        BASE_DIR / "cookies.txt",
        BASE_DIR / "cookies_master.txt",
        TEMP_DIR / "cookies.txt",
        TEMP_DIR / "cookies_master.txt",
        TEMP_DIR / "cookie_runtime.txt",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    return None
