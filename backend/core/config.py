import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Tự động nạp biến môi trường từ .env
load_dotenv()

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

DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL")
AVAILABLE_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]

# Audio Processing Configuration
AUDIO_SAMPLE_DURATION = int(os.getenv("AUDIO_SAMPLE_DURATION", "30"))
AUDIO_SAMPLE_OFFSET = int(os.getenv("AUDIO_SAMPLE_OFFSET", "45"))
BATCH_SLEEP_INTERVAL = float(os.getenv("BATCH_SLEEP_INTERVAL", "3.0"))

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
