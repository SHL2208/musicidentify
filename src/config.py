import os
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "data" / "temp"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Google AI Studio Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.1-flash-lite")
AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite"
]

# Audio Processing Configuration
AUDIO_SAMPLE_DURATION = int(os.getenv("AUDIO_SAMPLE_DURATION", "30"))
AUDIO_SAMPLE_OFFSET = int(os.getenv("AUDIO_SAMPLE_OFFSET", "45"))
BATCH_SLEEP_INTERVAL = float(os.getenv("BATCH_SLEEP_INTERVAL", "3.0"))

def get_ffmpeg_binary() -> str:
    """Trả về đường dẫn tới ffmpeg executable (từ PATH hoặc từ imageio-ffmpeg)."""
    import shutil
    ffmpeg_sys = shutil.which("ffmpeg")
    if ffmpeg_sys:
        return ffmpeg_sys
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"
