import base64
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Tự động nạp biến môi trường từ .env (khi chạy Local)
load_dotenv()

# Tự động đồng bộ toàn bộ Streamlit Secrets vào os.environ (hỗ trợ cả các cấu trúc lồng nhau)
try:
    import streamlit as st

    if hasattr(st, "secrets"):

        def _sync_secrets_to_env(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict) or hasattr(obj, "items"):
                for k, v in obj.items():
                    k_str = f"{prefix}_{k}".strip("_").upper()
                    if isinstance(v, (str, int, float, bool)):
                        if k_str not in os.environ:
                            os.environ[k_str] = str(v)
                        if str(k) not in os.environ:
                            os.environ[str(k)] = str(v)
                    elif isinstance(v, (dict, list)) or hasattr(v, "items"):
                        _sync_secrets_to_env(v, prefix=k_str)

        _sync_secrets_to_env(st.secrets)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = BASE_DIR / "data" / "temp"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_SNAPSHOT_DIR = BASE_DIR / "data" / "catalogs"
CATALOG_SNAPSHOT_PATH = CATALOG_SNAPSHOT_DIR / "catalog_snapshot.json"
CATALOG_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def get_configured_gemini_api_keys() -> list[str]:
    """
    Thu thập động danh sách Gemini API Keys theo thứ tự ưu tiên:
    1. Streamlit Secrets (st.secrets)
    2. Biến môi trường hệ thống (os.environ)
    3. File .env
    Hỗ trợ cả GEMINI_API_KEY_1..9, GEMINI_API_KEY, GOOGLE_API_KEY.
    """
    keys: list[str] = []

    # 1. Thử từ Streamlit Secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for i in range(1, 10):
                k = f"GEMINI_API_KEY_{i}"
                if k in st.secrets and str(st.secrets[k]).strip():
                    val = str(st.secrets[k]).strip()
                    if val not in keys:
                        keys.append(val)
            for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "api_key"]:
                if k in st.secrets and str(st.secrets[k]).strip():
                    val = str(st.secrets[k]).strip()
                    if val not in keys:
                        keys.append(val)
    except Exception:
        pass

    # 2. Thử từ os.environ
    for i in range(1, 10):
        val = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if val and val not in keys:
            keys.append(val)

    for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(k, "").strip()
        if val and val not in keys:
            keys.append(val)

    return keys


# Google AI Studio Configuration
GEMINI_API_KEYS: list[str] = get_configured_gemini_api_keys()
GEMINI_API_KEY: str = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# Model Selection: Gemini 3.8 Flash (chính) -> Gemini 3.7 Flash -> Gemini 3.6 Flash (dự phòng)
DEFAULT_GEMINI_MODEL = (
    os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite"
).strip()

# Thứ tự mô hình dự phòng tự động khi cạn Quota hoặc lỗi (Multi-tier Model Failover)
DEFAULT_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
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
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ]
    )
)

# ==============================================================================
# SECTION 3: AUDIO ENGINE CONFIGURATION
# ==============================================================================
AUDIO_SAMPLE_DURATION = int(os.getenv("AUDIO_SAMPLE_DURATION", "45"))
AUDIO_SAMPLE_OFFSET = int(os.getenv("AUDIO_SAMPLE_OFFSET", "45"))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "2"))
MULTI_SEGMENT_SLICE_DURATION = float(os.getenv("MULTI_SEGMENT_SLICE_DURATION", "15.0"))
MULTI_SEGMENT_POSITIONS = (0.20, 0.50, 0.75)
BATCH_SLEEP_INTERVAL = float(os.getenv("BATCH_SLEEP_INTERVAL", "0.5"))
PIPED_TIMEOUT_SECONDS = float(os.getenv("PIPED_TIMEOUT_SECONDS", "4.0"))
SPECTROGRAM_DPI = int(os.getenv("SPECTROGRAM_DPI", "80"))
ENABLE_CASCADED_EVALUATION = os.getenv("ENABLE_CASCADED_EVALUATION", "true").strip().lower() in (
    "true",
    "1",
    "yes",
)
ENABLE_RANGE_STREAM_DOWNLOAD = os.getenv(
    "ENABLE_RANGE_STREAM_DOWNLOAD", "true"
).strip().lower() in ("true", "1", "yes")

# ==============================================================================
# SECTION 4: ACOUSTIC ANALYSIS CONFIGURATION (THRESHOLDS & PARAMETERS)
# ==============================================================================
# 4.1. Pitch Micro-Jitter & Rigidity Analysis
PITCH_TARGET_SR = int(os.getenv("PITCH_TARGET_SR", "8000"))
PITCH_WINDOW_SEC = float(os.getenv("PITCH_WINDOW_SEC", "20.0"))
PITCH_FMIN = float(os.getenv("PITCH_FMIN", "80.0"))
PITCH_FMAX = float(os.getenv("PITCH_FMAX", "550.0"))
PITCH_VOICED_F0_MIN = float(os.getenv("PITCH_VOICED_F0_MIN", "85.0"))
PITCH_VOICED_F0_MAX = float(os.getenv("PITCH_VOICED_F0_MAX", "500.0"))
PITCH_MIN_VOICED_FRAMES = int(os.getenv("PITCH_MIN_VOICED_FRAMES", "30"))
PITCH_FRAME_LENGTH = int(os.getenv("PITCH_FRAME_LENGTH", "512"))
PITCH_HOP_LENGTH = int(os.getenv("PITCH_HOP_LENGTH", "128"))
PITCH_RIGID_DIFF_THRESHOLD = float(os.getenv("PITCH_RIGID_DIFF_THRESHOLD", "0.8"))
PITCH_STUDIO_AUTOTUNE_RIGIDITY = float(os.getenv("PITCH_STUDIO_AUTOTUNE_RIGIDITY", "0.35"))
PITCH_STUDIO_AUTOTUNE_JITTER_MAX = float(os.getenv("PITCH_STUDIO_AUTOTUNE_JITTER_MAX", "0.015"))
PITCH_STUDIO_AUTOTUNE_FORMANT_MIN = float(os.getenv("PITCH_STUDIO_AUTOTUNE_FORMANT_MIN", "300.0"))
PITCH_ANOMALY_RIGIDITY_THRESHOLD = float(os.getenv("PITCH_ANOMALY_RIGIDITY_THRESHOLD", "0.40"))
PITCH_ANOMALY_JITTER_THRESHOLD = float(os.getenv("PITCH_ANOMALY_JITTER_THRESHOLD", "0.015"))
PITCH_ANOMALY_RIGIDITY_SECONDARY = float(os.getenv("PITCH_ANOMALY_RIGIDITY_SECONDARY", "0.32"))
PITCH_MILD_AUTOTUNE_RIGIDITY = float(os.getenv("PITCH_MILD_AUTOTUNE_RIGIDITY", "0.25"))

# 4.2. Stem Bleed Analysis
STEM_BLEED_WINDOW_SEC = float(os.getenv("STEM_BLEED_WINDOW_SEC", "20.0"))
STEM_BLEED_HOP_LENGTH = int(os.getenv("STEM_BLEED_HOP_LENGTH", "512"))
STEM_BLEED_THRESHOLD = float(os.getenv("STEM_BLEED_THRESHOLD", "0.40"))

# 4.3. Spectral STFT & Ultrasonic Shelf Cutoff
SPECTRAL_STFT_N_FFT = int(os.getenv("SPECTRAL_STFT_N_FFT", "2048"))
SPECTRAL_STFT_HOP_LENGTH = int(os.getenv("SPECTRAL_STFT_HOP_LENGTH", "512"))
SPECTRAL_ROLLOFF_PERCENT_HIGH = float(os.getenv("SPECTRAL_ROLLOFF_PERCENT_HIGH", "0.95"))
SPECTRAL_ROLLOFF_PERCENT_MID = float(os.getenv("SPECTRAL_ROLLOFF_PERCENT_MID", "0.85"))
ULTRASONIC_SHELF_DROP_DB_THRESHOLD = float(os.getenv("ULTRASONIC_SHELF_DROP_DB_THRESHOLD", "35.0"))
ULTRASONIC_ROLLOFF_CUTOFF_HZ = float(os.getenv("ULTRASONIC_ROLLOFF_CUTOFF_HZ", "13500.0"))
ULTRASONIC_HIGH_FREQ_RATIO_MAX = float(os.getenv("ULTRASONIC_HIGH_FREQ_RATIO_MAX", "0.0002"))

# 4.4. Vocal Separation & Instrumental Detection
VOCAL_BAND_FREQ_MIN = float(os.getenv("VOCAL_BAND_FREQ_MIN", "300.0"))
VOCAL_BAND_FREQ_MAX = float(os.getenv("VOCAL_BAND_FREQ_MAX", "3400.0"))
INSTRUMENTAL_VOCAL_ENERGY_MAX = float(os.getenv("INSTRUMENTAL_VOCAL_ENERGY_MAX", "0.20"))

# 4.5. Modern AI Vocal Suspect Thresholds
MODERN_AI_ROLLOFF_MAX = float(os.getenv("MODERN_AI_ROLLOFF_MAX", "11800.0"))
MODERN_AI_RIGIDITY_MIN = float(os.getenv("MODERN_AI_RIGIDITY_MIN", "0.47"))
MODERN_AI_HARMONIC_MIN = float(os.getenv("MODERN_AI_HARMONIC_MIN", "0.80"))
MODERN_AI_FAKEPRINT_MIN = float(os.getenv("MODERN_AI_FAKEPRINT_MIN", "0.35"))
MODERN_AI_STEM_BLEED_MIN = float(os.getenv("MODERN_AI_STEM_BLEED_MIN", "0.32"))
MODERN_AI_ULTRASONIC_RIGIDITY_MIN = float(os.getenv("MODERN_AI_ULTRASONIC_RIGIDITY_MIN", "0.55"))

# 4.6. Compression Source Classification
COMPRESSION_LOSSY_ROLLOFF_MAX = float(os.getenv("COMPRESSION_LOSSY_ROLLOFF_MAX", "17500.0"))
COMPRESSION_STUDIO_ROLLOFF_MIN = float(os.getenv("COMPRESSION_STUDIO_ROLLOFF_MIN", "19000.0"))

# 4.7. Safeguard Physical Thresholds (dùng chung cho DetectionService & SignalAnalyzer)
SAFEGUARD_FAKEPRINT_OVERRIDE_HIGH = float(os.getenv("SAFEGUARD_FAKEPRINT_OVERRIDE_HIGH", "0.85"))
SAFEGUARD_STEM_BLEED_OVERRIDE_HIGH = float(os.getenv("SAFEGUARD_STEM_BLEED_OVERRIDE_HIGH", "0.50"))
SAFEGUARD_FAKEPRINT_MODERATE = float(os.getenv("SAFEGUARD_FAKEPRINT_MODERATE", "0.50"))
SAFEGUARD_STEM_BLEED_MODERATE = float(os.getenv("SAFEGUARD_STEM_BLEED_MODERATE", "0.35"))
SAFEGUARD_FAKEPRINT_STRONG = float(os.getenv("SAFEGUARD_FAKEPRINT_STRONG", "0.65"))
SAFEGUARD_STEM_BLEED_STRONG = float(os.getenv("SAFEGUARD_STEM_BLEED_STRONG", "0.30"))
SAFEGUARD_FAKEPRINT_RELIABLE = float(os.getenv("SAFEGUARD_FAKEPRINT_RELIABLE", "0.60"))
SAFEGUARD_HUMAN_FAKEPRINT_MAX = float(os.getenv("SAFEGUARD_HUMAN_FAKEPRINT_MAX", "0.25"))
SAFEGUARD_HUMAN_STEM_BLEED_MAX = float(os.getenv("SAFEGUARD_HUMAN_STEM_BLEED_MAX", "0.25"))

# ==============================================================================
# SECTION 5: HEURISTICS & RULES CONFIGURATION
# ==============================================================================
PRE_AI_CUTOFF_YEAR = int(os.getenv("PRE_AI_CUTOFF_YEAR", "2022"))
PRE_AI_CUTOFF_MONTH = int(os.getenv("PRE_AI_CUTOFF_MONTH", "11"))
PRE_AI_MATCH_SCORE_MIN = float(os.getenv("PRE_AI_MATCH_SCORE_MIN", "0.75"))
PRE_AI_RECENT_TARGET_YEAR = int(os.getenv("PRE_AI_RECENT_TARGET_YEAR", "2023"))

SUNO_EXPLICIT_KEYWORDS = (
    "created with suno",
    "#suno",
    "suno ai",
    "suno.com",
)
UDIO_EXPLICIT_KEYWORDS = (
    "created with udio",
    "#udio",
    "udio ai",
    "udio.com",
)
DERIVATIVE_KEYWORDS = (
    "cover",
    "remake",
    "remix",
    "ai cover",
)
GENERIC_ANONYMOUS_ARTISTS = {
    "",
    "unknown",
    "various artists",
    "nghệ sĩ ẩn danh",
    "ca sĩ ẩn danh",
    "ẩn danh",
    "anonymous",
    "nhiều nghệ sĩ",
    "nhiều ca sĩ",
    "ai channel",
    "ai music",
    "va",
    "none",
    "null",
}

AI_CONTEXT_PATTERN = (
    r"\b(?:suno|udio)\b|ai\s+music|ai\s+cover|tạo bằng ai|mô hình ai|prompt:|style:"
)
SUNO_PROMPT_TAGS_PATTERN = (
    r"\[\s*(?:verse\s*\d*|chorus|bridge|outro|intro|pre-chorus|hook|vocal chop|sub-?bass)[^\]]*\]"
)
BATCH_AI_NUMBER_PATTERN = r"bài\s*(?:số\s*)?\d+\s*/\s*\d+"
BATCH_AI_KEYWORDS_PATTERN = r"\b(?:suno|udio|ai)\b|ai\s+music|ai\s+cover|#ai|tạo bằng ai|sinh bằng ai|dự án\s+(?:sáng tác|nhạc|ai)"
NATIONAL_ARTIST_HONORIFICS_PATTERN = r"\b(nsnd|nsưt|nsut)\b"
COLLABORATOR_SPLIT_PATTERN = r"[,/&;+]|(?:\s+(?:x|ft\.?|feat\.?|vs\.?)\s+)"
PRODUCER_PREFIX_PATTERN = r"^prod\.?\s*"

# Cache TTL cho danh mục Database (giây)
CATALOG_CACHE_TTL_SECONDS = int(os.getenv("CATALOG_CACHE_TTL_SECONDS", "300"))
MAJOR_EXCLUSIVE_LABELS = ("sony", "universal", "umg", "vieent", "loops", "bmz")

# ==============================================================================
# SECTION 6: MEDIA CLOUD STORAGE & SERVER CONFIGURATION
# ==============================================================================
CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD = os.getenv(
    "CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD", "true"
).strip().lower() in ("true", "1", "yes")
UPLOAD_AUDIO_SAMPLES_TO_CLOUD = os.getenv(
    "UPLOAD_AUDIO_SAMPLES_TO_CLOUD", "false"
).strip().lower() in ("true", "1", "yes")

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


def _download_standalone_deno() -> Path | None:
    """
    Tự động tải bản phân phối Deno standalone chính thức từ GitHub releases
    nếu môi trường chưa có JavaScript runtime (Deno, Node >= 22, Bun, QuickJS).
    Deno là runtime JavaScript được yt-dlp khuyến nghị chính thức (enabled by default)
    để giải quyết triệt để YouTube n challenge / cipher signatures trên Cloud Datacenter.
    """
    import platform
    import urllib.request
    import zipfile

    system = platform.system().lower()
    machine = platform.machine().lower()

    if "linux" in system:
        asset = (
            "deno-aarch64-unknown-linux-gnu.zip"
            if ("aarch64" in machine or "arm64" in machine)
            else "deno-x86_64-unknown-linux-gnu.zip"
        )
    elif "darwin" in system:
        asset = (
            "deno-aarch64-apple-darwin.zip"
            if ("arm64" in machine or "aarch64" in machine)
            else "deno-x86_64-apple-darwin.zip"
        )
    elif "windows" in system or os.name == "nt":
        asset = "deno-x86_64-pc-windows-msvc.zip"
    else:
        return None

    bin_dir = TEMP_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "deno.exe" if os.name == "nt" else "deno"
    target_exe = bin_dir / exe_name

    if target_exe.exists() and (os.name == "nt" or os.access(target_exe, os.X_OK)):
        return target_exe

    url = f"https://github.com/denoland/deno/releases/latest/download/{asset}"
    zip_dest = bin_dir / f"deno_tmp_{os.getpid()}.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(zip_dest, "wb") as f:
                f.write(content)

        with zipfile.ZipFile(zip_dest, "r") as z:
            z.extractall(bin_dir)

        if target_exe.exists():
            if os.name != "nt":
                current_mode = os.stat(target_exe).st_mode
                os.chmod(target_exe, current_mode | 0o755)
            return target_exe
    except Exception:
        pass
    finally:
        if zip_dest.exists():
            zip_dest.unlink(missing_ok=True)

    return None


def _ensure_in_path(d: Path) -> None:
    """Đảm bảo thư mục binary nằm trong PATH môi trường."""
    d_str = str(d.resolve())
    curr_path = os.environ.get("PATH", "")
    paths = curr_path.split(os.pathsep)
    if d_str not in paths:
        os.environ["PATH"] = f"{d_str}{os.pathsep}{curr_path}"


def get_js_runtimes_config() -> dict[str, Any]:
    """
    Tự động phát hiện, xác minh và cấu hình JavaScript Runtime tương thích (Deno, Node.js >= 22, Bun, QuickJS)
    cho yt-dlp để giải mã JavaScript challenge (n challenge, cipher signature).
    Nếu trên môi trường Cloud (Linux) không có JS runtime hợp lệ hoặc Node.js < 22,
    hàm sẽ tự động tải và kích hoạt binary Deno standalone chính thức từ GitHub releases.
    """
    # 1. Kiểm tra Deno có sẵn trên hệ thống hoặc trong thư mục bin cục bộ
    cached_deno = TEMP_DIR / "bin" / ("deno.exe" if os.name == "nt" else "deno")
    deno_candidates = [
        shutil.which("deno"),
        str(cached_deno) if cached_deno.exists() else None,
        "/usr/bin/deno",
        str(Path.home() / ".deno" / "bin" / "deno"),
    ]
    for cand in deno_candidates:
        if cand and Path(cand).exists():
            try:
                from yt_dlp.utils._jsruntime import DenoJsRuntime

                rt = DenoJsRuntime(cand)
                if rt.info and rt.info.supported:
                    _ensure_in_path(Path(cand).parent)
                    return {"deno": {"path": str(cand)}}
            except Exception:
                pass

    # 2. Kiểm tra Node.js (phải đạt version >= 22 theo yêu cầu của yt-dlp để dùng permission model)
    node_candidates = [
        shutil.which("node"),
        shutil.which("nodejs"),
        "/usr/bin/node",
        "/usr/bin/nodejs",
        r"C:\Program Files\nodejs\node.exe",
    ]
    for cand in node_candidates:
        if cand and Path(cand).exists():
            try:
                from yt_dlp.utils._jsruntime import NodeJsRuntime

                rt = NodeJsRuntime(cand)
                if rt.info and rt.info.supported:
                    _ensure_in_path(Path(cand).parent)
                    return {"node": {"path": str(cand)}}
            except Exception:
                pass

    # 3. Kiểm tra Bun
    bun_cand = shutil.which("bun")
    if bun_cand and Path(bun_cand).exists():
        try:
            from yt_dlp.utils._jsruntime import BunJsRuntime

            rt = BunJsRuntime(bun_cand)
            if rt.info and rt.info.supported:
                return {"bun": {"path": str(bun_cand)}}
        except Exception:
            pass

    # 4. Kiểm tra QuickJS
    qjs_cand = shutil.which("qjs") or shutil.which("quickjs")
    if qjs_cand and Path(qjs_cand).exists():
        try:
            from yt_dlp.utils._jsruntime import QuickJsRuntime

            rt = QuickJsRuntime(qjs_cand)
            if rt.info and rt.info.supported:
                return {"quickjs": {"path": str(qjs_cand)}}
        except Exception:
            pass

    # 5. Nếu chưa có JS runtime nào đạt chuẩn yt-dlp, tự động tải Deno standalone binary
    downloaded_deno = _download_standalone_deno()
    if downloaded_deno and downloaded_deno.exists():
        try:
            from yt_dlp.utils._jsruntime import DenoJsRuntime

            rt = DenoJsRuntime(str(downloaded_deno))
            if rt.info and rt.info.supported:
                _ensure_in_path(downloaded_deno.parent)
                return {"deno": {"path": str(downloaded_deno)}}
        except Exception:
            pass

    return {}


def normalize_netscape_cookies(raw_text: str) -> tuple[str, str, int, int]:
    """
    Chuẩn hóa nội dung cookies sang định dạng chuẩn Netscape HTTP Cookie File cho yt-dlp.
    Tạo ra 2 phiên bản độc lập:
    1. Full Cookies (toàn bộ cookie gốc, giữ nguyên các token chuẩn kể cả SIDTS, phục hồi tab).
    2. Guest/Visitor Cookies (lọc chỉ giữ lại các token cookie khách an toàn trên Cloud Datacenter,
       loại bỏ hoàn toàn session tài khoản cá nhân SID/LOGIN_INFO/PSID/SIDCC để ngăn YouTube
       kích hoạt cơ chế chống chiếm đoạt tài khoản: 'The page needs to be reloaded').

    Trả về: (full_content, guest_content, full_count, guest_count)
    """
    if not raw_text or not raw_text.strip():
        return "", "", 0, 0

    content = raw_text.strip()

    # Bóc tách dấu ngoặc kép nếu người dùng dán cả ngoặc
    for q in ['"""', "'''"]:
        if content.startswith(q) and content.endswith(q):
            content = content[len(q) : -len(q)].strip()

    # Hỗ trợ tiền tố base64: hoặc tự động nhận diện Base64
    if content.startswith("base64:"):
        content = content[7:].strip()

    if "\t" not in content and "\n" not in content[:200] and len(content) > 50:
        try:
            decoded = base64.b64decode(content, validate=True).decode("utf-8", errors="replace")
            if any(
                k in decoded.lower()
                for k in ["youtube", "netscape", "domain", "true", "false", "sid"]
            ):
                content = decoded
        except Exception:
            pass

    # Xử lý chuỗi escape nếu có
    if "\\n" in content and "\n" not in content:
        content = content.replace("\\r\\n", "\n").replace("\\n", "\n")
    if "\\t" in content:
        content = content.replace("\\t", "\t")

    lines = content.splitlines()
    full_cookies_dict: dict[tuple[str, str, str], str] = {}
    guest_cookies_dict: dict[tuple[str, str, str], str] = {}

    # Danh sách cookie khách an toàn không mang định danh tài khoản cá nhân
    guest_safe_names = {
        "VISITOR_INFO1_LIVE",
        "VISITOR_PRIVACY_METADATA",
        "PREF",
        "YSC",
        "__Secure-BUCKET",
        "GPS",
        "SOCS",
        "CONSENT",
    }

    # Danh sách các token tài khoản cá nhân gây lỗi 'The page needs to be reloaded' khi dùng chéo IP
    account_session_names = {
        "LOGIN_INFO",
        "SID",
        "HSID",
        "SSID",
        "APISID",
        "SAPISID",
        "__Secure-1PAPISID",
        "__Secure-3PAPISID",
        "__Secure-1PSID",
        "__Secure-3PSID",
        "__Secure-1PSIDTS",
        "__Secure-3PSIDTS",
        "SIDCC",
        "__Secure-1PSIDCC",
        "__Secure-3PSIDCC",
        "__Secure-YNID",
        "__Secure-ROLLOUT_TOKEN",
        "DEVICE_INFO",
        "__Secure-YT_TVFAS",
        "__Secure-YT_DERP",
    }

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue

        domain = sub = path = sec = exp = name = val = ""
        # Trường hợp 1: Dòng đã có ít nhất 6 dấu tab chuẩn Netscape
        if line_str.count("\t") >= 6:
            parts = line_str.split("\t")
            if len(parts) >= 7 and parts[0]:
                domain, sub, path, sec, exp, name = parts[:6]
                val = parts[6]
            elif len(parts) == 6 and parts[0]:
                domain, sub, path, sec, exp, name = parts[:6]
                val = ""
        else:
            # Trường hợp 2: Dòng bị biến thành khoảng trắng do copy-paste qua web
            parts = re.split(r"\s+", line_str)
            if len(parts) >= 7 and parts[0].startswith(
                (".", "http", "www", "youtube", "google", "zing")
            ):
                domain, sub, path, sec, exp, name = parts[:6]
                val = " ".join(parts[6:])
            elif len(parts) == 6 and parts[0].startswith(
                (".", "http", "www", "youtube", "google", "zing")
            ):
                domain, sub, path, sec, exp, name = parts[:6]
                val = ""

        if domain and name:
            cookie_key = (domain.lower(), path, name)
            cookie_line = f"{domain}\t{sub}\t{path}\t{sec}\t{exp}\t{name}\t{val}"
            # Ghi đè theo cookie_key để tự động loại bỏ trùng lặp nếu người dùng paste nhiều lần
            full_cookies_dict[cookie_key] = cookie_line

            # Lọc cookie khách: chỉ giữ token khách an toàn, bỏ token session cá nhân
            if name in guest_safe_names or (
                name not in account_session_names and not name.startswith(("__Secure-", "SID"))
            ):
                guest_cookies_dict[cookie_key] = cookie_line

    header = "# Netscape HTTP Cookie File\n# This file is generated for yt-dlp.\n\n"
    full_cookies = list(full_cookies_dict.values())
    guest_cookies = list(guest_cookies_dict.values())
    full_content = (header + "\n".join(full_cookies) + "\n") if full_cookies else ""
    guest_content = (header + "\n".join(guest_cookies) + "\n") if guest_cookies else ""

    return full_content, guest_content, len(full_cookies), len(guest_cookies)


def find_cookie_content_from_secrets_or_env() -> str:
    """Tìm kiếm nội dung cookie YouTube từ các biến môi trường hoặc Streamlit Secrets."""
    candidate_keys = [
        "YOUTUBE_COOKIES_BASE64",
        "YOUTUBE_COOKIES_CONTENT",
        "YOUTUBE_COOKIES",
        "COOKIES_BASE64",
        "COOKIES_CONTENT",
        "COOKIES",
        "COOKIE",
        "cookies",
        "youtube_cookies",
        "COOKIES_TXT",
    ]
    for key in candidate_keys:
        val = os.getenv(key, "").strip()
        if val:
            return val

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in candidate_keys:
                if key in st.secrets and str(st.secrets[key]).strip():
                    return str(st.secrets[key]).strip()

            def search_secrets(obj: Any) -> str | None:
                if isinstance(obj, dict) or hasattr(obj, "items"):
                    for k, v in obj.items():
                        if isinstance(v, str):
                            k_l = str(k).lower()
                            if ("cookie" in k_l or "youtube" in k_l) and len(v.strip()) > 30:
                                return v.strip()
                            if (
                                ".youtube.com" in v
                                or "LOGIN_INFO" in v
                                or "Netscape" in v
                                or "__Secure-" in v
                            ):
                                return v.strip()
                        elif isinstance(v, (dict, list)) or hasattr(v, "items"):
                            res = search_secrets(v)
                            if res:
                                return res
                return None

            found = search_secrets(st.secrets)
            if found:
                return found
    except Exception:
        pass

    return ""


def get_all_cookie_file_paths(mode: str = "full") -> list[Path]:
    """
    Trả về danh sách tất cả các đường dẫn file cookie hợp lệ theo thứ tự ưu tiên (cookies.txt, cookies_master.txt...).
    Hỗ trợ hệ thống tự động fallback nếu một file cookie bị chặn hoặc hết hạn.
    """
    results: list[Path] = []
    seen_hashes: set[str] = set()

    # 1. Từ Secrets hoặc biến môi trường nếu có
    raw_content = find_cookie_content_from_secrets_or_env()
    if raw_content:
        full_c, guest_c, full_cnt, guest_cnt = normalize_netscape_cookies(raw_content)
        target_c = guest_c if mode == "guest" else full_c
        target_cnt = guest_cnt if mode == "guest" else full_cnt
        if target_c and target_cnt > 0:
            c_hash = hashlib.md5(target_c.encode("utf-8")).hexdigest()
            cookie_path = TEMP_DIR / f"cookie_env_{mode}.txt"
            try:
                cookie_path.write_text(target_c, encoding="utf-8")
                results.append(cookie_path)
                seen_hashes.add(c_hash)
            except Exception:
                pass

    # 2. Từ các file vật lý (ưu tiên cookies.txt rồi đến cookies_master.txt)
    candidates = [
        BASE_DIR / "cookies.txt",
        BASE_DIR / "cookies_master.txt",
        TEMP_DIR / "cookies.txt",
        TEMP_DIR / "cookies_master.txt",
    ]
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            try:
                content = cand.read_text(encoding="utf-8", errors="replace")
                full_c, guest_c, full_cnt, guest_cnt = normalize_netscape_cookies(content)
                target_c = guest_c if mode == "guest" else full_c
                target_cnt = guest_cnt if mode == "guest" else full_cnt
                if target_c and target_cnt > 0:
                    c_hash = hashlib.md5(target_c.encode("utf-8")).hexdigest()
                    if c_hash not in seen_hashes:
                        seen_hashes.add(c_hash)
                        stem = cand.stem
                        out_path = TEMP_DIR / f"cookie_{stem}_{mode}_runtime.txt"
                        out_path.write_text(target_c, encoding="utf-8")
                        results.append(out_path)
            except Exception:
                pass

    return results


def get_cookie_file_path(mode: str = "guest") -> Path | None:
    """
    Trả về đường dẫn file cookies.txt hợp lệ nhất nếu có.
    mode='guest': Cookie visitor/khách (VISITOR_INFO1_LIVE, PREF...).
    mode='full':  Toàn bộ cookies đầy đủ kể cả session tài khoản.
    """
    all_paths = get_all_cookie_file_paths(mode=mode)
    return all_paths[0] if all_paths else None


def get_cookie_status() -> tuple[bool, int, str]:
    """
    Kiểm tra trạng thái cấu hình cookies YouTube.
    Trả về: (has_cookies: bool, cookie_count: int, message: str)
    """
    guest_path = get_cookie_file_path(mode="guest")
    full_path = get_cookie_file_path(mode="full")
    if not guest_path and not full_path:
        return False, 0, "Chưa tìm thấy YouTube cookies trong Secrets/Môi trường."

    active_path = guest_path or full_path
    if active_path is None:
        return False, 0, "Chưa tìm thấy YouTube cookies trong Secrets/Môi trường."

    try:
        content = active_path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in content.splitlines() if line.strip() and not line.startswith("#")]
        mode_label = "Chế độ Cloud-Safe" if active_path == guest_path else "Chế độ Toàn phần"
        return True, len(lines), f"Đã nạp {len(lines)} cookies YouTube thành công ({mode_label})"
    except Exception as e:
        return False, 0, f"Lỗi đọc cookie: {e}"


def _normalize_proxy_url(proxy_url: str) -> str:
    """
    Chuẩn hóa URL proxy.
    LƯU Ý: Tuyệt đối không dùng '-rotate' đối với YouTube vì YouTube CDN (googlevideo)
    ràng buộc chặt chẽ IP ban đầu với link stream. Xoay IP liên tục giữa các chunk sẽ gây
    lỗi HTTP 403 Forbidden và kích hoạt hàng loạt retry/request lỗi lãng phí.
    """
    raw = proxy_url.strip()
    if "@p.webshare.io" in raw and "-rotate" in raw:
        # Tự động gỡ bỏ đuôi -rotate để giữ kết nối sticky/nhất quán IP
        raw = raw.replace("-rotate", "")
    return raw


def get_youtube_proxy() -> str | None:
    """
    Tìm kiếm cấu hình Proxy cho YouTube từ biến môi trường hoặc Streamlit Secrets.
    Hỗ trợ các định dạng: http://user:pass@host:port, http://host:port, socks5://...
    Thứ tự ưu tiên: YOUTUBE_PROXY -> PROXY -> HTTP_PROXY -> HTTPS_PROXY.
    """
    candidate_keys = ["YOUTUBE_PROXY", "PROXY", "HTTP_PROXY", "HTTPS_PROXY"]
    for key in candidate_keys:
        val = os.getenv(key, "").strip()
        if val:
            return _normalize_proxy_url(val)

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in candidate_keys:
                if key in st.secrets and str(st.secrets[key]).strip():
                    return _normalize_proxy_url(str(st.secrets[key]).strip())

            for k, v in st.secrets.items():
                if (
                    isinstance(v, str)
                    and "proxy" in str(k).lower()
                    and str(v).startswith(("http://", "https://", "socks"))
                ):
                    return _normalize_proxy_url(str(v).strip())
    except Exception:
        pass

    return None
