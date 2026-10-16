import base64
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


def normalize_netscape_cookies(raw_text: str) -> tuple[str, int]:
    """
    Chuẩn hóa nội dung cookies sang định dạng chuẩn Netscape HTTP Cookie File cho yt-dlp:
    1. Tự động giải mã Base64 (nếu chuỗi được mã hóa Base64 để chống lỗi trình soạn thảo web).
    2. Bóc tách dấu ngoặc kép TOML/Python thừa (''' hoặc \"\"\").
    3. Chuyển đổi escape characters ('\\n' -> '\n', '\\t' -> '\t').
    4. TỰ ĐỘNG PHỤC HỒI TAB: Trình duyệt web (khi copy-paste) thường biến dấu Tab thành 2-4 khoảng trắng (Spaces).
       yt-dlp bắt buộc phải có Tab chuẩn để phân tách 7 trường; nếu không có Tab yt-dlp sẽ âm thầm bỏ qua cookie.
       Hàm này tự động phân tách và ghép lại bằng Tab thật 100%.
    5. Đảm bảo có header '# Netscape HTTP Cookie File'.
    Trả về: (nội dung chuẩn hóa, số lượng cookie hợp lệ)
    """
    if not raw_text or not raw_text.strip():
        return "", 0

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
    valid_cookies: list[str] = []

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue

        # Nếu dòng đã có ít nhất 6 dấu tab chuẩn Netscape
        if line_str.count("\t") >= 6:
            parts = line_str.split("\t")
            if len(parts) >= 7 and parts[0]:
                cookie_name = parts[5].strip()
                # QUAN TRỌNG: Loại bỏ __Secure-1PSIDTS và __Secure-3PSIDTS
                # sidts là token phiên TLS-fingerprint của máy cá nhân. Đem sang Cloud Datacenter (AWS)
                # sẽ kích hoạt cơ chế chống bot của YouTube: "The page needs to be reloaded".
                if cookie_name in ("__Secure-1PSIDTS", "__Secure-3PSIDTS"):
                    continue
                valid_cookies.append("\t".join(parts[:7]))
            elif len(parts) == 6 and parts[0]:
                cookie_name = parts[5].strip()
                if cookie_name in ("__Secure-1PSIDTS", "__Secure-3PSIDTS"):
                    continue
                valid_cookies.append("\t".join(parts) + "\t")
            continue

        # Nếu bị biến thành khoảng trắng do copy-paste trên trình duyệt web:
        # Netscape format gồm 7 trường: domain, include_subdomains, path, secure, expires, name, value
        # 6 trường đầu tiên tuyệt đối không bao giờ chứa khoảng trắng!
        parts = re.split(r"\s+", line_str)
        if len(parts) >= 7 and parts[0].startswith(
            (".", "http", "www", "youtube", "google", "zing")
        ):
            domain, sub, path, sec, exp, name = parts[:6]
            if name in ("__Secure-1PSIDTS", "__Secure-3PSIDTS"):
                continue
            val = " ".join(parts[6:])
            valid_cookies.append(f"{domain}\t{sub}\t{path}\t{sec}\t{exp}\t{name}\t{val}")
        elif len(parts) == 6 and parts[0].startswith(
            (".", "http", "www", "youtube", "google", "zing")
        ):
            domain, sub, path, sec, exp, name = parts
            if name in ("__Secure-1PSIDTS", "__Secure-3PSIDTS"):
                continue
            valid_cookies.append(f"{domain}\t{sub}\t{path}\t{sec}\t{exp}\t{name}\t")

    if not valid_cookies:
        return "", 0

    header = "# Netscape HTTP Cookie File\n# This file is generated for yt-dlp.\n\n"
    normalized_content = header + "\n".join(valid_cookies) + "\n"
    return normalized_content, len(valid_cookies)


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


def get_cookie_file_path() -> Path | None:
    """
    Trả về đường dẫn file cookies.txt hợp lệ nếu có.
    Tự động phục hồi định dạng chuẩn Netscape từ Cloud Secrets hoặc local files.
    """
    raw_content = find_cookie_content_from_secrets_or_env()
    if raw_content:
        normalized, count = normalize_netscape_cookies(raw_content)
        if normalized and count > 0:
            runtime_cookie = TEMP_DIR / "cookie_runtime.txt"
            try:
                runtime_cookie.write_text(normalized, encoding="utf-8")
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
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
                normalized, count = normalize_netscape_cookies(content)
                if normalized and count > 0:
                    runtime_cookie = TEMP_DIR / "cookie_runtime.txt"
                    if candidate != runtime_cookie:
                        runtime_cookie.write_text(normalized, encoding="utf-8")
                    return runtime_cookie
            except Exception:
                return candidate

    return None


def get_cookie_status() -> tuple[bool, int, str]:
    """
    Kiểm tra trạng thái cấu hình cookies YouTube.
    Trả về: (has_cookies: bool, cookie_count: int, message: str)
    """
    cookie_path = get_cookie_file_path()
    if not cookie_path or not cookie_path.exists():
        return False, 0, "Chưa tìm thấy YouTube cookies trong Secrets/Môi trường."

    try:
        content = cookie_path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in content.splitlines() if line.strip() and not line.startswith("#")]
        return True, len(lines), f"Đã nạp {len(lines)} cookies YouTube thành công"
    except Exception as e:
        return False, 0, f"Lỗi đọc cookie: {e}"


def get_local_cookies_as_base64() -> str | None:
    """Đọc file cookies.txt cục bộ và chuyển đổi thành chuỗi Base64 để dán vào Secrets an toàn 100%."""
    for candidate in [BASE_DIR / "cookies.txt", BASE_DIR / "cookies_master.txt"]:
        if candidate.exists() and candidate.stat().st_size > 0:
            try:
                raw_bytes = candidate.read_bytes()
                return base64.b64encode(raw_bytes).decode("ascii")
            except Exception:
                pass
    return None

