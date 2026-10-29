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
    os.getenv("DEFAULT_GEMINI_MODEL",
              "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite"
).strip()

# Thứ tự mô hình dự phòng tự động khi cạn Quota hoặc lỗi (Multi-tier Model Failover)
DEFAULT_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

_env_fallback = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
if _env_fallback:
    GEMINI_FALLBACK_MODELS = [m.strip()
                              for m in _env_fallback.split(",") if m.strip()]
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

# Audio Processing Configuration
AUDIO_SAMPLE_DURATION = int(os.getenv("AUDIO_SAMPLE_DURATION", "30"))
AUDIO_SAMPLE_OFFSET = int(os.getenv("AUDIO_SAMPLE_OFFSET", "45"))
BATCH_SLEEP_INTERVAL = float(os.getenv("BATCH_SLEEP_INTERVAL", "1.0"))

# Performance Optimization Configuration
PIPED_TIMEOUT_SECONDS = float(os.getenv("PIPED_TIMEOUT_SECONDS", "4.0"))
SPECTROGRAM_DPI = int(os.getenv("SPECTROGRAM_DPI", "80"))
ENABLE_CASCADED_EVALUATION = os.getenv("ENABLE_CASCADED_EVALUATION", "true").strip().lower() in ("true", "1", "yes")
ENABLE_RANGE_STREAM_DOWNLOAD = os.getenv("ENABLE_RANGE_STREAM_DOWNLOAD", "false").strip().lower() in ("true", "1", "yes")

# Media Cloud Storage Configuration
CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD = os.getenv(
    "CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD", "true"
).strip().lower() in ("true", "1", "yes")
UPLOAD_AUDIO_SAMPLES_TO_CLOUD = os.getenv(
    "UPLOAD_AUDIO_SAMPLES_TO_CLOUD", "false"
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


def get_js_runtimes_config() -> dict[str, Any]:
    """
    Tự động phát hiện và cấu hình JavaScript Runtime cho yt-dlp (Node.js, Deno, Bun)
    hỗ trợ đầy đủ cả Windows và môi trường Linux / Debian của Streamlit Cloud.
    Gói 'nodejs' trên Debian cài binary tại /usr/bin/nodejs. Hàm này sẽ trỏ đúng binary
    và tự động tạo symlink 'node' vào thư mục PATH nếu cần.
    """
    node_exe = (
        shutil.which("node")
        or shutil.which("nodejs")
        or (Path("/usr/bin/nodejs") if Path("/usr/bin/nodejs").exists() else None)
        or (Path("/usr/bin/node") if Path("/usr/bin/node").exists() else None)
        or (
            Path(r"C:\Program Files\nodejs\node.exe")
            if Path(r"C:\Program Files\nodejs\node.exe").exists()
            else None
        )
    )

    if node_exe:
        node_str = str(node_exe)
        # Trên Linux nếu chỉ có nodejs mà không có lệnh node, tự động tạo symlink trợ giúp
        if os.name != "nt" and not shutil.which("node"):
            try:
                tmp_bin = TEMP_DIR / "bin"
                tmp_bin.mkdir(parents=True, exist_ok=True)
                node_sym = tmp_bin / "node"
                if not node_sym.exists():
                    node_sym.symlink_to(node_str)
                if str(tmp_bin) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{tmp_bin}:{os.environ.get('PATH', '')}"
            except Exception:
                pass
        return {"node": {"path": node_str}}

    deno_exe = (
        shutil.which("deno")
        or (Path("/usr/bin/deno") if Path("/usr/bin/deno").exists() else None)
        or (
            Path.home() / ".deno" / "bin" / "deno"
            if (Path.home() / ".deno" / "bin" / "deno").exists()
            else None
        )
    )
    if deno_exe:
        return {"deno": {"path": str(deno_exe)}}

    bun_exe = shutil.which("bun")
    if bun_exe:
        return {"bun": {"path": str(bun_exe)}}

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
            content = content[len(q): -len(q)].strip()

    # Hỗ trợ tiền tố base64: hoặc tự động nhận diện Base64
    if content.startswith("base64:"):
        content = content[7:].strip()

    if "\t" not in content and "\n" not in content[:200] and len(content) > 50:
        try:
            decoded = base64.b64decode(content, validate=True).decode(
                "utf-8", errors="replace")
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
                name not in account_session_names and not name.startswith(
                    ("__Secure-", "SID"))
            ):
                guest_cookies_dict[cookie_key] = cookie_line

    header = "# Netscape HTTP Cookie File\n# This file is generated for yt-dlp.\n\n"
    full_cookies = list(full_cookies_dict.values())
    guest_cookies = list(guest_cookies_dict.values())
    full_content = (header + "\n".join(full_cookies) +
                    "\n") if full_cookies else ""
    guest_content = (header + "\n".join(guest_cookies) +
                     "\n") if guest_cookies else ""

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
        full_c, guest_c, full_cnt, guest_cnt = normalize_netscape_cookies(
            raw_content)
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
                full_c, guest_c, full_cnt, guest_cnt = normalize_netscape_cookies(
                    content)
                target_c = guest_c if mode == "guest" else full_c
                target_cnt = guest_cnt if mode == "guest" else full_cnt
                if target_c and target_cnt > 0:
                    c_hash = hashlib.md5(target_c.encode("utf-8")).hexdigest()
                    if c_hash not in seen_hashes:
                        seen_hashes.add(c_hash)
                        stem = cand.stem
                        out_path = TEMP_DIR / \
                            f"cookie_{stem}_{mode}_runtime.txt"
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
        lines = [line for line in content.splitlines() if line.strip()
                 and not line.startswith("#")]
        mode_label = "Chế độ Cloud-Safe" if active_path == guest_path else "Chế độ Toàn phần"
        return True, len(lines), f"Đã nạp {len(lines)} cookies YouTube thành công ({mode_label})"
    except Exception as e:
        return False, 0, f"Lỗi đọc cookie: {e}"


def _normalize_proxy_url(proxy_url: str) -> str:
    """Tự động chuẩn hóa URL proxy (ví dụ Webshare backbone cần thêm -rotate nếu thiếu)."""
    raw = proxy_url.strip()
    if "@p.webshare.io" in raw:
        m = re.match(r"^(https?://)([^:@]+)(:[^@]+@p\.webshare\.io.*)$", raw)
        if m:
            proto, user, rest = m.groups()
            if "-" not in user:
                return f"{proto}{user}-rotate{rest}"
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
