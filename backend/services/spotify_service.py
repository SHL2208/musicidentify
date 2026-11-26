import base64
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any

from ..core.config import (
    SPOTIFY_BATCH_WORKERS,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_MAX_CALLS_PER_SECOND,
    SPOTIFY_SEARCH_LIMIT,
)
from ..core.interfaces.services.spotify_service_interface import ISpotifyService
from ..core.utils import normalize_vietnamese_text
from .spotify_cache_service import SpotifyCacheService

logger = logging.getLogger(__name__)

SPOTIFY_TRACK_ID_REGEX = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/track/|open\.spotify\.com/intl-[a-z]+/track/)([a-zA-Z0-9]+)",
    re.IGNORECASE,
)

# Regex lọc bỏ các từ khóa nhiễu trong tiêu đề bài hát khi so khớp (giữ lại hàm _strip_noise cho tương thích ngược)
NOISE_TITLE_REGEX = re.compile(
    r"(?i)\s*(?:[\(\[\{][^\)\]\}]*(?:official|music\s*video|mv|audio|lyric|video|remix|cover|live|beat|karaoke|prod|ft\.?|feat\.?)[^\)\]\}]*[\)\]\}]|\b(?:ft\.?|feat\.?)\s+.*$)",
    re.IGNORECASE,
)

# Regex trích xuất các cụm ngoặc (...), [...], {...} và phần phụ sau dấu ' - '
BRACKET_CHUNK_REGEX = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
FEATURED_SUFFIX_REGEX = re.compile(r"(?i)\s+\b(?:ft\.?|feat\.?)\s+.*$")
PACKAGING_ONLY_WORDS = {
    "official",
    "music",
    "video",
    "mv",
    "audio",
    "lyric",
    "lyrics",
    "visualizer",
    "teaser",
    "trailer",
    "hd",
    "4k",
    "full",
    "remaster",
    "remastered",
}
BEAT_KEYWORDS = {"beat", "instrumental", "inst", "karaoke", "playback"}
VARIANT_KEYWORDS = {
    "remix",
    "mix",
    "drill",
    "lofi",
    "acoustic",
    "live",
    "session",
    "cover",
    "ballad",
    "chacha",
    "speed",
    "slowed",
    "piano",
    "guitar",
    "remake",
    "vip",
    "edit",
    "mashup",
    "version",
    "ver",
    "pt",
    "part",
    "p2",
}
MODIFIER_STOPWORDS = (
    PACKAGING_ONLY_WORDS
    | BEAT_KEYWORDS
    | VARIANT_KEYWORDS
    | {"ft", "feat", "prod", "with", "by", "x", "va", "and", "the", "of", "nu", "nam", "up"}
)
VERSION_DECIMAL_ZERO_REGEX = re.compile(r"(\d+)\.0+\b")
ARTIST_SPLIT_REGEX = re.compile(r"(?i)\s*(?:,|;|&|\bx\b|\bft\.?\b|\bfeat\.?\b)\s*")


def _clean_str(text: str) -> str:
    """Loại bỏ dấu tiếng Việt và ký tự đặc biệt để so khớp chuỗi chuẩn xác."""
    return normalize_vietnamese_text(text, remove_special_chars=True).lower().strip()


def _normalize_version_str(text: str) -> str:
    """Chuẩn hóa số phiên bản dạng '2.0' thành '2' để khớp khi người dùng bỏ '.0'."""
    if not text:
        return ""
    return VERSION_DECIMAL_ZERO_REGEX.sub(r"\1", text)


def _strip_noise(text: str) -> str:
    """Loại bỏ các từ khóa nhiễu phổ biến khỏi tên bài hát."""
    if not text:
        return ""
    cleaned = NOISE_TITLE_REGEX.sub(" ", text)
    return " ".join(cleaned.split()).strip()


def _normalize_qualifier_tokens(text: str) -> list[str]:
    """Chuẩn hóa cụm mô tả phiên bản (ví dụ: 'Cha Cha Cha' -> 'chacha')."""
    norm = _clean_str(text)
    if not norm:
        return []
    norm = re.sub(r"\bcha\s+cha(?:\s+cha)?\b", "chacha", norm)
    norm = re.sub(r"\bchachacha\b", "chacha", norm)
    norm = re.sub(r"\blo\s+fi\b", "lofi", norm)
    norm = re.sub(r"\b(?:part|pt)\s+(\d+)\b", r"p\1", norm)
    norm = re.sub(r"\b(?:version|ver)\s+(\d+)\b", r"version \1", norm)
    return norm.split()


def _extract_version_profile(title: str) -> dict[str, Any]:
    """Tách tên bài hát gốc (base title) và hồ sơ phiên bản (Beat, Remix, Live Session, Remixer).

    Giúp phân biệt rạch ròi giữa Bản gốc, Beat/Instrumental, Live Session và các bản Remix khác nhau,
    tránh xóa nhầm thông tin phiên bản khi đối soát.
    """
    raw = (title or "").strip()
    if not raw:
        return {
            "base_title": "",
            "base_norm": "",
            "base_vnorm": "",
            "is_beat": False,
            "variant_tags": set(),
            "specific_modifiers": set(),
            "all_qualifier_tokens": set(),
        }

    qualifier_chunks: list[str] = []

    def _collect_bracket(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            return " "
        inner_tokens = _normalize_qualifier_tokens(inner)
        # Bỏ qua credit ft./feat. (không làm thay đổi phiên bản bài hát)
        if inner_tokens and inner_tokens[0] in ("ft", "feat"):
            return " "
        # Bỏ qua nếu chỉ toàn là từ khóa đóng gói MV/Official/Audio/Remaster
        if inner_tokens and all(t in PACKAGING_ONLY_WORDS for t in inner_tokens):
            return " "
        # Chỉ tách thành qualifier phiên bản nếu chứa từ khóa phiên bản/beat/remix/live/part
        if any(t in BEAT_KEYWORDS or t in VARIANT_KEYWORDS for t in inner_tokens):
            qualifier_chunks.append(inner)
            return " "
        # Giữ nguyên phụ đề trong ngoặc vào tên bài gốc (vd: 'Kẻ Say Tình 3 (Ngày Gió Giông)')
        return f" {inner} "

    base_work = BRACKET_CHUNK_REGEX.sub(_collect_bracket, raw)
    base_work = FEATURED_SUFFIX_REGEX.sub(" ", base_work).strip()

    # Tách phần mô tả phiên bản sau dấu ' - ' (vd: 'Sai Ngay Từ Đầu - Beat', 'Mừng Ngày Em Cưới - Am Remix - Beat')
    if " - " in base_work:
        parts = [p.strip() for p in base_work.split(" - ") if p.strip()]
        if len(parts) >= 2:
            kept_parts = [parts[0]]
            for seg in parts[1:]:
                seg_tokens = _normalize_qualifier_tokens(seg)
                if seg_tokens and all(t in PACKAGING_ONLY_WORDS for t in seg_tokens):
                    continue
                if any(t in BEAT_KEYWORDS or t in VARIANT_KEYWORDS for t in seg_tokens):
                    qualifier_chunks.append(seg)
                else:
                    kept_parts.append(seg)
            base_work = " - ".join(kept_parts)

    # Xử lý hậu tố phiên bản đứng cuối câu không có ngoặc/gạch (vd: 'Tên Bài Remix', 'Tên Bài Beat')
    base_tokens_raw = _normalize_qualifier_tokens(base_work)
    while len(base_tokens_raw) >= 2 and base_tokens_raw[-1] in (
        BEAT_KEYWORDS | {"remix", "mix", "karaoke", "instrumental", "lofi", "acoustic"}
    ):
        qualifier_chunks.append(base_tokens_raw.pop())
        base_work = " ".join(base_tokens_raw)

    all_q_tokens: list[str] = []
    for chunk in qualifier_chunks:
        all_q_tokens.extend(_normalize_qualifier_tokens(chunk))

    q_token_set = set(all_q_tokens)
    is_beat = bool(q_token_set & BEAT_KEYWORDS)
    variant_tags = q_token_set & VARIANT_KEYWORDS
    specific_modifiers = {
        t
        for t in q_token_set
        if t not in MODIFIER_STOPWORDS and (len(t) >= 2 or t.isdigit())
    }

    base_title = " ".join(base_work.split()).strip()
    base_norm = _clean_str(base_title)
    base_vnorm = _clean_str(_normalize_version_str(base_title))

    return {
        "base_title": base_title,
        "base_norm": base_norm,
        "base_vnorm": base_vnorm,
        "is_beat": is_beat,
        "variant_tags": variant_tags,
        "specific_modifiers": specific_modifiers,
        "all_qualifier_tokens": q_token_set,
    }


def _are_version_profiles_compatible(
    q_prof: dict[str, Any],
    c_prof: dict[str, Any],
    cand_artist_tokens: set[str],
    cand_artists_nospace: set[str],
) -> tuple[bool, float]:
    """Kiểm tra mức độ tương thích phiên bản (Bản gốc / Beat / Remix / Live Session / Remixer).

    Trả về: (có tương thích hay không, điểm thưởng/phạt phụ trội).
    """
    # 1. Ràng buộc cứng Beat/Instrumental: Beat chỉ khớp Beat, Vocal chỉ khớp Vocal
    if q_prof["is_beat"] != c_prof["is_beat"]:
        return False, 0.0

    q_vars: set[str] = q_prof["variant_tags"]
    c_vars: set[str] = c_prof["variant_tags"]
    q_mods: set[str] = q_prof["specific_modifiers"]
    c_mods: set[str] = c_prof["specific_modifiers"]

    # 2. Người dùng tìm bản gốc (không yêu cầu remix/live/cover/version...) -> từ chối bản phái sinh
    if not q_vars and not q_mods and not q_prof["is_beat"]:
        if c_vars or c_mods:
            return False, 0.0
        return True, 0.02

    # 3. Người dùng yêu cầu phiên bản cụ thể
    # 3a. Live / Session phải khớp Live / Session
    if q_vars & {"live", "session"}:
        if not (c_vars & {"live", "session"}) and not (q_mods & c_mods):
            return False, 0.0

    # 3b. Remix / Mix / Drill / Chacha / Cover / Lofi / Acoustic / Ballad / Version / P2
    if "chacha" in q_vars and "chacha" not in c_vars:
        return False, 0.0
    if (q_vars & {"remix", "mix"}) and not (
        c_vars & {"remix", "mix", "drill", "chacha", "lofi", "edit", "vip", "remake"}
    ) and not (q_mods & (c_mods | cand_artist_tokens | cand_artists_nospace)):
        return False, 0.0
    for strict_tag in ("cover", "acoustic", "lofi", "drill", "piano", "guitar", "p2"):
        if strict_tag in q_vars and strict_tag not in c_vars:
            return False, 0.0
    if (q_vars & {"version", "ver"}) and not (c_vars & {"version", "ver"}):
        return False, 0.0

    # Ngược lại: nếu query KHÔNG phải remix/live/cover nhưng candidate lại là bản phái sinh khác hẳn
    if not (q_vars & {"live", "session"}) and (c_vars & {"live", "session"}):
        return False, 0.0
    if not (q_vars & {"cover"}) and ("cover" in c_vars):
        return False, 0.0

    # 3c. Kiểm tra tên remixer / tên phiên bản cụ thể (vd: VANTRI vs MVS, NVT x Quann vs Tón, AM vs Original)
    if q_mods:
        cand_pool = c_prof["all_qualifier_tokens"] | cand_artist_tokens | cand_artists_nospace
        cand_joined = "".join(sorted(cand_pool))
        for mod in q_mods:
            if mod in cand_pool:
                continue
            # Hỗ trợ viết liền/cách (vd: 'indam' trong 'indamstage')
            if len(mod) >= 4 and mod in cand_joined:
                continue
            return False, 0.0

    # Thưởng nhẹ nếu khớp chính xác cả tập modifier (vd: (KTH Remix) khớp Beat KTH Remix thay vì KTH x ABC)
    bonus = 0.03 if (q_mods == c_mods and q_vars == c_vars) else 0.0
    return True, bonus


def _split_artists(artist_text: str) -> list[str]:
    """Tách chuỗi nhiều nghệ sĩ thành danh sách chuẩn hóa."""
    if not artist_text:
        return []
    parts = ARTIST_SPLIT_REGEX.split(artist_text)
    cleaned = [_clean_str(p) for p in parts if p and p.strip()]
    return [c for c in cleaned if c]


def _compute_artist_score(
    target_artist: str, cand_artist_str: str, cand_artists_raw: list[str]
) -> float:
    """Tính điểm khớp nghệ sĩ hỗ trợ danh sách nhiều nghệ sĩ và tên viết liền/cách."""
    target_norm = _clean_str(target_artist)
    cand_norm = _clean_str(cand_artist_str)
    if not target_norm:
        return 0.0

    base_seq_score = SequenceMatcher(None, target_norm, cand_norm).ratio()
    target_list = _split_artists(target_artist)
    cand_list = [_clean_str(a) for a in cand_artists_raw if _clean_str(a)]
    if not cand_list:
        cand_list = _split_artists(cand_artist_str)
    if not target_list or not cand_list:
        return base_seq_score

    def _pair_sim(t_a: str, c_a: str) -> float:
        if t_a == c_a:
            return 1.0
        t_ns = t_a.replace(" ", "")
        c_ns = c_a.replace(" ", "")
        if len(t_ns) >= 3 and len(c_ns) >= 3 and t_ns == c_ns:
            return 0.98
        if t_a in c_a or c_a in t_a:
            len_min = min(len(t_a), len(c_a))
            len_max = max(len(t_a), len(c_a))
            if len_max > 0 and (len_min / len_max) >= 0.60:
                return 0.88
        if len(t_ns) >= 4 and len(c_ns) >= 4 and (t_ns in c_ns or c_ns in t_ns):
            len_min = min(len(t_ns), len(c_ns))
            len_max = max(len(t_ns), len(c_ns))
            if len_max > 0 and (len_min / len_max) >= 0.70:
                return 0.85
        return SequenceMatcher(None, t_a, c_a).ratio()

    # Kiểm tra nghệ sĩ chính (vị trí đầu tiên) và tỷ lệ bao phủ
    primary_best = max(_pair_sim(target_list[0], ca) for ca in cand_list)
    matched_targets = sum(1 for ta in target_list if max(_pair_sim(ta, ca) for ca in cand_list) >= 0.80)
    matched_cands = sum(1 for ca in cand_list if max(_pair_sim(ta, ca) for ta in target_list) >= 0.80)

    target_cov = matched_targets / len(target_list)
    cand_cov = matched_cands / len(cand_list)

    # Nếu nghệ sĩ chính khớp hoặc đa số nghệ sĩ ứng viên khớp trong danh sách đầu vào
    if primary_best >= 0.85 and (cand_cov >= 0.50 or target_cov >= 0.50):
        overlap_score = 0.75 + 0.25 * max(cand_cov, target_cov)
        return max(base_seq_score, overlap_score)

    if matched_cands > 0 and cand_cov >= 0.50:
        return max(base_seq_score, 0.70 + 0.25 * cand_cov)

    return base_seq_score


def _contains_token_sequence(haystack_tokens: list[str], needle_tokens: list[str]) -> bool:
    """Kiểm tra danh sách token `needle_tokens` có xuất hiện liên tiếp trong `haystack_tokens` không."""
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    n_len = len(needle_tokens)
    for i in range(len(haystack_tokens) - n_len + 1):
        if haystack_tokens[i : i + n_len] == needle_tokens:
            return True
    return False


def _remove_matched_artist_tokens(
    query_norm: str, cand_artists_list: list[str]
) -> tuple[str, bool]:
    """Tách các token nghệ sĩ xuất hiện trong chuỗi truy vấn tự do (khi không có cột artist riêng).

    Hỗ trợ khớp cả tên nghệ sĩ liền/cách (ví dụ: 'richchoi' <-> 'Rich Choi').
    Trả về: (chuỗi truy vấn còn lại sau khi tách nghệ sĩ, cờ có khớp nghệ sĩ hay không).
    """
    q_tokens = query_norm.split()
    if len(q_tokens) < 2 or not cand_artists_list:
        return query_norm, False

    matched_indices: set[int] = set()
    matched_any = False

    # Ưu tiên khớp tên nghệ sĩ dài trước để tránh khớp nhầm từ ngắn
    sorted_artists = sorted(
        (a for a in cand_artists_list if a),
        key=lambda x: len(x.replace(" ", "")),
        reverse=True,
    )

    for art in sorted_artists:
        art_tokens = art.split()
        art_nospace = art.replace(" ", "")
        if len(art_nospace) < 3:
            continue

        # 1. Khớp cụm token liên tiếp của tên nghệ sĩ
        n_art = len(art_tokens)
        found_phrase = False
        if n_art <= len(q_tokens):
            for i in range(len(q_tokens) - n_art + 1):
                if any(idx in matched_indices for idx in range(i, i + n_art)):
                    continue
                if q_tokens[i : i + n_art] == art_tokens:
                    # Không xóa hết toàn bộ query nếu làm rỗng hoàn toàn
                    if len(matched_indices) + n_art < len(q_tokens):
                        matched_indices.update(range(i, i + n_art))
                        matched_any = True
                        found_phrase = True
                        break

        if found_phrase or len(art_nospace) < 4:
            continue

        # 2. Khớp tên nghệ sĩ viết liền không dấu cách (vd: 'richchoi' <-> 'Rich Choi')
        for i, tok in enumerate(q_tokens):
            if i in matched_indices or len(tok) < 4:
                continue
            if tok == art_nospace and len(matched_indices) + 1 < len(q_tokens):
                matched_indices.add(i)
                matched_any = True
                break

    if not matched_any:
        return query_norm, False

    remaining = [tok for i, tok in enumerate(q_tokens) if i not in matched_indices]
    return " ".join(remaining).strip(), True


def _compute_title_score(
    target_norm: str,
    target_vnorm: str,
    cand_norm: str,
    cand_vnorm: str,
    min_sub_ratio: float = 0.60,
) -> float:
    """Tính điểm tương đồng tiêu đề bài hát trên cả bản chuẩn hóa gốc và bản chuẩn hóa số phiên bản."""
    if not target_norm or not cand_norm:
        return 0.0

    score = max(
        SequenceMatcher(None, target_norm, cand_norm).ratio(),
        SequenceMatcher(None, target_vnorm, cand_vnorm).ratio(),
    )

    for t_str, c_str in ((target_norm, cand_norm), (target_vnorm, cand_vnorm)):
        if not t_str or not c_str:
            continue
        if t_str in c_str or c_str in t_str:
            len_min = min(len(t_str), len(c_str))
            len_max = max(len(t_str), len(c_str))
            if len_max > 0 and (len_min / len_max) >= min_sub_ratio:
                # Xác nhận ranh giới token để tránh khớp một phần từ (vd 'an' trong 'anh')
                t_tokens = t_str.split()
                c_tokens = c_str.split()
                if _contains_token_sequence(t_tokens, c_tokens) or _contains_token_sequence(
                    c_tokens, t_tokens
                ):
                    score = max(score, 0.85)
    return score


class _CallPacer:
    """Bộ điều tiết nhịp độ (Rate Limiting Pacer) tránh gửi request dồn dập vào Spotify API."""

    def __init__(self, max_rate: float = 5.0):
        self.min_interval = 1.0 / max(0.1, max_rate)
        self.last_call = 0.0
        self.lock = threading.Lock()

    def pace(self) -> None:
        """Tạm dừng nếu khoảng cách giữa các lần gọi quá ngắn."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


class SpotifyService(ISpotifyService):
    """
    Hiện thực ISpotifyService kết nối Spotify Web API (Client Credentials Flow)
    kèm bộ đệm SQLite Cache cục bộ (Positive + Negative Cache), điều tiết nhịp độ (Pacer),
    thuật toán so khớp độ tin cậy cao (High Precision Matching) và cơ chế Hybrid dự phòng.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout_seconds: float = 4.0,
        cache_service: SpotifyCacheService | None = None,
    ):
        self.client_id = (client_id if client_id is not None else SPOTIFY_CLIENT_ID).strip()
        self.client_secret = (
            client_secret if client_secret is not None else SPOTIFY_CLIENT_SECRET
        ).strip()
        self.timeout_seconds = timeout_seconds

        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = threading.Lock()
        self._premium_required: bool = False
        self._premium_warning_logged: bool = False
        self._quota_exceeded: bool = False
        self._quota_warning_logged: bool = False
        self._quota_reset_at: float = 0.0
        self.api_call_count: int = 0

        self.cache_service = cache_service or SpotifyCacheService()
        self._pacer = _CallPacer(max_rate=SPOTIFY_MAX_CALLS_PER_SECOND)

    def _get_access_token(self) -> str | None:
        """
        Lấy access token bằng Client Credentials Flow.
        Nếu có sẵn token còn hạn (> 60s), tái sử dụng ngay từ bộ nhớ đệm (In-memory Cache).
        """
        now = time.time()
        if self._access_token and now < (self._token_expires_at - 60):
            return self._access_token

        with self._token_lock:
            # Kiểm tra lại sau khi acquire lock (Double-checked locking)
            if self._access_token and now < (self._token_expires_at - 60):
                return self._access_token

            # 1. Ưu tiên Client Credentials Flow chính thức nếu có client_id & secret
            if self.client_id and self.client_secret:
                try:
                    auth_header = base64.b64encode(
                        f"{self.client_id}:{self.client_secret}".encode("utf-8")
                    ).decode("utf-8")
                    token_url = "https://accounts.spotify.com/api/token"
                    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(
                        "utf-8"
                    )

                    req = urllib.request.Request(
                        token_url,
                        data=data,
                        headers={
                            "Authorization": f"Basic {auth_header}",
                            "Content-Type": "application/x-www-form-urlencoded",
                            "User-Agent": "MusicIdentify-SpotifyService/1.0",
                        },
                        method="POST",
                    )

                    self.api_call_count += 1
                    with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        self._access_token = res_json.get("access_token")
                        expires_in = int(res_json.get("expires_in", 3600))
                        self._token_expires_at = time.time() + expires_in
                        logger.info("🟢 [SpotifyService] Đã cấp mới Access Token thành công.")
                        return self._access_token
                except Exception as e:
                    logger.warning(
                        "⚠️ [SpotifyService] Không thể xác thực Client Credentials: %s. Thử fallback...",
                        e,
                    )

            # 2. Fallback: Lấy token ẩn danh từ Spotify Web Client
            try:
                web_token_url = "https://open.spotify.com/get_access_token"
                req = urllib.request.Request(
                    web_token_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                )
                self.api_call_count += 1
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    anon_token = res_json.get("accessToken")
                    if anon_token:
                        self._access_token = anon_token
                        expires_in = int(res_json.get("accessTokenExpirationTimestampMs", 0)) / 1000
                        if expires_in > now:
                            self._token_expires_at = expires_in
                        else:
                            self._token_expires_at = now + 1800
                        logger.info("🟢 [SpotifyService] Đã lấy Public Web Token thành công.")
                        return self._access_token
            except Exception as e:
                logger.debug("ℹ️ [SpotifyService] Fallback anonymous token thất bại: %s", e)

        return None

    def _parse_track_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Chuẩn hóa một đối tượng track từ Spotify API về dict chuẩn của hệ thống."""
        track_id = item.get("id") or ""
        name = item.get("name") or ""
        artists_list = [
            a.get("name", "").strip() for a in item.get("artists", []) if a.get("name")
        ]
        artist_str = ", ".join(artists_list)

        album = item.get("album") or {}
        album_name = album.get("name") or ""
        release_date = album.get("release_date") or ""

        images = album.get("images") or []
        cover_url = images[0].get("url") if images else None

        duration_ms = item.get("duration_ms") or 0
        duration_sec = round(duration_ms / 1000.0, 2) if duration_ms else 0.0

        external_urls = item.get("external_urls") or {}
        spotify_url = external_urls.get("spotify") or (
            f"https://open.spotify.com/track/{track_id}" if track_id else ""
        )
        embed_url = f"https://open.spotify.com/embed/track/{track_id}" if track_id else ""
        preview_url = item.get("preview_url")
        isrc = (item.get("external_ids") or {}).get("isrc")
        popularity = item.get("popularity", 0)

        return {
            "id": track_id,
            "title": name,
            "artist": artist_str,
            "artists_list": artists_list,
            "album": album_name,
            "release_date": release_date,
            "spotify_url": spotify_url,
            "embed_url": embed_url,
            "preview_url": preview_url,
            "cover_url": cover_url,
            "duration_sec": duration_sec,
            "isrc": isrc,
            "popularity": popularity,
            "is_official_api": True,
        }

    def search_tracks(
        self, query: str, artist: str = "", limit: int = SPOTIFY_SEARCH_LIMIT
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm danh sách bài hát trên Spotify qua Web API.
        Tích hợp Rate Limiting Pacer và tự động phục hồi khi gặp HTTP 429 Retry-After.
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        # Nếu query vốn là link Spotify, trực tiếp phân giải bài hát
        if self.is_spotify_url(clean_q):
            t = self.get_track_by_url(clean_q)
            return [t] if t else []

        search_term = clean_q
        if artist and artist.lower() not in clean_q.lower():
            search_term = f"{clean_q} {artist}".strip()

        now = time.time()
        # Tự động mở lại Circuit Breaker nếu đã hết thời gian phạt từ Spotify (Auto-reset after Retry-After)
        if self._quota_exceeded and now >= self._quota_reset_at:
            self._quota_exceeded = False
            self._quota_warning_logged = False
            logger.info("🟢 [SpotifyService] Circuit Breaker: Hết thời gian phạt 429, tự động mở lại kết nối Spotify API.")

        token = self._get_access_token()
        if token and not self._premium_required and not self._quota_exceeded:
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    # Áp dụng Pacer điều tiết nhịp độ gọi API
                    self._pacer.pace()

                    encoded_q = urllib.parse.quote(search_term)
                    search_url = (
                        f"https://api.spotify.com/v1/search?q={encoded_q}&type=track&limit={limit}"
                    )
                    req = urllib.request.Request(
                        search_url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "User-Agent": "MusicIdentify-SpotifyService/1.0",
                        },
                    )
                    self.api_call_count += 1
                    with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        items = (data.get("tracks") or {}).get("items") or []
                        results = [self._parse_track_item(it) for it in items if it]
                        if results:
                            return results
                        return []
                except urllib.error.HTTPError as e:
                    err_body = ""
                    try:
                        err_body = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass

                    # Xử lý HTTP 401 Unauthorized (token hết hạn/bị thu hồi): làm mới token và thử lại
                    if e.code == 401 and attempt < max_retries:
                        logger.info("🔄 [SpotifyService] HTTP 401: Làm mới Access Token và thử lại...")
                        with self._token_lock:
                            self._access_token = None
                            self._token_expires_at = 0.0
                        refreshed = self._get_access_token()
                        if refreshed:
                            token = refreshed
                            continue

                    # Xử lý HTTP 429 Too Many Requests thông minh với Retry-After
                    if e.code == 429 or "quota_exceeded" in err_body.lower() or "too many requests" in err_body.lower():
                        retry_after = 2
                        if getattr(e, "headers", None):
                            try:
                                retry_after = max(1, int(e.headers.get("Retry-After", 2)))
                            except (ValueError, TypeError):
                                retry_after = 2

                        # Nếu retry_after hợp lý (<= 30s) và còn lượt thử, chờ và retry
                        if attempt < max_retries and retry_after <= 30:
                            logger.info(
                                "⏳ [SpotifyService] HTTP 429 Rate Limit: Tạm dừng %ss theo Retry-After (Thử lại %s/%s)...",
                                retry_after,
                                attempt + 1,
                                max_retries,
                            )
                            time.sleep(retry_after + 0.2)
                            continue
                        else:
                            self._quota_exceeded = True
                            self._quota_reset_at = time.time() + max(retry_after, 15.0)
                            if not self._quota_warning_logged:
                                self._quota_warning_logged = True
                                logger.warning(
                                    "⚠️ [SpotifyService] Spotify API thông báo hết hạn mức (HTTP 429: QUOTA_EXCEEDED). "
                                    "Tạm dừng gọi Spotify API trong %s giây (tự động hồi phục sau thời gian này).",
                                    max(retry_after, 15.0)
                                )
                            break
                    elif e.code in (500, 502, 503, 504) and attempt < max_retries:
                        wait_s = 0.6 * (attempt + 1)
                        logger.info(
                            "⏳ [SpotifyService] HTTP %s tạm thời từ Spotify API. Thử lại sau %.1fs (%s/%s)...",
                            e.code,
                            wait_s,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(wait_s)
                        continue
                    elif e.code == 403 and "premium" in err_body.lower():
                        self._premium_required = True
                        if not self._premium_warning_logged:
                            self._premium_warning_logged = True
                            logger.info(
                                "💡 [SpotifyService] Tài khoản chủ sở hữu App là bản Free. "
                                "Spotify yêu cầu gói Premium để gọi Web API danh mục bài hát."
                            )
                        break
                    else:
                        logger.warning(
                            "⚠️ [SpotifyService] Lỗi khi gọi Spotify Search API: %s (%s)",
                            e,
                            err_body.strip(),
                        )
                        break
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    if attempt < max_retries:
                        wait_s = 0.5 * (attempt + 1)
                        logger.info(
                            "⏳ [SpotifyService] Lỗi kết nối tạm thời (%s). Thử lại sau %.1fs (%s/%s)...",
                            e,
                            wait_s,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(wait_s)
                        continue
                    logger.warning("⚠️ [SpotifyService] Lỗi khi gọi Spotify Search API: %s", e)
                    break
                except Exception as e:
                    logger.warning("⚠️ [SpotifyService] Lỗi khi gọi Spotify Search API: %s", e)
                    break

        # Khi không có kết quả hợp lệ hoặc cạn quota, trả về fallback an toàn
        encoded_web = urllib.parse.quote(search_term)
        return [
            {
                "id": "search_fallback",
                "title": clean_q,
                "artist": artist or "",
                "artists_list": [artist] if artist else [],
                "album": "Spotify Web Search",
                "release_date": "",
                "spotify_url": f"https://open.spotify.com/search/{encoded_web}",
                "embed_url": "",
                "preview_url": None,
                "cover_url": None,
                "duration_sec": 0.0,
                "isrc": None,
                "popularity": 0,
                "is_official_api": False,
            }
        ]

    def get_track_by_url(self, url: str) -> dict[str, Any] | None:
        """Trích xuất thông tin bài hát từ URL hoặc URI của Spotify."""
        url = url.strip()
        match = SPOTIFY_TRACK_ID_REGEX.search(url)
        track_id = match.group(1) if match else None

        # 1. Thử lấy qua Spotify API nếu có token và track_id (khi chưa cạn quota)
        now = time.time()
        if self._quota_exceeded and now >= self._quota_reset_at:
            self._quota_exceeded = False
            self._quota_warning_logged = False

        token = self._get_access_token()
        if token and track_id and not self._quota_exceeded:
            try:
                self._pacer.pace()
                api_url = f"https://api.spotify.com/v1/tracks/{track_id}"
                req = urllib.request.Request(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "MusicIdentify-SpotifyService/1.0",
                    },
                )
                self.api_call_count += 1
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data:
                        return self._parse_track_item(data)
            except Exception as e:
                logger.debug("ℹ️ [SpotifyService] Lỗi lấy track qua API v1: %s", e)

        # 2. Fallback qua Spotify OEmbed API (Hoạt động công khai 100% không cần token)
        try:
            oembed_url = f"https://open.spotify.com/oembed?url={urllib.parse.quote(url)}"
            req = urllib.request.Request(
                oembed_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_title = data.get("title", "")
                title = raw_title
                artist = ""
                if " by " in raw_title:
                    parts = raw_title.split(" by ")
                    title = parts[0].strip()
                    artist = parts[1].strip()

                return {
                    "id": track_id or "",
                    "title": title or url,
                    "artist": artist,
                    "artists_list": [artist] if artist else [],
                    "album": "",
                    "release_date": "",
                    "spotify_url": url,
                    "embed_url": f"https://open.spotify.com/embed/track/{track_id}"
                    if track_id
                    else "",
                    "preview_url": None,
                    "cover_url": data.get("thumbnail_url"),
                    "duration_sec": 0.0,
                    "isrc": None,
                    "popularity": 0,
                    "is_official_api": False,
                }
        except Exception as e:
            logger.warning("⚠️ [SpotifyService] OEmbed fallback cũng thất bại: %s", e)

        return None

    def _score_candidates(
        self,
        official_candidates: list[dict[str, Any]],
        title: str,
        artist: str = "",
    ) -> tuple[dict[str, Any] | None, float]:
        """Chấm điểm và chọn ứng viên khớp nhất theo tiêu chuẩn High Precision + Version-Aware Matching."""
        q_prof = _extract_version_profile(title)
        target_title_norm = q_prof["base_norm"] or _clean_str(title)
        target_title_vnorm = q_prof["base_vnorm"] or _clean_str(_normalize_version_str(title))
        target_artist_norm = _clean_str(artist) if artist else ""

        best_score = -1.0
        best_cand: dict[str, Any] | None = None

        for cand in official_candidates:
            raw_cand_title = cand.get("title", "")
            c_prof = _extract_version_profile(raw_cand_title)
            cand_title_norm = c_prof["base_norm"] or _clean_str(raw_cand_title)
            cand_title_vnorm = c_prof["base_vnorm"] or _clean_str(
                _normalize_version_str(raw_cand_title)
            )

            raw_artists = cand.get("artists_list") or [
                p.strip() for p in str(cand.get("artist", "")).split(",") if p.strip()
            ]
            cand_artists_list = [_clean_str(a) for a in raw_artists if _clean_str(a)]
            cand_artist_tokens: set[str] = set()
            cand_artists_nospace: set[str] = set()
            for a in cand_artists_list:
                cand_artist_tokens.update(a.split())
                cand_artists_nospace.add(a.replace(" ", ""))

            # Kiểm tra tương thích phiên bản (Original vs Beat vs Remix vs Live Session vs Remixer)
            is_compatible, version_bonus = _are_version_profiles_compatible(
                q_prof, c_prof, cand_artist_tokens, cand_artists_nospace
            )
            if not is_compatible:
                continue

            if target_artist_norm:
                # Chế độ 1: Có ca sĩ mục tiêu rõ ràng
                title_score = _compute_title_score(
                    target_title_norm,
                    target_title_vnorm,
                    cand_title_norm,
                    cand_title_vnorm,
                    min_sub_ratio=0.55,
                )
                artist_score = _compute_artist_score(
                    artist, cand.get("artist", ""), raw_artists
                )

                # RÀNG BUỘC NGHIÊM NGẶT: Nếu có artist, cả Tên bài gốc lẫn Nghệ sĩ phải đạt ngưỡng cơ sở
                if title_score < 0.70 or artist_score < 0.55:
                    continue

                combined_score = title_score * 0.65 + artist_score * 0.35 + version_bonus
            else:
                # Chế độ 2: Truy vấn tự do không tách cột ca sĩ (vd: "panorama obito", "binh xăng to xuanpac")
                stripped_q_norm, artist_in_query = _remove_matched_artist_tokens(
                    target_title_norm, cand_artists_list
                )
                stripped_q_vnorm, artist_in_vquery = _remove_matched_artist_tokens(
                    target_title_vnorm, cand_artists_list
                )
                has_embedded_artist = artist_in_query or artist_in_vquery

                base_title_score = _compute_title_score(
                    target_title_norm,
                    target_title_vnorm,
                    cand_title_norm,
                    cand_title_vnorm,
                    min_sub_ratio=0.60,
                )

                if has_embedded_artist:
                    embedded_title_score = _compute_title_score(
                        stripped_q_norm,
                        stripped_q_vnorm,
                        cand_title_norm,
                        cand_title_vnorm,
                        min_sub_ratio=0.50,
                    )
                    title_score = max(base_title_score, embedded_title_score)
                    if title_score < 0.75:
                        continue
                    combined_score = title_score * 0.70 + 0.30 + version_bonus
                else:
                    title_score = base_title_score
                    if title_score < 0.75:
                        continue
                    combined_score = title_score + version_bonus

            # Ưu tiên cộng điểm nhẹ cho bài có độ phổ biến cao (tối đa 0.03)
            popularity = cand.get("popularity", 0)
            combined_score += (popularity / 100.0) * 0.03

            if combined_score > best_score:
                best_score = combined_score
                best_cand = cand

        min_required_score = 0.70 if target_artist_norm else 0.75
        if best_cand and best_score >= min_required_score:
            return best_cand, min(1.0, best_score)
        return None, 0.0

    def find_best_match(
        self, title: str, artist: str = "", limit: int = SPOTIFY_SEARCH_LIMIT
    ) -> dict[str, Any] | None:
        """
        Tìm kiếm và xếp hạng kết quả để chọn ra bài hát Spotify khớp nhất.
        Tích hợp:
        - Kiểm tra Persistent SQLite Cache trước (kể cả Negative Cache).
        - Tìm kiếm 1-Shot tối ưu kết hợp fallback có cấu trúc (`track:"..." artist:"..."` và `track:"..."`).
        - Bộ lọc so khớp chính xác cao nhận diện phiên bản (Original / Beat / Remix / Remixer / Live Session).
        - Tuyệt đối KHÔNG trả về candidates ngẫu nhiên nếu không đạt ngưỡng tối thiểu (trả về None).
        - Chỉ lưu Negative Cache khi API phản hồi hợp lệ (không lưu khi lỗi mạng/503/429).
        """
        clean_title = title.strip()
        if not clean_title:
            return None

        # 1. BƯỚC 1: Kiểm tra Persistent Cache (SQLite)
        is_hit, cached_result = self.cache_service.get(clean_title, artist=artist)
        if is_hit:
            return cached_result

        # 2. BƯỚC 2: Gọi Spotify Search API (Mặc định lấy tối thiểu 10 ứng viên trong 1 request)
        search_limit = max(limit, 10)
        candidates = self.search_tracks(query=clean_title, artist=artist, limit=search_limit)

        # Kiểm tra xem có phải lỗi API/hết quota/mất mạng không
        had_api_failure = any(c.get("id") == "search_fallback" for c in candidates)
        official_candidates = [c for c in candidates if c.get("is_official_api")]

        # 3. BƯỚC 3: So khớp xếp hạng bằng High Precision Matching Engine
        best_cand: dict[str, Any] | None = None
        best_score = 0.0

        if official_candidates:
            best_cand, best_score = self._score_candidates(
                official_candidates, title=clean_title, artist=artist
            )
            # Thử hoán đổi Title <-> Artist trên chính tập ứng viên sẵn có (0 extra API calls)
            # Hỗ trợ trường hợp người dùng nhập 'Ca sĩ - Tên bài'
            if not best_cand and artist.strip():
                best_cand, best_score = self._score_candidates(
                    official_candidates, title=artist.strip(), artist=clean_title
                )

        # 4. BƯỚC 4: Smart Structured Fallback Search
        # Chỉ chạy khi 1-shot không lỗi API nhưng chưa khớp được ứng viên nào
        if not best_cand and not had_api_failure and not self._quota_exceeded and not self._premium_required:
            fallback_queries: list[tuple[str, str]] = []
            q_prof = _extract_version_profile(clean_title)
            base_q = (q_prof["base_title"] or _strip_noise(clean_title) or clean_title).replace('"', "").strip()
            split_artists = _split_artists(artist) if artist.strip() else []
            primary_artist = split_artists[0] if split_artists else ""

            if artist.strip():
                # Ưu tiên 1: Tìm chính xác theo track:"..." + artist chính đầu tiên
                # Khắc phục trường hợp cột Ca sĩ có thêm tên kênh/đơn vị (vd: 'InĐậm.Stage', 'ThuyDung Bolero')
                if base_q and primary_artist:
                    fallback_queries.append((f'track:"{base_q}" artist:"{primary_artist}"', ""))
                # Ưu tiên 2: Tìm chính xác theo track:"..." (lấy toàn bộ phiên bản Remix/Beat/Live của bài)
                if base_q:
                    fallback_queries.append((f'track:"{base_q}"', ""))
            else:
                if base_q and base_q != clean_title:
                    fallback_queries.append((f'track:"{base_q}"', ""))
                words = base_q.split()
                if len(words) >= 5:
                    prefix_len = min(4, max(3, len(words) // 2 + 1))
                    fallback_queries.append((" ".join(words[:prefix_len]), ""))

            seen_fb: set[tuple[str, str]] = {(clean_title.lower(), artist.lower())}
            for fb_q, fb_a in fallback_queries:
                key_pair = (fb_q.lower(), fb_a.lower())
                if not fb_q or key_pair in seen_fb:
                    continue
                seen_fb.add(key_pair)

                fb_cands = self.search_tracks(query=fb_q, artist=fb_a, limit=search_limit)
                if any(c.get("id") == "search_fallback" for c in fb_cands):
                    had_api_failure = True
                    continue
                fb_official = [c for c in fb_cands if c.get("is_official_api")]
                if not fb_official:
                    continue

                # Luôn đối soát lại trên clean_title & artist gốc để giữ nguyên độ chính xác cao
                best_cand, best_score = self._score_candidates(
                    fb_official, title=clean_title, artist=artist
                )
                if not best_cand and artist.strip():
                    best_cand, best_score = self._score_candidates(
                        fb_official, title=artist.strip(), artist=clean_title
                    )
                if best_cand:
                    break

        if best_cand:
            best_cand["match_score"] = round(best_score, 2)
            # Lưu Positive Cache (TTL 7 ngày)
            self.cache_service.set(clean_title, artist=artist, data=best_cand, is_found=True)
            return best_cand

        # KHÔNG ĐẠT NGƯỠNG: Tuyệt đối KHÔNG trả về candidates[0]!
        # Chỉ lưu Negative Cache nếu API không bị lỗi (tránh cache nhầm lỗi 503/429/timeout)
        if not had_api_failure:
            self.cache_service.set(clean_title, artist=artist, data=None, is_found=False)
        return None

    def batch_search_tracks(
        self,
        queries: list[dict[str, Any] | str],
        max_workers: int = SPOTIFY_BATCH_WORKERS,
        on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tra cứu hàng loạt danh sách bài hát trên Spotify bằng Multi-threading Pool.
        Bảo toàn 100% thứ tự bài hát ban đầu, tích hợp Cache và chuẩn hóa trạng thái rõ ràng.
        Nếu không tìm thấy bài hát: Để trống link và ghi nhận rõ trạng thái Chưa tìm thấy.
        """
        if not queries:
            return []

        # Chuẩn hóa danh sách đầu vào
        normalized_inputs: list[tuple[int, str, str, dict[str, Any]]] = []
        for idx, item in enumerate(queries):
            raw_meta: dict[str, Any] = {}
            if isinstance(item, dict):
                q = str(
                    item.get("query")
                    or item.get("title")
                    or item.get("track_title")
                    or ""
                ).strip()
                artist = str(item.get("artist") or "").strip()
                raw_meta = dict(item)
            else:
                raw_str = str(item).strip()
                # Tách nếu có cú pháp 'Tên bài - Ca sĩ'
                if " - " in raw_str and not raw_str.startswith("http"):
                    parts = raw_str.split(" - ", 1)
                    q = parts[0].strip()
                    artist = parts[1].strip()
                else:
                    q = raw_str
                    artist = ""
            normalized_inputs.append((idx, q, artist, raw_meta))

        total_count = len(normalized_inputs)
        results: list[dict[str, Any] | None] = [None] * total_count
        completed_count = 0
        lock = threading.Lock()

        def _worker(task_tuple: tuple[int, str, str, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            i, query_val, artist_val, raw_m = task_tuple
            match_res: dict[str, Any] | None = None
            try:
                # Nếu chuỗi là URL Spotify trực tiếp
                if self.is_spotify_url(query_val):
                    match_res = self.get_track_by_url(query_val)
                elif query_val:
                    match_res = self.find_best_match(query_val, artist=artist_val)
            except Exception as exc:
                logger.warning("Lỗi tra cứu Spotify bài '%s': %s", query_val, exc)

            # Xác định cờ tìm thấy hợp lệ (chỉ True khi có URL bản thu chính thức)
            is_official_found = bool(
                match_res
                and match_res.get("spotify_url")
                and not str(match_res.get("id", "")).startswith("search_fallback")
            )

            # Nếu không tìm thấy hoặc là fallback search link: Để trống link và không gán thông tin bài lạ
            final_spotify_url = match_res.get("spotify_url") if is_official_found else None
            final_spotify_id = match_res.get("id") if is_official_found else None
            final_spotify_title = match_res.get("title") if is_official_found else None
            final_spotify_artist = match_res.get("artist") if is_official_found else None
            final_spotify_album = match_res.get("album") if is_official_found else None
            final_release_date = match_res.get("release_date") if is_official_found else None
            final_isrc = match_res.get("isrc") if is_official_found else None
            final_score = match_res.get("match_score", 1.0) if is_official_found else 0.0

            res_record: dict[str, Any] = {
                "index": i + 1,
                "input_query": query_val,
                "input_artist": artist_val,
                "spotify_id": final_spotify_id,
                "spotify_title": final_spotify_title,
                "spotify_artist": final_spotify_artist,
                "spotify_album": final_spotify_album,
                "spotify_release_date": final_release_date,
                "spotify_url": final_spotify_url,
                "spotify_isrc": final_isrc,
                "match_score": final_score,
                "cover_url": match_res.get("cover_url") if is_official_found else None,
                "duration_sec": match_res.get("duration_sec", 0.0) if is_official_found else 0.0,
                "popularity": match_res.get("popularity", 0) if is_official_found else 0,
                "embed_url": match_res.get("embed_url") if is_official_found else None,
                "is_found": is_official_found,
                "is_search_link": False,
                "raw_input_meta": raw_m,
            }
            return i, res_record

        # Thực thi đa luồng với ThreadPoolExecutor
        workers = max(1, min(max_workers, 8))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(_worker, inp): inp[0] for inp in normalized_inputs}
            for future in as_completed(future_to_idx):
                idx_res, res_item = future.result()
                results[idx_res] = res_item
                with lock:
                    completed_count += 1
                    if on_progress:
                        try:
                            on_progress(completed_count, total_count, res_item)
                        except Exception:
                            pass

        return [r for r in results if r is not None]

    @staticmethod
    def is_spotify_url(url: str) -> bool:
        """Kiểm tra một chuỗi URL có phải thuộc về Spotify hay không."""
        return bool(SPOTIFY_TRACK_ID_REGEX.search(url.strip()))
