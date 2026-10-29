import hashlib
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yt_dlp

from ..core.config import get_cookie_file_path, get_js_runtimes_config, get_youtube_proxy
from ..core.exceptions import InvalidInputException, TrackNotFoundException
from ..core.interfaces.services.resolver_interface import IInputResolver
from ..core.models.entities import InputType, TrackItem

URL_REGEX = re.compile(
    r"^(?:http|ftp)s?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)

YOUTUBE_REGEX = re.compile(r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+", re.IGNORECASE)

ZING_REGEX = re.compile(r"^(https?://)?(www\.)?(zingmp3\.vn|mp3\.zing\.vn)/.+", re.IGNORECASE)

SPOTIFY_REGEX = re.compile(
    r"^(https?://)?(open\.)?spotify\.com/track/[a-zA-Z0-9]+|spotify:track:[a-zA-Z0-9]+",
    re.IGNORECASE,
)

from ..core.utils import extract_first_entry as _extract_first_entry
from ..core.utils import normalize_vietnamese_text


def _normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản loại bỏ dấu và ký tự đặc biệt để so khớp chuỗi."""
    return normalize_vietnamese_text(text, remove_special_chars=True)


def _normalize_date_str(val: Any) -> str | None:
    """Chuẩn hóa giá trị ngày phát hành về định dạng chuẩn YYYY-MM-DD."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, (int, float)):
        try:
            val = int(val)
        except (ValueError, OverflowError):
            pass
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2].strip()
    if not s or s.lower() in ("none", "nan", "null", "n/a", "nat"):
        return None
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    if "t" in s.lower():
        s = re.split(r"[Tt]", s)[0]
    if " " in s:
        s = s.split(" ")[0]
    s = s.replace("/", "-")
    return s.strip() or None


def extract_content_id_metadata(description: str, tags: list[str] | None = None) -> dict[str, str]:
    """
    Trích xuất siêu dữ liệu bản quyền tự động từ YouTube Content ID trong description hoặc tags.
    Hỗ trợ nhận diện các mẫu Song / Artist / Album / Licensed to bằng cả tiếng Anh và tiếng Việt.
    """
    meta: dict[str, str] = {}
    if not description:
        return meta

    # 1. Trích xuất Tên bài hát (Song / Track / Bài hát)
    song_match = re.search(r"(?:Song|Bài hát|Track):\s*([^\n\r]+)", description, re.IGNORECASE)
    if song_match:
        meta["content_id_song"] = song_match.group(1).strip()

    # 2. Trích xuất Nghệ sĩ (Artist / Nghệ sĩ / Trình bày)
    artist_match = re.search(
        r"(?:Artist|Nghệ sĩ|Trình bày):\s*([^\n\r]+)", description, re.IGNORECASE
    )
    if artist_match:
        meta["content_id_artist"] = artist_match.group(1).strip()

    # 3. Trích xuất Album
    album_match = re.search(r"(?:Album):\s*([^\n\r]+)", description, re.IGNORECASE)
    if album_match:
        meta["content_id_album"] = album_match.group(1).strip()

    # 4. Trích xuất Đơn vị cấp phép (Licensed to YouTube by / Cấp phép cho YouTube bởi)
    licensed_match = re.search(
        r"(?:Licensed to YouTube by|Cấp phép cho YouTube bởi):\s*([^\n\r]+)",
        description,
        re.IGNORECASE,
    )
    if licensed_match:
        meta["content_id_licensed_by"] = licensed_match.group(1).strip()

    return meta


def generate_deterministic_track_id(
    query: str, input_type: InputType, resolved_url: str | None = None
) -> str:
    """Tạo ID định danh cố định (Deterministic Track ID) cho bài hát để chống trùng lặp."""
    target = (resolved_url or query or "").strip()

    # 1. YouTube: trích xuất video ID chuẩn (11 ký tự)
    if "youtube.com" in target or "youtu.be" in target:
        yt_match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", target)
        if yt_match:
            return f"yt_{yt_match.group(1)}"

    # 2. Zing MP3: trích xuất song ID (ví dụ ZW... hoặc tương tự)
    if "zingmp3.vn" in target:
        zing_match = re.search(r"(?:bai-hat\/[^\/]+\/|song\/)([A-Z0-9]{8,12})", target)
        if zing_match:
            return f"zing_{zing_match.group(1)}"

    # 3. SoundCloud: lấy slug hoặc hash URL
    if "soundcloud.com" in target:
        slug = target.split("soundcloud.com/")[-1].split("?")[0].strip("/").replace("/", "_")
        if slug:
            clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "", slug)[:24]
            return f"sc_{clean_slug}"

    # 4. Local file: Tính MD5 của file nếu file tồn tại
    if input_type == InputType.LOCAL_FILE or (
        not target.startswith("http") and Path(target).exists()
    ):
        p = Path(target)
        if p.exists() and p.is_file():
            try:
                hasher = hashlib.md5()
                with open(p, "rb") as f:
                    chunk = f.read(524288)
                    hasher.update(chunk)
                    if p.stat().st_size > 524288:
                        f.seek(-min(524288, p.stat().st_size), os.SEEK_END)
                        hasher.update(f.read(524288))
                return f"file_{hasher.hexdigest()[:12]}"
            except Exception:
                pass
        clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", p.stem)[:20]
        return f"file_{clean_stem}"

    # 5. Query văn bản hoặc URL khác: Chuẩn hóa và băm MD5
    norm_text = re.sub(r"\s+", " ", target.lower().strip())
    h = hashlib.md5(norm_text.encode("utf-8")).hexdigest()[:12]
    return f"trk_{h}"


class InputResolverService(IInputResolver):
    """Hiện thực phân giải đầu vào bài hát đa nền tảng (YouTube, Zing MP3, Spotify, Local File)."""

    def __init__(self) -> None:
        self._cache: dict[str, TrackItem] = {}

    def is_youtube_url(self, url: str) -> bool:
        """Kiểm tra URL có thuộc YouTube hay không."""
        return bool(YOUTUBE_REGEX.match(url.strip()))

    def is_zingmp3_url(self, url: str) -> bool:
        """Kiểm tra URL có thuộc Zing MP3 hay không."""
        return bool(ZING_REGEX.match(url.strip()))

    def is_spotify_url(self, url: str) -> bool:
        """Kiểm tra URL có thuộc Spotify hay không."""
        return bool(SPOTIFY_REGEX.match(url.strip()))

    def detect_input_type(self, query: str) -> InputType:
        """Phân loại chuỗi nhập vào là URL, File nội bộ, hay Tên bài hát."""
        cleaned = query.strip()
        if URL_REGEX.match(cleaned):
            return InputType.URL
        if os.path.isfile(cleaned):
            return InputType.LOCAL_FILE
        return InputType.SONG_NAME

    def validate_match(
        self,
        target_title: str,
        target_artist: str = "",
        candidate: dict[str, Any] | None = None,
        target_duration: float | None = None,
    ) -> tuple[bool, float]:
        """
        Xác thực độ trùng khớp thông minh giữa bài hát mục tiêu và ứng viên tìm kiếm (YouTube/Web).
        Sử dụng so khớp chuỗi mờ (Fuzzy title/artist) và kiểm tra dung sai thời lượng (Duration check).
        Trả về: (is_match: bool, similarity_score: float)
        """
        if not candidate:
            return False, 0.0

        cand_title = str(candidate.get("title") or "").strip()
        cand_uploader = str(candidate.get("uploader") or candidate.get("channel") or "").strip()
        cand_duration = candidate.get("duration")

        if not cand_title:
            return False, 0.0

        clean_target_title = _normalize_text(target_title)
        clean_cand_title = _normalize_text(cand_title)
        clean_target_artist = _normalize_text(target_artist) if target_artist else ""
        clean_cand_uploader = _normalize_text(cand_uploader)

        # 0. Loại bỏ ngay các video công chiếu chưa diễn ra (Premiere / Upcoming stream)
        live_status = str(candidate.get("live_status") or "").lower()
        if "upcoming" in live_status or candidate.get("is_live") is True:
            return False, 0.0

        # Tách danh sách sub-artists từ target_artist trước khi xóa dấu phẩy/kí tự đặc biệt
        sub_artists: list[str] = []
        if target_artist:
            raw_subs = [
                a.strip()
                for a in re.split(
                    r"[,/&|]|(?:\s+ft\.?\s+)|\s+feat\.?\s+|\s+x\s+",
                    target_artist,
                    flags=re.IGNORECASE,
                )
                if a.strip()
            ]
            sub_artists = [_normalize_text(a) for a in raw_subs if _normalize_text(a)]
        if not sub_artists and clean_target_artist:
            sub_artists = [clean_target_artist]

        # Lọc bỏ tên nghệ sĩ khỏi clean_target_title nếu bị dính do query gộp
        for sub_a in sub_artists:
            if sub_a and sub_a in clean_target_title:
                clean_target_title = clean_target_title.replace(sub_a, "").strip()
        for word in clean_target_artist.split():
            if len(word) > 2 and clean_target_title.endswith(word):
                clean_target_title = re.sub(
                    rf"\b{re.escape(word)}\b", "", clean_target_title
                ).strip()

        # 1. So sánh tiêu đề bài hát
        title_sim = SequenceMatcher(None, clean_target_title, clean_cand_title).ratio()

        # Kiểm tra tỷ lệ từ khóa (Token overlap)
        target_words = set(clean_target_title.split())
        cand_words = set(clean_cand_title.split())
        token_overlap = 0.0
        if target_words:
            token_overlap = len(target_words & cand_words) / len(target_words)
            title_sim = max(title_sim, token_overlap * 0.9)

            # Ràng buộc cứng về tiêu đề: Ứng viên phải chứa ít nhất 40% số từ của tiêu đề bài hát mục tiêu
            # Hoặc tỷ lệ SequenceMatcher phải >= 0.60. Nếu không thỏa mãn -> LOẠI NGAY LẬP TỨC!
            if token_overlap < 0.40 and title_sim < 0.60:
                return False, round(min(title_sim * 0.30, 0.30), 3)

        # Nếu không có target_artist riêng biệt, so sánh target_title với cả (cand_title + uploader)
        if not clean_target_artist and clean_cand_uploader:
            cand_full = f"{clean_cand_title} {clean_cand_uploader}".strip()
            full_sim = SequenceMatcher(None, clean_target_title, cand_full).ratio()
            cand_full_words = set(cand_full.split())
            if target_words:
                full_overlap = len(target_words & cand_full_words) / len(target_words)
                title_sim = max(title_sim, full_sim, full_overlap * 0.95)

        # 2. So sánh nghệ sĩ nếu có (hỗ trợ đa nghệ sĩ phân tách bởi dấu phẩy, ft, feat, x, &)
        artist_sim = 0.5
        if sub_artists:
            best_sub_sim = 0.0

            # Làm sạch tên uploader loại bỏ các hậu tố phổ biến của kênh
            uploader_core = re.sub(
                r"\b(official|topic|channel|vevo|records|music|entertainment|ent|audio|studio)\b",
                "",
                clean_cand_uploader,
            ).strip()

            for sub in sub_artists:
                # a) Khớp nguyên cụm sub-artist có ranh giới từ trong uploader
                if re.search(rf"\b{re.escape(sub)}\b", clean_cand_uploader) or (
                    uploader_core and re.search(rf"\b{re.escape(sub)}\b", uploader_core)
                ):
                    # Nếu uploader chứa các từ lạ không thuộc sub và không phải tiền/hậu tố kênh:
                    # Kiểm tra độ tương đồng chuỗi để chống nhận nhầm nghệ sĩ con (VD: 'anh vu' vs 'anh tuan vu')
                    if uploader_core:
                        up_words = set(uploader_core.split())
                        sub_w = set(sub.split())
                        if up_words - sub_w:
                            ratio_exact = SequenceMatcher(None, sub, uploader_core).ratio()
                            if ratio_exact < 0.80:
                                continue
                    best_sub_sim = max(best_sub_sim, 1.0)
                    break

                # b) Độ tương đồng SequenceMatcher trực tiếp với uploader_core
                if uploader_core:
                    sim_uploader = SequenceMatcher(None, sub, uploader_core).ratio()
                    if sim_uploader >= 0.85:
                        best_sub_sim = max(best_sub_sim, sim_uploader)

                # c) Khớp nguyên cụm sub-artist có ranh giới từ trong tiêu đề video
                if re.search(rf"\b{re.escape(sub)}\b", clean_cand_title):
                    best_sub_sim = max(best_sub_sim, 0.90)

                # d) Đối với tên nhóm/dự án dài (>= 3 từ, ví dụ 'Anh Trai Vượt Ngàn Chông Gai'):
                sub_words = set(sub.split())
                uploader_words = set(uploader_core.split()) if uploader_core else set()
                if len(sub_words) >= 3 and uploader_words:
                    jaccard = len(sub_words & uploader_words) / len(sub_words | uploader_words)
                    if jaccard >= 0.60:
                        best_sub_sim = max(best_sub_sim, jaccard)

            # e) Nghệ sĩ hợp tác phân tán giữa uploader và tiêu đề (VD: 'xuanpac yb rookie'
            # với uploader 'xuanpacdontlie' và tiêu đề 'ft. @Yb.rookie'):
            # Chỉ kích hoạt khi tiêu đề chứa từ khóa collab rõ ràng (ft., feat., x, &, with, @)
            # để tránh nhầm lẫn với trường hợp tên ca sĩ đơn bị trùng lặp bộ phận.
            collab_pattern = re.compile(
                r"(?:(?:\s+ft\.?\s+)|\s+feat\.?\s+|\s+featuring\s+|\s+x\s+|\s+&\s+|\s+with\s+|@)(.+)",
                re.IGNORECASE,
            )
            collab_match = collab_pattern.search(cand_title)
            if collab_match:
                collab_text = _normalize_text(collab_match.group(1))
                artist_tokens = [w for w in clean_target_artist.split() if len(w) > 1]
                if len(artist_tokens) >= 2:
                    in_uploader = [
                        tok
                        for tok in artist_tokens
                        if re.search(rf"\b{re.escape(tok)}\b", clean_cand_uploader)
                        or (len(tok) >= 4 and tok in clean_cand_uploader)
                    ]
                    in_collab_title = [
                        tok
                        for tok in artist_tokens
                        if re.search(rf"\b{re.escape(tok)}\b", collab_text)
                        or (len(tok) >= 4 and tok in collab_text)
                    ]
                    if (
                        in_uploader
                        and in_collab_title
                        and (set(in_uploader) != set(in_collab_title))
                    ):
                        matched_union = set(in_uploader) | set(in_collab_title)
                        token_ratio = len(matched_union) / len(artist_tokens)
                        if token_ratio >= 0.60:
                            best_sub_sim = max(best_sub_sim, token_ratio * 0.95)

            artist_sim = best_sub_sim

            # Nếu có chỉ định ca sĩ mà ứng viên không đạt độ khớp nghệ sĩ:
            if artist_sim < 0.60:
                if title_sim >= 0.75 or token_overlap >= 0.75:
                    return False, round(min(title_sim * 0.55, 0.55), 3)
                return False, round(min(title_sim * 0.35, 0.35), 3)

        combined_score = (title_sim * 0.7 + artist_sim * 0.3) if clean_target_artist else title_sim

        # 3. Kiểm tra độ lệch thời lượng (Duration Check)
        if cand_duration and float(cand_duration) > 600.0:
            return False, 0.0

        if target_duration and target_duration > 0 and cand_duration and float(cand_duration) > 0:
            duration_diff = abs(float(cand_duration) - float(target_duration))
            # Nếu lệch thời lượng quá 15 giây VÀ tỉ lệ lệch > 15% thời lượng mục tiêu
            if duration_diff > 15.0 and (duration_diff / float(target_duration) > 0.15):
                return False, round(combined_score * 0.2, 3)

        is_match = combined_score >= 0.60
        if clean_target_artist and artist_sim < 0.60:
            is_match = False
        return is_match, round(combined_score, 3)

    def _fetch_zing_metadata(self, url_or_query: str) -> dict[str, Any]:
        """Trích xuất thông tin bài hát từ Zing MP3 qua API hoặc URL slug."""
        clean_query = url_or_query.strip()
        # Nếu là URL, cố gắng bóc tách slug hoặc song code
        match_code = re.search(r"/([A-Za-z0-9]+)\.html", clean_query)
        song_code = match_code.group(1) if match_code else ""

        # Tìm kiếm qua Zing MP3 Autocomplete API
        search_kw = song_code or clean_query
        match_slug = re.search(r"/bai-hat/([^/]+)/", clean_query)
        if match_slug:
            search_kw = match_slug.group(1).replace("-", " ")

        ssl_ctx = None
        try:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        except Exception:
            ssl_ctx = None

        for proto in ["https", "http"]:
            api_url = f"{proto}://ac.mp3.zing.vn/complete?type=artist,song,key,album&query={urllib.parse.quote(search_kw)}"
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                open_kwargs: dict[str, Any] = {"timeout": 4}
                if ssl_ctx and proto == "https":
                    open_kwargs["context"] = ssl_ctx
                with urllib.request.urlopen(req, **open_kwargs) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    songs: list[dict[str, Any]] = []
                    for item in data.get("data", []):
                        if isinstance(item, dict) and "song" in item:
                            songs = item.get("song", [])
                            break
                    if songs:
                        s = songs[0]
                        return {
                            "title": s.get("name") or search_kw,
                            "artist": s.get("artist") or "",
                            "duration": float(s.get("duration", 0)),
                            "stream_url": f"https://zingmp3.vn/bai-hat/{s.get('id')}.html",
                            "is_vip": str(s.get("streamingStatus")) != "1"
                            or str(s.get("block", "")).lower() in ["true", "1"],
                        }
            except Exception:
                continue

        return {
            "title": search_kw,
            "artist": "",
            "duration": 0.0,
            "stream_url": clean_query if self.is_zingmp3_url(clean_query) else "",
            "is_vip": False,
        }

    def _fetch_spotify_metadata(self, url: str) -> dict[str, Any]:
        """Trích xuất thông tin bài hát Spotify qua API OEmbed chính thức."""
        api_url = f"https://open.spotify.com/oembed?url={urllib.parse.quote(url)}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                title = data.get("title", "")
                artist = ""
                # Oembed title thường có dạng "Title by Artist" hoặc "Title"
                if " by " in title:
                    parts = title.split(" by ")
                    title = parts[0].strip()
                    artist = parts[1].strip()
                return {
                    "title": title,
                    "artist": artist,
                    "duration": 0.0,
                    "preview_url": None,
                }
        except Exception as e:  # noqa: BLE001
            print(f"[InputResolverService] Lỗi gọi API Spotify OEmbed: {e}")

        return {
            "title": url,
            "artist": "",
            "duration": 0.0,
            "preview_url": None,
        }

    def search_zingmp3(self, song_name: str, target_artist: str = "") -> dict[str, Any]:
        """Tìm kiếm bài hát trên Zing MP3 và xác thực độ trùng khớp."""
        search_term = f"{song_name} {target_artist}".strip() if target_artist else song_name.strip()
        meta = self._fetch_zing_metadata(search_term)
        if not meta or not meta.get("stream_url") or meta.get("is_vip", False):
            return {}

        cand = {"title": meta.get("title", ""), "uploader": meta.get("artist", "")}
        is_match, score = self.validate_match(
            target_title=song_name,
            target_artist=target_artist,
            candidate=cand,
            target_duration=meta.get("duration"),
        )
        if not (is_match and score >= 0.50):
            return {}

        meta["match_score"] = score
        return meta

    def search_spotify(self, song_name: str) -> dict[str, Any]:
        """Tìm kiếm bài hát trên Spotify (hỗ trợ preview 30s hoặc canonical metadata)."""
        clean_q = song_name.strip()
        if self.is_spotify_url(clean_q):
            return self._fetch_spotify_metadata(clean_q)

        try:
            api_url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/track/{urllib.parse.quote(clean_q)}"
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                title = data.get("title", "")
                if title:
                    return {
                        "title": title,
                        "artist": "",
                        "duration": 0.0,
                        "preview_url": None,
                        "track_url": f"https://open.spotify.com/search/{urllib.parse.quote(clean_q)}",
                    }
        except Exception:  # noqa: BLE001, S110
            pass

        return {}

    def resolve_track(self, query: str, artist: str = "", force_rescan: bool = False) -> TrackItem:
        """Chuyển đổi một chuỗi truy vấn (YouTube, Zing, Spotify, Local, Text) thành TrackItem hoàn chỉnh."""
        query = query.strip()
        if not query:
            raise InvalidInputException("Chuỗi tìm kiếm không được để trống.")

        cache_key = f"{query.lower()}||{artist.strip().lower()}"
        if not force_rescan and cache_key in self._cache:
            c = self._cache[cache_key]
            return TrackItem(
                id=c.id,
                raw_query=c.raw_query,
                input_type=c.input_type,
                resolved_url=c.resolved_url,
                track_title=c.track_title,
                artist=c.artist,
                duration_sec=c.duration_sec,
                platform=c.platform,
                is_verified_match=c.is_verified_match,
                match_score=c.match_score,
                preview_url=c.preview_url,
            )

        item = self._do_resolve_track(query, artist)
        self._cache[cache_key] = item
        return item

    def _do_resolve_track(self, query: str, artist: str = "") -> TrackItem:
        """Thực hiện giải quyết bài hát chi tiết khi không có cache."""
        input_type = self.detect_input_type(query)
        track_id = generate_deterministic_track_id(query, input_type)

        if input_type == InputType.LOCAL_FILE:
            file_name = Path(query).stem
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=query,
                track_title=file_name,
                platform="local",
                is_verified_match=True,
            )

        if input_type == InputType.URL:
            # 1. Nhánh Zing MP3
            if self.is_zingmp3_url(query):
                meta = self._fetch_zing_metadata(query)
                title = meta.get("title") or query
                artist = meta.get("artist") or ""
                duration = meta.get("duration") or None

                is_vip = meta.get("is_vip", False)
                # Thử lấy URL stream trực tiếp qua yt-dlp nếu không phải VIP
                if not is_vip:
                    url_meta_title, url_meta_dur = self._get_url_metadata(query)
                    if url_meta_title:
                        return TrackItem(
                            id=track_id,
                            raw_query=query,
                            input_type=input_type,
                            resolved_url=query,
                            track_title=title,
                            artist=artist,
                            duration_sec=duration or url_meta_dur,
                            platform="zingmp3",
                            is_verified_match=True,
                        )

                # Fallback: Nếu bài VIP hoặc yt-dlp không tải được Zing, dùng Smart Bridge sang YouTube
                search_term = f"{title} {artist}".strip()
                res_url, _, res_dur, is_verified, score = self.search_youtube(
                    search_term, target_artist=artist, target_duration=duration
                )
                return TrackItem(
                    id=generate_deterministic_track_id(
                        query, input_type, resolved_url=res_url or query
                    ),
                    raw_query=query,
                    input_type=input_type,
                    resolved_url=res_url or query,
                    track_title=title,
                    artist=artist,
                    duration_sec=duration or res_dur,
                    platform="zingmp3_bridge",
                    is_verified_match=is_verified,
                    match_score=score,
                )

            # 2. Nhánh Spotify
            if self.is_spotify_url(query):
                meta = self._fetch_spotify_metadata(query)
                title = meta.get("title") or query
                artist = meta.get("artist") or ""
                duration = meta.get("duration") or None

                # Cầu nối Smart Bridge sang YouTube bằng Title + Artist chuẩn
                search_term = f"{title} {artist}".strip()
                res_url, _, res_dur, is_verified, score = self.search_youtube(
                    search_term, target_artist=artist, target_duration=duration
                )
                return TrackItem(
                    id=generate_deterministic_track_id(
                        query, input_type, resolved_url=res_url or query
                    ),
                    raw_query=query,
                    input_type=input_type,
                    resolved_url=res_url or query,
                    track_title=title,
                    artist=artist,
                    duration_sec=duration or res_dur,
                    platform="spotify_bridge",
                    is_verified_match=is_verified,
                    match_score=score,
                    preview_url=meta.get("preview_url"),
                )

            # 3. Nhánh YouTube hoặc URL thông thường
            track_title, duration = self._get_url_metadata(query)
            return TrackItem(
                id=generate_deterministic_track_id(query, input_type, resolved_url=query),
                raw_query=query,
                input_type=input_type,
                resolved_url=query,
                track_title=track_title or query,
                duration_sec=duration,
                platform="youtube" if self.is_youtube_url(query) else "web_url",
                is_verified_match=True,
            )

        # 4. Nhánh Tên bài hát tự do (Text query) - Chuỗi Fallback 3 tầng: YouTube -> Zing MP3 -> Spotify
        target_artist = artist.strip()
        target_title = query.strip()
        if not target_artist:
            for sep in [" - ", " – ", " — ", " | "]:
                if sep in target_title:
                    parts = target_title.split(sep, 1)
                    target_title = parts[0].strip()
                    target_artist = parts[1].strip()
                    break
        else:
            # Nếu target_artist đã được chỉ định, tách lọc bỏ phần tên nghệ sĩ khỏi target_title
            for sep in [" - ", " – ", " — ", " | "]:
                if sep in target_title:
                    parts = target_title.split(sep, 1)
                    target_title = parts[0].strip()
                    break
            for a_part in [target_artist] + [
                a.strip() for a in target_artist.split(",") if len(a.strip()) > 2
            ]:
                if a_part and a_part.lower() in target_title.lower():
                    target_title = re.sub(
                        re.escape(a_part), "", target_title, flags=re.IGNORECASE
                    ).strip(" -–—|/,")
            target_title = target_title.strip() or query.strip()

        # TẦNG 1: Ưu tiên tìm kiếm trên Zing MP3 (kho dữ liệu gốc cho bài hát V-Pop & dự án AI độc lập)
        zing_meta = self.search_zingmp3(target_title, target_artist=target_artist)
        if zing_meta and zing_meta.get("stream_url") and not zing_meta.get("is_vip", False):
            z_stream = zing_meta["stream_url"]
            return TrackItem(
                id=generate_deterministic_track_id(query, input_type, resolved_url=z_stream),
                raw_query=query,
                input_type=input_type,
                resolved_url=z_stream,
                track_title=zing_meta.get("title") or query,
                artist=zing_meta.get("artist", "") or target_artist or None,
                duration_sec=zing_meta.get("duration"),
                platform="zingmp3",
                is_verified_match=True,
                match_score=zing_meta.get("match_score", 0.90),
            )

        # TẦNG 2: Zing MP3 không tìm thấy / bị khóa VIP -> Tìm kiếm trên YouTube qua Smart Matcher
        yt_url, yt_title, yt_dur, yt_verified, yt_score = self.search_youtube(
            target_title, target_artist=target_artist
        )
        if yt_verified and yt_url:
            return TrackItem(
                id=generate_deterministic_track_id(query, input_type, resolved_url=yt_url),
                raw_query=query,
                input_type=input_type,
                resolved_url=yt_url,
                track_title=yt_title or query,
                artist=target_artist or None,
                duration_sec=yt_dur,
                platform="youtube_search",
                is_verified_match=True,
                match_score=yt_score,
            )

        # TẦNG 3: Zing MP3 không tìm thấy / bị khóa VIP -> Fallback sang Spotify
        spotify_meta = self.search_spotify(query)
        if spotify_meta and (spotify_meta.get("preview_url") or spotify_meta.get("track_url")):
            stream_url = spotify_meta.get("preview_url") or spotify_meta.get("track_url")
            return TrackItem(
                id=generate_deterministic_track_id(query, input_type, resolved_url=stream_url),
                raw_query=query,
                input_type=input_type,
                resolved_url=stream_url,
                track_title=spotify_meta.get("title") or query,
                artist=spotify_meta.get("artist", "") or target_artist or None,
                duration_sec=spotify_meta.get("duration"),
                platform="spotify",
                is_verified_match=True,
                match_score=0.90,
                preview_url=spotify_meta.get("preview_url"),
            )

        # TẦNG 4: NẾU CẢ 3 NỀN TẢNG ĐỀU KHÔNG TÌM THẤY BẢN THU XÁC THỰC -> BÁO LỖI TrackNotFoundException
        # Tuyệt đối không chấp nhận các video không liên quan hoặc lệch nghệ sĩ để tránh thẩm định sai bản thu.
        raise TrackNotFoundException(
            track_id=query,
            message=f"Không tìm thấy bản thu chính chủ. Không tìm thấy bài hát '{query}' trên YouTube, Zing MP3, hoặc Spotify.",
        )

    def search_youtube(
        self,
        song_name: str,
        target_artist: str = "",
        target_duration: float | None = None,
    ) -> tuple[str | None, str | None, float | None, bool, float]:
        """
        Tìm kiếm bài hát trên YouTube với bộ lọc Top 5, xác thực thông minh và Fallback đa Pass.
        Trả về: (url, title, duration, is_verified_match, match_score)
        """
        js_runtimes_cfg = get_js_runtimes_config()

        # Tìm kiếm YouTube hoàn toàn mở: không truyền cookie tài khoản cá nhân (để tránh lỗi session reload)
        # và chỉ định client web_embedded / mweb an toàn.
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "extractor_args": {"youtube": {"player_client": ["android", "web_embedded"]}},
        }
        if js_runtimes_cfg:
            ydl_opts["js_runtimes"] = js_runtimes_cfg
        yt_proxy = get_youtube_proxy()
        if yt_proxy:
            ydl_opts["proxy"] = yt_proxy

        # Danh sách các câu truy vấn tìm kiếm theo thứ tự ưu tiên
        candidate_queries: list[str] = []
        clean_title = re.sub(r"\(.*?\)|\[.*?\]", "", song_name).strip()

        # Pass 0: Tìm kiếm chính xác có trích dẫn ngoặc kép nghệ sĩ
        # (Giúp YouTube ưu tiên đúng kênh/nghệ sĩ chính chủ thay vì các bản mashup/remix/cover nổi tiếng khác)
        if target_artist:
            candidate_queries.append(f'{clean_title or song_name} "{target_artist}"')
            if clean_title and clean_title != song_name:
                candidate_queries.append(f'{song_name} "{target_artist}"')

        base_search = (
            f"{song_name} {target_artist}".strip()
            if target_artist and target_artist not in song_name
            else song_name.strip()
        )
        if base_search not in candidate_queries:
            candidate_queries.append(base_search)

        # Pass 2: Rút gọn hậu tố trong ngoặc như (Trung Thu EDM Version) và lược bỏ nghệ sĩ phụ sau dấu phẩy/ft
        primary_artist = ""
        if target_artist:
            raw_first = re.split(
                r"[,/&|]|(?:\s+ft\.?\s+)|\s+feat\.?\s+|\s+x\s+",
                target_artist,
                flags=re.IGNORECASE,
            )[0].strip()
            primary_artist = raw_first

        if clean_title and (
            clean_title != song_name or (primary_artist and primary_artist != target_artist)
        ):
            simplified = f"{clean_title} {primary_artist}".strip()
            if simplified and simplified not in candidate_queries:
                candidate_queries.append(simplified)

        # Pass 3: Chỉ tìm bằng clean_title nếu có chỉ định nghệ sĩ nhưng các pass trước có thể bị nhiễu
        if clean_title and clean_title not in candidate_queries:
            candidate_queries.append(clean_title)

        for query_str in candidate_queries:
            search_query = f"ytsearch5:{query_str}"
            try:
                with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                    info = ydl.extract_info(search_query, download=False)
                    if isinstance(info, dict) and "entries" in info:
                        entries: list[dict[str, Any]] = [
                            cast(dict[str, Any], e)
                            for e in info.get("entries", [])
                            if isinstance(e, dict)
                        ]
                        best_candidate: dict[str, Any] | None = None
                        best_score = -1.0

                        for candidate in entries:
                            is_match, score = self.validate_match(
                                target_title=song_name,
                                target_artist=target_artist,
                                candidate=candidate,
                                target_duration=target_duration,
                            )
                            if not is_match and primary_artist and primary_artist != target_artist:
                                is_p_match, p_score = self.validate_match(
                                    target_title=song_name,
                                    target_artist=primary_artist,
                                    candidate=candidate,
                                    target_duration=target_duration,
                                )
                                if is_p_match and p_score > score:
                                    is_match, score = is_p_match, p_score

                            if is_match and score > best_score:
                                best_score = score
                                best_candidate = candidate

                        if best_candidate:
                            cand_title = best_candidate.get("title")
                            cand_url = best_candidate.get("url") or best_candidate.get(
                                "webpage_url"
                            )
                            cand_dur = best_candidate.get("duration")
                            return cand_url, cand_title, cand_dur, True, best_score

            except Exception as e:  # noqa: BLE001
                print(f"[InputResolverService] Lỗi tìm kiếm YouTube cho '{query_str}': {e}")

        # Fallback cuối cùng nếu cả các Pass đều không tìm thấy verified match:
        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(f"ytsearch3:{clean_title or base_search}", download=False)
                if isinstance(info, dict) and "entries" in info:
                    entries = [
                        cast(dict[str, Any], e)
                        for e in info.get("entries", [])
                        if isinstance(e, dict)
                    ]
                    valid_entries = [
                        e
                        for e in entries
                        if "upcoming" not in str(e.get("live_status") or "").lower()
                        and not e.get("is_live")
                    ]
                    chosen = (
                        valid_entries[0] if valid_entries else (entries[0] if entries else None)
                    )
                    if chosen:
                        first_title = chosen.get("title")
                        first_url = chosen.get("url") or chosen.get("webpage_url")
                        first_dur = chosen.get("duration")
                        is_match, score = self.validate_match(
                            song_name, target_artist, chosen, target_duration
                        )
                        if is_match and score >= 0.65:
                            return first_url, first_title, first_dur, True, score
                        return None, song_name, None, False, score
        except Exception:  # noqa: BLE001, S110
            pass

        return None, song_name, None, False, 0.0

    def get_url_metadata(self, url: str) -> tuple[str | None, float | None]:
        """Trích xuất tiêu đề và thời lượng của một URL qua yt-dlp với OEmbed fallback."""
        cookie_file = get_cookie_file_path(mode="full") or get_cookie_file_path(mode="guest")
        js_runtimes_cfg = get_js_runtimes_config()

        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }
        if "youtube.com" in url or "youtu.be" in url:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": ["android", "web_embedded"]}}
        if cookie_file and cookie_file.exists():
            ydl_opts["cookiefile"] = str(cookie_file)
        if js_runtimes_cfg:
            ydl_opts["js_runtimes"] = js_runtimes_cfg
        yt_proxy = get_youtube_proxy()
        if yt_proxy:
            ydl_opts["proxy"] = yt_proxy

        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(url, download=False)
                if isinstance(info, dict):
                    if "entries" in info:
                        first = _extract_first_entry(info.get("entries"))
                        if first:
                            info = first
                    title = info.get("title")
                    dur = info.get("duration")
                    if title:
                        return title, dur
        except Exception:  # noqa: BLE001, S110
            pass

        # Fallback YouTube OEmbed API chính thức (Không bao giờ bị bot block, không cần cookie, phản hồi 0.05s)
        if "youtube.com" in url or "youtu.be" in url:
            try:
                oembed_url = (
                    f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
                )
                req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=4) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    oe_title = data.get("title")
                    if oe_title:
                        return oe_title, None
            except Exception:
                pass

        return None, None

    # Alias tương thích ngược
    _get_url_metadata = get_url_metadata

    def parse_batch_file(self, file_path: Path) -> list[TrackItem]:
        """Đọc file đầu vào (.txt, .csv, .xlsx) và trả về danh sách các TrackItem."""
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            with open(file_path, encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            tracks: list[TrackItem] = []
            for q in queries:
                input_type = self.detect_input_type(q)
                resolved_u = q if input_type in (InputType.URL, InputType.LOCAL_FILE) else None
                item_id = generate_deterministic_track_id(q, input_type, resolved_url=resolved_u)
                tracks.append(
                    TrackItem(
                        id=item_id,
                        raw_query=q,
                        input_type=input_type,
                        track_title=Path(q).stem if input_type == InputType.LOCAL_FILE else q,
                        resolved_url=resolved_u,
                    )
                )
            return tracks

        elif suffix == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except Exception:  # noqa: BLE001
                df = pd.read_csv(file_path, encoding="latin-1")
            return self._extract_tracks_from_df(df)

        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            return self._extract_tracks_from_df(df)

        else:
            raise InvalidInputException(
                f"Định dạng tệp không được hỗ trợ: {suffix}. Vui lòng dùng .txt, .csv, hoặc .xlsx"
            )

    def _extract_tracks_from_df(self, df: pd.DataFrame) -> list[TrackItem]:
        """Trích xuất danh sách TrackItem từ DataFrame (tự động nhận diện Title, Artist, URL, Song Type, Distributor, Release Date)."""
        url_cols = []
        title_cols = []
        artist_cols = []
        song_type_cols = []
        distributor_cols = []
        release_date_cols = []
        duration_cols = []

        for col in df.columns:
            c_low = str(col).strip().lower()
            if any(
                k in c_low
                for k in [
                    "url",
                    "link",
                    "đường dẫn",
                    "liên kết",
                    "youtube",
                    "source_url",
                    "webpage",
                ]
            ):
                url_cols.append(col)
            elif any(k in c_low for k in ["tên bài", "tiêu đề", "song title", "track", "title"]):
                title_cols.append(col)
            elif any(k in c_low for k in ["ca sĩ", "nghệ sĩ", "artist", "singer"]):
                artist_cols.append(col)
            elif any(
                k in c_low
                for k in [
                    "song type",
                    "song_type",
                    "loại bài",
                    "loại tác phẩm",
                    "work_type",
                    "work type",
                    "type",
                ]
            ):
                song_type_cols.append(col)
            elif any(
                k in c_low
                for k in [
                    "distributor",
                    "nhà phân phối",
                    "phân phối",
                    "publisher",
                    "label",
                    "nhãn",
                ]
            ):
                distributor_cols.append(col)
            elif any(
                k in c_low
                for k in [
                    "release_date",
                    "release date",
                    "ngày phát hành",
                    "ngay phat hanh",
                    "publish_date",
                    "upload_date",
                    "release",
                ]
            ):
                release_date_cols.append(col)
            elif any(
                k in c_low
                for k in [
                    "duration",
                    "thời lượng",
                    "thoi luong",
                    "length",
                    "thời gian",
                    "duration_sec",
                ]
            ):
                duration_cols.append(col)

        if not title_cols and not url_cols:
            queries = self._extract_queries_from_df(df)
            tracks: list[TrackItem] = []
            for q in queries:
                input_type = self.detect_input_type(q)
                resolved_u = q if input_type in (InputType.URL, InputType.LOCAL_FILE) else None
                item_id = generate_deterministic_track_id(q, input_type, resolved_url=resolved_u)
                tracks.append(
                    TrackItem(
                        id=item_id,
                        raw_query=q,
                        input_type=input_type,
                        track_title=Path(q).stem if input_type == InputType.LOCAL_FILE else q,
                        resolved_url=resolved_u,
                    )
                )
            return tracks

        tracks = []
        for _, row in df.iterrows():
            raw_u = ""
            for c in url_cols:
                if c in row and pd.notna(row[c]):
                    v = str(row[c]).strip()
                    if v and v.lower() not in ("none", "nan", "null", "n/a"):
                        raw_u = v
                        break

            raw_t = ""
            for c in title_cols:
                if c in row and pd.notna(row[c]):
                    v = str(row[c]).strip()
                    if v and v.lower() not in ("none", "nan", "null", "n/a"):
                        raw_t = v
                        break

            raw_a = ""
            for c in artist_cols:
                if c in row and pd.notna(row[c]):
                    v = str(row[c]).strip()
                    if v and v.lower() not in ("none", "nan", "null", "n/a"):
                        raw_a = v
                        break

            raw_st = ""
            for c in song_type_cols:
                if c in row and pd.notna(row[c]):
                    v = str(row[c]).strip()
                    if v and v.lower() not in ("none", "nan", "null", "n/a"):
                        raw_st = v
                        break

            raw_dist = ""
            for c in distributor_cols:
                if c in row and pd.notna(row[c]):
                    v = str(row[c]).strip()
                    if v and v.lower() not in ("none", "nan", "null", "n/a"):
                        raw_dist = v
                        break

            raw_rd = None
            for c in release_date_cols:
                if c in row and pd.notna(row[c]):
                    v = _normalize_date_str(row[c])
                    if v:
                        raw_rd = v
                        break

            raw_dur = None
            for c in duration_cols:
                if c in row and pd.notna(row[c]):
                    try:
                        raw_dur = float(row[c])
                        break
                    except (ValueError, TypeError):
                        pass

            if not raw_u and not raw_t:
                continue

            resolved_u = (
                raw_u
                if (
                    raw_u.startswith("http://")
                    or raw_u.startswith("https://")
                    or Path(raw_u).exists()
                )
                else None
            )
            effective_title = raw_t or (Path(raw_u).stem if raw_u else "")
            q = raw_u if resolved_u else (f"{raw_t} {raw_a}".strip() if raw_a else raw_t)
            input_type = self.detect_input_type(q)
            final_resolved_u = resolved_u or (
                q if input_type in (InputType.URL, InputType.LOCAL_FILE) else None
            )
            item_id = generate_deterministic_track_id(q, input_type, resolved_url=final_resolved_u)

            tracks.append(
                TrackItem(
                    id=item_id,
                    raw_query=q,
                    input_type=input_type,
                    track_title=effective_title,
                    artist=raw_a or None,
                    resolved_url=final_resolved_u,
                    song_type=raw_st or None,
                    distributor=raw_dist or None,
                    release_date=raw_rd,
                    duration_sec=raw_dur,
                )
            )
        return tracks

    def _extract_queries_from_df(self, df: pd.DataFrame) -> list[str]:
        """Tự động tìm kiếm cột chứa link hoặc tên bài hát từ DataFrame."""
        header_keywords = [
            "tên bài hát",
            "bài hát",
            "đường dẫn",
            "liên kết",
            "tiêu đề",
            "track",
            "query",
            "url",
            "link",
            "song",
            "title",
            "name",
        ]
        chosen_col = None
        has_named_header = False

        for col in df.columns:
            clean_name = col.strip().lower() if isinstance(col, str) else str(col).strip().lower()
            if (
                URL_REGEX.match(clean_name)
                or clean_name.startswith("http")
                or "youtu" in clean_name
            ):
                continue

            if clean_name in header_keywords or any(
                kw in clean_name for kw in ["bài hát", "tiêu đề", "đường dẫn", "liên kết", "track"]
            ):
                chosen_col = col
                has_named_header = True
                break

        if chosen_col is None:
            chosen_col = df.columns[0]

        items = [str(val).strip() for val in df[chosen_col].dropna() if str(val).strip()]

        col_str = chosen_col.strip() if isinstance(chosen_col, str) else str(chosen_col).strip()
        if not has_named_header and (
            URL_REGEX.match(col_str)
            or col_str.startswith("http")
            or "youtu" in col_str
            or (
                not col_str.lower().startswith("unnamed:")
                and col_str.lower() not in header_keywords
            )
        ):
            items.insert(0, col_str)

        return items
