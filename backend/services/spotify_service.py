import base64
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any

from ..core.config import SPOTIFY_BATCH_WORKERS, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
from ..core.interfaces.services.spotify_service_interface import ISpotifyService
from ..core.utils import normalize_vietnamese_text

logger = logging.getLogger(__name__)

SPOTIFY_TRACK_ID_REGEX = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/track/|open\.spotify\.com/intl-[a-z]+/track/)([a-zA-Z0-9]+)",
    re.IGNORECASE,
)


def _clean_str(text: str) -> str:
    """Loại bỏ dấu tiếng Việt và ký tự đặc biệt để so khớp chuỗi chuẩn xác."""
    return normalize_vietnamese_text(text, remove_special_chars=True).lower()


class SpotifyService(ISpotifyService):
    """
    Hiện thực ISpotifyService kết nối Spotify Web API (Client Credentials Flow)
    kèm cơ chế Hybrid dự phòng tự động (Fallback to Public / Web Search) nếu chưa cấu hình API Keys.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout_seconds: float = 4.0,
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
        self, query: str, artist: str = "", limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm danh sách bài hát trên Spotify.
        Nếu có token API: Gọi Spotify Search API.
        Nếu không có token / lỗi mạng: Trả về mục fallback có link mở Spotify Web Search.
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

        token = self._get_access_token()
        if token and not self._premium_required and not self._quota_exceeded:
            try:
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
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    items = (data.get("tracks") or {}).get("items") or []
                    results = [self._parse_track_item(it) for it in items if it]
                    if results:
                        return results
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if e.code == 429 or "quota_exceeded" in err_body.lower() or "too many requests" in err_body.lower():
                    self._quota_exceeded = True
                    if not self._quota_warning_logged:
                        self._quota_warning_logged = True
                        logger.warning(
                            "⚠️ [SpotifyService] Spotify API thông báo hết hạn mức (HTTP 429: QUOTA_EXCEEDED). "
                            "Hệ thống tự động kích hoạt Circuit Breaker: Tạm dừng gọi Spotify API trong phiên này "
                            "để tối ưu tốc độ và chuyển sang chế độ Search Link Fallback."
                        )
                elif e.code == 403 and "premium" in err_body.lower():
                    self._premium_required = True
                    if not self._premium_warning_logged:
                        self._premium_warning_logged = True
                        logger.info(
                            "💡 [SpotifyService] Tài khoản chủ sở hữu App là bản Free. "
                            "Spotify yêu cầu gói Premium để gọi Web API danh mục bài hát. "
                            "Hệ thống đã tự động chuyển sang chế độ Search Link Fallback."
                        )
                else:
                    logger.warning("⚠️ [SpotifyService] Lỗi khi gọi Spotify Search API: %s (%s)", e, err_body.strip())
            except Exception as e:
                logger.warning("⚠️ [SpotifyService] Lỗi khi gọi Spotify Search API: %s", e)

        # Fallback an toàn: Trả về mục tìm kiếm web để người dùng có thể nhấp mở trực tiếp
        encoded_web = urllib.parse.quote(search_term)
        return [
            {
                "id": "search_fallback",
                "title": clean_q,
                "artist": artist or "",
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
        token = self._get_access_token()
        if token and track_id and not self._quota_exceeded:
            try:
                api_url = f"https://api.spotify.com/v1/tracks/{track_id}"
                req = urllib.request.Request(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "MusicIdentify-SpotifyService/1.0",
                    },
                )
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

    def find_best_match(self, title: str, artist: str = "") -> dict[str, Any] | None:
        """
        Tìm kiếm và xếp hạng kết quả để chọn ra bài hát Spotify khớp nhất.
        Sử dụng SequenceMatcher tính toán độ tương đồng giữa (Title, Artist) mục tiêu
        và các ứng viên trả về từ Spotify.
        """
        clean_title = title.strip()
        if not clean_title:
            return None

        candidates = self.search_tracks(query=clean_title, artist=artist, limit=5)
        if not candidates:
            return None

        # Nếu chỉ có 1 ứng viên fallback
        if len(candidates) == 1 and not candidates[0].get("is_official_api"):
            return candidates[0]

        target_title_norm = _clean_str(clean_title)
        target_artist_norm = _clean_str(artist) if artist else ""

        best_score = -1.0
        best_cand: dict[str, Any] | None = None

        for cand in candidates:
            cand_title_norm = _clean_str(cand.get("title", ""))
            cand_artist_norm = _clean_str(cand.get("artist", ""))

            # So khớp tên bài hát
            title_score = SequenceMatcher(None, target_title_norm, cand_title_norm).ratio()
            # Khớp chuỗi con (nếu tên nằm trọn trong nhau)
            if target_title_norm in cand_title_norm or cand_title_norm in target_title_norm:
                title_score = max(title_score, 0.85)

            # So khớp nghệ sĩ nếu có
            if target_artist_norm:
                artist_score = SequenceMatcher(
                    None, target_artist_norm, cand_artist_norm
                ).ratio()
                if (
                    target_artist_norm in cand_artist_norm
                    or cand_artist_norm in target_artist_norm
                ):
                    artist_score = max(artist_score, 0.85)
                combined_score = title_score * 0.65 + artist_score * 0.35
            else:
                combined_score = title_score

            # Ưu tiên cộng điểm nếu có độ phổ biến cao
            popularity = cand.get("popularity", 0)
            combined_score += (popularity / 100.0) * 0.05

            if combined_score > best_score:
                best_score = combined_score
                best_cand = cand

        # Nếu điểm so khớp đạt ngưỡng tối thiểu >= 0.40
        if best_cand and best_score >= 0.40:
            best_cand["match_score"] = round(best_score, 2)
            return best_cand

        return candidates[0] if candidates else None

    def batch_search_tracks(
        self,
        queries: list[dict[str, Any] | str],
        max_workers: int = SPOTIFY_BATCH_WORKERS,
        on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tra cứu hàng loạt danh sách bài hát trên Spotify bằng Multi-threading Pool.
        Bảo toàn 100% thứ tự bài hát ban đầu và cập nhật tiến trình liên tục.
        """
        if not queries:
            return []

        # Chuẩn hóa danh sách đầu vào
        normalized_inputs: list[tuple[int, str, str, dict[str, Any]]] = []
        for idx, item in enumerate(queries):
            raw_meta: dict[str, Any] = {}
            if isinstance(item, dict):
                q = str(item.get("query") or item.get("title") or item.get("track_title") or "").strip()
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

            is_found = bool(
                match_res
                and match_res.get("spotify_url")
                and not match_res.get("id", "").startswith("search_fallback")
            )
            is_search_link = bool(
                match_res
                and match_res.get("spotify_url")
                and match_res.get("id", "").startswith("search_fallback")
            )

            res_record: dict[str, Any] = {
                "index": i + 1,
                "input_query": query_val,
                "input_artist": artist_val,
                "spotify_id": match_res.get("id") if match_res else None,
                "spotify_title": match_res.get("title") if match_res else None,
                "spotify_artist": match_res.get("artist") if match_res else None,
                "spotify_album": match_res.get("album") if match_res else None,
                "spotify_release_date": match_res.get("release_date") if match_res else None,
                "spotify_url": match_res.get("spotify_url") if match_res else None,
                "spotify_isrc": match_res.get("isrc") if match_res else None,
                "match_score": match_res.get("match_score", 1.0 if is_found else 0.0) if match_res else 0.0,
                "cover_url": match_res.get("cover_url") if match_res else None,
                "duration_sec": match_res.get("duration_sec", 0.0) if match_res else 0.0,
                "popularity": match_res.get("popularity", 0) if match_res else 0,
                "embed_url": match_res.get("embed_url") if match_res else None,
                "is_found": is_found,
                "is_search_link": is_search_link,
                "raw_input_meta": raw_m,
            }
            return i, res_record

        # Thực thi đa luồng với ThreadPoolExecutor
        workers = max(1, min(max_workers, 16))
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
