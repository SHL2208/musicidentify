import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import yt_dlp

from ..core.config import (
    AUDIO_SAMPLE_DURATION,
    AUDIO_SAMPLE_OFFSET,
    ENABLE_RANGE_STREAM_DOWNLOAD,
    get_all_cookie_file_paths,
    get_cookie_file_path,
    get_ffmpeg_binary,
    get_js_runtimes_config,
    get_youtube_proxy,
)
from ..core.exceptions import AudioProcessingException
from ..core.interfaces.repositories.storage_repository_interface import IStorageRepository
from ..core.interfaces.services.audio_engine_interface import IAudioEngine
from ..core.models.entities import InputType, TrackItem
from ..core.utils import extract_first_entry as _extract_first_entry


def _safe_log(msg: str) -> None:
    """In log an toàn tránh UnicodeEncodeError trên môi trường Windows console cp1252."""
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass


def clean_music_search_queries(title: str, artist: str = "") -> list[str]:
    """
    Làm sạch các từ khóa boilerplate YouTube và bóc tách phân đoạn thông minh
    để tìm kiếm chính xác 100% trên SoundCloud / Zing MP3 khi YouTube bị lỗi chặn.
    """
    if not title:
        return [artist] if artist else []

    t = re.sub(r"@[A-Za-z0-9_\.-]+", " ", title)
    t = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", " ", t)
    t = re.sub(
        r"\b(official\s*(audio|music\s*video|mv|video|lyric\s*video|visualizer)?|mv|video\s*lyric|audio|lyrics?)\b",
        " ",
        t,
        flags=re.I,
    )
    # Tách theo các loại gạch nối unicode (–, —, ―) và phân cách thường gặp (| / :)
    # Dấu '-' chỉ tách khi có khoảng trắng xung quanh (ví dụ 'A - B') để bảo vệ tên ghép (M-TP, T-Ara)
    segments = re.split(
        r"\s*[\u2013\u2014\u2015\|/:]\s*|\s+-\s+|\s+(?:ft\.?|feat\.?|featuring)\s+", t, flags=re.I
    )
    segments = [" ".join(s.split()).strip() for s in segments if s.strip()]

    queries: list[str] = []
    # 1. Phân đoạn đầu tiên (trong 90% video âm nhạc, đây chính là Tên bài hát thực tế)
    if segments:
        s0 = segments[0]
        s0_clean = re.sub(r"[\"\'~#,\+\.]", " ", s0).strip()
        if len(s0_clean) >= 2:
            queries.append(s0_clean)
            if artist and artist.lower() not in s0_clean.lower():
                queries.insert(0, f"{s0_clean} {artist}".strip())

    # 2. Toàn bộ chuỗi đã loại bỏ ký tự phân cách
    full_clean = " ".join(re.sub(r"[\u2013\u2014\u2015\|/:~#,\+\.]", " ", t).split()).strip()
    if full_clean and full_clean not in queries:
        queries.append(full_clean)
        if artist and artist.lower() not in full_clean.lower():
            queries.insert(0, f"{full_clean} {artist}".strip())

    # 3. Phân đoạn thứ hai (trong trường hợp cấu trúc video là 'Nghệ sĩ - Tên bài hát')
    if len(segments) > 1:
        s1 = segments[1]
        s1_clean = re.sub(r"[\"\'~#,\+\.]", " ", s1).strip()
        if len(s1_clean) >= 2 and len(s1_clean.split()) <= 6 and s1_clean not in queries:
            queries.append(s1_clean)

    return list(dict.fromkeys(queries))


class AudioEngineService(IAudioEngine):
    """Hiện thực dịch vụ xử lý và trích xuất âm thanh, tuân thủ interface IAudioEngine."""

    def __init__(self, storage_repo: IStorageRepository, ffmpeg_path: str | None = None):
        self.storage_repo = storage_repo
        self.temp_dir = self.storage_repo.get_temp_dir()
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_binary()

    def process_track_audio(self, track: TrackItem) -> tuple[Path, dict[str, Any] | None]:
        """
        Tải và trích xuất mẫu âm thanh đa phân đoạn (Multi-Segment) đại diện cho toàn bài hát.
        Trả về: (đường dẫn file .wav đa phân đoạn, metadata chi tiết)
        """
        raw_audio_path: Path | None = None
        metadata: dict[str, Any] = {}

        try:
            if track.input_type == InputType.LOCAL_FILE and os.path.isfile(track.raw_query):
                raw_audio_path = Path(track.raw_query)
                metadata = {
                    "title": Path(track.raw_query).stem,
                    "webpage_url": track.raw_query,
                    "url": track.raw_query,
                }
            else:
                url_to_download = track.resolved_url or (
                    track.raw_query if track.input_type == InputType.URL else None
                )
                if not url_to_download and track.input_type not in (
                    InputType.LOCAL_FILE,
                    InputType.URL,
                ):
                    from ..core.container import get_container

                    res_svc = get_container().resolver
                    try:
                        resolved_track = res_svc.resolve_track(
                            track.raw_query, artist=track.artist or ""
                        )
                    except TypeError:
                        resolved_track = res_svc.resolve_track(track.raw_query)

                    if resolved_track and getattr(resolved_track, "resolved_url", None):
                        res_url = resolved_track.resolved_url
                        if res_url and not res_url.startswith("<MagicMock"):
                            url_to_download = res_url
                            track.resolved_url = res_url
                        if getattr(resolved_track, "track_title", None) and isinstance(
                            resolved_track.track_title, str
                        ):
                            track.track_title = resolved_track.track_title
                        if getattr(resolved_track, "artist", None) and isinstance(
                            resolved_track.artist, str
                        ):
                            track.artist = resolved_track.artist
                        if getattr(resolved_track, "is_verified_match", None) is not None:
                            track.is_verified_match = bool(resolved_track.is_verified_match)
                        if getattr(resolved_track, "match_score", None) is not None and isinstance(
                            resolved_track.match_score, (int, float)
                        ):
                            track.match_score = float(resolved_track.match_score)
                        if getattr(resolved_track, "duration_sec", None) is not None and isinstance(
                            resolved_track.duration_sec, (int, float)
                        ):
                            track.duration_sec = float(resolved_track.duration_sec)
                        if getattr(resolved_track, "platform", None) and isinstance(
                            resolved_track.platform, str
                        ):
                            track.platform = resolved_track.platform

                if not url_to_download:
                    raise AudioProcessingException(
                        f"Không tìm thấy URL hợp lệ để tải bài hát '{track.track_title or track.raw_query}'"
                    )

                raw_audio_path = None
                metadata = {}
                err_str = ""
                try:
                    raw_audio_path, metadata = self._download_audio(url_to_download, track.id)
                except Exception as dl_err:
                    err_str = str(dl_err).lower()
                    target_t = track.track_title or track.raw_query
                    target_a = track.artist or ""

                    # Bóc tách tên bài sạch nếu target_t đang là URL
                    search_title = target_t
                    if (
                        not search_title
                        or search_title.startswith("http")
                        or "youtu" in search_title
                        or "zing" in search_title
                    ):
                        from ..core.container import get_container

                        res_svc = get_container().resolver
                        meta_title, _ = res_svc.get_url_metadata(url_to_download)
                        if meta_title and not meta_title.startswith("http"):
                            search_title = meta_title
                            track.track_title = search_title

                    # Nhánh 1: Tự động chuyển đổi sang YouTube nếu tải từ nguồn gốc thất bại (Zing MP3 VIP/403/429 hoặc lỗi stream)
                    if not raw_audio_path and (
                        "youtube.com" not in url_to_download and "youtu.be" not in url_to_download
                    ):
                        _safe_log(
                            f"[AudioEngineService] [FALLBACK] Tải URL gốc thất bại ({url_to_download}: {dl_err}). Đang tìm kiếm và chuyển đổi sang YouTube..."
                        )
                        from ..core.container import get_container

                        res_svc = get_container().resolver

                        # Bước 1.1: Tìm kiếm qua Smart Matcher trên YouTube
                        yt_url, yt_title, _, is_match, score = res_svc.search_youtube(
                            search_title or target_t, target_artist=target_a
                        )
                        if yt_url and (is_match or score >= 0.40):
                            try:
                                _safe_log(
                                    f"[AudioEngineService] [FALLBACK] Đang tải bản thu YouTube thay thế: {yt_url} (Score: {score:.2f})"
                                )
                                raw_audio_path, metadata = self._download_audio(yt_url, track.id)
                                if raw_audio_path:
                                    metadata["is_fallback_audio"] = True
                                    metadata["fallback_from_url"] = url_to_download
                                    metadata["fallback_platform"] = "youtube"
                                    metadata["webpage_url"] = yt_url
                                    track.resolved_url = yt_url
                                    if yt_title and not track.track_title:
                                        track.track_title = yt_title
                            except Exception as yt_err:
                                _safe_log(
                                    f"[AudioEngineService] [FALLBACK] Tải video YouTube ({yt_url}) gặp lỗi: {yt_err}"
                                )
                                raw_audio_path, metadata = None, {}

                        # Bước 1.2: Nếu search_youtube chưa tải được, thử các câu truy vấn làm sạch qua ytsearch3
                        if (
                            not raw_audio_path
                            and search_title
                            and not search_title.startswith("http")
                        ):
                            candidate_queries = clean_music_search_queries(search_title, target_a)
                            for q in candidate_queries:
                                try:
                                    yt_search_query = f"ytsearch3:{q}"
                                    _safe_log(
                                        f"[AudioEngineService] [FALLBACK] Thử tìm kiếm YouTube với từ khóa: {yt_search_query}"
                                    )
                                    raw_audio_path, metadata = self._download_audio(
                                        yt_search_query, track.id
                                    )
                                    if raw_audio_path:
                                        metadata["is_fallback_audio"] = True
                                        metadata["fallback_from_url"] = url_to_download
                                        metadata["fallback_platform"] = "youtube_clean_search"
                                        if metadata.get("webpage_url"):
                                            track.resolved_url = metadata["webpage_url"]
                                        break
                                except Exception as yt_q_err:
                                    _safe_log(
                                        f"[AudioEngineService] [FAIL] YouTube search query thất bại ({q}): {yt_q_err}"
                                    )

                    # Nhánh 2: Nếu chưa tải được (do YouTube bị chặn Datacenter/bot/format trên Cloud),
                    # Tự động Fallback sang SoundCloud (hoàn toàn mở, không bị bot block trên Cloud Datacenter)
                    if not raw_audio_path:
                        search_title = track.track_title or ""
                        if (
                            not search_title
                            or search_title.startswith("http")
                            or "youtu" in search_title
                        ):
                            from ..core.container import get_container

                            res_svc = get_container().resolver
                            meta_title, _ = res_svc.get_url_metadata(url_to_download)
                            if meta_title and not meta_title.startswith("http"):
                                search_title = meta_title

                        if search_title and not search_title.startswith("http"):
                            candidate_queries = clean_music_search_queries(search_title, target_a)
                            for q in candidate_queries:
                                try:
                                    sc_search = f"scsearch1:{q}"
                                    _safe_log(
                                        f"[AudioEngineService] [FALLBACK] YouTube tải thất bại. Đang thử tải từ SoundCloud: {sc_search}"
                                    )
                                    raw_audio_path, metadata = self._download_audio(
                                        sc_search, track.id
                                    )
                                    if raw_audio_path:
                                        metadata["is_fallback_audio"] = True
                                        metadata["fallback_from_url"] = url_to_download
                                        metadata["fallback_platform"] = "soundcloud"
                                        break
                                except Exception as sc_err:
                                    _safe_log(
                                        f"[AudioEngineService] [FAIL] SoundCloud fallback thất bại ({q}): {sc_err}"
                                    )

                    # Nhánh 3: Nếu SoundCloud cũng không có, thử tìm kiếm Zing MP3
                    if not raw_audio_path:
                        search_title = track.track_title or ""
                        if (
                            not search_title
                            or search_title.startswith("http")
                            or "youtu" in search_title
                        ):
                            from ..core.container import get_container

                            res_svc = get_container().resolver
                            meta_title, _ = res_svc.get_url_metadata(url_to_download)
                            if meta_title and not meta_title.startswith("http"):
                                search_title = meta_title

                        if search_title and not search_title.startswith("http"):
                            candidate_queries = clean_music_search_queries(search_title, target_a)
                            for q in candidate_queries:
                                try:
                                    from ..core.container import get_container

                                    res_svc = get_container().resolver
                                    zing_meta = res_svc.search_zingmp3(q, target_artist=target_a)
                                    if zing_meta and zing_meta.get("stream_url"):
                                        _safe_log(
                                            f"[AudioEngineService] [FALLBACK] Đang thử tải từ Zing MP3: {q}"
                                        )
                                        raw_audio_path, metadata = self._download_audio(
                                            zing_meta["stream_url"], track.id
                                        )
                                        if raw_audio_path and raw_audio_path.exists():
                                            metadata["is_fallback_audio"] = True
                                            metadata["fallback_from_url"] = url_to_download
                                            metadata["fallback_platform"] = "zingmp3"
                                            break
                                except Exception as zing_err:
                                    _safe_log(
                                        f"[AudioEngineService] [FAIL] Zing MP3 fallback thất bại ({q}): {zing_err}"
                                    )
            if not raw_audio_path:
                if "vip" in err_str or "zingmp3" in url_to_download.lower():
                    raise AudioProcessingException(
                        f"Bài hát '{track.track_title or track.raw_query}' trên Zing MP3 bị khóa VIP và không tìm thấy bản phát hành chính chủ trên YouTube."
                    )
                raise AudioProcessingException(
                    f"Không thể tải tệp âm thanh hợp lệ cho bài '{track.track_title or track.raw_query}'"
                )

            # Trích xuất mẫu đa phân đoạn (Verse 20%, Chorus 50%, Climax 75%)
            sample_path = self._slice_multisegment_sample(
                raw_audio_path, track.id, track.duration_sec
            )

            # Xóa file audio gốc tải về nếu là file tạm
            if (
                raw_audio_path != sample_path
                and raw_audio_path.exists()
                and raw_audio_path.is_relative_to(self.temp_dir)
            ):
                self.storage_repo.cleanup_file(raw_audio_path)

            return sample_path, metadata

        except Exception as e:
            if isinstance(e, AudioProcessingException):
                raise
            raise AudioProcessingException(
                f"Lỗi khi xử lý âm thanh bài '{track.track_title or track.raw_query}': {e}"
            ) from e

    def _try_stream_slice_download(self, url: str, track_id: str) -> tuple[Path, dict] | None:
        """
        Tối ưu siêu tốc (HTTP Range Request):
        Trích xuất direct stream URL từ yt-dlp mà không tải toàn bộ file video (download=False),
        sau đó dùng FFmpeg stream slice trực tiếp 45 giây. Giảm tải dữ liệu từ ~10MB xuống ~800KB.
        Nếu gặp bất kỳ sự cố nào (403, cipher, timeout), tự động trả về None để fallback sang tải thông thường.
        """
        if not ENABLE_RANGE_STREAM_DOWNLOAD:
            return None

        if not (
            isinstance(url, str)
            and url.startswith("http")
            and ("youtube.com" in url or "youtu.be" in url)
        ):
            return None

        out_path = self.temp_dir / f"{track_id}_raw.mp3"
        js_runtimes_cfg = get_js_runtimes_config()
        full_cookie_paths = get_all_cookie_file_paths(mode="full")
        cookie_file = str(full_cookie_paths[0]) if full_cookie_paths else None

        ydl_opts: dict[str, Any] = {
            "format": "bestaudio/ba/b[acodec!=none]/best[acodec!=none]",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 8,
            "nocheckcertificate": True,
        }
        if cookie_file and Path(cookie_file).exists():
            ydl_opts["cookiefile"] = cookie_file
        if js_runtimes_cfg:
            ydl_opts["js_runtimes"] = js_runtimes_cfg
        yt_proxy = get_youtube_proxy()
        if yt_proxy:
            ydl_opts["proxy"] = yt_proxy

        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(url, download=False)
                if not isinstance(info, dict):
                    return None
                if "entries" in info:
                    entries = list(info.get("entries") or [])
                    if entries:
                        info = entries[0]
                info_dict = dict(info) if isinstance(info, dict) else {}
                stream_url = info_dict.get("url")
                if not stream_url and "formats" in info_dict:
                    audio_fmts = [
                        f
                        for f in info_dict["formats"]
                        if f.get("url") and f.get("acodec") != "none"
                    ]
                    if audio_fmts:
                        audio_fmts.sort(
                            key=lambda x: x.get("abr") or x.get("tbr") or 0, reverse=True
                        )
                        stream_url = audio_fmts[0]["url"]

                if not stream_url or not str(stream_url).startswith("http"):
                    return None

                dur = float(info_dict.get("duration") or 0.0)
                if dur <= 0.0:
                    return None

                http_headers = info_dict.get("http_headers") or {}
                ua = http_headers.get("User-Agent")

                if dur <= 45.0:
                    cmd = [
                        self.ffmpeg_path,
                        "-y",
                    ]
                    if yt_proxy:
                        cmd.extend(["-http_proxy", yt_proxy])
                    if ua:
                        cmd.extend(["-user_agent", ua])
                    cmd.extend(
                        [
                            "-i",
                            str(stream_url),
                            "-t",
                            f"{dur:.2f}",
                            "-vn",
                            "-acodec",
                            "libmp3lame",
                            "-ab",
                            "192k",
                            "-ar",
                            "44100",
                            "-ac",
                            "2",
                            str(out_path),
                        ]
                    )
                else:
                    slice_dur = 15.0
                    start_1 = max(0.0, min(dur * 0.20 - slice_dur / 2, dur - slice_dur))
                    start_2 = max(0.0, min(dur * 0.50 - slice_dur / 2, dur - slice_dur))
                    start_3 = max(0.0, min(dur * 0.75 - slice_dur / 2, dur - slice_dur))

                    filter_complex = (
                        f"[0:a]atrim=start={start_1:.2f}:end={start_1 + slice_dur:.2f},asetpts=PTS-STARTPTS[a1];"
                        f"[0:a]atrim=start={start_2:.2f}:end={start_2 + slice_dur:.2f},asetpts=PTS-STARTPTS[a2];"
                        f"[0:a]atrim=start={start_3:.2f}:end={start_3 + slice_dur:.2f},asetpts=PTS-STARTPTS[a3];"
                        "[a1][a2][a3]concat=n=3:v=0:a=1[out]"
                    )
                    cmd = [
                        self.ffmpeg_path,
                        "-y",
                    ]
                    if yt_proxy:
                        cmd.extend(["-http_proxy", yt_proxy])
                    if ua:
                        cmd.extend(["-user_agent", ua])
                    cmd.extend(
                        [
                            "-i",
                            str(stream_url),
                            "-filter_complex",
                            filter_complex,
                            "-map",
                            "[out]",
                            "-vn",
                            "-acodec",
                            "libmp3lame",
                            "-ab",
                            "192k",
                            "-ar",
                            "44100",
                            "-ac",
                            "2",
                            str(out_path),
                        ]
                    )

                _safe_log(
                    f"[AudioEngineService] [STREAM_SLICE] Thử stream slice trực tiếp 45s từ CDN ({url})..."
                )
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=25,
                )
                if res.returncode == 0 and out_path.exists() and out_path.stat().st_size >= 10240:
                    _safe_log(
                        f"[AudioEngineService] [STREAM_SLICE] ✓ Hoàn thành stream slice siêu tốc ({out_path.stat().st_size // 1024} KB)"
                    )
                    if not info_dict.get("webpage_url"):
                        info_dict["webpage_url"] = url
                    return out_path, info_dict
                else:
                    _safe_log(
                        f"[AudioEngineService] [STREAM_SLICE] FFmpeg stream slice không thành công (code {res.returncode}), fallback..."
                    )
                    if out_path.exists():
                        out_path.unlink()
                    return None
        except Exception as slice_err:
            _safe_log(
                f"[AudioEngineService] [STREAM_SLICE] Stream slice thất bại: {slice_err}. Tự động fallback tải toàn bộ."
            )
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            return None

    def _download_audio(self, url: str, track_id: str) -> tuple[Path, dict]:
        """Tải luồng âm thanh tốt nhất bằng yt-dlp với cơ chế phòng thủ lỗi đa tầng và chống bot/format/cookie mismatch."""
        # 0. Thử nghiệm tối ưu tải trực tiếp bằng Stream Range Slicing
        stream_slice_result = self._try_stream_slice_download(url, track_id)
        if stream_slice_result:
            return stream_slice_result

        out_template = str(self.temp_dir / f"{track_id}_raw.%(ext)s")

        # Nạp cookies từ cấu hình: tìm tất cả các file cookie khả dụng (cookies.txt, cookies_master.txt)
        full_cookie_paths = get_all_cookie_file_paths(mode="full")
        guest_cookie_path = get_cookie_file_path(mode="guest")
        guest_cookie_file = (
            str(guest_cookie_path) if (guest_cookie_path and guest_cookie_path.exists()) else None
        )

        js_runtimes_cfg = get_js_runtimes_config()

        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )

        is_youtube = ("youtube.com" in url) or ("youtu.be" in url)

        # Ma trận chiến lược tải đa tầng (Multi-tier Strategy Matrix)
        if is_youtube:
            strategies: list[dict[str, Any]] = []

            # 1. NẾU CÓ COOKIE TÀI KHOẢN (cookies.txt, cookies_master.txt...): ƯU TIÊN SỐ 1!
            # Tải ngay bằng cookie tài khoản để bypass bot guard qua web/creator clients hỗ trợ cookie
            for c_path in full_cookie_paths:
                suffix = f"_{c_path.stem}" if len(full_cookie_paths) > 1 else ""
                strategies.extend(
                    [
                        {
                            "name": f"web_full_cookies{suffix}",
                            "client": ["web_embedded", "web"],
                            "cookie_file": str(c_path),
                        },
                        {
                            "name": f"web_creator_full_cookies{suffix}",
                            "client": ["web_creator"],
                            "cookie_file": str(c_path),
                        },
                        {
                            "name": f"mweb_full_cookies{suffix}",
                            "client": ["mweb"],
                            "cookie_file": str(c_path),
                        },
                        {
                            "name": f"default_full_cookies{suffix}",
                            "client": None,
                            "cookie_file": str(c_path),
                        },
                    ]
                )

            # 2. Các chiến lược không cookie / mobile app (chạy khi không có cookie hoặc fallback)
            strategies.extend(
                [
                    # 1. TV client KHÔNG cookie: Vượt qua rào cản bot guard trên Datacenter tốt nhất (không truyền cookie vào TV client)
                    {
                        "name": "tv_no_cookies",
                        "client": ["tv"],
                        "cookie_file": None,
                    },
                    # 2. Android client KHÔNG cookie: Vượt qua rào cản PO token / SABR
                    {
                        "name": "android_no_cookies",
                        "client": ["android"],
                        "cookie_file": None,
                    },
                    # 2. VisionOS client KHÔNG cookie: Lấy stream Opus 251 chất lượng cao
                    {
                        "name": "visionos_no_cookies",
                        "client": ["visionos"],
                        "cookie_file": None,
                    },
                    # 3. Android + iOS kết hợp
                    {
                        "name": "android_ios_hybrid",
                        "client": ["android", "ios"],
                        "cookie_file": None,
                    },
                    # 4. iOS client KHÔNG cookie (dự phòng)
                    {
                        "name": "ios_no_cookies",
                        "client": ["ios"],
                        "cookie_file": None,
                    },
                    # 5. mweb client (YouTube Mobile Web)
                    {
                        "name": "mweb_no_cookies",
                        "client": ["mweb"],
                        "cookie_file": None,
                    },
                    # 6. Client mặc định yt-dlp KHÔNG cookie
                    {
                        "name": "default_no_cookies",
                        "client": None,
                        "cookie_file": None,
                    },
                    # 7. Client mặc định CÓ Guest Cookies (Visitor Info) nếu có cấu hình
                    {
                        "name": "default_guest_cookies",
                        "client": None,
                        "cookie_file": guest_cookie_file,
                    },
                ]
            )

            # Loại bỏ các chiến lược có cookie nếu không có file cookie tương ứng
            strategies = [
                s
                for s in strategies
                if not (
                    s.get("cookie_file") is None
                    and "cookies" in s["name"]
                    and not s["name"].endswith("no_cookies")
                )
            ]
        else:
            # Đối với SoundCloud, Zing MP3 hoặc các nền tảng khác: Chỉ cần 1 lượt tải tiêu chuẩn
            strategies = [
                {
                    "name": "standard_platform",
                    "client": None,
                    "cookie_file": None,
                }
            ]

        last_exception = None
        for strat in strategies:
            strat_name = strat["name"]
            client_pref = strat.get("client")
            cookie_to_use = strat.get("cookie_file")

            def _reject_non_audio(
                info_dict: dict[str, Any], *, incomplete: bool = False
            ) -> str | None:
                if info_dict.get("ext") == "mhtml":
                    return "Bỏ qua định dạng storyboard mhtml không có âm thanh"
                if (
                    not incomplete
                    and info_dict.get("acodec") == "none"
                    and info_dict.get("vcodec") == "none"
                ):
                    return "Bỏ qua định dạng không có audio và video"
                return None

            ydl_opts: dict[str, Any] = {
                # Format: Ưu tiên audio thuần túy; nếu YouTube giấu audio thì tự động hạ xuống video có tiếng để FFmpeg bóc tách sang MP3
                "format": "bestaudio/ba/b/best",
                "match_filter": _reject_non_audio,
                "max_filesize": 150 * 1024 * 1024,
                "outtmpl": out_template,
                "quiet": True,
                "no_warnings": True,
                "ffmpeg_location": self.ffmpeg_path,
                "nocheckcertificate": True,
                "geo_bypass": True,
                "retries": 2,
                "socket_timeout": 20,
                "user_agent": user_agent,
                "http_headers": {
                    "User-Agent": user_agent,
                    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Referer": "https://www.youtube.com/"
                    if is_youtube
                    else "https://soundcloud.com/",
                },
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }

            # Bypass kiểm tra format availability — fix lỗi 'Requested format is not available'
            # khi YouTube trả về SABR-only stream mà yt-dlp chưa hỗ trợ phân giải
            if is_youtube:
                ydl_opts["check_formats"] = None  # --no-check-formats

            if is_youtube and client_pref:
                ydl_opts["extractor_args"] = {"youtube": {"player_client": client_pref}}
                if any(c in client_pref for c in ("android", "ios", "tv")):
                    # Để yt-dlp tự động thiết lập User-Agent và headers chính chủ của mobile app
                    ydl_opts.pop("user_agent", None)
                    if "http_headers" in ydl_opts:
                        ydl_opts["http_headers"].pop("User-Agent", None)

            if is_youtube:
                yt_proxy = get_youtube_proxy()
                if yt_proxy:
                    ydl_opts["proxy"] = yt_proxy

            if cookie_to_use and Path(cookie_to_use).exists():
                ydl_opts["cookiefile"] = str(cookie_to_use)

            if js_runtimes_cfg:
                ydl_opts["js_runtimes"] = js_runtimes_cfg

            try:
                with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if isinstance(info, dict) and "entries" in info:
                        raw_entries = info.get("entries")
                        entries_list = list(cast(Any, raw_entries or []))
                        if not entries_list:
                            raise FileNotFoundError(
                                f"Không tìm thấy kết quả tìm kiếm cho query: {url}"
                            )
                        first_entry = _extract_first_entry(entries_list)
                        if first_entry:
                            info = first_entry

                    info_dict = dict(info) if isinstance(info, dict) else {}
                    if not info_dict.get("webpage_url"):
                        video_id = info_dict.get("id")
                        if video_id and not str(video_id).startswith("http"):
                            info_dict["webpage_url"] = f"https://www.youtube.com/watch?v={video_id}"
                        elif info_dict.get("url") and str(info_dict["url"]).startswith("http"):
                            info_dict["webpage_url"] = info_dict["url"]
                        elif url.startswith("http"):
                            info_dict["webpage_url"] = url

                    # Tìm file kết quả (.mp3 do postprocessor tạo ra hoặc file gốc hợp lệ, loại bỏ storyboard)
                    candidates = [
                        c
                        for c in self.temp_dir.glob(f"{track_id}_raw.*")
                        if c.suffix.lower() not in (".mhtml", ".part", ".ytdl")
                    ]
                    if candidates:
                        mp3_cands = [c for c in candidates if c.suffix.lower() == ".mp3"]
                        raw_file = mp3_cands[0] if mp3_cands else candidates[0]
                        return raw_file, info_dict

                    raise FileNotFoundError(
                        f"Không tìm thấy file audio đã tải về cho ID {track_id}"
                    )
            except Exception as e:
                last_exception = e
                _safe_log(f"[AudioEngineService] Chiến lược '{strat_name}' thử nghiệm: {e}")
                continue

        raise AudioProcessingException(f"Lỗi tải âm thanh từ URL/Tìm kiếm: {last_exception}")

    @staticmethod
    def _extract_youtube_video_id(url: str) -> str | None:
        """Trích xuất YouTube Video ID (11 ký tự) từ các dạng URL YouTube."""
        patterns = [
            r"(?:youtube\.com/watch\?.*?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _slice_multisegment_sample(
        self, input_audio: Path, track_id: str, total_duration: float | None = None
    ) -> Path:
        """
        Trích xuất đa phân đoạn đại diện toàn bài (Multi-Segment):
        - Đoạn 1 (15s): Verse / Khúc đầu (20% thời lượng bài)
        - Đoạn 2 (15s): Điệp khúc / Chorus chính (50% thời lượng bài)
        - Đoạn 3 (15s): Chuyển đoạn / Climax / Bridge (75% thời lượng bài)
        """
        output_sample = self.temp_dir / f"{track_id}_multisegment.wav"

        # TỐI ƯU SIÊU TỐC: Dùng FFmpeg CLI để slice stream trực tiếp trong 0.05 giây
        dur = float(total_duration) if total_duration and total_duration > 0 else 0.0
        if dur <= 0.0:
            try:
                snd_info = sf.info(str(input_audio))
                dur = float(snd_info.duration)
            except Exception:
                dur = self._get_audio_duration_ffmpeg(input_audio)

        if dur > 0.0:
            try:
                if dur <= 45.0:
                    cmd = [
                        self.ffmpeg_path,
                        "-y",
                        "-i",
                        str(input_audio),
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        str(output_sample),
                    ]
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
                    )
                    if res.returncode == 0 and output_sample.exists():
                        return output_sample
                else:
                    slice_dur = 15.0
                    start_1 = max(0.0, min(dur * 0.20 - slice_dur / 2, dur - slice_dur))
                    start_2 = max(0.0, min(dur * 0.50 - slice_dur / 2, dur - slice_dur))
                    start_3 = max(0.0, min(dur * 0.75 - slice_dur / 2, dur - slice_dur))

                    filter_complex = (
                        f"[0:a]atrim=start={start_1:.2f}:end={start_1 + slice_dur:.2f},asetpts=PTS-STARTPTS[a1];"
                        f"[0:a]atrim=start={start_2:.2f}:end={start_2 + slice_dur:.2f},asetpts=PTS-STARTPTS[a2];"
                        f"[0:a]atrim=start={start_3:.2f}:end={start_3 + slice_dur:.2f},asetpts=PTS-STARTPTS[a3];"
                        "[a1][a2][a3]concat=n=3:v=0:a=1[out]"
                    )
                    cmd = [
                        self.ffmpeg_path,
                        "-y",
                        "-i",
                        str(input_audio),
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        "[out]",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        str(output_sample),
                    ]
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
                    )
                    if res.returncode == 0 and output_sample.exists():
                        return output_sample
            except Exception as ffmpeg_err:
                _safe_log(
                    f"[AudioEngineService] FFmpeg slice error, fallback to soundfile: {ffmpeg_err}"
                )

        try:
            data, sr = sf.read(str(input_audio))
            if data.ndim == 1:
                data = np.column_stack([data, data])

            total_frames = len(data)
            duration = total_frames / sr

            if duration <= 45.0:
                sf.write(str(output_sample), data, sr)
                return output_sample

            slice_dur = 15.0
            slice_frames = int(slice_dur * sr)
            fade_len = int(0.05 * sr)
            fade_in = np.linspace(0, 1, fade_len)[:, np.newaxis]
            fade_out = np.linspace(1, 0, fade_len)[:, np.newaxis]

            def extract_slice(center_ratio: float) -> np.ndarray:
                center_frame = int(center_ratio * total_frames)
                start_frame = max(0, center_frame - slice_frames // 2)
                end_frame = min(total_frames, start_frame + slice_frames)
                if end_frame - start_frame < slice_frames:
                    start_frame = max(0, end_frame - slice_frames)

                segment = data[start_frame:end_frame].copy()
                if len(segment) >= fade_len * 2:
                    segment[:fade_len] *= fade_in
                    segment[-fade_len:] *= fade_out
                return segment

            seg1 = extract_slice(0.20)
            seg2 = extract_slice(0.50)
            seg3 = extract_slice(0.75)

            composite = np.concatenate([seg1, seg2, seg3], axis=0)
            sf.write(str(output_sample), composite, sr)
            return output_sample

        except Exception as e:
            _safe_log(f"[AudioEngineService] Fallback sang FFmpeg đơn đoạn do lỗi: {e}")
            return self._slice_ffmpeg_fallback(input_audio, track_id, total_duration)

    def _get_audio_duration_ffmpeg(self, audio_path: Path) -> float:
        """Trích xuất độ dài audio bằng FFmpeg khi định dạng không được soundfile hỗ trợ trực tiếp."""
        try:
            cmd = [self.ffmpeg_path, "-i", str(audio_path)]
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
            if m:
                return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3))
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _slice_ffmpeg_fallback(
        self, input_audio: Path, track_id: str, total_duration: float | None = None
    ) -> Path:
        """Fallback FFmpeg cắt đoạn 30s đơn nếu xảy ra lỗi."""
        output_sample = self.temp_dir / f"{track_id}_30s.wav"
        offset = AUDIO_SAMPLE_OFFSET
        if total_duration:
            if total_duration <= AUDIO_SAMPLE_DURATION:
                offset = 0
            elif total_duration < (AUDIO_SAMPLE_OFFSET + AUDIO_SAMPLE_DURATION):
                offset = max(0, int(total_duration / 3))

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            str(offset),
            "-t",
            str(AUDIO_SAMPLE_DURATION),
            "-i",
            str(input_audio),
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_sample),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise AudioProcessingException(f"FFmpeg slicing failed: {result.stderr}")

        return output_sample
