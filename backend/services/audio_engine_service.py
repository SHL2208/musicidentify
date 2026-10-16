import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import yt_dlp

from ..core.config import (
    AUDIO_SAMPLE_DURATION,
    AUDIO_SAMPLE_OFFSET,
    get_cookie_file_path,
    get_ffmpeg_binary,
)
from ..core.exceptions import AudioProcessingException
from ..core.interfaces.repositories.storage_repository_interface import IStorageRepository
from ..core.interfaces.services.audio_engine_interface import IAudioEngine
from ..core.models.entities import InputType, TrackItem


def _extract_first_entry(entries: Any) -> dict | None:
    """Trích xuất phần tử đầu tiên từ entries của yt-dlp (hỗ trợ cả list, PagedList và Iterable)."""
    if not entries:
        return None
    try:
        entry = entries[0]
        return dict(entry) if isinstance(entry, dict) else None
    except Exception:
        try:
            entry = next(iter(entries))
            return dict(entry) if isinstance(entry, dict) else None
        except Exception:
            return None


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
                    "webpage_url": str(track.raw_query),
                    "url": str(track.raw_query),
                }
            else:
                url_to_download = track.resolved_url
                if not url_to_download and track.input_type not in (InputType.LOCAL_FILE, InputType.URL):
                    from ..core.container import get_container

                    res_svc = get_container().resolver
                    try:
                        resolved_track = res_svc.resolve_track(
                            track.raw_query, artist=track.artist or ""
                        )
                    except TypeError:
                        resolved_track = res_svc.resolve_track(track.raw_query)

                    if resolved_track and getattr(resolved_track, "resolved_url", None):
                        res_url = str(resolved_track.resolved_url)
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

                try:
                    raw_audio_path, metadata = self._download_audio(url_to_download, track.id)
                except Exception as dl_err:
                    err_str = str(dl_err).lower()
                    if (
                        "zingmp3" in url_to_download.lower()
                        or "vip" in err_str
                        or "premieres" in err_str
                        or "upcoming" in err_str
                    ):
                        from ..core.container import get_container

                        res_svc = get_container().resolver
                        target_t = track.track_title or track.raw_query
                        target_a = track.artist or ""
                        yt_url, _, _, is_match, score = res_svc.search_youtube(
                            target_t, target_artist=target_a
                        )

                        # BẢO VỆ CHẶT CHẼ NGUỒN ÂM THANH:
                        # Tuyệt đối không xóa tên ca sĩ để tải bừa bản cover của ca sĩ khác.
                        # Chỉ chấp nhận fallback sang YouTube nếu khớp chính xác nghệ sĩ mục tiêu (score >= 0.65).
                        if yt_url and is_match and score >= 0.65:
                            raw_audio_path, metadata = self._download_audio(yt_url, track.id)
                            metadata["is_fallback_audio"] = True
                            metadata["fallback_from_url"] = url_to_download
                        else:
                            raise AudioProcessingException(
                                f"Bản thu '{target_t}' ({target_a}) bị khóa VIP trên Zing MP3 và không tìm thấy bản phát hành chính chủ trên YouTube."
                            ) from dl_err
                    else:
                        raise

            # Trích xuất mẫu đa phân đoạn (Verse 20%, Chorus 50%, Climax 75%)
            sample_path = self._slice_multisegment_sample(
                raw_audio_path, track.id, track.duration_sec
            )

            # Xóa file audio gốc tải về nếu là file tạm
            if raw_audio_path != sample_path and raw_audio_path.is_relative_to(self.temp_dir):
                self.storage_repo.cleanup_file(raw_audio_path)

            return sample_path, metadata

        except Exception as e:
            if isinstance(e, AudioProcessingException):
                raise
            raise AudioProcessingException(
                f"Lỗi khi xử lý âm thanh bài '{track.track_title or track.raw_query}': {e}"
            ) from e

    def _download_audio(self, url: str, track_id: str) -> tuple[Path, dict]:
        """Tải luồng âm thanh tốt nhất bằng yt-dlp với cơ chế phòng thủ lỗi và chống 403 Forbidden."""
        out_template = str(self.temp_dir / f"{track_id}_raw.%(ext)s")

        client_fallbacks = [None, ["web"], ["mweb"], ["tv"]]
        node_bin = shutil.which("node") or (
            r"C:\Program Files\nodejs\node.exe"
            if Path(r"C:\Program Files\nodejs\node.exe").exists()
            else None
        )
        js_runtimes_cfg: dict[str, Any] = {"node": {}} if node_bin else {}

        # Tự động nạp cookies.txt hoặc YOUTUBE_COOKIES_CONTENT từ cấu hình
        cookie_path = get_cookie_file_path()
        cookie_file: str | None = str(cookie_path) if cookie_path else None

        last_exception = None
        for client_pref in client_fallbacks:
            ydl_opts: dict[str, Any] = {
                # Tải audio stream nén gốc nhẹ (~1-2MB) thay vì convert toàn bộ bài sang WAV 100MB
                "format": "ba[ext=m4a]/ba[ext=webm]/ba[ext=mp3]/bestaudio/ba",
                "max_filesize": 35 * 1024 * 1024,
                "outtmpl": out_template,
                "quiet": True,
                "no_warnings": True,
                "ffmpeg_location": self.ffmpeg_path,
                "nocheckcertificate": True,
                "retries": 1,
                "socket_timeout": 12,
                "remote_components": ["ejs:github"],
            }
            if client_pref:
                ydl_opts["extractor_args"] = {"youtube": {"player_client": client_pref}}
            if cookie_file:
                ydl_opts["cookiefile"] = cookie_file
            if js_runtimes_cfg:
                ydl_opts["js_runtimes"] = js_runtimes_cfg
            try:
                with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if isinstance(info, dict) and "entries" in info:
                        first_entry = _extract_first_entry(info.get("entries"))
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

                    candidates = list(self.temp_dir.glob(f"{track_id}_raw.*"))
                    if candidates:
                        raw_file = candidates[0]
                        return raw_file, info_dict

                    raise FileNotFoundError(
                        f"Không tìm thấy file audio đã tải về cho ID {track_id}"
                    )
            except Exception as e:
                last_exception = e
                continue

        raise AudioProcessingException(f"Lỗi tải âm thanh từ URL/Tìm kiếm: {last_exception}")

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
                print(
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
            print(f"[AudioEngineService] Fallback sang FFmpeg đơn đoạn do lỗi: {e}")
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
