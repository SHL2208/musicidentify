import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, cast
import numpy as np
import soundfile as sf
import yt_dlp

from ..core.config import AUDIO_SAMPLE_DURATION, AUDIO_SAMPLE_OFFSET, get_ffmpeg_binary
from ..core.models.entities import TrackItem, InputType
from ..core.interfaces.services.audio_engine_interface import IAudioEngine
from ..core.interfaces.repositories.storage_repository_interface import IStorageRepository
from ..core.exceptions import AudioProcessingException

def _extract_first_entry(entries: Any) -> Optional[dict]:
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

    def __init__(self, storage_repo: IStorageRepository, ffmpeg_path: Optional[str] = None):
        self.storage_repo = storage_repo
        self.temp_dir = self.storage_repo.get_temp_dir()
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_binary()

    def process_track_audio(self, track: TrackItem) -> Tuple[Path, Optional[Dict[str, Any]]]:
        """
        Tải và trích xuất mẫu âm thanh đa phân đoạn (Multi-Segment) đại diện cho toàn bài hát.
        Trả về: (đường dẫn file .wav đa phân đoạn, metadata chi tiết)
        """
        raw_audio_path: Optional[Path] = None
        metadata: Dict[str, Any] = {}

        try:
            if track.input_type == InputType.LOCAL_FILE and os.path.isfile(track.raw_query):
                raw_audio_path = Path(track.raw_query)
            else:
                url_to_download = track.resolved_url
                if not url_to_download or not url_to_download.startswith("http"):
                    url_to_download = f"ytsearch1:{track.raw_query}"
                raw_audio_path, metadata = self._download_audio(url_to_download, track.id)

            # Trích xuất mẫu đa phân đoạn (Verse 20%, Chorus 50%, Climax 75%)
            sample_path = self._slice_multisegment_sample(raw_audio_path, track.id, track.duration_sec)

            # Xóa file audio gốc tải về nếu là file tạm
            if raw_audio_path != sample_path and raw_audio_path.is_relative_to(self.temp_dir):
                self.storage_repo.cleanup_file(raw_audio_path)

            return sample_path, metadata

        except Exception as e:
            if isinstance(e, AudioProcessingException):
                raise
            raise AudioProcessingException(f"Lỗi khi xử lý âm thanh bài '{track.track_title or track.raw_query}': {e}") from e

    def _download_audio(self, url: str, track_id: str) -> Tuple[Path, dict]:
        """Tải luồng âm thanh tốt nhất bằng yt-dlp với cơ chế phòng thủ lỗi và chống 403 Forbidden."""
        out_template = str(self.temp_dir / f"{track_id}_raw.%(ext)s")
        
        client_fallbacks = [
            ['android', 'web'],
            ['mweb', 'android'],
            ['ios']
        ]

        last_exception = None
        for client_pref in client_fallbacks:
            ydl_opts = {
                'format': 'bestaudio/ba/b',
                'outtmpl': out_template,
                'quiet': True,
                'no_warnings': True,
                'ffmpeg_location': self.ffmpeg_path,
                'nocheckcertificate': True,
                'retries': 3,
                'socket_timeout': 30,
                'extractor_args': {
                    'youtube': {
                        'player_client': client_pref
                    }
                },
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
            }
            try:
                with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if isinstance(info, dict) and 'entries' in info:
                        first_entry = _extract_first_entry(info.get('entries'))
                        if first_entry:
                            info = first_entry

                    wav_file = self.temp_dir / f"{track_id}_raw.wav"
                    if not wav_file.exists():
                        candidates = list(self.temp_dir.glob(f"{track_id}_raw.*"))
                        if candidates:
                            wav_file = candidates[0]
                        else:
                            raise FileNotFoundError(f"Không tìm thấy file audio đã tải về cho ID {track_id}")

                    return wav_file, dict(info) if isinstance(info, dict) else {}
            except Exception as e:
                last_exception = e
                continue

        raise AudioProcessingException(f"Lỗi tải âm thanh từ URL/Tìm kiếm: {last_exception}")

    def _slice_multisegment_sample(self, input_audio: Path, track_id: str, total_duration: Optional[float] = None) -> Path:
        """
        Trích xuất đa phân đoạn đại diện toàn bài (Multi-Segment):
        - Đoạn 1 (15s): Verse / Khúc đầu (20% thời lượng bài)
        - Đoạn 2 (15s): Điệp khúc / Chorus chính (50% thời lượng bài)
        - Đoạn 3 (15s): Chuyển đoạn / Climax / Bridge (75% thời lượng bài)
        """
        output_sample = self.temp_dir / f"{track_id}_multisegment.wav"

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

    def _slice_ffmpeg_fallback(self, input_audio: Path, track_id: str, total_duration: Optional[float] = None) -> Path:
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
            "-ss", str(offset),
            "-t", str(AUDIO_SAMPLE_DURATION),
            "-i", str(input_audio),
            "-ar", "44100",
            "-ac", "2",
            str(output_sample)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise AudioProcessingException(f"FFmpeg slicing failed: {result.stderr}")

        return output_sample
