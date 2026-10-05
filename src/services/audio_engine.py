import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import yt_dlp

from ..config import TEMP_DIR, AUDIO_SAMPLE_DURATION, AUDIO_SAMPLE_OFFSET, get_ffmpeg_binary
from ..models.schemas import TrackItem, InputType

class AudioEngine:
    def __init__(self, temp_dir: Path = TEMP_DIR):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = get_ffmpeg_binary()

    def process_track_audio(self, track: TrackItem) -> Tuple[Path, Optional[dict]]:
        """
        Tải và cắt đoạn âm thanh 30s đại diện cho bài hát.
        Trả về: (đường dẫn file .wav 30s, metadata chi tiết)
        """
        raw_audio_path = None
        metadata = {}

        if track.input_type == InputType.LOCAL_FILE and os.path.isfile(track.raw_query):
            raw_audio_path = Path(track.raw_query)
        else:
            url_to_download = track.resolved_url or track.raw_query
            raw_audio_path, metadata = self._download_audio(url_to_download, track.id)

        # Cắt đoạn đại diện 30s
        sample_path = self._slice_30s_sample(raw_audio_path, track.id, track.duration_sec)

        # Xóa file audio gốc tải về (nếu là file tạm được tải từ web)
        if raw_audio_path != sample_path and raw_audio_path.is_relative_to(self.temp_dir):
            try:
                raw_audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        return sample_path, metadata

    def _download_audio(self, url: str, track_id: str) -> Tuple[Path, dict]:
        """Tải luồng âm thanh tốt nhất bằng yt-dlp."""
        out_template = str(self.temp_dir / f"{track_id}_raw.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_template,
            'quiet': True,
            'no_warnings': True,
            'ffmpeg_location': self.ffmpeg_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            wav_file = self.temp_dir / f"{track_id}_raw.wav"
            return wav_file, info

    def _slice_30s_sample(self, input_audio: Path, track_id: str, total_duration: Optional[float] = None) -> Path:
        """Sử dụng FFmpeg cắt đoạn 30s chuẩn 44.1kHz Stereo."""
        output_sample = self.temp_dir / f"{track_id}_30s.wav"

        # Tính thời điểm bắt đầu cắt (start offset)
        offset = AUDIO_SAMPLE_OFFSET
        if total_duration:
            if total_duration <= AUDIO_SAMPLE_DURATION:
                offset = 0
            elif total_duration < (AUDIO_SAMPLE_OFFSET + AUDIO_SAMPLE_DURATION):
                offset = max(0, int(total_duration / 3))

        cmd = [
            self.ffmpeg_path,
            "-y",  # overwrite output
            "-ss", str(offset),
            "-t", str(AUDIO_SAMPLE_DURATION),
            "-i", str(input_audio),
            "-ar", "44100",  # Giữ sample rate 44.1kHz để không mất dải tần cao > 16kHz
            "-ac", "2",      # 2 kênh stereo để đo phase
            str(output_sample)
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg slicing failed: {result.stderr}")

        return output_sample
