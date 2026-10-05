import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
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
        Tải và trích xuất mẫu âm thanh đa phân đoạn (Multi-Segment) đại diện cho toàn bài hát.
        Trả về: (đường dẫn file .wav đa phân đoạn, metadata chi tiết)
        """
        raw_audio_path = None
        metadata = {}

        if track.input_type == InputType.LOCAL_FILE and os.path.isfile(track.raw_query):
            raw_audio_path = Path(track.raw_query)
        else:
            url_to_download = track.resolved_url or track.raw_query
            raw_audio_path, metadata = self._download_audio(url_to_download, track.id)

        # Trích xuất mẫu đa phân đoạn (Verse 20%, Chorus 50%, Climax 75%)
        sample_path = self._slice_multisegment_sample(raw_audio_path, track.id, track.duration_sec)

        # Xóa file audio gốc tải về nếu là file tạm
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

    def _slice_multisegment_sample(self, input_audio: Path, track_id: str, total_duration: Optional[float] = None) -> Path:
        """
        Trích xuất đa phân đoạn đại diện toàn bài (Multi-Segment):
        - Đoạn 1 (15s): Verse / Khúc đầu (20% thời lượng bài)
        - Đoạn 2 (15s): Điệp khúc / Chorus chính (50% thời lượng bài)
        - Đoạn 3 (15s): Chuyển đoạn / Climax / Bridge (75% thời lượng bài)
        Ghép lại thành mẫu kiểm định 45s có crossfade 50ms chống giật, tránh rơi vào khoảng lặng hay intro lo-fi.
        """
        output_sample = self.temp_dir / f"{track_id}_multisegment.wav"

        try:
            data, sr = sf.read(str(input_audio))
            if data.ndim == 1:
                data = np.column_stack([data, data])

            total_frames = len(data)
            duration = total_frames / sr

            # Nếu bài hát ngắn <= 45 giây, lấy toàn bộ bài
            if duration <= 45.0:
                sf.write(str(output_sample), data, sr)
                return output_sample

            # Nếu bài hát dài hơn, cắt 3 lát 15s (tổng 45s)
            slice_dur = 15.0
            slice_frames = int(slice_dur * sr)
            fade_len = int(0.05 * sr)  # 50ms fade để chống click khi ghép
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

            # 3 vị trí vàng: Verse (20%), Chorus (50%), Climax/Bridge (75%)
            seg1 = extract_slice(0.20)
            seg2 = extract_slice(0.50)
            seg3 = extract_slice(0.75)

            composite = np.concatenate([seg1, seg2, seg3], axis=0)
            sf.write(str(output_sample), composite, sr)
            return output_sample

        except Exception as e:
            print(f"[AudioEngine] Fallback sang FFmpeg đơn đoạn do lỗi: {e}")
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
            raise RuntimeError(f"FFmpeg slicing failed: {result.stderr}")

        return output_sample
