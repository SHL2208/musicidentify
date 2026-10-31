import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.core.config import get_ffmpeg_binary
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.repositories.temp_storage_repository import TempStorageRepository
from backend.services.audio_engine_service import AudioEngineService
from backend.services.detection_service import DetectionService
from backend.services.signal_analyzer_service import is_verified_human_artist


class TestStreamSliceAndDBArtists(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage_repo = TempStorageRepository(temp_dir=self.temp_dir, output_dir=self.temp_dir)
        self.audio_engine = AudioEngineService(
            storage_repo=self.storage_repo, ffmpeg_path=get_ffmpeg_binary()
        )
        self.detection_service = DetectionService(
            audio_engine=self.audio_engine,
            signal_analyzer=None,
            classifier=None,
            resolver=None,
            track_repo=None,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_slice_multisegment_on_already_sliced_short_audio(self):
        """
        Kiểm tra lỗi Double-Slice: File đầu vào đã là bản cắt 45s (từ stream slice CDN),
        nhưng track.duration_sec là 240s (thời lượng cả bài gốc).
        _slice_multisegment_sample KHÔNG được dùng 240s để atrim quá độ dài file,
        gây ra file rỗng 0-byte (dẫn đến lỗi division by zero).
        """
        # Tạo 1 file wav giả lập 45 giây (44100Hz, stereo)
        sr = 44100
        dur_sec = 45.0
        t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)
        audio_data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo_data = np.column_stack([audio_data, audio_data])

        input_audio_path = self.temp_dir / "test_track_raw.mp3"
        # Ghi file audio giả lập 45s
        sf.write(str(input_audio_path), stereo_data, sr)

        # Gọi _slice_multisegment_sample với total_duration = 240.0s
        output_sample = self.audio_engine._slice_multisegment_sample(
            input_audio=input_audio_path,
            track_id="test_track",
            total_duration=240.0,
        )

        self.assertTrue(output_sample.exists(), "Output sample phải tồn tại")
        file_info = sf.info(str(output_sample))
        self.assertGreater(file_info.frames, 0, "Output sample không được rỗng 0-frame")
        self.assertGreaterEqual(file_info.duration, 40.0, "Thời lượng phải duy trì ~45s")
        # Đảm bảo librosa / stft không bị lỗi length=0
        data, _ = sf.read(str(output_sample))
        self.assertGreater(len(data), 0, "Dữ liệu âm thanh phải lớn hơn 0 mẫu")

    def test_verified_human_artist_loaded_from_db(self):
        """
        Kiểm tra danh mục nghệ sĩ xác thực:
        Phải nạp động từ CSDL (bảng verified_human_artists).
        Các nghệ sĩ như Vũ Cát Tường, Lân Nhã, Quốc Thiên phải được xác thực là người thật.
        """
        self.assertTrue(
            is_verified_human_artist("Vũ Cát Tường", "Ảo Ảnh"),
            "Vũ Cát Tường phải được xác thực là nghệ sĩ người thật từ DB dynamic catalog",
        )
        self.assertTrue(
            is_verified_human_artist("Lân Nhã", "Chớm Đông"),
            "Lân Nhã phải được xác thực là nghệ sĩ người thật từ DB dynamic catalog",
        )
        self.assertTrue(
            is_verified_human_artist("Quốc Thiên", "Hoa và Váy"),
            "Quốc Thiên phải được xác thực là nghệ sĩ người thật từ DB dynamic catalog",
        )

    def test_safeguard_does_not_override_when_human_bio_evidence_present(self):
        """
        Kiểm tra quy tắc 'Suy đoán vô tội' (Anti-FP):
        Bản thu có Auto-Tune phòng thu (rigidity >= 0.45) và nén YouTube rolloff < 11800Hz,
        nhưng Gemini đã phát hiện bằng chứng sinh học thanh quản tự nhiên (PROCESSED_HUMAN).
        Safeguard KHÔNG ĐƯỢC tự ý override thành AI.
        """
        track = TrackItem(
            id="trk_lan_nha",
            track_title="Chớm Đông",
            artist="Lân Nhã",
            raw_query="Chớm Đông Lân Nhã",
            input_type=InputType.SONG_NAME,
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=10233.0,  # Rolloff dưới 11.8kHz do Opus YouTube
            spectral_rolloff_85_hz=6500.0,
            pitch_rigidity_ratio=0.52,  # Rigidity >= 0.45 do Auto-tune phòng thu
            harmonic_ratio=0.82,
            fakeprint_score=0.15,  # Fakeprint thấp, không có dấu vân AI
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=False,
            stem_bleed_score=0.18,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            is_instrumental=False,
        )
        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.92,
            suspected_engine="None",
            reasons=[
                "Giọng hát có độ biến thiên formant tự nhiên kết hợp với hiệu chỉnh cao độ thương mại (Auto-Tune phòng thu) nhưng vẫn giữ được vi chuyển động sinh học và thanh quản tự nhiên."
            ],
            detailed_explanation="Bản ghi thể hiện cấu trúc sản xuất phòng thu với thanh quản sinh học tự nhiên.",
            acoustic_metrics=metrics,
        )

        final_res = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={"artist": "Lân Nhã", "title": "Chớm Đông"},
            metrics=metrics,
        )

        self.assertFalse(
            final_res.is_ai,
            "Bản thu của ca sĩ thật với bằng chứng sinh học không được bị override thành AI",
        )
        self.assertEqual(final_res.verdict_type, VerdictType.PROCESSED_HUMAN)


if __name__ == "__main__":
    unittest.main()
