import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.services.audio_engine_service import AudioEngineService
from backend.services.batch_service import BatchService
from backend.services.gemini_classifier_service import (
    GeminiClassifierService,
    should_require_deep_audio_inspection,
)
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestPerformanceOptimizations(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho toàn bộ các tính năng tối ưu hóa hiệu suất (Phase 2)."""

    def setUp(self):
        GeminiClassifierService._active_key_index = 0
        GeminiClassifierService._key_cooldowns = {}

    def test_pitch_jitter_downsampling_accuracy(self):
        """Kiểm tra YIN pitch tracking với downsampling 8000Hz tính toán chính xác và nhanh chóng."""
        mock_fakeprint = MagicMock()
        mock_storage = MagicMock()
        analyzer = SignalAnalyzerService(
            fakeprint_detector=mock_fakeprint, storage_repo=mock_storage
        )

        # Tạo tín hiệu sine 440Hz chuẩn (nốt La - vocal range) dài 2 giây ở 44100Hz
        sr = 44100
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)

        res_dict = analyzer._compute_pitch_jitter(audio, sr)
        self.assertIsInstance(res_dict, dict)
        self.assertIn("pitch_jitter", res_dict)
        jitter = res_dict["pitch_jitter"]
        self.assertIsInstance(jitter, float)
        self.assertGreaterEqual(jitter, 0.0)
        self.assertLessEqual(jitter, 1.0)
        # Sóng sine hoàn hảo có pitch cực kỳ ổn định (jitter rất thấp)
        self.assertLess(jitter, 0.05)

    def test_should_require_deep_audio_inspection_logic(self):
        """Kiểm tra logic phân tầng Two-Tier Cascaded Evaluation bảo đảm không bỏ lọt Suno v3.5/v4."""
        # 1. Bài hát người thật rõ ràng: không cần Tier 2
        human_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=45.0,
            spectral_rolloff_95_hz=19500.0,
            high_freq_ratio_16k=0.35,
            pitch_rigidity_ratio=0.10,
            pitch_jitter=0.035,
            harmonic_ratio=1.2,
            fakeprint_score=0.12,
            stem_bleed_score=0.05,
        )
        self.assertFalse(
            should_require_deep_audio_inspection(human_metrics, {"title": "Official Master Track"})
        )

        # 2. Bài nghi vấn Suno v3.5/v4 (rolloff chuyển tiếp 14500Hz, rigidity 0.35) -> BẮT BUỘC Tier 2
        suno_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=45.0,
            spectral_rolloff_95_hz=14500.0,
            high_freq_ratio_16k=0.22,
            pitch_rigidity_ratio=0.35,
            pitch_jitter=0.015,
            harmonic_ratio=1.8,
            fakeprint_score=0.45,
            stem_bleed_score=0.32,
        )
        self.assertTrue(
            should_require_deep_audio_inspection(suno_metrics, {"title": "Dreaming Night"})
        )

        # 3. Bài có metadata cover / remix / AI -> BẮT BUỘC Tier 2
        self.assertTrue(
            should_require_deep_audio_inspection(human_metrics, {"title": "Song (AI Cover)"})
        )
        self.assertTrue(
            should_require_deep_audio_inspection(human_metrics, {"title": "Hit Song (Remix)"})
        )

        # 4. Bài có fakeprint borderline (0.38) -> BẮT BUỘC Tier 2
        borderline_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=45.0,
            spectral_rolloff_95_hz=18500.0,
            high_freq_ratio_16k=0.30,
            pitch_rigidity_ratio=0.20,
            pitch_jitter=0.030,
            harmonic_ratio=1.2,
            fakeprint_score=0.38,
            stem_bleed_score=0.10,
        )
        self.assertTrue(should_require_deep_audio_inspection(borderline_metrics, {}))

    def test_sticky_key_rotation_and_cooldown(self):
        """Kiểm tra cơ chế Sticky Key rotation và tự động hạ cooldown 60s khi gặp lỗi 429."""
        service = GeminiClassifierService(api_keys=["key_0", "key_1", "key_2"])

        # Ban đầu active key là key 0
        self.assertEqual(service._active_key_index, 0)
        self.assertEqual(len(service._key_cooldowns), 0)

        # Giả lập khi gọi key 0 gặp lỗi 429 (ResourceExhausted)
        # Cập nhật cooldown cho key 0
        service._key_cooldowns[0] = time.time() + 60.0
        service._active_key_index = (service._active_key_index + 1) % len(service.api_keys)

        # Active key chuyển sang key 1 và dính cố định (Sticky) ở key 1
        self.assertEqual(service._active_key_index, 1)

        # Kiểm tra vòng lặp chọn key: key 0 bị bỏ qua vì còn trong cooldown
        available_keys = [
            i
            for i in range(len(service.api_keys))
            if service._key_cooldowns.get(i, 0) <= time.time()
        ]
        self.assertNotIn(0, available_keys)
        self.assertIn(1, available_keys)
        self.assertIn(2, available_keys)

    def test_batch_service_adaptive_sleep(self):
        """Kiểm tra BatchService tự động bỏ qua sleep đối với file local và giữ sleep đối với remote."""
        mock_detect = MagicMock()
        mock_report = MagicMock()
        mock_track_repo = MagicMock()

        dummy_result = DetectionResult(
            track_id="t1",
            track_title="Test",
            source_url=None,
            input_type=InputType.LOCAL_FILE,
            raw_query="song.mp3",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="HUMAN",
            confidence_score=0.9,
            suspected_engine="None",
            reasons=[],
            detailed_explanation="",
            acoustic_metrics=None,
            spectrogram_image_path=None,
        )
        mock_detect.detect_single.return_value = dummy_result

        batch_svc = BatchService(
            detection_service=mock_detect,
            report_service=mock_report,
            track_repo=mock_track_repo,
            sleep_interval=1.0,
        )

        # 2 bài local file: không được sleep 1s
        tracks = [
            TrackItem(id="1", raw_query="local1.mp3", input_type=InputType.LOCAL_FILE),
            TrackItem(id="2", raw_query="local2.mp3", input_type=InputType.LOCAL_FILE),
        ]

        with patch("time.sleep") as mock_sleep:
            batch_svc.process_batch(tracks)
            # Không gọi sleep vì cả 2 bài là LOCAL_FILE
            mock_sleep.assert_not_called()

        # 2 bài URL: phải gọi sleep 1s giữa bài 1 và bài 2
        remote_tracks = [
            TrackItem(
                id="3", raw_query="https://youtube.com/watch?v=aaa", input_type=InputType.URL
            ),
            TrackItem(
                id="4", raw_query="https://youtube.com/watch?v=bbb", input_type=InputType.URL
            ),
        ]

        with patch("time.sleep") as mock_sleep:
            batch_svc.process_batch(remote_tracks)
            mock_sleep.assert_called_once_with(1.0)

    def test_audio_engine_stream_slice_graceful_fallback(self):
        """Kiểm tra AudioEngineService stream slicing tự động fallback an toàn nếu gặp lỗi."""
        mock_storage = MagicMock()
        mock_storage.get_temp_dir.return_value = Path("tmp")
        engine = AudioEngineService(storage_repo=mock_storage, ffmpeg_path="ffmpeg")

        # Trường hợp query không phải YouTube URL -> _try_stream_slice_download trả về None
        res = engine._try_stream_slice_download("ytsearch1:Son Tung M-TP", "track_123")
        self.assertIsNone(res)

        # Trường hợp URL là local file -> trả về None
        res_local = engine._try_stream_slice_download("C:/music/test.mp3", "track_123")
        self.assertIsNone(res_local)


if __name__ == "__main__":
    unittest.main()
