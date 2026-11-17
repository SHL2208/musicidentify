import unittest
from backend.core.config import (
    OPUS_STREAMING_FAKEPRINT_MIN,
    OPUS_STREAMING_RIGIDITY_MIN,
    OPUS_STREAMING_STEM_BLEED_MIN,
)
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
    WorkType,
)
from backend.services.detection_service import DetectionService
from unittest.mock import MagicMock
from backend.services.gemini_classifier_service import (
    GeminiResponseSchemaEn,
    GeminiResponseSchemaVi,
)
from backend.services.work_type_resolver import resolve_work_type_from_metadata


class TestBenchmarkAccuracyEnhancements(unittest.TestCase):
    """Kiểm thử toàn diện các cải tiến cho Benchmark:
    1. Schema Few-Shot Scoring (Vocal/Beat 1-5).
    2. Ngưỡng Opus Streaming thích ứng (Rigidity >= 0.42, Bleed >= 0.26, Fakeprint >= 0.22).
    3. Chống va chạm Namespace Nhật Linh.
    4. Mở rộng REMIX_PATTERN.
    """

    def setUp(self):
        self.ds = DetectionService(
            audio_engine=MagicMock(),
            signal_analyzer=MagicMock(),
            classifier=MagicMock(),
            resolver=MagicMock(),
            track_repo=MagicMock(),
        )

    def test_few_shot_scoring_schema_fields(self):
        """Xác nhận schema Vi và En có đầy đủ trường vocal_ai_score và beat_ai_score (1-5)."""
        schema_vi = GeminiResponseSchemaVi(
            vocal_ai_score=4,
            beat_ai_score=5,
            verdict_type="AI_GENERATED",
            is_ai=True,
            confidence_score=0.95,
            suspected_engine="Suno",
            reasons=["Lý do 1", "Lý do 2"],
            detailed_explanation="Giải thích chi tiết",
        )
        self.assertEqual(schema_vi.vocal_ai_score, 4)
        self.assertEqual(schema_vi.beat_ai_score, 5)

        schema_en = GeminiResponseSchemaEn(
            vocal_ai_score=1,
            beat_ai_score=2,
            verdict_type="PROCESSED_HUMAN",
            is_ai=False,
            confidence_score=0.90,
            suspected_engine="None",
            reasons=["Reason 1", "Reason 2"],
            detailed_explanation="Detailed explanation",
        )
        self.assertEqual(schema_en.vocal_ai_score, 1)
        self.assertEqual(schema_en.beat_ai_score, 2)

    def test_opus_streaming_consensus_safeguard(self):
        """Xác nhận ngưỡng thích ứng Opus streaming (Rigidity >= 0.42, Bleed >= 0.26, FP >= 0.22)
        kích hoạt override AI thành công trên bài hát không có trong catalog."""
        ds = self.ds

        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=36.0,
            spectral_rolloff_95_hz=14000.0,
            spectral_rolloff_85_hz=10000.0,
            pitch_rigidity_ratio=0.44,  # >= 0.42
            stem_bleed_score=0.28,      # >= 0.26
            fakeprint_score=0.25,       # >= 0.22
            stereo_correlation=0.70,
            harmonic_ratio=0.75,
            compression_source="Lossy Streaming (Opus/AAC)",
            has_brickwall_cutoff=False,
            is_studio_autotune=False,
            vocal_formant_variance=1000.0,
        )

        initial_result = DetectionResult(
            track_id="test_opus_1",
            track_title="Synthetic Waves",
            artist="Suno Indie Bot",
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN",
            confidence_score=0.90,
            suspected_engine="None",
            reasons=["Initial perception was human"],
            detailed_explanation="Initial explanation",
            acoustic_metrics=metrics,
        )

        track = TrackItem(
            id="test_opus_1",
            input_type=InputType.SONG_NAME,
            track_title="Synthetic Waves",
            artist="Suno Indie Bot",
            raw_query="Synthetic Waves Suno Indie Bot",
        )

        final_res = ds._apply_cross_validation_safeguard(
            result=initial_result,
            track=track,
            metadata={"artist": "Suno Indie Bot", "title": "Synthetic Waves"},
            metrics=metrics,
        )

        self.assertTrue(final_res.is_ai)
        self.assertTrue(final_res.consensus_is_ai)
        self.assertIn("OPUS STREAMING", final_res.reasons[0].upper())

    def test_nhat_linh_human_artist_protection(self):
        """Xác nhận ca sĩ Nhật Linh là nghệ sĩ người thật và KHÔNG bị gán nhầm là AI project."""
        ds = self.ds

        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=36.0,
            spectral_rolloff_95_hz=16000.0,
            spectral_rolloff_85_hz=12000.0,
            pitch_rigidity_ratio=0.20,
            stem_bleed_score=0.10,
            fakeprint_score=0.05,
            stereo_correlation=0.85,
            harmonic_ratio=0.70,
            compression_source="Lossy Streaming (Opus/AAC)",
            has_brickwall_cutoff=False,
            is_studio_autotune=False,
            vocal_formant_variance=200000.0,
        )

        initial_result = DetectionResult(
            track_id="test_nhat_linh",
            track_title="Vui Buồn Thoáng Mây",
            artist="Nhật Linh",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.95,
            suspected_engine="None",
            reasons=["Giọng mộc sinh học"],
            detailed_explanation="Giải thích",
            acoustic_metrics=metrics,
        )

        track = TrackItem(
            id="test_nhat_linh",
            input_type=InputType.SONG_NAME,
            track_title="Vui Buồn Thoáng Mây",
            artist="Nhật Linh",
            raw_query="Vui Buồn Thoáng Mây Nhật Linh",
        )

        final_res = ds._apply_cross_validation_safeguard(
            result=initial_result,
            track=track,
            metadata={"artist": "Nhật Linh", "title": "Vui Buồn Thoáng Mây"},
            metrics=metrics,
        )

        # Tuyệt đối không bị gán nhãn AI
        self.assertFalse(final_res.is_ai)
        self.assertFalse(final_res.consensus_is_ai)
        self.assertNotIn("DỰ ÁN ÂM NHẠC AI", " ".join(final_res.reasons).upper())

    def test_remix_pattern_comprehensive(self):
        """Kiểm tra toàn bộ các token mới trong REMIX_PATTERN."""
        titles_and_artists = [
            ("Liên Khúc Mới Nhất Con Phải Tha Phương", "Lưu Hoài Nam"),
            ("Liên Khúc Giờ Người Nơi Đâu - Đò Bỏ Bến Xưa", "Ánh Như"),
            ("Happy Birthday - Party Celebration Mix", "Birthday Studio"),
            ("Ngẩn Ngơ Giữa Ngân Hà (Deep House #2)", "THIÊN ĐÌNH"),
            ("Vé một chiều (2025 Remastered)", "Dương Trường Giang"),
            ("The Greatest Games (WC 26) - Latin Dance", "Goal Vibes"),
        ]

        for title, artist in titles_and_artists:
            w_type, _, _, _, _ = resolve_work_type_from_metadata(title=title, artist=artist)
            self.assertEqual(
                w_type,
                WorkType.REMIX,
                f"Bản thu '{title}' phải được nhận diện là REMIX!",
            )


if __name__ == "__main__":
    unittest.main()
