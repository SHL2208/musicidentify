import unittest
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
    WorkType,
)
from backend.services.detection_service import DetectionService
from backend.services.work_type_resolver import resolve_work_type_from_metadata
from unittest.mock import MagicMock


class TestCoverAndSafeguardOverhaul(unittest.TestCase):
    def test_cover_songs_resolution(self):
        """Kiểm tra các bài Cover từng bị đoán nhầm là Original nay được giải mã chính xác thành COVER."""
        test_cases = [
            ("Chuyện Cũ Bỏ Qua", "Mai Phương Nghi"),
            ("Ông Hoàng Bơ", "Đinh Tùng Huy, Ca sĩ Giấu Mặt"),
            ("Bông Hồng Cài Áo", "Đông Phương"),
            ("Xin Đừng Lặng Im", "Marcus"),
            ("Mưa Chiều (Giọng Nữ)", "Phố Nhạc Việt"),
            ("Chúng Ta Chỉ Là Đã Từng Yêu", "Thư Thư"),
            ("Ai Là Người Mong Ta Thành Công", "Ngọc Hằng, LKT Media"),
            ("Cao Nguyên Gió Lạnh Mưa Mùa", "Mai Diễm My"),
            ("Cứ Bỏ Mặc Em Đi", "Dương Trúc Quân"),
            ("Thương Cánh Sim Tím", "Băng Nhi AI"),
            ("Trong Mắt Em", "HAKY"),
            ("Ăn Tết Văn Minh", "Nguyễn Thương"),
            ("O Christmas Tree", "Irma Lamb"),
            ("Chẳng Trách Em Đâu", "Đình Hùng Bolero"),
        ]

        for title, artist in test_cases:
            w_type, w_label, is_cov, orig_a, orig_t = resolve_work_type_from_metadata(
                title=title,
                artist=artist,
            )
            self.assertEqual(
                w_type,
                WorkType.COVER,
                f"Bài '{title}' của '{artist}' phải được phân loại là COVER, thực tế: {w_type}",
            )
            self.assertTrue(is_cov)

    def test_anti_fp_preserves_human_with_lossy_streaming(self):
        """Kiểm tra Safeguard không vu oan bài hát người thật có rolloff thấp do nén Opus streaming."""
        detection_service = DetectionService(
            audio_engine=MagicMock(),
            signal_analyzer=MagicMock(),
            classifier=MagicMock(),
            resolver=MagicMock(),
            track_repo=MagicMock(),
        )

        track = TrackItem(
            id="test_track_1",
            input_type=InputType.SONG_NAME,
            raw_query="Bài hát người thật",
            track_title="Bài hát người thật",
            artist="Ca sĩ người thật",
        )

        # Gemini phán đoán đúng là PROCESSED_HUMAN
        initial_result = DetectionResult(
            track_id="test_track_1",
            track_title="Bài hát người thật",
            input_type=InputType.SONG_NAME,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.92,
            suspected_engine="None",
            reasons=["Bản thu phòng thu thực tế"],
            detailed_explanation="Chi tiết phân tích",
        )

        # Metrics có rolloff thấp do nén YouTube Opus (8403 Hz), fakeprint 0.35 do hiệu ứng reverb, nhưng stem_bleed sạch (0.05)
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=36.0,
            spectral_rolloff_95_hz=8403.0,
            spectral_rolloff_85_hz=6000.0,
            pitch_rigidity_ratio=0.55,
            pitch_jitter=0.015,
            stem_bleed_score=0.05,
            fakeprint_score=0.35,
            fakeprint_detected=False,
            comb_signature_score=0.25,
            stereo_correlation=0.95,
            harmonic_ratio=0.85,
            has_brickwall_cutoff=False,
            modern_ai_vocal_suspect=True,
        )

        final_result = detection_service._apply_cross_validation_safeguard(
            result=initial_result,
            track=track,
            metadata={"title": "Bài hát người thật", "artist": "Ca sĩ người thật"},
            metrics=metrics,
        )

        # Không được bị override thành AI vì stem bleed sạch!
        self.assertFalse(final_result.is_ai)
        self.assertEqual(final_result.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_anti_fn_catches_ai_with_stem_bleed(self):
        """Kiểm tra Safeguard bắt đúng AI khi có stem bleed cao và không bị override nhầm về Human."""
        detection_service = DetectionService(
            audio_engine=MagicMock(),
            signal_analyzer=MagicMock(),
            classifier=MagicMock(),
            resolver=MagicMock(),
            track_repo=MagicMock(),
        )

        track = TrackItem(
            id="test_track_ai",
            input_type=InputType.SONG_NAME,
            raw_query="Please Don't Wake Me",
            track_title="Please Don't Wake Me",
            artist="Suno Singer",
        )

        # Ban đầu kết luận AI
        ai_result = DetectionResult(
            track_id="test_track_ai",
            track_title="Please Don't Wake Me",
            input_type=InputType.SONG_NAME,
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED",
            confidence_score=0.95,
            suspected_engine="Suno",
            reasons=["[HYBRID AI]: Thành phần Vocal là AI | Beat điện tử phòng thu"],
            detailed_explanation="Chi tiết phân tích",
        )

        # Metrics Suno đặc trưng: Stem bleed 0.36, fakeprint 0.28
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=36.0,
            spectral_rolloff_95_hz=11500.0,
            spectral_rolloff_85_hz=9000.0,
            pitch_rigidity_ratio=0.52,
            pitch_jitter=0.012,
            stem_bleed_score=0.36,
            fakeprint_score=0.28,
            fakeprint_detected=False,
            comb_signature_score=0.45,
            stereo_correlation=0.90,
            harmonic_ratio=0.80,
            has_brickwall_cutoff=False,
            modern_ai_vocal_suspect=True,
        )

        final_result = detection_service._apply_cross_validation_safeguard(
            result=ai_result,
            track=track,
            metadata={"title": "Please Don't Wake Me", "artist": "Suno Singer"},
            metrics=metrics,
        )

        # Không được hạ cấp về Human!
        self.assertTrue(final_result.is_ai)
        self.assertIn(final_result.verdict_type, (VerdictType.AI_GENERATED, VerdictType.HYBRID_AI))


if __name__ == "__main__":
    unittest.main()
