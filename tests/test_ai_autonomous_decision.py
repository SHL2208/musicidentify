"""Kiểm thử đảm bảo AI Gemini tự chủ ra phán quyết âm học mà không bị code Python can thiệp/ghi đè (No Python Override).

Tuân thủ nghiêm ngặt chuẩn TDD (Red-Green-Refactor) và PEP 8 / Ruff.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.services.detection_service import DetectionService
from backend.services.gemini_classifier_service import GeminiClassifierService


class TestAIAutonomousDecision(unittest.TestCase):
    """Bộ kiểm thử cho chính sách để AI Gemini tự chủ quyết định 100%."""

    def setUp(self) -> None:
        self.classifier = GeminiClassifierService(
            api_key="TEST_KEY",
            model_name="gemini-3.8-flash",
        )

    def test_gemini_verdict_human_is_preserved_without_python_override(self) -> None:
        """Kiểm tra: Khi Gemini nghe thấy giọng người và phán PROCESSED_HUMAN,

        dù bài hát có Rolloff < 12kHz và Fakeprint >= 0.24, code Python TUYỆT ĐỐI KHÔNG
        được tự ý lật ngược phán quyết thành AI.
        """
        mock_audio_engine = MagicMock()
        mock_signal_analyzer = MagicMock()
        mock_classifier = MagicMock()
        mock_resolver = MagicMock()
        mock_track_repo = MagicMock()
        mock_track_repo.find_result_by_query_or_url.return_value = None

        detection_service = DetectionService(
            audio_engine=mock_audio_engine,
            signal_analyzer=mock_signal_analyzer,
            classifier=mock_classifier,
            resolver=mock_resolver,
            track_repo=mock_track_repo,
        )

        # Mô phỏng bài hát người thật 'Ngàn Năm Hồn Việt' hoặc 'THÊU HOA DỆT GẤM'
        track = TrackItem(
            id="1189473843",
            raw_query="Ngàn Năm Hồn Việt Minh Sang",
            input_type=InputType.SONG_NAME,
            track_title="Ngàn Năm Hồn Việt",
            artist="Minh Sang",
            song_type="Original",
        )

        mock_audio_engine.process_track_audio.return_value = (None, {})
        mock_signal_analyzer.check_heuristics.return_value = None

        # Metrics thực tế của bài người: Rolloff 95% = 11131 Hz, Fakeprint = 0.437
        human_track_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.015,
            stereo_correlation=0.88,
            spectral_rolloff_95_hz=11131.0,
            spectral_rolloff_85_hz=9200.0,
            harmonic_ratio=0.76,
            fakeprint_score=0.437,
            fakeprint_detected=False,
            stem_bleed_score=0.18,
            stem_bleed_detected=False,
        )
        mock_signal_analyzer.analyze_audio.return_value = (human_track_metrics, None)

        # Gemini sau khi nghe trực tiếp và phân tích vocal khẳng định đây là nhạc người
        gemini_verdict = DetectionResult(
            track_id=track.id,
            track_title=track.track_title or "",
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.92,
            suspected_engine="None",
            reasons=[
                "Giọng hát có hơi thở sinh học, cộng hưởng khoang ngực tự nhiên của ca sĩ người thật."
            ],
            detailed_explanation="Không có dấu vết khuếch tán tiềm ẩn.",
            acoustic_metrics=human_track_metrics,
        )
        mock_classifier.classify_track.return_value = gemini_verdict

        final_result = detection_service.detect_single(track)

        # ĐÒI HỎI: Phán quyết của Gemini phải được giữ nguyên vẹn 100%, không bị Python override
        self.assertFalse(
            final_result.is_ai,
            "LỖI HỆ THỐNG: Code Python đã tự ý override phán quyết Người của Gemini thành AI!",
        )
        self.assertEqual(final_result.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_neutral_prompt_does_not_contain_coercive_directive(self) -> None:
        """Kiểm tra: Prompt gửi cho Gemini không được chứa câu lệnh đe dọa / ép cung

        như 'Do NOT classify as PROCESSED_HUMAN based on vocal breathing when this severe high-frequency cutoff is present!'.
        """
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.01,
            stereo_correlation=0.85,
            spectral_rolloff_95_hz=10500.0,
            spectral_rolloff_85_hz=8500.0,
            harmonic_ratio=0.75,
            fakeprint_score=0.30,
            fakeprint_detected=False,
            stem_bleed_score=0.15,
            stem_bleed_detected=False,
        )

        prompt = self.classifier._build_forensic_prompt(
            track_id="test_track",
            metrics=metrics,
            metadata={"title": "Test Pop Song", "song_type": "Original"},
        )

        coercive_phrase = "Do NOT classify as PROCESSED_HUMAN based on vocal breathing when this severe high-frequency cutoff is present"
        self.assertNotIn(
            coercive_phrase,
            prompt,
            "LỖI: Prompt vẫn đang chứa câu lệnh ép cung khiến Gemini bị thiên kiến kết tội oan!",
        )


if __name__ == "__main__":
    unittest.main()
