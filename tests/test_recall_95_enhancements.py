"""Bộ kiểm thử TDD cho mục tiêu nâng Recall nhận diện nhạc AI lên >95% và bảo vệ nhạc người thật.

Tuân thủ nghiêm ngặt PEP 8, Ruff và quy trình Red-Green-Refactor.
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
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestRecall95Enhancements(unittest.TestCase):
    """Test suite chứng minh các ca lỗi cũ (Red) và xác nhận cải tiến (Green)."""

    def setUp(self) -> None:
        self.mock_audio_engine = MagicMock()
        self.mock_signal_analyzer = MagicMock()
        self.mock_classifier = MagicMock()
        self.mock_resolver = MagicMock()
        self.mock_track_repo = MagicMock()

        # Giả lập danh mục AI và Whitelist Người thật
        self.mock_track_repo.get_ai_artist_projects.return_value = [
            {"artist_name": "YNTT", "project_type": "AI Music Channel"},
            {"artist_name": "THIÊN ĐÌNH", "project_type": "AI Music Project"},
            {"artist_name": "Kẻ Viết Nhạc", "project_type": "AI Creator"},
            {"artist_name": "Kikoly", "project_type": "AI Vocal Synthesizer"},
        ]
        self.mock_track_repo.get_distributor_catalog.return_value = []

        self.detection_service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.mock_track_repo,
        )

    def test_red_signal_analyzer_flags_suno_v35_ballad_as_modern_ai_suspect(self) -> None:
        """RED TEST 1: Suno v3.5 Nhạc Xuân/Ballad (Fakeprint 0.24, Bleed 0.0, Rolloff 10.5kHz, Rigidity 0.53).

        Hiện tại: Ngưỡng MODERN_AI_FAKEPRINT_MIN = 0.35 khiến bài này bị gán modern_ai_vocal_suspect = False.
        Kỳ vọng: Hệ thống phải nhận diện được mẫu hình Suno v3.5 và trả về modern_ai_vocal_suspect = True.
        """
        analyzer = SignalAnalyzerService(
            fakeprint_detector=MagicMock(),
            storage_repo=MagicMock(),
        )
        is_suspect = analyzer._is_modern_ai_vocal(
            rolloff_95=10492.0,
            rigidity=0.53,
            harmonic_ratio=0.87,
            fakeprint_score=0.24,
            stem_bleed=0.00,
            ultrasonic_cutoff=False,
        )
        self.assertTrue(
            is_suspect,
            "Suno v3.5 Nhạc Xuân/Ballad (Fakeprint 0.24, Rigidity 0.53) phải được gắn cờ modern_ai_vocal_suspect=True",
        )

    def test_red_safeguard_anti_fp_prevents_false_override_on_human_track(self) -> None:
        """RED TEST 2: Ca sĩ người thật (Nguyễn Văn Chung, Chu Hà My) bị dải cao thấp (YouTube Opus 6.4k-8.8k).

        Gemini đã thẩm âm chính xác là PROCESSED_HUMAN.
        Hiện tại: Code tự ý override sang HYBRID_AI do cờ Rolloff thấp + Rigidity.
        Kỳ vọng: Hệ thống phải tôn trọng phán quyết của Gemini và bảo toàn nhãn HUMAN (is_ai=False).
        """
        track = TrackItem(
            id="trk_human_nguyen_van_chung",
            raw_query="Ta Thương Một Người Nguyễn Văn Chung",
            track_title="Ta Thương Một Người",
            artist="Nguyễn Văn Chung",
            input_type=InputType.SONG_NAME,
        )
        # Thông số thực tế từ database của ca khúc Nguyễn Văn Chung (bị override nhầm sang AI)
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=6408.0,
            spectral_rolloff_85_hz=4200.0,
            stereo_correlation=0.91,
            harmonic_ratio=0.85,
            compression_source="Opus/AAC (YouTube/Web Streaming Lossy)",
            has_brickwall_cutoff=True,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_rigidity_ratio=0.69,
            pitch_jitter=0.06,
            fakeprint_score=0.21,
            fakeprint_detected=False,
            is_instrumental=False,
            vocal_energy_ratio=0.48,
            stem_bleed_score=0.04,
            stem_bleed_detected=False,
            ultrasonic_cutoff_detected=False,
            modern_ai_vocal_suspect=True,  # Bị nghi ngờ do dải cao thấp
        )

        llm_human_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=False,
            confidence_score=0.92,
            suspected_engine="None",
            reasons=["Giọng hát mộc tự nhiên với độ động và cộng hưởng ngực rõ rệt."],
            detailed_explanation="Bản ghi thể hiện giọng hát thật của con người.",
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
        )

        final_result = self.detection_service._apply_cross_validation_safeguard(
            result=llm_human_result,
            track=track,
            metadata={"title": track.track_title, "artist": track.artist},
            metrics=metrics,
        )

        # Phải bảo vệ nhạc người thật: is_ai phải là False!
        self.assertFalse(
            final_result.is_ai,
            "Nhạc người thật (Nguyễn Văn Chung) đã được Gemini xác nhận là Human thì code KHÔNG ĐƯỢC phép override thành AI",
        )
        self.assertEqual(final_result.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_red_ai_artist_catalog_overrides_suno_yntt(self) -> None:
        """RED TEST 3: Dự án âm nhạc AI YNTT (Chuyện Xuân Ngày Cũ).

        Nếu Gemini bị ảo giác Auto-Tune và đoán nhầm là PROCESSED_HUMAN,
        Hệ thống phát hiện nghệ sĩ nằm trong danh mục AI Catalog và nâng cấp lên AI (is_ai=True).
        """
        track = TrackItem(
            id="trk_ai_yntt_xuan",
            raw_query="Chuyện Xuân Ngày Cũ YNTT",
            track_title="Chuyện Xuân Ngày Cũ",
            artist="YNTT",
            input_type=InputType.SONG_NAME,
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=10492.0,
            spectral_rolloff_85_hz=7200.0,
            stereo_correlation=0.88,
            harmonic_ratio=0.87,
            compression_source="Opus/AAC (YouTube/Web Streaming Lossy)",
            has_brickwall_cutoff=False,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_rigidity_ratio=0.53,
            pitch_jitter=0.08,
            fakeprint_score=0.24,
            fakeprint_detected=False,
            is_instrumental=False,
            vocal_energy_ratio=0.50,
            stem_bleed_score=0.00,
            stem_bleed_detected=False,
            ultrasonic_cutoff_detected=False,
            modern_ai_vocal_suspect=False,  # Chưa bị cờ vật lý
        )
        # Giả lập Gemini bị đánh lừa là người thật
        llm_fooled_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=False,
            confidence_score=0.94,
            suspected_engine="None",
            reasons=["Giọng hát thể hiện kỹ thuật Auto-Tune thương mại của phòng thu."],
            detailed_explanation="Bản ghi thể hiện giọng hát phòng thu.",
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
        )

        final_result = self.detection_service._apply_cross_validation_safeguard(
            result=llm_fooled_result,
            track=track,
            metadata={"title": track.track_title, "artist": track.artist},
            metrics=metrics,
        )

        self.assertTrue(
            final_result.is_ai,
            "Ca khúc của dự án AI YNTT phải được hệ thống nhận diện là AI thông qua danh mục AI Catalog",
        )
        self.assertIn("YNTT", str(final_result.reasons))


if __name__ == "__main__":
    unittest.main()
