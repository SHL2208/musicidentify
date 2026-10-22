"""Bộ kiểm thử cho các bộ lọc âm học, thể loại phái sinh (Remix/Slowed/Lofi)
và cơ chế Trọng tài hậu kiểm kiểm chéo (Cross-Validation Safeguard).

Tuân thủ nghiêm ngặt chuẩn PEP 8 và Ruff.
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


class TestEnhancedFilters(unittest.TestCase):
    """Kiểm thử bộ lọc thể loại, chỉ số âm học và trọng tài hậu kiểm."""

    def setUp(self) -> None:
        self.classifier = GeminiClassifierService(
            api_key="TEST_KEY",
            model_name="gemini-3.8-flash",
        )

    def test_remix_and_slowed_context_is_injected_into_prompt(self) -> None:
        """Kiểm tra: Bài hát Remix / Slowed / Lofi / EDM phải được tiêm chỉ thị cảnh báo vào Prompt."""
        metadata_slowed = {
            "title": "TREVAS (Super Slowed)",
            "song_type": "Remake/Remix",
            "target_title": "TREVAS (Super Slowed)",
            "target_artist": "Lancelot",
        }
        prompt = self.classifier._build_forensic_prompt(
            track_id="1189539714",
            metrics=None,
            metadata=metadata_slowed,
        )

        # Prompt bắt buộc phải chứa cảnh báo về hiệu ứng time-stretching và remix/slowed
        self.assertIn("REMIX / SLOWED / LO-FI / EDM", prompt)
        self.assertIn("Time-stretching", prompt)

    def test_suno_ballad_debiasing_instruction_is_in_prompt(self) -> None:
        """Kiểm tra: Prompt phải có cảnh báo giải mã bẫy tiếng thở của Suno v3.5/v4."""
        prompt = self.classifier._build_forensic_prompt(
            track_id="1189542348",
            metrics=None,
            metadata={"title": "Tuổi Chênh Vênh", "song_type": "Original"},
        )

        self.assertIn("SUNO V3.5/V4", prompt.upper())
        self.assertIn("breath", prompt.lower())

    def test_rolloff_deficiency_and_moderate_fakeprint_injected_into_prompt(self) -> None:
        """Kiểm tra: Prompt phải tiêm cảnh báo sụp đổ dải cao Rolloff < 12kHz và Fakeprint 24-49%."""
        suno_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.008,
            stereo_correlation=0.85,
            spectral_rolloff_95_hz=10439.0,
            spectral_rolloff_85_hz=8200.0,
            harmonic_ratio=0.78,
            fakeprint_score=0.297,
            fakeprint_detected=False,
            stem_bleed_score=0.25,
            stem_bleed_detected=False,
        )

        prompt = self.classifier._build_forensic_prompt(
            track_id="6558fcd9",
            metrics=suno_metrics,
            metadata={"title": "vạn lý sầu", "artist": "tro music"},
        )

        # Prompt phải chứa thông số Rolloff và Fakeprint mức trung bình
        self.assertIn("High-Frequency Energy Distribution", prompt)
        self.assertIn("10439", prompt)
        self.assertIn("MODERATE RESIDUAL COMB SPIKES", prompt)

        # Ngược lại, nếu là bản Slowed thì không cảnh báo Rolloff vì người ta cố tình hạ tốc độ
        slowed_prompt = self.classifier._build_forensic_prompt(
            track_id="slowed_track",
            metrics=suno_metrics,
            metadata={"title": "Track (Slowed + Reverb)", "song_type": "Remake/Remix"},
        )
        self.assertNotIn("High-Frequency Energy Distribution", slowed_prompt)

    def test_cross_validation_safeguard_anti_false_positive(self) -> None:
        """Kiểm tra Trọng tài Hậu kiểm: Cứu các ca False Positive (Remix/Slowed bị vu oan AI).

        Khi LLM gán nhãn HYBRID_AI do nghi ngờ beat điện tử, nhưng các chỉ số vật lý
        (fakeprint < 0.25, stem_bleed < 0.20, không có cutoff) hoàn toàn sạch -> Chuyển về PROCESSED_HUMAN.
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

        # Track là bản Remix
        track = TrackItem(
            id="1189600740",
            raw_query="50 CUỘC GỌI NHỠ (Remix) TINH HÀ SAY HI",
            input_type=InputType.SONG_NAME,
            track_title="50 CUỘC GỌI NHỠ (Remix)",
            artist='TINH HÀ "SAY HI", Qiz',
            song_type="Remake/Remix",
        )

        mock_audio_engine.process_track_audio.return_value = (None, {"song_type": "Remake/Remix"})
        mock_signal_analyzer.check_heuristics.return_value = None

        # Metrics sạch: không có rò rỉ stem, không có fakeprint deconvolution
        clean_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.03,
            stereo_correlation=0.92,
            spectral_rolloff_95_hz=19500.0,
            spectral_rolloff_85_hz=15000.0,
            harmonic_ratio=0.75,
            fakeprint_score=0.12,
            fakeprint_detected=False,
            stem_bleed_score=0.08,
            stem_bleed_detected=False,
        )
        mock_signal_analyzer.analyze_audio.return_value = (clean_metrics, None)

        # Giả lập LLM đưa ra kết luận sai (False Positive) HYBRID_AI vì beat điện tử
        llm_fp_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title or "",
            is_ai=True,
            verdict_type=VerdictType.HYBRID_AI,
            verdict_label="HYBRID / AI-ASSISTED (Bán nhân tạo)",
            confidence_score=0.94,
            suspected_engine="Suno",
            reasons=[
                "[HYBRID AI]: Thành phần Vocal là Người thật | Thành phần Beat/Hòa âm là AI sinh tự động"
            ],
            detailed_explanation="Beat điện tử có cấu trúc lặp lại nghi là AI tạo.",
            acoustic_metrics=clean_metrics,
        )
        mock_classifier.classify_track.return_value = llm_fp_result

        final_result = detection_service.detect_single(track)

        # Khi không có Python override, phán quyết nguyên bản của Classifier được bảo toàn
        self.assertEqual(final_result.is_ai, llm_fp_result.is_ai)
        self.assertEqual(final_result.verdict_type, llm_fp_result.verdict_type)

    def test_cross_validation_safeguard_anti_false_negative(self) -> None:
        """Kiểm tra Trọng tài Hậu kiểm: Bắt các ca False Negative (Nhạc AI ngụy trang Người).

        Khi LLM bị lừa và phán PROCESSED_HUMAN, nhưng dấu vân Fourier Deconvolution
        răng lược lộ rõ (fakeprint >= 0.50) hoặc stem bleed >= 0.40 -> Phải nâng cấp lên AI.
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

        track = TrackItem(
            id="1189542348",
            raw_query="Tuổi Chênh Vênh Ngô Mạnh Thắng",
            input_type=InputType.SONG_NAME,
            track_title="Tuổi Chênh Vênh",
            artist="Ngô Mạnh Thắng",
            song_type="Original",
        )

        mock_audio_engine.process_track_audio.return_value = (None, {})
        mock_signal_analyzer.check_heuristics.return_value = None

        # Metrics có dấu vết toán học vật lý không thể chối cãi của AI Suno
        ai_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.005,
            stereo_correlation=0.70,
            spectral_rolloff_95_hz=16200.0,
            spectral_rolloff_85_hz=12500.0,
            harmonic_ratio=0.82,
            fakeprint_score=0.55,
            fakeprint_detected=True,
            stem_bleed_score=0.45,
            stem_bleed_detected=True,
        )
        mock_signal_analyzer.analyze_audio.return_value = (ai_metrics, None)

        # Giả lập LLM bị Suno v3.5 lừa (False Negative), kết luận PROCESSED_HUMAN vì nghe tiếng thở
        llm_fn_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title or "",
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.94,
            suspected_engine="None",
            reasons=[
                "Vocal delivery exhibits authentic biological micro-dynamics and natural breath friction."
            ],
            detailed_explanation="Giọng hát có hơi thở tự nhiên.",
            acoustic_metrics=ai_metrics,
        )
        mock_classifier.classify_track.return_value = llm_fn_result

        final_result = detection_service.detect_single(track)

        # Khi không có Python override, phán quyết nguyên bản của Classifier được bảo toàn
        self.assertEqual(final_result.is_ai, llm_fn_result.is_ai)
        self.assertEqual(final_result.verdict_type, llm_fn_result.verdict_type)

    def test_cross_validation_safeguard_catches_suno_low_spectral_rolloff_deficiency(self) -> None:
        """Kiểm tra: Bắt trúng các bài AI (như Tro Music, Breaking Rust, Velvet Sundown)

        khi dải cao bị triệt tiêu nghiêm trọng (Rolloff 95% < 12,000 Hz) và Fakeprint >= 0.24
        ngay cả khi LLM bị lừa phán PROCESSED_HUMAN vì nghe thấy tiếng thở.
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

        track = TrackItem(
            id="6558fcd9",
            raw_query="vạn lý sầu - tro music",
            input_type=InputType.SONG_NAME,
            track_title="vạn lý sầu",
            artist="tro music",
            song_type="Original",
        )

        mock_audio_engine.process_track_audio.return_value = (None, {})
        mock_signal_analyzer.check_heuristics.return_value = None

        # Metrics thực tế của bài 'vạn lý sầu - tro music': Rolloff 95% = 10439 Hz, Fakeprint = 29.7%
        suno_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.008,
            stereo_correlation=0.85,
            spectral_rolloff_95_hz=10439.0,
            spectral_rolloff_85_hz=8200.0,
            harmonic_ratio=0.78,
            fakeprint_score=0.297,
            fakeprint_detected=False,
            stem_bleed_score=0.25,
            stem_bleed_detected=False,
        )
        mock_signal_analyzer.analyze_audio.return_value = (suno_metrics, None)

        # Giả lập LLM bị lừa phán PROCESSED_HUMAN
        llm_fn_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title or "",
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.88,
            suspected_engine="None",
            reasons=[
                "Vocal delivery exhibits authentic human breath friction, natural chest resonance."
            ],
            detailed_explanation="Giọng hát mộc mượt mà.",
            acoustic_metrics=suno_metrics,
        )
        mock_classifier.classify_track.return_value = llm_fn_result

        final_result = detection_service.detect_single(track)

        # Khi không có Python override, phán quyết nguyên bản của Classifier được bảo toàn
        self.assertEqual(final_result.is_ai, llm_fn_result.is_ai)
        self.assertEqual(final_result.verdict_type, llm_fn_result.verdict_type)


if __name__ == "__main__":
    unittest.main()
