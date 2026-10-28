"""
Unit tests for reason language selection and UI cleanup (removal of test tab).
"""
import unittest
from unittest.mock import MagicMock, patch

from backend.api.schemas.dtos import BatchDetectRequest, TrackDetectRequest
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.services.detection_service import DetectionService
from backend.services.gemini_classifier_service import GeminiClassifierService


class TestReasonLanguage(unittest.TestCase):

    def test_dtos_language_field(self):
        """Test that DTOs accept language and default to 'vi'."""
        req_default = TrackDetectRequest(query="https://youtube.com/watch?v=123")
        self.assertEqual(req_default.language, "vi")

        req_en = TrackDetectRequest(query="https://youtube.com/watch?v=123", language="en")
        self.assertEqual(req_en.language, "en")

        batch_default = BatchDetectRequest(queries=["Song 1"])
        self.assertEqual(batch_default.language, "vi")

        batch_en = BatchDetectRequest(queries=["Song 1"], language="en")
        self.assertEqual(batch_en.language, "en")

    def test_gemini_prompt_language_directives(self):
        """Test that GeminiClassifierService forensic prompt includes language directives."""
        service = GeminiClassifierService(api_key="mock_key")

        # Real AcousticMetrics instance
        mock_metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=180.0,
            fakeprint_score=0.05,
            compression_source="Studio",
            pitch_jitter=0.08,
            pitch_rigidity_ratio=0.1,
            pitch_stability_label="Natural",
            stem_bleed_score=0.05,
            high_freq_ratio_16k=0.1,
            ultrasonic_cutoff_detected=False,
            is_instrumental=False,
            vocal_energy_ratio=0.4,
            harmonic_ratio=0.6,
            stereo_correlation=0.8,
            spectral_rolloff_95_hz=18000.0,
            has_brickwall_cutoff=False,
            fakeprint_detected=False,
            stem_bleed_detected=False,
            pitch_anomaly_detected=False,
            modern_ai_vocal_suspect=False,
            vocal_formant_variance=150.0,
        )

        # Test Vietnamese prompt
        prompt_vi = service._build_forensic_prompt(
            track_id="track_123", metrics=mock_metrics, metadata={}, language="vi"
        )
        self.assertIn("VIETNAMESE (Tiếng Việt)", prompt_vi)
        self.assertIn("MANDATORY OUTPUT LANGUAGE REQUIREMENT", prompt_vi)

        # Test English prompt
        prompt_en = service._build_forensic_prompt(
            track_id="track_123", metrics=mock_metrics, metadata={}, language="en"
        )
        self.assertIn("strictly in ENGLISH", prompt_en)
        self.assertIn("DO NOT use Vietnamese", prompt_en)

    def test_anti_fp_anti_fn_language_localization(self):
        """Test that safeguard reasons are localized in EN and VI."""
        service = DetectionService.__new__(DetectionService)

        track = TrackItem(
            id="track_1",
            raw_query="Son Tung MTP",
            input_type=InputType.SONG_NAME,
            track_title="Lac Troi",
            artist="Son Tung M-TP",
        )
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=180.0,
            fakeprint_score=0.1,
            stem_bleed_score=0.05,
        )

        res_base = DetectionResult(
            track_id="track_1",
            track_title="Lac Troi",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI GENERATED",
            confidence_score=0.85,
            suspected_engine="Suno",
            reasons=["Dấu hiệu AI tổng hợp giọng"],
            detailed_explanation="Giải thích chuyên sâu tiếng Việt",
        )

        with patch("backend.services.signal_analyzer_service.is_verified_human_artist", return_value=True):
            res_en = service._apply_cross_validation_safeguard(
                result=res_base.model_copy(deep=True),
                track=track,
                metadata={"artist": "Son Tung M-TP"},
                metrics=metrics,
                language="en",
            )
            self.assertFalse(res_en.is_ai)
            self.assertTrue(any("[HUMAN MUSIC SAFEGUARD]" in r for r in res_en.reasons))

            res_vi = service._apply_cross_validation_safeguard(
                result=res_base.model_copy(deep=True),
                track=track,
                metadata={"artist": "Son Tung M-TP"},
                metrics=metrics,
                language="vi",
            )
            self.assertFalse(res_vi.is_ai)
            self.assertTrue(any("[BẢO VỆ NHẠC NGƯỜI]" in r for r in res_vi.reasons))

    def test_adapt_result_language(self):
        """Test that _adapt_result_language delegates translation when language does not match."""
        service = DetectionService.__new__(DetectionService)
        mock_classifier = MagicMock()
        mock_classifier.translate_forensic_text.return_value = (
            ["Translated reason 1", "Translated reason 2"],
            "Translated explanation"
        )
        service.classifier = mock_classifier
        service.track_repo = MagicMock()

        # Vietnamese result
        res_vi = DetectionResult(
            track_id="track_1",
            track_title="Bản tình ca",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.9,
            suspected_engine="None",
            reasons=["Giọng hát tự nhiên", "Không có vết cắt tần số"],
            detailed_explanation="Bản thu có dải động học phong phú.",
        )

        # Adapt to English
        res_adapted = service._adapt_result_language(res_vi, language="en")
        mock_classifier.translate_forensic_text.assert_called_once()
        self.assertEqual(res_adapted.reasons, ["Translated reason 1", "Translated reason 2"])
        self.assertEqual(res_adapted.detailed_explanation, "Translated explanation")

    def test_frontend_no_test_tab(self):
        """Verify that frontend/app.py does not import or render the benchmark test tab."""
        with open("frontend/app.py", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("render_benchmark_view", content)
        self.assertNotIn("Chấm điểm Testcase (Benchmark)", content)
        self.assertNotIn("tab3", content)
        self.assertIn("tab1, tab2 = st.tabs(", content)


if __name__ == "__main__":
    unittest.main()
