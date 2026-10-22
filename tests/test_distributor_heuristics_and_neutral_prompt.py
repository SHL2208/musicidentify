import unittest
from unittest.mock import MagicMock, patch

from backend.core.models.entities import AcousticMetrics, InputType, TrackItem, VerdictType
from backend.services.gemini_classifier_service import GeminiClassifierService
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestDistributorHeuristicsAndNeutralPrompt(unittest.TestCase):
    def setUp(self):
        self.fakeprint_mock = MagicMock()
        self.fakeprint_mock.analyze_fakeprint.return_value = {
            "score": 0.0,
            "detected": False,
            "freqs": [],
            "residual_spectrum": [],
        }
        self.storage_mock = MagicMock()
        self.signal_analyzer = SignalAnalyzerService(
            fakeprint_detector=self.fakeprint_mock,
            storage_repo=self.storage_mock,
        )
        self.classifier = GeminiClassifierService(
            api_keys=["fake_key_1", "fake_key_2"],
            model_name="gemini-3.8-flash",
            fallback_models=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        )

    def test_distributor_blacklist_detection(self):
        """Kiểm tra các nhà phân phối không bị gắn nhãn cứng qua Heuristics Tầng 1 mà chuyển cho mô hình AI."""
        ai_distributors = [
            "INCOM",
            "TTE Fast Considerate",
        ]
        for dist in ai_distributors:
            meta = {
                "title": "Bản nhạc thử nghiệm",
                "artist": "Nghệ sĩ ẩn danh",
                "distributor": dist,
            }
            res = self.signal_analyzer.check_heuristics(meta)
            self.assertIsNone(
                res,
                f"Distributor '{dist}' không được gắn nhãn AI cứng tại Heuristics Tầng 1 để tránh lỗi phán quyết!",
            )

    def test_distributor_whitelist_protection(self):
        """Kiểm tra các hãng phân phối chính thống không bị gắn nhãn nhầm (Tránh False Positive)."""
        human_distributors = [
            "Warner Music Vietnam",
            "Sony Music",
            "Universal Music Group",
            "VieEnt",
            "BMZ",
        ]
        for dist in human_distributors:
            meta = {
                "title": "Bản tình ca người làm",
                "artist": "Ca Sĩ Nổi Tiếng",
                "distributor": dist,
            }
            res = self.signal_analyzer.check_heuristics(meta)
            if res is not None:
                self.assertFalse(res["is_ai"], f"Distributor '{dist}' không được gắn nhãn AI!")

    def test_track_entity_has_distributor_field(self):
        """Kiểm tra đối tượng TrackItem hỗ trợ thuộc tính distributor."""
        t = TrackItem(
            id="test_id",
            input_type=InputType.SONG_NAME,
            raw_query="Bài hát A",
            track_title="Bài hát A",
            artist="Ca sĩ B",
            distributor="INCOM",
        )
        self.assertEqual(t.distributor, "INCOM")

    def test_neutral_acoustic_prompt_formatting(self):
        """
        Kiểm tra Prompt giám định gửi Gemini:
        Không được chứa các nhãn kết luận đóng khung định kiến như:
        'CONFIRMED HUMAN VOCAL TRACT WITH STUDIO AUTO-TUNE'
        'Clean isolation consistent with conventional multitrack studio production'
        Thay vào đó phải hiển thị số đo kỹ thuật khách quan.
        """
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            spectral_rolloff_95_hz=16000.0,
            spectral_rolloff_85_hz=12000.0,
            stereo_correlation=0.95,
            harmonic_ratio=0.85,
            fakeprint_score=0.1,
            fakeprint_detected=False,
            compression_source="Lossy Streaming (44.1kHz / 160kbps MP3)",
            has_brickwall_cutoff=False,
            pitch_jitter=0.012,
            pitch_rigidity_ratio=0.45,
            pitch_anomaly_detected=False,
            pitch_stability_label="Studio Auto-Tuned (Natural Vocal Tract)",
            vocal_formant_variance=320.5,
            is_studio_autotune=True,
            stem_bleed_score=0.08,
            stem_bleed_detected=False,
        )

        prompt = self.classifier._build_forensic_prompt("track_123", metrics)

        # Đảm bảo không chứa nhãn mớm lời định kiến
        self.assertNotIn(
            "CONFIRMED HUMAN VOCAL TRACT WITH STUDIO AUTO-TUNE",
            prompt,
            "Prompt không được chứa nhãn mớm lời CONFIRMED HUMAN VOCAL TRACT!",
        )
        self.assertNotIn(
            "Clean isolation consistent with conventional multitrack studio production",
            prompt,
            "Prompt không được mớm lời Clean isolation consistent with conventional multitrack!",
        )
        # Đảm bảo có số đo kỹ thuật khách quan
        self.assertIn("Rigidity:", prompt)
        self.assertIn("Multi-Stem Bleed", prompt)

    @patch("time.sleep", return_value=None)
    def test_retry_on_503_unavailable(self, mock_sleep):
        """Kiểm tra cơ chế retry khi gặp lỗi 503 Unavailable trước khi failover."""
        mock_client = MagicMock()
        # Lần 1 ném 503, lần 2 thành công
        success_response = MagicMock()
        success_response.text = '{"is_ai": false, "verdict_type": "PROCESSED_HUMAN", "confidence_score": 0.9, "reasons": ["Human vocal"], "detailed_explanation": "Natural"}'
        mock_client.models.generate_content.side_effect = [
            Exception("503 UNAVAILABLE: This model is currently experiencing high demand"),
            success_response,
        ]

        with patch.object(self.classifier, "_get_client", return_value=mock_client):
            res = self.classifier.classify_track(
                track_id="test_retry",
                track_title="Test Retry Song",
            )
            self.assertEqual(res.verdict_type, VerdictType.PROCESSED_HUMAN)
            self.assertEqual(mock_client.models.generate_content.call_count, 2)
            mock_sleep.assert_called()


if __name__ == "__main__":
    unittest.main()
