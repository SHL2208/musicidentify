import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.core.config import ATTACH_SPECTROGRAM_TO_LLM
from backend.core.container import get_container
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.services.local_llm_classifier_service import LocalLLMClassifierService


class TestAsyncSpectrogramAndPureText(unittest.TestCase):
    """Kiểm thử cơ chế vẽ ảnh phổ bất đồng bộ và Pure Text Mode cho LLM Classifier."""

    def test_signal_analyzer_async_plot_returns_future_and_saves_png(self):
        """Kiểm tra analyze_audio khi async_plot=True trả về Future và vẽ ra file PNG hợp lệ trên đĩa."""
        audio_files = list(Path("data/audio_cache").glob("*.wav"))
        if not audio_files:
            self.skipTest("Không có file test audio trong data/audio_cache")

        test_audio = audio_files[0]
        container = get_container()
        analyzer = container.signal_analyzer

        metrics, plot_future_or_path = analyzer.analyze_audio(test_audio, async_plot=True)

        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, AcousticMetrics)
        # Xác nhận trả về Future (bất đồng bộ)
        self.assertTrue(hasattr(plot_future_or_path, "result"))

        # Thu nhận kết quả render
        image_path = plot_future_or_path.result(timeout=10)
        self.assertIsNotNone(image_path)
        self.assertTrue(Path(image_path).exists())
        self.assertGreater(Path(image_path).stat().st_size, 10000)  # PNG > 10KB

    def test_local_llm_classifier_pure_text_payload_when_disabled(self):
        """Kiểm tra khi ATTACH_SPECTROGRAM_TO_LLM=False, payload gửi LLM tuyệt đối không chứa image_url."""
        service = LocalLLMClassifierService(
            base_url="http://localhost:20128/v1",
            api_key="test_key",
            model_name="antigravity/gemini-3.8-flash-high",
        )

        captured_payload = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal captured_payload
            captured_payload = json
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict_type": "CONFIRMED_HUMAN", "is_ai": false, "confidence_score": 0.95, "reasons": ["Giọng mộc"], "detailed_explanation": "Chi tiết"}'
                        }
                    }
                ]
            }
            return mock_resp

        dummy_spec = Path("temp/dummy_spec.png")
        dummy_spec.parent.mkdir(parents=True, exist_ok=True)
        dummy_spec.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        try:
            with patch("backend.services.local_llm_classifier_service.ATTACH_SPECTROGRAM_TO_LLM", False):
                with patch("requests.post", side_effect=mock_post):
                    service.classify_track(
                        track_id="test_track",
                        track_title="Test Song",
                        spectrogram_path=dummy_spec,
                    )

            # Kiểm tra nội dung tin nhắn gửi đi
            messages = captured_payload.get("messages", [])
            self.assertGreaterEqual(len(messages), 2)
            user_msg = messages[-1]
            user_content = user_msg.get("content", [])

            # Phải là list chứa đúng 1 item text, KHÔNG ĐƯỢC có image_url
            item_types = [item.get("type") for item in user_content if isinstance(item, dict)]
            self.assertNotIn("image_url", item_types)
            self.assertIn("text", item_types)
            self.assertEqual(len(user_content), 1)
        finally:
            if dummy_spec.exists():
                dummy_spec.unlink()

    def test_local_llm_classifier_attaches_image_when_enabled(self):
        """Kiểm tra khi ATTACH_SPECTROGRAM_TO_LLM=True, payload gửi LLM có đính kèm image_url."""
        service = LocalLLMClassifierService(
            base_url="http://localhost:20128/v1",
            api_key="test_key",
            model_name="antigravity/gemini-3.8-flash-high",
        )

        captured_payload = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal captured_payload
            captured_payload = json
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict_type": "CONFIRMED_HUMAN", "is_ai": false, "confidence_score": 0.95, "reasons": ["Giọng mộc"], "detailed_explanation": "Chi tiết"}'
                        }
                    }
                ]
            }
            return mock_resp

        dummy_spec = Path("temp/dummy_spec_enabled.png")
        dummy_spec.parent.mkdir(parents=True, exist_ok=True)
        dummy_spec.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        try:
            with patch("backend.services.local_llm_classifier_service.ATTACH_SPECTROGRAM_TO_LLM", True):
                with patch("requests.post", side_effect=mock_post):
                    service.classify_track(
                        track_id="test_track",
                        track_title="Test Song",
                        spectrogram_path=dummy_spec,
                    )

            messages = captured_payload.get("messages", [])
            user_msg = messages[-1]
            user_content = user_msg.get("content", [])

            item_types = [item.get("type") for item in user_content if isinstance(item, dict)]
            self.assertIn("image_url", item_types)
        finally:
            if dummy_spec.exists():
                dummy_spec.unlink()

    def test_detection_service_always_persists_spectrogram_in_result(self):
        """Xác nhận DetectionService luôn luôn gán và lưu spectrogram_image_path đầy đủ trong kết quả."""
        audio_files = list(Path("data/audio_cache").glob("*.wav"))
        if not audio_files:
            self.skipTest("Không có file test audio trong data/audio_cache")

        test_audio = audio_files[0]
        container = get_container()
        detection_svc = container.detection_service

        mock_classifier = MagicMock()
        mock_classifier.classify_track.return_value = DetectionResult(
            track_id="test_detect_id",
            track_title="Test Song",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.99,
            suspected_engine="None",
            reasons=["Động lực thanh quản sinh học"],
            detailed_explanation="Bản thu mộc tự nhiên.",
        )

        track = TrackItem(
            id="test_detect_id",
            raw_query=str(test_audio),
            input_type=InputType.LOCAL_FILE,
            track_title="Test Local Audio",
        )

        with patch.object(detection_svc, "classifier", mock_classifier):
            res = detection_svc.detect_single(track, force_rescan=True)

        # Khẳng định tuyệt đối: Người dùng VẪN CÓ ẢNH ĐẦY ĐỦ trong kết quả!
        self.assertIsNotNone(res.spectrogram_image_path)
        self.assertTrue(Path(res.spectrogram_image_path).exists())
        self.assertEqual(Path(res.spectrogram_image_path).suffix, ".png")


if __name__ == "__main__":
    unittest.main()
