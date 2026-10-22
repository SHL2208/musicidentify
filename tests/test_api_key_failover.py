import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm root vào sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.core.models.entities import AcousticMetrics, VerdictType
from backend.services.gemini_classifier_service import GeminiClassifierService as GeminiClassifier


class TestApiKeyFailover(unittest.TestCase):
    def setUp(self):
        self.metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            spectral_rolloff_95_hz=16000.0,
            spectral_rolloff_85_hz=12000.0,
            stereo_correlation=0.92,
            harmonic_ratio=0.35,
            compression_source="YouTube Opus / Web Streaming",
            fakeprint_score=0.88,
            fakeprint_detected=True,
        )
        self.dummy_audio = Path(root_dir / "data" / "temp" / "dummy_audio.wav")
        self.dummy_audio.parent.mkdir(parents=True, exist_ok=True)
        self.dummy_audio.touch(exist_ok=True)

        self.dummy_spec = Path(root_dir / "data" / "temp" / "dummy_spec.png")
        self.dummy_spec.touch(exist_ok=True)

    def tearDown(self):
        if self.dummy_audio.exists():
            self.dummy_audio.unlink(missing_ok=True)
        if self.dummy_spec.exists():
            self.dummy_spec.unlink(missing_ok=True)

    def test_classifier_key_ordering(self):
        """Kiểm tra thứ tự khởi tạo API key: 1 > 2 > 3."""
        keys = ["KEY_ALPHA", "KEY_BETA", "KEY_GAMMA"]
        classifier = GeminiClassifier(api_keys=keys)
        self.assertEqual(classifier.api_keys, ["KEY_ALPHA", "KEY_BETA", "KEY_GAMMA"])
        self.assertEqual(classifier.api_key, "KEY_ALPHA")

    def test_failover_key1_fails_key2_succeeds(self):
        """Kiểm tra khi Key 1 bị lỗi (ví dụ: hết Quota 429), tự động chuyển sang Key 2 thành công."""
        call_history = []

        def mock_client_factory(api_key):
            client = MagicMock()
            if api_key == "KEY_1":
                # Giả lập lỗi ở Key 1 khi generate_content
                client.files.upload.return_value = MagicMock(name="uploaded_file_key1")
                client.models.generate_content.side_effect = Exception(
                    "429 ResourceExhausted: Quota exceeded for KEY_1"
                )
            elif api_key == "KEY_2":
                # Key 2 thành công
                client.files.upload.return_value = MagicMock(name="uploaded_file_key2")
                resp = MagicMock()
                resp.text = '{"verdict_type": "AI_GENERATED", "is_ai": true, "confidence_score": 0.95, "suspected_engine": "Suno v3.5", "reasons": ["Giọng nhựa AI", "Stem bleed"], "detailed_explanation": "Phát hiện dấu vết AI rõ rệt."}'
                client.models.generate_content.return_value = resp
            else:
                raise ValueError("Unexpected key")

            original_gen = client.models.generate_content

            def recorded_gen(*args, **kwargs):
                call_history.append(api_key)
                return original_gen(*args, **kwargs)

            client.models.generate_content = recorded_gen
            return client

        classifier = GeminiClassifier(api_keys=["KEY_1", "KEY_2", "KEY_3"])

        with patch.object(classifier, "_get_client", side_effect=mock_client_factory):
            result = classifier.classify_track(
                track_id="test_01",
                track_title="Demo Track",
                source_url="https://youtube.com/watch?v=123",
                audio_path=self.dummy_audio,
                spectrogram_path=self.dummy_spec,
                metrics=self.metrics,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.verdict_type, VerdictType.AI_GENERATED)
        self.assertEqual(result.confidence_score, 0.95)
        # Kiểm tra thứ tự gọi: KEY_1 thất bại -> chuyển sang KEY_2 và dừng (không cần gọi KEY_3)
        self.assertEqual(call_history, ["KEY_1", "KEY_2"])
        print("\n✅ TEST PASSED: Key 1 lỗi -> Key 2 tự động tiếp quản thành công!")

    def test_failover_key1_and_2_fail_key3_succeeds(self):
        """Kiểm tra khi cả Key 1 và Key 2 bị lỗi, Key 3 tiếp quản thành công."""
        call_history = []

        def mock_client_factory(api_key):
            client = MagicMock()
            if api_key in ("KEY_1", "KEY_2"):
                client.files.upload.return_value = MagicMock(name="uploaded_file")
                client.models.generate_content.side_effect = Exception(f"API Error on {api_key}")
            elif api_key == "KEY_3":
                client.files.upload.return_value = MagicMock(name="uploaded_file_key3")
                resp = MagicMock()
                resp.text = '{"verdict_type": "CONFIRMED_HUMAN", "is_ai": false, "confidence_score": 0.92, "suspected_engine": "None", "reasons": ["Vocal tự nhiên"], "detailed_explanation": "Giọng người thật."}'
                client.models.generate_content.return_value = resp

            original_gen = client.models.generate_content

            def recorded_gen(*args, **kwargs):
                call_history.append(api_key)
                return original_gen(*args, **kwargs)

            client.models.generate_content = recorded_gen
            return client

        classifier = GeminiClassifier(api_keys=["KEY_1", "KEY_2", "KEY_3"])

        with patch.object(classifier, "_get_client", side_effect=mock_client_factory):
            result = classifier.classify_track(
                track_id="test_02",
                track_title="Demo Human Song",
                source_url=None,
                audio_path=self.dummy_audio,
                spectrogram_path=self.dummy_spec,
                metrics=self.metrics,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.verdict_type, VerdictType.CONFIRMED_HUMAN)
        self.assertEqual(call_history, ["KEY_1", "KEY_2", "KEY_3"])
        print("✅ TEST PASSED: Key 1 và Key 2 lỗi -> Key 3 tự động tiếp quản thành công!")

    def test_all_keys_fail_raises_runtime_error(self):
        """Kiểm tra khi tất cả 3 key đều lỗi thì báo RuntimeError chi tiết."""

        def mock_client_factory(api_key):
            client = MagicMock()
            client.files.upload.return_value = MagicMock(name="uploaded_file")
            client.models.generate_content.side_effect = Exception(f"Total outage on {api_key}")
            return client

        classifier = GeminiClassifier(api_keys=["KEY_1", "KEY_2", "KEY_3"])

        with patch.object(classifier, "_get_client", side_effect=mock_client_factory):
            with self.assertRaises(RuntimeError) as ctx:
                classifier.classify_track(
                    track_id="test_03",
                    track_title="Demo Broken Song",
                    source_url=None,
                    audio_path=self.dummy_audio,
                    spectrogram_path=self.dummy_spec,
                    metrics=self.metrics,
                )
            self.assertIn("Tất cả 3 Gemini API Keys đều thất bại", str(ctx.exception))
        print("✅ TEST PASSED: Tất cả 3 key lỗi -> Báo lỗi đầy đủ chính xác!")

    def test_multi_model_failover_gemini38_to_37(self):
        """Kiểm tra khi Gemini 3.8 Flash cạn quota/lỗi ở cả 3 key, tự động chuyển sang Gemini 3.7 Flash (Key 1)."""
        call_log = []

        def mock_client_factory(api_key):
            client = MagicMock()

            def fake_generate_content(model, contents, config):
                call_log.append((model, api_key))
                if model == "gemini-3.8-flash":
                    raise Exception(
                        f"429 ResourceExhausted: Quota exceeded for {model} on {api_key}"
                    )
                elif model == "gemini-3.7-flash" and api_key == "KEY_1":
                    resp = MagicMock()
                    resp.text = '{"verdict_type": "AI_GENERATED", "is_ai": true, "confidence_score": 0.96, "suspected_engine": "Suno v3.5", "reasons": ["AI artifact"], "detailed_explanation": "Thẩm định qua 3.7 Flash"}'
                    return resp
                raise Exception("Unexpected call")

            client.models.generate_content.side_effect = fake_generate_content
            return client

        classifier = GeminiClassifier(
            api_keys=["KEY_1", "KEY_2", "KEY_3"],
            fallback_models=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        )

        with patch.object(classifier, "_get_client", side_effect=mock_client_factory):
            result = classifier.classify_track(
                track_id="test_model_failover_1",
                track_title="Test Model Failover Track",
                audio_path=self.dummy_audio,
                spectrogram_path=self.dummy_spec,
                metrics=self.metrics,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.model_used, "gemini-3.7-flash")
        # Kiểm tra đúng thứ tự: 3.8 (KEY_1, KEY_2, KEY_3) -> 3.7 (KEY_1)
        expected_calls = [
            ("gemini-3.8-flash", "KEY_1"),
            ("gemini-3.8-flash", "KEY_2"),
            ("gemini-3.8-flash", "KEY_3"),
            ("gemini-3.7-flash", "KEY_1"),
        ]
        self.assertEqual(call_log, expected_calls)
        print(
            "✅ TEST PASSED: Gemini 3.8 Flash (3 keys lỗi) -> Tự động chuyển Gemini 3.7 Flash thành công!"
        )

    def test_multi_model_failover_gemini38_and_37_to_36(self):
        """Kiểm tra khi cả 3.8 và 3.7 đều cạn quota ở cả 3 key, tự động chuyển sang 3.6 Flash."""
        call_log = []

        def mock_client_factory(api_key):
            client = MagicMock()

            def fake_generate_content(model, contents, config):
                call_log.append((model, api_key))
                if model in ("gemini-3.8-flash", "gemini-3.7-flash"):
                    raise Exception(f"429 Quota exhausted for {model} on {api_key}")
                elif model == "gemini-3.6-flash":
                    if api_key == "KEY_1":
                        raise Exception("KEY_1 temporary glitch")
                    elif api_key == "KEY_2":
                        resp = MagicMock()
                        resp.text = '{"verdict_type": "CONFIRMED_HUMAN", "is_ai": false, "confidence_score": 0.94, "suspected_engine": "None", "reasons": ["Organic breath"], "detailed_explanation": "Thẩm định qua 3.6 Flash"}'
                        return resp
                raise Exception("Unexpected call")

            client.models.generate_content.side_effect = fake_generate_content
            return client

        classifier = GeminiClassifier(
            api_keys=["KEY_1", "KEY_2", "KEY_3"],
            fallback_models=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        )

        with patch.object(classifier, "_get_client", side_effect=mock_client_factory):
            result = classifier.classify_track(
                track_id="test_model_failover_2",
                track_title="Test Deep Failover Track",
                audio_path=self.dummy_audio,
                spectrogram_path=self.dummy_spec,
                metrics=self.metrics,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.model_used, "gemini-3.6-flash")
        # Kiểm tra 3 cuộc gọi cho 3.8, 3 cuộc gọi cho 3.7, và 2 cuộc gọi cho 3.6 (Key 1 lỗi, Key 2 thành công)
        expected_calls = [
            ("gemini-3.8-flash", "KEY_1"),
            ("gemini-3.8-flash", "KEY_2"),
            ("gemini-3.8-flash", "KEY_3"),
            ("gemini-3.7-flash", "KEY_1"),
            ("gemini-3.7-flash", "KEY_2"),
            ("gemini-3.7-flash", "KEY_3"),
            ("gemini-3.6-flash", "KEY_1"),
            ("gemini-3.6-flash", "KEY_2"),
        ]
        self.assertEqual(call_log, expected_calls)
        print(
            "✅ TEST PASSED: Gemini 3.8 Flash (3 keys) + 3.7 Flash (3 keys) lỗi -> Chuyển 3.6 Flash (Key 2) thành công!"
        )

    def test_resolve_model_chain_ordering(self):
        """Kiểm tra thứ tự resolve mô hình khi override hoặc dùng mặc định."""
        classifier = GeminiClassifier(
            api_keys=["KEY_1"],
            fallback_models=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        )
        # Mặc định bắt đầu từ 3.8 -> 3.7 -> 3.6
        self.assertEqual(
            classifier._resolve_model_chain(),
            ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        )
        # Nếu chọn bắt đầu từ 3.7 -> lùi về 3.6
        self.assertEqual(
            classifier._resolve_model_chain(model_override="gemini-3.7-flash"),
            ["gemini-3.7-flash", "gemini-3.6-flash"],
        )
        # Nếu chỉ định model tùy biến -> đưa lên đầu rồi nối tiếp chuỗi fallback
        self.assertEqual(
            classifier._resolve_model_chain(model_override="custom-model"),
            ["custom-model", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        )
        print("✅ TEST PASSED: Chuỗi thứ tự mô hình _resolve_model_chain hoạt động chính xác!")


if __name__ == "__main__":
    unittest.main()
