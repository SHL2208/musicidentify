"""
Kiểm thử tự động cho Chiến lược 5: Đa mẫu âm thanh thông minh (Multi-Segment Adaptive Slicing)
và Tái cấu trúc Prompt thẩm âm '3 Lát Cắt - 3 Nhiệm Vụ'.

Tuân thủ nghiêm ngặt TDD (Test-Driven Development), PEP 8 và Ruff.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from backend.core.config import (
    MULTI_SEGMENT_CHORUS_DURATION_SEC,
    MULTI_SEGMENT_INTRO_DURATION_SEC,
    MULTI_SEGMENT_TOTAL_DURATION_SEC,
    MULTI_SEGMENT_VERSE_DURATION_SEC,
    get_ffmpeg_binary,
)
from backend.repositories.temp_storage_repository import TempStorageRepository
from backend.services.audio_engine_service import AudioEngineService
from backend.services.gemini_classifier_service import GeminiClassifierService


class TestMultiSegmentAdaptiveSlicing(unittest.TestCase):
    """Kiểm thử bộ trích xuất đa phân đoạn và cấu trúc prompt 3 lát cắt."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage_repo = TempStorageRepository(temp_dir=self.temp_dir, output_dir=self.temp_dir)
        self.audio_engine = AudioEngineService(
            storage_repo=self.storage_repo, ffmpeg_path=get_ffmpeg_binary()
        )
        self.classifier = GeminiClassifierService(api_key="mock_key", model_name="gemini-3.5-flash-lite")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_segment_constants(self) -> None:
        """Kiểm tra các hằng số phân đoạn 3 lát cắt được cấu hình chuẩn xác."""
        self.assertEqual(MULTI_SEGMENT_INTRO_DURATION_SEC, 6.0)
        self.assertEqual(MULTI_SEGMENT_VERSE_DURATION_SEC, 15.0)
        self.assertEqual(MULTI_SEGMENT_CHORUS_DURATION_SEC, 15.0)
        self.assertEqual(MULTI_SEGMENT_TOTAL_DURATION_SEC, 36.0)

    def test_slice_multisegment_extracts_intro_verse_chorus(self) -> None:
        """
        Kiểm tra hàm _slice_multisegment_sample trích xuất file âm thanh gồm 3 phân đoạn:
        - Intro: 0.0s - 6.0s
        - Verse: 15.0s (tại ~22% thời lượng)
        - Chorus: 15.0s (tại ~50% thời lượng)
        Tổng thời lượng file kết quả phải xấp xỉ 36.0 giây (+/- 0.5s).
        """
        sr = 44100
        total_sec = 180.0
        n_samples = int(sr * total_sec)
        t = np.linspace(0, total_sec, n_samples, endpoint=False)
        waveform = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo_data = np.column_stack([waveform, waveform])

        input_audio = self.temp_dir / "test_full_track_180s.wav"
        sf.write(str(input_audio), stereo_data, sr)

        output_path = self.audio_engine._slice_multisegment_sample(
            input_audio=input_audio, track_id="test_track_1", total_duration=total_sec
        )

        self.assertTrue(output_path.exists())
        info = sf.info(str(output_path))
        self.assertAlmostEqual(info.duration, 36.0, delta=0.5)

    def test_gemini_prompt_has_3segment_rubric(self) -> None:
        """Kiểm tra cả 2 phiên bản Tiếng Việt và Tiếng Anh của prompt Flash-Lite chứa cấu trúc 3 lát cắt."""
        prompt_vi = self.classifier._build_flash_lite_checklist_prompt(
            track_id="test_1", metrics=None, metadata=None, language="vi"
        )
        self.assertIn("0:00 - 0:06", prompt_vi)
        self.assertIn("0:06 - 0:21", prompt_vi)
        self.assertIn("0:21 - 0:36", prompt_vi)
        self.assertIn("INTRO", prompt_vi)
        self.assertIn("VERSE", prompt_vi)
        self.assertIn("CHORUS", prompt_vi)

        prompt_en = self.classifier._build_flash_lite_checklist_prompt(
            track_id="test_1", metrics=None, metadata=None, language="en"
        )
        self.assertIn("0:00 - 0:06", prompt_en)
        self.assertIn("0:06 - 0:21", prompt_en)
        self.assertIn("0:21 - 0:36", prompt_en)
        self.assertIn("INTRO", prompt_en)
        self.assertIn("VERSE", prompt_en)
        self.assertIn("CHORUS", prompt_en)

    def test_gemini_classify_track_preserves_full_36s_audio(self) -> None:
        """Kiểm tra classify_track không cắt mù về 18s mà gửi trọn vẹn composite audio 36s lên Gemini."""
        sr = 44100
        sec = 36.0
        t = np.linspace(0, sec, int(sr * sec), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_stereo = np.column_stack([audio, audio])
        audio_file = self.temp_dir / "sample_36s.wav"
        sf.write(str(audio_file), audio_stereo, sr)

        captured_contents = []

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "is_ai": False,
            "verdict_type": "CONFIRMED_HUMAN",
            "confidence_score": 0.95,
            "suspected_engine": "None",
            "reasons": ["[BẢO VỆ NHẠC NGƯỜI]: Giọng hát mộc tự nhiên trên cả 3 phân đoạn."],
            "detailed_explanation": "Thẩm định 36s cho thấy cấu trúc phòng thu thật.",
            "work_type": "ORIGINAL",
            "is_cover": False
        })

        def fake_generate_content(model, contents, config):
            captured_contents.extend(contents)
            return mock_response

        mock_client.models.generate_content.side_effect = fake_generate_content

        with patch.object(self.classifier, "_get_client", return_value=mock_client):
            res = self.classifier.classify_track(
                track_id="t36",
                track_title="Test 36s Track",
                audio_path=audio_file,
                spectrogram_path=None,
                metrics=None
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.verdict_label, "CONFIRMED HUMAN (Người làm)")
        # Kiểm tra payload chứa Part audio và dung lượng Part audio tương ứng file MP3 36s (> 500KB)
        audio_parts = [p for p in captured_contents if hasattr(p, "inline_data") and "audio" in getattr(p.inline_data, "mime_type", "")]
        self.assertTrue(len(audio_parts) > 0)
        data_len = len(audio_parts[0].inline_data.data)
        # 36s MP3 ở 160kbps ~ 700KB, trong khi nếu bị chặt về 18s chỉ ~350KB
        self.assertGreater(data_len, 500_000, f"Expected > 500KB for 36s MP3, got {data_len}")


if __name__ == "__main__":
    unittest.main()
