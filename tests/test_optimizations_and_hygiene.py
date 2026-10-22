"""Unit tests for performance optimizations, resolver caching, Excel sanitization, and database resilience.

Follows PEP 8, Ruff, and Clean Architecture standards.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from backend.repositories.file_report_repository import clean_excel_text
from backend.services.resolver_service import InputResolverService
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestOptimizationsAndHygiene(unittest.TestCase):
    """Test suite verifying performance enhancements and hygiene standards."""

    def setUp(self) -> None:
        self.resolver = InputResolverService()
        self.mock_fakeprint = MagicMock()
        self.mock_storage = MagicMock()
        self.mock_storage.get_temp_dir.return_value = Path("data/temp")
        self.analyzer = SignalAnalyzerService(
            fakeprint_detector=self.mock_fakeprint,
            storage_repo=self.mock_storage,
        )

    def test_resolver_cache_stores_and_reuses_results(self) -> None:
        """Kiểm tra InputResolverService lưu cache kết quả phân giải và tái sử dụng cho các lần sau."""
        query = "Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau"

        # Lần 1: Mock search_youtube và phân giải
        with (
            patch.object(
                self.resolver,
                "search_youtube",
                return_value=(
                    "https://youtu.be/fake_url",
                    "Đừng Làm Trái Tim Anh Đau",
                    210.0,
                    True,
                    0.95,
                ),
            ) as mock_yt,
            patch.object(
                self.resolver,
                "search_zingmp3",
                return_value=None,
            ),
        ):
            track_1 = self.resolver.resolve_track(query)
            self.assertEqual(track_1.resolved_url, "https://youtu.be/fake_url")
            self.assertEqual(mock_yt.call_count, 1)

            # Lần 2: Cùng query -> Phải lấy từ cache, KHÔNG gọi lại search_youtube
            track_2 = self.resolver.resolve_track(query)
            self.assertEqual(track_2.resolved_url, "https://youtu.be/fake_url")
            self.assertEqual(mock_yt.call_count, 1, "Cache phải chặn cuộc gọi tìm kiếm mạng lần 2!")

    def test_excel_formula_injection_sanitization(self) -> None:
        """Kiểm tra hàm clean_excel_text vô hiệu hóa các ký tự điều khiển công thức (=, +, -, @)."""
        malicious_inputs = [
            "=cmd|' /C calc'!A0",
            "+1+2",
            "-5+10",
            "@SUM(A1:A10)",
        ]
        for item in malicious_inputs:
            cleaned = clean_excel_text(item)
            self.assertTrue(
                cleaned.startswith("'"),
                f"Giá trị độc hại '{item}' phải được chèn tiền tố nháy đơn an toàn! Nhận được: '{cleaned}'",
            )

        # Chuỗi số học thông thường hoặc chuỗi an toàn không được bị phá hỏng
        self.assertEqual(clean_excel_text(123), 123)
        self.assertEqual(clean_excel_text("Sơn Tùng M-TP"), "Sơn Tùng M-TP")
        self.assertEqual(clean_excel_text(45.6), 45.6)

    def test_signal_analyzer_harmonic_ratio_and_rolloff_accuracy(self) -> None:
        """Kiểm tra tính toán harmonic_ratio, rolloff và các chỉ số âm học sau khi tối ưu STFT/HPSS."""
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Sóng hình sin 440Hz thuần khiết -> harmonic_ratio phải rất cao (> 0.90)
        pure_sine = 0.8 * np.sin(2 * np.pi * 440 * t)

        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        test_file = temp_dir / "test_synth_opt.wav"
        sf.write(str(test_file), pure_sine, sr)

        try:
            self.mock_fakeprint.analyze_waveform.return_value = {
                "fakeprint_score": 0.05,
                "fakeprint_detected": False,
                "residual_spectrum": np.array([]),
                "freqs": np.array([]),
            }

            metrics, img_path = self.analyzer.analyze_audio(test_file)
            self.assertGreater(
                metrics.harmonic_ratio, 0.85, "Sóng sin thuần phải có tỷ lệ sóng hài cao!"
            )
            self.assertEqual(metrics.sample_rate, 44100)
            self.assertFalse(metrics.fakeprint_detected)
            self.assertIsNotNone(metrics.spectral_rolloff_95_hz)
            self.assertGreater(metrics.spectral_rolloff_95_hz, 0.0)
        finally:
            if test_file.exists():
                test_file.unlink(missing_ok=True)
            if img_path and Path(img_path).exists():
                Path(img_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
