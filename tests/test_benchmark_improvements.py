"""Bộ kiểm thử cho kế hoạch cải thiện độ chính xác Benchmark & Detector (TDD).

Kiểm tra:
1. Hiệu chỉnh đường cong Fakeprint (Fakeprint Rescaling) nhạy hơn với AI thế hệ mới (prominence 0.10 - 0.14).
2. Xử lý tính toàn vẹn xác thực (Resolver Verification Integrity) để tránh đánh giá nhầm audio sai nguồn.
3. System Instruction của Gemini được bổ sung các tiêu chí nhận diện AI thế hệ mới (Suno v3.5/v4, Udio).

Tuân thủ nghiêm ngặt PEP 8 và Ruff.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from backend.core.models.entities import AcousticMetrics, InputType, TrackItem, VerdictType
from backend.core.utils import sanitize_vietnamese_ai_keyword
from backend.services.fakeprint_detector_service import FakeprintDetectorService
from backend.services.gemini_classifier_service import (
    SYSTEM_INSTRUCTION,
)
from backend.services.resolver_service import InputResolverService
from backend.services.signal_analyzer_service import SignalAnalyzerService
from benchmark_evaluator import run_benchmark


class TestBenchmarkAccuracyImprovements(unittest.TestCase):
    """Kiểm thử tự động cho các cải tiến phát hiện AI và đối soát benchmark."""

    def setUp(self) -> None:
        self.detector = FakeprintDetectorService()
        self.resolver = InputResolverService()

    def test_fakeprint_moderate_prominence_is_not_suppressed(self) -> None:
        """Kiểm tra độ lồi chu kỳ vừa phải (0.10 - 0.14) của AI thế hệ mới không bị dìm về 0%."""
        # Giả lập dạng sóng có đỉnh răng lược Fourier ở mức vừa phải (prominence ~ 0.12)
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Tín hiệu nền + sóng hài chu kỳ giả lập AI
        y = 0.3 * np.sin(2 * np.pi * 440 * t)
        # Thêm các vạch răng lược tuần hoàn nhân tạo ở dải 6kHz - 14kHz
        for freq in range(6000, 14000, 400):
            y += 0.05 * np.sin(2 * np.pi * freq * t)

        res = self.detector.analyze_waveform(y, sr=sr)
        # Với đường cong hiệu chỉnh mới, nếu phát hiện vết răng lược vừa phải,
        # điểm không được giậm chân tại mức < 0.25 (đặc trưng của thuật toán cũ quá gắt)
        comb_prom = res.get("comb_prominence", 0.0)
        score = res.get("fakeprint_score", 0.0)

        if comb_prom >= 0.10:
            self.assertGreaterEqual(
                score,
                0.40,
                f"LỖI: Độ lồi răng lược {comb_prom:.3f} bị dìm điểm Fakeprint xuống quá thấp ({score * 100:.1f}%)!",
            )

    def test_resolver_validate_match_rejects_irrelevant_audio(self) -> None:
        """Kiểm tra bộ lọc xác thực nguồn bài hát từ chối các video hoàn toàn lệch nội dung."""
        target_title = "Neon Summer Boulevard"
        target_artist = "Dark Suiltion"

        # Ứng viên YouTube hoàn toàn lạc đề (như podcast hoặc ca sĩ khác)
        irrelevant_cand = {
            "title": "10 nghịch lý cuộc sống - Kent Keith [Audio full]",
            "uploader": "Sách Nói Hay",
            "duration": 1800,
            "url": "https://www.youtube.com/watch?v=ydH71zgm0_Q",
        }

        is_match, score = self.resolver.validate_match(
            target_title=target_title,
            target_artist=target_artist,
            candidate=irrelevant_cand,
            target_duration=186,
        )

        self.assertFalse(is_match, "LỖI: Video lạc đề không được xem là trùng khớp!")
        self.assertLess(score, 0.40, "Điểm tương đồng của video lạc đề phải < 0.40")

    def test_gemini_system_instruction_contains_modern_ai_guidelines(self) -> None:
        """System instruction phải chứa hướng dẫn nhận diện lỗi đặc trưng của AI thế hệ mới."""
        instruction_lower = SYSTEM_INSTRUCTION.lower()

        # Phải nhắc đến các lỗi vi mô của mô hình sinh khuếch tán âm thanh (latent diffusion)
        has_diffusion_or_codec = (
            "diffusion" in instruction_lower
            or "codec" in instruction_lower
            or "neural" in instruction_lower
        )
        self.assertTrue(
            has_diffusion_or_codec,
            "LỖI: System Instruction thiếu định danh công nghệ Neural Audio Codec / Latent Diffusion!",
        )

        # Phải nhắc đến hiện tượng phrasing / phát âm / nuốt chữ cơ học của AI
        has_phrasing_guideline = (
            "phrasing" in instruction_lower
            or "consonant" in instruction_lower
            or "slur" in instruction_lower
            or "pronunciation" in instruction_lower
        )
        self.assertTrue(
            has_phrasing_guideline,
            "LỖI: System Instruction cần có hướng dẫn phân tích phát âm và phrasing của AI!",
        )

    def test_benchmark_flags_unverified_resolution_mismatch(self) -> None:
        """Benchmark phải gắn cờ cảnh báo khi video tải về không trùng khớp bài hát mục tiêu."""
        # Giả lập mẫu kiểm thử mà resolver không tìm thấy nguồn khớp (is_verified_match = False)
        sample = [
            {
                "media_id": "row_999",
                "title": "Neon Summer Boulevard",
                "artist": "Dark Suiltion",
                "query": "Neon Summer Boulevard Dark Suiltion",
                "gt_is_ai": True,
                "gt_label": "AI",
                "row_idx": 999,
            }
        ]

        mock_track = TrackItem(
            id="test1234",
            raw_query="Neon Summer Boulevard Dark Suiltion",
            input_type=InputType.SONG_NAME,
            resolved_url="https://youtube.com/watch?v=unrelated",
            track_title="10 nghich ly cuoc song",
            is_verified_match=False,
            match_score=0.15,
        )

        mock_detection_result = MagicMock()
        mock_detection_result.is_ai = False  # Audio người của video lạc đề
        mock_detection_result.verdict_label = "PROCESSED HUMAN (Người làm)"
        mock_detection_result.confidence_score = 0.95
        mock_detection_result.verdict_type = VerdictType.PROCESSED_HUMAN
        mock_detection_result.reasons = ["Người thật đọc sách"]
        mock_detection_result.suspected_engine = "None"
        mock_detection_result.source_url = "https://youtube.com/watch?v=unrelated"

        with (
            patch("benchmark_evaluator.get_container") as mock_get_container,
            patch("sys.stdout", new_callable=MagicMock),
        ):
            mock_container = MagicMock()
            mock_container.resolver.resolve_track.return_value = mock_track
            mock_container.detection_service.detect_single.return_value = mock_detection_result
            mock_get_container.return_value = mock_container

            report = run_benchmark(sample)

            # Benchmark phải có trường ghi nhận số bài không tìm thấy nguồn chuẩn xác (unverified_matches)
            self.assertIn(
                "unverified_matches",
                report,
                "LỖI: Báo cáo benchmark phải thống kê 'unverified_matches' để cảnh báo sai lệch nguồn âm thanh!",
            )
            self.assertEqual(
                report["unverified_matches"],
                1,
                "LỖI: Phải đếm được 1 bài không được xác minh nguồn!",
            )

    def test_sanitize_vietnamese_ai_keyword(self) -> None:
        """Kiểm tra việc làm sạch từ khóa 'ai' tiếng Việt khỏi thiên kiến của LLM."""
        cases = [
            ("NỖI BUỒN MANG TÊN AI", "NỖI BUỒN MANG TÊN [***]"),
            ("ai yêu ai hơn", "[***] yêu [***] hơn"),
            ("Có ai ở đó không?", "Có [***] ở đó không?"),
            ("BÀI HÁT HAY", "BÀI HÁT HAY"),
            ("Khái niệm", "Khái niệm"),
            ("Giai điệu", "Giai điệu"),
        ]
        for inp, expected in cases:
            self.assertEqual(sanitize_vietnamese_ai_keyword(inp), expected)

    def test_system_instruction_broad_ai_and_hpss_rules(self) -> None:
        """System instruction phải chứa quy tắc Broad AI, bóc tách HPSS và phân biệt Auto-Tune."""
        self.assertIn("HYBRID_AI", SYSTEM_INSTRUCTION)
        self.assertIn("[HYBRID AI]: Thành phần Vocal", SYSTEM_INSTRUCTION)
        self.assertIn("Tier 2: Harmonic Vocal", SYSTEM_INSTRUCTION)
        self.assertIn("Melodyne", SYSTEM_INSTRUCTION)
        self.assertIn("Auto-Tune", SYSTEM_INSTRUCTION)

    def test_resilient_sampling_pool_in_benchmark_evaluator(self) -> None:
        """Kiểm tra cơ chế tự động nạp bù mẫu cùng nhãn khi bài gốc gặp lỗi."""
        # 1 bài gốc AI gặp lỗi
        failing_sample = [
            {
                "media_id": "row_vip_error",
                "title": "VIP Song",
                "artist": "VIP Singer",
                "query": "VIP Song VIP Singer",
                "gt_is_ai": True,
                "gt_label": "AI",
                "row_idx": 10,
            }
        ]
        # Dataset dự phòng có 1 bài thay thế cùng nhãn AI
        backup_dataset = [
            failing_sample[0],
            {
                "media_id": "row_backup_ai",
                "title": "Backup AI Track",
                "artist": "AI Artist",
                "query": "Backup AI Track AI Artist",
                "gt_is_ai": True,
                "gt_label": "AI",
                "row_idx": 11,
            },
        ]

        mock_detection_success = MagicMock()
        mock_detection_success.is_ai = True
        mock_detection_success.verdict_label = "AI GENERATED"
        mock_detection_success.confidence_score = 0.99
        mock_detection_success.verdict_type = VerdictType.AI_GENERATED
        mock_detection_success.reasons = ["Synthetic vocal sheen"]
        mock_detection_success.suspected_engine = "Suno"
        mock_detection_success.source_url = "https://youtube.com/watch?v=backup"

        def mock_resolve_track(title, artist=""):
            if title == "VIP Song":
                raise RuntimeError("Zing MP3 VIP download error")
            return TrackItem(
                id="resolved_backup",
                raw_query=f"{title} {artist}",
                input_type=InputType.SONG_NAME,
                resolved_url="https://youtube.com/watch?v=backup",
                track_title=title,
                is_verified_match=True,
                match_score=0.98,
            )

        with (
            patch("benchmark_evaluator.get_container") as mock_get_container,
            patch("sys.stdout", new_callable=MagicMock),
        ):
            mock_container = MagicMock()
            mock_container.resolver.resolve_track.side_effect = mock_resolve_track
            mock_container.detection_service.detect_single.return_value = mock_detection_success
            mock_get_container.return_value = mock_container

            report = run_benchmark(samples=failing_sample, dataset=backup_dataset)

            # Phải hoàn thành mục tiêu 1 bài thành công nhờ nạp bù
            self.assertEqual(report["metrics"]["evaluated_successfully"], 1)
            self.assertEqual(report["metrics"]["errors"], 1)
            self.assertEqual(len(report["metrics"]["error_records"]), 1)
            err_entry = report["metrics"]["error_records"][0]
            self.assertEqual(err_entry["Tên bài lỗi (Song Title)"], "VIP Song")
            self.assertEqual(
                err_entry["Bài nạp bù thay thế (Replacement)"], "Backup AI Track AI Artist"
            )

    def test_forensic_plot_3_tiers(self) -> None:
        """Kiểm tra việc tạo ảnh phổ 3 tầng (3-tier forensic plot) chống che mờ âm học."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        y_harm = 0.4 * np.sin(2 * np.pi * 440 * t)

        metrics = AcousticMetrics(
            sample_rate=sr,
            duration_sec=duration,
            spectral_rolloff_85_hz=4000.0,
            spectral_rolloff_95_hz=8000.0,
            high_freq_ratio_16k=0.08,
            compression_source="Lossless",
        )
        fakeprint_res = {
            "freqs": np.linspace(0, 10000, 100),
            "residual_spectrum": np.random.rand(100),
            "fakeprint_score": 0.85,
            "comb_prominence": 0.15,
        }

        fakeprint_mock = MagicMock()
        storage_mock = MagicMock()
        storage_mock.get_temp_dir.return_value = Path("edge_temp")
        analyzer = SignalAnalyzerService(
            fakeprint_detector=fakeprint_mock, storage_repo=storage_mock
        )
        out_path = analyzer._generate_forensic_plot(
            mono_y=y,
            mono_harm=y_harm,
            sr=sr,
            audio_stem="test_tier3",
            metrics=metrics,
            fakeprint_res=fakeprint_res,
        )

        self.assertIsNotNone(out_path)
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 5000)

        # Dọn dẹp file và thư mục tạm
        if out_path.exists():
            out_path.unlink()
        edge_dir = Path("edge_temp")
        if edge_dir.exists():
            try:
                edge_dir.rmdir()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
