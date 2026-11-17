"""Kiểm thử tự động cho các khắc phục cốt lõi độ chính xác Benchmark & AI Detector:
1. Xác nhận triệt tiêu hoàn toàn Data Leakage trong phân loại Tác phẩm (Work Type).
2. Xác nhận Nghệ sĩ AI trong Catalog được nhận diện chính xác kể cả khi stream lossy.
3. Xác nhận Ca sĩ người thật đã cập nhật Whitelist được bảo vệ khỏi False Positive.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.core.models.entities import AcousticMetrics, InputType, TrackItem, VerdictType
from backend.services.audio_engine_service import AudioEngineService
from backend.services.detection_service import DetectionService
from backend.services.signal_analyzer_service import SignalAnalyzerService
from backend.services.work_type_resolver import (
    WorkType,
    resolve_work_type_from_metadata,
)


class TestBenchmarkCoreAccuracyFixes(unittest.TestCase):
    """Kiểm tra các sửa chữa cốt lõi cho hệ thống phân loại và benchmark."""

    def test_work_type_resolution_without_ground_truth_leakage(self) -> None:
        """Kiểm tra Work Type Resolver hoạt động chuẩn xác mà không cần metadata Ground Truth."""
        # 1. Bài hát gốc thông thường
        w1, _, is_c1, _, _ = resolve_work_type_from_metadata(
            title="Lạc Trôi",
            artist="Sơn Tùng M-TP",
            metadata={},  # Hoàn toàn không truyền Song Type
        )
        self.assertEqual(w1, WorkType.ORIGINAL)
        self.assertFalse(is_c1)

        # 2. Bài hát remix có từ khóa trong title
        w2, _, is_c2, _, _ = resolve_work_type_from_metadata(
            title="Nơi Này Có Anh (Remix)",
            artist="Sơn Tùng M-TP",
            metadata={},
        )
        self.assertEqual(w2, WorkType.REMIX)

        # 3. Bài hát cover kinh điển từ catalog
        w3, _, is_c3, orig_a, orig_t = resolve_work_type_from_metadata(
            title="Tuổi Nàng 15",
            artist="Anh Ba Trứng",
            metadata={},
        )
        self.assertEqual(w3, WorkType.COVER)
        self.assertTrue(is_c3)
        self.assertEqual(orig_a, "Giao Linh")

        # 4. Bài hát cover có từ khóa tone / cover
        w4, _, is_c4, _, _ = resolve_work_type_from_metadata(
            title="Miền Trung Thức Trắng (Tone Nữ)",
            artist="Bạch Duy Sơn",
            metadata={},
        )
        self.assertEqual(w4, WorkType.COVER)
        self.assertTrue(is_c4)

    def test_catalog_ai_artist_detection(self) -> None:
        """Kiểm tra các nghệ sĩ AI ảo trong catalog được phát hiện chính xác."""
        analyzer = SignalAnalyzerService(
            fakeprint_detector=MagicMock(), storage_repo=MagicMock()
        )
        ai_test_artists = [
            "Lil Pang Wag",
            "Mokai Tunr",
            "Tokai Yussie",
            "Dame Wing Hear",
            "Picture Record",
        ]
        for artist in ai_test_artists:
            meta = {"artist": artist, "title": "Sample AI Song"}
            res = analyzer.check_heuristics(meta)
            self.assertIsNotNone(
                res, f"Nghệ sĩ AI '{artist}' trong Catalog phải được nhận diện qua Heuristics!"
            )
            self.assertTrue(res["is_ai"])
            self.assertEqual(res["verdict_type"], VerdictType.AI_GENERATED)

    def test_verified_human_artists_protected_from_ai_verdict(self) -> None:
        """Kiểm tra ca sĩ người thật (Bảo Thạch, 2Can, Chí Thành Danh...) được bảo vệ."""
        analyzer = SignalAnalyzerService(
            fakeprint_detector=MagicMock(), storage_repo=MagicMock()
        )
        human_test_artists = [
            "Chí Thành Danh",
            "Bảo Thạch",
            "2Can",
            "Lâm Chấn Kiệt",
            "Thiên Khôi",
            "Quách Beem",
        ]
        for artist in human_test_artists:
            self.assertTrue(
                analyzer.is_verified_human_artist(artist),
                f"Ca sĩ người thật '{artist}' phải được xác nhận là Human Artist!",
            )

    def test_spotify_bridge_proactive_bypass(self) -> None:
        """Kiểm tra AudioEngine chủ động chuyển đổi Spotify sang YouTube, không gọi yt-dlp DRM."""
        mock_storage = MagicMock()
        mock_storage.get_temp_dir.return_value = Path(".")
        service = AudioEngineService(storage_repo=mock_storage)

        track = TrackItem(
            id="test-spotify-bypass",
            raw_query="https://open.spotify.com/track/7GcWeFdftkUOCtiFR2FSIS",
            resolved_url="https://open.spotify.com/track/7GcWeFdftkUOCtiFR2FSIS",
            input_type=InputType.URL,
            track_title="Silence KZen Serene",
            artist="KZen",
        )

        with patch.object(service, "_download_audio") as mock_dl:
            mock_dl.return_value = (
                Path("tests/test_audio.mp3"),
                {"webpage_url": "https://youtube.com/watch?v=123"},
            )
            with patch.object(service, "_slice_multisegment_sample") as mock_slice:
                mock_slice.return_value = Path("tests/test_audio.mp3")
                with patch("backend.core.container.get_container") as mock_cont:
                    mock_res = MagicMock()
                    mock_res.get_url_metadata.return_value = ("Silence KZen Serene", 180)
                    mock_res.search_youtube.return_value = (
                        "https://youtube.com/watch?v=123",
                        "Silence KZen Serene",
                        180,
                        True,
                        0.95,
                    )
                    mock_cont.return_value.resolver = mock_res

                    path, meta = service.process_track_audio(track)
                    self.assertIsNotNone(path)
                    self.assertEqual(meta.get("fallback_platform"), "spotify_bridge")
                    # Đảm bảo KHÔNG bao giờ gọi yt-dlp trực tiếp vào link spotify.com
                    spotify_called = any(
                        "spotify.com" in call.args[0] for call in mock_dl.call_args_list
                    )
                    self.assertFalse(spotify_called, "Tuyệt đối không được gọi yt-dlp tải trực tiếp Spotify DRM!")
                    # Chỉ tải 1 lần từ YouTube
                    self.assertEqual(len(mock_dl.call_args_list), 1)
                    self.assertIn("youtube.com", mock_dl.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
