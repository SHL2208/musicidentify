"""Bộ kiểm thử tập trung cấu hình Worker và tính năng Spotify Enrichment & Early-Exit.

Kiểm thử:
1. Quy hoạch tập trung cấu hình Worker trong config.py.
2. Tiền tra cứu Spotify trong ResolverService (làm sạch Title, Artist, bổ sung Release Date).
3. Pre-download Early-Exit trong DetectionService: phát hành trước 11/2022 kết luận CONFIRMED_HUMAN ngay.
4. Không phạt tăng tỷ lệ AI khi bài hát vắng mặt trên Spotify.
5. Bài hát Remix / Cover không được phép Early-Exit kể cả khi bài gốc trước 2022.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.core.config import (
    BENCHMARK_WORKERS,
    ENABLE_SPOTIFY_EARLY_EXIT,
    ENABLE_SPOTIFY_PRE_SEARCH,
    SPECTROGRAM_PLOTTER_WORKERS,
    SPOTIFY_BATCH_WORKERS,
)
from backend.core.interfaces.services.spotify_service_interface import ISpotifyService
from backend.core.models.entities import (
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
    WorkType,
)
from backend.repositories.in_memory_track_repository import InMemoryTrackRepository
from backend.services.detection_service import DetectionService
from backend.services.resolver_service import InputResolverService


class MockSpotifyService(ISpotifyService):
    """Mock SpotifyService để kiểm thử độc lập."""

    def __init__(self, track_to_return: dict | None = None) -> None:
        self.track_to_return = track_to_return

    def search_tracks(self, query: str, artist: str = "", limit: int = 5) -> list[dict]:
        return [self.track_to_return] if self.track_to_return else []

    def get_track_by_url(self, url: str) -> dict | None:
        return self.track_to_return

    def find_best_match(self, title: str, artist: str = "") -> dict | None:
        return self.track_to_return

    def batch_search_tracks(
        self,
        queries: list[str | dict],
        max_workers: int = 3,
        on_progress: any = None,
    ) -> list[dict]:
        return [self.track_to_return] if self.track_to_return else []


class TestWorkerConfigConsolidation(unittest.TestCase):
    """Kiểm tra việc tập trung hóa cấu hình Worker trong core/config.py."""

    def test_worker_configs_exist_and_valid(self) -> None:
        """Đảm bảo các hằng số worker tồn tại, kiểu int dương và boolean hợp lệ."""
        self.assertIsInstance(SPOTIFY_BATCH_WORKERS, int)
        self.assertGreater(SPOTIFY_BATCH_WORKERS, 0)

        self.assertIsInstance(SPECTROGRAM_PLOTTER_WORKERS, int)
        self.assertGreater(SPECTROGRAM_PLOTTER_WORKERS, 0)

        self.assertIsInstance(BENCHMARK_WORKERS, int)
        self.assertGreater(BENCHMARK_WORKERS, 0)

        self.assertIsInstance(ENABLE_SPOTIFY_PRE_SEARCH, bool)
        self.assertIsInstance(ENABLE_SPOTIFY_EARLY_EXIT, bool)

    def test_signal_analyzer_uses_spectrogram_worker_config(self) -> None:
        """Đảm bảo SignalAnalyzerService sử dụng cấu hình worker từ config."""
        from backend.services.signal_analyzer_service import _plot_executor

        self.assertEqual(_plot_executor._max_workers, SPECTROGRAM_PLOTTER_WORKERS)

    def test_benchmark_evaluator_has_workers_arg(self) -> None:
        """Đảm bảo benchmark_evaluator có tham số --workers và mặc định từ config."""
        result = subprocess.run(
            [sys.executable, "benchmark_evaluator.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn("--workers", result.stdout)
        self.assertIn(f"mặc định: {BENCHMARK_WORKERS}", result.stdout)


class TestSpotifyEnrichmentAndEarlyExit(unittest.TestCase):
    """Kiểm tra tiền tra cứu Spotify và Pre-download Early-Exit."""

    def setUp(self) -> None:
        self.repo = InMemoryTrackRepository()
        self.mock_audio_engine = MagicMock()
        self.mock_signal_analyzer = MagicMock()
        self.mock_classifier = MagicMock()
        self.mock_resolver = MagicMock()

    def test_pre_download_early_exit_for_pre_2022_track(self) -> None:
        """Bài hát phát hành trước 11/2022 (vd: 2018) phải Early-Exit ngay thành CONFIRMED_HUMAN mà không cần tải audio."""
        spotify_data = {
            "id": "spot_1999",
            "title": "Chạm Đáy Nỗi Đau",
            "artist": "ERIK",
            "album": "Chạm Đáy Nỗi Đau Single",
            "release_date": "2018-04-11",
            "distributor": "Universal Music",
            "spotify_url": "https://open.spotify.com/track/spot_1999",
            "embed_url": "https://open.spotify.com/embed/track/spot_1999",
            "cover_url": "https://i.scdn.co/cover.jpg",
            "duration_sec": 260.0,
        }
        mock_spotify = MockSpotifyService(track_to_return=spotify_data)

        service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.repo,
            spotify_service=mock_spotify,
        )

        track_item = TrackItem(
            id="cham_day_noi_dau_yt",
            raw_query="Chạm Đáy Nỗi Đau ERIK",
            input_type=InputType.URL,
            track_title="Chạm Đáy Nỗi Đau",
            artist="ERIK",
            resolved_url="https://youtube.com/watch?v=mock123",
            song_type="Original",
        )

        with patch("backend.services.detection_service.ENABLE_SPOTIFY_EARLY_EXIT", True):
            result = service.detect_single(track_item)

        # Kiểm tra kết quả
        self.assertEqual(result.verdict_type, VerdictType.CONFIRMED_HUMAN)
        self.assertFalse(result.is_ai)
        self.assertEqual(result.confidence_score, 0.99)
        self.assertEqual(result.model_used, "spotify-pre-ai-rule")
        self.assertIn("Spotify", result.detailed_explanation)
        self.assertIn("2018", result.detailed_explanation)

        # RẤT QUAN TRỌNG: Audio engine KHÔNG được gọi (tiết kiệm thời gian tải YouTube)
        self.mock_audio_engine.process_track_audio.assert_not_called()
        self.mock_classifier.classify_track.assert_not_called()

    def test_no_early_exit_for_remix_even_if_original_pre_2022(self) -> None:
        """Bài hát Remix phát hành bài gốc năm 2015 nhưng bản phối mới có thể dùng AI, không được Early-Exit."""
        spotify_data = {
            "id": "spot_remix",
            "title": "Lạc Trôi",
            "artist": "Sơn Tùng M-TP",
            "album": "Lạc Trôi",
            "release_date": "2017-01-01",
            "spotify_url": "https://open.spotify.com/track/spot_remix",
        }
        mock_spotify = MockSpotifyService(track_to_return=spotify_data)

        # Mock audio engine và classifier trả về kết quả bình thường
        self.mock_audio_engine.process_track_audio.return_value = (
            None,
            {"title": "Lạc Trôi (Remix 2024)", "artist": "Sơn Tùng M-TP"},
        )
        self.mock_signal_analyzer.check_heuristics.return_value = None
        self.mock_signal_analyzer.analyze.return_value = None

        mock_classifier_res = MagicMock()
        mock_classifier_res.is_ai = False
        mock_classifier_res.verdict_type = VerdictType.PROCESSED_HUMAN
        mock_classifier_res.verdict_label = "PROCESSED HUMAN"
        mock_classifier_res.confidence_score = 0.85
        mock_classifier_res.suspected_engine = "None"
        mock_classifier_res.reasons = ["Bản phối EDM phòng thu"]
        mock_classifier_res.detailed_explanation = "Bản remix Vinahouse người phối."
        mock_classifier_res.acoustic_metrics = None
        mock_classifier_res.model_used = "gemini-3.8-flash"
        mock_classifier_res.work_type = WorkType.REMIX
        self.mock_classifier.classify_track.return_value = mock_classifier_res

        service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.repo,
            spotify_service=mock_spotify,
        )

        track_item = TrackItem(
            id="lac_troi_remix",
            raw_query="Lạc Trôi Remix Sơn Tùng",
            input_type=InputType.URL,
            track_title="Lạc Trôi (Remix 2024)",
            artist="Sơn Tùng M-TP",
            resolved_url="https://youtube.com/watch?v=mockremix",
            song_type="Remix",
        )

        with (
            patch("backend.services.detection_service.ENABLE_SPOTIFY_EARLY_EXIT", True),
            patch.object(
                service,
                "_apply_cross_validation_safeguard",
                side_effect=lambda r, *args, **kwargs: r,
            ),
        ):
            result = service.detect_single(track_item)

        # Audio engine PHẢI được gọi để phân tích bản remix
        self.mock_audio_engine.process_track_audio.assert_called_once()
        self.assertEqual(result.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_no_penalty_when_absent_from_spotify(self) -> None:
        """Nếu bài hát không có trên Spotify, không được phạt tăng tỷ lệ AI."""
        # Spotify trả về None
        mock_spotify = MockSpotifyService(track_to_return=None)

        self.mock_audio_engine.process_track_audio.return_value = (
            None,
            {"title": "Bài Ca Không Có Trên Spotify", "artist": "Ca Sĩ Độc Lập"},
        )
        self.mock_signal_analyzer.check_heuristics.return_value = None
        self.mock_signal_analyzer.analyze.return_value = None

        mock_classifier_res = MagicMock()
        mock_classifier_res.is_ai = False
        mock_classifier_res.verdict_type = VerdictType.CONFIRMED_HUMAN
        mock_classifier_res.verdict_label = "CONFIRMED HUMAN"
        mock_classifier_res.confidence_score = 0.90
        mock_classifier_res.suspected_engine = "None"
        mock_classifier_res.reasons = ["Giai điệu acoustic người thật, biểu cảm phong phú."]
        mock_classifier_res.detailed_explanation = "Chi tiết thẩm âm con người."
        mock_classifier_res.acoustic_metrics = None
        mock_classifier_res.model_used = "gemini-3.8-flash"
        mock_classifier_res.work_type = WorkType.ORIGINAL
        self.mock_classifier.classify_track.return_value = mock_classifier_res

        service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.repo,
            spotify_service=mock_spotify,
        )

        track_item = TrackItem(
            id="indie_viet",
            raw_query="Bài Ca Không Có Trên Spotify Ca Sĩ Độc Lập",
            input_type=InputType.URL,
            track_title="Bài Ca Không Có Trên Spotify",
            artist="Ca Sĩ Độc Lập",
            resolved_url="https://youtube.com/watch?v=mock_indie",
            song_type="Original",
        )

        with patch.object(
            service,
            "_apply_cross_validation_safeguard",
            side_effect=lambda r, *args, **kwargs: r,
        ):
            result = service.detect_single(track_item)

        # Xác nhận: Vắng mặt trên Spotify không biến thành AI
        self.assertFalse(result.is_ai)
        self.assertEqual(result.verdict_type, VerdictType.CONFIRMED_HUMAN)


class TestResolverSpotifyPreSearch(unittest.TestCase):
    """Kiểm tra tiền tra cứu Spotify trong InputResolverService."""

    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_resolver_enriches_metadata_from_spotify(self, mock_search_yt: MagicMock) -> None:
        """Resolver tra cứu Spotify và làm sạch title, artist, gắn release_date."""
        mock_search_yt.return_value = (
            "https://www.youtube.com/watch?v=mockyt_son_tung",
            "Sơn Tùng M-TP - Nơi Này Có Anh (Official Music Video)",
            260.0,
            True,
            0.95,
        )

        spotify_data = {
            "id": "spot_123",
            "title": "Nơi Này Có Anh",
            "artist": "Sơn Tùng M-TP",
            "album": "m-tp M-TP",
            "release_date": "2017-02-14",
            "distributor": "M-TP Entertainment",
            "spotify_url": "https://open.spotify.com/track/spot_123",
        }
        mock_spotify = MockSpotifyService(track_to_return=spotify_data)

        resolver = InputResolverService(spotify_service=mock_spotify)

        with patch("backend.services.resolver_service.ENABLE_SPOTIFY_PRE_SEARCH", True):
            track = resolver.resolve_track("noi nay co anh son tung")

        self.assertIsInstance(track, TrackItem)
        # Metadata đã được làm sạch và bổ sung từ Spotify
        self.assertEqual(track.track_title, "Nơi Này Có Anh")
        self.assertEqual(track.artist, "Sơn Tùng M-TP")
        self.assertEqual(track.release_date, "2017-02-14")
        self.assertEqual(track.distributor, "M-TP Entertainment")
        self.assertEqual(track.spotify_url, "https://open.spotify.com/track/spot_123")
        self.assertIsNotNone(track.spotify_metadata)
        self.assertEqual(track.spotify_metadata.get("id"), "spot_123")


class TestSpotifyOnOffToggle(unittest.TestCase):
    """Kiểm tra cơ chế bật/tắt Check Spotify (enable_spotify=False) xuyên suốt các tầng."""

    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_resolver_skips_spotify_when_disabled_and_isolates_cache(
        self, mock_search_yt: MagicMock
    ) -> None:
        mock_search_yt.return_value = (
            "https://www.youtube.com/watch?v=mockyt_son_tung",
            "Sơn Tùng M-TP - Nơi Này Có Anh",
            260.0,
            True,
            0.95,
        )
        spotify_data = {
            "id": "spot_123",
            "title": "Nơi Này Có Anh",
            "artist": "Sơn Tùng M-TP",
            "release_date": "2017-02-14",
            "spotify_url": "https://open.spotify.com/track/spot_123",
        }
        mock_spotify = MagicMock(spec=ISpotifyService)
        mock_spotify.find_best_match.return_value = spotify_data
        mock_spotify.search_tracks.return_value = [spotify_data]

        resolver = InputResolverService(spotify_service=mock_spotify)

        with patch("backend.services.resolver_service.ENABLE_SPOTIFY_PRE_SEARCH", True):
            track_off = resolver.resolve_track("noi nay co anh", enable_spotify=False)
            self.assertIsNone(track_off.spotify_url)
            self.assertFalse(track_off.spotify_searched)
            mock_spotify.find_best_match.assert_not_called()
            mock_spotify.search_tracks.assert_not_called()

            track_on = resolver.resolve_track("noi nay co anh", enable_spotify=True)
            self.assertEqual(track_on.spotify_url, "https://open.spotify.com/track/spot_123")
            self.assertTrue(track_on.spotify_searched)

    def test_detection_service_skips_spotify_and_early_exit_when_disabled(self) -> None:
        repo = InMemoryTrackRepository()
        mock_audio_engine = MagicMock()
        mock_audio_engine.process_track_audio.return_value = (
            None,
            {"title": "Chạm Đáy Nỗi Đau", "artist": "ERIK"},
        )
        mock_signal_analyzer = MagicMock()
        mock_signal_analyzer.check_heuristics.return_value = None
        mock_signal_analyzer.analyze.return_value = None

        mock_classifier_res = MagicMock()
        mock_classifier_res.is_ai = False
        mock_classifier_res.verdict_type = VerdictType.CONFIRMED_HUMAN
        mock_classifier_res.verdict_label = "CONFIRMED HUMAN"
        mock_classifier_res.confidence_score = 0.91
        mock_classifier_res.suspected_engine = "None"
        mock_classifier_res.reasons = ["Âm sắc tự nhiên"]
        mock_classifier_res.detailed_explanation = "Người thật hát."
        mock_classifier_res.acoustic_metrics = None
        mock_classifier_res.model_used = "gemini-3.8-flash"
        mock_classifier_res.work_type = WorkType.ORIGINAL
        mock_classifier = MagicMock()
        mock_classifier.classify_track.return_value = mock_classifier_res

        mock_spotify = MagicMock(spec=ISpotifyService)
        mock_spotify.find_best_match.return_value = {
            "id": "spot_1999",
            "title": "Chạm Đáy Nỗi Đau",
            "artist": "ERIK",
            "release_date": "2018-04-11",
            "spotify_url": "https://open.spotify.com/track/spot_1999",
        }

        service = DetectionService(
            audio_engine=mock_audio_engine,
            signal_analyzer=mock_signal_analyzer,
            classifier=mock_classifier,
            resolver=MagicMock(),
            track_repo=repo,
            spotify_service=mock_spotify,
        )

        track_item = TrackItem(
            id="cham_day_off",
            raw_query="Chạm Đáy Nỗi Đau ERIK",
            input_type=InputType.URL,
            track_title="Chạm Đáy Nỗi Đau",
            artist="ERIK",
            resolved_url="https://youtube.com/watch?v=mock123",
            song_type="Original",
        )

        with (
            patch("backend.services.detection_service.ENABLE_SPOTIFY_SEARCH", True),
            patch("backend.services.detection_service.ENABLE_SPOTIFY_EARLY_EXIT", True),
            patch.object(
                service,
                "_apply_cross_validation_safeguard",
                side_effect=lambda r, *args, **kwargs: r,
            ),
        ):
            result = service.detect_single(track_item, enable_spotify=False)

        mock_spotify.find_best_match.assert_not_called()
        mock_audio_engine.process_track_audio.assert_called_once()
        mock_classifier.classify_track.assert_called_once()
        self.assertIsNone(result.spotify_url)
        self.assertNotEqual(result.model_used, "spotify-pre-ai-rule")

    def test_track_controller_and_batch_service_propagate_enable_spotify(self) -> None:
        from backend.api.controllers.track_controller import TrackController
        from backend.api.schemas.dtos import TrackDetectRequest
        from backend.services.batch_service import BatchService

        mock_resolver = MagicMock()
        track = TrackItem(
            id="t_prop",
            raw_query="Song Test",
            input_type=InputType.SONG_NAME,
            track_title="Song Test",
            resolved_url="https://youtube.com/watch?v=prop",
        )
        mock_resolver.resolve_track.return_value = track

        mock_detection = MagicMock()
        dummy_res = DetectionResult(
            track_id="t_prop",
            track_title="Song Test",
            input_type=InputType.SONG_NAME,
            raw_query="Song Test",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.9,
            suspected_engine="None",
            reasons=["OK"],
            detailed_explanation="OK",
        )
        mock_detection.detect_single.return_value = dummy_res

        controller = TrackController(detection_service=mock_detection, resolver=mock_resolver)
        req = TrackDetectRequest(query="Song Test", enable_spotify=False)
        resp = controller.detect_track(req)

        self.assertTrue(resp.success)
        mock_resolver.resolve_track.assert_called_once_with(
            "Song Test", force_rescan=False, enable_spotify=False
        )
        mock_detection.detect_single.assert_called_once_with(
            track,
            model_override=None,
            force_rescan=False,
            language="vi",
            enable_spotify=False,
        )

        mock_detection.detect_single.reset_mock()
        batch_svc = BatchService(
            detection_service=mock_detection,
            report_service=MagicMock(),
            track_repo=InMemoryTrackRepository(),
            sleep_interval=0.0,
        )
        batch_svc.process_batch([track], enable_spotify=False)
        mock_detection.detect_single.assert_called_once_with(
            track,
            model_override=None,
            force_rescan=False,
            language="vi",
            enable_spotify=False,
        )

    def test_batch_job_manager_and_ui_toggle_sync_enable_spotify(self) -> None:
        from backend.api.controllers.batch_controller import BatchController
        from backend.api.schemas.dtos import BatchDetectRequest
        from backend.core.models.entities import BatchReport
        from frontend.components.ui_compat import _SPOTIFY_TOGGLE_KEYS, _sync_spotify_toggle

        # 1. BatchController propagates enable_spotify=False
        mock_batch_svc = MagicMock()
        mock_batch_svc.process_batch.return_value = BatchReport(
            total_tracks=1,
            processed_count=1,
            ai_count=0,
            human_count=1,
            uncertain_count=0,
            failed_count=0,
            results=[],
        )
        mock_resolver = MagicMock()
        mock_resolver.detect_input_type.return_value = InputType.SONG_NAME

        batch_ctrl = BatchController(batch_service=mock_batch_svc, resolver=mock_resolver)
        req = BatchDetectRequest(queries=["Song A"], enable_spotify=False)
        resp = batch_ctrl.process_queries(req)
        self.assertTrue(resp.success, msg=resp.error)
        self.assertEqual(mock_batch_svc.process_batch.call_args.kwargs["enable_spotify"], False)

        # 2. UI toggle sync helper updates enable_spotify and sibling widget keys
        class AttrDict(dict):
            def __getattr__(self, name: str):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name: str, value) -> None:
                self[name] = value

        fake_state = AttrDict({k: True for k in _SPOTIFY_TOGGLE_KEYS})
        fake_state["enable_spotify"] = True
        fake_state["single_enable_spotify"] = False
        with patch("frontend.components.ui_compat.st.session_state", fake_state):
            _sync_spotify_toggle("single_enable_spotify")
            self.assertFalse(fake_state["enable_spotify"])
            self.assertFalse(fake_state["sb_enable_spotify"])
            self.assertFalse(fake_state["batch_enable_spotify"])


if __name__ == "__main__":
    unittest.main()
