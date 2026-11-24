"""Bộ kiểm thử tích hợp Spotify với InputResolverService và DetectionService.

Tuân thủ nghiêm ngặt PEP 8, unittest và Ruff.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from backend.core.interfaces.services.spotify_service_interface import ISpotifyService
from backend.core.models.entities import InputType, TrackItem, VerdictType, WorkType
from backend.repositories.in_memory_track_repository import InMemoryTrackRepository
from backend.services.detection_service import DetectionService
from backend.services.resolver_service import InputResolverService


class MockSpotifyService(ISpotifyService):
    """Giả lập ISpotifyService cho kiểm thử tích hợp."""

    def search_tracks(self, query: str, artist: str = "", limit: int = 5) -> list[dict]:
        return [
            {
                "id": "mock_id_999",
                "title": query,
                "artist": artist or "Nghệ Sĩ Gốc",
                "album": "Album Chính Thức",
                "release_date": "2024-01-01",
                "spotify_url": f"https://open.spotify.com/track/mock_id_999",
                "embed_url": "https://open.spotify.com/embed/track/mock_id_999",
                "preview_url": "https://p.scdn.co/preview.mp3",
                "cover_url": "https://i.scdn.co/cover.jpg",
                "duration_sec": 210.0,
                "is_official_api": True,
            }
        ]

    def get_track_by_url(self, url: str) -> dict | None:
        return {
            "id": "mock_url_track",
            "title": "Bản Thu Gốc",
            "artist": "Ca Sĩ Chính Chủ",
            "album": "Album Gốc",
            "release_date": "2023-12-01",
            "spotify_url": url,
            "embed_url": "https://open.spotify.com/embed/track/mock_url_track",
            "preview_url": None,
            "cover_url": "https://i.scdn.co/cover.jpg",
            "duration_sec": 195.0,
            "is_official_api": True,
        }

    def find_best_match(self, title: str, artist: str = "") -> dict | None:
        tracks = self.search_tracks(query=title, artist=artist, limit=1)
        return tracks[0] if tracks else None

    def batch_search_tracks(
        self,
        queries: list[str | dict],
        max_workers: int = 3,
        on_progress: Any = None,
    ) -> list[dict]:
        return []


class TestSpotifyResolverIntegration(unittest.TestCase):
    """Kiểm thử tích hợp Spotify với Resolver và DetectionService."""

    def setUp(self) -> None:
        self.mock_spotify = MockSpotifyService()
        self.resolver = InputResolverService(spotify_service=self.mock_spotify)
        self.track_repo = InMemoryTrackRepository()

    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_resolve_spotify_url_bridge(self, mock_search_yt: MagicMock) -> None:
        """Kiểm tra khi đưa vào URL Spotify, Resolver chuyển bridge sang YouTube và lưu giữ spotify_url."""
        mock_search_yt.return_value = (
            "https://www.youtube.com/watch?v=fake_yt",
            "Bản Thu Gốc - Ca Sĩ Chính Chủ",
            195.0,
            True,
            0.95,
        )

        spotify_url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        track = self.resolver.resolve_track(spotify_url)

        self.assertIsInstance(track, TrackItem)
        self.assertEqual(track.platform, "spotify_bridge")
        self.assertEqual(track.track_title, "Bản Thu Gốc")
        self.assertEqual(track.artist, "Ca Sĩ Chính Chủ")
        self.assertEqual(track.spotify_url, spotify_url)
        self.assertIsNotNone(track.spotify_metadata)
        self.assertEqual(track.spotify_metadata["id"], "mock_url_track")

    def test_detection_service_auto_enriches_spotify_url(self) -> None:
        """Kiểm tra DetectionService tự động tìm link Spotify cho bài hát và gán vào DetectionResult."""
        mock_audio_engine = MagicMock()
        mock_audio_engine.process_track_audio.return_value = (None, {"title": "Bài Ca Kỷ Niệm", "artist": "Ca Sĩ Test"})
        mock_signal_analyzer = MagicMock()
        mock_signal_analyzer.check_heuristics.return_value = None
        mock_signal_analyzer.analyze.return_value = None
        mock_classifier = MagicMock()

        detection_service = DetectionService(
            audio_engine=mock_audio_engine,
            signal_analyzer=mock_signal_analyzer,
            classifier=mock_classifier,
            resolver=self.resolver,
            track_repo=self.track_repo,
            spotify_service=self.mock_spotify,
        )

        track = TrackItem(
            id="track_test_spotify_enrich",
            raw_query="Bài Ca Kỷ Niệm",
            input_type=InputType.SONG_NAME,
            track_title="Bài Ca Kỷ Niệm",
            artist="Ca Sĩ Test",
            resolved_url="https://youtube.com/watch?v=123",
        )

        # Giả lập classifier trả về kết quả hợp lệ
        mock_res = MagicMock()
        mock_res.is_ai = False
        mock_res.verdict_type = VerdictType.CONFIRMED_HUMAN
        mock_res.verdict_label = "CONFIRMED HUMAN (Người làm)"
        mock_res.confidence_score = 0.95
        mock_res.suspected_engine = "None"
        mock_res.reasons = ["Âm thanh tự nhiên"]
        mock_res.detailed_explanation = "Chi tiết thẩm âm"
        mock_res.acoustic_metrics = None
        mock_res.model_used = "gemini-3.8-flash"
        mock_res.work_type = WorkType.ORIGINAL
        mock_classifier.classify.return_value = mock_res
        mock_signal_analyzer.analyze.return_value = None

        with patch.object(detection_service, "_apply_cross_validation_safeguard", side_effect=lambda r, *args, **kwargs: r):
            result = detection_service.detect_single(track)

        self.assertIsNotNone(result.spotify_url)
        self.assertIn("open.spotify.com/track/mock_id_999", result.spotify_url)
        self.assertIsNotNone(result.spotify_metadata)
        self.assertEqual(result.spotify_metadata["album"], "Album Chính Thức")


if __name__ == "__main__":
    unittest.main()
