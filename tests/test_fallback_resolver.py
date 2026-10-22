"""Bộ kiểm thử cho cơ chế Fallback tìm kiếm đa nền tảng (YouTube -> Zing MP3 -> Spotify -> Raise Exception).

Quy trình:
1. Nếu không tìm thấy trên YouTube: Chuyển sang tìm trên Zing MP3.
2. Nếu Zing MP3 cũng không tìm thấy / không tải được: Chuyển sang tìm trên Spotify.
3. Nếu cả 3 nền tảng đều không tìm thấy: Báo lỗi TrackNotFoundException (không tải bừa video rác).

Tuân thủ nghiêm ngặt PEP 8 và Ruff.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.core.exceptions import TrackNotFoundException
from backend.core.models.entities import TrackItem
from backend.services.resolver_service import InputResolverService


class TestFallbackResolver(unittest.TestCase):
    """Kiểm thử chuỗi dự phòng 3 tầng (YouTube -> Zing MP3 -> Spotify -> Error)."""

    def setUp(self) -> None:
        self.resolver = InputResolverService()

    @patch("backend.services.resolver_service.InputResolverService._get_url_metadata")
    @patch("backend.services.resolver_service.InputResolverService._fetch_zing_metadata")
    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_youtube_fails_fallback_to_zingmp3(
        self,
        mock_search_yt: MagicMock,
        mock_fetch_zing: MagicMock,
        mock_get_url_meta: MagicMock,
    ) -> None:
        """Nếu YouTube không tìm thấy bài hát, hệ thống phải tự động fallback sang Zing MP3."""
        # Giả lập YouTube không tìm thấy video khớp
        mock_search_yt.return_value = (None, "Neon Summer Boulevard", None, False, 0.15)

        # Giả lập Zing MP3 tìm thấy bài hát hợp lệ
        mock_fetch_zing.return_value = {
            "title": "Neon Summer Boulevard",
            "artist": "Dark Suiltion",
            "duration": 186.0,
            "stream_url": "https://zingmp3.vn/bai-hat/Z9U0UC0B.html",
            "is_vip": False,
        }
        mock_get_url_meta.return_value = ("Neon Summer Boulevard", 186.0)

        query = "Neon Summer Boulevard Dark Suiltion"
        track = self.resolver.resolve_track(query)

        self.assertIsInstance(track, TrackItem)
        self.assertEqual(track.platform, "zingmp3")
        self.assertEqual(track.track_title, "Neon Summer Boulevard")
        self.assertEqual(track.resolved_url, "https://zingmp3.vn/bai-hat/Z9U0UC0B.html")
        self.assertTrue(track.is_verified_match)

    @patch("backend.services.resolver_service.InputResolverService.search_spotify")
    @patch("backend.services.resolver_service.InputResolverService._fetch_zing_metadata")
    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_youtube_and_zing_fail_fallback_to_spotify(
        self,
        mock_search_yt: MagicMock,
        mock_fetch_zing: MagicMock,
        mock_search_spotify: MagicMock,
    ) -> None:
        """Nếu cả YouTube và Zing MP3 thất bại, hệ thống phải tự động fallback sang Spotify."""
        mock_search_yt.return_value = (None, "Midnight Sky", None, False, 0.20)
        mock_fetch_zing.return_value = {}  # Zing không có bài

        mock_search_spotify.return_value = {
            "title": "Midnight Sky",
            "artist": "Dark Suiltion",
            "duration": 210.0,
            "preview_url": "https://p.scdn.co/mp3-preview/fake123.mp3",
            "track_url": "https://open.spotify.com/track/fake123",
        }

        query = "Midnight Sky Dark Suiltion"
        track = self.resolver.resolve_track(query)

        self.assertIsInstance(track, TrackItem)
        self.assertEqual(track.platform, "spotify")
        self.assertEqual(track.track_title, "Midnight Sky")
        self.assertEqual(track.resolved_url, "https://p.scdn.co/mp3-preview/fake123.mp3")
        self.assertTrue(track.is_verified_match)

    @patch("backend.services.resolver_service.InputResolverService.search_spotify")
    @patch("backend.services.resolver_service.InputResolverService._fetch_zing_metadata")
    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_all_three_platforms_fail_raises_track_not_found(
        self,
        mock_search_yt: MagicMock,
        mock_fetch_zing: MagicMock,
        mock_search_spotify: MagicMock,
    ) -> None:
        """Nếu cả 3 nền tảng YouTube, Zing, Spotify đều không tìm thấy, PHẢI BÁO LỖI TrackNotFoundException."""
        mock_search_yt.return_value = (None, "Unknown Song", None, False, 0.10)
        mock_fetch_zing.return_value = {}
        mock_search_spotify.return_value = {}

        query = "Bài Hát Ảo Hoàn Toàn Không Tồn Tại 99999"

        with self.assertRaises(TrackNotFoundException) as ctx:
            self.resolver.resolve_track(query)

        self.assertIn("Không tìm thấy bài hát", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
