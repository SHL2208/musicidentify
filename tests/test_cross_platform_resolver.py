"""Bộ kiểm thử cho bộ phân giải đa nền tảng (Cross-Platform Resolver: YouTube, Zing MP3, Spotify).

Kiểm tra:
1. Nhận diện định dạng nguồn (YouTube, Zing MP3, Spotify, File, Text).
2. Trích xuất siêu dữ liệu chuẩn hóa từ Zing MP3 và Spotify.
3. Bộ lọc kiểm chứng thông minh (Smart Matcher) với đối soát chuỗi mờ (Fuzzy)
   và đối soát thời lượng (Duration check) để ngăn chặn 100% video rác.

Tuân thủ nghiêm ngặt PEP 8 và Ruff.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.core.models.entities import InputType, TrackItem
from backend.services.resolver_service import InputResolverService


class TestCrossPlatformResolver(unittest.TestCase):
    """Kiểm thử tính năng phân giải âm nhạc đa nền tảng và bộ lọc thông minh."""

    def setUp(self) -> None:
        self.resolver = InputResolverService()

    def test_detect_platform_types(self) -> None:
        """Kiểm tra nhận diện đúng nguồn URL: YouTube, Zing MP3, Spotify, Local File."""
        yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        zing_url = (
            "https://zingmp3.vn/bai-hat/Ke-Say-Tinh-3-Ngay-Gio-Giong-Quoc-Thien/Z9U0OE7I.html"
        )
        spotify_url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        local_file = "sample.mp3"
        query_text = "Nơi Này Có Anh Sơn Tùng M-TP"

        # Kiểm tra detect platform
        self.assertTrue(self.resolver.is_zingmp3_url(zing_url))
        self.assertTrue(self.resolver.is_spotify_url(spotify_url))
        self.assertTrue(self.resolver.is_youtube_url(yt_url))
        self.assertFalse(self.resolver.is_youtube_url(local_file))

        # Kiểm tra detect input type tổng quát
        self.assertEqual(self.resolver.detect_input_type(zing_url), InputType.URL)
        self.assertEqual(self.resolver.detect_input_type(spotify_url), InputType.URL)
        self.assertEqual(self.resolver.detect_input_type(yt_url), InputType.URL)
        self.assertEqual(self.resolver.detect_input_type(query_text), InputType.SONG_NAME)

    def test_smart_match_rejects_irrelevant_youtube_video(self) -> None:
        """Bộ lọc đối soát thông minh PHẢI TỪ CHỐI video không liên quan hoặc lệch thời lượng."""
        target_title = "Ocean Neon Nights"
        target_artist = "Dark Suiltion"
        target_duration = 180.0  # 3 phút

        # Trường hợp 1: Video game Deus Ex (sai tên hoàn toàn, sai thời lượng)
        candidate_1 = {
            "title": "Deus Ex: Mankind Divided - SM00 Neon Nights (Stealth)",
            "uploader": "Square Enix Music",
            "duration": 365.0,  # Lệch quá xa (hơn 3 phút)
            "url": "https://www.youtube.com/watch?v=fake_deus_ex",
        }
        is_match_1, score_1 = self.resolver.validate_match(
            target_title=target_title,
            target_artist=target_artist,
            candidate=candidate_1,
            target_duration=target_duration,
        )
        self.assertFalse(is_match_1, f"Video rác không được phép chấp nhận! Điểm: {score_1}")

        # Trường hợp 2: Video hướng dẫn sửa xe (như ca Golden Rearview dán keo xe)
        candidate_2 = {
            "title": "This Glue Makes Rearview Mirror Repairs PERMANENT",
            "uploader": "ChrisFix",
            "duration": 650.0,
            "url": "https://www.youtube.com/watch?v=fake_glue",
        }
        is_match_2, score_2 = self.resolver.validate_match(
            target_title="Golden Rearview",
            target_artist="Dark Suiltion",
            candidate=candidate_2,
            target_duration=195.0,
        )
        self.assertFalse(is_match_2, f"Video hướng dẫn sửa xe phải bị loại bỏ! Điểm: {score_2}")

    def test_smart_match_accepts_valid_official_track(self) -> None:
        """Bộ lọc đối soát thông minh PHẢI CHẤP NHẬN video chính chủ khớp tên và thời lượng."""
        target_title = "Kẻ Say Tình 3 (Ngày Gió Giông)"
        target_artist = "Quốc Thiên"
        target_duration = 261.0  # 4m21s

        candidate = {
            "title": "KẺ SAY TÌNH 3 (NGÀY GIÓ GIÔNG) - QUỐC THIÊN | OFFICIAL AUDIO",
            "uploader": "Quốc Thiên Official",
            "duration": 263.0,  # Lệch 2 giây (trong ngưỡng cho phép ±5s)
            "url": "https://www.youtube.com/watch?v=fake_valid",
        }
        is_match, score = self.resolver.validate_match(
            target_title=target_title,
            target_artist=target_artist,
            candidate=candidate,
            target_duration=target_duration,
        )
        self.assertTrue(is_match, f"Video chính chủ phải được chấp nhận! Điểm: {score}")
        self.assertGreaterEqual(score, 0.70)

    @patch(
        "backend.services.resolver_service.InputResolverService._get_url_metadata",
        return_value=("Kẻ Say Tình 3", 261.0),
    )
    @patch("backend.services.resolver_service.InputResolverService._fetch_zing_metadata")
    def test_resolve_zingmp3_track(
        self, mock_zing_meta: MagicMock, mock_url_meta: MagicMock
    ) -> None:
        """Kiểm tra bóc tách bài hát Zing MP3 thành TrackItem hoàn chỉnh."""
        mock_zing_meta.return_value = {
            "title": "Kẻ Say Tình 3",
            "artist": "Quốc Thiên",
            "duration": 261.0,
            "stream_url": "https://api.mp3.zing.vn/stream/fake_stream.mp3",
            "is_vip": False,
        }
        zing_url = "https://zingmp3.vn/bai-hat/Ke-Say-Tinh-3/Z9U0OE7I.html"
        track = self.resolver.resolve_track(zing_url)

        self.assertIsInstance(track, TrackItem)
        self.assertEqual(track.track_title, "Kẻ Say Tình 3")
        self.assertEqual(track.artist, "Quốc Thiên")
        self.assertEqual(track.duration_sec, 261.0)
        self.assertEqual(track.input_type, InputType.URL)

    @patch("backend.services.resolver_service.InputResolverService._fetch_spotify_metadata")
    def test_resolve_spotify_track(self, mock_spotify_meta: MagicMock) -> None:
        """Kiểm tra bóc tách bài hát Spotify thành TrackItem hoàn chỉnh."""
        mock_spotify_meta.return_value = {
            "title": "Blinding Lights",
            "artist": "The Weeknd",
            "duration": 200.0,
            "preview_url": "https://p.scdn.co/mp3-preview/fake_preview.mp3",
        }
        spotify_url = "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b"
        track = self.resolver.resolve_track(spotify_url)

        self.assertIsInstance(track, TrackItem)
        self.assertEqual(track.track_title, "Blinding Lights")
        self.assertEqual(track.artist, "The Weeknd")
        self.assertEqual(track.duration_sec, 200.0)


if __name__ == "__main__":
    unittest.main()
