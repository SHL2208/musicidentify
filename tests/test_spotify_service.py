"""Bộ kiểm thử cho SpotifyService (Client Credentials Flow, Metadata Parser, Best Match & Fallbacks).

Tuân thủ nghiêm ngặt PEP 8, unittest và Ruff.
"""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from backend.services.spotify_service import SpotifyService


class TestSpotifyService(unittest.TestCase):
    """Kiểm thử SpotifyService từ xác thực, tìm kiếm, đến so khớp bài hát tối ưu."""

    def setUp(self) -> None:
        self.service = SpotifyService(
            client_id="test_client_id",
            client_secret="test_client_secret",
            timeout_seconds=2.0,
        )

    def test_is_spotify_url(self) -> None:
        """Kiểm tra nhận diện các dạng URL Spotify hợp lệ và không hợp lệ."""
        valid_urls = [
            "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT",
            "https://open.spotify.com/intl-vi/track/4cOdK2wGLETKBW3PvgPWqT?si=abc",
            "spotify:track:4cOdK2wGLETKBW3PvgPWqT",
        ]
        invalid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://zingmp3.vn/bai-hat/Z9U0OE7I.html",
            "sample_audio.mp3",
            "Sơn Tùng M-TP - Nơi Này Có Anh",
        ]

        for u in valid_urls:
            self.assertTrue(SpotifyService.is_spotify_url(u), f"Phải nhận diện là Spotify URL: {u}")
        for u in invalid_urls:
            self.assertFalse(SpotifyService.is_spotify_url(u), f"Không được nhận diện là Spotify: {u}")

    def test_parse_track_item(self) -> None:
        """Kiểm tra hàm chuẩn hóa dữ liệu track từ Spotify Web API."""
        raw_spotify_track = {
            "id": "track_123",
            "name": "Đừng Làm Trái Tim Anh Đau",
            "artists": [{"name": "Sơn Tùng M-TP"}],
            "album": {
                "name": "Đừng Làm Trái Tim Anh Đau - Single",
                "release_date": "2024-06-08",
                "images": [{"url": "https://i.scdn.co/image/cover123.jpg"}],
            },
            "duration_ms": 315000,
            "external_urls": {"spotify": "https://open.spotify.com/track/track_123"},
            "preview_url": "https://p.scdn.co/mp3-preview/preview123.mp3",
            "external_ids": {"isrc": "VNXYZ2400001"},
            "popularity": 85,
        }

        parsed = self.service._parse_track_item(raw_spotify_track)
        self.assertEqual(parsed["id"], "track_123")
        self.assertEqual(parsed["title"], "Đừng Làm Trái Tim Anh Đau")
        self.assertEqual(parsed["artist"], "Sơn Tùng M-TP")
        self.assertEqual(parsed["album"], "Đừng Làm Trái Tim Anh Đau - Single")
        self.assertEqual(parsed["release_date"], "2024-06-08")
        self.assertEqual(parsed["cover_url"], "https://i.scdn.co/image/cover123.jpg")
        self.assertEqual(parsed["spotify_url"], "https://open.spotify.com/track/track_123")
        self.assertEqual(parsed["embed_url"], "https://open.spotify.com/embed/track/track_123")
        self.assertEqual(parsed["duration_sec"], 315.0)
        self.assertEqual(parsed["isrc"], "VNXYZ2400001")
        self.assertEqual(parsed["popularity"], 85)
        self.assertTrue(parsed["is_official_api"])

    @patch("urllib.request.urlopen")
    def test_get_access_token_success(self, mock_urlopen: MagicMock) -> None:
        """Kiểm tra lấy token thành công qua Client Credentials Flow."""
        token_response_payload = json.dumps({"access_token": "mock_token_abc_123", "expires_in": 3600}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = BytesIO(token_response_payload)

        token = self.service._get_access_token()
        self.assertEqual(token, "mock_token_abc_123")
        self.assertEqual(self.service._access_token, "mock_token_abc_123")

    @patch("urllib.request.urlopen")
    def test_search_tracks_with_official_api(self, mock_urlopen: MagicMock) -> None:
        """Kiểm tra tìm kiếm bài hát khi đã có token xác thực."""
        # Giả lập token có sẵn
        self.service._access_token = "valid_token"
        self.service._token_expires_at = 9999999999.0

        mock_search_data = {
            "tracks": {
                "items": [
                    {
                        "id": "abc123",
                        "name": "Nơi Này Có Anh",
                        "artists": [{"name": "Sơn Tùng M-TP"}],
                        "album": {"name": "m-tp M-TP", "release_date": "2017-02-14", "images": []},
                        "duration_ms": 260000,
                        "external_urls": {"spotify": "https://open.spotify.com/track/abc123"},
                        "popularity": 80,
                    }
                ]
            }
        }
        mock_urlopen.return_value.__enter__.return_value = BytesIO(json.dumps(mock_search_data).encode("utf-8"))

        results = self.service.search_tracks(query="Nơi Này Có Anh", artist="Sơn Tùng M-TP")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "abc123")
        self.assertEqual(results[0]["title"], "Nơi Này Có Anh")
        self.assertEqual(results[0]["artist"], "Sơn Tùng M-TP")

    def test_search_tracks_fallback_when_no_token(self) -> None:
        """Kiểm tra cơ chế Zero-Config Fallback trả về link mở Spotify Search an toàn."""
        service_no_keys = SpotifyService(client_id="", client_secret="")
        # Không có token và không có internet
        with patch.object(service_no_keys, "_get_access_token", return_value=None):
            results = service_no_keys.search_tracks(query="Blinding Lights", artist="The Weeknd")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], "search_fallback")
            self.assertIn("open.spotify.com/search", results[0]["spotify_url"])
            self.assertFalse(results[0]["is_official_api"])

    def test_find_best_match_ranking(self) -> None:
        """Kiểm tra thuật toán so khớp chuỗi chọn bài hát chính xác nhất từ danh sách ứng viên."""
        mock_candidates = [
            {
                "id": "1",
                "title": "Đừng Làm Trái Tim Anh Đau (Remix Vinahouse)",
                "artist": "DJ Hùng Remix",
                "popularity": 40,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/1",
            },
            {
                "id": "2",
                "title": "Đừng Làm Trái Tim Anh Đau",
                "artist": "Sơn Tùng M-TP",
                "popularity": 90,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/2",
            },
            {
                "id": "3",
                "title": "Anh Đau Lòng Quá",
                "artist": "Ca Sĩ Khác",
                "popularity": 10,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/3",
            },
        ]

        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            best = self.service.find_best_match("Đừng Làm Trái Tim Anh Đau", artist="Sơn Tùng M-TP")
            self.assertIsNotNone(best)
            self.assertEqual(best["id"], "2")
            self.assertEqual(best["artist"], "Sơn Tùng M-TP")

    def test_spotify_quota_exceeded_circuit_breaker(self) -> None:
        """Kiểm tra cơ chế Circuit Breaker tự ngắt khi gặp lỗi HTTP 429 Quota Exceeded."""
        import urllib.error

        self.service._access_token = "valid_token"
        self.service._token_expires_at = 9999999999.0
        self.assertFalse(self.service._quota_exceeded)

        # Giả lập phản hồi 429 từ Spotify API
        err_fp = BytesIO(b'{"error":{"status":429,"message":"Too many requuests","reason":"QUOTA_EXCEEDED"}}')
        http_429_err = urllib.error.HTTPError(
            url="https://api.spotify.com/v1/search",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=err_fp,
        )

        with patch("urllib.request.urlopen", side_effect=http_429_err):
            # Lần gọi đầu tiên: Gặp 429 và kích hoạt Circuit Breaker
            res1 = self.service.search_tracks("Test Song")
            self.assertTrue(self.service._quota_exceeded)
            self.assertEqual(res1[0]["id"], "search_fallback")

        # Lần gọi thứ hai: urllib.request.urlopen không được gọi vì circuit breaker đã ngắt
        with patch("urllib.request.urlopen") as mock_url:
            res2 = self.service.search_tracks("Another Song")
            mock_url.assert_not_called()
            self.assertEqual(res2[0]["id"], "search_fallback")


if __name__ == "__main__":
    unittest.main()
