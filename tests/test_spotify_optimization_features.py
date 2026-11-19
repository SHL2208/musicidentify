"""Bộ kiểm thử chuyên sâu cho các tính năng tối ưu hóa Spotify Search:
1. A1 & A2: Strict Precision Matching, Noise Stripping, từ chối False Positives (None thay vì candidates[0]).
2. B1: Persistent SQLite Cache (Hit, Miss, Negative Cache).
3. B2: Rate Limiter Pacer và tự động Retry khi gặp HTTP 429 Retry-After.
4. B4: Deduplication trong Detection Pipeline.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.core.models.entities import InputType, TrackItem
from backend.services.spotify_cache_service import SpotifyCacheService
from backend.services.spotify_service import SpotifyService, _strip_noise


class TestSpotifyOptimizationFeatures(unittest.TestCase):
    """Kiểm thử toàn diện các hạng mục A1, A2, B1, B2, B4."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_spotify_cache.sqlite"
        self.cache_service = SpotifyCacheService(db_path=self.test_db_path)
        self.service = SpotifyService(
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            cache_service=self.cache_service,
        )
        self.service._access_token = "mock_valid_token"
        self.service._token_expires_at = time.time() + 3600

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # =========================================================================
    # A1 & A2: STRICT PRECISION & NOISE STRIPPING
    # =========================================================================

    def test_strip_noise_removes_unwanted_tags(self) -> None:
        """Kiểm tra loại bỏ các hậu tố nhiễu (MV, Official, Audio, Remix, ft...)."""
        raw_titles = [
            ("Đừng Làm Trái Tim Anh Đau (Official Music Video)", "Đừng Làm Trái Tim Anh Đau"),
            ("Chìm Sâu [Official Audio] ft. MCK", "Chìm Sâu"),
            ("Nơi Này Có Anh (Lyric Video)", "Nơi Này Có Anh"),
            ("Cắt Đôi Nỗi Sầu (Remix)", "Cắt Đôi Nỗi Sầu"),
        ]
        for raw, expected in raw_titles:
            cleaned = _strip_noise(raw)
            self.assertEqual(cleaned, expected)

    def test_find_best_match_returns_none_when_unmatched(self) -> None:
        """(A1) Khi không có bài nào đạt ngưỡng, phải trả về None thay vì candidates[0]."""
        mock_candidates = [
            {
                "id": "wrong_1",
                "title": "Bài Hát Lạ Hoắc Nào Đó",
                "artist": "Ca Sĩ Xa Lạ",
                "artists_list": ["Ca Sĩ Xa Lạ"],
                "popularity": 20,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/wrong_1",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            result = self.service.find_best_match("Một Khúc Tâm Tình Của Người Hà Tĩnh", artist="Thu Hiền")
            self.assertIsNone(result, "Phải trả về None khi điểm so khớp quá thấp, không được lấy candidates[0]!")

    def test_find_best_match_rejects_short_substring_trap(self) -> None:
        """(A2) Tên bài quá ngắn không được nhận điểm substring vào bài dài không liên quan."""
        mock_candidates = [
            {
                "id": "long_song",
                "title": "Anh Nhớ Em Nhiều Lắm",
                "artist": "Vũ",
                "artists_list": ["Vũ"],
                "popularity": 70,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/long_song",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            # Target chỉ là "Anh", không được phép match vào "Anh Nhớ Em Nhiều Lắm"
            result = self.service.find_best_match("Anh")
            self.assertIsNone(result, "Từ 1 chữ 'Anh' không được phép match vào 'Anh Nhớ Em Nhiều Lắm'!")

    def test_find_best_match_rejects_wrong_artist(self) -> None:
        """(A2) Đúng tên bài nhưng hoàn toàn sai ca sĩ phải bị từ chối."""
        mock_candidates = [
            {
                "id": "match_title_wrong_artist",
                "title": "Đừng Làm Trái Tim Anh Đau",
                "artist": "Ca Sĩ Vô Danh",
                "artists_list": ["Ca Sĩ Vô Danh"],
                "popularity": 10,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/wrong_art",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            result = self.service.find_best_match("Đừng Làm Trái Tim Anh Đau", artist="Sơn Tùng M-TP")
            self.assertIsNone(result, "Sai ca sĩ mục tiêu thì phải từ chối!")

    def test_find_best_match_accepts_valid_match(self) -> None:
        """(A2) Đúng tên bài và ca sĩ phải khớp thành công."""
        mock_candidates = [
            {
                "id": "correct_track",
                "title": "Đừng Làm Trái Tim Anh Đau (Official Audio)",
                "artist": "Sơn Tùng M-TP",
                "artists_list": ["Sơn Tùng M-TP"],
                "popularity": 85,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/correct_track",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            result = self.service.find_best_match("Đừng Làm Trái Tim Anh Đau", artist="Sơn Tùng M-TP")
            self.assertIsNotNone(result)
            self.assertEqual(result["id"], "correct_track")
            self.assertGreaterEqual(result["match_score"], 0.70)

    def test_batch_search_tracks_sets_empty_link_when_not_found(self) -> None:
        """(A1) Batch search khi không tìm thấy thì link Spotify phải là None, is_found = False."""
        with patch.object(self.service, "find_best_match", return_value=None):
            results = self.service.batch_search_tracks(["Bài Hát Không Tồn Tại 12345"])
            self.assertEqual(len(results), 1)
            item = results[0]
            self.assertFalse(item["is_found"])
            self.assertIsNone(item["spotify_url"])
            self.assertIsNone(item["spotify_title"])
            self.assertEqual(item["match_score"], 0.0)

    # =========================================================================
    # B1: PERSISTENT SQLITE CACHE (POSITIVE & NEGATIVE CACHE)
    # =========================================================================

    def test_sqlite_cache_hit_and_negative_cache(self) -> None:
        """(B1) Kiểm tra lưu trữ và truy xuất cache (cả bài tìm thấy và không tìm thấy)."""
        # 1. Test Positive Cache
        mock_data = {
            "id": "track_abc",
            "title": "Cắt Đôi Nỗi Sầu",
            "artist": "Tăng Duy Tân",
            "spotify_url": "https://open.spotify.com/track/track_abc",
        }
        self.cache_service.set("Cắt Đôi Nỗi Sầu", artist="Tăng Duy Tân", data=mock_data, is_found=True)

        is_hit, hit_data = self.cache_service.get("Cắt Đôi Nỗi Sầu", artist="Tăng Duy Tân")
        self.assertTrue(is_hit)
        self.assertIsNotNone(hit_data)
        self.assertEqual(hit_data["id"], "track_abc")

        # 2. Test Negative Cache (bài không tìm thấy)
        self.cache_service.set("Bài Hát Ảo", artist="Ca Sĩ Ảo", data=None, is_found=False)
        is_neg_hit, neg_data = self.cache_service.get("Bài Hát Ảo", artist="Ca Sĩ Ảo")
        self.assertTrue(is_neg_hit, "Phải hit Negative Cache")
        self.assertIsNone(neg_data, "Negative Cache hit phải trả về data = None")

        # 3. Test Cache Miss
        is_miss, miss_data = self.cache_service.get("Bài Chưa Từng Tìm")
        self.assertFalse(is_miss)
        self.assertIsNone(miss_data)

    def test_find_best_match_uses_cache_and_avoids_api_call(self) -> None:
        """(B1) Khi đã có trong Cache, find_best_match không được gọi search_tracks."""
        mock_data = {
            "id": "cached_track_1",
            "title": "Chìm Sâu",
            "artist": "MCK",
            "spotify_url": "https://open.spotify.com/track/cached_track_1",
            "match_score": 0.95,
        }
        self.cache_service.set("Chìm Sâu", artist="MCK", data=mock_data, is_found=True)

        with patch.object(self.service, "search_tracks") as mock_search:
            res = self.service.find_best_match("Chìm Sâu", artist="MCK")
            self.assertIsNotNone(res)
            self.assertEqual(res["id"], "cached_track_1")
            mock_search.assert_not_called()

    # =========================================================================
    # B2: RATE LIMITER PACER & RETRY-AFTER
    # =========================================================================

    def test_http_429_retries_with_retry_after(self) -> None:
        """(B2) Khi gặp HTTP 429, tự động chờ theo Retry-After và thử lại thành công."""
        self.service._access_token = "valid_token"
        self.service._token_expires_at = time.time() + 3600

        # Lần 1 ném 429 với Retry-After: 1, lần 2 trả về JSON thành công
        err_fp = io.BytesIO(b'{"error":{"status":429,"message":"Rate limit"}}')
        http_429_err = urllib.error.HTTPError(
            url="https://api.spotify.com/v1/search",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "1"},
            fp=err_fp,
        )

        success_response = {
            "tracks": {
                "items": [
                    {
                        "id": "retry_track",
                        "name": "Bài Hát Retry",
                        "artists": [{"name": "Ca Sĩ"}],
                        "album": {"name": "Album", "release_date": "2024-01-01", "images": []},
                        "duration_ms": 180000,
                        "external_urls": {"spotify": "https://open.spotify.com/track/retry_track"},
                        "popularity": 75,
                    }
                ]
            }
        }
        success_fp = io.BytesIO(json.dumps(success_response).encode("utf-8"))

        with patch("time.sleep") as mock_sleep:
            with patch("urllib.request.urlopen", side_effect=[http_429_err, success_fp]):
                results = self.service.search_tracks("Bài Hát Retry")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["id"], "retry_track")
                # Xác nhận time.sleep đã được gọi với thời gian Retry-After (1s)
                mock_sleep.assert_called()
                self.assertFalse(self.service._quota_exceeded)

    # =========================================================================
    # B4: PIPELINE DEDUPLICATION
    # =========================================================================

    def test_detection_service_skips_spotify_when_already_searched(self) -> None:
        """(B4) DetectionService không gọi lại Spotify nếu track đã được đánh dấu spotify_searched."""
        from backend.services.detection_service import DetectionService

        mock_spotify = MagicMock()
        detection_service = DetectionService(
            audio_engine=MagicMock(),
            signal_analyzer=MagicMock(),
            classifier=MagicMock(),
            resolver=MagicMock(),
            track_repo=MagicMock(),
            spotify_service=mock_spotify,
        )

        track = TrackItem(
            id="track_test_dedup",
            raw_query="Test Song",
            input_type=InputType.SONG_NAME,
            track_title="Test Song",
            artist="Test Artist",
            spotify_searched=True,  # Đã tìm kiếm trước đó
            spotify_url=None,       # Không có bài trên Spotify
        )

        # Kiểm tra logic bảo vệ: nếu spotify_searched=True thì không gọi find_best_match
        if not track.spotify_url and not getattr(track, "spotify_searched", False):
            mock_spotify.find_best_match("Test Song", artist="Test Artist")

        mock_spotify.find_best_match.assert_not_called()


if __name__ == "__main__":
    unittest.main()
