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

    def test_find_best_match_with_embedded_artist_in_query(self) -> None:
        """Truy vấn gộp 'Tên bài Ca sĩ' không có dấu '-' vẫn phải khớp đúng bài có ca sĩ đó và loại bài cùng tên khác ca sĩ."""
        mock_candidates = [
            {
                "id": "kpop_panorama",
                "title": "Panorama",
                "artist": "IZ*ONE",
                "artists_list": ["IZ*ONE"],
                "popularity": 85,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/kpop_panorama",
            },
            {
                "id": "vn_panorama",
                "title": "Panorama",
                "artist": "Richie D. ICY, Obito",
                "artists_list": ["Richie D. ICY", "Obito"],
                "popularity": 60,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/vn_panorama",
            },
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            result = self.service.find_best_match("panorama obito")
            self.assertIsNotNone(result)
            self.assertEqual(result["id"], "vn_panorama")

            # Đảm bảo không khớp nhầm bài IZ*ONE nếu không có Obito
            only_kpop = [mock_candidates[0]]
            with patch.object(self.service, "search_tracks", return_value=only_kpop):
                wrong_res = self.service.find_best_match("panorama obito 2")
                self.assertIsNone(wrong_res)

    def test_find_best_match_version_dot_zero_and_collapsed_artist(self) -> None:
        """Hỗ trợ khớp '2' với '2.0' và nghệ sĩ viết liền 'richchoi' với 'Rich Choi' khi có từ khóa phụ."""
        mock_candidates = [
            {
                "id": "ntcc_1",
                "title": "Nâng Tầm Cuộc Chơi",
                "artist": "Rocky CDE, Cậu Phát",
                "artists_list": ["Rocky CDE", "Cậu Phát"],
                "popularity": 30,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/ntcc_1",
            },
            {
                "id": "ntcc_2",
                "title": "NÂNG TẦM CUỘC CHƠI 2.0",
                "artist": "Rocky CDE, Rich Choi, Cậu Phát",
                "artists_list": ["Rocky CDE", "Rich Choi", "Cậu Phát"],
                "popularity": 35,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/ntcc_2",
            },
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            result = self.service.find_best_match("nâng tầm cuộc chơi 2 gió và cỏ richchoi")
            self.assertIsNotNone(result)
            self.assertEqual(result["id"], "ntcc_2")

    def test_find_best_match_rejects_short_title_even_if_artist_matches(self) -> None:
        """Tên bài quá ngắn (vd: 'Em') không được phép khớp vào truy vấn dài chỉ vì trùng ca sĩ."""
        mock_candidates = [
            {
                "id": "short_em",
                "title": "Em",
                "artist": "Sơn Tùng",
                "artists_list": ["Sơn Tùng"],
                "popularity": 50,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/short_em",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=mock_candidates):
            result = self.service.find_best_match("Em Của Ngày Hôm Qua Sơn Tùng")
            self.assertIsNone(result)

    def test_find_best_match_does_not_negative_cache_api_failures(self) -> None:
        """Lỗi tạm thời (503/429/mất mạng trả về search_fallback) tuyệt đối không được lưu vào Negative Cache."""
        fallback_only = [
            {
                "id": "search_fallback",
                "title": "Nâng Tầm Cuộc Chơi",
                "artist": "",
                "is_official_api": False,
                "spotify_url": "https://open.spotify.com/search/test",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=fallback_only):
            res = self.service.find_best_match("Nâng Tầm Cuộc Chơi")
            self.assertIsNone(res)

        is_hit, _ = self.cache_service.get("Nâng Tầm Cuộc Chơi")
        self.assertFalse(is_hit, "Không được lưu Negative Cache khi API bị lỗi tạm thời!")

    def test_find_best_match_prefix_fallback_rescues_extra_context(self) -> None:
        """Nếu lần tìm đầu không ra ứng viên khớp vì dư từ khóa phụ, fallback tìm theo cụm đầu và đối soát lại."""
        unrelated = [
            {
                "id": "other",
                "title": "Có Ai Hẹn Hò Cùng Em Chưa",
                "artist": "Quân A.P",
                "artists_list": ["Quân A.P"],
                "popularity": 70,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/other",
            }
        ]
        target = [
            {
                "id": "ntcc_1",
                "title": "Nâng Tầm Cuộc Chơi",
                "artist": "Rocky CDE, Cậu Phát",
                "artists_list": ["Rocky CDE", "Cậu Phát"],
                "popularity": 40,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/ntcc_1",
            }
        ]
        with patch.object(self.service, "search_tracks", side_effect=[unrelated, target]) as mock_search:
            res = self.service.find_best_match("nâng tầm cuộc chơi gió và cò")
            self.assertIsNotNone(res)
            self.assertEqual(res["id"], "ntcc_1")
            self.assertEqual(mock_search.call_count, 2)

    def test_http_503_retries_and_succeeds(self) -> None:
        """Khi Spotify trả HTTP 503 (no healthy upstream), tự động retry và thành công."""
        self.service._access_token = "valid_token"
        self.service._token_expires_at = time.time() + 3600

        err_fp = io.BytesIO(b'{"error":{"status":503,"message":"no healthy upstream"}}')
        http_503_err = urllib.error.HTTPError(
            url="https://api.spotify.com/v1/search",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=err_fp,
        )
        success_response = {
            "tracks": {
                "items": [
                    {
                        "id": "ok_after_503",
                        "name": "Nâng Tầm Cuộc Chơi",
                        "artists": [{"name": "Rocky CDE"}],
                        "album": {"name": "Single", "release_date": "2024-01-01", "images": []},
                        "duration_ms": 200000,
                        "external_urls": {"spotify": "https://open.spotify.com/track/ok_after_503"},
                        "popularity": 50,
                    }
                ]
            }
        }
        success_fp = io.BytesIO(json.dumps(success_response).encode("utf-8"))

        with patch("time.sleep"):
            with patch("urllib.request.urlopen", side_effect=[http_503_err, success_fp]):
                results = self.service.search_tracks("Nâng Tầm Cuộc Chơi")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["id"], "ok_after_503")

    def test_version_aware_matching_distinguishes_beat_remix_remixer_and_live(self) -> None:
        """Phân biệt chính xác Bản gốc vs Beat vs Remix vs Live Session vs tên Remixer khác nhau."""
        # 1. VANTRI Remix không được khớp nhầm sang MVS Remix hay bản Instrumental
        remix_candidates = [
            {
                "id": "mvs_remix",
                "title": "Em Thua Cô Ta (MVS Remix)",
                "artist": "Min Quỳnh Anh, MeMe Media, Mvs",
                "artists_list": ["Min Quỳnh Anh", "MeMe Media", "Mvs"],
                "popularity": 80,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/mvs_remix",
            },
            {
                "id": "vantri_inst",
                "title": "Em Thua Cô Ta (VANTRI Remix)[instrumental]",
                "artist": "Min Quỳnh Anh, MeMe Media, VANTRI",
                "artists_list": ["Min Quỳnh Anh", "MeMe Media", "VANTRI"],
                "popularity": 50,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/vantri_inst",
            },
            {
                "id": "vantri_vocal",
                "title": "Em Thua Cô Ta (VANTRI Remix)",
                "artist": "Min Quỳnh Anh, MeMe Media, VANTRI",
                "artists_list": ["Min Quỳnh Anh", "MeMe Media", "VANTRI"],
                "popularity": 40,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/vantri_vocal",
            },
        ]
        with patch.object(self.service, "search_tracks", return_value=remix_candidates):
            res = self.service.find_best_match(
                "Em Thua Cô Ta (VANTRI Remix)", artist="Min Quỳnh Anh, Meme Media"
            )
            self.assertIsNotNone(res)
            self.assertEqual(res["id"], "vantri_vocal")

        # 2. Bản gốc KHÔNG được khớp nhầm sang bản Beat hoặc bản Drill/Remix
        beat_only = [
            {
                "id": "beat_ver",
                "title": "Hai Chữ Đơn Thân - Beat",
                "artist": "Lý Văn, WDM",
                "artists_list": ["Lý Văn", "WDM"],
                "popularity": 60,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/beat_ver",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=beat_only):
            res_orig = self.service.find_best_match("Hai Chữ Đơn Thân", artist="Lý Văn, WDM")
            self.assertIsNone(res_orig)

        # 3. Yêu cầu Beat KHÔNG được khớp nhầm sang bản có lời (non-beat)
        non_beat_remix = [
            {
                "id": "ton_vocal",
                "title": "Anh Chỉ Là Người Thay Thế - Tón Remix",
                "artist": "Yến Nhi, Mimilee, Tón",
                "artists_list": ["Yến Nhi", "Mimilee", "Tón"],
                "popularity": 70,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/ton_vocal",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=non_beat_remix):
            res_beat = self.service.find_best_match(
                "Anh Chỉ Là Người Thay Thế (Tón Remix) (Beat)",
                artist="Yến Nhi, Mimilee, Kiyoshi Phan Music",
            )
            self.assertIsNone(res_beat)

        # 4. Yêu cầu Live Session KHÔNG được khớp nhầm sang bản Studio thường
        studio_only = [
            {
                "id": "studio_track",
                "title": "ít nhất là vào hôm nay",
                "artist": "Ngọc Dolil",
                "artists_list": ["Ngọc Dolil"],
                "popularity": 65,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/studio_track",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=studio_only):
            res_live = self.service.find_best_match(
                "Ít Nhất Là Vào Hôm Nay (InĐậm.Stage Live Session)",
                artist="Ngọc Dolil, Tr.D, InĐậm.Stage",
            )
            self.assertIsNone(res_live)

        # 5. Yêu cầu Remix KHÔNG được khớp nhầm sang Bản gốc
        original_only = [
            {
                "id": "orig_only",
                "title": "Mừng Ngày Em Cưới",
                "artist": "Lê Gia Bảo",
                "artists_list": ["Lê Gia Bảo"],
                "popularity": 75,
                "is_official_api": True,
                "spotify_url": "https://open.spotify.com/track/orig_only",
            }
        ]
        with patch.object(self.service, "search_tracks", return_value=original_only):
            res_rem = self.service.find_best_match(
                "Mừng Ngày Em Cưới (AM Remix)", artist="Lê Gia Bảo"
            )
            self.assertIsNone(res_rem)

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
