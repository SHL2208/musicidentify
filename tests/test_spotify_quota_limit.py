"""Bộ kiểm thử cho công cụ kiểm tra hạn ngạch Spotify (SpotifyQuotaTester).

Kiểm tra:
1. Nạp danh sách bài hát từ file .txt và cấu trúc bài mẫu.
2. Xử lý kịch bản thành công liên tiếp, tính toán tỷ lệ % và thống kê.
3. Kịch bản phát hiện HTTP 429 và tính năng dừng sớm (stop_on_429).
4. Xuất kết quả kiểm thử ra JSON / Excel.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.test_spotify_quota_limit import (
    DEFAULT_SAMPLE_TRACKS,
    QuotaTestSummary,
    SpotifyQuotaTester,
    TrackTestResult,
)


class TestSpotifyQuotaTester(unittest.TestCase):
    """Kiểm thử hoạt động của SpotifyQuotaTester."""

    def setUp(self) -> None:
        self.tester = SpotifyQuotaTester(
            client_id="mock_id",
            client_secret="mock_secret",
            bypass_cache=True,
            delay_between_requests=0.0,
            stop_on_429=False,
        )

    def test_default_sample_tracks_valid(self) -> None:
        """Kiểm tra danh sách bài hát mẫu có sẵn đầy đủ thông tin."""
        self.assertGreaterEqual(len(DEFAULT_SAMPLE_TRACKS), 50)
        for t in DEFAULT_SAMPLE_TRACKS:
            self.assertTrue("title" in t and len(t["title"]) > 0)
            self.assertTrue("artist" in t)

    def test_load_tracks_from_text_file(self) -> None:
        """Kiểm tra nạp danh sách bài hát từ file text nhiều định dạng."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write("# Danh sách test\n")
            f.write("Chúng Ta Của Hiện Tại - Sơn Tùng M-TP\n")
            f.write("Lạ Lùng\n")
            f.write("Blank Space - Taylor Swift\n")
            temp_path = f.name

        try:
            loaded = self.tester.load_tracks_from_file(temp_path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["title"], "Chúng Ta Của Hiện Tại")
            self.assertEqual(loaded[0]["artist"], "Sơn Tùng M-TP")
            self.assertEqual(loaded[1]["title"], "Lạ Lùng")
            self.assertEqual(loaded[1]["artist"], "")
            self.assertEqual(loaded[2]["title"], "Blank Space")
            self.assertEqual(loaded[2]["artist"], "Taylor Swift")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_run_test_successful_flow(self) -> None:
        """Kiểm tra chạy test với mock service trả về kết quả thành công."""
        sample_inputs = [
            {"title": "Track 1", "artist": "Artist 1"},
            {"title": "Track 2", "artist": "Artist 2"},
            {"title": "Track Unknown", "artist": ""},
        ]

        def mock_find_best_match(title: str, artist: str = ""):
            if "Unknown" in title:
                return None
            return {
                "id": f"sp_{title}",
                "title": title,
                "artist": artist or "Artist",
                "spotify_url": f"https://open.spotify.com/track/{title}",
                "match_score": 0.95,
            }

        self.tester.service.find_best_match = MagicMock(side_effect=mock_find_best_match)

        summary = self.tester.run_test(sample_inputs, limit=3)

        self.assertEqual(summary.total_requested, 3)
        self.assertEqual(summary.total_executed, 3)
        self.assertEqual(summary.success_count, 2)
        self.assertEqual(summary.not_found_count, 1)
        self.assertEqual(summary.rate_limited_count, 0)
        self.assertEqual(summary.max_consecutive_success, 2)
        self.assertEqual(len(summary.results), 3)
        self.assertEqual(summary.results[0].status, "SUCCESS")
        self.assertEqual(summary.results[1].status, "SUCCESS")
        self.assertEqual(summary.results[2].status, "NOT_FOUND")

    def test_run_test_stop_on_429(self) -> None:
        """Kiểm tra chức năng dừng sớm khi gặp lỗi HTTP 429 Rate Limit."""
        self.tester.stop_on_429 = True
        sample_inputs = [
            {"title": "Track 1", "artist": "A1"},
            {"title": "Track 2", "artist": "A2"},
            {"title": "Track 3", "artist": "A3"},
        ]

        call_count = 0

        def mock_find_best_match_with_429(title: str, artist: str = ""):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Giả lập khi gặp 429, SpotifyService đánh dấu _quota_exceeded
                self.tester.service._quota_exceeded = True
                return None
            return {
                "id": "1",
                "title": title,
                "spotify_url": "https://open.spotify.com/track/1",
                "match_score": 1.0,
            }

        self.tester.service.find_best_match = MagicMock(side_effect=mock_find_best_match_with_429)

        summary = self.tester.run_test(sample_inputs)

        # Dừng ngay tại bài thứ 2, không chạy bài thứ 3
        self.assertEqual(summary.total_executed, 2)
        self.assertEqual(summary.success_count, 1)
        self.assertEqual(summary.rate_limited_count, 1)
        self.assertEqual(summary.rate_limit_encountered_at, 2)

    def test_export_results_json(self) -> None:
        """Kiểm tra xuất kết quả ra file JSON."""
        summary = QuotaTestSummary(
            total_requested=1,
            total_executed=1,
            success_count=1,
            results=[
                TrackTestResult(
                    index=1,
                    query="Song A",
                    artist="Artist B",
                    status="SUCCESS",
                    spotify_url="https://open.spotify.com/track/123",
                    match_score=0.98,
                    latency_ms=150.0,
                )
            ],
        )

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".json", delete=False) as f:
            temp_json = f.name

        try:
            self.tester.export_results(summary, json_path=temp_json)
            content = json.loads(Path(temp_json).read_text(encoding="utf-8"))
            self.assertEqual(len(content), 1)
            self.assertEqual(content[0]["Tên bài tìm kiếm"], "Song A")
            self.assertEqual(content[0]["Spotify URL"], "https://open.spotify.com/track/123")
        finally:
            Path(temp_json).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
