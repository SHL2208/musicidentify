"""Bộ kiểm thử cho chức năng Tìm kiếm Spotify Hàng loạt (Batch Spotify Search).

Kiểm tra:
1. SpotifyService.batch_search_tracks (xử lý đa luồng, bảo toàn thứ tự, gọi on_progress)
2. InputResolverService.parse_spotify_batch_input (bóc tách text nhiều dòng, DataFrame, file)
3. ReportService.export_spotify_batch_report (ghép kết quả 1-1 bảo toàn cột gốc, xuất Excel/CSV)
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock

import pandas as pd

from backend.services.report_service import ReportService
from backend.services.resolver_service import InputResolverService
from backend.services.spotify_service import SpotifyService


class TestSpotifyBatchSearch(unittest.TestCase):
    """Kiểm thử toàn diện quy trình tìm kiếm hàng loạt bài hát Spotify."""

    def setUp(self) -> None:
        self.spotify_service = SpotifyService(client_id="test_id", client_secret="test_secret")
        self.resolver_service = InputResolverService(spotify_service=self.spotify_service)
        self.report_service = ReportService(report_repo=MagicMock())

    def test_batch_search_tracks_preserves_order_and_callbacks(self) -> None:
        """Kiểm tra batch_search_tracks giữ đúng thứ tự và gọi progress callback."""
        queries = ["Bài 1 - Ca sĩ A", "Bài 2 - Ca sĩ B", "Bài 3 - Ca sĩ C"]

        # Mock find_best_match trả về kết quả giả lập theo query
        def mock_find_best_match(query: str, artist: str = "", limit: int = 5):
            idx = query.split()[1] if len(query.split()) > 1 else "0"
            return {
                "id": f"track_{idx}",
                "title": f"Track {idx}",
                "artist": artist or "Artist",
                "album": "Album Test",
                "release_date": "2024-01-01",
                "spotify_url": f"https://open.spotify.com/track/{idx}",
                "preview_url": f"https://p.scdn.co/{idx}",
                "embed_url": f"https://open.spotify.com/embed/track/{idx}",
                "duration_sec": 200.0,
                "popularity": 85,
                "isrc": "VN001",
                "match_score": 0.95,
            }

        self.spotify_service.find_best_match = MagicMock(side_effect=mock_find_best_match)

        callback_calls = []

        def mock_callback(completed: int, total: int, res: dict) -> None:
            callback_calls.append((completed, total, res.get("input_query")))

        results = self.spotify_service.batch_search_tracks(
            queries=queries,
            max_workers=2,
            on_progress=mock_callback,
        )

        # Kiểm tra kết quả trả về đúng số lượng và đúng thứ tự ban đầu
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["input_query"], "Bài 1")
        self.assertEqual(results[0]["spotify_title"], "Track 1")
        self.assertEqual(results[0]["spotify_url"], "https://open.spotify.com/track/1")

        self.assertEqual(results[1]["input_query"], "Bài 2")
        self.assertEqual(results[1]["spotify_title"], "Track 2")
        self.assertEqual(results[1]["spotify_url"], "https://open.spotify.com/track/2")

        self.assertEqual(results[2]["input_query"], "Bài 3")
        self.assertEqual(results[2]["spotify_title"], "Track 3")
        self.assertEqual(results[2]["spotify_url"], "https://open.spotify.com/track/3")

        # Kiểm tra callback được gọi đủ 3 lần
        self.assertEqual(len(callback_calls), 3)
        self.assertEqual(callback_calls[-1][0], 3)
        self.assertEqual(callback_calls[-1][1], 3)

    def test_parse_spotify_batch_input_multiline_text(self) -> None:
        """Kiểm tra parse danh sách từ text nhiều dòng, làm sạch đầu vào."""
        raw_text = """
        1. Em Của Ngày Hôm Qua - Sơn Tùng M-TP
        2. Cắt Đôi Nỗi Sầu - Tăng Duy Tân
        
        - Lạc Trôi
        * Nơi Này Có Anh
        """
        queries, df = self.resolver_service.parse_spotify_batch_input(raw_text)

        self.assertEqual(len(queries), 4)
        self.assertEqual(queries[0]["query"], "Em Của Ngày Hôm Qua")
        self.assertEqual(queries[0]["artist"], "Sơn Tùng M-TP")
        self.assertEqual(queries[1]["query"], "Cắt Đôi Nỗi Sầu")
        self.assertEqual(queries[1]["artist"], "Tăng Duy Tân")
        self.assertEqual(queries[2]["query"], "Lạc Trôi")
        self.assertEqual(queries[2]["artist"], "")
        self.assertEqual(queries[3]["query"], "Nơi Này Có Anh")
        self.assertEqual(queries[3]["artist"], "")

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)

    def test_parse_spotify_batch_input_dataframe(self) -> None:
        """Kiểm tra parse từ DataFrame có các tên cột phổ biến và cột tùy biến."""
        sample_data = {
            "Mã bài hát": ["BH01", "BH02"],
            "Tên bản ghi": ["Chúng Ta Của Hiện Tại", "Waiting For You"],
            "Ca sĩ": ["Sơn Tùng M-TP", "MONO"],
            "Ghi chú": ["Ưu tiên 1", "Ưu tiên 2"],
        }
        input_df = pd.DataFrame(sample_data)

        queries, returned_df = self.resolver_service.parse_spotify_batch_input(input_df)

        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0]["query"], "Chúng Ta Của Hiện Tại")
        self.assertEqual(queries[0]["artist"], "Sơn Tùng M-TP")
        self.assertEqual(queries[1]["query"], "Waiting For You")
        self.assertEqual(queries[1]["artist"], "MONO")
        # Bảo toàn 100% cột ban đầu
        self.assertListEqual(list(returned_df.columns), ["Mã bài hát", "Tên bản ghi", "Ca sĩ", "Ghi chú"])

    def test_export_spotify_batch_report_xlsx_and_csv(self) -> None:
        """Kiểm tra xuất báo cáo batch Spotify ra Excel và CSV giữ nguyên cột gốc."""
        original_df = pd.DataFrame({
            "Mã ID": ["ID_101", "ID_102"],
            "Tên bài": ["Bài Hát 1", "Bài Hát 2"],
            "Nhạc sĩ": ["NS A", "NS B"],
        })

        results = [
            {
                "index": 1,
                "input_query": "Bài Hát 1",
                "input_artist": "Ca sĩ X",
                "is_found": True,
                "spotify_id": "track1",
                "spotify_title": "Bài Hát 1 (Official)",
                "spotify_artist": "Ca sĩ X",
                "spotify_album": "Album Y",
                "spotify_release_date": "2023-01-01",
                "spotify_url": "https://open.spotify.com/track/track1",
                "spotify_preview_url": "https://p.scdn.co/preview1",
                "spotify_embed_url": "https://open.spotify.com/embed/track/track1",
                "spotify_isrc": "VN12345678",
                "duration_sec": 210.0,
                "popularity": 80,
                "match_score": 0.98,
            },
            {
                "index": 2,
                "input_query": "Bài Hát 2",
                "input_artist": "",
                "is_found": False,
                "spotify_id": None,
                "spotify_title": None,
                "spotify_artist": None,
                "spotify_album": None,
                "spotify_release_date": None,
                "spotify_url": "",
                "spotify_preview_url": None,
                "spotify_embed_url": None,
                "spotify_isrc": None,
                "duration_sec": 0.0,
                "popularity": 0,
                "match_score": 0.0,
            },
        ]

        # 1. Xuất file Excel (.xlsx)
        xlsx_bytes = self.report_service.export_spotify_batch_report(original_df, results, format="xlsx")
        self.assertIsInstance(xlsx_bytes, bytes)
        self.assertGreater(len(xlsx_bytes), 0)

        # Đọc lại từ buffer để đối soát dữ liệu
        read_df = pd.read_excel(io.BytesIO(xlsx_bytes), keep_default_na=False)
        self.assertEqual(len(read_df), 2)
        # Kiểm tra cột gốc được bảo toàn
        self.assertIn("Mã ID", read_df.columns)
        self.assertIn("Nhạc sĩ", read_df.columns)
        # Kiểm tra các cột Spotify mới
        self.assertIn("Link Spotify", read_df.columns)
        self.assertIn("Spotify Tìm thấy", read_df.columns)
        self.assertIn("Độ tương đồng", read_df.columns)

        # Hàng 1 tìm thấy
        self.assertEqual(read_df.iloc[0]["Spotify Tìm thấy"], "Có 🟢")
        self.assertEqual(read_df.iloc[0]["Link Spotify"], "https://open.spotify.com/track/track1")
        self.assertEqual(read_df.iloc[0]["Độ tương đồng"], "98%")

        # Hàng 2 không tìm thấy
        self.assertEqual(read_df.iloc[1]["Spotify Tìm thấy"], "Không ⚪")
        self.assertEqual(read_df.iloc[1]["Link Spotify"], "")

        # 2. Xuất file CSV
        csv_bytes = self.report_service.export_spotify_batch_report(original_df, results, format="csv")
        self.assertIsInstance(csv_bytes, bytes)
        self.assertGreater(len(csv_bytes), 0)
        # Đảm bảo có UTF-8 BOM
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
