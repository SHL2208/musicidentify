import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Thêm root vào sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.core.container import get_container
from backend.core.models.entities import (
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from frontend.components.batch_view import format_detection_row


class TestSongTypeMetadataPersistence(unittest.TestCase):
    """Kiểm thử tính toàn vẹn của việc trích xuất, lưu trữ CSDL và hiển thị Song Type, Distributor, Release Date."""

    def setUp(self):
        self.container = get_container(force_new=True)
        self.resolver = self.container.resolver
        self.track_repo = self.container.track_repository
        self.detection_service = self.container.detection_service
        self.report_service = self.container.report_service

    def test_df_extraction_song_type_and_metadata(self):
        """Kiểm tra việc tự động trích xuất các cột Song Type, Distributor, Release Date từ DataFrame."""
        df = pd.DataFrame(
            [
                {
                    "song title": "Bài Hát Test 1",
                    "artist": "Ca Sĩ A",
                    "Song Type": "Original",
                    "distributor": "Believe Music",
                    "release_date": "2023-05-15",
                },
                {
                    "song title": "Bài Hát Test 2",
                    "artist": "Ca Sĩ B",
                    "loại tác phẩm": "Cover",
                    "nhà phân phối": "Zing MP3",
                    "ngày phát hành": 20240101,  # Số nguyên YYYYMMDD
                },
            ]
        )
        tracks = self.resolver._extract_tracks_from_df(df)
        self.assertEqual(len(tracks), 2)

        # Track 1
        t1 = tracks[0]
        self.assertEqual(t1.track_title, "Bài Hát Test 1")
        self.assertEqual(t1.artist, "Ca Sĩ A")
        self.assertEqual(t1.song_type, "Original")
        self.assertEqual(t1.distributor, "Believe Music")
        self.assertEqual(t1.release_date, "2023-05-15")

        # Track 2
        t2 = tracks[1]
        self.assertEqual(t2.track_title, "Bài Hát Test 2")
        self.assertEqual(t2.artist, "Ca Sĩ B")
        self.assertEqual(t2.song_type, "Cover")
        self.assertEqual(t2.distributor, "Zing MP3")
        self.assertEqual(t2.release_date, "2024-01-01")

    def test_repository_persistence_and_model_mapping(self):
        """Kiểm tra việc lưu vào CSDL và ánh xạ ngược từ TrackModel sang DetectionResult."""
        track = TrackItem(
            id="test_meta_001",
            raw_query="Test Song Metadata",
            input_type=InputType.SONG_NAME,
            track_title="Test Song Metadata",
            artist="Artist X",
            song_type="REMIX",
            distributor="Sony Music Entertainment",
            release_date="2022-11-20",
        )
        saved_track = self.track_repo.save_track(track)
        self.assertEqual(saved_track.song_type, "REMIX")
        self.assertEqual(saved_track.distributor, "Sony Music Entertainment")
        self.assertEqual(saved_track.release_date, "2022-11-20")

        # Đọc lại từ CSDL
        loaded_track = self.track_repo.get_track("test_meta_001")
        self.assertIsNotNone(loaded_track)
        self.assertEqual(loaded_track.song_type, "REMIX")
        self.assertEqual(loaded_track.distributor, "Sony Music Entertainment")
        self.assertEqual(loaded_track.release_date, "2022-11-20")

        # Lưu DetectionResult và đọc lại
        result = DetectionResult(
            track_id="test_meta_001",
            track_title="Test Song Metadata",
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm)",
            confidence_score=0.92,
            suspected_engine="None",
            reasons=["Âm thanh tự nhiên"],
            detailed_explanation="Giải thích test",
            song_type="REMIX",
            distributor="Sony Music Entertainment",
            release_date="2022-11-20",
        )
        self.track_repo.save_result(result)

        loaded_result = self.track_repo.get_result("test_meta_001")
        self.assertIsNotNone(loaded_result)
        self.assertEqual(loaded_result.song_type, "REMIX")
        self.assertEqual(loaded_result.distributor, "Sony Music Entertainment")
        self.assertEqual(loaded_result.release_date, "2022-11-20")

    def test_detection_service_metadata_auto_enrichment(self):
        """Kiểm tra cơ chế làm giàu tự động khi metadata thiếu từ file đầu vào nhưng có từ YouTube Content ID."""
        track = TrackItem(
            id="test_auto_enrich_unique_999",
            raw_query="Bài Hát Chưa Có Metadata 999",
            input_type=InputType.URL,
            track_title="Bài Hát Chưa Có Metadata 999",
            artist="Ca Sĩ Test",
            resolved_url="https://www.youtube.com/watch?v=mock_unique_999",
            song_type=None,
            distributor=None,
            release_date=None,
        )
        self.track_repo.save_track(track)

        # Mock audio engine trả về metadata từ YouTube Content ID
        mock_meta = {
            "title": "Bài Hát Chưa Có Metadata (Official Music Video)",
            "content_id_licensed_by": "Universal Music Group (UMG)",
            "upload_date": "20231225",
            "work_type": "COVER",
        }
        self.detection_service.audio_engine.process_track_audio = MagicMock(
            return_value=(None, mock_meta)
        )
        self.detection_service.signal_analyzer.check_heuristics = MagicMock(
            return_value={
                "is_ai": False,
                "confidence": 0.95,
                "engine": "None",
                "reason": "Xác thực qua Content ID",
                "verdict_type": VerdictType.CONFIRMED_HUMAN,
            }
        )

        res = self.detection_service.detect_single(track, force_rescan=True)

        # Kiểm tra TrackItem & DB đã được cập nhật
        self.assertEqual(track.distributor, "Universal Music Group (UMG)")
        self.assertEqual(track.release_date, "2023-12-25")
        self.assertEqual(track.song_type, "COVER")

        # Kiểm tra DetectionResult nhận được metadata
        self.assertEqual(res.distributor, "Universal Music Group (UMG)")
        self.assertEqual(res.release_date, "2023-12-25")
        self.assertIn("COVER", res.song_type)

    def test_format_detection_row_ui(self):
        """Kiểm tra hàm format_detection_row trên UI sinh đủ các cột Song Type, Distributor, Release Date."""
        res = DetectionResult(
            track_id="test_ui_row",
            track_title="Hát Về Cây Lúa",
            source_url="https://youtube.com/watch?v=12345",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.98,
            suspected_engine="None",
            reasons=["Dải tần thoáng", "Giọng mộc"],
            detailed_explanation="Chi tiết",
            song_type="ORIGINAL",
            distributor="Hãng Đĩa Dihavina",
            release_date="2020-03-08",
        )
        row = format_detection_row(res)

        self.assertIn("Loại tác phẩm", row)
        self.assertIn("Nhà phân phối", row)
        self.assertIn("Ngày phát hành", row)
        self.assertEqual(row["Loại tác phẩm"], "ORIGINAL")
        self.assertEqual(row["Nhà phân phối"], "Hãng Đĩa Dihavina")
        self.assertEqual(row["Ngày phát hành"], "2020-03-08")

    def test_report_service_export_columns(self):
        """Kiểm tra file xuất báo cáo Excel chứa đầy đủ 3 cột metadata mới."""
        res = DetectionResult(
            track_id="test_export_row",
            track_title="Ca Khúc Báo Cáo",
            source_url="https://youtube.com/watch?v=export123",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED",
            confidence_score=0.99,
            suspected_engine="Suno v3.5",
            reasons=["Fakeprint cao", "Rigidity cao"],
            detailed_explanation="Chi tiết phân tích AI",
            song_type="REMIX",
            distributor="AI Music Network",
            release_date="2024-05-20",
        )
        out_file = self.report_service.export_report([res], format="xlsx")
        self.assertIsNotNone(out_file)
        self.assertTrue(Path(out_file).exists())

        # Đọc lại bằng pandas và kiểm tra cột
        df = pd.read_excel(out_file)
        self.assertIn("Loại tác phẩm (Song Type)", df.columns)
        self.assertIn("Nhà phân phối", df.columns)
        self.assertIn("Ngày phát hành", df.columns)

        self.assertEqual(df["Loại tác phẩm (Song Type)"].iloc[0], "REMIX")
        self.assertEqual(df["Nhà phân phối"].iloc[0], "AI Music Network")
        self.assertEqual(str(df["Ngày phát hành"].iloc[0]), "2024-05-20")

    def test_incomplete_track_save_does_not_wipe_db(self):
        """Kiểm tra việc lưu TrackItem bị khuyết dữ liệu (VD: từ file Excel thô) KHÔNG đè mất URL, duration, song_type trong DB."""
        # 1. Tạo và lưu bản ghi hoàn chỉnh trong CSDL
        initial_track = TrackItem(
            id="test_no_wipe_001",
            raw_query="Bài Hát Chống Mất Data",
            input_type=InputType.SONG_NAME,
            track_title="Bài Hát Chống Mất Data",
            artist="Ca Sĩ Bản Quyền",
            resolved_url="https://www.youtube.com/watch?v=nowipe123",
            duration_sec=245.5,
            song_type="ORIGINAL",
            distributor="Believe Music",
            release_date="2023-11-15",
            status="COMPLETED",
        )
        self.track_repo.save_track(initial_track)

        res = DetectionResult(
            track_id="test_no_wipe_001",
            track_title="Bài Hát Chống Mất Data",
            source_url="https://www.youtube.com/watch?v=nowipe123",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.99,
            suspected_engine="None",
            reasons=["Âm thanh tự nhiên"],
            detailed_explanation="Thẩm âm người thật",
            song_type="ORIGINAL",
            distributor="Believe Music",
            release_date="2023-11-15",
            duration_sec=245.5,
        )
        self.track_repo.save_result(res)

        # 2. Giả lập người dùng quét lại file Excel thô (chỉ có Tên bài + Ca sĩ, các trường khác đều None)
        raw_excel_track = TrackItem(
            id="test_no_wipe_001",
            raw_query="Bài Hát Chống Mất Data",
            input_type=InputType.SONG_NAME,
            track_title="Bài Hát Chống Mất Data",
            artist="Ca Sĩ Bản Quyền",
            resolved_url=None,
            duration_sec=None,
            song_type=None,
            distributor=None,
            release_date=None,
        )
        saved = self.track_repo.save_track(raw_excel_track)

        # 3. Đảm bảo dữ liệu quý giá trong CSDL không bị xóa đè thành None
        self.assertEqual(saved.resolved_url, "https://www.youtube.com/watch?v=nowipe123")
        self.assertEqual(saved.duration_sec, 245.5)
        self.assertEqual(saved.song_type, "ORIGINAL")
        self.assertEqual(saved.distributor, "Believe Music")
        self.assertEqual(saved.release_date, "2023-11-15")

        # 4. Khi detect_single chạy qua Step 0a (Cache Hit), kết quả trả về phải giữ nguyên đầy đủ
        cached_result = self.detection_service.detect_single(raw_excel_track, force_rescan=False)
        self.assertTrue(cached_result.is_cached)
        self.assertEqual(cached_result.source_url, "https://www.youtube.com/watch?v=nowipe123")
        self.assertEqual(cached_result.duration_sec, 245.5)
        self.assertEqual(cached_result.song_type, "ORIGINAL")
        self.assertEqual(cached_result.distributor, "Believe Music")


if __name__ == "__main__":
    unittest.main()
