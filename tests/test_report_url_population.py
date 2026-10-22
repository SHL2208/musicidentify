import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import openpyxl
import pandas as pd

from backend.core.models.entities import DetectionResult, InputType, VerdictType
from backend.repositories.file_report_repository import FileReportRepository
from backend.services.report_service import ReportService
from backend.services.resolver_service import InputResolverService


class TestReportUrlPopulation(unittest.TestCase):
    """Kiểm thử tính toàn vẹn và hiện diện của URL file nhạc trong file báo cáo report."""

    def setUp(self):
        self.output_dir = Path("data/temp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo = FileReportRepository(output_dir=self.output_dir)
        self.report_service = ReportService(report_repo=self.repo)

    def test_report_includes_source_url(self):
        """Đảm bảo cột 'Nguồn link' trong file Excel luôn có URL khi có source_url hoặc raw_query."""
        results = [
            DetectionResult(
                track_id="t1",
                track_title="Một Bóng Mình Ta - Ken Quách",
                raw_query="Một Bóng Mình Ta - Ken Quách",
                source_url="https://www.youtube.com/watch?v=19e48730abc",
                input_type=InputType.SONG_NAME,
                is_ai=True,
                verdict_type=VerdictType.HYBRID_AI,
                verdict_label="HYBRID / AI-ASSISTED (Bán nhân tạo)",
                confidence_score=0.94,
                suspected_engine="Suno",
                reasons=["Dấu vết vocoder"],
                detailed_explanation="Bản phân tích quang phổ chi tiết.",
            ),
            DetectionResult(
                track_id="t2",
                track_title="audio_local_test",
                raw_query="D:/music/sample_test.wav",
                source_url="D:/music/sample_test.wav",
                input_type=InputType.LOCAL_FILE,
                is_ai=False,
                verdict_type=VerdictType.CONFIRMED_HUMAN,
                verdict_label="CONFIRMED HUMAN (Người làm)",
                confidence_score=0.95,
                suspected_engine="None",
                reasons=["Âm học phòng thu tự nhiên"],
                detailed_explanation="Nhạc người làm chuẩn phòng thu.",
            ),
            DetectionResult(
                track_id="t3",
                track_title="Test Query URL Fallback",
                raw_query="https://soundcloud.com/artist/track-demo",
                source_url=None,  # source_url bị None nhưng raw_query là URL
                input_type=InputType.URL,
                is_ai=False,
                verdict_type=VerdictType.PROCESSED_HUMAN,
                verdict_label="PROCESSED HUMAN",
                confidence_score=0.90,
                suspected_engine="None",
                reasons=["Xử lý phòng thu"],
                detailed_explanation="Giải thích.",
            ),
        ]

        excel_path = self.report_service.export_report(results, format="xlsx")
        self.assertIsNotNone(excel_path)
        self.assertTrue(excel_path.exists())

        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))

        header = rows[0]
        self.assertIn("Nguồn link", header)
        link_idx = header.index("Nguồn link")

        # Row 1: YouTube URL
        row1 = rows[1]
        self.assertEqual(row1[link_idx], "https://www.youtube.com/watch?v=19e48730abc")

        # Row 2: Local file path
        row2 = rows[2]
        self.assertEqual(row2[link_idx], "D:/music/sample_test.wav")

        # Row 3: Fallback from raw_query
        row3 = rows[3]
        self.assertEqual(row3[link_idx], "https://soundcloud.com/artist/track-demo")

        excel_path.unlink(missing_ok=True)

    def test_parse_batch_file_with_url_column(self):
        """Đảm bảo parse_batch_file tự động nhận diện cột URL và gán vào resolved_url."""
        resolver = InputResolverService()
        test_file = self.output_dir / "test_batch_with_url.xlsx"

        df = pd.DataFrame(
            [
                {
                    "Tên bài hát": "Bài Test 1",
                    "Ca sĩ": "Ca sĩ A",
                    "URL": "https://www.youtube.com/watch?v=sample1",
                },
                {
                    "Tên bài hát": "Bài Test 2",
                    "Ca sĩ": "Ca sĩ B",
                    "URL": "https://www.youtube.com/watch?v=sample2",
                },
            ]
        )
        df.to_excel(test_file, index=False)

        tracks = resolver.parse_batch_file(test_file)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].resolved_url, "https://www.youtube.com/watch?v=sample1")
        self.assertEqual(tracks[0].track_title, "Bài Test 1")
        self.assertEqual(tracks[1].resolved_url, "https://www.youtube.com/watch?v=sample2")
        self.assertEqual(tracks[1].track_title, "Bài Test 2")

        test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
