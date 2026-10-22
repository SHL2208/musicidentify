from __future__ import annotations

import unittest
from pathlib import Path

import openpyxl

from backend.core.models.entities import (
    DetectionResult,
    InputType,
    VerdictType,
    WorkType,
)
from backend.repositories.file_report_repository import FileReportRepository
from backend.services.gemini_classifier_service import GeminiResponseSchema
from backend.services.report_service import ReportService
from backend.services.resolver_service import extract_content_id_metadata


class TestCoverOriginalDetection(unittest.TestCase):
    """Kiểm thử tính năng phân biệt Original vs Cover theo chuẩn PEP8 và Clean Architecture."""

    def test_work_type_enum_and_detection_result_fields(self) -> None:
        """Kiểm tra WorkType enum và các trường mới trong DetectionResult."""
        self.assertEqual(WorkType.ORIGINAL.value, "ORIGINAL")
        self.assertEqual(WorkType.COVER.value, "COVER")
        self.assertEqual(WorkType.REMIX.value, "REMIX")
        self.assertEqual(WorkType.UNCERTAIN.value, "UNCERTAIN")

        result = DetectionResult(
            track_id="test_001",
            track_title="Đừng Làm Trái Tim Anh Đau",
            raw_query="https://youtu.be/test12345",
            source_url="https://youtu.be/test12345",
            input_type=InputType.URL,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            is_ai=False,
            confidence_score=0.98,
            suspected_engine="None",
            reasons=["Âm thanh tự nhiên", "Chính chủ thể hiện"],
            detailed_explanation="Bản phát hành chính thức của nghệ sĩ.",
            work_type=WorkType.ORIGINAL,
            work_label="ORIGINAL (Tác phẩm chính chủ)",
            is_cover=False,
            original_artist="Sơn Tùng M-TP",
            original_song_title="Đừng Làm Trái Tim Anh Đau",
            cover_performer="Sơn Tùng M-TP",
            work_evidence="Chính chủ phát hành trên kênh YouTube chính thức",
        )

        self.assertEqual(result.work_type, WorkType.ORIGINAL)
        self.assertFalse(result.is_cover)
        self.assertEqual(result.original_artist, "Sơn Tùng M-TP")
        self.assertEqual(result.cover_performer, "Sơn Tùng M-TP")

    def test_extract_content_id_metadata(self) -> None:
        """Kiểm tra trích xuất YouTube Content ID từ description."""
        sample_description = """
        Provided to YouTube by Universal Music Group

        Song: Shape of You
        Artist: Ed Sheeran
        Album: ÷ (Divide)
        Licensed to YouTube by: WMG (on behalf of Warner Music UK)
        """
        extracted = extract_content_id_metadata(sample_description)
        self.assertEqual(extracted.get("content_id_song"), "Shape of You")
        self.assertEqual(extracted.get("content_id_artist"), "Ed Sheeran")

        # Trường hợp tiếng Việt
        sample_desc_vi = """
        Bài hát: Cắt Đôi Nỗi Sầu
        Nghệ sĩ: Tăng Duy Tân
        """
        extracted_vi = extract_content_id_metadata(sample_desc_vi)
        self.assertEqual(extracted_vi.get("content_id_song"), "Cắt Đôi Nỗi Sầu")
        self.assertEqual(extracted_vi.get("content_id_artist"), "Tăng Duy Tân")

    def test_gemini_response_schema_cover_fields(self) -> None:
        """Kiểm tra GeminiResponseSchema hỗ trợ các trường tác phẩm Original / Cover."""
        raw_json = {
            "verdict_type": "PROCESSED_HUMAN",
            "is_ai": False,
            "confidence_score": 0.95,
            "suspected_engine": "None",
            "reasons": ["Giọng hát thật có Auto-Tune", "Giai điệu cover ca khúc nổi tiếng"],
            "detailed_explanation": "Bản thu cover do Hương Ly thể hiện lại bài của Sơn Tùng M-TP.",
            "work_type": "COVER",
            "is_cover": True,
            "original_artist": "Sơn Tùng M-TP",
            "original_song_title": "Đừng Làm Trái Tim Anh Đau",
            "cover_performer": "Hương Ly",
        }
        parsed = GeminiResponseSchema.model_validate(raw_json)
        self.assertEqual(parsed.work_type, "COVER")
        self.assertTrue(parsed.is_cover)
        self.assertEqual(parsed.original_artist, "Sơn Tùng M-TP")
        self.assertEqual(parsed.cover_performer, "Hương Ly")

    def test_report_service_export_cover_columns(self) -> None:
        """Kiểm tra file Excel xuất ra có đầy đủ các cột Original / Cover."""
        out_dir = Path("data/temp/test_export")
        out_dir.mkdir(parents=True, exist_ok=True)
        repo = FileReportRepository(output_dir=out_dir)
        service = ReportService(report_repo=repo)

        r1 = DetectionResult(
            track_id="t1",
            track_title="Bài Hát Gốc",
            raw_query="query1",
            input_type=InputType.SONG_NAME,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN",
            is_ai=False,
            confidence_score=0.95,
            suspected_engine="None",
            reasons=["Bản thu gốc phòng thu"],
            detailed_explanation="Bản phát hành chính thức.",
            work_type=WorkType.ORIGINAL,
            work_label="ORIGINAL (Bản gốc)",
            is_cover=False,
            original_artist="Ca Sĩ A",
            original_song_title="Bài Hát Gốc",
        )

        r2 = DetectionResult(
            track_id="t2",
            track_title="Bài Hát Cover",
            raw_query="query2",
            input_type=InputType.SONG_NAME,
            verdict_type=VerdictType.HYBRID_AI,
            verdict_label="HYBRID AI",
            is_ai=True,
            confidence_score=0.92,
            suspected_engine="RVC/Diff-SVC",
            reasons=["AI Voice Clone"],
            detailed_explanation="Bản cover bằng giọng AI clone.",
            work_type=WorkType.COVER,
            work_label="COVER (Bản hát lại)",
            is_cover=True,
            original_artist="Ca Sĩ A",
            original_song_title="Bài Hát Gốc",
            cover_performer="AI Voice Clone",
        )

        exported_path = service.export_report([r1, r2], format="xlsx")
        self.assertIsNotNone(exported_path)
        self.assertTrue(exported_path.exists())

        wb = openpyxl.load_workbook(exported_path)
        sheet = wb.active
        headers = [cell.value for cell in sheet[1]]

        self.assertIn("Tác phẩm", headers)
        self.assertIn("Là bản Cover", headers)
        self.assertIn("Nghệ sĩ bài gốc", headers)
        self.assertIn("Tên bài gốc", headers)

        # Kiểm tra nội dung dòng 2 (r1) và dòng 3 (r2)
        row2 = [cell.value for cell in sheet[2]]
        row3 = [cell.value for cell in sheet[3]]

        cover_idx = headers.index("Là bản Cover")
        self.assertEqual(row2[cover_idx], "Không")
        self.assertEqual(row3[cover_idx], "Có (Cover)")


if __name__ == "__main__":
    unittest.main()
