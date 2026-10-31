"""Kiểm thử unit test cho chức năng import danh sách nghệ sĩ AI từ file Excel vào CSDL.

Sử dụng thư viện chuẩn unittest để tương thích hoàn toàn môi trường hiện tại.
Đảm bảo tuân thủ nghiêm ngặt PEP 8 và Ruff.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import openpyxl
from backend.migrations.import_ai_artists_from_excel import (
    generate_artist_regex,
    import_ai_artists_to_db,
    parse_ai_artists_from_excel,
)


class TestArtistRegexGeneration(unittest.TestCase):
    """Kiểm tra tính chính xác và an toàn của hàm tạo biểu thức chính quy."""

    def test_regex_matching_and_boundary(self) -> None:
        test_cases = [
            ("Athena", "artist: athena", "athenaaa"),
            ("Alaric Nóe", "alaric nóe, feat x", "alaric nóe2"),
            ("D&C", "d&c music", "d&c_music"),
            ("ammotined:", "ammotined:, artist2", "ammotined:extra"),
            ("Ashe$", "ashe$, singer", "ashe$band"),
            ("5SecondDecider", "by 5seconddecider", "5seconddecider123"),
            ("J-Steve Pham", "j-steve pham in concert", "j-steve phammy"),
            ("KEMM BƠ", "kemm bơ track", "kemm bơi"),
        ]

        for name, sample_matched, sample_unmatched in test_cases:
            with self.subTest(name=name):
                pattern = generate_artist_regex(name)
                compiled = re.compile(pattern)

                self.assertIsNotNone(
                    compiled.search(name.lower()),
                    f"Pattern {pattern} không khớp chính tên nó",
                )
                self.assertIsNotNone(
                    compiled.search(sample_matched.lower()),
                    f"Pattern {pattern} không khớp chuỗi hợp lệ: {sample_matched}",
                )
                self.assertIsNone(
                    compiled.search(sample_unmatched.lower()),
                    f"Pattern {pattern} bị false positive với: {sample_unmatched}",
                )


class TestExcelParsing(unittest.TestCase):
    """Kiểm tra khả năng đọc và lọc dữ liệu từ file Excel."""

    def test_parse_valid_and_broken_rows(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "NS AI"

            # Dòng 1: Header
            ws.append(["main", "AI YES"])
            # Dòng 2: Hợp lệ
            ws.append(["Artist A", "Yes"])
            # Dòng 3: Bị lỗi công thức #REF!
            ws.append(["#REF!", "Yes"])
            # Dòng 4: Hợp lệ có khoảng trắng thừa
            ws.append(["  Artist B  ", "Yes"])
            # Dòng 5: Ô trống / None
            ws.append([None, None])
            # Dòng 6: Trùng lặp Artist A (case-insensitive)
            ws.append(["artist a", "Yes"])

            wb.save(tmp_path)
            wb.close()

            artists = parse_ai_artists_from_excel(tmp_path, sheet_name="NS AI")

            self.assertEqual(artists, ["Artist A", "Artist B"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_parse_non_existent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            parse_ai_artists_from_excel("non_existent_file_xyz.xlsx")

    def test_parse_missing_sheet(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            wb = openpyxl.Workbook()
            wb.save(tmp_path)
            wb.close()

            with self.assertRaises(ValueError):
                parse_ai_artists_from_excel(tmp_path, sheet_name="SheetKhongTonTai")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestDatabaseImport(unittest.TestCase):
    """Kiểm tra logic thêm mới và cập nhật bản ghi vào CSDL."""

    @patch("backend.migrations.import_ai_artists_from_excel.parse_ai_artists_from_excel")
    @patch("backend.migrations.import_ai_artists_from_excel.get_db_session")
    def test_import_idempotency(self, mock_get_session: MagicMock, mock_parse: MagicMock) -> None:
        mock_parse.return_value = ["Artist New", "Artist Existing"]

        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        existing_record = MagicMock()
        existing_record.project_name = "Artist Existing"
        existing_record.regex_pattern = r"\bartist existing\b"
        existing_record.is_active = True

        mock_session.scalars.return_value.all.side_effect = [
            [existing_record],  # Lần 1: Lấy danh sách hiện tại
            [existing_record, MagicMock(is_active=True)],  # Lần 2: Đếm tổng sau khi commit
        ]

        stats = import_ai_artists_to_db("dummy.xlsx", sheet_name="NS AI")

        self.assertEqual(stats["total_parsed"], 2)
        self.assertEqual(stats["inserted"], 1)
        self.assertEqual(stats["unchanged"], 1)
        self.assertEqual(stats["updated"], 0)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
