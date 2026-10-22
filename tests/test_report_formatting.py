import sys
from pathlib import Path

import openpyxl

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import unittest

from backend.core.models.entities import DetectionResult, InputType, VerdictType
from backend.repositories.file_report_repository import FileReportRepository
from backend.services.report_service import ReportService, format_display_title


def test_format_display_title_cases():
    print("=== TEST 1: FORMAT_DISPLAY_TITLE LOGIC ===")

    # Case 1: Input là tên bài hát (InputType.SONG_NAME)
    res_song_name = DetectionResult(
        track_id="t1",
        track_title="Tây Thi - MCK",
        raw_query="Tây Thi - MCK",
        source_url="https://youtu.be/s_GrhB1YQVQ",
        input_type=InputType.SONG_NAME,
        is_ai=False,
        verdict_type=VerdictType.CONFIRMED_HUMAN,
        verdict_label="CONFIRMED HUMAN",
        confidence_score=0.9,
        suspected_engine="None",
        reasons=["Giọng hát tự nhiên"],
        detailed_explanation="Nhạc người làm",
    )
    title_1 = format_display_title(res_song_name)
    print(f"Case 1 (Song Name): '{title_1}'")
    assert title_1 == "Tây Thi - MCK", f"Expected 'Tây Thi - MCK', got '{title_1}'"

    # Case 2: Input là link URL có tên bài hát trích xuất từ metadata
    res_url_with_title = DetectionResult(
        track_id="t2",
        track_title="MCK - TÂY THI (Official Audio)",
        raw_query="https://youtu.be/s_GrhB1YQVQ?si=123",
        source_url="https://youtu.be/s_GrhB1YQVQ?si=123",
        input_type=InputType.URL,
        is_ai=False,
        verdict_type=VerdictType.CONFIRMED_HUMAN,
        verdict_label="CONFIRMED HUMAN",
        confidence_score=0.9,
        suspected_engine="None",
        reasons=["Giọng hát tự nhiên"],
        detailed_explanation="Nhạc người làm",
    )
    title_2 = format_display_title(res_url_with_title)
    print(f"Case 2 (URL + Title): '{title_2}'")
    assert title_2 == "MCK - TÂY THI (Official Audio) - https://youtu.be/s_GrhB1YQVQ?si=123", (
        f"Got '{title_2}'"
    )

    # Case 3: Input là link URL nhưng chưa/không lấy được tên (tên trùng link)
    res_url_raw = DetectionResult(
        track_id="t3",
        track_title="https://youtu.be/s_GrhB1YQVQ",
        raw_query="https://youtu.be/s_GrhB1YQVQ",
        source_url="https://youtu.be/s_GrhB1YQVQ",
        input_type=InputType.URL,
        is_ai=False,
        verdict_type=VerdictType.ERROR,
        verdict_label="ERROR",
        confidence_score=0.0,
        suspected_engine="N/A",
        reasons=["Lỗi tải mạng"],
        detailed_explanation="Lỗi",
    )
    title_3 = format_display_title(res_url_raw)
    print(f"Case 3 (URL only fallback): '{title_3}'")
    assert title_3 == "https://youtu.be/s_GrhB1YQVQ", f"Got '{title_3}'"

    # Case 4: Input là local file
    res_local = DetectionResult(
        track_id="t4",
        track_title="my_song_sample",
        raw_query="D:/data/my_song_sample.mp3",
        source_url="D:/data/my_song_sample.mp3",
        input_type=InputType.LOCAL_FILE,
        is_ai=False,
        verdict_type=VerdictType.PROCESSED_HUMAN,
        verdict_label="PROCESSED HUMAN",
        confidence_score=0.85,
        suspected_engine="None",
        reasons=["Mastering studio"],
        detailed_explanation="Nhạc người làm",
    )
    title_4 = format_display_title(res_local)
    print(f"Case 4 (Local File): '{title_4}'")
    assert title_4 == "my_song_sample", f"Got '{title_4}'"

    print("TEST 1 PASSED: format_display_title logic correct!\n")


def test_excel_export_column_values():
    print("=== TEST 2: EXCEL EXPORT INTEGRATION ===")
    test_out_dir = root_dir / "data" / "temp"
    repo = FileReportRepository(output_dir=test_out_dir)
    service = ReportService(report_repo=repo)

    results = [
        DetectionResult(
            track_id="1",
            track_title="Chìm Sâu - MCK",
            raw_query="Chìm Sâu - MCK",
            source_url="https://youtu.be/chim-sau",
            input_type=InputType.SONG_NAME,
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.95,
            suspected_engine="None",
            reasons=["Vocal tự nhiên"],
            detailed_explanation="Người làm",
        ),
        DetectionResult(
            track_id="2",
            track_title="Tại Vì Sao - RHYDER",
            raw_query="https://youtu.be/tai-vi-sao",
            source_url="https://youtu.be/tai-vi-sao",
            input_type=InputType.URL,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN",
            confidence_score=0.90,
            suspected_engine="None",
            reasons=["Auto-tune nghệ thuật"],
            detailed_explanation="Người làm có xử lý",
        ),
    ]

    excel_file = service.export_report(results, format="xlsx")
    assert excel_file is not None and excel_file.exists()

    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active
    assert ws is not None
    rows = list(ws.iter_rows(values_only=True))

    header = rows[0]
    print("Header:", header)
    title_col_idx = header.index("Tên bài hát")
    link_col_idx = header.index("Nguồn link")

    row1 = rows[1]
    print(f"Row 1: Tên='{row1[title_col_idx]}', Link='{row1[link_col_idx]}'")
    # Row 1 (Song name input): chỉ là tên bài hát
    assert row1[title_col_idx] == "Chìm Sâu - MCK"
    assert row1[link_col_idx] == "https://youtu.be/chim-sau"

    row2 = rows[2]
    print(f"Row 2: Tên='{row2[title_col_idx]}', Link='{row2[link_col_idx]}'")
    # Row 2 (URL input): tên bài hát + link
    assert row2[title_col_idx] == "Tại Vì Sao - RHYDER - https://youtu.be/tai-vi-sao"
    assert row2[link_col_idx] == "https://youtu.be/tai-vi-sao"

    print("TEST 2 PASSED: Excel rows match specifications exactly!\n")
    excel_file.unlink(missing_ok=True)


class TestReportFormatting(unittest.TestCase):
    """Kiểm thử định dạng báo cáo và xuất file Excel."""

    def test_format_display_title_cases(self) -> None:
        test_format_display_title_cases()

    def test_excel_export_column_values(self) -> None:
        test_excel_export_column_values()


if __name__ == "__main__":
    unittest.main()
