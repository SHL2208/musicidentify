import sys
import unittest
from pathlib import Path

import openpyxl
import pandas as pd

# Thêm root vào sys.path
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

from backend.core.container import get_container
from backend.core.models.entities import DetectionResult, InputType, TrackItem, VerdictType
from backend.services.gemini_classifier_service import GeminiClassifierService as GeminiClassifier

_default_container = get_container()
InputResolver = _default_container.resolver
SignalAnalyzer = _default_container.signal_analyzer


class BatchManager:
    """Helper wrapper for batch testing using backend DI container."""

    def __init__(self, gemini_classifier=None):
        self.container = get_container(custom_classifier=gemini_classifier, force_new=True)
        self.container.track_repository.find_result_by_query_or_url = lambda *args, **kwargs: None
        self.audio_engine = self.container.audio_engine
        self.batch_service = self.container.batch_service

    def process_batch(self, tracks, on_progress=None):
        return self.batch_service.process_batch(tracks, on_progress=on_progress)


class MockClassifier(GeminiClassifier):
    def __init__(self):
        super().__init__(api_key="mock_key")

    def classify_track(self, *args, **kwargs):
        track_id = kwargs.get("track_id", "mock_id")
        track_title = kwargs.get("track_title", "Mock Title")
        source_url = kwargs.get("source_url")
        return DetectionResult(
            track_id=track_id,
            track_title=track_title,
            source_url=source_url,
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.95,
            suspected_engine="Suno v3.5",
            reasons=["Dấu hiệu giọng hát AI", "Stem bleed"],
            detailed_explanation="Bản thu AI",
            acoustic_metrics=None,
            spectrogram_image_path=None,
        )


def test_batch_5_tracks_with_errors():
    print("=== TEST 1: BATCH PROCESSING 5 TRACKS (SUCCESS & FAILURES) ===")
    bm = BatchManager(gemini_classifier=MockClassifier())

    # Tạo 5 bài hát giả lập: 3 bài hợp lệ, 2 bài gây lỗi
    tracks = [
        TrackItem(
            id="t1", raw_query="Song 1", input_type=InputType.SONG_NAME, track_title="Song 1"
        ),
        TrackItem(
            id="t2",
            raw_query="Song 2 (Error)",
            input_type=InputType.URL,
            resolved_url="invalid://broken-url",
            track_title="Song 2 (Error)",
        ),
        TrackItem(
            id="t3", raw_query="Song 3", input_type=InputType.SONG_NAME, track_title="Song 3"
        ),
        TrackItem(
            id="t4",
            raw_query="Song 4 (Error)",
            input_type=InputType.LOCAL_FILE,
            track_title="Song 4 (Error)",
        ),
        TrackItem(
            id="t5", raw_query="Song 5", input_type=InputType.SONG_NAME, track_title="Song 5"
        ),
    ]

    # Mock audio_engine.process_track_audio để không cần kết nối mạng
    dummy_wav = Path(root_dir / "data" / "temp" / "dummy_test.wav")
    dummy_wav.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf

    sf.write(str(dummy_wav), np.zeros((44100 * 2, 2)), 44100)

    def mock_process_audio(track: TrackItem):
        if "Error" in (track.track_title or ""):
            raise RuntimeError(f"Simulated network download failure for {track.track_title}")
        return dummy_wav, {"upload_date": "20240101", "title": track.track_title}

    bm.audio_engine.process_track_audio = mock_process_audio

    progress_events = []

    def on_progress(idx, total, current_track, res, err):
        progress_events.append(
            (idx, current_track.track_title, res.verdict_label if res else None, err)
        )

    report = bm.process_batch(tracks, on_progress=on_progress)

    # 1. Kiểm tra số lượng kết quả
    print(f"Total input tracks: {len(tracks)}")
    print(f"Report total_tracks: {report.total_tracks}")
    print(f"Report processed_count: {report.processed_count}")
    print(f"Report failed_count: {report.failed_count}")
    print(f"Report results count: {len(report.results)}")

    assert len(report.results) == 5, f"Expected 5 results, got {len(report.results)}"
    assert report.total_tracks == 5
    assert report.failed_count == 2
    assert len(progress_events) == 5

    # 2. Kiểm tra từng dòng trong kết quả
    assert report.results[0].verdict_type == VerdictType.AI_GENERATED
    assert report.results[1].verdict_type == VerdictType.ERROR
    assert "Simulated network download failure" in report.results[1].reasons[0]
    assert report.results[2].verdict_type == VerdictType.AI_GENERATED
    assert report.results[3].verdict_type == VerdictType.ERROR
    assert report.results[4].verdict_type == VerdictType.AI_GENERATED

    # 3. Kiểm tra file Excel xuất ra
    assert report.report_file_path is not None
    excel_path = Path(report.report_file_path)
    assert excel_path.exists()

    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    assert ws is not None
    rows = list(ws.iter_rows(values_only=True))
    # 1 header + 5 data rows = 6 rows
    print(f"Excel total rows: {len(rows)} (1 header + 5 items)")
    assert len(rows) == 6, f"Expected 6 rows in Excel (1 header + 5 data), got {len(rows)}"

    for i in range(1, 6):
        stt = rows[i][0]
        title = rows[i][1]
        verdict = rows[i][2]
        print(f"  Row {i}: STT={stt}, Title={title}, Verdict={verdict}")
        assert stt == i

    print("TEST 1 PASSED: 5 tracks input -> Exactly 5 tracks output in report & Excel!\n")
    dummy_wav.unlink(missing_ok=True)
    excel_path.unlink(missing_ok=True)


def test_excel_header_preservation():
    print("=== TEST 2: EXCEL INPUT HEADER VS DATA DETECTION ===")
    temp_excel = root_dir / "data" / "temp" / "test_5_songs.xlsx"
    temp_excel.parent.mkdir(parents=True, exist_ok=True)

    # Trường hợp 1: File Excel KHÔNG CÓ header (5 bài hát trực tiếp từ dòng 1)
    df_no_head = pd.DataFrame(
        [
            ["https://youtu.be/song1"],
            ["https://youtu.be/song2"],
            ["https://youtu.be/song3"],
            ["https://youtu.be/song4"],
            ["https://youtu.be/song5"],
        ]
    )
    df_no_head.to_excel(temp_excel, header=False, index=False)

    tracks_no_head = InputResolver.parse_batch_file(temp_excel)
    print(f"File without header: Read {len(tracks_no_head)} tracks (Expected: 5)")
    assert len(tracks_no_head) == 5, f"Expected 5 tracks, got {len(tracks_no_head)}"

    # Trường hợp 2: File Excel CÓ header "Tên bài hát / Link"
    df_with_head = pd.DataFrame(
        {
            "Tên bài hát / Link": [
                "https://youtu.be/song1",
                "https://youtu.be/song2",
                "https://youtu.be/song3",
                "https://youtu.be/song4",
                "https://youtu.be/song5",
            ]
        }
    )
    df_with_head.to_excel(temp_excel, header=True, index=False)

    tracks_with_head = InputResolver.parse_batch_file(temp_excel)
    print(
        f"File with header 'Tên bài hát / Link': Read {len(tracks_with_head)} tracks (Expected: 5)"
    )
    assert len(tracks_with_head) == 5, f"Expected 5 tracks, got {len(tracks_with_head)}"

    print("TEST 2 PASSED: Both header and headerless Excel preserve all 5 tracks!\n")
    temp_excel.unlink(missing_ok=True)


def test_ai_heuristics():
    print("=== TEST 3: AI HEURISTICS & BULK NUMBERING DETECTION ===")
    # Bài hát có đánh số kiểu dự án AI (như bài user test: 'Lần Này Theo Lý Trí - HuyH (Bài số 83/100)')
    meta_batch = {
        "title": "Lần Này Theo Lý Trí - HuyH (Bài số 83/100)",
        "description": "Nhạc hay mỗi ngày, dự án sáng tác",
        "upload_date": "20240919",
    }
    h_res = SignalAnalyzer.check_heuristics(meta_batch)
    print(f"Heuristics for '(Bài số 83/100)': {h_res}")
    assert h_res is not None
    assert h_res["is_ai"] is True
    assert h_res["verdict_type"] == VerdictType.AI_GENERATED
    print("TEST 3 PASSED: Bulk AI numbering recognized successfully!\n")


class TestBatchCompleteness(unittest.TestCase):
    """Kiểm thử tính toàn vẹn và khả năng phục hồi lỗi của xử lý hàng loạt (Batch)."""

    def test_batch_5_tracks_with_errors(self) -> None:
        test_batch_5_tracks_with_errors()

    def test_excel_header_preservation(self) -> None:
        test_excel_header_preservation()

    def test_ai_heuristics(self) -> None:
        test_ai_heuristics()


if __name__ == "__main__":
    unittest.main()
