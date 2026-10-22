import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.core.interfaces.services.classifier_interface import IAIClassifier
from backend.core.interfaces.services.detection_service_interface import IDetectionService
from backend.core.models.entities import (
    AcousticMetrics,
    BatchReport,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.repositories.file_report_repository import FileReportRepository
from backend.repositories.in_memory_track_repository import InMemoryTrackRepository
from backend.services.batch_service import BatchService
from backend.services.report_service import ReportService


class MockUncertainClassifier(IAIClassifier):
    def classify_track(
        self,
        track_id: str,
        track_title: str,
        source_url: str | None = None,
        audio_path: Path | None = None,
        spectrogram_path: Path | None = None,
        metrics: AcousticMetrics | None = None,
        model_override: str | None = None,
        metadata: dict[str, Any] | None = None,
        input_type: Any | None = None,
        raw_query: str | None = None,
    ) -> DetectionResult:
        return DetectionResult(
            track_id=track_id,
            track_title=track_title,
            source_url=source_url,
            is_ai=False,
            verdict_type=VerdictType.UNCERTAIN,
            verdict_label="UNCERTAIN (Chưa thể kết luận)",
            confidence_score=0.45,
            suspected_engine="None",
            reasons=["Dữ liệu âm thanh bị nhiễu nặng, không thể kết luận chắc chắn."],
            detailed_explanation="Không đủ bằng chứng kỹ thuật để phân loại cho AI hay Người làm.",
        )


class MockDetectionService(IDetectionService):
    def __init__(self, classifier: IAIClassifier):
        self.classifier = classifier

    def detect_single(self, track: TrackItem, model_override=None, **kwargs):
        return self.classifier.classify_track(
            track_id=track.id,
            track_title=track.track_title or track.raw_query,
            source_url=track.resolved_url,
            input_type=track.input_type,
            raw_query=track.raw_query,
        )

    def detect_query(self, query: str, model_override=None, **kwargs):
        track = TrackItem(id="t1", raw_query=query, input_type=InputType.SONG_NAME)
        return self.detect_single(track, model_override=model_override, **kwargs)


class TestUncertainVerdict(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.track_repo = InMemoryTrackRepository()
        self.report_repo = FileReportRepository(output_dir=self.temp_dir)
        self.report_service = ReportService(report_repo=self.report_repo)
        self.classifier = MockUncertainClassifier()
        self.detection_service = MockDetectionService(self.classifier)
        self.batch_service = BatchService(
            detection_service=self.detection_service,
            report_service=self.report_service,
            track_repo=self.track_repo,
            sleep_interval=0.0,
        )

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verdict_type_enum(self):
        """Kiểm tra nhãn UNCERTAIN tồn tại trong VerdictType."""
        self.assertEqual(VerdictType.UNCERTAIN.value, "UNCERTAIN")
        self.assertIn("UNCERTAIN", [v.value for v in VerdictType])

    def test_batch_report_uncertain_counting(self):
        """Kiểm tra BatchService đếm chính xác số lượng bài có kết luận UNCERTAIN."""
        tracks = [
            TrackItem(id="u1", raw_query="Track 1", input_type=InputType.SONG_NAME),
            TrackItem(id="u2", raw_query="Track 2", input_type=InputType.SONG_NAME),
        ]
        report = self.batch_service.process_batch(tracks)

        self.assertIsInstance(report, BatchReport)
        self.assertEqual(report.total_tracks, 2)
        self.assertEqual(report.processed_count, 2)
        self.assertEqual(report.uncertain_count, 2)
        self.assertEqual(report.ai_count, 0)
        self.assertEqual(report.human_count, 0)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.results[0].verdict_type, VerdictType.UNCERTAIN)
        self.assertEqual(report.results[0].verdict_label, "UNCERTAIN (Chưa thể kết luận)")


if __name__ == "__main__":
    unittest.main()
