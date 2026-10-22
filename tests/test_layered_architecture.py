import shutil
import tempfile
import unittest
from pathlib import Path

from backend.api.controllers.batch_controller import BatchController
from backend.api.controllers.report_controller import ReportController
from backend.api.controllers.track_controller import TrackController
from backend.api.schemas.dtos import BatchDetectRequest, ExportReportRequest, TrackDetectRequest
from backend.core.container import AppContainer
from backend.core.interfaces.repositories.report_repository_interface import IReportRepository
from backend.core.interfaces.repositories.storage_repository_interface import IStorageRepository
from backend.core.interfaces.repositories.track_repository_interface import ITrackRepository
from backend.core.interfaces.services.audio_engine_interface import IAudioEngine
from backend.core.interfaces.services.batch_service_interface import IBatchService
from backend.core.interfaces.services.classifier_interface import IAIClassifier
from backend.core.interfaces.services.detection_service_interface import IDetectionService
from backend.core.interfaces.services.report_service_interface import IReportService
from backend.core.interfaces.services.resolver_interface import IInputResolver
from backend.core.interfaces.services.signal_analyzer_interface import ISignalAnalyzer

# Backend Core & Interfaces
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.repositories.file_report_repository import FileReportRepository

# Implementations
from backend.repositories.in_memory_track_repository import InMemoryTrackRepository
from backend.repositories.temp_storage_repository import TempStorageRepository
from backend.services.batch_service import BatchService
from backend.services.detection_service import DetectionService
from backend.services.report_service import ReportService


class MockAudioEngine(IAudioEngine):
    def process_track_audio(self, track: TrackItem):
        dummy_wav = Path(tempfile.gettempdir()) / f"{track.id}_test.wav"
        dummy_wav.touch(exist_ok=True)
        return dummy_wav, {"title": track.track_title, "upload_date": "20240101"}


class MockSignalAnalyzer(ISignalAnalyzer):
    def check_heuristics(self, metadata=None):
        return None

    def analyze_audio(self, audio_path: Path):
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            spectral_rolloff_95_hz=16500.0,
            spectral_rolloff_85_hz=12500.0,
            stereo_correlation=0.9,
            harmonic_ratio=0.4,
            fakeprint_score=0.75,
            fakeprint_detected=True,
            compression_source="Opus/AAC (YouTube/Web Streaming Lossy)",
            has_brickwall_cutoff=True,
        )
        return metrics, None


class MockClassifier(IAIClassifier):
    def classify_track(self, **kwargs):
        return DetectionResult(
            track_id=kwargs.get("track_id", "t1"),
            track_title=kwargs.get("track_title", "Mock Title"),
            source_url=kwargs.get("source_url"),
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.92,
            suspected_engine="Suno v3.5",
            reasons=["Dấu vết Deconvolution Fourier ISMIR 2025", "Vocal nhựa"],
            detailed_explanation="Bản ghi thẩm định mẫu từ mock classifier.",
        )


class MockResolver(IInputResolver):
    def detect_input_type(self, query: str) -> InputType:
        if query.startswith("http"):
            return InputType.URL
        return InputType.SONG_NAME

    def resolve_track(self, query: str, artist: str = "", force_rescan: bool = False) -> TrackItem:
        return TrackItem(
            id="mock_id_123",
            raw_query=query,
            input_type=self.detect_input_type(query),
            track_title=query,
            resolved_url=query
            if query.startswith("http")
            else f"https://youtube.com/watch?v={query}",
        )

    def parse_batch_file(self, file_path: Path):
        return [self.resolve_track("Song A"), self.resolve_track("Song B")]

    def search_youtube(self, song_name: str):
        return f"https://youtube.com/watch?v={song_name}", song_name, 180.0


class TestLayeredArchitecture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())

        # Layer 3: Repositories
        self.track_repo: ITrackRepository = InMemoryTrackRepository()
        self.report_repo: IReportRepository = FileReportRepository(output_dir=self.output_dir)
        self.storage_repo: IStorageRepository = TempStorageRepository(
            temp_dir=self.temp_dir, output_dir=self.output_dir
        )

        # Layer 2: Services through Interfaces
        self.audio_engine: IAudioEngine = MockAudioEngine()
        self.signal_analyzer: ISignalAnalyzer = MockSignalAnalyzer()
        self.classifier: IAIClassifier = MockClassifier()
        self.resolver: IInputResolver = MockResolver()
        self.report_service: IReportService = ReportService(report_repo=self.report_repo)
        self.detection_service: IDetectionService = DetectionService(
            audio_engine=self.audio_engine,
            signal_analyzer=self.signal_analyzer,
            classifier=self.classifier,
            resolver=self.resolver,
            track_repo=self.track_repo,
        )
        self.batch_service: IBatchService = BatchService(
            detection_service=self.detection_service,
            report_service=self.report_service,
            track_repo=self.track_repo,
            sleep_interval=0.01,
        )

        # Layer 1: Controllers
        self.track_controller = TrackController(
            detection_service=self.detection_service, resolver=self.resolver
        )
        self.batch_controller = BatchController(
            batch_service=self.batch_service, resolver=self.resolver
        )
        self.report_controller = ReportController(
            report_service=self.report_service, track_repo=self.track_repo
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_interfaces_conformance(self):
        """Kiểm tra tính tuân thủ nghiêm ngặt của các Interface (FAANG Standard)."""
        self.assertIsInstance(self.audio_engine, IAudioEngine)
        self.assertIsInstance(self.signal_analyzer, ISignalAnalyzer)
        self.assertIsInstance(self.classifier, IAIClassifier)
        self.assertIsInstance(self.resolver, IInputResolver)
        self.assertIsInstance(self.detection_service, IDetectionService)
        self.assertIsInstance(self.batch_service, IBatchService)
        self.assertIsInstance(self.report_service, IReportService)
        self.assertIsInstance(self.track_repo, ITrackRepository)
        self.assertIsInstance(self.report_repo, IReportRepository)
        self.assertIsInstance(self.storage_repo, IStorageRepository)

    def test_layer1_track_controller(self):
        """Kiểm tra Layer 1 (Controller) gọi Layer 2 (Service qua Interface) và trả về DTO chuẩn."""
        req = TrackDetectRequest(query="https://youtube.com/watch?v=sample")
        resp = self.track_controller.detect_track(req)

        self.assertTrue(resp.success)
        self.assertIsNotNone(resp.data)
        self.assertEqual(resp.data.track_title, "https://youtube.com/watch?v=sample")
        self.assertTrue(resp.data.is_ai)
        self.assertEqual(resp.data.verdict_type, VerdictType.AI_GENERATED)

        # Kiểm tra dữ liệu được lưu tự động vào Layer 3 (Repository)
        saved = self.track_repo.get_result(resp.data.track_id)
        self.assertIsNotNone(saved)
        self.assertEqual(saved.track_id, resp.data.track_id)

    def test_layer1_batch_controller_and_export(self):
        """Kiểm tra Layer 1 BatchController xử lý hàng loạt và xuất báo cáo qua ReportController."""
        req = BatchDetectRequest(queries=["Song 1", "Song 2", "Song 3"])
        resp = self.batch_controller.process_queries(req)

        self.assertTrue(resp.success)
        self.assertIsNotNone(resp.data)
        self.assertEqual(resp.data.total_tracks, 3)
        self.assertEqual(resp.data.processed_count, 3)
        self.assertEqual(resp.data.ai_count, 3)

        # Kiểm tra xuất báo cáo qua ReportController
        export_req = ExportReportRequest(format="xlsx")
        export_resp = self.report_controller.export_results(export_req)
        self.assertTrue(export_resp.success)
        self.assertTrue(Path(export_resp.data).exists())

    def test_di_container(self):
        """Kiểm tra Dependency Injection Container khởi tạo trơn tru toàn bộ 3 tầng."""
        container = AppContainer(
            custom_classifier=self.classifier, temp_dir=self.temp_dir, output_dir=self.output_dir
        )
        self.assertIsNotNone(container.track_controller)
        self.assertIsNotNone(container.batch_controller)
        self.assertIsNotNone(container.report_controller)
        self.assertIsNotNone(container.detection_service)
        self.assertIsNotNone(container.batch_service)
        self.assertIsNotNone(container.track_repository)

    def test_fastapi_rest_endpoints(self):
        """Kiểm tra các REST API endpoints của FastAPI server."""
        from fastapi.testclient import TestClient

        from backend.api.app import app

        client = TestClient(app)
        # 1. Health check
        res = client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

        # 2. History check
        res_hist = client.get("/api/v1/reports/history")
        self.assertEqual(res_hist.status_code, 200)
        self.assertTrue(res_hist.json()["success"])


if __name__ == "__main__":
    unittest.main()
