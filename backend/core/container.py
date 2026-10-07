from typing import Optional, List
from pathlib import Path

# Core Configurations
from .config import (
    TEMP_DIR,
    OUTPUT_DIR,
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
    DEFAULT_GEMINI_MODEL,
    BATCH_SLEEP_INTERVAL,
    get_ffmpeg_binary
)

# Interfaces
from .interfaces.services.audio_engine_interface import IAudioEngine
from .interfaces.services.fakeprint_detector_interface import IFakeprintDetector
from .interfaces.services.signal_analyzer_interface import ISignalAnalyzer
from .interfaces.services.classifier_interface import IAIClassifier
from .interfaces.services.resolver_interface import IInputResolver
from .interfaces.services.detection_service_interface import IDetectionService
from .interfaces.services.batch_service_interface import IBatchService
from .interfaces.services.report_service_interface import IReportService
from .interfaces.repositories.track_repository_interface import ITrackRepository
from .interfaces.repositories.report_repository_interface import IReportRepository
from .interfaces.repositories.storage_repository_interface import IStorageRepository

# Concrete Implementations
from ..repositories.in_memory_track_repository import InMemoryTrackRepository
from ..repositories.file_report_repository import FileReportRepository
from ..repositories.temp_storage_repository import TempStorageRepository
from ..services.audio_engine_service import AudioEngineService
from ..services.fakeprint_detector_service import FakeprintDetectorService
from ..services.signal_analyzer_service import SignalAnalyzerService
from ..services.gemini_classifier_service import GeminiClassifierService
from ..services.resolver_service import InputResolverService
from ..services.report_service import ReportService
from ..services.detection_service import DetectionService
from ..services.batch_service import BatchService
from ..api.controllers.track_controller import TrackController
from ..api.controllers.batch_controller import BatchController
from ..api.controllers.report_controller import ReportController

class AppContainer:
    """
    Dependency Injection Container (IoC) chuẩn quốc tế (FAANG Pattern).
    Khởi tạo và kết nối toàn bộ các Repository, Service qua Interface, và Controllers.
    Đảm bảo nguyên lý Dependency Inversion Principle (DIP).
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        default_model: Optional[str] = None,
        temp_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        custom_classifier: Optional[IAIClassifier] = None
    ):
        # 1. LAYER 3: REPOSITORIES
        self.storage_repository: IStorageRepository = TempStorageRepository(
            temp_dir=temp_dir or TEMP_DIR,
            output_dir=output_dir or OUTPUT_DIR
        )
        self.report_repository: IReportRepository = FileReportRepository(
            output_dir=output_dir or OUTPUT_DIR
        )
        self.track_repository: ITrackRepository = InMemoryTrackRepository()

        # 2. LAYER 2: SERVICES (Tất cả được gán kiểu là Interface)
        self.audio_engine: IAudioEngine = AudioEngineService(
            storage_repo=self.storage_repository,
            ffmpeg_path=get_ffmpeg_binary()
        )
        self.fakeprint_detector: IFakeprintDetector = FakeprintDetectorService()
        self.signal_analyzer: ISignalAnalyzer = SignalAnalyzerService(
            fakeprint_detector=self.fakeprint_detector,
            storage_repo=self.storage_repository
        )
        self.classifier: IAIClassifier = custom_classifier or GeminiClassifierService(
            api_keys=api_keys or GEMINI_API_KEYS,
            model_name=default_model or DEFAULT_GEMINI_MODEL
        )
        self.resolver: IInputResolver = InputResolverService()
        self.report_service: IReportService = ReportService(
            report_repo=self.report_repository
        )
        self.detection_service: IDetectionService = DetectionService(
            audio_engine=self.audio_engine,
            signal_analyzer=self.signal_analyzer,
            classifier=self.classifier,
            resolver=self.resolver,
            track_repo=self.track_repository
        )
        self.batch_service: IBatchService = BatchService(
            detection_service=self.detection_service,
            report_service=self.report_service,
            track_repo=self.track_repository,
            sleep_interval=BATCH_SLEEP_INTERVAL
        )

        # 3. LAYER 1: CONTROLLERS
        self.track_controller: TrackController = TrackController(
            detection_service=self.detection_service,
            resolver=self.resolver
        )
        self.batch_controller: BatchController = BatchController(
            batch_service=self.batch_service,
            resolver=self.resolver
        )
        self.report_controller: ReportController = ReportController(
            report_service=self.report_service,
            track_repo=self.track_repository
        )

# Global default container instance
_container_instance: Optional[AppContainer] = None

def get_container(
    api_keys: Optional[List[str]] = None,
    default_model: Optional[str] = None,
    custom_classifier: Optional[IAIClassifier] = None,
    force_new: bool = False
) -> AppContainer:
    """Singleton getter / factory cho Dependency Injection Container."""
    global _container_instance
    if _container_instance is None or force_new or custom_classifier or api_keys or default_model:
        _container_instance = AppContainer(
            api_keys=api_keys,
            default_model=default_model,
            custom_classifier=custom_classifier
        )
    return _container_instance
