from pathlib import Path

from ..api.controllers.batch_controller import BatchController
from ..api.controllers.report_controller import ReportController
from ..api.controllers.track_controller import TrackController
from ..repositories.file_report_repository import FileReportRepository

# Concrete Implementations
from ..repositories.in_memory_track_repository import InMemoryTrackRepository
from ..repositories.temp_storage_repository import TempStorageRepository
from ..services.audio_engine_service import AudioEngineService
from ..services.batch_service import BatchService
from ..services.detection_service import DetectionService
from ..services.fakeprint_detector_service import FakeprintDetectorService
from ..services.gemini_classifier_service import GeminiClassifierService
from ..services.report_service import ReportService
from ..services.resolver_service import InputResolverService
from ..services.signal_analyzer_service import SignalAnalyzerService

# Core Configurations
from .config import (
    BATCH_SLEEP_INTERVAL,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEYS,
    GEMINI_FALLBACK_MODELS,
    OUTPUT_DIR,
    TEMP_DIR,
    get_ffmpeg_binary,
)
from .interfaces.repositories.report_repository_interface import IReportRepository
from .interfaces.repositories.storage_repository_interface import IStorageRepository
from .interfaces.repositories.track_repository_interface import ITrackRepository

# Interfaces
from .interfaces.services.audio_engine_interface import IAudioEngine
from .interfaces.services.batch_service_interface import IBatchService
from .interfaces.services.classifier_interface import IAIClassifier
from .interfaces.services.detection_service_interface import IDetectionService
from .interfaces.services.fakeprint_detector_interface import IFakeprintDetector
from .interfaces.services.report_service_interface import IReportService
from .interfaces.services.resolver_interface import IInputResolver
from .interfaces.services.signal_analyzer_interface import ISignalAnalyzer


class AppContainer:
    """
    Dependency Injection Container (IoC) chuẩn quốc tế (FAANG Pattern).
    Khởi tạo và kết nối toàn bộ các Repository, Service qua Interface, và Controllers.
    Đảm bảo nguyên lý Dependency Inversion Principle (DIP).
    """

    def __init__(
        self,
        api_keys: list[str] | None = None,
        default_model: str | None = None,
        fallback_models: list[str] | None = None,
        temp_dir: Path | None = None,
        output_dir: Path | None = None,
        custom_classifier: IAIClassifier | None = None,
    ):
        # 1. LAYER 3: REPOSITORIES
        self.storage_repository: IStorageRepository = TempStorageRepository(
            temp_dir=temp_dir or TEMP_DIR, output_dir=output_dir or OUTPUT_DIR
        )
        self.report_repository: IReportRepository = FileReportRepository(
            output_dir=output_dir or OUTPUT_DIR
        )
        self.track_repository: ITrackRepository = InMemoryTrackRepository()

        # 2. LAYER 2: SERVICES (Tất cả được gán kiểu là Interface)
        self.audio_engine: IAudioEngine = AudioEngineService(
            storage_repo=self.storage_repository, ffmpeg_path=get_ffmpeg_binary()
        )
        self.fakeprint_detector: IFakeprintDetector = FakeprintDetectorService()
        self.signal_analyzer: ISignalAnalyzer = SignalAnalyzerService(
            fakeprint_detector=self.fakeprint_detector, storage_repo=self.storage_repository
        )
        self.classifier: IAIClassifier = custom_classifier or GeminiClassifierService(
            api_keys=api_keys or GEMINI_API_KEYS,
            model_name=default_model or DEFAULT_GEMINI_MODEL,
            fallback_models=fallback_models or GEMINI_FALLBACK_MODELS,
        )
        self.resolver: IInputResolver = InputResolverService()
        self.report_service: IReportService = ReportService(report_repo=self.report_repository)
        self.detection_service: IDetectionService = DetectionService(
            audio_engine=self.audio_engine,
            signal_analyzer=self.signal_analyzer,
            classifier=self.classifier,
            resolver=self.resolver,
            track_repo=self.track_repository,
        )
        self.batch_service: IBatchService = BatchService(
            detection_service=self.detection_service,
            report_service=self.report_service,
            track_repo=self.track_repository,
            sleep_interval=BATCH_SLEEP_INTERVAL,
        )

        # 3. LAYER 1: CONTROLLERS
        self.track_controller: TrackController = TrackController(
            detection_service=self.detection_service, resolver=self.resolver
        )
        self.batch_controller: BatchController = BatchController(
            batch_service=self.batch_service, resolver=self.resolver
        )
        self.report_controller: ReportController = ReportController(
            report_service=self.report_service, track_repo=self.track_repository
        )


# Global default container instance
_container_instance: AppContainer | None = None


def get_container(
    api_keys: list[str] | None = None,
    default_model: str | None = None,
    fallback_models: list[str] | None = None,
    custom_classifier: IAIClassifier | None = None,
    force_new: bool = False,
) -> AppContainer:
    """Singleton getter / factory cho Dependency Injection Container."""
    global _container_instance
    if (
        _container_instance is None
        or force_new
        or custom_classifier
        or api_keys
        or default_model
        or fallback_models
    ):
        _container_instance = AppContainer(
            api_keys=api_keys,
            default_model=default_model,
            fallback_models=fallback_models,
            custom_classifier=custom_classifier,
        )
    return _container_instance
