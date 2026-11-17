from pathlib import Path

from ..api.controllers.batch_controller import BatchController
from ..api.controllers.report_controller import ReportController
from ..api.controllers.track_controller import TrackController

# Concrete Implementations
from ..repositories.database_catalog_repository import DatabaseCatalogRepository
from ..repositories.database_track_repository import DatabaseTrackRepository
from ..repositories.file_report_repository import FileReportRepository
from ..repositories.in_memory_track_repository import InMemoryTrackRepository
from ..repositories.temp_storage_repository import TempStorageRepository
from ..services.audio_engine_service import AudioEngineService
from ..services.batch_service import BatchService
from ..services.detection_service import DetectionService
from ..services.fakeprint_detector_service import FakeprintDetectorService
from ..services.gemini_classifier_service import GeminiClassifierService
from ..services.local_llm_classifier_service import LocalLLMClassifierService
from ..services.routing_classifier_service import RoutingClassifierService
from ..services.report_service import ReportService
from ..services.resolver_service import InputResolverService
from ..services.signal_analyzer_service import SignalAnalyzerService
from ..services.spotify_service import SpotifyService

# Core Configurations
from .config import (
    BATCH_SLEEP_INTERVAL,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEYS,
    GEMINI_FALLBACK_MODELS,
    LOCAL_DEFAULT_MODEL,
    LOCAL_FALLBACK_MODELS,
    LOCAL_MODEL_API_KEY,
    LOCAL_MODEL_BASE_URL,
    OUTPUT_DIR,
    TEMP_DIR,
    USE_LOCAL_MODEL,
    get_ffmpeg_binary,
)
from .database import is_database_available
from .interfaces.repositories.catalog_repository_interface import ICatalogRepository
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
from .interfaces.services.spotify_service_interface import ISpotifyService


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
        self.catalog_repository: ICatalogRepository = DatabaseCatalogRepository()
        if is_database_available():
            self.track_repository: ITrackRepository = DatabaseTrackRepository()
        else:
            self.track_repository: ITrackRepository = InMemoryTrackRepository()

        # 2. LAYER 2: SERVICES (Tất cả được gán kiểu là Interface)
        self.audio_engine: IAudioEngine = AudioEngineService(
            storage_repo=self.storage_repository, ffmpeg_path=get_ffmpeg_binary()
        )
        self.fakeprint_detector: IFakeprintDetector = FakeprintDetectorService()
        self.signal_analyzer: ISignalAnalyzer = SignalAnalyzerService(
            fakeprint_detector=self.fakeprint_detector,
            storage_repo=self.storage_repository,
            catalog_repo=self.catalog_repository,
        )
        if custom_classifier:
            self.classifier: IAIClassifier = custom_classifier
        else:
            cloud_classifier = GeminiClassifierService(
                api_keys=api_keys or GEMINI_API_KEYS,
                model_name=default_model or DEFAULT_GEMINI_MODEL,
                fallback_models=fallback_models or GEMINI_FALLBACK_MODELS,
            )
            local_classifier = None
            if LOCAL_MODEL_BASE_URL and (LOCAL_MODEL_API_KEY or USE_LOCAL_MODEL):
                local_classifier = LocalLLMClassifierService(
                    base_url=LOCAL_MODEL_BASE_URL,
                    api_key=LOCAL_MODEL_API_KEY,
                    model_name=default_model if (default_model and "/" in default_model) else LOCAL_DEFAULT_MODEL,
                    fallback_models=LOCAL_FALLBACK_MODELS,
                )

            if local_classifier:
                self.classifier: IAIClassifier = RoutingClassifierService(
                    local_classifier=local_classifier,
                    cloud_classifier=cloud_classifier,
                    use_local_default=USE_LOCAL_MODEL,
                )
            else:
                self.classifier: IAIClassifier = cloud_classifier
        self.spotify_service: ISpotifyService = SpotifyService()
        self.resolver: IInputResolver = InputResolverService(spotify_service=self.spotify_service)
        self.report_service: IReportService = ReportService(report_repo=self.report_repository)
        self.detection_service: IDetectionService = DetectionService(
            audio_engine=self.audio_engine,
            signal_analyzer=self.signal_analyzer,
            classifier=self.classifier,
            resolver=self.resolver,
            track_repo=self.track_repository,
            spotify_service=self.spotify_service,
        )
        self.batch_service: IBatchService = BatchService(
            detection_service=self.detection_service,
            report_service=self.report_service,
            track_repo=self.track_repository,
            sleep_interval=BATCH_SLEEP_INTERVAL,
        )

        # 3. LAYER 1: CONTROLLERS
        self.track_controller: TrackController = TrackController(
            detection_service=self.detection_service,
            resolver=self.resolver,
            track_repo=self.track_repository,
            spotify_service=self.spotify_service,
        )
        self.batch_controller: BatchController = BatchController(
            batch_service=self.batch_service, resolver=self.resolver
        )
        self.report_controller: ReportController = ReportController(
            report_service=self.report_service, track_repo=self.track_repository
        )
        from ..services.batch_job_manager import get_batch_job_manager

        self.batch_job_manager = get_batch_job_manager(
            batch_service=self.batch_service,
            resolver=self.resolver,
            report_service=self.report_service,
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
