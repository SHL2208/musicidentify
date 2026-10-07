import logging
from typing import Optional

from ..core.models.entities import TrackItem, DetectionResult, VerdictType
from ..core.interfaces.services.detection_service_interface import IDetectionService
from ..core.interfaces.services.audio_engine_interface import IAudioEngine
from ..core.interfaces.services.signal_analyzer_interface import ISignalAnalyzer
from ..core.interfaces.services.classifier_interface import IAIClassifier
from ..core.interfaces.services.resolver_interface import IInputResolver
from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.exceptions import MusicIdentifyException

logger = logging.getLogger(__name__)

class DetectionService(IDetectionService):
    """
    Hiện thực nghiệp vụ điều phối nhận diện âm nhạc.
    Tuân thủ interface IDetectionService và kết nối các dịch vụ hoàn toàn thông qua Interface.
    """

    def __init__(
        self,
        audio_engine: IAudioEngine,
        signal_analyzer: ISignalAnalyzer,
        classifier: IAIClassifier,
        resolver: IInputResolver,
        track_repo: ITrackRepository
    ):
        self.audio_engine = audio_engine
        self.signal_analyzer = signal_analyzer
        self.classifier = classifier
        self.resolver = resolver
        self.track_repo = track_repo

    def detect_single(
        self,
        track: TrackItem,
        model_override: Optional[str] = None
    ) -> DetectionResult:
        """Thực hiện toàn bộ quy trình nhận diện 1 bài hát từ A đến Z."""
        self.track_repo.save_track(track)

        # 1. Tải và cắt phân đoạn âm thanh qua IAudioEngine
        audio_path, metadata = self.audio_engine.process_track_audio(track)

        # 2. Kiểm tra bộ lọc Heuristics qua ISignalAnalyzer
        heuristics_result = self.signal_analyzer.check_heuristics(metadata)
        if heuristics_result:
            is_ai = heuristics_result["is_ai"]
            v_type = heuristics_result.get(
                "verdict_type",
                VerdictType.CONFIRMED_HUMAN if not is_ai else VerdictType.AI_GENERATED
            )
            if v_type == VerdictType.CONFIRMED_HUMAN:
                verdict_label = "CONFIRMED HUMAN (Người làm)"
            elif v_type == VerdictType.PROCESSED_HUMAN:
                verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
            elif v_type == VerdictType.HYBRID_AI:
                verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
            else:
                verdict_label = "AI-GENERATED (Nhạc AI tạo)"

            # Tạo ảnh phổ và metrics
            try:
                metrics, spec_image_path = self.signal_analyzer.analyze_audio(audio_path)
            except Exception as exc:
                logger.warning(
                    "Không thể tạo ảnh phổ/metrics cho bài '%s': %s",
                    track.track_title or track.raw_query,
                    exc,
                )
                metrics = None
                spec_image_path = None

            result = DetectionResult(
                track_id=track.id,
                track_title=track.track_title or track.raw_query,
                source_url=track.resolved_url,
                is_ai=is_ai,
                verdict_type=v_type,
                verdict_label=verdict_label,
                confidence_score=heuristics_result["confidence"],
                suspected_engine=heuristics_result["engine"],
                reasons=[heuristics_result["reason"]],
                detailed_explanation=f"Xác thực qua siêu dữ liệu nguồn: {heuristics_result['reason']}",
                acoustic_metrics=metrics,
                spectrogram_image_path=str(spec_image_path) if spec_image_path else None
            )
            self.track_repo.save_result(result)
            return result

        # 3. Phân tích âm học và tạo ảnh Forensic 2 tầng qua ISignalAnalyzer
        metrics, spec_image_path = self.signal_analyzer.analyze_audio(audio_path)

        # 4. Gửi lên IAIClassifier để phân loại
        result = self.classifier.classify_track(
            track_id=track.id,
            track_title=track.track_title or track.raw_query,
            source_url=track.resolved_url,
            audio_path=audio_path,
            spectrogram_path=spec_image_path,
            metrics=metrics,
            model_override=model_override,
            metadata=metadata
        )

        self.track_repo.save_result(result)
        return result

    def detect_query(
        self,
        query: str,
        model_override: Optional[str] = None
    ) -> DetectionResult:
        """Phân giải chuỗi query tự do qua IInputResolver và thực hiện nhận diện bài hát."""
        track = self.resolver.resolve_track(query)
        return self.detect_single(track, model_override=model_override)
