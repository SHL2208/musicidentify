import logging
from pathlib import Path
from typing import Any

from ..core.config import (
    DEFAULT_GEMINI_MODEL,
    LOCAL_DEFAULT_MODEL,
    LOCAL_MODEL_API_KEY,
    LOCAL_MODEL_BASE_URL,
    USE_LOCAL_MODEL,
)
from ..core.interfaces.services.classifier_interface import IAIClassifier
from ..core.models.entities import AcousticMetrics, DetectionResult
from .gemini_classifier_service import GeminiClassifierService
from .local_llm_classifier_service import LocalLLMClassifierService

logger = logging.getLogger(__name__)


class RoutingClassifierService(IAIClassifier):
    """
    Bộ định tuyến giám định âm nhạc AI:
    - Nếu model bắt đầu bằng 'antigravity/' hoặc chứa '/' hoặc USE_LOCAL_MODEL=True: Chuyển tiếp tới Local Model Gateway (localhost).
    - Nếu model là Google Cloud Gemini thuần: Chuyển tiếp tới Google AI Studio Cloud Gemini.
    - Hỗ trợ failover: nếu một bên gặp sự cố kết nối, tự động thử bên còn lại (nếu có key/cấu hình).
    """

    def __init__(
        self,
        local_classifier: LocalLLMClassifierService | None = None,
        cloud_classifier: GeminiClassifierService | None = None,
        use_local_default: bool = False,
    ):
        self.local_classifier = local_classifier
        self.cloud_classifier = cloud_classifier
        self.use_local_default = use_local_default

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
        language: str = "vi",
    ) -> DetectionResult:
        m_str = str(model_override or "").strip()
        is_local_request = (
            ("/" in m_str)
            or ("local" in m_str.lower())
            or (self.use_local_default and not m_str)
        )

        # 1. Yêu cầu Local Model
        if is_local_request and self.local_classifier:
            try:
                return self.local_classifier.classify_track(
                    track_id=track_id,
                    track_title=track_title,
                    source_url=source_url,
                    audio_path=audio_path,
                    spectrogram_path=spectrogram_path,
                    metrics=metrics,
                    model_override=model_override,
                    metadata=metadata,
                    input_type=input_type,
                    raw_query=raw_query,
                    language=language,
                )
            except Exception as local_err:
                logger.warning(
                    f"[RoutingClassifier] Local Model ({m_str or LOCAL_DEFAULT_MODEL}) gặp sự cố: {local_err}. "
                    f"Thử chuyển tiếp sang Cloud Gemini..."
                )
                if self.cloud_classifier:
                    return self.cloud_classifier.classify_track(
                        track_id=track_id,
                        track_title=track_title,
                        source_url=source_url,
                        audio_path=audio_path,
                        spectrogram_path=spectrogram_path,
                        metrics=metrics,
                        model_override=None,
                        metadata=metadata,
                        input_type=input_type,
                        raw_query=raw_query,
                        language=language,
                    )
                raise

        # 2. Yêu cầu Cloud Gemini
        if self.cloud_classifier:
            try:
                return self.cloud_classifier.classify_track(
                    track_id=track_id,
                    track_title=track_title,
                    source_url=source_url,
                    audio_path=audio_path,
                    spectrogram_path=spectrogram_path,
                    metrics=metrics,
                    model_override=model_override,
                    metadata=metadata,
                    input_type=input_type,
                    raw_query=raw_query,
                    language=language,
                )
            except Exception as cloud_err:
                logger.warning(
                    f"[RoutingClassifier] Cloud Gemini gặp lỗi: {cloud_err}. "
                    f"Thử chuyển tiếp sang Local Model..."
                )
                if self.local_classifier:
                    return self.local_classifier.classify_track(
                        track_id=track_id,
                        track_title=track_title,
                        source_url=source_url,
                        audio_path=audio_path,
                        spectrogram_path=spectrogram_path,
                        metrics=metrics,
                        model_override=None,
                        metadata=metadata,
                        input_type=input_type,
                        raw_query=raw_query,
                        language=language,
                    )
                raise

        # 3. Nếu chỉ có Local classifier
        if self.local_classifier:
            return self.local_classifier.classify_track(
                track_id=track_id,
                track_title=track_title,
                source_url=source_url,
                audio_path=audio_path,
                spectrogram_path=spectrogram_path,
                metrics=metrics,
                model_override=model_override,
                metadata=metadata,
                input_type=input_type,
                raw_query=raw_query,
                language=language,
            )

        raise RuntimeError("Không có dịch vụ phân loại AI nào (cả Cloud và Local) khả dụng.")

