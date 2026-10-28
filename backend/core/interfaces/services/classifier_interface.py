from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from backend.core.models.entities import AcousticMetrics, DetectionResult


class IAIClassifier(ABC):
    """Interface định nghĩa phân loại thẩm âm sử dụng mô hình trí tuệ nhân tạo (Gemini)."""

    @abstractmethod
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
        """
        Thực hiện giám định đa tầng bằng AI kết hợp tín hiệu âm thanh và biểu đồ phổ tần.
        Có cơ chế tự động chuyển đổi khóa dự phòng (Failover 1 > 2 > 3).
        """
        pass
