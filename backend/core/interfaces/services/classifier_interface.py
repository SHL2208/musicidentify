from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from backend.core.models.entities import DetectionResult, AcousticMetrics

class IAIClassifier(ABC):
    """Interface định nghĩa phân loại thẩm âm sử dụng mô hình trí tuệ nhân tạo (Gemini)."""

    @abstractmethod
    def classify_track(
        self,
        track_id: str,
        track_title: str,
        source_url: Optional[str] = None,
        audio_path: Optional[Path] = None,
        spectrogram_path: Optional[Path] = None,
        metrics: Optional[AcousticMetrics] = None,
        model_override: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Thực hiện giám định đa tầng bằng AI kết hợp tín hiệu âm thanh và biểu đồ phổ tần.
        Có cơ chế tự động chuyển đổi khóa dự phòng (Failover 1 > 2 > 3).
        """
        pass
