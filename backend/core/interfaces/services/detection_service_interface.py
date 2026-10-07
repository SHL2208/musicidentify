from abc import ABC, abstractmethod
from typing import Optional
from backend.core.models.entities import TrackItem, DetectionResult

class IDetectionService(ABC):
    """Interface định nghĩa nghiệp vụ điều phối quy trình phân tích nhận diện đơn bài."""

    @abstractmethod
    def detect_single(
        self,
        track: TrackItem,
        model_override: Optional[str] = None
    ) -> DetectionResult:
        """Thực hiện toàn bộ quy trình nhận diện bài hát từ TrackItem đã chuẩn hóa."""
        pass

    @abstractmethod
    def detect_query(
        self,
        query: str,
        model_override: Optional[str] = None
    ) -> DetectionResult:
        """Phân giải chuỗi query tự do và thực hiện nhận diện bài hát."""
        pass
