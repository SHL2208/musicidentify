from abc import ABC, abstractmethod

from backend.core.models.entities import DetectionResult, TrackItem


class IDetectionService(ABC):
    """Interface định nghĩa nghiệp vụ điều phối quy trình phân tích nhận diện đơn bài."""

    @abstractmethod
    def detect_single(
        self,
        track: TrackItem,
        model_override: str | None = None,
        force_rescan: bool = False,
        language: str = "vi",
    ) -> DetectionResult:
        """Thực hiện toàn bộ quy trình nhận diện bài hát từ TrackItem đã chuẩn hóa."""
        pass

    @abstractmethod
    def detect_query(
        self,
        query: str,
        model_override: str | None = None,
        force_rescan: bool = False,
        language: str = "vi",
    ) -> DetectionResult:
        """Phân giải chuỗi query tự do và thực hiện nhận diện bài hát."""
        pass
