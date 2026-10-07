from abc import ABC, abstractmethod
from typing import List, Optional
from backend.core.models.entities import TrackItem, DetectionResult

class ITrackRepository(ABC):
    """Interface định nghĩa hợp đồng lưu trữ và truy xuất các bài hát và kết quả thẩm định."""

    @abstractmethod
    def save_track(self, track: TrackItem) -> TrackItem:
        """Lưu hoặc cập nhật thông tin bài hát."""
        pass

    @abstractmethod
    def get_track(self, track_id: str) -> Optional[TrackItem]:
        """Lấy bài hát theo ID."""
        pass

    @abstractmethod
    def save_result(self, result: DetectionResult) -> DetectionResult:
        """Lưu hoặc cập nhật kết quả thẩm định bài hát."""
        pass

    @abstractmethod
    def get_result(self, track_id: str) -> Optional[DetectionResult]:
        """Lấy kết quả thẩm định theo ID bài hát."""
        pass

    @abstractmethod
    def list_results(self) -> List[DetectionResult]:
        """Lấy toàn bộ kết quả thẩm định đã ghi nhận."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Xóa trắng dữ liệu phiên làm việc."""
        pass
