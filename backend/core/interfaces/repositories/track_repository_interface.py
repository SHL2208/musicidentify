from abc import ABC, abstractmethod

from backend.core.models.entities import DetectionResult, TrackItem


class ITrackRepository(ABC):
    """Interface định nghĩa hợp đồng lưu trữ và truy xuất các bài hát và kết quả thẩm định."""

    @abstractmethod
    def save_track(self, track: TrackItem) -> TrackItem:
        """Lưu hoặc cập nhật thông tin bài hát."""
        pass

    @abstractmethod
    def get_track(self, track_id: str) -> TrackItem | None:
        """Lấy bài hát theo ID."""
        pass

    @abstractmethod
    def save_result(self, result: DetectionResult) -> DetectionResult:
        """Lưu hoặc cập nhật kết quả thẩm định bài hát."""
        pass

    @abstractmethod
    def get_result(self, track_id: str) -> DetectionResult | None:
        """Lấy kết quả thẩm định theo ID bài hát."""
        pass

    @abstractmethod
    def list_results(self) -> list[DetectionResult]:
        """Lấy toàn bộ kết quả thẩm định đã ghi nhận."""
        pass

    @abstractmethod
    def find_result_by_query_or_url(
        self, query: str, resolved_url: str | None = None, track_id: str | None = None
    ) -> DetectionResult | None:
        """Tìm kiếm kết quả thẩm định đã có theo câu truy vấn, URL nguồn hoặc track_id."""
        pass

    @abstractmethod
    def correct_verdict(
        self,
        track_id: str,
        is_ai: bool,
        verdict_type: str,
        verdict_label: str,
        note: str | None = None,
    ) -> DetectionResult | None:
        """Đính chính phán quyết của bài hát theo can thiệp của con người."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Xóa trắng dữ liệu phiên làm việc."""
        pass
