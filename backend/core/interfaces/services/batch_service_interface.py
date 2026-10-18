from abc import ABC, abstractmethod
from collections.abc import Callable

from backend.core.models.entities import BatchReport, DetectionResult, TrackItem


class IBatchService(ABC):
    """Interface định nghĩa nghiệp vụ xử lý hàng loạt có điều tiết tốc độ (Rate Limiting)."""

    @abstractmethod
    def process_batch(
        self,
        tracks: list[TrackItem],
        model_override: str | None = None,
        scan_mode: str = "smart",
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
    ) -> BatchReport:
        """
        Xử lý danh sách bài hát có điều tiết tốc độ (Rate Limiting)
        và cập nhật tiến độ mà không bao giờ làm rơi rụng bài hát.
        scan_mode: 'smart' (bỏ qua bài đã có), 'force_all' (quét lại toàn bộ), 'retry_errors' (chỉ quét bài lỗi/chưa có).
        """
        pass
