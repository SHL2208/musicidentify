from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from backend.core.models.entities import TrackItem, DetectionResult, BatchReport

class IBatchService(ABC):
    """Interface định nghĩa nghiệp vụ xử lý hàng loạt có điều tiết tốc độ (Rate Limiting)."""

    @abstractmethod
    def process_batch(
        self,
        tracks: List[TrackItem],
        model_override: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, TrackItem, Optional[DetectionResult], Optional[str]], None]] = None
    ) -> BatchReport:
        """
        Xử lý danh sách bài hát có điều tiết tốc độ (Rate Limiting)
        và cập nhật tiến độ mà không bao giờ làm rơi rụng bài hát.
        """
        pass
