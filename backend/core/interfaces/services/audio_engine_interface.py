from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from backend.core.models.entities import TrackItem


class IAudioEngine(ABC):
    """Interface định nghĩa hợp đồng trích xuất và xử lý âm thanh bài hát."""

    @abstractmethod
    def process_track_audio(self, track: TrackItem) -> tuple[Path, dict[str, Any] | None]:
        """
        Tải và trích xuất mẫu âm thanh đa phân đoạn (Multi-Segment) đại diện cho toàn bài hát.
        Trả về: (đường dẫn file .wav đa phân đoạn, metadata chi tiết)
        """
        pass
