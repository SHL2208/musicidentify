from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
from backend.core.models.entities import TrackItem, InputType

class IInputResolver(ABC):
    """Interface định nghĩa phân giải dữ liệu đầu vào (URL, Tên bài, File local, Batch file)."""

    @abstractmethod
    def detect_input_type(self, query: str) -> InputType:
        """Nhận diện loại đầu vào là URL, LOCAL_FILE hay SONG_NAME."""
        pass

    @abstractmethod
    def resolve_track(self, query: str) -> TrackItem:
        """Chuyển đổi chuỗi đầu vào bất kỳ thành một TrackItem hoàn chỉnh."""
        pass

    @abstractmethod
    def parse_batch_file(self, file_path: Path) -> List[TrackItem]:
        """Đọc và trích xuất danh sách bài hát từ file .txt, .csv, hoặc .xlsx."""
        pass

    @abstractmethod
    def search_youtube(self, song_name: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Tìm kiếm link và metadata bài hát trên YouTube qua yt-dlp."""
        pass
