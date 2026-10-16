from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from backend.core.models.entities import InputType, TrackItem


class IInputResolver(ABC):
    """Interface định nghĩa phân giải dữ liệu đầu vào (URL, Tên bài, File local, Batch file)."""

    @abstractmethod
    def detect_input_type(self, query: str) -> InputType:
        """Nhận diện loại đầu vào là URL, LOCAL_FILE hay SONG_NAME."""

    @abstractmethod
    def resolve_track(self, query: str, artist: str = "") -> TrackItem:
        """Chuyển đổi chuỗi đầu vào bất kỳ thành một TrackItem hoàn chỉnh."""

    @abstractmethod
    def parse_batch_file(self, file_path: Path) -> list[TrackItem]:
        """Đọc và trích xuất danh sách bài hát từ file .txt, .csv, hoặc .xlsx."""

    @abstractmethod
    def search_youtube(
        self,
        song_name: str,
        target_artist: str = "",
        target_duration: float | None = None,
    ) -> tuple[str | None, str | None, float | None, bool, float]:
        """Tìm kiếm link và metadata bài hát trên YouTube qua yt-dlp.

        Trả về: (url, title, duration, is_verified_match, match_score)
        """

    def get_url_metadata(self, url: str) -> tuple[str | None, float | None]:
        """Trích xuất tiêu đề và thời lượng của một URL qua yt-dlp."""
        return None, None

    def search_zingmp3(self, song_name: str, target_artist: str = "") -> dict[str, Any]:
        """Tìm kiếm link và metadata bài hát trên Zing MP3."""
        return {}

    def search_spotify(self, song_name: str) -> dict[str, Any]:
        """Tìm kiếm link và metadata bài hát trên Spotify."""
        return {}

    def validate_match(
        self,
        target_title: str,
        target_artist: str = "",
        candidate: dict[str, Any] | None = None,
        target_duration: float | None = None,
    ) -> tuple[bool, float]:
        """Xác thực độ trùng khớp thông minh giữa bài hát mục tiêu và ứng viên tìm kiếm (YouTube/Web)."""
        return False, 0.0
