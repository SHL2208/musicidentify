from abc import ABC, abstractmethod
from typing import Any


class ISpotifyService(ABC):
    """Giao diện hợp đồng (Interface Contract) cho dịch vụ kết nối và tìm kiếm Spotify."""

    @abstractmethod
    def search_tracks(
        self, query: str, artist: str = "", limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm danh sách bài hát trên Spotify theo từ khóa (tên bài hát, ca sĩ).
        Trả về danh sách dict chuẩn hóa chứa thông tin track, URL, album cover, embed_url, ISRC,...
        """
        pass

    @abstractmethod
    def get_track_by_url(self, url: str) -> dict[str, Any] | None:
        """
        Lấy thông tin chi tiết bài hát Spotify từ đường link (https://open.spotify.com/track/...)
        hoặc URI (spotify:track:...).
        """
        pass

    @abstractmethod
    def find_best_match(
        self, title: str, artist: str = "", limit: int = 5
    ) -> dict[str, Any] | None:
        """
        Tìm kiếm và chọn ra bài hát khớp nhất (Best Match) dựa trên tên bài hát và nghệ sĩ.
        Sử dụng thuật toán so khớp chuỗi mờ (Fuzzy matching) để tối ưu độ chính xác.
        """
        pass

    @abstractmethod
    def batch_search_tracks(
        self,
        queries: list[dict[str, str]],
        max_workers: int = 4,
        on_progress: Any = None,
    ) -> list[dict[str, Any]]:
        """
        Tra cứu hàng loạt danh sách bài hát trên Spotify bằng Multi-threading Concurrency Pool.
        Mỗi query là một dict {"query": ..., "artist": ...} hoặc chuỗi query.
        """
        pass
