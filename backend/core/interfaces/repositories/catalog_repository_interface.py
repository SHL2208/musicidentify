from abc import ABC, abstractmethod
from typing import Any


class ICatalogRepository(ABC):
    """Interface định nghĩa truy xuất danh mục nghệ sĩ, dự án AI và nhà phân phối."""

    @abstractmethod
    def get_catalog_data(self) -> dict[str, Any]:
        """
        Lấy toàn bộ dữ liệu danh mục gồm:
        - human_artists: set[str] (tên chữ thường)
        - human_artists_unaccented: set[str] (tên không dấu chữ thường)
        - ai_artists: list[tuple[str, str]] (regex_pattern, project_name)
        - whitelist: list[str] (tên nhà phân phối whitelist chữ thường)
        - ai_distributors: list[str] (tên nhà phân phối AI chữ thường)
        - famous_cover_songs: dict[str, str] (tên bài hát -> nghệ sĩ gốc)
        - famous_remix_songs: set[str]
        - specialized_cover_channels: set[str]
        """
        pass

    @abstractmethod
    def get_famous_cover_catalog(self) -> dict[str, str]:
        """Lấy danh mục bài hát cover nổi tiếng (tên bài -> nghệ sĩ gốc)."""
        pass

    @abstractmethod
    def get_famous_remix_catalog(self) -> set[str]:
        """Lấy danh mục bài hát remix đặc thù."""
        pass

    @abstractmethod
    def get_specialized_cover_channels(self) -> set[str]:
        """Lấy danh mục các kênh YouTube/nghệ sĩ chuyên hát cover."""
        pass

    @abstractmethod
    def is_verified_human_artist(self, artist_str: str, title_str: str = "") -> bool:
        """Kiểm tra xem tên nghệ sĩ hoặc tiêu đề bài hát có thuộc ca sĩ người thật đã xác thực hay không."""
        pass

    @abstractmethod
    def add_verified_human_artist(
        self, name: str, aliases: str | None = None, notes: str | None = None
    ) -> bool:
        """Thêm mới ca sĩ người thật vào Database và làm mới cache."""
        pass

    @abstractmethod
    def add_ai_artist_project(
        self, name: str, regex_pattern: str | None = None, notes: str | None = None
    ) -> bool:
        """Thêm mới dự án nghệ sĩ ảo AI vào Database và làm mới cache."""
        pass

    @abstractmethod
    def refresh_cache(self) -> None:
        """Làm mới bộ nhớ đệm (in-memory cache) ngay lập tức từ DB."""
        pass
