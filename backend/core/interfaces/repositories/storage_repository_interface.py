from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class IStorageRepository(ABC):
    """Interface định nghĩa hợp đồng quản lý lưu trữ tệp tin tạm và tệp tin đầu ra."""

    @abstractmethod
    def create_temp_path(self, filename: str) -> Path:
        """Tạo đường dẫn an toàn trong thư mục tạm."""
        pass

    @abstractmethod
    def cleanup_file(self, path: Optional[Path]) -> bool:
        """Dọn dẹp file tạm an toàn không ném exception."""
        pass

    @abstractmethod
    def get_output_dir(self) -> Path:
        """Lấy thư mục xuất bản báo cáo."""
        pass

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """Lấy thư mục lưu trữ file tạm."""
        pass
