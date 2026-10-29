from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class IReportRepository(ABC):
    """Interface định nghĩa hợp đồng lưu trữ và xuất tệp báo cáo số liệu."""

    @abstractmethod
    def save_report_file(self, records: list[dict[str, Any]], format: str = "xlsx") -> Path:
        """Lưu trữ danh sách bản ghi ra tệp báo cáo dạng Excel (.xlsx) hoặc CSV (.csv)."""
        pass

    @abstractmethod
    def get_report_file(self, filename: str) -> Path | None:
        """Truy xuất đường dẫn tệp báo cáo theo tên file."""
        pass

    @abstractmethod
    def export_report_bytes(self, records: list[dict[str, Any]], format: str = "xlsx") -> bytes:
        """Xuất danh sách bản ghi ra mảng bytes trong bộ nhớ RAM (không ghi đĩa)."""
        pass

    @abstractmethod
    def prune_old_reports(self, max_keep: int = 10) -> int:
        """Dọn dẹp các tệp báo cáo cũ trong thư mục, chỉ giữ lại tối đa max_keep tệp mới nhất."""
        pass
