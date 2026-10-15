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
