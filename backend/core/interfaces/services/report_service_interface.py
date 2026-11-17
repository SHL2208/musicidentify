from abc import ABC, abstractmethod
from pathlib import Path

from backend.core.models.entities import DetectionResult


class IReportService(ABC):
    """Interface định nghĩa nghiệp vụ xuất bản báo cáo (Excel / CSV)."""

    @abstractmethod
    def export_report(self, results: list[DetectionResult], format: str = "xlsx") -> Path | None:
        """Xuất danh sách kết quả ra file Excel hoặc CSV."""
        pass

    @abstractmethod
    def export_report_bytes(
        self, results: list[DetectionResult], format: str = "xlsx"
    ) -> bytes | None:
        """Xuất danh sách kết quả ra mảng bytes trong RAM (Excel hoặc CSV)."""
        pass

    def export_spotify_batch_report(
        self, original_df: any, results: list[dict], format: str = "xlsx"
    ) -> bytes:
        """Ghép nối kết quả tra cứu Spotify vào DataFrame gốc (đối soát 1-1) và xuất ra bytes."""
        return b""
