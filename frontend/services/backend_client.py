from collections.abc import Callable
from pathlib import Path

from backend.api.schemas.dtos import BatchDetectRequest, ExportReportRequest, TrackDetectRequest
from backend.core.container import get_container
from backend.core.models.entities import BatchReport, DetectionResult, TrackItem


class BackendClient:
    """
    Adapter kết nối Frontend với Backend Layer.
    Tuân thủ nguyên lý Decoupling: Giao diện người dùng không thao tác trực tiếp với logic lõi
    mà thông qua Controller và DTOs.
    """

    def __init__(self, api_key_override: str | None = None, model_override: str | None = None):
        api_keys = [api_key_override] if api_key_override else None
        self.container = get_container(api_keys=api_keys, default_model=model_override)
        self.track_controller = self.container.track_controller
        self.batch_controller = self.container.batch_controller
        self.report_controller = self.container.report_controller

    def detect_single_track(self, query: str, model_override: str | None = None) -> DetectionResult:
        """Gọi Controller phân tích một bài hát đơn lẻ."""
        req = TrackDetectRequest(query=query, model_override=model_override)
        resp = self.track_controller.detect_track(req)
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def process_batch_queries(
        self,
        queries: list[str],
        model_override: str | None = None,
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
    ) -> BatchReport:
        """Gọi Controller xử lý lô bài hát từ danh sách queries."""
        req = BatchDetectRequest(queries=queries, model_override=model_override)
        resp = self.batch_controller.process_queries(req, on_progress=on_progress)
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def process_batch_file(
        self,
        file_path: Path,
        model_override: str | None = None,
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
    ) -> BatchReport:
        """Gọi Controller xử lý lô bài hát từ tệp (.txt, .csv, .xlsx)."""
        resp = self.batch_controller.process_file(
            file_path, model_override=model_override, on_progress=on_progress
        )
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def export_report(self, format: str = "xlsx") -> str | None:
        """Gọi Controller xuất báo cáo kết quả."""
        req = ExportReportRequest(format=format)
        resp = self.report_controller.export_results(req)
        return resp.data if resp.success else None

    def get_history(self) -> list[DetectionResult]:
        """Truy xuất lịch sử giám định."""
        resp = self.report_controller.get_history()
        return resp.data or []
