from pathlib import Path
from typing import List, Optional, Callable
from backend.core.container import get_container
from backend.core.models.entities import DetectionResult, BatchReport, TrackItem
from backend.api.schemas.dtos import TrackDetectRequest, BatchDetectRequest, ExportReportRequest

class BackendClient:
    """
    Adapter kết nối Frontend với Backend Layer.
    Tuân thủ nguyên lý Decoupling: Giao diện người dùng không thao tác trực tiếp với logic lõi
    mà thông qua Controller và DTOs.
    """

    def __init__(self, api_key_override: Optional[str] = None, model_override: Optional[str] = None):
        api_keys = [api_key_override] if api_key_override else None
        self.container = get_container(
            api_keys=api_keys,
            default_model=model_override
        )
        self.track_controller = self.container.track_controller
        self.batch_controller = self.container.batch_controller
        self.report_controller = self.container.report_controller

    def detect_single_track(self, query: str, model_override: Optional[str] = None) -> DetectionResult:
        """Gọi Controller phân tích một bài hát đơn lẻ."""
        req = TrackDetectRequest(query=query, model_override=model_override)
        resp = self.track_controller.detect_track(req)
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def process_batch_queries(
        self,
        queries: List[str],
        model_override: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, TrackItem, Optional[DetectionResult], Optional[str]], None]] = None
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
        model_override: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, TrackItem, Optional[DetectionResult], Optional[str]], None]] = None
    ) -> BatchReport:
        """Gọi Controller xử lý lô bài hát từ tệp (.txt, .csv, .xlsx)."""
        resp = self.batch_controller.process_file(file_path, model_override=model_override, on_progress=on_progress)
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def export_report(self, format: str = "xlsx") -> Optional[str]:
        """Gọi Controller xuất báo cáo kết quả."""
        req = ExportReportRequest(format=format)
        resp = self.report_controller.export_results(req)
        return resp.data if resp.success else None

    def get_history(self) -> List[DetectionResult]:
        """Truy xuất lịch sử giám định."""
        resp = self.report_controller.get_history()
        return resp.data or []
