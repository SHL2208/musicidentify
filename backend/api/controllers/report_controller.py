from typing import List, Optional
from backend.core.interfaces.services.report_service_interface import IReportService
from backend.core.interfaces.repositories.track_repository_interface import ITrackRepository
from backend.core.models.entities import DetectionResult
from ..schemas.dtos import ExportReportRequest, StandardResponse

class ReportController:
    """
    Controller phụ trách tạo và xuất dữ liệu báo cáo thẩm định.
    Gọi IReportService và ITrackRepository qua Interface.
    """

    def __init__(self, report_service: IReportService, track_repo: ITrackRepository):
        self.report_service = report_service
        self.track_repo = track_repo

    def export_results(self, request: ExportReportRequest) -> StandardResponse[str]:
        """Xuất toàn bộ kết quả hiện có trong kho dữ liệu ra file báo cáo."""
        try:
            results = self.track_repo.list_results()
            if not results:
                return StandardResponse[str](
                    success=False,
                    message="Chưa có dữ liệu thẩm định nào để xuất báo cáo.",
                    error="NO_DATA"
                )

            report_path = self.report_service.export_report(results, format=request.format)
            if not report_path:
                return StandardResponse[str](
                    success=False,
                    message="Không thể tạo tệp báo cáo.",
                    error="EXPORT_FAILED"
                )

            return StandardResponse[str](
                success=True,
                message="Đã tạo tệp báo cáo thành công",
                data=str(report_path)
            )
        except Exception as e:
            return StandardResponse[str](
                success=False,
                message="Lỗi trong quá trình xuất báo cáo",
                error=str(e)
            )

    def get_history(self) -> StandardResponse[List[DetectionResult]]:
        """Lấy danh sách toàn bộ kết quả đã thực hiện."""
        try:
            results = self.track_repo.list_results()
            return StandardResponse[List[DetectionResult]](
                success=True,
                message=f"Đã lấy {len(results)} kết quả thẩm định",
                data=results
            )
        except Exception as e:
            return StandardResponse[List[DetectionResult]](
                success=False,
                message="Lỗi khi truy xuất lịch sử",
                error=str(e)
            )
