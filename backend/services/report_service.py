from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.models.entities import DetectionResult, VerdictType
from ..core.interfaces.services.report_service_interface import IReportService
from ..core.interfaces.repositories.report_repository_interface import IReportRepository

class ReportService(IReportService):
    """
    Hiện thực dịch vụ xuất báo cáo.
    Tuân thủ interface IReportService và phụ thuộc vào IReportRepository.
    """

    def __init__(self, report_repo: IReportRepository):
        self.report_repo = report_repo

    def export_report(self, results: List[DetectionResult], format: str = "xlsx") -> Optional[Path]:
        """Xuất danh sách kết quả ra file Excel hoặc CSV."""
        if not results:
            return None

        records: List[Dict[str, Any]] = []
        for idx, r in enumerate(results):
            if r.verdict_type == VerdictType.ERROR:
                records.append({
                    "STT": idx + 1,
                    "Tên bài hát": r.track_title,
                    "Kết luận thẩm định": r.verdict_label,
                    "Phân loại kỹ thuật": "ERROR",
                    "Độ tin cậy (%)": "0.0%",
                    "Điểm Fakeprint Fourier (ISMIR 2025)": "N/A",
                    "Độ mở dải tần (Rolloff 95%)": "N/A",
                    "Đánh giá nén truyền tải": "N/A",
                    "Nền tảng nghi vấn": "N/A",
                    "Luận cứ chính / Chi tiết lỗi": r.reasons[0] if r.reasons else "Lỗi không xác định",
                    "Giải thích chi tiết": r.detailed_explanation,
                    "Nguồn link": r.source_url or "N/A",
                    "Thời gian quét": r.processed_at
                })
            else:
                fakeprint_str = (
                    f"{r.acoustic_metrics.fakeprint_score * 100:.1f}%"
                    if r.acoustic_metrics else "N/A"
                )
                rolloff_str = (
                    f"{r.acoustic_metrics.spectral_rolloff_95_hz:.0f} Hz"
                    if r.acoustic_metrics else "N/A"
                )
                comp_source = (
                    r.acoustic_metrics.compression_source
                    if r.acoustic_metrics else "N/A"
                )
                records.append({
                    "STT": idx + 1,
                    "Tên bài hát": r.track_title,
                    "Kết luận thẩm định": r.verdict_label,
                    "Phân loại kỹ thuật": r.verdict_type.value if hasattr(r.verdict_type, 'value') else str(r.verdict_type),
                    "Độ tin cậy (%)": f"{r.confidence_score * 100:.1f}%",
                    "Điểm Fakeprint Fourier (ISMIR 2025)": fakeprint_str,
                    "Độ mở dải tần (Rolloff 95%)": rolloff_str,
                    "Đánh giá nén truyền tải": comp_source,
                    "Nền tảng nghi vấn": r.suspected_engine,
                    "Luận cứ chính / Chi tiết lỗi": " | ".join(r.reasons),
                    "Giải thích chi tiết": r.detailed_explanation,
                    "Nguồn link": r.source_url or "N/A",
                    "Thời gian quét": r.processed_at
                })

        return self.report_repo.save_report_file(records, format=format)
