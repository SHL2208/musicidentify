from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List

from ..schemas.dtos import ExportReportRequest, StandardResponse
from backend.core.models.entities import DetectionResult
from backend.core.container import get_container

router = APIRouter(prefix="/reports", tags=["Reports"])

def get_report_controller():
    return get_container().report_controller

@router.post("/export", response_model=StandardResponse[str])
def export_report_file(
    request: ExportReportRequest,
    controller=Depends(get_report_controller)
):
    """Xuất danh sách kết quả ra tệp Excel hoặc CSV."""
    return controller.export_results(request)

@router.get("/history", response_model=StandardResponse[List[DetectionResult]])
def get_analysis_history(
    controller=Depends(get_report_controller)
):
    """Truy xuất lịch sử các kết quả giám định trong phiên."""
    return controller.get_history()

def get_report_repo():
    return get_container().report_repository

@router.get("/download/{filename}")
def download_report_file(
    filename: str,
    repo=Depends(get_report_repo)
):
    """Tải trực tiếp tệp báo cáo đã tạo về máy tính."""
    report_file = repo.get_report_file(filename)
    if not report_file or not report_file.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp báo cáo.")
    return FileResponse(
        path=str(report_file),
        filename=report_file.name,
        media_type="application/octet-stream"
    )
