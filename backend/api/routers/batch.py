import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from backend.core.container import get_container
from backend.core.models.entities import BatchReport

from ..schemas.dtos import BatchDetectRequest, StandardResponse

router = APIRouter(prefix="/batch", tags=["Batch"])


def get_batch_controller():
    return get_container().batch_controller


def get_storage_repo():
    return get_container().storage_repository


@router.post("/queries", response_model=StandardResponse[BatchReport])
def process_queries_batch(request: BatchDetectRequest, controller=Depends(get_batch_controller)):
    """Xử lý danh sách bài hát từ mảng văn bản URL / Tên bài hát."""
    return controller.process_queries(request)


@router.post("/upload", response_model=StandardResponse[BatchReport])
async def upload_and_process_batch_file(
    file: UploadFile = File(...),
    model_override: str | None = None,
    controller=Depends(get_batch_controller),
    storage=Depends(get_storage_repo),
):
    """Tải lên file danh sách (.txt, .csv, .xlsx) để quét tự động."""
    temp_dir = storage.get_temp_dir()
    file_ext = Path(file.filename or "upload.txt").suffix
    saved_path = temp_dir / f"batch_upload_{uuid.uuid4().hex[:8]}{file_ext}"

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    report_resp = controller.process_file(saved_path, model_override=model_override)
    storage.cleanup_file(saved_path)
    return report_resp
