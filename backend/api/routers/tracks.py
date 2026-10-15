from fastapi import APIRouter, Depends

from backend.core.container import get_container
from backend.core.models.entities import DetectionResult, TrackItem

from ..schemas.dtos import StandardResponse, TrackDetectRequest

router = APIRouter(prefix="/tracks", tags=["Tracks"])


def get_track_controller():
    return get_container().track_controller


@router.post("/detect", response_model=StandardResponse[DetectionResult])
def detect_single_track(request: TrackDetectRequest, controller=Depends(get_track_controller)):
    """Giám định một bài hát từ URL, Tên bài, hoặc Đường dẫn file."""
    return controller.detect_track(request)


@router.get("/resolve", response_model=StandardResponse[TrackItem])
def resolve_track_metadata(query: str, controller=Depends(get_track_controller)):
    """Tìm kiếm và phân giải siêu dữ liệu bài hát."""
    return controller.resolve_track_info(query)
