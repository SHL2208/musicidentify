from typing import Optional
from backend.core.interfaces.services.detection_service_interface import IDetectionService
from backend.core.interfaces.services.resolver_interface import IInputResolver
from backend.core.models.entities import DetectionResult, TrackItem
from ..schemas.dtos import TrackDetectRequest, StandardResponse

class TrackController:
    """
    Controller điều phối yêu cầu phân tích đơn bài hát.
    Tuân thủ mô hình 3 lớp: nhận request, gọi Service thông qua Interface, trả về DTO chuẩn.
    """

    def __init__(self, detection_service: IDetectionService, resolver: IInputResolver):
        self.detection_service = detection_service
        self.resolver = resolver

    def detect_track(self, request: TrackDetectRequest) -> StandardResponse[DetectionResult]:
        """Tiếp nhận yêu cầu giám định một bài hát."""
        try:
            track = self.resolver.resolve_track(request.query)
            result = self.detection_service.detect_single(track, model_override=request.model_override)
            return StandardResponse[DetectionResult](
                success=True,
                message=f"Đã hoàn thành giám định bài: {result.track_title}",
                data=result
            )
        except Exception as e:
            return StandardResponse[DetectionResult](
                success=False,
                message="Lỗi khi giám định bài hát",
                error=str(e)
            )

    def resolve_track_info(self, query: str) -> StandardResponse[TrackItem]:
        """Tiếp nhận yêu cầu tìm kiếm và phân giải thông tin nguồn nhạc."""
        try:
            track = self.resolver.resolve_track(query)
            return StandardResponse[TrackItem](
                success=True,
                message="Phân giải thông tin thành công",
                data=track
            )
        except Exception as e:
            return StandardResponse[TrackItem](
                success=False,
                message="Lỗi khi phân giải bài hát",
                error=str(e)
            )
