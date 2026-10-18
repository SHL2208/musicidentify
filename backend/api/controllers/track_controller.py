from backend.core.interfaces.repositories.track_repository_interface import ITrackRepository
from backend.core.interfaces.services.detection_service_interface import IDetectionService
from backend.core.interfaces.services.resolver_interface import IInputResolver
from backend.core.models.entities import DetectionResult, TrackItem

from ..schemas.dtos import StandardResponse, TrackCorrectionRequest, TrackDetectRequest


class TrackController:
    """
    Controller điều phối yêu cầu phân tích đơn bài hát.
    Tuân thủ mô hình 3 lớp: nhận request, gọi Service thông qua Interface, trả về DTO chuẩn.
    """

    def __init__(
        self,
        detection_service: IDetectionService,
        resolver: IInputResolver,
        track_repo: ITrackRepository | None = None,
    ):
        self.detection_service = detection_service
        self.resolver = resolver
        self.track_repo = track_repo

    def detect_track(self, request: TrackDetectRequest) -> StandardResponse[DetectionResult]:
        """Tiếp nhận yêu cầu giám định một bài hát (hỗ trợ Cache & Force Re-scan)."""
        try:
            force_rescan_flag = getattr(request, "force_rescan", False)
            try:
                track = self.resolver.resolve_track(request.query, force_rescan=force_rescan_flag)
            except TypeError:
                track = self.resolver.resolve_track(request.query)
            result = self.detection_service.detect_single(
                track,
                model_override=request.model_override,
                force_rescan=force_rescan_flag,
            )
            return StandardResponse[DetectionResult](
                success=True,
                message=f"Đã hoàn thành giám định bài: {result.track_title}",
                data=result,
            )
        except Exception as e:
            return StandardResponse[DetectionResult](
                success=False, message="Lỗi khi giám định bài hát", error=str(e)
            )

    def correct_track_verdict(
        self, request: TrackCorrectionRequest
    ) -> StandardResponse[DetectionResult]:
        """Tiếp nhận yêu cầu đính chính phán quyết của con người."""
        try:
            if not self.track_repo:
                return StandardResponse[DetectionResult](
                    success=False,
                    message="Kho lưu trữ chưa sẵn sàng để đính chính",
                    error="Repository not available",
                )
            updated_res = self.track_repo.correct_verdict(
                track_id=request.track_id,
                is_ai=request.is_ai,
                verdict_type=request.verdict_type,
                verdict_label=request.verdict_label,
                note=request.note,
            )
            if not updated_res:
                return StandardResponse[DetectionResult](
                    success=False,
                    message=f"Không tìm thấy bài hát có ID '{request.track_id}' để đính chính",
                )
            return StandardResponse[DetectionResult](
                success=True,
                message=f"Đã đính chính thành công cho bài: {updated_res.track_title}",
                data=updated_res,
            )
        except Exception as e:
            return StandardResponse[DetectionResult](
                success=False, message="Lỗi khi đính chính phán quyết", error=str(e)
            )

    def resolve_track_info(self, query: str) -> StandardResponse[TrackItem]:
        """Tiếp nhận yêu cầu tìm kiếm và phân giải thông tin nguồn nhạc."""
        try:
            track = self.resolver.resolve_track(query)
            return StandardResponse[TrackItem](
                success=True, message="Phân giải thông tin thành công", data=track
            )
        except Exception as e:
            return StandardResponse[TrackItem](
                success=False, message="Lỗi khi phân giải bài hát", error=str(e)
            )
