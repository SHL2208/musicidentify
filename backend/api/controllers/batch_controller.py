from collections.abc import Callable
from pathlib import Path

from backend.core.interfaces.services.batch_service_interface import IBatchService
from backend.core.interfaces.services.resolver_interface import IInputResolver
from backend.core.models.entities import BatchReport, DetectionResult, TrackItem

from ..schemas.dtos import BatchDetectRequest, StandardResponse


class BatchController:
    """
    Controller điều phối yêu cầu phân tích hàng loạt.
    Gọi IBatchService và IInputResolver qua Interface.
    """

    def __init__(self, batch_service: IBatchService, resolver: IInputResolver):
        self.batch_service = batch_service
        self.resolver = resolver

    def process_queries(
        self,
        request: BatchDetectRequest,
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
        stop_check: Callable[[], bool] | None = None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
    ) -> StandardResponse[BatchReport]:
        """Tiếp nhận danh sách truy vấn dạng text và thực hiện quét hàng loạt."""
        try:
            tracks: list[TrackItem] = []
            for q in request.queries:
                clean_q = q.strip()
                if not clean_q:
                    continue
                input_type = self.resolver.detect_input_type(clean_q)
                resolved_u = clean_q if input_type.value == "URL" else None
                from backend.services.resolver_service import generate_deterministic_track_id

                item_id = generate_deterministic_track_id(
                    clean_q, input_type, resolved_url=resolved_u
                )
                tracks.append(
                    TrackItem(
                        id=item_id,
                        raw_query=clean_q,
                        input_type=input_type,
                        track_title=Path(clean_q).stem
                        if input_type.value == "LOCAL_FILE"
                        else clean_q,
                        resolved_url=resolved_u,
                    )
                )

            scan_m = getattr(request, "scan_mode", "smart")
            lang = getattr(request, "language", "vi") or "vi"
            report = self.batch_service.process_batch(
                tracks=tracks,
                model_override=request.model_override,
                scan_mode=scan_m,
                language=lang,
                on_progress=on_progress,
                stop_check=stop_check,
                start_index=start_index,
                initial_results=initial_results,
            )
            return StandardResponse[BatchReport](
                success=True, message=f"Đã xử lý xong lô {report.total_tracks} bài hát", data=report
            )
        except Exception as e:
            return StandardResponse[BatchReport](
                success=False, message="Lỗi khi xử lý danh sách hàng loạt", error=str(e)
            )

    def process_file(
        self,
        file_path: Path,
        model_override: str | None = None,
        scan_mode: str = "smart",
        language: str = "vi",
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
        stop_check: Callable[[], bool] | None = None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
    ) -> StandardResponse[BatchReport]:
        """Tiếp nhận tệp dữ liệu (.txt, .csv, .xlsx) và thực hiện quét hàng loạt."""
        try:
            tracks = self.resolver.parse_batch_file(file_path)
            report = self.batch_service.process_batch(
                tracks=tracks,
                model_override=model_override,
                scan_mode=scan_mode,
                language=language,
                on_progress=on_progress,
                stop_check=stop_check,
                start_index=start_index,
                initial_results=initial_results,
            )
            return StandardResponse[BatchReport](
                success=True,
                message=f"Đã phân tích xong tệp {file_path.name} ({report.total_tracks} bài)",
                data=report,
            )
        except Exception as e:
            return StandardResponse[BatchReport](
                success=False, message=f"Lỗi khi xử lý tệp '{file_path.name}'", error=str(e)
            )
