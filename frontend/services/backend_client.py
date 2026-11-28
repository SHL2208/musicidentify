from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.api.schemas.dtos import BatchDetectRequest, ExportReportRequest, TrackDetectRequest
from backend.core.config import SPOTIFY_SEARCH_LIMIT
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
        self.batch_job_manager = self.container.batch_job_manager

    def detect_single_track(
        self,
        query: str,
        track_id: str | None = None,
        model_override: str | None = None,
        force_rescan: bool = False,
        language: str = "vi",
        enable_spotify: bool = True,
    ) -> DetectionResult:
        """Gọi Controller phân tích một bài hát đơn lẻ (hỗ trợ Cache và Force Re-scan)."""
        req = TrackDetectRequest(
            query=query,
            track_id=track_id,
            model_override=model_override,
            force_rescan=force_rescan,
            language=language,
            enable_spotify=enable_spotify,
        )
        resp = self.track_controller.detect_track(req)
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def search_spotify_tracks(
        self, query: str, artist: str = "", limit: int = SPOTIFY_SEARCH_LIMIT
    ) -> list[dict]:
        """Gọi Controller tìm kiếm danh sách bài hát trên Spotify."""
        resp = self.track_controller.search_spotify(query=query, artist=artist, limit=limit)
        if not resp.success:
            return []
        return resp.data or []

    def batch_search_spotify_tracks(
        self,
        queries: list[dict | str],
        max_workers: int = 4,
        on_progress: Callable[[int, int, dict], None] | None = None,
    ) -> list[dict]:
        """Gọi Controller tìm kiếm hàng loạt bài hát trên Spotify qua multi-threading."""
        resp = self.track_controller.batch_search_spotify(
            queries=queries, max_workers=max_workers, on_progress=on_progress
        )
        if not resp.success:
            return []
        return resp.data or []

    def parse_spotify_batch_input(self, input_data: str | Path) -> tuple[list[dict], Any]:
        """Bóc tách danh sách bài hát từ file hoặc text."""
        return self.container.resolver.parse_spotify_batch_input(input_data)

    def export_spotify_batch_report(
        self, original_df: Any, results: list[dict], format: str = "xlsx"
    ) -> bytes:
        """Xuất file báo cáo đối soát 1-1 cho danh sách Spotify tra cứu được."""
        return self.container.report_service.export_spotify_batch_report(
            original_df=original_df, results=results, format=format
        )

    def correct_verdict(
        self,
        track_id: str,
        is_ai: bool,
        verdict_type: str,
        verdict_label: str,
        note: str | None = None,
    ) -> DetectionResult:
        """Gọi Controller đính chính phán quyết của bài hát theo can thiệp con người."""
        from backend.api.schemas.dtos import TrackCorrectionRequest

        req = TrackCorrectionRequest(
            track_id=track_id,
            is_ai=is_ai,
            verdict_type=verdict_type,
            verdict_label=verdict_label,
            note=note,
        )
        resp = self.track_controller.correct_track_verdict(req)
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def process_batch_queries(
        self,
        queries: list[str],
        model_override: str | None = None,
        scan_mode: str = "smart",
        language: str = "vi",
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
        stop_check: Callable[[], bool] | None = None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
        enable_spotify: bool = True,
    ) -> BatchReport:
        """Gọi Controller xử lý lô bài hát từ danh sách queries."""
        req = BatchDetectRequest(
            queries=queries,
            model_override=model_override,
            scan_mode=scan_mode,
            language=language,
            enable_spotify=enable_spotify,
        )
        resp = self.batch_controller.process_queries(
            req,
            on_progress=on_progress,
            stop_check=stop_check,
            start_index=start_index,
            initial_results=initial_results,
        )
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def process_batch_file(
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
        enable_spotify: bool = True,
    ) -> BatchReport:
        """Gọi Controller xử lý lô bài hát từ tệp (.txt, .csv, .xlsx)."""
        resp = self.batch_controller.process_file(
            file_path,
            model_override=model_override,
            scan_mode=scan_mode,
            language=language,
            on_progress=on_progress,
            stop_check=stop_check,
            start_index=start_index,
            initial_results=initial_results,
            enable_spotify=enable_spotify,
        )
        if not resp.success or resp.data is None:
            raise RuntimeError(resp.error or resp.message)
        return resp.data

    def export_report(self, format: str = "xlsx") -> str | None:
        """Gọi Controller xuất báo cáo kết quả."""
        req = ExportReportRequest(format=format)
        resp = self.report_controller.export_results(req)
        return resp.data if resp.success else None

    def export_custom_results(
        self, results: list[DetectionResult], format: str = "xlsx"
    ) -> str | None:
        """Xuất danh sách kết quả bất kỳ ra file báo cáo Excel."""
        if not results:
            return None
        res_file = self.container.report_service.export_report(results, format=format)
        return str(res_file) if res_file else None

    def export_report_bytes(
        self, results: list[DetectionResult], format: str = "xlsx"
    ) -> bytes | None:
        """Xuất danh sách kết quả bất kỳ ra mảng bytes trong RAM (không ghi đĩa)."""
        if not results:
            return None
        return self.container.report_service.export_report_bytes(results, format=format)

    def get_history(self) -> list[DetectionResult]:
        """Truy xuất lịch sử giám định."""
        resp = self.report_controller.get_history()
        return resp.data or []

    # --- Tương tác với Background Batch Job Manager ---

    def get_batch_job_snapshot(self):
        """Lấy snapshot của tác vụ quét hàng loạt hiện tại."""
        return self.batch_job_manager.get_active_job_snapshot()

    def get_batch_queue_snapshots(self):
        """Lấy danh sách các tác vụ đang nằm trong hàng đợi chờ xử lý."""
        return self.batch_job_manager.get_queue_snapshots()

    def enqueue_batch_job(
        self,
        file_path: Path | str | None = None,
        queries: list[str] | None = None,
        scan_mode: str = "smart",
        model_override: str | None = None,
        job_name: str | None = None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
        language: str = "vi",
        enable_spotify: bool = True,
    ):
        """Gửi yêu cầu quét hàng loạt vào hàng đợi chạy ngầm."""
        return self.batch_job_manager.enqueue_job(
            file_path=file_path,
            queries=queries,
            scan_mode=scan_mode,
            model_override=model_override,
            job_name=job_name,
            start_index=start_index,
            initial_results=initial_results,
            language=language,
            enable_spotify=enable_spotify,
        )

    def cancel_queued_batch_job(self, job_id: str) -> bool:
        """Hủy một tác vụ trong hàng đợi."""
        return self.batch_job_manager.cancel_queued_job(job_id)

    def stop_batch_job(self) -> bool:
        """Yêu cầu dừng an toàn tác vụ quét ngầm đang chạy."""
        return self.batch_job_manager.request_stop()

    def resume_batch_job(self) -> tuple[bool, str]:
        """Tiếp tục chạy tác vụ đang bị tạm dừng (Circuit Breaker hoặc Dừng bởi người dùng)."""
        return self.batch_job_manager.resume_active_job()

    def reset_batch_job(self):
        """Xóa sạch trạng thái quét và hàng đợi để bắt đầu mới."""
        self.batch_job_manager.reset_job()

    def refresh_catalog_cache(self) -> None:
        """Làm mới bộ nhớ đệm danh mục (Artists, AI projects, Song Types)."""
        if hasattr(self.container, "catalog_repository"):
            self.container.catalog_repository.refresh_cache()

    def add_verified_human_artist(
        self, name: str, aliases: str | None = None, notes: str | None = None
    ) -> bool:
        """Thêm ca sĩ người thật vào Database và làm mới cache."""
        if hasattr(self.container, "catalog_repository"):
            return self.container.catalog_repository.add_verified_human_artist(
                name, aliases=aliases, notes=notes
            )
        return False

    def add_ai_artist_project(
        self, name: str, regex_pattern: str | None = None, notes: str | None = None
    ) -> bool:
        """Thêm dự án nghệ sĩ ảo AI vào Database và làm mới cache."""
        if hasattr(self.container, "catalog_repository"):
            return self.container.catalog_repository.add_ai_artist_project(
                name, regex_pattern=regex_pattern, notes=notes
            )
        return False
