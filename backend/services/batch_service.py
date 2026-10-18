import logging
import time
from collections.abc import Callable
from pathlib import Path

from ..core.config import BATCH_SLEEP_INTERVAL
from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.interfaces.services.batch_service_interface import IBatchService
from ..core.interfaces.services.detection_service_interface import IDetectionService
from ..core.interfaces.services.report_service_interface import IReportService
from ..core.models.entities import BatchReport, DetectionResult, InputType, TrackItem, VerdictType
from ..repositories.file_report_repository import clean_excel_text

logger = logging.getLogger(__name__)


class BatchService(IBatchService):
    """
    Hiện thực xử lý danh sách bài hát có điều tiết tốc độ (Rate Limiting).
    Tuân thủ interface IBatchService và liên kết qua các Service Interfaces.
    """

    def __init__(
        self,
        detection_service: IDetectionService,
        report_service: IReportService,
        track_repo: ITrackRepository,
        sleep_interval: float = BATCH_SLEEP_INTERVAL,
    ):
        self.detection_service = detection_service
        self.report_service = report_service
        self.track_repo = track_repo
        self.sleep_interval = sleep_interval

    def process_batch(
        self,
        tracks: list[TrackItem],
        model_override: str | None = None,
        scan_mode: str = "smart",
        on_progress: Callable[[int, int, TrackItem, DetectionResult | None, str | None], None]
        | None = None,
    ) -> BatchReport:
        """Xử lý danh sách bài hát có điều tiết tốc độ và cập nhật tiến độ."""
        total = len(tracks)
        results: list[DetectionResult] = []
        ai_count = 0
        human_count = 0
        uncertain_count = 0
        failed_count = 0

        for i, track in enumerate(tracks):
            track.status = "PROCESSING"
            self.track_repo.save_track(track)
            was_cached = False
            try:
                force_rescan = scan_mode == "force_all"
                if scan_mode == "retry_errors":
                    existing = self.track_repo.find_result_by_query_or_url(
                        track.raw_query, track.resolved_url
                    )
                    force_rescan = existing is None or existing.verdict_type == VerdictType.ERROR

                result = self.detection_service.detect_single(
                    track, model_override=model_override, force_rescan=force_rescan
                )
                was_cached = bool(getattr(result, "is_cached", False))
                results.append(result)
                track.status = "COMPLETED"
                self.track_repo.save_track(track)

                if result.verdict_type == VerdictType.UNCERTAIN:
                    uncertain_count += 1
                elif result.is_ai:
                    ai_count += 1
                else:
                    human_count += 1

                if on_progress:
                    on_progress(i + 1, total, track, result, None)

            except Exception as e:
                failed_count += 1
                error_msg = clean_excel_text(str(e)).strip()
                track.status = "FAILED"
                track.error_message = error_msg
                self.track_repo.save_track(track)

                logger.error(
                    "Lỗi xử lý bài '%s': %s",
                    track.track_title or track.raw_query,
                    error_msg,
                )
                print(
                    f"[BatchService] Lỗi xử lý bài '{track.track_title or track.raw_query}': {error_msg}"
                )

                err_source_url = track.resolved_url or (
                    track.raw_query
                    if (track.raw_query and str(track.raw_query).startswith("http"))
                    else None
                )
                if not err_source_url and track.input_type == InputType.LOCAL_FILE:
                    err_source_url = str(track.raw_query)

                error_result = DetectionResult(
                    track_id=track.id,
                    track_title=track.track_title or track.raw_query,
                    source_url=err_source_url,
                    input_type=track.input_type,
                    raw_query=track.raw_query,
                    is_ai=False,
                    verdict_type=VerdictType.ERROR,
                    verdict_label="❌ LỖI XỬ LÝ (ERROR)",
                    confidence_score=0.0,
                    suspected_engine="N/A",
                    reasons=[f"Lỗi kỹ thuật: {error_msg}"],
                    detailed_explanation=f"Không thể hoàn thành thẩm định bài hát do lỗi: {error_msg}",
                    acoustic_metrics=None,
                    spectrogram_image_path=None,
                )
                results.append(error_result)
                self.track_repo.save_result(error_result)

                if on_progress:
                    on_progress(i + 1, total, track, error_result, error_msg)

            finally:
                # Tự động dọn dẹp file âm thanh tạm của bài vừa xong để giữ đĩa luôn trống (chống tràn đĩa khi chạy 900 bài)
                if track.input_type != InputType.LOCAL_FILE and getattr(track, "audio_path", None):
                    try:
                        p = Path(str(track.audio_path))
                        if p.exists() and p.is_file():
                            p.unlink()
                    except Exception:
                        pass

            if i < total - 1 and not was_cached:
                time.sleep(self.sleep_interval)

        # Xuất báo cáo qua IReportService
        report_file = self.report_service.export_report(results, format="xlsx")

        return BatchReport(
            total_tracks=total,
            processed_count=len(results),
            ai_count=ai_count,
            human_count=human_count,
            uncertain_count=uncertain_count,
            failed_count=failed_count,
            results=results,
            report_file_path=str(report_file) if report_file else None,
        )
