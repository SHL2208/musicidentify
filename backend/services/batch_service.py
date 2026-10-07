import logging
import time
from typing import List, Optional, Callable

from ..core.config import BATCH_SLEEP_INTERVAL
from ..core.models.entities import TrackItem, DetectionResult, BatchReport, VerdictType
from ..core.interfaces.services.batch_service_interface import IBatchService
from ..core.interfaces.services.detection_service_interface import IDetectionService
from ..core.interfaces.services.report_service_interface import IReportService
from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
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
        sleep_interval: float = BATCH_SLEEP_INTERVAL
    ):
        self.detection_service = detection_service
        self.report_service = report_service
        self.track_repo = track_repo
        self.sleep_interval = sleep_interval

    def process_batch(
        self,
        tracks: List[TrackItem],
        model_override: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, TrackItem, Optional[DetectionResult], Optional[str]], None]] = None
    ) -> BatchReport:
        """Xử lý danh sách bài hát có điều tiết tốc độ và cập nhật tiến độ."""
        total = len(tracks)
        results: List[DetectionResult] = []
        ai_count = 0
        human_count = 0
        failed_count = 0

        for i, track in enumerate(tracks):
            track.status = "PROCESSING"
            self.track_repo.save_track(track)
            try:
                result = self.detection_service.detect_single(track, model_override=model_override)
                results.append(result)
                track.status = "COMPLETED"
                self.track_repo.save_track(track)

                if result.is_ai:
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
                print(f"[BatchService] Lỗi xử lý bài '{track.track_title or track.raw_query}': {error_msg}")

                error_result = DetectionResult(
                    track_id=track.id,
                    track_title=track.track_title or track.raw_query,
                    source_url=track.resolved_url or track.raw_query,
                    is_ai=False,
                    verdict_type=VerdictType.ERROR,
                    verdict_label="❌ LỖI XỬ LÝ (ERROR)",
                    confidence_score=0.0,
                    suspected_engine="N/A",
                    reasons=[f"Lỗi kỹ thuật: {error_msg}"],
                    detailed_explanation=f"Không thể hoàn thành thẩm định bài hát do lỗi: {error_msg}",
                    acoustic_metrics=None,
                    spectrogram_image_path=None
                )
                results.append(error_result)
                self.track_repo.save_result(error_result)

                if on_progress:
                    on_progress(i + 1, total, track, error_result, error_msg)

            if i < total - 1:
                time.sleep(self.sleep_interval)

        # Xuất báo cáo qua IReportService
        report_file = self.report_service.export_report(results, format="xlsx")

        return BatchReport(
            total_tracks=total,
            processed_count=len(results),
            ai_count=ai_count,
            human_count=human_count,
            failed_count=failed_count,
            results=results,
            report_file_path=str(report_file) if report_file else None
        )
