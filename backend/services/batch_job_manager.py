import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from backend.core.config import TEMP_DIR
from backend.core.interfaces.services.batch_service_interface import IBatchService
from backend.core.interfaces.services.report_service_interface import IReportService
from backend.core.interfaces.services.resolver_interface import IInputResolver
from backend.core.models.entities import BatchReport, DetectionResult, TrackItem, VerdictType

logger = logging.getLogger(__name__)

BATCH_STATE_FILE = TEMP_DIR / "batch_state.json"
CIRCUIT_BREAKER_THRESHOLD = 5


class JobStatus(str, Enum):
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def format_time_duration(seconds: float | None) -> str:
    """Định dạng số giây thành chuỗi hiển thị dễ đọc (VD: '3m 45s', '1h 12m')."""
    if not seconds or seconds <= 0:
        return "0s"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def format_detection_row(r: DetectionResult, stt: int | None = None) -> dict:
    """Định dạng kết quả giám định thành một hàng hiển thị đồng nhất trong Data Table."""
    origin_tag = "⚡ CSDL (Cached)" if getattr(r, "is_cached", False) else "🔄 Quét mới"
    corr_tag = "🧑‍🎤 Đã đính chính" if getattr(r, "is_human_corrected", False) else "-"
    ai_model_col = r.ai_model_verdict or ("AI" if getattr(r, "ai_model_is_ai", False) else "HUMAN")
    consensus_col = r.consensus_verdict or r.verdict_label
    conf = getattr(r, "confidence_score", 0.0)
    conf_str = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "0%"
    reasons = r.reasons or []
    song_type_val = r.song_type or getattr(r, "work_label", "-") or "-"
    distributor_val = r.distributor or "-"
    release_date_val = r.release_date or "-"

    dur_sec = getattr(r, "duration_sec", None)
    if (dur_sec is None or dur_sec <= 0) and r.acoustic_metrics:
        dur_sec = getattr(r.acoustic_metrics, "duration_sec", None)
    dur_str = format_time_duration(dur_sec) if dur_sec and dur_sec > 0 else "-"

    res_dict = {}
    if stt is not None:
        res_dict["STT"] = stt
    res_dict.update(
        {
            "Tên bài hát": r.track_title,
            "Thời lượng": dur_str,
            "Loại tác phẩm": song_type_val,
            "Nhà phân phối": distributor_val,
            "Ngày phát hành": release_date_val,
            "Đánh giá Model AI": ai_model_col,
            "Kết luận Tổng hợp": consensus_col,
            "Nguồn gốc": origin_tag,
            "Đính chính": corr_tag,
            "Độ tin cậy": conf_str,
            "Mô hình nghi vấn": r.suspected_engine or "N/A",
            "Luận cứ": " | ".join(reasons[:2]) if reasons else "N/A",
            "Nguồn link": r.source_url or "N/A",
        }
    )
    return res_dict


@dataclass
class BatchJobSnapshot:
    """Bản chụp trạng thái an toàn (Thread-safe Snapshot) để UI đọc mà không bị Race Condition."""

    job_id: str
    job_name: str
    status: JobStatus
    scan_mode: str
    total_tracks: int
    current_index: int
    current_track_title: str
    submitted_at: float
    start_time: float | None
    finish_time: float | None
    elapsed_sec: float
    speed_per_track: float
    eta_sec: float
    counts: dict[str, int]
    consecutive_errors: int
    pause_reason: str | None
    stop_requested: bool
    live_rows: list[dict]
    live_results: list[DetectionResult]
    report_file_path: str | None
    partial_excel_path: str | None
    error_message: str | None
    file_path: str | None = None
    queries: list[str] = field(default_factory=list)
    language: str = "vi"


class BatchJob:
    """Thực thể đại diện cho một tác vụ quét hàng loạt trong hàng đợi."""

    def __init__(
        self,
        job_id: str,
        job_name: str,
        file_path: Path | str | None = None,
        queries: list[str] | None = None,
        scan_mode: str = "smart",
        model_override: str | None = None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
        language: str = "vi",
    ):
        self.job_id = job_id
        self.job_name = job_name
        self.file_path = Path(file_path) if file_path else None
        self.queries = list(queries) if queries else []
        self.scan_mode = scan_mode
        self.model_override = model_override
        self.language = language
        self.status: JobStatus = JobStatus.QUEUED
        self.submitted_at: float = time.time()
        self.start_time: float | None = None
        self.finish_time: float | None = None

        self.total_tracks: int = 0
        self.current_index: int = start_index
        self.current_track_title: str = ""
        self.counts: dict[str, int] = {"ai": 0, "human": 0, "uncertain": 0, "failed": 0}
        self.consecutive_errors: int = 0
        self.pause_reason: str | None = None
        self.stop_requested: bool = False
        self.user_stop_requested: bool = False
        self.circuit_breaker_triggered: bool = False

        self.live_rows: list[dict] = []
        self.live_results: list[DetectionResult] = list(initial_results) if initial_results else []
        self.report_file_path: str | None = None
        self.partial_excel_path: str | None = None
        self.error_message: str | None = None

        # Khôi phục counts từ initial_results nếu có
        for idx, r in enumerate(self.live_results, 1):
            if getattr(r, "verdict_type", None) == VerdictType.UNCERTAIN:
                self.counts["uncertain"] += 1
            elif getattr(r, "verdict_type", None) == VerdictType.ERROR:
                self.counts["failed"] += 1
            elif r.is_ai:
                self.counts["ai"] += 1
            else:
                self.counts["human"] += 1
            self.live_rows.append(format_detection_row(r, stt=idx))

    def get_snapshot(self) -> BatchJobSnapshot:
        """Tạo bản sao dữ liệu an toàn để giao diện Streamlit render."""
        now = time.time()
        start_t = self.start_time or now
        elapsed = max(0.0, (self.finish_time or now) - start_t) if self.start_time else 0.0

        processed = len(self.live_results)
        speed = elapsed / max(1, processed) if processed > 0 else 0.0
        remaining = max(0, self.total_tracks - processed)
        eta = remaining * speed if speed > 0 else 0.0

        return BatchJobSnapshot(
            job_id=self.job_id,
            job_name=self.job_name,
            status=self.status,
            scan_mode=self.scan_mode,
            total_tracks=self.total_tracks,
            current_index=self.current_index,
            current_track_title=self.current_track_title,
            submitted_at=self.submitted_at,
            start_time=self.start_time,
            finish_time=self.finish_time,
            elapsed_sec=elapsed,
            speed_per_track=speed,
            eta_sec=eta,
            counts=dict(self.counts),
            consecutive_errors=self.consecutive_errors,
            pause_reason=self.pause_reason,
            stop_requested=self.stop_requested,
            live_rows=list(self.live_rows),
            live_results=list(self.live_results),
            report_file_path=self.report_file_path,
            partial_excel_path=self.partial_excel_path,
            error_message=self.error_message,
            file_path=str(self.file_path) if self.file_path else None,
            queries=list(self.queries),
            language=getattr(self, "language", "vi"),
        )

    def to_persistent_dict(self) -> dict:
        """Chuyển thành dict để ghi xuống đĩa batch_state.json."""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.status.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "queries": self.queries,
            "scan_mode": self.scan_mode,
            "language": getattr(self, "language", "vi"),
            "total_tracks": self.total_tracks,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "counts": self.counts,
            "consecutive_errors": self.consecutive_errors,
            "pause_reason": self.pause_reason,
            "live_rows": self.live_rows,
            "live_results": [r.model_dump() for r in self.live_results],
            "report_file_path": self.report_file_path,
            "partial_excel_path": self.partial_excel_path,
            "saved_at": time.time(),
        }


class BatchJobManager:
    """
    Trình điều phối tác vụ quét hàng loạt chạy ngầm (Background Worker & Job Queue Manager).
    Tuân thủ:
    1. Singleton & Thread-safe (RLock)
    2. Bounded Concurrency (Tối đa 1 worker chạy ngầm cùng lúc, chống sập server)
    3. FIFO Job Queue (Nhiều người nộp bài tự động xếp hàng)
    4. Circuit Breaker (Tạm dừng an toàn sau 5 lỗi liên tiếp)
    5. Disk State Persistence (Chống mất mát khi server tắt hoặc tab reload)
    """

    _instance: "BatchJobManager | None" = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        batch_service: IBatchService,
        resolver: IInputResolver,
        report_service: IReportService | None = None,
    ):
        self.batch_service = batch_service
        self.resolver = resolver
        self.report_service: IReportService | None = report_service or getattr(
            batch_service, "report_service", None
        )
        self._lock = threading.RLock()
        self._queue: deque[BatchJob] = deque()
        self._active_job: BatchJob | None = None
        self._last_completed_job: BatchJob | None = None
        self._worker_thread: threading.Thread | None = None

        # Nạp lại trạng thái từ đĩa nếu tồn tại
        self._restore_from_disk()

    @classmethod
    def get_instance(
        cls,
        batch_service: IBatchService | None = None,
        resolver: IInputResolver | None = None,
        report_service: IReportService | None = None,
    ) -> "BatchJobManager":
        """Lấy singleton instance của BatchJobManager."""
        with cls._singleton_lock:
            if cls._instance is None:
                if batch_service is None or resolver is None:
                    raise RuntimeError("BatchJobManager chưa được khởi tạo với dependencies!")
                cls._instance = cls(
                    batch_service=batch_service,
                    resolver=resolver,
                    report_service=report_service,
                )
            elif report_service is not None and cls._instance.report_service is None:
                cls._instance.report_service = report_service
            return cls._instance

    def enqueue_job(
        self,
        file_path: Path | str | None = None,
        queries: list[str] | None = None,
        scan_mode: str = "smart",
        model_override: str | None = None,
        job_name: str | None = None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
        language: str = "vi",
    ) -> tuple[bool, str, BatchJob]:
        """Thêm một lô quét mới vào hàng đợi."""
        with self._lock:
            # Tạo tên đại diện
            if not job_name:
                if file_path:
                    job_name = Path(file_path).name
                elif queries:
                    job_name = f"Lô {len(queries)} bài hát ({queries[0][:25]}...)"
                else:
                    job_name = "Lô bài hát không tên"

            job_id = str(uuid.uuid4())[:8]
            new_job = BatchJob(
                job_id=job_id,
                job_name=job_name,
                file_path=file_path,
                queries=queries,
                scan_mode=scan_mode,
                model_override=model_override,
                start_index=start_index,
                initial_results=initial_results,
                language=language,
            )

            # Đưa vào hàng đợi
            self._queue.append(new_job)
            queue_pos = len(self._queue)
            logger.info(
                "Đã tiếp nhận Job [%s - '%s'] vào hàng đợi (Vị trí #%d)",
                job_id,
                job_name,
                queue_pos,
            )

            # Kích hoạt worker nếu hiện tại chưa có job nào chạy
            self._ensure_worker_running()

            if self._active_job and self._active_job.status == JobStatus.RUNNING:
                msg = f"Đã xếp lô '{job_name}' vào hàng đợi (Vị trí #{queue_pos})."
            else:
                msg = f"Đã bắt đầu xử lý lô '{job_name}' trên luồng ngầm."

            return True, msg, new_job

    def get_active_job_snapshot(self) -> BatchJobSnapshot | None:
        """Lấy snapshot của job đang chạy, tạm dừng, hoặc job vừa hoàn tất gần nhất."""
        with self._lock:
            if self._active_job:
                return self._active_job.get_snapshot()
            if self._last_completed_job:
                return self._last_completed_job.get_snapshot()
            return None

    def get_queue_snapshots(self) -> list[BatchJobSnapshot]:
        """Lấy danh sách các job đang xếp hàng chờ."""
        with self._lock:
            return [job.get_snapshot() for job in self._queue]

    def cancel_queued_job(self, job_id: str) -> bool:
        """Hủy một job đang nằm trong hàng đợi chờ xử lý."""
        with self._lock:
            target_idx = None
            for idx, j in enumerate(self._queue):
                if j.job_id == job_id:
                    target_idx = idx
                    break
            if target_idx is not None:
                removed = self._queue[target_idx]
                del self._queue[target_idx]
                removed.status = JobStatus.STOPPED
                logger.info("Đã hủy Job [%s - '%s'] khỏi hàng đợi", job_id, removed.job_name)
                return True
            return False

    def request_stop(self) -> bool:
        """Gửi cờ dừng an toàn tới job đang chạy."""
        with self._lock:
            if self._active_job and self._active_job.status == JobStatus.RUNNING:
                self._active_job.stop_requested = True
                self._active_job.user_stop_requested = True
                logger.info("Đã gửi cờ yêu cầu dừng cho Job [%s]", self._active_job.job_id)
                return True
            return False

    def resume_active_job(self) -> tuple[bool, str]:
        """Tiếp tục chạy lại active job nếu đang ở trạng thái PAUSED hoặc STOPPED."""
        with self._lock:
            job = self._active_job
            if not job:
                return False, "Không có tác vụ nào đang dừng để tiếp tục."
            if job.status not in (JobStatus.PAUSED, JobStatus.STOPPED):
                return False, f"Tác vụ đang ở trạng thái {job.status.value}, không thể tiếp tục."

            job.status = JobStatus.QUEUED
            job.stop_requested = False
            job.user_stop_requested = False
            job.circuit_breaker_triggered = False
            job.consecutive_errors = 0
            job.pause_reason = None

            # Đưa job trở lại đầu hàng đợi để worker tiếp tục ngay
            self._queue.appendleft(job)
            self._active_job = None
            self._ensure_worker_running()
            return True, f"Đang tiếp tục quét lô '{job.job_name}'..."

    def reset_job(self):
        """Đặt lại trạng thái, xóa tệp lưu vết trên đĩa khi người dùng bấm Đặt lại."""
        with self._lock:
            if self._active_job and self._active_job.status == JobStatus.RUNNING:
                self._active_job.stop_requested = True
                self._active_job.user_stop_requested = True
            self._active_job = None
            self._last_completed_job = None
            self._queue.clear()
            self._clear_disk_state()
            logger.info("Đã đặt lại toàn bộ trạng thái BatchJobManager.")

    def _ensure_worker_running(self):
        """Khởi động worker thread nếu chưa chạy và có việc trong hàng đợi."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            if self._queue or (self._active_job and self._active_job.status == JobStatus.RUNNING):
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, name="BatchJobWorkerThread", daemon=True
                )
                self._worker_thread.start()
                logger.info("Đã kích hoạt Daemon Worker Thread cho BatchJobManager.")

    def _worker_loop(self):
        """Vòng lặp chính của Background Worker: Bốc lần lượt từng job trong hàng đợi để xử lý."""
        while True:
            job: BatchJob | None = None
            with self._lock:
                if not self._queue:
                    # Hết việc
                    self._worker_thread = None
                    break
                job = self._queue.popleft()
                self._active_job = job
                job.status = JobStatus.RUNNING
                if not job.start_time:
                    job.start_time = time.time()
                self._save_state_to_disk(job)

            logger.info("Worker bắt đầu thực thi Job [%s - '%s']", job.job_id, job.job_name)

            should_break = False
            try:
                self._execute_job(job)
            except Exception as e:
                logger.exception("Lỗi nghiêm trọng khi thực thi Job [%s]: %s", job.job_id, e)
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.finish_time = time.time()
                    self._save_state_to_disk(job)
            finally:
                with self._lock:
                    # Nếu job bị PAUSED do Circuit Breaker, dừng worker loop và giữ nguyên active_job
                    if job.status == JobStatus.PAUSED:
                        logger.warning(
                            "Job [%s] tạm dừng an toàn bởi Circuit Breaker: %s",
                            job.job_id,
                            job.pause_reason,
                        )
                        self._save_state_to_disk(job)
                        self._worker_thread = None
                        should_break = True

                    # Nếu job dừng do người dùng
                    if job.status == JobStatus.STOPPED:
                        logger.info("Job [%s] đã dừng theo yêu cầu của người dùng.", job.job_id)
                        self._last_completed_job = job
                        self._save_state_to_disk(job)
                    elif job.status == JobStatus.COMPLETED:
                        logger.info("Job [%s] đã hoàn tất 100%% thành công.", job.job_id)
                        self._last_completed_job = job
                        self._save_state_to_disk(job)

                    # Bàn giao việc tiếp theo
                    if not self._queue:
                        self._worker_thread = None
                        should_break = True

            if should_break:
                break

    def _execute_job(self, job: BatchJob):
        """Tiến hành phân tích danh sách bài hát của Job thông qua BatchService."""
        # 1. Phân giải danh sách bài hát (TrackItem)
        tracks: list[TrackItem] = []
        if job.file_path and job.file_path.exists():
            tracks = self.resolver.parse_batch_file(job.file_path)
        elif job.queries:
            for q in job.queries:
                clean_q = q.strip()
                if not clean_q:
                    continue
                input_t = self.resolver.detect_input_type(clean_q)
                resolved_u = clean_q if input_t.value == "URL" else None
                from backend.services.resolver_service import generate_deterministic_track_id

                item_id = generate_deterministic_track_id(clean_q, input_t, resolved_url=resolved_u)
                tracks.append(
                    TrackItem(
                        id=item_id,
                        raw_query=clean_q,
                        input_type=input_t,
                        track_title=Path(clean_q).stem
                        if input_t.value == "LOCAL_FILE"
                        else clean_q,
                        resolved_url=resolved_u,
                    )
                )

        job.total_tracks = len(tracks)
        start_idx = len(job.live_results)

        # 2. Callback cập nhật tiến độ
        def on_progress_callback(curr, total, track, res, err):
            with self._lock:
                job.current_index = curr
                job.total_tracks = total

                current_stt = len(job.live_rows) + 1
                if res:
                    job.current_track_title = (
                        res.track_title or track.track_title or track.raw_query
                    )
                    row = format_detection_row(res, stt=current_stt)
                    v_type = getattr(res, "verdict_type", None)

                    if v_type == VerdictType.ERROR:
                        job.counts["failed"] += 1
                        job.consecutive_errors += 1
                    else:
                        # Thành công: Reset circuit breaker
                        job.consecutive_errors = 0
                        if v_type == VerdictType.UNCERTAIN:
                            job.counts["uncertain"] += 1
                        elif res.is_ai:
                            job.counts["ai"] += 1
                        else:
                            job.counts["human"] += 1

                    job.live_results.append(res)
                else:
                    job.current_track_title = track.track_title or track.raw_query
                    job.counts["failed"] += 1
                    job.consecutive_errors += 1
                    dur_raw = getattr(track, "duration_sec", 0) or 0
                    dur_str = format_time_duration(dur_raw) if dur_raw > 0 else "-"
                    row = {
                        "STT": current_stt,
                        "Tên bài hát": track.track_title or track.raw_query,
                        "Thời lượng": dur_str,
                        "Loại tác phẩm": getattr(track, "song_type", "-") or "-",
                        "Nhà phân phối": getattr(track, "distributor", "-") or "-",
                        "Ngày phát hành": getattr(track, "release_date", "-") or "-",
                        "Đánh giá Model AI": "ERROR",
                        "Kết luận Tổng hợp": "❌ LỖI XỬ LÝ",
                        "Nguồn gốc": "🔄 Quét mới",
                        "Đính chính": "-",
                        "Độ tin cậy": "0%",
                        "Mô hình nghi vấn": "N/A",
                        "Luận cứ": str(err or "Lỗi không xác định"),
                        "Nguồn link": getattr(track, "resolved_url", "N/A") or "N/A",
                    }

                job.live_rows.append(row)

                # Kiểm tra Circuit Breaker (5 lỗi liên tiếp)
                if job.consecutive_errors >= CIRCUIT_BREAKER_THRESHOLD:
                    job.circuit_breaker_triggered = True
                    job.pause_reason = (
                        f"⚠️ Gặp {job.consecutive_errors} lỗi kỹ thuật liên tiếp "
                        f"(Có thể do YouTube chặn IP hoặc Gemini hết hạn ngạch API). "
                        f"Đã tạm dừng an toàn sau bài '{job.current_track_title}'."
                    )
                    job.stop_requested = True
                    logger.warning("Circuit Breaker kích hoạt: %s", job.pause_reason)

                # Lưu đĩa mỗi bài hoàn thành
                self._save_state_to_disk(job)

        # 3. Hàm kiểm tra tín hiệu dừng
        def stop_checker() -> bool:
            return job.stop_requested

        # 4. Thực thi quét hàng loạt
        batch_kwargs = {
            "tracks": tracks,
            "model_override": job.model_override,
            "scan_mode": job.scan_mode,
            "on_progress": on_progress_callback,
            "stop_check": stop_checker,
            "start_index": start_idx,
            "initial_results": job.live_results,
        }
        import inspect

        if "language" in inspect.signature(self.batch_service.process_batch).parameters:
            batch_kwargs["language"] = getattr(job, "language", "vi") or "vi"

        report: BatchReport = self.batch_service.process_batch(**batch_kwargs)

        with self._lock:
            job.finish_time = time.time()
            if report and report.report_file_path:
                job.report_file_path = report.report_file_path

            if job.circuit_breaker_triggered:
                job.status = JobStatus.PAUSED
                # Lỗi hệ thống: Đảm bảo có 1 bản báo cáo cứu hộ trên đĩa
                if not job.report_file_path and job.live_results and self.report_service:
                    try:
                        rescue_p = self.report_service.export_report(
                            job.live_results, format="xlsx"
                        )
                        if rescue_p:
                            job.report_file_path = str(rescue_p)
                    except Exception as err:
                        logger.error("Không thể xuất file cứu hộ: %s", err)
            elif job.user_stop_requested:
                job.status = JobStatus.STOPPED
                # Người dùng chủ động bấm dừng: KHÔNG lưu file đĩa (chỉ tải On-Demand khi cần)
                if job.report_file_path:
                    try:
                        p = Path(job.report_file_path)
                        if p.exists() and p.is_file():
                            p.unlink()
                    except Exception:
                        pass
                job.report_file_path = None
            else:
                job.status = JobStatus.COMPLETED

            self._save_state_to_disk(job)

    def _save_state_to_disk(self, job: BatchJob | None = None):
        """Lưu trạng thái an toàn xuống tệp batch_state.json."""
        target = job or self._active_job or self._last_completed_job
        if not target:
            return
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            data = target.to_persistent_dict()
            with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Không thể ghi batch_state.json: {e}")

    def _clear_disk_state(self):
        """Xóa tệp lưu vết trên đĩa."""
        try:
            if BATCH_STATE_FILE.exists():
                BATCH_STATE_FILE.unlink()
        except Exception as e:
            logger.warning(f"Không thể xóa batch_state.json: {e}")

    def _restore_from_disk(self):
        """Khôi phục trạng thái từ đĩa khi khởi động server."""
        if not BATCH_STATE_FILE.exists():
            return
        try:
            with open(BATCH_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if not data or not data.get("live_rows"):
                return

            status_str = data.get("status", JobStatus.STOPPED.value)
            # Nếu lúc server tắt job đang RUNNING, khi mở lại ta coi như PAUSED để người dùng bấm tiếp tục
            if status_str == JobStatus.RUNNING.value:
                status_str = JobStatus.PAUSED.value

            restored_results: list[DetectionResult] = []
            for r_dict in data.get("live_results", []):
                try:
                    restored_results.append(DetectionResult.model_validate(r_dict))
                except Exception:
                    pass

            job = BatchJob(
                job_id=data.get("job_id", "restored"),
                job_name=data.get("job_name", "Lô đã khôi phục từ đĩa"),
                file_path=data.get("file_path"),
                queries=data.get("queries"),
                scan_mode=data.get("scan_mode", "smart"),
                start_index=len(restored_results),
                initial_results=restored_results,
                language=data.get("language", "vi"),
            )
            job.status = JobStatus(status_str)
            job.total_tracks = data.get("total_tracks", len(restored_results))
            job.start_time = data.get("start_time")
            job.finish_time = data.get("finish_time")
            job.counts = data.get("counts", job.counts)
            job.consecutive_errors = data.get("consecutive_errors", 0)
            job.pause_reason = data.get("pause_reason")
            job.live_rows = data.get("live_rows", [])
            job.report_file_path = data.get("report_file_path")
            job.partial_excel_path = data.get("partial_excel_path")

            self._last_completed_job = job
            if job.status == JobStatus.PAUSED:
                self._active_job = job

            logger.info(
                "Đã khôi phục thành công trạng thái Job từ đĩa: %d bài hát", len(job.live_rows)
            )
        except Exception as e:
            logger.warning(f"Không thể khôi phục trạng thái từ đĩa: {e}")


def get_batch_job_manager(
    batch_service: IBatchService | None = None,
    resolver: IInputResolver | None = None,
    report_service: IReportService | None = None,
) -> BatchJobManager:
    """Helper lấy hoặc khởi tạo BatchJobManager singleton."""
    return BatchJobManager.get_instance(
        batch_service=batch_service, resolver=resolver, report_service=report_service
    )
