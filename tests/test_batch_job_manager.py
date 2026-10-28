import sys
import time
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.core.interfaces.services.batch_service_interface import IBatchService
from backend.core.interfaces.services.resolver_interface import IInputResolver
from backend.core.models.entities import (
    BatchReport,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.services.batch_job_manager import (
    CIRCUIT_BREAKER_THRESHOLD,
    BatchJobManager,
    JobStatus,
)


class MockResolver(IInputResolver):
    def detect_input_type(self, query: str) -> InputType:
        return InputType.SONG_NAME

    def resolve_track(self, query: str, artist: str = "", force_rescan: bool = False) -> TrackItem:
        return TrackItem(id="t_mock", raw_query=query, input_type=InputType.SONG_NAME, track_title=query)

    def parse_batch_file(self, file_path: Path) -> list[TrackItem]:
        return [
            TrackItem(id=f"t_{i}", raw_query=f"Track {i}", input_type=InputType.SONG_NAME, track_title=f"Track {i}")
            for i in range(1, 6)
        ]

    def search_youtube(
        self,
        song_name: str,
        target_artist: str = "",
        target_duration: float | None = None,
    ) -> tuple[str | None, str | None, float | None, bool, float]:
        return ("https://youtube.com/watch?v=mock", song_name, 180.0, True, 1.0)


class MockBatchService(IBatchService):
    def __init__(self, fail_all: bool = False, delay_per_track: float = 0.05):
        self.fail_all = fail_all
        self.delay_per_track = delay_per_track
        self.processed_tracks = []

    def process_batch(
        self,
        tracks: list[TrackItem],
        model_override: str | None = None,
        scan_mode: str = "smart",
        language: str = "vi",
        on_progress=None,
        stop_check=None,
        start_index: int = 0,
        initial_results: list[DetectionResult] | None = None,
        **kwargs,
    ) -> BatchReport:
        total = len(tracks)
        results = list(initial_results) if initial_results else []

        for i in range(start_index, total):
            if stop_check and stop_check():
                break

            track = tracks[i]
            if self.delay_per_track > 0:
                time.sleep(self.delay_per_track)

            if self.fail_all:
                err_res = DetectionResult(
                    track_id=track.id,
                    track_title=track.track_title or track.raw_query,
                    input_type=track.input_type,
                    raw_query=track.raw_query,
                    is_ai=False,
                    verdict_type=VerdictType.ERROR,
                    verdict_label="❌ LỖI",
                    confidence_score=0.0,
                    suspected_engine="None",
                    reasons=["Mock Error"],
                    detailed_explanation="Lỗi giả lập",
                )
                results.append(err_res)
                if on_progress:
                    on_progress(i + 1, total, track, err_res, "Mock Error")
            else:
                ok_res = DetectionResult(
                    track_id=track.id,
                    track_title=track.track_title or track.raw_query,
                    input_type=track.input_type,
                    raw_query=track.raw_query,
                    is_ai=False,
                    verdict_type=VerdictType.CONFIRMED_HUMAN,
                    verdict_label="HUMAN",
                    confidence_score=0.9,
                    suspected_engine="None",
                    reasons=["Good track"],
                    detailed_explanation="Thẩm định thành công",
                )
                results.append(ok_res)
                self.processed_tracks.append(track)
                if on_progress:
                    on_progress(i + 1, total, track, ok_res, None)

        return BatchReport(
            total_tracks=total,
            processed_count=len(results),
            ai_count=0,
            human_count=len(results) if not self.fail_all else 0,
            uncertain_count=0,
            failed_count=len(results) if self.fail_all else 0,
            results=results,
            report_file_path=str(root_dir / "data" / "temp" / "mock_report.xlsx"),
        )


class TestBatchJobManager(unittest.TestCase):
    def setUp(self):
        self.resolver = MockResolver()
        self.batch_service = MockBatchService(delay_per_track=0.02)
        # Create fresh manager instance
        self.manager = BatchJobManager(batch_service=self.batch_service, resolver=self.resolver)
        self.manager.reset_job()

    def tearDown(self):
        self.manager.reset_job()

    def test_enqueue_and_sequential_fifo_execution(self):
        """Kiểm tra hàng đợi FIFO: Job 1 chạy trước, Job 2 xếp hàng chờ và tự động chạy tiếp."""
        queries_1 = [f"Song A{i}" for i in range(1, 4)]
        queries_2 = [f"Song B{i}" for i in range(1, 3)]

        succ1, msg1, job1 = self.manager.enqueue_job(queries=queries_1, job_name="Job 1")
        self.assertTrue(succ1)

        succ2, msg2, job2 = self.manager.enqueue_job(queries=queries_2, job_name="Job 2")
        self.assertTrue(succ2)

        # Chờ worker thread bốc Job 1 ra làm active_job
        timeout = 1.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            if len(self.manager.get_queue_snapshots()) == 1:
                break
            time.sleep(0.01)

        # Job 2 phải nằm trong hàng đợi
        queue_snaps = self.manager.get_queue_snapshots()
        self.assertEqual(len(queue_snaps), 1)
        self.assertEqual(queue_snaps[0].job_id, job2.job_id)

        # Chờ worker hoàn tất cả 2 jobs
        timeout = 5.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            snap = self.manager.get_active_job_snapshot()
            q = self.manager.get_queue_snapshots()
            if not q and snap and snap.job_id == job2.job_id and snap.status == JobStatus.COMPLETED:
                break
            time.sleep(0.05)

        snap = self.manager.get_active_job_snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.job_id, job2.job_id)
        self.assertEqual(snap.status, JobStatus.COMPLETED)
        self.assertEqual(len(snap.live_results), 2)

    def test_cancel_queued_job(self):
        """Kiểm tra hủy job đang xếp hàng chờ."""
        queries_1 = [f"LongSong {i}" for i in range(1, 10)]
        queries_2 = ["Song To Cancel"]

        self.manager.enqueue_job(queries=queries_1, job_name="Running Job")
        _, _, job2 = self.manager.enqueue_job(queries=queries_2, job_name="Job To Cancel")

        # Chờ worker thread bốc Running Job ra làm active_job
        timeout = 1.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            if len(self.manager.get_queue_snapshots()) == 1:
                break
            time.sleep(0.01)

        # Xác nhận job 2 đang trong hàng đợi
        self.assertEqual(len(self.manager.get_queue_snapshots()), 1)

        # Hủy job 2
        cancelled = self.manager.cancel_queued_job(job2.job_id)
        self.assertTrue(cancelled)

        # Hàng đợi phải rỗng
        self.assertEqual(len(self.manager.get_queue_snapshots()), 0)

        # Dừng job 1
        self.manager.request_stop()

    def test_circuit_breaker_triggers_on_5_consecutive_errors(self):
        """Kiểm tra Circuit Breaker tự ngắt sau 5 lỗi liên tiếp."""
        failing_service = MockBatchService(fail_all=True, delay_per_track=0.01)
        manager = BatchJobManager(batch_service=failing_service, resolver=self.resolver)
        manager.reset_job()

        # Tạo danh sách 10 bài lỗi
        queries = [f"Failing Song {i}" for i in range(1, 11)]
        manager.enqueue_job(queries=queries, job_name="Failing Batch")

        # Chờ cho Circuit Breaker ngắt
        timeout = 5.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            snap = manager.get_active_job_snapshot()
            if snap and snap.status == JobStatus.PAUSED:
                break
            time.sleep(0.05)

        snap = manager.get_active_job_snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.status, JobStatus.PAUSED)
        self.assertGreaterEqual(snap.consecutive_errors, CIRCUIT_BREAKER_THRESHOLD)
        self.assertIn("Gặp 5 lỗi kỹ thuật liên tiếp", snap.pause_reason or "")
        # Phải dừng chính xác tại 5 bài, không quét tiếp bài 6-10
        self.assertEqual(len(snap.live_results), 5)

        manager.reset_job()

    def test_resume_paused_job(self):
        """Kiểm tra tiếp tục quét từ vị trí đã tạm dừng."""
        queries = [f"Song {i}" for i in range(1, 7)]
        self.manager.enqueue_job(queries=queries, job_name="Job To Stop And Resume")

        # Đợi xử lý 2 bài rồi gửi yêu cầu dừng
        timeout = 5.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            snap = self.manager.get_active_job_snapshot()
            if snap and len(snap.live_results) >= 2:
                self.manager.request_stop()
                break
            time.sleep(0.02)

        # Chờ job chuyển sang STOPPED
        while time.time() - start_t < timeout:
            snap = self.manager.get_active_job_snapshot()
            if snap and snap.status == JobStatus.STOPPED:
                break
            time.sleep(0.02)

        snap = self.manager.get_active_job_snapshot()
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.status, JobStatus.STOPPED)
        first_count = len(snap.live_results)

        # Bấm tiếp tục quét
        succ, msg = self.manager.resume_active_job()
        self.assertTrue(succ)

        # Chờ hoàn tất toàn bộ 6 bài
        while time.time() - start_t < timeout:
            snap = self.manager.get_active_job_snapshot()
            if snap and snap.status == JobStatus.COMPLETED:
                break
            time.sleep(0.02)

        final_snap = self.manager.get_active_job_snapshot()
        self.assertIsNotNone(final_snap)
        assert final_snap is not None
        self.assertEqual(final_snap.status, JobStatus.COMPLETED)
        self.assertEqual(len(final_snap.live_results), 6)

    def test_stt_numbering_starts_at_one(self):
        """Kiểm tra số thứ tự STT luôn bắt đầu từ 1 chứ không phải 0."""
        from backend.services.batch_job_manager import format_detection_row

        res = DetectionResult(
            track_id="t1",
            track_title="Song 1",
            is_ai=False,
            verdict_label="HUMAN",
            confidence_score=0.9,
            suspected_engine="None",
            reasons=["Natural"],
            detailed_explanation="Detail",
        )
        # Test format_detection_row với STT
        row = format_detection_row(res, stt=1)
        self.assertEqual(row.get("STT"), 1)

        # Chạy một job nhỏ và kiểm tra snapshot live_rows
        self.batch_service.delay_per_track = 0.01
        self.batch_service.fail_all = False

        succ, msg, job_id = self.manager.enqueue_job(queries=["Track 1", "Track 2"], job_name="Test STT Job")
        self.assertTrue(succ)
        timeout = 5.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            snap = self.manager.get_active_job_snapshot()
            if snap and snap.status == JobStatus.COMPLETED:
                break
            time.sleep(0.02)

        snap = self.manager.get_active_job_snapshot()
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(len(snap.live_rows), 2)
        # STT phải bắt đầu từ 1 và 2
        self.assertEqual(snap.live_rows[0].get("STT"), 1)
        self.assertEqual(snap.live_rows[1].get("STT"), 2)


if __name__ == "__main__":
    unittest.main()
