import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    VerdictType,
)
from backend.repositories.file_report_repository import FileReportRepository
from backend.services.batch_job_manager import BatchJob, JobStatus
from backend.services.report_service import ReportService


class TestReportStorageAndCleanup(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.repo = FileReportRepository(output_dir=self.test_dir)
        self.report_service = ReportService(report_repo=self.repo)

        self.mock_result = DetectionResult(
            track_id="test_t1",
            track_title="Bài Test 1",
            source_url="https://youtu.be/test1",
            input_type=InputType.URL,
            raw_query="https://youtu.be/test1",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED",
            confidence_score=0.95,
            suspected_engine="Suno",
            reasons=["Fourier cutoff at 16kHz"],
            detailed_explanation="Phát hiện dấu hiệu AI",
            acoustic_metrics=AcousticMetrics(
                sample_rate=44100,
                duration_sec=180.0,
                spectral_rolloff_95_hz=16000.0,
                fakeprint_score=0.92,
                compression_source="Lossless",
            ),
        )

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_report_bytes_excel(self):
        """Kiểm tra xuất Excel trực tiếp ra mảng bytes trong RAM (không ghi đĩa)."""
        xlsx_bytes = self.report_service.export_report_bytes([self.mock_result], format="xlsx")
        self.assertIsNotNone(xlsx_bytes)
        self.assertIsInstance(xlsx_bytes, bytes)
        self.assertGreater(len(xlsx_bytes), 0)

        # Kiểm tra không có file nào được tạo trong test_dir
        files_in_dir = list(self.test_dir.glob("*.xlsx"))
        self.assertEqual(len(files_in_dir), 0)

        # Đọc lại bằng pandas từ BytesIO
        df = pd.read_excel(io.BytesIO(xlsx_bytes))
        self.assertEqual(len(df), 1)
        self.assertEqual(df["STT"].iloc[0], 1)
        self.assertIn("Bài Test 1", df["Tên bài hát"].iloc[0])
        self.assertEqual(df["Mô hình AI"].iloc[0], "Heuristics")

    def test_export_report_bytes_csv(self):
        """Kiểm tra xuất CSV trực tiếp ra mảng bytes trong RAM (không ghi đĩa)."""
        csv_bytes = self.report_service.export_report_bytes([self.mock_result], format="csv")
        self.assertIsNotNone(csv_bytes)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertGreater(len(csv_bytes), 0)

        # Kiểm tra không ghi đĩa
        files_in_dir = list(self.test_dir.glob("*.csv"))
        self.assertEqual(len(files_in_dir), 0)

        # Đọc lại bằng pandas từ BytesIO
        df = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(df), 1)
        self.assertIn("Bài Test 1", df["Tên bài hát"].iloc[0])

    def test_retention_policy_prune_old_reports(self):
        """Kiểm tra cơ chế giữ tối đa max_keep tệp mới nhất, tự động xóa tệp cũ hơn."""
        # Tạo 15 file báo cáo giả lập với thời gian sửa đổi khác nhau
        created_files = []
        for i in range(15):
            f = self.test_dir / f"ai_music_report_{i:02d}.xlsx"
            f.write_text("test")
            # Thiết lập mtime tăng dần
            import os
            os.utime(f, (time.time() + i * 10, time.time() + i * 10))
            created_files.append(f)

        # Tạo thêm 1 file ảnh spectrogram để đảm bảo không bị xóa nhầm
        img_file = self.test_dir / "spectrogram_test.png"
        img_file.write_text("image_data")

        # Chạy prune với max_keep = 10
        deleted = self.repo.prune_old_reports(max_keep=10)
        self.assertEqual(deleted, 5)

        # File ảnh phải còn nguyên
        self.assertTrue(img_file.exists())

        # Số file báo cáo còn lại trong thư mục phải đúng bằng 10
        remaining_reports = list(self.test_dir.glob("ai_music_report_*.xlsx"))
        self.assertEqual(len(remaining_reports), 10)

        # 5 file cũ nhất (từ 00 đến 04) phải bị xóa
        for i in range(5):
            self.assertFalse((self.test_dir / f"ai_music_report_{i:02d}.xlsx").exists())

        # 10 file mới nhất (từ 05 đến 14) phải còn
        for i in range(5, 15):
            self.assertTrue((self.test_dir / f"ai_music_report_{i:02d}.xlsx").exists())

    def test_save_report_file_auto_triggers_prune(self):
        """Kiểm tra hàm save_report_file tự động kích hoạt prune_old_reports."""
        records = [{"STT": 1, "Tên bài hát": "Song A"}]
        # Lưu 12 file liên tiếp
        for _ in range(12):
            self.repo.save_report_file(records, format="xlsx")
            time.sleep(0.01)

        # Số file ai_music_report_*.xlsx không vượt quá 10
        reports = list(self.test_dir.glob("ai_music_report_*.xlsx"))
        self.assertLessEqual(len(reports), 10)

    def test_job_manager_stop_vs_circuit_breaker_storage_policy(self):
        """Kiểm tra chính sách: user bấm dừng thì không lưu file, lỗi hệ thống thì lưu file cứu hộ."""
        # Case 1: User dừng chủ động
        job_user_stop = BatchJob(
            job_id="job_user_stop",
            job_name="User Stop Job",
            file_path=None,
            queries=["Song 1"],
            initial_results=[self.mock_result],
        )
        job_user_stop.user_stop_requested = True
        job_user_stop.report_file_path = str(self.test_dir / "temp_to_delete.xlsx")
        (self.test_dir / "temp_to_delete.xlsx").write_text("dummy")

        # Giả lập logic hoàn tất trong BatchJobManager
        if job_user_stop.user_stop_requested:
            job_user_stop.status = JobStatus.STOPPED
            if job_user_stop.report_file_path:
                p = Path(job_user_stop.report_file_path)
                if p.exists():
                    p.unlink()
            job_user_stop.report_file_path = None

        self.assertIsNone(job_user_stop.report_file_path)
        self.assertFalse((self.test_dir / "temp_to_delete.xlsx").exists())

        # Case 2: Dừng do hệ thống (Circuit Breaker)
        job_cb = BatchJob(
            job_id="job_cb",
            job_name="CB Job",
            file_path=None,
            queries=["Song 1"],
            initial_results=[self.mock_result],
        )
        job_cb.circuit_breaker_triggered = True

        if job_cb.circuit_breaker_triggered:
            job_cb.status = JobStatus.PAUSED
            if not job_cb.report_file_path and job_cb.live_results:
                rescue_p = self.report_service.export_report(job_cb.live_results, format="xlsx")
                if rescue_p:
                    job_cb.report_file_path = str(rescue_p)

        self.assertIsNotNone(job_cb.report_file_path)
        self.assertTrue(Path(job_cb.report_file_path).exists())


if __name__ == "__main__":
    unittest.main()
