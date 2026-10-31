"""Bộ kiểm thử đơn vị (Unit Test) cho các chỉ số đo lường hiệu năng và ma trận nhầm lẫn.

Kiểm tra tính đúng đắn của các công thức toán học Machine Learning (Accuracy, Precision, Recall,
Specificity, F1-Score), khả năng chống lỗi chia cho 0, lưu vết sự cố, và kiểm tra hàm vẽ biểu đồ /
xuất file Excel đối soát với dữ liệu giả lập (100% Offline & Mock).

Tuân thủ nghiêm ngặt chuẩn PEP 8 và Ruff.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import openpyxl
import pandas as pd

# Tái sử dụng các module đánh giá đã được chuẩn hóa tại benchmark_evaluator
from benchmark_evaluator import (
    calculate_confusion_metrics,
    export_evaluation_excel,
    load_metrics_from_audit_report,
    plot_confusion_matrix,
    run_benchmark,
)


class TestConfusionMatrixEvaluation(unittest.TestCase):
    """Kiểm thử tự động tính toán chỉ số và vẽ biểu đồ ma trận nhầm lẫn."""

    def test_perfect_accuracy(self) -> None:
        """Kiểm tra trường hợp dự đoán chính xác 100%."""
        y_true = [True, True, False, False]
        y_pred = [True, True, False, False]
        metrics = calculate_confusion_metrics(y_true, y_pred)

        self.assertEqual(metrics.tp, 2)
        self.assertEqual(metrics.tn, 2)
        self.assertEqual(metrics.fp, 0)
        self.assertEqual(metrics.fn, 0)
        self.assertAlmostEqual(metrics.accuracy, 1.0)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1_score, 1.0)

    def test_seventy_percent_accuracy_scenario(self) -> None:
        """Kiểm tra trực tiếp kịch bản người dùng phản ánh: Tỉ lệ đạt ~70%.

        Ví dụ tập mẫu 20 bài: 14 bài đúng (70%), 6 bài sai (30%).
        """
        # 10 bài AI, 10 bài Con người
        # TP = 7, FN = 3 (Đúng 7/10 bài AI)
        # TN = 7, FP = 3 (Đúng 7/10 bài Người)
        # Accuracy = (7 + 7) / 20 = 70.0%
        y_true = [True] * 10 + [False] * 10
        y_pred = [True] * 7 + [False] * 3 + [False] * 7 + [True] * 3

        metrics = calculate_confusion_metrics(y_true, y_pred)

        self.assertEqual(metrics.total_samples, 20)
        self.assertEqual(metrics.tp, 7)
        self.assertEqual(metrics.tn, 7)
        self.assertEqual(metrics.fp, 3)
        self.assertEqual(metrics.fn, 3)
        self.assertAlmostEqual(metrics.accuracy, 0.70)
        self.assertAlmostEqual(metrics.precision, 0.70)
        self.assertAlmostEqual(metrics.recall, 0.70)
        self.assertAlmostEqual(metrics.specificity, 0.70)
        self.assertAlmostEqual(metrics.f1_score, 0.70)
        self.assertAlmostEqual(metrics.error_rate, 0.30)

    def test_zero_division_safety(self) -> None:
        """Kiểm tra độ bền khi tập mẫu rỗng hoặc không có dự đoán dương tính."""
        metrics = calculate_confusion_metrics([], [])
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1_score, 0.0)

    def test_plot_confusion_matrix_generation(self) -> None:
        """Kiểm tra chức năng vẽ và lưu file ảnh Confusion Matrix."""
        # Tạo metrics mẫu 70%
        y_true = [True] * 10 + [False] * 10
        y_pred = [True] * 7 + [False] * 3 + [False] * 7 + [True] * 3
        metrics = calculate_confusion_metrics(y_true, y_pred)

        temp_img = ROOT_DIR / "data" / "outputs" / "test_cm_output.png"
        saved_path = plot_confusion_matrix(
            metrics=metrics,
            output_path=temp_img,
            title="Test Confusion Matrix (70% Accuracy Verification)",
            model_name="Unit Test Verification Engine",
        )

        self.assertTrue(saved_path.exists())
        self.assertGreater(os.path.getsize(saved_path), 1000)

        # Dọn dẹp sau khi kiểm tra
        if saved_path.exists():
            saved_path.unlink()

    def test_mock_evaluation_pipeline(self) -> None:
        """Kiểm tra chạy đánh giá benchmark với chế độ mock mode."""
        dummy_dataset = [
            {
                "media_id": f"id_{i}",
                "title": f"Song {i}",
                "artist": f"Artist {i}",
                "query": f"Song {i} Artist {i}",
                "gt_is_ai": i % 2 == 0,
                "gt_label": "AI" if i % 2 == 0 else "HUMAN",
                "row_idx": i,
            }
            for i in range(20)
        ]

        benchmark_data = run_benchmark(
            samples=dummy_dataset,
            mock_mode=True,
            mock_accuracy=0.70,
        )
        metrics = benchmark_data["metrics"]
        results = benchmark_data["results"]

        self.assertEqual(metrics["total_samples"], 20)
        self.assertEqual(len(results), 20)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_error_records_in_confusion_metrics(self) -> None:
        """Kiểm tra việc lưu vết error_records khi tính toán metrics."""
        err_recs = [
            {
                "Mã bài lỗi (Media ID)": "test_id",
                "Tên bài lỗi (Song Title)": "Error Song",
                "Ca sĩ (Artist)": "Test Artist",
                "Nhãn thực tế (Ground Truth)": "AI",
                "Nguyên nhân sự cố (Error)": "Zing VIP audio download error",
                "Bài nạp bù thay thế (Replacement)": "Backup Song",
                "Thời điểm ghi nhận": "2026-10-15 00:00:00",
            }
        ]
        metrics = calculate_confusion_metrics([True], [True], error_count=1, error_records=err_recs)
        self.assertEqual(metrics.errors, 1)
        self.assertEqual(len(metrics.error_records), 1)
        self.assertEqual(metrics.error_records[0]["Tên bài lỗi (Song Title)"], "Error Song")

    def test_export_excel_with_error_records_sheet(self) -> None:
        """Kiểm tra xuất file Excel với đầy đủ Sheet 3 'Sự cố & Nạp bù'."""
        err_recs = [
            {
                "Mã bài lỗi (Media ID)": "id_err",
                "Tên bài lỗi (Song Title)": "Err Track",
                "Ca sĩ (Artist)": "VIP Singer",
                "Nhãn thực tế (Ground Truth)": "AI",
                "Nguyên nhân sự cố (Error)": "403 Forbidden",
                "Bài nạp bù thay thế (Replacement)": "Backup Track",
                "Thời điểm ghi nhận": "2026-10-15 00:00:00",
            }
        ]
        metrics = calculate_confusion_metrics([True], [True], error_count=1, error_records=err_recs)
        results = [
            {
                "media_id": "id_1",
                "song_title": "Track 1",
                "artist": "Artist 1",
                "query": "Track 1 Artist 1",
                "ground_truth": "AI",
                "prediction": "AI",
                "is_correct": "ĐÚNG",
                "confidence_score": "95%",
                "reasons": "AI synthetic artifacts",
            }
        ]
        temp_excel = ROOT_DIR / "data" / "outputs" / "test_error_export.xlsx"
        try:
            exported_path = export_evaluation_excel(metrics, results, output_path=temp_excel)
            self.assertTrue(Path(exported_path).exists())

            # Kiểm tra sheet names
            wb = openpyxl.load_workbook(str(exported_path), read_only=True)
            self.assertIn("Audit Details", wb.sheetnames)
            self.assertIn("Executive Summary", wb.sheetnames)
            self.assertIn("Sự cố & Nạp bù", wb.sheetnames)
            wb.close()
        finally:
            if temp_excel.exists():
                temp_excel.unlink()

    def test_load_metrics_from_audit_report_with_work_type_columns(self) -> None:
        """Kiểm tra đọc file audit có cả cột ground_truth và gt_work_type không bị nhận diện sai nhãn."""
        records = [
            {
                "media_id": "1",
                "ground_truth": "AI",
                "gt_work_type": "ORIGINAL",
                "pred_work_type": "ORIGINAL",
                "prediction": "AI",
            },
            {
                "media_id": "2",
                "ground_truth": "HUMAN",
                "gt_work_type": "COVER",
                "pred_work_type": "COVER",
                "prediction": "HUMAN",
            },
            {
                "media_id": "3",
                "ground_truth": "AI",
                "gt_work_type": "REMIX",
                "pred_work_type": "REMIX",
                "prediction": "HUMAN",
            },
            {
                "media_id": "4",
                "ground_truth": "HUMAN",
                "gt_work_type": "ORIGINAL",
                "pred_work_type": "ORIGINAL",
                "prediction": "AI",
            },
        ]
        temp_file = ROOT_DIR / "data" / "outputs" / "temp_test_audit.xlsx"
        try:
            with pd.ExcelWriter(temp_file, engine="openpyxl") as writer:
                pd.DataFrame(records).to_excel(writer, sheet_name="Audit Details", index=False)

            metrics, _ = load_metrics_from_audit_report(temp_file)
            self.assertEqual(metrics.total_samples, 4)
            self.assertEqual(metrics.tp, 1)  # id 1
            self.assertEqual(metrics.tn, 1)  # id 2
            self.assertEqual(metrics.fn, 1)  # id 3
            self.assertEqual(metrics.fp, 1)  # id 4
            self.assertAlmostEqual(metrics.accuracy, 0.5)
        finally:
            if temp_file.exists():
                temp_file.unlink()


if __name__ == "__main__":
    unittest.main()
