"""Bộ kiểm thử cho bộ lọc Heuristic (Signal Analyzer Heuristic Calibration).

Kiểm tra:
1. Từ khóa âm nhạc tiêu chuẩn ([Instrumental], [Guitar solo], [Piano solo])
   KHÔNG ĐƯỢC PHÉP bị gán nhãn là lệnh điều khiển AI Suno/Udio (tránh False Positives).
2. Quy tắc ngày phát hành (< 2022) không được kích hoạt bừa bãi khi video nguồn
   chưa được xác minh chính chủ hoặc khớp độ tin cậy thấp (tránh False Negatives).

Tuân thủ nghiêm ngặt PEP 8 và Ruff.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestHeuristicCalibration(unittest.TestCase):
    """Kiểm thử chuẩn hóa luật Heuristics để loại bỏ FP và FN phi lý."""

    def setUp(self) -> None:
        self.mock_fakeprint = MagicMock()
        self.mock_storage = MagicMock()
        self.analyzer = SignalAnalyzerService(
            fakeprint_detector=self.mock_fakeprint,
            storage_repo=self.mock_storage,
        )

    def test_instrumental_and_standard_tags_do_not_trigger_ai(self) -> None:
        """Thẻ [Instrumental] hay [Beat] không được coi là prompt Suno/Udio."""
        # Giả lập metadata bài hát người thật có từ [Instrumental] trong mô tả hoặc tiêu đề
        metadata_instrumental = {
            "title": "Thương Nhau Đến Thế (ZoneH Remix) [Instrumental]",
            "description": "Bản phối khí nhạc cụ [Instrumental] chất lượng cao dành cho biểu diễn karaoke.",
            "uploader": "Meme Media",
            "tags": ["remix", "instrumental", "beat"],
            "upload_date": "20240315",
        }
        res = self.analyzer.check_heuristics(metadata_instrumental)
        # Kết quả mong đợi: KHÔNG ĐƯỢC trả về is_ai=True
        if res is not None:
            self.assertFalse(
                res.get("is_ai"),
                f"LỖI: Bản phối [Instrumental] của người bị nhận nhầm thành AI Suno! Lý do: {res.get('reason')}",
            )

    def test_unverified_video_cannot_trigger_pre_2022_human_heuristic(self) -> None:
        """Nếu video nguồn là kết quả tìm kiếm không được xác minh (unverified/low confidence),

        quy tắc '< 2022' không được phép kích hoạt để bỏ qua bước phân tích âm thanh.
        """
        metadata_unverified = {
            "title": "Deus Ex: Mankind Divided - SM00 Neon Nights (Stealth)",
            "uploader": "Random Gamer",
            "upload_date": "20160823",
            "is_verified_match": False,  # Cờ đánh dấu video chưa được xác thực trùng khớp
        }
        res = self.analyzer.check_heuristics(metadata_unverified)
        # Nếu chưa được xác thực, không được kết luận sớm là CONFIRMED HUMAN mà phải chạy qua audio model
        if res is not None:
            self.assertNotEqual(
                res.get("reason"),
                "Bài hát phát hành năm 2016 (trước khi các mô hình sinh nhạc AI Suno/Udio ra đời).",
                "LỖI: Video rác chưa xác thực vẫn kích hoạt luật < 2022!",
            )

    def test_actual_suno_explicit_credit_triggers_ai(self) -> None:
        """Các bài có credit rõ ràng (#suno, #udio, created with suno) vẫn phải bắt chính xác."""
        metadata_suno = {
            "title": "Tình Đơn Phương AI Cover",
            "description": "Song generated and created with Suno AI v3.5 #suno #ai",
            "uploader": "AI Music Lab",
            "upload_date": "20240501",
        }
        res = self.analyzer.check_heuristics(metadata_suno)
        self.assertIsNotNone(res)
        self.assertTrue(res["is_ai"])
        self.assertEqual(res["engine"], "Suno")


if __name__ == "__main__":
    unittest.main()
