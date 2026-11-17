"""Bộ kiểm thử cho việc xuất báo cáo Excel/CSV và hiển thị Data Table có cột Link Spotify.

Tuân thủ nghiêm ngặt PEP 8, unittest và Ruff.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.core.models.entities import DetectionResult, VerdictType, WorkType
from backend.services.batch_job_manager import format_detection_row
from backend.services.report_service import ReportService


class TestSpotifyReportExport(unittest.TestCase):
    """Kiểm thử tính năng xuất báo cáo và hiển thị hàng bảng có cột Link Spotify."""

    def test_report_service_includes_spotify_link(self) -> None:
        """Kiểm tra build_report_records luôn có cột 'Link Spotify'."""
        mock_repo = MagicMock()
        report_service = ReportService(report_repo=mock_repo)

        res = DetectionResult(
            track_id="t1",
            track_title="Test Song",
            source_url="https://youtube.com/watch?v=123",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.95,
            suspected_engine="None",
            reasons=["Natural Voice"],
            detailed_explanation="Clean biological recording",
            spotify_url="https://open.spotify.com/track/abc456",
        )

        records = report_service.build_report_records([res])
        self.assertEqual(len(records), 1)
        self.assertIn("Link Spotify", records[0])
        self.assertEqual(records[0]["Link Spotify"], "https://open.spotify.com/track/abc456")
        self.assertIn("Độ khớp Spotify (%)", records[0])

    def test_format_detection_row_includes_spotify_link(self) -> None:
        """Kiểm tra format_detection_row trong batch view có cột 'Link Spotify' và 'Độ khớp Spotify (%)'."""
        res = DetectionResult(
            track_id="t2",
            track_title="Another Song",
            source_url="https://youtube.com/watch?v=789",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED",
            confidence_score=0.99,
            suspected_engine="Suno",
            reasons=["Fakeprint peak"],
            detailed_explanation="Suno generated track",
            spotify_url="https://open.spotify.com/track/xyz789",
            spotify_metadata={"match_score": 0.96},
        )

        row = format_detection_row(res, stt=1)
        self.assertIn("Link Spotify", row)
        self.assertEqual(row["Link Spotify"], "https://open.spotify.com/track/xyz789")
        self.assertIn("Độ khớp Spotify (%)", row)
        self.assertEqual(row["Độ khớp Spotify (%)"], "96%")


if __name__ == "__main__":
    unittest.main()
