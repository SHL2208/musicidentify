"""Unit tests for fixing evaluation regressions, distributor heuristics, and model consistency.

Strictly adheres to PEP 8 and Ruff guidelines.
"""

import unittest
from unittest.mock import MagicMock

from backend.core.models.entities import InputType, TrackItem, VerdictType
from backend.services.signal_analyzer_service import SignalAnalyzerService
from tests.test_confusion_matrix import load_testcase_dataset


class TestFixEvalRegression(unittest.TestCase):
    """Test suite ensuring distributor intelligence, anti-FP safeguards, and evaluation consistency."""

    def setUp(self) -> None:
        self.mock_fakeprint = MagicMock()
        self.mock_storage = MagicMock()
        self.analyzer = SignalAnalyzerService(
            fakeprint_detector=self.mock_fakeprint,
            storage_repo=self.mock_storage,
        )

    def test_test_confusion_matrix_loads_distributor(self) -> None:
        """Kiểm tra load_testcase_dataset trong test_confusion_matrix.py nạp đúng cột distributor."""
        dataset = load_testcase_dataset(file_path="testcase.xlsx")
        self.assertGreater(len(dataset), 0)

        # Kiểm tra các mục cụ thể có distributor
        incom_items = [d for d in dataset if d.get("distributor") == "INCOM"]
        self.assertGreater(
            len(incom_items), 0, "test_confusion_matrix.py phải nạp được bài có distributor INCOM"
        )

        sony_items = [d for d in dataset if d.get("distributor") == "Sony"]
        self.assertGreater(
            len(sony_items), 0, "test_confusion_matrix.py phải nạp được bài có distributor Sony"
        )

    def test_ai_distributor_blacklist_flags_incom_and_tte(self) -> None:
        """Kiểm tra INCOM và TTE Fast Considerate được nhận diện là AI."""
        meta_incom = {
            "title": "Cỏ Dại và Hoa Dành Dành",
            "artist": "F47",
            "distributor": "INCOM",
        }
        res_incom = self.analyzer.check_heuristics(meta_incom)
        self.assertIsNotNone(res_incom)
        self.assertTrue(res_incom["is_ai"])
        self.assertEqual(res_incom["verdict_type"], VerdictType.AI_GENERATED)

        meta_tte = {
            "title": "Miền Ký Ức",
            "artist": "Anh Vũ",
            "distributor": "TTE Fast Considerate",
        }
        res_tte = self.analyzer.check_heuristics(meta_tte)
        self.assertIsNotNone(res_tte)
        self.assertTrue(res_tte["is_ai"])

    def test_mixus_is_not_hard_blacklisted(self) -> None:
        """Kiểm tra Mixus KHÔNG bị đưa vào blacklist cứng (vì Mixus có hơn 50 bài người thật)."""
        meta_mixus_human = {
            "title": "Một Bài Hát Người Thật",
            "artist": "Ca Sĩ Thật",
            "distributor": "Mixus",
        }
        res = self.analyzer.check_heuristics(meta_mixus_human)
        # Không được tự động kết luận là AI nếu không có dấu hiệu AI rõ ràng
        self.assertIsNone(res)

    def test_human_distributor_whitelist_protection(self) -> None:
        """Kiểm tra hãng đĩa chính thống (Sony, Warner Music, BMZ) không bị gắn nhãn AI qua heuristics."""
        meta_sony = {
            "title": "CHUYỆN CHÀNG CÔ ĐƠN",
            "artist": "Anh Trai Vượt Ngàn Chông Gai",
            "distributor": "Sony",
        }
        res_sony = self.analyzer.check_heuristics(meta_sony)
        self.assertIsNone(res_sony)

        meta_warner = {
            "title": "Exposing me",
            "artist": "DANNY HUNNID",
            "distributor": "Warner Music Vietnam",
        }
        res_warner = self.analyzer.check_heuristics(meta_warner)
        self.assertIsNone(res_warner)

    def test_detection_service_preserves_distributor_on_empty_metadata(self) -> None:
        """Kiểm tra DetectionService vẫn gán distributor vào metadata kể cả khi audio_engine trả về metadata rỗng."""
        from backend.services.detection_service import DetectionService

        mock_audio_engine = MagicMock()
        mock_audio_engine.process_track_audio.return_value = (None, {})

        mock_classifier = MagicMock()
        mock_resolver = MagicMock()
        mock_track_repo = MagicMock()
        mock_track_repo.find_result_by_query_or_url.return_value = None

        det_svc = DetectionService(
            audio_engine=mock_audio_engine,
            signal_analyzer=self.analyzer,
            classifier=mock_classifier,
            resolver=mock_resolver,
            track_repo=mock_track_repo,
        )

        track = TrackItem(
            id="test_1",
            raw_query="Cỏ Dại và Hoa Dành Dành",
            input_type=InputType.SONG_NAME,
            track_title="Cỏ Dại và Hoa Dành Dành",
            artist="F47",
            distributor="INCOM",
        )

        # Phải bắt được qua Heuristics Tầng 1 và không cần gọi classifier
        result = det_svc.detect_single(track)
        self.assertTrue(result.is_ai)
        self.assertEqual(result.verdict_type, VerdictType.AI_GENERATED)
        mock_classifier.classify_track.assert_not_called()


if __name__ == "__main__":
    unittest.main()
