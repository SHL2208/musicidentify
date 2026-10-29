import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from backend.api.controllers.track_controller import TrackController
from backend.api.schemas.dtos import TrackDetectRequest
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.repositories.in_memory_track_repository import InMemoryTrackRepository
from backend.services.detection_service import DetectionService


class TestRescanUpdateDb(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.repo = InMemoryTrackRepository()
        self.mock_audio_engine = MagicMock()
        self.mock_audio_engine.process_track_audio.return_value = (self.tmp_path / "audio.mp3", {})
        self.mock_signal_analyzer = MagicMock()
        self.mock_signal_analyzer.analyze_audio.return_value = (
            AcousticMetrics(sample_rate=44100, duration_sec=120.0),
            self.tmp_path / "spec.png",
        )
        self.mock_signal_analyzer.check_heuristics.return_value = None
        self.mock_classifier = MagicMock()
        self.mock_resolver = MagicMock()
        self.service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.repo,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_rescan_updates_existing_record_instead_of_adding_new(self):
        """Kiểm tra khi force_rescan=True, hệ thống cập nhật bản ghi cũ thay vì thêm mới."""
        initial_res = DetectionResult(
            track_id="trk_initial123",
            track_title="Bài hát thử nghiệm",
            source_url="https://www.youtube.com/watch?v=12345678901",
            input_type=InputType.SONG_NAME,
            raw_query="Bài hát thử nghiệm",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.90,
            suspected_engine="None",
            reasons=["Âm học tự nhiên"],
            detailed_explanation="Human",
        )
        self.mock_classifier.classify_track.return_value = initial_res

        # 1. Quét lần đầu
        track1 = TrackItem(
            id="trk_initial123",
            raw_query="Bài hát thử nghiệm",
            input_type=InputType.SONG_NAME,
            resolved_url="https://www.youtube.com/watch?v=12345678901",
            track_title="Bài hát thử nghiệm",
        )
        self.service.detect_single(track1, force_rescan=False)
        self.assertEqual(len(self.repo.list_results()), 1)
        self.assertFalse(self.repo.get_result("trk_initial123").is_ai)

        # 2. Quét lại với force_rescan=True nhưng đối tượng track mới có ID khác (do YouTube resolver sinh ra yt_...)
        updated_res = DetectionResult(
            track_id="yt_12345678901",
            track_title="Bài hát thử nghiệm",
            source_url="https://www.youtube.com/watch?v=12345678901",
            input_type=InputType.URL,
            raw_query="Bài hát thử nghiệm",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI GENERATED",
            confidence_score=0.98,
            suspected_engine="Suno",
            reasons=["Dấu vết AI"],
            detailed_explanation="AI Detected",
        )
        self.mock_classifier.classify_track.return_value = updated_res
        track2 = TrackItem(
            id="yt_12345678901",
            raw_query="Bài hát thử nghiệm",
            input_type=InputType.URL,
            resolved_url="https://www.youtube.com/watch?v=12345678901",
            track_title="Bài hát thử nghiệm",
        )
        res2 = self.service.detect_single(track2, force_rescan=True)

        # Đảm bảo KHÔNG thêm bản ghi mới: Vẫn chỉ có đúng 1 bản ghi duy nhất trong DB!
        self.assertEqual(len(self.repo.list_results()), 1)
        # ID của kết quả được đồng bộ đè lên bản ghi cũ
        self.assertEqual(res2.track_id, "trk_initial123")
        self.assertTrue(self.repo.get_result("trk_initial123").is_ai)
        self.assertEqual(
            self.repo.get_result("trk_initial123").verdict_type, VerdictType.AI_GENERATED
        )

    def test_controller_detect_track_with_track_id_param(self):
        """Kiểm tra TrackController tiếp nhận track_id để quét lại và cập nhật đè chính xác."""
        controller = TrackController(
            detection_service=self.service, resolver=self.mock_resolver, track_repo=self.repo
        )
        self.mock_resolver.resolve_track.return_value = TrackItem(
            id="random_temp_id",
            raw_query="Bài hát test",
            input_type=InputType.URL,
            resolved_url="https://www.youtube.com/watch?v=99988877766",
        )
        mock_res = DetectionResult(
            track_id="random_temp_id",
            track_title="Bài hát test",
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="HUMAN",
            confidence_score=0.95,
            suspected_engine="None",
            reasons=["Âm học chuẩn"],
            detailed_explanation="Human voice detected",
        )
        self.mock_classifier.classify_track.return_value = mock_res

        resp = controller.detect_track(
            TrackDetectRequest(
                query="https://www.youtube.com/watch?v=99988877766",
                track_id="fixed_target_id",
                force_rescan=True,
            )
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.data.track_id, "fixed_target_id")
        self.assertEqual(len(self.repo.list_results()), 1)
        self.assertIsNotNone(self.repo.get_result("fixed_target_id"))

    def test_database_track_repository_rescan_updates_in_db(self):
        """Kiểm tra trực tiếp tầng CSDL DatabaseTrackRepository thực hiện UPDATE thay vì INSERT khi quét lại."""
        from contextlib import contextmanager

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from backend.core.database import Base
        from backend.core.models.db_models import DetectionResultModel, TrackModel
        from backend.repositories.database_track_repository import DatabaseTrackRepository

        test_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(test_engine)
        TestingSessionLocal = sessionmaker(bind=test_engine)

        @contextmanager
        def mock_get_db_session():
            session = TestingSessionLocal()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        with unittest.mock.patch(
            "backend.repositories.database_track_repository.get_db_session", mock_get_db_session
        ):
            repo = DatabaseTrackRepository()

            # 1. Lưu kết quả lần đầu
            res1 = DetectionResult(
                track_id="trk_db_test_01",
                track_title="Bài ban đầu",
                source_url="https://youtube.com/watch?v=first123",
                input_type=InputType.SONG_NAME,
                raw_query="Bài ban đầu",
                is_ai=False,
                verdict_type=VerdictType.CONFIRMED_HUMAN,
                verdict_label="HUMAN",
                confidence_score=0.90,
                suspected_engine="None",
                reasons=["Lý do 1"],
                detailed_explanation="Giải thích 1",
            )
            repo.save_result(res1)

            # Kiểm tra trong SQLite DB: có 1 track, 1 detection_result
            with mock_get_db_session() as s:
                tracks = s.query(TrackModel).all()
                results = s.query(DetectionResultModel).all()
                self.assertEqual(len(tracks), 1)
                self.assertEqual(len(results), 1)
                self.assertFalse(results[0].is_ai)
                self.assertEqual(tracks[0].track_title, "Bài ban đầu")

            # 2. Quét lại (force_rescan) và lưu kết quả mới đè lên cùng track_id
            res2 = DetectionResult(
                track_id="trk_db_test_01",
                track_title="Bài cập nhật sau quét lại",
                source_url="https://youtube.com/watch?v=first123",
                input_type=InputType.SONG_NAME,
                raw_query="Bài ban đầu",
                is_ai=True,
                verdict_type=VerdictType.AI_GENERATED,
                verdict_label="AI GENERATED",
                confidence_score=0.99,
                suspected_engine="Suno",
                reasons=["Dấu vết âm học AI mới"],
                detailed_explanation="AI",
            )
            repo.save_result(res2)

            # Kiểm tra trong SQLite DB: VẪN CHỈ CÓ 1 track, 1 detection_result (KHÔNG BỊ THÊM MỚI)!
            with mock_get_db_session() as s:
                tracks = s.query(TrackModel).all()
                results = s.query(DetectionResultModel).all()
                self.assertEqual(len(tracks), 1)
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].is_ai)
                self.assertEqual(results[0].verdict_type, "AI_GENERATED")
                self.assertEqual(tracks[0].track_title, "Bài cập nhật sau quét lại")


if __name__ == "__main__":
    unittest.main()
