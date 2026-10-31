import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.core.models.entities import (
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.repositories.database_track_repository import DatabaseTrackRepository
from backend.repositories.in_memory_track_repository import InMemoryTrackRepository
from backend.services.batch_service import BatchService
from backend.services.detection_service import DetectionService
from backend.services.resolver_service import generate_deterministic_track_id


class TestSmartDedupAndHumanFeedback(unittest.TestCase):
    """Bộ kiểm thử tính năng Deduplication thông minh và phản hồi đính chính từ con người."""

    def test_deterministic_track_id_youtube(self):
        """Kiểm tra sinh ID cố định chuẩn xác từ các định dạng link YouTube."""
        id1 = generate_deterministic_track_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", InputType.URL
        )
        id2 = generate_deterministic_track_id("https://youtu.be/dQw4w9WgXcQ", InputType.URL)
        id3 = generate_deterministic_track_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s", InputType.URL
        )

        self.assertEqual(id1, "yt_dQw4w9WgXcQ")
        self.assertEqual(id2, "yt_dQw4w9WgXcQ")
        self.assertEqual(id3, "yt_dQw4w9WgXcQ")

    def test_deterministic_track_id_zingmp3(self):
        """Kiểm tra sinh ID cố định từ link Zing MP3."""
        url = "https://zingmp3.vn/bai-hat/Chung-Ta-Cua-Hien-Tai-Son-Tung-M-TP/ZW7U0I0E.html"
        track_id = generate_deterministic_track_id(url, InputType.URL)
        self.assertEqual(track_id, "zing_ZW7U0I0E")

    def test_deterministic_track_id_local_file(self):
        """Kiểm tra sinh ID cố định từ mã băm nội dung file cục bộ (đổi tên file vẫn ra cùng 1 ID)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_a = tmp_path / "song1.wav"
            file_b = tmp_path / "renamed_copy.wav"

            sample_bytes = b"RIFF" + b"\x00" * 1024 + b"WAVEfmt "
            file_a.write_bytes(sample_bytes)
            file_b.write_bytes(sample_bytes)

            id_a = generate_deterministic_track_id(str(file_a), InputType.LOCAL_FILE)
            id_b = generate_deterministic_track_id(str(file_b), InputType.LOCAL_FILE)

            self.assertTrue(id_a.startswith("file_"))
            self.assertEqual(id_a, id_b)

    def test_in_memory_repository_deduplication_and_correction(self):
        """Kiểm tra tra cứu cache và đính chính phán quyết trong InMemoryTrackRepository."""
        repo = InMemoryTrackRepository()
        track = TrackItem(
            id="yt_dQw4w9WgXcQ",
            raw_query="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            input_type=InputType.URL,
            resolved_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            track_title="Never Gonna Give You Up",
        )
        repo.save_track(track)

        res = DetectionResult(
            track_id="yt_dQw4w9WgXcQ",
            track_title="Never Gonna Give You Up",
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.92,
            suspected_engine="Suno",
            reasons=["High rolloff cutoff"],
            detailed_explanation="AI artifacts detected",
        )
        repo.save_result(res)

        # 1. Tra cứu theo raw_query hoặc resolved_url
        found = repo.find_result_by_query_or_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIsNotNone(found)
        self.assertEqual(found.track_id, "yt_dQw4w9WgXcQ")

        # 2. Đính chính phán quyết của con người
        corrected = repo.correct_verdict(
            track_id="yt_dQw4w9WgXcQ",
            is_ai=False,
            verdict_type="CONFIRMED_HUMAN",
            verdict_label="CONFIRMED HUMAN (Người làm)",
            note="Rick Astley là ca sĩ thật 100%",
        )
        self.assertIsNotNone(corrected)
        self.assertFalse(corrected.is_ai)
        self.assertEqual(corrected.verdict_type, VerdictType.CONFIRMED_HUMAN)
        self.assertTrue(corrected.is_human_corrected)
        self.assertIn("Đính chính thủ công", corrected.reasons[0])

    def test_detection_service_cache_hit_and_force_rescan(self):
        """Kiểm tra DetectionService trả kết quả từ Cache và cho phép Force Re-scan."""
        mock_audio_engine = MagicMock()
        mock_signal_analyzer = MagicMock()
        mock_classifier = MagicMock()
        mock_resolver = MagicMock()
        repo = InMemoryTrackRepository()

        service = DetectionService(
            audio_engine=mock_audio_engine,
            signal_analyzer=mock_signal_analyzer,
            classifier=mock_classifier,
            resolver=mock_resolver,
            track_repo=repo,
        )

        track = TrackItem(
            id="yt_test123",
            raw_query="https://www.youtube.com/watch?v=test1234567",
            input_type=InputType.URL,
            resolved_url="https://www.youtube.com/watch?v=test1234567",
            track_title="Test Song",
        )

        mock_audio_engine.process_track_audio.return_value = (Path("temp/fake.wav"), {})
        mock_signal_analyzer.check_heuristics.return_value = None
        mock_signal_analyzer.analyze_audio.return_value = (None, None)
        initial_res = DetectionResult(
            track_id="yt_test123",
            track_title="Test Song",
            source_url=track.resolved_url,
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.95,
            suspected_engine="None",
            reasons=["Natural acoustics"],
            detailed_explanation="Human voice detected",
        )
        mock_classifier.classify_track.return_value = initial_res

        # LẦN 1: Quét lần đầu (Chưa có trong cache)
        res1 = service.detect_single(track, force_rescan=False)
        self.assertEqual(res1.track_id, "yt_test123")
        self.assertFalse(res1.is_cached)
        self.assertEqual(mock_audio_engine.process_track_audio.call_count, 1)

        # LẦN 2: Quét lại cùng bài nhưng KHÔNG force_rescan -> Trả ngay cache (Audio engine không bị gọi lại)
        res2 = service.detect_single(track, force_rescan=False)
        self.assertEqual(res2.track_id, "yt_test123")
        self.assertTrue(res2.is_cached)
        self.assertEqual(mock_audio_engine.process_track_audio.call_count, 1)

        # LẦN 3: Quét lại với force_rescan=True -> Bắt buộc chạy lại toàn bộ pipeline
        res3 = service.detect_single(track, force_rescan=True)
        self.assertEqual(res3.track_id, "yt_test123")
        self.assertEqual(mock_audio_engine.process_track_audio.call_count, 2)

    def test_batch_service_smart_scan_mode(self):
        """Kiểm tra BatchService xử lý danh sách hàng loạt ở chế độ 'smart' bỏ qua các bài đã có."""
        mock_detection_service = MagicMock()
        mock_report_service = MagicMock()
        repo = InMemoryTrackRepository()

        batch_svc = BatchService(
            detection_service=mock_detection_service,
            report_service=mock_report_service,
            track_repo=repo,
            sleep_interval=0.0,
        )

        tracks = [
            TrackItem(id="t1", raw_query="song 1", input_type=InputType.SONG_NAME),
            TrackItem(id="t2", raw_query="song 2", input_type=InputType.SONG_NAME),
        ]

        mock_detection_service.detect_single.side_effect = [
            DetectionResult(
                track_id="t1",
                track_title="song 1",
                is_ai=False,
                verdict_type=VerdictType.CONFIRMED_HUMAN,
                verdict_label="HUMAN",
                confidence_score=0.9,
                suspected_engine="None",
                reasons=[],
                detailed_explanation="",
                is_cached=True,
            ),
            DetectionResult(
                track_id="t2",
                track_title="song 2",
                is_ai=True,
                verdict_type=VerdictType.AI_GENERATED,
                verdict_label="AI",
                confidence_score=0.9,
                suspected_engine="Suno",
                reasons=[],
                detailed_explanation="",
                is_cached=False,
            ),
        ]

        report = batch_svc.process_batch(tracks, scan_mode="smart")
        self.assertEqual(report.total_tracks, 2)
        self.assertEqual(report.processed_count, 2)
        self.assertEqual(mock_detection_service.detect_single.call_count, 2)

    def test_database_track_repository_no_upload_audio_to_cloud(self):
        """Kiểm tra khi UPLOAD_AUDIO_SAMPLES_TO_CLOUD = False, không tải file WAV lên Supabase Storage và xóa file tạm."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with (
                patch(
                    "backend.repositories.database_track_repository.UPLOAD_AUDIO_SAMPLES_TO_CLOUD",
                    False,
                ),
                patch(
                    "backend.repositories.database_track_repository.CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD",
                    True,
                ),
                patch("backend.repositories.database_track_repository.TEMP_DIR", tmp_path),
            ):
                repo = DatabaseTrackRepository()
                mock_storage = MagicMock()
                repo._storage_service = mock_storage

                track_id = "yt_zero_storage_test"
                dummy_wav = tmp_path / f"{track_id}_multisegment.wav"
                dummy_wav.write_bytes(b"\x00" * 1024)

                res = DetectionResult(
                    track_id=track_id,
                    track_title="Zero Cloud Storage Song",
                    is_ai=False,
                    verdict_type=VerdictType.CONFIRMED_HUMAN,
                    verdict_label="HUMAN",
                    confidence_score=0.95,
                    suspected_engine="None",
                    reasons=["Clean"],
                    detailed_explanation="Clean",
                )

                repo.save_result(res)

                for call in mock_storage.upload_file.call_args_list:
                    self.assertNotEqual(call.kwargs.get("bucket"), "audio-samples")
                self.assertFalse(dummy_wav.exists())

    def test_rescan_updates_db_without_adding_duplicate(self):
        """Kiểm tra chức năng quét lại (force_rescan) cập nhật đè CSDL thay vì thêm bản ghi mới."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from backend.api.controllers.track_controller import TrackController
            from backend.api.schemas.dtos import TrackDetectRequest
            from backend.core.models.entities import AcousticMetrics

            repo = InMemoryTrackRepository()
            mock_audio_engine = MagicMock()
            mock_audio_engine.process_track_audio.return_value = (tmp_path / "audio.mp3", {})
            mock_signal_analyzer = MagicMock()
            mock_signal_analyzer.analyze_audio.return_value = (
                AcousticMetrics(sample_rate=44100, duration_sec=120.0),
                tmp_path / "spec.png",
            )
            mock_signal_analyzer.check_heuristics.return_value = None
            mock_classifier = MagicMock()
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
            mock_classifier.classify_track.return_value = initial_res
            mock_resolver = MagicMock()
            service = DetectionService(
                audio_engine=mock_audio_engine,
                signal_analyzer=mock_signal_analyzer,
                classifier=mock_classifier,
                resolver=mock_resolver,
                track_repo=repo,
            )

            # 1. Quét lần đầu
            track1 = TrackItem(
                id="trk_initial123",
                raw_query="Bài hát thử nghiệm",
                input_type=InputType.SONG_NAME,
                resolved_url="https://www.youtube.com/watch?v=12345678901",
                track_title="Bài hát thử nghiệm",
            )
            service.detect_single(track1, force_rescan=False)
            self.assertEqual(len(repo.list_results()), 1)
            self.assertFalse(repo.get_result("trk_initial123").is_ai)

            # 2. Quét lại với force_rescan=True nhưng đối tượng track mới có ID khác
            updated_res = DetectionResult(
                track_id="yt_12345678901",
                track_title="Bài hát thử nghiệm",
                source_url="https://www.youtube.com/watch?v=12345678901",
                input_type=InputType.SONG_NAME,
                raw_query="Bài hát thử nghiệm",
                is_ai=True,
                verdict_type=VerdictType.AI_GENERATED,
                verdict_label="AI GENERATED",
                confidence_score=0.98,
                suspected_engine="Suno",
                reasons=["Dấu vết AI"],
                detailed_explanation="AI Detected",
            )
            mock_classifier.classify_track.return_value = updated_res
            track2 = TrackItem(
                id="yt_12345678901",
                raw_query="Bài hát thử nghiệm",
                input_type=InputType.URL,
                resolved_url="https://www.youtube.com/watch?v=12345678901",
                track_title="Bài hát thử nghiệm",
            )
            res2 = service.detect_single(track2, force_rescan=True)

            # Đảm bảo KHÔNG thêm bản ghi mới (vẫn chỉ có 1 bản ghi duy nhất)
            self.assertEqual(len(repo.list_results()), 1)
            # ID của bài hát được đồng bộ cập nhật đè lên bản ghi cũ
            self.assertEqual(res2.track_id, "trk_initial123")
            self.assertTrue(repo.get_result("trk_initial123").is_ai)
            self.assertEqual(
                repo.get_result("trk_initial123").verdict_type, VerdictType.AI_GENERATED
            )

            # 3. Quét lại qua TrackController với tham số track_id
            controller = TrackController(
                detection_service=service, resolver=mock_resolver, track_repo=repo
            )
            mock_resolver.resolve_track.return_value = TrackItem(
                id="random_fresh_id",
                raw_query="Bài hát thử nghiệm",
                input_type=InputType.URL,
                resolved_url="https://www.youtube.com/watch?v=12345678901",
            )
            resp = controller.detect_track(
                TrackDetectRequest(
                    query="https://www.youtube.com/watch?v=12345678901",
                    track_id="trk_initial123",
                    force_rescan=True,
                )
            )
            self.assertTrue(resp.success)
            self.assertEqual(resp.data.track_id, "trk_initial123")
            self.assertEqual(len(repo.list_results()), 1)


if __name__ == "__main__":
    unittest.main()
