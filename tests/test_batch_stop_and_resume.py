import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.core.container import get_container
from backend.core.models.entities import DetectionResult, InputType, TrackItem, VerdictType
from backend.services.gemini_classifier_service import GeminiClassifierService


class MockClassifier(GeminiClassifierService):
    def __init__(self):
        super().__init__(api_key="mock_key")

    def classify_track(self, *args, **kwargs):
        track_id = kwargs.get("track_id", "mock_id")
        track_title = kwargs.get("track_title", "Mock Title")
        source_url = kwargs.get("source_url")
        return DetectionResult(
            track_id=track_id,
            track_title=track_title,
            source_url=source_url,
            is_ai=False,
            verdict_type=VerdictType.CONFIRMED_HUMAN,
            verdict_label="CONFIRMED HUMAN",
            confidence_score=0.9,
            suspected_engine="None",
            reasons=["Human performance"],
            detailed_explanation="Nhạc người thật",
            acoustic_metrics=None,
            spectrogram_image_path=None,
        )


class TestBatchStopAndResume(unittest.TestCase):
    def setUp(self):
        self.container = get_container(custom_classifier=MockClassifier(), force_new=True)
        self.container.track_repository.find_result_by_query_or_url = lambda *args, **kwargs: None
        self.batch_service = self.container.batch_service

        # Mock audio engine to avoid network calls
        dummy_wav = Path(root_dir / "data" / "temp" / "dummy_test_sr.wav")
        dummy_wav.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import soundfile as sf

        sf.write(str(dummy_wav), np.zeros((44100 * 2, 2)), 44100)
        self.container.audio_engine.process_track_audio = lambda track: (
            dummy_wav,
            {"upload_date": "20240101", "title": track.track_title},
        )

        self.tracks = [
            TrackItem(
                id=f"t_{i}",
                raw_query=f"Song {i}",
                input_type=InputType.SONG_NAME,
                track_title=f"Song {i}",
            )
            for i in range(1, 7)
        ]

    def test_stop_check_triggers_early_exit(self):
        # Stop after 2 tracks
        stop_flag = False

        def on_progress(idx, total, current_track, res, err):
            nonlocal stop_flag
            if idx == 2:
                stop_flag = True

        report = self.batch_service.process_batch(
            self.tracks,
            on_progress=on_progress,
            stop_check=lambda: stop_flag,
        )

        # Should have stopped and processed exactly 2 tracks out of 6
        self.assertEqual(report.total_tracks, 6)
        self.assertEqual(len(report.results), 2)
        self.assertEqual(report.processed_count, 2)
        self.assertIsNotNone(report.report_file_path)

    def test_resume_from_start_index(self):
        # First run processes 2 tracks
        stop_flag = False

        def on_progress(idx, total, current_track, res, err):
            nonlocal stop_flag
            if idx == 2:
                stop_flag = True

        first_report = self.batch_service.process_batch(
            self.tracks,
            on_progress=on_progress,
            stop_check=lambda: stop_flag,
        )
        self.assertEqual(len(first_report.results), 2)

        # Resume from track index 2 (0-indexed, meaning 3rd track) with initial_results
        second_report = self.batch_service.process_batch(
            self.tracks,
            start_index=2,
            initial_results=first_report.results,
        )

        # All 6 tracks should now be completed
        self.assertEqual(second_report.total_tracks, 6)
        self.assertEqual(len(second_report.results), 6)
        self.assertEqual(second_report.processed_count, 6)


if __name__ == "__main__":
    unittest.main()
