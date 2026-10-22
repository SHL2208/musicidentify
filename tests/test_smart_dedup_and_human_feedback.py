from pathlib import Path
from unittest.mock import MagicMock

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


def test_deterministic_track_id_youtube():
    """Kiểm tra sinh ID cố định chuẩn xác từ các định dạng link YouTube."""
    id1 = generate_deterministic_track_id(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", InputType.URL
    )
    id2 = generate_deterministic_track_id("https://youtu.be/dQw4w9WgXcQ", InputType.URL)
    id3 = generate_deterministic_track_id(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s", InputType.URL
    )

    assert id1 == "yt_dQw4w9WgXcQ"
    assert id2 == "yt_dQw4w9WgXcQ"
    assert id3 == "yt_dQw4w9WgXcQ"


def test_deterministic_track_id_zingmp3():
    """Kiểm tra sinh ID cố định từ link Zing MP3."""
    url = "https://zingmp3.vn/bai-hat/Chung-Ta-Cua-Hien-Tai-Son-Tung-M-TP/ZW7U0I0E.html"
    track_id = generate_deterministic_track_id(url, InputType.URL)
    assert track_id == "zing_ZW7U0I0E"


def test_deterministic_track_id_local_file(tmp_path: Path):
    """Kiểm tra sinh ID cố định từ mã băm nội dung file cục bộ (đổi tên file vẫn ra cùng 1 ID)."""
    file_a = tmp_path / "song1.wav"
    file_b = tmp_path / "renamed_copy.wav"

    sample_bytes = b"RIFF" + b"\x00" * 1024 + b"WAVEfmt "
    file_a.write_bytes(sample_bytes)
    file_b.write_bytes(sample_bytes)

    id_a = generate_deterministic_track_id(str(file_a), InputType.LOCAL_FILE)
    id_b = generate_deterministic_track_id(str(file_b), InputType.LOCAL_FILE)

    assert id_a.startswith("file_")
    assert id_a == id_b


def test_in_memory_repository_deduplication_and_correction():
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
    assert found is not None
    assert found.track_id == "yt_dQw4w9WgXcQ"

    # 2. Đính chính phán quyết của con người
    corrected = repo.correct_verdict(
        track_id="yt_dQw4w9WgXcQ",
        is_ai=False,
        verdict_type="CONFIRMED_HUMAN",
        verdict_label="CONFIRMED HUMAN (Người làm)",
        note="Rick Astley là ca sĩ thật 100%",
    )
    assert corrected is not None
    assert corrected.is_ai is False
    assert corrected.verdict_type == VerdictType.CONFIRMED_HUMAN
    assert corrected.is_human_corrected is True
    assert "Đính chính thủ công" in corrected.reasons[0]


def test_detection_service_cache_hit_and_force_rescan():
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

    # Setup mock cho lần chạy đầu tiên
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
    assert res1.track_id == "yt_test123"
    assert res1.is_cached is False
    assert mock_audio_engine.process_track_audio.call_count == 1

    # LẦN 2: Quét lại cùng bài nhưng KHÔNG force_rescan -> Trả ngay cache (Audio engine không bị gọi lại)
    res2 = service.detect_single(track, force_rescan=False)
    assert res2.track_id == "yt_test123"
    assert res2.is_cached is True
    # Số lần gọi audio engine vẫn là 1 (không tải lại audio)
    assert mock_audio_engine.process_track_audio.call_count == 1

    # LẦN 3: Quét lại với force_rescan=True -> Bắt buộc chạy lại toàn bộ pipeline
    res3 = service.detect_single(track, force_rescan=True)
    assert res3.track_id == "yt_test123"
    assert mock_audio_engine.process_track_audio.call_count == 2


def test_batch_service_smart_scan_mode():
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
    assert report.total_tracks == 2
    assert report.processed_count == 2
    assert mock_detection_service.detect_single.call_count == 2


def test_database_track_repository_no_upload_audio_to_cloud(tmp_path: Path, monkeypatch):
    """Kiểm tra khi UPLOAD_AUDIO_SAMPLES_TO_CLOUD = False, không tải file WAV 8MB lên Supabase Storage và xóa file tạm."""
    monkeypatch.setattr(
        "backend.repositories.database_track_repository.UPLOAD_AUDIO_SAMPLES_TO_CLOUD",
        False,
    )
    monkeypatch.setattr(
        "backend.repositories.database_track_repository.CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD",
        True,
    )

    repo = DatabaseTrackRepository()
    mock_storage = MagicMock()
    repo._storage_service = mock_storage

    track_id = "yt_zero_storage_test"
    dummy_wav = tmp_path / f"{track_id}_multisegment.wav"
    dummy_wav.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("backend.repositories.database_track_repository.TEMP_DIR", tmp_path)

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

    # Đảm bảo storage_service KHÔNG được gọi để tải audio-samples lên cloud
    for call in mock_storage.upload_file.call_args_list:
        assert call.kwargs.get("bucket") != "audio-samples"

    # Đảm bảo file WAV tạm đã được xóa sạch sau khi xử lý
    assert not dummy_wav.exists()
