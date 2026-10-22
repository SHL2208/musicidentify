import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

# Thêm root vào sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.core.config import TEMP_DIR, get_ffmpeg_binary
from backend.core.container import get_container
from backend.core.models.entities import InputType, VerdictType

_container = get_container()
InputResolver = _container.resolver
SignalAnalyzer = _container.signal_analyzer
FakeprintDetector = _container.fakeprint_detector


def test_imports_and_ffmpeg():
    print("Testing FFmpeg resolution...")
    ffmpeg_exe = get_ffmpeg_binary()
    print(f"FFmpeg binary located at: {ffmpeg_exe}")
    assert ffmpeg_exe is not None and len(ffmpeg_exe) > 0
    print("FFmpeg check: PASSED\n")


def test_resolver():
    print("Testing InputResolver...")
    url_type = InputResolver.detect_input_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert url_type == InputType.URL
    print("URL Detection: PASSED")

    song_type = InputResolver.detect_input_type("Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau")
    assert song_type == InputType.SONG_NAME
    print("Song Name Detection: PASSED\n")


def test_fakeprint_detector():
    print("Testing FakeprintDetector (Deezer ISMIR 2025)...")
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Âm thanh tự nhiên (tổng hợp một vài sóng hài mượt)
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    res_clean = FakeprintDetector.analyze_waveform(clean_audio, sr=sr)
    print(
        f"Clean Audio Fakeprint Score: {res_clean['fakeprint_score'] * 100:.1f}% (Detected: {res_clean['fakeprint_detected']})"
    )
    assert res_clean["fakeprint_detected"] is False
    print("Clean Audio Fakeprint: PASSED")

    # 2. Âm thanh giả lập vân deconvolution (Dirac comb spikes ở dải 6kHz - 14kHz)
    comb_signal = clean_audio.copy()
    for f in range(6000, 14000, 300):
        comb_signal += 0.08 * np.sin(2 * np.pi * f * t)
    res_comb = FakeprintDetector.analyze_waveform(comb_signal, sr=sr)
    print(
        f"Simulated AI Deconvolution Fakeprint Score: {res_comb['fakeprint_score'] * 100:.1f}% (Detected: {res_comb['fakeprint_detected']})"
    )
    assert res_comb["fakeprint_score"] > res_clean["fakeprint_score"]
    print("Deconvolution Fakeprint Discrimination: PASSED\n")


def test_signal_analyzer():
    print("Testing SignalAnalyzer with forensic metrics...")
    temp_wav = TEMP_DIR / "test_synth.wav"
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    signal = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 18000 * t)
    stereo = np.column_stack([signal, signal])
    sf.write(str(temp_wav), stereo, sr)

    metrics, spec_path = SignalAnalyzer.analyze_audio(temp_wav)
    print(f"Sample Rate: {metrics.sample_rate} Hz")
    print(f"Duration: {metrics.duration_sec} s")
    print(f"Spectral Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz")
    print(f"Spectral Rolloff 85%: {metrics.spectral_rolloff_85_hz:.0f} Hz")
    print(f"Harmonic Ratio: {metrics.harmonic_ratio * 100:.1f}%")
    print(f"Fakeprint Score: {metrics.fakeprint_score * 100:.1f}%")
    print(f"Compression Assessment: {metrics.compression_source}")
    print(f"Spectrogram generated at: {spec_path}")

    assert spec_path.exists()
    print("Signal Analyzer & 2-Panel Forensic Spectrogram: PASSED\n")

    # Cleanup
    temp_wav.unlink(missing_ok=True)
    spec_path.unlink(missing_ok=True)


def test_heuristics():
    print("Testing Heuristics Filter (4-tier)...")
    res_old = SignalAnalyzer.check_heuristics({"upload_date": "20200501", "title": "Old Song"})
    assert res_old is not None
    assert res_old["is_ai"] is False
    assert res_old["verdict_type"] == VerdictType.CONFIRMED_HUMAN
    print("Pre-2022 Heuristic (CONFIRMED_HUMAN): PASSED")

    res_suno = SignalAnalyzer.check_heuristics(
        {"description": "Made with #suno AI v3", "title": "Cyberpunk"}
    )
    assert res_suno is not None
    assert res_suno["is_ai"] is True
    assert res_suno["verdict_type"] == VerdictType.AI_GENERATED
    print("Suno Tag Heuristic (AI_GENERATED): PASSED")

    # Kiểm tra đảm bảo không tự động xác nhận con người chỉ vì thông tin ekip/credit (tránh trường hợp nhạc AI gắn fake credit)
    res_fake_credit = SignalAnalyzer.check_heuristics(
        {
            "upload_date": "20240101",
            "title": "Song With Fake Credits",
            "description": "Sáng tác: Nguyen Van A, Prod by: Studio B, Link: open.spotify.com/artist/xyz",
        }
    )
    assert res_fake_credit is None, "Fake human credit metadata must not bypass audio analysis!"
    print("Bypass Fake Human Credit check: PASSED (Properly delegated to audio analysis)\n")


class TestPipeline(unittest.TestCase):
    """Kiểm thử toàn diện pipeline giám định âm học (Forensic Pipeline)."""

    def test_imports_and_ffmpeg(self) -> None:
        test_imports_and_ffmpeg()

    def test_resolver(self) -> None:
        test_resolver()

    def test_fakeprint_detector(self) -> None:
        test_fakeprint_detector()

    def test_signal_analyzer(self) -> None:
        test_signal_analyzer()

    def test_heuristics(self) -> None:
        test_heuristics()


if __name__ == "__main__":
    unittest.main()
