import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.core.models.entities import AcousticMetrics
from backend.repositories.temp_storage_repository import TempStorageRepository
from backend.services.fakeprint_detector_service import FakeprintDetectorService
from backend.services.gemini_classifier_service import GeminiClassifierService
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestSolutions123(unittest.TestCase):
    def setUp(self):
        self.storage = TempStorageRepository()
        self.fakeprint = FakeprintDetectorService()
        self.analyzer = SignalAnalyzerService(self.fakeprint, self.storage)

    def test_solution1_spectrogram_has_no_ocr_leakage(self):
        """
        Giải pháp 1: Ảnh phổ không được chứa điểm % hay các nhãn kết luận thiên vị
        như 'PHỔ ĐỒ TỰ NHIÊN' hoặc 'PHÁT HIỆN HOA VĂN RĂNG LƯỢC' để chống Gemini OCR đọc trộm.
        """
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.01,
            stereo_correlation=0.9,
            spectral_rolloff_95_hz=16000.0,
            spectral_rolloff_85_hz=13000.0,
            harmonic_ratio=0.8,
            fakeprint_score=0.308,
            fakeprint_detected=False,
            pitch_jitter=0.01,
            pitch_rigidity_ratio=0.35,
            pitch_anomaly_detected=False,
            pitch_stability_label="Natural Biological",
        )

        dummy_audio = np.random.uniform(-0.1, 0.1, 44100 * 5)
        fakeprint_res = {
            "fakeprint_score": 0.308,
            "fakeprint_detected": False,
            "freqs": np.linspace(1000, 16000, 100),
            "residual_spectrum": np.random.uniform(0.1, 0.5, 100),
        }

        # Gọi hàm tạo biểu đồ phổ
        out_path = self.analyzer._generate_forensic_plot(
            audio_stem="test_audio",
            metrics=metrics,
            mono_y=dummy_audio,
            sr=44100,
            fakeprint_res=fakeprint_res,
            mono_harm=dummy_audio,
        )
        self.assertIsNotNone(out_path)
        self.assertTrue(Path(out_path).exists())

        # Kiểm tra tiêu đề Subplot 3 được gán trong hàm _generate_forensic_plot
        # Tiêu đề không được chứa "PHỔ ĐỒ TỰ NHIÊN" hay "30.8%"
        fig = plt.figure(figsize=(10, 8))
        tier3_title = self.analyzer._get_tier3_spectrogram_title(metrics)
        plt.close(fig)

        self.assertNotIn("PHỔ ĐỒ TỰ NHIÊN", tier3_title)
        self.assertNotIn("30.8%", tier3_title)
        self.assertNotIn("PHÁT HIỆN HOA VĂN RĂNG LƯỢC", tier3_title)
        self.assertIn("FOURIER", tier3_title.upper())

    def test_solution2_distinguishes_studio_autotune_from_ai_latent(self):
        """
        Giải pháp 2:
        - Giọng người thật dùng Auto-Tune (F0 phẳng nhưng Formant/khẩu hình chuyển động liên tục)
          -> is_studio_autotune = True, pitch_anomaly_detected = False
        - Giọng AI Suno/Udio (F0 phẳng và Formant bị tê liệt/đóng băng)
          -> is_studio_autotune = False, pitch_anomaly_detected = True
        """
        sr = 44100
        t = np.linspace(0, 5, sr * 5, endpoint=False)
        base_f0 = 220.0

        # 1. Giả lập AI: Sóng hài phẳng hoàn toàn, không biến thiên Formant
        ai_signal = np.sin(2 * np.pi * base_f0 * t)
        ai_res = self.analyzer._compute_pitch_jitter(ai_signal, sr=sr)

        self.assertIn("vocal_formant_variance", ai_res)
        self.assertIn("is_studio_autotune", ai_res)
        self.assertFalse(ai_res["is_studio_autotune"])
        self.assertTrue(ai_res["pitch_anomaly_detected"])
        self.assertEqual(ai_res["pitch_stability_label"], "Synthetic / Quantized Latent Grid")

        # 2. Giả lập Auto-Tune người thật: F0 phẳng nhưng có amplitude/formant modulation
        # Mô phỏng khẩu hình chuyển đổi A -> O -> I bằng cách điều chế lọc thông dải động
        mod_formant = 1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)  # Biến động 3Hz mô phỏng khẩu âm
        human_autotune_signal = np.sin(2 * np.pi * base_f0 * t) * mod_formant
        # Thêm hài âm bậc cao biến đổi
        human_autotune_signal += (
            0.3 * np.sin(2 * np.pi * 2 * base_f0 * t) * (1.0 - 0.5 * np.sin(2 * np.pi * 2.5 * t))
        )

        human_res = self.analyzer._compute_pitch_jitter(human_autotune_signal, sr=sr)
        self.assertTrue(human_res["is_studio_autotune"])
        self.assertFalse(human_res["pitch_anomaly_detected"])
        self.assertEqual(
            human_res["pitch_stability_label"], "Studio Auto-Tuned (Natural Vocal Tract)"
        )

    def test_solution3_stem_bleed_detection(self):
        """
        Giải pháp 3:
        - Tách Hài âm & Bộ gõ để phát hiện hiện tượng rò rỉ âm học (Stem Bleed)
          do mô hình sinh nguyên khối End-to-End (Suno/Udio) gây ra.
        """
        sr = 44100
        duration = 5
        t = np.linspace(0, duration, sr * duration, endpoint=False)

        # 1. Tín hiệu phòng thu sạch: Tiếng trống (Percussive) và Vocal (Harmonic) độc lập
        clean_vocal = np.sin(2 * np.pi * 300 * t)
        # Bộ gõ là các xung ngắn cách nhau 0.5s
        clean_drums = np.zeros_like(t)
        pulse_indices = np.arange(0, len(t), int(sr * 0.5))
        for idx in pulse_indices:
            clean_drums[idx : idx + 200] = np.random.uniform(-1, 1, 200)

        clean_bleed_res = self.analyzer._compute_stem_bleed(clean_vocal, clean_drums, sr=sr)
        self.assertIn("stem_bleed_score", clean_bleed_res)
        self.assertIn("stem_bleed_detected", clean_bleed_res)
        self.assertFalse(clean_bleed_res["stem_bleed_detected"])
        self.assertLess(clean_bleed_res["stem_bleed_score"], 0.40)

        # 2. Tín hiệu AI rò rỉ nặng (Intermodulation Bleed):
        # Năng lượng dải vocal bị giật và biến điệu đồng bộ theo xung trống
        drum_envelope = np.abs(clean_drums)
        leaked_vocal = clean_vocal * (1.0 + 2.0 * drum_envelope)
        ai_bleed_res = self.analyzer._compute_stem_bleed(leaked_vocal, clean_drums, sr=sr)
        self.assertTrue(ai_bleed_res["stem_bleed_detected"])
        self.assertGreaterEqual(ai_bleed_res["stem_bleed_score"], 0.40)

    def test_entities_and_gemini_prompt_incorporate_new_metrics(self):
        """
        Kiểm tra AcousticMetrics lưu đầy đủ trường mới và GeminiClassifierService đưa vào prompt.
        """
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            high_freq_ratio_16k=0.01,
            stereo_correlation=0.9,
            spectral_rolloff_95_hz=16000.0,
            spectral_rolloff_85_hz=13000.0,
            harmonic_ratio=0.8,
            fakeprint_score=0.2,
            fakeprint_detected=False,
            pitch_jitter=0.005,
            pitch_rigidity_ratio=0.48,
            pitch_anomaly_detected=False,
            pitch_stability_label="Studio Auto-Tuned (Natural Vocal Tract)",
            vocal_formant_variance=250000.0,
            is_studio_autotune=True,
            stem_bleed_score=0.62,
            stem_bleed_detected=True,
        )

        classifier = GeminiClassifierService(api_key="mock_key")
        prompt = classifier._build_forensic_prompt("Track #9999", metrics, None)

        self.assertIn("Stem Bleed", prompt)
        self.assertIn("Studio Auto-Tuned", prompt)
        self.assertIn("0.62", prompt)


if __name__ == "__main__":
    unittest.main()
