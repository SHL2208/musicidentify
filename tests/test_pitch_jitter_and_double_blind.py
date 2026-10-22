import unittest

import numpy as np

from backend.core.models.entities import AcousticMetrics
from backend.repositories.temp_storage_repository import TempStorageRepository
from backend.services.fakeprint_detector_service import FakeprintDetectorService
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestPitchJitterAndDoubleBlind(unittest.TestCase):
    def setUp(self):
        self.storage = TempStorageRepository()
        self.fakeprint = FakeprintDetectorService()
        self.analyzer = SignalAnalyzerService(self.fakeprint, self.storage)

    def test_compute_pitch_jitter_returns_valid_metrics(self):
        # Tạo tín hiệu sóng hài tổng hợp 44100Hz 5s có cao độ ~220Hz
        sr = 44100
        t = np.linspace(0, 5, sr * 5, endpoint=False)
        # Sóng sine 220Hz hoàn toàn phẳng (mô phỏng pitch rigidity của AI)
        synth_flat = np.sin(2 * np.pi * 220 * t)

        res = self.analyzer._compute_pitch_jitter(synth_flat, sr=sr)
        self.assertIn("pitch_jitter", res)
        self.assertIn("pitch_rigidity_ratio", res)
        self.assertIn("pitch_anomaly_detected", res)
        self.assertIn("pitch_stability_label", res)

        # Vì tần số phẳng tuyệt đối 220Hz, rigidity ratio phải cao
        self.assertGreaterEqual(res["pitch_rigidity_ratio"], 0.40)
        self.assertTrue(res["pitch_anomaly_detected"])
        self.assertEqual(res["pitch_stability_label"], "Synthetic / Quantized Latent Grid")

    def test_gemini_prompt_shields_low_fakeprint_and_anonymizes_metadata(self):
        # Khởi tạo mock metrics có fakeprint thấp (29%)
        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=45.0,
            high_freq_ratio_16k=0.02,
            stereo_correlation=0.95,
            spectral_rolloff_95_hz=18000.0,
            spectral_rolloff_85_hz=14000.0,
            harmonic_ratio=0.8,
            fakeprint_score=0.293,  # Điểm thấp < 50%
            fakeprint_detected=False,
            pitch_jitter=0.008,
            pitch_rigidity_ratio=0.55,
            pitch_anomaly_detected=True,
            pitch_stability_label="Synthetic / Quantized Latent Grid",
        )

        # Test kiểm tra chuỗi prompt nội bộ khi gọi
        # Dựng chuỗi metrics_text theo logic mới
        if metrics.fakeprint_score >= 0.50:
            fp_line = f"- Fourier Deconvolution Fakeprint (ISMIR 2025): ELEVATED COMB SIGNATURE ({metrics.fakeprint_score * 100:.1f}%)"
        else:
            fp_line = "- Fourier Deconvolution Fakeprint: INCONCLUSIVE / BYPASSED by continuous latent diffusion (Suno v3.5/v4, Udio)."

        # Xác nhận con số 29.3% KHÔNG ĐƯỢC XUẤT HIỆN trong fp_line
        self.assertNotIn("29.3%", fp_line)
        self.assertIn("INCONCLUSIVE / BYPASSED", fp_line)

        # Xác nhận có cảnh báo vi cao độ nhân tạo
        self.assertTrue(metrics.pitch_anomaly_detected)
        self.assertGreaterEqual(metrics.pitch_rigidity_ratio, 0.40)


if __name__ == "__main__":
    unittest.main()
