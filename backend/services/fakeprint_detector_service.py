from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, signal

from ..core.interfaces.services.fakeprint_detector_interface import IFakeprintDetector


class FakeprintDetectorService(IFakeprintDetector):
    """
    Hiện thực phát hiện dấu vết kiến trúc tầng Deconvolution / Transposed Convolution (Fakeprint)
    trong các mô hình sinh nhạc AI (Suno, Udio, Encodec, DAC...).

    Tuân thủ interface IFakeprintDetector, dựa trên nghiên cứu ISMIR 2025 (Deezer Research).
    Đã được chuẩn hóa và loại bỏ False Positives đối với bồi âm giọng hát / nhạc cụ tự nhiên của con người.
    """

    def analyze_waveform(self, y: np.ndarray, sr: int = 44100) -> dict[str, Any]:
        """
        Phân tích dạng sóng mono để trích xuất chỉ số Fakeprint Fourier.
        """
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Đảm bảo độ dài tối thiểu 5 giây để phân tích Fourier ổn định
        if len(y) < sr * 5:
            repeats = int(np.ceil((sr * 5) / len(y)))
            y = np.tile(y, repeats)

        # 1. Tính toán Phổ công suất STFT độ phân giải cao (N_FFT = 8192)
        n_fft = 8192
        hop_length = 2048

        window = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        num_frames = (len(y) - n_fft) // hop_length + 1
        if num_frames <= 0:
            num_frames = 1
            padded = np.pad(y, (0, max(0, n_fft - len(y))))
            frames = padded[:n_fft][np.newaxis, :]
        else:
            frames = np.lib.stride_tricks.as_strided(
                y, shape=(num_frames, n_fft), strides=(y.strides[0] * hop_length, y.strides[0])
            )

        windowed_frames = frames * window
        fft_frames = np.abs(np.fft.rfft(windowed_frames, axis=1)) ** 2
        avg_power = np.mean(fft_frames, axis=0) + 1e-12
        log_power_db = 10.0 * np.log10(avg_power)

        # 2. Giới hạn dải tần quan tâm theo nghiên cứu Deezer ISMIR 2025 (4.5kHz đến 16kHz)
        freq_mask = (freqs >= 4500) & (freqs <= min(16000, sr // 2))
        sub_freqs = freqs[freq_mask]
        sub_power = log_power_db[freq_mask]

        peak_total_power = float(np.max(log_power_db))
        # Nếu dải tần cao quá yếu (> 50 dB dưới đỉnh tín hiệu), đó chỉ là nhiễu số hoặc sidelobe cửa sổ
        if (
            len(sub_power) < 100
            or (np.max(sub_power) < peak_total_power - 50.0)
            or (np.mean(sub_power) < -80.0)
        ):
            return {
                "fakeprint_score": 0.0,
                "fakeprint_detected": False,
                "comb_prominence": 0.0,
                "peak_periodicity_hz": 0.0,
                "residual_spectrum": np.array([]),
                "freqs": sub_freqs,
            }

        df = sub_freqs[1] - sub_freqs[0] if len(sub_freqs) > 1 else 1.0

        # 3. Trích xuất phổ dư thừa (Artifact Fingerprint Residual)
        # Sử dụng bộ lọc hình thái học (Lower-hull + Median) để bám theo phong bì phổ tự nhiên
        kernel_size = max(5, int(1000 / df))
        if kernel_size % 2 == 0:
            kernel_size += 1

        lower_hull = ndimage.minimum_filter1d(sub_power, size=kernel_size)
        baseline = ndimage.median_filter(lower_hull, size=kernel_size)
        residual = np.maximum(0, sub_power - baseline)

        # 4. Phân tích bổ trợ Autocorrelation chuẩn hóa (Normalized Autocorrelation)
        detrended = residual - ndimage.uniform_filter1d(residual, size=kernel_size)
        std_d = np.std(detrended)
        if std_d > 1e-6:
            detrended = detrended / std_d

        autocorr = signal.correlate(detrended, detrended, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        norm_autocorr = autocorr / (autocorr[0] + 1e-12)

        lag_freqs = np.arange(len(norm_autocorr)) * df
        lag_mask = (lag_freqs >= 180) & (lag_freqs <= 2500)
        valid_autocorr = norm_autocorr[lag_mask]
        valid_lags = lag_freqs[lag_mask]

        # 5. Tìm kiếm đỉnh chu kỳ trong miền [180Hz, 2500Hz]
        # Kẹp phần tương quan không âm để tránh tính nhầm trough âm làm đội prominence ảo
        pos_autocorr = np.maximum(0.0, valid_autocorr)
        ac_peaks, ac_props = signal.find_peaks(
            pos_autocorr, height=0.08, prominence=0.05, distance=max(2, int(80 / df))
        )

        comb_prominence = 0.0
        peak_periodicity_hz = 0.0
        peak_height = 0.0

        if len(ac_peaks) > 0:
            best_idx = int(np.argmax(ac_props["prominences"]))
            comb_prominence = float(ac_props["prominences"][best_idx])
            peak_height = float(pos_autocorr[ac_peaks[best_idx]])
            peak_periodicity_hz = float(valid_lags[ac_peaks[best_idx]])

        # 6. Tính toán Fakeprint Score khách quan (Đường cong hiệu chỉnh liên tục)
        # Tương quan chu kỳ thực sự của Neural Vocoder (Suno/Udio) có cả độ cao (peak_height >= 0.20)
        # và độ nổi bật (prominence). Dao động tự nhiên của nhạc cụ/giọng hát người thường có height < 0.18.
        effective_metric = (comb_prominence * 0.6 + peak_height * 0.4) if peak_height > 0 else 0.0

        if effective_metric <= 0.12:
            # Miền âm thanh tự nhiên / không có đỉnh răng lược chu kỳ
            fakeprint_score = float(np.clip(effective_metric / 0.12 * 0.25, 0.0, 0.25))
        elif effective_metric <= 0.18:
            # Miền chuyển tiếp / nhạc điện tử, Auto-Tune phòng thu (0.25 -> 0.45)
            factor = (effective_metric - 0.12) / 0.06
            fakeprint_score = float(np.clip(0.25 + 0.20 * factor, 0.25, 0.45))
        elif effective_metric <= 0.25:
            # Miền răng lược rõ rệt của neural deconvolution (0.45 -> 0.75)
            factor = (effective_metric - 0.18) / 0.07
            fakeprint_score = float(np.clip(0.45 + 0.30 * factor, 0.45, 0.75))
        else:
            # Miền răng lược cực mạnh (> 0.25)
            factor = min(1.0, (effective_metric - 0.25) / 0.10)
            fakeprint_score = float(np.clip(0.75 + 0.25 * factor, 0.75, 1.0))

        fakeprint_detected = fakeprint_score >= 0.65

        return {
            "fakeprint_score": round(fakeprint_score, 3),
            "fakeprint_detected": fakeprint_detected,
            "comb_prominence": round(comb_prominence, 3),
            "peak_periodicity_hz": round(peak_periodicity_hz, 1),
            "residual_spectrum": residual,
            "freqs": sub_freqs,
        }
