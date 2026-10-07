import numpy as np
from scipy import signal, ndimage
from typing import Dict, Any

from ..core.interfaces.services.fakeprint_detector_interface import IFakeprintDetector

class FakeprintDetectorService(IFakeprintDetector):
    """
    Hiện thực phát hiện dấu vết kiến trúc tầng Deconvolution / Transposed Convolution (Fakeprint)
    trong các mô hình sinh nhạc AI (Suno, Udio, Stable Audio...).
    
    Tuân thủ interface IFakeprintDetector, dựa trên nghiên cứu ISMIR 2025 (Deezer Research).
    """

    def analyze_waveform(self, y: np.ndarray, sr: int = 44100) -> Dict[str, Any]:
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
                y,
                shape=(num_frames, n_fft),
                strides=(y.strides[0] * hop_length, y.strides[0])
            )
        
        windowed_frames = frames * window
        fft_frames = np.abs(np.fft.rfft(windowed_frames, axis=1)) ** 2
        avg_power = np.mean(fft_frames, axis=0) + 1e-12
        log_power_db = 10.0 * np.log10(avg_power)

        # 2. Giới hạn dải tần quan tâm (5kHz đến 18kHz)
        freq_mask = (freqs >= 4000) & (freqs <= min(18000, sr // 2))
        sub_freqs = freqs[freq_mask]
        sub_power = log_power_db[freq_mask]

        if len(sub_power) < 100:
            return {
                "fakeprint_score": 0.0,
                "fakeprint_detected": False,
                "comb_prominence": 0.0,
                "peak_periodicity_hz": 0.0,
                "residual_spectrum": np.array([]),
                "freqs": sub_freqs
            }

        # 3. Baseline Subtraction (Lower-Hull / Morphological Filtering)
        kernel_size = max(5, int(len(sub_power) * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        lower_hull = ndimage.minimum_filter1d(sub_power, size=kernel_size)
        smoothed_baseline = ndimage.median_filter(lower_hull, size=kernel_size)
        
        residual = np.maximum(0, sub_power - smoothed_baseline)

        # 4. Phân tích tính chu kỳ của các đỉnh (Cepstrum / Autocorrelation of Residuals)
        norm_residual = residual - np.mean(residual)
        std_res = np.std(norm_residual)
        if std_res > 1e-6:
            norm_residual = norm_residual / std_res

        autocorr = signal.correlate(norm_residual, norm_residual, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        lags = np.arange(len(autocorr))
        df = sub_freqs[1] - sub_freqs[0] if len(sub_freqs) > 1 else 1.0
        lag_freqs = lags * df

        # Tìm kiếm đỉnh chu kỳ từ 150Hz đến 2500Hz
        lag_mask = (lag_freqs >= 150) & (lag_freqs <= 2500)
        valid_lags = lags[lag_mask]
        valid_autocorr = autocorr[lag_mask]

        comb_prominence = 0.0
        peak_period_hz = 0.0

        if len(valid_autocorr) > 5:
            peaks, properties = signal.find_peaks(
                valid_autocorr,
                height=0.15,
                distance=max(2, int(100 / df)),
                prominence=0.08
            )
            if len(peaks) > 0:
                highest_idx = peaks[np.argmax(valid_autocorr[peaks])]
                comb_prominence = float(valid_autocorr[highest_idx])
                peak_period_hz = float(valid_lags[highest_idx] * df)

        # 5. Phân tích mật độ đỉnh nhọn cục bộ (Spectral Peak Density)
        local_peaks, _ = signal.find_peaks(residual, prominence=1.2, distance=max(2, int(150 / df)))
        peak_density = len(local_peaks) / (len(residual) + 1e-6)

        # 6. Tổng hợp điểm số Fakeprint Score (0.0 đến 1.0)
        prominence_factor = min(1.0, comb_prominence / 0.55)
        density_factor = min(1.0, peak_density / 0.08)
        
        fakeprint_score = float(np.clip(
            0.65 * prominence_factor + 0.35 * density_factor,
            0.0,
            1.0
        ))

        fakeprint_detected = bool(fakeprint_score >= 0.55 or (comb_prominence >= 0.35 and len(local_peaks) >= 6))

        return {
            "fakeprint_score": round(fakeprint_score, 3),
            "fakeprint_detected": fakeprint_detected,
            "comb_prominence": round(comb_prominence, 3),
            "peak_periodicity_hz": round(peak_period_hz, 1),
            "residual_spectrum": residual,
            "freqs": sub_freqs
        }
