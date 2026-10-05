import numpy as np
from scipy import signal, ndimage
from typing import Dict, Any, Tuple

class FakeprintDetector:
    """
    Phát hiện dấu vết kiến trúc tầng Deconvolution / Transposed Convolution (Fakeprint)
    trong các mô hình sinh nhạc AI (Suno, Udio, Stable Audio...).
    
    Nguyên lý khoa học:
    Dựa trên nghiên cứu tại ISMIR 2025 ('A Fourier Explanation of AI-music Artifacts' - Deezer Research).
    Mọi mạng neural decoder dùng để phóng đại biểu diễn ẩn (latent) thành sóng âm thanh
    đều dùng các tầng Deconvolution (upsampling bằng cách chèn số 0 và áp bộ lọc).
    Quá trình này tất yếu tạo ra các đỉnh phổ chu kỳ cách đều nhau (periodic spectral spikes / comb pattern)
    trong miền tần số Fourier, độc lập với việc âm thanh bị nén hay thể loại nhạc nào.
    """

    @staticmethod
    def analyze_waveform(y: np.ndarray, sr: int = 44100) -> Dict[str, Any]:
        """
        Phân tích dạng sóng mono để trích xuất chỉ số Fakeprint Fourier.
        Trả về dict:
            - fakeprint_score: float (0.0 đến 1.0)
            - fakeprint_detected: bool
            - comb_prominence: float
            - peak_periodicity_hz: float
            - residual_spectrum: np.ndarray (để vẽ đồ thị)
            - freqs: np.ndarray
        """
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Đảm bảo độ dài tối thiểu 5 giây để phân tích Fourier ổn định
        if len(y) < sr * 5:
            # Lặp lại tín hiệu nếu quá ngắn
            repeats = int(np.ceil((sr * 5) / len(y)))
            y = np.tile(y, repeats)

        # 1. Tính toán Phổ công suất STFT độ phân giải cao (N_FFT = 8192 hoặc 16384)
        n_fft = 8192
        hop_length = 2048
        
        # Cửa sổ Hanning
        window = np.hanning(n_fft)
        
        # Tính Welch-like averaged power spectrum
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        
        # Chia thành các frame và tính trung bình phổ công suất
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

        # 2. Giới hạn dải tần quan tâm (5kHz đến 18kHz - nơi các tầng deconvolution bộc lộ rõ nhất)
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
        # Loại bỏ đường bao hài âm tự nhiên của nhạc cụ, chỉ giữ lại các gai nhọn (peaks)
        kernel_size = max(5, int(len(sub_power) * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        lower_hull = ndimage.minimum_filter1d(sub_power, size=kernel_size)
        smoothed_baseline = ndimage.median_filter(lower_hull, size=kernel_size)
        
        # Phần dư đỉnh phổ (Peak residues)
        residual = np.maximum(0, sub_power - smoothed_baseline)

        # 4. Phân tích tính chu kỳ của các đỉnh (Cepstrum / Autocorrelation of Residuals)
        # Nếu là gai chu kỳ đều đặn của Deconvolution (Dirac comb), autocorrelation sẽ có đỉnh sắc nét
        norm_residual = residual - np.mean(residual)
        std_res = np.std(norm_residual)
        if std_res > 1e-6:
            norm_residual = norm_residual / std_res

        autocorr = signal.correlate(norm_residual, norm_residual, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Lấy nửa dương
        
        # Bỏ qua lag = 0 (tự tương quan tức thời)
        min_lag = int(len(sub_power) * 0.02)  # ít nhất cách một khoảng tần số
        max_lag = int(len(sub_power) * 0.40)
        
        if max_lag > min_lag:
            search_window = autocorr[min_lag:max_lag]
            peak_indices, properties = signal.find_peaks(search_window, prominence=0.15)
            
            if len(peak_indices) > 0:
                prominences = properties['prominences']
                max_prom_idx = np.argmax(prominences)
                max_prominence = float(prominences[max_prom_idx])
                best_lag = min_lag + peak_indices[max_prom_idx]
                
                # Tần số chu kỳ giữa các đỉnh (Hz)
                df = sub_freqs[1] - sub_freqs[0]
                periodicity_hz = float(best_lag * df)
            else:
                max_prominence = 0.0
                periodicity_hz = 0.0
        else:
            max_prominence = 0.0
            periodicity_hz = 0.0

        # 5. Đo tỷ lệ đỉnh nhọn so với nền (Peak-to-Average Ratio trên residual)
        q95 = np.percentile(residual, 95)
        median_res = np.median(residual) + 1e-6
        sparsity_ratio = float(q95 / median_res)

        # 6. Tính toán Fakeprint Score tổng hợp (0.0 đến 1.0)
        # Nhạc AI: có gai chu kỳ rõ rệt (prominence cao) và độ nhọn đỉnh tách biệt
        score_prom = np.clip(max_prominence / 0.8, 0.0, 1.0)
        score_sparsity = np.clip((sparsity_ratio - 2.5) / 5.0, 0.0, 1.0)
        
        raw_score = 0.65 * score_prom + 0.35 * score_sparsity
        fakeprint_score = float(np.clip(raw_score, 0.0, 1.0))
        
        # Ngưỡng phát hiện fakeprint: >= 0.60
        is_detected = fakeprint_score >= 0.60

        return {
            "fakeprint_score": round(fakeprint_score, 3),
            "fakeprint_detected": is_detected,
            "comb_prominence": round(max_prominence, 3),
            "peak_periodicity_hz": round(periodicity_hz, 1),
            "residual_spectrum": residual,
            "freqs": sub_freqs
        }
