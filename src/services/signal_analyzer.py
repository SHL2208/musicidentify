import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt

from ..models.schemas import AcousticMetrics, VerdictType
from .fakeprint_detector import FakeprintDetector

class SignalAnalyzer:
    @staticmethod
    def check_heuristics(metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra các quy tắc nhanh (Metadata, Ngày phát hành, Tags).
        Nếu thỏa mãn điều kiện chắc chắn, trả về kết quả ngay để tiết kiệm API quota.
        """
        if not metadata:
            return None

        # 1. Kiểm tra ngày phát hành (Trước tháng 11/2022 -> Chắc chắn 100% người làm)
        upload_date = metadata.get("upload_date")  # Định dạng 'YYYYMMDD'
        release_year = metadata.get("release_year")
        if upload_date and len(upload_date) >= 4:
            year = int(upload_date[:4])
            if year < 2022 or (year == 2022 and int(upload_date[4:6]) < 11):
                return {
                    "is_ai": False,
                    "verdict_type": VerdictType.CONFIRMED_HUMAN,
                    "confidence": 0.99,
                    "engine": "None",
                    "reason": f"Bài hát phát hành năm {year} (trước khi các mô hình sinh nhạc AI Suno/Udio ra đời)."
                }
        elif release_year and int(release_year) < 2022:
            return {
                "is_ai": False,
                "verdict_type": VerdictType.CONFIRMED_HUMAN,
                "confidence": 0.99,
                "engine": "None",
                "reason": f"Năm phát hành là {release_year} (trước kỷ nguyên nhạc AI thương mại)."
            }

        # 2. Kiểm tra hashtag / credit rõ ràng của AI trong mô tả
        description = (metadata.get("description") or "").lower()
        title = (metadata.get("title") or "").lower()
        uploader = (metadata.get("uploader") or "").lower()
        channel = (metadata.get("channel") or "").lower()
        tags = [str(t).lower() for t in metadata.get("tags") or []]
        combined_text = f"{title} {description} {uploader} {channel} {' '.join(tags)}"

        if "created with suno" in combined_text or "#suno" in combined_text or "suno ai" in combined_text:
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Suno",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Suno AI trong phần mô tả / tiêu đề."
            }
        if "created with udio" in combined_text or "#udio" in combined_text or "udio ai" in combined_text:
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Udio",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Udio AI trong phần mô tả / tiêu đề."
            }

        # 2b. Kiểm tra các thẻ lệnh điều khiển cấu trúc AI sinh nhạc (Suno / Udio Prompt Tags)
        # Người dùng Suno thường copy toàn bộ lời bài hát bao gồm các thẻ prompt điều khiển vào mô tả
        suno_prompt_pattern = re.compile(
            r'\[\s*(?:drums?|sub-?bass|beat drop|instrumental|guitar solo|piano solo|vocal chop|synth|fade out|drop|build-?up|tempo)[^\]]*\]',
            re.IGNORECASE
        )
        ai_prompt_tags = suno_prompt_pattern.findall(description)
        if ai_prompt_tags:
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.98,
                "engine": "Suno",
                "reason": f"Phát hiện thẻ lệnh điều khiển phối khí của AI Suno/Udio trong phần lời ({', '.join(ai_prompt_tags[:2])})."
            }

        return None

    @staticmethod
    def analyze_audio(audio_path: Path) -> Tuple[AcousticMetrics, Path]:
        """
        Tính toán các chỉ số âm học chuyên sâu & Dấu vân phổ Fourier (Fakeprint).
        Trả về: (AcousticMetrics, spectrogram_image_path)
        """
        data, sr = sf.read(str(audio_path))
        if data.ndim == 1:
            y_mono = data
            y_stereo = None
        else:
            y_stereo = data
            y_mono = np.mean(data, axis=1)

        duration = len(y_mono) / sr

        # 1. Tính toán Spectral Rolloff (85% và 95%) - Đo độ mở dải tần tự nhiên
        rolloff_95_series = librosa.feature.spectral_rolloff(y=y_mono, sr=sr, roll_percent=0.95)
        rolloff_95 = float(np.median(rolloff_95_series))

        rolloff_85_series = librosa.feature.spectral_rolloff(y=y_mono, sr=sr, roll_percent=0.85)
        rolloff_85 = float(np.median(rolloff_85_series))

        # 2. Phân tích thành phần Sóng hài (Harmonic) vs Tiếng gõ (Percussive)
        y_harm, y_perc = librosa.effects.hpss(y_mono)
        harm_energy = np.sum(y_harm ** 2)
        perc_energy = np.sum(y_perc ** 2) + 1e-10
        harmonic_ratio = float(harm_energy / (harm_energy + perc_energy))

        # 3. Tính toán tỷ lệ năng lượng >16kHz và kiểm tra nén Codec
        n_fft = 2048
        hop_length = 512
        D = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        high_freq_mask = freqs >= 16000
        high_freq_energy = np.sum(D[high_freq_mask, :] ** 2)
        total_energy = np.sum(D ** 2) + 1e-10
        high_freq_ratio = float(high_freq_energy / total_energy)

        # 4. Nhận diện dấu vết nén truyền tải Web (Opus/AAC trên YouTube)
        # Nếu rolloff 95% rơi vào khoảng 15.5kHz - 18kHz nhưng không có đỉnh deconvolution, đó là nén YouTube
        if rolloff_95 < 18000 and rolloff_85 > 8000:
            compression_source = "Opus/AAC (YouTube/Web Lossy Codec)"
            has_cutoff = True
        elif rolloff_95 >= 19000:
            compression_source = "Studio Master / High-Fidelity"
            has_cutoff = False
        else:
            compression_source = "Standard Digital Audio"
            has_cutoff = False

        # 5. Phân tích Dấu vân Fourier Deconvolution (Fakeprint Detector - ISMIR 2025)
        fakeprint_info = FakeprintDetector.analyze_waveform(y_mono, sr=sr)
        fakeprint_score = fakeprint_info["fakeprint_score"]
        # Nếu bài hát có tỷ lệ sóng hài cao (vocal autotune, synth, chords tự nhiên), yêu cầu điểm fakeprint phải cực kỳ rõ rệt
        if harmonic_ratio > 0.70:
            fakeprint_detected = fakeprint_score >= 0.88
        else:
            fakeprint_detected = fakeprint_info["fakeprint_detected"]

        # 6. Tương quan pha Stereo
        stereo_corr = 1.0
        if y_stereo is not None and y_stereo.shape[1] >= 2:
            left = y_stereo[:, 0]
            right = y_stereo[:, 1]
            if np.std(left) > 1e-6 and np.std(right) > 1e-6:
                corr = np.corrcoef(left, right)[0, 1]
                stereo_corr = float(corr)

        metrics = AcousticMetrics(
            sample_rate=sr,
            duration_sec=round(duration, 2),
            high_freq_ratio_16k=round(high_freq_ratio, 5),
            stereo_correlation=round(stereo_corr, 4),
            spectral_rolloff_95_hz=round(rolloff_95, 1),
            spectral_rolloff_85_hz=round(rolloff_85, 1),
            harmonic_ratio=round(harmonic_ratio, 3),
            fakeprint_score=round(fakeprint_score, 3),
            fakeprint_detected=fakeprint_detected,
            compression_source=compression_source,
            has_brickwall_cutoff=has_cutoff
        )

        # 7. Vẽ biểu đồ Forensic 2 tầng (Mel-Spectrogram + Phổ vân Deconvolution)
        spec_img_path = audio_path.with_suffix(".png")
        SignalAnalyzer._render_forensic_plot(
            y_mono=y_mono,
            sr=sr,
            output_path=spec_img_path,
            metrics=metrics,
            fakeprint_info=fakeprint_info
        )

        return metrics, spec_img_path

    @staticmethod
    def _render_forensic_plot(
        y_mono: np.ndarray,
        sr: int,
        output_path: Path,
        metrics: AcousticMetrics,
        fakeprint_info: Dict[str, Any]
    ):
        """Vẽ biểu đồ thẩm định đa tầng (Mel-Spectrogram + Fourier Fakeprint)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [2.2, 1.2]})

        # --- Panel 1: Mel-Spectrogram (0 - 22kHz) ---
        S = librosa.feature.melspectrogram(
            y=y_mono, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=0, fmax=sr//2
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(
            S_dB, x_axis='time', y_axis='hz', sr=sr, hop_length=512, fmin=0, fmax=sr//2, cmap='magma', ax=ax1
        )
        fig.colorbar(img, ax=ax1, format='%+2.0f dB')
        ax1.set_title(
            f"Mel-Spectrogram (Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f}Hz | Nguồn: {metrics.compression_source})",
            fontsize=11, fontweight='bold'
        )
        ax1.set_xlabel("Thời gian (giây)")
        ax1.set_ylabel("Tần số (Hz)")

        # --- Panel 2: Phổ vân Fourier Deconvolution (Deezer ISMIR 2025) ---
        sub_freqs = fakeprint_info.get("freqs", np.array([]))
        residual = fakeprint_info.get("residual_spectrum", np.array([]))

        if len(sub_freqs) > 0 and len(residual) > 0:
            ax2.plot(sub_freqs / 1000.0, residual, color='#00d2ff', linewidth=1.0, label='Đỉnh phổ thực tế (Peak Residues)')
            
            # Tô màu cảnh báo nếu có dấu vết gai nhọn chu kỳ
            status_text = f"ĐIỂM FAKEPRINT: {metrics.fakeprint_score*100:.1f}% ({'NGHI VẤN AI' if metrics.fakeprint_detected else 'CHUẨN TỰ NHIÊN'})"
            ax2.set_title(
                f"Phổ vân Deconvolution Fourier ISMIR 2025 — {status_text}",
                fontsize=10, fontweight='bold', color='#ff3366' if metrics.fakeprint_detected else '#00cc66'
            )
            ax2.set_xlabel("Tần số (kHz)")
            ax2.set_ylabel("Biên độ phần dư (dB)")
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='upper right')
        else:
            ax2.text(0.5, 0.5, "Không đủ mẫu để phân tích Fourier", horizontalalignment='center', verticalalignment='center')

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
