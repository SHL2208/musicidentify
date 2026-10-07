import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..core.models.entities import AcousticMetrics, VerdictType
from ..core.interfaces.services.signal_analyzer_interface import ISignalAnalyzer
from ..core.interfaces.services.fakeprint_detector_interface import IFakeprintDetector
from ..core.interfaces.repositories.storage_repository_interface import IStorageRepository
from ..core.exceptions import AudioProcessingException

class SignalAnalyzerService(ISignalAnalyzer):
    """
    Hiện thực phân tích âm học và biểu đồ quang phổ 2 tầng.
    Tuân thủ interface ISignalAnalyzer và nhận IFakeprintDetector qua Dependency Injection.
    """

    def __init__(self, fakeprint_detector: IFakeprintDetector, storage_repo: IStorageRepository):
        self.fakeprint_detector = fakeprint_detector
        self.storage_repo = storage_repo
        self.temp_dir = self.storage_repo.get_temp_dir()

    def check_heuristics(self, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra các quy tắc nhanh (Metadata, Ngày phát hành, Tags).
        Nếu thỏa mãn điều kiện chắc chắn, trả về kết quả ngay để tiết kiệm API quota.
        """
        if not metadata:
            return None

        # 1. Kiểm tra ngày phát hành (Trước tháng 11/2022 -> Chắc chắn 100% người làm)
        upload_date = metadata.get("upload_date")
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

        # 2. Kiểm tra hashtag / credit rõ ràng của AI trong mô tả hoặc tiêu đề
        description = (metadata.get("description") or "").lower()
        title = (metadata.get("title") or "").lower()
        uploader = (metadata.get("uploader") or "").lower()
        channel = (metadata.get("channel") or "").lower()
        tags = [str(t).lower() for t in metadata.get("tags") or []]
        combined_text = f"{title} {description} {uploader} {channel} {' '.join(tags)}"

        if "created with suno" in combined_text or "#suno" in combined_text or "suno ai" in combined_text or "suno.com" in combined_text:
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Suno",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Suno AI trong phần mô tả / tiêu đề."
            }
        if "created with udio" in combined_text or "#udio" in combined_text or "udio ai" in combined_text or "udio.com" in combined_text:
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Udio",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Udio AI trong phần mô tả / tiêu đề."
            }

        # 2b. Kiểm tra các thẻ lệnh điều khiển cấu trúc AI sinh nhạc
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

        # 2c. Kiểm tra định dạng đánh số chuỗi bài sản xuất hàng loạt bằng AI (VD: "Bài số 83/100")
        batch_ai_pattern = re.compile(r'bài\s*(?:số\s*)?\d+\s*/\s*\d+', re.IGNORECASE)
        match_batch = batch_ai_pattern.search(title) or batch_ai_pattern.search(description)
        if match_batch and ("ai" in combined_text or "suno" in combined_text or "nhạc" in combined_text):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.95,
                "engine": "Suno/AI",
                "reason": f"Phát hiện định dạng phát hành hàng loạt đặc trưng của dự án nhạc AI: '{match_batch.group()}'."
            }

        return None

    def analyze_audio(self, audio_path: Path) -> Tuple[AcousticMetrics, Optional[Path]]:
        """
        Phân tích dạng sóng âm thanh và xuất biểu đồ phổ 2 tầng:
        Tầng 1: Mel-Spectrogram toàn diện
        Tầng 2: Phổ vân Fourier Deconvolution (Deezer ISMIR 2025)
        """
        try:
            data, sr = sf.read(str(audio_path))
            if data.ndim == 1:
                mono_y = data
                y_left = data
                y_right = data
            else:
                y_left = data[:, 0]
                y_right = data[:, 1]
                mono_y = np.mean(data, axis=1)

            duration = len(mono_y) / sr

            # 1. Tính toán tương quan pha stereo L/R
            if data.ndim > 1:
                std_l = np.std(y_left)
                std_r = np.std(y_right)
                if std_l > 1e-6 and std_r > 1e-6:
                    correlation_matrix = np.corrcoef(y_left, y_right)
                    stereo_corr = float(correlation_matrix[0, 1])
                else:
                    stereo_corr = 1.0
            else:
                stereo_corr = 1.0

            # 2. Phân tích phổ tần STFT và Spectral Rolloff
            stft = np.abs(librosa.stft(mono_y, n_fft=2048, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

            rolloff_95 = float(np.mean(librosa.feature.spectral_rolloff(y=mono_y, sr=sr, roll_percent=0.95)))
            rolloff_85 = float(np.mean(librosa.feature.spectral_rolloff(y=mono_y, sr=sr, roll_percent=0.85)))

            # 3. Tỷ lệ năng lượng dải siêu cao tần > 16kHz
            high_freq_mask = freqs >= 16000
            total_power = np.sum(stft ** 2) + 1e-12
            high_freq_power = np.sum((stft[high_freq_mask, :]) ** 2)
            high_freq_ratio = float(high_freq_power / total_power)

            # 4. Phân tích thành phần sóng hài (Harmonic Ratio)
            y_harm, y_perc = librosa.effects.hpss(mono_y)
            harm_power = np.sum(y_harm ** 2) + 1e-12
            perc_power = np.sum(y_perc ** 2) + 1e-12
            harmonic_ratio = float(harm_power / (harm_power + perc_power))

            # 5. Phân tích vết vân Fourier qua Interface IFakeprintDetector
            fakeprint_res = self.fakeprint_detector.analyze_waveform(mono_y, sr=sr)
            fakeprint_score = fakeprint_res["fakeprint_score"]
            fakeprint_detected = fakeprint_res["fakeprint_detected"]

            # 6. Đánh giá nén truyền tải và vết ngắt tần
            has_cutoff = bool(rolloff_95 < 17500.0)
            if has_cutoff and not fakeprint_detected:
                compression_source = "Opus/AAC (YouTube/Web Streaming Lossy)"
            elif rolloff_95 >= 19000.0:
                compression_source = "Studio Master / Wide-Bandwidth"
            else:
                compression_source = "Standard Digital Audio (44.1kHz)"

            metrics = AcousticMetrics(
                sample_rate=sr,
                duration_sec=duration,
                high_freq_ratio_16k=round(high_freq_ratio, 4),
                stereo_correlation=round(stereo_corr, 3),
                spectral_rolloff_95_hz=round(rolloff_95, 1),
                spectral_rolloff_85_hz=round(rolloff_85, 1),
                harmonic_ratio=round(harmonic_ratio, 3),
                fakeprint_score=fakeprint_score,
                fakeprint_detected=fakeprint_detected,
                compression_source=compression_source,
                has_brickwall_cutoff=has_cutoff
            )

            # 7. Vẽ biểu đồ Forensic 2 tầng
            image_path = self._generate_forensic_plot(
                mono_y=mono_y,
                sr=sr,
                audio_stem=audio_path.stem,
                metrics=metrics,
                fakeprint_res=fakeprint_res
            )

            return metrics, image_path

        except Exception as e:
            raise AudioProcessingException(f"Lỗi khi trích xuất thông số âm học từ '{audio_path}': {e}") from e

    def _generate_forensic_plot(
        self,
        mono_y: np.ndarray,
        sr: int,
        audio_stem: str,
        metrics: AcousticMetrics,
        fakeprint_res: Dict[str, Any]
    ) -> Optional[Path]:
        """Tạo biểu đồ phổ hình 2 tầng chuyên nghiệp."""
        out_image_path = self.temp_dir / f"{audio_stem}_forensic.png"
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [2, 1.2]})
            fig.patch.set_facecolor('#0F172A')

            # TẦNG 1: Mel-Spectrogram
            S = librosa.feature.melspectrogram(y=mono_y, sr=sr, n_mels=128, fmax=sr//2)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img1 = librosa.display.specshow(
                S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr//2, ax=ax1, cmap='magma'
            )
            ax1.set_title(
                f"TẦNG 1: QUANG PHỔ MEL-SPECTROGRAM | Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz | {metrics.compression_source}",
                color='#38BDF8', fontsize=11, fontweight='bold', pad=10
            )
            ax1.set_facecolor('#0B0F19')
            ax1.tick_params(colors='#94A3B8')
            ax1.yaxis.label.set_color('#94A3B8')
            ax1.xaxis.label.set_color('#94A3B8')
            cbar1 = fig.colorbar(img1, ax=ax1, format='%+2.0f dB', pad=0.02)
            cbar1.ax.tick_params(colors='#94A3B8')

            # TẦNG 2: Phổ vân Fourier Deconvolution ISMIR 2025
            sub_freqs = fakeprint_res.get("freqs", np.array([]))
            residual = fakeprint_res.get("residual_spectrum", np.array([]))

            if len(sub_freqs) > 0 and len(residual) > 0:
                ax2.plot(sub_freqs / 1000.0, residual, color='#EC4899', linewidth=1.2, label='Fourier Comb Residue')
                ax2.set_xlim(sub_freqs[0] / 1000.0, sub_freqs[-1] / 1000.0)
                ax2.set_xlabel('Tần số (kHz)', color='#94A3B8', fontsize=10)
                ax2.set_ylabel('Cường độ vân nhọn', color='#94A3B8', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.3, color='#475569')

                verdict_fp = "⚠️ PHÁT HIỆN DẤU VÂN RĂNG LƯỢC DECONVOLUTION (SUNO/UDIO)" if metrics.fakeprint_detected else "✅ PHỔ NHẴN TỰ NHIÊN (CON NGƯỜI)"
                color_fp = '#F43F5E' if metrics.fakeprint_detected else '#10B981'
                ax2.set_title(
                    f"TẦNG 2: PHỔ VÂN FOURIER (ISMIR 2025 DEEZER) | Điểm: {metrics.fakeprint_score*100:.1f}% - {verdict_fp}",
                    color=color_fp, fontsize=10.5, fontweight='bold', pad=10
                )
            else:
                ax2.text(0.5, 0.5, "Không đủ dữ liệu Fourier để vẽ tầng 2", color='#94A3B8', ha='center', va='center')

            ax2.set_facecolor('#0B0F19')
            ax2.tick_params(colors='#94A3B8')

            plt.tight_layout()
            plt.savefig(str(out_image_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            return out_image_path

        except Exception as e:
            print(f"[SignalAnalyzerService] Không thể tạo ảnh phổ: {e}")
            plt.close('all')
            return None
