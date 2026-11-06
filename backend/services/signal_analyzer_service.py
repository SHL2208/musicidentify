import re
from pathlib import Path
from typing import Any

import librosa
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.config import (
    AI_CONTEXT_PATTERN,
    BATCH_AI_KEYWORDS_PATTERN,
    BATCH_AI_NUMBER_PATTERN,
    COMPRESSION_LOSSY_ROLLOFF_MAX,
    COMPRESSION_STUDIO_ROLLOFF_MIN,
    DERIVATIVE_KEYWORDS,
    GENERIC_ANONYMOUS_ARTISTS,
    INSTRUMENTAL_VOCAL_ENERGY_MAX,
    MODERN_AI_FAKEPRINT_MIN,
    MODERN_AI_HARMONIC_MIN,
    MODERN_AI_RIGIDITY_MIN,
    MODERN_AI_ROLLOFF_MAX,
    MODERN_AI_STEM_BLEED_MIN,
    MODERN_AI_ULTRASONIC_RIGIDITY_MIN,
    PITCH_ANOMALY_JITTER_THRESHOLD,
    PITCH_ANOMALY_RIGIDITY_SECONDARY,
    PITCH_ANOMALY_RIGIDITY_THRESHOLD,
    PITCH_FMAX,
    PITCH_FMIN,
    PITCH_FRAME_LENGTH,
    PITCH_HOP_LENGTH,
    PITCH_MILD_AUTOTUNE_RIGIDITY,
    PITCH_MIN_VOICED_FRAMES,
    PITCH_RIGID_DIFF_THRESHOLD,
    PITCH_STUDIO_AUTOTUNE_FORMANT_MIN,
    PITCH_STUDIO_AUTOTUNE_JITTER_MAX,
    PITCH_STUDIO_AUTOTUNE_RIGIDITY,
    PITCH_TARGET_SR,
    PITCH_VOICED_F0_MAX,
    PITCH_VOICED_F0_MIN,
    PITCH_WINDOW_SEC,
    PRE_AI_CUTOFF_MONTH,
    PRE_AI_CUTOFF_YEAR,
    PRE_AI_MATCH_SCORE_MIN,
    PRE_AI_RECENT_TARGET_YEAR,
    SPECTRAL_ROLLOFF_PERCENT_HIGH,
    SPECTRAL_ROLLOFF_PERCENT_MID,
    SPECTRAL_STFT_HOP_LENGTH,
    SPECTRAL_STFT_N_FFT,
    SPECTROGRAM_DPI,
    STEM_BLEED_HOP_LENGTH,
    STEM_BLEED_THRESHOLD,
    STEM_BLEED_WINDOW_SEC,
    SUNO_EXPLICIT_KEYWORDS,
    SUNO_PROMPT_TAGS_PATTERN,
    UDIO_EXPLICIT_KEYWORDS,
    ULTRASONIC_HIGH_FREQ_RATIO_MAX,
    ULTRASONIC_ROLLOFF_CUTOFF_HZ,
    ULTRASONIC_SHELF_DROP_DB_THRESHOLD,
    VOCAL_BAND_FREQ_MAX,
    VOCAL_BAND_FREQ_MIN,
)
from ..core.exceptions import AudioProcessingException
from ..core.interfaces.repositories.catalog_repository_interface import (
    ICatalogRepository,
)
from ..core.interfaces.repositories.storage_repository_interface import (
    IStorageRepository,
)
from ..core.interfaces.services.fakeprint_detector_interface import IFakeprintDetector
from ..core.interfaces.services.signal_analyzer_interface import ISignalAnalyzer
from ..core.models.entities import AcousticMetrics, VerdictType
from ..core.utils import normalize_vietnamese_text
from ..repositories.database_catalog_repository import DatabaseCatalogRepository

# Singleton Catalog Repository phục vụ tra cứu động từ Database (kèm cache)
_default_catalog_repo: ICatalogRepository | None = None


def _get_catalog_repo() -> ICatalogRepository:
    """Khởi tạo và trả về singleton Catalog Repository."""
    global _default_catalog_repo
    if _default_catalog_repo is None:
        _default_catalog_repo = DatabaseCatalogRepository()
    return _default_catalog_repo


def remove_vietnamese_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối sánh mềm tên nghệ sĩ."""
    return normalize_vietnamese_text(text).lower()


def is_verified_human_artist(artist_str: str, title_str: str = "") -> bool:
    """Xác thực xem nghệ sĩ hoặc tiêu đề có thuộc danh sách nghệ sĩ người thật từ CSDL động không."""
    return _get_catalog_repo().is_verified_human_artist(artist_str, title_str)


def get_dynamic_catalog() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Lấy danh mục nhà phân phối AI, Whitelist và nghệ sĩ động từ CSDL (kèm cache 5 phút)."""
    cat = _get_catalog_repo().get_catalog_data()
    return (
        cat.get("whitelist", []),
        cat.get("ai_distributors", []),
        cat.get("ai_artists", []),
    )


# Biến tương thích ngược cho các unit test cũ
_DEFAULT_HUMAN_WHITELIST = ["sony", "bmz", "vieent", "believe", "loops"]
_DEFAULT_PURE_AI_DISTRIBUTORS: list[str] = []
_DEFAULT_AI_ARTIST_REGEXES: list[tuple[str, str]] = []
_DEFAULT_SEED_HUMAN_ARTISTS: list[str] = []
_GENERIC_ANONYMOUS_ARTISTS = GENERIC_ANONYMOUS_ARTISTS


class SignalAnalyzerService(ISignalAnalyzer):
    """
    Hiện thực phân tích âm học và biểu đồ quang phổ 2 tầng.
    Tuân thủ interface ISignalAnalyzer và nhận IFakeprintDetector, ICatalogRepository qua DI.
    """

    def __init__(
        self,
        fakeprint_detector: IFakeprintDetector,
        storage_repo: IStorageRepository,
        catalog_repo: ICatalogRepository | None = None,
    ):
        self.fakeprint_detector = fakeprint_detector
        self.storage_repo = storage_repo
        self.catalog_repo = catalog_repo or _get_catalog_repo()
        self.temp_dir = self.storage_repo.get_temp_dir()

    def check_heuristics(self, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """
        Kiểm tra các quy tắc nhanh (Metadata, Ngày phát hành, Tags).
        Nếu thỏa mãn điều kiện chắc chắn, trả về kết quả ngay để tiết kiệm API quota.
        """
        if not metadata:
            return None

        # 1. Kiểm tra ngày phát hành (Trước tháng 11/2022 -> Chắc chắn 100% người làm)
        # ĐIỀU KIỆN TIÊN QUYẾT BẢO VỆ CHỐNG NHẬN DIỆN SAI BẢN COVER & NGUỒN FALLBACK:
        # a) Nguồn âm thanh PHẢI được xác thực chính xác với bài hát & nghệ sĩ mục tiêu (is_verified_match is True)
        # b) Không phải nguồn fallback âm thanh unverified (is_unverified_fallback is not True)
        # c) Điểm khớp nguồn phải cao (match_score >= 0.75)
        # d) TUYỆT ĐỐI KHÔNG ÁP DỤNG cho các bài hát COVER, REMAKE, REMIX, hoặc phiên bản phái sinh
        # e) TUYỆT ĐỐI KHÔNG ÁP DỤNG nếu thông tin mục tiêu có năm phát hành thực tế >= 2023
        is_verified = metadata.get("is_verified_match") is not False
        is_fallback = metadata.get("is_unverified_fallback") is True
        match_score = float(metadata.get("match_score", 1.0) or 0.0)

        # Phát hiện bài hát Cover/Remake/Phái sinh từ config
        song_type = str(metadata.get("song_type") or "").lower()
        title_text = f"{metadata.get('title', '')} {metadata.get('target_title', '')}".lower()
        query_text = str(metadata.get("raw_query") or "").lower()
        is_cover_or_derivative = any(
            k in song_type or k in title_text or k in query_text for k in DERIVATIVE_KEYWORDS
        )

        target_rel_date = str(metadata.get("target_release_date") or "")
        is_recent_target = False
        if target_rel_date and len(target_rel_date) >= 4:
            try:
                target_year = int(target_rel_date[:4])
                if target_year >= PRE_AI_RECENT_TARGET_YEAR:
                    is_recent_target = True
            except ValueError:
                pass

        if (
            is_verified
            and not is_fallback
            and match_score >= PRE_AI_MATCH_SCORE_MIN
            and not is_cover_or_derivative
            and not is_recent_target
        ):
            upload_date = metadata.get("upload_date")
            release_year = metadata.get("release_year")
            if upload_date and len(upload_date) >= 4:
                year = int(upload_date[:4])
                if year < PRE_AI_CUTOFF_YEAR or (
                    year == PRE_AI_CUTOFF_YEAR and int(upload_date[4:6]) < PRE_AI_CUTOFF_MONTH
                ):
                    return {
                        "is_ai": False,
                        "verdict_type": VerdictType.CONFIRMED_HUMAN,
                        "confidence": 0.99,
                        "engine": "None",
                        "reason": f"Bản phát hành chính thức năm {year} (trước khi các mô hình sinh nhạc AI Suno/Udio ra đời).",
                    }
            elif release_year and int(release_year) < PRE_AI_CUTOFF_YEAR:
                return {
                    "is_ai": False,
                    "verdict_type": VerdictType.CONFIRMED_HUMAN,
                    "confidence": 0.99,
                    "engine": "None",
                    "reason": f"Bản phát hành chính thức năm {release_year} (trước kỷ nguyên nhạc AI thương mại).",
                }

        # 2. Kiểm tra hashtag / credit rõ ràng của AI trong mô tả hoặc tiêu đề
        description = (metadata.get("description") or "").lower()
        title = (metadata.get("title") or "").lower()
        uploader = (metadata.get("uploader") or "").lower()
        channel = (metadata.get("channel") or "").lower()
        tags = [str(t).lower() for t in metadata.get("tags") or []]
        combined_text = f"{title} {description} {uploader} {channel} {' '.join(tags)}"

        if any(k in combined_text for k in SUNO_EXPLICIT_KEYWORDS):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Suno",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Suno AI trong phần mô tả / tiêu đề.",
            }
        if any(k in combined_text for k in UDIO_EXPLICIT_KEYWORDS):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Udio",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Udio AI trong phần mô tả / tiêu đề.",
            }

        # 2a. Kiểm tra nhãn ca sĩ ảo / giọng hát AI rõ ràng trong tiêu đề hoặc mô tả
        AI_SINGER_REGEX = (
            r"\b(ca\s*sĩ\s*a\.?i|ca\s*sĩ\s*ảo|ai\s*singer|virtual\s*singer|"
            r"giọng\s*hát\s*a\.?i|vocal\s*a\.?i)\b"
        )
        ai_singer_match = re.search(AI_SINGER_REGEX, combined_text, re.IGNORECASE)
        if ai_singer_match:
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.98,
                "engine": "AI Virtual Singer",
                "reason": (
                    f"Phát hiện nhãn xác nhận ca sĩ ảo / giọng hát AI trong phần tiêu đề hoặc mô tả "
                    f"('{ai_singer_match.group()}')."
                ),
            }

        # 2b. Kiểm tra các thẻ lệnh điều khiển cấu trúc AI sinh nhạc
        if re.search(AI_CONTEXT_PATTERN, combined_text, re.IGNORECASE):
            ai_prompt_tags = re.findall(SUNO_PROMPT_TAGS_PATTERN, description, re.IGNORECASE)
            if ai_prompt_tags:
                return {
                    "is_ai": True,
                    "verdict_type": VerdictType.AI_GENERATED,
                    "confidence": 0.98,
                    "engine": "Suno",
                    "reason": f"Phát hiện thẻ lệnh điều khiển cấu trúc của AI Suno/Udio trong phần lời ({', '.join(ai_prompt_tags[:2])}).",
                }

        # 2c. Kiểm tra định dạng đánh số chuỗi bài sản xuất hàng loạt bằng AI (VD: "Bài số 83/100")
        match_batch = re.search(BATCH_AI_NUMBER_PATTERN, title, re.IGNORECASE) or re.search(
            BATCH_AI_NUMBER_PATTERN, description, re.IGNORECASE
        )
        if match_batch and re.search(BATCH_AI_KEYWORDS_PATTERN, combined_text, re.IGNORECASE):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.95,
                "engine": "Suno/AI",
                "reason": f"Phát hiện định dạng phát hành hàng loạt đặc trưng của dự án nhạc AI: '{match_batch.group()}'.",
            }

        artist_lower = str(
            metadata.get("target_artist")
            or metadata.get("artist")
            or metadata.get("uploader")
            or ""
        ).lower()

        # 3. Các dự án nghệ sĩ ảo / AI sinh hàng loạt:
        # Không short-circuit trả về AI ngay tại đây để tránh False Positive (vu oan cho bài mộc).
        # Thay vào đó, gắn cờ nghi vấn vào metadata để Trọng tài Hậu kiểm đối soát âm học vật lý.
        catalog_data = self.catalog_repo.get_catalog_data()
        ai_artist_regexes = catalog_data.get("ai_artists") or []

        for pat, proj_name in ai_artist_regexes:
            try:
                if re.search(pat, artist_lower):
                    metadata["suspected_ai_project"] = proj_name
                    break
            except Exception:
                continue

        return None

    def is_verified_human_artist(self, artist_str: str, title_str: str = "") -> bool:
        """Xác thực nghệ sĩ người thật (ủy quyền cho Catalog Repository)."""
        return self.catalog_repo.is_verified_human_artist(artist_str, title_str)

    def _is_modern_ai_vocal(
        self,
        rolloff_95: float,
        rigidity: float,
        harmonic_ratio: float,
        fakeprint_score: float,
        stem_bleed: float,
        ultrasonic_cutoff: bool = False,
    ) -> bool:
        """Đánh giá nghi vấn giọng hát AI đời mới (Suno v3.5/v4, Udio) qua đa chỉ số âm học vật lý."""
        # 1. AI cổ điển / tạo sinh chung: Rolloff sụt giảm + Rigidity cao + (Fakeprint hoặc Bleed trung bình)
        legacy_suspect = (
            rolloff_95 < MODERN_AI_ROLLOFF_MAX
            and rigidity >= MODERN_AI_RIGIDITY_MIN
            and harmonic_ratio >= MODERN_AI_HARMONIC_MIN
            and (
                fakeprint_score >= MODERN_AI_FAKEPRINT_MIN or stem_bleed >= MODERN_AI_STEM_BLEED_MIN
            )
        ) or (ultrasonic_cutoff and rigidity >= MODERN_AI_ULTRASONIC_RIGIDITY_MIN)

        # 2. Suno v3.5/v4 Dân ca, Ballad, Nhạc Xuân tiếng Việt:
        # Đặc trưng: Rolloff lossy streaming <= 12500Hz, Rigidity >= 0.48, Harmonic >= 0.82, Fakeprint >= 0.20
        suno_v35_ballad_suspect = (
            rolloff_95 <= 12500.0
            and rigidity >= 0.48
            and harmonic_ratio >= 0.82
            and fakeprint_score >= 0.20
        )

        return bool(legacy_suspect or suno_v35_ballad_suspect)

    def _compute_pitch_jitter(self, mono_harm: np.ndarray, sr: int = 44100) -> dict[str, Any]:
        """
        Trích xuất chỉ số vi biến thiên cao độ giọng hát (Pitch Micro-Jitter) và độ cứng nhắc phẳng (Rigidity).
        Phát hiện dấu hiệu giọng hát sinh bởi AI (Suno/Udio) do thiếu micro-tremors cơ học của dây thanh đới.
        """
        try:
            target_sr = PITCH_TARGET_SR
            total_len = len(mono_harm)
            window_len = int(sr * PITCH_WINDOW_SEC)
            if total_len > window_len:
                start = (total_len - window_len) // 2
                raw_seg = mono_harm[start : start + window_len]
            else:
                raw_seg = mono_harm

            if sr != target_sr:
                y_vocal = librosa.resample(raw_seg, orig_sr=sr, target_sr=target_sr)
            else:
                y_vocal = raw_seg

            f0 = librosa.yin(
                y_vocal,
                fmin=PITCH_FMIN,
                fmax=PITCH_FMAX,
                sr=target_sr,
                frame_length=PITCH_FRAME_LENGTH,
                hop_length=PITCH_HOP_LENGTH,
            )
            voiced_mask = (f0 >= PITCH_VOICED_F0_MIN) & (f0 <= PITCH_VOICED_F0_MAX)
            voiced_f0 = f0[voiced_mask]

            if len(voiced_f0) < PITCH_MIN_VOICED_FRAMES:
                return {
                    "pitch_jitter": 0.0,
                    "pitch_rigidity_ratio": 0.0,
                    "pitch_anomaly_detected": False,
                    "pitch_stability_label": "Inconclusive / Instrumental",
                    "vocal_formant_variance": 0.0,
                    "is_studio_autotune": False,
                }

            # Phân tích biến thiên Formant trong dải thanh âm
            centroids = librosa.feature.spectral_centroid(
                y=y_vocal,
                sr=target_sr,
                n_fft=PITCH_FRAME_LENGTH,
                hop_length=PITCH_HOP_LENGTH,
            )
            formant_var = float(np.var(centroids)) if centroids.shape[1] > 1 else 0.0

            diffs = np.abs(np.diff(voiced_f0))
            mean_f0 = float(np.mean(voiced_f0))
            jitter = float(np.mean(diffs) / (mean_f0 + 1e-6))
            rigid_frames = np.sum(diffs < PITCH_RIGID_DIFF_THRESHOLD)
            rigidity_ratio = float(rigid_frames / (len(diffs) + 1e-6))

            is_studio_autotune = (
                rigidity_ratio >= PITCH_STUDIO_AUTOTUNE_RIGIDITY
                or jitter < PITCH_STUDIO_AUTOTUNE_JITTER_MAX
            ) and formant_var >= PITCH_STUDIO_AUTOTUNE_FORMANT_MIN

            is_anomaly = (
                rigidity_ratio >= PITCH_ANOMALY_RIGIDITY_THRESHOLD
                or (
                    jitter < PITCH_ANOMALY_JITTER_THRESHOLD
                    and rigidity_ratio > PITCH_ANOMALY_RIGIDITY_SECONDARY
                )
            ) and not is_studio_autotune

            if is_anomaly:
                label = "Synthetic / Quantized Latent Grid"
            elif is_studio_autotune:
                label = "Studio Auto-Tuned (Natural Vocal Tract)"
            elif rigidity_ratio > PITCH_MILD_AUTOTUNE_RIGIDITY:
                label = "Studio Auto-Tuned"
            else:
                label = "Natural Biological"

            return {
                "pitch_jitter": round(jitter, 4),
                "pitch_rigidity_ratio": round(rigidity_ratio, 3),
                "pitch_anomaly_detected": is_anomaly,
                "pitch_stability_label": label,
                "vocal_formant_variance": round(formant_var, 1),
                "is_studio_autotune": is_studio_autotune,
            }
        except Exception:  # noqa: BLE001
            return {
                "pitch_jitter": 0.0,
                "pitch_rigidity_ratio": 0.0,
                "pitch_anomaly_detected": False,
                "pitch_stability_label": "Error Calculating",
                "vocal_formant_variance": 0.0,
                "is_studio_autotune": False,
            }

    def _compute_stem_bleed(
        self, mono_harm: np.ndarray, mono_perc: np.ndarray, sr: int
    ) -> dict[str, Any]:
        """
        Phát hiện hiện tượng rò rỉ âm học đa kênh (Stem Bleed / Intermodulation Distortion)
        đặc trưng của mô hình sinh nhạc nguyên khối End-to-End Latent Diffusion (Suno/Udio).
        """
        try:
            if mono_harm is None or mono_perc is None or len(mono_harm) == 0 or len(mono_perc) == 0:
                return {"stem_bleed_score": 0.0, "stem_bleed_detected": False}

            max_len = int(sr * STEM_BLEED_WINDOW_SEC)
            if len(mono_harm) > max_len:
                start = (len(mono_harm) - max_len) // 2
                h_seg = mono_harm[start : start + max_len]
                p_seg = mono_perc[start : start + max_len]
            else:
                h_seg = mono_harm
                p_seg = mono_perc

            rms_perc = librosa.feature.rms(y=p_seg, hop_length=STEM_BLEED_HOP_LENGTH)[0]
            rms_harm = librosa.feature.rms(y=h_seg, hop_length=STEM_BLEED_HOP_LENGTH)[0]

            std_p = float(np.std(rms_perc))
            std_h = float(np.std(rms_harm))
            if std_p < 1e-6 or std_h < 1e-6:
                return {"stem_bleed_score": 0.0, "stem_bleed_detected": False}

            corr_matrix = np.corrcoef(rms_perc, rms_harm)
            corr = float(corr_matrix[0, 1])
            bleed_score = float(max(0.0, corr)) if not np.isnan(corr) else 0.0

            is_bleed = bleed_score >= STEM_BLEED_THRESHOLD
            return {
                "stem_bleed_score": round(bleed_score, 3),
                "stem_bleed_detected": is_bleed,
            }
        except Exception:  # noqa: BLE001
            return {"stem_bleed_score": 0.0, "stem_bleed_detected": False}

    def analyze_audio(self, audio_path: Path) -> tuple[AcousticMetrics, Path | None]:
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
            D = librosa.stft(mono_y, n_fft=SPECTRAL_STFT_N_FFT, hop_length=SPECTRAL_STFT_HOP_LENGTH)
            stft = np.abs(D)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=SPECTRAL_STFT_N_FFT)

            rolloff_95 = float(
                np.mean(
                    librosa.feature.spectral_rolloff(
                        S=stft, sr=sr, roll_percent=SPECTRAL_ROLLOFF_PERCENT_HIGH
                    )
                )
            )
            rolloff_85 = float(
                np.mean(
                    librosa.feature.spectral_rolloff(
                        S=stft, sr=sr, roll_percent=SPECTRAL_ROLLOFF_PERCENT_MID
                    )
                )
            )

            # 3. Tỷ lệ năng lượng dải siêu cao tần > 16kHz
            high_freq_mask = freqs >= 16000
            total_power = np.sum(stft**2) + 1e-12
            high_freq_power = np.sum((stft[high_freq_mask, :]) ** 2)
            high_freq_ratio = float(high_freq_power / total_power)

            # 3b. Phân tích bậc thềm suy giảm dải siêu cao tần (Ultrasonic Shelf & Cutoff)
            mask_mid_high = (freqs >= 12000) & (freqs < 16000)
            mask_ultra = (freqs >= 16000) & (freqs <= 20000)
            power_mid_high = (
                float(np.mean(stft[mask_mid_high, :] ** 2)) if np.any(mask_mid_high) else 0.0
            ) + 1e-12
            power_ultra = (
                float(np.mean(stft[mask_ultra, :] ** 2)) if np.any(mask_ultra) else 0.0
            ) + 1e-12
            shelf_drop_db = float(10 * np.log10(power_mid_high / power_ultra))

            ultrasonic_cutoff = (
                shelf_drop_db >= ULTRASONIC_SHELF_DROP_DB_THRESHOLD
                and rolloff_95 < ULTRASONIC_ROLLOFF_CUTOFF_HZ
                and high_freq_ratio < ULTRASONIC_HIGH_FREQ_RATIO_MAX
            )
            high_freq_slope = round(float(shelf_drop_db), 2)

            # 4. Phân tích thành phần sóng hài toàn bài (HPSS)
            H, P = librosa.decompose.hpss(D)
            harm_power = float(np.sum(np.abs(H) ** 2)) + 1e-12
            perc_power = float(np.sum(np.abs(P) ** 2)) + 1e-12
            harmonic_ratio = float(harm_power / (harm_power + perc_power))
            mono_harm = librosa.istft(H, hop_length=SPECTRAL_STFT_HOP_LENGTH, length=len(mono_y))
            mono_perc = librosa.istft(P, hop_length=SPECTRAL_STFT_HOP_LENGTH, length=len(mono_y))

            # 4b. Phân tích vi biến thiên cao độ thanh âm
            pitch_res = self._compute_pitch_jitter(mono_harm, sr=sr)

            # 4c. Phân tích hiện tượng rò rỉ âm học đa kênh
            stem_bleed_res = self._compute_stem_bleed(mono_harm, mono_perc, sr=sr)

            # 5. Phân tích vết vân Fourier qua Interface IFakeprintDetector
            fakeprint_res = self.fakeprint_detector.analyze_waveform(mono_y, sr=sr)
            fakeprint_score = fakeprint_res["fakeprint_score"]
            fakeprint_detected = fakeprint_res["fakeprint_detected"]

            # 5b. Nhận diện Nhạc không lời
            vocal_band_mask = (freqs >= VOCAL_BAND_FREQ_MIN) & (freqs <= VOCAL_BAND_FREQ_MAX)
            vocal_band_harm_power = (
                float(np.sum(np.abs(H[vocal_band_mask, :]) ** 2))
                if np.any(vocal_band_mask)
                else 0.0
            )
            vocal_energy_ratio = round(float(vocal_band_harm_power / harm_power), 4)

            pitch_label = str(pitch_res.get("pitch_stability_label", "")).lower()
            is_instrumental = (
                vocal_energy_ratio < INSTRUMENTAL_VOCAL_ENERGY_MAX
                or "instrumental" in pitch_label
                or pitch_res.get("pitch_rigidity_ratio", 0.0) == 0.0
            )

            # 5c. Cờ nhận diện AI Vocals đời mới
            rigidity = float(pitch_res.get("pitch_rigidity_ratio", 0.0) or 0.0)
            stem_bleed = float(stem_bleed_res.get("stem_bleed_score", 0.0) or 0.0)
            modern_ai_vocal_suspect = self._is_modern_ai_vocal(
                rolloff_95=rolloff_95,
                rigidity=rigidity,
                harmonic_ratio=harmonic_ratio,
                fakeprint_score=fakeprint_score,
                stem_bleed=stem_bleed,
                ultrasonic_cutoff=ultrasonic_cutoff,
            )

            # 5d. Phân tích độ mạch lạc pha (Phase Coherence) & Nén Clipping Master (Phase 2)
            phase_coherence = self._calculate_high_freq_phase_coherence(mono_y, sr)
            crest_factor_db, clipping_ratio, is_clipped_master = self._calculate_clipping_metrics(mono_y)

            # 6. Đánh giá nén truyền tải và vết ngắt tần
            has_cutoff = rolloff_95 < COMPRESSION_LOSSY_ROLLOFF_MAX
            if has_cutoff and not fakeprint_detected:
                compression_source = "Opus/AAC (YouTube/Web Streaming Lossy)"
            elif rolloff_95 >= COMPRESSION_STUDIO_ROLLOFF_MIN:
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
                has_brickwall_cutoff=has_cutoff,
                pitch_jitter=pitch_res["pitch_jitter"],
                pitch_rigidity_ratio=pitch_res["pitch_rigidity_ratio"],
                pitch_anomaly_detected=pitch_res["pitch_anomaly_detected"],
                pitch_stability_label=pitch_res["pitch_stability_label"],
                vocal_formant_variance=pitch_res.get("vocal_formant_variance"),
                is_studio_autotune=pitch_res.get("is_studio_autotune", False),
                stem_bleed_score=stem_bleed_res["stem_bleed_score"],
                stem_bleed_detected=stem_bleed_res["stem_bleed_detected"],
                ultrasonic_cutoff_detected=ultrasonic_cutoff,
                high_freq_rolloff_slope=high_freq_slope,
                is_instrumental=is_instrumental,
                vocal_energy_ratio=vocal_energy_ratio,
                modern_ai_vocal_suspect=modern_ai_vocal_suspect,
                phase_coherence_score=phase_coherence,
                crest_factor_db=crest_factor_db,
                clipping_ratio=clipping_ratio,
                is_clipped_master=is_clipped_master,
            )

            # 7. Vẽ biểu đồ Forensic 3 tầng
            image_path = self._generate_forensic_plot(
                mono_y=mono_y,
                mono_harm=mono_harm,
                sr=sr,
                audio_stem=audio_path.stem,
                metrics=metrics,
                fakeprint_res=fakeprint_res,
                stft_mag=stft,
                harm_stft_mag=np.abs(H),
            )

            return metrics, image_path

        except Exception as e:
            raise AudioProcessingException(
                f"Lỗi khi trích xuất thông số âm học từ '{audio_path}': {e}"
            ) from e

    def _calculate_high_freq_phase_coherence(
        self, y: np.ndarray, sr: int, fmin: float = 6000.0
    ) -> float:
        """Tính toán độ mạch lạc pha (Phase Coherence) dải cao (>6kHz).
        
        Vocal con người (kể cả qua Auto-Tune/Melodyne) có độ mạch lạc pha cao (>= 0.65).
        Neural Vocoder (Suno/Udio) bị phân tán pha ngẫu nhiên (Phase Smearing) dải cao (< 0.55).
        """
        try:
            if len(y) < sr:
                return 1.0
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            mag, phase = np.abs(D), np.angle(D)
            
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            hf_idx = np.where(freqs >= fmin)[0]
            if len(hf_idx) < 5 or phase.shape[1] < 3:
                return 1.0
                
            hf_phase = phase[hf_idx, :]
            hf_mag = mag[hf_idx, :]
            
            dphase = np.diff(hf_phase, axis=1)
            d2phase = np.diff(dphase, axis=1)
            d2phase_wrapped = np.angle(np.exp(1j * d2phase))
            
            weights = hf_mag[:, 2:] + 1e-9
            mean_circ_var = np.average(np.abs(d2phase_wrapped), weights=weights)
            
            coherence = max(0.0, min(1.0, 1.0 - (mean_circ_var / (np.pi * 0.75))))
            return float(round(coherence, 3))
        except Exception:
            return 1.0

    def _calculate_clipping_metrics(self, y: np.ndarray) -> tuple[float, float, bool]:
        """Tính toán hệ số đỉnh Crest Factor và tỷ lệ mẫu chạm trần clipping (Phonk/EDM)."""
        try:
            if len(y) == 0:
                return 12.0, 0.0, False
            peak = float(np.max(np.abs(y)))
            rms = float(np.sqrt(np.mean(y**2))) + 1e-9
            crest_factor_db = float(round(20.0 * np.log10(max(peak, 1e-9) / rms), 2))
            
            clipping_samples = int(np.sum(np.abs(y) >= 0.985))
            clipping_ratio = float(round(clipping_samples / max(len(y), 1), 4))
            is_clipped_master = bool(clipping_ratio >= 0.015 or (crest_factor_db < 7.0 and clipping_ratio >= 0.005))
            return crest_factor_db, clipping_ratio, is_clipped_master
        except Exception:
            return 12.0, 0.0, False

    def _get_tier3_spectrogram_title(self, metrics: AcousticMetrics) -> str:
        """Trả về tiêu đề trung tính khoa học cho Subplot 3 của ảnh phổ."""
        return "TẦNG 3: PHỔ DƯ SAI FOURIER (FOURIER RESIDUAL DECONVOLUTION SPECTRUM)"

    def _generate_forensic_plot(
        self,
        mono_y: np.ndarray,
        mono_harm: np.ndarray | None,
        sr: int,
        audio_stem: str,
        metrics: AcousticMetrics,
        fakeprint_res: dict[str, Any],
        stft_mag: np.ndarray | None = None,
        harm_stft_mag: np.ndarray | None = None,
    ) -> Path | None:
        """Tạo biểu đồ phổ hình 3 tầng chuyên nghiệp chống hiện tượng che mờ âm học."""
        out_image_path = self.temp_dir / f"{audio_stem}_forensic.png"
        try:
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(11, 9.8), gridspec_kw={"height_ratios": [1.5, 1.5, 1.1]}
            )
            fig.patch.set_facecolor("#0F172A")

            # TẦNG 1: Mel-Spectrogram toàn dải (Full Mix)
            if stft_mag is not None:
                S = librosa.feature.melspectrogram(S=stft_mag**2, sr=sr, n_mels=128, fmax=sr // 2)
            else:
                S = librosa.feature.melspectrogram(y=mono_y, sr=sr, n_mels=128, fmax=sr // 2)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img1 = librosa.display.specshow(
                S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=sr // 2, ax=ax1, cmap="magma"
            )
            ax1.set_title(
                f"TẦNG 1: QUANG PHỔ TOÀN DẢI (FULL AUDIO MIX) | Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz | {metrics.compression_source}",
                color="#38BDF8",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax1.set_facecolor("#0B0F19")
            ax1.tick_params(colors="#94A3B8")
            ax1.yaxis.label.set_color("#94A3B8")
            ax1.xaxis.label.set_color("#94A3B8")
            cbar1 = fig.colorbar(img1, ax=ax1, format="%+2.0f dB", pad=0.02)
            cbar1.ax.tick_params(colors="#94A3B8")

            # TẦNG 2: Mel-Spectrogram sóng hài
            if harm_stft_mag is not None:
                S_harm = librosa.feature.melspectrogram(
                    S=harm_stft_mag**2, sr=sr, n_mels=128, fmax=sr // 2
                )
            else:
                target_harm = mono_harm if mono_harm is not None else mono_y
                S_harm = librosa.feature.melspectrogram(
                    y=target_harm, sr=sr, n_mels=128, fmax=sr // 2
                )
            S_harm_dB = librosa.power_to_db(S_harm, ref=np.max)
            img2 = librosa.display.specshow(
                S_harm_dB, x_axis="time", y_axis="mel", sr=sr, fmax=sr // 2, ax=ax2, cmap="inferno"
            )
            ax2.set_title(
                "TẦNG 2: PHỔ SÓNG HÀI VOCAL (KHỬ TRỐNG / BỘ GÕ QUA HPSS) | Bóc tách giọng mộc, chống che mờ âm học",
                color="#A78BFA",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax2.set_facecolor("#0B0F19")
            ax2.tick_params(colors="#94A3B8")
            ax2.yaxis.label.set_color("#94A3B8")
            ax2.xaxis.label.set_color("#94A3B8")
            cbar2 = fig.colorbar(img2, ax=ax2, format="%+2.0f dB", pad=0.02)
            cbar2.ax.tick_params(colors="#94A3B8")

            # TẦNG 3: Phổ vân Fourier Deconvolution ISMIR 2025
            sub_freqs = fakeprint_res.get("freqs", np.array([]))
            residual = fakeprint_res.get("residual_spectrum", np.array([]))

            if len(sub_freqs) > 0 and len(residual) > 0:
                ax3.plot(
                    sub_freqs / 1000.0,
                    residual,
                    color="#EC4899",
                    linewidth=1.2,
                    label="Fourier Comb Residue",
                )
                ax3.set_xlim(sub_freqs[0] / 1000.0, sub_freqs[-1] / 1000.0)
                ax3.set_xlabel("Tần số (kHz)", color="#94A3B8", fontsize=9.5)
                ax3.set_ylabel("Cường độ vân", color="#94A3B8", fontsize=9.5)
                ax3.grid(True, linestyle="--", alpha=0.3, color="#475569")

                ax3.set_title(
                    self._get_tier3_spectrogram_title(metrics),
                    color="#38BDF8",
                    fontsize=10.0,
                    fontweight="bold",
                    pad=8,
                )
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "Không đủ dữ liệu Fourier để vẽ tầng 3",
                    color="#94A3B8",
                    ha="center",
                    va="center",
                )

            ax3.set_facecolor("#0B0F19")
            ax3.tick_params(colors="#94A3B8")

            plt.tight_layout()
            plt.savefig(
                str(out_image_path),
                dpi=SPECTROGRAM_DPI,
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            plt.close(fig)
            return out_image_path

        except Exception as e:  # noqa: BLE001
            print(f"[SignalAnalyzerService] Không thể tạo ảnh phổ: {e}")
            plt.close("all")
            return None
