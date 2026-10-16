import re
import time
from pathlib import Path
from typing import Any

import librosa
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.exceptions import AudioProcessingException
from ..core.interfaces.repositories.storage_repository_interface import (
    IStorageRepository,
)
from ..core.interfaces.services.fakeprint_detector_interface import IFakeprintDetector
from ..core.interfaces.services.signal_analyzer_interface import ISignalAnalyzer
from ..core.models.entities import AcousticMetrics, VerdictType

_DEFAULT_HUMAN_WHITELIST = [
    "sony",
    "warner music vietnam",
    "bmz",
    "vieent",
    "believe",
    "loops",
]

_DEFAULT_PURE_AI_DISTRIBUTORS = [
    "incom",
    "tte fast considerate",
]

_DEFAULT_AI_ARTIST_REGEXES = [
    (r"\blofi quê hương\b", "Lofi Quê Hương"),
    (r"\bhanzen\b", "HanZen"),
    (r"\ban deep\b", "An Deep"),
    (r"\bgmhd\b", "GMHD"),
    (r"\bsinbadai\b", "SinbadAI"),
    (r"\bđặng bảo trung\b", "Đặng Bảo Trung"),
    (r"\btiểu mai tử\b", "Tiểu Mai Tử"),
    (r"\bf47\b", "F47"),
    (r"\bhoàng k\b", "Hoàng K"),
    (r"\bhlc\b", "HLC"),
]

_catalog_cache: dict[str, Any] = {
    "timestamp": 0.0,
    "whitelist": list(_DEFAULT_HUMAN_WHITELIST),
    "ai_distributors": list(_DEFAULT_PURE_AI_DISTRIBUTORS),
    "ai_artists": list(_DEFAULT_AI_ARTIST_REGEXES),
}


def get_dynamic_catalog() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Lấy danh mục nhà phân phối AI, Whitelist và nghệ sĩ ảo động từ CSDL (kèm cache 5 phút)."""
    global _catalog_cache
    now = time.time()
    if now - _catalog_cache["timestamp"] < 300:
        return (
            _catalog_cache["whitelist"],
            _catalog_cache["ai_distributors"],
            _catalog_cache["ai_artists"],
        )

    try:
        from sqlalchemy import select

        from ..core.database import get_db_session, is_database_available
        from ..core.models.db_models import AIArtistProjectModel, DistributorCatalogModel

        if is_database_available():
            with get_db_session() as session:
                dist_records = session.scalars(
                    select(DistributorCatalogModel).where(
                        DistributorCatalogModel.is_active.is_(True)
                    )
                ).all()
                artist_records = session.scalars(
                    select(AIArtistProjectModel).where(AIArtistProjectModel.is_active.is_(True))
                ).all()

                whitelist = [
                    d.name.lower() for d in dist_records if d.catalog_type == "HUMAN_WHITELIST"
                ]
                ai_dist = [
                    d.name.lower() for d in dist_records if d.catalog_type == "AI_DISTRIBUTOR"
                ]
                ai_artists = [(a.regex_pattern, a.project_name) for a in artist_records]

                merged_wl = list(dict.fromkeys(_DEFAULT_HUMAN_WHITELIST + whitelist))
                merged_ai = list(dict.fromkeys(_DEFAULT_PURE_AI_DISTRIBUTORS + ai_dist))
                merged_artists = ai_artists if ai_artists else list(_DEFAULT_AI_ARTIST_REGEXES)

                _catalog_cache["whitelist"] = merged_wl
                _catalog_cache["ai_distributors"] = merged_ai
                _catalog_cache["ai_artists"] = merged_artists
                _catalog_cache["timestamp"] = now
    except Exception:
        pass

    return (
        _catalog_cache["whitelist"],
        _catalog_cache["ai_distributors"],
        _catalog_cache["ai_artists"],
    )


class SignalAnalyzerService(ISignalAnalyzer):
    """
    Hiện thực phân tích âm học và biểu đồ quang phổ 2 tầng.
    Tuân thủ interface ISignalAnalyzer và nhận IFakeprintDetector qua Dependency Injection.
    """

    def __init__(self, fakeprint_detector: IFakeprintDetector, storage_repo: IStorageRepository):
        self.fakeprint_detector = fakeprint_detector
        self.storage_repo = storage_repo
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
        # d) TUYỆT ĐỐI KHÔNG ÁP DỤNG cho các bài hát COVER, REMAKE, REMIX, hoặc phiên bản phái sinh:
        #    Vì bản cover ra mắt gần đây (2023-2026) có thể hoàn toàn được tạo bằng AI dù bài hát gốc phát hành từ năm 2014-2020!
        # e) TUYỆT ĐỐI KHÔNG ÁP DỤNG nếu thông tin mục tiêu có năm phát hành thực tế >= 2023 (hoặc năm 2026)
        is_verified = metadata.get("is_verified_match") is not False
        is_fallback = metadata.get("is_unverified_fallback") is True
        match_score = float(metadata.get("match_score", 1.0) or 0.0)

        # Phát hiện bài hát Cover/Remake/Phái sinh
        song_type = str(metadata.get("song_type") or "").lower()
        title_text = f"{metadata.get('title', '')} {metadata.get('target_title', '')}".lower()
        query_text = str(metadata.get("raw_query") or "").lower()
        is_cover_or_derivative = (
            "cover" in song_type
            or "remake" in song_type
            or "remix" in song_type
            or "cover" in title_text
            or "cover" in query_text
            or "remake" in title_text
            or "remix" in title_text
            or "ai cover" in title_text
            or "ai cover" in query_text
        )

        target_rel_date = str(metadata.get("target_release_date") or "")
        is_recent_target = False
        if target_rel_date and len(target_rel_date) >= 4:
            try:
                target_year = int(target_rel_date[:4])
                if target_year >= 2023:
                    is_recent_target = True
            except ValueError:
                pass

        if (
            is_verified
            and not is_fallback
            and match_score >= 0.75
            and not is_cover_or_derivative
            and not is_recent_target
        ):
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
                        "reason": f"Bản phát hành chính thức năm {year} (trước khi các mô hình sinh nhạc AI Suno/Udio ra đời).",
                    }
            elif release_year and int(release_year) < 2022:
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

        if (
            "created with suno" in combined_text
            or "#suno" in combined_text
            or "suno ai" in combined_text
            or "suno.com" in combined_text
        ):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Suno",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Suno AI trong phần mô tả / tiêu đề.",
            }
        if (
            "created with udio" in combined_text
            or "#udio" in combined_text
            or "udio ai" in combined_text
            or "udio.com" in combined_text
        ):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.99,
                "engine": "Udio",
                "reason": "Phát hiện nhãn xác nhận sáng tác bằng Udio AI trong phần mô tả / tiêu đề.",
            }

        # 2b. Kiểm tra các thẻ lệnh điều khiển cấu trúc AI sinh nhạc
        # Chỉ kích hoạt khi có bối cảnh từ khóa AI đi kèm (dùng word boundary để tránh nhận diện nhầm 'audio' thành 'udio')
        ai_context_pattern = re.compile(
            r"\b(?:suno|udio)\b|ai\s+music|ai\s+cover|tạo bằng ai|mô hình ai|prompt:|style:",
            re.IGNORECASE,
        )
        if ai_context_pattern.search(combined_text):
            suno_prompt_pattern = re.compile(
                r"\[\s*(?:verse\s*\d*|chorus|bridge|outro|intro|pre-chorus|hook|vocal chop|sub-?bass)[^\]]*\]",
                re.IGNORECASE,
            )
            ai_prompt_tags = suno_prompt_pattern.findall(description)
            if ai_prompt_tags:
                return {
                    "is_ai": True,
                    "verdict_type": VerdictType.AI_GENERATED,
                    "confidence": 0.98,
                    "engine": "Suno",
                    "reason": f"Phát hiện thẻ lệnh điều khiển cấu trúc của AI Suno/Udio trong phần lời ({', '.join(ai_prompt_tags[:2])}).",
                }

        # 2c. Kiểm tra định dạng đánh số chuỗi bài sản xuất hàng loạt bằng AI (VD: "Bài số 83/100")
        batch_ai_pattern = re.compile(r"bài\s*(?:số\s*)?\d+\s*/\s*\d+", re.IGNORECASE)
        match_batch = batch_ai_pattern.search(title) or batch_ai_pattern.search(description)
        ai_batch_pattern = re.compile(
            r"\b(?:suno|udio|ai)\b|ai\s+music|ai\s+cover|#ai|tạo bằng ai|sinh bằng ai|dự án\s+(?:sáng tác|nhạc|ai)",
            re.IGNORECASE,
        )
        if match_batch and ai_batch_pattern.search(combined_text):
            return {
                "is_ai": True,
                "verdict_type": VerdictType.AI_GENERATED,
                "confidence": 0.95,
                "engine": "Suno/AI",
                "reason": f"Phát hiện định dạng phát hành hàng loạt đặc trưng của dự án nhạc AI: '{match_batch.group()}'.",
            }

        # 2d. Kiểm tra nhà phân phối chuyên phát hành nhạc AI thương mại (Catalog / Distributor Intelligence)
        distributor = str(metadata.get("distributor") or "").strip()
        distributor_lower = distributor.lower()
        artist_lower = str(
            metadata.get("target_artist")
            or metadata.get("artist")
            or metadata.get("uploader")
            or ""
        ).lower()

        human_distributor_whitelist, pure_ai_distributors, ai_artist_regexes = get_dynamic_catalog()
        is_whitelisted_distributor = any(
            w in distributor_lower for w in human_distributor_whitelist
        )

        # 1. Các dự án nghệ sĩ ảo / AI sinh hàng loạt đã được xác thực 100% qua catalog
        for pat, proj_name in ai_artist_regexes:
            if re.search(pat, artist_lower):
                return {
                    "is_ai": True,
                    "verdict_type": VerdictType.AI_GENERATED,
                    "confidence": 0.98,
                    "engine": "AI Virtual Project",
                    "reason": f"Dự án âm nhạc '{proj_name}' là sản phẩm phát hành hàng loạt từ mô hình sinh nhạc AI.",
                }

        # 2. Kiểm tra nhà phân phối chuyên phát hành nhạc AI thương mại (Catalog / Distributor Intelligence)
        if not is_whitelisted_distributor:
            for ai_dist in pure_ai_distributors:
                if ai_dist in distributor_lower:
                    return {
                        "is_ai": True,
                        "verdict_type": VerdictType.AI_GENERATED,
                        "confidence": 0.98,
                        "engine": "AI Studio/Catalog",
                        "reason": f"Nhà phân phối '{distributor}' nằm trong danh mục đơn vị phát hành nhạc AI chuyên biệt.",
                    }

        return None

    def _compute_pitch_jitter(self, mono_harm: np.ndarray, sr: int = 44100) -> dict[str, Any]:
        """
        Trích xuất chỉ số vi biến thiên cao độ giọng hát (Pitch Micro-Jitter) và độ cứng nhắc phẳng (Rigidity).
        Phát hiện dấu hiệu giọng hát sinh bởi AI (Suno/Udio) do thiếu micro-tremors cơ học của dây thanh đới.
        """
        try:
            target_sr = 16000
            # Chọn phân đoạn 15-20s tiêu biểu ở giữa để tối ưu tốc độ CPU trước khi resample
            total_len = len(mono_harm)
            if total_len > sr * 20:
                start = (total_len - sr * 20) // 2
                raw_seg = mono_harm[start : start + sr * 20]
            else:
                raw_seg = mono_harm

            if sr != target_sr:
                y_vocal = librosa.resample(raw_seg, orig_sr=sr, target_sr=target_sr)
            else:
                y_vocal = raw_seg

            f0 = librosa.yin(
                y_vocal, fmin=80, fmax=550, sr=target_sr, frame_length=1024, hop_length=256
            )
            voiced_mask = (f0 >= 85) & (f0 <= 500)
            voiced_f0 = f0[voiced_mask]

            if len(voiced_f0) < 30:
                return {
                    "pitch_jitter": 0.0,
                    "pitch_rigidity_ratio": 0.0,
                    "pitch_anomaly_detected": False,
                    "pitch_stability_label": "Inconclusive / Instrumental",
                    "vocal_formant_variance": 0.0,
                    "is_studio_autotune": False,
                }

            # Phân tích biến thiên Formant trong dải thanh âm (1000 - 4000 Hz)
            # nhằm bóc tách Auto-Tune người thật với Lưới Latent nhân tạo của AI
            centroids = librosa.feature.spectral_centroid(
                y=y_vocal, sr=target_sr, n_fft=1024, hop_length=256
            )
            formant_var = float(np.var(centroids)) if centroids.shape[1] > 1 else 0.0

            diffs = np.abs(np.diff(voiced_f0))
            mean_f0 = float(np.mean(voiced_f0))
            jitter = float(np.mean(diffs) / (mean_f0 + 1e-6))
            rigid_frames = np.sum(diffs < 0.8)
            rigidity_ratio = float(rigid_frames / (len(diffs) + 1e-6))

            # Nếu cao độ phẳng (rigidity >= 0.35 hoặc jitter < 0.015) nhưng Formant chuyển động
            # linh hoạt (formant_var >= 300.0) do khẩu hình phát âm tự nhiên -> Auto-Tune phòng thu của người
            is_studio_autotune = (rigidity_ratio >= 0.35 or jitter < 0.015) and formant_var >= 300.0
            is_anomaly = (
                rigidity_ratio >= 0.40 or (jitter < 0.015 and rigidity_ratio > 0.32)
            ) and not is_studio_autotune

            if is_anomaly:
                label = "Synthetic / Quantized Latent Grid"
            elif is_studio_autotune:
                label = "Studio Auto-Tuned (Natural Vocal Tract)"
            elif rigidity_ratio > 0.25:
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
        So sánh tương quan giữa năng lượng xung bộ gõ (perc) và độ biến điệu biên độ
        dải vocal harmonic (harm).
        """
        try:
            if mono_harm is None or mono_perc is None or len(mono_harm) == 0 or len(mono_perc) == 0:
                return {"stem_bleed_score": 0.0, "stem_bleed_detected": False}

            max_len = sr * 20
            if len(mono_harm) > max_len:
                start = (len(mono_harm) - max_len) // 2
                h_seg = mono_harm[start : start + max_len]
                p_seg = mono_perc[start : start + max_len]
            else:
                h_seg = mono_harm
                p_seg = mono_perc

            hop = 512
            rms_perc = librosa.feature.rms(y=p_seg, hop_length=hop)[0]
            rms_harm = librosa.feature.rms(y=h_seg, hop_length=hop)[0]

            std_p = float(np.std(rms_perc))
            std_h = float(np.std(rms_harm))
            if std_p < 1e-6 or std_h < 1e-6:
                return {"stem_bleed_score": 0.0, "stem_bleed_detected": False}

            corr_matrix = np.corrcoef(rms_perc, rms_harm)
            corr = float(corr_matrix[0, 1])
            bleed_score = float(max(0.0, corr)) if not np.isnan(corr) else 0.0

            is_bleed = bleed_score >= 0.40
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

            # 2. Phân tích phổ tần STFT và Spectral Rolloff (1-Pass STFT)
            D = librosa.stft(mono_y, n_fft=2048, hop_length=512)
            stft = np.abs(D)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

            rolloff_95 = float(
                np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.95))
            )
            rolloff_85 = float(
                np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85))
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
            ultrasonic_cutoff = shelf_drop_db >= 25.0 and rolloff_95 < 18000.0
            high_freq_slope = round(float(shelf_drop_db), 2)

            # 4. Phân tích thành phần sóng hài toàn bài (HPSS Harmonic-Percussive Sound Separation)
            H, P = librosa.decompose.hpss(D)
            harm_power = float(np.sum(np.abs(H) ** 2)) + 1e-12
            perc_power = float(np.sum(np.abs(P) ** 2)) + 1e-12
            harmonic_ratio = float(harm_power / (harm_power + perc_power))
            mono_harm = librosa.istft(H, hop_length=512, length=len(mono_y))
            mono_perc = librosa.istft(P, hop_length=512, length=len(mono_y))

            # 4b. Phân tích vi biến thiên cao độ thanh âm (Pitch Micro-Jitter & Latent Rigidity)
            pitch_res = self._compute_pitch_jitter(mono_harm, sr=sr)

            # 4c. Phân tích hiện tượng rò rỉ âm học đa kênh (Stem Bleed / Intermodulation Distortion)
            stem_bleed_res = self._compute_stem_bleed(mono_harm, mono_perc, sr=sr)

            # 5. Phân tích vết vân Fourier qua Interface IFakeprintDetector
            fakeprint_res = self.fakeprint_detector.analyze_waveform(mono_y, sr=sr)
            fakeprint_score = fakeprint_res["fakeprint_score"]
            fakeprint_detected = fakeprint_res["fakeprint_detected"]

            # 6. Đánh giá nén truyền tải và vết ngắt tần
            has_cutoff = rolloff_95 < 17500.0
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
            )

            # 7. Vẽ biểu đồ Forensic 3 tầng (Full Mix + Harmonic Vocal HPSS + Fourier Fakeprint)
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

    def _get_tier3_spectrogram_title(self, metrics: AcousticMetrics) -> str:
        """
        Trả về tiêu đề trung tính khoa học cho Subplot 3 của ảnh phổ.
        Xóa bỏ toàn bộ từ ngữ định kiến ('PHỔ ĐỒ TỰ NHIÊN', 'RĂNG LƯỢC') và điểm phần trăm
        để tránh việc Gemini OCR đọc lén và bị thiên kiến xác nhận.
        """
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
        """Tạo biểu đồ phổ hình 3 tầng chuyên nghiệp chống hiện tượng che mờ âm học (Acoustic Masking)."""
        out_image_path = self.temp_dir / f"{audio_stem}_forensic.png"
        try:
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

            # TẦNG 2: Mel-Spectrogram sóng hài (Harmonic Vocal - Đã khử trống lân/bộ gõ qua HPSS)
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
                str(out_image_path), dpi=100, facecolor=fig.get_facecolor(), edgecolor="none"
            )
            plt.close(fig)
            return out_image_path

        except Exception as e:  # noqa: BLE001
            print(f"[SignalAnalyzerService] Không thể tạo ảnh phổ: {e}")
            plt.close("all")
            return None
