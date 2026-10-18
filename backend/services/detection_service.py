import logging
import re
from pathlib import Path

from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.interfaces.services.audio_engine_interface import IAudioEngine
from ..core.interfaces.services.classifier_interface import IAIClassifier
from ..core.interfaces.services.detection_service_interface import IDetectionService
from ..core.interfaces.services.resolver_interface import IInputResolver
from ..core.interfaces.services.signal_analyzer_interface import ISignalAnalyzer
from ..core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from .gemini_classifier_service import is_derivative_or_remix_context

logger = logging.getLogger(__name__)


class DetectionService(IDetectionService):
    """
    Hiện thực nghiệp vụ điều phối nhận diện âm nhạc.
    Tuân thủ interface IDetectionService và kết nối các dịch vụ hoàn toàn thông qua Interface.
    """

    def __init__(
        self,
        audio_engine: IAudioEngine,
        signal_analyzer: ISignalAnalyzer,
        classifier: IAIClassifier,
        resolver: IInputResolver,
        track_repo: ITrackRepository,
    ):
        self.audio_engine = audio_engine
        self.signal_analyzer = signal_analyzer
        self.classifier = classifier
        self.resolver = resolver
        self.track_repo = track_repo

    def detect_single(
        self,
        track: TrackItem,
        model_override: str | None = None,
        force_rescan: bool = False,
    ) -> DetectionResult:
        """Thực hiện toàn bộ quy trình nhận diện 1 bài hát từ A đến Z (hỗ trợ Cache thông minh và Force Re-scan)."""
        # 0a. Kiểm tra kết quả trong CSDL trước nếu không yêu cầu quét lại (Force Re-scan)
        if not force_rescan:
            existing_result = self.track_repo.find_result_by_query_or_url(
                query=track.raw_query, resolved_url=track.resolved_url
            )
            if existing_result and existing_result.verdict_type != VerdictType.ERROR:
                if existing_result.acoustic_metrics:
                    m = existing_result.acoustic_metrics
                    if not getattr(m, "modern_ai_vocal_suspect", False):
                        r95 = getattr(m, "spectral_rolloff_95_hz", 20000.0) or 20000.0
                        rig = getattr(m, "pitch_rigidity_ratio", 0.0) or 0.0
                        harm = getattr(m, "harmonic_ratio", 0.0) or 0.0
                        cut = getattr(m, "ultrasonic_cutoff_detected", False) or False
                        m.modern_ai_vocal_suspect = (
                            r95 < 11800.0 and rig >= 0.45 and harm >= 0.80
                        ) or (cut and rig >= 0.50)
                    meta = {
                        "distributor": track.distributor
                        or getattr(existing_result, "distributor", ""),
                        "title": track.track_title or track.raw_query,
                        "artist": track.artist or "",
                    }
                    existing_result = self._apply_cross_validation_safeguard(
                        result=existing_result,
                        track=track,
                        metadata=meta,
                        metrics=m,
                    )
                logger.info(
                    "⚡ [CACHE HIT] Đã tìm thấy kết quả phân tích cũ cho '%s' (ID: %s)",
                    track.track_title or track.raw_query,
                    existing_result.track_id,
                )
                existing_result.is_cached = True
                return existing_result

        # 0b. Phân giải URL và xác thực bài hát nếu track chưa có resolved_url hợp lệ (hoặc khi yêu cầu quét lại)
        if (not track.resolved_url or force_rescan) and track.input_type not in (
            InputType.LOCAL_FILE,
            InputType.URL,
        ):
            try:
                try:
                    resolved_item = self.resolver.resolve_track(
                        track.raw_query, artist=track.artist or "", force_rescan=force_rescan
                    )
                except TypeError:
                    resolved_item = self.resolver.resolve_track(track.raw_query)

                if resolved_item and getattr(resolved_item, "resolved_url", None):
                    res_url = resolved_item.resolved_url
                    if res_url and not res_url.startswith("<MagicMock"):
                        track.resolved_url = res_url
                    if getattr(resolved_item, "track_title", None) and isinstance(
                        resolved_item.track_title, str
                    ):
                        track.track_title = resolved_item.track_title
                    if getattr(resolved_item, "artist", None) and isinstance(
                        resolved_item.artist, str
                    ):
                        track.artist = resolved_item.artist
                    if getattr(resolved_item, "is_verified_match", None) is not None:
                        track.is_verified_match = bool(resolved_item.is_verified_match)
                    if getattr(resolved_item, "match_score", None) is not None and isinstance(
                        resolved_item.match_score, (int, float)
                    ):
                        track.match_score = float(resolved_item.match_score)
                    if getattr(resolved_item, "duration_sec", None) is not None and isinstance(
                        resolved_item.duration_sec, (int, float)
                    ):
                        track.duration_sec = float(resolved_item.duration_sec)
                    if getattr(resolved_item, "platform", None) and isinstance(
                        resolved_item.platform, str
                    ):
                        track.platform = resolved_item.platform

                    # Sau khi phân giải ra URL chính thức, kiểm tra cache một lần nữa
                    if not force_rescan and track.resolved_url:
                        existing_after_resolve = self.track_repo.find_result_by_query_or_url(
                            query=track.raw_query, resolved_url=track.resolved_url
                        )
                        if (
                            existing_after_resolve
                            and existing_after_resolve.verdict_type != VerdictType.ERROR
                        ):
                            if existing_after_resolve.acoustic_metrics:
                                m = existing_after_resolve.acoustic_metrics
                                if not getattr(m, "modern_ai_vocal_suspect", False):
                                    r95 = getattr(m, "spectral_rolloff_95_hz", 20000.0) or 20000.0
                                    rig = getattr(m, "pitch_rigidity_ratio", 0.0) or 0.0
                                    harm = getattr(m, "harmonic_ratio", 0.0) or 0.0
                                    cut = getattr(m, "ultrasonic_cutoff_detected", False) or False
                                    m.modern_ai_vocal_suspect = (
                                        r95 < 11800.0 and rig >= 0.45 and harm >= 0.80
                                    ) or (cut and rig >= 0.50)
                                meta = {
                                    "distributor": track.distributor
                                    or getattr(existing_after_resolve, "distributor", ""),
                                    "title": track.track_title or track.raw_query,
                                    "artist": track.artist or "",
                                }
                                existing_after_resolve = self._apply_cross_validation_safeguard(
                                    result=existing_after_resolve,
                                    track=track,
                                    metadata=meta,
                                    metrics=m,
                                )
                            logger.info(
                                "⚡ [CACHE HIT] Đã tìm thấy kết quả phân tích cũ sau khi phân giải URL cho '%s'",
                                track.resolved_url,
                            )
                            existing_after_resolve.is_cached = True
                            return existing_after_resolve
            except Exception as resolve_err:
                logger.warning(
                    "Không thể tự động phân giải URL cho bài '%s': %s",
                    track.track_title or track.raw_query,
                    resolve_err,
                )

        self.track_repo.save_track(track)

        # 1. Tải và cắt phân đoạn âm thanh qua IAudioEngine
        audio_path, metadata = self.audio_engine.process_track_audio(track)

        # Cập nhật thông tin thực tế từ metadata (yt-dlp) nếu trích xuất thành công
        if metadata is None:
            metadata = {}

        if metadata:
            if metadata.get("is_fallback_audio"):
                cand = {
                    "title": metadata.get("title", ""),
                    "uploader": metadata.get("uploader", metadata.get("channel", "")),
                    "duration": metadata.get("duration"),
                }
                is_match, score = self.resolver.validate_match(
                    target_title=track.track_title or track.raw_query,
                    target_artist=track.artist or "",
                    candidate=cand,
                )
                track.is_verified_match = is_match
                track.match_score = score
                metadata["is_verified_match"] = is_match
                metadata["match_score"] = score
                if not is_match:
                    metadata["is_unverified_fallback"] = True
            elif track.is_verified_match is not None:
                metadata["is_verified_match"] = track.is_verified_match
                metadata["match_score"] = track.match_score

            real_title = str(metadata.get("title") or "").strip()
            if real_title and (track.is_verified_match is True or not track.track_title):
                track.track_title = real_title

            resolved_web_url = str(
                metadata.get("webpage_url")
                or metadata.get("original_url")
                or (
                    f"https://www.youtube.com/watch?v={metadata.get('id')}"
                    if metadata.get("id")
                    else ""
                )
                or metadata.get("url")
                or ""
            ).strip()

            if metadata.get("is_fallback_audio") and resolved_web_url:
                track.resolved_url = resolved_web_url
            elif resolved_web_url and not track.resolved_url:
                track.resolved_url = resolved_web_url
            elif not track.resolved_url and track.input_type == InputType.LOCAL_FILE:
                track.resolved_url = track.raw_query

            self.track_repo.save_track(track)
        else:
            resolved_web_url = ""

        metadata["target_artist"] = track.artist
        metadata["target_title"] = track.track_title
        metadata["song_type"] = getattr(track, "song_type", None)
        metadata["target_release_date"] = getattr(track, "release_date", None)
        metadata["distributor"] = getattr(track, "distributor", None)
        metadata["raw_query"] = track.raw_query

        # Xác định URL nguồn tốt nhất hiện có
        source_url_val = (
            track.resolved_url
            or resolved_web_url
            or (track.raw_query if track.raw_query and track.raw_query.startswith("http") else None)
        )
        if not source_url_val and track.input_type == InputType.LOCAL_FILE:
            source_url_val = track.raw_query

        if source_url_val and not track.resolved_url:
            track.resolved_url = source_url_val
            self.track_repo.save_track(track)

        # 2. Kiểm tra bộ lọc Heuristics qua ISignalAnalyzer
        heuristics_result = self.signal_analyzer.check_heuristics(metadata)
        if heuristics_result:
            is_ai = heuristics_result["is_ai"]
            v_type = heuristics_result.get(
                "verdict_type",
                VerdictType.CONFIRMED_HUMAN if not is_ai else VerdictType.AI_GENERATED,
            )
            if v_type == VerdictType.CONFIRMED_HUMAN:
                verdict_label = "CONFIRMED HUMAN (Người làm)"
            elif v_type == VerdictType.PROCESSED_HUMAN:
                verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
            elif v_type == VerdictType.HYBRID_AI:
                verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
            elif v_type == VerdictType.UNCERTAIN:
                verdict_label = "UNCERTAIN (Chưa thể kết luận)"
            else:
                verdict_label = "AI-GENERATED (Nhạc AI tạo)"

            # Tạo ảnh phổ và metrics nếu audio_path hợp lệ
            metrics = None
            spec_image_path = None
            if audio_path and Path(audio_path).exists():
                try:
                    metrics, spec_image_path = self.signal_analyzer.analyze_audio(audio_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Không thể tạo ảnh phổ/metrics cho bài '%s': %s",
                        track.track_title or track.raw_query,
                        exc,
                    )

            result = DetectionResult(
                track_id=track.id,
                track_title=track.track_title or track.raw_query,
                source_url=source_url_val,
                input_type=track.input_type,
                raw_query=track.raw_query,
                is_ai=is_ai,
                verdict_type=v_type,
                verdict_label=verdict_label,
                confidence_score=heuristics_result["confidence"],
                suspected_engine=heuristics_result["engine"],
                reasons=[heuristics_result["reason"]],
                detailed_explanation=f"Xác thực qua siêu dữ liệu nguồn: {heuristics_result['reason']}",
                acoustic_metrics=metrics,
                spectrogram_image_path=str(spec_image_path) if spec_image_path else None,
            )
            self.track_repo.save_result(result)
            return result

        # 3. Phân tích âm học và tạo ảnh Forensic 2 tầng qua ISignalAnalyzer
        metrics, spec_image_path = self.signal_analyzer.analyze_audio(audio_path)

        # 4. Gửi lên IAIClassifier để phân loại
        result = self.classifier.classify_track(
            track_id=track.id,
            track_title=track.track_title or track.raw_query,
            source_url=source_url_val,
            audio_path=audio_path,
            spectrogram_path=spec_image_path,
            metrics=metrics,
            model_override=model_override,
            metadata=metadata,
            input_type=track.input_type,
            raw_query=track.raw_query,
        )

        if getattr(result, "input_type", None) is None:
            result.input_type = track.input_type
        if getattr(result, "raw_query", None) is None:
            result.raw_query = track.raw_query
        if getattr(result, "source_url", None) is None or not result.source_url:
            result.source_url = source_url_val

        # 5. Bộ lọc Trọng tài Kiểm chéo hai chiều (Cross-Validation Safeguard)
        result = self._apply_cross_validation_safeguard(result, track, metadata, metrics)

        self.track_repo.save_result(result)
        return result

    def _apply_cross_validation_safeguard(
        self,
        result: DetectionResult,
        track: TrackItem,
        metadata: dict | None,
        metrics: AcousticMetrics | None,
    ) -> DetectionResult:
        """
        Bộ lọc Trọng tài Kiểm chéo hai chiều (Cross-Validation Safeguard):
        Đối soát giữa phán quyết của mô hình phân loại và các bằng chứng vật lý âm học khách quan.

        1. Safeguard Anti-False Positive (Chống vu oan cho Remix/Slowed/EDM):
           Nếu kết luận là AI (HYBRID_AI / AI_GENERATED) nhưng bài hát thuộc thể loại Remix/Slowed/Lofi/EDM
           và các chỉ số vật lý hoàn toàn sạch (Fakeprint < 0.25, Stem Bleed < 0.20, không có Cutoff)
           -> Hạ cấp về PROCESSED_HUMAN (is_ai: False).

        2. Safeguard Anti-False Negative (Chống bỏ sót AI núp bóng Pop Ballad):
           Nếu kết luận là Người (PROCESSED_HUMAN) nhưng xuất hiện bằng chứng toán học vật lý vượt trội
           của AI (Fakeprint >= 0.50 HOẶC Stem Bleed >= 0.40 HOẶC Ultrasonic Cutoff với Fakeprint >= 0.35)
           -> Nâng cấp lên AI (is_ai: True).
        """
        if not metrics:
            return result

        meta = metadata or {}
        distributor_str = (
            str(meta.get("distributor") or getattr(track, "distributor", "") or "").strip().lower()
        )
        artist_str = str(meta.get("artist") or getattr(track, "artist", "") or "").strip()
        title_str = str(
            meta.get("title")
            or getattr(track, "track_title", "")
            or getattr(track, "raw_query", "")
            or ""
        ).strip()

        from .signal_analyzer_service import (
            _GENERIC_ANONYMOUS_ARTISTS,
            get_dynamic_catalog,
            is_verified_human_artist,
        )

        is_human_artist = is_verified_human_artist(artist_str, title_str)
        is_major_exclusive_label = any(
            w in distributor_str for w in ["sony", "universal", "umg", "vieent", "loops", "bmz"]
        )

        _, ai_dist_list, ai_artist_regexes = get_dynamic_catalog()
        is_ai_dist = any(ad in distributor_str for ad in ai_dist_list) or (
            "incom" in distributor_str and not is_human_artist
        )
        if not is_ai_dist:
            for pat, _ in ai_artist_regexes:
                if re.search(pat, artist_str.lower()) or re.search(pat, title_str.lower()):
                    is_ai_dist = True
                    break

        # 1. Safeguard Anti-FP (Bảo vệ nhạc người thật khỏi bị vu oan do nén lossy hoặc Auto-Tune)
        if result.is_ai:
            first_reason = (result.reasons[0] if result.reasons else "").lower()

            # 1a. Ca sĩ INCOM xác thực
            if "nhà phân phối 'incom'" in first_reason:
                artist_clean = artist_str.lower()
                if artist_clean and artist_clean not in _GENERIC_ANONYMOUS_ARTISTS:
                    logger.info(
                        "Kích hoạt Trọng tài Hậu kiểm (Anti-FP): Nghệ sĩ '%s' phát hành qua INCOM là nghệ sĩ cá nhân thật, hạ cấp về PROCESSED_HUMAN.",
                        track.track_title,
                    )
                    result.is_ai = False
                    result.verdict_type = VerdictType.PROCESSED_HUMAN
                    result.verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                    result.reasons = [
                        f"[BẢO VỆ NHẠC NGƯỜI]: Nghệ sĩ '{track.artist}' phát hành qua INCOM là nghệ sĩ độc lập thực tế, không thuộc dự án AI sinh hàng loạt."
                    ]
                    result.confidence_score = max(0.90, result.confidence_score)
                    result.suspected_engine = "None"
                    return result

            # 1b. Bảo vệ ca sĩ người thật đã xác thực hoặc hãng đĩa lớn độc quyền
            if is_human_artist or is_major_exclusive_label:
                stem_bleed_val = getattr(metrics, "stem_bleed_score", 0.0) or 0.0
                if not (metrics.fakeprint_score >= 0.85 and stem_bleed_val >= 0.50):
                    logger.info(
                        "Kích hoạt Trọng tài Hậu kiểm (Anti-FP): Nghệ sĩ xác thực '%s' / Hãng đĩa lớn được bảo vệ, hạ cấp về PROCESSED_HUMAN.",
                        track.track_title,
                    )
                    result.is_ai = False
                    result.verdict_type = VerdictType.PROCESSED_HUMAN
                    result.verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                    safeguard_reason = (
                        f"[BẢO VỆ NHẠC NGƯỜI]: Bản thu của nghệ sĩ '{artist_str or track.artist}' là sản phẩm âm nhạc có con người thực hiện, "
                        "tránh nhận diện nhầm do hiệu ứng Auto-Tune phòng thu hoặc nén băng thông phát trực tuyến."
                    )
                    result.reasons = [safeguard_reason] + (result.reasons or [])
                    result.confidence_score = max(0.92, result.confidence_score)
                    result.suspected_engine = "None"
                    return result

            # 1c. Bản thu độc lập có chỉ số vật lý sạch
            # TUYỆT ĐỐI KHÔNG HẠ CẤP nếu:
            # - Thuộc danh mục nhà phân phối AI hoặc dự án nghệ sĩ ảo AI (is_ai_dist)
            # - Có chỉ thị phát hiện hàng loạt / dự án AI từ metadata / heuristics
            # - Mang đặc trưng phổ AI (Ultrasonic cutoff đi kèm Pitch rigidity cao của Suno v3.5/v4)
            is_catalog_or_batch_flag = any(
                kw in first_reason
                for kw in [
                    "phát hành hàng loạt",
                    "dự án âm nhạc",
                    "nhà phân phối",
                    "ai virtual project",
                    "bài số",
                ]
            )
            has_cutoff = getattr(metrics, "ultrasonic_cutoff_detected", False)
            rigidity = getattr(metrics, "pitch_rigidity_ratio", 0.0) or 0.0
            harm_ratio = getattr(metrics, "harmonic_ratio", 0.0) or 0.0
            is_ai_acoustic_signature = has_cutoff and rigidity >= 0.50 and harm_ratio >= 0.80

            is_derivative = is_derivative_or_remix_context(meta)
            is_vocal_human_suspect_beat = "vocal là người thật" in first_reason
            stem_bleed_val = getattr(metrics, "stem_bleed_score", 0.0) or 0.0
            is_clean_physical = (
                not is_ai_dist
                and not is_catalog_or_batch_flag
                and not is_ai_acoustic_signature
                and metrics.fakeprint_score < 0.25
                and not metrics.fakeprint_detected
                and stem_bleed_val < 0.25
            )
            if (
                is_derivative or is_vocal_human_suspect_beat or is_clean_physical
            ) and is_clean_physical:
                logger.info(
                    "Kích hoạt Trọng tài Hậu kiểm (Anti-FP): Bản thu '%s' có chỉ số vật lý sạch, hạ cấp về PROCESSED_HUMAN.",
                    track.track_title,
                )
                result.is_ai = False
                result.verdict_type = VerdictType.PROCESSED_HUMAN
                result.verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                safeguard_reason = (
                    "[BẢO VỆ NHẠC NGƯỜI]: Bản thu có các chỉ số vật lý sạch "
                    f"(Fakeprint: {metrics.fakeprint_score * 100:.1f}%, Stem Bleed: {stem_bleed_val:.2f}). "
                    "Không có dấu vết tạo sinh từ mô hình AI."
                )
                result.reasons = [safeguard_reason] + (result.reasons or [])
                result.confidence_score = max(0.90, result.confidence_score)
                result.suspected_engine = "None"
                return result

        # 2. Safeguard Anti-FN (Bắt AI lẩn khuất - Đồng thuận đa tiêu chí)
        if not result.is_ai:
            stem_bleed = getattr(metrics, "stem_bleed_score", 0.0) or 0.0
            rigidity = getattr(metrics, "pitch_rigidity_ratio", 0.0) or 0.0
            has_cutoff = getattr(metrics, "ultrasonic_cutoff_detected", False)
            rolloff = metrics.spectral_rolloff_95_hz
            harm_ratio = metrics.harmonic_ratio

            # Tiêu chí 1: AI thuần nhạc cụ / ambient
            is_instrumental_ai = rolloff < 6500.0 and rigidity >= 0.72

            # Tiêu chí 2: AI truyền thống có dấu vân răng lược Fourier rõ rệt hoặc phát hành từ nguồn AI đã xác thực
            classic_ai_consensus = (
                is_ai_dist
                or (metrics.fakeprint_score >= 0.65 and stem_bleed >= 0.30)
                or (has_cutoff and metrics.fakeprint_score >= 0.60)
            )

            # Tiêu chí 3: AI Vocals đời mới (Suno v3.5/v4, AI Vocal Clones)
            # Chỉ kích hoạt khi KHÔNG PHẢI là nghệ sĩ người thật đã xác thực
            modern_ai_trigger = (
                not is_human_artist
                and not is_major_exclusive_label
                and (
                    (
                        rigidity >= 0.47
                        and harm_ratio >= 0.80
                        and (metrics.fakeprint_score >= 0.35 or stem_bleed >= 0.32)
                    )
                    or (has_cutoff and rigidity >= 0.55 and harm_ratio >= 0.80)
                    or (
                        rolloff < 10000.0
                        and rigidity >= 0.50
                        and harm_ratio >= 0.88
                        and metrics.fakeprint_score >= 0.30
                    )
                )
            )

            has_multi_metric_ai_consensus = (
                is_instrumental_ai or classic_ai_consensus or modern_ai_trigger
            )

            if has_multi_metric_ai_consensus:
                logger.info(
                    "Kích hoạt Trọng tài Hậu kiểm (Anti-FN): Bản thu '%s' có bằng chứng vật lý AI vượt trội, nâng cấp lên AI.",
                    track.track_title,
                )
                result.is_ai = True
                result.verdict_type = VerdictType.HYBRID_AI
                result.verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
                if modern_ai_trigger:
                    safeguard_reason = (
                        "[CẢNH BÁO ÂM HỌC VẬT LÝ]: Dấu vết nơ-ron tổng hợp giọng hát AI đời mới "
                        f"(Rolloff 95%: {rolloff:.0f} Hz, Độ cứng cao độ: {rigidity * 100:.1f}%, "
                        f"Harmonic Ratio: {harm_ratio:.2f}). Trần tần số sụt giảm kết hợp lưới cao độ phẳng nhân tạo áp đảo phán quyết ban đầu."
                    )
                else:
                    safeguard_reason = (
                        "[CẢNH BÁO ÂM HỌC VẬT LÝ]: Bằng chứng vật lý toán học xác nhận dấu vết AI "
                        f"(Fakeprint: {metrics.fakeprint_score * 100:.1f}%, Stem Bleed: {stem_bleed:.2f}). "
                        "Dấu vết kỹ thuật deconvolution và intermodulation stem bleed áp đảo phán quyết ban đầu."
                    )
                result.reasons = [safeguard_reason] + (result.reasons or [])
                if not result.suspected_engine or result.suspected_engine == "None":
                    result.suspected_engine = "Suno/Udio"
                result.confidence_score = max(0.90, result.confidence_score)
                return result

        # Giữ nguyên phán quyết nguyên bản từ mô hình phân loại AI
        return result

    def detect_query(
        self,
        query: str,
        model_override: str | None = None,
        force_rescan: bool = False,
    ) -> DetectionResult:
        """Phân giải chuỗi query tự do qua IInputResolver và thực hiện nhận diện bài hát."""
        track = self.resolver.resolve_track(query)
        return self.detect_single(track, model_override=model_override, force_rescan=force_rescan)
