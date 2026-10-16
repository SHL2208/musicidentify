import logging
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

    def detect_single(self, track: TrackItem, model_override: str | None = None) -> DetectionResult:
        """Thực hiện toàn bộ quy trình nhận diện 1 bài hát từ A đến Z."""
        # 0. Phân giải URL và xác thực bài hát nếu track chưa có resolved_url hợp lệ
        if (
            not track.resolved_url
            and track.input_type not in (InputType.LOCAL_FILE, InputType.URL)
        ):
            try:
                try:
                    resolved_item = self.resolver.resolve_track(
                        track.raw_query, artist=track.artist or ""
                    )
                except TypeError:
                    resolved_item = self.resolver.resolve_track(track.raw_query)

                if resolved_item and getattr(resolved_item, "resolved_url", None):
                    res_url = str(resolved_item.resolved_url)
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
            if real_title and (
                track.is_verified_match is True
                or not track.track_title
            ):
                track.track_title = real_title

            resolved_web_url = str(
                metadata.get("webpage_url")
                or metadata.get("original_url")
                or (f"https://www.youtube.com/watch?v={metadata.get('id')}" if metadata.get("id") else "")
                or metadata.get("url")
                or ""
            ).strip()

            if resolved_web_url and not track.resolved_url:
                track.resolved_url = resolved_web_url
            elif not track.resolved_url and track.input_type == InputType.LOCAL_FILE:
                track.resolved_url = str(track.raw_query)

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
            or (track.raw_query if track.raw_query and str(track.raw_query).startswith("http") else None)
        )
        if not source_url_val and track.input_type == InputType.LOCAL_FILE:
            source_url_val = str(track.raw_query)

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

        # 1. Safeguard Anti-FP
        is_remix_ctx = is_derivative_or_remix_context(metadata) or is_derivative_or_remix_context(
            {
                "song_type": getattr(track, "song_type", None),
                "title": track.track_title,
                "raw_query": track.raw_query,
            }
        )

        is_physically_clean = (
            metrics.fakeprint_score < 0.25
            and not metrics.fakeprint_detected
            and (metrics.stem_bleed_score is None or metrics.stem_bleed_score < 0.20)
            and not getattr(metrics, "ultrasonic_cutoff_detected", False)
        )

        if result.is_ai and is_remix_ctx and is_physically_clean:
            logger.info(
                "Kích hoạt Trọng tài Hậu kiểm (Anti-FP): Bản Remix '%s' có chỉ số vật lý sạch, hạ cấp về PROCESSED_HUMAN.",
                track.track_title,
            )
            result.is_ai = False
            result.verdict_type = VerdictType.PROCESSED_HUMAN
            result.verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
            safeguard_reason = (
                "[BẢO VỆ NHẠC NGƯỜI - REMIX/EDM]: Bản thu phái sinh/Remix có các chỉ số vật lý hoàn toàn sạch "
                f"(Fakeprint: {metrics.fakeprint_score * 100:.1f}%, Stem Bleed: {metrics.stem_bleed_score or 0.0:.2f}). "
                "Hiệu ứng beat điện tử và kéo dãn cao độ/thời gian là kỹ thuật DAW phòng thu chuẩn của con người."
            )
            result.reasons = [safeguard_reason] + (result.reasons or [])
            result.confidence_score = max(0.88, result.confidence_score)
            return result

        # 2. Safeguard Anti-FN
        has_low_rolloff_ai_deficiency = (
            metrics.spectral_rolloff_95_hz < 12000.0
            and metrics.fakeprint_score >= 0.24
            and not is_remix_ctx  # Đảm bảo không bắt nhầm bản Slowed đã giảm tốc độ mẫu
        )

        has_conclusive_ai_metrics = (
            (metrics.fakeprint_detected and metrics.fakeprint_score >= 0.50)
            or (
                getattr(metrics, "stem_bleed_detected", False)
                and getattr(metrics, "stem_bleed_score", 0.0) >= 0.40
            )
            or (
                getattr(metrics, "ultrasonic_cutoff_detected", False)
                and metrics.fakeprint_score >= 0.35
            )
            or has_low_rolloff_ai_deficiency
        )

        if not result.is_ai and has_conclusive_ai_metrics:
            logger.info(
                "Kích hoạt Trọng tài Hậu kiểm (Anti-FN): Bản thu '%s' có bằng chứng vật lý AI vượt trội, nâng cấp lên AI.",
                track.track_title,
            )
            result.is_ai = True
            if (
                (metrics.fakeprint_score >= 0.50 and getattr(metrics, "stem_bleed_score", 0.0) >= 0.40)
                or (
                    has_low_rolloff_ai_deficiency
                    and (metrics.fakeprint_score >= 0.24 or getattr(metrics, "ultrasonic_cutoff_detected", False))
                )
                or (
                    getattr(metrics, "ultrasonic_cutoff_detected", False)
                    and metrics.fakeprint_score >= 0.30
                )
            ):
                result.verdict_type = VerdictType.AI_GENERATED
                result.verdict_label = "AI-GENERATED (Nhạc AI tạo)"
            else:
                result.verdict_type = VerdictType.HYBRID_AI
                result.verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"

            if has_low_rolloff_ai_deficiency and not (
                (metrics.fakeprint_detected and metrics.fakeprint_score >= 0.50)
                or (
                    getattr(metrics, "stem_bleed_detected", False)
                    and getattr(metrics, "stem_bleed_score", 0.0) >= 0.40
                )
            ):
                safeguard_reason = (
                    "[CẢNH BÁO ÂM HỌC VẬT LÝ]: Phát hiện dấu hiệu triệt tiêu dải cao đặc thù của AI "
                    f"(Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz < 12kHz, Fakeprint: {metrics.fakeprint_score * 100:.1f}%). "
                    "Bản thu thương mại có dải cao bị bóp nghẹt là đặc trưng sinh âm thanh từ không gian tiềm ẩn của Suno/Udio."
                )
            else:
                safeguard_reason = (
                    "[CẢNH BÁO ÂM HỌC VẬT LÝ]: Bằng chứng vật lý toán học xác nhận dấu vết AI "
                    f"(Fakeprint: {metrics.fakeprint_score * 100:.1f}%, Stem Bleed: {metrics.stem_bleed_score or 0.0:.2f}). "
                    "Dấu vết kỹ thuật này áp đảo cảm nhận ban đầu về giọng hát."
                )

            result.reasons = [safeguard_reason] + (result.reasons or [])
            result.suspected_engine = (
                result.suspected_engine
                if result.suspected_engine and result.suspected_engine != "None"
                else "Suno/Udio"
            )
            result.confidence_score = max(0.90, result.confidence_score)
            return result

        return result

    def detect_query(self, query: str, model_override: str | None = None) -> DetectionResult:
        """Phân giải chuỗi query tự do qua IInputResolver và thực hiện nhận diện bài hát."""
        track = self.resolver.resolve_track(query)
        return self.detect_single(track, model_override=model_override)
