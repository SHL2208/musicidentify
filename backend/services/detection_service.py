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
    WorkType,
)
from .gemini_classifier_service import is_derivative_or_remix_context
from .work_type_resolver import resolve_work_type_from_metadata

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
                if getattr(existing_result, "work_type", WorkType.UNCERTAIN) == WorkType.UNCERTAIN:
                    meta_w = {
                        "distributor": track.distributor
                        or getattr(existing_result, "distributor", ""),
                        "title": track.track_title or track.raw_query,
                        "artist": track.artist or "",
                        "song_type": getattr(track, "song_type", None),
                    }
                    w_type, w_label, is_cov, orig_a, orig_t = resolve_work_type_from_metadata(
                        title=track.track_title or track.raw_query,
                        artist=track.artist or "",
                        metadata=meta_w,
                    )
                    existing_result.work_type = w_type
                    existing_result.work_label = w_label
                    existing_result.is_cover = is_cov
                    if not getattr(existing_result, "original_artist", None) and orig_a:
                        existing_result.original_artist = orig_a
                    if not getattr(existing_result, "original_song_title", None) and orig_t:
                        existing_result.original_song_title = orig_t
                    try:
                        self.track_repo.save_result(existing_result)
                    except Exception:  # noqa: BLE001, S110
                        pass
                # Đồng bộ 2 chiều các trường metadata giữa track và existing_result:
                # 1. URL
                if track.resolved_url and not getattr(existing_result, "source_url", None):
                    existing_result.source_url = track.resolved_url
                elif getattr(existing_result, "source_url", None) and not track.resolved_url:
                    track.resolved_url = existing_result.source_url

                # Nếu cả 2 đều chưa có resolved_url, tiến hành phân giải ngay để không bị miss link
                if not track.resolved_url and not getattr(existing_result, "source_url", None) and track.input_type not in (InputType.LOCAL_FILE,):
                    try:
                        resolved_cand = self.resolver.resolve_track(track.raw_query, artist=track.artist or "")
                        if resolved_cand and resolved_cand.resolved_url:
                            track.resolved_url = resolved_cand.resolved_url
                            existing_result.source_url = resolved_cand.resolved_url
                            if resolved_cand.duration_sec and not track.duration_sec:
                                track.duration_sec = resolved_cand.duration_sec
                    except Exception:
                        pass

                # 2. Tiêu đề & Ca sĩ
                if track.track_title and (not existing_result.track_title or existing_result.track_title == "N/A" or existing_result.track_title == existing_result.raw_query):
                    existing_result.track_title = track.track_title
                elif getattr(existing_result, "track_title", None) and existing_result.track_title != "N/A" and not track.track_title:
                    track.track_title = existing_result.track_title

                if getattr(existing_result, "cover_performer", None) and not track.artist:
                    track.artist = existing_result.cover_performer

                # 3. Thời lượng
                if track.duration_sec is not None and getattr(existing_result, "duration_sec", None) is None:
                    existing_result.duration_sec = track.duration_sec
                elif getattr(existing_result, "duration_sec", None) is not None and track.duration_sec is None:
                    track.duration_sec = existing_result.duration_sec
                elif existing_result.acoustic_metrics and existing_result.acoustic_metrics.duration_sec and track.duration_sec is None:
                    track.duration_sec = existing_result.acoustic_metrics.duration_sec
                    existing_result.duration_sec = existing_result.acoustic_metrics.duration_sec

                # 4. Loại tác phẩm (Song Type)
                if track.song_type and not getattr(existing_result, "song_type", None):
                    existing_result.song_type = track.song_type
                elif getattr(existing_result, "song_type", None) and not track.song_type:
                    track.song_type = existing_result.song_type
                elif not getattr(existing_result, "song_type", None) and getattr(existing_result, "work_label", None):
                    existing_result.song_type = existing_result.work_label
                    track.song_type = existing_result.work_label

                # 5. Nhà phân phối
                if track.distributor and not getattr(existing_result, "distributor", None):
                    existing_result.distributor = track.distributor
                elif getattr(existing_result, "distributor", None) and not track.distributor:
                    track.distributor = existing_result.distributor

                # 6. Ngày phát hành
                if track.release_date and not getattr(existing_result, "release_date", None):
                    existing_result.release_date = track.release_date
                elif getattr(existing_result, "release_date", None) and not track.release_date:
                    track.release_date = existing_result.release_date

                # Lưu cập nhật đồng bộ lại vào CSDL
                track.status = "COMPLETED"
                try:
                    self.track_repo.save_result(existing_result)
                    self.track_repo.save_track(track)
                except Exception:
                    pass

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
                            if (
                                getattr(existing_after_resolve, "work_type", WorkType.UNCERTAIN)
                                == WorkType.UNCERTAIN
                            ):
                                meta_w = {
                                    "distributor": track.distributor
                                    or getattr(existing_after_resolve, "distributor", ""),
                                    "title": track.track_title or track.raw_query,
                                    "artist": track.artist or "",
                                    "song_type": getattr(track, "song_type", None),
                                }
                                w_type, w_label, is_cov, orig_a, orig_t = (
                                    resolve_work_type_from_metadata(
                                        title=track.track_title or track.raw_query,
                                        artist=track.artist or "",
                                        metadata=meta_w,
                                    )
                                )
                                existing_after_resolve.work_type = w_type
                                existing_after_resolve.work_label = w_label
                                existing_after_resolve.is_cover = is_cov
                                if (
                                    not getattr(existing_after_resolve, "original_artist", None)
                                    and orig_a
                                ):
                                    existing_after_resolve.original_artist = orig_a
                                if (
                                    not getattr(existing_after_resolve, "original_song_title", None)
                                    and orig_t
                                ):
                                    existing_after_resolve.original_song_title = orig_t
                                try:
                                    self.track_repo.save_result(existing_after_resolve)
                                except Exception:  # noqa: BLE001, S110
                                    pass
                            # Đồng bộ 2 chiều các trường metadata giữa track và existing_after_resolve:
                            if track.resolved_url and not getattr(existing_after_resolve, "source_url", None):
                                existing_after_resolve.source_url = track.resolved_url
                            elif getattr(existing_after_resolve, "source_url", None) and not track.resolved_url:
                                track.resolved_url = existing_after_resolve.source_url

                            if track.track_title and (not existing_after_resolve.track_title or existing_after_resolve.track_title == "N/A" or existing_after_resolve.track_title == existing_after_resolve.raw_query):
                                existing_after_resolve.track_title = track.track_title
                            elif getattr(existing_after_resolve, "track_title", None) and existing_after_resolve.track_title != "N/A" and not track.track_title:
                                track.track_title = existing_after_resolve.track_title

                            if track.duration_sec is not None and getattr(existing_after_resolve, "duration_sec", None) is None:
                                existing_after_resolve.duration_sec = track.duration_sec
                            elif getattr(existing_after_resolve, "duration_sec", None) is not None and track.duration_sec is None:
                                track.duration_sec = existing_after_resolve.duration_sec
                            elif existing_after_resolve.acoustic_metrics and existing_after_resolve.acoustic_metrics.duration_sec and track.duration_sec is None:
                                track.duration_sec = existing_after_resolve.acoustic_metrics.duration_sec
                                existing_after_resolve.duration_sec = existing_after_resolve.acoustic_metrics.duration_sec

                            if track.song_type and not getattr(existing_after_resolve, "song_type", None):
                                existing_after_resolve.song_type = track.song_type
                            elif getattr(existing_after_resolve, "song_type", None) and not track.song_type:
                                track.song_type = existing_after_resolve.song_type
                            elif not getattr(existing_after_resolve, "song_type", None) and getattr(existing_after_resolve, "work_label", None):
                                existing_after_resolve.song_type = existing_after_resolve.work_label
                                track.song_type = existing_after_resolve.work_label

                            if track.distributor and not getattr(existing_after_resolve, "distributor", None):
                                existing_after_resolve.distributor = track.distributor
                            elif getattr(existing_after_resolve, "distributor", None) and not track.distributor:
                                track.distributor = existing_after_resolve.distributor

                            if track.release_date and not getattr(existing_after_resolve, "release_date", None):
                                existing_after_resolve.release_date = track.release_date
                            elif getattr(existing_after_resolve, "release_date", None) and not track.release_date:
                                track.release_date = existing_after_resolve.release_date

                            track.status = "COMPLETED"
                            try:
                                self.track_repo.save_result(existing_after_resolve)
                                self.track_repo.save_track(track)
                            except Exception:
                                pass

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

            # Auto-enrich distributor nếu chưa có từ file đầu vào
            if not getattr(track, "distributor", None):
                cand_dist = (
                    metadata.get("content_id_licensed_by")
                    or metadata.get("distributor")
                    or metadata.get("channel")
                    or metadata.get("uploader")
                )
                if cand_dist:
                    cand_dist_str = str(cand_dist).strip()
                    if cand_dist_str and cand_dist_str.lower() not in ("none", "nan", "null"):
                        track.distributor = cand_dist_str

            # Auto-enrich release_date nếu chưa có từ file đầu vào
            if not getattr(track, "release_date", None):
                cand_date = metadata.get("release_date") or metadata.get("upload_date")
                if cand_date:
                    from .resolver_service import _normalize_date_str

                    norm_date = _normalize_date_str(cand_date)
                    if norm_date:
                        track.release_date = norm_date

            # Auto-enrich song_type nếu chưa có từ file đầu vào
            if not getattr(track, "song_type", None):
                cand_st = metadata.get("song_type") or metadata.get("work_type")
                if cand_st:
                    cand_st_str = str(cand_st).strip()
                    if cand_st_str and cand_st_str.lower() not in ("none", "nan", "null"):
                        track.song_type = cand_st_str

            # Auto-enrich duration_sec nếu chưa có
            if getattr(track, "duration_sec", None) is None:
                cand_dur = metadata.get("duration")
                if cand_dur and isinstance(cand_dur, (int, float)) and cand_dur > 0:
                    track.duration_sec = float(cand_dur)

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

            v_type_str = v_type.value if hasattr(v_type, "value") else str(v_type)
            w_type, w_label, is_cov, orig_a, orig_t = resolve_work_type_from_metadata(
                title=track.track_title or track.raw_query,
                artist=track.artist or "",
                metadata=metadata,
            )
            if not getattr(track, "song_type", None) and w_type != WorkType.UNCERTAIN:
                track.song_type = w_type.value
                self.track_repo.save_track(track)

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
                ai_model_is_ai=None,
                ai_model_verdict="N/A (Heuristics)",
                consensus_is_ai=is_ai,
                consensus_verdict=v_type_str,
                work_type=w_type,
                work_label=w_label,
                is_cover=is_cov,
                original_artist=orig_a,
                original_song_title=orig_t,
                song_type=track.song_type or (w_type.value if w_type != WorkType.UNCERTAIN else None),
                distributor=track.distributor,
                release_date=track.release_date,
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

        # Ghi nhận kết quả nguyên bản do Model AI đánh giá độc lập trước khi qua các tầng Trọng tài / Tiêu chí cứng
        raw_ai_is_ai = result.is_ai
        raw_ai_verdict = (
            result.verdict_type.value
            if hasattr(result.verdict_type, "value")
            else str(result.verdict_type)
        )
        result.ai_model_is_ai = raw_ai_is_ai
        result.ai_model_verdict = raw_ai_verdict

        if getattr(result, "input_type", None) is None:
            result.input_type = track.input_type
        if getattr(result, "raw_query", None) is None:
            result.raw_query = track.raw_query
        if getattr(result, "source_url", None) is None or not result.source_url:
            result.source_url = source_url_val

        # 5. Bộ lọc Trọng tài Kiểm chéo hai chiều (Cross-Validation Safeguard)
        result = self._apply_cross_validation_safeguard(result, track, metadata, metrics)

        # 6. Đảm bảo phân loại Tác phẩm (Work Type) luôn được giải mã chuẩn xác nếu mô hình để trống
        if getattr(result, "work_type", WorkType.UNCERTAIN) == WorkType.UNCERTAIN:
            w_type, w_label, is_cov, orig_a, orig_t = resolve_work_type_from_metadata(
                title=track.track_title or track.raw_query,
                artist=track.artist or "",
                metadata=metadata,
                raw_gemini_work_type=str(getattr(result, "work_type", "UNCERTAIN")),
            )
            result.work_type = w_type
            result.work_label = w_label
            result.is_cover = is_cov
            if not result.original_artist and orig_a:
                result.original_artist = orig_a
            if not result.original_song_title and orig_t:
                result.original_song_title = orig_t

        # Đồng bộ kết luận tổng hợp cuối cùng
        if not getattr(track, "song_type", None) and result.work_type != WorkType.UNCERTAIN:
            track.song_type = result.work_type.value

        result.song_type = track.song_type or (
            result.work_type.value if result.work_type != WorkType.UNCERTAIN else None
        )
        result.distributor = track.distributor
        result.release_date = track.release_date

        dur_val = getattr(track, "duration_sec", None) or (
            result.acoustic_metrics.duration_sec if result.acoustic_metrics else None
        )
        if dur_val is not None:
            result.duration_sec = dur_val
            track.duration_sec = dur_val

        result.consensus_is_ai = result.is_ai
        result.consensus_verdict = (
            result.verdict_type.value
            if hasattr(result.verdict_type, "value")
            else str(result.verdict_type)
        )

        track.status = "COMPLETED"
        try:
            self.track_repo.save_track(track)
        except Exception:
            pass
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
            get_dynamic_catalog,
            is_verified_human_artist,
        )

        is_human_artist = is_verified_human_artist(artist_str, title_str)
        is_major_exclusive_label = any(
            w in distributor_str for w in ["sony", "universal", "umg", "vieent", "loops", "bmz"]
        )

        _, ai_dist_list, ai_artist_regexes = get_dynamic_catalog()
        is_ai_dist = any(ad in distributor_str for ad in ai_dist_list)
        if not is_ai_dist:
            for pat, _ in ai_artist_regexes:
                if re.search(pat, artist_str.lower()) or re.search(pat, title_str.lower()):
                    is_ai_dist = True
                    break

        # 1. Safeguard Anti-FP (Bảo vệ nhạc người thật khỏi bị vu oan do nén lossy hoặc Auto-Tune)
        if result.is_ai:
            first_reason = (result.reasons[0] if result.reasons else "").lower()

            # 1a. Bảo vệ ca sĩ người thật đã xác thực hoặc hãng đĩa lớn độc quyền
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
                    result.consensus_is_ai = False
                    result.consensus_verdict = "PROCESSED_HUMAN"
                    return result

            # 1b. Bản thu phái sinh/remix có vocal người thật và chỉ số vật lý sạch
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
            if (is_derivative or is_vocal_human_suspect_beat) and is_clean_physical:
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
                result.consensus_is_ai = False
                result.consensus_verdict = "PROCESSED_HUMAN"
                return result

        # 2. Safeguard Anti-FN (Bắt AI lẩn khuất - Đồng thuận đa tiêu chí)
        if not result.is_ai:
            stem_bleed = getattr(metrics, "stem_bleed_score", 0.0) or 0.0
            rigidity = getattr(metrics, "pitch_rigidity_ratio", 0.0) or 0.0
            has_cutoff = getattr(metrics, "ultrasonic_cutoff_detected", False)
            rolloff = metrics.spectral_rolloff_95_hz

            # Kiểm tra xem AI Model có phát hiện bằng chứng sinh học thanh quản tự nhiên không
            all_reasons_text = " ".join(
                (result.reasons or []) + [result.detailed_explanation or ""]
            ).lower()
            has_human_bio_evidence = any(
                kw in all_reasons_text
                for kw in [
                    "thanh quản sinh học",
                    "sinh học tự nhiên",
                    "vocal tract",
                    "biological",
                    "auto-tune phòng thu",
                    "auto-tuned (natural vocal tract)",
                    "natural vocal",
                    "giọng người thật",
                    "cấu trúc thanh quản",
                    "thanh quản",
                ]
            )

            # 2a. Tiêu chí 1: AI thuần nhạc cụ / ambient (Chỉ áp dụng khi bản thu không có vocal)
            is_instrumental_track = bool(getattr(metrics, "is_instrumental", False))
            is_instrumental_ai = is_instrumental_track and rolloff < 6500.0 and rigidity >= 0.75

            # 2b. Tiêu chí 2: AI có dấu vân răng lược hoặc stem bleed cao áp đảo
            strong_physical_ai = (
                (metrics.fakeprint_score >= 0.50 and stem_bleed >= 0.35)
                or (has_cutoff and metrics.fakeprint_score >= 0.60)
                or (metrics.fakeprint_score >= 0.65 and stem_bleed >= 0.30)
            )

            # 2c. Tiêu chí 3: Đồng thuận đa chỉ số mức trung bình (Borderline)
            harm_ratio = getattr(metrics, "harmonic_ratio", 0.0) or 0.0
            borderline_ai = (
                rigidity >= 0.47
                and harm_ratio >= 0.80
                and (metrics.fakeprint_score >= 0.35 and stem_bleed >= 0.32)
            )

            # Trọng tài Anti-FN can thiệp:
            # - Tuyệt đối không can thiệp nếu là nghệ sĩ người thật đã xác thực (trừ khi fakeprint cực đoan >= 0.85)
            # - Nếu có bằng chứng AI danh mục (is_ai_dist), AI nhạc cụ (is_instrumental_ai), hoặc AI vật lý mạnh (strong_physical_ai):
            #   Bằng chứng vật lý/danh mục có quyền phủ quyết (override) phán quyết sinh học của LLM.
            # - Nếu chỉ là bằng chứng mức trung bình (borderline_ai): Cần đảm bảo không có bằng chứng sinh học và không thuộc nhãn đĩa lớn.
            should_override_to_ai = False
            override_reason = ""

            if not is_human_artist:
                if is_ai_dist:
                    should_override_to_ai = True
                    override_reason = (
                        "[CẢNH BÁO DANH MỤC AI]: Nghệ sĩ hoặc bản phát hành thuộc danh mục dự án AI / nhà phân phối ảo. "
                        f"Chỉ số vật lý: Fakeprint {metrics.fakeprint_score * 100:.1f}%, Stem Bleed {stem_bleed:.2f}."
                    )
                elif is_instrumental_ai:
                    should_override_to_ai = True
                    override_reason = (
                        "[CẢNH BÁO ÂM HỌC NHẠC CỤ AI]: Tần số suy giảm đột ngột và độ cứng cao bất thường "
                        f"(Rolloff: {rolloff:.0f} Hz, Rigidity: {rigidity:.2f}), đặc trưng của mô hình tạo sinh nhạc cụ/ambient."
                    )
                elif strong_physical_ai:
                    should_override_to_ai = True
                    override_reason = (
                        "[CẢNH BÁO ÂM HỌC VẬT LÝ]: Bằng chứng vật lý toán học xác nhận dấu vết AI vượt trội "
                        f"(Fakeprint: {metrics.fakeprint_score * 100:.1f}%, Stem Bleed: {stem_bleed:.2f}). "
                        "Dấu vết kỹ thuật deconvolution và intermodulation stem bleed áp đảo phán quyết ban đầu."
                    )
                elif borderline_ai and not has_human_bio_evidence and not is_major_exclusive_label:
                    should_override_to_ai = True
                    override_reason = (
                        "[CẢNH BÁO ĐỒNG THUẬN ĐA TIÊU CHÍ]: Phát hiện đồng thuận các chỉ số vật lý AI "
                        f"(Rigidity: {rigidity:.2f}, Fakeprint: {metrics.fakeprint_score * 100:.1f}%, Stem Bleed: {stem_bleed:.2f})."
                    )

            if should_override_to_ai:
                logger.info(
                    "Kích hoạt Trọng tài Hậu kiểm (Anti-FN): Bản thu '%s' có bằng chứng vật lý AI / danh mục vượt trội, nâng cấp lên AI.",
                    track.track_title,
                )
                result.is_ai = True
                result.verdict_type = VerdictType.HYBRID_AI
                result.verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
                result.reasons = [override_reason] + (result.reasons or [])
                if not result.suspected_engine or result.suspected_engine == "None":
                    result.suspected_engine = "Suno/Udio"
                result.confidence_score = max(0.90, result.confidence_score)

        result.consensus_is_ai = result.is_ai
        result.consensus_verdict = (
            result.verdict_type.value
            if hasattr(result.verdict_type, "value")
            else str(result.verdict_type)
        )
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
