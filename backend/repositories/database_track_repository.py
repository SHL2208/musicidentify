import logging
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select

from ..core.config import (
    CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD,
    TEMP_DIR,
    UPLOAD_AUDIO_SAMPLES_TO_CLOUD,
)
from ..core.database import get_db_session
from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.models.db_models import (
    AcousticMetricsModel,
    AIArtistProjectModel,
    BenchmarkItemModel,
    BenchmarkRunModel,
    DetectionResultModel,
    DistributorCatalogModel,
    MediaAssetModel,
    TrackModel,
    get_utc_now,
)
from ..core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from ..services.supabase_storage_service import SupabaseStorageService

logger = logging.getLogger(__name__)


class DatabaseTrackRepository(ITrackRepository):
    """
    Hiện thực ITrackRepository với CSDL PostgreSQL / Supabase Cloud.
    Hỗ trợ lưu trữ vĩnh viễn dữ liệu bài hát, kết quả thẩm định, chỉ số âm học và catalog.
    Tự động đồng bộ các file media (audio và ảnh phổ đồ) lên Supabase Storage.
    """

    def __init__(self):
        self._in_memory_cache_tracks: dict[str, TrackItem] = {}
        self._in_memory_cache_results: dict[str, DetectionResult] = {}
        self._storage_service = SupabaseStorageService()

    def save_track(self, track: TrackItem) -> TrackItem:
        """Lưu hoặc cập nhật bài hát vào CSDL Supabase."""
        self._in_memory_cache_tracks[track.id] = track
        try:
            with get_db_session() as session:
                existing = session.get(TrackModel, track.id)
                input_type_val = (
                    track.input_type.value
                    if hasattr(track.input_type, "value")
                    else str(track.input_type)
                )

                if existing:
                    if track.raw_query:
                        existing.raw_query = track.raw_query
                    if input_type_val:
                        existing.input_type = input_type_val

                    # URL: Giữ lại URL đã phân giải nếu track mới chưa có
                    if track.resolved_url:
                        existing.resolved_url = track.resolved_url
                    elif existing.resolved_url:
                        track.resolved_url = existing.resolved_url

                    # Tiêu đề bài hát
                    if track.track_title:
                        existing.track_title = track.track_title
                    elif existing.track_title and not track.track_title:
                        track.track_title = existing.track_title

                    # Ca sĩ / Nghệ sĩ
                    if track.artist:
                        existing.artist = track.artist
                    elif existing.artist:
                        track.artist = existing.artist

                    # Thời lượng bài hát
                    if track.duration_sec is not None and track.duration_sec > 0:
                        existing.duration_sec = track.duration_sec
                    elif existing.duration_sec is not None:
                        track.duration_sec = existing.duration_sec

                    # Nền tảng
                    if track.platform:
                        existing.platform = track.platform
                    elif existing.platform:
                        track.platform = existing.platform

                    # Loại tác phẩm (Song Type)
                    if track.song_type:
                        existing.song_type = track.song_type
                    elif existing.song_type and not track.song_type:
                        track.song_type = existing.song_type

                    # Ngày phát hành
                    if track.release_date:
                        existing.release_date = track.release_date
                    elif existing.release_date and not track.release_date:
                        track.release_date = existing.release_date

                    # Nhà phân phối
                    if track.distributor:
                        existing.distributor = track.distributor
                    elif existing.distributor and not track.distributor:
                        track.distributor = existing.distributor

                    # Match score & verification
                    if track.is_verified_match is not None:
                        existing.is_verified_match = track.is_verified_match
                    elif existing.is_verified_match is not None:
                        track.is_verified_match = existing.is_verified_match

                    if track.match_score is not None:
                        existing.match_score = track.match_score
                    elif existing.match_score is not None:
                        track.match_score = existing.match_score

                    # Trạng thái: Không hạ cấp từ COMPLETED về PROCESSING nếu không có lỗi
                    if track.status:
                        if existing.status == "COMPLETED" and track.status == "PROCESSING":
                            pass  # Giữ nguyên COMPLETED
                        else:
                            existing.status = track.status
                    existing.updated_at = get_utc_now()
                else:
                    new_track = TrackModel(
                        id=track.id,
                        raw_query=track.raw_query,
                        input_type=input_type_val,
                        resolved_url=track.resolved_url,
                        track_title=track.track_title,
                        artist=track.artist,
                        duration_sec=track.duration_sec,
                        platform=track.platform,
                        song_type=track.song_type,
                        release_date=track.release_date,
                        distributor=track.distributor,
                        is_verified_match=track.is_verified_match,
                        match_score=track.match_score,
                        status=track.status,
                    )
                    session.add(new_track)
        except Exception as e:
            logger.warning(f"Lỗi khi lưu track vào CSDL (đã lưu cache): {e}")

        return track

    def get_track(self, track_id: str) -> TrackItem | None:
        """Lấy bài hát theo ID từ CSDL hoặc Cache."""
        if track_id in self._in_memory_cache_tracks:
            return self._in_memory_cache_tracks[track_id]

        try:
            with get_db_session() as session:
                tm = session.get(TrackModel, track_id)
                if tm:
                    inp_type = InputType.URL
                    try:
                        inp_type = InputType(tm.input_type)
                    except ValueError:
                        pass

                    track = TrackItem(
                        id=tm.id,
                        raw_query=tm.raw_query,
                        input_type=inp_type,
                        resolved_url=tm.resolved_url,
                        track_title=tm.track_title,
                        artist=tm.artist,
                        duration_sec=tm.duration_sec,
                        platform=tm.platform,
                        song_type=tm.song_type,
                        release_date=tm.release_date,
                        distributor=tm.distributor,
                        is_verified_match=tm.is_verified_match,
                        match_score=tm.match_score,
                        status=tm.status or "PENDING",
                    )
                    self._in_memory_cache_tracks[track_id] = track
                    return track
        except Exception as e:
            logger.warning(f"Lỗi khi lấy track từ CSDL: {e}")

        return None

    def save_result(self, result: DetectionResult) -> DetectionResult:
        """Lưu hoặc cập nhật kết quả thẩm định kèm các chỉ số âm học và asset."""
        self._in_memory_cache_results[result.track_id] = result
        try:
            with get_db_session() as session:
                # 1. Đảm bảo track cha tồn tại và đồng bộ đầy đủ siêu dữ liệu
                track_rec = session.get(TrackModel, result.track_id)
                dur_val = getattr(result, "duration_sec", None) or (
                    result.acoustic_metrics.duration_sec if result.acoustic_metrics else None
                )
                if not track_rec:
                    inp_type_str = (
                        result.input_type.value
                        if result.input_type and hasattr(result.input_type, "value")
                        else str(result.input_type or "SONG_NAME")
                    )
                    track_rec = TrackModel(
                        id=result.track_id,
                        raw_query=result.raw_query or result.track_title,
                        input_type=inp_type_str,
                        resolved_url=result.source_url,
                        track_title=result.track_title,
                        song_type=result.song_type,
                        distributor=result.distributor,
                        release_date=result.release_date,
                        duration_sec=dur_val,
                        status="COMPLETED",
                    )
                    session.add(track_rec)
                    session.flush()
                else:
                    if result.source_url:
                        track_rec.resolved_url = result.source_url
                    elif track_rec.resolved_url and not result.source_url:
                        result.source_url = track_rec.resolved_url

                    if result.track_title:
                        track_rec.track_title = result.track_title
                    elif track_rec.track_title and not result.track_title:
                        result.track_title = track_rec.track_title

                    if result.song_type:
                        track_rec.song_type = result.song_type
                    elif track_rec.song_type and not result.song_type:
                        result.song_type = track_rec.song_type

                    if result.distributor:
                        track_rec.distributor = result.distributor
                    elif track_rec.distributor and not result.distributor:
                        result.distributor = track_rec.distributor

                    if result.release_date:
                        track_rec.release_date = result.release_date
                    elif track_rec.release_date and not result.release_date:
                        result.release_date = track_rec.release_date

                    if dur_val is not None and dur_val > 0:
                        track_rec.duration_sec = dur_val
                    elif track_rec.duration_sec and getattr(result, "duration_sec", None) is None:
                        result.duration_sec = track_rec.duration_sec

                    track_rec.status = "COMPLETED"
                    track_rec.updated_at = get_utc_now()

                # 2. Lưu hoặc cập nhật DetectionResultModel
                res_id = f"res_{result.track_id}"
                existing_res = session.get(DetectionResultModel, res_id)
                v_type_str = (
                    result.verdict_type.value
                    if hasattr(result.verdict_type, "value")
                    else str(result.verdict_type)
                )

                if existing_res:
                    existing_res.is_ai = result.is_ai
                    existing_res.verdict_type = v_type_str
                    existing_res.verdict_label = result.verdict_label
                    existing_res.confidence_score = result.confidence_score
                    existing_res.suspected_engine = result.suspected_engine
                    existing_res.reasons = result.reasons or []
                    existing_res.detailed_explanation = result.detailed_explanation
                    existing_res.model_used = result.model_used
                    existing_res.ai_model_is_ai = result.ai_model_is_ai
                    existing_res.ai_model_verdict = result.ai_model_verdict
                    existing_res.consensus_is_ai = result.consensus_is_ai
                    existing_res.consensus_verdict = result.consensus_verdict
                    existing_res.processed_at = get_utc_now()
                else:
                    existing_res = DetectionResultModel(
                        id=res_id,
                        track_id=result.track_id,
                        is_ai=result.is_ai,
                        verdict_type=v_type_str,
                        verdict_label=result.verdict_label,
                        confidence_score=result.confidence_score,
                        suspected_engine=result.suspected_engine,
                        reasons=result.reasons or [],
                        detailed_explanation=result.detailed_explanation,
                        model_used=result.model_used,
                        ai_model_is_ai=result.ai_model_is_ai,
                        ai_model_verdict=result.ai_model_verdict,
                        consensus_is_ai=result.consensus_is_ai,
                        consensus_verdict=result.consensus_verdict,
                    )
                    session.add(existing_res)
                    session.flush()

                # 3. Lưu hoặc cập nhật AcousticMetricsModel
                if result.acoustic_metrics:
                    m = result.acoustic_metrics
                    metrics_id = f"m_{res_id}"
                    existing_m = session.get(AcousticMetricsModel, metrics_id)
                    if existing_m:
                        existing_m.sample_rate = m.sample_rate
                        existing_m.duration_sec = m.duration_sec
                        existing_m.fakeprint_score = m.fakeprint_score
                        existing_m.fakeprint_detected = m.fakeprint_detected
                        existing_m.spectral_rolloff_95_hz = m.spectral_rolloff_95_hz
                        existing_m.spectral_rolloff_85_hz = m.spectral_rolloff_85_hz
                        existing_m.harmonic_ratio = m.harmonic_ratio
                        existing_m.pitch_jitter = m.pitch_jitter
                        existing_m.pitch_rigidity_ratio = m.pitch_rigidity_ratio
                        existing_m.pitch_anomaly_detected = m.pitch_anomaly_detected
                        existing_m.pitch_stability_label = m.pitch_stability_label
                        existing_m.is_studio_autotune = m.is_studio_autotune
                        existing_m.stem_bleed_score = m.stem_bleed_score
                        existing_m.stem_bleed_detected = m.stem_bleed_detected
                        existing_m.ultrasonic_cutoff_detected = m.ultrasonic_cutoff_detected
                        existing_m.compression_source = m.compression_source
                    else:
                        new_m = AcousticMetricsModel(
                            id=metrics_id,
                            result_id=res_id,
                            sample_rate=m.sample_rate,
                            duration_sec=m.duration_sec,
                            fakeprint_score=m.fakeprint_score,
                            fakeprint_detected=m.fakeprint_detected,
                            spectral_rolloff_95_hz=m.spectral_rolloff_95_hz,
                            spectral_rolloff_85_hz=m.spectral_rolloff_85_hz,
                            harmonic_ratio=m.harmonic_ratio,
                            pitch_jitter=m.pitch_jitter,
                            pitch_rigidity_ratio=m.pitch_rigidity_ratio,
                            pitch_anomaly_detected=m.pitch_anomaly_detected,
                            pitch_stability_label=m.pitch_stability_label,
                            is_studio_autotune=m.is_studio_autotune,
                            stem_bleed_score=m.stem_bleed_score,
                            stem_bleed_detected=m.stem_bleed_detected,
                            ultrasonic_cutoff_detected=m.ultrasonic_cutoff_detected,
                            compression_source=m.compression_source,
                        )
                        session.add(new_m)

                # 4. Tải lên và Lưu MediaAssetModel cho ảnh quang phổ & file audio cắt lát
                cleanup_local = CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD

                # a) Ảnh phổ đồ (Spectrogram)
                if result.spectrogram_image_path:
                    spec_path = Path(result.spectrogram_image_path)
                    public_spec_url = str(result.spectrogram_image_path)

                    if spec_path.exists() and not public_spec_url.startswith("http"):
                        uploaded_url = self._storage_service.upload_file(
                            spec_path, bucket="spectrograms"
                        )
                        if uploaded_url.startswith("http"):
                            public_spec_url = uploaded_url
                            result.spectrogram_image_path = public_spec_url
                            if cleanup_local:
                                spec_path.unlink(missing_ok=True)

                    asset_id = f"asset_spec_{result.track_id}"
                    existing_asset = session.get(MediaAssetModel, asset_id)
                    if existing_asset:
                        existing_asset.public_url = public_spec_url
                    else:
                        new_asset = MediaAssetModel(
                            id=asset_id,
                            track_id=result.track_id,
                            asset_type="SPECTROGRAM_FORENSIC",
                            file_path=str(spec_path),
                            public_url=public_spec_url,
                        )
                        session.add(new_asset)

                # b) File Audio cắt lát đa phân đoạn (Multi-Segment Audio Sample)
                audio_sample_path = TEMP_DIR / f"{result.track_id}_multisegment.wav"
                if audio_sample_path.exists():
                    file_sz = audio_sample_path.stat().st_size
                    uploaded_audio_url = str(audio_sample_path)

                    if UPLOAD_AUDIO_SAMPLES_TO_CLOUD:
                        uploaded_audio_url = self._storage_service.upload_file(
                            audio_sample_path, bucket="audio-samples"
                        )
                        audio_asset_id = f"asset_audio_{result.track_id}"
                        existing_audio_asset = session.get(MediaAssetModel, audio_asset_id)
                        if existing_audio_asset:
                            existing_audio_asset.public_url = uploaded_audio_url
                        else:
                            new_audio_asset = MediaAssetModel(
                                id=audio_asset_id,
                                track_id=result.track_id,
                                asset_type="AUDIO_SAMPLE",
                                file_path=str(audio_sample_path),
                                public_url=uploaded_audio_url,
                                file_size=file_sz,
                            )
                            session.add(new_audio_asset)

                    if cleanup_local:
                        audio_sample_path.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Lỗi khi lưu DetectionResult vào CSDL (đã lưu cache): {e}")

        return result

    def get_result(self, track_id: str) -> DetectionResult | None:
        """Lấy kết quả thẩm định theo track_id."""
        if track_id in self._in_memory_cache_results:
            return self._in_memory_cache_results[track_id]

        try:
            with get_db_session() as session:
                res_id = f"res_{track_id}"
                rm = session.get(DetectionResultModel, res_id)
                if rm:
                    return self._map_model_to_detection_result(session, rm)
        except Exception as e:
            logger.warning(f"Lỗi khi lấy kết quả từ CSDL: {e}")

        return None

    def list_results(self) -> list[DetectionResult]:
        """Lấy toàn bộ kết quả thẩm định."""
        try:
            with get_db_session() as session:
                stmt = (
                    select(DetectionResultModel)
                    .order_by(DetectionResultModel.processed_at.desc())
                    .limit(100)
                )
                models = session.scalars(stmt).all()
                if models:
                    return [self._map_model_to_detection_result(session, m) for m in models]
        except Exception as e:
            logger.warning(f"Lỗi khi lấy danh sách kết quả từ CSDL: {e}")

        return list(self._in_memory_cache_results.values())

    def find_result_by_query_or_url(
        self, query: str, resolved_url: str | None = None, track_id: str | None = None
    ) -> DetectionResult | None:
        """Tìm kiếm kết quả thẩm định đã có theo câu truy vấn, URL nguồn hoặc track_id."""
        clean_q = (query or "").strip()
        clean_u = (resolved_url or "").strip()
        clean_id = (track_id or "").strip()
        if not clean_q and not clean_u and not clean_id:
            return None

        # 0. Nếu có track_id, kiểm tra cache bộ nhớ trước
        if clean_id and clean_id in self._in_memory_cache_results:
            cached = self._in_memory_cache_results[clean_id]
            cached.is_cached = True
            return cached

        # Trích xuất YouTube Video ID nếu có
        yt_id = None
        for cand in [clean_u, clean_q, clean_id]:
            if "youtube.com" in cand or "youtu.be" in cand:
                m = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", cand)
                if m:
                    yt_id = m.group(1)
                    break

        # 1. Kiểm tra cache bộ nhớ phiên làm việc trước
        for t_id, track in self._in_memory_cache_tracks.items():
            t_query = (track.raw_query or "").strip().lower()
            t_url = (track.resolved_url or "").strip().lower()
            t_title = (track.track_title or "").strip().lower()
            q_lower = clean_q.lower()
            u_lower = clean_u.lower()
            yt_match = yt_id and (yt_id in t_url or yt_id in t_query or t_id == f"yt_{yt_id}")
            if (
                (clean_id and t_id == clean_id)
                or yt_match
                or (u_lower and (u_lower == t_url or u_lower == t_query))
                or (q_lower and (q_lower == t_query or q_lower == t_url or q_lower == t_title))
            ):
                cached = self._in_memory_cache_results.get(t_id)
                if cached:
                    cached.is_cached = True
                    return cached

        # 2. Truy vấn từ CSDL Supabase / PostgreSQL
        try:
            with get_db_session() as session:
                conditions = []
                if clean_id:
                    conditions.append(TrackModel.id == clean_id)
                if yt_id:
                    conditions.append(TrackModel.id == f"yt_{yt_id}")
                    conditions.append(TrackModel.resolved_url.ilike(f"%{yt_id}%"))
                    conditions.append(TrackModel.raw_query.ilike(f"%{yt_id}%"))
                if clean_u:
                    conditions.append(TrackModel.resolved_url == clean_u)
                    conditions.append(TrackModel.raw_query == clean_u)
                if clean_q:
                    conditions.append(func.lower(TrackModel.raw_query) == clean_q.lower())
                    conditions.append(func.lower(TrackModel.track_title) == clean_q.lower())
                    conditions.append(TrackModel.resolved_url == clean_q)
                    conditions.append(TrackModel.id == clean_q)

                stmt = (
                    select(TrackModel)
                    .where(or_(*conditions))
                    .order_by(TrackModel.updated_at.desc())
                    .limit(1)
                )
                tm = session.scalars(stmt).first()
                if tm:
                    rm = getattr(tm, "detection_result", None) or session.get(
                        DetectionResultModel, f"res_{tm.id}"
                    )
                    if rm:
                        res = self._map_model_to_detection_result(session, rm)
                        res.is_cached = True
                        inp_type = InputType.URL
                        if tm.input_type:
                            try:
                                inp_type = InputType(tm.input_type)
                            except ValueError:
                                pass
                        self._in_memory_cache_tracks[tm.id] = TrackItem(
                            id=tm.id,
                            raw_query=tm.raw_query,
                            input_type=inp_type,
                            resolved_url=tm.resolved_url,
                            track_title=tm.track_title,
                            artist=tm.artist,
                            duration_sec=tm.duration_sec,
                            platform=tm.platform,
                            is_verified_match=tm.is_verified_match,
                            match_score=tm.match_score,
                            status=tm.status,
                            song_type=tm.song_type,
                            distributor=tm.distributor,
                            release_date=tm.release_date,
                        )
                        self._in_memory_cache_results[tm.id] = res
                        return res
        except Exception as e:
            logger.warning(f"Lỗi khi tra cứu bài hát từ CSDL: {e}")

        return None

    def correct_verdict(
        self,
        track_id: str,
        is_ai: bool,
        verdict_type: str,
        verdict_label: str,
        note: str | None = None,
    ) -> DetectionResult | None:
        """Cập nhật đính chính phán quyết của bài hát trực tiếp vào CSDL."""
        try:
            with get_db_session() as session:
                res_id = f"res_{track_id}"
                rm = session.get(DetectionResultModel, res_id)
                if not rm:
                    return None

                rm.is_ai = is_ai
                rm.verdict_type = verdict_type
                rm.verdict_label = verdict_label
                reasons_list = list(rm.reasons or [])
                correction_reason = (
                    f"Chuyên gia đính chính: {note}"
                    if note
                    else "Đã được đính chính thủ công bởi chuyên gia/người dùng"
                )
                reasons_list.insert(0, correction_reason)
                rm.reasons = reasons_list
                session.commit()

                res = self._map_model_to_detection_result(session, rm)
                res.is_human_corrected = True
                self._in_memory_cache_results[track_id] = res
                return res
        except Exception as e:
            logger.error(f"Lỗi khi đính chính phán quyết bài {track_id}: {e}")
            return None

    def clear(self) -> None:
        """Xóa cache bộ nhớ phiên làm việc."""
        self._in_memory_cache_tracks.clear()
        self._in_memory_cache_results.clear()

    # --- CÁC HÀM TIỆN ÍCH DÀNH CHO CATALOG & BENCHMARK ---

    def get_distributor_catalog(self) -> list[dict[str, Any]]:
        """Lấy danh sách các nhà phân phối AI & Whitelist từ CSDL."""
        try:
            with get_db_session() as session:
                stmt = select(DistributorCatalogModel).where(
                    DistributorCatalogModel.is_active.is_(True)
                )
                items = session.scalars(stmt).all()
                return [
                    {
                        "id": it.id,
                        "name": it.name,
                        "catalog_type": it.catalog_type,
                        "notes": it.notes,
                    }
                    for it in items
                ]
        except Exception as e:
            logger.warning(f"Lỗi lấy distributor catalog: {e}")
            return []

    def get_ai_artist_projects(self) -> list[dict[str, Any]]:
        """Lấy danh sách các dự án nghệ sĩ ảo AI sinh hàng loạt từ CSDL."""
        try:
            with get_db_session() as session:
                stmt = select(AIArtistProjectModel).where(AIArtistProjectModel.is_active.is_(True))
                items = session.scalars(stmt).all()
                return [
                    {
                        "id": it.id,
                        "project_name": it.project_name,
                        "regex_pattern": it.regex_pattern,
                        "suspected_engine": it.suspected_engine,
                        "confidence": it.confidence,
                    }
                    for it in items
                ]
        except Exception as e:
            logger.warning(f"Lỗi lấy ai_artist_projects catalog: {e}")
            return []

    def save_benchmark_run(self, summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
        """Lưu toàn bộ đợt chạy Benchmark vào CSDL."""
        run_id = f"bm_{uuid.uuid4().hex[:12]}"
        try:
            with get_db_session() as session:
                run_rec = BenchmarkRunModel(
                    id=run_id,
                    run_name=summary.get("run_name", "Benchmark Evaluation"),
                    dataset_source=summary.get("dataset_source", "testcase.xlsx"),
                    total_samples=summary.get("total_samples", len(results)),
                    evaluated_samples=summary.get("evaluated_successfully", len(results)),
                    error_count=summary.get("errors", 0),
                    tp=summary.get("TP", 0),
                    tn=summary.get("TN", 0),
                    fp=summary.get("FP", 0),
                    fn=summary.get("FN", 0),
                    accuracy=summary.get("accuracy", 0.0),
                    precision=summary.get("precision", 0.0),
                    recall=summary.get("recall", 0.0),
                    specificity=summary.get("specificity", 0.0),
                    f1_score=summary.get("f1_score", 0.0),
                    confusion_matrix_image_url=summary.get("confusion_matrix_image_path"),
                )
                session.add(run_rec)

                for idx, r in enumerate(results):
                    item_id = f"{run_id}_{idx}"
                    b_item = BenchmarkItemModel(
                        id=item_id,
                        benchmark_run_id=run_id,
                        media_id=str(r.get("Mã bài") or r.get("media_id") or ""),
                        track_title=str(r.get("Tên bài hát") or r.get("song_title") or ""),
                        artist=str(r.get("Ca sĩ") or r.get("artist") or ""),
                        ground_truth=str(r.get("Thực tế") or r.get("ground_truth") or ""),
                        prediction=str(r.get("Dự đoán") or r.get("prediction") or ""),
                        classification_type=str(
                            r.get("Phân loại") or r.get("classification_type") or ""
                        ),
                        is_correct=str(r.get("Kết quả") or r.get("is_correct") or ""),
                        source_url=str(r.get("Nguồn link") or r.get("source_url") or ""),
                    )
                    session.add(b_item)

                logger.info(f"Đã lưu đợt Benchmark {run_id} với {len(results)} bài vào CSDL.")
        except Exception as e:
            logger.error(f"Lỗi khi lưu Benchmark Run vào CSDL: {e}")

        return run_id

    def list_benchmark_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Lấy danh sách các đợt Benchmark đã chạy từ trước đến nay."""
        try:
            with get_db_session() as session:
                stmt = (
                    select(BenchmarkRunModel)
                    .order_by(BenchmarkRunModel.created_at.desc())
                    .limit(limit)
                )
                runs = session.scalars(stmt).all()
                return [
                    {
                        "id": r.id,
                        "run_name": r.run_name,
                        "dataset_source": r.dataset_source,
                        "total_samples": r.total_samples,
                        "evaluated_samples": r.evaluated_samples,
                        "accuracy": r.accuracy,
                        "precision": r.precision,
                        "recall": r.recall,
                        "f1_score": r.f1_score,
                        "tp": r.tp,
                        "tn": r.tn,
                        "fp": r.fp,
                        "fn": r.fn,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in runs
                ]
        except Exception as e:
            logger.warning(f"Lỗi lấy danh sách Benchmark runs: {e}")
            return []

    def _map_model_to_detection_result(
        self, session: Any, rm: DetectionResultModel
    ) -> DetectionResult:
        """Chuyển đổi từ DetectionResultModel sang Pydantic DetectionResult."""
        tm = session.get(TrackModel, rm.track_id)
        metrics_obj = None
        if rm.acoustic_metrics:
            am = rm.acoustic_metrics
            r95 = am.spectral_rolloff_95_hz or 0.0
            rig = am.pitch_rigidity_ratio or 0.0
            harm = am.harmonic_ratio or 0.0
            cut = am.ultrasonic_cutoff_detected or False
            is_suspect = (r95 < 11800.0 and rig >= 0.45 and harm >= 0.80) or (cut and rig >= 0.50)
            metrics_obj = AcousticMetrics(
                sample_rate=am.sample_rate or 44100,
                duration_sec=am.duration_sec or 30.0,
                fakeprint_score=am.fakeprint_score or 0.0,
                fakeprint_detected=am.fakeprint_detected or False,
                spectral_rolloff_95_hz=r95,
                spectral_rolloff_85_hz=am.spectral_rolloff_85_hz or 0.0,
                harmonic_ratio=harm,
                pitch_jitter=am.pitch_jitter or 0.0,
                pitch_rigidity_ratio=rig,
                pitch_anomaly_detected=am.pitch_anomaly_detected or False,
                pitch_stability_label=am.pitch_stability_label or "Natural Biological",
                is_studio_autotune=am.is_studio_autotune or False,
                stem_bleed_score=am.stem_bleed_score,
                stem_bleed_detected=am.stem_bleed_detected or False,
                ultrasonic_cutoff_detected=cut,
                compression_source=am.compression_source or "Studio Master / Uncompressed",
                modern_ai_vocal_suspect=is_suspect,
            )

        v_type = VerdictType.CONFIRMED_HUMAN
        try:
            v_type = VerdictType(rm.verdict_type)
        except ValueError:
            pass

        inp_type = InputType.URL
        if tm and tm.input_type:
            try:
                inp_type = InputType(tm.input_type)
            except ValueError:
                pass

        spec_url = None
        if tm and getattr(tm, "media_assets", None):
            for asset in tm.media_assets:
                if getattr(asset, "asset_type", None) == "SPECTROGRAM_FORENSIC":
                    spec_url = asset.public_url or asset.file_path
                    break

        reasons_list = rm.reasons or []
        is_corrected = any("đính chính" in str(r).lower() for r in reasons_list)

        return DetectionResult(
            track_id=rm.track_id,
            track_title=tm.track_title if tm else "N/A",
            source_url=tm.resolved_url if tm else None,
            input_type=inp_type,
            raw_query=tm.raw_query if tm else None,
            is_ai=rm.is_ai,
            verdict_type=v_type,
            verdict_label=rm.verdict_label,
            confidence_score=rm.confidence_score,
            suspected_engine=rm.suspected_engine or "None",
            reasons=reasons_list,
            detailed_explanation=rm.detailed_explanation or "",
            acoustic_metrics=metrics_obj,
            spectrogram_image_path=spec_url,
            processed_at=rm.processed_at.isoformat() if rm.processed_at else "",
            model_used=rm.model_used,
            is_human_corrected=is_corrected,
            ai_model_is_ai=rm.ai_model_is_ai,
            ai_model_verdict=rm.ai_model_verdict,
            consensus_is_ai=rm.consensus_is_ai,
            consensus_verdict=rm.consensus_verdict,
            song_type=tm.song_type if tm else None,
            distributor=tm.distributor if tm else None,
            release_date=tm.release_date if tm else None,
            duration_sec=tm.duration_sec if (tm and tm.duration_sec is not None) else (metrics_obj.duration_sec if metrics_obj else None),
        )
