import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..core.config import CLEANUP_LOCAL_MEDIA_AFTER_UPLOAD, TEMP_DIR
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
                    existing.raw_query = track.raw_query
                    existing.input_type = input_type_val
                    existing.resolved_url = track.resolved_url
                    existing.track_title = track.track_title
                    existing.artist = track.artist
                    existing.duration_sec = track.duration_sec
                    existing.platform = track.platform
                    existing.song_type = track.song_type
                    existing.release_date = track.release_date
                    existing.distributor = track.distributor
                    existing.is_verified_match = track.is_verified_match
                    existing.match_score = track.match_score
                    existing.status = track.status
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
                # 1. Đảm bảo track cha tồn tại
                track_rec = session.get(TrackModel, result.track_id)
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
                        status="COMPLETED",
                    )
                    session.add(track_rec)
                    session.flush()

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

                    if cleanup_local and uploaded_audio_url.startswith("http"):
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
            metrics_obj = AcousticMetrics(
                sample_rate=am.sample_rate or 44100,
                duration_sec=am.duration_sec or 30.0,
                fakeprint_score=am.fakeprint_score or 0.0,
                fakeprint_detected=am.fakeprint_detected or False,
                spectral_rolloff_95_hz=am.spectral_rolloff_95_hz or 0.0,
                spectral_rolloff_85_hz=am.spectral_rolloff_85_hz or 0.0,
                harmonic_ratio=am.harmonic_ratio or 0.0,
                pitch_jitter=am.pitch_jitter or 0.0,
                pitch_rigidity_ratio=am.pitch_rigidity_ratio or 0.0,
                pitch_anomaly_detected=am.pitch_anomaly_detected or False,
                pitch_stability_label=am.pitch_stability_label or "Natural Biological",
                is_studio_autotune=am.is_studio_autotune or False,
                stem_bleed_score=am.stem_bleed_score,
                stem_bleed_detected=am.stem_bleed_detected or False,
                ultrasonic_cutoff_detected=am.ultrasonic_cutoff_detected or False,
                compression_source=am.compression_source or "Studio Master / Uncompressed",
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
            reasons=rm.reasons or [],
            detailed_explanation=rm.detailed_explanation or "",
            acoustic_metrics=metrics_obj,
            spectrogram_image_path=None,
            processed_at=rm.processed_at.isoformat() if rm.processed_at else "",
            model_used=rm.model_used,
        )
