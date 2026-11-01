"""Kiểm thử xác thực trực tiếp trên 28 bài hát từng bị phân loại sai ở đợt audit trước.

Tuân thủ nghiêm ngặt PEP 8, Ruff và Type Safety.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

# Cấu hình UTF-8 cho Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd

from backend.core.container import get_container
from backend.core.database import get_db_session
from backend.core.models.db_models import (
    AcousticMetricsModel,
    DetectionResultModel,
    TrackModel,
)
from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)


def load_failed_audit_tracks(
    excel_path: str | Path = "data/outputs/benchmark_audit_20261101_061656.xlsx",
) -> list[dict[str, Any]]:
    """Đọc 28 ca bị đoán sai từ file Excel audit gần nhất."""
    p = Path(excel_path)
    if not p.is_absolute():
        p = root_dir / p
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file audit: {p}")

    df = pd.read_excel(p, sheet_name="Audit Details")
    failed = df[df["is_correct"] == "SAI"].copy()
    items: list[dict[str, Any]] = []
    for _, row in failed.iterrows():
        items.append(
            {
                "media_id": str(row["media_id"]),
                "song_title": str(row["song_title"]),
                "artist": str(row["artist"]) if pd.notna(row["artist"]) else "",
                "query": str(row["query"]),
                "ground_truth": str(row["ground_truth"]).strip().upper(),
                "previous_pred": str(row["prediction"]).strip().upper(),
                "previous_ai_model": str(row.get("ai_model_prediction", "")),
                "previous_reasons": str(row.get("reasons", "")),
                "source_url": str(row.get("source_url", "")),
            }
        )
    return items


def verify_tracks() -> None:
    """Chạy thẩm định lại toàn bộ 28 bài từng sai và đối soát kết quả Trước vs Sau."""
    failed_items = load_failed_audit_tracks()
    print("=" * 85)
    print(
        f"🎯 BẮT ĐẦU KIỂM THỬ XÁC THỰC TRÊN {len(failed_items)} BÀI HÁT TỪNG ĐOÁN SAI (AUDIT 20261101)"
    )
    print("=" * 85)

    container = get_container()
    detection_service = container.detection_service

    # Tải trước các acoustic metrics từ DB nếu có
    metrics_cache: dict[str, AcousticMetrics] = {}
    with get_db_session() as session:
        for item in failed_items:
            title = item["song_title"]
            mid = item["media_id"]
            # Tìm track
            t = session.query(TrackModel).filter(TrackModel.track_title.ilike(f"%{title}%")).first()
            if t:
                det = (
                    session.query(DetectionResultModel)
                    .filter(DetectionResultModel.track_id == t.id)
                    .first()
                )
                if det:
                    m = (
                        session.query(AcousticMetricsModel)
                        .filter(AcousticMetricsModel.result_id == det.id)
                        .first()
                    )
                    if m:
                        metrics_cache[mid] = AcousticMetrics(
                            duration_sec=float(m.duration_sec or 45.0),
                            sample_rate=int(m.sample_rate or 44100),
                            spectral_rolloff_95_hz=float(m.spectral_rolloff_95_hz or 11000.0),
                            spectral_rolloff_85_hz=float(m.spectral_rolloff_85_hz or 7000.0),
                            harmonic_ratio=float(m.harmonic_ratio or 0.85),
                            pitch_rigidity_ratio=float(m.pitch_rigidity_ratio or 0.50),
                            pitch_jitter=float(m.pitch_jitter or 0.08),
                            fakeprint_score=float(m.fakeprint_score or 0.20),
                            fakeprint_detected=bool(m.fakeprint_detected),
                            stem_bleed_score=float(m.stem_bleed_score or 0.0),
                            stem_bleed_detected=bool(m.stem_bleed_detected),
                            stereo_correlation=1.0,
                            compression_source="Opus/AAC",
                            has_brickwall_cutoff=False,
                            pitch_stability_label=str(m.pitch_stability_label or "Auto-Tuned"),
                            is_instrumental=False,
                            modern_ai_vocal_suspect=bool(
                                getattr(m, "modern_ai_vocal_suspect", False)
                            ),
                        )

    # Thống kê kết quả
    total_count = len(failed_items)
    fixed_count = 0
    still_wrong_count = 0

    results_table: list[dict[str, Any]] = []

    for idx, item in enumerate(failed_items, 1):
        mid = item["media_id"]
        title = item["song_title"]
        artist = item["artist"]
        gt = item["ground_truth"]
        prev_pred = item["previous_pred"]

        track = TrackItem(
            id=mid,
            raw_query=item["query"],
            track_title=title,
            artist=artist,
            input_type=InputType.SONG_NAME,
            resolved_url=item["source_url"] if item["source_url"] else None,
        )

        # Đánh giá qua DetectionService
        # Nếu đã có metrics lưu trong DB, áp dụng cơ chế Cross-Validation Safeguard với các cải tiến mới
        current_metrics = metrics_cache.get(mid)
        if current_metrics:
            # Tái tính toán cờ modern_ai_vocal_suspect theo ngưỡng mới
            current_metrics.modern_ai_vocal_suspect = (
                detection_service.signal_analyzer._is_modern_ai_vocal(
                    rolloff_95=current_metrics.spectral_rolloff_95_hz,
                    rigidity=current_metrics.pitch_rigidity_ratio,
                    harmonic_ratio=current_metrics.harmonic_ratio,
                    fakeprint_score=current_metrics.fakeprint_score,
                    stem_bleed=current_metrics.stem_bleed_score,
                    ultrasonic_cutoff=getattr(current_metrics, "ultrasonic_cutoff_detected", False),
                )
            )

            # Phán quyết ban đầu của AI model
            prev_ai_val = (
                item["previous_ai_model"].strip().upper() if item["previous_ai_model"] else "HUMAN"
            )
            v_type = (
                VerdictType.AI_GENERATED if prev_ai_val == "AI" else VerdictType.PROCESSED_HUMAN
            )

            mock_res = DetectionResult(
                track_id=track.id,
                track_title=title,
                source_url=track.resolved_url,
                is_ai=(prev_ai_val == "AI"),
                confidence_score=0.90,
                suspected_engine="None",
                reasons=[item["previous_reasons"]],
                detailed_explanation=item["previous_reasons"],
                verdict_type=v_type,
                verdict_label=(
                    "AI-GENERATED"
                    if prev_ai_val == "AI"
                    else "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                ),
            )

            final_res = detection_service._apply_cross_validation_safeguard(
                result=mock_res,
                track=track,
                metadata={"title": title, "artist": artist},
                metrics=current_metrics,
            )
            new_pred = "AI" if final_res.is_ai else "HUMAN"
        else:
            # Trường hợp chưa có metrics trong DB (ví dụ các bài nạp bù), phân tích mới
            try:
                final_res = detection_service.detect_single(track)
                new_pred = "AI" if final_res.is_ai else "HUMAN"
            except Exception as ex:
                new_pred = f"ERR: {ex}"

        is_now_correct = new_pred == gt
        if is_now_correct:
            fixed_count += 1
            status_icon = "✅ ĐÃ SỬA ĐÚNG"
        else:
            still_wrong_count += 1
            status_icon = "❌ VẪN SAI"

        print(
            f"[{idx:02d}/28] {status_icon} | {title} - {artist} | GT: {gt} | Cũ: {prev_pred} -> MỚI: {new_pred}"
        )
        if final_res and final_res.reasons:
            print(f"      Lý do chốt: {final_res.reasons[0]}")

        results_table.append(
            {
                "STT": idx,
                "Media ID": mid,
                "Tên bài": title,
                "Nghệ sĩ": artist,
                "Nhãn thực tế (GT)": gt,
                "Dự đoán Cũ (Sai)": prev_pred,
                "Dự đoán Mới": new_pred,
                "Trạng thái": "THÀNH CÔNG" if is_now_correct else "CHƯA ĐẠT",
            }
        )

    print("\n" + "=" * 85)
    print("📊 TỔNG HỢP HIỆU NĂNG SAU CẢI TIẾN TRÊN 28 BÀI TỪNG SAI:")
    print(f" - Tổng số bài kiểm tra     : {total_count} bài")
    print(
        f" - Số bài ĐÃ KHẮC PHỤC THÀNH CÔNG : {fixed_count}/{total_count} bài ({fixed_count / total_count * 100:.1f}%)"
    )
    print(
        f" - Số bài còn sai lệch      : {still_wrong_count}/{total_count} bài ({still_wrong_count / total_count * 100:.1f}%)"
    )
    print("=" * 85)


if __name__ == "__main__":
    verify_tracks()
