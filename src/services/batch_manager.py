import time
from pathlib import Path
from typing import List, Optional, Callable
import pandas as pd

from ..config import OUTPUT_DIR, BATCH_SLEEP_INTERVAL
from ..models.schemas import TrackItem, DetectionResult, BatchReport, VerdictType
from .audio_engine import AudioEngine
from .signal_analyzer import SignalAnalyzer
from .gemini_classifier import GeminiClassifier

class BatchManager:
    def __init__(self, gemini_classifier: Optional[GeminiClassifier] = None):
        self.audio_engine = AudioEngine()
        self.classifier = gemini_classifier or GeminiClassifier()

    def process_single_track(
        self,
        track: TrackItem,
        model_override: Optional[str] = None
    ) -> DetectionResult:
        """Thực hiện toàn bộ quy trình nhận diện 1 bài hát từ A đến Z."""
        # 1. Tải và cắt 30s âm thanh
        audio_path, metadata = self.audio_engine.process_track_audio(track)

        # 2. Kiểm tra bộ lọc Heuristics nhanh (Ngày phát hành, tags nhận diện AI)
        heuristics_result = SignalAnalyzer.check_heuristics(metadata)
        if heuristics_result:
            is_ai = heuristics_result["is_ai"]
            v_type = heuristics_result.get("verdict_type", VerdictType.CONFIRMED_HUMAN if not is_ai else VerdictType.AI_GENERATED)
            if v_type == VerdictType.CONFIRMED_HUMAN:
                verdict_label = "CONFIRMED HUMAN (Người làm)"
            elif v_type == VerdictType.PROCESSED_HUMAN:
                verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
            elif v_type == VerdictType.HYBRID_AI:
                verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
            else:
                verdict_label = "AI-GENERATED (Nhạc AI tạo)"

            # Tạo ảnh phổ và metrics để báo cáo đầy đủ
            try:
                metrics, spec_image_path = SignalAnalyzer.analyze_audio(audio_path)
            except Exception:
                metrics = None
                spec_image_path = None

            return DetectionResult(
                track_id=track.id,
                track_title=track.track_title or track.raw_query,
                source_url=track.resolved_url,
                is_ai=is_ai,
                verdict_type=v_type,
                verdict_label=verdict_label,
                confidence_score=heuristics_result["confidence"],
                suspected_engine=heuristics_result["engine"],
                reasons=[heuristics_result["reason"]],
                detailed_explanation=f"Xác thực qua siêu dữ liệu nguồn: {heuristics_result['reason']}",
                acoustic_metrics=metrics,
                spectrogram_image_path=str(spec_image_path) if spec_image_path else None
            )

        # 3. Phân tích âm học và tạo ảnh Forensic 2 tầng (Mel-Spectrogram + Fourier Fakeprint)
        metrics, spec_image_path = SignalAnalyzer.analyze_audio(audio_path)

        # 4. Gửi lên Gemini Flash Lite để phân loại 4 tầng
        result = self.classifier.classify_track(
            track_id=track.id,
            track_title=track.track_title or track.raw_query,
            source_url=track.resolved_url,
            audio_path=audio_path,
            spectrogram_path=spec_image_path,
            metrics=metrics,
            model_override=model_override
        )

        return result

    def process_batch(
        self,
        tracks: List[TrackItem],
        model_override: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, TrackItem, Optional[DetectionResult], Optional[str]], None]] = None
    ) -> BatchReport:
        """Xử lý danh sách bài hát có điều tiết tốc độ (Rate Limiting) và cập nhật tiến độ."""
        total = len(tracks)
        results: List[DetectionResult] = []
        ai_count = 0
        human_count = 0
        failed_count = 0

        for i, track in enumerate(tracks):
            track.status = "PROCESSING"
            try:
                result = self.process_single_track(track, model_override=model_override)
                results.append(result)
                track.status = "COMPLETED"

                if result.is_ai:
                    ai_count += 1
                else:
                    human_count += 1

                if on_progress:
                    on_progress(i + 1, total, track, result, None)

            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                track.status = "FAILED"
                track.error_message = error_msg
                print(f"[BatchManager] Lỗi xử lý bài '{track.track_title or track.raw_query}': {error_msg}")

                if on_progress:
                    on_progress(i + 1, total, track, None, error_msg)

            # Nghỉ giữa các lượt gọi API để tránh vượt quá Rate Limit của gói Free Tier
            if i < total - 1:
                time.sleep(BATCH_SLEEP_INTERVAL)

        # Xuất báo cáo tự động
        report_file = self.export_report(results)

        return BatchReport(
            total_tracks=total,
            processed_count=len(results),
            ai_count=ai_count,
            human_count=human_count,
            failed_count=failed_count,
            results=results,
            report_file_path=str(report_file) if report_file else None
        )

    def export_report(self, results: List[DetectionResult], format: str = "xlsx") -> Optional[Path]:
        """Xuất danh sách kết quả ra file Excel hoặc CSV."""
        if not results:
            return None

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        data = []
        for r in results:
            fakeprint_str = (
                f"{r.acoustic_metrics.fakeprint_score * 100:.1f}%"
                if r.acoustic_metrics else "N/A"
            )
            rolloff_str = (
                f"{r.acoustic_metrics.spectral_rolloff_95_hz:.0f} Hz"
                if r.acoustic_metrics else "N/A"
            )
            comp_source = (
                r.acoustic_metrics.compression_source
                if r.acoustic_metrics else "N/A"
            )
            data.append({
                "Tên bài hát": r.track_title,
                "Kết luận thẩm định": r.verdict_label,
                "Phân loại kỹ thuật": r.verdict_type.value if hasattr(r.verdict_type, 'value') else str(r.verdict_type),
                "Độ tin cậy (%)": f"{r.confidence_score * 100:.1f}%",
                "Điểm Fakeprint Fourier (ISMIR 2025)": fakeprint_str,
                "Độ mở dải tần (Rolloff 95%)": rolloff_str,
                "Đánh giá nén truyền tải": comp_source,
                "Nền tảng nghi vấn": r.suspected_engine,
                "Luận cứ chính": " | ".join(r.reasons),
                "Giải thích chi tiết": r.detailed_explanation,
                "Nguồn link": r.source_url or "N/A",
                "Thời gian quét": r.processed_at
            })

        df = pd.DataFrame(data)

        if format.lower() == "csv":
            out_path = OUTPUT_DIR / f"ai_music_report_{timestamp}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            out_path = OUTPUT_DIR / f"ai_music_report_{timestamp}.xlsx"
            df.to_excel(out_path, index=False, engine="openpyxl")

        return out_path
