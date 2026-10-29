from pathlib import Path
from typing import Any

from ..core.interfaces.repositories.report_repository_interface import IReportRepository
from ..core.interfaces.services.report_service_interface import IReportService
from ..core.models.entities import DetectionResult, InputType, VerdictType


def format_display_title(r: DetectionResult) -> str:
    """
    Định dạng hiển thị cho cột 'Tên bài hát' trong file báo cáo (.xlsx / .csv):
    - Nếu input là tên bài hát (hoặc file local): hiển thị tên bài hát.
    - Nếu input là link URL: hiển thị 'Tên bài hát - Link'.
      Nếu chưa trích xuất được tên bài hát thực sự (hoặc tên bị trùng link), chỉ hiển thị Link.
    """
    raw_query = (r.raw_query or "").strip()
    source_url = (r.source_url or "").strip()
    track_title = (r.track_title or "").strip()

    # Nhận diện xem đầu vào có phải là link hay không
    is_url_input = False
    if r.input_type == InputType.URL:
        is_url_input = True
    elif raw_query.startswith("http://") or raw_query.startswith("https://"):
        is_url_input = True
    elif (not raw_query or raw_query == track_title) and (
        track_title.startswith("http://") or track_title.startswith("https://")
    ):
        is_url_input = True

    if is_url_input:
        # Đường link URL gốc
        url = ""
        if r.input_type == InputType.URL:
            url = source_url or raw_query or track_title
        elif raw_query.startswith("http://") or raw_query.startswith("https://"):
            url = raw_query
        elif source_url.startswith("http://") or source_url.startswith("https://"):
            url = source_url
        elif track_title.startswith("http://") or track_title.startswith("https://"):
            url = track_title

        # Tên bài hát thực tế (loại trừ nếu giá trị đó là link URL)
        real_title = ""
        if (
            track_title
            and track_title != url
            and not track_title.startswith("http://")
            and not track_title.startswith("https://")
        ):
            real_title = track_title
        elif (
            raw_query
            and raw_query != url
            and not raw_query.startswith("http://")
            and not raw_query.startswith("https://")
        ):
            real_title = raw_query

        if real_title and url:
            return f"{real_title} - {url}"
        elif real_title:
            return real_title
        elif url:
            return url
        return track_title or "N/A"

    # Input không phải là link (nhập tên bài hát hoặc file local):
    # Chỉ hiển thị thuần túy tên bài hát
    if (
        track_title
        and not track_title.startswith("http://")
        and not track_title.startswith("https://")
    ):
        return track_title
    if raw_query and not raw_query.startswith("http://") and not raw_query.startswith("https://"):
        return raw_query
    return track_title or raw_query or "N/A"


class ReportService(IReportService):
    """
    Hiện thực dịch vụ xuất báo cáo.
    Tuân thủ interface IReportService và phụ thuộc vào IReportRepository.
    """

    def __init__(self, report_repo: IReportRepository):
        self.report_repo = report_repo

    def build_report_records(self, results: list[DetectionResult]) -> list[dict[str, Any]]:
        """Xây dựng danh sách bản ghi chuẩn hóa từ danh sách DetectionResult."""
        records: list[dict[str, Any]] = []
        for idx, r in enumerate(results):
            display_title = format_display_title(r)
            url_str = (r.source_url or "").strip() or (
                (r.raw_query or "").strip()
                if (r.raw_query or "").strip().startswith("http://")
                or (r.raw_query or "").strip().startswith("https://")
                else ""
            )
            if not url_str:
                url_str = "N/A"
            if not r.source_url and url_str != "N/A":
                r.source_url = url_str

            source_origin = (
                "CSDL đã lưu (Cached)"
                if getattr(r, "is_cached", False)
                else "Vừa quét mới (Fresh Scan)"
            )
            corrected_str = (
                "Có (Đã đính chính)" if getattr(r, "is_human_corrected", False) else "Không"
            )

            ai_eval = getattr(r, "ai_model_verdict", None)
            if not ai_eval:
                ai_bool = getattr(r, "ai_model_is_ai", None)
                ai_eval = (
                    "AI_GENERATED"
                    if ai_bool is True
                    else ("PROCESSED_HUMAN" if ai_bool is False else "N/A")
                )

            cons_verdict = getattr(r, "consensus_verdict", None) or r.verdict_label

            dur_val = getattr(r, "duration_sec", None) or (
                r.acoustic_metrics.duration_sec if r.acoustic_metrics else None
            )
            dur_display = (
                f"{int(dur_val // 60)}m {int(dur_val % 60)}s"
                if (dur_val and dur_val > 0)
                else "N/A"
            )

            song_type_display = r.song_type or getattr(r, "work_label", None) or "N/A"
            if song_type_display in ("None", ""):
                song_type_display = "N/A"

            if r.verdict_type == VerdictType.ERROR:
                records.append(
                    {
                        "STT": idx + 1,
                        "Tên bài hát": display_title,
                        "Thời lượng": dur_display,
                        "Loại tác phẩm (Song Type)": song_type_display if song_type_display != "N/A" else "ERROR",
                        "Nhà phân phối": r.distributor or "N/A",
                        "Ngày phát hành": r.release_date or "N/A",
                        "Đánh giá Model AI": ai_eval,
                        "Kết luận Tổng hợp": cons_verdict,
                        "Kết luận thẩm định": r.verdict_label,
                        "Phân loại kỹ thuật": "ERROR",
                        "Tác phẩm": getattr(r, "work_label", "ERROR"),
                        "Là bản Cover": "Không",
                        "Nghệ sĩ bài gốc": "N/A",
                        "Tên bài gốc": "N/A",
                        "Độ tin cậy (%)": "0.0%",
                        "Nguồn kết quả": source_origin,
                        "Đã đính chính": corrected_str,
                        "Điểm Fakeprint Fourier (ISMIR 2025)": "N/A",
                        "Độ mở dải tần (Rolloff 95%)": "N/A",
                        "Đánh giá nén truyền tải": "N/A",
                        "Nền tảng nghi vấn": "N/A",
                        "Mô hình AI": r.model_used or "N/A",
                        "Luận cứ chính / Chi tiết lỗi": r.reasons[0]
                        if r.reasons
                        else "Lỗi không xác định",
                        "Giải thích chi tiết": r.detailed_explanation,
                        "Nguồn link": url_str,
                        "Thời gian quét": r.processed_at,
                    }
                )
            else:
                fakeprint_str = (
                    f"{r.acoustic_metrics.fakeprint_score * 100:.1f}%"
                    if r.acoustic_metrics
                    else "N/A"
                )
                rolloff_str = (
                    f"{r.acoustic_metrics.spectral_rolloff_95_hz:.0f} Hz"
                    if r.acoustic_metrics
                    else "N/A"
                )
                comp_source = r.acoustic_metrics.compression_source if r.acoustic_metrics else "N/A"
                is_cov_str = "Có (Cover)" if getattr(r, "is_cover", False) else "Không"
                records.append(
                    {
                        "STT": idx + 1,
                        "Tên bài hát": display_title,
                        "Thời lượng": dur_display,
                        "Loại tác phẩm (Song Type)": song_type_display,
                        "Nhà phân phối": r.distributor or "N/A",
                        "Ngày phát hành": r.release_date or "N/A",
                        "Đánh giá Model AI": ai_eval,
                        "Kết luận Tổng hợp": cons_verdict,
                        "Kết luận thẩm định": r.verdict_label,
                        "Phân loại kỹ thuật": r.verdict_type.value
                        if hasattr(r.verdict_type, "value")
                        else str(r.verdict_type),
                        "Tác phẩm": getattr(r, "work_label", "Chưa rõ tác phẩm"),
                        "Là bản Cover": is_cov_str,
                        "Nghệ sĩ bài gốc": getattr(r, "original_artist", "") or "N/A",
                        "Tên bài gốc": getattr(r, "original_song_title", "") or "N/A",
                        "Độ tin cậy (%)": f"{r.confidence_score * 100:.1f}%",
                        "Nguồn kết quả": source_origin,
                        "Đã đính chính": corrected_str,
                        "Điểm Fakeprint Fourier (ISMIR 2025)": fakeprint_str,
                        "Độ mở dải tần (Rolloff 95%)": rolloff_str,
                        "Đánh giá nén truyền tải": comp_source,
                        "Nền tảng nghi vấn": r.suspected_engine,
                        "Mô hình AI": r.model_used or "Heuristics",
                        "Luận cứ chính / Chi tiết lỗi": " | ".join(r.reasons),
                        "Giải thích chi tiết": r.detailed_explanation,
                        "Nguồn link": url_str,
                        "Thời gian quét": r.processed_at,
                    }
                )

        return records

    def export_report(self, results: list[DetectionResult], format: str = "xlsx") -> Path | None:
        """Xuất danh sách kết quả ra file Excel hoặc CSV."""
        if not results:
            return None
        records = self.build_report_records(results)
        return self.report_repo.save_report_file(records, format=format)

    def export_report_bytes(
        self, results: list[DetectionResult], format: str = "xlsx"
    ) -> bytes | None:
        """Xuất danh sách kết quả ra mảng bytes trong RAM (Excel hoặc CSV)."""
        if not results:
            return None
        records = self.build_report_records(results)
        return self.report_repo.export_report_bytes(records, format=format)
