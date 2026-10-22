import json
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.core.config import TEMP_DIR
from backend.core.models.entities import BatchReport, DetectionResult, VerdictType
from frontend.components.ui_compat import get_width_kwargs
from frontend.services.backend_client import BackendClient

logger = logging.getLogger(__name__)

BATCH_STATE_FILE = TEMP_DIR / "batch_state.json"


def save_batch_persistent_state(
    file_path: Path | str | None,
    queries: list[str],
    scan_mode: str,
    total_tracks: int,
    start_time: float | None,
    counts: dict,
    live_rows: list[dict],
    live_results: list[DetectionResult],
    report_file_path: str | None = None,
    partial_excel_path: str | None = None,
):
    """Lưu trạng thái quét hàng loạt xuống đĩa để chịu lỗi khi người dùng reload / đóng tab."""
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "file_path": str(file_path) if file_path else None,
            "queries": queries,
            "scan_mode": scan_mode,
            "total_tracks": total_tracks,
            "start_time": start_time,
            "counts": counts,
            "live_rows": live_rows,
            "live_results": [r.model_dump() for r in live_results],
            "report_file_path": report_file_path,
            "partial_excel_path": partial_excel_path,
            "saved_at": time.time(),
        }
        with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Không thể lưu trạng thái batch xuống đĩa: {e}")


def load_batch_persistent_state() -> dict | None:
    """Nạp lại trạng thái quét từ đĩa nếu tồn tại."""
    if not BATCH_STATE_FILE.exists():
        return None
    try:
        with open(BATCH_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Không thể đọc trạng thái batch từ đĩa: {e}")
        return None


def clear_batch_persistent_state():
    """Xóa bỏ trạng thái quét trên đĩa khi người dùng bấm Đặt lại."""
    try:
        if BATCH_STATE_FILE.exists():
            BATCH_STATE_FILE.unlink()
    except Exception as e:
        logger.warning(f"Không thể xóa tệp trạng thái batch: {e}")


def format_time_duration(seconds: float) -> str:
    """Định dạng số giây thành chuỗi hiển thị dễ đọc (VD: '3m 45s', '1h 12m')."""
    if not seconds or seconds <= 0:
        return "0s"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def format_detection_row(r: DetectionResult) -> dict:
    """Định dạng kết quả giám định thành một hàng hiển thị đồng nhất trong Data Table."""
    origin_tag = "⚡ CSDL (Cached)" if getattr(r, "is_cached", False) else "🔄 Quét mới"
    corr_tag = "🧑‍🎤 Đã đính chính" if getattr(r, "is_human_corrected", False) else "-"
    ai_model_col = r.ai_model_verdict or ("AI" if getattr(r, "ai_model_is_ai", False) else "HUMAN")
    consensus_col = r.consensus_verdict or r.verdict_label
    conf = getattr(r, "confidence_score", 0.0)
    conf_str = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "0%"
    reasons = r.reasons or []
    song_type_val = r.song_type or getattr(r, "work_label", "-") or "-"
    distributor_val = r.distributor or "-"
    release_date_val = r.release_date or "-"

    dur_sec = getattr(r, "duration_sec", None)
    if (dur_sec is None or dur_sec <= 0) and r.acoustic_metrics:
        dur_sec = getattr(r.acoustic_metrics, "duration_sec", None)
    dur_str = format_time_duration(dur_sec) if dur_sec and dur_sec > 0 else "-"

    return {
        "Tên bài hát": r.track_title,
        "Thời lượng": dur_str,
        "Loại tác phẩm": song_type_val,
        "Nhà phân phối": distributor_val,
        "Ngày phát hành": release_date_val,
        "Đánh giá Model AI": ai_model_col,
        "Kết luận Tổng hợp": consensus_col,
        "Nguồn gốc": origin_tag,
        "Đính chính": corr_tag,
        "Độ tin cậy": conf_str,
        "Mô hình nghi vấn": r.suspected_engine or "N/A",
        "Luận cứ": " | ".join(reasons[:2]) if reasons else "N/A",
        "Nguồn link": r.source_url or "N/A",
    }


def render_batch_view(client: BackendClient):
    """Hiển thị giao diện quét hàng loạt bài hát với cơ chế bền bỉ (Session Persistence & Resilience)."""
    st.markdown("### 📁 Quét Hàng Loạt (Batch Mode)")
    st.write(
        "Tải lên tệp (.txt, .csv, .xlsx) hoặc nhập trực tiếp danh sách bài hát / link YouTube. "
        "Hệ thống có cơ chế tự động khôi phục và tiếp tục quét khi tab bị ngắt quãng hoặc reload."
    )

    # Tự động nạp lại trạng thái nếu session vừa bị reload nhưng đĩa vẫn còn dữ liệu
    if not st.session_state.get("batch_live_rows"):
        saved_state = load_batch_persistent_state()
        if saved_state and saved_state.get("live_rows"):
            if saved_state.get("file_path") and not st.session_state.get("batch_last_file_path"):
                st.session_state.batch_last_file_path = saved_state.get("file_path")
            if saved_state.get("queries") and not st.session_state.get("batch_last_queries"):
                st.session_state.batch_last_queries = saved_state.get("queries")
            if "batch_last_scan_mode" not in st.session_state:
                st.session_state.batch_last_scan_mode = saved_state.get("scan_mode") or "smart"
            st.session_state.batch_total_tracks = saved_state.get("total_tracks") or len(saved_state["live_rows"])
            st.session_state.batch_start_time = saved_state.get("start_time")
            st.session_state.batch_live_counts = saved_state.get("counts") or {
                "ai": 0,
                "human": 0,
                "uncertain": 0,
                "failed": 0,
            }
            st.session_state.batch_live_rows = saved_state.get("live_rows") or []
            restored_results = []
            for r_dict in saved_state.get("live_results", []):
                try:
                    restored_results.append(DetectionResult.model_validate(r_dict))
                except Exception:
                    pass
            st.session_state.batch_live_results = restored_results
            st.session_state.batch_partial_excel_path = saved_state.get("partial_excel_path")
            rep_path = saved_state.get("report_file_path")
            if rep_path and Path(rep_path).exists():
                st.session_state.batch_report = BatchReport(
                    total_tracks=st.session_state.batch_total_tracks,
                    processed_count=len(restored_results),
                    ai_count=st.session_state.batch_live_counts.get("ai", 0),
                    human_count=st.session_state.batch_live_counts.get("human", 0),
                    uncertain_count=st.session_state.batch_live_counts.get("uncertain", 0),
                    failed_count=st.session_state.batch_live_counts.get("failed", 0),
                    results=restored_results,
                    report_file_path=rep_path,
                )

    if st.session_state.get("batch_live_rows"):
        curr_len = len(st.session_state.batch_live_rows)
        exp_total = st.session_state.get("batch_total_tracks", curr_len)
        if exp_total > curr_len:
            st.info(
                f"💾 **Đã tự động khôi phục tiến trình quét dang dở ({curr_len}/{exp_total} bài).** "
                "Bạn có thể bấm **▶️ Tiếp tục quét** bên dưới để chạy tiếp các bài còn lại mà không mất dữ liệu."
            )
        else:
            st.success(
                f"💾 **Đã khôi phục kết quả quét gần nhất ({curr_len} bài).** "
                "Dữ liệu và file báo cáo Excel được lưu giữ đầy đủ bên dưới."
            )

    upload_mode = st.radio(
        "Phương thức nhập dữ liệu:",
        ["Tải lên tệp (.txt, .csv, .xlsx)", "Nhập danh sách trực tiếp"],
        horizontal=True,
    )

    tracks_to_process = []
    file_path = None

    if upload_mode == "Tải lên tệp (.txt, .csv, .xlsx)":
        uploaded_file = st.file_uploader(
            "Chọn tệp danh sách bài hát:", type=["txt", "csv", "xlsx", "xls"]
        )
        if uploaded_file:
            temp_path = TEMP_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_path = temp_path
            st.session_state.batch_last_file_path = str(temp_path)
            st.success(f"Đã nạp file: {uploaded_file.name}")
        elif st.session_state.get("batch_last_file_path"):
            cand = Path(st.session_state.batch_last_file_path)
            if cand.exists():
                file_path = cand
                st.info(f"💡 Đang dùng tệp đã nạp trước đó: **{cand.name}**")
    else:
        text_queries = st.text_area(
            "Nhập danh sách link hoặc tên bài hát (mỗi dòng 1 bài):",
            value="\n".join(st.session_state.get("batch_last_queries", [])),
            placeholder="https://www.youtube.com/watch?v=...\nSơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau\nBài hát 3...",
            height=200,
        )
        if text_queries.strip():
            tracks_to_process = [q.strip() for q in text_queries.split("\n") if q.strip()]
            st.session_state.batch_last_queries = tracks_to_process

    # Tùy chọn chế độ quét hàng loạt
    scan_mode_choice = st.radio(
        "🎯 **Chế độ quét hàng loạt:**",
        options=[
            "⚡ Quét gia số thông minh (Tự động dùng lại CSDL đã có - Tiết kiệm thời gian & Quota)",
            "🔄 Quét lại toàn bộ (Bỏ qua CSDL và ép buộc AI phân tích lại - Cập nhật đè CSDL)",
            "🛠️ Chỉ quét lại bài bị lỗi hoặc chưa có kết luận",
        ],
        index=0,
    )
    scan_mode_map = {
        "⚡ Quét gia số thông minh (Tự động dùng lại CSDL đã có - Tiết kiệm thời gian & Quota)": "smart",
        "🔄 Quét lại toàn bộ (Bỏ qua CSDL và ép buộc AI phân tích lại - Cập nhật đè CSDL)": "force_all",
        "🛠️ Chỉ quét lại bài bị lỗi hoặc chưa có kết luận": "retry_errors",
    }
    selected_scan_mode = scan_mode_map[scan_mode_choice]
    st.session_state.batch_last_scan_mode = selected_scan_mode

    # Trạng thái hiện tại
    has_live_data = bool(st.session_state.get("batch_live_rows"))
    has_report = bool(st.session_state.get("batch_report"))
    live_count = len(st.session_state.get("batch_live_rows", []))
    total_expected = st.session_state.get("batch_total_tracks", 0)
    is_partially_done = has_live_data and (total_expected > 0 and live_count < total_expected)

    st.divider()

    # KHU VỰC ĐIỀU KHIỂN CHÍNH (ACTION BUTTONS)
    col_act1, col_act2, col_act3, col_act4 = st.columns([2, 2, 2, 2])

    start_clicked = False
    resume_clicked = False
    stop_clicked = False
    reset_clicked = False
    retry_errors_clicked = False

    with col_act1:
        if is_partially_done:
            resume_clicked = st.button(
                f"▶️ Tiếp tục quét ({live_count}/{total_expected})",
                type="primary",
                use_container_width=True,
                help="Tiếp tục quét các bài còn lại từ vị trí dang dở mà không quét lại các bài đã xong.",
            )
        else:
            start_clicked = st.button(
                "🚀 Bắt đầu Quét Hàng Loạt",
                type="primary",
                use_container_width=True,
            )

    with col_act2:
        if has_live_data:
            stop_clicked = st.button(
                "🛑 Dừng Quét (Lưu kết quả)",
                use_container_width=True,
                help="Dừng quét an toàn sau bài hiện tại và giữ nguyên kết quả các bài đã hoàn thành.",
            )

    with col_act3:
        failed_count_now = st.session_state.get("batch_live_counts", {}).get("failed", 0)
        if failed_count_now > 0:
            retry_errors_clicked = st.button(
                f"🛠️ Quét lại {failed_count_now} bài lỗi",
                use_container_width=True,
                help="Chỉ quét lại các bài có nhãn lỗi kỹ thuật hoặc chưa hoàn tất.",
            )

    with col_act4:
        if has_live_data or has_report:
            reset_clicked = st.button(
                "🔄 Đặt lại từ đầu",
                use_container_width=True,
                help="Xóa kết quả hiển thị hiện tại để bắt đầu lượt quét hoàn toàn mới.",
            )

    # Xử lý Đặt lại (Reset)
    if reset_clicked:
        clear_batch_persistent_state()
        st.session_state.batch_live_rows = []
        st.session_state.batch_live_results = []
        st.session_state.batch_report = None
        st.session_state.batch_is_running = False
        st.session_state.batch_stop_requested = False
        st.session_state.batch_total_tracks = 0
        st.session_state.batch_partial_excel_path = None
        st.toast("Đã đặt lại tiến trình quét!", icon="🔄")
        st.rerun()

    # Xử lý Dừng quét (Stop)
    if stop_clicked:
        st.session_state.batch_stop_requested = True
        st.toast("🛑 Đã gửi lệnh dừng! Hệ thống sẽ chốt kết quả sau bài đang chạy dở...", icon="🛑")

    # Xử lý Bắt đầu mới hoặc Tiếp tục quét
    should_run = start_clicked or resume_clicked or retry_errors_clicked

    if should_run:
        target_file = file_path
        target_queries = tracks_to_process

        if not target_file and not target_queries:
            st.warning("Vui lòng tải lên tệp (.xlsx, .csv) hoặc nhập danh sách bài hát trước!")
            return

        start_index = 0
        initial_results: list[DetectionResult] = []

        if start_clicked:
            # Bắt đầu lượt quét hoàn toàn mới
            clear_batch_persistent_state()
            st.session_state.batch_live_rows = []
            st.session_state.batch_live_results = []
            st.session_state.batch_report = None
            st.session_state.batch_partial_excel_path = None
            st.session_state.batch_stop_requested = False
            st.session_state.batch_live_counts = {
                "ai": 0,
                "human": 0,
                "uncertain": 0,
                "failed": 0,
            }
            st.session_state.batch_start_time = time.time()
        elif resume_clicked:
            # Tiếp tục từ vị trí đã dừng
            st.session_state.batch_stop_requested = False
            start_index = len(st.session_state.batch_live_results)
            initial_results = list(st.session_state.batch_live_results)
            st.session_state.batch_start_time = time.time()
            st.info(f"▶️ Đang tiếp tục quét từ bài thứ {start_index + 1}...")
        elif retry_errors_clicked:
            # Quét lại các bài lỗi
            st.session_state.batch_stop_requested = False
            selected_scan_mode = "retry_errors"

        st.session_state.batch_is_running = True

        # Vùng hiển thị tiến trình thời gian thực
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        eta_box = st.empty()
        live_metrics_placeholder = st.empty()
        live_table_header = st.empty()
        live_table_placeholder = st.empty()

        def stop_checker() -> bool:
            return bool(st.session_state.get("batch_stop_requested", False))

        def update_progress(curr, total, track, res, err):
            st.session_state.batch_total_tracks = total
            pct = min(1.0, max(0.0, curr / total)) if total > 0 else 0.0
            progress_bar.progress(pct)

            counts = st.session_state.batch_live_counts
            start_t = st.session_state.get("batch_start_time") or time.time()
            elapsed_sec = max(0.1, time.time() - start_t)
            speed_per_track = elapsed_sec / max(1, (curr - start_index))
            remaining_tracks = max(0, total - curr)
            eta_sec = remaining_tracks * speed_per_track

            elapsed_str = format_time_duration(elapsed_sec)
            eta_str = format_time_duration(eta_sec)

            if res:
                cached_tag = " [⚡ CSDL]" if getattr(res, "is_cached", False) else ""
                status_box.markdown(
                    f"**[{curr}/{total}]** ✅ **{res.track_title}** ➔ `{res.verdict_label}`{cached_tag}"
                )
                row = format_detection_row(res)
                if getattr(res, "verdict_type", None) == VerdictType.UNCERTAIN:
                    counts["uncertain"] += 1
                elif getattr(res, "verdict_type", None) == VerdictType.ERROR:
                    counts["failed"] += 1
                elif res.is_ai:
                    counts["ai"] += 1
                else:
                    counts["human"] += 1
                st.session_state.batch_live_results.append(res)
            else:
                status_box.markdown(
                    f"**[{curr}/{total}]** ❌ **{track.raw_query}** ➔ `Lỗi: {err}`"
                )
                counts["failed"] += 1
                dur_raw = getattr(track, "duration_sec", 0) or 0
                dur_str = format_time_duration(dur_raw) if dur_raw > 0 else "-"
                row = {
                    "Tên bài hát": track.track_title or track.raw_query,
                    "Thời lượng": dur_str,
                    "Loại tác phẩm": getattr(track, "song_type", "-") or "-",
                    "Nhà phân phối": getattr(track, "distributor", "-") or "-",
                    "Ngày phát hành": getattr(track, "release_date", "-") or "-",
                    "Đánh giá Model AI": "ERROR",
                    "Kết luận Tổng hợp": "❌ LỖI XỬ LÝ",
                    "Nguồn gốc": "🔄 Quét mới",
                    "Đính chính": "-",
                    "Độ tin cậy": "0%",
                    "Mô hình nghi vấn": "N/A",
                    "Luận cứ": str(err or "Lỗi không xác định"),
                    "Nguồn link": getattr(track, "resolved_url", "N/A") or "N/A",
                }

            st.session_state.batch_live_rows.append(row)
            st.session_state.batch_live_counts = counts

            # Lưu bền vững xuống đĩa để chống mất dữ liệu khi reload tab
            save_batch_persistent_state(
                file_path=target_file,
                queries=target_queries,
                scan_mode=selected_scan_mode,
                total_tracks=total,
                start_time=st.session_state.get("batch_start_time"),
                counts=counts,
                live_rows=st.session_state.batch_live_rows,
                live_results=st.session_state.batch_live_results,
                partial_excel_path=st.session_state.get("batch_partial_excel_path"),
            )

            # Cập nhật thông số thời gian & tốc độ (ETA Dashboard)
            with eta_box.container():
                ec1, ec2, ec3 = st.columns(3)
                ec1.caption(f"⏱️ Đã chạy: **{elapsed_str}**")
                ec2.caption(f"⚡ Tốc độ: **{speed_per_track:.1f}s / bài**")
                ec3.caption(f"⏳ Ước tính còn lại (ETA): **~{eta_str}**")

            # Cập nhật metric cards
            with live_metrics_placeholder.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Đã quét", f"{curr}/{total}")
                m2.metric("Nghi vấn AI", counts["ai"])
                m3.metric("Người làm", counts["human"])
                m4.metric("Chưa rõ", counts["uncertain"])
                m5.metric("Lỗi / Bỏ qua", counts["failed"])

            # Cập nhật Data Table trực tiếp
            live_table_header.markdown(
                f"##### 📋 Bảng Dữ Liệu Thời Gian Thực ({len(st.session_state.batch_live_rows)}/{total} bài)"
            )
            live_table_placeholder.dataframe(
                pd.DataFrame(st.session_state.batch_live_rows),
                **get_width_kwargs(st.dataframe),
            )

        try:
            if target_file:
                report = client.process_batch_file(
                    file_path=target_file,
                    model_override=st.session_state.model_name,
                    scan_mode=selected_scan_mode,
                    on_progress=update_progress,
                    stop_check=stop_checker,
                    start_index=start_index,
                    initial_results=initial_results,
                )
            else:
                report = client.process_batch_queries(
                    queries=target_queries,
                    model_override=st.session_state.model_name,
                    scan_mode=selected_scan_mode,
                    on_progress=update_progress,
                    stop_check=stop_checker,
                    start_index=start_index,
                    initial_results=initial_results,
                )

            st.session_state.batch_report = report
            st.session_state.batch_is_running = False

            save_batch_persistent_state(
                file_path=target_file,
                queries=target_queries,
                scan_mode=selected_scan_mode,
                total_tracks=getattr(report, "total_tracks", len(st.session_state.batch_live_rows)),
                start_time=st.session_state.get("batch_start_time"),
                counts=st.session_state.batch_live_counts,
                live_rows=st.session_state.batch_live_rows,
                live_results=st.session_state.batch_live_results,
                report_file_path=getattr(report, "report_file_path", None),
                partial_excel_path=st.session_state.get("batch_partial_excel_path"),
            )

            if st.session_state.get("batch_stop_requested"):
                st.warning(
                    f"🛑 Đã dừng quét theo yêu cầu tại {len(st.session_state.batch_live_rows)}/{report.total_tracks} bài. "
                    "Kết quả đã được bảo toàn đầy đủ bên dưới."
                )
            else:
                st.success("🎉 Đã hoàn thành quét toàn bộ danh sách!")

            st.rerun()

        except Exception as e:
            st.session_state.batch_is_running = False
            st.error(f"Lỗi trong quá trình quét hàng loạt: {e}")

    # HIỂN THỊ KẾT QUẢ ĐÃ LƯU BỀN VỮNG (PERSISTENT DATA TABLE & REPORT)
    if st.session_state.get("batch_live_rows"):
        live_rows = st.session_state.batch_live_rows
        counts = st.session_state.get("batch_live_counts", {})
        total_trk = st.session_state.get("batch_total_tracks", len(live_rows))

        st.subheader("📊 Báo Cáo & Dữ Liệu Hiện Tại")

        # 5 Cards thống kê
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tổng bài đã quét", f"{len(live_rows)} / {total_trk}")
        c2.metric("Nghi vấn AI", counts.get("ai", 0), delta="AI Detected", delta_color="inverse")
        c3.metric("Người làm", counts.get("human", 0), delta="Human", delta_color="normal")
        c4.metric("Chưa rõ", counts.get("uncertain", 0))
        c5.metric("Lỗi / Bỏ qua", counts.get("failed", 0))

        # Nút tải Excel luôn khả dụng
        excel_path_to_download = None
        if (
            st.session_state.get("batch_report")
            and getattr(st.session_state.batch_report, "report_file_path", None)
            and Path(st.session_state.batch_report.report_file_path).exists()
        ):
            excel_path_to_download = st.session_state.batch_report.report_file_path
        elif (
            st.session_state.get("batch_partial_excel_path")
            and Path(st.session_state.batch_partial_excel_path).exists()
        ):
            excel_path_to_download = st.session_state.batch_partial_excel_path
        elif st.session_state.get("batch_live_results"):
            # Chỉ tạo file một lần khi cần tải mà chưa có file trên đĩa
            try:
                excel_path_to_download = client.export_custom_results(
                    st.session_state.batch_live_results, format="xlsx"
                )
                st.session_state.batch_partial_excel_path = excel_path_to_download
            except Exception:
                pass

        if excel_path_to_download and Path(excel_path_to_download).exists():
            with open(excel_path_to_download, "rb") as f:
                st.download_button(
                    label=f"📥 Tải xuống File Báo Cáo Excel Hiện Tại ({len(live_rows)} bài)",
                    data=f.read(),
                    file_name=Path(excel_path_to_download).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        # Bảng dữ liệu chi tiết
        st.markdown(f"##### 📋 Danh Sách Chi Tiết ({len(live_rows)} bài)")
        st.dataframe(pd.DataFrame(live_rows), **get_width_kwargs(st.dataframe))

        # Can thiệp nhanh từng bài trong danh sách
        live_results = st.session_state.get("batch_live_results", [])
        if live_results:
            with st.expander(
                "✍️ Can thiệp & Đính chính nhanh từng bài trong danh sách", expanded=False
            ):
                st.caption("Chọn bài hát để đính chính kết quả hoặc quét lại độc lập:")
                track_labels = [
                    f"{idx + 1}. {r.track_title} [{r.verdict_label}]"
                    for idx, r in enumerate(live_results)
                ]
                selected_idx = st.selectbox(
                    "Chọn bài cần can thiệp:",
                    range(len(track_labels)),
                    format_func=lambda i: track_labels[i],
                )
                if selected_idx is not None and selected_idx < len(live_results):
                    target_res = live_results[selected_idx]
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        if st.button(
                            "🧑‍🎤 Đính chính: Là Người",
                            key=f"b_corr_h_{target_res.track_id}",
                            use_container_width=True,
                        ):
                            try:
                                updated = client.correct_verdict(
                                    track_id=target_res.track_id,
                                    is_ai=False,
                                    verdict_type="CONFIRMED_HUMAN",
                                    verdict_label="CONFIRMED HUMAN (Người làm)",
                                    note="Đính chính từ bảng danh sách hàng loạt",
                                )
                                live_results[selected_idx] = updated
                                st.session_state.batch_live_results = live_results
                                st.session_state.batch_live_rows[selected_idx] = (
                                    format_detection_row(updated)
                                )
                                save_batch_persistent_state(
                                    file_path=st.session_state.get("batch_last_file_path"),
                                    queries=st.session_state.get("batch_last_queries", []),
                                    scan_mode=st.session_state.get("batch_last_scan_mode", "smart"),
                                    total_tracks=st.session_state.get("batch_total_tracks", len(live_results)),
                                    start_time=st.session_state.get("batch_start_time"),
                                    counts=st.session_state.get("batch_live_counts", {}),
                                    live_rows=st.session_state.batch_live_rows,
                                    live_results=st.session_state.batch_live_results,
                                    report_file_path=getattr(st.session_state.get("batch_report"), "report_file_path", None),
                                    partial_excel_path=st.session_state.get("batch_partial_excel_path"),
                                )
                                st.toast(
                                    f"✅ Đã đính chính '{updated.track_title}' thành Người thật!",
                                    icon="🧑‍🎤",
                                )
                                st.rerun()
                            except Exception as err:
                                st.error(f"Lỗi: {err}")
                    with col_b2:
                        if st.button(
                            "🤖 Đính chính: Là Nhạc AI",
                            key=f"b_corr_a_{target_res.track_id}",
                            use_container_width=True,
                        ):
                            try:
                                updated = client.correct_verdict(
                                    track_id=target_res.track_id,
                                    is_ai=True,
                                    verdict_type="AI_GENERATED",
                                    verdict_label="AI-GENERATED (Nhạc AI tạo)",
                                    note="Đính chính từ bảng danh sách hàng loạt",
                                )
                                live_results[selected_idx] = updated
                                st.session_state.batch_live_results = live_results
                                st.session_state.batch_live_rows[selected_idx] = (
                                    format_detection_row(updated)
                                )
                                save_batch_persistent_state(
                                    file_path=st.session_state.get("batch_last_file_path"),
                                    queries=st.session_state.get("batch_last_queries", []),
                                    scan_mode=st.session_state.get("batch_last_scan_mode", "smart"),
                                    total_tracks=st.session_state.get("batch_total_tracks", len(live_results)),
                                    start_time=st.session_state.get("batch_start_time"),
                                    counts=st.session_state.get("batch_live_counts", {}),
                                    live_rows=st.session_state.batch_live_rows,
                                    live_results=st.session_state.batch_live_results,
                                    report_file_path=getattr(st.session_state.get("batch_report"), "report_file_path", None),
                                    partial_excel_path=st.session_state.get("batch_partial_excel_path"),
                                )
                                st.toast(
                                    f"🤖 Đã đính chính '{updated.track_title}' thành Nhạc AI!",
                                    icon="🤖",
                                )
                                st.rerun()
                            except Exception as err:
                                st.error(f"Lỗi: {err}")
                    with col_b3:
                        if st.button(
                            "🔄 Quét lại bài này",
                            key=f"b_rescan_{target_res.track_id}",
                            use_container_width=True,
                        ):
                            with st.spinner(f"Đang quét lại '{target_res.track_title}'..."):
                                try:
                                    rescan_query = (
                                        target_res.raw_query
                                        or target_res.track_title
                                        or target_res.source_url
                                    )
                                    fresh = client.detect_single_track(
                                        query=rescan_query,
                                        model_override=st.session_state.model_name,
                                        force_rescan=True,
                                    )
                                    live_results[selected_idx] = fresh
                                    st.session_state.batch_live_results = live_results
                                    st.session_state.batch_live_rows[selected_idx] = (
                                        format_detection_row(fresh)
                                    )
                                    save_batch_persistent_state(
                                        file_path=st.session_state.get("batch_last_file_path"),
                                        queries=st.session_state.get("batch_last_queries", []),
                                        scan_mode=st.session_state.get("batch_last_scan_mode", "smart"),
                                        total_tracks=st.session_state.get("batch_total_tracks", len(live_results)),
                                        start_time=st.session_state.get("batch_start_time"),
                                        counts=st.session_state.get("batch_live_counts", {}),
                                        live_rows=st.session_state.batch_live_rows,
                                        live_results=st.session_state.batch_live_results,
                                        report_file_path=getattr(st.session_state.get("batch_report"), "report_file_path", None),
                                        partial_excel_path=st.session_state.get("batch_partial_excel_path"),
                                    )
                                    st.toast(
                                        f"🎉 Đã quét lại xong '{fresh.track_title}'!", icon="✅"
                                    )
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Lỗi: {err}")
