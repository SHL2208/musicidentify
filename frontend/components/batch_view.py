from pathlib import Path

import pandas as pd
import streamlit as st

from backend.core.config import TEMP_DIR
from backend.core.models.entities import BatchReport
from frontend.services.backend_client import BackendClient


def render_batch_view(client: BackendClient):
    """Hiển thị giao diện quét hàng loạt bài hát."""
    st.markdown("### 📁 Quét Hàng Loạt (Batch Mode)")
    st.write(
        "Tải lên tệp (.txt, .csv, .xlsx) hoặc nhập trực tiếp danh sách bài hát / link YouTube."
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
            st.success(f"Đã nạp file: {uploaded_file.name}")
    else:
        text_queries = st.text_area(
            "Nhập danh sách link hoặc tên bài hát (mỗi dòng 1 bài):",
            placeholder="https://www.youtube.com/watch?v=...\nSơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau\nBài hát 3...",
            height=200,
        )
        if text_queries.strip():
            tracks_to_process = [q.strip() for q in text_queries.split("\n") if q.strip()]

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

    if st.button("🚀 Bắt đầu Quét Hàng Loạt", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(curr, total, track, res, err):
            pct = curr / total
            progress_bar.progress(pct)
            if res:
                cached_tag = " [⚡ CSDL]" if getattr(res, "is_cached", False) else ""
                status_text.text(
                    f"[{curr}/{total}] ✅ {res.track_title} -> {res.verdict_label}{cached_tag}"
                )
            else:
                status_text.text(f"[{curr}/{total}] ❌ Lỗi: {track.raw_query} ({err})")

        with st.spinner("Đang xử lý toàn bộ danh sách..."):
            try:
                if file_path:
                    report = client.process_batch_file(
                        file_path=file_path,
                        model_override=st.session_state.model_name,
                        scan_mode=selected_scan_mode,
                        on_progress=update_progress,
                    )
                elif tracks_to_process:
                    report = client.process_batch_queries(
                        queries=tracks_to_process,
                        model_override=st.session_state.model_name,
                        scan_mode=selected_scan_mode,
                        on_progress=update_progress,
                    )
                else:
                    st.warning("Vui lòng tải lên tệp hoặc nhập ít nhất 1 bài hát!")
                    return

                st.session_state.batch_report = report
                st.success("🎉 Đã hoàn thành quét toàn bộ danh sách!")

            except Exception as e:
                st.error(f"Lỗi khi thực hiện quét hàng loạt: {e}")

    # Hiển thị báo cáo kết quả
    if st.session_state.batch_report:
        report: BatchReport = st.session_state.batch_report
        st.divider()
        st.subheader("📊 Tổng kết Kết quả")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tổng số bài", report.total_tracks)
        c2.metric("Nghi vấn AI", report.ai_count, delta="AI Detected", delta_color="inverse")
        c3.metric("Người làm", report.human_count, delta="Human", delta_color="normal")
        c4.metric("Chưa rõ", getattr(report, "uncertain_count", 0))
        c5.metric("Lỗi xử lý", report.failed_count)

        # Bảng dữ liệu chi tiết
        table_rows = []
        for r in report.results:
            origin_tag = "⚡ CSDL (Cached)" if getattr(r, "is_cached", False) else "🔄 Quét mới"
            corr_tag = "🧑‍🎤 Đã đính chính" if getattr(r, "is_human_corrected", False) else "-"
            ai_model_col = r.ai_model_verdict or ("AI" if r.ai_model_is_ai else "HUMAN")
            consensus_col = r.consensus_verdict or r.verdict_label
            table_rows.append(
                {
                    "Tên bài hát": r.track_title,
                    "Nguồn link": r.source_url or "N/A",
                    "Đánh giá Model AI": ai_model_col,
                    "Kết luận Tổng hợp": consensus_col,
                    "Nguồn gốc": origin_tag,
                    "Đính chính": corr_tag,
                    "Độ tin cậy": f"{r.confidence_score * 100:.0f}%",
                    "Mô hình nghi vấn": r.suspected_engine,
                    "Luận cứ": " | ".join(r.reasons[:2]),
                }
            )

        from frontend.components.ui_compat import get_width_kwargs

        st.dataframe(pd.DataFrame(table_rows), **get_width_kwargs(st.dataframe))

        # Can thiệp nhanh từng bài trong danh sách
        with st.expander("✍️ Can thiệp & Đính chính nhanh từng bài trong danh sách", expanded=False):
            st.caption("Chọn bài hát để đính chính kết quả hoặc quét lại độc lập:")
            track_labels = [
                f"{idx + 1}. {r.track_title} [{r.verdict_label}]"
                for idx, r in enumerate(report.results)
            ]
            selected_idx = st.selectbox(
                "Chọn bài cần can thiệp:",
                range(len(track_labels)),
                format_func=lambda i: track_labels[i],
            )
            if selected_idx is not None and selected_idx < len(report.results):
                target_res = report.results[selected_idx]
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
                            report.results[selected_idx] = updated
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
                            report.results[selected_idx] = updated
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
                                report.results[selected_idx] = fresh
                                st.toast(f"🎉 Đã quét lại xong '{fresh.track_title}'!", icon="✅")
                                st.rerun()
                            except Exception as err:
                                st.error(f"Lỗi: {err}")

        if report.report_file_path and Path(report.report_file_path).exists():
            with open(report.report_file_path, "rb") as f:
                st.download_button(
                    label="📥 Tải xuống File Báo cáo Excel (.xlsx)",
                    data=f.read(),
                    file_name=Path(report.report_file_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
