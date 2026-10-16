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

    if st.button("🚀 Bắt đầu Quét Hàng Loạt", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(curr, total, track, res, err):
            pct = curr / total
            progress_bar.progress(pct)
            if res:
                status_text.text(f"[{curr}/{total}] ✅ {res.track_title} -> {res.verdict_label}")
            else:
                status_text.text(f"[{curr}/{total}] ❌ Lỗi: {track.raw_query} ({err})")

        with st.spinner("Đang xử lý toàn bộ danh sách..."):
            try:
                if file_path:
                    report = client.process_batch_file(
                        file_path=file_path,
                        model_override=st.session_state.model_name,
                        on_progress=update_progress,
                    )
                elif tracks_to_process:
                    report = client.process_batch_queries(
                        queries=tracks_to_process,
                        model_override=st.session_state.model_name,
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
            table_rows.append(
                {
                    "Tên bài hát": r.track_title,
                    "Nguồn link": r.source_url or "N/A",
                    "Kết luận": r.verdict_label,
                    "Độ tin cậy": f"{r.confidence_score * 100:.0f}%",
                    "Mô hình nghi vấn": r.suspected_engine,
                    "Luận cứ": " | ".join(r.reasons[:2]),
                }
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        if report.report_file_path and Path(report.report_file_path).exists():
            with open(report.report_file_path, "rb") as f:
                st.download_button(
                    label="📥 Tải xuống File Báo cáo Excel (.xlsx)",
                    data=f.read(),
                    file_name=Path(report.report_file_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
