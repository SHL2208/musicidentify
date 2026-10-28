import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend.core.config import TEMP_DIR
from backend.services.batch_job_manager import (
    JobStatus,
    format_detection_row,
    format_time_duration,
)
from frontend.components.ui_compat import get_width_kwargs
from frontend.services.backend_client import BackendClient

logger = logging.getLogger(__name__)


def render_notification_script(enable: bool = True):
    """Nhúng Web Notification & Web Audio API (chuông Chime) để báo động khi quét xong."""
    if not enable:
        return
    script = """
    <script>
    if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission();
    }
    window.playBatchChime = function() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const now = ctx.currentTime;
            const osc1 = ctx.createOscillator();
            const osc2 = ctx.createOscillator();
            const gain = ctx.createGain();
            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(587.33, now); // D5
            osc2.type = 'sine';
            osc2.frequency.setValueAtTime(880.00, now + 0.15); // A5
            gain.gain.setValueAtTime(0.2, now);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.9);
            osc1.connect(gain);
            osc2.connect(gain);
            gain.connect(ctx.destination);
            osc1.start(now);
            osc1.stop(now + 0.2);
            osc2.start(now + 0.15);
            osc2.stop(now + 0.9);
        } catch(e) {}
    };
    </script>
    """
    if hasattr(st, "html"):
        st.html(script, unsafe_allow_javascript=True)
    else:
        components.html(script, height=0)


def trigger_completion_alert(job_name: str, total_tracks: int):
    """Kích hoạt chuông và desktop notification khi job hoàn thành."""
    safe_name = job_name.replace('"', '\\"')
    script = f"""
    <script>
    if (typeof window.playBatchChime === 'function') {{
        window.playBatchChime();
    }} else if (typeof window.parent !== 'undefined' && typeof window.parent.playBatchChime === 'function') {{
        window.parent.playBatchChime();
    }}
    const notifObj = window.Notification || (window.parent && window.parent.Notification);
    if (notifObj && notifObj.permission === "granted") {{
        new notifObj("🎵 AI Music Detector - Hoàn tất!", {{
            body: "Đã quét xong lô '{safe_name}' ({total_tracks} bài)!",
            icon: "https://emojicdn.elk.sh/🎵"
        }});
    }}
    </script>
    """
    if hasattr(st, "html"):
        st.html(script, unsafe_allow_javascript=True)
    else:
        components.html(script, height=0)


def render_batch_view(client: BackendClient):
    """
    Giao diện Quét Hàng Loạt chạy ngầm độc lập (Decoupled Background Worker & FIFO Queue).
    Hỗ trợ chuyển tab, tắt tab, F5 reload mà không bị mất tiến trình quét.
    """
    st.markdown("### 📁 Quét Hàng Loạt (Batch Mode)")
    st.write(
        "Tải lên tệp (.xlsx, .csv, .txt) hoặc nhập trực tiếp danh sách bài hát / link YouTube. "
        "Hệ thống chạy trên **Luồng ngầm độc lập (Background Worker)**: Bạn có thể chuyển tab, "
        "thu nhỏ trình duyệt hoặc làm việc khác thoải mái mà không lo bị ngắt quãng."
    )

    # 1. Tùy chọn thông báo Desktop & Âm thanh
    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt1:
        enable_notifications = st.toggle(
            "🔔 Bật thông báo Desktop & Chuông khi quét xong (Dù đang ở tab khác)",
            value=st.session_state.get("batch_enable_notifications", True),
            key="batch_notif_toggle",
            help="Hệ thống sẽ gửi Desktop Notification và phát tiếng chuông nhẹ khi hoàn thành 100%.",
        )
        st.session_state.batch_enable_notifications = enable_notifications
    with col_opt2:
        st.caption("🛡️ Bounded Concurrency: 1 Worker")

    render_notification_script(enable_notifications)

    # 2. Truy xuất trạng thái từ Background Job Manager
    active_snapshot = client.get_batch_job_snapshot()
    queue_snapshots = client.get_batch_queue_snapshots()

    is_running = bool(active_snapshot and active_snapshot.status == JobStatus.RUNNING)
    is_paused = bool(active_snapshot and active_snapshot.status == JobStatus.PAUSED)
    is_stopped = bool(active_snapshot and active_snapshot.status == JobStatus.STOPPED)
    is_completed = bool(active_snapshot and active_snapshot.status == JobStatus.COMPLETED)
    has_results = bool(active_snapshot and active_snapshot.live_rows)

    # 3. Hiển thị Panel Hàng Đợi (Queue Drawer / Panel) nếu có task đang chờ
    if queue_snapshots:
        with st.expander(f"📬 Hàng Đợi Đang Chờ ({len(queue_snapshots)} lô bài hát)", expanded=True):
            st.info("Hệ thống xử lý tuần tự từng lô để đảm bảo CPU/RAM ổn định và tránh bị YouTube chặn IP.")
            for q_idx, q_job in enumerate(queue_snapshots):
                qc1, qc2, qc3, qc4 = st.columns([1, 4, 2, 2])
                with qc1:
                    st.markdown(f"**#{q_idx + 1}**")
                with qc2:
                    st.markdown(f"**{q_job.job_name}** ({q_job.total_tracks or 'Chờ đếm'} bài)")
                with qc3:
                    st.caption(f"Chế độ: `{q_job.scan_mode}`")
                with qc4:
                    if st.button("❌ Hủy yêu cầu", key=f"cancel_q_{q_job.job_id}", use_container_width=True):
                        if client.cancel_queued_batch_job(q_job.job_id):
                            st.toast(f"Đã hủy lô '{q_job.job_name}' khỏi hàng đợi!", icon="🗑️")
                            st.rerun()

    # 4. Form nộp lô mới (Chỉ hiển thị hoặc cho phép khi không bị trùng)
    with st.expander("➕ Nộp Lô Bài Hát Mới", expanded=not is_running and not has_results):
        upload_mode = st.radio(
            "Phương thức nhập dữ liệu:",
            ["Tải lên tệp (.txt, .csv, .xlsx)", "Nhập danh sách trực tiếp"],
            horizontal=True,
            key="batch_input_mode",
        )

        tracks_to_process = []
        file_path = None

        if upload_mode == "Tải lên tệp (.txt, .csv, .xlsx)":
            uploaded_file = st.file_uploader(
                "Chọn tệp danh sách bài hát:", type=["txt", "csv", "xlsx", "xls"], key="batch_file_uploader"
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
                height=180,
                key="batch_text_queries",
            )
            if text_queries.strip():
                tracks_to_process = [q.strip() for q in text_queries.split("\n") if q.strip()]
                st.session_state.batch_last_queries = tracks_to_process

        scan_mode_choice = st.radio(
            "🎯 **Chế độ quét hàng loạt:**",
            options=[
                "⚡ Quét gia số thông minh (Tự động dùng lại CSDL đã có - Tiết kiệm thời gian & Quota)",
                "🔄 Quét lại toàn bộ (Bỏ qua CSDL và ép buộc AI phân tích lại - Cập nhật đè CSDL)",
                "🛠️ Chỉ quét lại bài bị lỗi hoặc chưa có kết luận",
            ],
            index=0,
            key="batch_scan_mode_radio",
        )
        scan_mode_map = {
            "⚡ Quét gia số thông minh (Tự động dùng lại CSDL đã có - Tiết kiệm thời gian & Quota)": "smart",
            "🔄 Quét lại toàn bộ (Bỏ qua CSDL và ép buộc AI phân tích lại - Cập nhật đè CSDL)": "force_all",
            "🛠️ Chỉ quét lại bài bị lỗi hoặc chưa có kết luận": "retry_errors",
        }
        selected_scan_mode = scan_mode_map[scan_mode_choice]

        col_submit, col_hint = st.columns([2, 3])
        with col_submit:
            start_btn_label = "🚀 Bắt đầu Quét Hàng Loạt" if not is_running else "➕ Xếp Vào Hàng Đợi (Đang Quét)"
            btn_type = "primary" if not is_running else "secondary"
            if st.button(start_btn_label, type=btn_type, use_container_width=True, key="btn_start_batch"):
                target_file = file_path
                target_queries = tracks_to_process
                if not target_file and not target_queries:
                    st.warning("Vui lòng tải lên tệp (.xlsx, .csv) hoặc nhập danh sách bài hát trước!")
                else:
                    success, msg, job = client.enqueue_batch_job(
                        file_path=target_file,
                        queries=target_queries,
                        scan_mode=selected_scan_mode,
                        model_override=st.session_state.model_name,
                        language=st.session_state.get("reason_language", "vi"),
                    )
                    if success:
                        st.toast(msg, icon="🚀")
                        st.rerun()
                    else:
                        st.error(msg)
        with col_hint:
            if is_running:
                st.caption("⏳ Hệ thống đang chạy 1 lô. Lô mới sẽ tự động vào hàng đợi và chạy tiếp khi xong.")
            else:
                st.caption("🚀 Quá trình quét chạy trên luồng ngầm độc lập. Bạn có thể chuyển tab thoải mái.")

    st.divider()

    # 5. KHU VỰC HIỂN THỊ TIẾN TRÌNH THỜI GIAN THỰC & ĐIỀU KHIỂN
    # Định nghĩa hàm render tiến trình (hỗ trợ st.fragment tự refresh nếu có)
    def render_progress_dashboard():
        curr_snap = client.get_batch_job_snapshot()
        if not curr_snap:
            st.info("Chưa có tác vụ quét nào. Hãy tải tệp hoặc nhập danh sách ở trên để bắt đầu.")
            return

        c_status = curr_snap.status
        c_running = c_status == JobStatus.RUNNING
        c_paused = c_status == JobStatus.PAUSED
        c_stopped = c_status == JobStatus.STOPPED
        c_completed = c_status == JobStatus.COMPLETED

        # Banner trạng thái
        if c_running:
            st.markdown(
                f"#### 🚀 Đang Quét Ngầm: **{curr_snap.job_name}** "
                f"({curr_snap.current_index}/{curr_snap.total_tracks} bài)"
            )
            pct = (
                min(1.0, max(0.0, curr_snap.current_index / curr_snap.total_tracks))
                if curr_snap.total_tracks > 0
                else 0.0
            )
            st.progress(pct)

            col_inf1, col_inf2, col_inf3 = st.columns(3)
            col_inf1.caption(f"⏱️ Đã chạy: **{format_time_duration(curr_snap.elapsed_sec)}**")
            col_inf2.caption(f"⚡ Tốc độ: **{curr_snap.speed_per_track:.1f}s / bài**")
            col_inf3.caption(f"⏳ Dự kiến còn: **~{format_time_duration(curr_snap.eta_sec)}**")

            if curr_snap.current_track_title:
                st.markdown(f"Đang xử lý: 🎵 `{curr_snap.current_track_title}`")

        elif c_paused:
            st.warning(
                f"⚠️ **TẠM DỪNG AN TOÀN (Circuit Breaker):**\n\n"
                f"{curr_snap.pause_reason or 'Hệ thống đã tự động dừng để bảo toàn danh sách.'}\n\n"
                f"Đã xử lý: **{len(curr_snap.live_rows)}/{curr_snap.total_tracks} bài**. "
                f"Bạn có thể kiểm tra lại kết nối/API key rồi bấm **▶️ Tiếp tục quét** bên dưới."
            )
        elif c_stopped:
            st.info(
                f"🛑 **Tiến trình quét đã dừng:** Đã hoàn thành {len(curr_snap.live_rows)}/{curr_snap.total_tracks} bài. "
                "Kết quả đã được bảo toàn đầy đủ."
            )
        elif c_completed:
            st.success(
                f"🎉 **ĐÃ HOÀN TẤT 100%:** Quét xong toàn bộ **{curr_snap.total_tracks} bài** của lô '{curr_snap.job_name}'!"
            )
            # Phát âm thanh & Notification nếu vừa hoàn thành
            last_notified = st.session_state.get("batch_last_notified_job_id")
            if last_notified != curr_snap.job_id and st.session_state.get("batch_enable_notifications", True):
                st.session_state.batch_last_notified_job_id = curr_snap.job_id
                trigger_completion_alert(curr_snap.job_name, curr_snap.total_tracks)

        # 5 Thẻ Metrics
        counts = curr_snap.counts
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Đã quét", f"{len(curr_snap.live_rows)}/{curr_snap.total_tracks}")
        m2.metric("Nghi vấn AI", counts.get("ai", 0), delta="AI Detected", delta_color="inverse")
        m3.metric("Người làm", counts.get("human", 0), delta="Human", delta_color="normal")
        m4.metric("Chưa rõ", counts.get("uncertain", 0))
        m5.metric("Lỗi / Bỏ qua", counts.get("failed", 0))

        # Nút điều khiển
        st.write("")
        act_c1, act_c2, act_c3, act_c4 = st.columns([2, 2, 2, 2])

        with act_c1:
            if c_running:
                if st.button("🛑 Dừng Quét (Lưu kết quả)", type="primary", use_container_width=True, key="btn_stop_job"):
                    if client.stop_batch_job():
                        st.toast("🛑 Đang dừng quét an toàn sau bài hiện tại...", icon="🛑")
                        st.rerun()
            elif c_paused or c_stopped:
                remain_c = curr_snap.total_tracks - len(curr_snap.live_rows)
                if st.button(
                    f"▶️ Tiếp tục quét ({remain_c} bài còn lại)",
                    type="primary",
                    use_container_width=True,
                    key="btn_resume_job",
                ):
                    succ, r_msg = client.resume_batch_job()
                    if succ:
                        st.toast(r_msg, icon="▶️")
                        st.rerun()
                    else:
                        st.error(r_msg)

        with act_c2:
            if not c_running and curr_snap.live_rows:
                if st.button("🔄 Đặt lại từ đầu", use_container_width=True, key="btn_reset_job"):
                    client.reset_batch_job()
                    st.toast("Đã đặt lại trạng thái!", icon="🔄")
                    st.rerun()

        # Nút tải Excel
        excel_path = curr_snap.report_file_path or curr_snap.partial_excel_path
        if not excel_path and curr_snap.live_results:
            try:
                excel_path = client.export_custom_results(curr_snap.live_results, format="xlsx")
                curr_snap.partial_excel_path = excel_path
            except Exception:
                pass

        if excel_path and Path(excel_path).exists():
            with act_c3:
                with open(excel_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Tải Excel ({len(curr_snap.live_rows)} bài)",
                        data=f.read(),
                        file_name=Path(excel_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="btn_download_excel",
                    )

        # Bảng dữ liệu chi tiết
        st.markdown(f"##### 📋 Danh Sách Chi Tiết ({len(curr_snap.live_rows)} bài)")
        if curr_snap.live_rows:
            df_rows = pd.DataFrame(curr_snap.live_rows)
            # Luôn đảm bảo cột STT bắt đầu từ 1 và đứng ở vị trí đầu tiên
            if "STT" not in df_rows.columns:
                df_rows.insert(0, "STT", range(1, len(df_rows) + 1))
            else:
                df_rows["STT"] = range(1, len(df_rows) + 1)
                cols = ["STT"] + [c for c in df_rows.columns if c != "STT"]
                df_rows = df_rows[cols]
            st.dataframe(df_rows, hide_index=True, **get_width_kwargs(st.dataframe))
        else:
            st.dataframe(pd.DataFrame(curr_snap.live_rows), hide_index=True, **get_width_kwargs(st.dataframe))

        # Can thiệp nhanh từng bài khi không đang chạy
        if not c_running and curr_snap.live_results:
            with st.expander("✍️ Can thiệp & Đính chính nhanh từng bài trong danh sách", expanded=False):
                st.caption("Chọn bài hát để đính chính kết quả hoặc quét lại độc lập:")
                live_res_list = curr_snap.live_results
                track_labels = [
                    f"{idx + 1}. {r.track_title} [{r.verdict_label}]" for idx, r in enumerate(live_res_list)
                ]
                selected_idx = st.selectbox(
                    "Chọn bài cần can thiệp:",
                    range(len(track_labels)),
                    format_func=lambda i: track_labels[i],
                    key="batch_corr_select",
                )
                if selected_idx is not None and selected_idx < len(live_res_list):
                    target_res = live_res_list[selected_idx]
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
                                live_res_list[selected_idx] = updated
                                curr_snap.live_rows[selected_idx] = format_detection_row(updated, stt=selected_idx + 1)
                                st.toast(f"✅ Đã đính chính '{updated.track_title}' thành Người thật!", icon="🧑‍🎤")
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
                                live_res_list[selected_idx] = updated
                                curr_snap.live_rows[selected_idx] = format_detection_row(updated, stt=selected_idx + 1)
                                st.toast(f"🤖 Đã đính chính '{updated.track_title}' thành Nhạc AI!", icon="🤖")
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
                                    rescan_q = (
                                        target_res.source_url
                                        or target_res.raw_query
                                        or target_res.track_title
                                        or ""
                                    )
                                    if not rescan_q:
                                        st.error("Không tìm thấy thông tin query của bài hát để quét lại.")
                                        return
                                    fresh = client.detect_single_track(
                                        query=rescan_q,
                                        track_id=target_res.track_id,
                                        model_override=st.session_state.model_name,
                                        force_rescan=True,
                                    )
                                    live_res_list[selected_idx] = fresh
                                    curr_snap.live_rows[selected_idx] = format_detection_row(fresh, stt=selected_idx + 1)
                                    st.toast(f"🎉 Đã quét lại xong '{fresh.track_title}'!", icon="✅")
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Lỗi: {err}")

    # Bọc trong st.fragment(run_every="2s") nếu Streamlit hỗ trợ, giúp tự làm mới khi đang quét
    if hasattr(st, "fragment") and is_running:

        @st.fragment(run_every="2s")
        def live_fragment():
            render_progress_dashboard()

        live_fragment()
    else:
        render_progress_dashboard()
