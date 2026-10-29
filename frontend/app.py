import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from backend.core.config import TEMP_DIR
from frontend.components.batch_view import render_batch_view

# UI Components
from frontend.components.header import render_header
from frontend.components.metrics_display import render_metrics_display
from frontend.components.sidebar import render_sidebar
from frontend.components.spectrogram_viewer import render_spectrogram_viewer
from frontend.components.ui_compat import get_width_kwargs
from frontend.components.verdict_card import render_verdict_card
from frontend.services.backend_client import BackendClient

# State & Services
from frontend.state.session_state import init_session_state


def main():
    # Page Config
    st.set_page_config(
        page_title="AI Music Detector - Deep Forensic & Gemini Flash Lite",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load CSS Styles
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Initialize Session State
    init_session_state()

    # Render Components
    render_sidebar()
    render_header()

    # Initialize Backend Client
    api_key_override = st.session_state.get("gemini_api_key_override") or None
    client = BackendClient(
        api_key_override=api_key_override, model_override=st.session_state.model_name
    )

    # Tabs
    tab1, tab2 = st.tabs(
        [
            "🔍 Thẩm định Đơn bài (Single Track)",
            "📁 Quét Hàng loạt (Batch Mode)",
        ]
    )

    # TAB 1: SINGLE TRACK
    with tab1:
        current_reason_lang = st.session_state.get("reason_language", "vi")
        lang_display = "Tiếng Việt" if current_reason_lang == "vi" else "English"
        st.caption(
            f"🌐 Ngôn ngữ Luận cứ kết quả: **{lang_display}** (thay đổi tại thanh bên ⚙️ Cấu hình)"
        )
        input_method = st.radio(
            "Phương thức thẩm định:",
            ["🔗 Nhập Link YouTube / Tên bài hát", "📤 Tải file âm thanh lên (.mp3, .wav, .m4a)"],
            horizontal=True,
        )

        query_to_process = None
        target_display_name = ""

        if input_method == "🔗 Nhập Link YouTube / Tên bài hát":
            col_input, col_btn = st.columns([4, 1])
            with col_input:
                query_input = st.text_input(
                    "Nhập liên kết YouTube, tên bài hát, hoặc URL âm thanh:",
                    placeholder="VD: https://www.youtube.com/watch?v=... hoặc 'Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau'",
                    key="single_query_input",
                )
            with col_btn:
                st.write("")
                st.write("")
                submit_btn = st.button(
                    "🚀 Bắt đầu Giám định",
                    type="primary",
                    key="btn_url",
                    **get_width_kwargs(st.button),
                )
            if submit_btn and query_input.strip():
                query_to_process = query_input.strip()
                target_display_name = query_input.strip()
        else:
            col_file, col_btn = st.columns([4, 1])
            with col_file:
                uploaded_audio = st.file_uploader(
                    "Tải lên tệp âm thanh trực tiếp (hoạt động 100% không phụ thuộc YouTube/Cloud bot filter):",
                    type=["mp3", "wav", "m4a", "flac", "ogg", "aac"],
                    key="single_audio_file",
                )
            with col_btn:
                st.write("")
                st.write("")
                submit_file_btn = st.button(
                    "🚀 Bắt đầu Giám định",
                    type="primary",
                    key="btn_file",
                    **get_width_kwargs(st.button),
                )
            if submit_file_btn and uploaded_audio:
                save_path = TEMP_DIR / uploaded_audio.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                query_to_process = str(save_path)
                target_display_name = uploaded_audio.name

        if query_to_process:
            req_lang = st.session_state.get("reason_language", "vi")
            with st.spinner(
                f"Đang thẩm định: '{target_display_name}' bằng {st.session_state.model_name} ({'Tiếng Việt' if req_lang == 'vi' else 'English'})..."
            ):
                try:
                    result = client.detect_single_track(
                        query=query_to_process,
                        model_override=st.session_state.model_name,
                        language=req_lang,
                    )
                    st.session_state.single_result = result
                except Exception as e:
                    err_msg = str(e)
                    if (
                        "Sign in to confirm you’re not a bot" in err_msg
                        or "HTTP Error 403" in err_msg
                        or "403: Forbidden" in err_msg
                        or "The page needs to be reloaded" in err_msg
                        or "Requested format is not available" in err_msg
                    ):
                        st.error(
                            "❌ **YouTube đã chặn tải trực tiếp từ máy chủ Cloud (Bot Detection / HTTP 403 Forbidden).**"
                        )
                        st.info(
                            """
💡 **2 Cách khắc phục:**

1. **Tải tệp âm thanh trực tiếp (Khuyên dùng - Hoạt động 100%):**
   Chuyển sang tùy chọn **'📤 Tải file âm thanh lên (.mp3, .wav, .m4a)'** ở trên. Tải file nhạc về máy rồi tải lên đây để giám định ngay lập tức mà không phụ thuộc YouTube/Cloud bot filter.

2. **Cấu hình Cookies cho Streamlit Cloud:**
   Thêm nội dung file `cookies.txt` vào **Streamlit Cloud -> App Settings -> Secrets** với biến `YOUTUBE_COOKIES_CONTENT`.
"""
                        )
                    else:
                        st.error(f"❌ Lỗi khi phân tích: {e}")

        # Hiển thị kết quả
        if st.session_state.single_result:
            res = st.session_state.single_result

            st.divider()

            st.markdown(f"### 🎵 Kết Quả Thẩm Định: **{res.track_title}**")
            if res.source_url:
                st.caption(f"🔗 Nguồn: [{res.source_url}]({res.source_url})")

            # Thông báo nguồn gốc kết quả (Cache / Đính chính)
            if getattr(res, "is_human_corrected", False):
                st.success(
                    "🧑‍🎤 **Đã qua thẩm định & đính chính bởi con người:** Kết quả này đã được người dùng/chuyên gia kiểm tra và ghi đè nhãn chuẩn."
                )
            elif getattr(res, "is_cached", False):
                st.info(
                    "⚡ **Kết quả được tải tức thì từ CSDL (Cached):** Bài hát này đã từng được thẩm định trước đó. Nếu bạn muốn phân tích lại từ đầu bằng AI, bấm nút '🔄 Quét lại bài này' bên dưới."
                )

            # Thẻ phán quyết
            render_verdict_card(res)

            # Khu vực tương tác: Quét lại hoặc Đính chính
            st.markdown("##### 🛠️ Thao Tác & Đính Chính Kết Quả")
            col_act1, col_act2, col_act3 = st.columns([1, 1, 1])

            with col_act1:
                if st.button(
                    "🔄 Quét lại bài này",
                    key=f"rescan_{res.track_id}",
                    use_container_width=True,
                ):
                    with st.spinner("Đang tải lại âm thanh và chạy lại toàn bộ mô hình AI..."):
                        try:
                            rescan_query = res.source_url or res.raw_query or res.track_title
                            fresh_res = client.detect_single_track(
                                query=rescan_query,
                                track_id=res.track_id,
                                model_override=st.session_state.model_name,
                                force_rescan=True,
                                language=st.session_state.get("reason_language", "vi"),
                            )
                            st.session_state.single_result = fresh_res
                            st.toast("🎉 Đã quét lại thành công và cập nhật CSDL!", icon="✅")
                            st.rerun()
                        except Exception as re_err:
                            st.error(f"Lỗi khi quét lại: {re_err}")

            with col_act2:
                if st.button(
                    "🧑‍🎤 Đính chính: Là Người",
                    key=f"corr_human_{res.track_id}",
                    use_container_width=True,
                ):
                    try:
                        updated = client.correct_verdict(
                            track_id=res.track_id,
                            is_ai=False,
                            verdict_type="CONFIRMED_HUMAN",
                            verdict_label="CONFIRMED HUMAN (Người làm)",
                            note="Xác nhận bởi người dùng: Tác phẩm của người thật",
                        )
                        st.session_state.single_result = updated
                        st.toast("✅ Đã đính chính: Tác phẩm của Người thật!", icon="🧑‍🎤")
                        st.rerun()
                    except Exception as corr_err:
                        st.error(f"Lỗi khi đính chính: {corr_err}")

            with col_act3:
                if st.button(
                    "🤖 Đính chính: Là Nhạc AI",
                    key=f"corr_ai_{res.track_id}",
                    use_container_width=True,
                ):
                    try:
                        updated = client.correct_verdict(
                            track_id=res.track_id,
                            is_ai=True,
                            verdict_type="AI_GENERATED",
                            verdict_label="AI-GENERATED (Nhạc AI tạo)",
                            note="Xác nhận bởi người dùng: Sản phẩm sinh từ AI",
                        )
                        st.session_state.single_result = updated
                        st.toast("🤖 Đã đính chính: Tác phẩm do AI tạo!", icon="🤖")
                        st.rerun()
                    except Exception as corr_err:
                        st.error(f"Lỗi khi đính chính: {corr_err}")

            # Thông số âm học
            if res.acoustic_metrics:
                render_metrics_display(res.acoustic_metrics)

            # Biểu đồ quang phổ & Audio preview
            render_spectrogram_viewer(res.spectrogram_image_path)

            # Luận cứ và giải thích chi tiết
            col_reasons, col_detail = st.columns([1, 1])
            is_en_lang = st.session_state.get("reason_language", "vi") == "en"
            with col_reasons:
                st.subheader(
                    "📌 Key Forensic Reasons" if is_en_lang else "📌 Luận cứ Giám định Chính"
                )
                for r in res.reasons:
                    st.markdown(f"- **{r}**")

            with col_detail:
                st.subheader(
                    "📝 Detailed Forensic Report" if is_en_lang else "📝 Báo cáo Thẩm âm Chi tiết"
                )
                st.info(res.detailed_explanation)

    # TAB 2: BATCH SCAN
    with tab2:
        render_batch_view(client)


if __name__ == "__main__":
    main()
