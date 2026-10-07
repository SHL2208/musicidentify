import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

# State & Services
from frontend.state.session_state import init_session_state
from frontend.services.backend_client import BackendClient

# UI Components
from frontend.components.header import render_header
from frontend.components.sidebar import render_sidebar
from frontend.components.verdict_card import render_verdict_card
from frontend.components.metrics_display import render_metrics_display
from frontend.components.spectrogram_viewer import render_spectrogram_viewer
from frontend.components.batch_view import render_batch_view

# Page Config
st.set_page_config(
    page_title="AI Music Detector - Deep Forensic & Gemini Flash Lite",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Styles
css_file = Path(__file__).parent / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
init_session_state()

# Render Components
render_sidebar()
render_header()

# Initialize Backend Client
client = BackendClient(
    model_override=st.session_state.model_name
)

# Tabs
tab1, tab2 = st.tabs(["🔍 Thẩm định Đơn bài (Single Track)", "📁 Quét Hàng loạt (Batch Mode)"])

# TAB 1: SINGLE TRACK
with tab1:
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query_input = st.text_input(
            "Nhập liên kết YouTube, tên bài hát, hoặc đường dẫn file âm thanh:",
            placeholder="VD: https://www.youtube.com/watch?v=... hoặc 'Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau'",
            key="single_query_input"
        )
    with col_btn:
        st.write("")
        st.write("")
        submit_btn = st.button("🚀 Bắt đầu Giám định", type="primary", use_container_width=True)

    if submit_btn and query_input.strip():
        with st.spinner(f"Đang thẩm định: '{query_input.strip()}' bằng {st.session_state.model_name}..."):
            try:
                result = client.detect_single_track(
                    query=query_input.strip(),
                    model_override=st.session_state.model_name
                )
                st.session_state.single_result = result
            except Exception as e:
                st.error(f"❌ Lỗi khi phân tích: {e}")

    # Hiển thị kết quả
    if st.session_state.single_result:
        res = st.session_state.single_result
        st.divider()

        st.markdown(f"### 🎵 Kết Quả Thẩm Định: **{res.track_title}**")
        if res.source_url:
            st.caption(f"🔗 Nguồn: [{res.source_url}]({res.source_url})")

        # Thẻ phán quyết
        render_verdict_card(res)

        # Thông số âm học
        if res.acoustic_metrics:
            render_metrics_display(res.acoustic_metrics)

        # Biểu đồ quang phổ & Audio preview
        render_spectrogram_viewer(res.spectrogram_image_path)

        # Luận cứ và giải thích chi tiết
        col_reasons, col_detail = st.columns([1, 1])
        with col_reasons:
            st.subheader("📌 Luận cứ Giám định Chính")
            for idx, r in enumerate(res.reasons, 1):
                st.markdown(f"- **{r}**")

        with col_detail:
            st.subheader("📝 Báo cáo Thẩm âm Chi tiết")
            st.info(res.detailed_explanation)

# TAB 2: BATCH SCAN
with tab2:
    render_batch_view(client)
