import streamlit as st
from backend.core.config import (
    DEFAULT_GEMINI_MODEL,
    AVAILABLE_MODELS,
    AUDIO_SAMPLE_DURATION,
    AUDIO_SAMPLE_OFFSET
)

def render_sidebar():
    """Hiển thị bảng điều khiển cấu hình bên sườn."""
    with st.sidebar:
        st.header("⚙️ Cấu hình Hệ thống")

        model_selected = st.selectbox(
            "Mô hình AI Thẩm định:",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_GEMINI_MODEL) if DEFAULT_GEMINI_MODEL in AVAILABLE_MODELS else 0,
            help="gemini-3.1-flash-lite cho phản hồi siêu tốc, gemini-3.5-flash-lite cho phân tích sâu."
        )
        st.session_state.model_name = model_selected

        st.divider()
        st.subheader("📊 Thông số Giám định")
        st.info(f"""
        - **Thời lượng mẫu:** {AUDIO_SAMPLE_DURATION} giây (3 phân đoạn)
        - **Phân đoạn ghép:** Verse (20%) + Chorus (50%) + Bridge (75%)
        - **Thuật toán phổ vân:** Deezer Research ISMIR 2025
        - **Tốc độ làm mượt:** 50ms Crossfade chống giật
        """)

        st.divider()
        st.caption("Music Identify Agent v2.0 - FAANG Enterprise Architecture")
