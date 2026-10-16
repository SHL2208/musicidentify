import streamlit as st

from backend.core.config import (
    AUDIO_SAMPLE_DURATION,
    AVAILABLE_MODELS,
    DEFAULT_GEMINI_MODEL,
)


def render_sidebar():
    """Hiển thị bảng điều khiển cấu hình bên sườn."""
    with st.sidebar:
        st.header("⚙️ Cấu hình Hệ thống")

        model_selected = st.selectbox(
            "Mô hình AI Thẩm định:",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_GEMINI_MODEL)
            if DEFAULT_GEMINI_MODEL in AVAILABLE_MODELS
            else 0,
            help="gemini-3.8-flash xịn nhất, 3.5-flash-lite nhanh nhất",
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
