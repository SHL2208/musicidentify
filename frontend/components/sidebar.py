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

        # Tùy chọn ngôn ngữ cho Luận cứ (Reason)
        lang_options = ["Tiếng Việt", "English"]
        curr_lang = st.session_state.get("reason_language", "vi")
        lang_idx = 0 if curr_lang in ("vi", "Tiếng Việt") else 1
        selected_lang = st.selectbox(
            "🌐 Ngôn ngữ Luận cứ (Reason):",
            options=lang_options,
            index=lang_idx,
            help="Chọn ngôn ngữ hiển thị luận cứ phân tích (Reasons) và giải thích chi tiết của AI.",
            key="sb_reason_language",
        )
        st.session_state.reason_language = "vi" if selected_lang == "Tiếng Việt" else "en"

        st.divider()
        st.subheader("📊 Thông số Giám định")
        st.info(f"""
        - **Thời lượng mẫu:** {AUDIO_SAMPLE_DURATION} giây (3 phân đoạn)
        - **Phân đoạn ghép:** Verse (20%) + Chorus (50%) + Bridge (75%)
        - **Thuật toán phổ vân:** Deezer Research ISMIR 2025
        - **Tốc độ làm mượt:** 50ms Crossfade chống giật
        """)

        st.divider()
        st.subheader("🗄️ Danh mục & Bộ nhớ đệm")
        from backend.core.database import is_database_available

        db_online = is_database_available()
        if db_online:
            st.success("🟢 Database: Supabase Online")
        else:
            st.warning("🟠 Chế độ: Offline Snapshot")

        if st.button(
            "🔄 Làm mới Cache Danh mục",
            use_container_width=True,
            help="Tải lại danh mục Ca sĩ, Dự án AI, Bài hát Cover/Remix từ Supabase",
        ):
            try:
                from frontend.services.backend_client import BackendClient

                client = BackendClient()
                client.refresh_catalog_cache()
                st.toast("✅ Đã làm mới bộ nhớ đệm danh mục thành công!", icon="🚀")
            except Exception as e:
                st.error(f"Lỗi khi làm mới cache: {e}")
