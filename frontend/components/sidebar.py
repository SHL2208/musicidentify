import streamlit as st

from backend.core.config import (
    AUDIO_SAMPLE_DURATION,
    AVAILABLE_MODELS,
    DEFAULT_GEMINI_MODEL,
    get_cookie_status,
    get_local_cookies_as_base64,
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
        st.subheader("🍪 Trạng thái YouTube Cookies")
        has_cookies, count, msg = get_cookie_status()
        if has_cookies:
            st.success(f"🟢 {msg}")
        else:
            st.warning(f"🟡 {msg}")

        with st.expander("🔑 Hướng dẫn cấu hình Cookie trên Streamlit"):
            st.markdown(
                """
                **Cách nạp Cookie vào Streamlit Secrets để không bao giờ bị lỗi Tab/Space:**
                1. Mở Streamlit Cloud App -> **Settings** -> **Secrets**.
                2. Dán mã Base64 bên dưới vào:
                ```toml
                YOUTUBE_COOKIES_BASE64 = "..."
                ```
                *(Mã hóa Base64 giúp chống 100% tình trạng trình duyệt tự ý đổi Tab thành Space)*
                """
            )
            b64_str = get_local_cookies_as_base64()
            if b64_str:
                st.text_area(
                    "Mã Base64 cookie của bạn (chỉ cần copy dán vào Secrets):",
                    value=b64_str,
                    height=90,
                )

        st.divider()
        st.subheader("📊 Thông số Giám định")
        st.info(f"""
        - **Thời lượng mẫu:** {AUDIO_SAMPLE_DURATION} giây (3 phân đoạn)
        - **Phân đoạn ghép:** Verse (20%) + Chorus (50%) + Bridge (75%)
        - **Thuật toán phổ vân:** Deezer Research ISMIR 2025
        - **Tốc độ làm mượt:** 50ms Crossfade chống giật
        """)

