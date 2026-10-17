import streamlit as st

from backend.core.config import (
    AUDIO_SAMPLE_DURATION,
    AVAILABLE_MODELS,
    DEFAULT_GEMINI_MODEL,
    get_configured_gemini_api_keys,
    get_cookie_status,
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

        # 1. Trạng thái & Nhập Gemini API Key
        st.divider()
        st.subheader("🔑 Gemini API Key")
        configured_keys = get_configured_gemini_api_keys()
        user_override = st.session_state.get("gemini_api_key_override", "").strip()

        active_keys_count = len(configured_keys) + (
            1 if user_override and user_override not in configured_keys else 0
        )

        if active_keys_count > 0:
            st.success(f"✅ Đã kết nối ({active_keys_count} key khả dụng)")
        else:
            st.error("⚠️ Chưa có Gemini API Key!")

        custom_key_input = st.text_input(
            "Nhập/Đổi API Key trực tiếp:",
            value=user_override,
            type="password",
            placeholder="Dán key AQ... hoặc AIzaSy...",
            help="Lấy API Key hoàn toàn miễn phí tại https://aistudio.google.com/app/apikey",
        )
        if custom_key_input.strip() != user_override:
            st.session_state.gemini_api_key_override = custom_key_input.strip()
            st.rerun()

        with st.expander("📖 Hướng dẫn lưu vĩnh viễn trên Streamlit Cloud"):
            st.markdown(
                """
1. Mở trang quản lý app tại [share.streamlit.io](https://share.streamlit.io)
2. Chọn app của bạn -> bấm dấu **⋮** -> **Settings** -> **Secrets**
3. Dán dòng sau vào ô Secrets rồi bấm **Save**:
```toml
GEMINI_API_KEY_1 = "AQ.xxxxxx"
```
                """
            )

        # 2. Trạng thái YouTube Cookies trên Cloud
        st.divider()
        st.subheader("🍪 YouTube Cookies")
        has_cookies, cookie_count, cookie_msg = get_cookie_status()
        if has_cookies:
            st.info(f"🛡️ {cookie_msg}")
        else:
            st.warning("⚡ Chưa nạp cookie YouTube (Sẽ tự động fallback sang SoundCloud/Zing MP3)")

        st.divider()
        st.subheader("📊 Thông số Giám định")
        st.info(f"""
        - **Thời lượng mẫu:** {AUDIO_SAMPLE_DURATION} giây (3 phân đoạn)
        - **Phân đoạn ghép:** Verse (20%) + Chorus (50%) + Bridge (75%)
        - **Thuật toán phổ vân:** Deezer Research ISMIR 2025
        - **Tốc độ làm mượt:** 50ms Crossfade chống giật
        """)
