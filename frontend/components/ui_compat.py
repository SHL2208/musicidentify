from typing import Any

import streamlit as st


def get_width_kwargs(func: Any = None, stretch: bool = True) -> dict[str, Any]:
    """
    Trả về tham số độ rộng tương thích mọi phiên bản Streamlit (tránh cảnh báo deprecation use_container_width).
    Streamlit >= 1.40 đã deprecate 'use_container_width' và thay bằng 'width="stretch"' hoặc 'width="content"'.
    """
    try:
        ver_parts = [int(p) for p in st.__version__.split(".")[:2] if p.isdigit()]
        if len(ver_parts) >= 2 and (ver_parts[0] > 1 or (ver_parts[0] == 1 and ver_parts[1] >= 40)):
            return {"width": "stretch" if stretch else "content"}
    except Exception:
        pass

    return {"use_container_width": stretch}


_SPOTIFY_TOGGLE_KEYS = (
    "sb_enable_spotify",
    "single_enable_spotify",
    "batch_enable_spotify",
)


def _sync_spotify_toggle(source_key: str) -> None:
    """Đồng bộ trạng thái On/Off Check Spotify giữa Sidebar và các Tab."""
    new_val = bool(st.session_state.get(source_key, True))
    st.session_state.enable_spotify = new_val
    for k in _SPOTIFY_TOGGLE_KEYS:
        if k != source_key and k in st.session_state:
            st.session_state[k] = new_val


def render_spotify_toggle(
    key: str,
    label: str = "🎧 Check Spotify khi thẩm định AI",
    help_text: str = (
        "Bật để tự động tra cứu Spotify (làm sạch tên bài/nghệ sĩ, gắn link/metadata Spotify "
        "và bỏ qua tải audio nếu bản gốc phát hành trước 11/2022). "
        "Tắt để AI thẩm định trực tiếp mà không gọi Spotify."
    ),
) -> bool:
    """Hiển thị nút gạt On/Off Check Spotify và giữ đồng bộ giữa mọi vị trí trên UI."""
    if "enable_spotify" not in st.session_state:
        st.session_state.enable_spotify = True
    if key not in st.session_state:
        st.session_state[key] = bool(st.session_state.enable_spotify)

    val = st.toggle(
        label,
        key=key,
        on_change=_sync_spotify_toggle,
        args=(key,),
        help=help_text,
    )
    st.session_state.enable_spotify = bool(val)
    return bool(val)
