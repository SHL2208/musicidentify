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
