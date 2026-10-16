import inspect
from typing import Any

import streamlit as st


def get_width_kwargs(func: Any = st.button, stretch: bool = True) -> dict[str, Any]:
    """
    Trả về tham số độ rộng tương thích mọi phiên bản Streamlit (tránh cảnh báo deprecation use_container_width).
    """
    try:
        sig = inspect.signature(func)
        if "width" in sig.parameters:
            return {"width": "stretch" if stretch else "auto"}
    except Exception:
        pass
    return {"use_container_width": stretch}
