
import streamlit as st

from backend.core.config import DEFAULT_GEMINI_MODEL


def init_session_state():
    """Khởi tạo trạng thái phiên làm việc chuẩn hóa cho frontend."""
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_GEMINI_MODEL
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "batch_report" not in st.session_state:
        st.session_state.batch_report = None
    if "history_results" not in st.session_state:
        st.session_state.history_results = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
