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
    if "batch_is_running" not in st.session_state:
        st.session_state.batch_is_running = False
    if "batch_stop_requested" not in st.session_state:
        st.session_state.batch_stop_requested = False
    if "batch_live_rows" not in st.session_state:
        st.session_state.batch_live_rows = []
    if "batch_live_results" not in st.session_state:
        st.session_state.batch_live_results = []
    if "batch_live_counts" not in st.session_state:
        st.session_state.batch_live_counts = {
            "ai": 0,
            "human": 0,
            "uncertain": 0,
            "failed": 0,
        }
    if "batch_last_file_path" not in st.session_state:
        st.session_state.batch_last_file_path = None
    if "batch_last_queries" not in st.session_state:
        st.session_state.batch_last_queries = []
    if "batch_last_scan_mode" not in st.session_state:
        st.session_state.batch_last_scan_mode = "smart"
    if "batch_start_time" not in st.session_state:
        st.session_state.batch_start_time = None
    if "batch_partial_excel_path" not in st.session_state:
        st.session_state.batch_partial_excel_path = None
    if "history_results" not in st.session_state:
        st.session_state.history_results = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    if "reason_language" not in st.session_state:
        st.session_state.reason_language = "vi"
