import streamlit as st


def render_header():
    """Hiển thị phần tiêu đề thương hiệu chuyên nghiệp."""
    st.markdown('<div class="main-header">🎵 AI MUSIC DETECTOR AGENT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hệ thống Giám định Âm học Đa Tầng (Deep Audio Forensics + Fourier Fakeprint ISMIR 2025 + Gemini Flash Lite)</div>',
        unsafe_allow_html=True
    )
