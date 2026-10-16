import streamlit as st

from backend.core.models.entities import AcousticMetrics


def render_metrics_display(metrics: AcousticMetrics):
    """Hiển thị các chỉ số vật lý âm học và giám định pháp y tín hiệu."""
    st.subheader("🔬 Thông số Âm học Pháp y (Audio Forensics Metrics)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fp_color = "normal" if not metrics.fakeprint_detected else "inverse"
        fp_tag = "⚠️ NGHI VẤN AI" if metrics.fakeprint_detected else "✅ TỰ NHIÊN"
        st.metric(
            label="Phổ vân Fourier (ISMIR 2025)",
            value=f"{metrics.fakeprint_score * 100:.1f}%",
            delta=fp_tag,
            delta_color=fp_color,
            help="Thuật toán phát hiện vết tầng Deconvolution đặc trưng của Suno/Udio (Deezer Research 2025).",
        )

    with col2:
        st.metric(
            label="Độ mở dải tần (Rolloff 95%)",
            value=f"{metrics.spectral_rolloff_95_hz:.0f} Hz",
            delta="Lossless/Studio"
            if metrics.spectral_rolloff_95_hz >= 18500
            else "Bị nén dải cao",
            help="Tần số giới hạn 95% năng lượng. Nhạc phòng thu thật thường vượt 18.5kHz.",
        )

    with col3:
        st.metric(
            label="Tỷ lệ sóng hài (Harmonic)",
            value=f"{metrics.harmonic_ratio * 100:.1f}%",
            help="Tỷ lệ năng lượng giữa thành phần sóng nhạc du dương so với thành phần gõ/ồn.",
        )

    with col4:
        st.metric(
            label="Tương quan pha Stereo",
            value=f"{metrics.stereo_correlation:.2f}",
            help="Độ tương quan pha L/R. AI thường có sự phân kỳ stereo hoặc mono giả lập.",
        )

    st.markdown(
        f"""
    <div style="background: rgba(30, 41, 59, 0.5); padding: 0.6rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-top: 0.5rem; font-size: 0.9rem;">
        📦 <b>Đánh giá định dạng truyền tải:</b> <span style="color: #38BDF8;">{metrics.compression_source}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
