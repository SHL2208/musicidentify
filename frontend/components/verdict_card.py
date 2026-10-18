import streamlit as st

from backend.core.models.entities import DetectionResult, VerdictType


def render_verdict_card(result: DetectionResult):
    """Hiển thị thẻ phán quyết kép: Đánh giá Model AI & Kết luận Tổng hợp."""
    v_type = result.verdict_type

    # 1. Cấu hình giao diện cho Cột 2 (Kết luận Tổng hợp)
    if v_type == VerdictType.CONFIRMED_HUMAN:
        box_class = "verdict-box-human"
        title_color = "#10B981"
        icon = "🧑‍🎤"
        tier_title = "CONFIRMED HUMAN (Con người sáng tác & Biểu diễn)"
        desc = "Âm thanh tự nhiên, dải tần thoáng rộng, không phát hiện dấu vết sinh sóng AI."
    elif v_type == VerdictType.PROCESSED_HUMAN:
        box_class = "verdict-box-processed"
        title_color = "#F59E0B"
        icon = "🎛️"
        tier_title = "PROCESSED HUMAN (Người làm - Can thiệp phòng thu/Kỹ thuật số)"
        desc = "Ca sĩ người thật có can thiệp Auto-Tune/EDM/nén OTT hoặc codec YouTube Opus/AAC."
    elif v_type == VerdictType.HYBRID_AI:
        box_class = "verdict-box-hybrid"
        title_color = "#F97316"
        icon = "🧬"
        tier_title = "HYBRID / AI-ASSISTED (Tác phẩm lai ghép Người & AI)"
        desc = "Sự kết hợp giữa người hòa âm và vocal AI clone, hoặc ca sĩ thật hát trên beat sinh từ AI."
    elif v_type == VerdictType.AI_GENERATED:
        box_class = "verdict-box-ai"
        title_color = "#EF4444"
        icon = "🤖"
        tier_title = "AI-GENERATED (Nhạc sinh hoàn toàn bằng AI)"
        desc = "Tác phẩm sinh trực tiếp từ các mô hình Suno/Udio (hòa bệt âm sắc, dính noise cymbals, phổ răng lược)."
    elif v_type == VerdictType.UNCERTAIN:
        box_class = "verdict-box-uncertain"
        title_color = "#94A3B8"
        icon = "❔"
        tier_title = "UNCERTAIN (Chưa đủ căn cứ kết luận)"
        desc = "Dữ liệu âm thanh chưa đủ rõ ràng, bị nhiễu nặng hoặc có dấu hiệu mâu thuẫn để phân loại chắc chắn."
    else:
        box_class = "verdict-box-ai"
        title_color = "#EF4444"
        icon = "❌"
        tier_title = "ERROR (Lỗi thẩm định)"
        desc = "Có lỗi kỹ thuật xảy ra trong quá trình xử lý bài hát này."

    # 2. Cấu hình giao diện cho Cột 1 (Đánh giá Model AI)
    ai_raw_is_ai = getattr(result, "ai_model_is_ai", result.is_ai)
    ai_raw_verdict = (
        getattr(result, "ai_model_verdict", None)
        or ("AI_GENERATED" if ai_raw_is_ai else "CONFIRMED_HUMAN")
    )
    if ai_raw_is_ai:
        ai_box_class = "verdict-box-ai"
        ai_color = "#EF4444"
        ai_icon = "🤖"
        ai_title = f"AI MODEL: {ai_raw_verdict}"
        ai_desc = "Mô hình AI phân tích tín hiệu nhận diện các đặc trưng sinh sóng nhân tạo."
    else:
        ai_box_class = "verdict-box-human"
        ai_color = "#10B981"
        ai_icon = "🧑‍🎤"
        ai_title = f"AI MODEL: {ai_raw_verdict}"
        ai_desc = "Mô hình AI nhận định âm thanh tự nhiên mang đặc tính thanh nhạc của con người."

    # Bảng phân tích kép (Dual Verdict Display)
    st.markdown("#### 🎯 Kết Quả Đối Soát Kép (Dual Output)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
        <div class="{ai_box_class}" style="min-height: 220px;">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 0.2rem;">
                Cột 1: Đánh Giá Model AI
            </div>
            <div class="verdict-title" style="color: {ai_color}; font-size: 1.35rem;">
                {ai_icon} {ai_title}
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; margin-top: 0.4rem;">
                Phân loại thô: <span style="color: {ai_color};">{'Có yếu tố AI' if ai_raw_is_ai else 'Con người'}</span>
            </div>
            <div style="margin-top: 0.6rem; font-size: 0.9rem; color: #CBD5E1;">{ai_desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        consensus_text = getattr(result, "consensus_verdict", None) or tier_title
        st.markdown(
            f"""
        <div class="{box_class}" style="min-height: 220px;">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 0.2rem;">
                Cột 2: Kết Luận Tổng Hợp
            </div>
            <div class="verdict-title" style="color: {title_color}; font-size: 1.35rem;">
                {icon} {consensus_text}
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-top: 0.4rem;">
                Độ tin cậy: <span style="color: {title_color};">{result.confidence_score * 100:.1f}%</span>
            </div>
            <div class="badge-engine">Mô hình nghi vấn: {result.suspected_engine}</div>
            <div style="margin-top: 0.6rem; font-size: 0.9rem; color: #CBD5E1;">{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Cảnh báo nếu có sự điều chỉnh giữa Model AI và Tiêu chí tổng hợp
    if ai_raw_is_ai != result.is_ai:
        st.info(
            f"ℹ️ **Điều chỉnh qua Tiêu chí cứng:** Model AI nhận định là `{ai_raw_verdict}`, "
            f"nhưng Kết luận Tổng hợp được xác lập là `{result.verdict_label}` nhờ bộ lọc kiểm định nguồn âm/nghệ sĩ/vật lý âm học."
        )

