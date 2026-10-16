import streamlit as st

from backend.core.models.entities import DetectionResult, VerdictType


def render_verdict_card(result: DetectionResult):
    """Hiển thị thẻ phán quyết 4 tầng với phong cách giao diện trực quan cao cấp."""
    v_type = result.verdict_type

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

    st.markdown(
        f"""
    <div class="{box_class}">
        <div class="verdict-title" style="color: {title_color};">{icon} {tier_title}</div>
        <div style="font-size: 1.15rem; font-weight: 700; margin-top: 0.3rem;">
            Độ tin cậy: <span style="color: {title_color};">{result.confidence_score * 100:.1f}%</span>
        </div>
        <div class="badge-engine">Mô hình nghi vấn: {result.suspected_engine}</div>
        <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #CBD5E1;">{desc}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
