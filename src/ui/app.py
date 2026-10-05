import os
import sys
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import module
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
from PIL import Image

from src.config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    AVAILABLE_MODELS,
    TEMP_DIR,
    OUTPUT_DIR
)
from src.models.schemas import VerdictType
from src.services.resolver import InputResolver
from src.services.gemini_classifier import GeminiClassifier
from src.services.batch_manager import BatchManager

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="AI Music Detector - Deep Forensic & Gemini Flash Lite",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Modern Glassmorphism)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #8A2BE2 50%, #00C0F2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .verdict-box-ai {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-human {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(59, 130, 246, 0.15) 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-processed {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(59, 130, 246, 0.15) 100%);
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-hybrid {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.18) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 2px solid #F97316;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-title {
        font-size: 1.6rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
    }
    .badge-engine {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        background: #374151;
        color: #F3F4F6;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CẤU HÌNH -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/sound-waves.png", width=64)
    st.title("Cấu hình Agent")

    api_key_env = GEMINI_API_KEY
    user_api_key = st.text_input(
        "Google AI Studio API Key:",
        value=api_key_env,
        type="password",
        help="Lấy miễn phí tại: https://aistudio.google.com/app/apikey"
    )

    if not user_api_key:
        st.warning("⚠️ Vui lòng nhập API Key để bắt đầu.")
        st.markdown("[👉 Lấy API Key miễn phí tại Google AI Studio](https://aistudio.google.com/app/apikey)")
    else:
        st.success("✅ Đã kết nối API Key")

    selected_model = st.selectbox(
        "Mô hình AI:",
        AVAILABLE_MODELS,
        index=0 if DEFAULT_GEMINI_MODEL == "gemini-3.1-flash-lite" else 1,
        help="Gemini 3.1 Flash Lite: Tối ưu tốc độ cao nhất.\nGemini 3.5 Flash Lite: Thẩm âm sâu và đối chiếu phổ âm kỹ càng."
    )

    st.markdown("---")
    st.markdown("### 🔬 Công nghệ thẩm định")
    st.markdown("""
    - **Fakeprint Fourier ISMIR 2025:** Bắt dấu vết gai chu kỳ Deconvolution của Suno/Udio.
    - **Phân loại 4 nhãn:** Chấm dứt bắt oan nghệ sĩ dùng Auto-Tune hay beat EDM.
    - **Khử nhiễu Codec Web:** Tự động phát hiện nén Opus/AAC của YouTube.
    """)

    if st.button("🗑️ Dọn dẹp bộ nhớ tạm"):
        deleted = 0
        for f in TEMP_DIR.glob("*"):
            if f.is_file() and f.name != ".gitkeep":
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    pass
        st.toast(f"Đã dọn dẹp {deleted} tệp tạm thời!")

classifier = GeminiClassifier(api_key=user_api_key, model_name=selected_model) if user_api_key else None
batch_manager = BatchManager(gemini_classifier=classifier) if classifier else None

# ----------------- HEADER CHÍNH -----------------
st.markdown('<div class="main-header">🎵 AI Music Detector Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống Giám định Âm thanh AI Đa tầng (Fourier Fakeprint ISMIR 2025 & Gemini Flash Lite)</div>', unsafe_allow_html=True)

# ----------------- TABS ĐIỀU HƯỚNG -----------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Kiểm tra 1 bài hát",
    "📋 Quét danh sách bài hát",
    "📁 Tải file Excel / CSV"
])

# ================= TAB 1: KIỂM TRA 1 BÀI =================
with tab1:
    st.markdown("##### Nhập đường link hoặc tên bài hát cần kiểm tra:")
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query = st.text_input(
            "Link hoặc Tên bài:",
            placeholder="VD: https://www.youtube.com/watch?v=... hoặc 'Sơn Tùng - Đừng làm trái tim anh đau'",
            label_visibility="collapsed"
        )
    with col_btn:
        btn_scan = st.button("🚀 Phân Tích", type="primary", use_container_width=True)

    if btn_scan:
        if not user_api_key:
            st.error("Vui lòng nhập Google AI Studio API Key ở thanh bên trái!")
        elif not query.strip():
            st.warning("Vui lòng nhập link bài hát hoặc tên bài hát!")
        else:
            with st.spinner("Đang phân giải link, tính toán vân Fourier Deconvolution và đối chiếu âm học..."):
                try:
                    track = InputResolver.resolve_track(query.strip())
                    st.info(f"Đang kiểm tra: **{track.track_title}** ({track.input_type.value})")

                    result = batch_manager.process_single_track(track, model_override=selected_model)

                    st.markdown("---")
                    col_verdict, col_metrics = st.columns([1, 1])

                    with col_verdict:
                        v_type = result.verdict_type
                        if v_type == VerdictType.AI_GENERATED:
                            st.markdown(f"""
                            <div class="verdict-box-ai">
                                <div style="color: #EF4444; font-size: 1.05rem; font-weight: bold;">KẾT QUẢ THẨM ĐỊNH (AI)</div>
                                <div class="verdict-title" style="color: #EF4444;">🤖 BÀI HÁT DO AI TẠO HOÀN TOÀN</div>
                                <div style="font-size: 1.05rem;">Độ tin cậy: <b>{result.confidence_score * 100:.1f}%</b></div>
                                <div class="badge-engine">Nền tảng nghi vấn: {result.suspected_engine}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif v_type == VerdictType.PROCESSED_HUMAN:
                            st.markdown(f"""
                            <div class="verdict-box-processed">
                                <div style="color: #F59E0B; font-size: 1.05rem; font-weight: bold;">KẾT QUẢ THẨM ĐỊNH (HUMAN DIGITAL)</div>
                                <div class="verdict-title" style="color: #F59E0B;">🎛️ BÀI HÁT DO NGƯỜI LÀM (KỸ THUẬT SỐ)</div>
                                <div style="font-size: 1.05rem;">Độ tin cậy: <b>{result.confidence_score * 100:.1f}%</b></div>
                                <div class="badge-engine">Đặc trưng: Auto-Tune / VST Synthesizer / Nén Web</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif v_type == VerdictType.HYBRID_AI:
                            st.markdown(f"""
                            <div class="verdict-box-hybrid">
                                <div style="color: #F97316; font-size: 1.05rem; font-weight: bold;">KẾT QUẢ THẨM ĐỊNH (LAI GHÉP)</div>
                                <div class="verdict-title" style="color: #F97316;">🧬 TÁC PHẨM LAI GHÉP (BÁN NHÂN TẠO)</div>
                                <div style="font-size: 1.05rem;">Độ tin cậy: <b>{result.confidence_score * 100:.1f}%</b></div>
                                <div class="badge-engine">Đặc trưng: Kết hợp Người làm + Công cụ AI</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="verdict-box-human">
                                <div style="color: #10B981; font-size: 1.05rem; font-weight: bold;">KẾT QUẢ THẨM ĐỊNH (HUMAN)</div>
                                <div class="verdict-title" style="color: #10B981;">👨‍🎤 BÀI HÁT DO CON NGƯỜI LÀM THỰC THỤ</div>
                                <div style="font-size: 1.05rem;">Độ tin cậy: <b>{result.confidence_score * 100:.1f}%</b></div>
                                <div class="badge-engine">Nguồn gốc: Human Studio Recording</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### 🎯 Các luận điểm chính:")
                        for reason in result.reasons:
                            st.markdown(f"- {reason}")

                        st.markdown("#### 📝 Đánh giá thẩm định chi tiết:")
                        st.info(result.detailed_explanation)

                    with col_metrics:
                        st.markdown("#### 📊 Bằng chứng Phổ âm học & Fakeprint Fourier:")
                        if result.spectrogram_image_path and os.path.isfile(result.spectrogram_image_path):
                            img = Image.open(result.spectrogram_image_path)
                            st.image(img, use_container_width=True, caption="Biểu đồ Mel-Spectrogram & Phổ vân Fourier ISMIR 2025")

                        if result.acoustic_metrics:
                            m = result.acoustic_metrics
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                fakeprint_status = "⚠️ NGHI VẤN AI" if m.fakeprint_detected else "✅ TỰ NHIÊN"
                                st.metric("Điểm Fakeprint Fourier", f"{m.fakeprint_score * 100:.1f}%", fakeprint_status)
                            with col_m2:
                                st.metric("Độ mở dải tần Rolloff 95%", f"{m.spectral_rolloff_95_hz:.0f} Hz")

                            col_m3, col_m4 = st.columns(2)
                            with col_m3:
                                st.metric("Đánh giá nén truyền tải", m.compression_source)
                            with col_m4:
                                st.metric("Tỷ lệ sóng hài (Harmonic)", f"{m.harmonic_ratio * 100:.1f}%")

                            if "Opus/AAC" in m.compression_source:
                                st.caption("ℹ️ *Dải cao bài hát bị suy giảm tự nhiên do nén video trên YouTube (Opus/AAC). Hệ thống đã tự động bù trừ để không phán nhầm cho nghệ sĩ.*")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")

# ================= TAB 2: QUÉT DANH SÁCH =================
with tab2:
    st.markdown("##### Dán danh sách các bài hát hoặc đường link (Mỗi dòng 1 bài):")
    batch_text = st.text_area(
        "Danh sách bài hát:",
        height=150,
        placeholder="https://www.youtube.com/watch?v=...\nSơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau\nĐen Vâu - Trốn Tìm\nSuno AI Pop Song 2024",
        label_visibility="collapsed"
    )
    btn_batch_scan = st.button("🚀 Quét Toàn Bộ Danh Sách", type="primary")

    if btn_batch_scan:
        if not user_api_key:
            st.error("Vui lòng nhập Google AI Studio API Key ở thanh bên trái!")
        elif not batch_text.strip():
            st.warning("Vui lòng dán danh sách bài hát!")
        else:
            tracks = InputResolver.parse_batch_text(batch_text)
            st.info(f"Đã tìm thấy **{len(tracks)} bài hát** trong danh sách. Bắt đầu phân tích...")

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            table_placeholder = st.empty()

            live_results = []

            def on_progress(idx, total, current_track, res, err):
                pct = idx / total
                progress_bar.progress(pct)
                if res:
                    status_text.text(f"[{idx}/{total}] Đã quét xong: {res.track_title} -> {res.verdict_label}")
                    live_results.append({
                        "Tên bài hát": res.track_title,
                        "Kết luận": res.verdict_label,
                        "Phân loại": res.verdict_type.value if hasattr(res.verdict_type, 'value') else str(res.verdict_type),
                        "Độ tin cậy": f"{res.confidence_score*100:.1f}%",
                        "Nền tảng": res.suspected_engine,
                        "Lý do": " | ".join(res.reasons)
                    })
                else:
                    status_text.text(f"[{idx}/{total}] Lỗi khi quét: {current_track.raw_query}")
                table_placeholder.dataframe(pd.DataFrame(live_results), use_container_width=True)

            report = batch_manager.process_batch(
                tracks,
                model_override=selected_model,
                on_progress=on_progress
            )

            st.success(f"🎉 Hoàn thành quét {report.total_tracks} bài! (Phát hiện: {report.ai_count} AI, {report.human_count} Con người, {report.failed_count} Lỗi)")

            if report.report_file_path and os.path.isfile(report.report_file_path):
                with open(report.report_file_path, "rb") as f:
                    st.download_button(
                        label="📥 Tải Báo Cáo Excel Kết Quả",
                        data=f,
                        file_name=os.path.basename(report.report_file_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# ================= TAB 3: TẢI FILE =================
with tab3:
    st.markdown("##### Tải lên file chứa danh sách bài hát (.xlsx, .csv, .txt):")
    uploaded_file = st.file_uploader(
        "Kéo thả file vào đây:",
        type=["csv", "xlsx", "xls", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        save_path = TEMP_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            file_tracks = InputResolver.parse_batch_file(save_path)
            st.success(f"Đã đọc thành công **{len(file_tracks)} bài hát** từ file `{uploaded_file.name}`.")

            preview_data = [{"STT": i+1, "Tên / Link": t.raw_query, "Loại": t.input_type.value} for i, t in enumerate(file_tracks[:10])]
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
            if len(file_tracks) > 10:
                st.caption(f"... và {len(file_tracks) - 10} bài hát khác.")

            btn_run_file = st.button("🚀 Bắt Đầu Xử Lý File Hàng Loạt", type="primary")

            if btn_run_file:
                if not user_api_key:
                    st.error("Vui lòng nhập Google AI Studio API Key ở thanh bên trái!")
                else:
                    file_prog = st.progress(0.0)
                    file_status = st.empty()
                    file_table = st.empty()
                    file_results = []

                    def on_file_progress(idx, total, current_track, res, err):
                        file_prog.progress(idx / total)
                        if res:
                            file_status.text(f"[{idx}/{total}] Đã quét: {res.track_title} -> {res.verdict_label}")
                            file_results.append({
                                "Tên bài hát": res.track_title,
                                "Kết luận": res.verdict_label,
                                "Phân loại": res.verdict_type.value if hasattr(res.verdict_type, 'value') else str(res.verdict_type),
                                "Độ tin cậy": f"{res.confidence_score*100:.1f}%",
                                "Nền tảng": res.suspected_engine,
                                "Lý do": " | ".join(res.reasons)
                            })
                        else:
                            file_status.text(f"[{idx}/{total}] Lỗi: {current_track.raw_query}")
                        file_table.dataframe(pd.DataFrame(file_results), use_container_width=True)

                    report = batch_manager.process_batch(
                        file_tracks,
                        model_override=selected_model,
                        on_progress=on_file_progress
                    )

                    st.success(f"🎉 Hoàn thành xử lý file! (AI: {report.ai_count} | Người: {report.human_count} | Lỗi: {report.failed_count})")

                    if report.report_file_path and os.path.isfile(report.report_file_path):
                        with open(report.report_file_path, "rb") as f:
                            st.download_button(
                                label="📥 Tải File Báo Cáo Kết Quả Excel (.xlsx)",
                                data=f,
                                file_name=os.path.basename(report.report_file_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
