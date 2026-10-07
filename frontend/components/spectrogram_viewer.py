import streamlit as st
from pathlib import Path
from PIL import Image

def render_spectrogram_viewer(image_path_str: str | None, audio_path_str: str | None = None):
    """Hiển thị biểu đồ phổ 2 tầng và khung nghe thử mẫu âm thanh."""
    if audio_path_str and Path(audio_path_str).exists():
        st.subheader("🔊 Mẫu Âm thanh Đa Phân đoạn (Multi-Segment)")
        st.audio(str(audio_path_str), format="audio/wav")

    if image_path_str and Path(image_path_str).exists():
        st.subheader("📈 Biểu đồ Pháp y Âm thanh 2 Tầng")
        img = Image.open(image_path_str)
        st.image(
            img,
            caption="Tầng 1: Mel-Spectrogram toàn diện | Tầng 2: Phổ vân Fourier Deconvolution ISMIR 2025",
            use_container_width=True
        )
