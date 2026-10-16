from pathlib import Path

import streamlit as st
from PIL import Image


def render_spectrogram_viewer(image_path_str: str | None, audio_path_str: str | None = None):
    """Hiển thị biểu đồ phổ 3 tầng và khung nghe thử mẫu âm thanh (hỗ trợ cả Supabase Cloud URL và file local)."""
    if audio_path_str:
        is_audio_url = str(audio_path_str).startswith("http://") or str(audio_path_str).startswith("https://")
        if is_audio_url or Path(audio_path_str).exists():
            st.subheader("🔊 Mẫu Âm thanh Đa Phân đoạn (Multi-Segment)")
            st.audio(str(audio_path_str), format="audio/wav")

    if image_path_str:
        is_img_url = str(image_path_str).startswith("http://") or str(image_path_str).startswith("https://")
        if is_img_url:
            st.subheader("📈 Biểu đồ Pháp y Âm thanh 3 Tầng (Cloud Storage)")
            st.image(
                str(image_path_str),
                caption="Tầng 1: Mel Full Mix | Tầng 2: Harmonic Vocal HPSS | Tầng 3: Phổ vân Fourier Deconvolution ISMIR 2025",
                use_container_width=True
            )
        elif Path(image_path_str).exists():
            st.subheader("📈 Biểu đồ Pháp y Âm thanh 3 Tầng")
            img = Image.open(image_path_str)
            st.image(
                img,
                caption="Tầng 1: Mel Full Mix | Tầng 2: Harmonic Vocal HPSS | Tầng 3: Phổ vân Fourier Deconvolution ISMIR 2025",
                use_container_width=True
            )
