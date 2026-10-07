from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from backend.core.models.entities import AcousticMetrics

class ISignalAnalyzer(ABC):
    """Interface định nghĩa phân tích âm học và giám định pháp y tín hiệu âm thanh."""

    @abstractmethod
    def check_heuristics(self, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra các quy tắc Heuristics nhanh (Metadata, Ngày phát hành, AI tags).
        Nếu thỏa mãn chắc chắn, trả về dict kết luận ngay để tiết kiệm tài nguyên.
        """
        pass

    @abstractmethod
    def analyze_audio(self, audio_path: Path) -> Tuple[AcousticMetrics, Optional[Path]]:
        """
        Phân tích chuyên sâu tín hiệu âm học và tạo ảnh Forensic 2 tầng
        (Mel-Spectrogram + Phổ vân Deconvolution Fourier ISMIR 2025).
        Trả về: (AcousticMetrics, spectrogram_image_path)
        """
        pass
