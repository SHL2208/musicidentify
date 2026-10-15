from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class IFakeprintDetector(ABC):
    """Interface định nghĩa thuật toán phát hiện dấu vết deconvolution Fourier (ISMIR 2025)."""

    @abstractmethod
    def analyze_waveform(self, y: np.ndarray, sr: int = 44100) -> dict[str, Any]:
        """
        Phân tích dạng sóng mono để trích xuất chỉ số Fakeprint Fourier.
        Trả về dict chứa các trường:
            - fakeprint_score: float (0.0 đến 1.0)
            - fakeprint_detected: bool
            - comb_prominence: float
            - peak_periodicity_hz: float
            - residual_spectrum: np.ndarray
            - freqs: np.ndarray
        """
        pass
