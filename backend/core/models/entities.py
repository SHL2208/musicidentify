from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class InputType(str, Enum):
    URL = "URL"
    SONG_NAME = "SONG_NAME"
    LOCAL_FILE = "LOCAL_FILE"

class VerdictType(str, Enum):
    CONFIRMED_HUMAN = "CONFIRMED_HUMAN"      # Nhạc do con người sáng tác & thu âm tự nhiên
    PROCESSED_HUMAN = "PROCESSED_HUMAN"      # Nhạc người làm có can thiệp phòng thu/Auto-tune/EDM/nén YouTube
    HYBRID_AI = "HYBRID_AI"                  # Tác phẩm lai ghép (Người phối khí + giọng AI, hoặc ngược lại)
    AI_GENERATED = "AI_GENERATED"            # Nhạc do AI sinh ra hoàn toàn (Suno, Udio...)
    ERROR = "ERROR"                          # Lỗi xảy ra trong quá trình xử lý / phân tích

class TrackItem(BaseModel):
    id: str
    raw_query: str
    input_type: InputType
    resolved_url: Optional[str] = None
    track_title: Optional[str] = None
    artist: Optional[str] = None
    duration_sec: Optional[float] = None
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message: Optional[str] = None

class AcousticMetrics(BaseModel):
    sample_rate: int
    duration_sec: float
    high_freq_ratio_16k: float = Field(
        default=0.0,
        description="Tỷ lệ năng lượng dải tần > 16kHz so với toàn dải"
    )
    stereo_correlation: float = Field(
        default=1.0,
        description="Hệ số tương quan pha giữa hai kênh L/R (-1.0 đến 1.0)"
    )
    spectral_rolloff_95_hz: float = Field(
        default=0.0,
        description="Tần số ngưỡng 95% năng lượng âm thanh (Hz) - Đo độ mở dải tần tự nhiên"
    )
    spectral_rolloff_85_hz: float = Field(
        default=0.0,
        description="Tần số ngưỡng 85% năng lượng âm thanh (Hz)"
    )
    harmonic_ratio: float = Field(
        default=0.0,
        description="Tỷ lệ thành phần sóng hài âm nhạc (Harmonic) so với thành phần gõ/ồn (Percussive)"
    )
    fakeprint_score: float = Field(
        default=0.0,
        description="Điểm vết vân Fourier deconvolution (Deezer ISMIR 2025 algorithm: 0.0 - 1.0)"
    )
    fakeprint_detected: bool = Field(
        default=False,
        description="True nếu phát hiện dấu vân răng lược deconvolution đặc trưng của Suno/Udio"
    )
    compression_source: str = Field(
        default="Studio Master / Uncompressed",
        description="Dấu hiệu nén truyền tải: 'Opus/AAC (YouTube/Web)' hoặc 'Studio/Lossless'"
    )
    has_brickwall_cutoff: bool = Field(
        default=False,
        description="True nếu có vết ngắt dải tần đột ngột (do nén web hoặc AI đời cũ)"
    )

class DetectionResult(BaseModel):
    track_id: str
    track_title: str
    source_url: Optional[str] = None
    is_ai: bool = Field(
        description="True nếu thiên về AI (AI_GENERATED hoặc HYBRID_AI), False nếu là con người"
    )
    verdict_type: VerdictType = Field(
        default=VerdictType.CONFIRMED_HUMAN,
        description="Nhãn phân loại 4 tầng: CONFIRMED_HUMAN, PROCESSED_HUMAN, HYBRID_AI, AI_GENERATED"
    )
    verdict_label: str = Field(
        description="Nhãn hiển thị trực quan (VD: 'CONFIRMED HUMAN', 'PROCESSED HUMAN', 'AI GENERATED')"
    )
    confidence_score: float = Field(
        description="Độ tin cậy từ 0.0 (0%) đến 1.0 (100%)"
    )
    suspected_engine: str = Field(
        description="Nền tảng nghi vấn, ví dụ: 'Suno v3.5/v4', 'Udio v1.5', 'Stable Audio', hoặc 'None'"
    )
    reasons: List[str] = Field(
        description="Các luận điểm chính chứng minh kết luận"
    )
    detailed_explanation: str = Field(
        description="Giải thích thẩm âm chuyên sâu về giọng hát, phổ tần, âm sắc và cấu trúc"
    )
    acoustic_metrics: Optional[AcousticMetrics] = None
    spectrogram_image_path: Optional[str] = None
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class BatchReport(BaseModel):
    total_tracks: int
    processed_count: int
    ai_count: int
    human_count: int
    failed_count: int
    results: List[DetectionResult]
    report_file_path: Optional[str] = None
