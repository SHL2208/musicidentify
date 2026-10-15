from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class InputType(str, Enum):
    URL = "URL"
    SONG_NAME = "SONG_NAME"
    LOCAL_FILE = "LOCAL_FILE"


class VerdictType(str, Enum):
    CONFIRMED_HUMAN = "CONFIRMED_HUMAN"  # Nhạc do con người sáng tác & thu âm tự nhiên
    PROCESSED_HUMAN = (
        "PROCESSED_HUMAN"  # Nhạc người làm có can thiệp phòng thu/Auto-tune/EDM/nén YouTube
    )
    HYBRID_AI = "HYBRID_AI"  # Tác phẩm lai ghép (Người phối khí + giọng AI, hoặc ngược lại)
    AI_GENERATED = "AI_GENERATED"  # Nhạc do AI sinh ra hoàn toàn (Suno, Udio...)
    UNCERTAIN = "UNCERTAIN"  # Chưa đủ căn cứ kết luận hoặc độ tin cậy thấp
    ERROR = "ERROR"  # Lỗi xảy ra trong quá trình xử lý / phân tích


class TrackItem(BaseModel):
    id: str
    raw_query: str
    input_type: InputType
    resolved_url: str | None = None
    track_title: str | None = None
    artist: str | None = None
    duration_sec: float | None = None
    platform: str | None = None
    is_verified_match: bool | None = None
    match_score: float | None = None
    preview_url: str | None = None
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message: str | None = None
    song_type: str | None = None
    release_date: str | None = None


class AcousticMetrics(BaseModel):
    sample_rate: int
    duration_sec: float
    high_freq_ratio_16k: float = Field(
        default=0.0, description="Tỷ lệ năng lượng dải tần > 16kHz so với toàn dải"
    )
    stereo_correlation: float = Field(
        default=1.0, description="Hệ số tương quan pha giữa hai kênh L/R (-1.0 đến 1.0)"
    )
    spectral_rolloff_95_hz: float = Field(
        default=0.0,
        description="Tần số ngưỡng 95% năng lượng âm thanh (Hz) - Đo độ mở dải tần tự nhiên",
    )
    spectral_rolloff_85_hz: float = Field(
        default=0.0, description="Tần số ngưỡng 85% năng lượng âm thanh (Hz)"
    )
    harmonic_ratio: float = Field(
        default=0.0,
        description="Tỷ lệ thành phần sóng hài âm nhạc (Harmonic) so với thành phần gõ/ồn (Percussive)",
    )
    fakeprint_score: float = Field(
        default=0.0,
        description="Điểm vết vân Fourier deconvolution (Deezer ISMIR 2025 algorithm: 0.0 - 1.0)",
    )
    fakeprint_detected: bool = Field(
        default=False,
        description="True nếu phát hiện dấu vân răng lược deconvolution đặc trưng của Suno/Udio",
    )
    compression_source: str = Field(
        default="Studio Master / Uncompressed",
        description="Dấu hiệu nén truyền tải: 'Opus/AAC (YouTube/Web)' hoặc 'Studio/Lossless'",
    )
    has_brickwall_cutoff: bool = Field(
        default=False,
        description="True nếu có vết ngắt dải tần đột ngột (do nén web hoặc AI đời cũ)",
    )
    pitch_jitter: float = Field(
        default=0.0,
        description="Chỉ số vi biến thiên cao độ giọng hát (Pitch Micro-Jitter: 0.0 - 1.0)",
    )
    pitch_rigidity_ratio: float = Field(
        default=0.0,
        description="Tỷ lệ cao độ phẳng cứng nhắc bất thường đặc trưng của giọng AI (0.0 - 1.0)",
    )
    pitch_anomaly_detected: bool = Field(
        default=False,
        description="True nếu phát hiện bất thường về vi cao độ giọng hát (quá phẳng hoặc lượng tử hóa nhân tạo)",
    )
    pitch_stability_label: str = Field(
        default="Natural Biological",
        description="Đánh giá đặc tính vi cao độ: 'Natural Biological', 'Studio Auto-Tuned', hoặc 'Synthetic Latent Quantized'",
    )
    vocal_formant_variance: float | None = Field(
        default=None,
        description="Phương sai biến thiên Formant trong dải giọng (Hz^2) - Bóc tách Auto-Tune người vs AI",
    )
    is_studio_autotune: bool = Field(
        default=False,
        description="True nếu phát hiện ca sĩ người thật dùng Auto-Tune phòng thu (F0 phẳng nhưng Formant biến thiên theo khẩu hình)",
    )
    stem_bleed_score: float | None = Field(
        default=None,
        description="Hệ số tương quan rò rỉ âm học giữa bộ gõ và dải vocal (0.0 - 1.0)",
    )
    stem_bleed_detected: bool = Field(
        default=False,
        description="True nếu phát hiện hiện tượng dính chùm âm học (Stem Bleed) đặc trưng của AI nguyên khối Suno/Udio",
    )


class DetectionResult(BaseModel):
    track_id: str
    track_title: str
    source_url: str | None = None
    input_type: InputType | None = None
    raw_query: str | None = None
    is_ai: bool = Field(
        description="True nếu thiên về AI (AI_GENERATED hoặc HYBRID_AI), False nếu là con người"
    )
    verdict_type: VerdictType = Field(
        default=VerdictType.CONFIRMED_HUMAN,
        description="Nhãn phân loại: CONFIRMED_HUMAN, PROCESSED_HUMAN, HYBRID_AI, AI_GENERATED, UNCERTAIN",
    )
    verdict_label: str = Field(
        description="Nhãn hiển thị trực quan (VD: 'CONFIRMED HUMAN', 'PROCESSED HUMAN', 'AI GENERATED')"
    )
    confidence_score: float = Field(description="Độ tin cậy từ 0.0 (0%) đến 1.0 (100%)")
    suspected_engine: str = Field(
        description="Nền tảng nghi vấn, ví dụ: 'Suno v3.5/v4', 'Udio v1.5', 'Stable Audio', hoặc 'None'"
    )
    reasons: list[str] = Field(description="Các luận điểm chính chứng minh kết luận")
    detailed_explanation: str = Field(
        description="Giải thích thẩm âm chuyên sâu về giọng hát, phổ tần, âm sắc và cấu trúc"
    )
    acoustic_metrics: AcousticMetrics | None = None
    spectrogram_image_path: str | None = None
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchReport(BaseModel):
    total_tracks: int
    processed_count: int
    ai_count: int
    human_count: int
    uncertain_count: int = 0
    failed_count: int
    results: list[DetectionResult]
    report_file_path: str | None = None
