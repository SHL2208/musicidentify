from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from ..database import Base


def get_utc_now() -> datetime:
    """Trả về thời gian UTC có múi giờ chuẩn."""
    return datetime.now(timezone.utc)


class TrackModel(Base):
    """Bảng lưu trữ bài hát và siêu dữ liệu đã phân giải."""

    __tablename__ = "tracks"

    id = Column(String(64), primary_key=True)
    raw_query = Column(Text, nullable=False)
    input_type = Column(String(20), nullable=False)
    resolved_url = Column(Text, nullable=True)
    track_title = Column(String(255), nullable=True)
    artist = Column(String(255), nullable=True)
    duration_sec = Column(Float, nullable=True)
    platform = Column(String(50), nullable=True)
    song_type = Column(String(50), nullable=True)
    release_date = Column(String(50), nullable=True)
    distributor = Column(String(255), nullable=True)
    is_verified_match = Column(Boolean, nullable=True)
    match_score = Column(Float, nullable=True)
    status = Column(String(20), default="PENDING")
    created_at = Column(DateTime(timezone=True), default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), default=get_utc_now, onupdate=get_utc_now)

    detection_result = relationship(
        "DetectionResultModel", back_populates="track", uselist=False, cascade="all, delete-orphan"
    )
    media_assets = relationship(
        "MediaAssetModel", back_populates="track", cascade="all, delete-orphan"
    )


class DetectionResultModel(Base):
    """Bảng lưu trữ kết luận thẩm định từ AI / Heuristics."""

    __tablename__ = "detection_results"

    id = Column(String(64), primary_key=True)
    track_id = Column(
        String(64), ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    is_ai = Column(Boolean, nullable=False, index=True)
    verdict_type = Column(String(30), nullable=False)
    verdict_label = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    suspected_engine = Column(String(50), default="None")
    reasons = Column(JSON, default=list)
    detailed_explanation = Column(Text, nullable=True)
    model_used = Column(String(50), nullable=True)
    processed_at = Column(DateTime(timezone=True), default=get_utc_now)

    track = relationship("TrackModel", back_populates="detection_result")
    acoustic_metrics = relationship(
        "AcousticMetricsModel",
        back_populates="detection_result",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AcousticMetricsModel(Base):
    """Bảng lưu trữ thông số âm học và pháp y chi tiết."""

    __tablename__ = "acoustic_metrics"

    id = Column(String(64), primary_key=True)
    result_id = Column(
        String(64),
        ForeignKey("detection_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sample_rate = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)
    fakeprint_score = Column(Float, default=0.0)
    fakeprint_detected = Column(Boolean, default=False)
    spectral_rolloff_95_hz = Column(Float, default=0.0)
    spectral_rolloff_85_hz = Column(Float, default=0.0)
    harmonic_ratio = Column(Float, default=0.0)
    pitch_jitter = Column(Float, default=0.0)
    pitch_rigidity_ratio = Column(Float, default=0.0)
    pitch_anomaly_detected = Column(Boolean, default=False)
    pitch_stability_label = Column(String(50), nullable=True)
    is_studio_autotune = Column(Boolean, default=False)
    stem_bleed_score = Column(Float, nullable=True)
    stem_bleed_detected = Column(Boolean, default=False)
    ultrasonic_cutoff_detected = Column(Boolean, default=False)
    compression_source = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    detection_result = relationship("DetectionResultModel", back_populates="acoustic_metrics")


class MediaAssetModel(Base):
    """Bảng quản lý file âm thanh và ảnh biểu đồ quang phổ."""

    __tablename__ = "media_assets"

    id = Column(String(64), primary_key=True)
    track_id = Column(
        String(64), ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_type = Column(String(30), nullable=False)  # 'AUDIO_SAMPLE', 'SPECTROGRAM_FORENSIC'
    file_path = Column(Text, nullable=False)
    public_url = Column(Text, nullable=True)
    file_size = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    track = relationship("TrackModel", back_populates="media_assets")


class DistributorCatalogModel(Base):
    """Bảng danh mục nhà phân phối âm nhạc AI / Whitelist."""

    __tablename__ = "distributor_catalog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    catalog_type = Column(String(50), nullable=False)  # 'AI_DISTRIBUTOR', 'HUMAN_WHITELIST'
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)


class AIArtistProjectModel(Base):
    """Bảng danh mục dự án nghệ sĩ ảo AI sinh hàng loạt."""

    __tablename__ = "ai_artist_projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_name = Column(String(255), unique=True, nullable=False)
    regex_pattern = Column(String(255), nullable=False)
    suspected_engine = Column(String(50), default="AI Virtual Project")
    confidence = Column(Float, default=0.98)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)


class BenchmarkRunModel(Base):
    """Bảng lưu trữ các đợt chạy Benchmark chấm điểm."""

    __tablename__ = "benchmark_runs"

    id = Column(String(64), primary_key=True)
    run_name = Column(String(255), nullable=True)
    dataset_source = Column(String(255), default="testcase.xlsx")
    total_samples = Column(Integer, nullable=False)
    evaluated_samples = Column(Integer, nullable=False)
    error_count = Column(Integer, default=0)
    tp = Column(Integer, default=0)
    tn = Column(Integer, default=0)
    fp = Column(Integer, default=0)
    fn = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    specificity = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    confusion_matrix_image_url = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    items = relationship(
        "BenchmarkItemModel", back_populates="benchmark_run", cascade="all, delete-orphan"
    )


class BenchmarkItemModel(Base):
    """Bảng chi tiết từng bài hát trong một đợt benchmark."""

    __tablename__ = "benchmark_items"

    id = Column(String(64), primary_key=True)
    benchmark_run_id = Column(
        String(64), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    media_id = Column(String(64), nullable=True)
    track_title = Column(String(255), nullable=True)
    artist = Column(String(255), nullable=True)
    ground_truth = Column(String(10), nullable=False)
    prediction = Column(String(10), nullable=False)
    classification_type = Column(String(20), nullable=True)
    is_correct = Column(String(10), nullable=True)
    source_url = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    benchmark_run = relationship("BenchmarkRunModel", back_populates="items")
