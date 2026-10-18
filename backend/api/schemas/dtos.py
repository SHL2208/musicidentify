from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class StandardResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "Success"
    data: T | None = None
    error: str | None = None


class TrackDetectRequest(BaseModel):
    query: str = Field(description="URL bài hát, tên bài hát, hoặc đường dẫn file âm thanh")
    model_override: str | None = Field(
        default=None, description="Mô hình Gemini chỉ định (nếu muốn ghi đè)"
    )
    force_rescan: bool = Field(
        default=False, description="Bỏ qua cache và buộc phân tích lại từ đầu"
    )


class BatchDetectRequest(BaseModel):
    queries: list[str] = Field(description="Danh sách link URL hoặc tên bài hát cần quét")
    model_override: str | None = Field(default=None, description="Mô hình Gemini chỉ định")
    scan_mode: str = Field(
        default="smart",
        description="Chế độ quét: 'smart' (bỏ qua bài có sẵn), 'force_all' (quét lại toàn bộ), 'retry_errors' (chỉ quét lỗi)",
    )


class TrackCorrectionRequest(BaseModel):
    track_id: str = Field(description="ID bài hát cần đính chính")
    is_ai: bool = Field(description="True nếu là AI, False nếu là Người thật")
    verdict_type: str = Field(
        description="Nhãn VerdictType (CONFIRMED_HUMAN, AI_GENERATED, HYBRID_AI)"
    )
    verdict_label: str = Field(description="Nhãn hiển thị trực quan")
    note: str | None = Field(default=None, description="Lý do hoặc ghi chú đính chính")


class ExportReportRequest(BaseModel):
    format: str = Field(default="xlsx", description="Định dạng xuất báo cáo: 'xlsx' hoặc 'csv'")


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    ffmpeg_available: bool
    configured_api_keys_count: int
    default_model: str
