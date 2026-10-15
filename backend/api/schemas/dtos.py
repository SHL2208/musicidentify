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


class BatchDetectRequest(BaseModel):
    queries: list[str] = Field(description="Danh sách link URL hoặc tên bài hát cần quét")
    model_override: str | None = Field(default=None, description="Mô hình Gemini chỉ định")


class ExportReportRequest(BaseModel):
    format: str = Field(default="xlsx", description="Định dạng xuất báo cáo: 'xlsx' hoặc 'csv'")


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    ffmpeg_available: bool
    configured_api_keys_count: int
    default_model: str
