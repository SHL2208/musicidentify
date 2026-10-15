from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEYS, get_ffmpeg_binary
from ..core.exceptions import MusicIdentifyException
from .routers.batch import router as batch_router
from .routers.reports import router as reports_router
from .routers.tracks import router as tracks_router
from .schemas.dtos import HealthResponse

app = FastAPI(
    title="AI Music Detector API",
    description="Enterprise Multi-Tier Forensic Audio Detection API (FAANG Clean Architecture)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(MusicIdentifyException)
async def custom_exception_handler(request: Request, exc: MusicIdentifyException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "details": exc.details,
            "error": exc.__class__.__name__,
        },
    )


# Include Routers
app.include_router(tracks_router, prefix="/api/v1")
app.include_router(batch_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Kiểm tra tình trạng sức khỏe của API server."""
    ffmpeg_exe = get_ffmpeg_binary()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        ffmpeg_available=bool(ffmpeg_exe and len(ffmpeg_exe) > 0),
        configured_api_keys_count=len(GEMINI_API_KEYS),
        default_model=DEFAULT_GEMINI_MODEL,
    )
