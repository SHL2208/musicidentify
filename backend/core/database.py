import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()


def get_database_url() -> str:
    """Chuẩn hóa URL kết nối PostgreSQL cho SQLAlchemy (hỗ trợ cả psycopg2 và SSL)."""
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        return ""

    url = raw_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    # Đảm bảo có sslmode=require khi kết nối Supabase Cloud
    if "sslmode=" not in url and ("supabase" in url or "aws" in url):
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    return url


DATABASE_URL = get_database_url()

engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Tự động ping kiểm tra kết nối sống trước khi query
            pool_recycle=300,  # Tái tạo kết nối sau 5 phút tránh bị cloud timeout
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Khởi tạo SQLAlchemy Engine kết nối tới Database thành công.")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo SQLAlchemy engine: {e}")
        engine = None
        SessionLocal = None


@contextmanager
def get_db_session() -> Generator:
    """Context manager cung cấp Database Session an toàn tự động commit/rollback/close."""
    if SessionLocal is None:
        raise RuntimeError("Database connection chưa được cấu hình hoặc DATABASE_URL không hợp lệ.")
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_database_available() -> bool:
    """Kiểm tra kết nối CSDL có khả dụng hay không."""
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database health check thất bại: {e}")
        return False
