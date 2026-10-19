"""Script đồng bộ danh mục dự án nghệ sĩ ảo AI và vô hiệu hóa Warner Music Vietnam Whitelist.

Đảm bảo tuân thủ nghiêm ngặt PEP 8 và Ruff.
Thực thi an toàn trên CSDL Supabase Cloud thông qua SQLAlchemy session.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import select

from backend.core.database import get_db_session
from backend.core.models.db_models import AIArtistProjectModel, DistributorCatalogModel
from backend.services.signal_analyzer_service import _DEFAULT_AI_ARTIST_REGEXES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sync_catalog() -> None:
    """Thực hiện đồng bộ dữ liệu vào CSDL PostgreSQL Supabase."""
    logger.info("Bắt đầu đồng bộ danh mục AI Artists và cập nhật Whitelist trên DB...")

    try:
        with get_db_session() as session:
            # 1. Cập nhật Warner Music Vietnam trong distributor_catalog
            stmt_warner = select(DistributorCatalogModel).where(
                DistributorCatalogModel.name.ilike("%warner%")
            )
            warner_records = session.scalars(stmt_warner).all()
            for rec in warner_records:
                logger.info(
                    "Tìm thấy bản ghi nhà phân phối: %s (Type: %s, Active: %s)",
                    rec.name,
                    rec.catalog_type,
                    rec.is_active,
                )
                rec.is_active = False
                rec.catalog_type = "NEUTRAL"
                rec.notes = "Vô hiệu hóa Whitelist cứng do Warner phân phối nhiều catalogue AI"
            if warner_records:
                logger.info(
                    "Đã vô hiệu hóa %d bản ghi Warner Music khỏi Whitelist.", len(warner_records)
                )

            # 2. Đồng bộ các dự án AI Artists mới
            inserted_count = 0
            existing_count = 0

            for pattern, name in _DEFAULT_AI_ARTIST_REGEXES:
                stmt_artist = select(AIArtistProjectModel).where(
                    AIArtistProjectModel.project_name.ilike(name)
                )
                existing = session.scalars(stmt_artist).first()
                if existing:
                    existing_count += 1
                    if not existing.is_active or existing.regex_pattern != pattern:
                        existing.is_active = True
                        existing.regex_pattern = pattern
                        logger.info("Đã cập nhật regex cho nghệ sĩ AI: %s -> %s", name, pattern)
                else:
                    new_item = AIArtistProjectModel(
                        project_name=name,
                        regex_pattern=pattern,
                        suspected_engine="AI Virtual Project",
                        confidence=0.98,
                        is_active=True,
                    )
                    session.add(new_item)
                    inserted_count += 1
                    logger.info("Đã thêm mới nghệ sĩ AI vào CSDL: %s (%s)", name, pattern)

            session.commit()
            logger.info(
                "Hoàn tất đồng bộ! Thêm mới: %d, Đã tồn tại/cập nhật: %d",
                inserted_count,
                existing_count,
            )

    except Exception as e:
        logger.error("Lỗi trong quá trình đồng bộ CSDL: %s", e)
        raise


if __name__ == "__main__":
    sync_catalog()
