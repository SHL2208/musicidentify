"""Đồng bộ danh mục nghệ sĩ AI và Whitelist người thật vào CSDL & Snapshot.

Tuân thủ nghiêm ngặt PEP 8, Ruff và Type Safety.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

# Đảm bảo UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.core.config import CATALOG_SNAPSHOT_PATH
from backend.core.database import get_db_session
from backend.core.models.db_models import (
    AIArtistProjectModel,
    VerifiedHumanArtistModel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NEW_AI_PROJECTS = [
    {
        "project_name": "YNTT",
        "regex_pattern": r"\b(yntt)\b",
        "suspected_engine": "Suno v3.5/v4 (Vietnamese Folk/Ballad)",
        "confidence": 0.98,
    },
    {
        "project_name": "THIÊN ĐÌNH",
        "regex_pattern": r"(thiên\s*đình|thien\s*dinh)",
        "suspected_engine": "Suno v3.5/v4",
        "confidence": 0.98,
    },
    {
        "project_name": "Kẻ Viết Nhạc",
        "regex_pattern": r"(kẻ\s*viết\s*nhạc|ke\s*viet\s*nhac)",
        "suspected_engine": "Suno/Udio",
        "confidence": 0.98,
    },
    {
        "project_name": "Kikoly",
        "regex_pattern": r"\b(kikoly)\b",
        "suspected_engine": "AI Vocal Synthesizer",
        "confidence": 0.98,
    },
    {
        "project_name": "TrapNova",
        "regex_pattern": r"\b(trapnova)\b",
        "suspected_engine": "AI Music Generator",
        "confidence": 0.98,
    },
    {
        "project_name": "Seorin",
        "regex_pattern": r"\b(seorin)\b",
        "suspected_engine": "AI Virtual Idol",
        "confidence": 0.98,
    },
    {
        "project_name": "Jennmi Nguyen",
        "regex_pattern": r"\b(jennmi\s*nguyen)\b",
        "suspected_engine": "Suno/Udio",
        "confidence": 0.98,
    },
    {
        "project_name": "Nhạc Phẩm Để Đời",
        "regex_pattern": r"(nhạc\s*phẩm\s*để\s*đời|nhac\s*pham\s*de\s*doi)",
        "suspected_engine": "Suno/Udio AI Bolero",
        "confidence": 0.98,
    },
    {
        "project_name": "Diễm Son",
        "regex_pattern": r"(diễm\s*son|diem\s*son)",
        "suspected_engine": "Suno v3.5/v4 EDM",
        "confidence": 0.98,
    },
    {
        "project_name": "Nhật Linh",
        "regex_pattern": r"(nhật\s*linh|nhat\s*linh)",
        "suspected_engine": "AI Virtual Vocal",
        "confidence": 0.98,
    },
    {
        "project_name": "Ca sĩ Giấu Mặt",
        "regex_pattern": r"(ca\s*sĩ\s*giấu\s*mặt|ca\s*si\s*giau\s*mat)",
        "suspected_engine": "AI Vocal Clone / Remix",
        "confidence": 0.98,
    },
    {
        "project_name": "Quang Bình",
        "regex_pattern": r"(quang\s*bình|quang\s*binh)",
        "suspected_engine": "Suno AI Cover",
        "confidence": 0.98,
    },
    {
        "project_name": "Hải Đăng",
        "regex_pattern": r"(hải\s*đăng|hai\s*dang)",
        "suspected_engine": "AI Pop Music Generator",
        "confidence": 0.98,
    },
]

NEW_HUMAN_ARTISTS = [
    "Nguyễn Văn Chung",
    "Chu Hà My",
    "Anh Ba Trứng",
    "INSOLENT, Tô",
    "Phạm Vũ Luân",
    "Gia Huy Singer",
    "VuNhann",
    "Olew",
]


def sync_database() -> None:
    """Nạp các nghệ sĩ vào database PostgreSQL / Supabase."""
    try:
        from sqlalchemy import text

        with get_db_session() as session:
            # Sửa sequence nếu bị out of sync
            try:
                session.execute(
                    text(
                        "SELECT setval('verified_human_artists_id_seq', "
                        "COALESCE((SELECT MAX(id) FROM verified_human_artists), 0) + 1, false);"
                    )
                )
                session.execute(
                    text(
                        "SELECT setval('ai_artist_projects_id_seq', "
                        "COALESCE((SELECT MAX(id) FROM ai_artist_projects), 0) + 1, false);"
                    )
                )
                session.commit()
            except Exception as seq_err:
                logger.debug("Không thể setval sequence: %s", seq_err)

            # 1. Thêm AI Projects
            for ai in NEW_AI_PROJECTS:
                existing = (
                    session.query(AIArtistProjectModel)
                    .filter(AIArtistProjectModel.project_name == ai["project_name"])
                    .first()
                )
                if not existing:
                    new_item = AIArtistProjectModel(
                        project_name=ai["project_name"],
                        regex_pattern=ai["regex_pattern"],
                        suspected_engine=ai["suspected_engine"],
                        confidence=ai["confidence"],
                        is_active=True,
                    )
                    session.add(new_item)
                    logger.info("Đã thêm AI Project: %s", ai["project_name"])
                else:
                    existing.regex_pattern = ai["regex_pattern"]
                    existing.is_active = True

            # 2. Thêm Human Artists
            for h in NEW_HUMAN_ARTISTS:
                existing_h = (
                    session.query(VerifiedHumanArtistModel)
                    .filter(VerifiedHumanArtistModel.artist_name.ilike(h))
                    .first()
                )
                if not existing_h:
                    new_h = VerifiedHumanArtistModel(artist_name=h)
                    session.add(new_h)
                    logger.info("Đã thêm Verified Human Artist: %s", h)

            session.commit()
            logger.info("Đồng bộ Database thành công!")
    except Exception as e:
        logger.warning("Không thể kết nối DB trực tiếp (sẽ cập nhật Snapshot): %s", e)


def sync_snapshot() -> None:
    """Cập nhật dữ liệu vào file JSON snapshot cục bộ."""
    if not CATALOG_SNAPSHOT_PATH.exists():
        logger.warning("Không tìm thấy %s", CATALOG_SNAPSHOT_PATH)
        return

    with open(CATALOG_SNAPSHOT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # 1. Cập nhật Human Artists
    human_set = set(data.get("human_artists", []))
    for h in NEW_HUMAN_ARTISTS:
        human_set.add(h.lower().strip())
    data["human_artists"] = sorted(list(human_set))

    # 2. Cập nhật AI Artists
    ai_list = data.get("ai_artists", [])
    existing_patterns = {item[0] for item in ai_list if isinstance(item, (list, tuple))}
    for ai in NEW_AI_PROJECTS:
        pat = ai["regex_pattern"]
        if pat not in existing_patterns:
            ai_list.append([pat, ai["project_name"]])
            existing_patterns.add(pat)
    data["ai_artists"] = ai_list

    with open(CATALOG_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        "Đã cập nhật snapshot: %d human artists, %d ai artists.",
        len(data["human_artists"]),
        len(data["ai_artists"]),
    )


if __name__ == "__main__":
    sync_database()
    sync_snapshot()
