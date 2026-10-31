"""Kịch bản đọc và import danh mục nghệ sĩ/dự án AI từ file Excel vào CSDL.

Đảm bảo tuân thủ nghiêm ngặt chuẩn PEP 8 và Ruff.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import openpyxl
from sqlalchemy import select

from ..core.database import get_db_session
from ..core.models.db_models import AIArtistProjectModel

logger = logging.getLogger(__name__)


def generate_artist_regex(name: str) -> str:
    """Tạo biểu thức chính quy nhận diện tên nghệ sĩ an toàn với ranh giới từ mềm."""
    cleaned = name.lower().strip()
    special_chars = r".^$*+?{}[]\|()"
    escaped = "".join(f"\\{c}" if c in special_chars else c for c in cleaned)
    left_b = r"\b" if re.match(r"^\w", cleaned) else r"(?<!\S)"
    right_b = r"\b" if re.search(r"\w$", cleaned) else r"(?!\w)"
    return f"{left_b}{escaped}{right_b}"


def parse_ai_artists_from_excel(file_path: str | Path, sheet_name: str = "NS AI") -> list[str]:
    """Đọc và lọc danh sách nghệ sĩ AI từ sheet Excel."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    wb = openpyxl.load_workbook(str(path_obj), data_only=True)
    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(f"Không tìm thấy sheet '{sheet_name}' trong file: {file_path}")

    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return []

    headers = [str(h).strip().lower() if h is not None else "" for h in rows[0]]
    artist_col = 0
    ai_col = 1 if len(headers) > 1 else -1

    for idx, h in enumerate(headers):
        if h in ("main", "artist", "nghệ sĩ", "tên"):
            artist_col = idx
        elif "ai" in h or "yes" in h:
            ai_col = idx

    seen: set[str] = set()
    result: list[str] = []

    for row in rows[1:]:
        if not row or artist_col >= len(row):
            continue
        val = row[artist_col]
        if val is None:
            continue
        artist_name = str(val).strip()
        if (
            not artist_name
            or artist_name.startswith("#")
            or artist_name.lower() in ("none", "nan", "null")
        ):
            continue

        if ai_col != -1 and ai_col < len(row):
            ai_val = str(row[ai_col] or "").strip().lower()
            if ai_val and "yes" not in ai_val and ai_val != "y":
                continue

        name_key = artist_name.lower()
        if name_key not in seen:
            seen.add(name_key)
            result.append(artist_name)

    return result


def import_ai_artists_to_db(file_path: str | Path, sheet_name: str = "NS AI") -> dict[str, int]:
    """Nhập danh sách nghệ sĩ AI vào CSDL PostgreSQL Supabase (Idempotent)."""
    artists = parse_ai_artists_from_excel(file_path, sheet_name=sheet_name)
    stats = {
        "total_parsed": len(artists),
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
    }

    with get_db_session() as session:
        existing_stmt = select(AIArtistProjectModel)
        existing_records = list(session.scalars(existing_stmt).all())
        existing_map: dict[str, Any] = {
            r.project_name.strip().lower(): r
            for r in existing_records
            if getattr(r, "project_name", None)
        }

        for artist in artists:
            pattern = generate_artist_regex(artist)
            key = artist.strip().lower()
            if key in existing_map:
                record = existing_map[key]
                needs_update = False
                if not record.is_active:
                    record.is_active = True
                    needs_update = True
                if record.regex_pattern != pattern:
                    record.regex_pattern = pattern
                    needs_update = True

                if needs_update:
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                new_record = AIArtistProjectModel(
                    project_name=artist,
                    regex_pattern=pattern,
                    suspected_engine="AI Virtual Project",
                    confidence=0.98,
                    is_active=True,
                )
                session.add(new_record)
                existing_map[key] = new_record
                stats["inserted"] += 1

        session.commit()

    return stats


if __name__ == "__main__":
    import sys

    target_file = sys.argv[1] if len(sys.argv) > 1 else "testcase4.xlsx"
    res = import_ai_artists_to_db(target_file, sheet_name="NS AI")
    print(f"Hoàn tất import nghệ sĩ AI: {res}")
