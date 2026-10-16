import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import OUTPUT_DIR
from ..core.exceptions import RepositoryException
from ..core.interfaces.repositories.report_repository_interface import IReportRepository

ILLEGAL_CHARACTERS_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def clean_excel_text(val: Any) -> Any:
    """Lọc bỏ ANSI escape codes, các ký tự điều khiển không hợp lệ và phòng chống CSV/Excel Formula Injection."""
    if isinstance(val, str):
        val = ANSI_ESCAPE_RE.sub("", val)
        val = ILLEGAL_CHARACTERS_RE.sub("", val)
        if val and val[0] in ("=", "+", "-", "@"):
            val = f"'{val}"
    return val


class FileReportRepository(IReportRepository):
    """Hiện thực lưu trữ và xuất tệp báo cáo ra đĩa cứng (Excel hoặc CSV)."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_report_file(self, records: list[dict[str, Any]], format: str = "xlsx") -> Path:
        if not records:
            raise RepositoryException("Không có dữ liệu bản ghi để xuất báo cáo.")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame(records)

            if hasattr(df, "map"):
                df = df.map(clean_excel_text)
            else:
                legacy_applymap = getattr(df, "applymap", None)
                if callable(legacy_applymap):
                    df = legacy_applymap(clean_excel_text)

            if format.lower() == "csv":
                out_path = self.output_dir / f"ai_music_report_{timestamp}.csv"
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
            else:
                out_path = self.output_dir / f"ai_music_report_{timestamp}.xlsx"
                df.to_excel(out_path, index=False, engine="openpyxl")

            return out_path
        except Exception as e:
            raise RepositoryException(f"Lỗi khi ghi tệp báo cáo: {e}") from e

    def get_report_file(self, filename: str) -> Path | None:
        target = self.output_dir / filename
        return target if target.exists() and target.is_file() else None
