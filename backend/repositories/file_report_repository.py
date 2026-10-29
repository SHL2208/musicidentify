import io
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import OUTPUT_DIR
from ..core.exceptions import RepositoryException
from ..core.interfaces.repositories.report_repository_interface import IReportRepository
from ..core.utils import clean_excel_text


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

            self.prune_old_reports(max_keep=10)
            return out_path
        except Exception as e:
            raise RepositoryException(f"Lỗi khi ghi tệp báo cáo: {e}") from e

    def export_report_bytes(self, records: list[dict[str, Any]], format: str = "xlsx") -> bytes:
        if not records:
            raise RepositoryException("Không có dữ liệu bản ghi để xuất báo cáo.")

        try:
            df = pd.DataFrame(records)

            if hasattr(df, "map"):
                df = df.map(clean_excel_text)
            else:
                legacy_applymap = getattr(df, "applymap", None)
                if callable(legacy_applymap):
                    df = legacy_applymap(clean_excel_text)

            if format.lower() == "csv":
                csv_str = df.to_csv(index=False, encoding="utf-8-sig")
                return csv_str.encode("utf-8-sig") if isinstance(csv_str, str) else csv_str
            else:
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)
                return buffer.getvalue()
        except Exception as e:
            raise RepositoryException(f"Lỗi khi xuất dữ liệu báo cáo ra bộ nhớ: {e}") from e

    def prune_old_reports(self, max_keep: int = 10) -> int:
        """
        Dọn dẹp các tệp báo cáo cũ trong self.output_dir.
        Chỉ giữ lại tối đa max_keep tệp báo cáo mới nhất (ai_music_report_*).
        Bảo toàn các tệp khác (như spectrogram .png).
        Trả về số lượng tệp đã xóa.
        """
        if not self.output_dir.exists():
            return 0

        report_files = [
            f
            for f in self.output_dir.iterdir()
            if f.is_file()
            and f.name.startswith("ai_music_report_")
            and f.suffix.lower() in [".xlsx", ".csv"]
        ]

        if len(report_files) <= max_keep:
            return 0

        # Sắp xếp theo st_mtime giảm dần (mới nhất lên đầu)
        report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        files_to_delete = report_files[max_keep:]
        deleted_count = 0
        for f in files_to_delete:
            try:
                f.unlink()
                deleted_count += 1
            except Exception:
                pass

        return deleted_count

    def get_report_file(self, filename: str) -> Path | None:
        target = self.output_dir / filename
        return target if target.exists() and target.is_file() else None
