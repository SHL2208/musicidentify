from pathlib import Path
from typing import Optional
from ..core.config import TEMP_DIR, OUTPUT_DIR
from ..core.interfaces.repositories.storage_repository_interface import IStorageRepository

class TempStorageRepository(IStorageRepository):
    """Hiện thực quản lý đường dẫn và dọn dẹp các tệp tin tạm trong hệ thống."""

    def __init__(self, temp_dir: Path = TEMP_DIR, output_dir: Path = OUTPUT_DIR):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_temp_path(self, filename: str) -> Path:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        return self.temp_dir / filename

    def cleanup_file(self, path: Optional[Path]) -> bool:
        if path and isinstance(path, Path) and path.exists() and path.is_file():
            try:
                path.unlink(missing_ok=True)
                return True
            except Exception:
                return False
        return False

    def get_output_dir(self) -> Path:
        return self.output_dir

    def get_temp_dir(self) -> Path:
        return self.temp_dir
