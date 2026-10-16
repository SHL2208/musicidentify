import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseStorageService:
    """Dịch vụ tải tệp tin đa phương tiện lên Supabase Object Storage."""

    def __init__(self):
        raw_url = os.getenv("SUPABASE_URL", "").strip()
        # Chuẩn hóa Base URL: loại bỏ /rest/v1/ nếu người dùng vô tình copy thừa
        self.supabase_url = raw_url.removesuffix("/rest/v1/").removesuffix("/rest/v1").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        self.client = None

        if self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client

                self.client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Khởi tạo kết nối Supabase Storage thành công.")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo Supabase Storage Client: {e}")

    def upload_file(self, local_path: Path | str, bucket: str = "spectrograms") -> str:
        """
        Tải file lên bucket Supabase Storage và trả về Public URL.
        Nếu tải lỗi hoặc chưa cấu hình, trả về đường dẫn file local an toàn.
        """
        path_obj = Path(local_path)
        if not path_obj.exists():
            return str(local_path)

        if not self.client:
            return str(local_path)

        file_name = path_obj.name
        content_type = "image/png" if file_name.endswith(".png") else "audio/wav"

        try:
            with open(path_obj, "rb") as f:
                file_bytes = f.read()

            # Upload vào bucket (upsert=True để tự động ghi đè nếu trùng tên)
            self.client.storage.from_(bucket).upload(
                path=file_name,
                file=file_bytes,
                file_options={"content-type": content_type, "upsert": "true"},
            )

            # Lấy URL công khai
            public_url = self.client.storage.from_(bucket).get_public_url(file_name)
            logger.info(
                f"Đã upload '{file_name}' lên Supabase Storage bucket '{bucket}': {public_url}"
            )
            return public_url
        except Exception as e:
            logger.warning(
                f"Lỗi khi upload lên Supabase Storage ({file_name}): {e}. Dùng file local."
            )
            return str(local_path)
