"""Test suite for dynamic Song Type Catalog, Repository enhancements, and Work Type Resolver.

Ensures TDD compliance: Failing test first, then implementation, zero ruff issues.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.core.interfaces.repositories.catalog_repository_interface import ICatalogRepository
from backend.core.models.db_models import SongTypeCatalogModel
from backend.core.models.entities import WorkType
from backend.repositories.database_catalog_repository import DatabaseCatalogRepository
from backend.services.work_type_resolver import (
    resolve_work_type_from_metadata,
)


class TestDynamicSongTypeCatalog(unittest.TestCase):
    """Test suite verifying database catalog models, repository querying, and resolver integration."""

    def test_song_type_catalog_model_attributes(self) -> None:
        """Kiểm tra model SongTypeCatalogModel có đầy đủ các trường yêu cầu."""
        model = SongTypeCatalogModel(
            entry_type="COVER_SONG",
            name="Sóng Gió",
            original_artist="Jack - J97",
            notes="Nhạc trẻ",
            is_active=True,
        )
        self.assertEqual(model.entry_type, "COVER_SONG")
        self.assertEqual(model.name, "Sóng Gió")
        self.assertEqual(model.original_artist, "Jack - J97")
        self.assertTrue(model.is_active)

    def test_repository_provides_song_type_catalogs(self) -> None:
        """Kiểm tra repository nạp và cung cấp đúng danh mục bài hát cover/remix/channel."""
        repo = DatabaseCatalogRepository()
        cover_catalog = repo.get_famous_cover_catalog()
        remix_catalog = repo.get_famous_remix_catalog()
        cover_channels = repo.get_specialized_cover_channels()

        self.assertIsInstance(cover_catalog, dict)
        self.assertIsInstance(remix_catalog, set)
        self.assertIsInstance(cover_channels, set)

        # Kiểm tra một số bài tiêu biểu có trong snapshot / DB
        self.assertIn("ngày mai người ta lấy chồng", cover_catalog)
        self.assertIn("cùng một người", remix_catalog)
        self.assertIn("hương ly", cover_channels)

    def test_repository_offline_fallback_logs_warning(self) -> None:
        """Kiểm tra repository ghi log cảnh báo khi Database Supabase offline và nạp snapshot."""
        repo = DatabaseCatalogRepository()
        with (
            patch(
                "backend.repositories.database_catalog_repository.is_database_available",
                return_value=False,
            ),
            patch("backend.repositories.database_catalog_repository.logger.warning") as mock_warn,
        ):
            repo.reload_catalogs(force=True)
            mock_warn.assert_called()
            # Xác nhận có log cảnh báo offline snapshot
            warning_messages = [call.args[0] for call in mock_warn.call_args_list if call.args]
            self.assertTrue(
                any(
                    "[CATALOG ALERT]" in msg or "OFFLINE SNAPSHOT" in msg
                    for msg in warning_messages
                ),
                f"Expected warning log not found in: {warning_messages}",
            )

    def test_work_type_resolver_uses_dynamic_repository(self) -> None:
        """Kiểm tra work_type_resolver sử dụng ICatalogRepository thay vì hardcode tĩnh."""
        mock_repo = MagicMock(spec=ICatalogRepository)
        mock_repo.get_famous_cover_catalog.return_value = {
            "bài ca hy vọng": "Nghệ sĩ A",
        }
        mock_repo.get_famous_remix_catalog.return_value = {
            "vũ điệu hoang dã",
        }
        mock_repo.get_specialized_cover_channels.return_value = {
            "cover channel x",
        }

        # 1. Test Cover resolution từ repo mock
        w_type, label, is_cover, orig_singer, orig_title = resolve_work_type_from_metadata(
            title="Bài Ca Hy Vọng - Ca sĩ B",
            artist="Ca sĩ B",
            catalog_repo=mock_repo,
        )
        self.assertEqual(w_type, WorkType.COVER)
        self.assertTrue(is_cover)
        self.assertEqual(orig_singer, "Nghệ sĩ A")

        # 2. Test Remix resolution từ repo mock
        w_type, label, is_cover, orig_singer, orig_title = resolve_work_type_from_metadata(
            title="Vũ Điệu Hoang Dã",
            artist="Ca sĩ C",
            catalog_repo=mock_repo,
        )
        self.assertEqual(w_type, WorkType.REMIX)

        # 3. Test Channel cover resolution từ repo mock
        w_type, label, is_cover, orig_singer, orig_title = resolve_work_type_from_metadata(
            title="Một bài hát bất kỳ",
            artist="Cover Channel X",
            catalog_repo=mock_repo,
        )
        self.assertEqual(w_type, WorkType.COVER)

    def test_add_artist_and_refresh_cache(self) -> None:
        """Kiểm tra API thêm ca sĩ vào repository và làm mới cache thành công."""
        repo = DatabaseCatalogRepository()
        self.assertTrue(hasattr(repo, "add_verified_human_artist"))
        self.assertTrue(hasattr(repo, "add_ai_artist_project"))
        self.assertTrue(hasattr(repo, "refresh_cache"))

        # Kiểm tra xử lý an toàn khi offline (trả về False, không văng ngoại lệ)
        with patch(
            "backend.repositories.database_catalog_repository.is_database_available",
            return_value=False,
        ):
            res_human = repo.add_verified_human_artist("Ca Sĩ Thử Nghiệm")
            res_ai = repo.add_ai_artist_project("Dự Án AI Thử Nghiệm")
            self.assertFalse(res_human)
            self.assertFalse(res_ai)

    def test_backend_client_catalog_integration(self) -> None:
        """Kiểm tra BackendClient kết nối thành công với catalog repository."""
        from frontend.services.backend_client import BackendClient

        client = BackendClient()
        self.assertTrue(hasattr(client, "refresh_catalog_cache"))
        self.assertTrue(hasattr(client, "add_verified_human_artist"))
        self.assertTrue(hasattr(client, "add_ai_artist_project"))

        # Gọi làm mới cache qua client
        client.refresh_catalog_cache()


if __name__ == "__main__":
    unittest.main()
