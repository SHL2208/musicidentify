"""Hiện thực DatabaseCatalogRepository quản lý danh mục nghệ sĩ và nhà phân phối từ Database Supabase.

Tích hợp In-Memory Caching (TTL) và Cơ chế Hybrid Snapshot dự phòng ngoại tuyến.
Tuân thủ nghiêm ngặt Clean Architecture, PEP 8 và Ruff.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

from sqlalchemy import select

from ..core.config import (
    CATALOG_CACHE_TTL_SECONDS,
    CATALOG_SNAPSHOT_PATH,
    COLLABORATOR_SPLIT_PATTERN,
    NATIONAL_ARTIST_HONORIFICS_PATTERN,
    PRODUCER_PREFIX_PATTERN,
)
from ..core.database import get_db_session, is_database_available
from ..core.interfaces.repositories.catalog_repository_interface import ICatalogRepository
from ..core.models.db_models import (
    AIArtistProjectModel,
    DistributorCatalogModel,
    SongTypeCatalogModel,
    VerifiedHumanArtistModel,
)
from ..core.utils import normalize_vietnamese_text

logger = logging.getLogger(__name__)


def remove_vietnamese_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối sánh mềm tên nghệ sĩ."""
    return normalize_vietnamese_text(text).lower()


class DatabaseCatalogRepository(ICatalogRepository):
    """
    Repository truy xuất danh mục nghệ sĩ và nhà phân phối từ Database.
    Sở hữu cơ chế In-Memory Cache và Hybrid Snapshot tự phục hồi an toàn (Crash-proof).
    """

    def __init__(self, ttl_seconds: int = CATALOG_CACHE_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, Any] = {
            "timestamp": 0.0,
            "whitelist": [],
            "ai_distributors": [],
            "ai_artists": [],
            "human_artists": set(),
            "human_artists_unaccented": set(),
            "famous_cover_songs": {},
            "famous_remix_songs": set(),
            "specialized_cover_channels": set(),
        }
        self._lock = threading.Lock()
        # Nạp dữ liệu ban đầu
        self._ensure_cache()

    def _load_from_snapshot(self) -> dict[str, Any]:
        """Tải dữ liệu danh mục từ file snapshot cục bộ khi DB offline."""
        if not CATALOG_SNAPSHOT_PATH.exists():
            logger.warning("Không tìm thấy file snapshot tại %s", CATALOG_SNAPSHOT_PATH)
            return {
                "whitelist": [],
                "ai_distributors": [],
                "ai_artists": [],
                "human_artists": set(),
                "human_artists_unaccented": set(),
                "famous_cover_songs": {},
                "famous_remix_songs": set(),
                "specialized_cover_channels": set(),
            }

        try:
            with open(CATALOG_SNAPSHOT_PATH, encoding="utf-8") as f:
                data = json.load(f)
            human_set = set(data.get("human_artists", []))
            unacc_set = set(data.get("human_artists_unaccented", []))
            ai_list = [(p, n) for p, n in data.get("ai_artists", [])]
            wl = list(data.get("whitelist", []))
            ai_dist = list(data.get("ai_distributors", []))
            famous_covers = dict(data.get("famous_cover_songs", {}))
            famous_remixes = set(data.get("famous_remix_songs", []))
            cover_chans = set(data.get("specialized_cover_channels", []))
            logger.info("Đã tải thành công %d nghệ sĩ từ file snapshot cục bộ.", len(human_set))
            return {
                "whitelist": wl,
                "ai_distributors": ai_dist,
                "ai_artists": ai_list,
                "human_artists": human_set,
                "human_artists_unaccented": unacc_set,
                "famous_cover_songs": famous_covers,
                "famous_remix_songs": famous_remixes,
                "specialized_cover_channels": cover_chans,
            }
        except Exception as e:
            logger.error("Lỗi khi đọc file snapshot cục bộ: %s", e)
            return {
                "whitelist": [],
                "ai_distributors": [],
                "ai_artists": [],
                "human_artists": set(),
                "human_artists_unaccented": set(),
                "famous_cover_songs": {},
                "famous_remix_songs": set(),
                "specialized_cover_channels": set(),
            }

    def _save_snapshot(
        self,
        human_names: set[str],
        human_unacc: set[str],
        ai_artists: list[tuple[str, str]],
        whitelist: list[str],
        ai_dist: list[str],
        famous_covers: dict[str, str] | None = None,
        famous_remixes: set[str] | None = None,
        cover_chans: set[str] | None = None,
    ) -> None:
        """Ghi lưu trữ dữ liệu vào file snapshot an toàn dưới nền."""
        try:
            CATALOG_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": time.time(),
                "human_artists": sorted(list(human_names)),
                "human_artists_unaccented": sorted(list(human_unacc)),
                "ai_artists": ai_artists,
                "whitelist": sorted(whitelist),
                "ai_distributors": sorted(ai_dist),
                "famous_cover_songs": famous_covers or {},
                "famous_remix_songs": sorted(list(famous_remixes or set())),
                "specialized_cover_channels": sorted(list(cover_chans or set())),
            }
            with open(CATALOG_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Không thể tự động ghi snapshot: %s", e)

    def _ensure_cache(self, force: bool = False) -> None:
        """Kiểm tra và làm mới bộ nhớ đệm nếu đã hết thời gian hiệu lực (TTL)."""
        now = time.time()
        if not force and (now - self._cache["timestamp"] < self.ttl_seconds):
            return

        with self._lock:
            # Kiểm tra lại sau khi nhận lock
            if not force and (now - self._cache["timestamp"] < self.ttl_seconds):
                return

            if is_database_available():
                try:
                    with get_db_session() as session:
                        # 1. Truy vấn nhà phân phối
                        dist_records = session.scalars(
                            select(DistributorCatalogModel).where(
                                DistributorCatalogModel.is_active.is_(True)
                            )
                        ).all()
                        whitelist = [
                            d.name.lower()
                            for d in dist_records
                            if d.catalog_type == "HUMAN_WHITELIST"
                        ]
                        ai_dist = [
                            d.name.lower()
                            for d in dist_records
                            if d.catalog_type == "AI_DISTRIBUTOR"
                        ]

                        # 2. Truy vấn dự án nghệ sĩ ảo AI
                        artist_records = session.scalars(
                            select(AIArtistProjectModel).where(
                                AIArtistProjectModel.is_active.is_(True)
                            )
                        ).all()
                        ai_artists = [(a.regex_pattern, a.project_name) for a in artist_records]

                        # 3. Truy vấn ca sĩ người thật
                        human_artist_records = session.scalars(
                            select(VerifiedHumanArtistModel).where(
                                VerifiedHumanArtistModel.is_active.is_(True)
                            )
                        ).all()

                        db_human_names = set()
                        db_human_unacc = set()
                        for har in human_artist_records:
                            n_low = har.artist_name.lower().strip()
                            db_human_names.add(n_low)
                            db_human_unacc.add(remove_vietnamese_accents(n_low))
                            if har.aliases:
                                for alias in har.aliases.split(","):
                                    a_clean = alias.strip().lower()
                                    if a_clean:
                                        db_human_names.add(a_clean)
                                        db_human_unacc.add(remove_vietnamese_accents(a_clean))

                        # 4. Truy vấn danh mục tác phẩm (Cover/Remix/Channel)
                        song_type_records = session.scalars(
                            select(SongTypeCatalogModel).where(
                                SongTypeCatalogModel.is_active.is_(True)
                            )
                        ).all()
                        famous_covers: dict[str, str] = {}
                        famous_remixes: set[str] = set()
                        cover_chans: set[str] = set()

                        for st in song_type_records:
                            st_name_low = st.name.strip().lower()
                            if st.entry_type == "COVER_SONG":
                                famous_covers[st_name_low] = (st.original_artist or "").strip()
                            elif st.entry_type == "REMIX_SONG":
                                famous_remixes.add(st_name_low)
                            elif st.entry_type == "COVER_CHANNEL":
                                cover_chans.add(st_name_low)

                        self._cache["whitelist"] = whitelist
                        self._cache["ai_distributors"] = ai_dist
                        self._cache["ai_artists"] = ai_artists
                        self._cache["human_artists"] = db_human_names
                        self._cache["human_artists_unaccented"] = db_human_unacc
                        self._cache["famous_cover_songs"] = famous_covers
                        self._cache["famous_remix_songs"] = famous_remixes
                        self._cache["specialized_cover_channels"] = cover_chans
                        self._cache["timestamp"] = now

                        # Tự động cập nhật snapshot cục bộ khi query DB thành công
                        self._save_snapshot(
                            db_human_names,
                            db_human_unacc,
                            ai_artists,
                            whitelist,
                            ai_dist,
                            famous_covers,
                            famous_remixes,
                            cover_chans,
                        )
                        return
                except Exception as e:
                    logger.warning("Lỗi truy vấn Database khi làm mới cache: %s", e)

            # Fallback sang snapshot khi DB offline hoặc query lỗi
            logger.warning(
                "[CATALOG ALERT] Database Supabase hiện không khả dụng. "
                "Hệ thống đang chạy ở chế độ OFFLINE SNAPSHOT từ '%s'.",
                CATALOG_SNAPSHOT_PATH,
            )
            snapshot_data = self._load_from_snapshot()
            self._cache["whitelist"] = snapshot_data.get("whitelist", [])
            self._cache["ai_distributors"] = snapshot_data.get("ai_distributors", [])
            self._cache["ai_artists"] = snapshot_data.get("ai_artists", [])
            self._cache["human_artists"] = snapshot_data.get("human_artists", set())
            self._cache["human_artists_unaccented"] = snapshot_data.get(
                "human_artists_unaccented", set()
            )
            self._cache["famous_cover_songs"] = snapshot_data.get("famous_cover_songs", {})
            self._cache["famous_remix_songs"] = snapshot_data.get("famous_remix_songs", set())
            self._cache["specialized_cover_channels"] = snapshot_data.get(
                "specialized_cover_channels", set()
            )
            self._cache["timestamp"] = now

    def reload_catalogs(self, force: bool = True) -> None:
        """Nạp lại danh mục từ DB hoặc Snapshot."""
        self._ensure_cache(force=force)

    def refresh_cache(self) -> None:
        """Làm mới bộ nhớ đệm ngay lập tức."""
        self.reload_catalogs(force=True)

    def get_famous_cover_catalog(self) -> dict[str, str]:
        """Lấy danh mục bài hát cover nổi tiếng (tên bài -> nghệ sĩ gốc)."""
        self._ensure_cache()
        return self._cache.get("famous_cover_songs") or {}

    def get_famous_remix_catalog(self) -> set[str]:
        """Lấy danh mục bài hát remix đặc thù."""
        self._ensure_cache()
        return self._cache.get("famous_remix_songs") or set()

    def get_specialized_cover_channels(self) -> set[str]:
        """Lấy danh mục các kênh YouTube/nghệ sĩ chuyên hát cover."""
        self._ensure_cache()
        return self._cache.get("specialized_cover_channels") or set()

    def add_verified_human_artist(
        self, name: str, aliases: str | None = None, notes: str | None = None
    ) -> bool:
        """Thêm mới ca sĩ người thật vào Database và làm mới cache."""
        clean_name = (name or "").strip()
        if not clean_name:
            return False

        if not is_database_available():
            logger.warning("Không thể thêm nghệ sĩ '%s': Database offline.", clean_name)
            return False

        try:
            with get_db_session() as session:
                existing = session.scalar(
                    select(VerifiedHumanArtistModel).where(
                        VerifiedHumanArtistModel.artist_name.ilike(clean_name)
                    )
                )
                if existing:
                    if not existing.is_active:
                        existing.is_active = True
                    if aliases and not existing.aliases:
                        existing.aliases = aliases
                else:
                    new_artist = VerifiedHumanArtistModel(
                        artist_name=clean_name,
                        aliases=aliases,
                        notes=notes or "Added from UI/API",
                        is_active=True,
                    )
                    session.add(new_artist)
                session.commit()
            self.refresh_cache()
            return True
        except Exception as e:
            logger.error("Lỗi khi thêm ca sĩ '%s' vào Database: %s", clean_name, e)
            return False

    def add_ai_artist_project(
        self, name: str, regex_pattern: str | None = None, notes: str | None = None
    ) -> bool:
        """Thêm mới dự án nghệ sĩ ảo AI vào Database và làm mới cache."""
        clean_name = (name or "").strip()
        if not clean_name:
            return False

        pat = regex_pattern or re.escape(clean_name)
        if not is_database_available():
            logger.warning("Không thể thêm dự án AI '%s': Database offline.", clean_name)
            return False

        try:
            with get_db_session() as session:
                existing = session.scalar(
                    select(AIArtistProjectModel).where(
                        AIArtistProjectModel.project_name.ilike(clean_name)
                    )
                )
                if existing:
                    if not existing.is_active:
                        existing.is_active = True
                    existing.regex_pattern = pat
                else:
                    new_proj = AIArtistProjectModel(
                        project_name=clean_name,
                        regex_pattern=pat,
                        suspected_engine="AI Virtual Project",
                        confidence=0.98,
                        notes=notes or "Added from UI/API",
                        is_active=True,
                    )
                    session.add(new_proj)
                session.commit()
            self.refresh_cache()
            return True
        except Exception as e:
            logger.error("Lỗi khi thêm dự án AI '%s' vào Database: %s", clean_name, e)
            return False

    def get_catalog_data(self) -> dict[str, Any]:
        """Lấy toàn bộ dữ liệu danh mục nghệ sĩ và distributor."""
        self._ensure_cache()
        return self._cache

    def is_verified_human_artist(self, artist_str: str, title_str: str = "") -> bool:
        """Xác thực xem nghệ sĩ hoặc tiêu đề có thuộc danh sách nghệ sĩ người thật không."""
        a_low = (artist_str or "").lower().strip()
        t_low = (title_str or "").lower().strip()
        a_unacc = remove_vietnamese_accents(a_low)
        t_unacc = remove_vietnamese_accents(t_low)

        self._ensure_cache()
        ai_artists = self._cache.get("ai_artists") or []

        # 1. Tuyệt đối không xác thực nếu thuộc danh sách các dự án nghệ sĩ ảo AI đã biết
        for pat, _ in ai_artists:
            try:
                if re.search(pat, a_low) or re.search(pat, t_low):
                    return False
            except Exception:
                continue

        # 1b. Danh hiệu nghệ sĩ Quốc gia (NSND, NSƯT / NSUT) -> Nghệ sĩ người thật
        if (
            re.search(NATIONAL_ARTIST_HONORIFICS_PATTERN, a_low)
            or re.search(NATIONAL_ARTIST_HONORIFICS_PATTERN, a_unacc)
            or re.search(NATIONAL_ARTIST_HONORIFICS_PATTERN, t_low)
            or re.search(NATIONAL_ARTIST_HONORIFICS_PATTERN, t_unacc)
        ):
            return True

        verified_artists = self._cache.get("human_artists") or set()
        verified_artists_unacc = self._cache.get("human_artists_unaccented") or set()

        # 2. Tách các nghệ sĩ theo dấu phân cách hợp tác (tránh nhầm lẫn 'Anh Vũ' với 'Vũ')
        tokens = [
            tok.strip().rstrip(".")
            for tok in re.split(COLLABORATOR_SPLIT_PATTERN, a_low)
            if tok.strip()
        ]

        for tok in tokens:
            clean_tok = re.sub(PRODUCER_PREFIX_PATTERN, "", tok).strip()
            clean_tok_unacc = remove_vietnamese_accents(clean_tok)
            if (
                clean_tok in verified_artists
                or clean_tok_unacc in verified_artists_unacc
                or tok in verified_artists
                or remove_vietnamese_accents(tok) in verified_artists_unacc
            ):
                return True

        # 3. Khớp các cụm từ nghệ sĩ / chương trình dài (VD: 'anh trai say hi', 'vũ cát tường')
        for kn in verified_artists:
            kn_unacc = remove_vietnamese_accents(kn)
            if len(kn.split()) > 1:
                if kn in a_low or kn in t_low or kn_unacc in a_unacc or kn_unacc in t_unacc:
                    return True
            else:
                # Nghệ sĩ đơn từ (VD: 'toharmony', 'arson')
                if kn == a_low or kn_unacc == a_unacc:
                    return True

        return False
