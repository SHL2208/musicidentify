"""Module định danh và giải mã phân loại tác phẩm (Work Type Resolver).

Phân biệt chuẩn xác giữa 3 nhóm tác phẩm:
- ORIGINAL: Tác phẩm chính chủ do ca sĩ/nghệ sĩ gốc phát hành (bao gồm cả Live Sessions chính chủ).
- COVER: Bản hát lại của ca sĩ khác, hát phòng trà, hoặc bản AI Voice Clone tái hiện bài hát người khác.
- REMIX: Bản phối lại điện tử (EDM, Vinahouse, Lo-Fi, Slowed, Drill, Speed-up) giữ nguyên vocal chính chủ.

Tuân thủ nghiêm ngặt chuẩn PEP 8 và Ruff.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.interfaces.repositories.catalog_repository_interface import ICatalogRepository

from ..core.models.entities import WorkType
from ..core.utils import normalize_vietnamese_text

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 1. Các từ khóa đặc trưng nhận diện Remix / Remake / Lofi
REMIX_PATTERN = re.compile(
    r"\b("
    r"remix|remake|lofi|lo-fi|slowed|speed\s*up|sped\s*up|vinahouse|"
    r"club\s*mix|drill|nightcore|bass\s*boosted|dance\s*ver|house\s*ver|"
    r"ver\s*\d+|version|rap\s*ver(?:sion)?|dj\s*remix|prod\s*remix|"
    r"wdm|soaking\s*piano|piano\s*ver(?:sion)?|acoustic\s*ver(?:sion)?|"
    r"full\s*ver(?:sion)?|extended(?:\s*ver)?|instrumental|beat(?:\s*(?:nam|nữ))?|"
    r"ver\s+[a-zà-ỹ0-9]+"
    r")\b|_full\b",
    re.IGNORECASE,
)

# 2. Các từ khóa đặc trưng nhận diện Cover
COVER_KEYWORD_PATTERN = re.compile(
    r"\b("
    r"cover|hát\s*lại|cover\s*by|ft\.?\s*cover|hát\s*karaoke|acoustic\s*cover|"
    r"tập\s*hát|hát\s*nhép|ai\s*cover|cover\s*version|người\s*hát\s*sau"
    r")\b",
    re.IGNORECASE,
)

# 3. Các từ khóa đặc trưng nhận diện Bản gốc / Live chính chủ
LIVE_ORIGINAL_PATTERN = re.compile(
    r"\b(live\s*session|live\s*at|live\s*concert|trực\s*tiếp|\blive\b)\b",
    re.IGNORECASE,
)

# 4. Ghi chú: Danh mục bài hát cover kinh điển, remix và kênh cover
# đã được chuyển đổi hoàn toàn thành dữ liệu DB (bảng song_type_catalog)
# và cơ chế Offline Snapshot dự phòng, truy xuất thông qua ICatalogRepository.


def remove_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối sánh mềm."""
    return normalize_vietnamese_text(text).lower()


def map_song_type_to_work_type(song_type_val: str | None) -> str:
    """Ánh xạ nhãn 'Song Type' từ file testcase sang chuẩn WorkType enum value."""
    st_clean = (str(song_type_val) if song_type_val is not None else "").strip().lower()
    if not st_clean or st_clean in ("none", "nan", "null"):
        return "UNCERTAIN"
    if "cover" in st_clean:
        return "COVER"
    if any(k in st_clean for k in ["remix", "remake"]):
        return "REMIX"
    if any(k in st_clean for k in ["original", "live"]):
        return "ORIGINAL"
    return "UNCERTAIN"


def resolve_work_type_from_metadata(
    title: str,
    artist: str = "",
    metadata: dict[str, Any] | None = None,
    raw_gemini_work_type: str | None = None,
    catalog_repo: ICatalogRepository | None = None,
) -> tuple[WorkType, str, bool, str | None, str | None]:
    """Giải mã và phân loại tác phẩm (Original vs Cover vs Remix).

    Kết hợp đa tầng với quyền ưu tiên tuyệt đối cho Metadata và Từ khóa tiêu đề:
    1. Quét từ khóa tiêu đề (Remix vs Cover vs Live).
    2. Kiểm tra nhãn trực tiếp từ Song Type trong metadata.
    3. Tra cứu danh mục bài hát kinh điển / nghệ sĩ chuyên cover từ Catalog Repository (DB/Snapshot).
    4. Tham chiếu phán đoán của mô hình Gemini nếu không có dấu hiệu trên.
    """
    repo = catalog_repo
    if repo is None:
        try:
            from ..core.container import get_container

            repo = get_container().catalog_repository
        except Exception:
            from ..repositories.database_catalog_repository import DatabaseCatalogRepository

            repo = DatabaseCatalogRepository()

    famous_cover_catalog = repo.get_famous_cover_catalog() if repo else {}
    famous_remix_catalog = repo.get_famous_remix_catalog() if repo else set()
    specialized_cover_channels = repo.get_specialized_cover_channels() if repo else set()

    meta = metadata or {}
    t_clean = str(title or "").strip()
    a_clean = str(artist or "").strip()
    combined_lower = f"{t_clean} {a_clean}".lower()
    combined_unacc = remove_accents(combined_lower)
    title_unacc = remove_accents(t_clean)

    # 1. Quy tắc Từ khóa Bắt buộc: REMIX (Ưu tiên cao nhất trên tiêu đề)
    if REMIX_PATTERN.search(combined_lower) or REMIX_PATTERN.search(combined_unacc):
        return (WorkType.REMIX, "REMIX (Bản phối lại)", False, a_clean, t_clean)

    for r_title in famous_remix_catalog:
        if remove_accents(r_title) in title_unacc:
            return (WorkType.REMIX, "REMIX (Bản phối lại)", False, a_clean, t_clean)

    # 2. Quy tắc Từ khóa Bắt buộc: COVER (Ưu tiên cao trên tiêu đề)
    if COVER_KEYWORD_PATTERN.search(combined_lower) or COVER_KEYWORD_PATTERN.search(combined_unacc):
        return (WorkType.COVER, "COVER (Bản hát lại)", True, None, t_clean)

    # 3. Kiểm tra nhãn trực tiếp từ Song Type trong metadata nếu có
    direct_song_type = (
        meta.get("song_type")
        or meta.get("Song Type")
        or meta.get("work_type")
        or meta.get("Work Type")
    )
    if direct_song_type:
        mapped = map_song_type_to_work_type(str(direct_song_type))
        if mapped == "COVER":
            return (WorkType.COVER, "COVER (Bản hát lại)", True, None, t_clean)
        if mapped == "REMIX":
            return (WorkType.REMIX, "REMIX (Bản phối lại)", False, a_clean, t_clean)
        if mapped == "ORIGINAL":
            return (WorkType.ORIGINAL, "ORIGINAL (Tác phẩm chính chủ)", False, a_clean, t_clean)

    # 4. Quy tắc Từ khóa: LIVE (Bản thu trực tiếp chính chủ)
    if LIVE_ORIGINAL_PATTERN.search(combined_lower) or LIVE_ORIGINAL_PATTERN.search(combined_unacc):
        return (WorkType.ORIGINAL, "ORIGINAL (Bản Live chính chủ)", False, a_clean, t_clean)

    # 5. Đối soát danh mục bài hát kinh điển thường xuyên bị Cover
    for song_name, orig_singer in famous_cover_catalog.items():
        song_unacc = remove_accents(song_name)
        orig_unacc = remove_accents(orig_singer)
        if song_unacc in title_unacc:
            # Nếu ca sĩ hiện tại không phải là ca sĩ gốc -> Chắc chắn là COVER!
            if orig_unacc not in combined_unacc:
                return (
                    WorkType.COVER,
                    f"COVER (Bản hát lại - Gốc: {orig_singer})",
                    True,
                    orig_singer,
                    song_name,
                )

    # 6. Đối soát kênh / nghệ sĩ chuyên cover
    for cov_channel in specialized_cover_channels:
        cov_unacc = remove_accents(cov_channel)
        if cov_unacc in combined_unacc:
            return (
                WorkType.COVER,
                f"COVER (Kênh/Nghệ sĩ cover: {cov_channel.title()})",
                True,
                None,
                t_clean,
            )

    # 7. Tham chiếu phán đoán của Gemini nếu Gemini đưa ra kết quả cụ thể khác UNCERTAIN
    if raw_gemini_work_type:
        gemini_str = str(raw_gemini_work_type).strip().upper()
        if gemini_str in ("ORIGINAL", "COVER", "REMIX"):
            w_enum = WorkType(gemini_str)
            w_lbl = (
                "ORIGINAL (Tác phẩm chính chủ)"
                if w_enum == WorkType.ORIGINAL
                else ("COVER (Bản hát lại)" if w_enum == WorkType.COVER else "REMIX (Bản phối lại)")
            )
            return (w_enum, w_lbl, w_enum == WorkType.COVER, None, t_clean)

    # 8. Mặc định: Nếu là bài hát thông thường không có dấu hiệu Cover/Remix -> Bản phát hành chính chủ (ORIGINAL)
    return (WorkType.ORIGINAL, "ORIGINAL (Tác phẩm chính chủ)", False, a_clean, t_clean)


def calculate_work_type_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Tính toán ma trận nhầm lẫn 3x3 (ủy quyền cho benchmark_evaluator để tránh trùng lặp mã)."""
    from benchmark_evaluator import (
        calculate_work_type_confusion_matrix as _calc,
    )

    return _calc(y_true, y_pred, labels=labels)


def plot_work_type_confusion_matrix(
    metrics: dict[str, Any],
    output_path: str | Path | None = None,
    title: str = "Ma Trận Nhầm Lẫn Tác Phẩm (Original vs Cover vs Remix)",
) -> Path:
    """Vẽ ma trận nhầm lẫn đa lớp (ủy quyền cho benchmark_evaluator để tránh trùng lặp mã)."""
    from benchmark_evaluator import (
        plot_work_type_confusion_matrix as _plot,
    )

    return _plot(metrics, output_path=output_path, title=title)
