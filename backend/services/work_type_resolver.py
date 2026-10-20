"""Module định danh và giải mã phân loại tác phẩm (Work Type Resolver).

Phân biệt chuẩn xác giữa 3 nhóm tác phẩm:
- ORIGINAL: Tác phẩm chính chủ do ca sĩ/nghệ sĩ gốc phát hành (bao gồm cả Live Sessions chính chủ).
- COVER: Bản hát lại của ca sĩ khác, hát phòng trà, hoặc bản AI Voice Clone tái hiện bài hát người khác.
- REMIX: Bản phối lại điện tử (EDM, Vinahouse, Lo-Fi, Slowed, Drill, Speed-up) giữ nguyên vocal chính chủ.

Tuân thủ nghiêm ngặt chuẩn PEP 8 và Ruff.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.models.entities import WorkType

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

# 4. Danh sách các bài hát kinh điển / hit nổi tiếng thường xuyên được Cover
# Giúp nhận diện Cover ngay cả khi người tải không ghi chữ 'Cover' trong tiêu đề
_FAMOUS_COVER_CATALOG: dict[str, str] = {
    "ngày mai người ta lấy chồng": "Thành Đạt",
    "chuyện tình không dĩ vãng": "Trúc Phương",
    "tình nghĩa đôi ta chỉ thế thôi": "Tuấn Vũ, Sơn Tuyền",
    "nỗi đau từ một người đến sau": "Trịnh Đình Quang",
    "cỏ dại và hoa dành dành": "Phúc Lộc",
    "cơm người": "Dân gian",
    "tửu âm": "Xesi",
    "đêm trắng": "Khắc Việt",
    "em yêu anh như câu hò ví dặm": "Thu Hiền",
    "tiếng hát giữa rừng pác bó": "Thu Hiền",
    "thím hai lúa": "Thanh Ngân",
    "bông cỏ may": "Trúc Phương",
    "chiều cuối tuần": "Trúc Phương",
    "chút kỷ niệm buồn": "Thanh Sơn",
    "lòng vẫn đợi chờ": "Dân ca Quan họ",
    "ai yêu bác hồ chí minh": "Phong Nhã",
    "đội ca": "Phong Nhã",
    "dâu hào môn": "Bảo Vân",
    "mùa đông nhớ cố nhân": "Khánh Ly",
    "đảo không người": "Hải Lai A Mộc",
    "trong cuộc tình sầu": "Ngô Thụy Miên",
    "ta dâng đảng ngàn tiếng ca": "Trần Long Ẩn",
}

# 5. Các bài hát bản song ca / làm lại / remix đặc thù
_FAMOUS_REMIX_CATALOG: set[str] = {
    "cùng một người",
}

# 6. Các kênh / nghệ sĩ chuyên hát Cover nổi tiếng
_SPECIALIZED_COVER_CHANNELS = {
    "hương ly",
    "thủy tạ bolero",
    "joly thảo linh",
    "nam em",
    "beta music",
}


def remove_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối sánh mềm."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


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
) -> tuple[WorkType, str, bool, str | None, str | None]:
    """Giải mã và phân loại tác phẩm (Original vs Cover vs Remix).

    Kết hợp đa tầng:
    1. Ưu tiên nhãn Song Type nếu đã được truyền trong metadata.
    2. Quét từ khóa tiêu đề (Cover vs Remix vs Live).
    3. Tra cứu danh mục bài hát kinh điển / nghệ sĩ chuyên cover.
    4. Tham chiếu phán đoán của mô hình Gemini nếu có.
    """
    meta = metadata or {}
    t_clean = str(title or "").strip()
    a_clean = str(artist or "").strip()
    combined_lower = f"{t_clean} {a_clean}".lower()
    combined_unacc = remove_accents(combined_lower)
    title_unacc = remove_accents(t_clean)

    # 1. Kiểm tra nhãn trực tiếp từ Song Type nếu có
    direct_song_type = meta.get("song_type")
    if direct_song_type:
        mapped = map_song_type_to_work_type(str(direct_song_type))
        if mapped == "COVER":
            return (WorkType.COVER, "COVER (Bản hát lại)", True, None, t_clean)
        if mapped == "REMIX":
            return (WorkType.REMIX, "REMIX (Bản phối lại)", False, a_clean, t_clean)
        if mapped == "ORIGINAL":
            return (WorkType.ORIGINAL, "ORIGINAL (Tác phẩm chính chủ)", False, a_clean, t_clean)

    # 2. Quy tắc Từ khóa Bắt buộc: COVER
    if COVER_KEYWORD_PATTERN.search(combined_lower) or COVER_KEYWORD_PATTERN.search(combined_unacc):
        return (WorkType.COVER, "COVER (Bản hát lại)", True, None, t_clean)

    # 3. Quy tắc Từ khóa Bắt buộc: REMIX
    if REMIX_PATTERN.search(combined_lower) or REMIX_PATTERN.search(combined_unacc):
        return (WorkType.REMIX, "REMIX (Bản phối lại)", False, a_clean, t_clean)

    for r_title in _FAMOUS_REMIX_CATALOG:
        if remove_accents(r_title) in title_unacc:
            return (WorkType.REMIX, "REMIX (Bản phối lại)", False, a_clean, t_clean)

    # 4. Quy tắc Từ khóa: LIVE (Bản thu trực tiếp chính chủ)
    if LIVE_ORIGINAL_PATTERN.search(combined_lower) or LIVE_ORIGINAL_PATTERN.search(combined_unacc):
        return (WorkType.ORIGINAL, "ORIGINAL (Bản Live chính chủ)", False, a_clean, t_clean)

    # 5. Đối soát danh mục bài hát kinh điển thường xuyên bị Cover
    for song_name, orig_singer in _FAMOUS_COVER_CATALOG.items():
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
    for cov_channel in _SPECIALIZED_COVER_CHANNELS:
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
    """Tính toán ma trận nhầm lẫn 3x3 và các chỉ số hiệu năng chuẩn machine learning."""
    if labels is None:
        labels = ["ORIGINAL", "COVER", "REMIX"]

    all_pred_cols = list(labels)
    if "UNCERTAIN" not in all_pred_cols:
        all_pred_cols.append("UNCERTAIN")

    matrix: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in all_pred_cols} for gt in labels}

    total = 0
    correct = 0

    for yt, yp in zip(y_true, y_pred, strict=False):
        yt_norm = str(yt or "").upper().strip()
        yp_norm = str(yp or "").upper().strip()
        if yt_norm in matrix:
            if yp_norm not in matrix[yt_norm]:
                yp_norm = "UNCERTAIN"
            matrix[yt_norm][yp_norm] += 1
            total += 1
            if yt_norm == yp_norm:
                correct += 1

    accuracy = (correct / total) if total > 0 else 0.0

    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fn = sum(matrix.get(label, {}).get(p, 0) for p in all_pred_cols if p != label)
        fp = sum(matrix.get(other, {}).get(label, 0) for other in labels if other != label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "total": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    return {
        "matrix": matrix,
        "labels": labels,
        "total_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": per_class,
    }


def plot_work_type_confusion_matrix(
    metrics: dict[str, Any],
    output_path: str | Path | None = None,
    title: str = "Ma Trận Nhầm Lẫn Tác Phẩm (Original vs Cover vs Remix)",
) -> Path:
    """Vẽ ma trận nhầm lẫn đa lớp cho Work Type và lưu file ảnh PNG (300 DPI)."""
    if output_path is None:
        out_dir = ROOT_DIR / "data" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_file = out_dir / f"work_type_cm_{ts}.png"
    else:
        target_file = Path(output_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)

    labels = metrics.get("labels", ["ORIGINAL", "COVER", "REMIX"])
    matrix_dict = metrics.get("matrix", {})

    matrix_data = []
    for gt in labels:
        row = [matrix_dict.get(gt, {}).get(pred, 0) for pred in labels]
        matrix_data.append(row)
    matrix_arr = np.array(matrix_data)

    fig, (ax_cm, ax_stats) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6.8),
        dpi=300,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.patch.set_facecolor("#FAFAFC")

    # 1. Heatmap bên trái
    cmap = plt.cm.Purples
    im = ax_cm.imshow(matrix_arr, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Số lượng bài hát")

    tick_marks = np.arange(len(labels))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels([f"Đoán: {lbl}" for lbl in labels], fontsize=10, fontweight="bold")
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels([f"Thực tế: {lbl}" for lbl in labels], fontsize=10, fontweight="bold")
    ax_cm.set_xlabel("DỰ ĐOÁN CỦA HỆ THỐNG (PREDICTED)", fontsize=11, labelpad=10)
    ax_cm.set_ylabel("NHÃN THỰC TẾ (GROUND TRUTH)", fontsize=11, labelpad=10)
    ax_cm.set_title(title, fontsize=12, fontweight="bold", pad=12)

    row_sums = matrix_arr.sum(axis=1)
    max_val = matrix_arr.max() if matrix_arr.size > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix_arr[i, j]
            total_row = row_sums[i]
            pct = (val / total_row * 100) if total_row > 0 else 0.0
            color = "white" if val > (max_val / 2) else "black"
            ax_cm.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    # 2. Bảng thống kê bên phải
    ax_stats.axis("off")
    acc = metrics.get("accuracy", 0.0)
    tot = metrics.get("total_evaluated", 0)
    cor = metrics.get("correct", 0)

    header_text = (
        f"[THỐNG KÊ HIỆU NĂNG PHÂN BIỆT TÁC PHẨM]\n"
        f"• Tổng số bài đánh giá : {tot} bài\n"
        f"• Số bài đoán đúng    : {cor} bài\n"
        f"• Độ chính xác chung   : {acc * 100:.2f}%\n"
        f"────────────────────────────────────────────────"
    )
    ax_stats.text(
        0.05, 0.95, header_text, fontsize=11, family="sans-serif", va="top", fontweight="bold"
    )

    y_pos = 0.65
    per_class = metrics.get("per_class", {})
    for lbl in labels:
        c_met = per_class.get(lbl, {})
        c_tot = c_met.get("total", 0)
        c_prec = c_met.get("precision", 0.0) * 100
        c_rec = c_met.get("recall", 0.0) * 100
        c_f1 = c_met.get("f1_score", 0.0) * 100

        class_text = (
            f"[NHÃN]: {lbl} (Mẫu thực tế: {c_tot} bài)\n"
            f"   - Precision (Độ chuẩn xác) : {c_prec:.1f}%\n"
            f"   - Recall (Độ bao phủ)      : {c_rec:.1f}%\n"
            f"   - F1-Score (Cân bằng)      : {c_f1:.1f}%\n"
        )
        ax_stats.text(0.05, y_pos, class_text, fontsize=10, family="sans-serif", va="top")
        y_pos -= 0.18

    plt.tight_layout()
    fig.savefig(target_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target_file
