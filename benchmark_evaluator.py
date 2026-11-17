"""
Công cụ Đánh Giá & Chấm Điểm Độ Chính Xác (Benchmark & Accuracy Evaluator)
Sử dụng dữ liệu Ground Truth từ file testcase.xlsx (cột 'AI Yes').

Các chỉ số đo lường:
- Confusion Matrix: True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
- Accuracy (Độ chính xác tổng thể)
- Precision (Độ tin cậy khi gắn nhãn AI - tránh bắt oan)
- Recall / Sensitivity (Độ nhạy bắt nhạc AI - tránh lọt lưới)
- Specificity (Độ đặc hiệu - khả năng nhận diện đúng nhạc con người)
- F1-Score (Điểm điều hòa cân bằng giữa Precision và Recall)

Cách dùng:
  python benchmark_evaluator.py --sample 20 --balanced
  python benchmark_evaluator.py --sample 50
  python benchmark_evaluator.py --all
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Thêm root vào sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001, S110
        pass

from backend.core.config import BENCHMARK_WORKERS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

from backend.core.container import get_container
from backend.core.models.entities import (
    DetectionResult,
    TrackItem,
    VerdictType,
)
from backend.services.work_type_resolver import (
    map_song_type_to_work_type,
    resolve_work_type_from_metadata,
)


@dataclass
class ConfusionMetrics:
    """Chứa toàn bộ các chỉ số thống kê của ma trận nhầm lẫn."""

    total_samples: int
    evaluated_samples: int
    errors: int
    tp: int  # True Positive (Thực tế AI, Đoán AI)
    tn: int  # True Negative (Thực tế Human, Đoán Human)
    fp: int  # False Positive (Thực tế Human, Đoán AI - Báo động giả)
    fn: int  # False Negative (Thực tế AI, Đoán Human - Lọt lưới)
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    balanced_accuracy: float
    error_rate: float
    error_records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi dữ liệu thống kê thành dictionary."""
        return asdict(self)


def calculate_confusion_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    error_count: int = 0,
    error_records: list[dict[str, Any]] | None = None,
) -> ConfusionMetrics:
    """Tính toán ma trận nhầm lẫn và các chỉ số đo lường hiệu năng.

    Parameters:
        y_true: Danh sách nhãn thực tế (True = AI, False = Human).
        y_pred: Danh sách nhãn dự đoán (True = AI, False = Human).
        error_count: Số lượng mẫu gặp sự cố kết nối/xử lý lỗi.
        error_records: Danh sách chi tiết các bài hát gặp lỗi và bài nạp bù.

    Returns:
        ConfusionMetrics: Đối tượng chứa đầy đủ các chỉ số thống kê.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài y_true ({len(y_true)}) và y_pred ({len(y_pred)}) phải bằng nhau.")

    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if not yt and not yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt and not yp)

    evaluated = tp + tn + fp + fn
    total = evaluated + error_count

    accuracy = (tp + tn) / evaluated if evaluated > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2 if evaluated > 0 else 0.0
    error_rate = (fp + fn) / evaluated if evaluated > 0 else 0.0

    return ConfusionMetrics(
        total_samples=total,
        evaluated_samples=evaluated,
        errors=error_count,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1_score,
        balanced_accuracy=balanced_acc,
        error_rate=error_rate,
        error_records=error_records or [],
    )


def plot_confusion_matrix(
    metrics: ConfusionMetrics,
    output_path: str | Path | None = None,
    title: str = "Confusion Matrix & Accuracy Assessment",
    model_name: str | None = None,
) -> Path:
    """Vẽ ma trận nhầm lẫn 2x2 chuyên nghiệp và bảng thống kê chỉ số hiệu năng."""
    if output_path is None:
        out_dir = root_dir / "data" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_file = out_dir / f"confusion_matrix_{ts}.png"
    else:
        target_file = Path(output_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)

    actual_human_total = metrics.tn + metrics.fp
    actual_ai_total = metrics.fn + metrics.tp

    tn_pct = (metrics.tn / actual_human_total * 100) if actual_human_total > 0 else 0.0
    fp_pct = (metrics.fp / actual_human_total * 100) if actual_human_total > 0 else 0.0
    fn_pct = (metrics.fn / actual_ai_total * 100) if actual_ai_total > 0 else 0.0
    tp_pct = (metrics.tp / actual_ai_total * 100) if actual_ai_total > 0 else 0.0

    fig, (ax_cm, ax_stats) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 6.8),
        dpi=300,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.patch.set_facecolor("#FAFAFC")

    cmap = plt.cm.Blues
    norm_matrix = np.array(
        [
            [tn_pct / 100.0, fp_pct / 100.0],
            [fn_pct / 100.0, tp_pct / 100.0],
        ]
    )

    im = ax_cm.imshow(norm_matrix, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Tỉ lệ theo nhóm thực tế (%)")

    classes = ["HUMAN", "AI"]
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels([f"Đoán: {c}" for c in classes], fontsize=11, fontweight="bold")
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels([f"Thực tế: {c}" for c in classes], fontsize=11, fontweight="bold")
    ax_cm.set_xlabel("DỰ ĐOÁN CỦA HỆ THỐNG (PREDICTED)", fontsize=11, labelpad=10)
    ax_cm.set_ylabel("NHÃN THỰC TẾ (GROUND TRUTH)", fontsize=11, labelpad=10)
    ax_cm.set_title("MA TRẬN NHẦM LẪN (2x2 MATRIX)", fontsize=13, fontweight="bold", pad=12)

    cell_info = [
        [
            {
                "label": "TRUE NEGATIVE (TN)",
                "count": metrics.tn,
                "pct": tn_pct,
                "desc": "✓ Đúng: Nhạc người",
            },
            {
                "label": "FALSE POSITIVE (FP)",
                "count": metrics.fp,
                "pct": fp_pct,
                "desc": "✗ Báo động giả (Vu oan)",
            },
        ],
        [
            {
                "label": "FALSE NEGATIVE (FN)",
                "count": metrics.fn,
                "pct": fn_pct,
                "desc": "✗ Lọt lưới (Bỏ sót AI)",
            },
            {
                "label": "TRUE POSITIVE (TP)",
                "count": metrics.tp,
                "pct": tp_pct,
                "desc": "✓ Đúng: Nhạc AI",
            },
        ],
    ]

    for i in range(2):
        for j in range(2):
            info = cell_info[i][j]
            val_pct = norm_matrix[i, j]
            text_color = "white" if val_pct > 0.55 else "#1A202C"

            txt = f"{info['label']}\nSố lượng: {info['count']} ({info['pct']:.1f}%)\n{info['desc']}"
            ax_cm.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10.5,
                fontweight="bold",
                linespacing=1.4,
            )

    ax_stats.set_facecolor("#FFFFFF")
    ax_stats.spines["top"].set_visible(False)
    ax_stats.spines["right"].set_visible(False)
    ax_stats.spines["left"].set_visible(False)
    ax_stats.spines["bottom"].set_color("#E2E8F0")

    metrics_list = [
        ("Accuracy (Độ chính xác)", metrics.accuracy),
        ("Precision (Độ tin cậy báo AI)", metrics.precision),
        ("Recall (Độ nhạy bắt AI)", metrics.recall),
        ("Specificity (Bảo vệ nhạc người)", metrics.specificity),
        ("F1-Score (Cân bằng P/R)", metrics.f1_score),
        ("Balanced Accuracy", metrics.balanced_accuracy),
    ]

    y_pos = np.arange(len(metrics_list))
    names = [m[0] for m in metrics_list]
    values = [m[1] * 100 for m in metrics_list]

    bar_colors = ["#10B981" if v >= 80.0 else "#F59E0B" if v >= 60.0 else "#EF4444" for v in values]

    bars = ax_stats.barh(
        y_pos,
        values,
        height=0.55,
        color=bar_colors,
        edgecolor="#CBD5E1",
        alpha=0.9,
    )
    ax_stats.set_yticks(y_pos)
    ax_stats.set_yticklabels(names, fontsize=10.5, fontweight="bold")
    ax_stats.set_xlim(0, 118)
    ax_stats.set_xlabel("Tỷ lệ phần trăm (%)", fontsize=11, labelpad=8)
    ax_stats.set_title("BẢNG CHỈ SỐ ĐO LƯỜNG HIỆU NĂNG", fontsize=13, fontweight="bold", pad=12)
    ax_stats.invert_yaxis()

    for bar, val in zip(bars, values, strict=True):
        width = bar.get_width()
        ax_stats.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#1E293B",
        )

    summary_text = (
        f"• Tổng số mẫu kiểm thử: {metrics.total_samples} bài "
        f"(Thành công: {metrics.evaluated_samples} | Lỗi: {metrics.errors})\n"
        f"• Dự đoán đúng: {metrics.tp + metrics.tn} bài ({metrics.accuracy * 100:.1f}%) "
        f"| Dự đoán sai: {metrics.fp + metrics.fn} bài ({metrics.error_rate * 100:.1f}%)\n"
        f"• Báo động giả (FP): {metrics.fp} bài | Bỏ sót AI (FN): {metrics.fn} bài\n"
        f"• Mô hình đánh giá: {model_name or 'Project Core AI Engine'}"
    )

    fig.text(
        0.5,
        0.025,
        summary_text,
        ha="center",
        fontsize=9.8,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F1F5F9",
            "edgecolor": "#CBD5E1",
            "alpha": 0.9,
        },
    )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98, color="#0F172A")
    plt.tight_layout(rect=(0.02, 0.12, 0.98, 0.93))

    plt.savefig(target_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return target_file


def calculate_work_type_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Tính toán ma trận nhầm lẫn đa lớp và chỉ số hiệu năng cho Work Type."""
    if labels is None:
        labels = ["ORIGINAL", "COVER", "REMIX"]

    all_pred_cols = list(labels)
    if "UNCERTAIN" not in all_pred_cols:
        all_pred_cols.append("UNCERTAIN")

    matrix: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in all_pred_cols} for gt in labels}

    total = 0
    correct = 0

    for yt, yp in zip(y_true, y_pred, strict=False):
        yt_norm = (yt or "").upper().strip()
        yp_norm = (yp or "").upper().strip()
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
        out_dir = root_dir / "data" / "outputs"
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


def load_testcase_dataset(
    file_path: str | Path = "testcase.xlsx", sheet_name: str = "Media_W38"
) -> list[dict[str, Any]]:
    """Đọc dữ liệu từ file Excel testcase và trích xuất danh sách bài hát kèm nhãn Ground Truth."""
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        if (root_dir / path_obj).exists():
            path_obj = root_dir / path_obj
        elif (root_dir / "data" / path_obj).exists():
            path_obj = root_dir / "data" / path_obj
        else:
            path_obj = root_dir / path_obj

    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    wb = openpyxl.load_workbook(str(path_obj), data_only=True)
    if sheet_name not in wb.sheetnames:
        sheet_name = wb.sheetnames[0]

    sheet = wb[sheet_name]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"Sheet '{sheet_name}' không có dữ liệu.")

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]

    # Tìm các cột cần thiết
    title_idx = -1
    artist_idx = -1
    ai_col_idx = -1
    media_id_idx = -1
    note_idx = -1
    song_type_idx = -1
    release_date_idx = -1
    distributor_idx = -1

    for idx, h in enumerate(headers):
        h_low = h.lower()
        if "song title" in h_low or "tên bài" in h_low or "title" in h_low:
            title_idx = idx
        elif "artist" in h_low or "ca sĩ" in h_low or "nghệ sĩ" in h_low:
            artist_idx = idx
        elif "ai yes" in h_low or h_low == "ai":
            ai_col_idx = idx
        elif "media_id" in h_low or h_low == "id":
            media_id_idx = idx
        elif "note" in h_low or "ghi chú" in h_low:
            note_idx = idx
        elif "song type" in h_low or "loại bài" in h_low or "type" in h_low:
            song_type_idx = idx
        elif "release" in h_low or "ngày phát hành" in h_low:
            release_date_idx = idx
        elif "distributor" in h_low or "nhà phân phối" in h_low:
            distributor_idx = idx

    if title_idx == -1:
        title_idx = 1 if len(headers) > 1 else 0

    dataset: list[dict[str, Any]] = []
    for r_idx, r in enumerate(rows[1:], start=2):
        title = str(r[title_idx]).strip() if r[title_idx] is not None else ""
        if not title:
            continue

        artist = (
            str(r[artist_idx]).strip() if (artist_idx != -1 and r[artist_idx] is not None) else ""
        )
        media_id = (
            str(r[media_id_idx]).strip()
            if (media_id_idx != -1 and r[media_id_idx] is not None)
            else f"row_{r_idx}"
        )
        note = str(r[note_idx]).strip() if (note_idx != -1 and r[note_idx] is not None) else ""
        song_type = (
            str(r[song_type_idx]).strip()
            if (song_type_idx != -1 and r[song_type_idx] is not None)
            else ""
        )
        release_date = (
            str(r[release_date_idx]).strip()
            if (release_date_idx != -1 and r[release_date_idx] is not None)
            else ""
        )

        distributor = (
            str(r[distributor_idx]).strip()
            if (distributor_idx != -1 and r[distributor_idx] is not None)
            else ""
        )

        ai_val = (
            str(r[ai_col_idx]).strip().lower()
            if (ai_col_idx != -1 and r[ai_col_idx] is not None)
            else ""
        )
        # Nhãn thực tế (Ground Truth)
        is_ai_ground_truth = "yes" in ai_val or ai_val == "y"
        gt_work_type = map_song_type_to_work_type(song_type)

        query = f"{title} {artist}".strip()
        dataset.append(
            {
                "media_id": media_id,
                "title": title,
                "artist": artist,
                "query": query,
                "gt_is_ai": is_ai_ground_truth,
                "gt_label": "AI" if is_ai_ground_truth else "HUMAN",
                "gt_work_type": gt_work_type,
                "note": note,
                "song_type": song_type,
                "release_date": release_date,
                "distributor": distributor,
                "row_idx": r_idx,
            }
        )

    return dataset


def select_samples(
    dataset: list[dict[str, Any]],
    sample_size: int | None = 20,
    balanced: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Lấy mẫu bài hát để thẩm định (hỗ trợ chế độ cân bằng phân tầng 50% AI - 50% Human)."""
    if sample_size is None or sample_size >= len(dataset):
        return dataset

    random.seed(seed)
    ai_items = [d for d in dataset if d["gt_is_ai"]]
    human_items = [d for d in dataset if not d["gt_is_ai"]]

    if balanced:
        half = sample_size // 2
        selected_ai = random.sample(ai_items, min(half, len(ai_items)))
        remaining = sample_size - len(selected_ai)
        selected_human = random.sample(human_items, min(remaining, len(human_items)))
        combined = selected_ai + selected_human
        random.shuffle(combined)
        return combined
    else:
        return random.sample(dataset, sample_size)


def load_metrics_from_audit_report(
    report_file_path: str | Path,
) -> tuple[ConfusionMetrics, list[dict[str, Any]]]:
    """Đọc dữ liệu đối soát từ file Excel kết quả kiểm tra trước đó để vẽ Confusion Matrix.

    Giúp người dùng tận dụng kết quả đã test để vẽ biểu đồ ngay lập tức
    mà không cần phải chạy lại toàn bộ mô hình tốn thời gian.
    """
    path_obj = Path(report_file_path)
    if not path_obj.is_absolute():
        path_obj = root_dir / path_obj

    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file báo cáo: {path_obj}")

    with pd.ExcelFile(str(path_obj)) as xl:
        target_sheet = "Audit Details" if "Audit Details" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(xl, sheet_name=target_sheet)

    # Tìm các cột Ground Truth và Prediction cho phân loại AI
    gt_col = None
    pred_col = None

    # 1. Tìm cột Ground Truth (ưu tiên nhãn AI thực tế, tránh các cột tác phẩm như gt_work_type)
    gt_exact = [
        "ground_truth",
        "ground_truth_ai",
        "gt_label",
        "nhãn thực tế",
        "thực tế",
        "gt_is_ai",
        "is_ai",
    ]
    for cand in gt_exact:
        for col in df.columns:
            if f"{col}".strip().lower() == cand:
                gt_col = col
                break
        if gt_col:
            break

    if gt_col is None:
        for col in df.columns:
            c_low = f"{col}".strip().lower()
            if (
                ("ground" in c_low or "thực tế" in c_low or c_low == "gt")
                and "work" not in c_low
                and "type" not in c_low
            ):
                gt_col = col
                break

    # 2. Tìm cột Prediction (ưu tiên kết quả dự đoán AI/Human, tránh pred_work_type hay các cột is_correct)
    pred_exact = [
        "prediction",
        "consensus_prediction",
        "ai_model_prediction",
        "dự đoán",
        "kết quả dự đoán",
    ]
    for cand in pred_exact:
        for col in df.columns:
            if f"{col}".strip().lower() == cand:
                pred_col = col
                break
        if pred_col:
            break

    if pred_col is None:
        for col in df.columns:
            c_low = f"{col}".strip().lower()
            if (
                ("prediction" in c_low or "dự đoán" in c_low or c_low == "pred")
                and "work" not in c_low
                and "type" not in c_low
                and "is_correct" not in c_low
            ):
                pred_col = col
                break

    if gt_col is None or pred_col is None:
        raise ValueError(
            f"Không tìm thấy cột Ground Truth hoặc Prediction trong {path_obj}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    y_true: list[bool] = []
    y_pred: list[bool] = []
    errors = 0
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        gt_val = str(row[gt_col]).strip().upper()
        pred_val = str(row[pred_col]).strip().upper()

        if "ERROR" in pred_val or "LỖI" in pred_val:
            errors += 1
            continue

        gt_is_ai = "AI" in gt_val or "YES" in gt_val
        pred_is_ai = "AI" in pred_val or "YES" in pred_val

        y_true.append(gt_is_ai)
        y_pred.append(pred_is_ai)
        records.append(dict(row))

    metrics = calculate_confusion_metrics(y_true, y_pred, error_count=errors)
    return metrics, records


def run_benchmark(
    samples: list[dict[str, Any]],
    dataset: list[dict[str, Any]] | None = None,
    model_override: str | None = None,
    force_rescan: bool = False,
    mock_mode: bool = False,
    mock_accuracy: float = 0.70,
    allow_replenish: bool = True,
    max_replacements: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Thực hiện thẩm định toàn bộ mẫu và tính toán các chỉ số độ chính xác.
    Hỗ trợ cơ chế Nạp Bù Mẫu Tự Động (Resilient Sampling Pool):
    Khi 1 bài hát gặp lỗi tải/Zing VIP, tự động lấy bài kế tiếp cùng nhãn từ dataset dự phòng
    để đảm bảo đủ chỉ tiêu số bài thành công, đồng thời lưu vết toàn bộ bài lỗi.
    Hỗ trợ chế độ mock_mode để kiểm thử và trực quan hóa ngoại tuyến mà không gọi API.
    """
    results: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    y_true_work: list[str] = []
    y_pred_work: list[str] = []

    tp = 0  # Thực tế AI, Đoán AI (True Positive)
    tn = 0  # Thực tế Người, Đoán Người (True Negative)
    fp = 0  # Thực tế Người, Đoán AI (False Positive - Vu oan)
    fn = 0  # Thực tế AI, Đoán Người (False Negative - Bỏ sót)
    errors = 0
    unverified_matches = 0  # Số bài không tìm thấy nguồn audio khớp trên YouTube/Web

    evaluation_queue = list(samples)

    if mock_mode:
        random.seed(42)
        print("=" * 75)
        print(
            f"⚡ ĐANG CHẠY CHẾ ĐỘ MÔ PHỎNG (MOCK MODE): {len(samples)} BÀI HÁT (ACCURACY: {mock_accuracy * 100:.1f}%)"
        )
        print("=" * 75)
        for idx, item in enumerate(evaluation_queue, 1):
            gt_is_ai = bool(item.get("gt_is_ai", False))
            gt = item.get("gt_label", "AI" if gt_is_ai else "HUMAN")
            query = item.get("query", f"{item.get('title', '')} {item.get('artist', '')}".strip())

            is_correct = random.random() < mock_accuracy
            pred_is_ai = gt_is_ai if is_correct else not gt_is_ai
            pred_label = "AI" if pred_is_ai else "HUMAN"

            gt_work = item.get("gt_work_type", "ORIGINAL")
            pred_work = (
                gt_work if is_correct else ("COVER" if gt_work == "ORIGINAL" else "ORIGINAL")
            )
            if gt_work != "UNCERTAIN":
                y_true_work.append(gt_work)
                y_pred_work.append(pred_work)

            if gt_is_ai and pred_is_ai:
                metric_type = "TP (Đúng: Phát hiện AI)"
                tp += 1
            elif not gt_is_ai and not pred_is_ai:
                metric_type = "TN (Đúng: Xác nhận Người)"
                tn += 1
            elif not gt_is_ai and pred_is_ai:
                metric_type = "FP (Sai: Vu oan nhạc người)"
                fp += 1
            else:
                metric_type = "FN (Sai: Bỏ sót nhạc AI)"
                fn += 1

            results.append(
                {
                    "media_id": item.get("media_id", f"mock_{idx}"),
                    "song_title": item.get("title", ""),
                    "artist": item.get("artist", ""),
                    "query": query,
                    "ground_truth": gt,
                    "gt_work_type": gt_work,
                    "pred_work_type": pred_work,
                    "work_is_correct": "ĐÚNG" if pred_work == gt_work else "SAI",
                    "work_label": pred_work,
                    "original_artist": "Mock Artist",
                    "original_song_title": "Mock Title",
                    "is_cover": pred_work == "COVER",
                    "ai_model_prediction": pred_label,
                    "ai_model_verdict": ("AI_GENERATED" if pred_is_ai else "PROCESSED_HUMAN"),
                    "ai_model_is_correct": "ĐÚNG" if is_correct else "SAI",
                    "consensus_prediction": pred_label,
                    "consensus_verdict": ("AI_GENERATED" if pred_is_ai else "PROCESSED_HUMAN"),
                    "prediction": pred_label,
                    "is_correct": "ĐÚNG" if is_correct else "SAI",
                    "classification_type": metric_type,
                    "is_verified_source": True,
                    "source_verification": "ĐÃ XÁC THỰC",
                    "match_score": "100.0%",
                    "verdict_type": ("AI_GENERATED" if pred_is_ai else "PROCESSED_HUMAN"),
                    "verdict_label": ("AI GENERATED" if pred_is_ai else "PROCESSED HUMAN"),
                    "confidence_score": f"{mock_accuracy * 100:.1f}%",
                    "suspected_engine": "Suno" if pred_is_ai else "None",
                    "reasons": ["Mock detection simulation"],
                    "audio_url": "https://youtube.com/watch?v=mock",
                    "note": item.get("note", ""),
                }
            )
    else:
        container = get_container()
        detection_service = container.detection_service
        resolver = container.resolver
        attempted_ids = {s["media_id"] for s in samples}

        print("=" * 75)
        print(f"🚀 BẮT ĐẦU CHẤM ĐIỂM TESTCASE (MỤC TIÊU: {len(samples)} BÀI HÁT THÀNH CÔNG)")
        print("=" * 75)

        target_successful = len(samples)
        if max_replacements is None:
            max_replacements = target_successful

        print_lock = threading.Lock()

        def _evaluate_item_logic(item: dict[str, Any], curr_i: int) -> dict[str, Any] | None:
            nonlocal tp, tn, fp, fn, errors, unverified_matches
            gt = item["gt_label"]
            query = item["query"]
            with print_lock:
                success_count = sum(1 for r in results if r.get("prediction") not in ("ERROR", "LỖI"))
                print(
                    f"\n[{curr_i}/{len(evaluation_queue)}] [Thành công: {success_count}/{target_successful} | Lỗi: {errors}] Thẩm định: '{query}' | Ground Truth: {gt}"
                )

            track: TrackItem | None = None
            try:
                try:
                    track = resolver.resolve_track(
                        item["title"], artist=item.get("artist", ""), force_rescan=force_rescan
                    )
                except TypeError:
                    track = resolver.resolve_track(item["title"], artist=item.get("artist", ""))

                track.song_type = None  # Không nạp trước Ground Truth để đảm bảo tính khách quan
                track.release_date = str(item.get("release_date") or "")
                track.distributor = str(item.get("distributor") or "")
                try:
                    res: DetectionResult = detection_service.detect_single(
                        track, force_rescan=force_rescan, model_override=model_override
                    )
                except TypeError:
                    res = detection_service.detect_single(track, model_override=model_override)

                # Đọc lại trạng thái xác thực nguồn sau khi audio_engine đã chạy và xử lý fallback
                is_verified = track.is_verified_match is not False
                if not is_verified or (track.match_score is not None and track.match_score < 0.70):
                    with print_lock:
                        unverified_matches += 1
                    score_pct = (
                        f"{track.match_score * 100:.1f}%"
                        if track.match_score is not None
                        else "0.0%"
                    )
                    raise ValueError(
                        f"Nguồn âm thanh không khớp bản gốc (Score: {score_pct} < 70.0% / Unverified). "
                        "Từ chối đưa vào đối soát để bảo toàn độ sạch của benchmark, kích hoạt nạp bù bài khác."
                    )

                pred_is_ai = res.is_ai
                pred_label = "AI" if pred_is_ai else "HUMAN"
                is_correct = pred_is_ai == item["gt_is_ai"]

                # Đánh giá Work Type (Original vs Cover vs Remix) - Luôn giải mã qua metadata/tiêu đề để đảm bảo chính xác
                gt_work = item.get("gt_work_type", "UNCERTAIN")
                raw_w_type = getattr(res, "work_type", None)
                pred_work_raw = (
                    raw_w_type.value
                    if hasattr(raw_w_type, "value")
                    else (str(raw_w_type) if raw_w_type else "UNCERTAIN")
                )
                # Bóc tách hoàn toàn nhãn kiểm thử (Ground Truth) để tránh Data Leakage
                clean_metadata = {
                    k: v
                    for k, v in item.items()
                    if k
                    not in (
                        "song_type",
                        "Song Type",
                        "work_type",
                        "Work Type",
                        "gt_work_type",
                        "AI YES",
                        "NOTE",
                        "gt_is_ai",
                        "gt_label",
                    )
                }
                resolved_w, resolved_label, is_cov, orig_a, orig_t = (
                    resolve_work_type_from_metadata(
                        title=item.get("title") or track.track_title or track.raw_query,
                        artist=item.get("artist") or track.artist or "",
                        metadata=clean_metadata,
                        raw_gemini_work_type=pred_work_raw,
                    )
                )
                pred_work = resolved_w.value
                res.work_type = resolved_w
                res.work_label = resolved_label
                res.is_cover = is_cov
                if orig_a and not getattr(res, "original_artist", None):
                    res.original_artist = orig_a
                if orig_t and not getattr(res, "original_song_title", None):
                    res.original_song_title = orig_t

                work_is_correct = (pred_work == gt_work) if gt_work != "UNCERTAIN" else None

                # Phân loại Confusion Matrix
                if item["gt_is_ai"] and pred_is_ai:
                    metric_type = "TP (Đúng: Phát hiện AI)"
                elif not item["gt_is_ai"] and not pred_is_ai:
                    metric_type = "TN (Đúng: Xác nhận Người)"
                elif not item["gt_is_ai"] and pred_is_ai:
                    metric_type = "FP (Sai: Vu oan nhạc người)"
                else:
                    metric_type = "FN (Sai: Bỏ sót nhạc AI)"

                icon = "✅" if is_correct else "❌"
                match_status_text = "ĐÃ XÁC THỰC" if is_verified else "SAI LỆCH (UNVERIFIED)"
                ai_model_is_ai = getattr(res, "ai_model_is_ai", res.is_ai)
                ai_pred_label = "AI" if ai_model_is_ai else "HUMAN"
                ai_verdict_str = res.ai_model_verdict or ai_pred_label
                ai_is_correct = "ĐÚNG" if ai_model_is_ai == item["gt_is_ai"] else "SAI"

                consensus_str = res.consensus_verdict or (
                    "AI" if getattr(res, "consensus_is_ai", res.is_ai) else "HUMAN"
                )

                work_icon = (
                    "💿" if pred_work == "ORIGINAL" else ("🎙️" if pred_work == "COVER" else "🎛️")
                )

                rec = {
                    "media_id": item["media_id"],
                    "song_title": item["title"],
                    "artist": item["artist"],
                    "query": query,
                    "ground_truth": gt,
                    "gt_work_type": gt_work,
                    "pred_work_type": pred_work,
                    "work_is_correct": "ĐÚNG"
                    if work_is_correct is True
                    else ("SAI" if work_is_correct is False else "N/A"),
                    "work_label": getattr(res, "work_label", "Chưa rõ tác phẩm"),
                    "original_artist": getattr(res, "original_artist", "") or "N/A",
                    "original_song_title": getattr(res, "original_song_title", "") or "N/A",
                    "is_cover": getattr(res, "is_cover", False),
                    "ai_model_prediction": ai_pred_label,
                    "ai_model_verdict": ai_verdict_str,
                    "ai_model_is_correct": ai_is_correct,
                    "consensus_prediction": pred_label,
                    "consensus_verdict": consensus_str,
                    "prediction": pred_label,
                    "is_correct": "ĐÚNG" if is_correct else "SAI",
                    "classification_type": metric_type,
                    "is_verified_source": is_verified,
                    "source_verification": match_status_text,
                    "match_score": (
                        f"{track.match_score * 100:.1f}%"
                        if track.match_score is not None
                        else "N/A"
                    ),
                    "verdict_type": (
                        res.verdict_type.value
                        if hasattr(res.verdict_type, "value")
                        else str(res.verdict_type)
                    ),
                    "verdict_label": res.verdict_label,
                    "confidence_score": f"{res.confidence_score * 100:.1f}%",
                    "suspected_engine": res.suspected_engine,
                    "source_url": (res.source_url or getattr(track, "resolved_url", "") or ""),
                    "reasons": " | ".join(res.reasons or []),
                    "error": "",
                }

                with print_lock:
                    if item["gt_is_ai"] and pred_is_ai:
                        tp += 1
                    elif not item["gt_is_ai"] and not pred_is_ai:
                        tn += 1
                    elif not item["gt_is_ai"] and pred_is_ai:
                        fp += 1
                    else:
                        fn += 1

                    if gt_work != "UNCERTAIN":
                        y_true_work.append(gt_work)
                        y_pred_work.append(pred_work)

                    results.append(rec)
                    print(
                        f"       -> Model AI: {ai_verdict_str} | Tổng hợp: {consensus_str} ({res.verdict_label}) | {icon} {metric_type}"
                    )
                    print(
                        f"       -> Tác phẩm: {work_icon} {getattr(res, 'work_label', pred_work)} (Thực tế: {gt_work})"
                    )
                    print(
                        f"       -> Nguồn audio: {match_status_text} | Độ tin cậy: {res.confidence_score * 100:.1f}%"
                    )
                    print(f"       -> Lý do: {res.reasons[0] if res.reasons else 'N/A'}")

                return None

            except Exception as e:
                with print_lock:
                    errors += 1
                    print(f"       -> ❌ LỖI XỬ LÝ NGUỒN: {e}")

                    replacement = None
                    replenish_status = ""
                    curr_success = sum(1 for r in results if r.get("prediction") not in ("ERROR", "LỖI"))

                    if not allow_replenish:
                        replenish_status = "Chế độ --no-replenish đang bật (không nạp bù)"
                    elif not dataset:
                        replenish_status = "Không có dataset dự phòng để nạp bù"
                    elif curr_success >= target_successful:
                        replenish_status = f"Đã đạt đủ mục tiêu {target_successful} bài thành công"
                    elif errors > max_replacements:
                        replenish_status = f"Đã đạt giới hạn nạp bù an toàn tối đa ({max_replacements} bài)"
                    else:
                        candidates_pool = [
                            cand
                            for cand in dataset
                            if (
                                cand.get("gt_is_ai") == item.get("gt_is_ai")
                                and cand.get("media_id") not in attempted_ids
                            )
                        ]
                        if candidates_pool:
                            replacement = random.choice(candidates_pool)
                            attempted_ids.add(replacement["media_id"])
                        else:
                            replenish_status = f"Hết bài dự phòng cùng nhãn {gt} trong dataset"

                    error_records.append(
                        {
                            "Mã bài lỗi (Media ID)": item.get("media_id", ""),
                            "Tên bài lỗi (Song Title)": item.get("title", ""),
                            "Ca sĩ (Artist)": item.get("artist", ""),
                            "Nhãn thực tế (Ground Truth)": gt,
                            "Nguyên nhân sự cố (Error)": str(e),
                            "Bài nạp bù thay thế (Replacement)": (
                                replacement["query"] if replacement else f"Không nạp bù ({replenish_status})"
                            ),
                            "Thời điểm ghi nhận": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    if replacement:
                        print(
                            f"       -> 🔄 ĐÃ NẠP BÙ MẪU THAY THẾ: '{replacement['query']}' (cùng nhãn {gt} | Hàng đợi: {len(evaluation_queue)} bài)"
                        )
                    else:
                        print(f"       -> ⚠️ {replenish_status} (ghi nhận 1 lỗi vào báo cáo)")
                        results.append(
                            {
                                "media_id": item["media_id"],
                                "song_title": item["title"],
                                "artist": item["artist"],
                                "query": query,
                                "ground_truth": gt,
                                "ai_model_prediction": "ERROR",
                                "ai_model_verdict": "ERROR",
                                "ai_model_is_correct": "LỖI",
                                "consensus_prediction": "ERROR",
                                "consensus_verdict": "ERROR",
                                "prediction": "ERROR",
                                "is_correct": "LỖI",
                                "classification_type": "ERROR",
                                "is_verified_source": False,
                                "source_verification": "LỖI",
                                "match_score": "0%",
                                "verdict_type": "ERROR",
                                "verdict_label": "LỖI HỆ THỐNG",
                                "confidence_score": "0%",
                                "suspected_engine": "N/A",
                                "source_url": (
                                    getattr(track, "resolved_url", "")
                                    if track and getattr(track, "resolved_url", None)
                                    else ""
                                ),
                                "reasons": str(e),
                                "error": str(e),
                            }
                        )
                return replacement

        if workers > 1:
            active_workers = max(1, min(workers, 16))
            print("=" * 75)
            print(
                f"🚀 BẮT ĐẦU CHẤM ĐIỂM TESTCASE ĐA LUỒNG (MỤC TIÊU: {len(samples)} BÀI HÁT THÀNH CÔNG | WORKERS: {active_workers})"
            )
            print("=" * 75)

            queue_idx = 0
            with ThreadPoolExecutor(max_workers=active_workers) as executor:
                futures = {}
                for itm in evaluation_queue:
                    queue_idx += 1
                    futures[executor.submit(_evaluate_item_logic, itm, queue_idx)] = itm

                while futures:
                    done_set = []
                    for fut in as_completed(futures):
                        done_set.append(fut)
                        try:
                            repl = fut.result()
                            with print_lock:
                                curr_succ = sum(
                                    1 for r in results if r.get("prediction") not in ("ERROR", "LỖI")
                                )
                                if repl and curr_succ < target_successful:
                                    queue_idx += 1
                                    evaluation_queue.append(repl)
                                    new_f = executor.submit(_evaluate_item_logic, repl, queue_idx)
                                    futures[new_f] = repl
                        except Exception:
                            pass
                        with print_lock:
                            curr_succ = sum(
                                1 for r in results if r.get("prediction") not in ("ERROR", "LỖI")
                            )
                            if curr_succ >= target_successful:
                                break
                    for df in done_set:
                        futures.pop(df, None)
                    with print_lock:
                        curr_succ = sum(
                            1 for r in results if r.get("prediction") not in ("ERROR", "LỖI")
                        )
                        if curr_succ >= target_successful:
                            for f in futures:
                                f.cancel()
                            break
        else:
            i = 0
            while i < len(evaluation_queue):
                success_count = sum(1 for r in results if r.get("prediction") not in ("ERROR", "LỖI"))
                if success_count >= target_successful:
                    print(
                        f"\n🎉 [HOÀN THÀNH] Đã thẩm định thành công đủ mục tiêu {target_successful} bài hát! Dừng hàng đợi."
                    )
                    break

                if i >= len(evaluation_queue):
                    break

                item = evaluation_queue[i]
                i += 1
                repl = _evaluate_item_logic(item, i)
                if repl:
                    evaluation_queue.append(repl)

                if i < len(evaluation_queue):
                    time.sleep(2.5)

    # Tính toán các chỉ số
    evaluated_total = tp + tn + fp + fn
    accuracy = ((tp + tn) / evaluated_total) if evaluated_total > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Đánh giá chỉ số phân biệt Tác phẩm (Work Type: Original vs Cover vs Remix)
    work_type_metrics = calculate_work_type_confusion_matrix(y_true_work, y_pred_work)
    work_type_cm_img_path: str | None = None
    if work_type_metrics.get("total_evaluated", 0) > 0:
        try:
            img_file = plot_work_type_confusion_matrix(work_type_metrics)
            work_type_cm_img_path = str(img_file)
        except Exception as plot_err:  # noqa: BLE001
            print(f"⚠️ Không thể tạo ảnh ma trận nhầm lẫn tác phẩm: {plot_err}")

    # Tính riêng chỉ số cho tập các bài tìm ĐÚNG nguồn âm thanh (Strictly Verified Tracks)
    verified_results = [r for r in results if r.get("is_verified_source")]
    v_tp = sum(1 for r in verified_results if r["ground_truth"] == "AI" and r["prediction"] == "AI")
    v_tn = sum(
        1 for r in verified_results if r["ground_truth"] == "HUMAN" and r["prediction"] == "HUMAN"
    )
    v_fp = sum(
        1 for r in verified_results if r["ground_truth"] == "HUMAN" and r["prediction"] == "AI"
    )
    v_fn = sum(
        1 for r in verified_results if r["ground_truth"] == "AI" and r["prediction"] == "HUMAN"
    )
    v_total = v_tp + v_tn + v_fp + v_fn
    v_acc = ((v_tp + v_tn) / v_total) if v_total > 0 else 0.0
    v_rec = (v_tp / (v_tp + v_fn)) if (v_tp + v_fn) > 0 else 0.0
    v_prec = (v_tp / (v_tp + v_fp)) if (v_tp + v_fp) > 0 else 0.0
    v_f1 = (2 * v_prec * v_rec / (v_prec + v_rec)) if (v_prec + v_rec) > 0 else 0.0

    metrics_summary = {
        "total_samples": len(samples),
        "evaluated_successfully": evaluated_total,
        "errors": errors,
        "unverified_matches": unverified_matches,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "error_records": error_records,
        "replaced_errors_count": len(error_records),
        "work_type_metrics": work_type_metrics,
        "work_type_cm_image_path": work_type_cm_img_path,
        "verified_metrics": {
            "verified_samples": v_total,
            "TP": v_tp,
            "TN": v_tn,
            "FP": v_fp,
            "FN": v_fn,
            "accuracy": v_acc,
            "recall": v_rec,
            "precision": v_prec,
            "f1_score": v_f1,
        },
    }

    return {
        "metrics": metrics_summary,
        "results": results,
        "unverified_matches": unverified_matches,
    }


def print_summary_report(
    metrics: dict[str, Any] | ConfusionMetrics,
    results: list[dict[str, Any]] | None = None,
):
    """In bảng báo cáo tổng kết chi tiết ra màn hình console."""
    if hasattr(metrics, "to_dict"):
        m = metrics.to_dict()
    elif isinstance(metrics, dict):
        m = metrics
    else:
        m = vars(metrics)

    tp = m.get("TP", m.get("tp", 0))
    tn = m.get("TN", m.get("tn", 0))
    fp = m.get("FP", m.get("fp", 0))
    fn = m.get("FN", m.get("fn", 0))
    total_samples = m.get("total_samples", len(results) if results else (tp + tn + fp + fn))
    evaluated = m.get("evaluated_successfully", m.get("evaluated_samples", (tp + tn + fp + fn)))
    errors = m.get("errors", 0)

    print("\n" + "=" * 75)
    print("                📊 BẢNG TỔNG KẾT ĐÁNH GIÁ ĐỘ CHÍNH XÁC")
    print("=" * 75)
    print(f" Tổng số bài kiểm tra    : {total_samples}")
    print(f" Số bài thành công        : {evaluated}")
    print(f" Số bài bị lỗi kết nối    : {errors}")
    if m.get("replaced_errors_count", 0) > 0:
        print(
            f" 🔄 Số bài lỗi đã tự động nạp bù : {m['replaced_errors_count']} bài (Chi tiết tại Sheet 'Sự cố & Nạp bù')"
        )
    print("-" * 75)
    print(" 🔢 MA TRẬN NHẦM LẪN AI/HUMAN (CONFUSION MATRIX):")
    print(f"   • True Positive  (TP - Đoán đúng AI)   : {tp} bài")
    print(f"   • True Negative  (TN - Đoán đúng Người): {tn} bài")
    print(f"   • False Positive (FP - Vu oan Người)   : {fp} bài (⚠️ Báo động giả)")
    print(f"   • False Negative (FN - Bỏ sót AI)      : {fn} bài (⚠️ Lọt lưới)")
    print("-" * 75)
    print(" 🎯 CÁC CHỈ SỐ ĐO LƯỜNG PHÂN BIỆT AI:")
    print(f"   ⭐ Accuracy (Độ chính xác tổng)        : {m.get('accuracy', 0.0) * 100:.2f}%")
    print(f"   🎯 Precision (Độ tin cậy khi báo AI)   : {m.get('precision', 0.0) * 100:.2f}%")
    print(f"   📡 Recall (Độ nhạy bắt bài AI)         : {m.get('recall', 0.0) * 100:.2f}%")
    print(f"   🛡️ Specificity (Độ đặc hiệu nhạc người): {m.get('specificity', 0.0) * 100:.2f}%")
    print(f"   ⚖️ F1-Score (Cân bằng Precision/Recall): {m.get('f1_score', 0.0) * 100:.2f}%")
    print("=" * 75)

    wt_metrics = m.get("work_type_metrics")
    if wt_metrics and wt_metrics.get("total_evaluated", 0) > 0:
        print("\n" + "=" * 75)
        print("      💿 BẢNG ĐỐI SOÁT PHÂN BIỆT TÁC PHẨM (ORIGINAL vs COVER vs REMIX)")
        print("=" * 75)
        print(f" Tổng số bài có nhãn tác phẩm : {wt_metrics['total_evaluated']} bài")
        print(f" Số bài nhận diện đúng tác phẩm: {wt_metrics['correct']} bài")
        print(f" ⭐ Độ chính xác tác phẩm chung : {wt_metrics['accuracy'] * 100:.2f}%")
        print("-" * 75)
        per_class = wt_metrics.get("per_class", {})
        for lbl, c_met in per_class.items():
            print(
                f"   🏷️ {lbl:<8}: Mẫu={c_met['total']:<2} | "
                f"Precision={c_met['precision'] * 100:.1f}% | "
                f"Recall={c_met['recall'] * 100:.1f}% | "
                f"F1-Score={c_met['f1_score'] * 100:.1f}%"
            )
        if m.get("work_type_cm_image_path"):
            print(f" 🖼️ Ma trận nhầm lẫn tác phẩm đã lưu: {m['work_type_cm_image_path']}")
        print("=" * 75)

    if m.get("unverified_matches", 0) > 0:
        print("\n" + "!" * 75)
        print(
            f" ⚠️ CẢNH BÁO NGUỒN ÂM THANH: Có {m['unverified_matches']} bài không tìm thấy trên YouTube (tải video lạc đề)!"
        )
        vm = m.get("verified_metrics", {})
        if vm and vm.get("verified_samples", 0) > 0:
            print(f" 🎯 CHỈ SỐ TRÊN TẬP ĐÃ XÁC THỰC ĐÚNG NGUỒN ({vm['verified_samples']} BÀI):")
            print(f"   ⭐ Verified Accuracy  : {vm['accuracy'] * 100:.2f}%")
            print(f"   🎯 Verified Precision : {vm['precision'] * 100:.2f}%")
            print(f"   📡 Verified Recall    : {vm['recall'] * 100:.2f}%")
            print(f"   ⚖️ Verified F1-Score  : {vm['f1_score'] * 100:.2f}%")
        print("!" * 75)

    # In danh sách bài sai (Error Analysis)
    if results:
        wrong_items = [r for r in results if r.get("is_correct") == "SAI"]
        if wrong_items:
            print("\n🔍 PHÂN TÍCH CÁC TRƯỜNG HỢP DỰ ĐOÁN SAI:")
            for idx, w in enumerate(wrong_items, 1):
                src_note = " [⚠️ SAI LỆCH NGUỒN]" if not w.get("is_verified_source") else ""
                print(f"  {idx}. '{w.get('song_title', '')}' - {w.get('artist', '')}{src_note}")
                print(
                    f"     Thực tế: {w.get('ground_truth', 'N/A')} | Model AI: {w.get('ai_model_verdict', 'N/A')} | Tổng hợp: {w.get('prediction', 'N/A')} ({w.get('classification_type', 'N/A')})"
                )
                print(f"     Lý do model đưa ra: {w.get('reasons', 'N/A')}")
        else:
            print("\n🎉 Tuyệt vời! Không có bài nào bị dự đoán sai trong tập mẫu kiểm tra!")
    print("=" * 75)


def export_audit_excel(
    metrics: dict[str, Any] | ConfusionMetrics,
    results: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    """Xuất file Excel đối soát chi tiết kết quả benchmark kèm Sheet 3 Sự cố & Nạp bù."""
    if hasattr(metrics, "to_dict"):
        m = metrics.to_dict()
    elif isinstance(metrics, dict):
        m = metrics
    else:
        m = vars(metrics)

    if output_path is not None:
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = root_dir / "data" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = out_dir / f"benchmark_audit_{timestamp}.xlsx"

    with pd.ExcelWriter(str(file_path), engine="openpyxl") as writer:
        # Sheet 1: Chi tiết
        df_details = pd.DataFrame(results)
        df_details.to_excel(writer, sheet_name="Audit Details", index=False)

        # Tính toán độ chính xác riêng của mô hình AI thuần túy
        ai_model_correct = sum(1 for r in results if r.get("ai_model_is_correct") == "ĐÚNG")
        ai_model_total = sum(1 for r in results if r.get("ai_model_is_correct") in ("ĐÚNG", "SAI"))
        ai_model_acc_str = (
            f"{(ai_model_correct / ai_model_total * 100):.2f}%" if ai_model_total > 0 else "N/A"
        )

        # Sheet 2: Tóm tắt
        summary_rows = [
            {
                "Chỉ số": "Tổng số bài mẫu thành công",
                "Giá trị": m.get("total_samples", len(results)),
            },
            {
                "Chỉ số": "Số bài thành công",
                "Giá trị": m.get(
                    "evaluated_successfully", m.get("evaluated_samples", len(results))
                ),
            },
            {
                "Chỉ số": "Số bài gặp sự cố (Đã nạp bù)",
                "Giá trị": m.get("replaced_errors_count", m.get("errors", 0)),
            },
            {
                "Chỉ số": "Số bài sai lệch nguồn âm thanh (Unverified)",
                "Giá trị": m.get("unverified_matches", 0),
            },
            {
                "Chỉ số": "Độ chính xác Thuần AI Model (Raw Gemini)",
                "Giá trị": ai_model_acc_str,
            },
            {
                "Chỉ số": "Độ chính xác Tổng hợp (Consensus Final)",
                "Giá trị": f"{m.get('accuracy', 0.0) * 100:.2f}%",
            },
            {"Chỉ số": "True Positive (Đoán đúng AI)", "Giá trị": m.get("TP", m.get("tp", 0))},
            {"Chỉ số": "True Negative (Đoán đúng Người)", "Giá trị": m.get("TN", m.get("tn", 0))},
            {"Chỉ số": "False Positive (Vu oan Người)", "Giá trị": m.get("FP", m.get("fp", 0))},
            {"Chỉ số": "False Negative (Bỏ sót AI)", "Giá trị": m.get("FN", m.get("fn", 0))},
            {
                "Chỉ số": "Accuracy (Độ chính xác AI)",
                "Giá trị": f"{m.get('accuracy', 0.0) * 100:.2f}%",
            },
            {
                "Chỉ số": "Precision (Độ tin cậy AI)",
                "Giá trị": f"{m.get('precision', 0.0) * 100:.2f}%",
            },
            {"Chỉ số": "Recall (Độ nhạy bắt AI)", "Giá trị": f"{m.get('recall', 0.0) * 100:.2f}%"},
            {
                "Chỉ số": "Specificity (Bảo vệ nhạc người)",
                "Giá trị": f"{m.get('specificity', 0.0) * 100:.2f}%",
            },
            {
                "Chỉ số": "F1-Score (Phân biệt AI)",
                "Giá trị": f"{m.get('f1_score', 0.0) * 100:.2f}%",
            },
        ]

        wt = m.get("work_type_metrics")
        if wt and wt.get("total_evaluated", 0) > 0:
            summary_rows.extend(
                [
                    {
                        "Chỉ số": "──────── PHÂN BIỆT TÁC PHẨM ────────",
                        "Giá trị": "───────────────────",
                    },
                    {
                        "Chỉ số": "Tổng số bài đánh giá Tác phẩm",
                        "Giá trị": wt["total_evaluated"],
                    },
                    {
                        "Chỉ số": "Độ chính xác Tác phẩm (Work Type Accuracy)",
                        "Giá trị": f"{wt['accuracy'] * 100:.2f}%",
                    },
                ]
            )
            per_class = wt.get("per_class", {})
            for lbl, c_met in per_class.items():
                summary_rows.append(
                    {
                        "Chỉ số": f"Nhãn {lbl} (Precision / Recall / F1)",
                        "Giá trị": (
                            f"{c_met['precision'] * 100:.1f}% / "
                            f"{c_met['recall'] * 100:.1f}% / "
                            f"{c_met['f1_score'] * 100:.1f}%"
                        ),
                    }
                )
            if m.get("work_type_cm_image_path"):
                summary_rows.append(
                    {
                        "Chỉ số": "File ảnh Ma Trận Nhầm Lẫn Tác Phẩm",
                        "Giá trị": m["work_type_cm_image_path"],
                    }
                )

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_excel(writer, sheet_name="Executive Summary", index=False)

        # Sheet 3: Sự cố & Nạp bù (Error Audit & Replacement)
        error_recs = m.get("error_records", [])
        if error_recs:
            df_err = pd.DataFrame(error_recs)
            df_err.to_excel(writer, sheet_name="Sự cố & Nạp bù", index=False)

    return file_path


# Alias tương thích ngược cho các bộ test
export_evaluation_excel = export_audit_excel
print_cli_report = print_summary_report


def main():
    parser = argparse.ArgumentParser(
        description="Chấm điểm độ chính xác AI Music Detector với testcase.xlsx"
    )
    parser.add_argument(
        "--file", default="testcase.xlsx", help="Đường dẫn file excel (mặc định: testcase.xlsx)"
    )
    parser.add_argument(
        "--sheet", default="Media_W38", help="Tên sheet chứa dữ liệu (mặc định: Media_W38)"
    )
    parser.add_argument(
        "--sample", type=int, default=20, help="Số lượng bài mẫu muốn test (mặc định: 20)"
    )
    parser.add_argument(
        "--balanced", action="store_true", default=True, help="Lấy mẫu cân bằng 50/50 AI và Người"
    )
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ dataset (bỏ qua --sample)")
    parser.add_argument(
        "--model", type=str, default=None, help="Mô hình Gemini override (tùy chọn)"
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        default=False,
        help="Bắt buộc quét mới, không dùng cache cũ",
    )
    parser.add_argument(
        "--check-work-type",
        action="store_true",
        help="Kiểm tra và đối soát độ chính xác phân biệt Original / Cover / Remix với Ground Truth Song Type",
    )
    parser.add_argument(
        "--from-report",
        type=str,
        default=None,
        help="Đọc kết quả từ file Excel audit trước đó để vẽ Confusion Matrix trực tiếp",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Chạy ở chế độ mô phỏng (offline/mock) để kiểm tra nhanh mà không gọi API",
    )
    parser.add_argument(
        "--mock-acc",
        type=float,
        default=0.70,
        help="Tỉ lệ chính xác giả lập khi chạy --mock (mặc định: 0.70)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu hình ảnh Confusion Matrix",
    )
    parser.add_argument(
        "--no-replenish",
        action="store_true",
        help="Không nạp bù bài hát khi gặp lỗi (chỉ đánh giá đúng số lượng bài mẫu ban đầu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=BENCHMARK_WORKERS,
        help=f"Số luồng chạy đồng thời (mặc định: {BENCHMARK_WORKERS})",
    )

    args = parser.parse_args()

    if args.check_work_type:
        from check_work_type_accuracy import check_work_type_accuracy

        check_work_type_accuracy(testcase_path=args.file)
        return

    if args.from_report:
        print(f"📖 Đang đọc kết quả đối soát từ: {args.from_report}")
        metrics_cm, results = load_metrics_from_audit_report(args.from_report)
        print_summary_report(metrics_cm, results)
        saved_img = plot_confusion_matrix(
            metrics=metrics_cm,
            output_path=args.output,
            title="Confusion Matrix (Từ Báo Cáo Đối Soát)",
            model_name="Imported Audit Report",
        )
        print(f"\n🖼️  Đã xuất hình ảnh Confusion Matrix tại: {saved_img}\n")
        return

    print(f"📖 Đang đọc dữ liệu từ '{args.file}' [Sheet: '{args.sheet}']...")
    dataset = load_testcase_dataset(file_path=args.file, sheet_name=args.sheet)

    total_ai = sum(1 for d in dataset if d["gt_is_ai"])
    total_human = sum(1 for d in dataset if not d["gt_is_ai"])
    print(f"✅ Đã tải {len(dataset)} bài hát (AI: {total_ai} bài | Con người: {total_human} bài)")

    if args.all:
        samples = dataset
        print(f"⚠️ Chế độ FULL: Sẽ thẩm định toàn bộ {len(samples)} bài hát.")
    else:
        samples = select_samples(dataset, sample_size=args.sample, balanced=args.balanced)
        sample_ai = sum(1 for s in samples if s["gt_is_ai"])
        sample_human = sum(1 for s in samples if not s["gt_is_ai"])
        print(
            f"🎯 Lấy mẫu thẩm định: {len(samples)} bài (AI: {sample_ai} bài | Con người: {sample_human} bài)"
        )

    if args.mock:
        print(f"⚡ Đang chạy chế độ MOCK (Mô phỏng tỉ lệ {args.mock_acc * 100:.1f}%)...")
    else:
        print("🚀 Đang thẩm định qua hệ thống AI Music Detector...")

    benchmark_data = run_benchmark(
        samples,
        dataset=dataset,
        model_override=args.model,
        force_rescan=args.force_rescan,
        mock_mode=args.mock,
        mock_accuracy=args.mock_acc,
        allow_replenish=not args.no_replenish,
        workers=args.workers,
    )
    metrics = benchmark_data["metrics"]
    results = benchmark_data["results"]

    print_summary_report(metrics, results)
    audit_file = export_audit_excel(metrics, results)
    print(f"\n📁 Đã xuất báo cáo đối soát chi tiết tại: {audit_file}")

    # Xuất cả ảnh Confusion Matrix 2x2 AI/Human
    cm_metrics = calculate_confusion_metrics(
        y_true=[r["ground_truth"] == "AI" for r in results],
        y_pred=[r["prediction"] == "AI" for r in results],
        error_count=metrics.get("errors", 0),
        error_records=metrics.get("error_records", []),
    )
    saved_cm_img = plot_confusion_matrix(
        metrics=cm_metrics,
        output_path=args.output,
        title=f"AI Music Detector - Confusion Matrix ({cm_metrics.accuracy * 100:.1f}% Accuracy)",
        model_name="Mock Engine" if args.mock else "Project Core AI Engine",
    )
    print(f"🖼️  Đã xuất hình ảnh Confusion Matrix 2x2 tại: {saved_cm_img}\n")


if __name__ == "__main__":
    main()
