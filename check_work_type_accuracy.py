"""Công cụ kiểm tra và đối soát độ chính xác phân biệt Tác phẩm (Original vs Cover vs Remix).

Đối soát trực tiếp giữa kết quả thẩm định của hệ thống với nhãn Ground Truth (cột 'Song Type')
từ file testcase (testcase.xlsx, testcase2.xlsx, testcase3.xlsx).

Các tính năng chính:
- Tự động nạp file audit mới nhất trong 'data/outputs/' nếu không truyền đường dẫn.
- Đối chiếu nhãn Ground Truth 'Song Type' chuẩn hóa sang ORIGINAL / COVER / REMIX.
- Tính toán ma trận nhầm lẫn 3x3 và các chỉ số Machine Learning (Accuracy, Precision, Recall, F1).
- Liệt kê bảng soi lỗi chi tiết từng bài đoán sai và phân tích nguyên nhân.
- Xuất báo cáo đối soát ra Excel và vẽ hình ảnh Ma trận nhầm lẫn (300 DPI).

Cách sử dụng:
  python check_work_type_accuracy.py
  python check_work_type_accuracy.py --audit data/outputs/evaluation_audit_20261019_231301.xlsx
  python check_work_type_accuracy.py --testcase testcase3.xlsx --export
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import openpyxl
import pandas as pd

from backend.services.work_type_resolver import (
    calculate_work_type_confusion_matrix,
    map_song_type_to_work_type,
    plot_work_type_confusion_matrix,
    resolve_work_type_from_metadata,
)


def find_latest_audit_file() -> Path | None:
    """Tìm file audit Excel mới nhất trong thư mục data/outputs."""
    patterns = [
        str(ROOT_DIR / "data" / "outputs" / "evaluation_audit_*.xlsx"),
        str(ROOT_DIR / "data" / "outputs" / "benchmark_audit_*.xlsx"),
    ]
    all_files: list[str] = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))

    valid_files = [
        f
        for f in set(all_files)
        if not os.path.basename(f).startswith("~$")
        and not os.path.basename(f).startswith("work_type_")
    ]
    if not valid_files:
        fallback_files = [
            f
            for f in glob.glob(str(ROOT_DIR / "data" / "outputs" / "*.xlsx"))
            if not os.path.basename(f).startswith("~$")
            and not os.path.basename(f).startswith("work_type_")
        ]
        if not fallback_files:
            return None
        valid_files = fallback_files

    valid_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return Path(valid_files[0])


def load_testcase_song_type_map(testcase_path: str | Path | None = None) -> dict[str, str]:
    """Tạo từ điển ánh xạ Ground Truth {query_key: song_type} từ file testcase."""
    candidates = []
    if testcase_path:
        candidates.append(Path(testcase_path))
    candidates.extend(
        [
            ROOT_DIR / "testcase3.xlsx",
            ROOT_DIR / "testcase2.xlsx",
            ROOT_DIR / "testcase.xlsx",
            ROOT_DIR / "data" / "testcase3.xlsx",
            ROOT_DIR / "data" / "testcase.xlsx",
        ]
    )

    found_path = None
    for cand in candidates:
        if cand.exists():
            found_path = cand
            break

    if not found_path:
        return {}

    wb = openpyxl.load_workbook(str(found_path), data_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return {}

    headers = [str(h).strip().lower() if h is not None else "" for h in rows[0]]
    title_idx = next(
        (
            i
            for i, h in enumerate(headers)
            if any(k in h for k in ["song title", "tên bài", "title"])
        ),
        0,
    )
    artist_idx = next(
        (i for i, h in enumerate(headers) if any(k in h for k in ["artist", "ca sĩ", "nghệ sĩ"])),
        1,
    )
    type_idx = next(
        (
            i
            for i, h in enumerate(headers)
            if any(k in h for k in ["song type", "loại bài", "type"])
        ),
        -1,
    )

    if type_idx == -1:
        return {}

    gt_map: dict[str, str] = {}
    for r in rows[1:]:
        title = str(r[title_idx] or "").strip()
        artist = str(r[artist_idx] or "").strip()
        st_val = str(r[type_idx] or "").strip()
        if not title:
            continue
        key1 = f"{title.lower()} {artist.lower()}".strip()
        key2 = title.lower().strip()
        mapped = map_song_type_to_work_type(st_val)
        gt_map[key1] = mapped
        gt_map[key2] = mapped

    return gt_map


def check_work_type_accuracy(
    audit_path: str | Path | None = None,
    testcase_path: str | Path | None = None,
    re_resolve: bool = True,
    export: bool = True,
) -> dict[str, Any]:
    """Thực hiện đối soát toàn diện độ chính xác phân biệt tác phẩm."""
    if audit_path is None:
        latest = find_latest_audit_file()
        if not latest:
            raise FileNotFoundError(
                "Không tìm thấy file audit nào trong thư mục 'data/outputs/'. Vui lòng chỉ định --audit."
            )
        target_audit = latest
    else:
        target_audit = Path(audit_path)

    if not target_audit.exists():
        raise FileNotFoundError(f"Không tìm thấy file audit: {target_audit}")

    print("=" * 78)
    print("      🔍 CÔNG CỤ ĐỐI SOÁT ĐỘ CHÍNH XÁC PHÂN BIỆT TÁC PHẨM")
    print("      (ORIGINAL vs COVER vs REMIX so với Ground Truth 'Song Type')")
    print("=" * 78)
    print(f"📁 File Audit đang đối soát : {target_audit.name}")

    df = pd.read_excel(str(target_audit), sheet_name=0)
    testcase_gt_map = load_testcase_song_type_map(testcase_path)
    if testcase_gt_map:
        print(f"📖 Đã nạp từ điển Ground Truth từ testcase ({len(testcase_gt_map)} mục)")

    title_col = next(
        (c for c in df.columns if any(k in str(c).lower() for k in ["song_title", "title", "tên"])),
        None,
    )
    artist_col = next(
        (c for c in df.columns if any(k in str(c).lower() for k in ["artist", "ca sĩ"])), None
    )
    gt_work_col = next(
        (
            c
            for c in df.columns
            if any(k in str(c).lower() for k in ["gt_work_type", "gt_work", "song_type"])
        ),
        None,
    )
    pred_work_col = next(
        (
            c
            for c in df.columns
            if any(
                k in str(c).lower() for k in ["pred_work_type", "predicted_work_type", "pred_work"]
            )
            or (
                "work_type" in str(c).lower()
                and not any(x in str(c).lower() for x in ["gt", "ground", "true"])
            )
        ),
        None,
    )

    audit_records: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    for idx, row in df.iterrows():
        title = str(row[title_col] or "").strip() if title_col else f"Song #{idx + 1}"
        artist = str(row[artist_col] or "").strip() if artist_col else ""
        query_key = f"{title.lower()} {artist.lower()}".strip()

        # 1. Xác định Ground Truth Work Type
        gt_work = "UNCERTAIN"
        if gt_work_col and pd.notna(row[gt_work_col]):
            gt_work = str(row[gt_work_col]).strip().upper()

        if gt_work in ("UNCERTAIN", "", "NAN"):
            gt_work = (
                testcase_gt_map.get(query_key) or testcase_gt_map.get(title.lower()) or "UNCERTAIN"
            )

        # 2. Xác định Predicted Work Type
        pred_work = "UNCERTAIN"
        if pred_work_col and pd.notna(row[pred_work_col]):
            pred_work = str(row[pred_work_col]).strip().upper()

        # Nếu mô hình trước đó trả về UNCERTAIN hoặc bật chế độ re_resolve -> Dùng Work Type Resolver tối ưu
        resolver_w_type, resolver_label, is_cov, orig_a, orig_t = resolve_work_type_from_metadata(
            title=title, artist=artist, metadata=dict(row)
        )

        effective_pred = pred_work
        if effective_pred in ("UNCERTAIN", "ERROR", "", "NAN") or re_resolve:
            effective_pred = resolver_w_type.value

        is_correct = (effective_pred == gt_work) if gt_work != "UNCERTAIN" else None

        rec = {
            "media_id": row.get("media_id", f"row_{idx + 1}"),
            "song_title": title,
            "artist": artist,
            "ground_truth_song_type": gt_work,
            "predicted_work_type": effective_pred,
            "work_is_correct": "ĐÚNG"
            if is_correct is True
            else ("SAI" if is_correct is False else "N/A"),
            "work_label": resolver_label,
            "is_cover": is_cov,
            "original_artist": orig_a or row.get("original_artist", "N/A"),
            "original_song_title": orig_t or row.get("original_song_title", "N/A"),
            "raw_audit_pred_work_type": pred_work,
        }
        audit_records.append(rec)

        if gt_work != "UNCERTAIN":
            y_true.append(gt_work)
            y_pred.append(effective_pred)

    metrics = calculate_work_type_confusion_matrix(y_true, y_pred)

    # In kết quả báo cáo
    print("\n" + "-" * 78)
    print("      📊 KẾT QUẢ ĐỐI SOÁT PHÂN BIỆT TÁC PHẨM (WORK TYPE)")
    print("-" * 78)
    print(f"• Tổng số bài hát kiểm tra : {len(audit_records)} bài")
    print(f"• Số bài có nhãn đối soát  : {metrics['total_evaluated']} bài")
    print(f"• Số bài nhận diện đúng    : {metrics['correct']} bài")
    print(f"⭐ ĐỘ CHÍNH XÁC (ACCURACY) : {metrics['accuracy'] * 100:.2f}%\n")

    print("🔢 MA TRẬN NHẦM LẪN 3x3 (CONFUSION MATRIX):")
    labels = metrics["labels"]
    matrix_dict = metrics["matrix"]
    header_str = f"{'Thực tế / Dự đoán':<22} | " + " | ".join(
        f"{lbl:<10}" for lbl in labels + ["UNCERTAIN"]
    )
    print(header_str)
    print("-" * len(header_str))
    for gt_lbl in labels:
        row_str = f"Thực tế: {gt_lbl:<12} | "
        vals = [f"{matrix_dict.get(gt_lbl, {}).get(p, 0):<10}" for p in labels + ["UNCERTAIN"]]
        row_str += " | ".join(vals)
        print(row_str)

    print("\n🎯 CHỈ SỐ THEO TỪNG NHÓM TÁC PHẨM:")
    for lbl in labels:
        c_met = metrics["per_class"].get(lbl, {})
        tot = c_met.get("total", 0)
        p = c_met.get("precision", 0.0) * 100
        r = c_met.get("recall", 0.0) * 100
        f1 = c_met.get("f1_score", 0.0) * 100
        print(
            f"   🏷️ {lbl:<8}: Mẫu={tot:<2} | Precision={p:>5.1f}% | Recall={r:>5.1f}% | F1-Score={f1:>5.1f}%"
        )

    # Liệt kê các ca sai
    wrong_cases = [r for r in audit_records if r["work_is_correct"] == "SAI"]
    if wrong_cases:
        print("\n" + "!" * 78)
        print(f"❌ DANH SÁCH {len(wrong_cases)} CA NHẬN DIỆN SAI TÁC PHẨM CẦN LƯU Ý:")
        print("!" * 78)
        for w_idx, w in enumerate(wrong_cases, 1):
            print(f" [{w_idx:02d}] {w['song_title']} - {w['artist']}")
            print(
                f"      -> Nhãn thực tế (GT): {w['ground_truth_song_type']} | Hệ thống đoán: {w['predicted_work_type']}"
            )
            print(f"      -> Chi tiết nhãn     : {w['work_label']}")
    else:
        print("\n🎉 XUẤT SẮC: 100% các bài hát có nhãn đều được phân biệt đúng tác phẩm!")

    # Xuất file báo cáo Excel và ảnh ma trận
    if export:
        out_dir = ROOT_DIR / "data" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Vẽ và lưu ảnh
        img_path = out_dir / f"work_type_cm_{ts}.png"
        plot_work_type_confusion_matrix(metrics, output_path=img_path)
        print(f"\n🖼️  Ảnh ma trận nhầm lẫn 3x3 đã lưu tại: {img_path}")

        # 2. Xuất file Excel
        excel_path = out_dir / f"work_type_audit_{ts}.xlsx"
        with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
            df_det = pd.DataFrame(audit_records)
            df_det.to_excel(writer, sheet_name="Work Type Audit Details", index=False)

            summary_rows = [
                {"Chỉ số": "Tổng số bài kiểm tra", "Giá trị": len(audit_records)},
                {"Chỉ số": "Số bài có nhãn đối soát", "Giá trị": metrics["total_evaluated"]},
                {"Chỉ số": "Số bài đoán đúng tác phẩm", "Giá trị": metrics["correct"]},
                {
                    "Chỉ số": "Độ chính xác tác phẩm (Work Type Accuracy)",
                    "Giá trị": f"{metrics['accuracy'] * 100:.2f}%",
                },
            ]
            for lbl in labels:
                c = metrics["per_class"].get(lbl, {})
                summary_rows.append(
                    {
                        "Chỉ số": f"Nhãn {lbl} (Precision / Recall / F1)",
                        "Giá trị": f"{c['precision'] * 100:.1f}% / {c['recall'] * 100:.1f}% / {c['f1_score'] * 100:.1f}%",
                    }
                )
            df_sum = pd.DataFrame(summary_rows)
            df_sum.to_excel(writer, sheet_name="Executive Summary", index=False)

        print(f"📁 Báo cáo Excel chi tiết đã lưu tại    : {excel_path}\n")

    return {
        "metrics": metrics,
        "records": audit_records,
        "wrong_cases": wrong_cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra và đối soát độ chính xác phân biệt Original / Cover / Remix"
    )
    parser.add_argument(
        "--audit",
        type=str,
        default=None,
        help="Đường dẫn file audit excel (mặc định: tự tìm file evaluation_audit mới nhất)",
    )
    parser.add_argument(
        "--testcase",
        type=str,
        default=None,
        help="Đường dẫn file testcase chứa cột Song Type (testcase3.xlsx, testcase2.xlsx)",
    )
    parser.add_argument(
        "--no-re-resolve",
        action="store_true",
        help="Không sử dụng bộ giải mã Heuristic cải tiến, giữ nguyên giá trị pred_work_type từ file audit",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Không xuất file Excel và ảnh PNG",
    )

    args = parser.parse_args()
    check_work_type_accuracy(
        audit_path=args.audit,
        testcase_path=args.testcase,
        re_resolve=not args.no_re_resolve,
        export=not args.no_export,
    )


if __name__ == "__main__":
    main()
