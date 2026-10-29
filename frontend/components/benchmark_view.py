import os
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.core.config import BASE_DIR, TEMP_DIR
from backend.core.container import get_container
from benchmark_evaluator import (
    calculate_work_type_confusion_matrix,
    export_audit_excel,
    load_testcase_dataset,
    plot_work_type_confusion_matrix,
    select_samples,
)
from frontend.components.ui_compat import get_width_kwargs


def render_benchmark_view():
    """Hiển thị giao diện Chấm điểm & Đánh giá Độ chính xác (Benchmark Mode)."""
    st.markdown("### 🎯 Chấm Điểm Độ Chính Xác (Accuracy & Benchmark)")
    st.write(
        "Sử dụng tập dữ liệu kiểm thử thực tế có gắn nhãn Ground Truth (như file `testcase.xlsx` với cột `AI Yes`) "
        "để tự động đối soát, tính toán độ chuẩn xác (Accuracy, Precision, Recall, F1-Score) và ma trận nhầm lẫn (Confusion Matrix)."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        available_files = []
        for cand in [
            "data/testcase.xlsx",
            "data/testcase2.xlsx",
            "data/testcase3.xlsx",
            "testcase.xlsx",
        ]:
            cand_path = BASE_DIR / cand
            if cand_path.exists():
                available_files.append((cand, str(cand_path)))

        options_map = {f"Dùng {name}": path for name, path in available_files}
        options_map["Tải lên file Excel khác"] = "upload"

        file_source = st.radio("Nguồn file test sample:", list(options_map.keys()), horizontal=True)

        if file_source == "Tải lên file Excel khác":
            uploaded = st.file_uploader("Chọn file Excel testcase (.xlsx):", type=["xlsx", "xls"])
            if uploaded:
                save_path = TEMP_DIR / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                selected_file_path = str(save_path)
            else:
                selected_file_path = None
        else:
            selected_file_path = options_map[file_source]

    with col2:
        sheet_name = st.text_input("Tên Sheet chứa Ground Truth:", value="Media_W38")

    if not selected_file_path or not os.path.exists(selected_file_path):
        st.info("💡 Vui lòng đảm bảo file testcase đã sẵn sàng.")
        return

    st.divider()

    # Cấu hình lấy mẫu
    c_sample, c_balance, c_seed = st.columns([2, 2, 1])
    with c_sample:
        sample_mode = st.radio(
            "Chế độ kiểm thử:",
            ["Lấy mẫu nhanh (Stratified Sample)", "Toàn bộ danh sách (Full)"],
            horizontal=True,
        )
        if sample_mode == "Lấy mẫu nhanh (Stratified Sample)":
            sample_count = st.slider(
                "Số lượng bài hát muốn thẩm định:", min_value=4, max_value=100, value=10, step=2
            )
        else:
            sample_count = None
            st.warning(
                "⚠️ Chạy toàn bộ (>1,600 bài) sẽ mất vài tiếng và tiêu tốn nhiều request YouTube/Gemini!"
            )

    with c_balance:
        st.write("")
        st.write("")
        balanced = st.checkbox("Cân bằng phân tầng 50% AI - 50% Người (Khuyên dùng)", value=True)

    with c_seed:
        seed_val = st.number_input("Random Seed:", value=42, min_value=1, max_value=9999)

    # Nút kích hoạt
    if st.button("🚀 Bắt đầu Chấm Điểm Dự Án", type="primary", **get_width_kwargs(st.button)):
        # Reset kết quả cũ để tập trung theo dõi tiến trình thời gian thực
        st.session_state["benchmark_summary"] = None
        st.session_state["benchmark_results"] = None

        try:
            with st.spinner("Đang nạp dữ liệu từ file testcase..."):
                dataset = load_testcase_dataset(file_path=selected_file_path, sheet_name=sheet_name)

            if sample_mode == "Lấy mẫu nhanh (Stratified Sample)":
                samples = select_samples(
                    dataset, sample_size=sample_count, balanced=balanced, seed=seed_val
                )
            else:
                samples = dataset

            s_ai = sum(1 for s in samples if s["gt_is_ai"])
            s_human = sum(1 for s in samples if not s["gt_is_ai"])
            st.success(
                f"Đã chọn {len(samples)} bài để thẩm định đối soát (AI: {s_ai} | Con người: {s_human})"
            )

            # Tiến trình chạy & Placeholders thời gian thực
            progress_bar = st.progress(0.0)
            status_box = st.empty()
            live_metrics_placeholder = st.empty()
            live_table_header = st.empty()
            live_table_placeholder = st.empty()

            container = get_container()
            detection_service = container.detection_service
            resolver = container.resolver

            results = []
            y_true_work = []
            y_pred_work = []
            tp, tn, fp, fn, errors = 0, 0, 0, 0, 0

            for i, item in enumerate(samples, 1):
                pct = i / len(samples)
                progress_bar.progress(pct)
                status_box.text(
                    f"[{i}/{len(samples)}] Đang thẩm định: '{item['query']}' (Ground Truth: {item['gt_label']})..."
                )

                track = None
                try:
                    track = resolver.resolve_track(item["title"], artist=item.get("artist", ""))
                    track.song_type = item.get("song_type") or ""
                    track.release_date = str(item.get("release_date") or "")
                    track.distributor = str(item.get("distributor") or "")
                    res = detection_service.detect_single(
                        track, model_override=st.session_state.get("model_name")
                    )

                    pred_is_ai = res.is_ai
                    pred_label = "AI" if pred_is_ai else "HUMAN"
                    is_correct = pred_is_ai == item["gt_is_ai"]

                    if item["gt_is_ai"] and pred_is_ai:
                        m_type = "TP (Đoán đúng AI)"
                        tp += 1
                    elif not item["gt_is_ai"] and not pred_is_ai:
                        m_type = "TN (Đoán đúng Người)"
                        tn += 1
                    elif not item["gt_is_ai"] and pred_is_ai:
                        m_type = "FP (Vu oan Người)"
                        fp += 1
                    else:
                        m_type = "FN (Bỏ sót AI)"
                        fn += 1

                    # Đối soát Work Type (Original vs Cover vs Remix)
                    gt_work = item.get("gt_work_type", "UNCERTAIN")
                    raw_w_type = getattr(res, "work_type", None)
                    pred_work = (
                        raw_w_type.value
                        if hasattr(raw_w_type, "value")
                        else (str(raw_w_type) if raw_w_type else "UNCERTAIN")
                    )
                    work_is_correct = (pred_work == gt_work) if gt_work != "UNCERTAIN" else None
                    if gt_work != "UNCERTAIN":
                        y_true_work.append(gt_work)
                        y_pred_work.append(pred_work)

                    ai_model_is_ai = getattr(res, "ai_model_is_ai", res.is_ai)
                    ai_model_pred = "AI" if ai_model_is_ai else "HUMAN"
                    ai_model_val = res.ai_model_verdict or ai_model_pred
                    consensus_val = res.consensus_verdict or pred_label
                    results.append(
                        {
                            "Mã bài": item["media_id"],
                            "Tên bài hát": item["title"],
                            "Ca sĩ": item["artist"],
                            "Nguồn link": res.source_url
                            or getattr(track, "resolved_url", "N/A")
                            or "N/A",
                            "Thực tế": item["gt_label"],
                            "Model AI Dự đoán": ai_model_pred,
                            "Đánh giá Model AI": ai_model_val,
                            "Kết luận Tổng hợp": consensus_val,
                            "Dự đoán": pred_label,
                            "Kết quả": "✅ ĐÚNG" if is_correct else "❌ SAI",
                            "Phân loại": m_type,
                            "Nhãn chi tiết": res.verdict_label,
                            "Thực tế Tác phẩm": gt_work,
                            "Dự đoán Tác phẩm": pred_work,
                            "Kết quả Tác phẩm": (
                                "✅ ĐÚNG"
                                if work_is_correct is True
                                else ("❌ SAI" if work_is_correct is False else "N/A")
                            ),
                            "Tác phẩm": getattr(res, "work_label", "Chưa rõ tác phẩm"),
                            "Nghệ sĩ bài gốc": getattr(res, "original_artist", "") or "N/A",
                            "Tên bài gốc": getattr(res, "original_song_title", "") or "N/A",
                            "Độ tin cậy": f"{res.confidence_score * 100:.1f}%",
                            "Nghi vấn nền tảng": res.suspected_engine,
                            "Luận cứ": " | ".join(res.reasons or [])[:120],
                        }
                    )
                except Exception as ex:
                    errors += 1
                    results.append(
                        {
                            "Mã bài": item["media_id"],
                            "Tên bài hát": item["title"],
                            "Ca sĩ": item["artist"],
                            "Nguồn link": getattr(track, "resolved_url", "N/A") if track else "N/A",
                            "Thực tế": item["gt_label"],
                            "Đánh giá Model AI": "ERROR",
                            "Kết luận Tổng hợp": "ERROR",
                            "Dự đoán": "ERROR",
                            "Kết quả": "❌ LỖI",
                            "Phân loại": "ERROR",
                            "Nhãn chi tiết": "Lỗi kỹ thuật",
                            "Thực tế Tác phẩm": item.get("gt_work_type", "UNCERTAIN"),
                            "Dự đoán Tác phẩm": "ERROR",
                            "Kết quả Tác phẩm": "❌ LỖI",
                            "Tác phẩm": "LỖI HỆ THỐNG",
                            "Nghệ sĩ bài gốc": "N/A",
                            "Tên bài gốc": "N/A",
                            "Độ tin cậy": "0%",
                            "Nghi vấn nền tảng": "N/A",
                            "Luận cứ": str(ex),
                        }
                    )

                # Cập nhật số liệu đếm trực tiếp
                with live_metrics_placeholder.container():
                    bm1, bm2, bm3, bm4 = st.columns(4)
                    bm1.metric("Tiến độ", f"{i}/{len(samples)}")
                    bm2.metric("Đúng (TP + TN)", tp + tn)
                    bm3.metric("Sai (FP + FN)", fp + fn)
                    eval_so_far = tp + tn + fp + fn
                    curr_acc = (
                        f"{(tp + tn) / eval_so_far * 100:.1f}%" if eval_so_far > 0 else "0.0%"
                    )
                    bm4.metric("Accuracy tạm tính", curr_acc)

                # Cập nhật Data Table trực tiếp theo thời gian thực (Incremental Stream Table)
                live_table_header.markdown(
                    f"##### 📋 Bảng Đối Soát Trực Tiếp ({len(results)}/{len(samples)} bài)"
                )
                live_table_placeholder.dataframe(
                    pd.DataFrame(results), **get_width_kwargs(st.dataframe)
                )

            progress_bar.progress(1.0)
            status_box.text("🎉 Đã hoàn thành đối soát toàn bộ tập mẫu!")

            # Tính toán metrics
            evaluated_total = tp + tn + fp + fn
            acc = ((tp + tn) / evaluated_total) if evaluated_total > 0 else 0.0
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            # Tính toán ma trận nhầm lẫn Work Type
            work_type_metrics = calculate_work_type_confusion_matrix(y_true_work, y_pred_work)
            work_type_cm_path: str | None = None
            if work_type_metrics.get("total_evaluated", 0) > 0:
                try:
                    cm_target = TEMP_DIR / f"work_type_cm_{seed_val}.png"
                    plot_work_type_confusion_matrix(work_type_metrics, output_path=cm_target)
                    work_type_cm_path = str(cm_target)
                except Exception as plot_err:  # noqa: BLE001
                    st.warning(f"Không thể tạo ảnh ma trận nhầm lẫn tác phẩm: {plot_err}")

            benchmark_summary = {
                "total_samples": len(samples),
                "evaluated_successfully": evaluated_total,
                "errors": errors,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1_score": f1,
                "work_type_metrics": work_type_metrics,
                "work_type_cm_image_path": work_type_cm_path,
            }

            st.session_state["benchmark_summary"] = benchmark_summary
            st.session_state["benchmark_results"] = results

            # Xuất file Excel
            audit_file = export_audit_excel(benchmark_summary, results)
            st.session_state["benchmark_file"] = audit_file

            # Tự động đồng bộ lịch sử đợt chấm điểm lên CSDL Supabase Cloud
            if hasattr(container.track_repository, "save_benchmark_run"):
                try:
                    run_id = container.track_repository.save_benchmark_run(
                        benchmark_summary, results
                    )
                    st.toast(f"☁️ Đã lưu đợt Benchmark '{run_id}' lên Supabase Cloud!", icon="✅")
                except Exception as db_save_err:  # noqa: BLE001
                    st.warning(f"Không thể lưu đợt benchmark vào Cloud DB: {db_save_err}")

            st.toast("🎉 Đã hoàn thành đối soát toàn bộ tập mẫu!", icon="✅")
            st.rerun()

        except Exception as e:  # noqa: BLE001
            st.error(f"Lỗi trong quá trình thẩm định benchmark: {e}")

    # Hiển thị kết quả đánh giá nếu có
    if "benchmark_summary" in st.session_state and st.session_state["benchmark_summary"]:
        bm = st.session_state["benchmark_summary"]
        res_list = st.session_state["benchmark_results"]

        st.divider()
        st.subheader("📊 Kết Quả Chấm Điểm Mô Hình (Evaluation Metrics)")

        # Thẻ chỉ số chính AI vs Human
        st.markdown("#### 🤖 1. Hiệu Năng Phân Biệt Nhạc AI vs Nhạc Con Người")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⭐ AI Accuracy", f"{bm['accuracy'] * 100:.1f}%")
        m2.metric("🎯 Precision (Tránh bắt oan)", f"{bm['precision'] * 100:.1f}%")
        m3.metric("📡 Recall (Độ nhạy bắt AI)", f"{bm['recall'] * 100:.1f}%")
        m4.metric("⚖️ F1-Score (Cân bằng)", f"{bm['f1_score'] * 100:.1f}%")

        # Ma trận nhầm lẫn AI/Human
        c_tp, c_tn, c_fp, c_fn = st.columns(4)
        c_tp.success(f"**True Positive (TP)**: {bm['TP']} bài\n*(Thực tế AI ➡️ Đoán đúng AI)*")
        c_tn.info(f"**True Negative (TN)**: {bm['TN']} bài\n*(Thực tế Người ➡️ Đoán đúng Người)*")
        c_fp.error(f"**False Positive (FP)**: {bm['FP']} bài\n*(Báo nhầm nhạc người là AI ⚠️)*")
        c_fn.warning(f"**False Negative (FN)**: {bm['FN']} bài\n*(Bỏ sót / lọt lưới nhạc AI ⚠️)*")

        # Thẻ chỉ số Work Type (Original vs Cover vs Remix)
        wt = bm.get("work_type_metrics")
        if wt and wt.get("total_evaluated", 0) > 0:
            st.divider()
            st.markdown("#### 💿 2. Hiệu Năng Phân Biệt Tác Phẩm (Original vs Cover vs Remix)")
            cw1, cw2, cw3, cw4 = st.columns(4)
            cw1.metric(
                "⭐ Work Accuracy",
                f"{wt['accuracy'] * 100:.1f}%",
                f"{wt['correct']}/{wt['total_evaluated']} bài",
            )
            p_class = wt.get("per_class", {})
            cov_stat = p_class.get("COVER", {})
            orig_stat = p_class.get("ORIGINAL", {})
            rem_stat = p_class.get("REMIX", {})

            cw2.metric(
                "🎙️ Cover Recall",
                f"{cov_stat.get('recall', 0.0) * 100:.1f}%",
                f"Prec: {cov_stat.get('precision', 0.0) * 100:.1f}%",
            )
            cw3.metric(
                "💿 Original Recall",
                f"{orig_stat.get('recall', 0.0) * 100:.1f}%",
                f"Prec: {orig_stat.get('precision', 0.0) * 100:.1f}%",
            )
            cw4.metric(
                "🎛️ Remix Recall",
                f"{rem_stat.get('recall', 0.0) * 100:.1f}%",
                f"Prec: {rem_stat.get('precision', 0.0) * 100:.1f}%",
            )

            cm_img = bm.get("work_type_cm_image_path")
            if cm_img and Path(cm_img).exists():
                st.markdown("##### 🖼️ Ma Trận Nhầm Lẫn Tác Phẩm (Multi-class Confusion Matrix)")
                st.image(
                    cm_img,
                    caption="Ma Trận Nhầm Lẫn Đối Soát Tác Phẩm (Original vs Cover vs Remix) - 300 DPI",
                    **get_width_kwargs(st.image),
                )

        # Bảng chi tiết
        st.divider()
        st.markdown("#### 📋 Bảng Đối Soát Chi Tiết Từng Bài Hát")
        df_res = pd.DataFrame(res_list)
        st.dataframe(df_res, **get_width_kwargs(st.dataframe))

        # Nút tải file
        audit_path = st.session_state.get("benchmark_file")
        if audit_path and Path(audit_path).exists():
            with open(audit_path, "rb") as f:
                st.download_button(
                    label="📥 Tải xuống File Báo Cáo Đối Soát (.xlsx)",
                    data=f.read(),
                    file_name=Path(audit_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    **get_width_kwargs(st.download_button),
                )

    # Hiển thị lịch sử các lần chạy Benchmark từ Cloud DB (nếu có)
    container = get_container()
    if hasattr(container.track_repository, "list_benchmark_runs"):
        past_runs = container.track_repository.list_benchmark_runs(limit=10)
        if past_runs:
            st.divider()
            with st.expander(
                "📜 Xem Lịch Sử Các Đợt Chấm Điểm Trên Cloud (Supabase)", expanded=False
            ):
                df_history = pd.DataFrame(past_runs)
                if not df_history.empty:
                    cols_to_show = [
                        "id",
                        "total_samples",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1_score",
                        "created_at",
                    ]
                    existing_cols = [c for c in cols_to_show if c in df_history.columns]
                    st.dataframe(
                        df_history[existing_cols].rename(
                            columns={
                                "id": "Mã đợt test",
                                "total_samples": "Số mẫu",
                                "accuracy": "Accuracy",
                                "precision": "Precision",
                                "recall": "Recall",
                                "f1_score": "F1-Score",
                                "created_at": "Thời gian chạy",
                            }
                        ),
                        **get_width_kwargs(st.dataframe),
                    )
