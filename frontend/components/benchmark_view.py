import os
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.core.config import BASE_DIR, TEMP_DIR
from backend.core.container import get_container
from benchmark_evaluator import (
    export_audit_excel,
    load_testcase_dataset,
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

            # Tiến trình chạy
            progress_bar = st.progress(0.0)
            status_box = st.empty()

            container = get_container()
            detection_service = container.detection_service
            resolver = container.resolver

            results = []
            tp, tn, fp, fn, errors = 0, 0, 0, 0, 0

            for i, item in enumerate(samples, 1):
                pct = i / len(samples)
                progress_bar.progress(pct)
                status_box.text(
                    f"[{i}/{len(samples)}] Đang thẩm định: '{item['query']}' (Ground Truth: {item['gt_label']})..."
                )

                track = None
                try:
                    track = resolver.resolve_track(item["query"])
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

                    ai_model_val = res.ai_model_verdict or (
                        "AI" if res.ai_model_is_ai else "HUMAN"
                    )
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
                            "Đánh giá Model AI": ai_model_val,
                            "Kết luận Tổng hợp": consensus_val,
                            "Dự đoán": pred_label,
                            "Kết quả": "✅ ĐÚNG" if is_correct else "❌ SAI",
                            "Phân loại": m_type,
                            "Nhãn chi tiết": res.verdict_label,
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
                            "Độ tin cậy": "0%",
                            "Nghi vấn nền tảng": "N/A",
                            "Luận cứ": str(ex),
                        }
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
                except Exception as db_save_err:
                    st.warning(f"Không thể lưu đợt benchmark vào Cloud DB: {db_save_err}")

        except Exception as e:
            st.error(f"Lỗi trong quá trình thẩm định benchmark: {e}")

    # Hiển thị kết quả đánh giá nếu có
    if "benchmark_summary" in st.session_state and st.session_state["benchmark_summary"]:
        bm = st.session_state["benchmark_summary"]
        res_list = st.session_state["benchmark_results"]

        st.divider()
        st.subheader("📊 Kết Quả Chấm Điểm Mô Hình (Evaluation Metrics)")

        # Thẻ chỉ số chính
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⭐ Accuracy (Độ chính xác)", f"{bm['accuracy'] * 100:.1f}%")
        m2.metric("🎯 Precision (Tránh bắt oan)", f"{bm['precision'] * 100:.1f}%")
        m3.metric("📡 Recall (Độ nhạy bắt AI)", f"{bm['recall'] * 100:.1f}%")
        m4.metric("⚖️ F1-Score (Cân bằng)", f"{bm['f1_score'] * 100:.1f}%")

        # Ma trận nhầm lẫn
        st.markdown("#### 🔢 Ma Trận Nhầm Lẫn (Confusion Matrix)")
        c_tp, c_tn, c_fp, c_fn = st.columns(4)
        c_tp.success(f"**True Positive (TP)**: {bm['TP']} bài\n*(Thực tế AI ➡️ Đoán đúng AI)*")
        c_tn.info(f"**True Negative (TN)**: {bm['TN']} bài\n*(Thực tế Người ➡️ Đoán đúng Người)*")
        c_fp.error(f"**False Positive (FP)**: {bm['FP']} bài\n*(Báo nhầm nhạc người là AI ⚠️)*")
        c_fn.warning(f"**False Negative (FN)**: {bm['FN']} bài\n*(Bỏ sót / lọt lưới nhạc AI ⚠️)*")

        # Bảng chi tiết
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
