import re
import time
import urllib.parse
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend.core.config import TEMP_DIR
from frontend.components.ui_compat import get_width_kwargs
from frontend.services.backend_client import BackendClient


def render_spotify_view(client: BackendClient):
    """
    Giao diện Tra cứu bài hát trên Spotify đa năng:
    - Chế độ 1: Tra cứu đơn bài (Chi tiết top 5 kết quả + iFrame Embed Player)
    - Chế độ 2: Tra cứu hàng loạt (Nhập Text hoặc File .xlsx/.csv/.txt -> Đa luồng 3-5 workers -> Bảng đối soát 1-1 -> Tải Excel/CSV -> Đưa sang Quét AI Hàng loạt)
    """
    st.markdown("### 🎧 Tra Cứu & Khám Phá Bài Hát Trên Spotify")
    st.caption(
        "Tìm kiếm các bài hát chính thức trên Spotify, trích xuất siêu dữ liệu phòng thu (Album, Ngày phát hành, ISRC), "
        "nghe thử bản thu và chuyển nhanh sang luồng Thẩm định AI."
    )

    mode = st.radio(
        "Chế độ tra cứu:",
        ["🔍 Tra cứu đơn bài", "📁 Tra cứu hàng loạt (Batch Search)"],
        horizontal=True,
        key="spotify_search_mode_toggle",
    )

    st.divider()

    # =========================================================================
    # CHẾ ĐỘ 1: TRA CỨU ĐƠN BÀI (SINGLE SEARCH)
    # =========================================================================
    if mode == "🔍 Tra cứu đơn bài":
        col_search, col_limit, col_btn = st.columns([5, 2, 2])
        with col_search:
            search_query = st.text_input(
                "Tên bài hát hoặc ca sĩ cần tìm:",
                placeholder="VD: Sơn Tùng M-TP Đừng Làm Trái Tim Anh Đau, hoặc Blinding Lights...",
                key="spotify_tab_query",
            )
        with col_limit:
            limit = st.selectbox(
                "Số lượng kết quả:",
                options=[5, 10, 15],
                index=0,
                key="spotify_tab_limit",
            )
        with col_btn:
            st.write("")
            st.write("")
            search_clicked = st.button(
                "🔍 Tìm kiếm",
                type="primary",
                key="btn_spotify_search",
                **get_width_kwargs(st.button),
            )

        if search_clicked and search_query.strip():
            with st.spinner(f"Đang tìm kiếm '{search_query.strip()}' trên Spotify..."):
                try:
                    results = client.search_spotify_tracks(
                        query=search_query.strip(), limit=int(limit)
                    )
                    st.session_state.spotify_search_results = results
                    if not results:
                        st.info("ℹ️ Không tìm thấy bài hát nào phù hợp trên Spotify.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi tìm kiếm Spotify: {e}")

        results = st.session_state.get("spotify_search_results") or []
        if results:
            if any(it.get("id") == "search_fallback" for it in results):
                st.info(
                    "💡 **Lưu ý:** Tài khoản Spotify tạo App hiện là gói **Miễn phí (Free)**. "
                    "Spotify yêu cầu tài khoản chủ App phải có gói **Premium** để cấp quyền truy xuất API danh mục bài hát. "
                    "Hệ thống đã **tạo sẵn đường link tìm kiếm Spotify chính xác 100%** bên dưới để bạn nhấp mở nghe trực tiếp!"
                )

            st.markdown(f"#### 🎵 Kết quả tìm kiếm ({len(results)} bài hát):")

            for idx, item in enumerate(results):
                track_title = item.get("title") or "N/A"
                artist = item.get("artist") or "Chưa rõ nghệ sĩ"
                album = item.get("album") or ""
                release_date = item.get("release_date") or ""
                cover_url = item.get("cover_url")
                spotify_url = item.get("spotify_url") or ""
                embed_url = item.get("embed_url")
                popularity = item.get("popularity", 0)

                dur_sec = item.get("duration_sec", 0.0)
                dur_text = (
                    f"{int(dur_sec // 60)}m {int(dur_sec % 60):02d}s" if dur_sec > 0 else "-"
                )

                with st.container():
                    col_art, col_info, col_actions = st.columns([1.5, 5, 3.5])

                    with col_art:
                        if cover_url:
                            st.image(cover_url, use_container_width=True)
                        else:
                            st.markdown(
                                """
                                <div style="background: rgba(255,255,255,0.05); height: 110px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                                    🎵
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with col_info:
                        st.markdown(f"##### **{idx + 1}. {track_title}**")
                        meta_badges = [f"👤 **{artist}**"]
                        if album:
                            meta_badges.append(f"💿 Album: {album}")
                        if release_date:
                            meta_badges.append(f"📅 {release_date}")
                        if dur_text != "-":
                            meta_badges.append(f"⏱️ {dur_text}")
                        if popularity > 0:
                            meta_badges.append(f"🔥 Độ phổ biến: {popularity}%")

                        st.markdown(" • ".join(meta_badges))
                        if spotify_url:
                            st.code(spotify_url, language=None)

                    with col_actions:
                        st.write("")
                        col_btn1, col_btn2 = st.columns([1, 1.3])
                        with col_btn1:
                            if spotify_url:
                                st.link_button(
                                    "🟢 Mở link",
                                    spotify_url,
                                    use_container_width=True,
                                )
                        with col_btn2:
                            detect_btn = st.button(
                                "🚀 Giám định AI",
                                key=f"detect_spot_{idx}_{item.get('id', idx)}",
                                type="primary",
                                use_container_width=True,
                                help="Chuyển bài hát này sang Tab Thẩm định để phân tích bằng AI",
                            )
                            if detect_btn:
                                target_q = spotify_url or f"{track_title} {artist}".strip()
                                st.session_state["single_query_input"] = target_q
                                with st.spinner(
                                    f"Đang thẩm định bài '{track_title}' bằng AI..."
                                ):
                                    try:
                                        req_lang = st.session_state.get("reason_language", "vi")
                                        res = client.detect_single_track(
                                            query=target_q,
                                            model_override=st.session_state.model_name,
                                            language=req_lang,
                                        )
                                        st.session_state.single_result = res
                                        st.toast(
                                            f"✅ Đã giám định xong: {track_title}!", icon="🎉"
                                        )
                                        st.info(
                                            f"🎉 **Đã hoàn thành giám định:** Phán quyết `{res.verdict_label}` "
                                            f"(Độ tin cậy: {res.confidence_score*100:.1f}%). "
                                            f"Xem chi tiết đầy đủ tại Tab **'🔍 Thẩm định Đơn bài'**!"
                                        )
                                    except Exception as err:
                                        st.error(f"❌ Lỗi khi thẩm định: {err}")

                    if embed_url:
                        components.iframe(
                            f"{embed_url}?utm_source=generator&theme=0",
                            height=80,
                            scrolling=False,
                        )
                    st.markdown("---")

    # =========================================================================
    # CHẾ ĐỘ 2: TRA CỨU HÀNG LOẠT (BATCH SPOTIFY SEARCH)
    # =========================================================================
    else:
        st.markdown("#### 📁 Tra Cứu Danh Sách Bài Hát Hàng Loạt")
        st.caption(
            "Tra cứu đồng thời hàng chục bài hát trên Spotify bằng Multi-threading Pool. "
            "Hỗ trợ nhập danh sách text hoặc tải tệp Excel/CSV để đối soát dữ liệu 1-1."
        )

        batch_input_method = st.radio(
            "Phương thức nhập dữ liệu:",
            ["📝 Nhập trực tiếp danh sách text", "📁 Tải lên tệp (.xlsx, .csv, .txt)"],
            horizontal=True,
            key="spotify_batch_input_method",
        )

        parsed_items: list[dict] = []
        parsed_orig_df: pd.DataFrame | None = None

        if batch_input_method == "📝 Nhập trực tiếp danh sách text":
            text_input = st.text_area(
                "Nhập danh sách bài hát (mỗi dòng 1 bài hoặc 'Tên bài - Ca sĩ'):",
                placeholder=(
                    "Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau\n"
                    "Chìm Sâu - MCK\n"
                    "Tại Vì Sao - RHYDER\n"
                    "Blinding Lights - The Weeknd\n"
                    "Shape of You - Ed Sheeran"
                ),
                height=160,
                key="spotify_batch_textarea",
            )
            if text_input.strip():
                parsed_items, parsed_orig_df = client.parse_spotify_batch_input(text_input.strip())
                st.caption(f"📋 Đã nhận diện được **{len(parsed_items)}** bài hát từ danh sách text.")
        else:
            uploaded_file = st.file_uploader(
                "Tải lên file danh sách bài hát (.xlsx, .csv, .txt):",
                type=["xlsx", "xls", "csv", "txt"],
                key="spotify_batch_file_uploader",
            )
            if uploaded_file:
                save_path = TEMP_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                parsed_items, parsed_orig_df = client.parse_spotify_batch_input(save_path)
                st.success(
                    f"✅ Đã tải file: **{uploaded_file.name}** (Nhận diện được **{len(parsed_items)}** bài hát)."
                )
                if parsed_orig_df is not None and len(parsed_orig_df) > 0:
                    with st.expander("👀 Xem trước dữ liệu file gốc (3 dòng đầu):"):
                        st.dataframe(parsed_orig_df.head(3), use_container_width=True)

        col_cfg1, col_cfg2 = st.columns([3, 1])
        with col_cfg1:
            workers_count = st.slider(
                "⚡ Tốc độ xử lý đa luồng (Số workers song song):",
                min_value=1,
                max_value=8,
                value=4,
                help="3-5 luồng là tối ưu nhất cho Spotify API để tra cứu cực nhanh mà không bị chạm trần Rate Limit.",
                key="spotify_batch_workers",
            )
        with col_cfg2:
            st.write("")
            st.write("")
            start_batch_btn = st.button(
                "🚀 Bắt đầu Tra cứu Hàng loạt",
                type="primary",
                disabled=len(parsed_items) == 0,
                use_container_width=True,
                key="btn_start_spotify_batch",
            )

        # Xử lý khi nhấn nút Tra cứu Hàng loạt
        if start_batch_btn and parsed_items:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            start_time = time.time()

            def _on_batch_progress(completed: int, total: int, item: dict):
                pct = completed / max(1, total)
                progress_bar.progress(pct)
                status_text.caption(
                    f"⏳ Đang tra cứu {completed}/{total}: **{item.get('input_query')}**..."
                )

            with st.spinner(f"Đang tra cứu đồng thời {len(parsed_items)} bài trên Spotify..."):
                try:
                    batch_res = client.batch_search_spotify_tracks(
                        queries=parsed_items,
                        max_workers=workers_count,
                        on_progress=_on_batch_progress,
                    )
                    elapsed_time = round(time.time() - start_time, 2)
                    st.session_state.spotify_batch_results = batch_res
                    st.session_state.spotify_batch_orig_df = parsed_orig_df
                    st.session_state.spotify_batch_elapsed = elapsed_time

                    progress_bar.progress(1.0)
                    status_text.empty()
                    st.toast(
                        f"🎉 Hoàn tất tra cứu {len(batch_res)} bài trong {elapsed_time}s!",
                        icon="✅",
                    )
                except Exception as err:
                    st.error(f"❌ Lỗi khi tra cứu hàng loạt: {err}")

        # Hiển thị kết quả tra cứu hàng loạt đã lưu trong Session State
        batch_results = st.session_state.get("spotify_batch_results") or []
        orig_df = st.session_state.get("spotify_batch_orig_df")
        elapsed = st.session_state.get("spotify_batch_elapsed", 0.0)

        if batch_results:
            st.divider()
            found_count = sum(1 for r in batch_results if r.get("is_found"))
            search_link_count = sum(
                1 for r in batch_results if not r.get("is_found") and r.get("spotify_url")
            )
            not_found_count = len(batch_results) - found_count - search_link_count

            if search_link_count > 0 and found_count == 0:
                st.info(
                    "💡 **Lưu ý về tài khoản Spotify:** Tài khoản Spotify tạo App hiện tại là tài khoản **Miễn phí (Free)**. "
                    "Theo chính sách mới của Spotify, máy chủ Spotify yêu cầu tài khoản chủ App phải có gói **Spotify Premium** "
                    "mới cấp quyền truy xuất Web API phòng thu (ảnh bìa, ISRC, thời lượng). "
                    "Hệ thống đã **tự động tạo sẵn đường link tìm kiếm Spotify chính xác 100%** cho từng bài hát dưới đây — "
                    "bạn có thể bấm vào link để mở nghe trực tiếp bài hát trên Spotify!"
                )

            # Thẻ thống kê nhanh
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Tổng bài tra cứu", len(batch_results))
            with m_col2:
                st.metric("Bản thu chính thức 🟢", found_count)
            with m_col3:
                st.metric("Link tìm kiếm sẵn 🔍", search_link_count)
            with m_col4:
                st.metric("Thời gian xử lý ⏱️", f"{elapsed}s")

            # Xây dựng bảng hiển thị Data Table
            display_rows = []
            for r in batch_results:
                dur = r.get("duration_sec", 0.0)
                dur_str = f"{int(dur // 60)}m {int(dur % 60):02d}s" if dur > 0 else "-"
                score = r.get("match_score", 0.0)
                score_str = f"{score * 100:.0f}%" if isinstance(score, (int, float)) else "0%"

                if r.get("is_found"):
                    row_status = "🟢 Tìm thấy"
                elif r.get("spotify_url") and "open.spotify.com/search" in r.get("spotify_url", ""):
                    row_status = "🔍 Link tìm kiếm sẵn"
                else:
                    row_status = "⚪ Chưa khớp"

                display_rows.append(
                    {
                        "STT": r.get("index"),
                        "Trạng thái": row_status,
                        "Từ khóa gốc": r.get("input_query"),
                        "Ca sĩ gốc": r.get("input_artist") or "-",
                        "Tên bài trên Spotify": r.get("spotify_title") or "-",
                        "Nghệ sĩ Spotify": r.get("spotify_artist") or "-",
                        "Album": r.get("spotify_album") or "-",
                        "Ngày phát hành": r.get("spotify_release_date") or "-",
                        "Thời lượng": dur_str,
                        "Độ khớp": score_str,
                        "Link Spotify": r.get("spotify_url") or "",
                    }
                )

            st.markdown("##### 📊 Bảng Kết Quả Đối Soát Chi Tiết")
            table_df = pd.DataFrame(display_rows)
            st.dataframe(
                table_df,
                column_config={
                    "Link Spotify": st.column_config.LinkColumn(
                        "Link Spotify",
                        help="Link bài hát dạng text URL đầy đủ (nhấp đúp để sao chép hoặc nhấp để mở)",
                    )
                },
                use_container_width=True,
                hide_index=True,
            )

            # Khối hiển thị toàn bộ danh sách Link dạng Text thuần để sao chép nhanh
            with st.expander("📋 Xem & Sao chép nhanh danh sách toàn bộ Link Spotify (Dạng Text)", expanded=False):
                links_text = "\n".join(
                    str(r.get("spotify_url", "")).strip()
                    for r in batch_results
                    if r.get("spotify_url")
                )
                st.caption(
                    "Toàn bộ đường link đầy đủ dạng text (mỗi dòng 1 link). "
                    "Bạn có thể bôi đen hoặc bấm nút sao chép ở góc phải của khung dưới đây:"
                )
                st.code(links_text, language=None)

            # Các nút thao tác xuất file và chuyển tiếp sang Quét AI
            st.markdown("##### 🛠️ Thao Tác Báo Cáo & Chuyển Sang Giám Định AI")
            act_col1, act_col2, act_col3 = st.columns([1, 1, 1.5])

            with act_col1:
                try:
                    excel_bytes = client.export_spotify_batch_report(
                        original_df=orig_df, results=batch_results, format="xlsx"
                    )
                    st.download_button(
                        label="📥 Tải báo cáo Excel (.xlsx)",
                        data=excel_bytes,
                        file_name=f"spotify_batch_report_{int(time.time())}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Lỗi tạo Excel: {e}")

            with act_col2:
                try:
                    csv_bytes = client.export_spotify_batch_report(
                        original_df=orig_df, results=batch_results, format="csv"
                    )
                    st.download_button(
                        label="📥 Tải báo cáo CSV (.csv)",
                        data=csv_bytes,
                        file_name=f"spotify_batch_report_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Lỗi tạo CSV: {e}")

            with act_col3:
                bridge_btn = st.button(
                    "🚀 Đưa toàn bộ vào Quét AI Hàng loạt",
                    type="primary",
                    use_container_width=True,
                    help="Chuyển toàn bộ danh sách link Spotify/tên bài này sang Tab 2 để tiến hành phân tích AI hàng loạt",
                )
                if bridge_btn:
                    # Trích xuất danh sách link/tên bài chuẩn
                    ai_batch_queries = []
                    for r in batch_results:
                        target = r.get("spotify_url") or r.get("input_query")
                        if target:
                            ai_batch_queries.append(target)

                    st.session_state.batch_last_queries = ai_batch_queries
                    st.toast(
                        f"🎉 Đã chuyển {len(ai_batch_queries)} bài sang Tab Quét Hàng Loạt!",
                        icon="🚀",
                    )
                    st.success(
                        f"✅ **Đã nạp {len(ai_batch_queries)} bài vào Tab 'Quét Hàng loạt (Batch Mode)'.** "
                        f"Vui lòng bấm sang **Tab 2** để xem danh sách và bắt đầu thẩm định AI!"
                    )
