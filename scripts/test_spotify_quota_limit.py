"""Công cụ kiểm thử hạn mức & khả năng lấy link Spotify Track hàng loạt (Spotify Quota & Rate Limit Tester).

Mục đích:
- Test trực tiếp với Spotify Web API để đo số lượng bài hát có thể lấy link thành công trước khi chạm Rate Limit.
- Thống kê tỷ lệ thành công (Success Rate), độ trễ (Latency ms), và thông tin chặn nếu gặp HTTP 429 (Retry-After).
- Tự động cô lập Cache (Bypass SQLite Cache) để bảo đảm 100% request được gửi trực tiếp đến máy chủ Spotify.

Cách sử dụng:
    python scripts/test_spotify_quota_limit.py --limit 30
    python scripts/test_spotify_quota_limit.py --limit 50 --delay 0.1
    python scripts/test_spotify_quota_limit.py --file my_songs.txt --stop-on-429
    python scripts/test_spotify_quota_limit.py --limit 20 --export-excel results.xlsx
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 cho Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Thêm thư mục gốc của project vào sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.core.config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
)
from backend.services.spotify_cache_service import SpotifyCacheService
from backend.services.spotify_service import SpotifyService

# ==============================================================================
# DANH SÁCH BÀI HÁT MẪU (100 bài đa dạng V-Pop, US-UK, Ballad, Rap, Pop)
# ==============================================================================
DEFAULT_SAMPLE_TRACKS: list[dict[str, str]] = [
    # --- V-Pop & Rap Việt ---
    {"title": "Chúng Ta Của Hiện Tại", "artist": "Sơn Tùng M-TP"},
    {"title": "Nơi Này Có Anh", "artist": "Sơn Tùng M-TP"},
    {"title": "Lạc Trôi", "artist": "Sơn Tùng M-TP"},
    {"title": "Đừng Làm Trái Tim Anh Đau", "artist": "Sơn Tùng M-TP"},
    {"title": "Em Của Ngày Hôm Qua", "artist": "Sơn Tùng M-TP"},
    {"title": "Nấu Ăn Cho Em", "artist": "Đen"},
    {"title": "Trốn Tìm", "artist": "Đen"},
    {"title": "Mang Tiền Về Cho Mẹ", "artist": "Đen"},
    {"title": "Đi Về Nhà", "artist": "Đen, JustaTee"},
    {"title": "Bài Này Chill Phết", "artist": "Đen, Min"},
    {"title": "Bước Qua Mùa Cô Đơn", "artist": "Vũ."},
    {"title": "Lạ Lùng", "artist": "Vũ."},
    {"title": "Bước Qua Nhau", "artist": "Vũ."},
    {"title": "Để Mị Nói Cho Mà Nghe", "artist": "Hoàng Thùy Linh"},
    {"title": "See Tình", "artist": "Hoàng Thùy Linh"},
    {"title": "Gieo Quẻ", "artist": "Hoàng Thùy Linh"},
    {"title": "Cắt Đôi Nỗi Sầu", "artist": "Tăng Duy Tân"},
    {"title": "Bên Trên Tầng Lầu", "artist": "Tăng Duy Tân"},
    {"title": "Dạ Vũ", "artist": "Tăng Duy Tân"},
    {"title": "Thị Mầu", "artist": "Hòa Minzy"},
    {"title": "Rời Bỏ", "artist": "Hòa Minzy"},
    {"title": "Waiting For You", "artist": "MONO"},
    {"title": "Em Là", "artist": "MONO"},
    {"title": "vaicaunoicokhiennguoithaydoi", "artist": "GREY D"},
    {"title": "đưa em về nhàa", "artist": "GREY D"},
    {"title": "Có Chàng Trai Viết Lên Cây", "artist": "Phan Mạnh Quỳnh"},
    {"title": "Ngày Chưa Giông Bão", "artist": "Bùi Lan Hương"},
    {"title": "Nàng Thơ", "artist": "Hoàng Dũng"},
    {"title": "Yếu Đuối", "artist": "Hoàng Dũng"},
    {"title": "Đi Giữa Trời Rực Rỡ", "artist": "Ngô Lan Hương"},
    {"title": "Từng Là", "artist": "Vũ Cát Tường"},
    {"title": "Vết Mưa", "artist": "Vũ Cát Tường"},
    {"title": "Có Ai Thương Em Như Anh", "artist": "Tóc Tiên"},
    {"title": "Ngày Đầu Tiên", "artist": "Đức Phúc"},
    {"title": "Hơn Cả Yêu", "artist": "Đức Phúc"},
    {"title": "Ánh Nắng Của Anh", "artist": "Đức Phúc"},
    {"title": "Chạm Đáy Nỗi Đau", "artist": "ERIK"},
    {"title": "Sau Tất Cả", "artist": "ERIK"},
    {"title": "Bật Tình Yêu Lên", "artist": "Tăng Duy Tân, Hòa Minzy"},
    {"title": "Chịu Cách Mình Nói Thua", "artist": "Rhyder"},
    {"title": "Hào Quang", "artist": "Rhyder"},
    {"title": "Ngủ Một Mình", "artist": "HIEUTHUHAI"},
    {"title": "Không Thể Say", "artist": "HIEUTHUHAI"},
    {"title": "Vệ Tinh", "artist": "HIEUTHUHAI"},
    {"title": "Thích Em Hơi Nhiều", "artist": "Wren Evans"},
    {"title": "Tò Te Tí", "artist": "Wren Evans"},
    {"title": "Cứ Chill Thôi", "artist": "Chillies, Suni Hạ Linh, Rhymastic"},
    {"title": "Mascara", "artist": "Chillies"},
    {"title": "Vùng Ký Ức", "artist": "Chillies"},
    {"title": "Có Em", "artist": "Madihu, Low G"},
    # --- US-UK & Quốc tế ---
    {"title": "Shape of You", "artist": "Ed Sheeran"},
    {"title": "Perfect", "artist": "Ed Sheeran"},
    {"title": "Bad Habits", "artist": "Ed Sheeran"},
    {"title": "Blank Space", "artist": "Taylor Swift"},
    {"title": "Cruel Summer", "artist": "Taylor Swift"},
    {"title": "Anti-Hero", "artist": "Taylor Swift"},
    {"title": "Love Story", "artist": "Taylor Swift"},
    {"title": "Blinding Lights", "artist": "The Weeknd"},
    {"title": "Starboy", "artist": "The Weeknd"},
    {"title": "Save Your Tears", "artist": "The Weeknd"},
    {"title": "Someone Like You", "artist": "Adele"},
    {"title": "Rolling in the Deep", "artist": "Adele"},
    {"title": "Easy On Me", "artist": "Adele"},
    {"title": "Uptown Funk", "artist": "Mark Ronson, Bruno Mars"},
    {"title": "That's What I Like", "artist": "Bruno Mars"},
    {"title": "Die With A Smile", "artist": "Lady Gaga, Bruno Mars"},
    {"title": "bad guy", "artist": "Billie Eilish"},
    {"title": "BIRDS OF A FEATHER", "artist": "Billie Eilish"},
    {"title": "lovely", "artist": "Billie Eilish, Khalid"},
    {"title": "Attention", "artist": "Charlie Puth"},
    {"title": "We Don't Talk Anymore", "artist": "Charlie Puth, Selena Gomez"},
    {"title": "See You Again", "artist": "Wiz Khalifa, Charlie Puth"},
    {"title": "Believer", "artist": "Imagine Dragons"},
    {"title": "Demons", "artist": "Imagine Dragons"},
    {"title": "Counting Stars", "artist": "OneRepublic"},
    {"title": "Viva La Vida", "artist": "Coldplay"},
    {"title": "Yellow", "artist": "Coldplay"},
    {"title": "Something Just Like This", "artist": "The Chainsmokers, Coldplay"},
    {"title": "Closer", "artist": "The Chainsmokers, Halsey"},
    {"title": "Faded", "artist": "Alan Walker"},
    {"title": "Alone", "artist": "Alan Walker"},
    {"title": "Levitating", "artist": "Dua Lipa"},
    {"title": "Don't Start Now", "artist": "Dua Lipa"},
    {"title": "Stay", "artist": "The Kid LAROI, Justin Bieber"},
    {"title": "Ghost", "artist": "Justin Bieber"},
    {"title": "Peaches", "artist": "Justin Bieber"},
    {"title": "Flowers", "artist": "Miley Cyrus"},
    {"title": "As It Was", "artist": "Harry Styles"},
    {"title": "Watermelon Sugar", "artist": "Harry Styles"},
    {"title": "Until I Found You", "artist": "Stephen Sanchez"},
    {"title": "Dynamite", "artist": "BTS"},
    {"title": "Butter", "artist": "BTS"},
    {"title": "How You Like That", "artist": "BLACKPINK"},
    {"title": "Pink Venom", "artist": "BLACKPINK"},
    {"title": "Ditto", "artist": "NewJeans"},
    {"title": "OMG", "artist": "NewJeans"},
    {"title": "Super Shy", "artist": "NewJeans"},
    {"title": "Golden Hour", "artist": "JVKE"},
    {"title": "Glimpse of Us", "artist": "Joji"},
    {"title": "Snooze", "artist": "SZA"},
]


@dataclass
class TrackTestResult:
    """Kết quả kiểm thử cho một bài hát."""

    index: int
    query: str
    artist: str
    status: str  # 'SUCCESS', 'NOT_FOUND', 'RATE_LIMITED', 'ERROR'
    spotify_url: str = ""
    spotify_id: str = ""
    spotify_title: str = ""
    spotify_artist: str = ""
    match_score: float = 0.0
    latency_ms: float = 0.0
    retry_after: int = 0
    error_message: str = ""


@dataclass
class QuotaTestSummary:
    """Bảng tổng hợp thống kê sau khi hoàn tất kiểm thử."""

    total_requested: int = 0
    total_executed: int = 0
    success_count: int = 0
    not_found_count: int = 0
    rate_limited_count: int = 0
    error_count: int = 0
    total_time_sec: float = 0.0
    avg_latency_ms: float = 0.0
    requests_per_sec: float = 0.0
    max_consecutive_success: int = 0
    rate_limit_encountered_at: int | None = None
    results: list[TrackTestResult] = field(default_factory=list)


class SpotifyQuotaTester:
    """Lớp điều phối kiểm tra hạn ngạch Spotify Web API."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        bypass_cache: bool = True,
        delay_between_requests: float = 0.05,
        stop_on_429: bool = False,
    ):
        self.client_id = client_id or SPOTIFY_CLIENT_ID
        self.client_secret = client_secret or SPOTIFY_CLIENT_SECRET
        self.bypass_cache = bypass_cache
        self.delay = max(0.0, delay_between_requests)
        self.stop_on_429 = stop_on_429

        # Nếu bypass_cache=True, sử dụng SQLite in-memory (:memory:) hoàn toàn độc lập
        # để mọi request đều gọi trực tiếp tới máy chủ Spotify mà không đọc cache cũ
        cache_service = (
            SpotifyCacheService(":memory:") if self.bypass_cache else SpotifyCacheService()
        )
        self.service = SpotifyService(
            client_id=self.client_id,
            client_secret=self.client_secret,
            timeout_seconds=5.0,
            cache_service=cache_service,
        )

    def load_tracks_from_file(self, file_path: str | Path) -> list[dict[str, str]]:
        """Nạp danh sách bài hát từ file txt, csv, hoặc xlsx."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {p}")

        tracks: list[dict[str, str]] = []
        ext = p.suffix.lower()

        if ext in (".csv", ".tsv"):
            import pandas as pd

            df = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",")
            for _, row in df.iterrows():
                title = str(
                    row.get("title")
                    or row.get("query")
                    or row.get("track_title")
                    or row.get("song_title")
                    or row.iloc[0]
                ).strip()
                artist = str(row.get("artist") or row.get("singer") or "").strip()
                if title:
                    tracks.append({"title": title, "artist": artist})

        elif ext in (".xlsx", ".xls"):
            import pandas as pd

            df = pd.read_excel(p)
            for _, row in df.iterrows():
                title = str(
                    row.get("title")
                    or row.get("query")
                    or row.get("track_title")
                    or row.get("song_title")
                    or row.iloc[0]
                ).strip()
                artist = str(row.get("artist") or row.get("singer") or "").strip()
                if title:
                    tracks.append({"title": title, "artist": artist})

        else:  # Text file (.txt)
            content = p.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if " - " in line and not line.startswith("http"):
                    parts = line.split(" - ", 1)
                    tracks.append({"title": parts[0].strip(), "artist": parts[1].strip()})
                else:
                    tracks.append({"title": line, "artist": ""})

        return tracks

    def run_test(
        self,
        tracks: list[dict[str, str]],
        limit: int | None = None,
    ) -> QuotaTestSummary:
        """Thực thi kiểm tra lần lượt từng bài hát và đo lường hạn ngạch."""
        if limit and limit > 0:
            target_tracks = tracks[:limit]
        else:
            target_tracks = tracks

        total = len(target_tracks)
        summary = QuotaTestSummary(total_requested=total)

        print("\n" + "=" * 78)
        print("🚀 BẮT ĐẦU KIỂM THỬ HẠN MỨC SPOTIFY WEB API")
        print("=" * 78)
        auth_type = (
            "Client Credentials Flow (Official)"
            if (self.client_id and self.client_secret)
            else "Anonymous Web Token (Fallback)"
        )
        print(f"🔑 Chế độ xác thực   : {auth_type}")
        print(f"📦 Số bài cần kiểm tra: {total} bài")
        print(f"⚡ Trạng thái Cache   : {'BYPASS (Gửi 100% request lên API)' if self.bypass_cache else 'ENABLED (Dùng Cache SQLite)'}")
        print(f"⏱️  Khoảng cách (delay): {self.delay}s / request")
        print(f"🛑 Dừng khi gặp 429   : {'BẬT (Stop on 429)' if self.stop_on_429 else 'TẮT (Tiếp tục thử)'}")
        print("-" * 78)

        start_time = time.time()
        current_consecutive_success = 0
        total_latency = 0.0

        for idx, item in enumerate(target_tracks, start=1):
            title = item.get("title", "").strip()
            artist = item.get("artist", "").strip()
            display_query = f"'{title}'" + (f" - {artist}" if artist else "")

            track_start = time.time()
            res_obj: dict[str, Any] | None = None
            err_msg = ""
            status = "NOT_FOUND"

            try:
                # Kiểm tra trạng thái Circuit Breaker trước khi gọi
                if self.service._quota_exceeded:
                    status = "RATE_LIMITED"
                    err_msg = "Spotify API Quota Exceeded (Circuit Breaker đã kích hoạt)"
                else:
                    res_obj = self.service.find_best_match(title, artist=artist)

                    # Kiểm tra xem có bị gán cờ quota sau lần gọi này không
                    if self.service._quota_exceeded:
                        status = "RATE_LIMITED"
                        err_msg = "Spotify trả về HTTP 429 Too Many Requests"
                    elif res_obj and res_obj.get("spotify_url"):
                        # Bản thu chính thức tìm thấy thành công
                        if not str(res_obj.get("id", "")).startswith("search_fallback"):
                            status = "SUCCESS"
                        else:
                            status = "NOT_FOUND"
                    else:
                        status = "NOT_FOUND"

            except Exception as e:
                status = "ERROR"
                err_msg = str(e)

            track_latency_ms = (time.time() - track_start) * 1000.0
            total_latency += track_latency_ms

            # Xử lý kết quả
            tr = TrackTestResult(
                index=idx,
                query=title,
                artist=artist,
                status=status,
                latency_ms=round(track_latency_ms, 1),
                error_message=err_msg,
            )

            if status == "SUCCESS" and res_obj:
                tr.spotify_url = res_obj.get("spotify_url", "")
                tr.spotify_id = res_obj.get("id", "")
                tr.spotify_title = res_obj.get("title", "")
                tr.spotify_artist = res_obj.get("artist", "")
                tr.match_score = float(res_obj.get("match_score", 0.0))

                current_consecutive_success += 1
                summary.max_consecutive_success = max(
                    summary.max_consecutive_success, current_consecutive_success
                )
                summary.success_count += 1
                icon = "✅"
                info = f"-> {tr.spotify_url} | Score: {tr.match_score:.2f}"
            elif status == "NOT_FOUND":
                current_consecutive_success = 0
                summary.not_found_count += 1
                icon = "⚪"
                info = "(Chưa khớp trên Spotify)"
            elif status == "RATE_LIMITED":
                current_consecutive_success = 0
                summary.rate_limited_count += 1
                if summary.rate_limit_encountered_at is None:
                    summary.rate_limit_encountered_at = idx
                icon = "🛑"
                info = f"[HTTP 429] {err_msg}"
            else:  # ERROR
                current_consecutive_success = 0
                summary.error_count += 1
                icon = "❌"
                info = f"Lỗi: {err_msg}"

            summary.results.append(tr)
            summary.total_executed = idx

            # Log trực tiếp trạng thái từng bài
            print(
                f"[{idx:>3}/{total}] {icon} {status:<12} ({track_latency_ms:>5.0f}ms) "
                f"{display_query[:35]:<35} {info}"
            )

            # Dừng nếu cấu hình stop_on_429
            if status == "RATE_LIMITED" and self.stop_on_429:
                print(f"\n⚠️  [DỪNG SỚM] Phát hiện HTTP 429 Rate Limit tại bài thứ {idx}!")
                break

            # Delay an toàn giữa các request nếu có
            if self.delay > 0 and idx < total:
                time.sleep(self.delay)

        total_time = time.time() - start_time
        summary.total_time_sec = round(total_time, 2)
        if summary.total_executed > 0:
            summary.avg_latency_ms = round(total_latency / summary.total_executed, 1)
            summary.requests_per_sec = round(summary.total_executed / max(0.001, total_time), 2)

        self._print_summary_report(summary)
        return summary

    def _print_summary_report(self, summary: QuotaTestSummary) -> None:
        """In bảng tổng kết kết quả kiểm thử định dạng đẹp."""
        print("\n" + "=" * 78)
        print("📊 BẢNG TỔNG KẾT KẾT QUẢ KIỂM THỬ SPOTIFY API QUOTA")
        print("=" * 78)
        total_exec = summary.total_executed
        succ_pct = (summary.success_count / total_exec * 100) if total_exec else 0.0
        nf_pct = (summary.not_found_count / total_exec * 100) if total_exec else 0.0
        rl_pct = (summary.rate_limited_count / total_exec * 100) if total_exec else 0.0

        api_calls = getattr(self.service, "api_call_count", total_exec)
        ratio = (api_calls / max(1, total_exec)) if total_exec else 1.0

        print(f"Tổng số bài đã chạy          : {total_exec} / {summary.total_requested}")
        print(f"📡 Tổng HTTP Requests gửi API : {api_calls} requests ({ratio:.2f} req/bài)")
        print(f"✅ Lấy link Spotify THÀNH CÔNG: {summary.success_count:>4} bài ({succ_pct:.1f}%)")
        print(f"⚪ Không tìm thấy / Chưa khớp : {summary.not_found_count:>4} bài ({nf_pct:.1f}%)")
        print(f"🛑 Chạm giới hạn Rate Limit  : {summary.rate_limited_count:>4} bài ({rl_pct:.1f}%)")
        print(f"❌ Lỗi ngoại lệ khác         : {summary.error_count:>4} bài")
        print("-" * 78)
        print(f"⏱️  Tổng thời gian thực thi   : {summary.total_time_sec} giây")
        print(f"⚡ Tốc độ trung bình          : {summary.requests_per_sec} requests/giây")
        print(f"📈 Độ trễ trung bình / bài    : {summary.avg_latency_ms} ms")
        print(f"🔥 Chuỗi thành công liên tục  : {summary.max_consecutive_success} bài")

        if summary.rate_limit_encountered_at:
            print(f"⚠️  Bị chặn Rate Limit ở bài  : Bài thứ {summary.rate_limit_encountered_at}")
        else:
            print("🎉 KHÔNG HỀ BỊ CHẶN RATE LIMIT! (Quota & Pacer hoàn toàn ổn định)")

        print("=" * 78 + "\n")

    def export_results(
        self,
        summary: QuotaTestSummary,
        excel_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> None:
        """Xuất kết quả kiểm thử ra file Excel hoặc JSON."""
        data_rows = []
        for r in summary.results:
            data_rows.append(
                {
                    "STT": r.index,
                    "Tên bài tìm kiếm": r.query,
                    "Nghệ sĩ": r.artist,
                    "Trạng thái": r.status,
                    "Spotify Track Title": r.spotify_title,
                    "Spotify Artist": r.spotify_artist,
                    "Spotify URL": r.spotify_url,
                    "Spotify ID": r.spotify_id,
                    "Điểm khớp (Match Score)": r.match_score,
                    "Độ trễ (ms)": r.latency_ms,
                    "Ghi chú lỗi": r.error_message,
                }
            )

        if excel_path:
            import pandas as pd

            df = pd.DataFrame(data_rows)
            out_p = Path(excel_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(out_p, index=False)
            print(f"💾 Đã lưu chi tiết kiểm thử vào Excel: {out_p.resolve()}")

        if json_path:
            import json

            out_p = Path(json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(
                json.dumps(data_rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"💾 Đã lưu chi tiết kiểm thử vào JSON: {out_p.resolve()}")


def main() -> None:
    """Hàm chạy dòng lệnh chính."""
    parser = argparse.ArgumentParser(
        description="Kiểm tra số lượng bài hát có thể lấy link Spotify thành công và đo Rate Limit."
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Số lượng bài muốn test (Mặc định: 20 bài)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.05,
        help="Thời gian nghỉ giữa các bài tính bằng giây (Mặc định: 0.05s)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Đường dẫn tới file danh sách bài hát (.txt, .csv, .xlsx). Nếu bỏ trống sẽ dùng danh sách bài mẫu 100 bài có sẵn.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Sử dụng cache SQLite có sẵn thay vì bypass. Mặc định là bypass cache để đo trực tiếp API.",
    )
    parser.add_argument(
        "--stop-on-429",
        action="store_true",
        help="Dừng kiểm tra ngay lập tức khi phát hiện HTTP 429 Rate Limit.",
    )
    parser.add_argument(
        "--export-excel",
        type=str,
        default=None,
        help="Đường dẫn file Excel để xuất chi tiết kết quả (ví dụ: results.xlsx).",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Đường dẫn file JSON để xuất chi tiết kết quả (ví dụ: results.json).",
    )

    args = parser.parse_args()

    tester = SpotifyQuotaTester(
        bypass_cache=not args.use_cache,
        delay_between_requests=args.delay,
        stop_on_429=args.stop_on_429,
    )

    if args.file:
        print(f"📂 Đang nạp danh sách bài hát từ file: {args.file}...")
        tracks = tester.load_tracks_from_file(args.file)
    else:
        tracks = DEFAULT_SAMPLE_TRACKS

    summary = tester.run_test(tracks=tracks, limit=args.limit)

    if args.export_excel or args.export_json:
        tester.export_results(
            summary=summary,
            excel_path=args.export_excel,
            json_path=args.export_json,
        )


if __name__ == "__main__":
    main()
