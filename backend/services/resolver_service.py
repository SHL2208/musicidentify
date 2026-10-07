import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, cast
import pandas as pd
import yt_dlp

from ..core.models.entities import TrackItem, InputType
from ..core.interfaces.services.resolver_interface import IInputResolver
from ..core.exceptions import InvalidInputException

URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://'
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
    r'(?::\d+)?'
    r'(?:/?|[/?]\S+)$', re.IGNORECASE
)

def _extract_first_entry(entries: Any) -> Optional[dict]:
    """Trích xuất phần tử đầu tiên từ entries của yt-dlp."""
    if not entries:
        return None
    try:
        entry = entries[0]
        return dict(entry) if isinstance(entry, dict) else None
    except Exception:
        try:
            entry = next(iter(entries))
            return dict(entry) if isinstance(entry, dict) else None
        except Exception:
            return None

class InputResolverService(IInputResolver):
    """Hiện thực phân giải đầu vào bài hát, tuân thủ interface IInputResolver."""

    def detect_input_type(self, query: str) -> InputType:
        """Phân loại chuỗi nhập vào là URL, File nội bộ, hay Tên bài hát."""
        cleaned = query.strip()
        if URL_REGEX.match(cleaned):
            return InputType.URL
        if os.path.isfile(cleaned):
            return InputType.LOCAL_FILE
        return InputType.SONG_NAME

    def resolve_track(self, query: str) -> TrackItem:
        """Chuyển đổi một chuỗi truy vấn (link, tên, file) thành TrackItem hoàn chỉnh."""
        query = query.strip()
        if not query:
            raise InvalidInputException("Chuỗi tìm kiếm không được để trống.")

        track_id = str(uuid.uuid4())[:8]
        input_type = self.detect_input_type(query)

        if input_type == InputType.URL:
            track_title, duration = self._get_url_metadata(query)
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=query,
                track_title=track_title or query,
                duration_sec=duration
            )

        elif input_type == InputType.LOCAL_FILE:
            file_name = Path(query).stem
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=query,
                track_title=file_name
            )

        else:
            resolved_url, title, duration = self.search_youtube(query)
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=resolved_url,
                track_title=title or query,
                duration_sec=duration
            )

    def search_youtube(self, song_name: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Tìm kiếm bài hát trên YouTube bằng yt-dlp ytsearch và trả về (url, title, duration)."""
        ydl_opts: Dict[str, Any] = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
        }
        search_query = f"ytsearch1:{song_name}"
        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(search_query, download=False)
                if isinstance(info, dict) and 'entries' in info:
                    first_entry = _extract_first_entry(info.get('entries'))
                    if first_entry:
                        title = first_entry.get('title')
                        url = first_entry.get('url') or first_entry.get('webpage_url')
                        duration = first_entry.get('duration')
                        return url, title, duration
        except Exception as e:
            print(f"[InputResolverService] Lỗi tìm kiếm YouTube cho '{song_name}': {e}")
        return None, song_name, None

    def _get_url_metadata(self, url: str) -> Tuple[Optional[str], Optional[float]]:
        """Lấy title và duration từ một liên kết âm thanh."""
        ydl_opts: Dict[str, Any] = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
        }
        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(url, download=False)
                if isinstance(info, dict):
                    if 'entries' in info:
                        first = _extract_first_entry(info.get('entries'))
                        if first:
                            info = first
                    return info.get('title'), info.get('duration')
        except Exception:
            pass
        return None, None

    def parse_batch_file(self, file_path: Path) -> List[TrackItem]:
        """Đọc file đầu vào (.txt, .csv, .xlsx) và trả về danh sách các TrackItem."""
        suffix = file_path.suffix.lower()
        queries: List[str] = []

        if suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        elif suffix == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(file_path, encoding="latin-1")
            queries = self._extract_queries_from_df(df)

        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            queries = self._extract_queries_from_df(df)

        else:
            raise InvalidInputException(f"Định dạng tệp không được hỗ trợ: {suffix}. Vui lòng dùng .txt, .csv, hoặc .xlsx")

        tracks: List[TrackItem] = []
        for q in queries:
            item_id = str(uuid.uuid4())[:8]
            input_type = self.detect_input_type(q)
            tracks.append(TrackItem(
                id=item_id,
                raw_query=q,
                input_type=input_type,
                track_title=Path(q).stem if input_type == InputType.LOCAL_FILE else q,
                resolved_url=q if input_type == InputType.URL else None
            ))

        return tracks

    def _extract_queries_from_df(self, df: pd.DataFrame) -> List[str]:
        """Tự động tìm kiếm cột chứa link hoặc tên bài hát từ DataFrame."""
        header_keywords = [
            "tên bài hát", "bài hát", "đường dẫn", "liên kết", "tiêu đề",
            "track", "query", "url", "link", "song", "title", "name"
        ]
        chosen_col = None
        has_named_header = False

        for col in df.columns:
            clean_name = str(col).strip().lower()
            # Nếu tên cột là một URL, nó chắc chắn không phải là header mà là dữ liệu của hàng đầu
            if URL_REGEX.match(clean_name) or clean_name.startswith("http") or "youtu" in clean_name:
                continue

            if clean_name in header_keywords or any(kw in clean_name for kw in ["bài hát", "tiêu đề", "đường dẫn", "liên kết", "track"]):
                chosen_col = col
                has_named_header = True
                break

        if chosen_col is None:
            chosen_col = df.columns[0]

        items = [str(val).strip() for val in df[chosen_col].dropna() if str(val).strip()]

        # Nếu cột đầu tiên là dữ liệu (không có header hoặc file không có header)
        col_str = str(chosen_col).strip()
        if not has_named_header:
            if URL_REGEX.match(col_str) or col_str.startswith("http") or "youtu" in col_str:
                items.insert(0, col_str)
            elif not col_str.lower().startswith("unnamed:") and col_str.lower() not in header_keywords:
                items.insert(0, col_str)

        return items
