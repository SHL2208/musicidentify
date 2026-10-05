import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import yt_dlp

from ..models.schemas import TrackItem, InputType

URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE
)

class InputResolver:
    @staticmethod
    def detect_input_type(query: str) -> InputType:
        """Phân loại chuỗi nhập vào là URL, File nội bộ, hay Tên bài hát."""
        cleaned = query.strip()
        if URL_REGEX.match(cleaned):
            return InputType.URL
        if os.path.isfile(cleaned):
            return InputType.LOCAL_FILE
        return InputType.SONG_NAME

    @staticmethod
    def resolve_track(query: str) -> TrackItem:
        """Chuyển đổi một chuỗi truy vấn (link, tên, file) thành TrackItem hoàn chỉnh."""
        query = query.strip()
        track_id = str(uuid.uuid4())[:8]
        input_type = InputResolver.detect_input_type(query)

        if input_type == InputType.URL:
            # Lấy thông tin video/audio từ URL
            track_title, duration = InputResolver._get_url_metadata(query)
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=query,
                track_title=track_title or query,
                duration_sec=duration
            )

        elif input_type == InputType.LOCAL_FILE:
            # File âm thanh local
            file_name = Path(query).stem
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=query,
                track_title=file_name
            )

        else:
            # Tên bài hát -> Tìm kiếm link YouTube phù hợp nhất
            resolved_url, title, duration = InputResolver.search_youtube(query)
            return TrackItem(
                id=track_id,
                raw_query=query,
                input_type=input_type,
                resolved_url=resolved_url,
                track_title=title or query,
                duration_sec=duration
            )

    @staticmethod
    def search_youtube(song_name: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Tìm kiếm bài hát trên YouTube bằng yt-dlp ytsearch và trả về (url, title, duration)."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
        }
        search_query = f"ytsearch1:{song_name}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)
                if 'entries' in info and len(info['entries']) > 0:
                    entry = info['entries'][0]
                    url = entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}"
                    title = entry.get('title')
                    duration = entry.get('duration')
                    return url, title, duration
        except Exception as e:
            print(f"[Resolver] Lỗi khi search YouTube '{song_name}': {e}")
        return None, song_name, None

    @staticmethod
    def _get_url_metadata(url: str) -> Tuple[Optional[str], Optional[float]]:
        """Lấy tiêu đề và thời lượng của một URL qua yt-dlp."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('title'), info.get('duration')
        except Exception:
            return None, None

    @staticmethod
    def parse_batch_text(text: str) -> List[TrackItem]:
        """Phân tích danh sách nhiều dòng bài hát hoặc link dán vào."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        tracks = []
        for line in lines:
            tracks.append(InputResolver.resolve_track(line))
        return tracks

    @staticmethod
    def parse_batch_file(file_path: Path) -> List[TrackItem]:
        """Đọc danh sách từ file .txt, .csv, hoặc .xlsx."""
        suffix = file_path.suffix.lower()
        raw_queries: List[str] = []

        if suffix in [".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_queries = [line.strip() for line in f if line.strip()]

        elif suffix in [".csv"]:
            df = pd.read_csv(file_path)
            raw_queries = InputResolver._extract_queries_from_df(df)

        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            raw_queries = InputResolver._extract_queries_from_df(df)

        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {suffix}. Hãy dùng .txt, .csv hoặc .xlsx")

        tracks = []
        for q in raw_queries:
            tracks.append(InputResolver.resolve_track(q))
        return tracks

    @staticmethod
    def _extract_queries_from_df(df: pd.DataFrame) -> List[str]:
        """Tự động tìm cột chứa tên hoặc link bài hát trong DataFrame."""
        candidate_cols = ["link", "url", "song", "title", "bài hát", "tên bài hát", "track", "name"]
        selected_col = None
        for col in df.columns:
            if str(col).strip().lower() in candidate_cols:
                selected_col = col
                break
        
        if selected_col is None:
            # Chọn cột đầu tiên nếu không khớp tên nào
            selected_col = df.columns[0]

        return [str(val).strip() for val in df[selected_col].dropna() if str(val).strip()]
