from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.models.entities import DetectionResult, TrackItem


class InMemoryTrackRepository(ITrackRepository):
    """Lưu trữ dữ liệu bài hát và kết quả thẩm định trong bộ nhớ phiên làm việc (Thread-safe dict)."""

    def __init__(self):
        self._tracks: dict[str, TrackItem] = {}
        self._results: dict[str, DetectionResult] = {}

    def save_track(self, track: TrackItem) -> TrackItem:
        existing = self._tracks.get(track.id)
        if existing:
            if not track.resolved_url and existing.resolved_url:
                track.resolved_url = existing.resolved_url
            if not track.duration_sec and existing.duration_sec:
                track.duration_sec = existing.duration_sec
            if not track.song_type and existing.song_type:
                track.song_type = existing.song_type
            if not track.distributor and existing.distributor:
                track.distributor = existing.distributor
            if not track.release_date and existing.release_date:
                track.release_date = existing.release_date
            if not track.artist and existing.artist:
                track.artist = existing.artist
            if (not track.track_title or track.track_title == track.raw_query) and existing.track_title:
                track.track_title = existing.track_title
            if existing.status == "COMPLETED" and track.status == "PROCESSING":
                track.status = "COMPLETED"
        self._tracks[track.id] = track
        return track

    def get_track(self, track_id: str) -> TrackItem | None:
        return self._tracks.get(track_id)

    def save_result(self, result: DetectionResult) -> DetectionResult:
        self._results[result.track_id] = result
        track = self._tracks.get(result.track_id)
        if track:
            if result.source_url and not track.resolved_url:
                track.resolved_url = result.source_url
            if result.song_type and not track.song_type:
                track.song_type = result.song_type
            if result.distributor and not track.distributor:
                track.distributor = result.distributor
            if result.release_date and not track.release_date:
                track.release_date = result.release_date
            dur_val = getattr(result, "duration_sec", None) or (
                result.acoustic_metrics.duration_sec if result.acoustic_metrics else None
            )
            if dur_val is not None and not track.duration_sec:
                track.duration_sec = dur_val
        return result

    def get_result(self, track_id: str) -> DetectionResult | None:
        return self._results.get(track_id)

    def list_results(self) -> list[DetectionResult]:
        return list(self._results.values())

    def find_result_by_query_or_url(
        self, query: str, resolved_url: str | None = None
    ) -> DetectionResult | None:
        import re
        clean_q = (query or "").strip().lower()
        clean_u = (resolved_url or "").strip().lower()
        yt_id = None
        for cand in [clean_u, clean_q]:
            if "youtube.com" in cand or "youtu.be" in cand:
                m = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", cand)
                if m:
                    yt_id = m.group(1).lower()
                    break

        for track_id, track in self._tracks.items():
            t_query = (track.raw_query or "").strip().lower()
            t_url = (track.resolved_url or "").strip().lower()
            t_title = (track.track_title or "").strip().lower()
            yt_match = yt_id and (yt_id in t_url or yt_id in t_query or track_id.lower() == f"yt_{yt_id}")
            if (
                yt_match
                or (clean_u and (clean_u == t_url or clean_u == t_query))
                or (clean_q and (clean_q == t_query or clean_q == t_url or clean_q == t_title))
            ):
                res = self._results.get(track_id)
                if res:
                    return res
        return None

    def correct_verdict(
        self,
        track_id: str,
        is_ai: bool,
        verdict_type: str,
        verdict_label: str,
        note: str | None = None,
    ) -> DetectionResult | None:
        res = self._results.get(track_id)
        if not res:
            return None
        from ..core.models.entities import VerdictType

        try:
            vt = VerdictType(verdict_type)
        except ValueError:
            vt = VerdictType.AI_GENERATED if is_ai else VerdictType.CONFIRMED_HUMAN

        res.is_ai = is_ai
        res.verdict_type = vt
        res.verdict_label = verdict_label
        res.is_human_corrected = True
        correction_reason = (
            f"Đính chính thủ công: {note}" if note else "Đã được đính chính thủ công bởi người dùng"
        )
        res.reasons.insert(0, correction_reason)
        return res

    def clear(self) -> None:
        self._tracks.clear()
        self._results.clear()
