from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.models.entities import DetectionResult, TrackItem


class InMemoryTrackRepository(ITrackRepository):
    """Lưu trữ dữ liệu bài hát và kết quả thẩm định trong bộ nhớ phiên làm việc (Thread-safe dict)."""

    def __init__(self):
        self._tracks: dict[str, TrackItem] = {}
        self._results: dict[str, DetectionResult] = {}

    def save_track(self, track: TrackItem) -> TrackItem:
        self._tracks[track.id] = track
        return track

    def get_track(self, track_id: str) -> TrackItem | None:
        return self._tracks.get(track_id)

    def save_result(self, result: DetectionResult) -> DetectionResult:
        self._results[result.track_id] = result
        return result

    def get_result(self, track_id: str) -> DetectionResult | None:
        return self._results.get(track_id)

    def list_results(self) -> list[DetectionResult]:
        return list(self._results.values())

    def find_result_by_query_or_url(
        self, query: str, resolved_url: str | None = None
    ) -> DetectionResult | None:
        clean_q = (query or "").strip().lower()
        clean_u = (resolved_url or "").strip().lower()
        for track_id, track in self._tracks.items():
            t_query = (track.raw_query or "").strip().lower()
            t_url = (track.resolved_url or "").strip().lower()
            if (clean_u and (clean_u == t_url or clean_u == t_query)) or (
                clean_q and (clean_q == t_query or clean_q == t_url)
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
