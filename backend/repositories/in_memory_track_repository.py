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

    def clear(self) -> None:
        self._tracks.clear()
        self._results.clear()
