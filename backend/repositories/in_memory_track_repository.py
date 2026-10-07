from typing import List, Optional, Dict
from ..core.interfaces.repositories.track_repository_interface import ITrackRepository
from ..core.models.entities import TrackItem, DetectionResult

class InMemoryTrackRepository(ITrackRepository):
    """Lưu trữ dữ liệu bài hát và kết quả thẩm định trong bộ nhớ phiên làm việc (Thread-safe dict)."""

    def __init__(self):
        self._tracks: Dict[str, TrackItem] = {}
        self._results: Dict[str, DetectionResult] = {}

    def save_track(self, track: TrackItem) -> TrackItem:
        self._tracks[track.id] = track
        return track

    def get_track(self, track_id: str) -> Optional[TrackItem]:
        return self._tracks.get(track_id)

    def save_result(self, result: DetectionResult) -> DetectionResult:
        self._results[result.track_id] = result
        return result

    def get_result(self, track_id: str) -> Optional[DetectionResult]:
        return self._results.get(track_id)

    def list_results(self) -> List[DetectionResult]:
        return list(self._results.values())

    def clear(self) -> None:
        self._tracks.clear()
        self._results.clear()
