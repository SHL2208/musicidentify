class MusicIdentifyException(RuntimeError):
    """Base exception class for music identify application."""
    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class AudioProcessingException(MusicIdentifyException):
    """Raised when audio download, extraction, or slicing fails."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=422, details=details)

class ClassificationException(MusicIdentifyException):
    """Raised when AI classification fails."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=502, details=details)

class RepositoryException(MusicIdentifyException):
    """Raised when data storage or retrieval operations fail."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=500, details=details)

class TrackNotFoundException(MusicIdentifyException):
    """Raised when a requested track cannot be found."""
    def __init__(self, track_id: str):
        super().__init__(f"Track with ID '{track_id}' not found.", status_code=404, details={"track_id": track_id})

class InvalidInputException(MusicIdentifyException):
    """Raised when input validation fails."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, details=details)
