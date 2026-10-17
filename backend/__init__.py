"""
AI Music Detector Agent - Backend Package
Enterprise Clean Layered Architecture (FAANG Standard)
"""


def __getattr__(name: str):
    if name in ("AppContainer", "get_container"):
        from .core.container import AppContainer, get_container

        return AppContainer if name == "AppContainer" else get_container
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_container", "AppContainer"]
