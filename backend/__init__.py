"""
AI Music Detector Agent - Backend Package
Enterprise Clean Layered Architecture (FAANG Standard)
"""

from .core.container import AppContainer, get_container

__all__ = ["get_container", "AppContainer"]
