"""
UI package for the Single-Flow Personal Assistant.
Contains PyQt6-based user interface components.
"""

from .main_window import MainWindow, SidePanelWidget
from .chat_web_view import ChatWebWidget

__all__ = [
    'MainWindow',
    'ChatWebWidget',
    'SidePanelWidget'
]
