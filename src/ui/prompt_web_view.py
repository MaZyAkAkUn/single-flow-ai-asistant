import json
import logging
from typing import Dict, Any, List
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject
from PyQt6.QtWidgets import QApplication

from .base_web_view import BaseWebView

logger = logging.getLogger(__name__)

class PromptBridge(QObject):
    """Bridge for Prompt Web View."""
    
    # Signals
    refreshRequested = pyqtSignal()
    editRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot()
    def requestRefresh(self):
        self.refreshRequested.emit()
        
    @pyqtSlot()
    def requestEdit(self):
        self.editRequested.emit()
        
    @pyqtSlot(str)
    def copyToClipboard(self, text: str):
        cb = QApplication.clipboard()
        cb.setText(text)
        
    @pyqtSlot(str)
    def log(self, message: str):
        logger.debug(f"JS [Prompt]: {message}")


class PromptWebWidget(BaseWebView):
    """
    QWebEngineView-based prompt panel.
    """

    edit_requested = pyqtSignal() # Propagate to Main Window

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = PromptBridge()
        self.current_prompt_data = None
        
        self.init_web_channel({"backend": self.bridge})
        self.load_ui("prompt.html")
        
        self.bridge.refreshRequested.connect(self.refresh_prompt)
        self.bridge.editRequested.connect(self.edit_requested.emit)

    def on_ui_ready(self):
        if self.current_prompt_data:
            self.set_current_prompt(self.current_prompt_data)

    def set_current_prompt(self, prompt_data: Dict[str, Any]):
        """Set the current prompt and update JS."""
        self.current_prompt_data = prompt_data
        
        # We might need to parse XML content here to JSON if we want sophisticated rendering,
        # OR we pass the raw XML string and let JS parse/render it.
        # JS has DOMParser which is great for XML.
        # Let's pass the raw structured prompt string + metadata.
        
        payload = {
            'structured_prompt': prompt_data.get('structured_prompt', ''),
            'user_input': prompt_data.get('user_input', ''),
            'timestamp': str(prompt_data.get('timestamp', '')),
        }
        
        payload_json = json.dumps(payload)
        self.run_js(f"updatePrompt({payload_json});")
        self.run_js(f"addToHistory({payload_json});")

    def display_prompt(self, prompt_text: str):
        """Display raw text prompt."""
        # Wrap in mock structure
        payload = {
            'structured_prompt': f"<SystemPrompt><Raw>{prompt_text}</Raw></SystemPrompt>",
            'user_input': '',
            'timestamp': 'Now'
        }
        self.set_current_prompt(payload)
        
    def refresh_prompt(self):
        # Re-send current data if valid
        if self.current_prompt_data:
            self.set_current_prompt(self.current_prompt_data)
