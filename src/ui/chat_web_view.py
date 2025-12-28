import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject

from .base_web_view import BaseWebView

logger = logging.getLogger(__name__)

class WebBridge(QObject):
    """Bridge for communication between Python and JavaScript."""

    # Signal to notify Python about an action requested from JS
    # Arguments: message_id, action_type
    actionRequested = pyqtSignal(str, str)

    # Signal for text input from JS
    textInputReceived = pyqtSignal(str)

    # Signal for audio control (input)
    audioToggleRequested = pyqtSignal()

    # Signal for audio player control
    audioControlRequested = pyqtSignal(str, float)

    # Signal for draft operations
    draftActionRequested = pyqtSignal(str, str)  # action, content (if applicable)

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot(str, str, str)
    def receiveMessage(self, category: str, action: str, content: str = ""):
        """Receive action from JavaScript with optional content."""
        if category == 'draft':
            logger.debug(f"Draft action received from JS: {action}")
            self.draftActionRequested.emit(action, content)
        else:
            # Handle as regular message
            logger.debug(f"Action received from JS: {action} for {category}")
            self.actionRequested.emit(category, action)

    @pyqtSlot(str)
    def receiveTextInput(self, text: str):
        """Receive text input from JavaScript."""
        logger.info(f"Text Input received from JS: {text[:50]}...")
        self.textInputReceived.emit(text)

    @pyqtSlot()
    def toggleAudioRecording(self):
        """Toggle audio recording request from JavaScript."""
        logger.debug("Audio toggle requested from JS")
        self.audioToggleRequested.emit()

    @pyqtSlot(str, float)
    def controlAudio(self, command: str, value: float):
        """Control audio playback from JavaScript."""
        self.audioControlRequested.emit(command, value)

    @pyqtSlot(str)
    def copyToClipboard(self, text: str):
        """Copy text to system clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        logger.info("Text copied to clipboard via WebBridge")

class ChatWebWidget(BaseWebView):
    """
    QWebEngineView-based chat widget replacment for MessageListWidget.
    Renders chat using HTML/JS/CSS for better formatting and performance.
    """
    
    # Signals matching the original interface (or adapted)
    regenerate_requested = pyqtSignal(dict)
    edit_requested = pyqtSignal(dict)
    show_prompt_requested = pyqtSignal(dict)
    speak_text_requested = pyqtSignal(dict)
    
    # New Signals for Input Integration
    input_received = pyqtSignal(str)
    audio_toggled = pyqtSignal()
    open_audio_settings_requested = pyqtSignal()
    
    # Audio Player Control
    audio_control_received = pyqtSignal(str, float)

    # Draft operations
    draft_action_received = pyqtSignal(str, str)  # action, content

    # UI ready signal
    ui_ready = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.messages_map: Dict[str, Dict[str, Any]] = {} # ID -> Message Data
        self.messages: List[Dict[str, Any]] = [] # Ordered list to match old API
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the web view and channel."""
        self.bridge = WebBridge()
        
        # Connect signals
        self.bridge.actionRequested.connect(self.handle_js_action)
        self.bridge.textInputReceived.connect(self.input_received)
        self.bridge.audioToggleRequested.connect(self.audio_toggled)
        self.bridge.audioControlRequested.connect(self.audio_control_received)
        self.bridge.draftActionRequested.connect(self.draft_action_received)
        
        self.init_web_channel({"backend": self.bridge})
        self.load_ui("chat.html")
    
    def on_ui_ready(self):
        """Handle page load completion."""
        logger.info("Chat Web Page loaded successfully")
        # BaseWebView handles pending JS execution

        # Emit signal that UI is ready (for combo loading, etc.)
        self.ui_ready.emit()
        
    def add_message(self, message_data: Dict[str, Any]):
        """Add a message to the chat view."""
        # Ensure message has an ID
        if 'id' not in message_data:
            message_data['id'] = str(uuid.uuid4())
            
        msg_id = message_data['id']
        self.messages_map[msg_id] = message_data
        self.messages.append(message_data)
        
        # Prepare data for JS
        role = message_data.get('role', 'unknown')
        content = message_data.get('content', '')
        pinned = message_data.get('pinned', False)
        timestamp = message_data.get('timestamp', '')
        
        # Call JS function
        safe_content = json.dumps(content)
        safe_id = json.dumps(msg_id)
        safe_role = json.dumps(role)
        safe_timestamp = json.dumps(str(timestamp))
        safe_pinned = "true" if pinned else "false"
        
        js_code = f"appendMessage({safe_id}, {safe_role}, {safe_content}, {safe_pinned}, {safe_timestamp});"
        self.run_js(js_code)
        
    def update_streaming_message(self, content: str, reasoning: Optional[str] = None):
        """Update the last message content (used for streaming)."""
        safe_content = json.dumps(content)
        safe_reasoning = json.dumps(reasoning) if reasoning else "null"
        js_code = f"updateLastMessage({safe_content}, {safe_reasoning});"
        self.run_js(js_code)

        # Update internal data of the last message
        if self.messages:
            self.messages[-1]['content'] = content
            if reasoning:
                self.messages[-1]['reasoning'] = reasoning

    def update_message(self, message_data: Dict[str, Any]):
        """Update a message in the chat view."""
        msg_id = message_data.get('id')
        if msg_id and msg_id in self.messages_map:
            # Update the stored message data
            self.messages_map[msg_id] = message_data

            # Find and update in the ordered list
            for i, msg in enumerate(self.messages):
                if msg.get('id') == msg_id:
                    self.messages[i] = message_data
                    break

            # Update in JS - this will trigger gallery finalization
            content = message_data.get('content', '')
            safe_content = json.dumps(content)
            safe_id = json.dumps(msg_id)
            js_code = f"updateMessage({safe_id}, {safe_content});"
            self.run_js(js_code)

    def clear_messages(self):
        """Clear all messages."""
        self.messages_map.clear()
        self.messages.clear()
        self.run_js("clearChat();")

    def remove_message(self, message_data: Dict[str, Any]):
        """Remove a specific message."""
        msg_id = message_data.get('id')
        if msg_id and msg_id in self.messages_map:
            # Remove from map
            del self.messages_map[msg_id]
            # Remove from list
            self.messages = [m for m in self.messages if m.get('id') != msg_id]
            
            # Remove from JS CMD
            safe_id = json.dumps(msg_id)
            self.run_js(f"removeMessage({safe_id});")

    def handle_js_action(self, message_id: str, action: str):
        """Handle actions triggered from the web view."""
        # Special case for global actions (like settings)
        if message_id == 'global':
            if action == 'open_audio_settings':
                self.open_audio_settings_requested.emit()
            return

        message_data = self.messages_map.get(message_id)
        if not message_data:
            logger.warning(f"Received action {action} for unknown message ID {message_id}")
            return
            
        if action == 'regenerate':
            self.regenerate_requested.emit(message_data)
        elif action == 'edit':
            self.edit_requested.emit(message_data)
        elif action == 'speak':
            self.speak_text_requested.emit(message_data)
        elif action == 'show_prompt':
            self.show_prompt_requested.emit(message_data)
        elif action == 'copy':
            # Copy content to clipboard
            content = message_data.get('content', '')
            QApplication.clipboard().setText(content)
            logger.info("Message content copied to clipboard via 'copy' action")
            
    # --- New UI Control Methods ---
    
    def set_input_enabled(self, enabled: bool):
        """Enable or disable the input area."""
        js_code = f"setInputEnabled({'true' if enabled else 'false'});"
        self.run_js(js_code)
        
    def set_process_state(self, state: str, message: str = ""):
        """Set the process indicator state."""
        safe_state = json.dumps(state)
        safe_message = json.dumps(message)
        js_code = f"setProcessState({safe_state}, {safe_message});"
        self.run_js(js_code)
        
    def update_audio_status(self, is_recording: bool):
        """Update the audio recording status in UI."""
        js_code = f"updateAudioStatus({'true' if is_recording else 'false'});"
        self.run_js(js_code)

    def update_audio_player_state(self, state: str, position: int, duration: int):
        """
        Update the audio player state in the UI.

        Args:
            state: 'playing', 'paused', 'stopped'
            position: current position in ms
            duration: total duration in ms
        """
        safe_state = json.dumps(state)
        js_code = f"updateAudioPlayerState({safe_state}, {position}, {duration});"
        self.run_js(js_code)

    def set_input_value(self, value: str):
        """
        Set the value of the message input field.

        Args:
            value: The text to set in the input field
        """
        safe_value = json.dumps(value)
        js_code = f"setInputValue({safe_value});"
        self.run_js(js_code)

    def load_combos(self, combos: Dict[str, Any], default_combo_id: str = ""):
        """
        Load available combos into the chat UI.

        Args:
            combos: Dictionary of combo_id -> combo data
            default_combo_id: The ID of the default combo to select
        """
        safe_combos = json.dumps(combos)
        safe_default = json.dumps(default_combo_id)
        js_code = f"loadCombos({safe_combos}, {safe_default});"
        self.run_js(js_code)

    def initialize_lazy_loading(self, total_message_count: int):
        """
        Initialize the UI for lazy loading with the total message count.

        Args:
            total_message_count: Total number of messages available in conversation history
        """
        js_code = f"initializeLazyLoading({total_message_count});"
        self.run_js(js_code)

    def load_message_page(self, messages: List[Dict[str, Any]], prepend: bool = False):
        """
        Load a page of messages into the UI.

        Args:
            messages: List of message dictionaries to add
            prepend: If True, add messages at the top (older messages), otherwise append at bottom
        """
        safe_messages = json.dumps(messages)
        js_code = f"loadMessagePage({safe_messages}, {str(prepend).lower()});"
        self.run_js(js_code)

        # Update internal state
        if prepend:
            # Add older messages at the beginning
            self.messages = messages + self.messages
            for msg in messages:
                msg_id = msg.get('id', str(uuid.uuid4()))
                msg['id'] = msg_id
                self.messages_map[msg_id] = msg
        else:
            # Add newer messages at the end (normal case)
            for msg in messages:
                self.add_message(msg)

    # Lazy loading signal
    load_more_messages_requested = pyqtSignal()

    @pyqtSlot()
    def request_load_more_messages(self):
        """Handle request from JavaScript to load more messages."""
        self.load_more_messages_requested.emit()
