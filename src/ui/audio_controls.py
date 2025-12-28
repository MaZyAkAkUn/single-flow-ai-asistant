"""
Audio controls UI components for voice input (speech-to-text).
Provides buttons and controls for speech-to-text functionality.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from ..core.logging_config import get_logger

logger = get_logger(__name__)



class VoiceInputWidget(QWidget):
    """Widget for voice input (speech-to-text) controls."""

    voice_input_started = pyqtSignal()
    voice_input_finished = pyqtSignal(str)  # Emitted with transcribed text
    voice_input_error = pyqtSignal(str)  # Emitted with error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_listening = False
        self.listen_timer = QTimer()
        self.listen_timer.timeout.connect(self.check_listening_status)
        self.init_ui()

    def init_ui(self):
        """Initialize the voice input UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Status label
        self.status_label = QLabel("Voice Input")
        self.status_label.setStyleSheet("font-size: 10px; color: #666;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Recording indicator
        self.recording_indicator = QLabel("🎤")
        self.recording_indicator.setStyleSheet("font-size: 24px; color: #666;")
        self.recording_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.recording_indicator)

        # Control button
        self.voice_button = QPushButton("🎙️")
        self.voice_button.setFixedSize(40, 40)
        self.voice_button.setToolTip("Hold to record voice input")
        self.voice_button.setCheckable(True)
        self.voice_button.clicked.connect(self.on_voice_button_clicked)
        layout.addWidget(self.voice_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Set fixed size
        self.setFixedSize(80, 80)

    def on_voice_button_clicked(self):
        """Handle voice button clicks."""
        if self.voice_button.isChecked():
            self.start_voice_input()
        else:
            self.stop_voice_input()

    def start_voice_input(self):
        """Start voice input recording."""
        self.is_listening = True
        self.voice_button.setChecked(True)
        self.recording_indicator.setStyleSheet("font-size: 24px; color: #e74c3c;")
        self.status_label.setText("Listening...")
        self.voice_input_started.emit()

        # Start timer to check listening status
        self.listen_timer.start(100)

    def stop_voice_input(self):
        """Stop voice input recording."""
        self.is_listening = False
        self.voice_button.setChecked(False)
        self.recording_indicator.setStyleSheet("font-size: 24px; color: #666;")
        self.status_label.setText("Voice Input")
        self.listen_timer.stop()

    def check_listening_status(self):
        """Check if voice input is still active."""
        # This would be called by a timer to check the ASR status
        # For now, we'll simulate it
        pass

    def set_transcription_result(self, text: str):
        """
        Set the transcription result.

        Args:
            text: Transcribed text
        """
        if text:
            self.voice_input_finished.emit(text)
            self.status_label.setText("Success")
        else:
            self.voice_input_error.emit("No speech detected")
            self.status_label.setText("No speech")

        # Reset after a short delay
        QTimer.singleShot(2000, lambda: self.status_label.setText("Voice Input"))

    def set_error(self, error_message: str):
        """
        Set an error state.

        Args:
            error_message: Error message to display
        """
        self.voice_input_error.emit(error_message)
        self.status_label.setText("Error")
        self.recording_indicator.setStyleSheet("font-size: 24px; color: #e74c3c;")

        # Reset after a short delay
        QTimer.singleShot(3000, lambda: self.reset_ui())

    def reset_ui(self):
        """Reset the UI to default state."""
        self.stop_voice_input()
        self.status_label.setText("Voice Input")
        self.recording_indicator.setStyleSheet("font-size: 24px; color: #666;")


class AudioControlsWidget(QWidget):
    """Widget for voice input (speech-to-text) controls."""

    voice_input_requested = pyqtSignal()  # Request to start voice input
    voice_input_cancelled = pyqtSignal()  # Request to cancel voice input

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the voice input controls UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Voice input widget
        self.voice_widget = VoiceInputWidget()
        layout.addWidget(self.voice_widget)

        # Settings button
        self.settings_button = QPushButton("⚙️")
        self.settings_button.setFixedSize(32, 32)
        self.settings_button.setToolTip("Audio settings")
        layout.addWidget(self.settings_button)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def connect_signals(self):
        """Connect widget signals."""
        self.voice_widget.voice_input_started.connect(self.on_voice_input_started)
        self.voice_widget.voice_input_finished.connect(self.on_voice_input_finished)
        self.voice_widget.voice_input_error.connect(self.on_voice_input_error)

    def on_voice_input_started(self):
        """Handle voice input start."""
        self.voice_input_requested.emit()

    def on_voice_input_finished(self, text: str):
        """Handle voice input completion with text."""
        # Emit the transcribed text (handled by parent widget)
        pass

    def on_voice_input_error(self, error: str):
        """Handle voice input error."""
        logger.error(f"Voice input error: {error}")

    def start_voice_input(self):
        """Start voice input recording."""
        self.voice_widget.start_voice_input()

    def stop_voice_input(self):
        """Stop voice input recording."""
        self.voice_widget.stop_voice_input()

    def set_voice_transcription(self, text: str):
        """
        Set the result of voice transcription.

        Args:
            text: Transcribed text
        """
        self.voice_widget.set_transcription_result(text)

    def set_voice_error(self, error: str):
        """
        Set a voice input error.

        Args:
            error: Error message
        """
        self.voice_widget.set_error(error)

    def is_voice_listening(self) -> bool:
        """
        Check if voice input is currently listening.

        Returns:
            True if listening, False otherwise
        """
        return self.voice_widget.is_listening
